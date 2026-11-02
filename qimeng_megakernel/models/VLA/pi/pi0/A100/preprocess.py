"""Pi0 checkpoint loading, official observation transforms, and A100 ABI packing.

This module is the host-side source of truth for the three pybind ABIs in
``kernel/mk_{siglip,gemma,ae}.cu``.  It converts the real OpenPI PyTorch
safetensors layout into the padded, folded, interleaved, and split-K layouts
consumed by those kernels.  It does not import verifier helpers, PyVM code, or
generated inputs.

The returned pack contains checkpoint-populated tensors plus zeroed storage for
runtime inputs and scratch.  It deliberately does not create instructions or
compose module launches; those remain separate production capabilities.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

from .schedule.pi0_config import Pi0A100Config as C
from .schedule.weight_io import (
    AE,
    GEMMA,
    PROJECTOR,
    VISION,
    checkpoint_path,
    fold_rms_weight,
    interleave_qk_rows,
)


SIGLIP_ABI_ORDER = (
    "vision_x",
    "siglip_x_normed",
    "ln_gamma",
    "ln_beta",
    "qkv_w",
    "qkv_b",
    "qkv_buf",
    "attn_out",
    "oproj_w",
    "oproj_b",
    "fc1_w",
    "fc1_b",
    "fc1_buf",
    "fc2_w",
    "fc2_b",
    "patch_in",
    "patch_w",
    "patch_b",
    "proj_w",
    "proj_b",
    "proj_out",
    "ln_stats",
    "ln_fold_qkv",
    "ln_fold_fc1",
    "Bar",
)

GEMMA_ABI_ORDER = (
    "encoder_x",
    "rms_factor",
    "qkv_w",
    "qkv_buf",
    "rope",
    "k_cache",
    "v_cache",
    # Materialized attention reuses the former split-attention ABI slots.
    "attn_scores",
    "attn_vt",
    "attn_out",
    "oproj_w",
    "gate_w",
    "up_w",
    "gate_buf",
    "up_buf",
    "down_w",
    "Bar",
)

AE_ABI_ORDER = (
    "x",
    "x_acc",
    "rms_factor",
    "qkv_w",
    "qkv_buf",
    "rope",
    "k_cache",
    "v_cache",
    "attn_partial",
    "attn_stats",
    "attn_out",
    "oproj_w",
    "gate_w",
    "up_w",
    "gate_buf",
    "ffn_buf",
    "down_w",
    "noise",
    "act_h",
    "act_w_in",
    "act_w_mlp",
    "act_w_out",
    "act_bias",
    "Bar",
)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_BF16 = torch.bfloat16
_F32 = torch.float32
_I32 = torch.int32

_SIGLIP_WIDTH_PAD = 1280
_SIGLIP_HEAD_DIM = 72
_SIGLIP_HEAD_DIM_PAD = 80
_SIGLIP_QKV_PAD = 3840
_SIGLIP_MLP_PAD = 4608
_SIGLIP_LN_STATS_ROWS = 42
_PATCH_K = 3 * 14 * 14
_PATCH_K_PAD = 768

_GEMMA_RMS_ROWS = 34
_AE_M_PAD = 64
_AE_K_SPLIT = 256
_AE_OPROJ_SPLITS = C.encoder_width // _AE_K_SPLIT
_AE_DOWN_SPLITS = C.decoder_mlp_width // _AE_K_SPLIT
_AE_ACT_BIAS_ROWS = 13
_AE_TIME_BIAS_ROWS = 10

# These ABI tensors intentionally contain both checkpoint-derived and runtime-
# materialized regions.  No other tensor may silently belong to two classes.
_PARTIALLY_DERIVED_BY_MODULE = {
    "siglip": frozenset({"vision_x"}),
    "gemma": frozenset(),
    "ae": frozenset({"act_bias"}),
}


@dataclass(frozen=True)
class CheckpointTensorSpec:
    """One tensor required to construct the production megakernel pack."""

    key: str
    shape: tuple[int, ...]
    dtype: torch.dtype = _BF16

    @property
    def nbytes(self) -> int:
        return math.prod(self.shape) * torch.empty((), dtype=self.dtype).element_size()


@dataclass(frozen=True)
class CheckpointInventory:
    """Validated identity and complete required-tensor inventory."""

    path: Path
    sha256: str
    file_size: int
    checkpoint_tensor_count: int
    required_tensor_count: int
    required_tensor_bytes: int
    unused_checkpoint_keys: tuple[str, ...]


@dataclass
class MegakernelTensorSet:
    """Named tensors for one pybind ABI, excluding the instruction tensor."""

    module: str
    abi_order: tuple[str, ...]
    tensors: dict[str, torch.Tensor]
    checkpoint_populated: frozenset[str]
    runtime_populated: frozenset[str]
    scratch: frozenset[str]

    def __post_init__(self) -> None:
        try:
            allowed_for_module = _PARTIALLY_DERIVED_BY_MODULE[self.module]
        except KeyError as exc:
            raise ValueError(f"unsupported megakernel module: {self.module!r}") from exc
        names = frozenset(self.tensors)
        expected = frozenset(self.abi_order)
        if names != expected or len(self.abi_order) != len(expected):
            raise ValueError(
                f"{self.module} ABI tensor names differ: "
                f"missing={sorted(expected - names)}, extra={sorted(names - expected)}"
            )
        classified = self.checkpoint_populated | self.runtime_populated | self.scratch
        if classified != names:
            raise ValueError(
                f"{self.module} ABI classification differs: "
                f"unclassified={sorted(names - classified)}, unknown={sorted(classified - names)}"
            )
        scratch_overlap = self.scratch & (self.checkpoint_populated | self.runtime_populated)
        if scratch_overlap:
            raise ValueError(
                f"{self.module} scratch tensors cannot have another classification: "
                f"{sorted(scratch_overlap)}"
            )
        partially_derived = self.checkpoint_populated & self.runtime_populated
        allowed = allowed_for_module & names
        if partially_derived != allowed:
            raise ValueError(
                f"{self.module} partially-derived tensors differ: "
                f"expected={sorted(allowed)}, got={sorted(partially_derived)}"
            )
        for name, tensor in self.tensors.items():
            if not tensor.is_contiguous():
                raise ValueError(f"{self.module}.{name} must be contiguous")

    def abi_arguments(self) -> tuple[torch.Tensor, ...]:
        """Return tensors in the exact ``bind_kernel`` order."""
        return tuple(self.tensors[name] for name in self.abi_order)

    @property
    def partially_derived(self) -> frozenset[str]:
        """Tensors with checkpoint-derived and request/runtime-populated regions."""
        return self.checkpoint_populated & self.runtime_populated

    @property
    def nbytes(self) -> int:
        return sum(tensor.numel() * tensor.element_size() for tensor in self.tensors.values())


@dataclass
class Pi0A100TensorPack:
    """Checkpoint-backed static weights and ABI storage for all three kernels."""

    inventory: CheckpointInventory
    num_views: int
    prompt_len: int
    chunk: int
    flow_steps: int
    ae_layout: dict[str, int]
    siglip: MegakernelTensorSet
    gemma: MegakernelTensorSet
    ae: MegakernelTensorSet
    runtime_aux: dict[str, torch.Tensor]

    @property
    def nbytes(self) -> int:
        aux = sum(t.numel() * t.element_size() for t in self.runtime_aux.values())
        return self.siglip.nbytes + self.gemma.nbytes + self.ae.nbytes + aux


def _requirements() -> tuple[CheckpointTensorSpec, ...]:
    specs: list[CheckpointTensorSpec] = [
        CheckpointTensorSpec(f"{VISION}.embeddings.patch_embedding.weight", (1152, 3, 14, 14)),
        CheckpointTensorSpec(f"{VISION}.embeddings.patch_embedding.bias", (1152,)),
        CheckpointTensorSpec(f"{VISION}.embeddings.position_embedding.weight", (256, 1152)),
        CheckpointTensorSpec(f"{VISION}.post_layernorm.weight", (1152,)),
        CheckpointTensorSpec(f"{VISION}.post_layernorm.bias", (1152,)),
        CheckpointTensorSpec(f"{PROJECTOR}.weight", (2048, 1152)),
        CheckpointTensorSpec(f"{PROJECTOR}.bias", (2048,)),
    ]
    for layer in range(C.vision_layers):
        prefix = f"{VISION}.encoder.layers.{layer}"
        specs.extend(
            [
                CheckpointTensorSpec(f"{prefix}.layer_norm1.weight", (1152,)),
                CheckpointTensorSpec(f"{prefix}.layer_norm1.bias", (1152,)),
                CheckpointTensorSpec(f"{prefix}.layer_norm2.weight", (1152,)),
                CheckpointTensorSpec(f"{prefix}.layer_norm2.bias", (1152,)),
            ]
        )
        for projection in ("q", "k", "v"):
            specs.append(
                CheckpointTensorSpec(f"{prefix}.self_attn.{projection}_proj.weight", (1152, 1152))
            )
            specs.append(
                CheckpointTensorSpec(f"{prefix}.self_attn.{projection}_proj.bias", (1152,))
            )
        specs.extend(
            [
                CheckpointTensorSpec(f"{prefix}.self_attn.out_proj.weight", (1152, 1152)),
                CheckpointTensorSpec(f"{prefix}.self_attn.out_proj.bias", (1152,)),
                CheckpointTensorSpec(f"{prefix}.mlp.fc1.weight", (4304, 1152)),
                CheckpointTensorSpec(f"{prefix}.mlp.fc1.bias", (4304,)),
                CheckpointTensorSpec(f"{prefix}.mlp.fc2.weight", (1152, 4304)),
                CheckpointTensorSpec(f"{prefix}.mlp.fc2.bias", (1152,)),
            ]
        )

    for prefix, width, mlp_width in (
        (GEMMA, C.encoder_width, C.encoder_mlp_width),
        (AE, C.decoder_width, C.decoder_mlp_width),
    ):
        for layer in range(C.encoder_layers):
            layer_prefix = f"{prefix}.layers.{layer}"
            specs.extend(
                [
                    CheckpointTensorSpec(f"{layer_prefix}.input_layernorm.weight", (width,)),
                    CheckpointTensorSpec(
                        f"{layer_prefix}.post_attention_layernorm.weight", (width,)
                    ),
                    CheckpointTensorSpec(
                        f"{layer_prefix}.self_attn.q_proj.weight", (2048, width)
                    ),
                    CheckpointTensorSpec(
                        f"{layer_prefix}.self_attn.k_proj.weight", (256, width)
                    ),
                    CheckpointTensorSpec(
                        f"{layer_prefix}.self_attn.v_proj.weight", (256, width)
                    ),
                    CheckpointTensorSpec(
                        f"{layer_prefix}.self_attn.o_proj.weight", (width, 2048)
                    ),
                    CheckpointTensorSpec(
                        f"{layer_prefix}.mlp.gate_proj.weight", (mlp_width, width)
                    ),
                    CheckpointTensorSpec(
                        f"{layer_prefix}.mlp.up_proj.weight", (mlp_width, width)
                    ),
                    CheckpointTensorSpec(
                        f"{layer_prefix}.mlp.down_proj.weight", (width, mlp_width)
                    ),
                ]
            )
        specs.append(CheckpointTensorSpec(f"{prefix}.norm.weight", (width,)))

    specs.extend(
        [
            CheckpointTensorSpec("action_in_proj.weight", (1024, 32)),
            CheckpointTensorSpec("action_in_proj.bias", (1024,)),
            CheckpointTensorSpec("action_out_proj.weight", (32, 1024)),
            CheckpointTensorSpec("action_out_proj.bias", (32,)),
            CheckpointTensorSpec("action_time_mlp_in.weight", (1024, 2048)),
            CheckpointTensorSpec("action_time_mlp_in.bias", (1024,)),
            CheckpointTensorSpec("action_time_mlp_out.weight", (1024, 1024)),
            CheckpointTensorSpec("action_time_mlp_out.bias", (1024,)),
            CheckpointTensorSpec("state_proj.weight", (1024, 32)),
            CheckpointTensorSpec("state_proj.bias", (1024,)),
            # PaliGemma ties input embeddings to lm_head; the checkpoint stores the
            # shared matrix under lm_head rather than embed_tokens.
            CheckpointTensorSpec("paligemma_with_expert.paligemma.lm_head.weight", (257152, 2048)),
            CheckpointTensorSpec("paligemma_with_expert.gemma_expert.lm_head.weight", (257152, 1024)),
        ]
    )
    return tuple(specs)


REQUIRED_CHECKPOINT_TENSORS = _requirements()


def sha256_file(path: str | Path, *, block_size: int = 8 << 20) -> str:
    """Hash an on-disk checkpoint without materializing it in memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_checkpoint_inventory(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> CheckpointInventory:
    """Validate every tensor used by the production pack and record file identity."""
    if expected_sha256 is not None and _SHA256.fullmatch(expected_sha256) is None:
        raise ValueError("expected_sha256 must be a lowercase SHA-256 when provided")
    resolved = checkpoint_path(path).resolve()
    actual_sha256 = sha256_file(resolved)
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise ValueError(
            f"Pi0 checkpoint SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}"
        )

    with safe_open(str(resolved), framework="pt", device="cpu") as handle:
        checkpoint_keys = frozenset(handle.keys())
        required_keys = frozenset(spec.key for spec in REQUIRED_CHECKPOINT_TENSORS)
        for spec in REQUIRED_CHECKPOINT_TENSORS:
            if spec.key not in checkpoint_keys:
                raise ValueError(f"Pi0 checkpoint has no required tensor {spec.key!r}")
            tensor_slice = handle.get_slice(spec.key)
            shape = tuple(int(dim) for dim in tensor_slice.get_shape())
            if shape != spec.shape:
                raise ValueError(
                    f"Pi0 checkpoint tensor {spec.key!r} has shape {shape}; expected {spec.shape}"
                )
            dtype = str(tensor_slice.get_dtype()).upper()
            if dtype != "BF16":
                raise ValueError(
                    f"Pi0 checkpoint tensor {spec.key!r} has dtype {dtype}; expected BF16"
                )

    return CheckpointInventory(
        path=resolved,
        sha256=actual_sha256,
        file_size=resolved.stat().st_size,
        checkpoint_tensor_count=len(checkpoint_keys),
        required_tensor_count=len(REQUIRED_CHECKPOINT_TENSORS),
        required_tensor_bytes=sum(spec.nbytes for spec in REQUIRED_CHECKPOINT_TENSORS),
        unused_checkpoint_keys=tuple(sorted(checkpoint_keys - required_keys)),
    )


class _TensorReader:
    def __init__(self, handle: object, device: torch.device):
        self._handle = handle
        self._device = device

    def get(self, key: str) -> torch.Tensor:
        tensor = self._handle.get_tensor(key)  # type: ignore[attr-defined]
        return tensor.to(device=self._device).contiguous()


def _zeros(
    shape: Sequence[int],
    *,
    device: torch.device,
    dtype: torch.dtype = _BF16,
) -> torch.Tensor:
    return torch.zeros(tuple(int(dim) for dim in shape), device=device, dtype=dtype)


def _bias_tiles(bias: torch.Tensor, padded: int) -> torch.Tensor:
    if padded % C.tile_n:
        raise ValueError(f"bias padded width must be divisible by {C.tile_n}, got {padded}")
    out = torch.zeros(padded, device=bias.device, dtype=_BF16)
    out[: bias.numel()] = bias.reshape(-1).to(_BF16)
    return out.reshape(padded // C.tile_n, C.tile_n)


def _head_pad_rows(source: torch.Tensor) -> torch.Tensor:
    """Pad every SigLIP 72-row head to 80 rows."""
    if source.shape[0] != C.vision_num_heads * _SIGLIP_HEAD_DIM:
        raise ValueError(f"unexpected SigLIP head-row shape {tuple(source.shape)}")
    tail = tuple(source.shape[1:])
    out = torch.zeros(
        (C.vision_num_heads, _SIGLIP_HEAD_DIM_PAD, *tail),
        device=source.device,
        dtype=source.dtype,
    )
    out[:, :_SIGLIP_HEAD_DIM] = source.reshape(C.vision_num_heads, _SIGLIP_HEAD_DIM, *tail)
    return out.reshape(C.vision_num_heads * _SIGLIP_HEAD_DIM_PAD, *tail)


def _head_pad_columns(source: torch.Tensor) -> torch.Tensor:
    """Pad every SigLIP 72-column head to 80 columns."""
    if source.ndim != 2 or source.shape[1] != C.vision_width:
        raise ValueError(f"unexpected SigLIP head-column shape {tuple(source.shape)}")
    out = torch.zeros(
        source.shape[0], C.vision_num_heads, _SIGLIP_HEAD_DIM_PAD,
        device=source.device, dtype=source.dtype,
    )
    out[:, :, :_SIGLIP_HEAD_DIM] = source.reshape(
        source.shape[0], C.vision_num_heads, _SIGLIP_HEAD_DIM
    )
    return out.reshape(source.shape[0], C.vision_num_heads * _SIGLIP_HEAD_DIM_PAD)


def _pack_ln_fold(correction: torch.Tensor, bias: torch.Tensor, padded: int) -> torch.Tensor:
    """Pack LnFold rows as ``[c_tile0,b_tile0,c_tile1,b_tile1,...]``."""
    if correction.shape != (padded,) or bias.shape != (padded,):
        raise ValueError("LnFold vectors must already include output padding")
    tiles = padded // C.tile_n
    out = torch.empty(2 * tiles, C.tile_n, device=correction.device, dtype=_BF16)
    out[0::2] = correction.reshape(tiles, C.tile_n).to(_BF16)
    out[1::2] = bias.reshape(tiles, C.tile_n).to(_BF16)
    return out


def _pack_ln_fold_adjacent(
    correction: torch.Tensor, bias: torch.Tensor, padded: int
) -> torch.Tensor:
    """Pack one bf16 ``[correction, bias]`` pair per output column."""
    if correction.shape != (padded,) or bias.shape != (padded,):
        raise ValueError("LnFold vectors must already include output padding")
    return torch.stack((correction, bias), dim=1).to(_BF16).contiguous()


def _fold_siglip_linear(
    weight: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    bias: torch.Tensor,
    *,
    output_pad: int,
    input_pad: int,
    head_pad_output: bool = False,
    adjacent_fold: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fold LayerNorm affine into one SigLIP weight and its LnFold table."""
    folded = weight.float() * gamma.float().unsqueeze(0)
    correction = folded.sum(dim=1)
    folded_bias = (weight.float() * beta.float().unsqueeze(0)).sum(dim=1) + bias.float()
    if head_pad_output:
        folded = _head_pad_rows(folded)
        correction = _head_pad_rows(correction)
        folded_bias = _head_pad_rows(folded_bias)

    packed_weight = _zeros((output_pad, input_pad), device=weight.device)
    packed_weight[: folded.shape[0], : folded.shape[1]] = folded.to(_BF16)
    c_padded = torch.zeros(output_pad, device=weight.device, dtype=_F32)
    b_padded = torch.zeros_like(c_padded)
    c_padded[: correction.numel()] = correction.reshape(-1)
    b_padded[: folded_bias.numel()] = folded_bias.reshape(-1)
    packer = _pack_ln_fold_adjacent if adjacent_fold else _pack_ln_fold
    return packed_weight, packer(c_padded, b_padded, output_pad)


def _to_split_planes(weight: torch.Tensor, splits: int) -> torch.Tensor:
    """Output-major ``[N,K]`` -> physical ``[splits,N,K/splits]``."""
    if weight.ndim != 2 or weight.shape[1] % splits:
        raise ValueError(f"cannot split weight shape {tuple(weight.shape)} into {splits} K planes")
    segment = weight.shape[1] // splits
    return weight.reshape(weight.shape[0], splits, segment).permute(1, 0, 2).contiguous()


def _build_siglip(reader: _TensorReader, *, num_views: int, device: torch.device) -> MegakernelTensorSet:
    seq = C.siglip_seq_for(num_views)
    layers = C.vision_layers
    tensors = {
        "vision_x": _zeros((1, 1, seq, _SIGLIP_WIDTH_PAD), device=device),
        "siglip_x_normed": _zeros((1, 1, seq, _SIGLIP_WIDTH_PAD), device=device),
        "ln_gamma": _zeros((1, 1, 2 * layers + 1, C.vision_width), device=device),
        "ln_beta": _zeros((1, 1, 2 * layers + 1, C.vision_width), device=device),
        "qkv_w": _zeros((layers, _SIGLIP_QKV_PAD, _SIGLIP_WIDTH_PAD), device=device),
        "qkv_b": _zeros((1, layers, _SIGLIP_QKV_PAD // C.tile_n, C.tile_n), device=device),
        "qkv_buf": _zeros((1, 1, seq, _SIGLIP_QKV_PAD), device=device),
        "attn_out": _zeros((1, 1, seq, _SIGLIP_WIDTH_PAD), device=device),
        "oproj_w": _zeros((layers, _SIGLIP_WIDTH_PAD, _SIGLIP_WIDTH_PAD), device=device),
        "oproj_b": _zeros((1, layers, _SIGLIP_WIDTH_PAD // C.tile_n, C.tile_n), device=device),
        "fc1_w": _zeros((layers, _SIGLIP_MLP_PAD, _SIGLIP_WIDTH_PAD), device=device),
        "fc1_b": _zeros((1, layers, _SIGLIP_MLP_PAD // C.tile_n, C.tile_n), device=device),
        "fc1_buf": _zeros((1, 1, seq, _SIGLIP_MLP_PAD), device=device),
        "fc2_w": _zeros((layers, _SIGLIP_WIDTH_PAD, _SIGLIP_MLP_PAD), device=device),
        "fc2_b": _zeros((1, layers, _SIGLIP_WIDTH_PAD // C.tile_n, C.tile_n), device=device),
        "patch_in": _zeros((1, 1, seq, _PATCH_K_PAD), device=device),
        "patch_w": _zeros((1, _SIGLIP_WIDTH_PAD, _PATCH_K_PAD), device=device),
        "patch_b": _zeros((1, 1, _SIGLIP_WIDTH_PAD // C.tile_n, C.tile_n), device=device),
        "proj_w": _zeros((1, C.encoder_width, _SIGLIP_WIDTH_PAD), device=device),
        "proj_b": _zeros((1, 1, C.encoder_width // C.tile_n, C.tile_n), device=device),
        "proj_out": _zeros((1, 1, seq, C.encoder_width), device=device),
        "ln_stats": _zeros((1, 1, _SIGLIP_LN_STATS_ROWS, seq), device=device, dtype=_F32),
        "ln_fold_qkv": _zeros((layers, _SIGLIP_QKV_PAD, 2), device=device),
        "ln_fold_fc1": _zeros((layers, _SIGLIP_MLP_PAD, 2), device=device),
        "Bar": _zeros((1, layers, C.siglip_barrier_rows, C.barrier_slots), device=device, dtype=_I32),
    }

    position = reader.get(f"{VISION}.embeddings.position_embedding.weight")
    tensors["vision_x"][0, 0, :, : C.vision_width] = position.repeat(num_views, 1)

    patch = reader.get(f"{VISION}.embeddings.patch_embedding.weight")
    # PyTorch Conv2d stores O,C,H,W; the runtime patch rows flatten H,W,C.
    patch_hwc = patch.permute(0, 2, 3, 1).reshape(C.vision_width, _PATCH_K)
    tensors["patch_w"][0, : C.vision_width, :_PATCH_K] = patch_hwc
    patch_bias = reader.get(f"{VISION}.embeddings.patch_embedding.bias")
    tensors["patch_b"][0, 0] = _bias_tiles(patch_bias, _SIGLIP_WIDTH_PAD)

    tensors["ln_gamma"][0, 0, -1] = reader.get(f"{VISION}.post_layernorm.weight")
    tensors["ln_beta"][0, 0, -1] = reader.get(f"{VISION}.post_layernorm.bias")
    projector = reader.get(f"{PROJECTOR}.weight")
    tensors["proj_w"][0, :, : C.vision_width] = projector
    tensors["proj_b"][0, 0] = _bias_tiles(reader.get(f"{PROJECTOR}.bias"), C.encoder_width)

    for layer in range(layers):
        prefix = f"{VISION}.encoder.layers.{layer}"
        gamma1 = reader.get(f"{prefix}.layer_norm1.weight")
        beta1 = reader.get(f"{prefix}.layer_norm1.bias")
        gamma2 = reader.get(f"{prefix}.layer_norm2.weight")
        beta2 = reader.get(f"{prefix}.layer_norm2.bias")
        tensors["ln_gamma"][0, 0, 2 * layer] = gamma1
        tensors["ln_beta"][0, 0, 2 * layer] = beta1
        tensors["ln_gamma"][0, 0, 2 * layer + 1] = gamma2
        tensors["ln_beta"][0, 0, 2 * layer + 1] = beta2

        qkv_weights = torch.cat(
            [reader.get(f"{prefix}.self_attn.{name}_proj.weight") for name in ("q", "k", "v")]
        )
        qkv_bias = torch.cat(
            [reader.get(f"{prefix}.self_attn.{name}_proj.bias") for name in ("q", "k", "v")]
        )
        folded_sections, fold_sections = [], []
        for section in range(3):
            start = section * C.vision_width
            folded, fold = _fold_siglip_linear(
                qkv_weights[start : start + C.vision_width],
                gamma1,
                beta1,
                qkv_bias[start : start + C.vision_width],
                output_pad=C.vision_num_heads * _SIGLIP_HEAD_DIM_PAD,
                input_pad=_SIGLIP_WIDTH_PAD,
                head_pad_output=True,
                adjacent_fold=True,
            )
            folded_sections.append(folded)
            fold_sections.append(fold)
        tensors["qkv_w"][layer] = torch.cat(folded_sections)
        # Each section is [1280,2] with one adjacent c/b pair per padded output
        # column.  Concatenation produces the global Q/K/V column order.
        tensors["ln_fold_qkv"][layer] = torch.cat(fold_sections)

        oproj = reader.get(f"{prefix}.self_attn.out_proj.weight")
        tensors["oproj_w"][layer, : C.vision_width, :] = _head_pad_columns(oproj)
        tensors["oproj_b"][0, layer] = _bias_tiles(
            reader.get(f"{prefix}.self_attn.out_proj.bias"), _SIGLIP_WIDTH_PAD
        )

        fc1, fc1_fold = _fold_siglip_linear(
            reader.get(f"{prefix}.mlp.fc1.weight"),
            gamma2,
            beta2,
            reader.get(f"{prefix}.mlp.fc1.bias"),
            output_pad=_SIGLIP_MLP_PAD,
            input_pad=_SIGLIP_WIDTH_PAD,
            adjacent_fold=True,
        )
        tensors["fc1_w"][layer] = fc1
        tensors["ln_fold_fc1"][layer] = fc1_fold

        # Fc2 is one K=4608 GEMM, so the weight is the padded output-major matrix as it
        # stands.  It used to be sliced into six 768-wide K segments stacked on dim 0 to
        # feed the split loop; the fused kernel wants exactly the matrix that slicing
        # started from, so the repack is gone rather than replaced.
        tensors["fc2_w"][layer, : C.vision_width, : C.vision_mlp_width] = reader.get(
            f"{prefix}.mlp.fc2.weight"
        )
        tensors["fc2_b"][0, layer] = _bias_tiles(
            reader.get(f"{prefix}.mlp.fc2.bias"), _SIGLIP_WIDTH_PAD
        )

    checkpoint_populated = frozenset(
        {
            "vision_x",
            "ln_gamma",
            "ln_beta",
            "qkv_w",
            "oproj_w",
            "oproj_b",
            "fc1_w",
            "fc2_w",
            "fc2_b",
            "patch_w",
            "patch_b",
            "proj_w",
            "proj_b",
            "ln_fold_qkv",
            "ln_fold_fc1",
        }
    )
    runtime_populated = frozenset({"vision_x", "patch_in"})
    scratch = frozenset(tensors) - checkpoint_populated - runtime_populated
    return MegakernelTensorSet(
        module="siglip",
        abi_order=SIGLIP_ABI_ORDER,
        tensors=tensors,
        checkpoint_populated=checkpoint_populated,
        runtime_populated=runtime_populated,
        scratch=scratch,
    )


def _fold_gemma_layer(reader: _TensorReader, prefix: str, layer: int) -> tuple[torch.Tensor, ...]:
    layer_prefix = f"{prefix}.layers.{layer}"
    attn_gamma = reader.get(f"{layer_prefix}.input_layernorm.weight")
    ffn_gamma = reader.get(f"{layer_prefix}.post_attention_layernorm.weight")
    q = fold_rms_weight(reader.get(f"{layer_prefix}.self_attn.q_proj.weight"), attn_gamma)
    k = fold_rms_weight(reader.get(f"{layer_prefix}.self_attn.k_proj.weight"), attn_gamma)
    v = fold_rms_weight(reader.get(f"{layer_prefix}.self_attn.v_proj.weight"), attn_gamma)
    q, k = interleave_qk_rows(q, k)
    return (
        torch.cat((q, k, v)).to(_BF16),
        reader.get(f"{layer_prefix}.self_attn.o_proj.weight"),
        fold_rms_weight(reader.get(f"{layer_prefix}.mlp.gate_proj.weight"), ffn_gamma).to(_BF16),
        fold_rms_weight(reader.get(f"{layer_prefix}.mlp.up_proj.weight"), ffn_gamma).to(_BF16),
        reader.get(f"{layer_prefix}.mlp.down_proj.weight"),
    )


def _build_gemma(
    reader: _TensorReader,
    *,
    num_views: int,
    prompt_len: int,
    device: torch.device,
) -> MegakernelTensorSet:
    seq = C.encoder_seq_for(num_views, prompt_len)
    seq_pad = C.encoder_seq_pad_for(seq)
    if seq_pad > C.gemma_max_seq_pad:
        raise ValueError(
            f"Gemma seq_pad={seq_pad} exceeds A100 ABI maximum {C.gemma_max_seq_pad}"
        )
    layers = C.encoder_layers
    tensors = {
        "encoder_x": _zeros((1, 1, seq_pad, C.encoder_width), device=device),
        "rms_factor": _zeros((1, 1, _GEMMA_RMS_ROWS, C.gemma_max_seq_pad), device=device, dtype=_F32),
        "qkv_w": _zeros((layers, C.encoder_qkv_out, C.encoder_width), device=device),
        "qkv_buf": _zeros((1, 1, seq_pad, C.encoder_qkv_out), device=device),
        "rope": _zeros((1, 1, C.gemma_max_seq_pad, C.head_dim), device=device),
        "k_cache": _zeros((layers, C.gemma_max_seq_pad, C.head_dim), device=device),
        "v_cache": _zeros((layers, C.gemma_max_seq_pad, C.head_dim), device=device),
        # `attn_scores` is the S/P matrix, one [seq_pad, seq_pad] block per query
        # head, laid out flat because it has a second tenant: `gemma_qkv_rope`
        # borrows it as fp32 Q scratch, which is what `attn_partial` used to be
        # for.  That tenancy wants `seq_pad * encoder_width` floats, i.e. 4096
        # rows of bf16, and attention wants `num_heads * seq_pad`; at one view
        # the borrower is the larger of the two, so take the max.
        "attn_scores": _zeros(
            (1, 1, max(C.num_heads * seq_pad, 2 * C.encoder_width), seq_pad),
            device=device,
        ),
        "attn_vt": _zeros((1, 1, C.head_dim, seq_pad), device=device),
        "attn_out": _zeros((1, 1, seq_pad, C.encoder_width), device=device),
        "oproj_w": _zeros((layers, C.encoder_width, C.encoder_width), device=device),
        "gate_w": _zeros((layers, C.encoder_mlp_width, C.encoder_width), device=device),
        "up_w": _zeros((layers, C.encoder_mlp_width, C.encoder_width), device=device),
        "gate_buf": _zeros((1, 1, seq_pad, C.encoder_mlp_width), device=device),
        "up_buf": _zeros((1, 1, seq_pad, C.encoder_mlp_width), device=device),
        "down_w": _zeros((layers, C.encoder_width, C.encoder_mlp_width), device=device),
        "Bar": _zeros((1, layers, C.gemma_barrier_rows, C.barrier_slots), device=device, dtype=_I32),
    }
    for layer in range(layers):
        qkv, oproj, gate, up, down = _fold_gemma_layer(reader, GEMMA, layer)
        tensors["qkv_w"][layer] = qkv
        tensors["oproj_w"][layer] = oproj
        tensors["gate_w"][layer] = gate
        tensors["up_w"][layer] = up
        tensors["down_w"][layer] = down

    checkpoint_populated = frozenset({"qkv_w", "oproj_w", "gate_w", "up_w", "down_w"})
    runtime_populated = frozenset({"encoder_x", "rope"})
    scratch = frozenset(tensors) - checkpoint_populated - runtime_populated
    return MegakernelTensorSet(
        module="gemma",
        abi_order=GEMMA_ABI_ORDER,
        tensors=tensors,
        checkpoint_populated=checkpoint_populated,
        runtime_populated=runtime_populated,
        scratch=scratch,
    )


def _sinusoidal_time_embedding(
    times: torch.Tensor,
    *,
    width: int,
) -> torch.Tensor:
    if width % 2:
        raise ValueError("time embedding width must be even")
    fraction = torch.linspace(0.0, 1.0, width // 2, device=times.device, dtype=_F32)
    period = 4e-3 * (4.0 / 4e-3) ** fraction
    angle = times.float().unsqueeze(1) * ((2.0 * math.pi) / period).unsqueeze(0)
    return torch.cat((torch.sin(angle), torch.cos(angle)), dim=1)


def _build_action_head(
    reader: _TensorReader,
    *,
    flow_steps: int,
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    if flow_steps != _AE_TIME_BIAS_ROWS:
        raise ValueError(
            f"current AE ABI has {_AE_TIME_BIAS_ROWS} time-bias rows; got flow_steps={flow_steps}"
        )
    action_in_w = reader.get("action_in_proj.weight")
    action_in_b = reader.get("action_in_proj.bias")
    time_in_w = reader.get("action_time_mlp_in.weight")
    time_in_b = reader.get("action_time_mlp_in.bias")
    action_half, time_half = time_in_w[:, : C.decoder_width], time_in_w[:, C.decoder_width :]

    fused_in = action_half.float() @ action_in_w.float()
    fixed_action_bias = action_half.float() @ action_in_b.float()
    times = 1.0 - torch.arange(flow_steps, device=device, dtype=_F32) / float(flow_steps)
    time_embeddings = _sinusoidal_time_embedding(times, width=C.decoder_width)
    time_bias = (
        fixed_action_bias.unsqueeze(0)
        + time_embeddings @ time_half.float().T
        + time_in_b.float().unsqueeze(0)
    )

    act_w_in = _zeros((1, 1, C.decoder_width, _AE_K_SPLIT), device=device)
    act_w_in[0, 0, :, : C.action_dim] = fused_in.to(_BF16)
    act_w_mlp = reader.get("action_time_mlp_out.weight").reshape(
        1, 1, C.decoder_width, C.decoder_width
    )

    # ACT_OUT uses store_add into noise.  Fold both the expert's final RMSNorm
    # affine and the fixed 10-step Euler delta into the projection.
    final_norm = reader.get(f"{AE}.norm.weight")
    dt = -1.0 / float(flow_steps)
    act_w_out = (
        reader.get("action_out_proj.weight").float()
        * (1.0 + final_norm.float()).unsqueeze(0)
        * dt
    ).to(_BF16).reshape(1, 1, C.action_dim, C.decoder_width)

    act_bias = _zeros((1, 1, _AE_ACT_BIAS_ROWS, C.decoder_width), device=device, dtype=_F32)
    act_bias[0, 0, :flow_steps] = time_bias
    act_bias[0, 0, 10] = reader.get("action_time_mlp_out.bias").float()
    act_bias[0, 0, 11, : C.action_dim] = reader.get("action_out_proj.bias").float() * dt
    # Row 12 is state_h and is intentionally populated by the runtime composer.
    # Keep the checkpoint-derived recipe tensors.  Flow times belong to the
    # runtime schedule, so a caller can rematerialize rows 0..9 explicitly.
    return (
        act_w_in,
        act_w_mlp,
        act_w_out,
        act_bias,
        time_half.contiguous(),
        fixed_action_bias.contiguous(),
        time_in_b.float().contiguous(),
    )


def _build_ae(
    reader: _TensorReader,
    *,
    num_views: int,
    prompt_len: int,
    chunk: int,
    flow_steps: int,
    device: torch.device,
) -> tuple[MegakernelTensorSet, dict[str, torch.Tensor], dict[str, int]]:
    if chunk not in C.CHUNK_CHOICES:
        raise ValueError(f"AE chunk must be one of {C.CHUNK_CHOICES}, got {chunk}")
    layout = C.ae_kv_layout_for(num_views, chunk, prompt_len)
    prefix_blocks = math.ceil(layout["prefix_len"] / C.ae_kv_block)
    attn_spans = math.ceil(prefix_blocks / 2) + 1
    layers = C.decoder_layers
    tensors = {
        "x": _zeros((1, 1, _AE_M_PAD, C.decoder_width), device=device),
        "x_acc": _zeros((1, 1, _AE_M_PAD, C.decoder_width), device=device, dtype=_F32),
        "rms_factor": _zeros((1, 1, 3, _AE_M_PAD), device=device, dtype=_F32),
        "qkv_w": _zeros((1, layers, C.encoder_qkv_out, C.decoder_width), device=device),
        "qkv_buf": _zeros((1, 1, _AE_M_PAD, C.encoder_qkv_out), device=device),
        "rope": _zeros((1, 1, _AE_M_PAD, C.head_dim), device=device),
        "k_cache": _zeros((1, layers, layout["kv_rows"], C.head_dim), device=device),
        "v_cache": _zeros((1, layers, layout["kv_rows"], C.head_dim), device=device),
        "attn_partial": _zeros((1, attn_spans, _AE_M_PAD, C.encoder_width), device=device),
        "attn_stats": _zeros((1, attn_spans, 2 * C.num_heads, _AE_M_PAD), device=device, dtype=_F32),
        "attn_out": _zeros((1, 1, _AE_OPROJ_SPLITS * _AE_M_PAD, _AE_K_SPLIT), device=device),
        "oproj_w": _zeros((1, layers * _AE_OPROJ_SPLITS, C.decoder_width, _AE_K_SPLIT), device=device),
        "gate_w": _zeros((1, layers, C.decoder_mlp_width, C.decoder_width), device=device),
        "up_w": _zeros((1, layers, C.decoder_mlp_width, C.decoder_width), device=device),
        "gate_buf": _zeros((1, 1, _AE_DOWN_SPLITS * _AE_M_PAD, _AE_K_SPLIT), device=device),
        "ffn_buf": _zeros((1, 1, _AE_DOWN_SPLITS * _AE_M_PAD, _AE_K_SPLIT), device=device),
        "down_w": _zeros((1, layers * _AE_DOWN_SPLITS, C.decoder_width, _AE_K_SPLIT), device=device),
        "noise": _zeros((1, 1, _AE_M_PAD, _AE_K_SPLIT), device=device),
        "act_h": _zeros((1, 1, _AE_M_PAD, C.decoder_width), device=device),
        "act_w_in": _zeros((1, 1, C.decoder_width, _AE_K_SPLIT), device=device),
        "act_w_mlp": _zeros((1, 1, C.decoder_width, C.decoder_width), device=device),
        "act_w_out": _zeros((1, 1, C.action_dim, C.decoder_width), device=device),
        "act_bias": _zeros((1, 1, _AE_ACT_BIAS_ROWS, C.decoder_width), device=device, dtype=_F32),
        "Bar": _zeros((1, layers, C.ae_barrier_rows, C.barrier_slots), device=device, dtype=_I32),
    }
    for layer in range(layers):
        qkv, oproj, gate, up, down = _fold_gemma_layer(reader, AE, layer)
        tensors["qkv_w"][0, layer] = qkv
        tensors["oproj_w"][0, layer * _AE_OPROJ_SPLITS : (layer + 1) * _AE_OPROJ_SPLITS] = (
            _to_split_planes(oproj, _AE_OPROJ_SPLITS)
        )
        tensors["gate_w"][0, layer] = gate
        tensors["up_w"][0, layer] = up
        tensors["down_w"][0, layer * _AE_DOWN_SPLITS : (layer + 1) * _AE_DOWN_SPLITS] = (
            _to_split_planes(down, _AE_DOWN_SPLITS)
        )

    action = _build_action_head(reader, flow_steps=flow_steps, device=device)
    for name, value in zip(
        ("act_w_in", "act_w_mlp", "act_w_out", "act_bias"), action[:4], strict=True
    ):
        tensors[name] = value.contiguous()

    checkpoint_populated = frozenset(
        {
            "qkv_w",
            "oproj_w",
            "gate_w",
            "up_w",
            "down_w",
            "act_w_in",
            "act_w_mlp",
            "act_w_out",
            "act_bias",
        }
    )
    runtime_populated = frozenset({"rope", "k_cache", "v_cache", "noise", "act_bias"})
    scratch = frozenset(tensors) - checkpoint_populated - runtime_populated
    return (
        MegakernelTensorSet(
            module="ae",
            abi_order=AE_ABI_ORDER,
            tensors=tensors,
            checkpoint_populated=checkpoint_populated,
            runtime_populated=runtime_populated,
            scratch=scratch,
        ),
        {
            "ae_time_bias_weight": action[4],
            "ae_action_bias_contribution": action[5],
            "ae_time_mlp_in_bias": action[6],
        },
        layout,
    )


def canonical_flow_times(
    flow_steps: int = _AE_TIME_BIAS_ROWS,
    *,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Return the OpenPI Euler schedule's input times ``1.0, 0.9, ..., 0.1``."""
    if type(flow_steps) is not int or flow_steps != _AE_TIME_BIAS_ROWS:
        raise ValueError(
            f"flow_steps must be the exact integer {_AE_TIME_BIAS_ROWS}; got {flow_steps!r}"
        )
    return 1.0 - torch.arange(flow_steps, device=device, dtype=_F32) / float(flow_steps)


def _require_finite_outside_capture(value: torch.Tensor, message: str) -> None:
    """Validate eager inputs without introducing a host sync during graph capture.

    Every materializer is run once on the warmup stream before capture.  Request
    values are additionally validated on CPU at the public inference boundary,
    so replay never relies on an unchecked tensor.
    """
    if value.is_cuda and torch.cuda.is_current_stream_capturing():
        return
    if not torch.isfinite(value).all():
        raise ValueError(message)


def materialize_ae_time_biases(
    pack: Pi0A100TensorPack,
    *,
    times: torch.Tensor | None = None,
) -> None:
    """Materialize schedule-derived AE ``act_bias`` rows 0..9 in place.

    The weight/bias recipe is checkpoint-static; the timestep vector is a
    runtime schedule input.  This separation prevents a zero or seeded bias
    table from masquerading as a valid checkpoint pack.
    """
    device = pack.ae.tensors["act_bias"].device
    selected = canonical_flow_times(pack.flow_steps, device=device) if times is None else times
    selected = torch.as_tensor(selected, device=device, dtype=_F32)
    if selected.shape != (pack.flow_steps,):
        raise ValueError(f"AE flow times must be finite [{pack.flow_steps}], got {tuple(selected.shape)}")
    _require_finite_outside_capture(
        selected, "AE flow times must contain only finite values"
    )
    embeddings = _sinusoidal_time_embedding(selected, width=C.decoder_width)
    time_weight = pack.runtime_aux["ae_time_bias_weight"].float()
    action_bias = pack.runtime_aux["ae_action_bias_contribution"].float()
    time_bias = pack.runtime_aux["ae_time_mlp_in_bias"].float()
    pack.ae.tensors["act_bias"][0, 0, : pack.flow_steps] = (
        action_bias.unsqueeze(0) + embeddings @ time_weight.T + time_bias.unsqueeze(0)
    )


def materialize_ae_state(pack: Pi0A100TensorPack, state: torch.Tensor) -> None:
    """Materialize request-derived ``state_h`` into ``act_bias`` row 12."""
    device = pack.ae.tensors["act_bias"].device
    value = torch.as_tensor(state, device=device, dtype=_F32)
    if value.shape == (1, C.action_dim):
        value = value[0]
    if value.shape != (C.action_dim,):
        raise ValueError(f"normalized Pi0 state must be finite [{C.action_dim}], got {tuple(value.shape)}")
    _require_finite_outside_capture(
        value, "normalized Pi0 state must contain only finite values"
    )
    weight = pack.runtime_aux["state_proj_weight"].float()
    bias = pack.runtime_aux["state_proj_bias"].float()
    pack.ae.tensors["act_bias"][0, 0, 12] = weight @ value + bias


def materialize_ae_noise(pack: Pi0A100TensorPack, noise: torch.Tensor) -> None:
    """Pack request noise into AE rows 1..chunk and zero every ABI pad lane."""
    target = pack.ae.tensors["noise"]
    value = torch.as_tensor(noise, device=target.device, dtype=_BF16)
    expected = (pack.chunk, C.action_dim)
    if value.shape != expected:
        raise ValueError(f"Pi0 noise must be finite {expected}, got {tuple(value.shape)}")
    _require_finite_outside_capture(value, "Pi0 noise must contain only finite values")
    target.zero_()
    target[0, 0, 1 : 1 + pack.chunk, : C.action_dim] = value


def repeat_siglip_position_embedding(
    position: torch.Tensor,
    *,
    num_views: int,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """Create a fresh padded ``vision_x`` base from an immutable position table."""
    if type(num_views) is not int or num_views not in C.VIEWS_CHOICES:
        raise ValueError(
            f"num_views must be an exact integer in {C.VIEWS_CHOICES}; got {num_views!r}"
        )
    target_device = position.device if device is None else torch.device(device)
    source = torch.as_tensor(position, device=target_device, dtype=_BF16)
    expected = (C.vision_tokens_per_view, C.vision_width)
    if source.shape != expected:
        raise ValueError(f"SigLIP position embedding must be {expected}, got {tuple(source.shape)}")
    _require_finite_outside_capture(
        source, "SigLIP position embedding must contain only finite values"
    )
    seq = C.siglip_seq_for(num_views)
    out = _zeros((1, 1, seq, _SIGLIP_WIDTH_PAD), device=target_device)
    out[0, 0, :, : C.vision_width] = source.repeat(num_views, 1)
    return out


def materialize_siglip_position(pack: Pi0A100TensorPack) -> None:
    """Reset mutable ``vision_x`` from the independent checkpoint position table."""
    position = pack.runtime_aux["siglip_position_embedding"]
    fresh = repeat_siglip_position_embedding(
        position,
        num_views=pack.num_views,
        device=pack.siglip.tensors["vision_x"].device,
    )
    pack.siglip.tensors["vision_x"].copy_(fresh)


def build_checkpoint_tensor_pack(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    num_views: int = 3,
    prompt_len: int = 20,
    chunk: int = C.production_chunk,
    flow_steps: int = 10,
    device: str | torch.device = "cpu",
) -> Pi0A100TensorPack:
    """Build all three A100 ABI packs from one verified local checkpoint.

    ``instructions`` are intentionally absent from each ABI tuple.  The pybind
    wrappers accept them as the final argument, after :meth:`abi_arguments`.
    """
    if type(num_views) is not int or num_views not in C.VIEWS_CHOICES:
        raise ValueError(
            f"num_views must be an exact integer in {C.VIEWS_CHOICES}; got {num_views!r}"
        )
    if type(chunk) is not int or chunk not in C.CHUNK_CHOICES:
        raise ValueError(f"chunk must be an exact integer in {C.CHUNK_CHOICES}; got {chunk!r}")
    if type(flow_steps) is not int or flow_steps != _AE_TIME_BIAS_ROWS:
        raise ValueError(
            f"flow_steps must be the exact integer {_AE_TIME_BIAS_ROWS}; got {flow_steps!r}"
        )
    if type(prompt_len) is not int or prompt_len <= 0:
        raise ValueError("prompt_len must be a positive integer")
    target = torch.device(device)
    inventory = inspect_checkpoint_inventory(path, expected_sha256=expected_sha256)

    with safe_open(str(inventory.path), framework="pt", device="cpu") as handle:
        reader = _TensorReader(handle, target)
        siglip = _build_siglip(reader, num_views=num_views, device=target)
        gemma = _build_gemma(
            reader,
            num_views=num_views,
            prompt_len=prompt_len,
            device=target,
        )
        ae, ae_aux, ae_layout = _build_ae(
            reader,
            num_views=num_views,
            prompt_len=prompt_len,
            chunk=chunk,
            flow_steps=flow_steps,
            device=target,
        )
        # Gemma writes the immutable prefix into the same physical K/V storage
        # that the action expert reads and extends.  The AE ABI keeps its leading
        # singleton plane, so these are zero-copy views of Gemma's cache.
        ae.tensors["k_cache"] = gemma.tensors["k_cache"].unsqueeze(0)
        ae.tensors["v_cache"] = gemma.tensors["v_cache"].unsqueeze(0)

        runtime_aux = {
            "siglip_position_embedding": reader.get(
                f"{VISION}.embeddings.position_embedding.weight"
            ).clone(),
            "encoder_token_embedding": reader.get(
                "paligemma_with_expert.paligemma.lm_head.weight"
            ),
            "encoder_final_norm": reader.get(f"{GEMMA}.norm.weight"),
            "state_proj_weight": reader.get("state_proj.weight"),
            "state_proj_bias": reader.get("state_proj.bias"),
            **ae_aux,
        }

    return Pi0A100TensorPack(
        inventory=inventory,
        num_views=num_views,
        prompt_len=prompt_len,
        chunk=chunk,
        flow_steps=flow_steps,
        ae_layout=ae_layout,
        siglip=siglip,
        gemma=gemma,
        ae=ae,
        runtime_aux=runtime_aux,
    )


@dataclass(frozen=True)
class ModelInputs:
    """Dataset-independent tensors after the official Pi0 input boundary."""

    dataset: str
    images: torch.Tensor
    state: torch.Tensor
    prompt_tokens: torch.Tensor
    raw_state: np.ndarray

    @property
    def prompt_len(self) -> int:
        return int(self.prompt_tokens.numel())


def _checkpoint_file(path: str | Path) -> Path:
    return checkpoint_path(path).resolve()


def _norm_stats(checkpoint: str | Path, dataset: str) -> dict[str, Any]:
    """Resolve the checkpoint-owned statistics for the selected public dataset."""
    if dataset not in {"libero", "droid"}:
        raise ValueError(f"unsupported Pi0 normalization dataset {dataset!r}")
    root = _checkpoint_file(checkpoint).parent / "assets"
    matches = sorted(root.glob("**/norm_stats.json"))
    selected = [
        path
        for path in matches
        if dataset in {part.casefold() for part in path.relative_to(root).parts[:-1]}
    ]
    if not selected and len(matches) == 1:
        selected = matches
    if len(selected) != 1:
        raise FileNotFoundError(
            f"expected one {dataset} norm_stats.json below {root}; found {selected or matches}"
        )
    payload = json.loads(selected[0].read_text(encoding="utf-8"))
    try:
        stats = payload["norm_stats"]
        for field in ("state", "actions"):
            for moment in ("mean", "std"):
                values = np.asarray(stats[field][moment], dtype=np.float32)
                if values.shape != (C.action_dim,) or not np.isfinite(values).all():
                    raise ValueError(
                        f"{selected[0]} {field}.{moment} must be finite [{C.action_dim}]"
                    )
    except KeyError as exc:
        raise ValueError(f"malformed Pi0 normalization file: {selected[0]}") from exc
    return stats


def _tokenizer_path() -> Path:
    value = os.environ.get("PI0_TOKENIZER")
    if not value:
        raise RuntimeError("PI0_TOKENIZER must point to paligemma_tokenizer.model")
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"PI0_TOKENIZER is unavailable: {path}")
    return path


def _tokenize_prompt(prompt: str) -> torch.Tensor:
    try:
        import sentencepiece  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("sentencepiece is required for Pi0 prompt tokenization") from exc
    processor = sentencepiece.SentencePieceProcessor(
        model_file=str(_tokenizer_path())
    )
    cleaned = prompt.strip().replace("_", " ").replace("\n", " ")
    tokens = processor.encode(cleaned, add_bos=True) + processor.encode("\n")
    if not tokens:
        raise ValueError("prompt produced no active tokens")
    return torch.tensor(tokens[:48], dtype=torch.int64)


def _image(value: Any) -> torch.Tensor:
    array = np.asarray(value)
    if array.ndim != 3:
        raise ValueError(f"Pi0 image must be rank 3, got {array.shape}")
    if array.shape[0] == 3 and array.shape[-1] != 3:
        array = np.moveaxis(array, 0, -1)
    if array.shape[-1] != 3:
        raise ValueError(f"Pi0 image must have three channels, got {array.shape}")
    if np.issubdtype(array.dtype, np.floating):
        if not np.isfinite(array).all():
            raise ValueError("Pi0 image must contain only finite values")
        array = (array * 255.0).clip(0, 255).astype(np.uint8)
    if array.dtype != np.uint8:
        array = array.astype(np.uint8)
    tensor = torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1)[None].float()
    height, width = int(tensor.shape[-2]), int(tensor.shape[-1])
    ratio = max(width / 224.0, height / 224.0)
    resized_h, resized_w = int(height / ratio), int(width / ratio)
    tensor = F.interpolate(tensor, (resized_h, resized_w), mode="bilinear", align_corners=False)
    tensor = tensor.round().clamp_(0, 255)
    pad_h, pad_w = 224 - resized_h, 224 - resized_w
    tensor = F.pad(tensor, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))
    return tensor[0].permute(1, 2, 0).div_(127.5).sub_(1.0).contiguous()


def _normalize_state(state: np.ndarray, stats: Mapping[str, Any]) -> torch.Tensor:
    padded = np.zeros(C.action_dim, dtype=np.float32)
    if state.ndim != 1 or state.size > C.action_dim:
        raise ValueError(f"Pi0 state must be a vector of at most {C.action_dim} values")
    if not np.isfinite(state).all():
        raise ValueError("Pi0 state must contain only finite values")
    padded[: state.size] = state.astype(np.float32, copy=False)
    mean = np.asarray(stats["state"]["mean"], dtype=np.float32)
    std = np.asarray(stats["state"]["std"], dtype=np.float32)
    return torch.from_numpy((padded - mean) / (std + 1e-6))


def preprocess_observation(
    checkpoint: str | Path,
    observation: Mapping[str, Any],
    *,
    views: int,
) -> ModelInputs:
    """Apply the LIBERO or DROID input mapping, then the common Pi0 transforms."""
    if views not in C.VIEWS_CHOICES:
        raise ValueError(f"views must be one of {C.VIEWS_CHOICES}, got {views}")
    if "observation/image" in observation:
        dataset = "libero"
        base = observation["observation/image"]
        wrist = observation["observation/wrist_image"]
        raw_state = np.asarray(observation["observation/state"], dtype=np.float32)
    elif "observation/exterior_image_1_left" in observation:
        dataset = "droid"
        base = observation["observation/exterior_image_1_left"]
        wrist = observation["observation/wrist_image_left"]
        raw_state = np.concatenate(
            (
                np.asarray(observation["observation/joint_position"], dtype=np.float32),
                np.asarray(observation["observation/gripper_position"], dtype=np.float32).reshape(-1),
            )
        )
    else:
        raise ValueError("observation is neither official LIBERO nor DROID format")
    prompt = observation.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("observation prompt must be a non-empty string")
    image_values = [base, wrist, wrist]
    images = torch.stack([_image(value) for value in image_values[:views]])
    return ModelInputs(
        dataset=dataset,
        images=images,
        state=_normalize_state(raw_state, _norm_stats(checkpoint, dataset)),
        prompt_tokens=_tokenize_prompt(prompt),
        raw_state=raw_state.copy(),
    )


def postprocess_actions(
    checkpoint: str | Path,
    model_actions: torch.Tensor | np.ndarray,
    *,
    dataset: str,
    raw_state: np.ndarray,
) -> np.ndarray:
    """Apply checkpoint unnormalization and the official dataset output mapping."""
    actions = np.asarray(
        model_actions.detach().float().cpu() if isinstance(model_actions, torch.Tensor) else model_actions,
        dtype=np.float32,
    ).copy()
    if actions.ndim != 2 or actions.shape[1] != C.action_dim:
        raise ValueError(
            f"Pi0 model actions must have shape [chunk, {C.action_dim}], got {actions.shape}"
        )
    if not np.isfinite(actions).all():
        raise ValueError("Pi0 model actions must contain only finite values")
    stats = _norm_stats(checkpoint, dataset)["actions"]
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    actions = actions * (std + 1e-6) + mean
    if dataset == "libero":
        actions[:, :6] += raw_state[None, :6]
        return actions[:, :7]
    if dataset == "droid":
        return actions[:, :8]
    raise ValueError(f"unsupported Pi0 output dataset {dataset!r}")


__all__ = [
    "AE_ABI_ORDER",
    "GEMMA_ABI_ORDER",
    "REQUIRED_CHECKPOINT_TENSORS",
    "SIGLIP_ABI_ORDER",
    "CheckpointInventory",
    "CheckpointTensorSpec",
    "MegakernelTensorSet",
    "ModelInputs",
    "Pi0A100TensorPack",
    "build_checkpoint_tensor_pack",
    "canonical_flow_times",
    "inspect_checkpoint_inventory",
    "materialize_ae_state",
    "materialize_ae_noise",
    "materialize_ae_time_biases",
    "materialize_siglip_position",
    "postprocess_actions",
    "preprocess_observation",
    "repeat_siglip_position_embedding",
    "sha256_file",
]
