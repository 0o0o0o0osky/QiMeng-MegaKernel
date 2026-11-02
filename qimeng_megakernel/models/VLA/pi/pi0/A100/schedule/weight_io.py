"""Pi0 safetensors bridge for the A100 baseline verifier.

This module intentionally depends only on torch, safetensors, and the in-repo
Llama Q/K interleave helper (no OpenPI import).  OpenPI is a development-only
oracle isolated in the verifier's ``_openpi_runner``.
All linear matrices returned here are output-major ``[N, K]``.
Gemma/AE Q/K rows are interleaved at load so ``apply_rope_interleaved`` matches
HF ``rotate_half`` math.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from safetensors import safe_open

from qimeng_megakernel.models.LLM.llama.preprocess import _interleave_qk_weight_indices

from .pi0_config import Pi0A100Config

VISION = "paligemma_with_expert.paligemma.model.vision_tower.vision_model"
GEMMA = "paligemma_with_expert.paligemma.model.language_model"
AE = "paligemma_with_expert.gemma_expert.model"
PROJECTOR = "paligemma_with_expert.paligemma.model.multi_modal_projector.linear"


@dataclass(frozen=True)
class Pi0WeightBundle:
    """Lazy, checkpoint-backed Pi0 weights with a stable checkpoint path."""

    path: Path
    device: str = "cpu"

    def keys(self) -> list[str]:
        with safe_open(str(self.path), framework="pt", device=self.device) as handle:
            return list(handle.keys())

    def tensor(self, key: str) -> torch.Tensor:
        with safe_open(str(self.path), framework="pt", device=self.device) as handle:
            if key not in handle.keys():
                raise KeyError(f"Pi0 checkpoint has no tensor {key!r}")
            return handle.get_tensor(key).contiguous()


def checkpoint_path(path: str | Path | None = None) -> Path:
    """Resolve an explicit path or the ``PI0_CKPT`` environment variable."""
    configured = path or os.environ.get("PI0_CKPT")
    if configured is None:
        raise FileNotFoundError("set PI0_CKPT to a local Pi0 safetensors checkpoint")
    source = Path(configured).expanduser()
    if source.is_dir():
        source /= "model.safetensors"
    if not source.is_file():
        raise FileNotFoundError(f"Pi0 safetensors checkpoint not found: {source}")
    return source


def load_pi0_safetensors(path: str | Path | None = None, *, device: str = "cpu") -> Pi0WeightBundle:
    return Pi0WeightBundle(checkpoint_path(path), device)


def fold_rms_weight(weight: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    """Fold Gemma ``RMSNorm(x) * (1 + gamma)`` into output-major weight columns."""
    if weight.ndim != 2 or gamma.ndim != 1 or weight.shape[1] != gamma.shape[0]:
        raise ValueError(
            f"RMS fold expects weight [N,K] and gamma [K], got {tuple(weight.shape)} "
            f"and {tuple(gamma.shape)}"
        )
    return (weight.float() * (1.0 + gamma.float()).unsqueeze(0)).contiguous()


def _linear(bundle: Pi0WeightBundle, key: str, gamma: torch.Tensor | None = None) -> torch.Tensor:
    """Load an OpenPI ``nn.Linear.weight`` in native output-major ``[N,K]``."""
    weight = bundle.tensor(key)
    if gamma is not None:
        # Gemma RMSNorm uses (1 + w); folding it into W preserves the baseline
        # APIs, which deliberately keep RMS normalization parameter-free.
        weight = fold_rms_weight(weight, gamma)
    return weight.contiguous()


def _layers(indices: Iterable[int] | None, count: int) -> list[int]:
    result = list(range(count)) if indices is None else sorted(set(int(i) for i in indices))
    if not result or result[0] < 0 or result[-1] >= count:
        raise ValueError(f"layer indices must be in [0, {count})")
    return result


def to_siglip_embeddings(bundle: Pi0WeightBundle) -> dict[str, torch.Tensor]:
    """Patch embed, position embed, post-LN, and multi-modal projector."""
    return {
        "patch_w": bundle.tensor(f"{VISION}.embeddings.patch_embedding.weight"),
        "patch_b": bundle.tensor(f"{VISION}.embeddings.patch_embedding.bias"),
        "pos_emb": bundle.tensor(f"{VISION}.embeddings.position_embedding.weight"),
        "post_ln_g": bundle.tensor(f"{VISION}.post_layernorm.weight"),
        "post_ln_b": bundle.tensor(f"{VISION}.post_layernorm.bias"),
        "proj_w": bundle.tensor(f"{PROJECTOR}.weight"),
        "proj_b": bundle.tensor(f"{PROJECTOR}.bias"),
    }


def to_siglip_baseline_weights(bundle: Pi0WeightBundle, *, layers: Iterable[int] | None = None) -> dict[str, object]:
    """Return SigLIP baseline layer dictionaries and embedding boundary weights."""
    out_layers: list[dict[str, torch.Tensor]] = []
    for layer in _layers(layers, Pi0A100Config.vision_layers):
        p = f"{VISION}.encoder.layers.{layer}"
        out_layers.append({
            "g1": bundle.tensor(f"{p}.layer_norm1.weight"),
            "b1": bundle.tensor(f"{p}.layer_norm1.bias"),
            "Wqkv": torch.cat([_linear(bundle, f"{p}.self_attn.{n}_proj.weight") for n in ("q", "k", "v")]),
            "bqkv": torch.cat([bundle.tensor(f"{p}.self_attn.{n}_proj.bias") for n in ("q", "k", "v")]),
            "Wo": _linear(bundle, f"{p}.self_attn.out_proj.weight"),
            "bo": bundle.tensor(f"{p}.self_attn.out_proj.bias"),
            "g2": bundle.tensor(f"{p}.layer_norm2.weight"),
            "b2": bundle.tensor(f"{p}.layer_norm2.bias"),
            "W1": _linear(bundle, f"{p}.mlp.fc1.weight"),
            "b1f": bundle.tensor(f"{p}.mlp.fc1.bias"),
            "W2": _linear(bundle, f"{p}.mlp.fc2.weight"),
            "b2f": bundle.tensor(f"{p}.mlp.fc2.bias"),
        })
    return {
        "layers": out_layers,
        "layer_indices": _layers(layers, Pi0A100Config.vision_layers),
        "embeddings": to_siglip_embeddings(bundle),
    }


def interleave_qk_rows(wq: torch.Tensor, wk: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Permute Q/K output rows from HF ``rotate_half`` layout to interleaved RoPE."""
    q_idx, k_idx = _interleave_qk_weight_indices(
        Pi0A100Config.num_heads,
        Pi0A100Config.num_kv_heads,
        Pi0A100Config.head_dim,
        wq.device,
    )
    return wq[q_idx].contiguous(), wk[k_idx].contiguous()


# Private alias retained for downstream imports predating the production pack.
_interleave_qk_rows = interleave_qk_rows


def _gemma_layer(bundle: Pi0WeightBundle, prefix: str, layer: int) -> tuple[torch.Tensor, ...]:
    """Load one Gemma/AE layer; Q/K rows interleaved for ``apply_rope_interleaved``."""
    p = f"{prefix}.layers.{layer}"
    attn_gamma = bundle.tensor(f"{p}.input_layernorm.weight")
    ffn_gamma = bundle.tensor(f"{p}.post_attention_layernorm.weight")
    wq = _linear(bundle, f"{p}.self_attn.q_proj.weight", attn_gamma)
    wk = _linear(bundle, f"{p}.self_attn.k_proj.weight", attn_gamma)
    wv = _linear(bundle, f"{p}.self_attn.v_proj.weight", attn_gamma)
    wq, wk = interleave_qk_rows(wq, wk)
    return (
        torch.cat([wq, wk, wv]),
        _linear(bundle, f"{p}.self_attn.o_proj.weight"),
        _linear(bundle, f"{p}.mlp.gate_proj.weight", ffn_gamma),
        _linear(bundle, f"{p}.mlp.up_proj.weight", ffn_gamma),
        _linear(bundle, f"{p}.mlp.down_proj.weight"),
    )


def to_gemma_baseline_weights(bundle: Pi0WeightBundle, *, layers: Iterable[int] | None = None) -> dict[str, object]:
    indices = _layers(layers, Pi0A100Config.encoder_layers)
    return {
        "layers": [_gemma_layer(bundle, GEMMA, i) for i in indices],
        "layer_indices": indices,
        "final_norm": bundle.tensor(f"{GEMMA}.norm.weight"),
    }


def to_ae_boundary_weights(bundle: Pi0WeightBundle) -> dict[str, torch.Tensor]:
    """Top-level action / state projection weights used at flow-step boundaries."""
    out: dict[str, torch.Tensor] = {}
    for name in (
        "action_in_proj.weight",
        "action_in_proj.bias",
        "action_out_proj.weight",
        "action_out_proj.bias",
        "action_time_mlp_in.weight",
        "action_time_mlp_in.bias",
        "action_time_mlp_out.weight",
        "action_time_mlp_out.bias",
        "state_proj.weight",
        "state_proj.bias",
    ):
        try:
            out[name.replace(".", "_")] = bundle.tensor(name)
        except KeyError:
            continue
    out["final_norm"] = bundle.tensor(f"{AE}.norm.weight")
    return out


def to_ae_baseline_weights(bundle: Pi0WeightBundle, *, layers: Iterable[int] | None = None) -> dict[str, object]:
    """Return Pi0 action-expert weights; Pi0 has RMSNorm, not Pi0.5 AdaRMS."""
    indices = _layers(layers, Pi0A100Config.decoder_layers)
    grouped = {"qkv": [], "oproj": [], "gate": [], "up": [], "down": []}
    for layer in indices:
        qkv, oproj, gate, up, down = _gemma_layer(bundle, AE, layer)
        grouped["qkv"].append(qkv)
        grouped["oproj"].append(oproj)
        grouped["gate"].append(gate)
        grouped["up"].append(up)
        grouped["down"].append(down)
    return {
        "layers": grouped,
        "layer_indices": indices,
        "boundary": to_ae_boundary_weights(bundle),
    }
