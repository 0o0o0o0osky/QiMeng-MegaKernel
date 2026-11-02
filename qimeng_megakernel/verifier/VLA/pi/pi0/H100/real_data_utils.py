"""Real-checkpoint inputs shared by the Pi0 H100 stage tests."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
from safetensors import safe_open
import torch

from qimeng_megakernel.models.VLA.pi.pi0.H100.preprocess import (
    Pi0H100TensorPack,
    build_checkpoint_tensor_pack,
    materialize_ae_time_biases,
)
from qimeng_megakernel.models.VLA.pi.pi0.H100.schedule.fused import (
    ae_ops,
    gemma_ops,
    siglip_ops,
)
from qimeng_megakernel.models.VLA.pi.pi0.H100.schedule.globs import (
    AeGlobals,
    GemmaGlobals,
    SiglipGlobals,
)
from qimeng_megakernel.models.VLA.pi.pi0.H100.schedule.instructions import (
    python_vm,
    scheduler,
)
from qimeng_megakernel.models.VLA.pi.pi0.H100.schedule.pi0_config import (
    Pi0H100Config as C,
)
from qimeng_megakernel.models.VLA.pi.pi0.H100.schedule.weight_io import (
    PROJECTOR,
    VISION,
)


_CHECKPOINT_VALUE = os.environ.get("PI0_CKPT", "")
CHECKPOINT = Path(_CHECKPOINT_VALUE).expanduser() if _CHECKPOINT_VALUE else None
LOCAL_CHECKPOINT_AVAILABLE = CHECKPOINT is not None and CHECKPOINT.is_file()


def require_checkpoint() -> Path:
    if not LOCAL_CHECKPOINT_AVAILABLE or CHECKPOINT is None:
        pytest.skip("set PI0_CKPT to the local Pi0 safetensors checkpoint")
    return CHECKPOINT


def require_h100() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    props = torch.cuda.get_device_properties("cuda:0")
    if (props.major, props.minor, props.multi_processor_count) != (9, 0, C.sm_count):
        pytest.skip("Pi0 H100 tests require compute 9.0 with 132 SMs")


def make_pack(*, views: int = 2, chunk: int = 20, prompt_len: int = 20) -> Pi0H100TensorPack:
    require_h100()
    return build_checkpoint_tensor_pack(
        require_checkpoint(),
        num_views=views,
        prompt_len=prompt_len,
        chunk=chunk,
        flow_steps=C.num_flow_steps,
        device="cuda:0",
    )


def seeded(
    *shape: int,
    seed: int,
    device: str | torch.device = "cuda:0",
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(shape, generator=generator, dtype=torch.float32).to(device, dtype=dtype)


def _read(handle: Any, key: str, device: torch.device) -> torch.Tensor:
    return handle.get_tensor(key).to(device=device).contiguous()


def _siglip_weights(
    handle: Any,
    *,
    layers: int,
    device: torch.device,
) -> list[dict[str, torch.Tensor]]:
    result: list[dict[str, torch.Tensor]] = []
    for layer in range(layers):
        prefix = f"{VISION}.encoder.layers.{layer}"
        qkv_w = torch.cat(
            [_read(handle, f"{prefix}.self_attn.{name}_proj.weight", device) for name in ("q", "k", "v")]
        )
        qkv_b = torch.cat(
            [_read(handle, f"{prefix}.self_attn.{name}_proj.bias", device) for name in ("q", "k", "v")]
        )
        result.append(
            {
                "g1": _read(handle, f"{prefix}.layer_norm1.weight", device),
                "b1": _read(handle, f"{prefix}.layer_norm1.bias", device),
                "Wqkv": qkv_w,
                "bqkv": qkv_b,
                "Wo": _read(handle, f"{prefix}.self_attn.out_proj.weight", device),
                "bo": _read(handle, f"{prefix}.self_attn.out_proj.bias", device),
                "g2": _read(handle, f"{prefix}.layer_norm2.weight", device),
                "b2": _read(handle, f"{prefix}.layer_norm2.bias", device),
                "W1": _read(handle, f"{prefix}.mlp.fc1.weight", device),
                "b1f": _read(handle, f"{prefix}.mlp.fc1.bias", device),
                "W2": _read(handle, f"{prefix}.mlp.fc2.weight", device),
                "b2f": _read(handle, f"{prefix}.mlp.fc2.bias", device),
            }
        )
    return result


def make_siglip_globs(
    *,
    views: int = 2,
    layers: int = 1,
    seed: int = 100,
    include_head_tail: bool = False,
) -> tuple[SiglipGlobals, list[dict[str, torch.Tensor]]]:
    require_h100()
    checkpoint = require_checkpoint()
    device = torch.device("cuda:0")
    seq = C.siglip_seq_for(views)
    with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
        weights = _siglip_weights(handle, layers=layers, device=device)
        position = _read(handle, f"{VISION}.embeddings.position_embedding.weight", device).repeat(views, 1)
        patch_w = _read(handle, f"{VISION}.embeddings.patch_embedding.weight", device)
        patch_w = patch_w.permute(0, 2, 3, 1).reshape(C.vision_width, -1)
        patch_b = _read(handle, f"{VISION}.embeddings.patch_embedding.bias", device)
        final_gamma = _read(handle, f"{VISION}.post_layernorm.weight", device)
        final_beta = _read(handle, f"{VISION}.post_layernorm.bias", device)
        proj_w = _read(handle, f"{PROJECTOR}.weight", device)
        proj_b = _read(handle, f"{PROJECTOR}.bias", device)

    x = position if include_head_tail else seeded(seq, C.vision_width, seed=seed)
    gamma = torch.zeros(1, 1, 2 * layers + 1, C.vision_width, device=device, dtype=x.dtype)
    beta = torch.zeros_like(gamma)
    for layer, weight in enumerate(weights):
        gamma[0, 0, 2 * layer] = weight["g1"]
        beta[0, 0, 2 * layer] = weight["b1"]
        gamma[0, 0, 2 * layer + 1] = weight["g2"]
        beta[0, 0, 2 * layer + 1] = weight["b2"]
    gamma[0, 0, -1] = final_gamma
    beta[0, 0, -1] = final_beta

    globs = SiglipGlobals(
        vision_x=x.reshape(1, 1, seq, C.vision_width).clone(),
        siglip_x_normed=torch.zeros(1, 1, seq, C.vision_width, device=device, dtype=x.dtype),
        qkv_buf=torch.zeros(1, 1, seq, 3 * C.vision_num_heads * C.siglip_head_dim_pad, device=device, dtype=x.dtype),
        attn_out=torch.zeros(1, 1, seq, C.vision_num_heads * C.siglip_head_dim_pad, device=device, dtype=x.dtype),
        fc1_buf=torch.zeros(1, 1, seq, C.vision_mlp_width, device=device, dtype=x.dtype),
        ln_gamma=gamma,
        ln_beta=beta,
        patch_in=seeded(seq, 3 * 14 * 14, seed=seed + 1).reshape(
            1, 1, seq, 3 * 14 * 14
        ),
        patch_w=patch_w.unsqueeze(0),
        patch_b=patch_b.reshape(1, 1, -1),
        proj_w=proj_w.unsqueeze(0),
        proj_b=proj_b.reshape(1, 1, -1),
        proj_out=torch.zeros(1, 1, seq, C.encoder_width, device=device, dtype=x.dtype),
        ln_attn_w=torch.stack([weight["g1"] for weight in weights]).unsqueeze(0),
        ln_attn_b=torch.stack([weight["b1"] for weight in weights]).unsqueeze(0),
        ln_ffn_w=torch.stack([weight["g2"] for weight in weights]).unsqueeze(0),
        ln_ffn_b=torch.stack([weight["b2"] for weight in weights]).unsqueeze(0),
        qkv_w=torch.stack([weight["Wqkv"] for weight in weights]).unsqueeze(0),
        qkv_b=torch.stack([weight["bqkv"] for weight in weights]).unsqueeze(0),
        oproj_w=torch.stack([weight["Wo"] for weight in weights]).unsqueeze(0),
        oproj_b=torch.stack([weight["bo"] for weight in weights]).unsqueeze(0),
        fc1_w=torch.stack([weight["W1"] for weight in weights]).unsqueeze(0),
        fc1_b=torch.stack([weight["b1f"] for weight in weights]).unsqueeze(0),
        fc2_w=torch.stack([weight["W2"] for weight in weights]).unsqueeze(0),
        fc2_b=torch.stack([weight["b2f"] for weight in weights]).unsqueeze(0),
        barriers=torch.zeros(1, layers, C.siglip_barrier_rows, C.barrier_slots, device=device, dtype=torch.int32),
    )
    globs._pyvm_meta = python_vm.PyvmMeta(num_views=views)
    return globs, weights


def _copy_tensors(tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: tensor.clone() for name, tensor in tensors.items()}


def make_gemma_globs(
    pack: Pi0H100TensorPack,
    *,
    layers: int,
    seed: int = 200,
) -> GemmaGlobals:
    tensors = _copy_tensors(pack.gemma.tensors)
    seq = C.encoder_seq_for(pack.num_views, pack.prompt_len)
    rope = gemma_ops.make_rope_table(
        tensors["rope"].shape[-2], device=tensors["rope"].device
    )
    tensors["rope"].zero_()
    tensors["rope"][0, 0].copy_(rope)
    tensors["encoder_x"].zero_()
    tensors["encoder_x"][0, 0, :seq] = seeded(seq, C.encoder_width, seed=seed)
    globs = GemmaGlobals(
        encoder_x=tensors["encoder_x"],
        qkv_buf=tensors["qkv_buf"],
        attn_out=tensors["attn_out"],
        gate_buf=tensors["gate_buf"],
        up_buf=tensors["up_buf"],
        rms_factor=tensors["rms_factor"],
        k_cache=tensors["k_cache"],
        v_cache=tensors["v_cache"],
        rope=tensors["rope"],
        qkv_w=tensors["qkv_w"][:layers],
        oproj_w=tensors["oproj_w"][:layers],
        gate_w=tensors["gate_w"][:layers],
        up_w=tensors["up_w"][:layers],
        down_w=tensors["down_w"][:layers],
        barriers=torch.zeros(1, layers, C.gemma_barrier_rows, C.barrier_slots, device=tensors["encoder_x"].device, dtype=torch.int32),
    )
    globs._pyvm_meta = python_vm.PyvmMeta(seq_len=seq, num_views=pack.num_views)
    return globs


def prepare_gemma_native(pack: Pi0H100TensorPack, *, seed: int = 200) -> None:
    tensors = pack.gemma.tensors
    seq = C.encoder_seq_for(pack.num_views, pack.prompt_len)
    for name in pack.gemma.scratch:
        tensors[name].zero_()
    tensors["encoder_x"].zero_()
    tensors["encoder_x"][0, 0, :seq] = seeded(seq, C.encoder_width, seed=seed)
    tensors["rope"].zero_()
    tensors["rope"][0, 0].copy_(
        gemma_ops.make_rope_table(tensors["rope"].shape[-2], device=tensors["rope"].device)
    )


def gemma_weights(globs: GemmaGlobals, layers: int) -> list[tuple[torch.Tensor, ...]]:
    return [
        (
            globs.qkv_w[layer],
            globs.oproj_w[layer],
            globs.gate_w[layer],
            globs.up_w[layer],
            globs.down_w[layer],
        )
        for layer in range(layers)
    ]


def make_ae_globs(
    pack: Pi0H100TensorPack,
    *,
    layers: int,
    seed: int = 300,
) -> AeGlobals:
    tensors = _copy_tensors(pack.ae.tensors)
    layout = pack.ae_layout
    tensors["x_acc"].zero_()
    tensors["x_acc"][0, 0, : layout["seq"]] = seeded(
        layout["seq"], C.decoder_width, seed=seed, dtype=torch.float32
    )
    tensors["rope"].zero_()
    full_rope = gemma_ops.make_rope_table(
        layout["prefix_len"] + layout["m_pad"], device=tensors["rope"].device
    )
    tensors["rope"][0, 0].copy_(full_rope[layout["prefix_len"] :])
    prefix_k = seeded(layout["prefix_len"], C.head_dim, seed=seed + 1)
    prefix_v = seeded(layout["prefix_len"], C.head_dim, seed=seed + 2)
    tensors["k_cache"].zero_()
    tensors["v_cache"].zero_()
    tensors["k_cache"][0, :layers, : layout["prefix_len"]] = prefix_k
    tensors["v_cache"][0, :layers, : layout["prefix_len"]] = prefix_v
    globs = AeGlobals(
        x=tensors["x"],
        x_acc=tensors["x_acc"],
        qkv_buf=tensors["qkv_buf"],
        attn_out=tensors["attn_out"],
        gate_buf=tensors["gate_buf"],
        ffn_buf=tensors["ffn_buf"],
        rms_factor=tensors["rms_factor"],
        k_cache=tensors["k_cache"],
        v_cache=tensors["v_cache"],
        rope=tensors["rope"],
        qkv_w=tensors["qkv_w"],
        oproj_w=tensors["oproj_w"],
        gate_w=tensors["gate_w"],
        up_w=tensors["up_w"],
        down_w=tensors["down_w"],
        noise=tensors["noise"],
        act_h=tensors["act_h"],
        act_w_in=tensors["act_w_in"],
        act_w_mlp=tensors["act_w_mlp"],
        act_w_out=tensors["act_w_out"],
        act_bias=tensors["act_bias"],
        barriers=torch.zeros(1, layers, C.ae_barrier_rows, C.barrier_slots, device=tensors["x"].device, dtype=torch.int32),
    )
    globs._pyvm_meta = python_vm.PyvmMeta(layout=layout)
    return globs


def prepare_ae_native(pack: Pi0H100TensorPack, *, seed: int = 300) -> None:
    tensors = pack.ae.tensors
    layout = pack.ae_layout
    for name in pack.ae.scratch:
        tensors[name].zero_()
    tensors["x_acc"].zero_()
    tensors["x_acc"][0, 0, : layout["seq"]] = seeded(
        layout["seq"], C.decoder_width, seed=seed, dtype=torch.float32
    )
    tensors["rope"].zero_()
    full_rope = gemma_ops.make_rope_table(
        layout["prefix_len"] + layout["m_pad"], device=tensors["rope"].device
    )
    tensors["rope"][0, 0].copy_(full_rope[layout["prefix_len"] :])
    tensors["k_cache"].zero_()
    tensors["v_cache"].zero_()
    prefix_k = seeded(layout["prefix_len"], C.head_dim, seed=seed + 1)
    prefix_v = seeded(layout["prefix_len"], C.head_dim, seed=seed + 2)
    tensors["k_cache"][0, :, : layout["prefix_len"]] = prefix_k
    tensors["v_cache"][0, :, : layout["prefix_len"]] = prefix_v
    materialize_ae_time_biases(pack)


def ae_weights(globs: AeGlobals, layers: int) -> dict[str, list[torch.Tensor]]:
    weights: dict[str, list[torch.Tensor]] = {
        "qkv": [], "oproj": [], "gate": [], "up": [], "down": []
    }
    for layer in range(layers):
        weights["qkv"].append(globs.qkv_w[0, layer])
        weights["gate"].append(globs.gate_w[0, layer])
        weights["up"].append(globs.up_w[0, layer])
        weights["oproj"].append(
            globs.oproj_w[0, layer * 8 : (layer + 1) * 8]
            .permute(1, 0, 2).reshape(C.decoder_width, -1)[:, : C.encoder_width]
        )
        weights["down"].append(
            globs.down_w[0, layer * 16 : (layer + 1) * 16]
            .permute(1, 0, 2).reshape(C.decoder_width, -1)[:, : C.decoder_mlp_width]
        )
    return weights


def ae_prefix(globs: AeGlobals, layers: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    layout = globs._pyvm_meta.layout
    return [
        (
            globs.k_cache[0, layer, : layout["prefix_len"]].clone(),
            globs.v_cache[0, layer, : layout["prefix_len"]].clone(),
        )
        for layer in range(layers)
    ]


def prepare_siglip_native(
    pack: Pi0H100TensorPack,
    *,
    seed: int,
    include_head_tail: bool,
) -> None:
    tensors = pack.siglip.tensors
    for name in pack.siglip.scratch:
        tensors[name].zero_()
    if include_head_tail:
        position = pack.runtime_aux["siglip_position_embedding"].repeat(pack.num_views, 1)
        tensors["vision_x"].zero_()
        tensors["vision_x"][0, 0, :, : C.vision_width] = position
        tensors["patch_in"].zero_()
        tensors["patch_in"][0, 0, :, : 3 * 14 * 14] = seeded(
            C.siglip_seq_for(pack.num_views), 3 * 14 * 14, seed=seed + 1
        )
    else:
        tensors["vision_x"].zero_()
        tensors["vision_x"][0, 0, :, : C.vision_width] = seeded(
            C.siglip_seq_for(pack.num_views), C.vision_width, seed=seed
        )


def instruction_tensor(instructions: list[Any], *, device: torch.device) -> torch.Tensor:
    queues, _ = scheduler.pack_instructions(instructions, sm_count=C.sm_count)
    return torch.tensor(queues, dtype=torch.int32, device=device).unsqueeze(0).contiguous()


def launch_native(mk: Any, tensor_set: Any, instructions: list[Any]) -> None:
    device = tensor_set.abi_arguments()[0].device
    encoded = instruction_tensor(instructions, device=device)
    mk(*tensor_set.abi_arguments(), encoded, stream=torch.cuda.current_stream(device))
    torch.cuda.synchronize(device)


def run_ae_fused(globs: AeGlobals, *, layers: int) -> torch.Tensor:
    layout = globs._pyvm_meta.layout
    return ae_ops.fused_ae_stack(
        globs.x_acc[0, 0].float(),
        ae_weights(globs, layers),
        globs.rope[0, 0],
        ae_prefix(globs, layers),
        layout=layout,
        n_layers=layers,
    )


def run_siglip_fused(
    x: torch.Tensor,
    weights: list[dict[str, torch.Tensor]],
    *,
    num_views: int,
) -> torch.Tensor:
    """Run fused SigLIP math with the same BF16 landings as its ABI buffers."""
    hidden = x.to(torch.bfloat16)
    for weight in weights:
        normed = siglip_ops.layernorm(hidden, weight["g1"], weight["b1"])
        qkv = (
            normed.float() @ weight["Wqkv"].float().T + weight["bqkv"].float()
        ).to(torch.bfloat16)
        attn = siglip_ops.fused_siglip_attention(
            qkv,
            num_views=num_views,
            tokens_per_view=C.vision_tokens_per_view,
            num_heads=C.vision_num_heads,
            head_dim=C.siglip_head_dim,
        ).to(torch.bfloat16)
        hidden = (
            hidden.float()
            + attn.float() @ weight["Wo"].float().T
            + weight["bo"].float()
        ).to(torch.bfloat16)
        normed = siglip_ops.layernorm(hidden, weight["g2"], weight["b2"])
        mlp = torch.nn.functional.gelu(
            normed.float() @ weight["W1"].float().T + weight["b1f"].float(),
            approximate="tanh",
        ).to(torch.bfloat16)
        hidden = (
            hidden.float()
            + mlp.float() @ weight["W2"].float().T
            + weight["b2f"].float()
        ).to(torch.bfloat16)
    return hidden


def run_gemma_fused(
    x: torch.Tensor,
    weights: list[tuple[torch.Tensor, ...]],
    rope: torch.Tensor,
    *,
    seq_len: int,
) -> torch.Tensor:
    """Run fused Gemma math with the same BF16 landings as its ABI buffers."""
    hidden = x.to(torch.bfloat16)
    for wqkv, wo, wg, wu, wd in weights:
        q, k, v = gemma_ops.fused_qkv_rope(hidden, wqkv, rope)
        attn = gemma_ops.fused_mqa_bidirectional(
            q.to(torch.bfloat16),
            k.to(torch.bfloat16),
            v.to(torch.bfloat16),
            seq_len,
        ).to(torch.bfloat16)
        hidden = gemma_ops.fused_oproj_residual(hidden, attn, wo).to(torch.bfloat16)
        hidden = gemma_ops.fused_ffn_gate_up_down(hidden, wg, wu, wd).to(
            torch.bfloat16
        )
    return hidden


__all__ = [
    "C",
    "LOCAL_CHECKPOINT_AVAILABLE",
    "ae_prefix",
    "ae_weights",
    "gemma_weights",
    "instruction_tensor",
    "launch_native",
    "make_ae_globs",
    "make_gemma_globs",
    "make_pack",
    "make_siglip_globs",
    "prepare_ae_native",
    "prepare_gemma_native",
    "prepare_siglip_native",
    "require_checkpoint",
    "require_h100",
    "run_ae_fused",
    "run_gemma_fused",
    "run_siglip_fused",
    "seeded",
]
