"""Python VM handlers for pi0 H100 instructions (tiled path, LLM-style).

Handlers execute instructions against Globals. Phase-coalescing: the first
instruction of each (layer, opcode) group runs the matching fused reference;
subsequent tiles only bump barriers so fused_to_pyvm matches fused full-ops.

The opcode 8--10 tail handlers lock local math and barrier semantics.  The
existing opcode 1--7 VM uses verifier-oriented dense layouts, so a full SigLIP
VM run is not the checkpoint-pack production reference and must not be used as
a production fallback.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch

from qimeng_megakernel.models.instruction import Instruction, run_pyvm_sequence

from ..fused import ae_ops, gemma_ops, siglip_ops
from ..globs import AeGlobals, GemmaGlobals, SiglipGlobals
from ..pi0_config import Pi0H100Config as C
from . import instruction_defs as I


_AE_INSTRUCTION_TYPES = (
    I.AeRmsnAttn,
    I.AeRmsnFfn,
    I.AeQkvRope,
    I.AeAttnPartial,
    I.AeAttnReduce,
    I.AeOproj,
    I.AeGate,
    I.AeUpMul,
    I.AeDown,
    I.AeFfnDual,
    I.AeActIn,
    I.AeActMlp,
    I.AeRmsnOut,
    I.AeActOut,
)
_AE_ACT_STATE_ROW = 12
_SIGLIP_INSTRUCTION_TYPES = (
    I.SiglipLnAttn,
    I.SiglipLnFfn,
    I.SiglipQkv,
    I.SiglipAttn,
    I.SiglipOproj,
    I.SiglipFc1,
    I.SiglipFc2,
    I.SiglipPatch,
    I.SiglipFinalNorm,
    I.SiglipProjector,
)
_GEMMA_INSTRUCTION_TYPES = (
    I.GemmaRmsnAttn,
    I.GemmaRmsnFfn,
    I.GemmaQkvRope,
    I.GemmaAttnPartial,
    I.GemmaAttnReduce,
    I.GemmaOproj,
    I.GemmaGate,
    I.GemmaUpMul,
    I.GemmaDown,
    I.GemmaDownSplitK,
)


@dataclass
class PyvmMeta:
    """Runtime context attached to globs for fused_to_pyvm (seq_len, layout, …)."""

    seq_len: int = C.encoder_seq
    num_views: int = C.num_views
    layout: dict[str, int] | None = None


def _meta(globs: Any) -> PyvmMeta:
    m = getattr(globs, "_pyvm_meta", None)
    if m is None:
        m = PyvmMeta()
        globs._pyvm_meta = m
    return m


def _reset_pyvm_state(globs: Any) -> None:
    globs._pyvm_ran = set()


def _bump_barrier(globs: Any, layer: int, row: int, slot: int = 0, delta: int = 1) -> None:
    layer = min(layer, globs.barriers.shape[1] - 1)
    globs.barriers[0, layer, row, slot] += delta


def _check_dep(globs: Any, inst: Instruction) -> None:
    dep_layer = min(getattr(inst, "dep_layer", 0), globs.barriers.shape[1] - 1)
    dep_row = getattr(inst, "dep_row", 0)
    dep_expected = getattr(inst, "dep_expected", 0)
    if dep_expected > 0:
        got = int(globs.barriers[0, dep_layer, dep_row, 0].item())
        assert got >= dep_expected, (
            f"barrier [{dep_layer},{dep_row}]={got} < dep_expected={dep_expected}"
        )


def _dependency_identity(inst: Instruction) -> tuple[int, int, int]:
    """Stable phase epoch carried by the instruction's cumulative dependency."""
    return (
        int(getattr(inst, "dep_layer", 0)),
        int(getattr(inst, "dep_row", 0)),
        int(getattr(inst, "dep_expected", 0)),
    )


def _instruction_barrier_row(inst: Instruction) -> int:
    if isinstance(inst, _AE_INSTRUCTION_TYPES):
        return I.ae_barrier_row(inst.opcode())
    if isinstance(inst, _SIGLIP_INSTRUCTION_TYPES):
        return I.siglip_barrier_row(inst.opcode())
    if isinstance(inst, _GEMMA_INSTRUCTION_TYPES):
        return I.gemma_barrier_row(inst.opcode())
    raise TypeError(f"unsupported Pi0/H100 instruction type: {type(inst).__name__}")


def _run_once(globs: Any, key: tuple, fn: Callable[[], None], inst: Instruction) -> None:
    _check_dep(globs, inst)
    ran: set[tuple] = globs._pyvm_ran
    epoch_key = (*key, ("dep", *_dependency_identity(inst)))
    if epoch_key not in ran:
        fn()
        ran.add(epoch_key)
    row = _instruction_barrier_row(inst)
    layer = getattr(inst, "layer_idx", 0)
    _bump_barrier(globs, layer, row)


# ── SigLIP ──────────────────────────────────────────────────────────────────


def _siglip_attn_valid_idx(device: torch.device | str | None = None) -> torch.Tensor:
    heads, hd, hd_pad = C.vision_num_heads, C.siglip_head_dim, C.siglip_head_dim_pad
    return torch.cat(
        [torch.arange(h * hd_pad, h * hd_pad + hd, device=device) for h in range(heads)]
    )


def _pad_qkv_dense(qkv_dense: torch.Tensor) -> torch.Tensor:
    """Dense ``[S, 3*H*D]`` -> head-padded ``[S, 3*H*D_pad]`` for attn kernels."""
    heads, hd, hd_pad = C.vision_num_heads, C.siglip_head_dim, C.siglip_head_dim_pad
    seq = qkv_dense.shape[0]
    out = torch.zeros(seq, 3 * heads * hd_pad, device=qkv_dense.device, dtype=qkv_dense.dtype)
    for sec in range(3):
        for h in range(heads):
            o_dense = sec * heads * hd + h * hd
            o_pad = sec * heads * hd_pad + h * hd_pad
            out[:, o_pad : o_pad + hd] = qkv_dense[:, o_dense : o_dense + hd]
    return out


def _handle_siglip_ln_attn(globs: SiglipGlobals, inst: I.SiglipLnAttn) -> None:
    def _run():
        assert globs.vision_x is not None and globs.ln_attn_w is not None
        x = globs.vision_x[0, 0]
        gamma, beta = globs.ln_attn_w[0, inst.layer_idx], globs.ln_attn_b[0, inst.layer_idx]
        normed = siglip_ops.layernorm(x, gamma, beta)
        if globs.qkv_buf is not None:
            globs.qkv_buf[0, 0, :, : x.shape[-1]] = normed.to(globs.qkv_buf.dtype)

    _run_once(globs, ("siglip", inst.layer_idx, "ln_attn"), _run, inst)


def _handle_siglip_qkv(globs: SiglipGlobals, inst: I.SiglipQkv) -> None:
    def _run():
        assert globs.qkv_buf is not None and globs.qkv_w is not None
        if globs.qkv_buf.abs().sum() > 0:
            x = globs.qkv_buf[0, 0, :, : C.vision_width]
        else:
            x = globs.vision_x[0, 0, :, : C.vision_width]
        w, b = globs.qkv_w[0, inst.layer_idx], globs.qkv_b[0, inst.layer_idx]
        y = x.float() @ w.float().T + b.float()
        padded = _pad_qkv_dense(y.to(globs.qkv_buf.dtype))
        globs.qkv_buf[0, 0, :, : padded.shape[-1]] = padded

    _run_once(globs, ("siglip", inst.layer_idx, "qkv"), _run, inst)


def _handle_siglip_attn(globs: SiglipGlobals, inst: I.SiglipAttn) -> None:
    meta = _meta(globs)

    def _run():
        assert globs.qkv_buf is not None and globs.attn_out is not None
        qkv = globs.qkv_buf[0, 0]
        out = siglip_ops.fused_siglip_attention_head_padded(qkv, num_views=meta.num_views)
        globs.attn_out[0, 0, :, : out.shape[-1]] = out.to(globs.attn_out.dtype)

    _run_once(globs, ("siglip", inst.layer_idx, "attn"), _run, inst)


def _handle_siglip_oproj(globs: SiglipGlobals, inst: I.SiglipOproj) -> None:
    def _run():
        assert globs.attn_out is not None and globs.vision_x is not None
        x = globs.vision_x[0, 0].float()
        valid_idx = _siglip_attn_valid_idx(globs.attn_out.device)
        attn = globs.attn_out[0, 0, :, valid_idx].float()
        wo, bo = globs.oproj_w[0, inst.layer_idx], globs.oproj_b[0, inst.layer_idx]
        y = x + attn @ wo.float().T + bo.float()
        globs.vision_x[0, 0, :, : C.vision_width] = y.to(globs.vision_x.dtype)

    _run_once(globs, ("siglip", inst.layer_idx, "oproj"), _run, inst)


def _handle_siglip_fc1(globs: SiglipGlobals, inst: I.SiglipFc1) -> None:
    def _run():
        assert globs.vision_x is not None and globs.fc1_buf is not None
        assert globs.ln_ffn_w is not None and globs.ln_ffn_b is not None
        x = globs.vision_x[0, 0, :, : C.vision_width]
        gamma, beta = globs.ln_ffn_w[0, inst.layer_idx], globs.ln_ffn_b[0, inst.layer_idx]
        x = siglip_ops.layernorm(x, gamma, beta)
        w, b = globs.fc1_w[0, inst.layer_idx], globs.fc1_b[0, inst.layer_idx]
        h = torch.nn.functional.gelu(x.float() @ w.float().T + b.float(), approximate="tanh")
        globs.fc1_buf[0, 0, :, : h.shape[-1]] = h.to(globs.fc1_buf.dtype)

    _run_once(globs, ("siglip", inst.layer_idx, "fc1"), _run, inst)


def _handle_siglip_fc2(globs: SiglipGlobals, inst: I.SiglipFc2) -> None:
    def _run():
        assert globs.vision_x is not None and globs.fc1_buf is not None
        x = globs.vision_x[0, 0, :, : C.vision_width].float()
        h = globs.fc1_buf[0, 0, :, : C.vision_mlp_width].float()
        w, b = globs.fc2_w[0, inst.layer_idx], globs.fc2_b[0, inst.layer_idx]
        y = x + h @ w.float().T + b.float()
        globs.vision_x[0, 0, :, : C.vision_width] = y.to(globs.vision_x.dtype)

    _run_once(globs, ("siglip", inst.layer_idx, "fc2"), _run, inst)


def _packed_bias(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.reshape(-1)


def _handle_siglip_patch(globs: SiglipGlobals, inst: I.SiglipPatch) -> None:
    def _run():
        assert globs.patch_in is not None and globs.patch_w is not None
        assert globs.patch_b is not None and globs.vision_x is not None
        patch_rows = globs.patch_in[0, 0]
        position = globs.vision_x[0, 0]
        output = siglip_ops.patch_linear_residual(
            patch_rows,
            globs.patch_w[0],
            _packed_bias(globs.patch_b[0, 0]),
            position,
        )
        globs.vision_x[0, 0].copy_(output.to(globs.vision_x.dtype))

    _run_once(globs, ("siglip", inst.layer_idx, "patch"), _run, inst)


def _handle_siglip_final_norm(globs: SiglipGlobals, inst: I.SiglipFinalNorm) -> None:
    def _run():
        assert globs.vision_x is not None and globs.siglip_x_normed is not None
        assert globs.ln_gamma is not None and globs.ln_beta is not None
        x = globs.vision_x[0, 0, :, : C.vision_width]
        gamma = globs.ln_gamma[0, 0, inst.weight_row]
        beta = globs.ln_beta[0, 0, inst.weight_row]
        output = siglip_ops.final_norm(x, gamma, beta)
        globs.siglip_x_normed.zero_()
        globs.siglip_x_normed[0, 0, :, : C.vision_width] = output.to(
            globs.siglip_x_normed.dtype
        )

    _run_once(globs, ("siglip", inst.layer_idx, "final_norm"), _run, inst)


def _handle_siglip_projector(globs: SiglipGlobals, inst: I.SiglipProjector) -> None:
    def _run():
        assert globs.siglip_x_normed is not None and globs.proj_w is not None
        assert globs.proj_b is not None and globs.proj_out is not None
        output = siglip_ops.projector(
            globs.siglip_x_normed[0, 0],
            globs.proj_w[0],
            _packed_bias(globs.proj_b[0, 0]),
        )
        globs.proj_out[0, 0].copy_(output.to(globs.proj_out.dtype))

    _run_once(globs, ("siglip", inst.layer_idx, "projector"), _run, inst)


def _nyi_siglip(name: str):
    def _handler(globs: SiglipGlobals, instruction: Instruction) -> None:
        _run_once(globs, ("siglip", "nyi", name), lambda: None, instruction)

    return _handler


SIGLIP_HANDLERS: dict[type[Instruction], Callable] = {
    I.SiglipLnAttn: _handle_siglip_ln_attn,
    I.SiglipLnFfn: _nyi_siglip("SiglipLnFfn"),
    I.SiglipQkv: _handle_siglip_qkv,
    I.SiglipAttn: _handle_siglip_attn,
    I.SiglipOproj: _handle_siglip_oproj,
    I.SiglipFc1: _handle_siglip_fc1,
    I.SiglipFc2: _handle_siglip_fc2,
    I.SiglipPatch: _handle_siglip_patch,
    I.SiglipFinalNorm: _handle_siglip_final_norm,
    I.SiglipProjector: _handle_siglip_projector,
}


# ── Gemma ─────────────────────────────────────────────────────────────────


def _handle_gemma_rmsn_attn(globs: GemmaGlobals, inst: I.GemmaRmsnAttn) -> None:
    def _run():
        assert globs.encoder_x is not None and globs.rms_factor is not None
        x = globs.encoder_x[0, 0, : _meta(globs).seq_len]
        inv = torch.rsqrt((x.float() ** 2).mean(-1) + C.rms_eps)
        globs.rms_factor[0, 0, 0, : _meta(globs).seq_len] = inv

    _run_once(globs, ("gemma", inst.layer_idx, "rmsn_attn"), _run, inst)


def _handle_gemma_rmsn_ffn(globs: GemmaGlobals, inst: I.GemmaRmsnFfn) -> None:
    def _run():
        assert globs.encoder_x is not None and globs.rms_factor is not None
        x = globs.encoder_x[0, 0, : _meta(globs).seq_len]
        inv = torch.rsqrt((x.float() ** 2).mean(-1) + C.rms_eps)
        globs.rms_factor[0, 0, 1, : _meta(globs).seq_len] = inv

    _run_once(globs, ("gemma", inst.layer_idx, "rmsn_ffn"), _run, inst)


def _handle_gemma_qkv_rope(globs: GemmaGlobals, inst: I.GemmaQkvRope) -> None:
    meta = _meta(globs)

    def _run():
        assert globs.encoder_x is not None and globs.qkv_w is not None
        x = globs.encoder_x[0, 0, : meta.seq_len]
        w = globs.qkv_w[inst.layer_idx]
        rope = globs.rope[0, 0, : meta.seq_len]
        q, k, v = gemma_ops.fused_qkv_rope(x, w, rope)
        y = torch.cat([q, k, v], dim=-1)
        globs.qkv_buf[0, 0, : meta.seq_len, : y.shape[-1]] = y.to(globs.qkv_buf.dtype)
        globs.k_cache[inst.layer_idx, : meta.seq_len] = k.to(globs.k_cache.dtype)
        globs.v_cache[inst.layer_idx, : meta.seq_len] = v.to(globs.v_cache.dtype)

    _run_once(globs, ("gemma", inst.layer_idx, "qkv"), _run, inst)


def _handle_gemma_attn_partial(globs: GemmaGlobals, inst: I.GemmaAttnPartial) -> None:
    _run_once(
        globs,
        ("gemma", inst.layer_idx, "attn_p", inst.partial_idx, inst.head_idx),
        lambda: None,
        inst,
    )


def _handle_gemma_attn_reduce(globs: GemmaGlobals, inst: I.GemmaAttnReduce) -> None:
    meta = _meta(globs)

    def _run():
        assert globs.qkv_buf is not None and globs.k_cache is not None
        q = globs.qkv_buf[0, 0, : meta.seq_len, : C.num_heads * C.head_dim]
        k = globs.k_cache[inst.layer_idx, : meta.seq_len]
        v = globs.v_cache[inst.layer_idx, : meta.seq_len]
        a = gemma_ops.fused_mqa_bidirectional(q, k, v, meta.seq_len)
        globs.attn_out[0, 0, : meta.seq_len, : a.shape[-1]] = a.to(globs.attn_out.dtype)

    _run_once(globs, ("gemma", inst.layer_idx, "attn_r"), _run, inst)


def _handle_gemma_oproj(globs: GemmaGlobals, inst: I.GemmaOproj) -> None:
    meta = _meta(globs)

    def _run():
        x = globs.encoder_x[0, 0, : meta.seq_len]
        attn = globs.attn_out[0, 0, : meta.seq_len]
        wo = globs.oproj_w[inst.layer_idx]
        y = gemma_ops.fused_oproj_residual(x, attn, wo)
        globs.encoder_x[0, 0, : meta.seq_len] = y.to(globs.encoder_x.dtype)

    _run_once(globs, ("gemma", inst.layer_idx, "oproj"), _run, inst)


def _handle_gemma_gate(globs: GemmaGlobals, inst: I.GemmaGate) -> None:
    meta = _meta(globs)

    def _run():
        x = globs.encoder_x[0, 0, : meta.seq_len]
        wg, wu, wd = (
            globs.gate_w[inst.layer_idx],
            globs.up_w[inst.layer_idx],
            globs.down_w[inst.layer_idx],
        )
        y = gemma_ops.fused_ffn_gate_up_down(x, wg, wu, wd)
        globs.encoder_x[0, 0, : meta.seq_len] = y.to(globs.encoder_x.dtype)

    _run_once(globs, ("gemma", inst.layer_idx, "ffn"), _run, inst)


def _handle_gemma_down(globs: GemmaGlobals, inst: I.GemmaDown) -> None:
    _run_once(globs, ("gemma", inst.layer_idx, "down"), lambda: None, inst)


def _handle_gemma_down_splitk(globs: GemmaGlobals, inst: I.GemmaDownSplitK) -> None:
    _handle_gemma_down(globs, inst)  # type: ignore[arg-type]


GEMMA_HANDLERS: dict[type[Instruction], Callable] = {
    I.GemmaRmsnAttn: _handle_gemma_rmsn_attn,
    I.GemmaRmsnFfn: _handle_gemma_rmsn_ffn,
    I.GemmaQkvRope: _handle_gemma_qkv_rope,
    I.GemmaAttnPartial: _handle_gemma_attn_partial,
    I.GemmaAttnReduce: _handle_gemma_attn_reduce,
    I.GemmaOproj: _handle_gemma_oproj,
    I.GemmaGate: _handle_gemma_gate,
    I.GemmaUpMul: _handle_gemma_gate,
    I.GemmaDown: _handle_gemma_down,
    I.GemmaDownSplitK: _handle_gemma_down_splitk,
}


# ── AE ──────────────────────────────────────────────────────────────────────


def _ae_layout(globs: AeGlobals) -> dict[str, int]:
    layout = _meta(globs).layout
    return layout if layout is not None else C.ae_kv_layout_for()


def _to_ksplit_major(a: torch.Tensor, splits: int, k_seg: int = 256) -> torch.Tensor:
    m_pad = a.shape[0]
    out = torch.zeros(splits * m_pad, k_seg, device=a.device, dtype=a.dtype)
    for s in range(splits):
        out[s * m_pad : (s + 1) * m_pad] = a[:, s * k_seg : (s + 1) * k_seg]
    return out


def _from_ksplit_major(p: torch.Tensor, splits: int, k_seg: int = 256) -> torch.Tensor:
    m_pad = p.shape[0] // splits
    out = torch.zeros(m_pad, splits * k_seg, device=p.device, dtype=torch.float32)
    for s in range(splits):
        out[:, s * k_seg : (s + 1) * k_seg] = p[s * m_pad : (s + 1) * m_pad].float()
    return out


def _handle_ae_rmsn_attn(globs: AeGlobals, inst: I.AeRmsnAttn) -> None:
    layout = _ae_layout(globs)

    def _run():
        x = globs.x_acc[0, 0, : layout["m_pad"]]
        inv = torch.rsqrt((x.float() ** 2).mean(-1) + C.rms_eps)
        globs.rms_factor[0, 0, 0, : layout["m_pad"]] = inv
        globs.x[0, 0, : layout["m_pad"]] = x.bfloat16()

    _run_once(globs, ("ae", inst.layer_idx, "rmsn_attn"), _run, inst)


def _handle_ae_rmsn_ffn(globs: AeGlobals, inst: I.AeRmsnFfn) -> None:
    layout = _ae_layout(globs)

    def _run():
        x = globs.x_acc[0, 0, : layout["m_pad"]]
        inv = torch.rsqrt((x.float() ** 2).mean(-1) + C.rms_eps)
        globs.rms_factor[0, 0, 1, : layout["m_pad"]] = inv
        globs.x[0, 0, : layout["m_pad"]] = x.bfloat16()

    _run_once(globs, ("ae", inst.layer_idx, "rmsn_ffn"), _run, inst)


def _handle_ae_qkv_rope(globs: AeGlobals, inst: I.AeQkvRope) -> None:
    layout = _ae_layout(globs)
    row0 = layout["state_key"]

    def _run():
        x = globs.x[0, 0, : layout["m_pad"]]
        w = globs.qkv_w[0, inst.layer_idx]
        rope = globs.rope[0, 0, : layout["m_pad"]]
        inv = globs.rms_factor[0, 0, 0, : layout["m_pad"]]
        y = inv[:, None] * (x.float() @ w.float().T)
        q_width = C.num_heads * C.head_dim
        q = gemma_ops.apply_rope_interleaved(y[:, :q_width], rope)
        k = gemma_ops.apply_rope_interleaved(
            y[:, q_width : q_width + C.head_dim], rope
        )
        v = y[:, q_width + C.head_dim :]
        y = torch.cat([q, k, v], dim=-1)
        globs.qkv_buf[0, 0, : layout["m_pad"], : y.shape[-1]] = y.to(globs.qkv_buf.dtype)
        globs.k_cache[0, inst.layer_idx, row0 : row0 + layout["m_pad"]] = k.to(
            globs.k_cache.dtype
        )
        globs.v_cache[0, inst.layer_idx, row0 : row0 + layout["m_pad"]] = v.to(
            globs.v_cache.dtype
        )

    _run_once(globs, ("ae", inst.layer_idx, "qkv"), _run, inst)


def _handle_ae_attn_partial(globs: AeGlobals, inst: I.AeAttnPartial) -> None:
    _run_once(
        globs,
        ("ae", inst.layer_idx, "attn_p", inst.partial_idx, inst.head_idx),
        lambda: None,
        inst,
    )


def _handle_ae_attn_reduce(globs: AeGlobals, inst: I.AeAttnReduce) -> None:
    layout = _ae_layout(globs)

    def _run():
        q = globs.qkv_buf[0, 0, : layout["m_pad"], : C.num_heads * C.head_dim]
        kc = globs.k_cache[0, inst.layer_idx]
        vc = globs.v_cache[0, inst.layer_idx]
        a = ae_ops.fused_ae_attn(
            q,
            kc,
            vc,
            prefix_len=layout["prefix_len"],
            state_key=layout["state_key"],
            self_end=layout["self_end"],
            m_pad=layout["m_pad"],
        )
        flat = a.reshape(layout["m_pad"], -1).bfloat16()
        for h in range(C.num_heads):
            globs.attn_out[0, 0, h * layout["m_pad"] : (h + 1) * layout["m_pad"], : C.head_dim] = (
                flat[:, h * C.head_dim : (h + 1) * C.head_dim].to(globs.attn_out.dtype)
            )

    _run_once(globs, ("ae", inst.layer_idx, "attn_r"), _run, inst)


def _handle_ae_oproj(globs: AeGlobals, inst: I.AeOproj) -> None:
    layout = _ae_layout(globs)
    oproj_splits = 8

    def _run():
        attn = torch.zeros(layout["m_pad"], C.num_heads * C.head_dim, device=globs.x_acc.device)
        for h in range(C.num_heads):
            attn[:, h * C.head_dim : (h + 1) * C.head_dim] = globs.attn_out[
                0, 0, h * layout["m_pad"] : (h + 1) * layout["m_pad"], : C.head_dim
            ].float()
        x_acc = globs.x_acc[0, 0, : layout["m_pad"]]
        wo_planes = globs.oproj_w[0, inst.layer_idx * oproj_splits : (inst.layer_idx + 1) * oproj_splits]
        wo = wo_planes.permute(1, 0, 2).reshape(C.decoder_width, -1)[:, : C.num_heads * C.head_dim]
        globs.x_acc[0, 0, : layout["m_pad"]] = (x_acc + attn @ wo.float().T).float()

    _run_once(globs, ("ae", inst.layer_idx, "oproj"), _run, inst)


def _handle_ae_ffn_dual(globs: AeGlobals, inst: I.AeFfnDual) -> None:
    layout = _ae_layout(globs)
    down_splits = 16

    def _run():
        x = globs.x[0, 0, : layout["m_pad"]]
        wg = globs.gate_w[0, inst.layer_idx]
        wu = globs.up_w[0, inst.layer_idx]
        inv = globs.rms_factor[0, 0, 1, : layout["m_pad"]]
        gate = torch.nn.functional.gelu(
            inv[:, None] * (x.float() @ wg.float().T), approximate="tanh"
        )
        h = gate * (inv[:, None] * (x.float() @ wu.float().T))
        m_pad = layout["m_pad"]
        globs.ffn_buf[0, 0] = _to_ksplit_major(
            h.to(globs.ffn_buf.dtype), down_splits
        )

    _run_once(globs, ("ae", inst.layer_idx, "ffn_dual"), _run, inst)


def _handle_ae_down(globs: AeGlobals, inst: I.AeDown) -> None:
    layout = _ae_layout(globs)
    down_splits = 16

    def _run():
        m_pad = layout["m_pad"]
        h = _from_ksplit_major(globs.ffn_buf[0, 0], down_splits)[
            :, : C.decoder_mlp_width
        ]
        wd_planes = globs.down_w[0, inst.layer_idx * down_splits : (inst.layer_idx + 1) * down_splits]
        wd = wd_planes.permute(1, 0, 2).reshape(C.decoder_width, -1)[:, : C.decoder_mlp_width]
        x_acc = globs.x_acc[0, 0, : layout["m_pad"]]
        globs.x_acc[0, 0, : layout["m_pad"]] = (x_acc + h @ wd.float().T).float()

    _run_once(globs, ("ae", inst.layer_idx, "down"), _run, inst)


def _handle_ae_act_in(globs: AeGlobals, inst: I.AeActIn) -> None:
    def _run():
        assert globs.noise is not None and globs.act_h is not None
        assert globs.act_w_in is not None and globs.act_bias is not None
        weight = globs.act_w_in[0, inst.plane]
        noise = globs.noise[0, 0, :, : weight.shape[-1]]
        bias = globs.act_bias[0, 0, inst.bias_row, : weight.shape[-2]]
        globs.act_h[0, 0, :, : weight.shape[-2]] = ae_ops.fused_ae_act_in(
            noise,
            weight,
            bias,
        )

    _run_once(globs, ("ae", inst.layer_idx, "act_in"), _run, inst)


def _handle_ae_act_mlp(globs: AeGlobals, inst: I.AeActMlp) -> None:
    def _run():
        assert globs.act_h is not None and globs.x_acc is not None
        assert globs.act_w_mlp is not None and globs.act_bias is not None
        weight = globs.act_w_mlp[0, inst.plane]
        act_h = globs.act_h[0, 0, :, : weight.shape[-1]]
        bias = globs.act_bias[0, 0, inst.bias_row, : weight.shape[-2]]
        state_h = globs.act_bias[0, 0, _AE_ACT_STATE_ROW, : weight.shape[-2]]
        globs.x_acc[0, 0, :, : weight.shape[-2]] = ae_ops.fused_ae_act_mlp(
            act_h,
            weight,
            bias,
            state_h,
            n_valid_row=inst.n_valid_row,
        )

    _run_once(globs, ("ae", inst.layer_idx, "act_mlp"), _run, inst)


def _handle_ae_rmsn_out(globs: AeGlobals, inst: I.AeRmsnOut) -> None:
    def _run():
        assert globs.x_acc is not None and globs.x is not None
        assert globs.rms_factor is not None
        x, inv = ae_ops.fused_ae_rmsn_landing(globs.x_acc[0, 0])
        globs.x[0, 0, :, : x.shape[-1]] = x
        globs.rms_factor[0, 0, 2, : inv.shape[0]] = inv

    _run_once(globs, ("ae", inst.layer_idx, "rmsn_out"), _run, inst)


def _handle_ae_act_out(globs: AeGlobals, inst: I.AeActOut) -> None:
    def _run():
        assert globs.noise is not None and globs.x is not None
        assert globs.act_w_out is not None and globs.act_bias is not None
        assert globs.rms_factor is not None
        weight = globs.act_w_out[0, inst.plane]
        width = weight.shape[-2]
        updated = ae_ops.fused_ae_act_out(
            globs.noise[0, 0, :, :width],
            globs.x[0, 0, :, : weight.shape[-1]],
            weight,
            globs.act_bias[0, 0, inst.bias_row, :width],
            globs.rms_factor[0, 0, inst.norm_sel],
        )
        globs.noise[0, 0, :, :width] = updated

    _run_once(globs, ("ae", inst.layer_idx, "act_out"), _run, inst)


AE_HANDLERS: dict[type[Instruction], Callable] = {
    I.AeRmsnAttn: _handle_ae_rmsn_attn,
    I.AeRmsnFfn: _handle_ae_rmsn_ffn,
    I.AeQkvRope: _handle_ae_qkv_rope,
    I.AeAttnPartial: _handle_ae_attn_partial,
    I.AeAttnReduce: _handle_ae_attn_reduce,
    I.AeOproj: _handle_ae_oproj,
    I.AeGate: _handle_ae_ffn_dual,
    I.AeUpMul: _handle_ae_ffn_dual,
    I.AeDown: _handle_ae_down,
    I.AeFfnDual: _handle_ae_ffn_dual,
    I.AeActIn: _handle_ae_act_in,
    I.AeActMlp: _handle_ae_act_mlp,
    I.AeRmsnOut: _handle_ae_rmsn_out,
    I.AeActOut: _handle_ae_act_out,
}


def run_siglip_pyvm(globs: SiglipGlobals, instructions: list[Instruction]) -> None:
    _reset_pyvm_state(globs)
    run_pyvm_sequence(globs, instructions, SIGLIP_HANDLERS)


def run_gemma_pyvm(globs: GemmaGlobals, instructions: list[Instruction]) -> None:
    _reset_pyvm_state(globs)
    run_pyvm_sequence(globs, instructions, GEMMA_HANDLERS)


def run_ae_pyvm(globs: AeGlobals, instructions: list[Instruction]) -> None:
    _reset_pyvm_state(globs)
    run_pyvm_sequence(globs, instructions, AE_HANDLERS)


def run_gemma_layer_fused_fallback(
    globs: GemmaGlobals,
    *,
    layer_idx: int,
    seq_len: int,
    wqkv,
    wo,
    wg,
    wu,
    wd,
    rope_table,
) -> None:
    """Convenience: execute one Gemma layer via fused (no tiling) into globs.encoder_x."""
    assert globs.encoder_x is not None
    x = globs.encoder_x[0, 0] if globs.encoder_x.dim() >= 3 else globs.encoder_x
    y = gemma_ops.fused_gemma_layer(x, wqkv, wo, wg, wu, wd, rope_table, seq_len)
    x.copy_(y.to(dtype=x.dtype))


INSTRUCTION_TO_HANDLER: dict[type[Instruction], Callable] = {
    **SIGLIP_HANDLERS,
    **GEMMA_HANDLERS,
    **AE_HANDLERS,
}


__all__ = [
    "PyvmMeta",
    "SIGLIP_HANDLERS",
    "GEMMA_HANDLERS",
    "AE_HANDLERS",
    "INSTRUCTION_TO_HANDLER",
    "run_siglip_pyvm",
    "run_gemma_pyvm",
    "run_ae_pyvm",
    "run_gemma_layer_fused_fallback",
    "_reset_pyvm_state",
    "siglip_ops",
    "gemma_ops",
    "ae_ops",
]
