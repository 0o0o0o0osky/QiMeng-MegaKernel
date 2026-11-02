"""Python VM handlers for pi0 A100 instructions (tiled path, LLM-style).

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

from ..fused import ae_ops, attention_ops, gemma_ops, siglip_ops
from ..globs import AeGlobals, GemmaGlobals, SiglipGlobals
from ..pi0_config import Pi0A100Config as C
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
    I.AeResidualReduce,
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
    I.GemmaAttnQk,
    I.GemmaAttnSoftmax,
    I.GemmaAttnPv,
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
    raise TypeError(f"unsupported Pi0/A100 instruction type: {type(inst).__name__}")


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


def _handle_siglip_ln_attn(globs: SiglipGlobals, inst: I.SiglipLnAttn) -> None:
    def _run():
        assert globs.vision_x is not None and globs.siglip_x_normed is not None
        assert globs.ln_attn_w is not None and globs.ln_attn_b is not None
        x = globs.vision_x[0, 0]
        gamma, beta = globs.ln_attn_w[0, inst.layer_idx], globs.ln_attn_b[0, inst.layer_idx]
        normed = siglip_ops.layernorm(x, gamma, beta)
        globs.siglip_x_normed[0, 0, :, : x.shape[-1]] = normed.to(globs.siglip_x_normed.dtype)

    _run_once(globs, ("siglip", inst.layer_idx, "ln_attn"), _run, inst)


def _handle_siglip_ln_ffn(globs: SiglipGlobals, inst: I.SiglipLnFfn) -> None:
    def _run():
        assert globs.vision_x is not None and globs.siglip_x_normed is not None
        assert globs.ln_ffn_w is not None and globs.ln_ffn_b is not None
        x = globs.vision_x[0, 0]
        gamma, beta = globs.ln_ffn_w[0, inst.layer_idx], globs.ln_ffn_b[0, inst.layer_idx]
        normed = siglip_ops.layernorm(x, gamma, beta)
        globs.siglip_x_normed[0, 0, :, : x.shape[-1]] = normed.to(globs.siglip_x_normed.dtype)

    _run_once(globs, ("siglip", inst.layer_idx, "ln_ffn"), _run, inst)


def _handle_siglip_qkv(globs: SiglipGlobals, inst: I.SiglipQkv) -> None:
    def _run():
        assert globs.siglip_x_normed is not None and globs.qkv_buf is not None
        assert globs.qkv_w is not None and globs.qkv_b is not None
        x = globs.siglip_x_normed[0, 0, :, : C.vision_width]
        w, b = globs.qkv_w[0, inst.layer_idx], globs.qkv_b[0, inst.layer_idx]
        padded = siglip_ops.fused_siglip_qkv(x, w, b)
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
        x = globs.vision_x[0, 0]
        wo, bo = globs.oproj_w[0, inst.layer_idx], globs.oproj_b[0, inst.layer_idx]
        y = siglip_ops.fused_siglip_oproj_residual(x, globs.attn_out[0, 0], wo, bo)
        globs.vision_x[0, 0, :, : C.vision_width] = y

    _run_once(globs, ("siglip", inst.layer_idx, "oproj"), _run, inst)


def _handle_siglip_fc1(globs: SiglipGlobals, inst: I.SiglipFc1) -> None:
    def _run():
        assert globs.siglip_x_normed is not None and globs.fc1_buf is not None
        x = globs.siglip_x_normed[0, 0, :, : C.vision_width]
        w, b = globs.fc1_w[0, inst.layer_idx], globs.fc1_b[0, inst.layer_idx]
        h = siglip_ops.fused_siglip_fc1(x, w, b)
        globs.fc1_buf[0, 0, :, : h.shape[-1]] = h.to(globs.fc1_buf.dtype)

    _run_once(globs, ("siglip", inst.layer_idx, "fc1"), _run, inst)


def _handle_siglip_fc2(globs: SiglipGlobals, inst: I.SiglipFc2) -> None:
    def _run():
        assert globs.vision_x is not None and globs.fc1_buf is not None
        x = globs.vision_x[0, 0, :, : C.vision_width]
        h = globs.fc1_buf[0, 0, :, : C.vision_mlp_width]
        w, b = globs.fc2_w[0, inst.layer_idx], globs.fc2_b[0, inst.layer_idx]
        y = siglip_ops.fused_siglip_fc2_residual(x, h, w, b)
        globs.vision_x[0, 0, :, : C.vision_width] = y

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


SIGLIP_HANDLERS: dict[type[Instruction], Callable] = {
    I.SiglipLnAttn: _handle_siglip_ln_attn,
    I.SiglipLnFfn: _handle_siglip_ln_ffn,
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
        inv = gemma_ops.rms_inv(x).squeeze(-1)
        globs.rms_factor[0, 0, 0, : _meta(globs).seq_len] = inv

    _run_once(globs, ("gemma", inst.layer_idx, "rmsn_attn"), _run, inst)


def _handle_gemma_rmsn_ffn(globs: GemmaGlobals, inst: I.GemmaRmsnFfn) -> None:
    def _run():
        assert globs.encoder_x is not None and globs.rms_factor is not None
        x = globs.encoder_x[0, 0, : _meta(globs).seq_len]
        inv = gemma_ops.rms_inv(x).squeeze(-1)
        globs.rms_factor[0, 0, 1, : _meta(globs).seq_len] = inv

    _run_once(globs, ("gemma", inst.layer_idx, "rmsn_ffn"), _run, inst)


def _handle_gemma_qkv_rope(globs: GemmaGlobals, inst: I.GemmaQkvRope) -> None:
    meta = _meta(globs)

    def _run():
        assert globs.encoder_x is not None and globs.qkv_w is not None
        x = globs.encoder_x[0, 0, : meta.seq_len]
        w = globs.qkv_w[inst.layer_idx]
        rope = globs.rope[0, 0, : meta.seq_len]
        factor = globs.rms_factor[0, 0, 0, : meta.seq_len]
        q, k, v = gemma_ops.fused_qkv_rope_with_factor(x, w, rope, factor)
        y = gemma_ops.pack_qkv(q, k, v)
        globs.qkv_buf[0, 0, : meta.seq_len, : y.shape[-1]] = y.to(globs.qkv_buf.dtype)
        globs.k_cache[inst.layer_idx, : meta.seq_len] = k.to(globs.k_cache.dtype)
        globs.v_cache[inst.layer_idx, : meta.seq_len] = v.to(globs.v_cache.dtype)

    _run_once(globs, ("gemma", inst.layer_idx, "qkv"), _run, inst)


def _gemma_seq_pad(globs: GemmaGlobals) -> int:
    """The padded row count the kernel calls `g.seq`."""
    return int(globs.encoder_x.shape[-2])


def _gemma_scores(globs: GemmaGlobals, seq_pad: int) -> torch.Tensor:
    """`attn_scores` as [head, row, key].

    The buffer is allocated flat because `gemma_qkv_rope` also borrows it, so
    the head axis is a view rather than a dimension.
    """
    return globs.attn_scores.view(-1)[
        : C.num_heads * seq_pad * seq_pad
    ].view(C.num_heads, seq_pad, seq_pad)


def _handle_gemma_attn_qk(globs: GemmaGlobals, inst: I.GemmaAttnQk) -> None:
    """S = Q K^T scaled, keys past `seq_len` set to -inf, plus Vt.

    The whole phase runs once, as every tiled Gemma phase does here: the VM
    checks the schedule's dependency structure, not its tiling.
    """
    seq_pad = _gemma_seq_pad(globs)

    def _run():
        scores = _gemma_scores(globs, seq_pad)
        q_all = globs.qkv_buf[0, 0, :seq_pad, : C.num_heads * C.head_dim].float()
        k = globs.k_cache[inst.layer_idx, :seq_pad].float()
        scale = C.head_dim**-0.5
        for head in range(C.num_heads):
            qh = q_all[:, head * C.head_dim : (head + 1) * C.head_dim]
            logits = (qh @ k.T) * scale
            logits[:, inst.seq_len :] = float("-inf")
            scores[head] = logits.to(scores.dtype)
        vt = globs.v_cache[inst.layer_idx, :seq_pad].clone()
        vt[inst.seq_len :] = 0
        globs.attn_vt[0, 0, : C.head_dim, :seq_pad] = vt.T.to(globs.attn_vt.dtype)

    _run_once(globs, ("gemma", inst.layer_idx, "attn_qk"), _run, inst)


def _handle_gemma_attn_softmax(globs: GemmaGlobals, inst: I.GemmaAttnSoftmax) -> None:
    seq_pad = _gemma_seq_pad(globs)

    def _run():
        scores = _gemma_scores(globs, seq_pad)
        probabilities = torch.softmax(scores.float(), dim=-1)
        scores.copy_(probabilities.to(scores.dtype))

    _run_once(globs, ("gemma", inst.layer_idx, "attn_sm"), _run, inst)


def _handle_gemma_attn_pv(globs: GemmaGlobals, inst: I.GemmaAttnPv) -> None:
    seq_pad = _gemma_seq_pad(globs)

    def _run():
        probabilities = _gemma_scores(globs, seq_pad).float()
        vt = globs.attn_vt[0, 0, : C.head_dim, :seq_pad].float()
        for head in range(C.num_heads):
            head0 = head * C.head_dim
            output = probabilities[head] @ vt.T
            globs.attn_out[0, 0, :seq_pad, head0 : head0 + C.head_dim] = output.to(
                globs.attn_out.dtype
            )

    _run_once(globs, ("gemma", inst.layer_idx, "attn_pv"), _run, inst)


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
        factor = globs.rms_factor[0, 0, inst.norm_sel, : meta.seq_len]
        gate = gemma_ops.fused_gate_projection(
            x, globs.gate_w[inst.layer_idx], factor
        )
        hidden = gemma_ops.fused_up_mul(
            x,
            globs.up_w[inst.layer_idx],
            factor,
            gate,
        )
        globs.up_buf[0, 0, : meta.seq_len] = hidden.to(globs.up_buf.dtype)

    _run_once(globs, ("gemma", inst.layer_idx, "gate"), _run, inst)


def _handle_gemma_up_mul(globs: GemmaGlobals, inst: I.GemmaUpMul) -> None:
    _run_once(
        globs,
        ("gemma", inst.layer_idx, "up_mul"),
        lambda: None,
        inst,
    )


def _handle_gemma_down(globs: GemmaGlobals, inst: I.GemmaDown) -> None:
    meta = _meta(globs)

    def _run():
        residual = globs.encoder_x[0, 0, : meta.seq_len]
        hidden = globs.up_buf[0, 0, : meta.seq_len]
        output = gemma_ops.fused_down_residual(
            residual,
            hidden,
            globs.down_w[inst.layer_idx],
        )
        globs.encoder_x[0, 0, : meta.seq_len] = output.to(globs.encoder_x.dtype)

    _run_once(globs, ("gemma", inst.layer_idx, "down"), _run, inst)


def _handle_gemma_down_splitk(globs: GemmaGlobals, inst: I.GemmaDownSplitK) -> None:
    _handle_gemma_down(globs, inst)  # type: ignore[arg-type]


GEMMA_HANDLERS: dict[type[Instruction], Callable] = {
    I.GemmaRmsnAttn: _handle_gemma_rmsn_attn,
    I.GemmaRmsnFfn: _handle_gemma_rmsn_ffn,
    I.GemmaQkvRope: _handle_gemma_qkv_rope,
    I.GemmaAttnQk: _handle_gemma_attn_qk,
    I.GemmaAttnSoftmax: _handle_gemma_attn_softmax,
    I.GemmaAttnPv: _handle_gemma_attn_pv,
    I.GemmaOproj: _handle_gemma_oproj,
    I.GemmaGate: _handle_gemma_gate,
    I.GemmaUpMul: _handle_gemma_up_mul,
    I.GemmaDown: _handle_gemma_down,
    I.GemmaDownSplitK: _handle_gemma_down_splitk,
}


# ── AE ──────────────────────────────────────────────────────────────────────


def _ae_layout(globs: AeGlobals) -> dict[str, int]:
    layout = _meta(globs).layout
    return layout if layout is not None else C.ae_kv_layout_for()


def _handle_ae_rmsn_attn(globs: AeGlobals, inst: I.AeRmsnAttn) -> None:
    layout = _ae_layout(globs)

    def _run():
        x = globs.x_acc[0, 0, : layout["m_pad"]]
        landed, inv = ae_ops.fused_ae_rmsn_landing(x)
        globs.rms_factor[0, 0, 0, : layout["m_pad"]] = inv
        globs.x[0, 0, : layout["m_pad"]] = landed

    _run_once(globs, ("ae", inst.layer_idx, "rmsn_attn"), _run, inst)


def _handle_ae_rmsn_ffn(globs: AeGlobals, inst: I.AeRmsnFfn) -> None:
    layout = _ae_layout(globs)

    def _run():
        x = globs.x_acc[0, 0, : layout["m_pad"]]
        landed, inv = ae_ops.fused_ae_rmsn_landing(x)
        globs.rms_factor[0, 0, 1, : layout["m_pad"]] = inv
        globs.x[0, 0, : layout["m_pad"]] = landed

    _run_once(globs, ("ae", inst.layer_idx, "rmsn_ffn"), _run, inst)


def _handle_ae_qkv_rope(globs: AeGlobals, inst: I.AeQkvRope) -> None:
    layout = _ae_layout(globs)
    row0 = layout["state_key"]

    def _run():
        x = globs.x[0, 0, : layout["m_pad"]]
        w = globs.qkv_w[0, inst.layer_idx]
        rope = globs.rope[0, 0, : layout["m_pad"]]
        inv = globs.rms_factor[0, 0, 0, : layout["m_pad"]]
        y = ae_ops.fused_ae_linear_with_factor(x, w, inv)
        q_width = C.num_heads * C.head_dim
        q = gemma_ops.apply_rope_interleaved(y[:, :q_width], rope)
        k = gemma_ops.apply_rope_interleaved(
            y[:, q_width : q_width + C.head_dim], rope
        )
        v = y[:, q_width + C.head_dim :]
        y = gemma_ops.pack_qkv(q, k, v)
        globs.qkv_buf[0, 0, : layout["m_pad"], : y.shape[-1]] = y.to(globs.qkv_buf.dtype)
        globs.k_cache[0, inst.layer_idx, row0 : row0 + layout["m_pad"]] = k.to(
            globs.k_cache.dtype
        )
        globs.v_cache[0, inst.layer_idx, row0 : row0 + layout["m_pad"]] = v.to(
            globs.v_cache.dtype
        )

    _run_once(globs, ("ae", inst.layer_idx, "qkv"), _run, inst)


def _handle_ae_attn_partial(globs: AeGlobals, inst: I.AeAttnPartial) -> None:
    layout = _ae_layout(globs)

    def _run():
        start = inst.kv_blk0 * C.ae_kv_block
        end = min(start + inst.kv_nblk * C.ae_kv_block, inst.seq_len)
        head0 = inst.head_idx * C.head_dim
        q = globs.qkv_buf[0, 0, : layout["m_pad"], head0 : head0 + C.head_dim]
        k = globs.k_cache[0, inst.layer_idx, start:end]
        v = globs.v_cache[0, inst.layer_idx, start:end]
        partial, maximum, row_sum = attention_ops.ae_split_attention_partial(
            q,
            k,
            v,
            key_start=start,
            prefix_len=inst.prefix_len,
            state_key=inst.state_key,
            visible_end=inst.seq_len,
            scale=C.head_dim**-0.5,
        )
        globs.attn_partial[
            0, inst.partial_idx, : layout["m_pad"], head0 : head0 + C.head_dim
        ] = partial.to(globs.attn_partial.dtype)
        globs.attn_stats[
            0, inst.partial_idx, inst.head_idx, : layout["m_pad"]
        ] = maximum
        globs.attn_stats[
            0, inst.partial_idx, C.num_heads + inst.head_idx, : layout["m_pad"]
        ] = row_sum

    _run_once(
        globs,
        ("ae", inst.layer_idx, "attn_p", inst.partial_idx, inst.head_idx),
        _run,
        inst,
    )


def _handle_ae_attn_reduce(globs: AeGlobals, inst: I.AeAttnReduce) -> None:
    layout = _ae_layout(globs)

    def _run():
        head0 = inst.head_idx * C.head_dim
        partials = globs.attn_partial[
            0, : inst.n_splits, : layout["m_pad"], head0 : head0 + C.head_dim
        ]
        maxima = globs.attn_stats[
            0, : inst.n_splits, inst.head_idx, : layout["m_pad"]
        ]
        sums = globs.attn_stats[
            0, : inst.n_splits, C.num_heads + inst.head_idx, : layout["m_pad"]
        ]
        output = attention_ops.reduce_attention_partials(partials, maxima, sums)
        globs.attn_out[
            0,
            0,
            inst.head_idx * layout["m_pad"] : (inst.head_idx + 1) * layout["m_pad"],
            : C.head_dim,
        ] = output.to(globs.attn_out.dtype)

    _run_once(globs, ("ae", inst.layer_idx, "attn_r", inst.head_idx), _run, inst)


def _handle_ae_oproj(globs: AeGlobals, inst: I.AeOproj) -> None:
    layout = _ae_layout(globs)
    oproj_splits = 8

    def _run():
        attn = ae_ops.unpack_attention_head_major(
            globs.attn_out[0, 0], rows=layout["m_pad"]
        )
        x_acc = globs.x_acc[0, 0, : layout["m_pad"]]
        wo_planes = globs.oproj_w[0, inst.layer_idx * oproj_splits : (inst.layer_idx + 1) * oproj_splits]
        wo = ae_ops.merge_k_split_weight(wo_planes, logical_k=C.num_heads * C.head_dim)
        globs.x_acc[0, 0, : layout["m_pad"]] = ae_ops.fused_ae_oproj_residual(
            x_acc, attn, wo
        )

    _run_once(globs, ("ae", inst.layer_idx, "oproj"), _run, inst)


def _handle_ae_gate(globs: AeGlobals, inst: I.AeGate) -> None:
    layout = _ae_layout(globs)

    def _run():
        gate = ae_ops.fused_ae_gate_with_factor(
            globs.x[0, 0, : layout["m_pad"]],
            globs.gate_w[0, inst.layer_idx],
            globs.rms_factor[0, 0, inst.norm_sel, : layout["m_pad"]],
        )
        globs.gate_buf[0, 0] = ae_ops.pack_k_split_major(gate.to(globs.gate_buf.dtype))

    _run_once(globs, ("ae", inst.layer_idx, "gate"), _run, inst)


def _handle_ae_up_mul(globs: AeGlobals, inst: I.AeUpMul) -> None:
    layout = _ae_layout(globs)

    def _run():
        gate = ae_ops.unpack_k_split_major(globs.gate_buf[0, 0], splits=16)
        hidden = ae_ops.fused_ae_up_mul_with_factor(
            globs.x[0, 0, : layout["m_pad"]],
            globs.up_w[0, inst.layer_idx],
            globs.rms_factor[0, 0, inst.norm_sel, : layout["m_pad"]],
            gate,
        )
        globs.ffn_buf[0, 0] = ae_ops.pack_k_split_major(hidden)

    _run_once(globs, ("ae", inst.layer_idx, "up_mul"), _run, inst)


def _handle_ae_ffn_dual(globs: AeGlobals, inst: I.AeFfnDual) -> None:
    layout = _ae_layout(globs)

    def _run():
        hidden = ae_ops.fused_ae_ffn_dual_with_factor(
            globs.x[0, 0, : layout["m_pad"]],
            globs.gate_w[0, inst.layer_idx],
            globs.up_w[0, inst.layer_idx],
            globs.rms_factor[0, 0, inst.norm_sel, : layout["m_pad"]],
        )
        globs.ffn_buf[0, 0] = ae_ops.pack_k_split_major(hidden)

    _run_once(globs, ("ae", inst.layer_idx, "ffn_dual"), _run, inst)


def _handle_ae_down(globs: AeGlobals, inst: I.AeDown) -> None:
    layout = _ae_layout(globs)
    down_splits = 16

    def _run():
        h = ae_ops.unpack_k_split_major(globs.ffn_buf[0, 0], splits=down_splits)
        wd_planes = globs.down_w[0, inst.layer_idx * down_splits : (inst.layer_idx + 1) * down_splits]
        wd = ae_ops.merge_k_split_weight(wd_planes, logical_k=C.decoder_mlp_width)
        x_acc = globs.x_acc[0, 0, : layout["m_pad"]]
        globs.x_acc[0, 0, : layout["m_pad"]] = ae_ops.fused_ae_down_residual(
            x_acc, h, wd
        )

    _run_once(globs, ("ae", inst.layer_idx, "down"), _run, inst)


def _handle_ae_residual_reduce(
    globs: AeGlobals, inst: I.AeResidualReduce
) -> None:
    # The Python VM evaluates OProj/Down as a full unsplit matmul, so its
    # deterministic scratch reduction is represented by a scheduling no-op.
    _run_once(
        globs,
        ("ae", inst.layer_idx, "residual_reduce", inst.dep_expected),
        lambda: None,
        inst,
    )


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
    I.AeGate: _handle_ae_gate,
    I.AeUpMul: _handle_ae_up_mul,
    I.AeDown: _handle_ae_down,
    I.AeFfnDual: _handle_ae_ffn_dual,
    I.AeActIn: _handle_ae_act_in,
    I.AeActMlp: _handle_ae_act_mlp,
    I.AeRmsnOut: _handle_ae_rmsn_out,
    I.AeActOut: _handle_ae_act_out,
    I.AeResidualReduce: _handle_ae_residual_reduce,
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
