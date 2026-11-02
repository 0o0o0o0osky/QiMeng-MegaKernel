"""Multi-batch PyVM handlers (B<=16). Barriers: row1 split-KV per-batch slices; row2 reduction per-batch slots."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from einops import einsum
from torch import Tensor

from qimeng_megakernel.models.instruction import Instruction
from qimeng_megakernel.models.LLM.decode_state import (
    decode_pos_id,
    instruction_active,
    row_active,
)

from ..globs import Globals
from ..model_config import LlamaConfig
from .instruction_defs import (
    DOWNPROJ_TILE_N,
    OPROJ_TILE_N,
    TILE_M_GRANULARITY,
    UPGATE_TILE_N,
    DecodeAttentionReduction,
    DecodeAttentionSplitKV,
    DownProj_GEMM_Residual,
    OProj_GEMM_Residual,
    RMS_LMHead_GEMM,
    RMS_QKVProj_GEMM_RoPE,
    RMS_UpGateProj_GEMM_SiLU,
)


def _barrier_stride() -> int:
    return LlamaConfig.num_attention_heads + 2 * LlamaConfig.num_key_value_heads


def _set_attn_partial_valid(
    globals: Globals,
    batch_idx: int,
    head_start: int,
    head_end: int,
    partial_idx: int,
    valid: bool,
) -> None:
    """Mark the SplitKV partial slots used by the reduction."""
    pv = globals.attn_partial_valid
    if pv is None:
        raise ValueError("multi-batch PyVM requires attn_partial_valid")
    pv[batch_idx, head_start:head_end, partial_idx] = valid


def _finish_splitkv_partial(
    globals: Globals,
    instruction: DecodeAttentionSplitKV,
    head_start: int,
    head_end: int,
    out: Tensor,
    *,
    valid: bool,
) -> None:
    """Publish the partial and, for P=1, directly satisfy OProj's reduction row."""
    b = instruction.batch_idx
    _set_attn_partial_valid(
        globals, b, head_start, head_end, instruction.partial_idx, valid
    )
    partial_row = _head_barrier_slice(
        globals, instruction.layer_idx, instruction.opcode() - 1, b
    )
    partial_row[head_start:head_end] += 1
    if instruction.num_partials == 1:
        dst = globals.attn_out[b].view(LlamaConfig.num_attention_heads, -1)
        if valid:
            dst[head_start:head_end] = out.to(globals.attn_out.dtype)
        else:
            dst[head_start:head_end].zero_()
        globals.barriers[
            instruction.layer_idx, DecodeAttentionReduction.opcode() - 1, b
        ] += head_end - head_start


def _head_barrier_slice(globs: Globals, layer: int, row: int, batch_idx: int) -> Tensor:
    s = _barrier_stride()
    return globs.barriers[layer, row, batch_idx * s : (batch_idx + 1) * s]


def _prev_barrier_row(instruction: Instruction) -> int:
    p = instruction.prev_opcode()
    return (p[0] - 1) if isinstance(p, tuple) else (p - 1)


def _range_from_idx(idx: int, granularity: int) -> tuple[int, int]:
    start = granularity * idx
    end = start + granularity
    return start, end


def _range_from_span(start_idx: int, end_idx: int, granularity: int) -> tuple[int, int]:
    start = granularity * start_idx
    end = granularity * end_idx
    return start, end


def _m_span(instruction) -> tuple[int, int]:
    m_start = instruction.tile_M_idx * instruction.tile_M_granularity
    m_end = min(m_start + instruction.tile_M_granularity, instruction.batch_size)
    return m_start, m_end


def _gemm_range(
    mat: Tensor,
    vec_2d: Tensor,
    out_start: int,
    out_end: int,
    in_start: int | None = None,
    in_end: int | None = None,
) -> Tensor:
    if in_start is not None and in_end is not None:
        m = mat[out_start:out_end, in_start:in_end]
        v = vec_2d[:, in_start:in_end]
    else:
        m = mat[out_start:out_end]
        v = vec_2d
    return einsum(m, v, "o i, b i -> b o")


def _rms_norm(inp: Tensor, weight: Tensor, eps: float) -> Tensor:
    input_dtype = inp.dtype
    x = inp.to(torch.float32)
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return weight * x.to(input_dtype)


def _rotate_half_interleaved(x: Tensor) -> Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    new_x1 = -x2
    new_x2 = x1
    stacked = torch.stack((new_x1, new_x2), dim=-1)
    return stacked.view_as(x)


def _apply_rotary_pos_emb_interleaved(
    q: Tensor, k: Tensor, cos: Tensor, sin: Tensor, unsqueeze_dim: int = 1
):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (_rotate_half_interleaved(q) * sin)
    k_embed = (k * cos) + (_rotate_half_interleaved(k) * sin)
    return q_embed, k_embed


def rms_qkvproj_gemm_rope(globals: Globals, instruction: RMS_QKVProj_GEMM_RoPE):
    layer_idx = instruction.layer_idx
    B = instruction.batch_size

    if layer_idx > 0:
        op0 = globals.barriers[layer_idx - 1, _prev_barrier_row(instruction)]
        expected = LlamaConfig.hidden_size // DOWNPROJ_TILE_N
        assert int(op0[0].item()) == expected, (op0[0].item(), expected)

    post_ln = _rms_norm(
        globals.hidden_states,
        globals.attn_norm_weights[layer_idx],
        eps=LlamaConfig.rms_norm_eps,
    )
    k_start = LlamaConfig.num_attention_heads * LlamaConfig.head_dim
    v_start = k_start + LlamaConfig.num_key_value_heads * LlamaConfig.head_dim
    head_dim = LlamaConfig.head_dim
    if globals.active_mask is not None:
        for b in range(B):
            if not row_active(globals, b):
                globals.q_post_rope[b].zero_()

    barriers = globals.barriers[instruction.layer_idx, instruction.opcode() - 1]

    if instruction.tile_M_granularity == TILE_M_GRANULARITY:
        start, end = _range_from_span(
            instruction.tile_N_start_idx,
            instruction.tile_N_end_idx,
            instruction.tile_N_granularity,
        )
        in_start, in_end = 0, LlamaConfig.hidden_size
    else:
        start, end = _range_from_idx(
            instruction.tile_M_idx, instruction.tile_M_granularity
        )
        in_start, in_end = _range_from_span(
            instruction.tile_N_start_idx,
            instruction.tile_N_end_idx,
            instruction.tile_N_granularity,
        )

    chunk_start = start
    while chunk_start < end:
        chunk_end_cand = min(chunk_start + head_dim, end)
        if chunk_start < k_start:
            seg_mode = "q"
            chunk_end = min(chunk_end_cand, k_start)
        elif chunk_start < v_start:
            seg_mode = "k"
            chunk_end = min(chunk_end_cand, v_start)
        else:
            seg_mode = "v"
            chunk_end = chunk_end_cand
        chunk_len = chunk_end - chunk_start

        matmul_output = _gemm_range(
            globals.qkv_weights[layer_idx],
            post_ln,
            chunk_start,
            chunk_end,
            in_start,
            in_end,
        )

        for b in range(B):
            if not row_active(globals, b):
                continue
            out = matmul_output[b]
            pos_id_b = decode_pos_id(globals, b)
            if seg_mode in ("q", "k"):
                full_head = torch.zeros(
                    1,
                    head_dim,
                    device=globals.hidden_states.device,
                    dtype=out.dtype,
                )
                head_segment = chunk_start % head_dim
                full_head[:, head_segment : head_segment + chunk_len] = out.view(1, -1)
                full_head_with_rope, _ = _apply_rotary_pos_emb_interleaved(
                    q=full_head,
                    k=full_head,
                    cos=globals.rope_cos[pos_id_b],
                    sin=globals.rope_sin[pos_id_b],
                    unsqueeze_dim=0,
                )
                out = full_head_with_rope[
                    :, head_segment : head_segment + chunk_len
                ].view(-1)

            if seg_mode == "q":
                globals.q_post_rope[b, chunk_start:chunk_end] = out.to(
                    globals.q_post_rope.dtype
                )
            elif seg_mode == "k":
                start_in_k = chunk_start - k_start
                end_in_k = chunk_end - k_start
                globals.k_cache[layer_idx, b, pos_id_b].view(-1)[
                    start_in_k:end_in_k
                ] = out.to(globals.k_cache.dtype)
            else:
                start_in_v = chunk_start - v_start
                end_in_v = chunk_end - v_start
                globals.v_cache[layer_idx, b, pos_id_b].view(-1)[
                    start_in_v:end_in_v
                ] = out.to(globals.v_cache.dtype)

        if chunk_len != head_dim or chunk_start % head_dim != 0:
            raise RuntimeError(
                "QKV WGMMA N tiles must publish complete 64-element heads"
            )
        if seg_mode == "q":
            barrier_slot = chunk_start // head_dim
        elif seg_mode == "k":
            barrier_slot = (
                LlamaConfig.num_attention_heads + (chunk_start - k_start) // head_dim
            )
        else:
            barrier_slot = (
                LlamaConfig.num_attention_heads
                + LlamaConfig.num_key_value_heads
                + (chunk_start - v_start) // head_dim
            )
        barriers[barrier_slot] += 1

        chunk_start = chunk_end


def decode_attention_split_kv(globals: Globals, instruction: DecodeAttentionSplitKV):
    gqa_ratio = LlamaConfig.num_attention_heads // LlamaConfig.num_key_value_heads
    b = instruction.batch_idx
    op_barriers = globals.barriers[
        instruction.layer_idx, _prev_barrier_row(instruction)
    ]
    for i in range(gqa_ratio):
        assert op_barriers[instruction.kv_head_idx * gqa_ratio + i] == 1
    assert op_barriers[LlamaConfig.num_attention_heads + instruction.kv_head_idx] == 1
    assert (
        op_barriers[
            LlamaConfig.num_attention_heads
            + LlamaConfig.num_key_value_heads
            + instruction.kv_head_idx
        ]
        == 1
    )

    kv_block_size = LlamaConfig.attn_kv_block_size
    layer_idx = instruction.layer_idx
    kv_head_idx = instruction.kv_head_idx

    head_start = kv_head_idx * gqa_ratio
    head_end = head_start + gqa_ratio
    seq_len = (
        instruction.seq_len
        if instruction.seq_len >= 0
        else (decode_pos_id(globals, b) + 1)
    )
    if not instruction_active(instruction.is_active, globals, b):
        zl = torch.full(
            (gqa_ratio,),
            -float("inf"),
            device=globals.hidden_states.device,
            dtype=torch.float32,
        )
        zo = torch.zeros(
            (gqa_ratio, LlamaConfig.head_dim),
            device=globals.hidden_states.device,
            dtype=torch.float32,
        )
        globals.attn_lse_intermediates[
            b, head_start:head_end, instruction.partial_idx
        ] = zl
        globals.attn_out_intermediates[
            b, head_start:head_end, instruction.partial_idx
        ] = zo
        _finish_splitkv_partial(
            globals, instruction, head_start, head_end, zo, valid=False
        )
        return

    total_blocks = max(1, math.ceil(seq_len / kv_block_size))
    blocks_per_partial = max(1, math.ceil(total_blocks / instruction.num_partials))

    start_block = instruction.partial_idx * blocks_per_partial
    end_block = min(start_block + blocks_per_partial, total_blocks)

    start_token = start_block * kv_block_size
    end_token = min(end_block * kv_block_size, seq_len)

    if start_token >= end_token:
        zl = torch.full(
            (gqa_ratio,),
            -float("inf"),
            device=globals.hidden_states.device,
            dtype=torch.float32,
        )
        zo = torch.zeros(
            (gqa_ratio, LlamaConfig.head_dim),
            device=globals.hidden_states.device,
            dtype=torch.float32,
        )
        globals.attn_lse_intermediates[
            b, head_start:head_end, instruction.partial_idx
        ] = zl
        globals.attn_out_intermediates[
            b, head_start:head_end, instruction.partial_idx
        ] = zo
        _finish_splitkv_partial(
            globals, instruction, head_start, head_end, zo, valid=False
        )
        return

    k = globals.k_cache[layer_idx, b, start_token:end_token, kv_head_idx]
    v = globals.v_cache[layer_idx, b, start_token:end_token, kv_head_idx]

    q = globals.q_post_rope[b].view(LlamaConfig.num_attention_heads, -1)[
        head_start:head_end
    ]

    qk = einsum(q.float(), k.float(), "h i, k i -> h k")
    scaled_qk = qk * (1 / math.sqrt(LlamaConfig.head_dim))
    softmax = torch.softmax(scaled_qk, dim=-1)
    lse = torch.log2(torch.sum(torch.exp(scaled_qk), dim=-1))
    out = einsum(softmax.float(), v.float(), "h k, k o -> h o")

    globals.attn_lse_intermediates[b, head_start:head_end, instruction.partial_idx] = (
        lse
    )
    globals.attn_out_intermediates[b, head_start:head_end, instruction.partial_idx] = (
        out
    )
    _finish_splitkv_partial(globals, instruction, head_start, head_end, out, valid=True)


def decode_attention_reduction(globals: Globals, instruction: DecodeAttentionReduction):
    head_start_idx = instruction.head_start_idx
    b = instruction.batch_idx
    op_slice = _head_barrier_slice(
        globals, instruction.layer_idx, _prev_barrier_row(instruction), b
    )
    if not instruction_active(instruction.is_active, globals, b):
        globals.attn_out[b].view(LlamaConfig.num_attention_heads, -1)[
            head_start_idx : head_start_idx + LlamaConfig.attn_reduction_size
        ].zero_()
        globals.barriers[instruction.layer_idx, instruction.opcode() - 1, b] += (
            LlamaConfig.attn_reduction_size
        )
        return

    assert op_slice[head_start_idx] == instruction.num_partials

    # Must be 1-D: a 2-D index makes PyTorch return shape [R, R, K] instead of [R, K],
    # which breaks denom (R,R) vs reduced O (R,K,D) in torch.where.
    indices_to_reduce = torch.as_tensor(
        instruction.reduction_list,
        dtype=torch.long,
        device=globals.hidden_states.device,
    ).reshape(-1)

    lses = globals.attn_lse_intermediates[
        b,
        head_start_idx : head_start_idx + LlamaConfig.attn_reduction_size,
        indices_to_reduce,
    ]
    outs = globals.attn_out_intermediates[
        b,
        head_start_idx : head_start_idx + LlamaConfig.attn_reduction_size,
        indices_to_reduce,
    ]

    pv = globals.attn_partial_valid
    if pv is None:
        raise ValueError("multi-batch PyVM requires attn_partial_valid")
    valid = pv[
        b,
        head_start_idx : head_start_idx + LlamaConfig.attn_reduction_size,
        indices_to_reduce,
    ].to(dtype=torch.bool)

    # Log2-domain merge: only ``valid`` partials contribute (no (-inf)-(-inf)).
    neg_big = torch.tensor(-1e9, device=lses.device, dtype=lses.dtype)
    max_lse = torch.max(torch.where(valid, lses, neg_big), dim=-1, keepdim=True).values
    has_active = valid.any(dim=-1, keepdim=True)
    max_lse = torch.where(
        has_active, max_lse, torch.full_like(max_lse, -float("inf"))
    )
    adjusted_factors = torch.where(
        valid, (lses - max_lse).exp2(), torch.zeros_like(lses)
    )
    denom = adjusted_factors.sum(dim=-1, keepdim=True)
    # outs: [R, P, D] — sum over partials P (dim -2), not dim=1 (breaks if extra leading dims).
    num = (outs * adjusted_factors.unsqueeze(-1)).sum(dim=-2)
    # Same normalization as single-batch PyVM: plain divide (denom [R,1] broadcasts to num [R,D]).
    den1 = denom.squeeze(-1)
    if den1.ndim != 1:
        raise RuntimeError(
            "decode_attention_reduction: expected denom.squeeze(-1) to be 1D per head; "
            f"got den1.shape={tuple(den1.shape)} (denom={tuple(denom.shape)}, "
            f"adjusted_factors={tuple(adjusted_factors.shape)}, lses={tuple(lses.shape)}, "
            f"outs={tuple(outs.shape)}, reduction_list={instruction.reduction_list!r})"
        )
    reduced = num / denom

    if instruction.is_terminal:
        globals.attn_out[b].view(LlamaConfig.num_attention_heads, -1)[
            head_start_idx : head_start_idx + LlamaConfig.attn_reduction_size
        ] = reduced.to(globals.attn_out.dtype)
    else:
        # log2(sum_k 2^(lse_k - max)) + max_lse  ==  max_lse + log2(denom); not natural log.
        new_lse_dead = torch.zeros(
            (reduced.shape[0],), device=lses.device, dtype=lses.dtype
        )
        new_lse = torch.where(
            has_active.squeeze(-1),
            max_lse.squeeze(-1) + torch.log2(den1),
            new_lse_dead,
        )
        output_slot = instruction.output_partial_idx
        globals.attn_lse_intermediates[
            b,
            head_start_idx : head_start_idx + LlamaConfig.attn_reduction_size,
            output_slot,
        ] = new_lse
        globals.attn_out_intermediates[
            b,
            head_start_idx : head_start_idx + LlamaConfig.attn_reduction_size,
            output_slot,
        ] = reduced
        pv[
            b,
            head_start_idx : head_start_idx + LlamaConfig.attn_reduction_size,
            output_slot,
        ] = has_active.squeeze(-1)

    globals.barriers[instruction.layer_idx, instruction.opcode() - 1, b] += (
        LlamaConfig.attn_reduction_size
    )


def oproj_gemm_residual(globals: Globals, instruction: OProj_GEMM_Residual):
    row2 = globals.barriers[
        instruction.layer_idx, _prev_barrier_row(instruction), : instruction.batch_size
    ]
    assert torch.all(row2 == LlamaConfig.num_attention_heads), row2.tolist()

    assert instruction.reduction_block_idx == 0
    m_start, m_end = _m_span(instruction)
    mat = globals.o_proj_weights[instruction.layer_idx]
    vec = globals.attn_out[m_start:m_end]
    residual = globals.hidden_states
    in_full = (0, LlamaConfig.hidden_size)
    next_op_barriers = globals.barriers[instruction.layer_idx, instruction.opcode() - 1]

    for block_idx in range(instruction.tile_N_start_idx, instruction.tile_N_end_idx):
        start, end = _range_from_idx(block_idx, instruction.tile_N_granularity)
        gemm_out = _gemm_range(
            mat,
            vec,
            start,
            end,
            in_start=in_full[0],
            in_end=in_full[1],
        )
        go = gemm_out.to(residual.dtype)
        for mi in range(m_start, m_end):
            if not row_active(globals, mi):
                go[mi - m_start].zero_()
        residual[m_start:m_end, start:end] += go
        next_op_barriers[0] += 1


def down_proj_residual(globals: Globals, instruction: DownProj_GEMM_Residual):
    op_barriers = globals.barriers[
        instruction.layer_idx, _prev_barrier_row(instruction)
    ]
    expected_upgate = LlamaConfig.intermediate_size // UPGATE_TILE_N
    assert op_barriers[0] == expected_upgate

    m_start, m_end = _m_span(instruction)
    mat = globals.down_proj_weights[instruction.layer_idx]
    vec = globals.silu_out[m_start:m_end]
    residual = globals.hidden_states
    next_op_barriers = globals.barriers[instruction.layer_idx, instruction.opcode() - 1]
    in_full = (0, LlamaConfig.intermediate_size)

    for block_idx in range(instruction.tile_N_start_idx, instruction.tile_N_end_idx):
        start, end = _range_from_idx(block_idx, instruction.tile_N_granularity)
        gemm_out = _gemm_range(
            mat,
            vec,
            start,
            end,
            in_start=in_full[0],
            in_end=in_full[1],
        )
        go = gemm_out.to(residual.dtype)
        for mi in range(m_start, m_end):
            if not row_active(globals, mi):
                go[mi - m_start].zero_()
        residual[m_start:m_end, start:end] += go
        next_op_barriers[0] += 1


def rms_upgate_proj_gemm_silu(globals: Globals, instruction: RMS_UpGateProj_GEMM_SiLU):
    op_barriers = globals.barriers[
        instruction.layer_idx, _prev_barrier_row(instruction)
    ]
    expected_o = LlamaConfig.hidden_size // OPROJ_TILE_N
    assert op_barriers[0] == expected_o

    post_ln = _rms_norm(
        globals.hidden_states,
        globals.mlp_norm_weights[instruction.layer_idx],
        eps=LlamaConfig.rms_norm_eps,
    )
    m_start, m_end = _m_span(instruction)
    post_b = post_ln[m_start:m_end]
    barriers = globals.barriers[instruction.layer_idx, instruction.opcode() - 1]

    for block_idx in instruction.tile_N_block_idxs:
        start, end = _range_from_idx(block_idx, instruction.tile_N_granularity)
        up_mat = _gemm_range(
            globals.up_proj_weights[instruction.layer_idx],
            post_b,
            start,
            end,
        )
        gate_mat = _gemm_range(
            globals.gate_proj_weights[instruction.layer_idx],
            post_b,
            start,
            end,
        )
        post_silu = F.silu(gate_mat) * up_mat
        ps = post_silu.to(globals.silu_out.dtype)
        for mi in range(m_start, m_end):
            if not row_active(globals, mi):
                ps[mi - m_start].zero_()
        globals.silu_out[m_start:m_end, start:end] = ps
        barriers[0] += 1


def rms_lmhead_gemm(globals: Globals, instruction: RMS_LMHead_GEMM):
    op_tail = globals.barriers[
        LlamaConfig.num_hidden_layers - 1,
        _prev_barrier_row(instruction),
    ]
    expected_down = LlamaConfig.hidden_size // DOWNPROJ_TILE_N
    assert op_tail[0] == expected_down

    post_ln = _rms_norm(
        globals.hidden_states,
        globals.lm_head_norm_weights,
        eps=LlamaConfig.rms_norm_eps,
    )

    for block_idx in range(
        instruction.start_output_block_idx, instruction.end_output_block_idx
    ):
        start, end = _range_from_idx(block_idx, instruction.tile_N_granularity)
        logits_slice = einsum(
            globals.lm_head_weights[start:end],
            post_ln,
            "o i, b i -> b o",
        )
        ls = logits_slice.to(globals.logits.dtype)
        if globals.active_mask is not None and ls.dim() >= 1:
            for bb in range(ls.shape[0]):
                if not row_active(globals, bb):
                    ls[bb] = -1e9
        globals.logits[:, start:end] = ls


INSTRUCTION_TO_HANDLER = {
    RMS_QKVProj_GEMM_RoPE: rms_qkvproj_gemm_rope,
    DecodeAttentionSplitKV: decode_attention_split_kv,
    DecodeAttentionReduction: decode_attention_reduction,
    OProj_GEMM_Residual: oproj_gemm_residual,
    RMS_UpGateProj_GEMM_SiLU: rms_upgate_proj_gemm_silu,
    DownProj_GEMM_Residual: down_proj_residual,
    RMS_LMHead_GEMM: rms_lmhead_gemm,
}
