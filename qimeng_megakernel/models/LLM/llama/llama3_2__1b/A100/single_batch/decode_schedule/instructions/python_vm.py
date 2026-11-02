import math

import torch
import torch.nn.functional as F
from einops import einsum
from torch import Tensor
from qimeng_megakernel.models.instruction import Instruction
from ..globs import Globals
from ..model_config import LlamaConfig
from .instruction_defs import (
    RMS_QKVProj_GEMV_RoPE,
    DecodeAttentionSplitKV,
    DecodeAttentionReduction,
    OProj_GEMV_Residual,
    RMS_UpGateProj_GEMV_SiLU,
    DownProj_GEMV_Residual,
    RMS_LMHead_GEMV,
    BARRIER_ROW,
    DOWNPROJ_BARRIER_COUNT,
    TILE_N_GRANULARITY,
    UPGATE_BLOCKS_PER_SIGNAL_BUCKET,
)


def _prev_barrier_row(instruction: Instruction) -> int:
    p = instruction.prev_opcode()
    return BARRIER_ROW(p[0] if isinstance(p, tuple) else p)


def _range_from_idx(idx: int, granularity: int):
    start = granularity * idx
    end = start + granularity
    return start, end


def _range_from_span(start_idx: int, end_idx: int, granularity: int):
    start = granularity * start_idx
    end = granularity * end_idx
    return start, end


def _gemv_range(
    mat: Tensor,
    vec: Tensor,
    out_start: int,
    out_end: int,
    in_start: int | None = None,
    in_end: int | None = None,
):
    # Layout: mat(n, k), vec(k); GEMV out(n) = mat @ vec. k contiguous for CUDA; slice along n.
    if in_start is not None and in_end is not None:
        mat = mat[out_start:out_end, in_start:in_end]
        vec = vec[in_start:in_end]
    else:
        mat = mat[out_start:out_end]

    out = einsum(mat, vec, "o i, i -> o")
    return out


def _rms_norm(inp: Tensor, weight: Tensor, eps: float):
    input_dtype = inp.dtype
    inp = inp.to(torch.float32)
    variance = inp.pow(2).mean(-1, keepdim=True)
    inp = inp * torch.rsqrt(variance + eps)

    return weight * inp.to(input_dtype)


def _gemv_with_residual(
    mat: Tensor,
    vec: Tensor,
    residual: Tensor,
    out_start: int,
    out_end: int,
    in_start: int,
    in_end: int,
):
    gemv_out = _gemv_range(
        mat=mat,
        vec=vec,
        out_start=out_start,
        out_end=out_end,
        in_start=in_start,
        in_end=in_end,
    )
    residual[out_start:out_end] += gemv_out.to(residual.dtype)

def _rotate_half_interleaved(x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]

    new_x1 = -x2
    new_x2 = x1

    stacked = torch.stack((new_x1, new_x2), dim=-1)
    return stacked.view_as(x)

def _apply_rotary_pos_emb_interleaved(
    q, k, cos, sin, position_ids=None, unsqueeze_dim=1
):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (_rotate_half_interleaved(q) * sin)
    k_embed = (k * cos) + (_rotate_half_interleaved(k) * sin)
    return q_embed, k_embed


def oproj_gemv_residual(globals: Globals, instruction: OProj_GEMV_Residual):
    # Barrier check
    op_barriers = globals.barriers[instruction.layer_idx, _prev_barrier_row(instruction)]
    assert op_barriers[0] == LlamaConfig.num_attention_heads
    assert instruction.reduction_block_idx == 0

    mat = globals.o_proj_weights[instruction.layer_idx]
    vec = globals.attn_out
    residual = globals.hidden_states
    in_full = 0, LlamaConfig.hidden_size
    next_op_barriers = globals.barriers[
        instruction.layer_idx, BARRIER_ROW(instruction.opcode())
    ]

    for block_idx in range(instruction.tile_N_start_idx, instruction.tile_N_end_idx):
        start, end = _range_from_idx(block_idx, instruction.tile_N_granularity)
        gemv_out = _gemv_range(
            mat=mat,
            vec=vec,
            out_start=start,
            out_end=end,
            in_start=in_full[0],
            in_end=in_full[1],
        )
        residual[start:end] += gemv_out.to(residual.dtype)
        next_op_barriers[0] += 1


def down_proj_residual(globals: Globals, instruction: DownProj_GEMV_Residual):
    # This split consumes only its matching UpGate bucket.
    op_barriers = globals.barriers[instruction.layer_idx, _prev_barrier_row(instruction)]
    expected_upgate_blocks = (
        instruction.split_K_granularity // instruction.tile_N_granularity
    )
    assert expected_upgate_blocks == UPGATE_BLOCKS_PER_SIGNAL_BUCKET
    assert op_barriers[instruction.split_K_idx] == expected_upgate_blocks

    mat = globals.down_proj_weights[instruction.layer_idx]
    vec = globals.silu_out
    residual = globals.hidden_states
    # split_K: each instruction uses one slice of input (same as CUDA kernel)
    in_start = instruction.split_K_idx * instruction.split_K_granularity
    in_end = min(
        (instruction.split_K_idx + 1) * instruction.split_K_granularity,
        LlamaConfig.intermediate_size,
    )
    next_op_barriers = globals.barriers[
        instruction.layer_idx, BARRIER_ROW(instruction.opcode())
    ]

    # tile_N = output row blocks (HiddenDim), same semantics as kernel
    for block_idx in range(instruction.tile_N_start_idx, instruction.tile_N_end_idx):
        start, end = _range_from_idx(block_idx, instruction.tile_N_granularity)
        gemv_out = _gemv_range(
            mat=mat,
            vec=vec,
            out_start=start,
            out_end=end,
            in_start=in_start,
            in_end=in_end,
        )
        residual[start:end] += gemv_out.to(residual.dtype)
        next_op_barriers[0] += 1


def rms_upgate_proj_gemv_silu(
    globals: Globals, instruction: RMS_UpGateProj_GEMV_SiLU
):
    # Barrier check
    op_barriers = globals.barriers[instruction.layer_idx, _prev_barrier_row(instruction)]
    # Expected: one barrier count per hidden block (hidden_size / tile_N_granularity).
    expected_o_proj_blocks = LlamaConfig.hidden_size // instruction.tile_N_granularity
    assert op_barriers[0] == expected_o_proj_blocks

    post_ln = _rms_norm(
        inp=globals.hidden_states,
        weight=globals.mlp_norm_weights[instruction.layer_idx],
        eps=LlamaConfig.rms_norm_eps,
    )

    barriers = globals.barriers[
        instruction.layer_idx, BARRIER_ROW(instruction.opcode())
    ]

    for block_idx in instruction.tile_N_block_idxs:
        start, end = _range_from_idx(block_idx, instruction.tile_N_granularity)

        up_matvec = _gemv_range(
            mat=globals.up_proj_weights[instruction.layer_idx],
            vec=post_ln,
            out_start=start,
            out_end=end,
        )

        gate_matvec = _gemv_range(
            mat=globals.gate_proj_weights[instruction.layer_idx],
            vec=post_ln,
            out_start=start,
            out_end=end,
        )

        post_silu = F.silu(gate_matvec) * up_matvec

        globals.silu_out[start:end] = post_silu

        barriers[block_idx // UPGATE_BLOCKS_PER_SIGNAL_BUCKET] += 1


def rms_qkvproj_gemv_rope(
    globals: Globals, instruction: RMS_QKVProj_GEMV_RoPE
):
    layer_idx = instruction.layer_idx

    # Barrier check: every output tile from every DownProj split-K has arrived.
    if layer_idx > 0:
        op_barriers = globals.barriers[layer_idx - 1, _prev_barrier_row(instruction)]
        assert op_barriers[0] == DOWNPROJ_BARRIER_COUNT

    post_ln = _rms_norm(
        inp=globals.hidden_states,
        weight=globals.attn_norm_weights[layer_idx],
        eps=LlamaConfig.rms_norm_eps,
    )

    pos_id = globals.position_id

    k_start = LlamaConfig.num_attention_heads * LlamaConfig.head_dim
    v_start = k_start + LlamaConfig.num_key_value_heads * LlamaConfig.head_dim

    barriers = globals.barriers[
        instruction.layer_idx, BARRIER_ROW(instruction.opcode())
    ]

    # GEMV: when tile_M_granularity==1, output range is given by tile_N (N = output blocks);
    # input is full. Otherwise tile_M = output chunk, tile_N = input chunk.
    if instruction.tile_M_granularity == 1:
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

    head_dim = LlamaConfig.head_dim
    blocks_per_head = head_dim // instruction.tile_N_granularity

    # One instruction may span Q, K, and V output blocks. The fused PyTorch
    # reference applies one GEMV per block, then RoPE for Q/K, before the write.
    chunk_start = start
    while chunk_start < end:
        # Clip chunk_end so this chunk stays within one segment (Q / K / V)
        # A scheduled row span may cross a head boundary when QKV rows are
        # distributed across SMs.  Keep each RoPE slice inside one head.
        next_head_boundary = ((chunk_start // head_dim) + 1) * head_dim
        chunk_end_cand = min(next_head_boundary, end)
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

        # GEMV for this block only
        matmul_output = _gemv_range(
            mat=globals.qkv_weights[layer_idx],
            vec=post_ln,
            out_start=chunk_start,
            out_end=chunk_end,
            in_start=in_start,
            in_end=in_end,
        )
        out = matmul_output

        # RoPE for Q/K (full head; place partial chunk at correct position within head)
        if seg_mode in ["q", "k"]:
            # Create full head buffer and place chunk at correct position
            full_head = torch.zeros(
                1,
                head_dim,
                device=globals.hidden_states.device,
                dtype=out.dtype,
            )
            head_segment = chunk_start % head_dim
            full_head_start = head_segment
            full_head_end = full_head_start + chunk_len

            full_head[:, full_head_start:full_head_end] = out.view(1, -1)
            full_head_with_rope, _ = _apply_rotary_pos_emb_interleaved(
                q=full_head,
                k=full_head,
                cos=globals.rope_cos[pos_id],
                sin=globals.rope_sin[pos_id],
                unsqueeze_dim=0,
            )
            out = full_head_with_rope[:, full_head_start:full_head_end].view(-1)

        if seg_mode == "q":
            globals.q_post_rope[chunk_start:chunk_end] = out.to(globals.q_post_rope.dtype)
        elif seg_mode == "k":
            start_in_k = chunk_start - k_start
            end_in_k = chunk_end - k_start
            globals.k_cache[layer_idx, :, pos_id].view(-1)[start_in_k:end_in_k] = (
                out.to(globals.k_cache.dtype)
            )
        else:
            start_in_v = chunk_start - v_start
            end_in_v = chunk_end - v_start
            globals.v_cache[layer_idx, :, pos_id].view(-1)[start_in_v:end_in_v] = (
                out.to(globals.v_cache.dtype)
            )

        # Barrier: updated per N-direction output block (tile_N)
        if instruction.tile_M_granularity == 1:
            block_start_idx = chunk_start // instruction.tile_N_granularity
            block_end_idx = (chunk_end + instruction.tile_N_granularity - 1) // instruction.tile_N_granularity
            for block_idx in range(block_start_idx, block_end_idx):
                barriers[block_idx // blocks_per_head] += 1
        else:
            block_idx = instruction.tile_M_idx
            barriers[block_idx // blocks_per_head] += 1

        chunk_start = chunk_end


def rms_lmhead_gemv(globals: Globals, instruction: RMS_LMHead_GEMV):
    op_barriers = globals.barriers[
        LlamaConfig.num_hidden_layers - 1, _prev_barrier_row(instruction)
    ]
    # Expected: every output tile from every final-layer split-K has arrived.
    assert op_barriers[0] == DOWNPROJ_BARRIER_COUNT

    post_ln = _rms_norm(
        inp=globals.hidden_states,
        weight=globals.lm_head_norm_weights,
        eps=LlamaConfig.rms_norm_eps,
    )

    for block_idx in range(
        instruction.start_output_block_idx, instruction.end_output_block_idx
    ):
        start, end = _range_from_idx(block_idx, instruction.tile_N_granularity)

        matmul_output = einsum(
            globals.lm_head_weights[start:end],
            post_ln,
            "o i, i -> o",
        )

        globals.logits[start:end] = matmul_output


def decode_attention_split_kv(globals: Globals, instruction: DecodeAttentionSplitKV):
    gqa_ratio = LlamaConfig.num_attention_heads // LlamaConfig.num_key_value_heads

    # Barrier check
    op_barriers = globals.barriers[instruction.layer_idx, _prev_barrier_row(instruction)]
    blocks_per_head = LlamaConfig.head_dim // TILE_N_GRANULARITY
    for i in range(gqa_ratio):
        assert op_barriers[instruction.kv_head_idx * gqa_ratio + i] == blocks_per_head
    assert op_barriers[LlamaConfig.num_attention_heads + instruction.kv_head_idx] == blocks_per_head
    assert (
        op_barriers[
            LlamaConfig.num_attention_heads
            + LlamaConfig.num_key_value_heads
            + instruction.kv_head_idx
        ]
        == blocks_per_head
    )

    kv_block_size = LlamaConfig.attn_kv_block_size
    seq_len = (globals.position_id or 0) + 1
    layer_idx = instruction.layer_idx
    kv_head_idx = instruction.kv_head_idx

    total_blocks = math.ceil(seq_len / kv_block_size)
    blocks_per_partial = math.ceil(total_blocks / instruction.num_partials)

    start_block = instruction.partial_idx * blocks_per_partial
    end_block = min(start_block + blocks_per_partial, total_blocks)

    start_token = start_block * kv_block_size
    end_token = min(end_block * kv_block_size, seq_len)

    k = globals.k_cache[layer_idx, 0, start_token:end_token, kv_head_idx]
    v = globals.v_cache[layer_idx, 0, start_token:end_token, kv_head_idx]

    head_start = kv_head_idx * gqa_ratio
    head_end = head_start + gqa_ratio

    q = globals.q_post_rope.view(LlamaConfig.num_attention_heads, -1)[
        head_start:head_end
    ]

    qk = einsum(q.float(), k.float(), "h i, k i -> h k")
    scaled_qk = qk * (1 / math.sqrt(LlamaConfig.head_dim))

    # casting the output of the softmax to 16-bit causes small numerical differences
    softmax = torch.softmax(scaled_qk, dim=-1)

    lse = torch.log2(torch.sum(torch.exp(scaled_qk), dim=-1))

    out = einsum(softmax.float(), v.float(), "h k, k o -> h o")

    globals.attn_lse_intermediates[head_start:head_end, instruction.partial_idx] = (
        lse
    )
    globals.attn_out_intermediates[head_start:head_end, instruction.partial_idx] = (
        out
    )

    # Barrier update
    barriers = globals.barriers[
        instruction.layer_idx, BARRIER_ROW(instruction.opcode())
    ]
    barriers[head_start:head_end] += 1


def decode_attention_reduction(globals: Globals, instruction: DecodeAttentionReduction):
    head_start_idx = instruction.head_start_idx

    # Barrier check
    op_barriers = globals.barriers[instruction.layer_idx, _prev_barrier_row(instruction)]
    assert op_barriers[head_start_idx] == instruction.num_partials

    indices_to_reduce = torch.tensor(
        instruction.reduction_list,
        dtype=torch.long,
        device=globals.hidden_states.device,
    )

    lses = globals.attn_lse_intermediates[
        head_start_idx : head_start_idx + LlamaConfig.attn_reduction_size,
        indices_to_reduce,
    ]
    outs = globals.attn_out_intermediates[
        head_start_idx : head_start_idx + LlamaConfig.attn_reduction_size,
        indices_to_reduce,
    ]

    max_lse = torch.max(lses, dim=-1, keepdim=True).values

    adjusted_factors = (lses - max_lse).exp2()
    new_denominator = adjusted_factors.sum(dim=-1, keepdim=True)

    reduced = (outs * adjusted_factors.unsqueeze(-1)).sum(dim=1) / new_denominator

    if instruction.is_terminal:
        globals.attn_out.view(LlamaConfig.num_attention_heads, -1)[
            head_start_idx : head_start_idx + LlamaConfig.attn_reduction_size
        ] = reduced
    else:
        new_lse = new_denominator.log()
        output_slot = instruction.output_partial_idx
        globals.attn_lse_intermediates[
            head_start_idx : head_start_idx + LlamaConfig.attn_reduction_size,
            output_slot,
        ] = new_lse
        globals.attn_out_intermediates[
            head_start_idx : head_start_idx + LlamaConfig.attn_reduction_size,
            output_slot,
        ] = reduced

    # Barrier update
    next_op_barriers = globals.barriers[
        instruction.layer_idx, BARRIER_ROW(instruction.opcode())
    ]
    next_op_barriers[0] += LlamaConfig.attn_reduction_size  # the dumb way


INSTRUCTION_TO_HANDLER = {
    RMS_QKVProj_GEMV_RoPE: rms_qkvproj_gemv_rope,
    DecodeAttentionSplitKV: decode_attention_split_kv,
    DecodeAttentionReduction: decode_attention_reduction,
    OProj_GEMV_Residual: oproj_gemv_residual,
    RMS_UpGateProj_GEMV_SiLU: rms_upgate_proj_gemv_silu,
    DownProj_GEMV_Residual: down_proj_residual,
    RMS_LMHead_GEMV: rms_lmhead_gemv,
}
