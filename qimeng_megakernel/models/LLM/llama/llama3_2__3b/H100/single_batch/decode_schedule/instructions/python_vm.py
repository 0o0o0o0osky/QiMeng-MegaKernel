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
    TILE_N_GRANULARITY,
    DOWNPROJ_SPLIT_K_STARTS,
    DOWNPROJ_SPLIT_K_SIZES,
    UPGATE_BLOCKS_PER_SIGNAL_BUCKET,
    DOWNPROJ_BARRIER_COUNT,
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
    # This split consumes exactly one UpGate segment, so it waits only for the
    # matching per-segment slot.  The ragged tail has 128 arrivals; the two full
    # segments have 192 each.
    op_barriers = globals.barriers[instruction.layer_idx, _prev_barrier_row(instruction)]
    split_idx = instruction.split_K_idx
    expected_upgate_blocks = UPGATE_BLOCKS_PER_SIGNAL_BUCKET[split_idx]
    assert op_barriers[split_idx] == expected_upgate_blocks, (
        f"downproj split_K={split_idx} expected {expected_upgate_blocks} "
        f"upgate arrivals in slot {split_idx}, got {op_barriers[split_idx]}"
    )

    mat = globals.down_proj_weights[instruction.layer_idx]
    vec = globals.silu_out
    residual = globals.hidden_states
    in_start = DOWNPROJ_SPLIT_K_STARTS[split_idx]
    in_end = in_start + DOWNPROJ_SPLIT_K_SIZES[split_idx]
    assert in_end <= LlamaConfig.intermediate_size
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
            in_start=in_start,
            in_end=in_end,
        )
        residual[start:end] += gemv_out.to(residual.dtype)

    # Each instruction contributes its output span; all three splits cover 192
    # output tiles, so opcode 6 slot 0 reaches 576.
    next_op_barriers[0] += instruction.tile_N_end_idx - instruction.tile_N_start_idx


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

        bucket = block_idx // UPGATE_BLOCKS_PER_SIGNAL_BUCKET[0]
        barriers[bucket] += 1


def rms_qkvproj_gemv_rope(
    globals: Globals, instruction: RMS_QKVProj_GEMV_RoPE
):
    layer_idx = instruction.layer_idx

    # Verify that the previous layer's down projection has completed.
    if layer_idx > 0:
        op_barriers = globals.barriers[layer_idx - 1, _prev_barrier_row(instruction)]
        expected_prev_downproj_blocks = DOWNPROJ_BARRIER_COUNT
        actual = op_barriers[0].item() if op_barriers.dim() > 0 else op_barriers.item()
        assert actual == expected_prev_downproj_blocks, (
            f"layer_idx={layer_idx} prev barrier row {_prev_barrier_row(instruction)}: "
            f"expected {expected_prev_downproj_blocks}, got {actual}"
        )

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
    # Process 16 (tile_N_granularity) elements at a time so full_head slice never exceeds head boundary.
    chunk_start = start
    while chunk_start < end:
        # Clip chunk_end: at most tile_N_granularity per step, and stay within one segment (Q/K/V) and within one head
        chunk_end_cand = min(chunk_start + instruction.tile_N_granularity, end)
        if chunk_start < k_start:
            seg_mode = "q"
            chunk_end = min(chunk_end_cand, k_start)
        elif chunk_start < v_start:
            seg_mode = "k"
            chunk_end = min(chunk_end_cand, v_start)
        else:
            seg_mode = "v"
            chunk_end = chunk_end_cand
        # Stay within current head for RoPE (full_head has head_dim columns)
        if seg_mode in ["q", "k"]:
            head_end = (chunk_start // head_dim + 1) * head_dim
            chunk_end = min(chunk_end, head_end)
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
    # The language-model head depends on the final layer's down projection.
    expected_downproj_blocks = DOWNPROJ_BARRIER_COUNT
    assert op_barriers[0].item() == expected_downproj_blocks

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

    # Barrier check: one Q head's Q/K/V blocks ready
    op_barriers = globals.barriers[instruction.layer_idx, _prev_barrier_row(instruction)]
    blocks_per_head = LlamaConfig.head_dim // TILE_N_GRANULARITY
    abs_head = instruction.attention_head_idx
    assert op_barriers[abs_head] == blocks_per_head
    kv_head_idx = abs_head // gqa_ratio
    assert op_barriers[LlamaConfig.num_attention_heads + kv_head_idx] == blocks_per_head
    assert (
        op_barriers[
            LlamaConfig.num_attention_heads
            + LlamaConfig.num_key_value_heads
            + kv_head_idx
        ]
        == blocks_per_head
    )

    kv_block_size = LlamaConfig.attn_kv_block_size
    seq_len = (globals.position_id or 0) + 1
    layer_idx = instruction.layer_idx
    kv_head_idx = instruction.attention_head_idx // gqa_ratio

    total_blocks = math.ceil(seq_len / kv_block_size)
    blocks_per_partial = math.ceil(total_blocks / instruction.num_partials)

    start_block = instruction.partial_idx * blocks_per_partial
    end_block = min(start_block + blocks_per_partial, total_blocks)

    start_token = start_block * kv_block_size
    end_token = min(end_block * kv_block_size, seq_len)

    k = globals.k_cache[layer_idx, 0, start_token:end_token, kv_head_idx]
    v = globals.v_cache[layer_idx, 0, start_token:end_token, kv_head_idx]

    # Single Q head (per-Q-head)
    abs_head = instruction.attention_head_idx
    q = globals.q_post_rope.view(LlamaConfig.num_attention_heads, -1)[abs_head : abs_head + 1]

    qk = einsum(q.float(), k.float(), "h i, k i -> h k")
    scaled_qk = qk * (1 / math.sqrt(LlamaConfig.head_dim))

    softmax = torch.softmax(scaled_qk, dim=-1)

    lse = torch.log2(torch.sum(torch.exp(scaled_qk), dim=-1))

    out = einsum(softmax.float(), v.float(), "h k, k o -> h o")

    globals.attn_lse_intermediates[abs_head, instruction.partial_idx] = lse.squeeze(0)
    globals.attn_out_intermediates[abs_head, instruction.partial_idx] = out.squeeze(0)

    # Barrier update: one head done for this partial
    barriers = globals.barriers[
        instruction.layer_idx, BARRIER_ROW(instruction.opcode())
    ]
    barriers[abs_head] += 1


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
