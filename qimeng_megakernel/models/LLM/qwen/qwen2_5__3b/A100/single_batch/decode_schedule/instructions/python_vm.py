"""Instruction handlers for Qwen2.5 3B A100 single-batch decode."""
import math

import torch
import torch.nn.functional as F
from einops import einsum
from torch import Tensor
from qimeng_megakernel.models.instruction import Instruction

from ..globs import Globals
from ..model_config import Qwen2Config
from .instruction_defs import (
    RMS_QKVProj_GEMV_RoPE,
    DecodeAttentionSplitKV,
    DecodeAttentionReduction,
    OProj_GEMV_Residual,
    RMS_UpGateProj_GEMV_SiLU,
    DownProj_GEMV_Residual,
    RMS_LMHead_GEMV,
    TILE_N_GRANULARITY,
    SPLIT_K_GRANULARITY,
    NUM_DOWNPROJ_SPLIT_K,
    UPGATE_BLOCKS_PER_SIGNAL_BUCKET,
    BARRIER_ROW,
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
    vec = vec.reshape(-1).to(mat.dtype)
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
    residual = residual.reshape(-1)
    residual[out_start:out_end] += gemv_out.to(residual.dtype)


def _rotate_half_interleaved(x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    new_x1 = -x2
    new_x2 = x1
    stacked = torch.stack((new_x1, new_x2), dim=-1)
    return stacked.view_as(x)


def _apply_rotary_pos_emb_interleaved(q, k, cos, sin, unsqueeze_dim=1):
    """Apply interleaved RoPE with the same layout as the kernel."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (_rotate_half_interleaved(q) * sin)
    k_embed = (k * cos) + (_rotate_half_interleaved(k) * sin)
    return q_embed, k_embed


def oproj_gemv_residual(globs: Globals, instruction: OProj_GEMV_Residual):
    op_barriers = globs.barriers[instruction.layer_idx, _prev_barrier_row(instruction)]
    assert op_barriers[0] == Qwen2Config.num_attention_heads
    assert instruction.reduction_block_idx == 0

    mat = globs.o_proj_weights[instruction.layer_idx]
    vec = globs.attn_out.reshape(-1)
    residual = globs.hidden_states.reshape(-1)
    in_full = 0, Qwen2Config.hidden_size
    next_op_barriers = globs.barriers[
        instruction.layer_idx, BARRIER_ROW(instruction.opcode())
    ]

    for block_idx in range(instruction.tile_N_start_idx, instruction.tile_N_end_idx):
        start, end = _range_from_idx(block_idx, instruction.tile_N_granularity)
        _gemv_with_residual(
            mat=mat,
            vec=vec,
            residual=residual,
            out_start=start,
            out_end=end,
            in_start=in_full[0],
            in_end=in_full[1],
        )
        next_op_barriers[0] += 1


def down_proj_residual(globs: Globals, instruction: DownProj_GEMV_Residual):
    op_barriers = globs.barriers[instruction.layer_idx, _prev_barrier_row(instruction)]
    assert (
        op_barriers[instruction.split_K_idx]
        == UPGATE_BLOCKS_PER_SIGNAL_BUCKET
    )

    mat = globs.down_proj_weights[instruction.layer_idx]
    vec = globs.silu_out.reshape(-1)
    residual = globs.hidden_states.reshape(-1)
    in_start = instruction.split_K_idx * SPLIT_K_GRANULARITY
    in_end = min(in_start + SPLIT_K_GRANULARITY, Qwen2Config.intermediate_size)
    next_op_barriers = globs.barriers[
        instruction.layer_idx, BARRIER_ROW(instruction.opcode())
    ]

    for block_idx in range(instruction.tile_N_start_idx, instruction.tile_N_end_idx):
        start, end = _range_from_idx(block_idx, instruction.tile_N_granularity)
        _gemv_with_residual(
            mat=mat,
            vec=vec,
            residual=residual,
            out_start=start,
            out_end=end,
            in_start=in_start,
            in_end=in_end,
        )
    next_op_barriers[0] += instruction.tile_N_end_idx - instruction.tile_N_start_idx


def rms_upgate_proj_gemv_silu(globs: Globals, instruction: RMS_UpGateProj_GEMV_SiLU):
    op_barriers = globs.barriers[instruction.layer_idx, _prev_barrier_row(instruction)]
    expected_o_proj_blocks = Qwen2Config.hidden_size // instruction.tile_N_granularity
    assert op_barriers[0] == expected_o_proj_blocks

    post_ln = _rms_norm(
        inp=globs.hidden_states,
        weight=globs.mlp_norm_weights[instruction.layer_idx],
        eps=Qwen2Config.rms_norm_eps,
    )
    barriers = globs.barriers[
        instruction.layer_idx, BARRIER_ROW(instruction.opcode())
    ]
    post_ln_flat = post_ln.reshape(-1)

    for block_idx in instruction.tile_N_block_idxs:
        start, end = _range_from_idx(block_idx, instruction.tile_N_granularity)
        up_matvec = _gemv_range(
            mat=globs.up_proj_weights[instruction.layer_idx],
            vec=post_ln_flat,
            out_start=start,
            out_end=end,
        )
        gate_matvec = _gemv_range(
            mat=globs.gate_proj_weights[instruction.layer_idx],
            vec=post_ln_flat,
            out_start=start,
            out_end=end,
        )
        post_silu = F.silu(gate_matvec) * up_matvec
        globs.silu_out.reshape(-1)[start:end] = post_silu.to(globs.silu_out.dtype)
        bucket = block_idx // UPGATE_BLOCKS_PER_SIGNAL_BUCKET
        barriers[bucket] += 1


def rms_qkvproj_gemv_rope(globs: Globals, instruction: RMS_QKVProj_GEMV_RoPE):
    layer_idx = instruction.layer_idx

    if layer_idx > 0:
        op_barriers = globs.barriers[layer_idx - 1, _prev_barrier_row(instruction)]
        expected_prev_downproj_blocks = (Qwen2Config.hidden_size // TILE_N_GRANULARITY) * NUM_DOWNPROJ_SPLIT_K
        actual = op_barriers[0].item() if op_barriers.dim() > 0 else op_barriers.item()
        assert actual == expected_prev_downproj_blocks, (
            f"layer_idx={layer_idx} prev barrier: expected {expected_prev_downproj_blocks}, got {actual}"
        )

    post_ln = _rms_norm(
        inp=globs.hidden_states,
        weight=globs.attn_norm_weights[layer_idx],
        eps=Qwen2Config.rms_norm_eps,
    )
    pos_id = int(globs.position_id) if globs.position_id is not None else 0

    k_start = Qwen2Config.num_attention_heads * Qwen2Config.head_dim
    v_start = k_start + Qwen2Config.num_key_value_heads * Qwen2Config.head_dim

    barriers = globs.barriers[
        instruction.layer_idx, BARRIER_ROW(instruction.opcode())
    ]

    if instruction.tile_M_granularity == 1:
        start, end = _range_from_span(
            instruction.tile_N_start_idx,
            instruction.tile_N_end_idx,
            instruction.tile_N_granularity,
        )
        in_start, in_end = 0, Qwen2Config.hidden_size
    else:
        start, end = _range_from_idx(instruction.tile_M_idx, instruction.tile_M_granularity)
        in_start, in_end = _range_from_span(
            instruction.tile_N_start_idx,
            instruction.tile_N_end_idx,
            instruction.tile_N_granularity,
        )

    head_dim = Qwen2Config.head_dim
    blocks_per_head = head_dim // instruction.tile_N_granularity
    post_ln_flat = post_ln.reshape(-1)

    chunk_start = start
    while chunk_start < end:
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
        if seg_mode in ["q", "k"]:
            head_end = (chunk_start // head_dim + 1) * head_dim
            chunk_end = min(chunk_end, head_end)
        chunk_len = chunk_end - chunk_start

        matmul_output = _gemv_range(
            mat=globs.qkv_weights[layer_idx],
            vec=post_ln_flat,
            out_start=chunk_start,
            out_end=chunk_end,
            in_start=in_start,
            in_end=in_end,
        )
        if globs.qkv_bias is not None:
            if globs.qkv_bias.dim() == 2:
                bias_chunk = globs.qkv_bias[layer_idx, chunk_start:chunk_end].to(matmul_output.dtype)
            else:
                block_idx = chunk_start // instruction.tile_N_granularity
                bias_chunk = globs.qkv_bias[layer_idx, block_idx].to(matmul_output.dtype)[:chunk_len]
            out = matmul_output + bias_chunk.reshape(-1)
        else:
            out = matmul_output

        if seg_mode in ["q", "k"]:
            full_head = torch.zeros(1, head_dim, device=globs.hidden_states.device, dtype=out.dtype)
            head_segment = chunk_start % head_dim
            full_head_start = head_segment
            full_head_end = full_head_start + chunk_len
            full_head[:, full_head_start:full_head_end] = out.view(1, -1)
            full_head_with_rope, _ = _apply_rotary_pos_emb_interleaved(
                q=full_head,
                k=full_head,
                cos=globs.rope_cos[pos_id],
                sin=globs.rope_sin[pos_id],
                unsqueeze_dim=0,
            )
            out = full_head_with_rope[:, full_head_start:full_head_end].view(-1)

        if seg_mode == "q":
            globs.q_post_rope.reshape(-1)[chunk_start:chunk_end] = out.to(globs.q_post_rope.dtype)
        elif seg_mode == "k":
            start_in_k = chunk_start - k_start
            kv_head = start_in_k // head_dim
            in_head = start_in_k % head_dim
            if globs.k_cache.dim() == 5:
                globs.k_cache[layer_idx, 0, pos_id, kv_head, in_head : in_head + chunk_len] = out.to(globs.k_cache.dtype)
            else:
                globs.k_cache[layer_idx, pos_id, kv_head, in_head : in_head + chunk_len] = out.to(globs.k_cache.dtype)
        else:
            start_in_v = chunk_start - v_start
            kv_head = start_in_v // head_dim
            in_head = start_in_v % head_dim
            if globs.v_cache.dim() == 5:
                globs.v_cache[layer_idx, 0, pos_id, kv_head, in_head : in_head + chunk_len] = out.to(globs.v_cache.dtype)
            else:
                globs.v_cache[layer_idx, pos_id, kv_head, in_head : in_head + chunk_len] = out.to(globs.v_cache.dtype)

        if instruction.tile_M_granularity == 1:
            block_start_idx = chunk_start // instruction.tile_N_granularity
            block_end_idx = (chunk_end + instruction.tile_N_granularity - 1) // instruction.tile_N_granularity
            for block_idx in range(block_start_idx, block_end_idx):
                barriers[block_idx // blocks_per_head] += 1
        else:
            barriers[instruction.tile_M_idx // blocks_per_head] += 1
        chunk_start = chunk_end


def rms_lmhead_gemv(globs: Globals, instruction: RMS_LMHead_GEMV):
    op_barriers = globs.barriers[Qwen2Config.num_hidden_layers - 1, _prev_barrier_row(instruction)]
    expected_downproj_blocks = (Qwen2Config.hidden_size // TILE_N_GRANULARITY) * NUM_DOWNPROJ_SPLIT_K
    assert op_barriers[0].item() == expected_downproj_blocks

    post_ln = _rms_norm(
        inp=globs.hidden_states,
        weight=globs.lm_head_norm_weights,
        eps=Qwen2Config.rms_norm_eps,
    )

    for block_idx in range(instruction.start_output_block_idx, instruction.end_output_block_idx):
        start, end = _range_from_idx(block_idx, instruction.tile_N_granularity)
        matmul_output = einsum(
            globs.lm_head_weights[start:end],
            post_ln.reshape(-1),
            "o i, i -> o",
        )
        globs.logits.reshape(-1)[start:end] = matmul_output.to(globs.logits.dtype)


def decode_attention_split_kv(globs: Globals, instruction: DecodeAttentionSplitKV):
    """A100: split by KV head; one instruction handles gqa_ratio Q heads over a KV sequence slice."""
    gqa_ratio = Qwen2Config.num_attention_heads // Qwen2Config.num_key_value_heads
    op_barriers = globs.barriers[instruction.layer_idx, _prev_barrier_row(instruction)]
    blocks_per_head = Qwen2Config.head_dim // TILE_N_GRANULARITY

    kv_head_idx = instruction.kv_head_idx
    head_start = kv_head_idx * gqa_ratio
    head_end = head_start + gqa_ratio

    for i in range(gqa_ratio):
        assert op_barriers[head_start + i] == blocks_per_head
    assert op_barriers[Qwen2Config.num_attention_heads + kv_head_idx] == blocks_per_head
    assert op_barriers[Qwen2Config.num_attention_heads + Qwen2Config.num_key_value_heads + kv_head_idx] == blocks_per_head

    kv_block_size = Qwen2Config.attn_kv_block_size
    seq_len = pos_id = int(globs.position_id) if globs.position_id is not None else 0
    seq_len += 1
    layer_idx = instruction.layer_idx

    total_blocks = math.ceil(seq_len / kv_block_size)
    blocks_per_partial = math.ceil(total_blocks / instruction.num_partials)
    start_block = instruction.partial_idx * blocks_per_partial
    end_block = min(start_block + blocks_per_partial, total_blocks)
    start_token = start_block * kv_block_size
    end_token = min(end_block * kv_block_size, seq_len)

    if globs.k_cache.dim() == 5:
        k = globs.k_cache[layer_idx, 0, start_token:end_token, kv_head_idx, :]
        v = globs.v_cache[layer_idx, 0, start_token:end_token, kv_head_idx, :]
    else:
        k = globs.k_cache[layer_idx, start_token:end_token, kv_head_idx, :]
        v = globs.v_cache[layer_idx, start_token:end_token, kv_head_idx, :]

    q = globs.q_post_rope.view(Qwen2Config.num_attention_heads, -1)[head_start:head_end]

    qk = einsum(q.float(), k.float(), "h i, k i -> h k")
    scaled_qk = qk * (1 / math.sqrt(Qwen2Config.head_dim))
    softmax = torch.softmax(scaled_qk, dim=-1)
    lse = torch.logsumexp(scaled_qk.float(), dim=-1) / math.log(2)
    out = einsum(softmax.float(), v.float(), "h k, k o -> h o")

    if globs.attn_lse_intermediates.dim() == 2:
        globs.attn_lse_intermediates[head_start:head_end, instruction.partial_idx] = lse
    elif globs.attn_lse_intermediates.dim() == 4:
        globs.attn_lse_intermediates[0, 0, head_start:head_end, instruction.partial_idx] = lse
    else:
        globs.attn_lse_intermediates[0, head_start:head_end, instruction.partial_idx] = lse
    if globs.attn_out_intermediates.dim() == 3:
        globs.attn_out_intermediates[head_start:head_end, instruction.partial_idx, :] = out
    elif globs.attn_out_intermediates.dim() == 4:
        globs.attn_out_intermediates[0, head_start:head_end, instruction.partial_idx, :] = out
    else:
        globs.attn_out_intermediates[0, head_start:head_end, instruction.partial_idx, :] = out

    barriers = globs.barriers[
        instruction.layer_idx, BARRIER_ROW(instruction.opcode())
    ]
    barriers[head_start:head_end] += 1


def decode_attention_reduction(globs: Globals, instruction: DecodeAttentionReduction):
    head_start_idx = instruction.head_start_idx
    op_barriers = globs.barriers[instruction.layer_idx, _prev_barrier_row(instruction)]
    assert op_barriers[head_start_idx] == instruction.num_partials

    indices_to_reduce = torch.tensor(
        instruction.reduction_list,
        dtype=torch.long,
        device=globs.hidden_states.device,
    )
    size = Qwen2Config.attn_reduction_size
    if globs.attn_lse_intermediates.dim() == 2:
        lses = globs.attn_lse_intermediates[head_start_idx : head_start_idx + size, indices_to_reduce]
        outs = globs.attn_out_intermediates[head_start_idx : head_start_idx + size, indices_to_reduce, :]
    else:
        lses = globs.attn_lse_intermediates[0, 0, head_start_idx : head_start_idx + size, indices_to_reduce]
        outs = globs.attn_out_intermediates[0, head_start_idx : head_start_idx + size, indices_to_reduce, :]

    max_lse = torch.max(lses, dim=-1, keepdim=True).values
    adjusted_factors = (lses - max_lse).exp2()
    new_denominator = adjusted_factors.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    reduced = (outs * adjusted_factors.unsqueeze(-1)).sum(dim=1) / new_denominator

    if instruction.is_terminal:
        globs.attn_out.view(Qwen2Config.num_attention_heads, -1)[
            head_start_idx : head_start_idx + size
        ] = reduced.to(globs.attn_out.dtype)
    else:
        new_lse = new_denominator.log()
        output_slot = instruction.output_partial_idx
        if globs.attn_lse_intermediates.dim() == 2:
            globs.attn_lse_intermediates[head_start_idx : head_start_idx + size, output_slot] = new_lse.squeeze(-1)
            globs.attn_out_intermediates[head_start_idx : head_start_idx + size, output_slot, :] = reduced
        else:
            globs.attn_lse_intermediates[0, 0, head_start_idx : head_start_idx + size, output_slot] = new_lse.squeeze(-1)
            globs.attn_out_intermediates[0, head_start_idx : head_start_idx + size, output_slot, :] = reduced

    next_op_barriers = globs.barriers[
        instruction.layer_idx, BARRIER_ROW(instruction.opcode())
    ]
    next_op_barriers[0] += size


INSTRUCTION_TO_HANDLER = {
    RMS_QKVProj_GEMV_RoPE: rms_qkvproj_gemv_rope,
    DecodeAttentionSplitKV: decode_attention_split_kv,
    DecodeAttentionReduction: decode_attention_reduction,
    OProj_GEMV_Residual: oproj_gemv_residual,
    RMS_UpGateProj_GEMV_SiLU: rms_upgate_proj_gemv_silu,
    DownProj_GEMV_Residual: down_proj_residual,
    RMS_LMHead_GEMV: rms_lmhead_gemv,
}
