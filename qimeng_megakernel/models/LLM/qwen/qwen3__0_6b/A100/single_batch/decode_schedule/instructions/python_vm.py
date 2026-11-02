"""Qwen3-0.6B PyVM. Adapted from Qwen2.5-3B; key differences:
- rms_qkvproj_gemv_rope: no qkv_bias; applies Q-norm and K-norm (per-head RMSNorm) after GEMV
  and before RoPE. Each instruction covers exactly one head (128 elements).
- All config uses Qwen3Config (hidden=1024, intermediate=3072, 16Q/8KV heads, 28 layers).
- DownProj split-K geometry comes from the route instruction layout.
"""
import math

import torch
import torch.nn.functional as F
from einops import einsum
from torch import Tensor
from qimeng_megakernel.models.instruction import Instruction

from ..globs import Globals
from ..model_config import Qwen3Config
from .instruction_defs import (
    RMS_QKVProj_GEMV,
    DecodeAttentionSplitKV,
    DecodeAttentionReduction,
    OProj_GEMV_Residual,
    RMS_UpGateProj_GEMV_SiLU,
    DownProj_GEMV_Residual,
    RMS_LMHead_GEMV,
    TILE_N_GRANULARITY,
    SPLIT_K_GRANULARITY,
    NUM_DOWNPROJ_SPLIT_K,
    QKV_RAW_READY_SLOT_BASE,
    QKV_RAW_BLOCKS_PER_GROUP,
    qkv_raw_group_for_block,
    ATTENTION_PREPARED_SLOT_BASE,
    ATTENTION_PREPARE_PRODUCER_PARTIAL,
    UPGATE_BLOCKS_PER_SIGNAL_BUCKET,
)


def _prev_barrier_row(instruction: Instruction) -> int:
    p = instruction.prev_opcode()
    return (p[0] - 1) if isinstance(p, tuple) else (p - 1)


def _range_from_idx(idx: int, granularity: int):
    start = granularity * idx
    end = start + granularity
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
    return einsum(mat, vec, "o i, i -> o")


def _rms_norm(inp: Tensor, weight: Tensor, eps: float):
    input_dtype = inp.dtype
    inp = inp.to(torch.float32)
    variance = inp.pow(2).mean(-1, keepdim=True)
    inp = inp * torch.rsqrt(variance + eps)
    return weight * inp.to(input_dtype)


def _per_head_rms_norm(x: Tensor, weight: Tensor, eps: float) -> Tensor:
    """Apply RMSNorm per head. x: [n_heads, head_dim], weight: [head_dim]."""
    x_f32 = x.to(torch.float32)
    variance = x_f32.pow(2).mean(-1, keepdim=True)
    x_normed = x_f32 * torch.rsqrt(variance + eps)
    return (weight.to(torch.float32) * x_normed).to(x.dtype)


def _gemv_with_residual(
    mat: Tensor,
    vec: Tensor,
    residual: Tensor,
    out_start: int,
    out_end: int,
    in_start: int,
    in_end: int,
):
    gemv_out = _gemv_range(mat, vec, out_start, out_end, in_start, in_end)
    residual = residual.reshape(-1)
    residual[out_start:out_end] += gemv_out.to(residual.dtype)


def _rotate_half_interleaved(x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    stacked = torch.stack((-x2, x1), dim=-1)
    return stacked.view_as(x)


def _apply_rotary_pos_emb_interleaved(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (_rotate_half_interleaved(q) * sin)
    k_embed = (k * cos) + (_rotate_half_interleaved(k) * sin)
    return q_embed, k_embed


def rms_qkvproj_gemv(globs: Globals, instruction: RMS_QKVProj_GEMV):
    """RMS + QKV GEMV for tile range [start, end); writes to qkv_raw. No Q-norm, K-norm, or RoPE."""
    layer_idx = instruction.layer_idx

    if layer_idx > 0:
        op_barriers = globs.barriers[layer_idx - 1, _prev_barrier_row(instruction)]
        expected_prev = (Qwen3Config.hidden_size // TILE_N_GRANULARITY) * NUM_DOWNPROJ_SPLIT_K
        actual = op_barriers[0].item() if op_barriers.dim() > 0 else op_barriers.item()
        assert actual == expected_prev, (
            f"layer {layer_idx}: prev-layer DownProj barrier {actual} != {expected_prev}"
        )

    post_ln = _rms_norm(
        globs.hidden_states.reshape(-1),
        globs.attn_norm_weights[layer_idx],
        eps=Qwen3Config.rms_norm_eps,
    )
    start = instruction.tile_N_start_idx * TILE_N_GRANULARITY
    end = instruction.tile_N_end_idx * TILE_N_GRANULARITY

    post_ln_flat = post_ln.reshape(-1)
    out = _gemv_range(globs.qkv_weights[layer_idx], post_ln_flat, start, end)
    globs.qkv_raw.reshape(-1)[start:end] = out.to(globs.qkv_raw.dtype)

    barriers = globs.barriers[layer_idx, instruction.opcode() - 1]
    for block_idx in range(
        instruction.tile_N_start_idx, instruction.tile_N_end_idx
    ):
        group_idx = qkv_raw_group_for_block(block_idx)
        barriers[QKV_RAW_READY_SLOT_BASE + group_idx] += 1


# Alias for backward compat (instruction type is RMS_QKVProj_GEMV; same handler).
def rms_qkvproj_gemv_rope(globs: Globals, instruction: RMS_QKVProj_GEMV):
    rms_qkvproj_gemv(globs, instruction)


def oproj_gemv_residual(globs: Globals, instruction: OProj_GEMV_Residual):
    op_barriers = globs.barriers[instruction.layer_idx, _prev_barrier_row(instruction)]
    assert op_barriers[0] == Qwen3Config.num_attention_heads
    assert instruction.reduction_block_idx == 0

    mat = globs.o_proj_weights[instruction.layer_idx]
    vec = globs.attn_out.reshape(-1)
    residual = globs.hidden_states.reshape(-1)
    # Qwen3: n_heads * head_dim = 2048 ≠ hidden_size = 1024; in_full spans full attn output
    in_full = 0, Qwen3Config.num_attention_heads * Qwen3Config.head_dim
    next_op_barriers = globs.barriers[instruction.layer_idx, instruction.opcode() - 1]

    for block_idx in range(instruction.tile_N_start_idx, instruction.tile_N_end_idx):
        start, end = _range_from_idx(block_idx, instruction.tile_N_granularity)
        _gemv_with_residual(
            mat=mat, vec=vec, residual=residual,
            out_start=start, out_end=end,
            in_start=in_full[0], in_end=in_full[1],
        )
        next_op_barriers[0] += 1


def rms_upgate_proj_gemv_silu(globs: Globals, instruction: RMS_UpGateProj_GEMV_SiLU):
    op_barriers = globs.barriers[instruction.layer_idx, _prev_barrier_row(instruction)]
    expected_o_proj_blocks = Qwen3Config.hidden_size // instruction.tile_N_granularity
    assert op_barriers[0] == expected_o_proj_blocks

    post_ln = _rms_norm(
        inp=globs.hidden_states,
        weight=globs.mlp_norm_weights[instruction.layer_idx],
        eps=Qwen3Config.rms_norm_eps,
    )
    barriers = globs.barriers[instruction.layer_idx, instruction.opcode() - 1]
    post_ln_flat = post_ln.reshape(-1)

    for block_idx in instruction.tile_N_block_idxs:
        start, end = _range_from_idx(block_idx, instruction.tile_N_granularity)
        up_matvec = _gemv_range(globs.up_proj_weights[instruction.layer_idx], post_ln_flat, start, end)
        gate_matvec = _gemv_range(globs.gate_proj_weights[instruction.layer_idx], post_ln_flat, start, end)
        post_silu = F.silu(gate_matvec) * up_matvec
        globs.silu_out.reshape(-1)[start:end] = post_silu.to(globs.silu_out.dtype)
        signal_bucket = block_idx // UPGATE_BLOCKS_PER_SIGNAL_BUCKET
        barriers[signal_bucket] += 1


def down_proj_residual(globs: Globals, instruction: DownProj_GEMV_Residual):
    op_barriers = globs.barriers[instruction.layer_idx, _prev_barrier_row(instruction)]
    expected_upgate_blocks = UPGATE_BLOCKS_PER_SIGNAL_BUCKET
    assert op_barriers[instruction.split_K_idx] == expected_upgate_blocks

    mat = globs.down_proj_weights[instruction.layer_idx]
    vec = globs.silu_out.reshape(-1)
    residual = globs.hidden_states.reshape(-1)
    in_start = instruction.split_K_idx * SPLIT_K_GRANULARITY
    in_end = min(in_start + SPLIT_K_GRANULARITY, Qwen3Config.intermediate_size)
    next_op_barriers = globs.barriers[instruction.layer_idx, instruction.opcode() - 1]

    for block_idx in range(instruction.tile_N_start_idx, instruction.tile_N_end_idx):
        start, end = _range_from_idx(block_idx, instruction.tile_N_granularity)
        _gemv_with_residual(
            mat=mat, vec=vec, residual=residual,
            out_start=start, out_end=end,
            in_start=in_start, in_end=in_end,
        )
    next_op_barriers[0] += instruction.tile_N_end_idx - instruction.tile_N_start_idx


def decode_attention_split_kv(globs: Globals, instruction: DecodeAttentionSplitKV):
    """A100: split by KV head. Reads raw QKV from qkv_raw, applies Q-norm/K-norm/RoPE inline, then attention."""
    n_heads = Qwen3Config.num_attention_heads
    n_kv = Qwen3Config.num_key_value_heads
    head_dim = Qwen3Config.head_dim
    gqa_ratio = n_heads // n_kv  # 2
    k_start_elem = n_heads * head_dim
    v_start_elem = k_start_elem + n_kv * head_dim
    eps = Qwen3Config.rms_norm_eps

    op_barriers = globs.barriers[instruction.layer_idx, _prev_barrier_row(instruction)]
    raw_ready_slot = QKV_RAW_READY_SLOT_BASE + instruction.kv_head_idx
    assert op_barriers[raw_ready_slot].item() == QKV_RAW_BLOCKS_PER_GROUP, (
        f"QKV group {instruction.kv_head_idx} barrier "
        f"{op_barriers[raw_ready_slot].item()} != {QKV_RAW_BLOCKS_PER_GROUP}"
    )

    pos_id = int(globs.position_id) if globs.position_id is not None else 0
    seq_len = pos_id + 1
    layer_idx = instruction.layer_idx
    kv_head_idx = instruction.kv_head_idx
    head_start = kv_head_idx * gqa_ratio
    head_end = head_start + gqa_ratio

    kv_block_size = Qwen3Config.attn_kv_block_size
    total_blocks = math.ceil(seq_len / kv_block_size)
    blocks_per_partial = math.ceil(total_blocks / instruction.num_partials)
    start_block = instruction.partial_idx * blocks_per_partial
    end_block = min(start_block + blocks_per_partial, total_blocks)
    start_token = start_block * kv_block_size
    end_token = min(end_block * kv_block_size, seq_len)

    # 1. The unique partial-0 producer prepares this KV group's two Q heads
    #    and one K/V head, then publishes the prepared slot.
    qkv_raw = globs.qkv_raw.reshape(-1)
    prepared_slot = ATTENTION_PREPARED_SLOT_BASE + kv_head_idx
    if instruction.partial_idx == ATTENTION_PREPARE_PRODUCER_PARTIAL:
        raw_q = qkv_raw[
            head_start * head_dim : head_end * head_dim
        ].view(gqa_ratio, head_dim)
        q_normed = _per_head_rms_norm(
            raw_q, globs.q_norm_weights[layer_idx], eps
        )
        q_for_rope = q_normed.unsqueeze(0)
        q_rope, _ = _apply_rotary_pos_emb_interleaved(
            q_for_rope,
            q_for_rope,
            globs.rope_cos[pos_id],
            globs.rope_sin[pos_id],
            unsqueeze_dim=0,
        )
        globs.q_post_rope.reshape(n_heads, head_dim)[head_start:head_end] = (
            q_rope.squeeze(0).to(globs.q_post_rope.dtype)
        )

        k_offset = k_start_elem + kv_head_idx * head_dim
        raw_k = qkv_raw[k_offset : k_offset + head_dim].view(1, head_dim)
        k_normed = _per_head_rms_norm(raw_k, globs.k_norm_weights[layer_idx], eps)
        k_for_rope = k_normed.unsqueeze(0)
        _, k_rope = _apply_rotary_pos_emb_interleaved(
            k_for_rope, k_for_rope,
            globs.rope_cos[pos_id], globs.rope_sin[pos_id],
            unsqueeze_dim=0,
        )
        k_slot = k_rope.squeeze(0).to(globs.k_cache.dtype)
        if globs.k_cache.dim() == 5:
            globs.k_cache[layer_idx, 0, pos_id, kv_head_idx, :] = k_slot
        else:
            globs.k_cache[layer_idx, pos_id, kv_head_idx, :] = k_slot

        v_offset = v_start_elem + kv_head_idx * head_dim
        raw_v = qkv_raw[v_offset : v_offset + head_dim].to(globs.v_cache.dtype)
        if globs.v_cache.dim() == 5:
            globs.v_cache[layer_idx, 0, pos_id, kv_head_idx, :] = raw_v
        else:
            globs.v_cache[layer_idx, pos_id, kv_head_idx, :] = raw_v

        op_barriers[prepared_slot] += 1

    assert op_barriers[prepared_slot].item() == 1
    q = globs.q_post_rope.reshape(n_heads, head_dim)[head_start:head_end].float()

    # 2. Attention over [start_token, end_token)
    if globs.k_cache.dim() == 5:
        k = globs.k_cache[layer_idx, 0, start_token:end_token, kv_head_idx, :]
        v = globs.v_cache[layer_idx, 0, start_token:end_token, kv_head_idx, :]
    else:
        k = globs.k_cache[layer_idx, start_token:end_token, kv_head_idx, :]
        v = globs.v_cache[layer_idx, start_token:end_token, kv_head_idx, :]

    qk = einsum(q.float(), k.float(), "h i, k i -> h k")
    scaled_qk = qk * (1 / math.sqrt(Qwen3Config.head_dim))
    softmax = torch.softmax(scaled_qk, dim=-1)
    lse = torch.logsumexp(scaled_qk.float(), dim=-1) / math.log(2)
    out = einsum(softmax.float(), v.float(), "h k, k o -> h o")

    if globs.attn_lse_intermediates.dim() == 2:
        globs.attn_lse_intermediates[head_start:head_end, instruction.partial_idx] = lse
    else:
        globs.attn_lse_intermediates[0, 0, head_start:head_end, instruction.partial_idx] = lse
    if globs.attn_out_intermediates.dim() == 3:
        globs.attn_out_intermediates[head_start:head_end, instruction.partial_idx, :] = out
    else:
        globs.attn_out_intermediates[0, head_start:head_end, instruction.partial_idx, :] = out

    barriers = globs.barriers[instruction.layer_idx, instruction.opcode() - 1]
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
    size = Qwen3Config.attn_reduction_size
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
        globs.attn_out.view(Qwen3Config.num_attention_heads, -1)[
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

    next_op_barriers = globs.barriers[instruction.layer_idx, instruction.opcode() - 1]
    next_op_barriers[0] += size


def rms_lmhead_gemv(globs: Globals, instruction: RMS_LMHead_GEMV):
    op_barriers = globs.barriers[Qwen3Config.num_hidden_layers - 1, _prev_barrier_row(instruction)]
    expected_downproj_blocks = (Qwen3Config.hidden_size // TILE_N_GRANULARITY) * NUM_DOWNPROJ_SPLIT_K
    assert op_barriers[0].item() == expected_downproj_blocks

    post_ln = _rms_norm(
        inp=globs.hidden_states,
        weight=globs.lm_head_norm_weights,
        eps=Qwen3Config.rms_norm_eps,
    )

    for block_idx in range(instruction.start_output_block_idx, instruction.end_output_block_idx):
        start, end = _range_from_idx(block_idx, instruction.tile_N_granularity)
        matmul_output = einsum(
            globs.lm_head_weights[start:end],
            post_ln.reshape(-1),
            "o i, i -> o",
        )
        globs.logits.reshape(-1)[start:end] = matmul_output.to(globs.logits.dtype)


INSTRUCTION_TO_HANDLER = {
    RMS_QKVProj_GEMV: rms_qkvproj_gemv,
    DecodeAttentionSplitKV: decode_attention_split_kv,
    DecodeAttentionReduction: decode_attention_reduction,
    OProj_GEMV_Residual: oproj_gemv_residual,
    RMS_UpGateProj_GEMV_SiLU: rms_upgate_proj_gemv_silu,
    DownProj_GEMV_Residual: down_proj_residual,
    RMS_LMHead_GEMV: rms_lmhead_gemv,
}
