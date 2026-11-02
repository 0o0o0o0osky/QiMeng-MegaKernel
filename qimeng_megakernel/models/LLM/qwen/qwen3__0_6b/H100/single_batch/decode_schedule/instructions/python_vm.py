"""Qwen3 0.6B H100 Python VM."""
import math

import torch
import torch.nn.functional as F
from einops import einsum
from torch import Tensor
from qimeng_megakernel.models.instruction import Instruction

from ..globs import Globals
from ..model_config import Qwen3Config
from .instruction_defs import (
    BARRIER_ROW,
    RMS_QKVProj_GEMV,
    DecodeAttentionSplitKV,
    DecodeAttentionReduction,
    OProj_GEMV_Residual,
    RMS_UpGateProj_GEMV_SiLU,
    DownProj_GEMV_Residual,
    RMS_LMHead_GEMV,
    RMS_QKV_BLOCK_SIZE,
    ATTENTION_PREPARE_PRODUCER_PARTIAL,
    TILE_N_GRANULARITY,
    NUM_DOWNPROJ_SPLIT_K,
    SPLIT_K_GRANULARITY,
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


def _rms_norm_float(inp: Tensor, weight: Tensor, eps: float) -> Tensor:
    """Native Op2 RMS: keep the normalized value and weight product in fp32."""
    inp_float = inp.to(torch.float32)
    variance = inp_float.pow(2).mean(-1, keepdim=True)
    return weight.to(torch.float32) * (
        inp_float * torch.rsqrt(variance + eps)
    )


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
    """Interleaved RoPE; out_even = q*cos - q_odd*sin, out_odd = q*cos + q_even*sin."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (_rotate_half_interleaved(q) * sin)
    k_embed = (k * cos) + (_rotate_half_interleaved(k) * sin)
    return q_embed, k_embed


def oproj_gemv_residual(globs: Globals, instruction: OProj_GEMV_Residual):
    op_barriers = globs.barriers[instruction.layer_idx, _prev_barrier_row(instruction)]
    assert op_barriers[0] == Qwen3Config.num_attention_heads
    assert instruction.reduction_block_idx == 0

    mat = globs.o_proj_weights[instruction.layer_idx]
    vec = globs.attn_out.reshape(-1)
    residual = globs.hidden_states.reshape(-1)
    attn_size = Qwen3Config.num_attention_heads * Qwen3Config.head_dim
    in_full = 0, attn_size
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
    assert 0 <= instruction.split_K_idx < NUM_DOWNPROJ_SPLIT_K
    op_barriers = globs.barriers[instruction.layer_idx, _prev_barrier_row(instruction)]
    assert op_barriers[instruction.split_K_idx] == UPGATE_BLOCKS_PER_SIGNAL_BUCKET

    mat = globs.down_proj_weights[instruction.layer_idx]
    vec = globs.silu_out.reshape(-1)
    residual = globs.hidden_states.reshape(-1)
    assert instruction.split_K_granularity == SPLIT_K_GRANULARITY
    in_start = instruction.split_K_idx * SPLIT_K_GRANULARITY
    in_end = min(in_start + SPLIT_K_GRANULARITY, Qwen3Config.intermediate_size)
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
    expected_o_proj_blocks = Qwen3Config.hidden_size // instruction.tile_N_granularity
    assert op_barriers[0] == expected_o_proj_blocks

    post_ln = _rms_norm(
        inp=globs.hidden_states,
        weight=globs.mlp_norm_weights[instruction.layer_idx],
        eps=Qwen3Config.rms_norm_eps,
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
        barriers[block_idx // UPGATE_BLOCKS_PER_SIGNAL_BUCKET] += 1


def _rms_qkv_barrier_slot(block_idx: int) -> int:
    """Map one native QKV output block to its Q/K/V head barrier slot."""
    blocks_per_head = Qwen3Config.head_dim // RMS_QKV_BLOCK_SIZE
    q_blocks = (
        Qwen3Config.num_attention_heads
        * Qwen3Config.head_dim
        // RMS_QKV_BLOCK_SIZE
    )
    kv_blocks = (
        Qwen3Config.num_key_value_heads
        * Qwen3Config.head_dim
        // RMS_QKV_BLOCK_SIZE
    )
    if block_idx < q_blocks:
        return block_idx // blocks_per_head
    if block_idx < q_blocks + kv_blocks:
        return (
            Qwen3Config.num_attention_heads
            + (block_idx - q_blocks) // blocks_per_head
        )
    return (
        Qwen3Config.num_attention_heads
        + Qwen3Config.num_key_value_heads
        + (block_idx - q_blocks - kv_blocks) // blocks_per_head
    )


def rms_qkvproj_gemv(globs: Globals, instruction: RMS_QKVProj_GEMV) -> None:
    """Execute native Op1 semantics: RMSNorm + QKV GEMV into ``qkv_raw``."""
    layer_idx = instruction.layer_idx

    if layer_idx > 0:
        op_barriers = globs.barriers[layer_idx - 1, _prev_barrier_row(instruction)]
        expected_prev_downproj_blocks = (
            Qwen3Config.hidden_size // TILE_N_GRANULARITY
        ) * NUM_DOWNPROJ_SPLIT_K
        actual = op_barriers[0].item() if op_barriers.dim() > 0 else op_barriers.item()
        assert actual == expected_prev_downproj_blocks, (
            f"layer_idx={layer_idx} prev barrier: expected {expected_prev_downproj_blocks}, got {actual}"
        )

    assert instruction.tile_N_granularity == RMS_QKV_BLOCK_SIZE
    post_ln = _rms_norm(
        inp=globs.hidden_states,
        weight=globs.attn_norm_weights[layer_idx],
        eps=Qwen3Config.rms_norm_eps,
    )
    start, end = _range_from_span(
        instruction.tile_N_start_idx,
        instruction.tile_N_end_idx,
        RMS_QKV_BLOCK_SIZE,
    )
    assert 0 <= start < end <= Qwen3Config.qkv_out_dim
    output = _gemv_range(
        mat=globs.qkv_weights[layer_idx],
        vec=post_ln.reshape(-1),
        out_start=start,
        out_end=end,
    )
    globs.qkv_raw.reshape(-1)[start:end].copy_(output.to(globs.qkv_raw.dtype))

    barriers = globs.barriers[layer_idx, BARRIER_ROW(instruction.opcode())]
    for block_idx in range(
        instruction.tile_N_start_idx,
        instruction.tile_N_end_idx,
    ):
        barriers[_rms_qkv_barrier_slot(block_idx)] += 1

def rms_lmhead_gemv(globs: Globals, instruction: RMS_LMHead_GEMV):
    op_barriers = globs.barriers[Qwen3Config.num_hidden_layers - 1, _prev_barrier_row(instruction)]
    expected_downproj_blocks = (
        Qwen3Config.hidden_size // TILE_N_GRANULARITY
    ) * NUM_DOWNPROJ_SPLIT_K
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


def _qknorm_rope_writecache_for_attention_head(
    globs: Globals,
    *,
    layer_idx: int,
    attention_head_idx: int,
) -> None:
    """Mirror the native Op2 pre-attention fusion for one Q/GQA head."""
    head_dim = Qwen3Config.head_dim
    gqa_ratio = Qwen3Config.num_attention_heads // Qwen3Config.num_key_value_heads
    kv_head_idx = attention_head_idx // gqa_ratio
    q_dim = Qwen3Config.num_attention_heads * head_dim
    kv_dim = Qwen3Config.num_key_value_heads * head_dim
    q_start = attention_head_idx * head_dim
    k_start = q_dim + kv_head_idx * head_dim
    v_start = q_dim + kv_dim + kv_head_idx * head_dim
    pos_id = int(globs.position_id) if globs.position_id is not None else 0

    qkv_raw = globs.qkv_raw.reshape(-1)
    q = _rms_norm_float(
        qkv_raw[q_start : q_start + head_dim].view(1, head_dim),
        globs.q_norm_weights[layer_idx],
        Qwen3Config.rms_norm_eps,
    )
    k = _rms_norm_float(
        qkv_raw[k_start : k_start + head_dim].view(1, head_dim),
        globs.k_norm_weights[layer_idx],
        Qwen3Config.rms_norm_eps,
    )
    q_rope, k_rope = _apply_rotary_pos_emb_interleaved(
        q,
        k,
        globs.rope_cos[pos_id],
        globs.rope_sin[pos_id],
        unsqueeze_dim=0,
    )

    globs.q_post_rope.reshape(Qwen3Config.num_attention_heads, head_dim)[
        attention_head_idx
    ].copy_(q_rope.reshape(-1).to(globs.q_post_rope.dtype))
    v = qkv_raw[v_start : v_start + head_dim]
    if globs.k_cache.dim() == 5:
        globs.k_cache[layer_idx, 0, pos_id, kv_head_idx, :].copy_(
            k_rope.reshape(-1).to(globs.k_cache.dtype)
        )
        globs.v_cache[layer_idx, 0, pos_id, kv_head_idx, :].copy_(
            v.to(globs.v_cache.dtype)
        )
    else:
        globs.k_cache[layer_idx, pos_id, kv_head_idx, :].copy_(
            k_rope.reshape(-1).to(globs.k_cache.dtype)
        )
        globs.v_cache[layer_idx, pos_id, kv_head_idx, :].copy_(
            v.to(globs.v_cache.dtype)
        )


def decode_attention_split_kv(globs: Globals, instruction: DecodeAttentionSplitKV):
    gqa_ratio = Qwen3Config.num_attention_heads // Qwen3Config.num_key_value_heads
    op_barriers = globs.barriers[instruction.layer_idx, _prev_barrier_row(instruction)]
    blocks_per_head = Qwen3Config.head_dim // RMS_QKV_BLOCK_SIZE
    abs_head = instruction.attention_head_idx
    assert abs_head % gqa_ratio == 0
    kv_head_idx = abs_head // gqa_ratio
    if instruction.partial_idx == ATTENTION_PREPARE_PRODUCER_PARTIAL:
        for head_idx in range(abs_head, abs_head + gqa_ratio):
            assert op_barriers[head_idx] == blocks_per_head
        assert op_barriers[Qwen3Config.num_attention_heads + kv_head_idx] == blocks_per_head
        assert op_barriers[Qwen3Config.num_attention_heads + Qwen3Config.num_key_value_heads + kv_head_idx] == blocks_per_head

        for head_idx in range(abs_head, abs_head + gqa_ratio):
            _qknorm_rope_writecache_for_attention_head(
                globs,
                layer_idx=instruction.layer_idx,
                attention_head_idx=head_idx,
            )
            op_barriers[head_idx] += 1

    prepared_head_count = blocks_per_head + 1
    for head_idx in range(abs_head, abs_head + gqa_ratio):
        assert op_barriers[head_idx] == prepared_head_count

    kv_block_size = Qwen3Config.attn_kv_block_size
    seq_len = pos_id = int(globs.position_id) if globs.position_id is not None else 0
    seq_len += 1
    layer_idx = instruction.layer_idx
    kv_head_idx = instruction.attention_head_idx // gqa_ratio

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

    q = globs.q_post_rope.view(Qwen3Config.num_attention_heads, -1)[
        abs_head : abs_head + gqa_ratio
    ]

    qk = einsum(q.float(), k.float(), "h i, k i -> h k")
    scaled_qk = qk * (1 / math.sqrt(Qwen3Config.head_dim))
    softmax = torch.softmax(scaled_qk, dim=-1)
    lse = torch.logsumexp(scaled_qk.float(), dim=-1) / math.log(2)
    out = einsum(softmax.float(), v.float(), "h k, k o -> h o")

    if instruction.num_partials == 1:
        globs.attn_out.view(Qwen3Config.num_attention_heads, -1)[
            abs_head : abs_head + gqa_ratio
        ] = out.to(globs.attn_out.dtype)
        next_op_barriers = globs.barriers[
            instruction.layer_idx,
            BARRIER_ROW(DecodeAttentionReduction.opcode()),
        ]
        next_op_barriers[0] += gqa_ratio
        return

    for local_head in range(gqa_ratio):
        head_idx = abs_head + local_head
        if globs.attn_lse_intermediates.dim() == 2:
            globs.attn_lse_intermediates[head_idx, instruction.partial_idx] = lse[local_head]
        elif globs.attn_lse_intermediates.dim() == 4:
            globs.attn_lse_intermediates[0, 0, head_idx, instruction.partial_idx] = lse[local_head]
        else:
            globs.attn_lse_intermediates[0, head_idx, instruction.partial_idx] = lse[local_head]
        if globs.attn_out_intermediates.dim() == 3:
            globs.attn_out_intermediates[head_idx, instruction.partial_idx, :] = out[local_head]
        elif globs.attn_out_intermediates.dim() == 4:
            globs.attn_out_intermediates[0, head_idx, instruction.partial_idx, :] = out[local_head]
        else:
            globs.attn_out_intermediates[0, head_idx, instruction.partial_idx, :] = out[local_head]

    barriers = globs.barriers[
        instruction.layer_idx, BARRIER_ROW(instruction.opcode())
    ]
    for head_idx in range(abs_head, abs_head + gqa_ratio):
        barriers[head_idx] += 1
    if barriers[abs_head] == instruction.num_partials:
        for head_idx in range(abs_head, abs_head + gqa_ratio):
            assert barriers[head_idx] == instruction.num_partials
            if globs.attn_lse_intermediates.dim() == 2:
                lses = globs.attn_lse_intermediates[
                    head_idx, : instruction.num_partials
                ]
            elif globs.attn_lse_intermediates.dim() == 4:
                lses = globs.attn_lse_intermediates[
                    0, 0, head_idx, : instruction.num_partials
                ]
            else:
                lses = globs.attn_lse_intermediates[
                    0, head_idx, : instruction.num_partials
                ]

            if globs.attn_out_intermediates.dim() == 3:
                outs = globs.attn_out_intermediates[
                    head_idx, : instruction.num_partials, :
                ]
            else:
                outs = globs.attn_out_intermediates[
                    0, head_idx, : instruction.num_partials, :
                ]

            max_lse = torch.max(lses)
            factors = (lses - max_lse).exp2()
            denominator = factors.sum().clamp(min=1e-12)
            reduced = (outs.float() * factors[:, None]).sum(dim=0) / denominator
            globs.attn_out.view(Qwen3Config.num_attention_heads, -1)[head_idx] = (
                reduced.to(globs.attn_out.dtype)
            )
        next_op_barriers = globs.barriers[
            instruction.layer_idx,
            BARRIER_ROW(DecodeAttentionReduction.opcode()),
        ]
        next_op_barriers[0] += gqa_ratio


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

    next_op_barriers = globs.barriers[
        instruction.layer_idx, BARRIER_ROW(instruction.opcode())
    ]
    next_op_barriers[0] += size

INSTRUCTION_TO_HANDLER = {
    RMS_QKVProj_GEMV: rms_qkvproj_gemv,
    DecodeAttentionSplitKV: decode_attention_split_kv,
    DecodeAttentionReduction: decode_attention_reduction,
    OProj_GEMV_Residual: oproj_gemv_residual,
    RMS_UpGateProj_GEMV_SiLU: rms_upgate_proj_gemv_silu,
    DownProj_GEMV_Residual: down_proj_residual,
    RMS_LMHead_GEMV: rms_lmhead_gemv,
}
