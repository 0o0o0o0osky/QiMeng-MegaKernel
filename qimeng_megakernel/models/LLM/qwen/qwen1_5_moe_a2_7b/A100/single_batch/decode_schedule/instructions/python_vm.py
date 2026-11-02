"""PyVM reference for the A100 single-batch opcodes and barrier behavior."""
import math

import torch
import torch.nn.functional as F
from einops import einsum
from torch import Tensor
from qimeng_megakernel.models.instruction import Instruction

from ..globs import Globals
from ..model_config import QwenMoeConfig
from .instruction_defs import (
    RMS_QKVProj_GEMV_RoPE,
    DecodeAttentionSplitKV,
    DecodeAttentionReduction,
    OProj_GEMV_Residual,
    Router_GEMV,
    Router_SoftmaxTopK,
    SharedExpert_UpGate_SiLU,
    SharedExpert_DownProj,
    RoutedExpert_UpGate_SiLU,
    RoutedExpert_DownProj,
    MoE_Combine_Commit,
    RMS_LMHead_GEMV,
    TILE_N_GRANULARITY,
    SPLIT_K_GRANULARITY_SHARED,
    NUM_DOWNPROJ_SPLIT_K_SHARED,
)

NUM_HIDDEN_BLOCKS = QwenMoeConfig.hidden_size // TILE_N_GRANULARITY
NUM_MOE_INTERMEDIATE_BLOCKS = QwenMoeConfig.moe_intermediate_size // TILE_N_GRANULARITY
NUM_ROUTED_UPGATE_BLOCKS = QwenMoeConfig.num_experts_per_tok * NUM_MOE_INTERMEDIATE_BLOCKS
NUM_ROUTED_DOWNPROJ_BLOCKS = QwenMoeConfig.num_experts_per_tok * NUM_HIDDEN_BLOCKS
NUM_SHARED_DOWNPROJ_ACCUM_BLOCKS = NUM_DOWNPROJ_SPLIT_K_SHARED * NUM_HIDDEN_BLOCKS
COMBINE_DONE_VALUE = NUM_HIDDEN_BLOCKS


def _prev_barrier_row(instruction: Instruction) -> int:
    p = instruction.prev_opcode()
    return (p[0] - 1) if isinstance(p, tuple) else (p - 1)


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
    return torch.matmul(mat, vec)


def _gemv(mat: Tensor, vec: Tensor) -> Tensor:
    """Match the fused linear reference's fp32 accumulation for a row slice."""
    return torch.matmul(mat, vec.reshape(-1).to(mat.dtype))


def _rms_norm(inp: Tensor, weight: Tensor, eps: float):
    input_dtype = inp.dtype
    inp = inp.to(torch.float32)
    variance = inp.pow(2).mean(-1, keepdim=True)
    inp = inp * torch.rsqrt(variance + eps)
    return weight * inp.to(input_dtype)


def _rotate_half_interleaved(x: Tensor) -> Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    stacked = torch.stack((-x2, x1), dim=-1)
    return stacked.view_as(x)


def _apply_rotary_pos_emb_interleaved(
    q: Tensor, k: Tensor, cos: Tensor, sin: Tensor, unsqueeze_dim: int = 1
):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (_rotate_half_interleaved(q) * sin)
    k_embed = (k * cos) + (_rotate_half_interleaved(k) * sin)
    return q_embed, k_embed


# --------- Attention (1-4) ---------


def rms_qkvproj_gemv_rope(globs: Globals, instruction: RMS_QKVProj_GEMV_RoPE) -> None:
    layer_idx = instruction.layer_idx
    if layer_idx > 0:
        op_barriers = globs.barriers[layer_idx - 1, _prev_barrier_row(instruction)]
        expected = COMBINE_DONE_VALUE
        actual = op_barriers[0].item() if op_barriers.dim() > 0 else op_barriers.item()
        assert actual == expected, (
            f"layer {layer_idx}: prev-layer MoE combine barrier {actual} != {expected}"
        )

    post_ln = _rms_norm(
        globs.hidden_states.reshape(-1),
        globs.attn_norm_weights[layer_idx],
        eps=QwenMoeConfig.rms_norm_eps,
    )
    post_ln_flat = post_ln.reshape(-1)
    start, end = _range_from_span(
        instruction.tile_N_start_idx,
        instruction.tile_N_end_idx,
        instruction.tile_N_granularity,
    )
    pos_id = int(globs.position_id) if globs.position_id is not None else 0
    k_start = QwenMoeConfig.num_attention_heads * QwenMoeConfig.head_dim
    v_start = k_start + QwenMoeConfig.num_key_value_heads * QwenMoeConfig.head_dim
    barriers = globs.barriers[instruction.layer_idx, instruction.opcode() - 1]
    head_dim = QwenMoeConfig.head_dim
    blocks_per_head = head_dim // instruction.tile_N_granularity

    chunk_start = start
    while chunk_start < end:
        chunk_end_cand = min(chunk_start + instruction.tile_N_granularity, end)
        if chunk_start < k_start:
            seg_mode = "q"
            chunk_end = min(chunk_end_cand, k_start)
            head_end = (chunk_start // head_dim + 1) * head_dim
            chunk_end = min(chunk_end, head_end)
        elif chunk_start < v_start:
            seg_mode = "k"
            chunk_end = min(chunk_end_cand, v_start)
            head_end = (chunk_start // head_dim + 1) * head_dim
            chunk_end = min(chunk_end, head_end)
        else:
            seg_mode = "v"
            chunk_end = chunk_end_cand
        chunk_len = chunk_end - chunk_start

        weight_chunk = globs.qkv_weights[layer_idx, chunk_start:chunk_end]
        if globs.qkv_bias is not None:
            if globs.qkv_bias.dim() == 2:
                bias_chunk = globs.qkv_bias[layer_idx, chunk_start:chunk_end].to(
                    weight_chunk.dtype
                )
            else:
                block_idx = chunk_start // instruction.tile_N_granularity
                bias_chunk = globs.qkv_bias[layer_idx, block_idx].to(
                    weight_chunk.dtype
                )[:chunk_len]
            out = F.linear(post_ln_flat, weight_chunk, bias_chunk.reshape(-1))
        else:
            out = F.linear(post_ln_flat, weight_chunk)

        if seg_mode in ["q", "k"]:
            full_head = torch.zeros(
                1, head_dim, device=globs.hidden_states.device, dtype=out.dtype
            )
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
            globs.q_post_rope.reshape(-1)[chunk_start:chunk_end] = out.to(
                globs.q_post_rope.dtype
            )
        elif seg_mode == "k":
            start_in_k = chunk_start - k_start
            kv_head = start_in_k // head_dim
            in_head = start_in_k % head_dim
            globs.k_cache[layer_idx, 0, pos_id, kv_head, in_head : in_head + chunk_len] = (
                out.to(globs.k_cache.dtype)
            )
        else:
            start_in_v = chunk_start - v_start
            kv_head = start_in_v // head_dim
            in_head = start_in_v % head_dim
            globs.v_cache[layer_idx, 0, pos_id, kv_head, in_head : in_head + chunk_len] = (
                out.to(globs.v_cache.dtype)
            )

        block_start_idx = chunk_start // instruction.tile_N_granularity
        block_end_idx = (
            chunk_end + instruction.tile_N_granularity - 1
        ) // instruction.tile_N_granularity
        for block_idx in range(block_start_idx, block_end_idx):
            barriers[block_idx // blocks_per_head] += 1
        chunk_start = chunk_end


def decode_attention_split_kv(globs: Globals, instruction: DecodeAttentionSplitKV) -> None:
    op_barriers = globs.barriers[instruction.layer_idx, _prev_barrier_row(instruction)]
    blocks_per_head = QwenMoeConfig.head_dim // TILE_N_GRANULARITY
    gqa_ratio = QwenMoeConfig.num_attention_heads // QwenMoeConfig.num_key_value_heads
    for i in range(gqa_ratio):
        assert op_barriers[instruction.kv_head_idx * gqa_ratio + i] == blocks_per_head
    assert op_barriers[QwenMoeConfig.num_attention_heads + instruction.kv_head_idx] == blocks_per_head
    assert (
        op_barriers[
            QwenMoeConfig.num_attention_heads
            + QwenMoeConfig.num_key_value_heads
            + instruction.kv_head_idx
        ]
        == blocks_per_head
    )

    kv_block_size = QwenMoeConfig.attn_kv_block_size
    seq_len = int(globs.position_id) + 1 if globs.position_id is not None else 1
    layer_idx = instruction.layer_idx
    kv_head_idx = instruction.kv_head_idx
    total_blocks = math.ceil(seq_len / kv_block_size)
    blocks_per_partial = math.ceil(total_blocks / instruction.num_partials)
    start_block = instruction.partial_idx * blocks_per_partial
    end_block = min(start_block + blocks_per_partial, total_blocks)
    start_token = start_block * kv_block_size
    end_token = min(end_block * kv_block_size, seq_len)

    head_start = kv_head_idx * gqa_ratio
    head_end = head_start + gqa_ratio
    q = globs.q_post_rope.view(QwenMoeConfig.num_attention_heads, -1)[head_start:head_end]

    if end_token <= start_token:
        # Empty partial: write neutral for reduction (lse = -inf so 2^lse=0, out = 0)
        dev = globs.hidden_states.device
        lse = torch.full(
            (head_end - head_start,),
            float("-inf"),
            device=dev,
            dtype=torch.float32,
        )
        out = torch.zeros(
            head_end - head_start,
            QwenMoeConfig.head_dim,
            device=dev,
            dtype=globs.attn_out_intermediates.dtype,
        )
    else:
        k = globs.k_cache[layer_idx, 0, start_token:end_token, kv_head_idx, :]
        v = globs.v_cache[layer_idx, 0, start_token:end_token, kv_head_idx, :]
        qk = einsum(q.float(), k.float(), "h i, k i -> h k")
        scaled_qk = qk * (1 / math.sqrt(QwenMoeConfig.head_dim))
        softmax = torch.softmax(scaled_qk, dim=-1)
        lse = torch.logsumexp(scaled_qk.float(), dim=-1) / math.log(2)
        out = einsum(softmax.float(), v.float(), "h k, k o -> h o")

    if globs.attn_lse_intermediates.dim() == 2:
        globs.attn_lse_intermediates[head_start:head_end, instruction.partial_idx] = lse
        globs.attn_out_intermediates[head_start:head_end, instruction.partial_idx, :] = out
    else:
        globs.attn_lse_intermediates[0, 0, head_start:head_end, instruction.partial_idx] = lse
        globs.attn_out_intermediates[0, head_start:head_end, instruction.partial_idx, :] = out

    barriers = globs.barriers[instruction.layer_idx, instruction.opcode() - 1]
    barriers[head_start:head_end] += 1


def decode_attention_reduction(globs: Globals, instruction: DecodeAttentionReduction) -> None:
    head_start_idx = instruction.head_start_idx
    op_barriers = globs.barriers[instruction.layer_idx, _prev_barrier_row(instruction)]
    assert op_barriers[head_start_idx] == instruction.num_partials

    indices_to_reduce = torch.tensor(
        instruction.reduction_list, dtype=torch.long, device=globs.hidden_states.device
    )
    size = QwenMoeConfig.attn_reduction_size
    if globs.attn_lse_intermediates.dim() == 2:
        lses = globs.attn_lse_intermediates[head_start_idx : head_start_idx + size, indices_to_reduce]
        outs = globs.attn_out_intermediates[head_start_idx : head_start_idx + size, indices_to_reduce, :]
    else:
        lses = globs.attn_lse_intermediates[0, 0, head_start_idx : head_start_idx + size, indices_to_reduce]
        outs = globs.attn_out_intermediates[0, head_start_idx : head_start_idx + size, indices_to_reduce, :]

    max_lse = torch.max(lses, dim=-1, keepdim=True).values
    adjusted_factors = (lses - max_lse).exp2()
    new_denominator = adjusted_factors.sum(dim=-1, keepdim=True)
    reduced = (outs * adjusted_factors.unsqueeze(-1)).sum(dim=1) / new_denominator

    if instruction.is_terminal:
        globs.attn_out.view(QwenMoeConfig.num_attention_heads, -1)[
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


def oproj_gemv_residual(globs: Globals, instruction: OProj_GEMV_Residual) -> None:
    op_barriers = globs.barriers[instruction.layer_idx, _prev_barrier_row(instruction)]
    assert op_barriers[0] == QwenMoeConfig.num_attention_heads
    assert instruction.reduction_block_idx == 0

    mat = globs.o_proj_weights[instruction.layer_idx]
    vec = globs.attn_out.reshape(-1)
    residual = globs.hidden_states.reshape(-1)
    in_full = (0, QwenMoeConfig.num_attention_heads * QwenMoeConfig.head_dim)
    next_op_barriers = globs.barriers[instruction.layer_idx, instruction.opcode() - 1]

    for block_idx in range(instruction.tile_N_start_idx, instruction.tile_N_end_idx):
        start, end = _range_from_idx(block_idx, instruction.tile_N_granularity)
        gemv_out = _gemv_range(
            mat, vec, start, end, in_start=in_full[0], in_end=in_full[1]
        )
        # MK does float32 add then truncates to bf16 (single truncation).
        # Match that: (f32_gemv + f32_residual).to(bf16)
        residual[start:end] = (gemv_out.to(torch.float32) + residual[start:end].to(torch.float32)).to(residual.dtype)
        next_op_barriers[0] += 1


# --------- MoE (8-14) ---------


def router_gemv(globs: Globals, instruction: Router_GEMV) -> None:
    op_barriers = globs.barriers[instruction.layer_idx, _prev_barrier_row(instruction)]
    expected_o_proj_blocks = QwenMoeConfig.hidden_size // TILE_N_GRANULARITY
    assert op_barriers[0] == expected_o_proj_blocks

    normed = _rms_norm(
        globs.hidden_states.reshape(-1),
        globs.mlp_norm_weights[instruction.layer_idx],
        eps=QwenMoeConfig.rms_norm_eps,
    )
    num_experts = QwenMoeConfig.num_experts
    if globs.router_logits_buf is None or globs.router_logits_buf.shape[0] != num_experts:
        globs.router_logits_buf = torch.zeros(
            num_experts, device=normed.device, dtype=torch.float32
        )
    logits_buf = globs.router_logits_buf
    start_e = instruction.tile_N_start_idx * instruction.tile_N_granularity
    end_e = min(instruction.tile_N_end_idx * instruction.tile_N_granularity, num_experts)
    router_w = globs.router_weights[instruction.layer_idx]
    for e in range(start_e, end_e):
        logits_buf[e] = (router_w[e].float() * normed.float()).sum().item()
    barriers = globs.barriers[instruction.layer_idx, instruction.opcode() - 1]
    barriers[0] += end_e - start_e
    if instruction.write_normed_hidden:
        globs.normed_hidden_states.copy_(normed.to(globs.normed_hidden_states.dtype))
        barriers[1] += 1


def router_softmax_topk(globs: Globals, instruction: Router_SoftmaxTopK) -> None:
    op_barriers = globs.barriers[instruction.layer_idx, _prev_barrier_row(instruction)]
    assert op_barriers[0] == QwenMoeConfig.num_experts, (
        f"Router_GEMV barrier {op_barriers[0].item()} "
        f"!= {QwenMoeConfig.num_experts}"
    )
    logits_buf = globs.router_logits_buf
    assert logits_buf is not None and logits_buf.shape[0] == QwenMoeConfig.num_experts
    probs = F.softmax(logits_buf, dim=-1)
    top_probs, top_indices = torch.topk(probs, QwenMoeConfig.num_experts_per_tok, dim=-1)
    globs.expert_indices.copy_(top_indices.to(globs.expert_indices.dtype))
    globs.expert_weights.copy_(top_probs.to(globs.expert_weights.dtype))
    globs.barriers[instruction.layer_idx, instruction.opcode() - 1][0] = 1


def shared_expert_upgate_silu(globs: Globals, instruction: SharedExpert_UpGate_SiLU) -> None:
    if instruction.init_shared_out:
        globs.shared_out.zero_()

    op_barriers = globs.barriers[instruction.layer_idx, _prev_barrier_row(instruction)]
    assert op_barriers[1] == 1

    normed = globs.normed_hidden_states.reshape(-1)
    barriers = globs.barriers[instruction.layer_idx, instruction.opcode() - 1]
    out_buf = globs.shared_expert_silu_out.reshape(-1)
    for block_idx in range(instruction.tile_N_start_idx, instruction.tile_N_end_idx):
        start, end = _range_from_idx(block_idx, instruction.tile_N_granularity)
        gate = _gemv_range(
            globs.shared_expert_gate_proj_weights[instruction.layer_idx],
            normed, start, end,
        )
        up = _gemv_range(
            globs.shared_expert_up_proj_weights[instruction.layer_idx],
            normed, start, end,
        )
        out_buf[start:end] = (F.silu(gate.to(torch.float32)) * up.to(torch.float32)).to(out_buf.dtype)
        barriers[0] += 1


def shared_expert_downproj(globs: Globals, instruction: SharedExpert_DownProj) -> None:
    op_barriers = globs.barriers[instruction.layer_idx, _prev_barrier_row(instruction)]
    expected_blocks = QwenMoeConfig.intermediate_size // instruction.tile_N_granularity
    assert op_barriers[0] == expected_blocks

    # shared_out was already zeroed by the op9 SM with tile_N_start_idx==0.
    mat = globs.shared_expert_down_proj_weights[instruction.layer_idx]
    vec = globs.shared_expert_silu_out.reshape(-1)
    in_start = instruction.split_K_idx * instruction.split_K_granularity
    in_end = min(
        (instruction.split_K_idx + 1) * instruction.split_K_granularity,
        QwenMoeConfig.intermediate_size,
    )
    next_op_barriers = globs.barriers[instruction.layer_idx, instruction.opcode() - 1]
    shared_out = globs.shared_out.reshape(-1)

    for block_idx in range(instruction.tile_N_start_idx, instruction.tile_N_end_idx):
        start, end = _range_from_idx(block_idx, instruction.tile_N_granularity)
        gemv_out = _gemv_range(mat, vec, start, end, in_start, in_end)
        shared_out[start:end] = (shared_out[start:end].to(torch.float32) + gemv_out.to(torch.float32)).to(shared_out.dtype)
        next_op_barriers[block_idx] += 1


def routed_expert_upgate_silu(globs: Globals, instruction: RoutedExpert_UpGate_SiLU) -> None:
    op_barriers = globs.barriers[instruction.layer_idx, _prev_barrier_row(instruction)]
    assert op_barriers[0] == 1  # Router completed

    if globs.normed_hidden_states is not None:
        normed = globs.normed_hidden_states.reshape(-1)
    else:
        normed = _rms_norm(
            globs.hidden_states.reshape(-1),
            globs.mlp_norm_weights[instruction.layer_idx],
            eps=QwenMoeConfig.rms_norm_eps,
        )
    eid = globs.expert_indices[instruction.expert_idx].item()
    gate_up = globs.expert_gate_up_proj_weights[instruction.layer_idx, eid]
    mid = QwenMoeConfig.moe_intermediate_size
    barriers = globs.barriers[instruction.layer_idx, instruction.opcode() - 1]
    silu_out = globs.silu_out.reshape(-1)
    routed_silu_buf = globs.routed_silu_buffer
    if (
        routed_silu_buf is None
        or routed_silu_buf.shape[0] != QwenMoeConfig.num_experts_per_tok
        or routed_silu_buf.shape[1] != QwenMoeConfig.moe_intermediate_size
        or routed_silu_buf.device != silu_out.device
        or routed_silu_buf.dtype != silu_out.dtype
    ):
        routed_silu_buf = torch.zeros(
            (QwenMoeConfig.num_experts_per_tok, QwenMoeConfig.moe_intermediate_size),
            device=silu_out.device,
            dtype=silu_out.dtype,
        )
        globs.routed_silu_buffer = routed_silu_buf
    block_count = instruction.tile_N_end_idx - instruction.tile_N_start_idx
    for block_idx in range(instruction.tile_N_start_idx, instruction.tile_N_end_idx):
        start, end = _range_from_idx(block_idx, instruction.tile_N_granularity)
        gate_vec = _gemv_range(gate_up[:mid], normed, start, end)
        up_vec = _gemv_range(gate_up[mid:], normed, start, end)
        silu_slice = (F.silu(gate_vec.to(torch.float32)) * up_vec.to(torch.float32)).to(
            silu_out.dtype
        )
        # Keep legacy single buffer for debug/tests and write per-expert buffer for op12.
        silu_out[start:end] = silu_slice
        routed_silu_buf[instruction.expert_idx, start:end] = silu_slice
    barriers[0] += block_count


def routed_expert_downproj(globs: Globals, instruction: RoutedExpert_DownProj) -> None:
    op_barriers = globs.barriers[instruction.layer_idx, _prev_barrier_row(instruction)]
    # All routed upgate tile blocks must have completed
    assert op_barriers[0] == NUM_ROUTED_UPGATE_BLOCKS, (
        f"RoutedUpGate barrier {op_barriers[0].item()} != {NUM_ROUTED_UPGATE_BLOCKS}"
    )
    eid = int(globs.expert_indices[instruction.expert_idx].item())
    w = globs.expert_weights[instruction.expert_idx].to(torch.float32)
    mat = globs.expert_down_proj_weights[instruction.layer_idx, eid].contiguous()
    routed_silu_buf = globs.routed_silu_buffer
    if routed_silu_buf is not None:
        vec = routed_silu_buf[instruction.expert_idx].reshape(-1).contiguous()
    else:
        vec = globs.silu_out.reshape(-1).contiguous()
    routed_out_f32_flat = globs.routed_out_f32.reshape(-1)
    next_op_barriers = globs.barriers[instruction.layer_idx, instruction.opcode() - 1]

    for block_idx in range(instruction.tile_N_start_idx, instruction.tile_N_end_idx):
        start, end = _range_from_idx(block_idx, instruction.tile_N_granularity)
        gemv_out = _gemv_range(mat, vec, start, end)
        routed_out_f32_flat[start:end] += gemv_out.to(torch.float32) * w
        next_op_barriers[block_idx] += 1


def moe_combine_commit(globs: Globals, instruction: MoE_Combine_Commit) -> None:
    """Consume complete per-tile fan-in and publish the next hidden state."""
    layer_idx = instruction.layer_idx
    shared_barriers = globs.barriers[
        layer_idx, SharedExpert_DownProj.opcode() - 1
    ]
    routed_barriers = globs.barriers[
        layer_idx, RoutedExpert_DownProj.opcode() - 1
    ]
    for tile_idx in range(
        instruction.tile_N_start_idx, instruction.tile_N_end_idx
    ):
        assert shared_barriers[tile_idx] == NUM_DOWNPROJ_SPLIT_K_SHARED
        assert routed_barriers[tile_idx] == QwenMoeConfig.num_experts_per_tok

    gate_w = globs.shared_expert_gate_weights[layer_idx].reshape(-1).float()
    normed = globs.normed_hidden_states.reshape(-1).float()
    gate = torch.sigmoid((gate_w * normed).sum())
    start, end = _range_from_span(
        instruction.tile_N_start_idx,
        instruction.tile_N_end_idx,
        instruction.tile_N_granularity,
    )
    hidden = globs.hidden_states.reshape(-1)
    routed = globs.routed_out_f32.reshape(-1)
    shared = globs.shared_out.reshape(-1)
    hidden[start:end] = (
        hidden[start:end].float()
        + routed[start:end]
        + gate * shared[start:end].float()
    ).to(hidden.dtype)
    routed[start:end] = 0.0
    globs.barriers[layer_idx, instruction.opcode() - 1][0] += (
        instruction.tile_N_end_idx - instruction.tile_N_start_idx
    )


# --------- LM head (7) ---------


def rms_lmhead_gemv(globs: Globals, instruction: RMS_LMHead_GEMV) -> None:
    op_barriers = globs.barriers[
        QwenMoeConfig.num_hidden_layers - 1, _prev_barrier_row(instruction)
    ]
    assert op_barriers[0].item() == COMBINE_DONE_VALUE

    post_ln = _rms_norm(
        globs.hidden_states.reshape(-1),
        globs.lm_head_norm_weights,
        eps=QwenMoeConfig.rms_norm_eps,
    )
    for block_idx in range(instruction.start_output_block_idx, instruction.end_output_block_idx):
        start, end = _range_from_idx(block_idx, instruction.tile_N_granularity)
        out = _gemv(globs.lm_head_weights[start:end], post_ln)
        globs.logits.reshape(-1)[start:end] = out.to(globs.logits.dtype)


INSTRUCTION_TO_HANDLER = {
    RMS_QKVProj_GEMV_RoPE: rms_qkvproj_gemv_rope,
    DecodeAttentionSplitKV: decode_attention_split_kv,
    DecodeAttentionReduction: decode_attention_reduction,
    OProj_GEMV_Residual: oproj_gemv_residual,
    Router_GEMV: router_gemv,
    Router_SoftmaxTopK: router_softmax_topk,
    SharedExpert_UpGate_SiLU: shared_expert_upgate_silu,
    SharedExpert_DownProj: shared_expert_downproj,
    RoutedExpert_UpGate_SiLU: routed_expert_upgate_silu,
    RoutedExpert_DownProj: routed_expert_downproj,
    MoE_Combine_Commit: moe_combine_commit,
    RMS_LMHead_GEMV: rms_lmhead_gemv,
}
