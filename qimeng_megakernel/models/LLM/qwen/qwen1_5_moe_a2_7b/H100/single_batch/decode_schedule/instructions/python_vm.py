"""PyVM handlers for Qwen1.5-MoE-A2.7B H100 single-batch decode.

Opcodes 1-4 implement attention, opcode 7 implements LM head, and opcodes
8-14 implement the MoE block. Expected barrier counts come from
``model_config.barrier_expected_table()``. Router math and routed accumulation
stay fp32; shared split-K accumulation follows the bf16 buffer ABI.
"""

from __future__ import annotations

import math
import os

import torch
import torch.nn.functional as F
from torch import Tensor

from ..globs import Globals
from ..model_config import (
    BARRIER_ROW_ATTN_PARTIAL,
    BARRIER_ROW_ATTN_REDUCTION,
    BARRIER_ROW_COMBINE_COMMIT,
    BARRIER_ROW_O_PROJ,
    BARRIER_ROW_QKV_ROPE,
    BARRIER_ROW_ROUTED_DOWNPROJ,
    BARRIER_ROW_ROUTED_UPGATE,
    BARRIER_ROW_ROUTER_GEMV,
    BARRIER_ROW_ROUTER_TOPK,
    BARRIER_ROW_SHARED_DOWNPROJ,
    BARRIER_ROW_SHARED_UPGATE,
    BARRIER_SLOT_K_BASE,
    BARRIER_SLOT_NORMED_HIDDEN,
    BARRIER_SLOT_ROUTER_LOGITS,
    BARRIER_SLOT_V_BASE,
    QwenMoeConfig,
    barrier_expected_table,
)
from .instruction_defs import (
    TILE_N_GRANULARITY,
    DecodeAttentionReduction,
    DecodeAttentionSplitKV,
    MoE_Combine_Commit,
    OProj_GEMV_Residual,
    RMS_LMHead_GEMV,
    RMS_QKVProj_GEMV_RoPE,
    RoutedExpert_DownProj,
    RoutedExpert_UpGate_SiLU,
    Router_GEMV,
    Router_SoftmaxTopK,
    SharedExpert_DownProj,
    SharedExpert_UpGate_SiLU,
)

#: Barrier counts shared by the scheduler and PyVM handlers.
EXPECTED = barrier_expected_table()

#: Reading a barrier counter forces a device sync, so the ~27k spin checks in a full
#: decode step cost real wall-clock.  They are on by default (the verifier relies on
#: them); set QWEN_MOE_A2_7B_PYVM_BARRIER_CHECKS=0 to run the handlers without them.
#: Barrier *signals* are always written -- only the waits are skippable.
BARRIER_CHECKS = os.environ.get(
    "QWEN_MOE_A2_7B_PYVM_BARRIER_CHECKS", "1"
).strip().lower() not in {"0", "false", "no", "off"}


# ---------------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------------


def _bar(globs: Globals, layer_idx: int, row: int) -> Tensor:
    return globs.barriers[layer_idx, row]


def _wait(globs: Globals, layer_idx: int, row: int, slot: int, expected: int, what: str) -> None:
    """PyVM stand-in for the megakernel spin-wait.

    Instructions are executed in the scheduler's (topological) emission order, so the
    dependency is satisfied by construction; the counter is asserted to prove the
    schedule and handlers agree on the configured dependency count.
    """
    if not BARRIER_CHECKS:
        return
    actual = int(globs.barriers[layer_idx, row, slot].item())
    assert actual == expected, (
        f"{what}: barriers[{layer_idx}, {row}, {slot}] = {actual} != {expected}"
    )


def _elems(start_block: int, end_block: int, granularity: int = TILE_N_GRANULARITY):
    """[block_start, block_end) -> [element_start, element_end)."""
    return start_block * granularity, end_block * granularity


def _gemv(mat: Tensor, vec: Tensor) -> Tensor:
    """`F.linear(vec, mat)` for one row slice of `mat`.

    Kept as a real matmul (not `(mat * vec).sum(-1)`) so the accumulator stays fp32 and a
    row slice is numerically identical to the corresponding slice of the fused
    `F.linear` over the full matrix -- rows of a GEMV are independent.
    """
    return torch.matmul(mat, vec.to(mat.dtype))


def _rms_norm(inp: Tensor, weight: Tensor, eps: float) -> Tensor:
    """Identical to `fused/operations.py::_rms_norm` (fp32 variance, cast back, scale)."""
    input_dtype = inp.dtype
    x = inp.to(torch.float32)
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return weight * x.to(input_dtype)


def _rotate_half_interleaved(x: Tensor) -> Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).view_as(x)


def _apply_rope_interleaved(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Interleaved RoPE on a head-aligned slice.

    Pairs are (2i, 2i+1), so restricting to any slice that starts at an even offset and
    stays inside one head gives exactly the corresponding slice of the full-head result.
    Tiles are 16 wide and head_dim is 128, so every tile satisfies that.
    """
    return (x * cos) + (_rotate_half_interleaved(x) * sin)


# ---------------------------------------------------------------------------------
# opcode 1 -- RMS_QKVProj_GEMV_RoPE
# ---------------------------------------------------------------------------------


def rms_qkvproj_gemv_rope(globs: Globals, ins: RMS_QKVProj_GEMV_RoPE) -> None:
    """RMSNorm + QKV GEMV + bias + interleaved RoPE -> q_post_rope / k_cache / v_cache.

    Waits (layer > 0): `barriers[L-1, 13, 0] == 128` (previous layer committed).
    Signals: `barriers[L, 0, block // 8] += 1` per output tile block.
    """
    layer_idx = ins.layer_idx
    if layer_idx > 0:
        _wait(
            globs,
            layer_idx - 1,
            BARRIER_ROW_COMBINE_COMMIT,
            0,
            EXPECTED["op1@row13.slot0"],
            "op1 prev-layer MoE combine",
        )

    cfg = QwenMoeConfig
    head_dim = cfg.head_dim
    k_start = cfg.num_attention_heads * head_dim
    v_start = k_start + cfg.num_key_value_heads * head_dim
    g = ins.tile_N_granularity
    blocks_per_head = head_dim // g
    pos_id = int(globs.position_id) if globs.position_id is not None else 0

    post_ln = _rms_norm(
        globs.hidden_states.reshape(-1),
        globs.attn_norm_weights[layer_idx],
        eps=cfg.rms_norm_eps,
    )

    n_start, n_end = _elems(ins.tile_N_start_idx, ins.tile_N_end_idx, g)
    weight = globs.qkv_weights[layer_idx][n_start:n_end]
    out = _gemv(weight, post_ln)
    if globs.qkv_bias is not None:
        out = out + globs.qkv_bias[layer_idx].reshape(-1)[n_start:n_end].to(out.dtype)

    cos = globs.rope_cos[pos_id]
    sin = globs.rope_sin[pos_id]
    bar = _bar(globs, layer_idx, BARRIER_ROW_QKV_ROPE)

    for block_idx in range(ins.tile_N_start_idx, ins.tile_N_end_idx):
        s = block_idx * g
        e = s + g
        chunk = out[s - n_start : e - n_start]
        if s < k_start:
            off = s % head_dim
            globs.q_post_rope.reshape(-1)[s:e] = _apply_rope_interleaved(
                chunk, cos[off : off + g], sin[off : off + g]
            ).to(globs.q_post_rope.dtype)
        elif s < v_start:
            in_k = s - k_start
            kv_head, off = divmod(in_k, head_dim)
            globs.k_cache[layer_idx, 0, pos_id, kv_head, off : off + g] = (
                _apply_rope_interleaved(chunk, cos[off : off + g], sin[off : off + g]).to(
                    globs.k_cache.dtype
                )
            )
        else:
            in_v = s - v_start
            kv_head, off = divmod(in_v, head_dim)
            globs.v_cache[layer_idx, 0, pos_id, kv_head, off : off + g] = chunk.to(
                globs.v_cache.dtype
            )
        # Slot layout: Q heads 0..15, K heads 16..31, V heads 32..47 (48 slots of 8 tiles).
        bar[block_idx // blocks_per_head] += 1


# ---------------------------------------------------------------------------------
# opcode 2 -- DecodeAttentionSplitKV
# ---------------------------------------------------------------------------------


def decode_attention_split_kv(globs: Globals, ins: DecodeAttentionSplitKV) -> None:
    """One (kv_head, partial) split-KV attention shard -> attn_{out,lse}_intermediates.

    Waits: `barriers[L, 0, {q_head, 16+kv_head, 32+kv_head}] == 8`.
    Signals: `barriers[L, 1, q_head] += 1`.
    """
    cfg = QwenMoeConfig
    layer_idx = ins.layer_idx
    kv_head_idx = ins.kv_head_idx
    gqa = cfg.num_attention_heads // cfg.num_key_value_heads  # MHA -> 1

    for i in range(gqa):
        _wait(
            globs, layer_idx, BARRIER_ROW_QKV_ROPE, kv_head_idx * gqa + i,
            EXPECTED["op2@row0.q_head"], "op2 Q head",
        )
    _wait(
        globs, layer_idx, BARRIER_ROW_QKV_ROPE, BARRIER_SLOT_K_BASE + kv_head_idx,
        EXPECTED["op2@row0.k_head"], "op2 K head",
    )
    _wait(
        globs, layer_idx, BARRIER_ROW_QKV_ROPE, BARRIER_SLOT_V_BASE + kv_head_idx,
        EXPECTED["op2@row0.v_head"], "op2 V head",
    )

    kv_block_size = cfg.attn_kv_block_size
    seq_len = (int(globs.position_id) if globs.position_id is not None else 0) + 1
    total_blocks = math.ceil(seq_len / kv_block_size)
    blocks_per_partial = math.ceil(total_blocks / ins.num_partials)
    start_block = ins.partial_idx * blocks_per_partial
    end_block = min(start_block + blocks_per_partial, total_blocks)
    start_token = start_block * kv_block_size
    end_token = min(end_block * kv_block_size, seq_len)

    head_start = kv_head_idx * gqa
    head_end = head_start + gqa
    q = globs.q_post_rope.view(cfg.num_attention_heads, -1)[head_start:head_end]

    if end_token <= start_token:
        # Empty partial: neutral element for the LSE reduction (2**-inf = 0).
        lse = torch.full(
            (gqa,), float("-inf"), device=q.device, dtype=torch.float32
        )
        out = torch.zeros(
            gqa, cfg.head_dim, device=q.device,
            dtype=globs.attn_out_intermediates.dtype,
        )
    else:
        k = globs.k_cache[layer_idx, 0, start_token:end_token, kv_head_idx, :].float()
        v = globs.v_cache[layer_idx, 0, start_token:end_token, kv_head_idx, :].float()
        scaled_qk = torch.matmul(q.float(), k.transpose(0, 1)) / math.sqrt(cfg.head_dim)
        probs = torch.softmax(scaled_qk, dim=-1)
        lse = torch.logsumexp(scaled_qk, dim=-1) / math.log(2)
        out = torch.matmul(probs, v)

    globs.attn_lse_intermediates[head_start:head_end, ins.partial_idx] = lse
    globs.attn_out_intermediates[head_start:head_end, ins.partial_idx, :] = out.to(
        globs.attn_out_intermediates.dtype
    )

    _bar(globs, layer_idx, BARRIER_ROW_ATTN_PARTIAL)[head_start:head_end] += 1


# ---------------------------------------------------------------------------------
# opcode 3 -- DecodeAttentionReduction
# ---------------------------------------------------------------------------------


def decode_attention_reduction(globs: Globals, ins: DecodeAttentionReduction) -> None:
    """LSE-weighted reduction over the split-KV partials of 4 Q heads -> attn_out.

    Waits: `barriers[L, 1, head_start + i] == num_partials`, i in 0..3.
    Signals: `barriers[L, 2, 0] += 4`.
    """
    cfg = QwenMoeConfig
    layer_idx = ins.layer_idx
    size = cfg.attn_reduction_size
    head_start = ins.head_start_idx

    for i in range(size):
        _wait(
            globs, layer_idx, BARRIER_ROW_ATTN_PARTIAL, head_start + i,
            ins.num_partials, "op3 attention partial",
        )

    idx = torch.tensor(
        ins.reduction_list, dtype=torch.long, device=globs.hidden_states.device
    )
    lses = globs.attn_lse_intermediates[head_start : head_start + size][:, idx]
    outs = globs.attn_out_intermediates[head_start : head_start + size][:, idx, :]

    max_lse = torch.max(lses, dim=-1, keepdim=True).values
    factors = (lses - max_lse).exp2()
    denom = factors.sum(dim=-1, keepdim=True)
    reduced = (outs * factors.unsqueeze(-1)).sum(dim=1) / denom

    if ins.is_terminal:
        globs.attn_out.view(cfg.num_attention_heads, -1)[
            head_start : head_start + size
        ] = reduced.to(globs.attn_out.dtype)
    else:
        slot = ins.output_partial_idx
        globs.attn_lse_intermediates[head_start : head_start + size, slot] = (
            denom.log2() + max_lse
        ).squeeze(-1)
        globs.attn_out_intermediates[head_start : head_start + size, slot, :] = reduced

    _bar(globs, layer_idx, BARRIER_ROW_ATTN_REDUCTION)[0] += size


# ---------------------------------------------------------------------------------
# opcode 4 -- OProj_GEMV_Residual
# ---------------------------------------------------------------------------------


def oproj_gemv_residual(globs: Globals, ins: OProj_GEMV_Residual) -> None:
    """o_proj GEMV + residual into hidden_states.

    Waits: `barriers[L, 2, 0] == 16`.  Signals: `barriers[L, 3, 0] += tile blocks`.
    """
    layer_idx = ins.layer_idx
    _wait(
        globs, layer_idx, BARRIER_ROW_ATTN_REDUCTION, 0,
        EXPECTED["op4@row2.slot0"], "op4 attention reduction",
    )
    assert ins.reduction_block_idx == 0, "single reduction block for this model"

    n_start, n_end = _elems(ins.tile_N_start_idx, ins.tile_N_end_idx, ins.tile_N_granularity)
    out = _gemv(globs.o_proj_weights[layer_idx][n_start:n_end], globs.attn_out.reshape(-1))
    hidden = globs.hidden_states.reshape(-1)
    # Same rounding as `fused_oproj_gemv_residual` (`hidden_states += out.to(dtype)`).
    hidden[n_start:n_end] = hidden[n_start:n_end] + out.to(hidden.dtype)

    _bar(globs, layer_idx, BARRIER_ROW_O_PROJ)[0] += (
        ins.tile_N_end_idx - ins.tile_N_start_idx
    )


# ---------------------------------------------------------------------------------
# opcode 8 -- Router_GEMV
# ---------------------------------------------------------------------------------


def router_gemv(globs: Globals, ins: Router_GEMV) -> None:
    """MLP RMSNorm + router logits for experts [start, end).

    `tile_N_*` counts experts (granularity 1).  The instruction with
    `write_normed_hidden == 1` additionally publishes `normed_hidden_states` and raises
    `barriers[L, 7, 1]` -- see module docstring D2.

    Waits: `barriers[L, 3, 0] == 128`.
    Signals: `barriers[L, 7, 0] += (end - start)`; plus `barriers[L, 7, 1] += 1` once.
    """
    layer_idx = ins.layer_idx
    _wait(
        globs, layer_idx, BARRIER_ROW_O_PROJ, 0,
        EXPECTED["op8@row3.slot0"], "op8 o_proj residual",
    )

    normed = _rms_norm(
        globs.hidden_states.reshape(-1),
        globs.mlp_norm_weights[layer_idx],
        eps=QwenMoeConfig.rms_norm_eps,
    )

    e_start = ins.tile_N_start_idx * ins.tile_N_granularity
    e_end = min(ins.tile_N_end_idx * ins.tile_N_granularity, QwenMoeConfig.num_experts)
    # Router logits stay fp32 through GEMV and top-k selection.
    globs.router_logits_buf[e_start:e_end] = torch.matmul(
        globs.router_weights[layer_idx][e_start:e_end].float(), normed.float()
    )

    bar = _bar(globs, layer_idx, BARRIER_ROW_ROUTER_GEMV)
    bar[BARRIER_SLOT_ROUTER_LOGITS] += e_end - e_start
    if ins.write_normed_hidden:
        globs.normed_hidden_states.reshape(-1).copy_(
            normed.to(globs.normed_hidden_states.dtype)
        )
        bar[BARRIER_SLOT_NORMED_HIDDEN] += 1


# ---------------------------------------------------------------------------------
# opcode 9 -- Router_SoftmaxTopK
# ---------------------------------------------------------------------------------


def router_softmax_topk(globs: Globals, ins: Router_SoftmaxTopK) -> None:
    """fp32 softmax over the 60 router logits -> top-4 slots.  NOT renormalised.

    `norm_topk_prob = False`: the four routing weights are raw probabilities
    from the full 60-way softmax.

    Waits: `barriers[L, 7, 0] == 60`.  Signals: `barriers[L, 8, 0] += 1`.
    """
    layer_idx = ins.layer_idx
    _wait(
        globs, layer_idx, BARRIER_ROW_ROUTER_GEMV, BARRIER_SLOT_ROUTER_LOGITS,
        EXPECTED["op9@row7.slot0"], "op9 router logits",
    )

    logits = globs.router_logits_buf.reshape(-1)
    assert logits.dtype == torch.float32, "router_logits_buf must stay fp32"
    probs = F.softmax(logits, dim=-1)
    top_probs, top_indices = torch.topk(probs, QwenMoeConfig.num_experts_per_tok, dim=-1)
    globs.expert_indices.copy_(top_indices.to(globs.expert_indices.dtype))
    globs.expert_weights.copy_(top_probs.to(globs.expert_weights.dtype))

    _bar(globs, layer_idx, BARRIER_ROW_ROUTER_TOPK)[0] += 1


# ---------------------------------------------------------------------------------
# opcode 10 -- SharedExpert_UpGate_SiLU
# ---------------------------------------------------------------------------------


def shared_expert_upgate_silu(globs: Globals, ins: SharedExpert_UpGate_SiLU) -> None:
    """Shared expert gate/up GEMV + SiLU over shared intermediate blocks [start, end).

    Reads `normed_hidden_states` (published by opcode 8) -- it does NOT recompute the
    RMSNorm.  `init_shared_out == 1` zeroes `shared_out` so the opcode 11 split-K
    segments can all accumulate without a first-writer special case.

    Waits: `barriers[L, 7, 1] == 1`.  Signals: `barriers[L, 9, 0] += tile blocks`.
    """
    layer_idx = ins.layer_idx
    _wait(
        globs, layer_idx, BARRIER_ROW_ROUTER_GEMV, BARRIER_SLOT_NORMED_HIDDEN,
        EXPECTED["op10@row7.slot1"], "op10 normed_hidden_states published",
    )
    if ins.init_shared_out:
        globs.shared_out.zero_()

    normed = globs.normed_hidden_states.reshape(-1)
    n_start, n_end = _elems(ins.tile_N_start_idx, ins.tile_N_end_idx, ins.tile_N_granularity)
    gate = _gemv(globs.shared_expert_gate_proj_weights[layer_idx][n_start:n_end], normed)
    up = _gemv(globs.shared_expert_up_proj_weights[layer_idx][n_start:n_end], normed)
    out = globs.shared_expert_silu_out.reshape(-1)
    out[n_start:n_end] = (F.silu(gate.float()) * up.float()).to(out.dtype)

    _bar(globs, layer_idx, BARRIER_ROW_SHARED_UPGATE)[0] += (
        ins.tile_N_end_idx - ins.tile_N_start_idx
    )


# ---------------------------------------------------------------------------------
# opcode 11 -- SharedExpert_DownProj
# ---------------------------------------------------------------------------------


def shared_expert_downproj(globs: Globals, ins: SharedExpert_DownProj) -> None:
    """Shared expert down_proj, one K slice of 1408 -> accumulate into shared_out.

    Waits: `barriers[L, 9, 0] == 352`.
    Signals: `barriers[L, 10, t] += 1` for every output tile `t` it touches, so each of
    the 128 slots ends at exactly 4 (one per split-K segment) -- that is what opcode 14
    spins on.
    """
    layer_idx = ins.layer_idx
    _wait(
        globs, layer_idx, BARRIER_ROW_SHARED_UPGATE, 0,
        EXPECTED["op11@row9.slot0"], "op11 shared gate/up/SiLU",
    )

    k_start = ins.split_K_idx * ins.split_K_granularity
    k_end = min(k_start + ins.split_K_granularity, QwenMoeConfig.intermediate_size)
    n_start, n_end = _elems(ins.tile_N_start_idx, ins.tile_N_end_idx, ins.tile_N_granularity)

    mat = globs.shared_expert_down_proj_weights[layer_idx][n_start:n_end, k_start:k_end]
    vec = globs.shared_expert_silu_out.reshape(-1)[k_start:k_end]
    partial = _gemv(mat, vec)

    shared_out = globs.shared_out.reshape(-1)
    shared_out[n_start:n_end] = shared_out[n_start:n_end] + partial.to(shared_out.dtype)

    bar = _bar(globs, layer_idx, BARRIER_ROW_SHARED_DOWNPROJ)
    bar[ins.tile_N_start_idx : ins.tile_N_end_idx] += 1


# ---------------------------------------------------------------------------------
# opcode 12 -- RoutedExpert_UpGate_SiLU
# ---------------------------------------------------------------------------------


def routed_expert_upgate_silu(globs: Globals, ins: RoutedExpert_UpGate_SiLU) -> None:
    """Routed expert gate/up GEMV + SiLU for ONE top-k slot.

    `ins.expert_idx` is the top-k **slot** 0..3; the physical expert id is
    `expert_indices[slot]`.  Reads `normed_hidden_states`, writes
    `routed_silu_buffer[slot, start:end]`.

    Waits: `barriers[L, 8, 0] == 1`.  Signals: `barriers[L, 11, 0] += tile blocks`.
    """
    layer_idx = ins.layer_idx
    _wait(
        globs, layer_idx, BARRIER_ROW_ROUTER_TOPK, 0,
        EXPECTED["op12@row8.slot0"], "op12 router top-k",
    )

    mid = QwenMoeConfig.moe_intermediate_size
    eid = int(globs.expert_indices[ins.expert_idx].item())
    gate_up = globs.expert_gate_up_proj_weights_flat[
        layer_idx * QwenMoeConfig.num_experts + eid
    ]
    normed = globs.normed_hidden_states.reshape(-1)
    n_start, n_end = _elems(ins.tile_N_start_idx, ins.tile_N_end_idx, ins.tile_N_granularity)

    gate = _gemv(gate_up[n_start:n_end], normed)
    up = _gemv(gate_up[mid + n_start : mid + n_end], normed)
    buf = globs.routed_silu_buffer
    buf[ins.expert_idx, n_start:n_end] = (F.silu(gate.float()) * up.float()).to(buf.dtype)

    _bar(globs, layer_idx, BARRIER_ROW_ROUTED_UPGATE)[0] += (
        ins.tile_N_end_idx - ins.tile_N_start_idx
    )


# ---------------------------------------------------------------------------------
# opcode 13 -- RoutedExpert_DownProj  (pure GEMV on H100; see module docstring D1)
# ---------------------------------------------------------------------------------


def routed_expert_downproj(globs: Globals, ins: RoutedExpert_DownProj) -> None:
    """Routed expert down_proj GEMV x routing weight -> fp32 accumulate into routed_out_f32.

    There is deliberately **no combine tail** here (H100 delta D1): the combine is
    opcode 14.  `routed_out_f32` is not zeroed here either -- opcode 14 of the previous
    layer left it at zero (and L1 allocated it zeroed).

    Waits: `barriers[L, 11, 0] == 352`.
    Signals: `barriers[L, 12, t] += 1` per output tile, so each of the 128 slots ends at
    exactly 4 (one per top-k slot).
    """
    layer_idx = ins.layer_idx
    _wait(
        globs, layer_idx, BARRIER_ROW_ROUTED_UPGATE, 0,
        EXPECTED["op13@row11.slot0"], "op13 routed gate/up/SiLU",
    )

    eid = int(globs.expert_indices[ins.expert_idx].item())
    weight = globs.expert_weights[ins.expert_idx].float()
    mat = globs.expert_down_proj_weights_flat[
        layer_idx * QwenMoeConfig.num_experts + eid
    ]
    vec = globs.routed_silu_buffer[ins.expert_idx].reshape(-1)
    n_start, n_end = _elems(ins.tile_N_start_idx, ins.tile_N_end_idx, ins.tile_N_granularity)

    partial = _gemv(mat[n_start:n_end], vec)
    accum = globs.routed_out_f32.reshape(-1)
    accum[n_start:n_end] += partial.float() * weight

    bar = _bar(globs, layer_idx, BARRIER_ROW_ROUTED_DOWNPROJ)
    bar[ins.tile_N_start_idx : ins.tile_N_end_idx] += 1


# ---------------------------------------------------------------------------------
# opcode 14 -- MoE_Combine_Commit
# ---------------------------------------------------------------------------------


def moe_combine_commit(globs: Globals, ins: MoE_Combine_Commit) -> None:
    """Combine the shared and routed branches into hidden_states, one hidden tile.

    Per-tile sequence:
      1. `gate = sigmoid(dot(shared_expert_gate_weights[L], normed_hidden_states))`
         -- `shared_expert_gate_weights` is `(L, 1, 2048)`, so `[L]` is `(1, 2048)`.
      2. spin on `barriers[L, 10, t] >= 4` and `barriers[L, 12, t] >= 4`;
      3. `hidden_states[t] += routed_out_f32[t] + gate * shared_out[t]`;
      4. `routed_out_f32[t] = 0` (next layer starts from a clean accumulator);
      5. `barriers[L, 13, 0] += 1`.

    Each instruction computes the gate directly from route-owned tensors.
    """
    layer_idx = ins.layer_idx
    gate_w = globs.shared_expert_gate_weights[layer_idx].reshape(-1).float()
    normed = globs.normed_hidden_states.reshape(-1).float()
    gate = torch.sigmoid((gate_w * normed).sum())

    for t in range(ins.tile_N_start_idx, ins.tile_N_end_idx):
        _wait(
            globs, layer_idx, BARRIER_ROW_SHARED_DOWNPROJ, t,
            EXPECTED["op14@row10.tile"], f"op14 shared split-K tile {t}",
        )
        _wait(
            globs, layer_idx, BARRIER_ROW_ROUTED_DOWNPROJ, t,
            EXPECTED["op14@row12.tile"], f"op14 routed expert tile {t}",
        )

    n_start, n_end = _elems(ins.tile_N_start_idx, ins.tile_N_end_idx, ins.tile_N_granularity)
    hidden = globs.hidden_states.reshape(-1)
    routed = globs.routed_out_f32.reshape(-1)
    shared = globs.shared_out.reshape(-1)
    # fp32 throughout, one rounding to bf16 at the end.
    hidden[n_start:n_end] = (
        hidden[n_start:n_end].float()
        + routed[n_start:n_end]
        + gate * shared[n_start:n_end].float()
    ).to(hidden.dtype)
    routed[n_start:n_end] = 0.0

    _bar(globs, layer_idx, BARRIER_ROW_COMBINE_COMMIT)[0] += (
        ins.tile_N_end_idx - ins.tile_N_start_idx
    )


# ---------------------------------------------------------------------------------
# opcode 7 -- RMS_LMHead_GEMV  (terminal)
# ---------------------------------------------------------------------------------


def rms_lmhead_gemv(globs: Globals, ins: RMS_LMHead_GEMV) -> None:
    """Final RMSNorm + lm_head GEMV -> logits.

    No `layer_idx` field: opcode 7 always waits on the LAST layer's combine commit.
    Terminal instruction -- it signals nothing (barrier row 6 stays permanently zero).
    """
    _wait(
        globs, QwenMoeConfig.num_hidden_layers - 1, BARRIER_ROW_COMBINE_COMMIT, 0,
        EXPECTED["op7@row13.slot0"], "op7 last-layer MoE combine",
    )

    post_ln = _rms_norm(
        globs.hidden_states.reshape(-1),
        globs.lm_head_norm_weights,
        eps=QwenMoeConfig.rms_norm_eps,
    )
    n_start, n_end = _elems(
        ins.start_output_block_idx, ins.end_output_block_idx, ins.tile_N_granularity
    )
    n_end = min(n_end, QwenMoeConfig.vocab_size)
    if n_end <= n_start:
        return
    out = _gemv(globs.lm_head_weights[n_start:n_end], post_ln)
    globs.logits.reshape(-1)[n_start:n_end] = out.to(globs.logits.dtype)


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

__all__ = [
    "INSTRUCTION_TO_HANDLER",
    "EXPECTED",
    "BARRIER_CHECKS",
    "rms_qkvproj_gemv_rope",
    "decode_attention_split_kv",
    "decode_attention_reduction",
    "oproj_gemv_residual",
    "router_gemv",
    "router_softmax_topk",
    "shared_expert_upgate_silu",
    "shared_expert_downproj",
    "routed_expert_upgate_silu",
    "routed_expert_downproj",
    "moe_combine_commit",
    "rms_lmhead_gemv",
]
