"""Instruction definitions for Qwen1.5-MoE-A2.7B H100 single-batch decode.

Opcode map (rows in `barriers[layer, row, slot]` are always `opcode - 1`):

    1  RMS_QKVProj_GEMV_RoPE     7  RMS_LMHead_GEMV         11 SharedExpert_DownProj
    2  DecodeAttentionSplitKV    8  Router_GEMV             12 RoutedExpert_UpGate_SiLU
    3  DecodeAttentionReduction  9  Router_SoftmaxTopK      13 RoutedExpert_DownProj
    4  OProj_GEMV_Residual      10  SharedExpert_UpGate_SiLU 14 MoE_Combine_Commit

Opcodes 5 / 6 (dense MLP) are unused for this model.

Opcode 14 schedules the shared/routed combine one hidden tile per instruction.
Opcode 8 publishes ``normed_hidden_states`` once per layer for opcodes 10, 12,
and 14 while all router logits remain independently schedulable.

Field-order invariant
--------------------
Dataclass field order == the order `Instruction.serialize()` emits ints ==
`globs.instructions[sm, q, k]` column order == the order the megakernel reads
`inst.field_at(k)`.  Word 0 is always the opcode; the dataclasses below are the
authoritative 32-int instruction-layout definitions.
"""

from dataclasses import dataclass

from qimeng_megakernel.models.instruction import Instruction

from ..model_config import QwenMoeConfig

# Granularity constants: single source of truth is `model_config.QwenMoeConfig`.
# They are re-exported here because dataclass field defaults are bound at class creation.
TILE_M_GRANULARITY = QwenMoeConfig.tile_m_granularity  # 1
TILE_N_GRANULARITY = QwenMoeConfig.tile_n_granularity  # 16
NUM_DOWNPROJ_SPLIT_K_SHARED = QwenMoeConfig.num_downproj_split_k_shared  # 4
SPLIT_K_GRANULARITY_SHARED = QwenMoeConfig.split_k_granularity_shared  # 1408
MOE_INTERMEDIATE_SIZE = QwenMoeConfig.moe_intermediate_size  # 1408
# Router GEMV addresses experts one at a time -> its tile_N unit is "one expert".
ROUTER_TILE_N_GRANULARITY = 1


# ---------------------------------------------------------------------------------
# MoE opcodes 8-14 (declared first so opcodes 1 and 7 can reference opcode 14)
# ---------------------------------------------------------------------------------


@dataclass
class Router_GEMV(Instruction):
    """RMSNorm(hidden) + router logits for experts [tile_N_start_idx, tile_N_end_idx).

    `tile_N_*` counts **experts**, hence `tile_N_granularity = 1`.

    Signals `barriers[L, 7, 0] += (end - start)` (one per expert logit written).

    If `write_normed_hidden == 1` (exactly one instruction per layer, the one with
    `tile_N_start_idx == 0`), it ALSO stores its RMSNorm result to
    `normed_hidden_states` and then signals `barriers[L, 7, 1] += 1`.  See module
    docstring D2.
    """

    layer_idx: int
    tile_N_start_idx: int
    tile_N_end_idx: int
    write_normed_hidden: int = 0
    tile_N_granularity: int = ROUTER_TILE_N_GRANULARITY

    @classmethod
    def opcode(cls) -> int:
        return 8

    @classmethod
    def prev_opcode(cls) -> tuple[int, ...]:
        return (OProj_GEMV_Residual.opcode(),)


@dataclass
class Router_SoftmaxTopK(Instruction):
    """softmax(router_logits_buf) -> top-4 -> expert_indices / expert_weights.

    `norm_topk_prob = False` for this model, so the top-k weights are NOT renormalised.
    `expert_indices[slot]` holds the *physical* expert id; every downstream instruction
    addresses experts by **top-k slot** (0..3) and dereferences the physical id on device.
    """

    layer_idx: int

    @classmethod
    def opcode(cls) -> int:
        return 9

    @classmethod
    def prev_opcode(cls) -> tuple[int, ...]:
        return (Router_GEMV.opcode(),)


@dataclass
class SharedExpert_UpGate_SiLU(Instruction):
    """Shared expert gate/up GEMV + SiLU over shared intermediate blocks [start, end).

    Reads `normed_hidden_states` (published by opcode 8, see module docstring D2) --
    it does NOT recompute the RMSNorm.

    If `init_shared_out == 1` (exactly one instruction per layer, the one with
    `tile_N_start_idx == 0`) it first zeroes `shared_out`, so the opcode 11 split-K
    segments can all accumulate without a first-writer special case.  This is safe
    because opcode 11 waits for *every* opcode 10 tile block.
    """

    layer_idx: int
    tile_M_idx: int
    tile_N_start_idx: int
    tile_N_end_idx: int
    init_shared_out: int = 0
    tile_M_granularity: int = TILE_M_GRANULARITY
    tile_N_granularity: int = TILE_N_GRANULARITY

    @classmethod
    def opcode(cls) -> int:
        return 10

    @classmethod
    def prev_opcode(cls) -> tuple[int, ...]:
        # D2: waits on barriers[L, 7, 1] (normed_hidden_states published), which
        # transitively implies o_proj (opcode 4) is complete because opcode 8 waits on it.
        return (Router_GEMV.opcode(),)


@dataclass
class SharedExpert_DownProj(Instruction):
    """Shared expert down_proj: shared_expert_silu_out -> shared_out, split-K over 5632.

    `split_K_granularity = 1408` is deliberately identical to A100 -- the L4 K=1408
    pipeline design depends on it.

    Accumulates into `shared_out` and signals `barriers[L, 10, t] += 1` for each output
    tile `t` it touches, so each of the 128 slots reaches exactly 4 (one per split-K).
    """

    layer_idx: int
    tile_M_idx: int
    tile_N_start_idx: int
    tile_N_end_idx: int
    split_K_idx: int
    tile_M_granularity: int = TILE_M_GRANULARITY
    tile_N_granularity: int = TILE_N_GRANULARITY
    split_K_granularity: int = SPLIT_K_GRANULARITY_SHARED

    @classmethod
    def opcode(cls) -> int:
        return 11

    @classmethod
    def prev_opcode(cls) -> tuple[int, ...]:
        return (SharedExpert_UpGate_SiLU.opcode(),)


@dataclass
class RoutedExpert_UpGate_SiLU(Instruction):
    """Routed expert gate/up GEMV + SiLU for ONE top-k slot.

    `expert_idx` is the **top-k slot (0..3)**, not the physical expert id; the physical
    id is `expert_indices[expert_idx]`, dereferenced on device.

    Reads `normed_hidden_states` (D2).  Writes `routed_silu_buffer[expert_idx, :]`.
    """

    layer_idx: int
    expert_idx: int
    tile_N_start_idx: int
    tile_N_end_idx: int
    tile_M_granularity: int = TILE_M_GRANULARITY
    tile_N_granularity: int = TILE_N_GRANULARITY

    @classmethod
    def opcode(cls) -> int:
        return 12

    @classmethod
    def prev_opcode(cls) -> tuple[int, ...]:
        return (Router_SoftmaxTopK.opcode(),)


@dataclass
class RoutedExpert_DownProj(Instruction):
    """Routed expert down_proj GEMV for ONE top-k slot -- **pure GEMV** on H100.

    `expert_idx` is the top-k slot (0..3).  Accumulates
    `routed_out_f32[row] += expert_weights[expert_idx] * dot(...)` with fp32 atomicAdd,
    then signals `barriers[L, 12, t] += 1` per output tile `t`, so each of the 128 slots
    reaches exactly 4 (one per top-k slot).

    NOTE (H100 delta D1): there is deliberately **no** `fuse_combine` field.  The combine
    is opcode 14.
    """

    layer_idx: int
    expert_idx: int
    tile_M_idx: int
    tile_N_start_idx: int
    tile_N_end_idx: int
    tile_M_granularity: int = TILE_M_GRANULARITY
    tile_N_granularity: int = TILE_N_GRANULARITY

    @classmethod
    def opcode(cls) -> int:
        return 13

    @classmethod
    def prev_opcode(cls) -> tuple[int, ...]:
        return (RoutedExpert_UpGate_SiLU.opcode(),)


@dataclass
class MoE_Combine_Commit(Instruction):
    """MoE combine + commit for hidden tiles [tile_N_start_idx, tile_N_end_idx).

    One instruction owns each 16-element hidden tile, for 128 per layer.

    Per tile `t`, the consumer:
      1. computes `gate = sigmoid(dot(shared_expert_gate_weights[L], normed_hidden_states))`
         (`shared_expert_gate_weights` has source shape `(L, 1, 2048)`);
      2. spins until `barriers[L, 10, t] >= 4` (all shared split-K segments landed) and
         `barriers[L, 12, t] >= 4` (all 4 routed expert slots landed);
      3. writes `hidden_states[t] += routed_out_f32[t] + gate * shared_out[t]`;
      4. zeroes `routed_out_f32[t]` so the next layer starts from a clean accumulator;
      5. signals `barriers[L, 13, 0] += 1`.

    Row 13 slot 0 therefore reaches 128 per layer, which is exactly what the next layer's
    opcode 1 (and the terminal opcode 7) waits for.
    """

    layer_idx: int
    tile_N_start_idx: int
    tile_N_end_idx: int
    tile_N_granularity: int = TILE_N_GRANULARITY

    @classmethod
    def opcode(cls) -> int:
        return 14

    @classmethod
    def prev_opcode(cls) -> tuple[int, ...]:
        return (
            SharedExpert_DownProj.opcode(),
            RoutedExpert_DownProj.opcode(),
        )


# ---------------------------------------------------------------------------------
# Attention + LM head (opcodes 1-4, 7).  Opcodes 5 / 6 unused (dense MLP).
# ---------------------------------------------------------------------------------


@dataclass
class RMS_QKVProj_GEMV_RoPE(Instruction):
    """RMSNorm + QKV GEMV + RoPE; writes q_post_rope, k_cache, v_cache.

    Signals `barriers[L, 0, head_slot] += 1` per output tile, where
    `head_slot = global_block_idx // 8` over the 48 head slots
    (Q: 0..15, K: 16..31, V: 32..47) -- 8 tiles of 16 rows per 128-wide head.
    """

    layer_idx: int
    tile_M_idx: int
    tile_N_start_idx: int
    tile_N_end_idx: int
    tile_M_granularity: int = TILE_M_GRANULARITY
    tile_N_granularity: int = TILE_N_GRANULARITY

    @classmethod
    def opcode(cls) -> int:
        return 1

    @classmethod
    def prev_opcode(cls) -> tuple[int, ...]:
        return (MoE_Combine_Commit.opcode(),)


@dataclass
class DecodeAttentionSplitKV(Instruction):
    """Decode attention, split-KV. One instruction per (kv_head, partial).

    MHA: 16 Q heads == 16 KV heads, so the GQA group is 1.
    """

    layer_idx: int
    kv_head_idx: int
    num_partials: int
    partial_idx: int

    @classmethod
    def opcode(cls) -> int:
        return 2

    @classmethod
    def prev_opcode(cls) -> tuple[int, ...]:
        return (RMS_QKVProj_GEMV_RoPE.opcode(),)


@dataclass
class DecodeAttentionReduction(Instruction):
    """LSE-weighted reduction over `num_partials` split-KV partials, 4 Q heads each."""

    layer_idx: int
    head_start_idx: int
    num_partials: int
    reduction_list: list[int]
    is_terminal: int = 1  # 1 = True, kept int for serialize()
    output_partial_idx: int = 0

    @classmethod
    def opcode(cls) -> int:
        return 3

    @classmethod
    def prev_opcode(cls) -> tuple[int, ...]:
        return (DecodeAttentionSplitKV.opcode(),)


@dataclass
class OProj_GEMV_Residual(Instruction):
    """o_proj GEMV with residual add into hidden_states."""

    layer_idx: int
    tile_M_idx: int
    tile_N_start_idx: int
    tile_N_end_idx: int
    reduction_block_idx: int
    tile_M_granularity: int = TILE_M_GRANULARITY
    tile_N_granularity: int = TILE_N_GRANULARITY

    @classmethod
    def opcode(cls) -> int:
        return 4

    @classmethod
    def prev_opcode(cls) -> tuple[int, ...]:
        return (DecodeAttentionReduction.opcode(),)


# Opcodes 5, 6 are unused for this model (dense MLP replaced by MoE opcodes 8-14).


@dataclass
class RMS_LMHead_GEMV(Instruction):
    """Final RMSNorm + lm_head GEMV -> logits. Emitted once, after the last layer."""

    start_output_block_idx: int
    end_output_block_idx: int
    tile_M_granularity: int = TILE_M_GRANULARITY
    tile_N_granularity: int = TILE_N_GRANULARITY

    @classmethod
    def opcode(cls) -> int:
        return 7

    @classmethod
    def prev_opcode(cls) -> tuple[int, ...]:
        return (MoE_Combine_Commit.opcode(),)


#: opcode -> dataclass, for tests / tooling that need to walk the whole table.
INSTRUCTION_BY_OPCODE = {
    1: RMS_QKVProj_GEMV_RoPE,
    2: DecodeAttentionSplitKV,
    3: DecodeAttentionReduction,
    4: OProj_GEMV_Residual,
    7: RMS_LMHead_GEMV,
    8: Router_GEMV,
    9: Router_SoftmaxTopK,
    10: SharedExpert_UpGate_SiLU,
    11: SharedExpert_DownProj,
    12: RoutedExpert_UpGate_SiLU,
    13: RoutedExpert_DownProj,
    14: MoE_Combine_Commit,
}
