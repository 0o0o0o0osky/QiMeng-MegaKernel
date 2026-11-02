"""Scheduler for Qwen1.5-MoE-A2.7B H100 single-batch decode.

Per-layer instruction table (132 SMs, `num_partials = 8`):

    op  name                     formula                                count
    --  -----------------------  -------------------------------------  -----
     1  RMS_QKV_ROPE             split(6144/16 = 384 tiles, 132 SMs)      132
     2  ATTN_PARTIAL             16 kv_head x 8 partial                   128
     3  ATTN_REDUCTION           16 heads / 1 per instruction              16
     4  O_PROJ_RESIDUAL          split(2048/16 = 128 tiles, 132 SMs)      128
     8  ROUTER_GEMV              1 per expert                              60
     9  ROUTER_SOFTMAX_TOPK      1                                          1
    10  SHARED_UPGATE_SILU       split(5632/16 = 352 tiles, 132 SMs)      132
    11  SHARED_DOWNPROJ          4 split-K x split(128 tiles, 33 SMs)     132
    12  ROUTED_UPGATE_SILU       4 slots  x split(88 tiles, 33 SMs)       132
    13  ROUTED_DOWNPROJ          4 slots  x split(128 tiles, 33 SMs)      132
    14  MOE_COMBINE_COMMIT       1 per hidden tile                        128
                                                             per layer   1121

    7  RMS_LMHEAD_GEMV  split(151936/16 = 9496 tiles, 132 SMs)            132  (once)

    total = 24 * 1121 + 132 = 27036,  queue depth = ceil(27036 / 132) = 205

Emission order is a strict topological order of the dependency DAG, and instructions are
handed to SMs round-robin, which preserves that order inside every per-SM queue.  Since
each instruction waits only on earlier phases, so per-SM in-order consumption remains
topologically valid.
"""

from dataclasses import fields

import torch

from qimeng_megakernel.models.instruction import Instruction

from ..globs import Globals
from ..model_config import (
    BARRIER_ROW_ATTN_PARTIAL,
    BARRIER_ROW_ATTN_REDUCTION,
    BARRIER_ROW_COMBINE_COMMIT,
    BARRIER_ROW_LM_HEAD,
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
    get_num_partials_for_device,
    get_sm_count_for_device,
)
from .instruction_defs import (
    NUM_DOWNPROJ_SPLIT_K_SHARED,
    SPLIT_K_GRANULARITY_SHARED,
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

INTS_PER_INSTRUCTION = 32
ATTN_REDUCTION_SIZE = QwenMoeConfig.attn_reduction_size
#: A `reduction_list` is serialized as [len, *items] after opcode + 3 scalar fields,
#: and two trailing scalars follow it, hence the -5 headroom check.
MAX_REDUCTION_LIST = min(INTS_PER_INSTRUCTION - 5, 27)


class NoOp(Instruction):
    @classmethod
    def opcode(cls) -> int:
        return 0

    @classmethod
    def prev_opcode(cls) -> tuple[int, ...]:
        return ()


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _get_device(globs: Globals) -> torch.device:
    for field in fields(globs):
        value = getattr(globs, field.name)
        if isinstance(value, torch.Tensor):
            return value.device
    return torch.device("cpu")


def _get_sm_count(device: torch.device) -> int:
    """Delegates to `model_config`: one place decides the SM count, as with num_partials."""
    return get_sm_count_for_device(device)


def _serialize_and_pad(instruction: Instruction) -> list[int]:
    serialized = instruction.serialize()
    num_padding = INTS_PER_INSTRUCTION - len(serialized)
    assert num_padding >= 0, (
        f"{type(instruction).__name__} serializes to {len(serialized)} ints, "
        f"exceeding INTS_PER_INSTRUCTION={INTS_PER_INSTRUCTION}"
    )
    return serialized + [0] * num_padding


def tensorize_instructions(
    globs: Globals, instruction_queues: list[list[Instruction]]
) -> torch.Tensor:
    num_sms = len(instruction_queues)
    max_queue_len = max((len(q) for q in instruction_queues), default=0)
    for queue in instruction_queues:
        queue.extend([NoOp()] * (max_queue_len - len(queue)))
    flattened = []
    for queue in instruction_queues:
        flattened.extend(_serialize_and_pad(ins) for ins in queue)
    device = _get_device(globs)
    serialized = torch.tensor(flattened, dtype=torch.int32, device=device).view(
        num_sms, -1, INTS_PER_INSTRUCTION
    )
    globs.instructions = serialized
    return serialized


def _split_n_tiles(num_n_tiles: int, sm_count: int) -> list[tuple[int, int]]:
    """Split `num_n_tiles` contiguous tiles across at most `sm_count` instructions."""
    if num_n_tiles <= 0:
        return []
    groups = min(num_n_tiles, sm_count)
    base = num_n_tiles // groups
    remainder = num_n_tiles % groups
    ranges: list[tuple[int, int]] = []
    start = 0
    for i in range(groups):
        size = base + (1 if i < remainder else 0)
        end = start + size
        if size > 0:
            ranges.append((start, end))
        start = end
    return ranges


def _round_robin_assign_to_sms(
    instructions: list[Instruction], sm_count: int
) -> list[list[Instruction]]:
    sm_queues: list[list[Instruction]] = [[] for _ in range(sm_count)]
    for i, instruction in enumerate(instructions):
        sm_queues[i % sm_count].append(instruction)
    return sm_queues


def _sum_tile_blocks(instructions: list[Instruction]) -> int:
    total = 0
    for ins in instructions:
        if hasattr(ins, "tile_N_start_idx") and hasattr(ins, "tile_N_end_idx"):
            total += ins.tile_N_end_idx - ins.tile_N_start_idx
    return total


# ---------------------------------------------------------------------------------
# Attention + o_proj (opcodes 1-4)
# ---------------------------------------------------------------------------------


def schedule_qkvproj_gemv(layer_idx: int, sm_count: int) -> list[Instruction]:
    num_tiles = _ceil_div(QwenMoeConfig.qkv_out_dim, TILE_N_GRANULARITY)  # 384
    instructions: list[Instruction] = []
    for n_start, n_end in _split_n_tiles(num_tiles, sm_count):
        instructions.append(
            RMS_QKVProj_GEMV_RoPE(
                layer_idx=layer_idx,
                tile_M_idx=0,
                tile_N_start_idx=n_start,
                tile_N_end_idx=n_end,
            )
        )
    assert _sum_tile_blocks(instructions) == QwenMoeConfig.num_qkv_blocks
    return instructions


def schedule_attention_split_kv(layer_idx: int, num_partials: int) -> list[Instruction]:
    instructions: list[Instruction] = []
    for kv_head_idx in range(QwenMoeConfig.num_key_value_heads):
        for partial_idx in range(num_partials):
            instructions.append(
                DecodeAttentionSplitKV(
                    layer_idx=layer_idx,
                    kv_head_idx=kv_head_idx,
                    num_partials=num_partials,
                    partial_idx=partial_idx,
                )
            )
    return instructions


def schedule_attention_reduction(layer_idx: int, num_partials: int) -> list[Instruction]:
    assert num_partials <= MAX_REDUCTION_LIST, (
        f"reduction_list of {num_partials} would overflow the 32-int instruction word"
    )
    instructions: list[Instruction] = []
    for head_start_idx in range(
        0, QwenMoeConfig.num_attention_heads, ATTN_REDUCTION_SIZE
    ):
        instructions.append(
            DecodeAttentionReduction(
                layer_idx=layer_idx,
                head_start_idx=head_start_idx,
                num_partials=num_partials,
                reduction_list=list(range(num_partials)),
            )
        )
    return instructions


def schedule_oproj_gemv(layer_idx: int, sm_count: int) -> list[Instruction]:
    instructions: list[Instruction] = []
    num_n_tiles = _ceil_div(QwenMoeConfig.hidden_size, TILE_N_GRANULARITY)  # 128
    for n_start, n_end in _split_n_tiles(num_n_tiles, sm_count):
        instructions.append(
            OProj_GEMV_Residual(
                layer_idx=layer_idx,
                tile_M_idx=0,
                tile_N_start_idx=n_start,
                tile_N_end_idx=n_end,
                reduction_block_idx=0,
            )
        )
    assert _sum_tile_blocks(instructions) == QwenMoeConfig.num_hidden_blocks
    return instructions


# ---------------------------------------------------------------------------------
# MoE (opcodes 8 -> 9 -> 10 -> 11 -> 12 -> 13 -> 14)
# ---------------------------------------------------------------------------------


def schedule_router_gemv(layer_idx: int, sm_count: int) -> list[Instruction]:
    """One instruction per expert -- 60, deliberately NOT coarsened.

    Router weights are 60 x 2048 x 2 B = 245 KiB per layer, so bandwidth is a non-issue;
    what matters is critical-path latency, because no routed-expert weight byte can be
    fetched until top-k resolves.  The redundancy that *does* matter -- 60 copies of the
    same RMSNorm -- is addressed by having instruction 0 publish `normed_hidden_states`
    for opcodes 10 / 12 / 14 (module docstring D2 in `instruction_defs.py`).
    """
    num_experts = QwenMoeConfig.num_experts
    worker_count = max(1, min(QwenMoeConfig.router_gemv_workers, num_experts))
    instructions: list[Instruction] = []
    for e_start, e_end in _split_n_tiles(num_experts, worker_count):
        instructions.append(
            Router_GEMV(
                layer_idx=layer_idx,
                tile_N_start_idx=e_start,
                tile_N_end_idx=e_end,
                write_normed_hidden=1 if e_start == 0 else 0,
            )
        )
    assert _sum_tile_blocks(instructions) == num_experts
    assert sum(ins.write_normed_hidden for ins in instructions) == 1
    return instructions


def schedule_router_softmax_topk(layer_idx: int) -> list[Instruction]:
    return [Router_SoftmaxTopK(layer_idx=layer_idx)]


def schedule_shared_upgate(layer_idx: int, sm_count: int) -> list[Instruction]:
    instructions: list[Instruction] = []
    num_n_tiles = _ceil_div(QwenMoeConfig.intermediate_size, TILE_N_GRANULARITY)  # 352
    for n_start, n_end in _split_n_tiles(num_n_tiles, sm_count):
        instructions.append(
            SharedExpert_UpGate_SiLU(
                layer_idx=layer_idx,
                tile_M_idx=0,
                tile_N_start_idx=n_start,
                tile_N_end_idx=n_end,
                init_shared_out=1 if n_start == 0 else 0,
            )
        )
    assert _sum_tile_blocks(instructions) == QwenMoeConfig.num_shared_upgate_blocks
    assert sum(ins.init_shared_out for ins in instructions) == 1
    return instructions


def schedule_shared_downproj(layer_idx: int, sm_count: int) -> list[Instruction]:
    """`num_split_k` groups of `sm_count // num_split_k` SMs; each group covers all 128
    output tiles for its own K slice.  132 / 4 = 33 -> 4 x 33 = 132 instructions."""
    instructions: list[Instruction] = []
    num_split_k = NUM_DOWNPROJ_SPLIT_K_SHARED
    num_n_tiles = _ceil_div(QwenMoeConfig.hidden_size, TILE_N_GRANULARITY)  # 128
    sms_per_group = max(1, sm_count // num_split_k)
    for split_k_idx in range(num_split_k):
        for seg_start, seg_end in _split_n_tiles(num_n_tiles, sms_per_group):
            instructions.append(
                SharedExpert_DownProj(
                    layer_idx=layer_idx,
                    tile_M_idx=0,
                    tile_N_start_idx=seg_start,
                    tile_N_end_idx=seg_end,
                    split_K_idx=split_k_idx,
                )
            )
    assert (
        _sum_tile_blocks(instructions)
        == QwenMoeConfig.num_shared_downproj_accum_blocks
    )
    return instructions


def schedule_routed_upgate(layer_idx: int, sm_count: int) -> list[Instruction]:
    instructions: list[Instruction] = []
    num_slots = QwenMoeConfig.num_experts_per_tok
    num_tiles_per_expert = _ceil_div(
        QwenMoeConfig.moe_intermediate_size, TILE_N_GRANULARITY
    )  # 88
    sms_per_expert = max(1, sm_count // num_slots)
    for expert_idx in range(num_slots):
        for n_start, n_end in _split_n_tiles(num_tiles_per_expert, sms_per_expert):
            instructions.append(
                RoutedExpert_UpGate_SiLU(
                    layer_idx=layer_idx,
                    expert_idx=expert_idx,
                    tile_N_start_idx=n_start,
                    tile_N_end_idx=n_end,
                )
            )
    assert _sum_tile_blocks(instructions) == QwenMoeConfig.num_routed_upgate_blocks
    return instructions


def schedule_routed_downproj(layer_idx: int, sm_count: int) -> list[Instruction]:
    """Schedule routed-expert DownProj GEMV; opcode 14 owns the combine."""
    instructions: list[Instruction] = []
    num_slots = QwenMoeConfig.num_experts_per_tok
    num_n_tiles = _ceil_div(QwenMoeConfig.hidden_size, TILE_N_GRANULARITY)  # 128
    sms_per_expert = max(1, sm_count // num_slots)
    for expert_idx in range(num_slots):
        for seg_start, seg_end in _split_n_tiles(num_n_tiles, sms_per_expert):
            instructions.append(
                RoutedExpert_DownProj(
                    layer_idx=layer_idx,
                    expert_idx=expert_idx,
                    tile_M_idx=0,
                    tile_N_start_idx=seg_start,
                    tile_N_end_idx=seg_end,
                )
            )
    assert _sum_tile_blocks(instructions) == QwenMoeConfig.num_routed_downproj_blocks
    return instructions


def schedule_moe_combine_commit(layer_idx: int, sm_count: int) -> list[Instruction]:
    """One instruction per hidden tile (128 <= 132 SMs, exactly one wave)."""
    instructions: list[Instruction] = []
    num_n_tiles = _ceil_div(QwenMoeConfig.hidden_size, TILE_N_GRANULARITY)  # 128
    for n_start, n_end in _split_n_tiles(num_n_tiles, sm_count):
        instructions.append(
            MoE_Combine_Commit(
                layer_idx=layer_idx,
                tile_N_start_idx=n_start,
                tile_N_end_idx=n_end,
            )
        )
    assert _sum_tile_blocks(instructions) == QwenMoeConfig.num_hidden_blocks
    return instructions


def schedule_lmhead_gemv(sm_count: int) -> list[Instruction]:
    instructions: list[Instruction] = []
    num_logit_tiles = _ceil_div(QwenMoeConfig.vocab_size, TILE_N_GRANULARITY)  # 9496
    for n_start, n_end in _split_n_tiles(num_logit_tiles, sm_count):
        instructions.append(
            RMS_LMHead_GEMV(
                start_output_block_idx=n_start,
                end_output_block_idx=n_end,
            )
        )
    return instructions


# ---------------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------------


def schedule_layer(layer_idx: int, sm_count: int, num_partials: int) -> list[Instruction]:
    """One layer, in strict topological order."""
    instructions: list[Instruction] = []
    instructions.extend(schedule_qkvproj_gemv(layer_idx, sm_count))
    instructions.extend(schedule_attention_split_kv(layer_idx, num_partials))
    instructions.extend(schedule_attention_reduction(layer_idx, num_partials))
    instructions.extend(schedule_oproj_gemv(layer_idx, sm_count))
    instructions.extend(schedule_router_gemv(layer_idx, sm_count))
    instructions.extend(schedule_router_softmax_topk(layer_idx))
    instructions.extend(schedule_shared_upgate(layer_idx, sm_count))
    instructions.extend(schedule_shared_downproj(layer_idx, sm_count))
    instructions.extend(schedule_routed_upgate(layer_idx, sm_count))
    instructions.extend(schedule_routed_downproj(layer_idx, sm_count))
    instructions.extend(schedule_moe_combine_commit(layer_idx, sm_count))
    return instructions


def build_instruction_list(
    globs: Globals, num_layers: int | None = None
) -> list[Instruction]:
    device = _get_device(globs)
    sm_count = _get_sm_count(device)
    num_partials = get_num_partials_for_device(device)
    layers = (
        num_layers if num_layers is not None else QwenMoeConfig.num_hidden_layers
    )
    instructions: list[Instruction] = []
    for layer_idx in range(layers):
        instructions.extend(schedule_layer(layer_idx, sm_count, num_partials))
    if layers == QwenMoeConfig.num_hidden_layers:
        instructions.extend(schedule_lmhead_gemv(sm_count))
    return instructions


def build_instruction_queues(
    globs: Globals, num_layers: int | None = None
) -> list[list[Instruction]]:
    device = _get_device(globs)
    sm_count = _get_sm_count(device)
    instructions = build_instruction_list(globs, num_layers=num_layers)
    return _round_robin_assign_to_sms(instructions, sm_count)


def build(globs: Globals, num_layers: int | None = None) -> torch.Tensor:
    queues = build_instruction_queues(globs, num_layers=num_layers)
    return tensorize_instructions(globs, queues)


# ---------------------------------------------------------------------------------
# Barrier model shared by the scheduler, PyVM, and native kernel audit.
#
# `barrier_signals_of` / `barrier_waits_of` describe each instruction's
# dependency edges. Schedule-shape tests verify that every wait is fully signalled.
# ---------------------------------------------------------------------------------


def barrier_signals_of(ins: Instruction) -> list[tuple[int, int, int, int]]:
    """-> list of (layer, row, slot, amount) this instruction atomically adds."""
    op = ins.opcode()
    out: list[tuple[int, int, int, int]] = []
    if op == 1:
        per_head = QwenMoeConfig.num_qkv_blocks_per_head
        for block in range(ins.tile_N_start_idx, ins.tile_N_end_idx):
            out.append((ins.layer_idx, BARRIER_ROW_QKV_ROPE, block // per_head, 1))
    elif op == 2:
        # MHA: GQA group == 1, so the Q head served equals the KV head index.
        gqa = QwenMoeConfig.num_attention_heads // QwenMoeConfig.num_key_value_heads
        for i in range(gqa):
            q_head = ins.kv_head_idx * gqa + i
            out.append((ins.layer_idx, BARRIER_ROW_ATTN_PARTIAL, q_head, 1))
    elif op == 3:
        out.append(
            (ins.layer_idx, BARRIER_ROW_ATTN_REDUCTION, 0, ATTN_REDUCTION_SIZE)
        )
    elif op == 4:
        out.append(
            (
                ins.layer_idx,
                BARRIER_ROW_O_PROJ,
                0,
                ins.tile_N_end_idx - ins.tile_N_start_idx,
            )
        )
    elif op == 7:
        # Terminal op: nothing downstream, so it signals nothing.  Barrier row 6
        # (BARRIER_ROW_LM_HEAD) stays permanently zero and is reserved for a future
        # consumer (e.g. a sampling opcode).
        pass
    elif op == 8:
        out.append(
            (
                ins.layer_idx,
                BARRIER_ROW_ROUTER_GEMV,
                BARRIER_SLOT_ROUTER_LOGITS,
                ins.tile_N_end_idx - ins.tile_N_start_idx,
            )
        )
        if ins.write_normed_hidden:
            out.append(
                (
                    ins.layer_idx,
                    BARRIER_ROW_ROUTER_GEMV,
                    BARRIER_SLOT_NORMED_HIDDEN,
                    1,
                )
            )
    elif op == 9:
        out.append((ins.layer_idx, BARRIER_ROW_ROUTER_TOPK, 0, 1))
    elif op == 10:
        out.append(
            (
                ins.layer_idx,
                BARRIER_ROW_SHARED_UPGATE,
                0,
                ins.tile_N_end_idx - ins.tile_N_start_idx,
            )
        )
    elif op == 11:
        for t in range(ins.tile_N_start_idx, ins.tile_N_end_idx):
            out.append((ins.layer_idx, BARRIER_ROW_SHARED_DOWNPROJ, t, 1))
    elif op == 12:
        out.append(
            (
                ins.layer_idx,
                BARRIER_ROW_ROUTED_UPGATE,
                0,
                ins.tile_N_end_idx - ins.tile_N_start_idx,
            )
        )
    elif op == 13:
        for t in range(ins.tile_N_start_idx, ins.tile_N_end_idx):
            out.append((ins.layer_idx, BARRIER_ROW_ROUTED_DOWNPROJ, t, 1))
    elif op == 14:
        out.append(
            (
                ins.layer_idx,
                BARRIER_ROW_COMBINE_COMMIT,
                0,
                ins.tile_N_end_idx - ins.tile_N_start_idx,
            )
        )
    else:
        raise ValueError(f"unknown opcode {op}")
    return out


def barrier_waits_of(ins: Instruction) -> list[tuple[int, int, int, int]]:
    """-> list of (layer, row, slot, expected) this instruction spins on."""
    op = ins.opcode()
    c = QwenMoeConfig
    out: list[tuple[int, int, int, int]] = []
    if op == 1:
        if ins.layer_idx > 0:
            out.append(
                (
                    ins.layer_idx - 1,
                    BARRIER_ROW_COMBINE_COMMIT,
                    0,
                    c.num_hidden_blocks,
                )
            )
    elif op == 2:
        per_head = c.num_qkv_blocks_per_head
        gqa = c.num_attention_heads // c.num_key_value_heads
        for i in range(gqa):
            q_head = ins.kv_head_idx * gqa + i
            out.append((ins.layer_idx, BARRIER_ROW_QKV_ROPE, q_head, per_head))
        out.append(
            (
                ins.layer_idx,
                BARRIER_ROW_QKV_ROPE,
                BARRIER_SLOT_K_BASE + ins.kv_head_idx,
                per_head,
            )
        )
        out.append(
            (
                ins.layer_idx,
                BARRIER_ROW_QKV_ROPE,
                BARRIER_SLOT_V_BASE + ins.kv_head_idx,
                per_head,
            )
        )
    elif op == 3:
        for i in range(ATTN_REDUCTION_SIZE):
            out.append(
                (
                    ins.layer_idx,
                    BARRIER_ROW_ATTN_PARTIAL,
                    ins.head_start_idx + i,
                    ins.num_partials,
                )
            )
    elif op == 4:
        out.append(
            (
                ins.layer_idx,
                BARRIER_ROW_ATTN_REDUCTION,
                0,
                c.num_attention_heads,
            )
        )
    elif op == 7:
        out.append(
            (
                c.num_hidden_layers - 1,
                BARRIER_ROW_COMBINE_COMMIT,
                0,
                c.num_hidden_blocks,
            )
        )
    elif op == 8:
        out.append((ins.layer_idx, BARRIER_ROW_O_PROJ, 0, c.num_hidden_blocks))
    elif op == 9:
        out.append(
            (
                ins.layer_idx,
                BARRIER_ROW_ROUTER_GEMV,
                BARRIER_SLOT_ROUTER_LOGITS,
                c.num_experts,
            )
        )
    elif op == 10:
        # D2: waiting on "normed_hidden_states published" transitively implies o_proj
        # is complete, because opcode 8 waits on the full o_proj residual.
        out.append(
            (ins.layer_idx, BARRIER_ROW_ROUTER_GEMV, BARRIER_SLOT_NORMED_HIDDEN, 1)
        )
    elif op == 11:
        out.append(
            (
                ins.layer_idx,
                BARRIER_ROW_SHARED_UPGATE,
                0,
                c.num_shared_upgate_blocks,
            )
        )
    elif op == 12:
        out.append((ins.layer_idx, BARRIER_ROW_ROUTER_TOPK, 0, 1))
    elif op == 13:
        out.append(
            (
                ins.layer_idx,
                BARRIER_ROW_ROUTED_UPGATE,
                0,
                c.num_routed_upgate_blocks,
            )
        )
    elif op == 14:
        for t in range(ins.tile_N_start_idx, ins.tile_N_end_idx):
            out.append(
                (
                    ins.layer_idx,
                    BARRIER_ROW_SHARED_DOWNPROJ,
                    t,
                    c.num_downproj_split_k_shared,
                )
            )
            out.append(
                (
                    ins.layer_idx,
                    BARRIER_ROW_ROUTED_DOWNPROJ,
                    t,
                    c.num_experts_per_tok,
                )
            )
    else:
        raise ValueError(f"unknown opcode {op}")
    return out


def barrier_census(
    instructions: list[Instruction],
) -> tuple[dict[tuple[int, int, int], int], dict[tuple[int, int, int], set[int]]]:
    """-> (signals[(layer,row,slot)] = total, demands[(layer,row,slot)] = {expected...})."""
    signals: dict[tuple[int, int, int], int] = {}
    demands: dict[tuple[int, int, int], set[int]] = {}
    for ins in instructions:
        for layer, row, slot, amount in barrier_signals_of(ins):
            signals[(layer, row, slot)] = signals.get((layer, row, slot), 0) + amount
        for layer, row, slot, expected in barrier_waits_of(ins):
            demands.setdefault((layer, row, slot), set()).add(expected)
    return signals, demands


#: Every opcode that the H100 schedule actually emits, in emission order within a layer.
EMITTED_OPCODES_PER_LAYER = (1, 2, 3, 4, 8, 9, 10, 11, 12, 13, 14)
#: Emitted once, after the last layer.
EMITTED_OPCODES_TERMINAL = (7,)

#: Expected instruction count per opcode per layer on H100 (132 SMs, num_partials = 8).
#: Asserted by `test_9_schedule_shape.py`; also the table L4 implements against.
EXPECTED_COUNTS_PER_LAYER = {
    1: 132,
    2: 128,
    3: 16,
    4: 128,
    8: 60,
    9: 1,
    10: 132,
    11: 132,
    12: 132,
    13: 132,
    14: 128,
}
EXPECTED_COUNT_PER_LAYER_TOTAL = 1121
EXPECTED_LM_HEAD_COUNT = 132
EXPECTED_TOTAL_INSTRUCTIONS = 24 * EXPECTED_COUNT_PER_LAYER_TOTAL + EXPECTED_LM_HEAD_COUNT
EXPECTED_QUEUE_DEPTH = 205


__all__ = [
    "INTS_PER_INSTRUCTION",
    "NoOp",
    "build",
    "build_instruction_list",
    "build_instruction_queues",
    "tensorize_instructions",
    "schedule_layer",
    "schedule_qkvproj_gemv",
    "schedule_attention_split_kv",
    "schedule_attention_reduction",
    "schedule_oproj_gemv",
    "schedule_router_gemv",
    "schedule_router_softmax_topk",
    "schedule_shared_upgate",
    "schedule_shared_downproj",
    "schedule_routed_upgate",
    "schedule_routed_downproj",
    "schedule_moe_combine_commit",
    "schedule_lmhead_gemv",
    "barrier_signals_of",
    "barrier_waits_of",
    "barrier_census",
    "EMITTED_OPCODES_PER_LAYER",
    "EMITTED_OPCODES_TERMINAL",
    "EXPECTED_COUNTS_PER_LAYER",
    "EXPECTED_COUNT_PER_LAYER_TOTAL",
    "EXPECTED_TOTAL_INSTRUCTIONS",
    "EXPECTED_QUEUE_DEPTH",
]
