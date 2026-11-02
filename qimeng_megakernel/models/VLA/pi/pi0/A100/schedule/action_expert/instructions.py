from __future__ import annotations

from dataclasses import dataclass

import torch

DINST_WIDTH = 32
DINST_NO_BARRIER = -1
SPLITK_REDUCE_PHASE = -1


class DOP:
    NOOP = 0
    STATE_IN = 300
    ACTION_IN = 301
    ACTION_MLP = 302
    # Values 303, 305, 307, and 310 remain reserved for ABI compatibility.
    QKV_ROPE_KV = 304
    ATTN_QK = 312       # materialized: scores = Qpacked @ K^T
    ATTN_SOFTMAX = 313  # materialized: row-softmax over keys
    ATTN_PV = 314       # materialized: attn_out = scores @ V(^T)
    OPROJ_RESIDUAL = 306
    GATE_UP_DUAL = 308
    DOWN_RESIDUAL = 309
    ACTION_OUT = 311


class BR:
    STATE_IN_DONE = 0
    ACTION_IN_DONE = 1
    ACTION_MLP_DONE = 2
    # Rows 3, 7, 9, and 13 are reserved.
    QKV_ROPE_KV_DONE = 4
    ATTN_QK_DONE = 5
    ATTN_DONE = 6
    OPROJ_RESIDUAL_DONE = 8
    GATE_UP_DUAL_DONE = 10
    DOWN_PARTIAL_DONE = 11
    DOWN_RESIDUAL_DONE = 12
    ACTION_OUT_DONE = 14
    ATTN_SM_DONE = 15         # materialized softmax phase (appended row)


kDecBarrierRows = 16
# Materialized attention uses slot=band for QK and softmax completion;
# every other barrier row uses slot 0.
# Keep this value consistent with ``kDecBarrierSlots`` in
# ``kernel/action_expert/globals.cuh``.
kDecBarrierSlots = 8


def decoder_barrier_byte_offset(flow_step: int, layer_slot: int, row: int, slot: int, num_layer_slots: int) -> int:
    idx = ((flow_step * num_layer_slots + layer_slot) * kDecBarrierRows + row) * kDecBarrierSlots + slot
    return idx * 4


@dataclass
class DecoderInstruction:
    opcode: int = DOP.NOOP
    tile_m_start: int = 0
    tile_m_end: int = 0
    tile_n_start: int = 0
    tile_n_end: int = 0
    dep_counter_offset: int = DINST_NO_BARRIER
    dep_expected_count: int = 0
    out_counter_offset: int = DINST_NO_BARRIER
    M: int = 0
    N: int = 0
    K: int = 0
    layer_idx: int = 0
    flow_step: int = 0
    k_split_idx: int = 0
    k_splits: int = 0


FIELD_LAYOUT: dict[str, int] = {
    "opcode": 0,
    "tile_m_start": 1,
    "tile_m_end": 2,
    "tile_n_start": 3,
    "tile_n_end": 4,
    "dep_counter_offset": 5,
    "dep_expected_count": 6,
    "out_counter_offset": 7,
    "M": 8,
    "N": 9,
    "K": 10,
    "layer_idx": 11,
    "flow_step": 17,
    "k_split_idx": 18,
    "k_splits": 19,
}


def serialize_instructions(queues: list[list[DecoderInstruction]], device: str = "cuda") -> torch.Tensor:
    if not queues:
        return torch.empty((0, 0, DINST_WIDTH), dtype=torch.int32, device=device)
    max_len = max((len(queue) for queue in queues), default=0)
    max_len = max(max_len, 1)
    packed = torch.zeros(
        (len(queues), max_len, DINST_WIDTH), dtype=torch.int32
    )
    for cta_idx, queue in enumerate(queues):
        for pc, instruction in enumerate(queue):
            for field, slot in FIELD_LAYOUT.items():
                packed[cta_idx, pc, slot] = int(getattr(instruction, field))
    return packed.to(device)
