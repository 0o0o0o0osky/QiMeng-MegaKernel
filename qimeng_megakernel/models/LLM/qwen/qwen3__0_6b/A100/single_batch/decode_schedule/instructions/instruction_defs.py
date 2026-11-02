"""Instruction definitions for Qwen3 0.6B A100 single-batch decode."""
from dataclasses import dataclass

from qimeng_megakernel.models.instruction import Instruction
from ..model_config import Qwen3Config


TILE_M_GRANULARITY = 1
TILE_N_GRANULARITY = 16
RMS_QKV_BLOCK_SIZE = TILE_N_GRANULARITY

# Raw Q/K/V readiness is partitioned by the dependency group consumed by one
# KV-head SplitKV family.  This avoids making unrelated groups contend on one
# counter while preserving the ordinary (row, slot, expected) accounting.
QKV_RAW_READY_SLOT_BASE = 0
QKV_RAW_READY_SLOTS = Qwen3Config.num_key_value_heads
ATTENTION_PREPARED_SLOT_BASE = QKV_RAW_READY_SLOT_BASE + QKV_RAW_READY_SLOTS
ATTENTION_PREPARE_PRODUCER_PARTIAL = 0
ATTENTION_REDUCTION_WORKSPACE_CAPACITY = 16

QKV_BLOCKS_PER_HEAD = Qwen3Config.head_dim // RMS_QKV_BLOCK_SIZE
QKV_GQA_GROUP = Qwen3Config.num_attention_heads // Qwen3Config.num_key_value_heads
QKV_RAW_BLOCKS_PER_GROUP = (QKV_GQA_GROUP + 2) * QKV_BLOCKS_PER_HEAD
assert Qwen3Config.head_dim % RMS_QKV_BLOCK_SIZE == 0
assert Qwen3Config.num_attention_heads % Qwen3Config.num_key_value_heads == 0


def qkv_raw_group_for_block(block_idx: int) -> int:
    """Map a block in concatenated [Q, K, V] output order to its KV group."""
    q_blocks = Qwen3Config.num_attention_heads * QKV_BLOCKS_PER_HEAD
    kv_blocks = Qwen3Config.num_key_value_heads * QKV_BLOCKS_PER_HEAD
    total_blocks = q_blocks + 2 * kv_blocks
    if not 0 <= block_idx < total_blocks:
        raise ValueError(f"QKV raw block {block_idx} is outside [0, {total_blocks})")
    if block_idx < q_blocks:
        q_head_idx = block_idx // QKV_BLOCKS_PER_HEAD
        return q_head_idx // QKV_GQA_GROUP
    if block_idx < q_blocks + kv_blocks:
        return (block_idx - q_blocks) // QKV_BLOCKS_PER_HEAD
    return (block_idx - q_blocks - kv_blocks) // QKV_BLOCKS_PER_HEAD

# Route-selected exact split-K profile.  Two 1536-column buckets fit the A100
# single-CTA SMEM budget and halve residual atomic contributions versus 4x768.
NUM_DOWNPROJ_SPLIT_K = 2
SPLIT_K_GRANULARITY = 1536
UPGATE_SIGNAL_BUCKETS = NUM_DOWNPROJ_SPLIT_K
UPGATE_BLOCKS_PER_SIGNAL_BUCKET = SPLIT_K_GRANULARITY // TILE_N_GRANULARITY
assert SPLIT_K_GRANULARITY % TILE_N_GRANULARITY == 0

FIRST_OPCODE = 1
LAST_OPCODE = 7


def BARRIER_ROW(opcode: int) -> int:
    if type(opcode) is not int:
        raise TypeError(f"opcode must be int, got {type(opcode).__name__}")
    if not FIRST_OPCODE <= opcode <= LAST_OPCODE:
        raise ValueError(
            f"opcode {opcode} is outside the opcode range "
            f"[{FIRST_OPCODE}, {LAST_OPCODE}]"
        )
    return opcode - FIRST_OPCODE


@dataclass
class RMS_QKVProj_GEMV(Instruction):
    """RMSNorm + QKV GEMV into ``qkv_raw`` in 16-element output blocks."""

    layer_idx: int
    tile_M_idx: int
    tile_N_start_idx: int
    tile_N_end_idx: int
    tile_M_granularity: int = TILE_M_GRANULARITY
    tile_N_granularity: int = RMS_QKV_BLOCK_SIZE

    @classmethod
    def opcode(cls) -> int:
        return 1

    @classmethod
    def prev_opcode(cls) -> tuple[int, ...]:
        return (DownProj_GEMV_Residual.opcode(),)


RMS_QKVProj_GEMV_RoPE = RMS_QKVProj_GEMV


@dataclass
class DecodeAttentionSplitKV(Instruction):
    """One KV head, its two-Q-head GQA group, and one SplitKV partial."""

    layer_idx: int
    kv_head_idx: int
    num_partials: int
    partial_idx: int

    @classmethod
    def opcode(cls) -> int:
        return 2

    @classmethod
    def prev_opcode(cls) -> tuple[int, ...]:
        return (RMS_QKVProj_GEMV.opcode(),)


@dataclass
class DecodeAttentionReduction(Instruction):
    layer_idx: int
    head_start_idx: int
    num_partials: int
    reduction_list: list[int]
    is_terminal: bool = True
    output_partial_idx: int = 0

    @classmethod
    def opcode(cls) -> int:
        return 3

    @classmethod
    def prev_opcode(cls) -> tuple[int, ...]:
        return (DecodeAttentionSplitKV.opcode(),)


@dataclass
class OProj_GEMV_Residual(Instruction):
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


@dataclass
class RMS_UpGateProj_GEMV_SiLU(Instruction):
    layer_idx: int
    tile_M_idx: int
    tile_N_block_idxs: list[int]
    tile_M_granularity: int = TILE_M_GRANULARITY
    tile_N_granularity: int = TILE_N_GRANULARITY

    @classmethod
    def opcode(cls) -> int:
        return 5

    @classmethod
    def prev_opcode(cls) -> tuple[int, ...]:
        return (OProj_GEMV_Residual.opcode(),)


@dataclass
class DownProj_GEMV_Residual(Instruction):
    layer_idx: int
    tile_M_idx: int
    tile_N_start_idx: int
    tile_N_end_idx: int
    split_K_idx: int
    tile_M_granularity: int = TILE_M_GRANULARITY
    tile_N_granularity: int = TILE_N_GRANULARITY
    split_K_granularity: int = SPLIT_K_GRANULARITY

    @classmethod
    def opcode(cls) -> int:
        return 6

    @classmethod
    def prev_opcode(cls) -> tuple[int, ...]:
        return (RMS_UpGateProj_GEMV_SiLU.opcode(),)


@dataclass
class RMS_LMHead_GEMV(Instruction):
    start_output_block_idx: int
    end_output_block_idx: int
    tile_M_granularity: int = TILE_M_GRANULARITY
    tile_N_granularity: int = TILE_N_GRANULARITY

    @classmethod
    def opcode(cls) -> int:
        return 7

    @classmethod
    def prev_opcode(cls) -> tuple[int, ...]:
        return (DownProj_GEMV_Residual.opcode(),)
