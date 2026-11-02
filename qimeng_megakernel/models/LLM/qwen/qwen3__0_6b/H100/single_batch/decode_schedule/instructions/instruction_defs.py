"""Instruction definitions for Qwen3 0.6B H100 decode."""
from qimeng_megakernel.models.instruction import Instruction
from dataclasses import dataclass

# All serialized output-block indices are consumed directly by native 16-row
# TMA/matvec pipelines. Keep this output block independent from higher-level head sizes.
TILE_M_GRANULARITY = 1
TILE_N_GRANULARITY = 16
RMS_QKV_BLOCK_SIZE = TILE_N_GRANULARITY

# Attention Op2 keeps QKNorm+RoPE/cache preparation inside the attention opcode,
# but exactly one split partial owns that prologue for each Q head.
ATTENTION_PREPARE_PRODUCER_PARTIAL = 0

# Native DownProj splits the 3072-column reduction into three 1024-column segments.
# ``split_K_idx`` is serialized at word[5] and is part of the instruction ABI.
NUM_DOWNPROJ_SPLIT_K = 3
SPLIT_K_GRANULARITY = 1024

# UpGate produces one 16-element block at a time.  Each DownProj split reads
# exactly one 1024-element slice, so the dependency is three independent
# 64-block buckets rather than one device-wide 192-block barrier.
UPGATE_SIGNAL_BUCKETS = NUM_DOWNPROJ_SPLIT_K
UPGATE_BLOCKS_PER_SIGNAL_BUCKET = SPLIT_K_GRANULARITY // TILE_N_GRANULARITY
assert SPLIT_K_GRANULARITY % TILE_N_GRANULARITY == 0

# Python mirror of decode_kernel/instruction_list.cuh::BARRIER_ROW. Only the
# seven opcodes in this instruction layout may be mapped; validating the bounds keeps
# malformed instructions from silently indexing an unrelated barrier row.
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
    """RMSNorm + QKV GEMV into ``qkv_raw``.

    ``tile_N_start_idx`` and ``tile_N_end_idx`` serialize as a half-open range of
    ``RMS_QKV_BLOCK_SIZE``-element output blocks. Q/K norm, RoPE, and KV writes
    belong to the following attention instruction, matching the native pipeline.
    """

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

@dataclass
class DecodeAttentionSplitKV(Instruction):
    """Decode attention over one KV head and its two-Q-head GQA group.

    ``attention_head_idx`` is the first Q head in the group and is therefore
    always even for Qwen3 0.6B.  The serialized field layout is unchanged.
    """
    layer_idx: int
    attention_head_idx: int
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
    """o proj gemv residual."""
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
    """rmsnorm + up gate proj gemv + silu."""
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
    """Down proj GEMV residual over one of three native split-K segments."""
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
