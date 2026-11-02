"""Instruction defs for Qwen2.5 3B decode. Copy from LLaMA 3B; attention per-Q-head; downproj 6 segments (5*2048+768)."""
from qimeng_megakernel.models.instruction import Instruction
from dataclasses import dataclass

# Granularity constants: single source of truth for instruction_defs and scheduler arithmetic.
TILE_M_GRANULARITY = 1
TILE_N_GRANULARITY = 16
# DownProj split-K: five full 2048-wide segments plus the 768 tail.
# UpGate publishes each matching input bucket independently, so DownProj can
# start a segment as soon as that segment (rather than all 11008 values) is ready.
NUM_DOWNPROJ_SPLIT_K = 6
SPLIT_K_GRANULARITY = 2048
DOWNPROJ_SPLIT_K_STARTS = tuple(
    split_k_idx * SPLIT_K_GRANULARITY
    for split_k_idx in range(NUM_DOWNPROJ_SPLIT_K)
)
DOWNPROJ_SPLIT_K_SIZES = (2048, 2048, 2048, 2048, 2048, 768)
UPGATE_BLOCKS_PER_SIGNAL_BUCKET = (128, 128, 128, 128, 128, 48)
DOWNPROJ_BARRIER_COUNT = 128 * NUM_DOWNPROJ_SPLIT_K

assert len(DOWNPROJ_SPLIT_K_STARTS) == len(DOWNPROJ_SPLIT_K_SIZES)
assert len(DOWNPROJ_SPLIT_K_SIZES) == len(UPGATE_BLOCKS_PER_SIGNAL_BUCKET)
assert sum(DOWNPROJ_SPLIT_K_SIZES) == 11008
assert sum(UPGATE_BLOCKS_PER_SIGNAL_BUCKET) == 11008 // TILE_N_GRANULARITY

# Python mirror of decode_kernel/instruction_list.cuh::BARRIER_ROW.  Only the
# seven opcodes in this instruction layout may be mapped; keeping the bounds here
# prevents an invalid opcode from silently indexing an unrelated barrier row.
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
class RMS_QKVProj_GEMV_RoPE(Instruction):
    """
    rmsnorm + qkv projection gemv + rope on q and k + append k and v to kv cache.
    Qwen: qkv_bias is applied before RoPE (in python_vm).
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
        return (DownProj_GEMV_Residual.opcode(),)


@dataclass
class DecodeAttentionSplitKV(Instruction):
    """
    decode attention, use split-kv strategy (per-Q-head, same instruction layout as LLaMA 3B).
    """
    layer_idx: int
    attention_head_idx: int  # Q head index 0..num_attention_heads-1
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
    """
    o proj gemv residual
    """
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
    """
    rmsnorm + up gate proj gemv + silu
    """
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
    """
    down proj gemv residual (split-K over intermediate dim: 6 segments 5*2048+768).
    """
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
