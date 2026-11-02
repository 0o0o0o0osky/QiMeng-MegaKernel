from qimeng_megakernel.models.instruction import Instruction
from dataclasses import dataclass

# Granularity constants: single source of truth for instruction_defs and scheduler arithmetic.
TILE_M_GRANULARITY = 1
TILE_N_GRANULARITY = 16
SPLIT_K_GRANULARITY = 2048
NUM_DOWNPROJ_SPLIT_K = 8192 // SPLIT_K_GRANULARITY
DOWNPROJ_BARRIER_COUNT = (3072 // TILE_N_GRANULARITY) * NUM_DOWNPROJ_SPLIT_K

# UpGate publishes exactly the slice consumed by one DownProj split-K
# instruction.  The slot is the split-K index; the count is the number of
# 16-element UpGate output blocks in that slice.
assert SPLIT_K_GRANULARITY % TILE_N_GRANULARITY == 0
UPGATE_BLOCKS_PER_SIGNAL_BUCKET = (
    SPLIT_K_GRANULARITY // TILE_N_GRANULARITY
)
assert NUM_DOWNPROJ_SPLIT_K == 4
assert DOWNPROJ_BARRIER_COUNT == 768

FIRST_OPCODE = 1
LAST_OPCODE = 7


def BARRIER_ROW(opcode: int) -> int:
    """Map a non-padding opcode to its zero-based barrier row."""
    if type(opcode) is not int:
        raise TypeError(f"opcode must be int, got {type(opcode).__name__}")
    if not FIRST_OPCODE <= opcode <= LAST_OPCODE:
        raise ValueError(
            f"opcode must be in [{FIRST_OPCODE}, {LAST_OPCODE}], got {opcode}"
        )
    return opcode - FIRST_OPCODE


@dataclass
class RMS_QKVProj_GEMV_RoPE(Instruction):
    """
    rmsnorm + qkv projection gemv + rope on q and k + append k and v to kv cache
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
    decode attention, use split-kv strategy
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
    DownProj split-K GEMV with an in-place BF16 residual update.

    Split ``b`` consumes only UpGate signal bucket ``b``.  The scheduler,
    Python VM, and CUDA kernel must therefore use ``split_K_idx`` as the
    barrier slot and ``UPGATE_BLOCKS_PER_SIGNAL_BUCKET`` as its expected count.
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
