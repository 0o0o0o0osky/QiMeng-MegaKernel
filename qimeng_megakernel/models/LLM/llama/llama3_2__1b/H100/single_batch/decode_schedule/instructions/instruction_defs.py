from qimeng_megakernel.models.instruction import Instruction
from dataclasses import dataclass

# Granularity constants: single source of truth for instruction_defs and scheduler arithmetic.
TILE_M_GRANULARITY = 1
TILE_N_GRANULARITY = 16
SPLIT_K_GRANULARITY = 2048

# UPGATE -> DOWNPROJ dependency granularity.  downproj split-K instruction b
# reads exactly silu_out[b*SPLIT_K_GRANULARITY : (b+1)*SPLIT_K_GRANULARITY],
# produced by exactly the upgate output blocks in
# [b*UPGATE_BLOCKS_PER_SIGNAL_BUCKET, (b+1)*UPGATE_BLOCKS_PER_SIGNAL_BUCKET).
# So the edge is UPGATE_SIGNAL_BUCKETS independent buckets, not one device-wide
# barrier.  Must stay in lockstep with decode_kernel/globals.cuh -- see the
# numbered list there.
UPGATE_BLOCKS_PER_SIGNAL_BUCKET = SPLIT_K_GRANULARITY // TILE_N_GRANULARITY  # 128
assert SPLIT_K_GRANULARITY % TILE_N_GRANULARITY == 0, (
    "a upgate output block must not straddle two buckets"
)

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

    Every split writes its
    contribution directly to hidden_states and opcode 6 publishes the total
    number of completed output tiles in barrier slot 0.
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
