"""Instruction definitions for Qwen2.5 3B A100 single-batch decode."""
from dataclasses import dataclass

from qimeng_megakernel.models.instruction import Instruction
from ..model_config import Qwen2Config

# Granularity constants: single source of truth for instruction_defs and scheduler arithmetic.
TILE_M_GRANULARITY = 1
TILE_N_GRANULARITY = 16
NUM_DOWNPROJ_SPLIT_K = 4
SPLIT_K_GRANULARITY = 2752
UPGATE_SIGNAL_BUCKETS = NUM_DOWNPROJ_SPLIT_K
UPGATE_BLOCKS_PER_SIGNAL_BUCKET = SPLIT_K_GRANULARITY // TILE_N_GRANULARITY
ATTENTION_REDUCTION_WORKSPACE_CAPACITY = 25
DOWNPROJ_BARRIER_COUNT = (
    Qwen2Config.hidden_size // TILE_N_GRANULARITY
) * NUM_DOWNPROJ_SPLIT_K
assert Qwen2Config.intermediate_size == (
    NUM_DOWNPROJ_SPLIT_K * SPLIT_K_GRANULARITY
)
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
    Decode attention, split-kv by KV head (A100). One instruction = (kv_head_idx, partial_idx);
    handles gqa_ratio Q heads over a slice of the KV sequence.
    """
    layer_idx: int
    kv_head_idx: int  # 0 .. num_key_value_heads - 1
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
    """Consume one UpGate bucket and accumulate one split-K contribution."""
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
