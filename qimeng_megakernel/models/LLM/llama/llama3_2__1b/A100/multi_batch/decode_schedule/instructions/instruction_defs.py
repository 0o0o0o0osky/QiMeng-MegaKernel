"""Instruction definitions for the selected 1B A100 M>1 decode route."""

from dataclasses import dataclass

from qimeng_megakernel.models.instruction import Instruction

TILE_M_GRANULARITY = 16
TILE_N_GRANULARITY = 16


@dataclass
class QKVProj_GEMM_RoPE(Instruction):
    """QKV GEMM + RoPE + KV write from the route-owned attention RMS scratch."""

    layer_idx: int
    batch_size: int
    tile_M_idx: int
    tile_N_start_idx: int
    tile_N_end_idx: int
    tile_M_granularity: int = TILE_M_GRANULARITY
    tile_N_granularity: int = TILE_N_GRANULARITY

    @classmethod
    def opcode(cls) -> int:
        return 2

    @classmethod
    def prev_opcode(cls) -> tuple[int, ...]:
        return (AttnRMS.opcode(),)


@dataclass
class DecodeAttentionSplitKV(Instruction):
    layer_idx: int
    batch_idx: int
    kv_head_idx: int
    num_partials: int
    partial_idx: int
    # seq_len: KV attention length for this row (-1 => use decode_pos_id_for_batch+1 at runtime).
    seq_len: int = -1
    # is_active: 0=barrier-only no-op; 1=force active (ignore active_mask); -1=row_active_decode (default).
    is_active: int = -1

    @classmethod
    def opcode(cls) -> int:
        return 3

    @classmethod
    def prev_opcode(cls) -> tuple[int, ...]:
        return (QKVProj_GEMM_RoPE.opcode(),)


@dataclass
class DecodeAttentionReduction(Instruction):
    layer_idx: int
    batch_idx: int
    head_start_idx: int
    num_partials: int
    reduction_list: list[int]
    is_terminal: bool = True
    output_partial_idx: int = 0
    # valid_partials: len(reduction_list) when filtered (-1 => len(reduction_list)).
    valid_partials: int = -1
    is_active: int = -1

    @classmethod
    def opcode(cls) -> int:
        return 4

    @classmethod
    def prev_opcode(cls) -> tuple[int, ...]:
        return (DecodeAttentionSplitKV.opcode(),)


@dataclass
class OProj_GEMM_Residual(Instruction):
    layer_idx: int
    batch_size: int
    tile_M_idx: int
    tile_N_start_idx: int
    tile_N_end_idx: int
    reduction_block_idx: int
    tile_M_granularity: int = TILE_M_GRANULARITY
    tile_N_granularity: int = TILE_N_GRANULARITY

    @classmethod
    def opcode(cls) -> int:
        return 5

    @classmethod
    def prev_opcode(cls) -> tuple[int, ...]:
        return (DecodeAttentionReduction.opcode(),)


@dataclass
class UpGateProj_GEMM_SiLU(Instruction):
    layer_idx: int
    batch_size: int
    tile_M_idx: int
    tile_N_block_idxs: list[int]
    tile_M_granularity: int = TILE_M_GRANULARITY
    tile_N_granularity: int = TILE_N_GRANULARITY

    @classmethod
    def opcode(cls) -> int:
        return 7

    @classmethod
    def prev_opcode(cls) -> tuple[int, ...]:
        return (MlpRMS.opcode(),)


@dataclass
class DownProj_GEMM_Residual(Instruction):
    """Down proj + residual; full K (no split-K)."""

    layer_idx: int
    batch_size: int
    tile_M_idx: int
    tile_N_start_idx: int
    tile_N_end_idx: int
    tile_M_granularity: int = TILE_M_GRANULARITY
    tile_N_granularity: int = TILE_N_GRANULARITY

    @classmethod
    def opcode(cls) -> int:
        return 8

    @classmethod
    def prev_opcode(cls) -> tuple[int, ...]:
        return (UpGateProj_GEMM_SiLU.opcode(),)


@dataclass
class LMHeadProj_GEMM(Instruction):
    batch_size: int
    start_output_block_idx: int
    end_output_block_idx: int
    tile_M_granularity: int = TILE_M_GRANULARITY
    tile_N_granularity: int = TILE_N_GRANULARITY

    @classmethod
    def opcode(cls) -> int:
        return 10

    @classmethod
    def prev_opcode(cls) -> tuple[int, ...]:
        return (LMHeadRMS.opcode(),)


# This route materializes its three RMS results as explicit row instructions.


@dataclass
class AttnRMS(Instruction):
    """One CTA per batch row; CUDA reads inst[1]=layer, inst[2]=batch_size, inst[6]=batch_row."""

    layer_idx: int
    batch_size: int
    res0: int = 0
    res1: int = 0
    res2: int = 0
    batch_row: int = 0

    @classmethod
    def opcode(cls) -> int:
        return 1

    @classmethod
    def prev_opcode(cls) -> tuple[int, ...]:
        return (DownProj_GEMM_Residual.opcode(),)


@dataclass
class MlpRMS(Instruction):
    """One CTA per batch row; CUDA reads inst[1]=layer, inst[2]=batch_size, inst[6]=batch_row."""

    layer_idx: int
    batch_size: int
    res0: int = 0
    res1: int = 0
    res2: int = 0
    batch_row: int = 0

    @classmethod
    def opcode(cls) -> int:
        return 6

    @classmethod
    def prev_opcode(cls) -> tuple[int, ...]:
        return (OProj_GEMM_Residual.opcode(),)


@dataclass
class LMHeadRMS(Instruction):
    """Final LM-head RMS scratch; CUDA reads inst[1]=batch_size, inst[6]=batch_row."""

    batch_size: int
    res0: int = 0
    res1: int = 0
    res2: int = 0
    res3: int = 0
    batch_row: int = 0

    @classmethod
    def opcode(cls) -> int:
        return 9

    @classmethod
    def prev_opcode(cls) -> tuple[int, ...]:
        return (DownProj_GEMM_Residual.opcode(),)
