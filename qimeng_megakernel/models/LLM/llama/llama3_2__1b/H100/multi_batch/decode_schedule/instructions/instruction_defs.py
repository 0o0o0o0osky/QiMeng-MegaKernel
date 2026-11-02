"""Hopper multi-batch instruction ABI: logical B<=16, one physical WGMMA M=64 tile."""

from dataclasses import dataclass

from qimeng_megakernel.models.instruction import Instruction

TILE_M_GRANULARITY = 64
TILE_N_GRANULARITY = 128
QKV_TILE_N = 128
OPROJ_TILE_N = 128
UPGATE_TILE_N = 64
DOWNPROJ_TILE_N = 128
LMHEAD_TILE_N = 128


@dataclass
class RMS_QKVProj_GEMM_RoPE(Instruction):
    """RMSNorm + QKV GEMM + RoPE + KV write. batch_size <= TILE_M_GRANULARITY (one M-tile)."""

    layer_idx: int
    batch_size: int
    tile_M_idx: int
    tile_N_start_idx: int
    tile_N_end_idx: int
    tile_M_granularity: int = TILE_M_GRANULARITY
    tile_N_granularity: int = QKV_TILE_N

    @classmethod
    def opcode(cls) -> int:
        return 1

    @classmethod
    def prev_opcode(cls) -> tuple[int, ...]:
        return (DownProj_GEMM_Residual.opcode(),)


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
        return 2

    @classmethod
    def prev_opcode(cls) -> tuple[int, ...]:
        return (RMS_QKVProj_GEMM_RoPE.opcode(),)


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
        return 3

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
    tile_N_granularity: int = OPROJ_TILE_N

    @classmethod
    def opcode(cls) -> int:
        return 4

    @classmethod
    def prev_opcode(cls) -> tuple[int, ...]:
        return (DecodeAttentionReduction.opcode(),)


@dataclass
class RMS_UpGateProj_GEMM_SiLU(Instruction):
    layer_idx: int
    batch_size: int
    tile_M_idx: int
    tile_N_block_idxs: list[int]
    tile_M_granularity: int = TILE_M_GRANULARITY
    tile_N_granularity: int = UPGATE_TILE_N

    @classmethod
    def opcode(cls) -> int:
        return 5

    @classmethod
    def prev_opcode(cls) -> tuple[int, ...]:
        return (OProj_GEMM_Residual.opcode(),)


@dataclass
class DownProj_GEMM_Residual(Instruction):
    """Down proj + residual; full K (no split-K)."""

    layer_idx: int
    batch_size: int
    tile_M_idx: int
    tile_N_start_idx: int
    tile_N_end_idx: int
    tile_M_granularity: int = TILE_M_GRANULARITY
    tile_N_granularity: int = DOWNPROJ_TILE_N

    @classmethod
    def opcode(cls) -> int:
        return 6

    @classmethod
    def prev_opcode(cls) -> tuple[int, ...]:
        return (RMS_UpGateProj_GEMM_SiLU.opcode(),)


@dataclass
class RMS_LMHead_GEMM(Instruction):
    batch_size: int
    start_output_block_idx: int
    end_output_block_idx: int
    tile_M_granularity: int = TILE_M_GRANULARITY
    tile_N_granularity: int = LMHEAD_TILE_N

    @classmethod
    def opcode(cls) -> int:
        return 7

    @classmethod
    def prev_opcode(cls) -> tuple[int, ...]:
        return (DownProj_GEMM_Residual.opcode(),)


# Opcodes 8–10 remain part of the public Python-VM ABI.  The H100 persistent
# runtime follows the single-batch fusion boundaries and does not emit them.


@dataclass
class RMS_Attn_Norm(Instruction):
    """One CTA per batch row; CUDA reads inst[1]=layer, inst[2]=batch_size, inst[6]=batch_row."""

    layer_idx: int
    batch_size: int
    res0: int = 0
    res1: int = 0
    res2: int = 0
    batch_row: int = 0

    @classmethod
    def opcode(cls) -> int:
        return 8

    @classmethod
    def prev_opcode(cls) -> tuple[int, ...]:
        return (DownProj_GEMM_Residual.opcode(),)


@dataclass
class RMS_Mlp_Norm(Instruction):
    """One CTA per batch row; CUDA reads inst[1]=layer, inst[2]=batch_size, inst[6]=batch_row."""

    layer_idx: int
    batch_size: int
    res0: int = 0
    res1: int = 0
    res2: int = 0
    batch_row: int = 0

    @classmethod
    def opcode(cls) -> int:
        return 9

    @classmethod
    def prev_opcode(cls) -> tuple[int, ...]:
        return (OProj_GEMM_Residual.opcode(),)


@dataclass
class RMS_LMHead_Norm(Instruction):
    """Final LM-head RMS scratch; CUDA reads inst[1]=batch_size, inst[6]=batch_row."""

    batch_size: int
    res0: int = 0
    res1: int = 0
    res2: int = 0
    res3: int = 0
    batch_row: int = 0

    @classmethod
    def opcode(cls) -> int:
        return 10

    @classmethod
    def prev_opcode(cls) -> tuple[int, ...]:
        return (DownProj_GEMM_Residual.opcode(),)
