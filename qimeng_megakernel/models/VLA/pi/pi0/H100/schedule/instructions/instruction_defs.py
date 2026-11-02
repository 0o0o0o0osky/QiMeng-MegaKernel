"""Instruction-layout-accurate dataclasses for the Pi0 H100 kernels.

The three modules have independent opcode spaces.  ``Instruction.serialize``
emits dataclass fields in declaration order, so every field below corresponds
to the CUDA parser word at the same offset (after word 0, the opcode).

Keep apparently-unused/reserved fields: several kernels read sparse words
(notably SigLIP ``flags`` at word 11), and dropping the gaps silently shifts
the host/device ABI.
"""
from __future__ import annotations

from dataclasses import dataclass

from qimeng_megakernel.models.instruction import Instruction


@dataclass
class Pi0Instruction(Instruction):
    """Pi0 H100 instruction base; dependency edges are scheduler-owned."""

    @classmethod
    def prev_opcode(cls) -> tuple[int, ...]:
        return ()


def _contiguous_barrier_row(opcode: int, *, last_opcode: int, namespace: str) -> int:
    """Map one namespace-local contiguous opcode to its barrier row."""
    if type(opcode) is not int or not 1 <= opcode <= last_opcode:
        raise ValueError(f"unsupported {namespace} opcode for barrier row: {opcode!r}")
    return opcode - 1


# ── SigLIP ──────────────────────────────────────────────────────────────────


@dataclass
class _SiglipLnInstruction(Pi0Instruction):
    # CUDA: siglip_ln.cu / siglip_stats in siglip_ln.cu
    layer_idx: int = 0       # [1]
    token_row: int = 0       # [2]
    weight_row: int = 0      # [3] (stats_row for the folded-LN variant)
    dep_layer: int = 0       # [4]
    dep_row: int = 0         # [5]
    dep_expected: int = 0    # [6]
    num_tokens: int = 1      # [7]
    sq_row: int = 0          # [8] (0 selects the CUDA fallback weight_row + 1)
    reserved_9: int = 0      # [9]
    reserved_10: int = 0     # [10]
    flags: int = 0           # [11]


@dataclass
class SiglipLnAttn(_SiglipLnInstruction):
    @classmethod
    def opcode(cls) -> int:
        return 1


@dataclass
class SiglipLnFfn(_SiglipLnInstruction):
    @classmethod
    def opcode(cls) -> int:
        return 2


@dataclass
class _SiglipGemmInstruction(Pi0Instruction):
    # CUDA: siglip_gemm.cu::parsed_instruction
    layer_idx: int = 0       # [1]
    tile_m: int = 0          # [2] local_row
    tile_n: int = 0          # [3] local_col
    dep_layer: int = 0       # [4]
    dep_row: int = 0         # [5]
    dep_expected: int = 0    # [6]
    norm_sel: int = 0        # [7]
    out_row: int = 0         # [8] (ignored by implicit-output variants)
    out_col: int = 0         # [9] (ignored by implicit-output variants)
    bias_col: int = 0        # [10] (ignored by implicit-output variants)
    flags: int = 0           # [11]


@dataclass
class SiglipQkv(_SiglipGemmInstruction):
    @classmethod
    def opcode(cls) -> int:
        return 3


@dataclass
class SiglipAttn(Pi0Instruction):
    # CUDA: siglip_attn.cu::parsed_instruction
    layer_idx: int = 0       # [1]
    view_idx: int = 0        # [2]
    q_tile: int = 0          # [3]
    head_idx: int = 0        # [4]
    dep_layer: int = 0       # [5]
    dep_row: int = 0         # [6]
    dep_expected: int = 0    # [7]
    reserved_8: int = 0      # [8]
    reserved_9: int = 0      # [9]
    reserved_10: int = 0     # [10]
    flags: int = 0           # [11]

    @classmethod
    def opcode(cls) -> int:
        return 4


@dataclass
class SiglipOproj(_SiglipGemmInstruction):
    @classmethod
    def opcode(cls) -> int:
        return 5


@dataclass
class SiglipFc1(_SiglipGemmInstruction):
    @classmethod
    def opcode(cls) -> int:
        return 6


@dataclass
class SiglipFc2(_SiglipGemmInstruction):
    @classmethod
    def opcode(cls) -> int:
        return 7


@dataclass
class SiglipPatch(_SiglipGemmInstruction):
    @classmethod
    def opcode(cls) -> int:
        return 8


@dataclass
class SiglipFinalNorm(_SiglipLnInstruction):
    @classmethod
    def opcode(cls) -> int:
        return 9


@dataclass
class SiglipProjector(_SiglipGemmInstruction):
    @classmethod
    def opcode(cls) -> int:
        return 10


OPCODE_NAMES_SIGLIP = {
    1: "LN_ATTN",
    2: "LN_FFN",
    3: "QKV",
    4: "ATTN",
    5: "OPROJ",
    6: "FC1",
    7: "FC2",
    8: "PATCH",
    9: "FINAL_NORM",
    10: "PROJECTOR",
}

_SIGLIP_OPCODE_SEQUENCE = tuple(range(1, 11))
if tuple(OPCODE_NAMES_SIGLIP) != _SIGLIP_OPCODE_SEQUENCE:
    raise RuntimeError("SigLIP opcode table must remain contiguous from 1 through 10")


def siglip_barrier_row(opcode: int) -> int:
    """Return the validated canonical SigLIP barrier row for a local opcode."""
    return _contiguous_barrier_row(
        opcode, last_opcode=_SIGLIP_OPCODE_SEQUENCE[-1], namespace="SigLIP"
    )


# ── Gemma encoder ───────────────────────────────────────────────────────────


@dataclass
class _GemmaRmsInstruction(Pi0Instruction):
    # CUDA: gemma_rmsn.cu::parsed_instruction
    layer_idx: int = 0       # [1]
    token_row: int = 0       # [2]
    reserved_3: int = 0      # [3]
    dep_layer: int = 0       # [4]
    dep_row: int = 0         # [5]
    dep_expected: int = 0    # [6]
    num_tokens: int = 1      # [7]


@dataclass
class GemmaRmsnAttn(_GemmaRmsInstruction):
    @classmethod
    def opcode(cls) -> int:
        return 1


@dataclass
class GemmaRmsnFfn(_GemmaRmsInstruction):
    @classmethod
    def opcode(cls) -> int:
        return 2


@dataclass
class _GemmaGemmInstruction(Pi0Instruction):
    # CUDA: gemma_qkv_rope.cu / gemma_resid_gemm.cu
    layer_idx: int = 0       # [1]
    tile_m: int = 0          # [2] local_row
    tile_n: int = 0          # [3] local_col
    dep_layer: int = 0       # [4]
    dep_row: int = 0         # [5]
    dep_expected: int = 0    # [6]
    norm_sel: int = 0        # [7]


@dataclass
class GemmaQkvRope(_GemmaGemmInstruction):
    @classmethod
    def opcode(cls) -> int:
        return 3


@dataclass
class GemmaAttnPartial(Pi0Instruction):
    # CUDA: gemma_attn.cu::gemma_attn_partial::parsed_instruction
    layer_idx: int = 0       # [1]
    q_tile: int = 0          # [2]
    head_idx: int = 0        # [3]
    partial_idx: int = 0     # [4] kv_split
    dep_layer: int = 0       # [5]
    dep_row: int = 0         # [6]
    dep_expected: int = 0    # [7]
    kv_blk0: int = 0         # [8]
    kv_nblk: int = 0         # [9]
    seq_len: int = 0         # [10]

    @classmethod
    def opcode(cls) -> int:
        return 4


@dataclass
class GemmaAttnReduce(Pi0Instruction):
    # CUDA: gemma_attn.cu::gemma_attn_reduce::parsed_instruction
    layer_idx: int = 0       # [1]
    q_tile: int = 0          # [2]
    head_idx: int = 0        # [3]
    half: int = 0            # [4]
    dep_layer: int = 0       # [5]
    dep_row: int = 0         # [6]
    dep_expected: int = 0    # [7]
    num_partials: int = 1    # [8] n_splits

    @classmethod
    def opcode(cls) -> int:
        return 5


@dataclass
class GemmaOproj(_GemmaGemmInstruction):
    @classmethod
    def opcode(cls) -> int:
        return 6


@dataclass
class GemmaGate(_GemmaGemmInstruction):
    # CUDA's production opcode 7 is gemma_ffn_dual and also reads word 8.
    band_slot: int = 0       # [8]

    @classmethod
    def opcode(cls) -> int:
        return 7


@dataclass
class GemmaUpMul(_GemmaGemmInstruction):
    @classmethod
    def opcode(cls) -> int:
        return 8


@dataclass
class GemmaDown(_GemmaGemmInstruction):
    @classmethod
    def opcode(cls) -> int:
        return 9


@dataclass
class GemmaDownSplitK(Pi0Instruction):
    # CUDA: gemma_down_splitk.cu::parsed_instruction
    layer_idx: int = 0       # [1]
    tile_m: int = 0          # [2] local_row
    tile_n: int = 0          # [3] local_col
    dep_layer: int = 0       # [4]
    dep_row: int = 0         # [5]
    dep_expected: int = 0    # [6]
    k_split: int = 0         # [7]
    dep_slot: int = 0        # [8]

    @classmethod
    def opcode(cls) -> int:
        return 10


# ── Action expert ───────────────────────────────────────────────────────────


class _AeLogicalLayer:
    """Expose the logical layer while word 1 carries a weight plane."""

    bar_layer: int

    @property
    def layer_idx(self) -> int:
        return self.bar_layer


@dataclass
class _AeRmsInstruction(_AeLogicalLayer, Pi0Instruction):
    # CUDA: ae_rmsn.cu::parsed_instruction
    plane: int = 0           # [1] (unused by this kernel)
    token_row: int = 0       # [2]
    reserved_3: int = 0      # [3]
    dep_layer: int = 0       # [4]
    dep_row: int = 0         # [5]
    dep_expected: int = 0    # [6]
    bar_layer: int = 0       # [7] logical layer / publication plane
    num_tokens: int = 1      # [8]


@dataclass
class AeRmsnAttn(_AeRmsInstruction):
    @classmethod
    def opcode(cls) -> int:
        return 1


@dataclass
class AeRmsnFfn(_AeRmsInstruction):
    @classmethod
    def opcode(cls) -> int:
        return 2


@dataclass
class AeQkvRope(_AeLogicalLayer, Pi0Instruction):
    # CUDA: ae_qkv_rope.cu::parsed_instruction
    plane: int = 0           # [1] weight plane
    local_row: int = 0       # [2]
    local_col: int = 0       # [3]
    dep_layer: int = 0       # [4]
    dep_row: int = 0         # [5]
    dep_expected: int = 0    # [6]
    bar_layer: int = 0       # [7] logical layer / publication plane
    reserved_8: int = 0      # [8]
    kv_row_blk: int = 0      # [9]
    norm_sel: int = 0        # [10]

    @property
    def tile_n(self) -> int:
        return self.local_col

    @classmethod
    def opcode(cls) -> int:
        return 3


@dataclass
class AeAttnPartial(Pi0Instruction):
    # CUDA: ae_attn.cu::ae_attn_partial::parsed_instruction
    layer_idx: int = 0       # [1]
    head_idx: int = 0        # [2]
    partial_idx: int = 0     # [3] kv_split
    dep_layer: int = 0       # [4]
    dep_row: int = 0         # [5]
    dep_expected: int = 0    # [6]
    kv_blk0: int = 0         # [7]
    kv_nblk: int = 0         # [8]
    seq_len: int = 0         # [9]
    prefix_len: int = 0      # [10]
    state_key: int = 0       # [11]

    @classmethod
    def opcode(cls) -> int:
        return 4


@dataclass
class AeAttnReduce(Pi0Instruction):
    # CUDA: ae_attn.cu::ae_attn_reduce::parsed_instruction
    layer_idx: int = 0       # [1]
    head_idx: int = 0        # [2]
    col_blk: int = 0         # [3]
    dep_layer: int = 0       # [4]
    dep_row: int = 0         # [5]
    dep_expected: int = 0    # [6]
    n_splits: int = 1        # [7]

    @classmethod
    def opcode(cls) -> int:
        return 5


@dataclass
class _AeGemmInstruction(_AeLogicalLayer, Pi0Instruction):
    # CUDA: ae_gemm.cu / ae_ffn_dual.cu
    plane: int = 0           # [1] weight plane
    local_row: int = 0       # [2]
    local_col: int = 0       # [3]
    dep_layer: int = 0       # [4]
    dep_row: int = 0         # [5]
    dep_expected: int = 0    # [6]
    bar_layer: int = 0       # [7] logical layer / publication plane
    out_row: int = 0         # [8]
    out_col: int = 0         # [9]
    norm_sel: int = 0        # [10]

    @property
    def tile_n(self) -> int:
        return self.local_col


@dataclass
class AeOproj(_AeGemmInstruction):
    @classmethod
    def opcode(cls) -> int:
        return 6


@dataclass
class AeGate(_AeGemmInstruction):
    @classmethod
    def opcode(cls) -> int:
        return 7


@dataclass
class AeUpMul(_AeGemmInstruction):
    @classmethod
    def opcode(cls) -> int:
        return 8


@dataclass
class AeDown(_AeGemmInstruction):
    @classmethod
    def opcode(cls) -> int:
        return 9


@dataclass
class AeFfnDual(_AeGemmInstruction):
    @classmethod
    def opcode(cls) -> int:
        return 10


@dataclass
class _AeActionInstruction(_AeLogicalLayer, Pi0Instruction):
    # CUDA: ae_action.cu::parsed_instruction
    plane: int = 0           # [1]
    local_row: int = 0       # [2]
    local_col: int = 0       # [3]
    dep_layer: int = 0       # [4]
    dep_row: int = 0         # [5]
    dep_expected: int = 0    # [6]
    bar_layer: int = 0       # [7]
    out_row: int = 0         # [8]
    out_col: int = 0         # [9]
    bias_row: int = 0        # [10]
    norm_sel: int = 0        # [11]
    n_valid_row: int = 0     # [12]


@dataclass
class AeActIn(_AeActionInstruction):
    @classmethod
    def opcode(cls) -> int:
        return 11


@dataclass
class AeActMlp(_AeActionInstruction):
    @classmethod
    def opcode(cls) -> int:
        return 12


@dataclass
class AeRmsnOut(_AeRmsInstruction):
    @classmethod
    def opcode(cls) -> int:
        return 13


@dataclass
class AeActOut(_AeActionInstruction):
    @classmethod
    def opcode(cls) -> int:
        return 14


OPCODE_NAMES_SIGLIP = {
    1: "LN_ATTN",
    2: "LN_FFN",
    3: "QKV",
    4: "ATTN",
    5: "OPROJ",
    6: "FC1",
    7: "FC2",
    8: "PATCH",
    9: "FINAL_NORM",
    10: "PROJECTOR",
}

OPCODE_NAMES_GEMMA = {
    1: "RMSN_ATTN",
    2: "RMSN_FFN",
    3: "QKV_ROPE",
    4: "ATTN_PARTIAL",
    5: "ATTN_REDUCE",
    6: "OPROJ",
    7: "GATE",
    8: "UP_MUL",
    9: "DOWN",
    10: "DOWN_SPLITK",
}

_GEMMA_OPCODE_SEQUENCE = tuple(range(1, 11))
if tuple(OPCODE_NAMES_GEMMA) != _GEMMA_OPCODE_SEQUENCE:
    raise RuntimeError("Gemma opcode table must remain contiguous from 1 through 10")


def gemma_barrier_row(opcode: int) -> int:
    """Return the validated canonical Gemma barrier row for a local opcode."""
    return _contiguous_barrier_row(
        opcode, last_opcode=_GEMMA_OPCODE_SEQUENCE[-1], namespace="Gemma"
    )

OPCODE_NAMES_AE = {
    1: "RMSN_ATTN",
    2: "RMSN_FFN",
    3: "QKV_ROPE",
    4: "ATTN_PARTIAL",
    5: "ATTN_REDUCE",
    6: "OPROJ",
    7: "GATE",
    8: "UP_MUL",
    9: "DOWN",
    10: "FFN_DUAL",
    11: "ACT_IN",
    12: "ACT_MLP",
    13: "RMSN_OUT",
    14: "ACT_OUT",
}

_AE_OPCODE_SEQUENCE = tuple(range(1, 15))
if tuple(OPCODE_NAMES_AE) != _AE_OPCODE_SEQUENCE:
    raise RuntimeError("AE opcode table must remain contiguous from 1 through 14")


def ae_barrier_row(opcode: int) -> int:
    """Return the validated canonical AE barrier row for a local opcode."""
    return _contiguous_barrier_row(
        opcode, last_opcode=_AE_OPCODE_SEQUENCE[-1], namespace="AE"
    )
