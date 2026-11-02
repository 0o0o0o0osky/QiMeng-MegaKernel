"""Instruction layout for Qwen1.5-MoE A100 single-batch decode.

Dataclass field order is the serialized 32-int instruction format read by the CUDA kernel.
Opcodes 1-4 and 7 own attention/LM-head; opcodes 8-14 own the MoE route.
"""

from dataclasses import dataclass

from qimeng_megakernel.models.instruction import Instruction

from ..model_config import QwenMoeConfig

TILE_M_GRANULARITY = QwenMoeConfig.tile_m_granularity
TILE_N_GRANULARITY = QwenMoeConfig.tile_n_granularity
NUM_DOWNPROJ_SPLIT_K_SHARED = QwenMoeConfig.num_downproj_split_k_shared
SPLIT_K_GRANULARITY_SHARED = QwenMoeConfig.split_k_granularity_shared
MOE_INTERMEDIATE_SIZE = QwenMoeConfig.moe_intermediate_size
ROUTER_TILE_N_GRANULARITY = 1


# -------- MoE opcodes 8-14 (defined first so 1,7 can reference 14) --------


@dataclass
class Router_GEMV(Instruction):
    """RMSNorm + router logits for a contiguous physical-expert range."""
    layer_idx: int
    tile_N_start_idx: int
    tile_N_end_idx: int
    write_normed_hidden: int = 0
    tile_N_granularity: int = ROUTER_TILE_N_GRANULARITY

    @classmethod
    def opcode(cls) -> int:
        return 8

    @classmethod
    def prev_opcode(cls) -> tuple[int, ...]:
        return (OProj_GEMV_Residual.opcode(),)


@dataclass
class Router_SoftmaxTopK(Instruction):
    """After all Router_GEMV (op8) complete: softmax(router_logits_buf) → topk(4) → expert_indices/weights."""
    layer_idx: int

    @classmethod
    def opcode(cls) -> int:
        return 9

    @classmethod
    def prev_opcode(cls) -> tuple[int, ...]:
        return (Router_GEMV.opcode(),)


@dataclass
class SharedExpert_UpGate_SiLU(Instruction):
    """Shared gate/up GEMV + SiLU using opcode 8's normalized hidden state."""
    layer_idx: int
    tile_M_idx: int
    tile_N_start_idx: int
    tile_N_end_idx: int
    init_shared_out: int = 0
    tile_M_granularity: int = TILE_M_GRANULARITY
    tile_N_granularity: int = TILE_N_GRANULARITY

    @classmethod
    def opcode(cls) -> int:
        return 10

    @classmethod
    def prev_opcode(cls) -> tuple[int, ...]:
        return (Router_GEMV.opcode(),)


@dataclass
class SharedExpert_DownProj(Instruction):
    """Shared expert down_proj: intermediate → shared_out. Split-K over 5632."""
    layer_idx: int
    tile_M_idx: int
    tile_N_start_idx: int
    tile_N_end_idx: int
    split_K_idx: int
    tile_M_granularity: int = TILE_M_GRANULARITY
    tile_N_granularity: int = TILE_N_GRANULARITY
    split_K_granularity: int = SPLIT_K_GRANULARITY_SHARED

    @classmethod
    def opcode(cls) -> int:
        return 11

    @classmethod
    def prev_opcode(cls) -> tuple[int, ...]:
        return (SharedExpert_UpGate_SiLU.opcode(),)


@dataclass
class RoutedExpert_UpGate_SiLU(Instruction):
    """Routed expert gate+up+SiLU for one expert_idx. Writes per-expert intermediate. Contiguous tiles [start, end)."""
    layer_idx: int
    expert_idx: int
    tile_N_start_idx: int
    tile_N_end_idx: int
    tile_M_granularity: int = TILE_M_GRANULARITY
    tile_N_granularity: int = TILE_N_GRANULARITY

    @classmethod
    def opcode(cls) -> int:
        return 12

    @classmethod
    def prev_opcode(cls) -> tuple[int, ...]:
        return (Router_SoftmaxTopK.opcode(),)


@dataclass
class RoutedExpert_DownProj(Instruction):
    """Pure routed down-projection into the FP32 contribution accumulator."""
    layer_idx: int
    expert_idx: int
    tile_M_idx: int
    tile_N_start_idx: int
    tile_N_end_idx: int
    tile_M_granularity: int = TILE_M_GRANULARITY
    tile_N_granularity: int = TILE_N_GRANULARITY

    @classmethod
    def opcode(cls) -> int:
        return 13

    @classmethod
    def prev_opcode(cls) -> tuple[int, ...]:
        return (RoutedExpert_UpGate_SiLU.opcode(),)


@dataclass
class MoE_Combine_Commit(Instruction):
    """Combine shared/routed contributions and commit contiguous hidden tiles."""
    layer_idx: int
    tile_N_start_idx: int
    tile_N_end_idx: int
    tile_N_granularity: int = TILE_N_GRANULARITY

    @classmethod
    def opcode(cls) -> int:
        return 14

    @classmethod
    def prev_opcode(cls) -> tuple[int, ...]:
        return (SharedExpert_DownProj.opcode(), RoutedExpert_DownProj.opcode())


# -------- Attention + LM head (1-4, 7); 5,6 unused --------


@dataclass
class RMS_QKVProj_GEMV_RoPE(Instruction):
    """RMS + QKV GEMV + RoPE; writes q_post_rope, k_cache, v_cache. TILE_N=16."""
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
        return (MoE_Combine_Commit.opcode(),)


@dataclass
class DecodeAttentionSplitKV(Instruction):
    """Decode attention split-KV by KV head."""
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
    is_terminal: int = 1  # 1 = True for serialize compatibility
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


# Opcodes 5, 6 unused for this model (dense MLP replaced by MoE 8-12)


@dataclass
class RMS_LMHead_GEMV(Instruction):
    """Final layer: RMSNorm + lm_head GEMV → logits."""
    start_output_block_idx: int
    end_output_block_idx: int
    tile_M_granularity: int = TILE_M_GRANULARITY
    tile_N_granularity: int = TILE_N_GRANULARITY

    @classmethod
    def opcode(cls) -> int:
        return 7

    @classmethod
    def prev_opcode(cls) -> tuple[int, ...]:
        return (MoE_Combine_Commit.opcode(),)


INSTRUCTION_BY_OPCODE = {
    1: RMS_QKVProj_GEMV_RoPE,
    2: DecodeAttentionSplitKV,
    3: DecodeAttentionReduction,
    4: OProj_GEMV_Residual,
    7: RMS_LMHead_GEMV,
    8: Router_GEMV,
    9: Router_SoftmaxTopK,
    10: SharedExpert_UpGate_SiLU,
    11: SharedExpert_DownProj,
    12: RoutedExpert_UpGate_SiLU,
    13: RoutedExpert_DownProj,
    14: MoE_Combine_Commit,
}
