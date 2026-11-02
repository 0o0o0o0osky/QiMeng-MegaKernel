"""Qwen1.5-MoE-A2.7B H100 single-batch model and schedule constants.

``get_num_partials_for_device`` is shared by preprocess and the scheduler so the
attention instruction fan-out and intermediate-buffer shape stay identical.
"""

from __future__ import annotations

import torch


class QwenMoeConfig:
    num_hidden_layers = 24
    hidden_size = 2048
    head_dim = 128
    num_attention_heads = 16
    num_key_value_heads = 16
    num_heads = num_attention_heads + num_key_value_heads * 2  # 48, attention barrier slots
    num_attn_barrier_slots = num_heads  # 48: max slot used by attention ops
    qkv_out_dim = num_heads * head_dim
    vocab_size = 151936
    rms_norm_eps = 1e-6
    # Shared expert (one MLP per layer)
    intermediate_size = 5632  # shared_expert_intermediate_size
    # Routed experts (60 experts, 4 active per token)
    moe_intermediate_size = 1408
    num_experts = 60
    num_experts_per_tok = 4
    norm_topk_prob = False
    # Attention split/reduction
    attn_kv_block_size = 16
    #: Q heads merged by one opcode-3 instruction. This matches the CUDA
    #: Q_HEADS_PER_INSTRUCTION and its barrier signal count.
    attn_reduction_size = 1

    # L3 tiling granularity. Instruction tile_N fields index output-row blocks.
    tile_m_granularity = 1
    tile_n_granularity = 16

    # Scheduler instruction counts and barrier values derive from these block counts.
    num_hidden_blocks = hidden_size // tile_n_granularity  # 128
    num_qkv_blocks = qkv_out_dim // tile_n_granularity  # 384
    num_qkv_blocks_per_head = head_dim // tile_n_granularity  # 8
    num_shared_upgate_blocks = intermediate_size // tile_n_granularity  # 352
    num_moe_intermediate_blocks = moe_intermediate_size // tile_n_granularity  # 88
    num_routed_upgate_blocks = num_experts_per_tok * num_moe_intermediate_blocks  # 352
    num_routed_downproj_blocks = num_experts_per_tok * num_hidden_blocks  # 512
    num_lm_head_blocks = -(-vocab_size // tile_n_granularity)  # ceil -> 9496
    # Shared-expert DownProj split-K follows the K=1408 WGMMA page geometry.
    num_downproj_split_k_shared = 4
    split_k_granularity_shared = intermediate_size // num_downproj_split_k_shared  # 1408
    num_shared_downproj_accum_blocks = num_downproj_split_k_shared * num_hidden_blocks  # 512
    # Router GEMV uses one worker per expert to minimize its top-k critical path.
    router_gemv_workers = num_experts  # 60

    # Barrier layout: barriers[L, num_opcode_rows, barrier_slots]
    # Per-tile combine uses one slot per hidden tile = hidden_size / 16 = 128
    barrier_slots = num_hidden_blocks  # 128 >= num_attn_barrier_slots(48)

    # Barrier rows cover opcodes 1..14 plus one spare row.
    max_opcode = 14
    num_opcode_rows = max_opcode + 1  # 15

    # H100 hardware profile used by attention fan-out sizing.
    h100_sm_count = 132
    # Upper bound on the split-KV reduction list length: it is serialized into the
    # 32-int instruction word (INTS_PER_INSTRUCTION - 5 = 27 slots).
    max_num_partials = 27

    @classmethod
    def update_from_hf(cls, config):
        cls.num_hidden_layers = config.num_hidden_layers
        cls.hidden_size = config.hidden_size
        cls.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        cls.num_attention_heads = config.num_attention_heads
        cls.num_key_value_heads = getattr(
            config, "num_key_value_heads", config.num_attention_heads
        )
        cls.num_heads = cls.num_attention_heads + cls.num_key_value_heads * 2
        cls.num_attn_barrier_slots = cls.num_heads
        cls.qkv_out_dim = cls.num_heads * cls.head_dim
        cls.vocab_size = config.vocab_size
        cls.rms_norm_eps = getattr(config, "rms_norm_eps", 1e-6)
        cls.intermediate_size = getattr(
            config, "shared_expert_intermediate_size", config.intermediate_size
        )
        cls.moe_intermediate_size = getattr(config, "moe_intermediate_size", 1408)
        cls.num_experts = getattr(config, "num_experts", 60)
        cls.num_experts_per_tok = getattr(config, "num_experts_per_tok", 4)
        cls.norm_topk_prob = getattr(config, "norm_topk_prob", False)
        if hasattr(config, "attn_kv_block_size"):
            cls.attn_kv_block_size = config.attn_kv_block_size
        if hasattr(config, "attn_reduction_size"):
            cls.attn_reduction_size = config.attn_reduction_size
        cls._recompute_derived()

    @classmethod
    def update_from_dict(cls, config: dict):
        cls.num_hidden_layers = config["num_hidden_layers"]
        cls.hidden_size = config["hidden_size"]
        cls.head_dim = config.get("head_dim") or (
            config["hidden_size"] // config["num_attention_heads"]
        )
        cls.num_attention_heads = config["num_attention_heads"]
        cls.num_key_value_heads = config.get(
            "num_key_value_heads", cls.num_attention_heads
        )
        cls.num_heads = cls.num_attention_heads + cls.num_key_value_heads * 2
        cls.num_attn_barrier_slots = cls.num_heads
        cls.qkv_out_dim = cls.num_heads * cls.head_dim
        cls.vocab_size = config["vocab_size"]
        cls.rms_norm_eps = config.get("rms_norm_eps", 1e-6)
        cls.intermediate_size = config.get(
            "shared_expert_intermediate_size", config.get("intermediate_size", 5632)
        )
        cls.moe_intermediate_size = config.get("moe_intermediate_size", 1408)
        cls.num_experts = config.get("num_experts", 60)
        cls.num_experts_per_tok = config.get("num_experts_per_tok", 4)
        cls.norm_topk_prob = config.get("norm_topk_prob", False)
        cls._recompute_derived()

    @classmethod
    def _recompute_derived(cls):
        """Recompute scheduler block counts and barrier dimensions."""
        g = cls.tile_n_granularity
        cls.num_hidden_blocks = cls.hidden_size // g
        cls.num_qkv_blocks = cls.qkv_out_dim // g
        cls.num_qkv_blocks_per_head = cls.head_dim // g
        cls.num_shared_upgate_blocks = cls.intermediate_size // g
        cls.num_moe_intermediate_blocks = cls.moe_intermediate_size // g
        cls.num_routed_upgate_blocks = (
            cls.num_experts_per_tok * cls.num_moe_intermediate_blocks
        )
        cls.num_routed_downproj_blocks = (
            cls.num_experts_per_tok * cls.num_hidden_blocks
        )
        cls.num_lm_head_blocks = -(-cls.vocab_size // g)
        cls.split_k_granularity_shared = (
            cls.intermediate_size // cls.num_downproj_split_k_shared
        )
        cls.num_shared_downproj_accum_blocks = (
            cls.num_downproj_split_k_shared * cls.num_hidden_blocks
        )
        cls.router_gemv_workers = cls.num_experts
        cls.barrier_slots = cls.num_hidden_blocks
        cls.num_opcode_rows = cls.max_opcode + 1

    @classmethod
    def as_dict(cls):
        return {
            "num_hidden_layers": cls.num_hidden_layers,
            "hidden_size": cls.hidden_size,
            "head_dim": cls.head_dim,
            "num_attention_heads": cls.num_attention_heads,
            "num_key_value_heads": cls.num_key_value_heads,
            "qkv_out_dim": cls.qkv_out_dim,
            "vocab_size": cls.vocab_size,
            "rms_norm_eps": cls.rms_norm_eps,
            "intermediate_size": cls.intermediate_size,
            "moe_intermediate_size": cls.moe_intermediate_size,
            "num_experts": cls.num_experts,
            "num_experts_per_tok": cls.num_experts_per_tok,
            "norm_topk_prob": cls.norm_topk_prob,
            "num_opcode_rows": cls.num_opcode_rows,
            "barrier_slots": cls.barrier_slots,
            "tile_n_granularity": cls.tile_n_granularity,
            "num_hidden_blocks": cls.num_hidden_blocks,
            "split_k_granularity_shared": cls.split_k_granularity_shared,
        }


# Barrier map: barriers[layer, opcode - 1, slot], shape [24, 15, 128].
# Expected values count tile-block signals and remain stable when scheduler work is
# repartitioned across instructions.

BARRIER_ROW_QKV_ROPE = 0  # opcode 1
BARRIER_ROW_ATTN_PARTIAL = 1  # opcode 2
BARRIER_ROW_ATTN_REDUCTION = 2  # opcode 3
BARRIER_ROW_O_PROJ = 3  # opcode 4
BARRIER_ROW_LM_HEAD = 6  # opcode 7 (terminal, nobody waits on it)
BARRIER_ROW_ROUTER_GEMV = 7  # opcode 8
BARRIER_ROW_ROUTER_TOPK = 8  # opcode 9
BARRIER_ROW_SHARED_UPGATE = 9  # opcode 10
BARRIER_ROW_SHARED_DOWNPROJ = 10  # opcode 11
BARRIER_ROW_ROUTED_UPGATE = 11  # opcode 12
BARRIER_ROW_ROUTED_DOWNPROJ = 12  # opcode 13
BARRIER_ROW_COMBINE_COMMIT = 13  # opcode 14

#: Next layer's opcode 1 (and the terminal LM head) waits on this row/slot.
PREV_LAYER_BARRIER_ROW = BARRIER_ROW_COMBINE_COMMIT
PREV_LAYER_EXPECTED = QwenMoeConfig.num_hidden_blocks  # 128

#: Slot layout inside row 0 (opcode 1).  48 slots = 16 Q + 16 K + 16 V heads.
BARRIER_SLOT_Q_BASE = 0
BARRIER_SLOT_K_BASE = QwenMoeConfig.num_attention_heads  # 16
BARRIER_SLOT_V_BASE = (
    QwenMoeConfig.num_attention_heads + QwenMoeConfig.num_key_value_heads
)  # 32

#: Slot layout inside row 7 (opcode 8).
#:   slot 0 -> "all router logits written" (counts experts)
#:   slot 1 -> "normed_hidden_states published" (written once by the op8 with
#:             tile_N_start_idx == 0 and consumed by opcode 10)
BARRIER_SLOT_ROUTER_LOGITS = 0
BARRIER_SLOT_NORMED_HIDDEN = 1


def barrier_row(opcode: int) -> int:
    """Map an opcode to its barrier row."""
    if not 1 <= opcode <= QwenMoeConfig.max_opcode:
        raise ValueError(f"opcode {opcode} outside 1..{QwenMoeConfig.max_opcode}")
    return opcode - 1


def barrier_expected_table() -> dict[str, int]:
    """`(consumer, barrier)` -> expected count.  Authoritative source for L4.

    Keys are `"op<consumer>@row<row>.slot<slot-kind>"`.  Values are the number of
    *tile-block signals* the consumer must observe before it may proceed.
    """
    c = QwenMoeConfig
    return {
        # opcode 1 (layer L>0) waits on the previous layer's combine commit
        "op1@row13.slot0": c.num_hidden_blocks,  # 128
        # opcode 2 waits on its own Q head tile blocks, and on K/V head tile blocks
        "op2@row0.q_head": c.num_qkv_blocks_per_head,  # 8
        "op2@row0.k_head": c.num_qkv_blocks_per_head,  # 8
        "op2@row0.v_head": c.num_qkv_blocks_per_head,  # 8
        # opcode 3 waits on all split-KV partials of each of its 4 Q heads
        "op3@row1.q_head": get_num_partials_for_device(None),  # 8
        # opcode 4 waits on every attention-reduction head
        "op4@row2.slot0": c.num_attention_heads,  # 16
        # opcode 8 waits on the full o_proj residual
        "op8@row3.slot0": c.num_hidden_blocks,  # 128
        # opcode 9 waits on all router logits
        "op9@row7.slot0": c.num_experts,  # 60
        # opcode 10 waits ONLY on normed_hidden_states being published (which
        # transitively implies o_proj complete, because op8 waits on row 3)
        "op10@row7.slot1": 1,
        # opcode 11 waits on the whole shared gate/up/SiLU intermediate
        "op11@row9.slot0": c.num_shared_upgate_blocks,  # 352
        # opcode 12 waits on router top-k
        "op12@row8.slot0": 1,
        # opcode 13 waits on all routed gate/up/SiLU tiles of all 4 slots
        "op13@row11.slot0": c.num_routed_upgate_blocks,  # 352
        # opcode 14 waits per hidden tile: 4 shared split-K + 4 routed expert slots
        "op14@row10.tile": c.num_downproj_split_k_shared,  # 4
        "op14@row12.tile": c.num_experts_per_tok,  # 4
        # opcode 7 waits on the last layer's combine commit
        "op7@row13.slot0": c.num_hidden_blocks,  # 128
    }


def _get_sm_count(device: torch.device | None) -> int:
    """Return the H100 schedule SM count and reject smaller CUDA devices."""
    if device is None or getattr(device, "type", None) != "cuda":
        return QwenMoeConfig.h100_sm_count
    try:
        sm_count = int(torch.cuda.get_device_properties(device).multi_processor_count)
    except Exception:
        return QwenMoeConfig.h100_sm_count
    if sm_count < QwenMoeConfig.h100_sm_count:
        raise ValueError(
            f"This decode schedule targets H100 ({QwenMoeConfig.h100_sm_count} SMs); "
            f"device reports {sm_count}."
        )
    return min(sm_count, QwenMoeConfig.h100_sm_count)


def get_sm_count_for_device(device: torch.device | None = None) -> int:
    """Return the SM count used by the scheduler."""
    return _get_sm_count(device)


def get_num_partials_for_device(device: torch.device | None = None) -> int:
    """Return the one-wave Split-KV fan-out used by preprocess and scheduler.

    On a 132-SM H100 with 16 KV heads this selects eight partials, producing 128
    attention CTAs. The same value sizes both attention intermediate buffers.
    """
    sm_count = _get_sm_count(device)
    n_kv = QwenMoeConfig.num_key_value_heads
    return max(1, min(QwenMoeConfig.max_num_partials, sm_count // n_kv))
