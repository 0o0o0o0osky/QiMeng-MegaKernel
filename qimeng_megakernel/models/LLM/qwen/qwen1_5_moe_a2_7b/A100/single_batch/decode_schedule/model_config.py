"""Qwen1.5-MoE-A2.7B A100 single-batch model and schedule constants."""

from __future__ import annotations

import torch


class QwenMoeConfig:
    num_hidden_layers = 24
    hidden_size = 2048
    head_dim = 128
    num_attention_heads = 16
    num_key_value_heads = 16
    num_heads = num_attention_heads + num_key_value_heads * 2
    num_attn_barrier_slots = num_heads
    qkv_out_dim = num_heads * head_dim
    vocab_size = 151936
    rms_norm_eps = 1e-6

    intermediate_size = 5632
    moe_intermediate_size = 1408
    num_experts = 60
    num_experts_per_tok = 4
    norm_topk_prob = False

    attn_kv_block_size = 16
    # One A100 attention-reduction CTA owns four Q heads.
    attn_reduction_size = 4

    tile_m_granularity = 1
    tile_n_granularity = 16
    num_downproj_split_k_shared = 4
    max_opcode = 14
    max_num_partials = 27
    a100_sm_count = 108

    num_hidden_blocks = hidden_size // tile_n_granularity
    num_qkv_blocks = qkv_out_dim // tile_n_granularity
    num_qkv_blocks_per_head = head_dim // tile_n_granularity
    num_shared_upgate_blocks = intermediate_size // tile_n_granularity
    num_moe_intermediate_blocks = moe_intermediate_size // tile_n_granularity
    num_routed_upgate_blocks = num_experts_per_tok * num_moe_intermediate_blocks
    num_routed_downproj_blocks = num_experts_per_tok * num_hidden_blocks
    num_lm_head_blocks = -(-vocab_size // tile_n_granularity)
    split_k_granularity_shared = intermediate_size // num_downproj_split_k_shared
    num_shared_downproj_accum_blocks = (
        num_downproj_split_k_shared * num_hidden_blocks
    )
    router_gemv_workers = num_experts
    barrier_slots = num_hidden_blocks
    num_opcode_rows = max_opcode + 1

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


# barriers[layer, opcode - 1, slot], shape [L, 15, 128]
BARRIER_ROW_QKV_ROPE = 0
BARRIER_ROW_ATTN_PARTIAL = 1
BARRIER_ROW_ATTN_REDUCTION = 2
BARRIER_ROW_O_PROJ = 3
BARRIER_ROW_LM_HEAD = 6
BARRIER_ROW_ROUTER_GEMV = 7
BARRIER_ROW_ROUTER_TOPK = 8
BARRIER_ROW_SHARED_UPGATE = 9
BARRIER_ROW_SHARED_DOWNPROJ = 10
BARRIER_ROW_ROUTED_UPGATE = 11
BARRIER_ROW_ROUTED_DOWNPROJ = 12
BARRIER_ROW_COMBINE_COMMIT = 13

PREV_LAYER_BARRIER_ROW = BARRIER_ROW_COMBINE_COMMIT
PREV_LAYER_EXPECTED = QwenMoeConfig.num_hidden_blocks

BARRIER_SLOT_Q_BASE = 0
BARRIER_SLOT_K_BASE = QwenMoeConfig.num_attention_heads
BARRIER_SLOT_V_BASE = (
    QwenMoeConfig.num_attention_heads + QwenMoeConfig.num_key_value_heads
)
BARRIER_SLOT_ROUTER_LOGITS = 0
BARRIER_SLOT_NORMED_HIDDEN = 1


def barrier_row(opcode: int) -> int:
    if not 1 <= opcode <= QwenMoeConfig.max_opcode:
        raise ValueError(f"opcode {opcode} outside 1..{QwenMoeConfig.max_opcode}")
    return opcode - 1


def get_sm_count_for_device(device: torch.device | None = None) -> int:
    """Return the 108-SM A100/A800 schedule width."""
    if device is None or getattr(device, "type", None) != "cuda":
        return QwenMoeConfig.a100_sm_count
    try:
        sm_count = int(torch.cuda.get_device_properties(device).multi_processor_count)
    except Exception:
        return QwenMoeConfig.a100_sm_count
    if sm_count < QwenMoeConfig.a100_sm_count:
        raise ValueError(
            f"This decode schedule targets A100/A800 "
            f"({QwenMoeConfig.a100_sm_count} SMs); device reports {sm_count}."
        )
    return min(sm_count, QwenMoeConfig.a100_sm_count)


def get_num_partials_for_device(device: torch.device | None = None) -> int:
    """Size Split-KV to approximately one A100 wave."""
    sm_count = get_sm_count_for_device(device)
    n_kv = QwenMoeConfig.num_key_value_heads
    return max(
        1,
        min(QwenMoeConfig.max_num_partials, (sm_count + n_kv - 1) // n_kv),
    )


def barrier_expected_table() -> dict[str, int]:
    c = QwenMoeConfig
    return {
        "op1@row13.slot0": c.num_hidden_blocks,
        "op2@row0.q_head": c.num_qkv_blocks_per_head,
        "op2@row0.k_head": c.num_qkv_blocks_per_head,
        "op2@row0.v_head": c.num_qkv_blocks_per_head,
        "op3@row1.q_head": get_num_partials_for_device(None),
        "op4@row2.slot0": c.num_attention_heads,
        "op8@row3.slot0": c.num_hidden_blocks,
        "op9@row7.slot0": c.num_experts,
        "op10@row7.slot1": 1,
        "op11@row9.slot0": c.num_shared_upgate_blocks,
        "op12@row8.slot0": 1,
        "op13@row11.slot0": c.num_routed_upgate_blocks,
        "op14@row10.tile": c.num_downproj_split_k_shared,
        "op14@row12.tile": c.num_experts_per_tok,
        "op7@row13.slot0": c.num_hidden_blocks,
    }
