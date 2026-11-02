"""A100 single-batch schedule; MoE emission order is 8→9→10→11→12→13→14."""
from dataclasses import fields

import torch

from qimeng_megakernel.models.instruction import Instruction
from ..globs import Globals
from ..model_config import (
    QwenMoeConfig,
    get_num_partials_for_device,
    get_sm_count_for_device,
)
from .instruction_defs import (
    RMS_QKVProj_GEMV_RoPE,
    DecodeAttentionSplitKV,
    DecodeAttentionReduction,
    OProj_GEMV_Residual,
    Router_GEMV,
    Router_SoftmaxTopK,
    SharedExpert_UpGate_SiLU,
    SharedExpert_DownProj,
    RoutedExpert_UpGate_SiLU,
    RoutedExpert_DownProj,
    MoE_Combine_Commit,
    RMS_LMHead_GEMV,
    TILE_N_GRANULARITY,
    SPLIT_K_GRANULARITY_SHARED,
    NUM_DOWNPROJ_SPLIT_K_SHARED,
)

INTS_PER_INSTRUCTION = 32
ATTN_REDUCTION_SIZE = QwenMoeConfig.attn_reduction_size
MAX_REDUCTION_LIST = min(INTS_PER_INSTRUCTION - 5, 27)


class NoOp(Instruction):
    @classmethod
    def opcode(cls) -> int:
        return 0

    @classmethod
    def prev_opcode(cls) -> tuple[int, ...]:
        return ()


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _get_device(globs: Globals) -> torch.device:
    for field in fields(globs):
        value = getattr(globs, field.name)
        if isinstance(value, torch.Tensor):
            return value.device
    return torch.device("cpu")


def _get_sm_count(device: torch.device) -> int:
    return get_sm_count_for_device(device)


def _serialize_and_pad(instruction: Instruction) -> list[int]:
    serialized = instruction.serialize()
    num_padding = INTS_PER_INSTRUCTION - len(serialized)
    assert num_padding >= 0, (
        f"{type(instruction).__name__} serializes to {len(serialized)} ints, "
        f"exceeding INTS_PER_INSTRUCTION={INTS_PER_INSTRUCTION}"
    )
    return serialized + [0] * num_padding


def tensorize_instructions(
    globs: Globals, instruction_queues: list[list[Instruction]]
) -> torch.Tensor:
    num_sms = len(instruction_queues)
    max_queue_len = max((len(q) for q in instruction_queues), default=0)
    for queue in instruction_queues:
        queue.extend([NoOp()] * (max_queue_len - len(queue)))
    flattened = []
    for queue in instruction_queues:
        flattened.extend(_serialize_and_pad(ins) for ins in queue)
    device = _get_device(globs)
    serialized = torch.tensor(flattened, dtype=torch.int32, device=device).view(
        num_sms, -1, INTS_PER_INSTRUCTION
    )
    globs.instructions = serialized
    return serialized


def _chunk_list(items: list[int], chunk_size: int) -> list[list[int]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def _split_n_tiles(num_n_tiles: int, sm_count: int) -> list[tuple[int, int]]:
    if num_n_tiles <= 0:
        return []
    groups = min(num_n_tiles, sm_count)
    base = num_n_tiles // groups
    remainder = num_n_tiles % groups
    ranges: list[tuple[int, int]] = []
    start = 0
    for i in range(groups):
        size = base + (1 if i < remainder else 0)
        end = start + size
        if size > 0:
            ranges.append((start, end))
        start = end
    return ranges


def _round_robin_assign_to_sms(
    instructions: list[Instruction], sm_count: int
) -> list[list[Instruction]]:
    sm_queues = [[] for _ in range(sm_count)]
    for i, instruction in enumerate(instructions):
        sm_queues[i % sm_count].append(instruction)
    return sm_queues


def _sum_tile_blocks(instructions: list[Instruction]) -> int:
    total = 0
    for ins in instructions:
        if hasattr(ins, "tile_N_start_idx") and hasattr(ins, "tile_N_end_idx"):
            total += getattr(ins, "tile_N_end_idx") - getattr(ins, "tile_N_start_idx")
    return total


# --------- Attention (1-4) ---------


def schedule_qkvproj_gemv(layer_idx: int, sm_count: int) -> list[Instruction]:
    num_tiles = _ceil_div(QwenMoeConfig.qkv_out_dim, TILE_N_GRANULARITY)
    instructions: list[Instruction] = []
    for n_start, n_end in _split_n_tiles(num_tiles, sm_count):
        instructions.append(
            RMS_QKVProj_GEMV_RoPE(
                layer_idx=layer_idx,
                tile_M_idx=0,
                tile_N_start_idx=n_start,
                tile_N_end_idx=n_end,
            )
        )
    expected = _ceil_div(QwenMoeConfig.qkv_out_dim, TILE_N_GRANULARITY)
    assert _sum_tile_blocks(instructions) == expected
    return instructions


def schedule_attention_split_kv(
    layer_idx: int, num_partials: int
) -> list[Instruction]:
    instructions: list[Instruction] = []
    for kv_head_idx in range(QwenMoeConfig.num_key_value_heads):
        for partial_idx in range(num_partials):
            instructions.append(
                DecodeAttentionSplitKV(
                    layer_idx=layer_idx,
                    kv_head_idx=kv_head_idx,
                    num_partials=num_partials,
                    partial_idx=partial_idx,
                )
            )
    return instructions


def schedule_attention_reduction(
    layer_idx: int, num_partials: int
) -> list[Instruction]:
    instructions: list[Instruction] = []
    for head_start_idx in range(
        0, QwenMoeConfig.num_attention_heads, ATTN_REDUCTION_SIZE
    ):
        instructions.append(
            DecodeAttentionReduction(
                layer_idx=layer_idx,
                head_start_idx=head_start_idx,
                num_partials=num_partials,
                reduction_list=list(range(num_partials)),
            )
        )
    return instructions


def schedule_oproj_gemv(layer_idx: int, sm_count: int) -> list[Instruction]:
    instructions: list[Instruction] = []
    num_n_tiles = _ceil_div(QwenMoeConfig.hidden_size, TILE_N_GRANULARITY)
    for n_start, n_end in _split_n_tiles(num_n_tiles, sm_count):
        instructions.append(
            OProj_GEMV_Residual(
                layer_idx=layer_idx,
                tile_M_idx=0,
                tile_N_start_idx=n_start,
                tile_N_end_idx=n_end,
                reduction_block_idx=0,
            )
        )
    assert _sum_tile_blocks(instructions) == QwenMoeConfig.num_hidden_blocks
    return instructions


# --------- MoE (8→9→10→11→12→13) ---------


def schedule_router_gemv(layer_idx: int, sm_count: int) -> list[Instruction]:
    """Use one router instruction per physical expert to minimize top-k latency."""
    num_experts = QwenMoeConfig.num_experts
    instructions: list[Instruction] = []
    worker_count = max(1, min(QwenMoeConfig.router_gemv_workers, num_experts))
    for e_start, e_end in _split_n_tiles(num_experts, worker_count):
        instructions.append(
            Router_GEMV(
                layer_idx=layer_idx,
                tile_N_start_idx=e_start,
                tile_N_end_idx=e_end,
                write_normed_hidden=1 if e_start == 0 else 0,
            )
        )
    assert _sum_tile_blocks(instructions) == num_experts
    assert sum(ins.write_normed_hidden for ins in instructions) == 1
    return instructions


def schedule_router_softmax_topk(layer_idx: int) -> list[Instruction]:
    return [Router_SoftmaxTopK(layer_idx=layer_idx)]


def schedule_shared_upgate(layer_idx: int, sm_count: int) -> list[Instruction]:
    instructions: list[Instruction] = []
    num_n_tiles = _ceil_div(QwenMoeConfig.intermediate_size, TILE_N_GRANULARITY)
    for n_start, n_end in _split_n_tiles(num_n_tiles, sm_count):
        instructions.append(
            SharedExpert_UpGate_SiLU(
                layer_idx=layer_idx,
                tile_M_idx=0,
                tile_N_start_idx=n_start,
                tile_N_end_idx=n_end,
                init_shared_out=1 if n_start == 0 else 0,
            )
        )
    expected = _ceil_div(QwenMoeConfig.intermediate_size, TILE_N_GRANULARITY)
    assert _sum_tile_blocks(instructions) == expected
    assert sum(ins.init_shared_out for ins in instructions) == 1
    return instructions


def schedule_shared_downproj(layer_idx: int, sm_count: int) -> list[Instruction]:
    """4 groups × 27 SMs = 108 instructions (split-K over 5632, 128 output tiles per group)."""
    instructions: list[Instruction] = []
    num_split_k = NUM_DOWNPROJ_SPLIT_K_SHARED
    num_n_tiles = _ceil_div(QwenMoeConfig.hidden_size, TILE_N_GRANULARITY)
    sms_per_group = max(1, sm_count // num_split_k)
    for split_k_idx in range(num_split_k):
        group_start_sm = split_k_idx * sms_per_group
        group_end_sm = (
            min((split_k_idx + 1) * sms_per_group, sm_count)
            if split_k_idx < num_split_k - 1
            else sm_count
        )
        group_sm_count = max(1, group_end_sm - group_start_sm)
        for seg_start, seg_end in _split_n_tiles(num_n_tiles, group_sm_count):
            instructions.append(
                SharedExpert_DownProj(
                    layer_idx=layer_idx,
                    tile_M_idx=0,
                    tile_N_start_idx=seg_start,
                    tile_N_end_idx=seg_end,
                    split_K_idx=split_k_idx,
                )
            )
    # split-K accumulates hidden blocks across 4 groups
    assert (
        _sum_tile_blocks(instructions)
        == QwenMoeConfig.num_shared_downproj_accum_blocks
    )
    return instructions


def schedule_routed_upgate(layer_idx: int, sm_count: int) -> list[Instruction]:
    instructions: list[Instruction] = []
    num_experts = QwenMoeConfig.num_experts_per_tok
    num_tiles_per_expert = _ceil_div(
        QwenMoeConfig.moe_intermediate_size, TILE_N_GRANULARITY
    )
    sms_per_expert = max(1, sm_count // num_experts)
    for expert_idx in range(num_experts):
        for n_start, n_end in _split_n_tiles(num_tiles_per_expert, sms_per_expert):
            instructions.append(
                RoutedExpert_UpGate_SiLU(
                    layer_idx=layer_idx,
                    expert_idx=expert_idx,
                    tile_N_start_idx=n_start,
                    tile_N_end_idx=n_end,
                )
            )
    expected = QwenMoeConfig.num_routed_upgate_blocks
    assert _sum_tile_blocks(instructions) == expected
    return instructions


def schedule_routed_downproj(layer_idx: int, sm_count: int) -> list[Instruction]:
    """Schedule pure routed-expert DownProj GEMVs; opcode 14 owns combine."""
    instructions: list[Instruction] = []
    num_experts = QwenMoeConfig.num_experts_per_tok
    num_n_tiles = _ceil_div(QwenMoeConfig.hidden_size, TILE_N_GRANULARITY)
    sms_per_expert = max(1, sm_count // num_experts)
    for expert_idx in range(num_experts):
        for seg_start, seg_end in _split_n_tiles(num_n_tiles, sms_per_expert):
            instructions.append(
                RoutedExpert_DownProj(
                    layer_idx=layer_idx,
                    expert_idx=expert_idx,
                    tile_M_idx=0,
                    tile_N_start_idx=seg_start,
                    tile_N_end_idx=seg_end,
                )
            )
    assert _sum_tile_blocks(instructions) == QwenMoeConfig.num_routed_downproj_blocks
    return instructions


def schedule_moe_combine_commit(
    layer_idx: int, sm_count: int
) -> list[Instruction]:
    """Assign every hidden tile to exactly one physical opcode-14 instruction."""
    instructions: list[Instruction] = []
    for n_start, n_end in _split_n_tiles(QwenMoeConfig.num_hidden_blocks, sm_count):
        instructions.append(
            MoE_Combine_Commit(
                layer_idx=layer_idx,
                tile_N_start_idx=n_start,
                tile_N_end_idx=n_end,
            )
        )
    assert _sum_tile_blocks(instructions) == QwenMoeConfig.num_hidden_blocks
    return instructions


def schedule_lmhead_gemv(sm_count: int) -> list[Instruction]:
    instructions: list[Instruction] = []
    num_logit_tiles = _ceil_div(QwenMoeConfig.vocab_size, TILE_N_GRANULARITY)
    for n_start, n_end in _split_n_tiles(num_logit_tiles, sm_count):
        instructions.append(
            RMS_LMHead_GEMV(
                start_output_block_idx=n_start,
                end_output_block_idx=n_end,
            )
        )
    return instructions


def build_instruction_list(
    globs: Globals, num_layers: int | None = None
) -> list[Instruction]:
    device = _get_device(globs)
    sm_count = _get_sm_count(device)
    layers = (
        num_layers
        if num_layers is not None
        else QwenMoeConfig.num_hidden_layers
    )
    num_partials = get_num_partials_for_device(device)
    instructions: list[Instruction] = []
    for layer_idx in range(layers):
        instructions.extend(schedule_qkvproj_gemv(layer_idx, sm_count))
        instructions.extend(
            schedule_attention_split_kv(layer_idx, num_partials)
        )
        instructions.extend(
            schedule_attention_reduction(layer_idx, num_partials)
        )
        instructions.extend(schedule_oproj_gemv(layer_idx, sm_count))
        instructions.extend(schedule_router_gemv(layer_idx, sm_count))
        instructions.extend(schedule_router_softmax_topk(layer_idx))
        instructions.extend(schedule_shared_upgate(layer_idx, sm_count))
        instructions.extend(schedule_shared_downproj(layer_idx, sm_count))
        instructions.extend(schedule_routed_upgate(layer_idx, sm_count))
        instructions.extend(schedule_routed_downproj(layer_idx, sm_count))
        instructions.extend(schedule_moe_combine_commit(layer_idx, sm_count))
    if layers == QwenMoeConfig.num_hidden_layers:
        instructions.extend(schedule_lmhead_gemv(sm_count))
    return instructions


def build_instruction_queues(
    globs: Globals, num_layers: int | None = None
) -> list[list[Instruction]]:
    device = _get_device(globs)
    sm_count = _get_sm_count(device)
    instructions = build_instruction_list(globs, num_layers=num_layers)
    return _round_robin_assign_to_sms(instructions, sm_count)


def build(globs: Globals, num_layers: int | None = None) -> torch.Tensor:
    queues = build_instruction_queues(globs, num_layers=num_layers)
    return tensorize_instructions(globs, queues)
