import math
from dataclasses import dataclass, fields

import torch

from qimeng_megakernel.models.instruction import Instruction
from ..globs import Globals
from ..model_config import LlamaConfig
from .instruction_defs import (
    RMS_QKVProj_GEMV_RoPE,
    DecodeAttentionSplitKV,
    DecodeAttentionReduction,
    OProj_GEMV_Residual,
    RMS_UpGateProj_GEMV_SiLU,
    DownProj_GEMV_Residual,
    RMS_LMHead_GEMV,
    TILE_N_GRANULARITY,
    SPLIT_K_GRANULARITY,
    UPGATE_BLOCKS_PER_SIGNAL_BUCKET,
)

INTS_PER_INSTRUCTION = 32
A100_SM_COUNT = 108
ATTN_REDUCTION_SIZE = LlamaConfig.attn_reduction_size
MAX_TILE_BLOCKS_PER_INSTRUCTION = INTS_PER_INSTRUCTION - 6
ATTENTION_REDUCTION_WORKSPACE_CAPACITY = 16
# Static instruction layout for reduction:
# [opcode, layer, head, count, list_len, list..., terminal, output_partial]
ATTENTION_REDUCTION_LIST_CAPACITY = INTS_PER_INSTRUCTION - 7
MAX_REDUCTION_LIST = ATTENTION_REDUCTION_LIST_CAPACITY
MAX_ATTENTION_PARTIALS = min(
    ATTENTION_REDUCTION_WORKSPACE_CAPACITY,
    MAX_REDUCTION_LIST,
)

assert ATTN_REDUCTION_SIZE == (
    LlamaConfig.num_attention_heads // LlamaConfig.num_key_value_heads
)


@dataclass
class NoOp(Instruction):
    @classmethod
    def opcode(cls) -> int:
        return 0

    @classmethod
    def prev_opcode(cls) -> tuple[int, ...]:
        return ()


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _assert_div(a: int, b: int) -> int:
    assert a % b == 0
    return a // b


def _get_device(globs: Globals) -> torch.device:
    for field in fields(globs):
        value = getattr(globs, field.name)
        if isinstance(value, torch.Tensor):
            return value.device
    return torch.device("cpu")


def _get_sm_count(device: torch.device) -> int:
    """A100-specific: returns SM count, capped at 108; errors if < 108."""
    if device.type != "cuda":
        return 1
    try:
        sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    except Exception:
        return 1
    if sm_count < A100_SM_COUNT:
        raise ValueError(
            f"This schedule targets A100 ({A100_SM_COUNT} SMs). "
            f"Device has {sm_count} SMs, which is fewer than required."
        )
    return min(sm_count, A100_SM_COUNT)


def pick_num_partials(sm_count: int) -> int:
    """Choose a wave-safe SplitKV fanout for the A100 route.

    One attention wave contains ``num_key_value_heads * num_partials``
    instructions.  Use floor division so the default never spills a handful
    of partials into a second wave.  KV-length-specific breakpoints require an
    A100/A800 sweep and intentionally do not inherit H100 thresholds.
    """
    wave_cap = max(1, sm_count // LlamaConfig.num_key_value_heads)
    return min(MAX_ATTENTION_PARTIALS, wave_cap)


def _validate_num_partials(num_partials: int) -> None:
    if not 1 <= num_partials <= MAX_ATTENTION_PARTIALS:
        raise ValueError(
            f"num_partials must be in [1, {MAX_ATTENTION_PARTIALS}], "
            f"got {num_partials}"
        )


def _serialize_and_pad(instruction: Instruction) -> list[int]:
    serialized = instruction.serialize()
    num_padding = INTS_PER_INSTRUCTION - len(serialized)
    assert num_padding >= 0
    return serialized + [0] * num_padding


def tensorize_instructions(
    globs: Globals, instruction_queues: list[list[Instruction]]
) -> torch.Tensor:
    num_sms = len(instruction_queues)
    max_queue_len = max((len(queue) for queue in instruction_queues), default=0)
    for queue in instruction_queues:
        queue.extend([NoOp()] * (max_queue_len - len(queue)))

    flattened = []
    for queue in instruction_queues:
        flattened.extend(_serialize_and_pad(instruction) for instruction in queue)

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


def schedule_qkvproj_gemv(layer_idx: int, sm_count: int) -> list[Instruction]:
    # GEMV: one instruction per SM, each covers a range of tile_N (output blocks); persistent
    # kernel style (load activation once per SM). M_idx=0, tile_M_granularity=1; granularity fixed.
    instructions: list[Instruction] = []
    num_output_blocks = _ceil_div(LlamaConfig.qkv_out_dim, TILE_N_GRANULARITY)
    for n_start, n_end in _split_n_tiles(num_output_blocks, sm_count):
        instructions.append(
            RMS_QKVProj_GEMV_RoPE(
                layer_idx=layer_idx,
                tile_M_idx=0,
                tile_N_start_idx=n_start,
                tile_N_end_idx=n_end,
            )
        )
    return instructions


def schedule_attention_split_kv(
    layer_idx: int, num_partials: int
) -> list[Instruction]:
    """Emit SplitKV instructions: one per (kv_head, partial). Each handles gqa_ratio Q heads over
    a slice of the KV sequence; num_partials is chosen so num_kv_heads * num_partials ~ sm_count."""
    _validate_num_partials(num_partials)
    instructions: list[Instruction] = []
    for kv_head_idx in range(LlamaConfig.num_key_value_heads):
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
    layer_idx: int,
    num_partials: int,
) -> list[Instruction]:
    _validate_num_partials(num_partials)
    instructions: list[Instruction] = []
    for head_start_idx in range(0, LlamaConfig.num_attention_heads, ATTN_REDUCTION_SIZE):
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
    num_n_tiles = _ceil_div(LlamaConfig.hidden_size, TILE_N_GRANULARITY)
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
    return instructions


def schedule_upgate(layer_idx: int, sm_count: int) -> list[Instruction]:
    """Assign strided UpGate blocks so split-K buckets become ready in order.

    DownProj split ``b`` consumes exactly the UpGate blocks in bucket ``b``.
    Striding each instruction across the output dimension lets early buckets
    fill while other UpGate work is still in flight, instead of making every
    instruction drain one contiguous bucket before the next can progress.
    """
    instructions: list[Instruction] = []
    num_n_tiles = _ceil_div(LlamaConfig.intermediate_size, TILE_N_GRANULARITY)
    assert num_n_tiles % UPGATE_BLOCKS_PER_SIGNAL_BUCKET == 0
    groups = min(num_n_tiles, sm_count)
    for group_idx in range(groups):
        tile_block_idxs = list(range(group_idx, num_n_tiles, groups))
        for chunk in _chunk_list(tile_block_idxs, MAX_TILE_BLOCKS_PER_INSTRUCTION):
            instructions.append(
                RMS_UpGateProj_GEMV_SiLU(
                    layer_idx=layer_idx,
                    tile_M_idx=0,
                    tile_N_block_idxs=chunk,
                )
            )
    return instructions


def schedule_downproj(layer_idx: int, sm_count: int) -> list[Instruction]:
    """Partition the input K dimension and distribute each split over output tiles."""
    instructions: list[Instruction] = []
    num_split_k = _ceil_div(LlamaConfig.intermediate_size, SPLIT_K_GRANULARITY)
    num_n_tiles = _ceil_div(LlamaConfig.hidden_size, TILE_N_GRANULARITY)

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
                DownProj_GEMV_Residual(
                    layer_idx=layer_idx,
                    tile_M_idx=0,
                    tile_N_start_idx=seg_start,
                    tile_N_end_idx=seg_end,
                    split_K_idx=split_k_idx,
                )
            )
    return instructions


def schedule_lmhead_gemv(sm_count: int) -> list[Instruction]:
    instructions: list[Instruction] = []
    num_logit_tiles = _ceil_div(LlamaConfig.vocab_size, TILE_N_GRANULARITY)
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
    """
    Build a flat list of instructions in dependency order (no round-robin).
    Used by tests to run_pyvm_sequence(globs, instructions, handler_map).
    """
    device = _get_device(globs)
    sm_count = _get_sm_count(device)
    layers = num_layers if num_layers is not None else LlamaConfig.num_hidden_layers
    num_partials = pick_num_partials(sm_count)
    instructions: list[Instruction] = []
    for layer_idx in range(layers):
        instructions.extend(schedule_qkvproj_gemv(layer_idx, sm_count))
        instructions.extend(schedule_attention_split_kv(layer_idx, num_partials))
        instructions.extend(schedule_attention_reduction(layer_idx, num_partials))
        instructions.extend(schedule_oproj_gemv(layer_idx, sm_count))
        instructions.extend(schedule_upgate(layer_idx, sm_count))
        instructions.extend(schedule_downproj(layer_idx, sm_count))
    if layers == LlamaConfig.num_hidden_layers:
        instructions.extend(schedule_lmhead_gemv(sm_count))
    return instructions


def build_instruction_queues(globs: Globals, num_layers: int | None = None):
    device = _get_device(globs)
    sm_count = _get_sm_count(device)
    layers = num_layers if num_layers is not None else LlamaConfig.num_hidden_layers
    num_partials = pick_num_partials(sm_count)
    instructions: list[Instruction] = []
    for layer_idx in range(layers):
        instructions.extend(schedule_qkvproj_gemv(layer_idx, sm_count))
        instructions.extend(schedule_attention_split_kv(layer_idx, num_partials))
        instructions.extend(schedule_attention_reduction(layer_idx, num_partials))
        instructions.extend(schedule_oproj_gemv(layer_idx, sm_count))
        instructions.extend(schedule_upgate(layer_idx, sm_count))
        instructions.extend(schedule_downproj(layer_idx, sm_count))
    if layers == LlamaConfig.num_hidden_layers:
        instructions.extend(schedule_lmhead_gemv(sm_count))
    return _round_robin_assign_to_sms(instructions, sm_count)


def build(globs: Globals, num_layers: int | None = None) -> torch.Tensor:
    queues = build_instruction_queues(globs, num_layers=num_layers)
    return tensorize_instructions(globs, queues)
