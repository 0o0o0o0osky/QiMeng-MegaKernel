"""Scheduler for Qwen3 0.6B H100 decode."""
from dataclasses import dataclass, fields

import torch

from qimeng_megakernel.models.instruction import Instruction
from ..globs import Globals
from ..model_config import Qwen3Config
from .instruction_defs import (
    RMS_QKVProj_GEMV,
    DecodeAttentionSplitKV,
    DecodeAttentionReduction,
    OProj_GEMV_Residual,
    RMS_UpGateProj_GEMV_SiLU,
    DownProj_GEMV_Residual,
    RMS_LMHead_GEMV,
    RMS_QKV_BLOCK_SIZE,
    TILE_N_GRANULARITY,
    NUM_DOWNPROJ_SPLIT_K,
)

INTS_PER_INSTRUCTION = 32
ATTN_REDUCTION_SIZE = Qwen3Config.attn_reduction_size
MAX_TILE_BLOCKS_PER_INSTRUCTION = INTS_PER_INSTRUCTION - 6
MAX_REDUCTION_LIST = INTS_PER_INSTRUCTION - 5


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


def _get_device(globs: Globals) -> torch.device:
    for field in fields(globs):
        value = getattr(globs, field.name)
        if isinstance(value, torch.Tensor):
            return value.device
    return torch.device("cpu")


H100_SM_COUNT = 132

_PARTIALS_BY_KV: tuple[tuple[int, int], ...] = (
    (224, 1),
    (768, 6),
    (2048, 12),
)
_PARTIALS_LONG_KV = 12
RUNTIME_P1_KV_MAX = 128
RUNTIME_MID_KV_MAX = 512
ADAPTIVE_MIN_KV_CAPACITY = 224


def _get_sm_count(device: torch.device) -> int:
    if device.type != "cuda":
        return 1
    try:
        sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    except Exception:
        return 1
    if sm_count < H100_SM_COUNT:
        raise ValueError(
            f"This schedule targets H100 ({H100_SM_COUNT} SMs). "
            f"Device has {sm_count} SMs, which is fewer than required."
        )
    return min(sm_count, H100_SM_COUNT)


def pick_num_partials(sm_count: int, kv_len: int | None = None) -> int:
    """Choose per-GQA-group SplitKV fanout for the maximum KV length."""
    wave_cap = max(1, sm_count // Qwen3Config.num_key_value_heads)
    cap = min(MAX_REDUCTION_LIST, wave_cap)
    if kv_len is None or kv_len <= 0:
        return cap
    num_partials = _PARTIALS_LONG_KV
    for kv_hi, candidate in _PARTIALS_BY_KV:
        if kv_len <= kv_hi:
            num_partials = candidate
            break
    total_kv_blocks = _ceil_div(kv_len, Qwen3Config.attn_kv_block_size)
    return max(1, min(cap, num_partials, total_kv_blocks))


def _max_kv_len(globs: Globals) -> int | None:
    """Return the largest KV length used by this prebuilt decode stream."""
    kv_len = 0
    pos = getattr(globs, "position_id", None)
    if pos is not None:
        kv_len = int(pos) + 1
    k_cache = getattr(globs, "k_cache", None)
    if isinstance(k_cache, torch.Tensor):
        if k_cache.dim() == 5:
            kv_len = max(kv_len, int(k_cache.shape[2]))
        elif k_cache.dim() == 4:
            kv_len = max(kv_len, int(k_cache.shape[1]))
    return kv_len if kv_len > 0 else None


def _serialize_and_pad(instruction: Instruction) -> list[int]:
    serialized = instruction.serialize()
    num_padding = INTS_PER_INSTRUCTION - len(serialized)
    assert num_padding >= 0
    return serialized + [0] * num_padding


def tensorize_instructions(
    globs: Globals, instruction_queues: list[list[Instruction]]
) -> torch.Tensor:
    device = _get_device(globs)
    serialized = _tensorize_queues(instruction_queues, device)
    globs.instructions = serialized

    adaptive_spec = getattr(globs, "_qwen3_adaptive_instruction_spec", None)
    if adaptive_spec is None:
        # Explicit verifier/op streams must never inherit a full-model adaptive
        # schedule previously attached to the same Globals object.
        if hasattr(globs, "_qwen3_instruction_tensors_by_kv_bucket"):
            delattr(globs, "_qwen3_instruction_tensors_by_kv_bucket")
        return serialized

    layers, base_num_partials = adaptive_spec
    sm_count = _get_sm_count(device)
    partial_options = {
        pick_num_partials(sm_count, kv_hi)
        for kv_hi, _ in _PARTIALS_BY_KV
    }
    partial_options.add(pick_num_partials(sm_count, _PARTIALS_BY_KV[-1][0] + 1))
    variants = {base_num_partials: serialized}
    for num_partials in sorted(partial_options):
        if num_partials == base_num_partials:
            continue
        queues = _build_instruction_queues_with_num_partials(
            sm_count, layers, num_partials
        )
        variants[num_partials] = _tensorize_queues(queues, device)
    globs._qwen3_instruction_tensors_by_kv_bucket = tuple(
        variants[pick_num_partials(sm_count, kv_hi)]
        for kv_hi, _ in _PARTIALS_BY_KV
    )
    delattr(globs, "_qwen3_adaptive_instruction_spec")
    return serialized


def _tensorize_queues(
    instruction_queues: list[list[Instruction]], device: torch.device
) -> torch.Tensor:
    num_sms = len(instruction_queues)
    max_queue_len = max((len(queue) for queue in instruction_queues), default=0)
    noop = _serialize_and_pad(NoOp())
    flattened: list[list[int]] = []
    for queue in instruction_queues:
        flattened.extend(_serialize_and_pad(instruction) for instruction in queue)
        flattened.extend(noop for _ in range(max_queue_len - len(queue)))

    return torch.tensor(flattened, dtype=torch.int32, device=device).view(
        num_sms, -1, INTS_PER_INSTRUCTION
    )


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
    """Partition the full QKV output in native 16-element output blocks."""
    instructions: list[Instruction] = []
    num_output_blocks = _ceil_div(
        Qwen3Config.qkv_out_dim,
        RMS_QKV_BLOCK_SIZE,
    )
    for n_start, n_end in _split_n_tiles(num_output_blocks, sm_count):
        instructions.append(
            RMS_QKVProj_GEMV(
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
    """One instruction handles both Q heads belonging to a KV head."""
    instructions: list[Instruction] = []
    gqa_ratio = (
        Qwen3Config.num_attention_heads // Qwen3Config.num_key_value_heads
    )
    for kv_head_idx in range(Qwen3Config.num_key_value_heads):
        for partial_idx in range(num_partials):
            instructions.append(
                DecodeAttentionSplitKV(
                    layer_idx=layer_idx,
                    attention_head_idx=kv_head_idx * gqa_ratio,
                    num_partials=num_partials,
                    partial_idx=partial_idx,
                )
            )
    return instructions


def schedule_attention_reduction(
    layer_idx: int,
    num_partials: int,
) -> list[Instruction]:
    """Last-arrival SplitKV partials perform reduction in place."""
    return []


def schedule_oproj_gemv(layer_idx: int, sm_count: int) -> list[Instruction]:
    instructions: list[Instruction] = []
    num_n_tiles = _ceil_div(Qwen3Config.hidden_size, TILE_N_GRANULARITY)
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
    """Assign one strided block list per SM for early bucket completion.

    The first block on every SM covers one contiguous 132-block wave.  The
    remaining 60 blocks form the second wave, so DownProj buckets 0 and 1 can
    become ready while bucket 2 is still being produced.
    """
    instructions: list[Instruction] = []
    num_n_tiles = _ceil_div(Qwen3Config.intermediate_size, TILE_N_GRANULARITY)
    groups = min(num_n_tiles, sm_count)
    for sm in range(groups):
        tile_block_idxs = list(range(sm, num_n_tiles, groups))
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
    """Schedule three native split-K segments with exactly-once N-block coverage.

    H100 has 132 SMs, so each split receives 44 work groups. Every output block is
    produced once per split and therefore contributes three barrier arrivals.
    """
    instructions: list[Instruction] = []
    num_split_k = NUM_DOWNPROJ_SPLIT_K
    num_n_tiles = _ceil_div(Qwen3Config.hidden_size, TILE_N_GRANULARITY)
    sms_per_group = max(1, sm_count // num_split_k)

    for split_k_idx in range(num_split_k):
        group_start_sm = split_k_idx * sms_per_group
        if split_k_idx < num_split_k - 1:
            group_end_sm = min((split_k_idx + 1) * sms_per_group, sm_count)
        else:
            group_end_sm = sm_count
        group_sm_count = max(1, group_end_sm - group_start_sm)

        for n_start, n_end in _split_n_tiles(num_n_tiles, group_sm_count):
            instructions.append(
                DownProj_GEMV_Residual(
                    layer_idx=layer_idx,
                    tile_M_idx=0,
                    tile_N_start_idx=n_start,
                    tile_N_end_idx=n_end,
                    split_K_idx=split_k_idx,
                )
            )
    return instructions


def schedule_lmhead_gemv(sm_count: int) -> list[Instruction]:
    instructions: list[Instruction] = []
    num_logit_tiles = _ceil_div(Qwen3Config.vocab_size, TILE_N_GRANULARITY)
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
    """Build a flat list of instructions in dependency order (no round-robin)."""
    device = _get_device(globs)
    sm_count = _get_sm_count(device)
    layers = num_layers if num_layers is not None else Qwen3Config.num_hidden_layers
    num_partials = pick_num_partials(sm_count, _max_kv_len(globs))
    instructions: list[Instruction] = []
    for layer_idx in range(layers):
        instructions.extend(schedule_qkvproj_gemv(layer_idx, sm_count))
        instructions.extend(schedule_attention_split_kv(layer_idx, num_partials))
        instructions.extend(schedule_attention_reduction(layer_idx, num_partials))
        instructions.extend(schedule_oproj_gemv(layer_idx, sm_count))
        instructions.extend(schedule_upgate(layer_idx, sm_count))
        instructions.extend(schedule_downproj(layer_idx, sm_count))
    if layers == Qwen3Config.num_hidden_layers:
        instructions.extend(schedule_lmhead_gemv(sm_count))
    return instructions


def build_instruction_queues(globs: Globals, num_layers: int | None = None):
    device = _get_device(globs)
    sm_count = _get_sm_count(device)
    layers = num_layers if num_layers is not None else Qwen3Config.num_hidden_layers
    max_kv_len = _max_kv_len(globs)
    num_partials = pick_num_partials(sm_count, max_kv_len)
    queues = _build_instruction_queues_with_num_partials(
        sm_count, layers, num_partials
    )
    if hasattr(globs, "_qwen3_adaptive_instruction_spec"):
        delattr(globs, "_qwen3_adaptive_instruction_spec")
    if (
        layers == Qwen3Config.num_hidden_layers
        and max_kv_len is not None
        and max_kv_len > ADAPTIVE_MIN_KV_CAPACITY
    ):
        # Consumed by tensorize_instructions during decode setup.  call_mk then
        # selects among these immutable tensors from the actual token position.
        globs._qwen3_adaptive_instruction_spec = (layers, num_partials)
    return queues


def _build_instruction_queues_with_num_partials(
    sm_count: int, layers: int, num_partials: int
) -> list[list[Instruction]]:
    instructions: list[Instruction] = []
    for layer_idx in range(layers):
        instructions.extend(schedule_qkvproj_gemv(layer_idx, sm_count))
        instructions.extend(schedule_attention_split_kv(layer_idx, num_partials))
        instructions.extend(schedule_attention_reduction(layer_idx, num_partials))
        instructions.extend(schedule_oproj_gemv(layer_idx, sm_count))
        instructions.extend(schedule_upgate(layer_idx, sm_count))
        instructions.extend(schedule_downproj(layer_idx, sm_count))
    if layers == Qwen3Config.num_hidden_layers:
        instructions.extend(schedule_lmhead_gemv(sm_count))
    return _round_robin_assign_to_sms(instructions, sm_count)


def build(globs: Globals, num_layers: int | None = None) -> torch.Tensor:
    queues = build_instruction_queues(globs, num_layers=num_layers)
    return tensorize_instructions(globs, queues)
