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
)

INTS_PER_INSTRUCTION = 32
ATTN_REDUCTION_SIZE = 4
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


def _assert_div(a: int, b: int) -> int:
    assert a % b == 0
    return a // b


def _get_device(globs: Globals) -> torch.device:
    for field in fields(globs):
        value = getattr(globs, field.name)
        if isinstance(value, torch.Tensor):
            return value.device
    return torch.device("cpu")


H100_SM_COUNT = 132


def _get_sm_count(device: torch.device) -> int:
    """H100-specific: returns SM count, capped at 132; errors if < 132."""
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


# Split-KV fan-out is bounded by one CTA wave and the serialized reduction list.
# The KV buckets select more parallelism as the reduction length grows. P=1 writes
# the final attention output directly and omits the identity reduction instruction.
_PARTIALS_BY_KV: tuple[tuple[int, int], ...] = (
    (224, 1),
    (768, 4),
    (2048, 8),
    (4096, 12),
)
_PARTIALS_LONG_KV = 16


def pick_num_partials(sm_count: int, kv_len: int | None = None) -> int:
    """Select KV-adaptive Split-K fan-out within wave and instruction-capacity limits.

    ``kv_len`` is the maximum KV length covered by the prebuilt instruction stream.
    """
    wave_cap = max(1, sm_count // LlamaConfig.num_key_value_heads)
    cap = min(MAX_REDUCTION_LIST, wave_cap)
    if kv_len is None or kv_len <= 0:
        return cap
    num_partials = _PARTIALS_LONG_KV
    for kv_hi, candidate in _PARTIALS_BY_KV:
        if kv_len <= kv_hi:
            num_partials = candidate
            break
    # Each partial owns at least one 16-token KV block.
    total_kv_blocks = _ceil_div(kv_len, LlamaConfig.attn_kv_block_size)
    return max(1, min(cap, num_partials, total_kv_blocks))


def _max_kv_len(globs: Globals) -> int | None:
    """Return the maximum KV length used by this decode request.

    The decode entry point pads the sequence dimension of ``k_cache`` to
    ``prompt_len + max_new_tokens``. Its length is therefore the final KV
    length for the request. ``position_id + 1`` provides a lower bound.
    """
    kv_len = 0
    pos = getattr(globs, "position_id", None)
    if pos is not None:
        kv_len = int(pos) + 1
    k_cache = getattr(globs, "k_cache", None)
    if isinstance(k_cache, torch.Tensor):
        if k_cache.dim() == 5:      # [layers, batch, seq, kv_heads, head_dim]
            kv_len = max(kv_len, int(k_cache.shape[2]))
        elif k_cache.dim() == 4:    # [layers, seq, kv_heads, head_dim] for MegaKernel
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
    num_output_blocks = _ceil_div(LlamaConfig.qkv_out_dim, TILE_N_GRANULARITY)  # 192 for 3072/16
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
    """Emit the AttentionReduction phase — **except when num_partials == 1**.

    With P == 1 the single SplitKV partial per kv head already covers the whole
    KV range, so the reduction is an identity merge: it would cost one extra
    barrier phase per layer plus an O/LSE global round trip for nothing.

    Barrier conservation when this returns []:
      * row 1 (SplitKV -> Reduction) gets no producer and no consumer;
      * SplitKV instead signals row 2 slot 0 with +GQA_RATIO each
        (decode_attention_splitkv.cu / python_vm.decode_attention_split_kv),
        so o_proj still sees exactly num_attention_heads (= 8 x 4) arrivals on
        the same row/slot it always waited on. o_proj is unchanged.
    The kernel takes its skip branch off the same `num_partials` instruction
    field, so scheduler and kernel cannot disagree.
    """
    if num_partials == 1:
        return []
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
    """Up/Gate GEMV + SiLU, one instruction per SM, **strided** block assignment.

    SM i owns blocks {i, i+sm_count, i+2*sm_count, ...} rather than a contiguous
    run.  Two reasons, both about the UPGATE -> DOWNPROJ edge:

    1. **Bucket fill order.**  The kernel signals bucket
       ``block_idx // UPGATE_BLOCKS_PER_SIGNAL_BUCKET``, and downproj split-K
       instruction ``b`` waits on bucket ``b`` alone.  With contiguous
       assignment every one of an SM's blocks lands in the same bucket, so a
       bucket only fills once a whole contiguous quarter of the SMs has
       finished, providing little improvement over a device-wide barrier.
       Strided, each SM's *first* block lands in bucket 0, its next in bucket 1,
       and so on, so bucket b fills at roughly (b+1)/num_buckets of the phase
       and downproj can start on bucket 0 while upgate is still working.
    2. **DRAM locality.**  At any instant the 132 SMs are then reading
       up/gate weight rows {t*sm_count .. t*sm_count + 131} -- a contiguous
       run -- instead of a stride-4 comb with holes.

    Ascending order within the list matters: the storer signals in list order,
    so ascending keeps buckets filling low-to-high, which is the order the
    downproj queue consumes them in.
    """
    instructions: list[Instruction] = []
    num_n_tiles = _ceil_div(LlamaConfig.intermediate_size, TILE_N_GRANULARITY)
    groups = min(num_n_tiles, sm_count)
    for sm in range(groups):
        tile_block_idxs = list(range(sm, num_n_tiles, groups))  # ascending
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
    """
    Split-K GEMV contributions update the BF16 residual
    in place and publish completion through opcode 6.
    
    tile_N indexes OUTPUT row blocks (HiddenDim / TILE_N_GRANULARITY = 128).
    split_K partitions the INPUT (IntermDim) into SPLIT_K_GRANULARITY chunks (4).
    
    SM mapping keeps activations resident:
    - Partition ``sm_count`` SMs into ``num_split_k`` groups.
    - Assign one split-K partition to each group.
    - Distribute the 128 output tiles evenly within each group.
    - Emit approximately ``sm_count`` fusion instructions in total.
    
    For 108 SMs and four split-K partitions, groups 0 through 3 use SM ranges
    0--26, 27--53, 54--80, and 81--107, respectively. Each group processes
    output tiles 0--31 for its split-K partition.
    """
    instructions: list[Instruction] = []
    num_split_k = _ceil_div(LlamaConfig.intermediate_size, SPLIT_K_GRANULARITY)
    num_n_tiles = _ceil_div(LlamaConfig.hidden_size, TILE_N_GRANULARITY)  # 128 output tiles
    
    # Assign one split-K partition to each SM group.
    sms_per_group = max(1, sm_count // num_split_k)  # Avoid division by zero.
    
    for split_k_idx in range(num_split_k):
        # SM range assigned to the current split-K partition.
        group_start_sm = split_k_idx * sms_per_group
        group_end_sm = min((split_k_idx + 1) * sms_per_group, sm_count) if split_k_idx < num_split_k - 1 else sm_count
        group_sm_count = max(1, group_end_sm - group_start_sm)
        
        # Distribute the output tiles across the current SM group.
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
    num_partials = pick_num_partials(sm_count, _max_kv_len(globs))
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
    num_partials = pick_num_partials(sm_count, _max_kv_len(globs))
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
