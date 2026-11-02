"""Multi-batch instruction scheduler (B<=16, single M-tile). DownProj mirrors OProj (no split-K)."""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch

from qimeng_megakernel.models.instruction import Instruction
from qimeng_megakernel.models.LLM.decode_state import decode_pos_id, row_active

from ..globs import Globals
from ..model_config import LlamaConfig
from .instruction_defs import (
    TILE_M_GRANULARITY,
    TILE_N_GRANULARITY,
    QKVProj_GEMM_RoPE,
    DecodeAttentionSplitKV,
    DecodeAttentionReduction,
    OProj_GEMM_Residual,
    UpGateProj_GEMM_SiLU,
    DownProj_GEMM_Residual,
    LMHeadProj_GEMM,
    AttnRMS,
    MlpRMS,
    LMHeadRMS,
)

INTS_PER_INSTRUCTION = 32
ATTN_REDUCTION_SIZE = 4
MAX_TILE_BLOCKS_PER_INSTRUCTION = INTS_PER_INSTRUCTION - 6
MAX_REDUCTION_LIST = INTS_PER_INSTRUCTION - 5

A100_SM_COUNT = 108
CTAS_PER_SM = 2


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


def _get_sm_count(device: torch.device) -> int:
    if device.type != "cuda":
        return 1
    try:
        sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    except Exception:
        return 1
    if sm_count < A100_SM_COUNT:
        raise ValueError(
            f"This multi_batch schedule targets A100 ({A100_SM_COUNT} SMs). "
            f"Device has {sm_count} SMs."
        )
    return min(sm_count, A100_SM_COUNT)


def _get_worker_count(device: torch.device) -> int:
    return _get_sm_count(device) * CTAS_PER_SM


def _serialize_and_pad(instruction: Instruction) -> list[int]:
    serialized = instruction.serialize()
    num_padding = INTS_PER_INSTRUCTION - len(serialized)
    assert num_padding >= 0
    return serialized + [0] * num_padding


def tensorize_instructions(
    globs: Globals,
    instruction_queues: list[list[Instruction]],
) -> torch.Tensor:
    num_workers = len(instruction_queues)
    expected_workers = _get_worker_count(_get_device(globs))
    if num_workers != expected_workers:
        raise ValueError(
            f"A100 multi-batch kernel needs {expected_workers} instruction queues, "
            f"got {num_workers}"
        )
    max_queue_len = max((len(queue) for queue in instruction_queues), default=0)
    for queue in instruction_queues:
        queue.extend([NoOp()] * (max_queue_len - len(queue)))

    flattened: list[int] = []
    for queue in instruction_queues:
        flattened.extend(_serialize_and_pad(instruction) for instruction in queue)

    device = _get_device(globs)
    serialized = torch.tensor(flattened, dtype=torch.int32, device=device).view(
        num_workers, -1, INTS_PER_INSTRUCTION
    )
    globs.instructions = serialized
    return serialized


def _split_n_tiles(num_n_tiles: int, worker_count: int) -> list[tuple[int, int]]:
    if num_n_tiles <= 0:
        return []
    groups = min(num_n_tiles, worker_count)
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


def _chunk_list(items: list[int], chunk_size: int) -> list[list[int]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def _round_robin_assign_to_workers(
    instructions: list[Instruction], worker_count: int
) -> list[list[Instruction]]:
    worker_queues = [[] for _ in range(worker_count)]
    for i, instruction in enumerate(instructions):
        worker_queues[i % worker_count].append(instruction)
    return worker_queues


def _append_worker_synchronized_wave(
    flat: list[Instruction], worker_count: int, wave: list[Instruction]
) -> None:
    if len(wave) != worker_count:
        raise ValueError(
            f"Worker wave must have length {worker_count} (got {len(wave)}); "
            "row RMS norms need one instruction slot per persistent worker per timestep."
        )
    flat.extend(wave)


def _append_instructions_as_worker_waves(
    flat: list[Instruction], worker_count: int, segment: list[Instruction]
) -> None:
    if not segment:
        return
    for i in range(0, len(segment), worker_count):
        chunk = list(segment[i : i + worker_count])
        if len(chunk) < worker_count:
            chunk.extend([NoOp()] * (worker_count - len(chunk)))
        flat.extend(chunk)


def _assert_batch(batch_size: int) -> None:
    if batch_size < 1 or batch_size > TILE_M_GRANULARITY:
        raise ValueError(
            f"multi_batch decode_schedule expects 1 <= batch_size <= {TILE_M_GRANULARITY}, got {batch_size}"
        )


def schedule_attn_rms(layer_idx: int, batch_size: int) -> list[Instruction]:
    _assert_batch(batch_size)
    return [
        AttnRMS(layer_idx=layer_idx, batch_size=batch_size, batch_row=row)
        for row in range(batch_size)
    ]


def schedule_qkvproj_gemm(layer_idx: int, batch_size: int, worker_count: int) -> list[Instruction]:
    _assert_batch(batch_size)
    instructions: list[Instruction] = []
    num_output_blocks = _ceil_div(LlamaConfig.qkv_out_dim, TILE_N_GRANULARITY)
    for n_start, n_end in _split_n_tiles(num_output_blocks, worker_count):
        instructions.append(
            QKVProj_GEMM_RoPE(
                layer_idx=layer_idx,
                batch_size=batch_size,
                tile_M_idx=0,
                tile_N_start_idx=n_start,
                tile_N_end_idx=n_end,
            )
        )
    return instructions


def schedule_attention_split_kv(
    layer_idx: int, batch_size: int, num_partials: int, globs: Globals
) -> list[Instruction]:
    _assert_batch(batch_size)
    instructions: list[Instruction] = []
    kv_bs = LlamaConfig.attn_kv_block_size
    for batch_idx in range(batch_size):
        row_on = row_active(globs, batch_idx)
        seq_len = (
            int(globs.kv_lengths[batch_idx].item())
            if globs.kv_lengths is not None
            else decode_pos_id(globs, batch_idx) + 1
        )
        total_blocks = max(1, _ceil_div(seq_len, kv_bs)) if row_on else 0
        blocks_per_partial = max(1, _ceil_div(total_blocks, num_partials)) if row_on else 1

        for kv_head_idx in range(LlamaConfig.num_key_value_heads):
            for partial_idx in range(num_partials):
                start_block = partial_idx * blocks_per_partial
                end_block = min(start_block + blocks_per_partial, total_blocks)
                start_token = start_block * kv_bs
                end_token = min(end_block * kv_bs, seq_len)
                has_work = row_on and start_token < end_token and end_token > 0
                instructions.append(
                    DecodeAttentionSplitKV(
                        layer_idx=layer_idx,
                        batch_idx=batch_idx,
                        kv_head_idx=kv_head_idx,
                        num_partials=num_partials,
                        partial_idx=partial_idx,
                        seq_len=seq_len,
                        is_active=1 if has_work else 0,
                    )
                )
    return instructions


def schedule_attention_split_kv_runtime(
    layer_idx: int, batch_size: int, num_partials: int
) -> list[Instruction]:
    """Static decode template: runtime position_ids/active_mask decide real work."""
    _assert_batch(batch_size)
    instructions: list[Instruction] = []
    for batch_idx in range(batch_size):
        for kv_head_idx in range(LlamaConfig.num_key_value_heads):
            for partial_idx in range(num_partials):
                instructions.append(
                    DecodeAttentionSplitKV(
                        layer_idx=layer_idx,
                        batch_idx=batch_idx,
                        kv_head_idx=kv_head_idx,
                        num_partials=num_partials,
                        partial_idx=partial_idx,
                        seq_len=-1,
                        is_active=-1,
                    )
                )
    return instructions


def schedule_attention_reduction(
    layer_idx: int, batch_size: int, num_partials: int, globs: Globals
) -> list[Instruction]:
    _assert_batch(batch_size)
    instructions: list[Instruction] = []
    for batch_idx in range(batch_size):
        row_on = row_active(globs, batch_idx)
        for head_start_idx in range(0, LlamaConfig.num_attention_heads, ATTN_REDUCTION_SIZE):
            instructions.append(
                DecodeAttentionReduction(
                    layer_idx=layer_idx,
                    batch_idx=batch_idx,
                    head_start_idx=head_start_idx,
                    num_partials=num_partials,
                    reduction_list=list(range(num_partials)),
                    valid_partials=num_partials,
                    is_active=1 if row_on else 0,
                )
            )
    return instructions


def schedule_attention_reduction_runtime(
    layer_idx: int, batch_size: int, num_partials: int
) -> list[Instruction]:
    """Static decode template: merge all partials, filter with partial_valid at runtime."""
    _assert_batch(batch_size)
    instructions: list[Instruction] = []
    full_reduction_list = list(range(num_partials))
    for batch_idx in range(batch_size):
        for head_start_idx in range(0, LlamaConfig.num_attention_heads, ATTN_REDUCTION_SIZE):
            instructions.append(
                DecodeAttentionReduction(
                    layer_idx=layer_idx,
                    batch_idx=batch_idx,
                    head_start_idx=head_start_idx,
                    num_partials=num_partials,
                    reduction_list=full_reduction_list,
                    valid_partials=-1,
                    is_active=-1,
                )
            )
    return instructions


def schedule_oproj_gemm(layer_idx: int, batch_size: int, worker_count: int) -> list[Instruction]:
    _assert_batch(batch_size)
    instructions: list[Instruction] = []
    num_n_tiles = _ceil_div(LlamaConfig.hidden_size, TILE_N_GRANULARITY)
    for n_start, n_end in _split_n_tiles(num_n_tiles, worker_count):
        instructions.append(
            OProj_GEMM_Residual(
                layer_idx=layer_idx,
                batch_size=batch_size,
                tile_M_idx=0,
                tile_N_start_idx=n_start,
                tile_N_end_idx=n_end,
                reduction_block_idx=0,
            )
        )
    return instructions


def schedule_upgate(layer_idx: int, batch_size: int, worker_count: int) -> list[Instruction]:
    _assert_batch(batch_size)
    instructions: list[Instruction] = []
    num_n_tiles = _ceil_div(LlamaConfig.intermediate_size, TILE_N_GRANULARITY)
    for n_start, n_end in _split_n_tiles(num_n_tiles, worker_count):
        tile_block_idxs = list(range(n_start, n_end))
        for chunk in _chunk_list(tile_block_idxs, MAX_TILE_BLOCKS_PER_INSTRUCTION):
            instructions.append(
                UpGateProj_GEMM_SiLU(
                    layer_idx=layer_idx,
                    batch_size=batch_size,
                    tile_M_idx=0,
                    tile_N_block_idxs=chunk,
                )
            )
    return instructions


def schedule_mlp_rms(layer_idx: int, batch_size: int) -> list[Instruction]:
    _assert_batch(batch_size)
    return [
        MlpRMS(layer_idx=layer_idx, batch_size=batch_size, batch_row=row)
        for row in range(batch_size)
    ]


def schedule_downproj(layer_idx: int, batch_size: int, worker_count: int) -> list[Instruction]:
    """Same N-tile split as OProj; full K per instruction (no split-K)."""
    _assert_batch(batch_size)
    instructions: list[Instruction] = []
    num_n_tiles = _ceil_div(LlamaConfig.hidden_size, TILE_N_GRANULARITY)
    for n_start, n_end in _split_n_tiles(num_n_tiles, worker_count):
        instructions.append(
            DownProj_GEMM_Residual(
                layer_idx=layer_idx,
                batch_size=batch_size,
                tile_M_idx=0,
                tile_N_start_idx=n_start,
                tile_N_end_idx=n_end,
            )
        )
    return instructions


def schedule_lmhead_gemm(batch_size: int, worker_count: int) -> list[Instruction]:
    _assert_batch(batch_size)
    instructions: list[Instruction] = []
    num_logit_tiles = _ceil_div(LlamaConfig.vocab_size, TILE_N_GRANULARITY)
    for n_start, n_end in _split_n_tiles(num_logit_tiles, worker_count):
        instructions.append(
            LMHeadProj_GEMM(
                batch_size=batch_size,
                start_output_block_idx=n_start,
                end_output_block_idx=n_end,
            )
        )
    return instructions


def schedule_lmhead_rms(batch_size: int) -> list[Instruction]:
    _assert_batch(batch_size)
    return [LMHeadRMS(batch_size=batch_size, batch_row=row) for row in range(batch_size)]


def _max_kv_blocks_over_active_rows(globs: Globals, batch_size: int) -> int:
    kv_bs = LlamaConfig.attn_kv_block_size
    mx = 1
    for b in range(batch_size):
        if not row_active(globs, b):
            continue
        seq_len = (
            int(globs.kv_lengths[b].item())
            if globs.kv_lengths is not None
            else decode_pos_id(globs, b) + 1
        )
        mx = max(mx, _ceil_div(seq_len, kv_bs))
    return mx


def _num_partials_for_globs(globs: Globals, batch_size: int, worker_count: int) -> int:
    active_b = sum(1 for b in range(batch_size) if row_active(globs, b))
    if active_b <= 0:
        return 1

    kvh = LlamaConfig.num_key_value_heads
    max_blocks = _max_kv_blocks_over_active_rows(globs, batch_size)
    max_blocks = max(1, max_blocks)

    # Dynamic policy: keep split-KV task count close to the persistent worker count.
    # target_tasks ~= active_b * kvh * num_partials, clamped by available SMs.
    target_tasks = min(worker_count, active_b * kvh * max_blocks)
    num_partials = _ceil_div(target_tasks, active_b * kvh)

    return min(MAX_REDUCTION_LIST, max(1, min(num_partials, max_blocks)))


def _num_partials_for_max_seq_len(max_seq_len: int, batch_size: int, worker_count: int) -> int:
    kvh = LlamaConfig.num_key_value_heads
    max_blocks = max(1, _ceil_div(max_seq_len, LlamaConfig.attn_kv_block_size))
    target_tasks = min(worker_count, batch_size * kvh * max_blocks)
    num_partials = _ceil_div(target_tasks, batch_size * kvh)
    return min(MAX_REDUCTION_LIST, max(1, min(num_partials, max_blocks)))


def _batch_size_from_globs(globs: Globals) -> int:
    hs = globs.hidden_states
    if hs is None:
        raise ValueError("globs.hidden_states is None")
    if hs.ndim == 1:
        return 1
    return int(hs.shape[0])


def build_instruction_list(
    globs: Globals,
    num_layers: int | None = None,
    batch_size: int | None = None,
) -> list[Instruction]:
    device = _get_device(globs)
    worker_count = _get_worker_count(device)
    B = batch_size if batch_size is not None else _batch_size_from_globs(globs)
    _assert_batch(B)
    layers = num_layers if num_layers is not None else LlamaConfig.num_hidden_layers
    num_partials = _num_partials_for_globs(globs, B, worker_count)
    instructions: list[Instruction] = []
    for layer_idx in range(layers):
        instructions.extend(schedule_attn_rms(layer_idx, B))
        instructions.extend(schedule_qkvproj_gemm(layer_idx, B, worker_count))
        instructions.extend(schedule_attention_split_kv(layer_idx, B, num_partials, globs))
        instructions.extend(schedule_attention_reduction(layer_idx, B, num_partials, globs))
        instructions.extend(schedule_oproj_gemm(layer_idx, B, worker_count))
        instructions.extend(schedule_mlp_rms(layer_idx, B))
        instructions.extend(schedule_upgate(layer_idx, B, worker_count))
        instructions.extend(schedule_downproj(layer_idx, B, worker_count))
    if layers == LlamaConfig.num_hidden_layers:
        instructions.extend(schedule_lmhead_rms(B))
        instructions.extend(schedule_lmhead_gemm(B, worker_count))
    return instructions


def build_instruction_queues(
    globs: Globals,
    num_layers: int | None = None,
    batch_size: int | None = None,
):
    device = _get_device(globs)
    worker_count = _get_worker_count(device)
    instructions = build_instruction_list(
        globs,
        num_layers=num_layers,
        batch_size=batch_size,
    )
    return _round_robin_assign_to_workers(instructions, worker_count)


def build_runtime_instruction_queues(
    globs: Globals,
    num_layers: int | None = None,
    batch_size: int | None = None,
):
    """
    Build a static decode template once and rely on runtime position_ids/active_mask.
    Intended for multi-batch decode hot path to avoid per-step Python rebuild.
    """
    device = _get_device(globs)
    worker_count = _get_worker_count(device)
    B = batch_size if batch_size is not None else _batch_size_from_globs(globs)
    _assert_batch(B)
    layers = num_layers if num_layers is not None else LlamaConfig.num_hidden_layers
    max_seq_len = int(globs.k_cache.shape[2]) if globs.k_cache is not None else 1
    num_partials = _num_partials_for_max_seq_len(max_seq_len, B, worker_count)
    instructions: list[Instruction] = []
    for layer_idx in range(layers):
        _append_worker_synchronized_wave(
            instructions,
            worker_count,
            [AttnRMS(layer_idx=layer_idx, batch_size=B, batch_row=br) for br in range(B)]
            + [NoOp()] * (worker_count - B),
        )
        _append_instructions_as_worker_waves(
            instructions, worker_count, schedule_qkvproj_gemm(layer_idx, B, worker_count)
        )
        _append_instructions_as_worker_waves(
            instructions, worker_count, schedule_attention_split_kv_runtime(layer_idx, B, num_partials)
        )
        _append_instructions_as_worker_waves(
            instructions, worker_count, schedule_attention_reduction_runtime(layer_idx, B, num_partials)
        )
        _append_instructions_as_worker_waves(
            instructions, worker_count, schedule_oproj_gemm(layer_idx, B, worker_count)
        )
        _append_worker_synchronized_wave(
            instructions,
            worker_count,
            [MlpRMS(layer_idx=layer_idx, batch_size=B, batch_row=br) for br in range(B)]
            + [NoOp()] * (worker_count - B),
        )
        _append_instructions_as_worker_waves(
            instructions, worker_count, schedule_upgate(layer_idx, B, worker_count)
        )
        _append_instructions_as_worker_waves(
            instructions, worker_count, schedule_downproj(layer_idx, B, worker_count)
        )
    if layers == LlamaConfig.num_hidden_layers:
        _append_worker_synchronized_wave(
            instructions,
            worker_count,
            [LMHeadRMS(batch_size=B, batch_row=br) for br in range(B)]
            + [NoOp()] * (worker_count - B),
        )
        _append_instructions_as_worker_waves(
            instructions, worker_count, schedule_lmhead_gemm(B, worker_count)
        )
    return _round_robin_assign_to_workers(instructions, worker_count)
