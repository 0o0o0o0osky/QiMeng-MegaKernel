"""Hopper multi-batch schedule for one logical B<=16 / physical M=64 tile."""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch

from qimeng_megakernel.models.instruction import Instruction
from qimeng_megakernel.models.LLM.decode_state import decode_pos_id, row_active

from ..globs import MAX_BATCH_SIZE, Globals
from ..model_config import LlamaConfig
from .instruction_defs import (
    DOWNPROJ_TILE_N,
    LMHEAD_TILE_N,
    OPROJ_TILE_N,
    QKV_TILE_N,
    UPGATE_TILE_N,
    DecodeAttentionReduction,
    DecodeAttentionSplitKV,
    DownProj_GEMM_Residual,
    OProj_GEMM_Residual,
    RMS_Attn_Norm,
    RMS_LMHead_GEMM,
    RMS_LMHead_Norm,
    RMS_Mlp_Norm,
    RMS_QKVProj_GEMM_RoPE,
    RMS_UpGateProj_GEMM_SiLU,
)

INTS_PER_INSTRUCTION = 32
ATTN_REDUCTION_SIZE = 4
MAX_TILE_BLOCKS_PER_INSTRUCTION = INTS_PER_INSTRUCTION - 6
MAX_REDUCTION_LIST = INTS_PER_INSTRUCTION - 5

H100_SM_COUNT = 132


@dataclass
class NoOp(Instruction):
    @classmethod
    def opcode(cls) -> int:
        return 0

    @classmethod
    def prev_opcode(cls) -> tuple[int, ...]:
        return ()


@dataclass
class _RangeInstruction(Instruction):
    """Internal CUDA range record; public/Python-VM instructions stay intact."""

    operation: int
    layer: int
    begin: int
    end: int
    stride: int
    dep_layer: int
    dep_row: int
    dep_expected: int
    dep_slot: int
    publish_row: int
    publish_slot: int
    batch_size: int

    def opcode(self) -> int:
        return self.operation

    @classmethod
    def prev_opcode(cls) -> tuple[int, ...]:
        return ()

    def serialize(self) -> list[int]:
        return [
            self.operation,
            self.layer,
            self.begin,
            self.end,
            self.stride,
            self.dep_layer,
            self.dep_row,
            self.dep_expected,
            self.dep_slot,
            self.publish_row,
            self.publish_slot,
            self.batch_size,
        ]


OPCODE_QKV = 1
OPCODE_ATTENTION = 2
OPCODE_OPROJ = 4
OPCODE_UPGATE = 5
OPCODE_DOWN = 6
OPCODE_LMHEAD = 7
OPCODE_RMS_ATTN = 8
OPCODE_RMS_MLP = 9
OPCODE_RMS_LMHEAD = 10
OPCODE_DOWN_REDUCE = 11


BAR_QKV = 0
BAR_ATTENTION = 1
BAR_OPROJ = 2
BAR_UPGATE = 3
BAR_DOWN = 4
BAR_LAYER_READY = 5
BAR_RMS_ATTN = 7
BAR_RMS_MLP = 8
BAR_RMS_LMHEAD = 9


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
    if sm_count < H100_SM_COUNT:
        raise ValueError(
            f"This multi_batch schedule targets H100 ({H100_SM_COUNT} SMs). "
            f"Device has {sm_count} SMs."
        )
    return min(sm_count, H100_SM_COUNT)


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
    flattened: list[list[int]] = []
    for queue in instruction_queues:
        padded = list(queue) + [NoOp()] * (max_queue_len - len(queue))
        flattened.extend(_serialize_and_pad(instruction) for instruction in padded)

    device = _get_device(globs)
    serialized = torch.tensor(flattened, dtype=torch.int32, device=device).view(
        num_sms, -1, INTS_PER_INSTRUCTION
    )
    globs.instructions = serialized
    return serialized


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


def _chunk_list(items: list[int], chunk_size: int) -> list[list[int]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def _round_robin_assign_to_sms(
    instructions: list[Instruction], sm_count: int
) -> list[list[Instruction]]:
    sm_queues = [[] for _ in range(sm_count)]
    for i, instruction in enumerate(instructions):
        sm_queues[i % sm_count].append(instruction)
    return sm_queues


def _append_sm_synchronized_wave(
    flat: list[Instruction], sm_count: int, wave: list[Instruction]
) -> None:
    if len(wave) != sm_count:
        raise ValueError(
            f"SM wave must have length {sm_count} (got {len(wave)}); "
            "row RMS norms need one instruction slot per SM per timestep."
        )
    flat.extend(wave)


def _append_instructions_as_sm_waves(
    flat: list[Instruction], sm_count: int, segment: list[Instruction]
) -> None:
    if not segment:
        return
    for i in range(0, len(segment), sm_count):
        chunk = list(segment[i : i + sm_count])
        if len(chunk) < sm_count:
            chunk.extend([NoOp()] * (sm_count - len(chunk)))
        flat.extend(chunk)


def _assert_batch(batch_size: int) -> None:
    if batch_size < 2 or batch_size > MAX_BATCH_SIZE:
        raise ValueError(
            f"H100 multi_batch expects 2 <= batch_size <= {MAX_BATCH_SIZE}, got {batch_size}"
        )


def schedule_qkvproj_gemm(
    layer_idx: int, batch_size: int, sm_count: int
) -> list[Instruction]:
    _assert_batch(batch_size)
    instructions: list[Instruction] = []
    num_output_blocks = _ceil_div(LlamaConfig.qkv_out_dim, QKV_TILE_N)
    for n_start, n_end in _split_n_tiles(num_output_blocks, sm_count):
        instructions.append(
            RMS_QKVProj_GEMM_RoPE(
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
        blocks_per_partial = (
            max(1, _ceil_div(total_blocks, num_partials)) if row_on else 1
        )

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
    """Runtime decode template: position_ids/active_mask decide real work."""
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
        for head_start_idx in range(
            0, LlamaConfig.num_attention_heads, ATTN_REDUCTION_SIZE
        ):
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
    """Runtime template: merge all partials, filter with partial_valid at runtime."""
    _assert_batch(batch_size)
    instructions: list[Instruction] = []
    full_reduction_list = list(range(num_partials))
    for batch_idx in range(batch_size):
        for head_start_idx in range(
            0, LlamaConfig.num_attention_heads, ATTN_REDUCTION_SIZE
        ):
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


def schedule_oproj_gemm(
    layer_idx: int, batch_size: int, sm_count: int
) -> list[Instruction]:
    _assert_batch(batch_size)
    instructions: list[Instruction] = []
    num_n_tiles = _ceil_div(LlamaConfig.hidden_size, OPROJ_TILE_N)
    for n_start, n_end in _split_n_tiles(num_n_tiles, sm_count):
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


def schedule_upgate(
    layer_idx: int, batch_size: int, sm_count: int
) -> list[Instruction]:
    _assert_batch(batch_size)
    instructions: list[Instruction] = []
    num_n_tiles = _ceil_div(LlamaConfig.intermediate_size, UPGATE_TILE_N)
    for n_start, n_end in _split_n_tiles(num_n_tiles, sm_count):
        tile_block_idxs = list(range(n_start, n_end))
        for chunk in _chunk_list(tile_block_idxs, MAX_TILE_BLOCKS_PER_INSTRUCTION):
            instructions.append(
                RMS_UpGateProj_GEMM_SiLU(
                    layer_idx=layer_idx,
                    batch_size=batch_size,
                    tile_M_idx=0,
                    tile_N_block_idxs=chunk,
                )
            )
    return instructions


def schedule_downproj(
    layer_idx: int, batch_size: int, sm_count: int
) -> list[Instruction]:
    """Same N-tile split as OProj; full K per instruction (no split-K)."""
    _assert_batch(batch_size)
    instructions: list[Instruction] = []
    num_n_tiles = _ceil_div(LlamaConfig.hidden_size, DOWNPROJ_TILE_N)
    for n_start, n_end in _split_n_tiles(num_n_tiles, sm_count):
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


def schedule_lmhead_gemm(batch_size: int, sm_count: int) -> list[Instruction]:
    _assert_batch(batch_size)
    instructions: list[Instruction] = []
    num_logit_tiles = _ceil_div(LlamaConfig.vocab_size, LMHEAD_TILE_N)
    # One WGMMA tile owns one VM instruction.  The old A100/CUDA-core schedule
    # packed 7--8 vocabulary tiles into each instruction, but the Hopper
    # matmul pipeline owns one output page and one page lifecycle per
    # instruction.  Packing multiple tiles would either overwrite that page or
    # hold it across unrelated work.  The wave builder below distributes these
    # 1002 tiles over the 132 persistent CTAs.
    del sm_count
    for n_start in range(num_logit_tiles):
        n_end = n_start + 1
        instructions.append(
            RMS_LMHead_GEMM(
                batch_size=batch_size,
                start_output_block_idx=n_start,
                end_output_block_idx=n_end,
            )
        )
    return instructions


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


def _num_partials_for_globs(globs: Globals, batch_size: int, sm_count: int) -> int:
    active_b = sum(1 for b in range(batch_size) if row_active(globs, b))
    if active_b <= 0:
        return 1

    kvh = LlamaConfig.num_key_value_heads
    max_blocks = _max_kv_blocks_over_active_rows(globs, batch_size)
    max_blocks = max(1, max_blocks)

    # Use the largest one-wave floor. B=16 gives P=1 and 128 attention CTAs.
    num_partials = max(1, sm_count // (active_b * kvh))

    return min(MAX_REDUCTION_LIST, max(1, min(num_partials, max_blocks)))


def _num_partials_for_max_seq_len(
    max_seq_len: int, batch_size: int, sm_count: int
) -> int:
    kvh = LlamaConfig.num_key_value_heads
    max_blocks = max(1, _ceil_div(max_seq_len, LlamaConfig.attn_kv_block_size))
    num_partials = max(1, sm_count // (batch_size * kvh))
    return min(MAX_REDUCTION_LIST, max(1, min(num_partials, max_blocks)))


def _batch_size_from_globs(globs: Globals) -> int:
    hs = globs.hidden_states
    if hs is None:
        raise ValueError("globs.hidden_states is None")
    if hs.ndim == 1:
        return 1
    return int(hs.shape[0])


def build_instruction_list(
    globs: Globals, num_layers: int | None = None, batch_size: int | None = None
) -> list[Instruction]:
    device = _get_device(globs)
    sm_count = _get_sm_count(device)
    B = batch_size if batch_size is not None else _batch_size_from_globs(globs)
    _assert_batch(B)
    layers = num_layers if num_layers is not None else LlamaConfig.num_hidden_layers
    num_partials = _num_partials_for_globs(globs, B, sm_count)
    instructions: list[Instruction] = []
    for layer_idx in range(layers):
        instructions.extend(schedule_qkvproj_gemm(layer_idx, B, sm_count))
        instructions.extend(
            schedule_attention_split_kv(layer_idx, B, num_partials, globs)
        )
        if num_partials > 1:
            instructions.extend(
                schedule_attention_reduction(layer_idx, B, num_partials, globs)
            )
        instructions.extend(schedule_oproj_gemm(layer_idx, B, sm_count))
        instructions.extend(schedule_upgate(layer_idx, B, sm_count))
        instructions.extend(schedule_downproj(layer_idx, B, sm_count))
    if layers == LlamaConfig.num_hidden_layers:
        instructions.extend(schedule_lmhead_gemm(B, sm_count))
    return instructions


def build_instruction_queues(
    globs: Globals, num_layers: int | None = None, batch_size: int | None = None
):
    return build_runtime_instruction_queues(
        globs, num_layers=num_layers, batch_size=batch_size
    )


def _range_wave(
    operation: int,
    *,
    layer: int,
    count: int,
    batch_size: int,
    sm_count: int,
    dep_layer: int,
    dep_row: int,
    dep_expected: int,
    publish_row: int,
    dep_slot: int = 0,
    publish_slot: int = 0,
) -> list[Instruction]:
    """Emit exactly one range record per SM for a synchronized phase."""
    if count < 0:
        raise ValueError("count must be non-negative")
    wave: list[Instruction] = []
    for sm_id in range(sm_count):
        if sm_id >= count:
            wave.append(NoOp())
            continue
        wave.append(
            _RangeInstruction(
                operation=operation,
                layer=layer,
                begin=sm_id,
                end=count,
                stride=sm_count,
                dep_layer=dep_layer,
                dep_row=dep_row,
                dep_expected=dep_expected,
                dep_slot=dep_slot,
                publish_row=publish_row,
                publish_slot=publish_slot,
                batch_size=batch_size,
            )
        )
    return wave


def build_runtime_instruction_queues(
    globs: Globals, num_layers: int | None = None, batch_size: int | None = None
):
    """Build the canonical two-slot VM template used by the CUDA kernel."""
    device = _get_device(globs)
    sm_count = _get_sm_count(device)
    B = batch_size if batch_size is not None else _batch_size_from_globs(globs)
    _assert_batch(B)
    layers = num_layers if num_layers is not None else LlamaConfig.num_hidden_layers
    if layers < 1 or layers > LlamaConfig.num_hidden_layers:
        raise ValueError("num_layers is outside the model")

    batch_tiles = 1 if B <= 8 else 2
    qkv_count = _ceil_div(LlamaConfig.qkv_out_dim, 64) * batch_tiles
    attention_count = B * LlamaConfig.num_key_value_heads
    oproj_count = _ceil_div(LlamaConfig.hidden_size, 64) * batch_tiles
    upgate_count = _ceil_div(LlamaConfig.intermediate_size, 64)
    down_count = _ceil_div(LlamaConfig.hidden_size, 64) * 4
    down_reduce_count = _ceil_div(LlamaConfig.hidden_size, 64)

    flat: list[Instruction] = []
    for layer_idx in range(layers):
        prev_layer = max(layer_idx - 1, 0)
        flat.extend(
            _range_wave(
                OPCODE_QKV,
                layer=layer_idx,
                count=qkv_count,
                batch_size=B,
                sm_count=sm_count,
                dep_layer=prev_layer,
                dep_row=BAR_LAYER_READY,
                dep_expected=0 if layer_idx == 0 else down_reduce_count,
                publish_row=BAR_RMS_ATTN,
            )
        )
        flat.extend(
            _range_wave(
                OPCODE_ATTENTION,
                layer=layer_idx,
                count=attention_count,
                batch_size=B,
                sm_count=sm_count,
                dep_layer=layer_idx,
                dep_row=BAR_QKV,
                dep_expected=6,
                publish_row=BAR_ATTENTION,
            )
        )
        flat.extend(
            _range_wave(
                OPCODE_OPROJ,
                layer=layer_idx,
                count=oproj_count,
                batch_size=B,
                sm_count=sm_count,
                dep_layer=layer_idx,
                dep_row=BAR_ATTENTION,
                dep_expected=8 * LlamaConfig.num_key_value_heads,
                publish_row=BAR_OPROJ,
            )
        )
        flat.extend(
            _range_wave(
                OPCODE_UPGATE,
                layer=layer_idx,
                count=upgate_count,
                batch_size=B,
                sm_count=sm_count,
                dep_layer=layer_idx,
                dep_row=BAR_OPROJ,
                dep_expected=LlamaConfig.hidden_size // 64,
                publish_row=BAR_RMS_MLP,
            )
        )
        flat.extend(
            _range_wave(
                OPCODE_DOWN,
                layer=layer_idx,
                count=down_count,
                batch_size=B,
                sm_count=sm_count,
                dep_layer=layer_idx,
                dep_row=BAR_UPGATE,
                dep_expected=upgate_count // 4,
                publish_row=BAR_DOWN,
            )
        )
        flat.extend(
            _range_wave(
                OPCODE_DOWN_REDUCE,
                layer=layer_idx,
                count=down_reduce_count,
                batch_size=B,
                sm_count=sm_count,
                dep_layer=layer_idx,
                dep_row=BAR_DOWN,
                dep_expected=4,
                publish_row=BAR_LAYER_READY,
            )
        )
    if layers == LlamaConfig.num_hidden_layers:
        final_layer = layers - 1
        lmhead_count = _ceil_div(_ceil_div(LlamaConfig.vocab_size, 64), 2)
        flat.extend(
            _range_wave(
                OPCODE_LMHEAD,
                layer=final_layer,
                count=lmhead_count,
                batch_size=B,
                sm_count=sm_count,
                dep_layer=final_layer,
                dep_row=BAR_LAYER_READY,
                dep_expected=down_reduce_count,
                publish_row=BAR_RMS_LMHEAD,
                publish_slot=1,
            )
        )
    return _round_robin_assign_to_sms(flat, sm_count)
