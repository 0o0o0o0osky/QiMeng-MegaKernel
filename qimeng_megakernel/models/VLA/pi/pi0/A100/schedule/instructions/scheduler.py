"""Shape-specialized pi0 A100 schedulers for dependency-ordered module instructions."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Iterable, TypeVar

from qimeng_megakernel.models.instruction import Instruction

from ..pi0_config import Pi0A100Config as C
from . import instruction_defs as I

T = TypeVar("T")


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


class Schedule(list):
    """An instruction list that remembers how many queues it was ordered for.

    ``_round_robin_flatten`` lays work out so that the i-th instruction belongs to
    CTA ``i % queues``, and ``pack_instructions`` re-derives that same assignment.
    Carrying the count prevents packing a schedule for a different launch grid.
    """

    __slots__ = ("queues",)

    def __init__(self, items: Iterable[Instruction], queues: int) -> None:
        super().__init__(items)
        self.queues = queues


def pack_instructions(
    instructions: Iterable[Instruction],
    *,
    sm_count: int | None = None,
    instruction_width: int | None = None,
):
    """Round-robin pack serialized instructions into [1, sm_count, depth, width] queues.

    ``sm_count`` defaults to whatever the schedule was distributed over.

    Returns ``(tensor_list_python, depth)`` — caller converts to torch on device.
    """
    if sm_count is None:
        sm_count = getattr(instructions, "queues", None) or C.sm_count
    instruction_width = C.instruction_width if instruction_width is None else instruction_width
    queues: list[list[list[int]]] = [[] for _ in range(sm_count)]
    for i, inst in enumerate(instructions):
        words = list(inst.serialize())
        if len(words) > instruction_width:
            raise ValueError(f"instruction wider than {instruction_width}: {words}")
        words = words + [0] * (instruction_width - len(words))
        queues[i % sm_count].append(words)
    depth = max((len(q) for q in queues), default=0)
    for q in queues:
        while len(q) < depth:
            q.append([0] * instruction_width)
    return queues, depth


def _round_robin_flatten(work: Sequence[T], sm_count: int) -> Schedule:
    queues: list[list[T]] = [[] for _ in range(sm_count)]
    for i, item in enumerate(work):
        queues[i % sm_count].append(item)
    depth = max((len(q) for q in queues), default=0)
    out: list[T] = []
    for j in range(depth):
        for sm in range(sm_count):
            if j < len(queues[sm]):
                out.append(queues[sm][j])
    return Schedule(out, sm_count)


def _distribute(work: Sequence[Instruction], sm_count: int | None = None) -> Schedule:
    return _round_robin_flatten(work, C.sm_count if sm_count is None else sm_count)


# Each module has its own queue count; keep separate distribution helpers.
def _distribute_siglip(work: Sequence[Instruction], sm_count: int | None = None) -> Schedule:
    """Distribute SigLIP work over its production grid."""
    return _round_robin_flatten(work, C.siglip_queues if sm_count is None else sm_count)


def _distribute_gemma(work: Sequence[Instruction], sm_count: int | None = None) -> Schedule:
    """Distribute Gemma work over its production grid."""
    return _round_robin_flatten(work, C.gemma_queues if sm_count is None else sm_count)


def _distribute_ae(work: Sequence[Instruction], sm_count: int | None = None) -> Schedule:
    """Distribute AE work over its production grid."""
    return _round_robin_flatten(work, C.ae_queues if sm_count is None else sm_count)


# ── SigLIP ──────────────────────────────────────────────────────────────────

LN_BATCH = 8
SIGLIP_TILE_M = 128
SIGLIP_TILE_N = 128
# 64 query rows per attention instruction: 4 tiles x 16 heads = 64 instructions at
# 1 view where 128 rows gave 32, and 32 instructions leave 76 of the 108 CTAs with
# nothing to do for the whole phase.  Must match kBlockM in siglip_attn.cu.
SIGLIP_Q_TILE_ROWS = 64
SIGLIP_ACT_PAD = 1280
SIGLIP_QKV_N_PAD = 3840
SIGLIP_MLP_N_PAD = 4608
SIGLIP_FC2_SPLITS = 6
SIGLIP_FC2_K_SEG = SIGLIP_MLP_N_PAD // SIGLIP_FC2_SPLITS
# Per-layer GEMMs emit one instruction per 64x64 MMA tile.
SIGLIP_MMA_TILE = 64
SIGLIP_QKV_TILE_N = 128
SIGLIP_SEG_TILES = SIGLIP_FC2_K_SEG // SIGLIP_MMA_TILE
SIGLIP_LN_STATS_TILES = SIGLIP_ACT_PAD // SIGLIP_TILE_N
SIGLIP_LN_STATS_ATTN_ROW = 2 * SIGLIP_LN_STATS_TILES
SIGLIP_PATCH_K = 3 * 14 * 14
SIGLIP_PATCH_K_PAD = 768
SIGLIP_FINAL_NORM_WEIGHT_ROW = 2 * C.vision_layers


def _exact_positive_int(name: str, value: int) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be an exact positive integer; got {value!r}")
    return value


def _exact_siglip_views(num_views: int) -> int:
    if type(num_views) is not int or num_views not in C.VIEWS_CHOICES:
        raise ValueError(
            f"num_views must be an exact integer in {C.VIEWS_CHOICES}; got {num_views!r}"
        )
    return num_views


def _exact_dependency(name: str, dep: tuple[int, int, int]) -> tuple[int, int, int]:
    if (
        type(dep) is not tuple
        or len(dep) != 3
        or any(type(value) is not int or value < 0 for value in dep)
    ):
        raise ValueError(
            f"{name} must be an exact non-negative integer triple; got {dep!r}"
        )
    return dep


def _dependency(
    dep: tuple[int, int, int] | None,
    dep_expected: int,
) -> tuple[int, int, int]:
    """Compatibility shim for old per-function ``dep_expected`` callers."""
    return (0, 0, dep_expected) if dep is None else dep


def schedule_siglip_ln_attn(
    layer_idx: int = 0,
    *,
    seq_len: int | None = None,
    stats_row: int = SIGLIP_LN_STATS_ATTN_ROW,
    dep: tuple[int, int, int] | None = None,
    dep_expected: int = 0,
    flags: int = 0,
    sm_count: int | None = None,
) -> list[Instruction]:
    seq_len = C.siglip_seq_pad if seq_len is None else seq_len
    dep_layer, dep_row, expected = _dependency(dep, dep_expected)
    groups = [list(range(i, min(i + LN_BATCH, seq_len))) for i in range(0, seq_len, LN_BATCH)]
    work: list[Instruction] = []
    for grp in groups:
        work.append(
            I.SiglipLnAttn(
                layer_idx=layer_idx,
                token_row=grp[0],
                weight_row=stats_row,
                dep_layer=dep_layer,
                dep_row=dep_row,
                dep_expected=expected,
                num_tokens=grp[-1] + 1 - grp[0],
                flags=flags,
            )
        )
    return _distribute_siglip(work, sm_count)


def schedule_siglip_ln_ffn(
    layer_idx: int = 0,
    *,
    seq_len: int | None = None,
    stats_row: int = 0,
    sq_row: int = 10,
    dep: tuple[int, int, int] | None = None,
    dep_expected: int = 0,
    flags: int = 0,
    sm_count: int | None = None,
) -> list[Instruction]:
    """Schedule the second LayerNorm as its own dependency-visible phase."""
    seq_len = C.siglip_seq_pad if seq_len is None else seq_len
    dep_layer, dep_row, expected = _dependency(dep, dep_expected)
    work = [
        I.SiglipLnFfn(
            layer_idx=layer_idx,
            token_row=start,
            weight_row=stats_row,
            dep_layer=dep_layer,
            dep_row=dep_row,
            dep_expected=expected,
            num_tokens=min(LN_BATCH, seq_len - start),
            sq_row=sq_row,
            flags=flags,
        )
        for start in range(0, seq_len, LN_BATCH)
    ]
    return _distribute_siglip(work, sm_count)


def schedule_siglip_qkv(
    layer_idx: int = 0,
    *,
    seq_len: int | None = None,
    dep_expected: int = 0,
    dep: tuple[int, int, int] | None = None,
    norm_sel: int = SIGLIP_LN_STATS_ATTN_ROW,
    flags: int = 0,
    sm_count: int | None = None,
) -> list[Instruction]:
    seq_len = C.siglip_seq_pad if seq_len is None else seq_len
    dep_layer, dep_row, expected = _dependency(dep, dep_expected)
    work = [
        I.SiglipQkv(
            layer_idx=layer_idx,
            tile_m=m,
            tile_n=n,
            dep_layer=dep_layer,
            dep_row=dep_row,
            dep_expected=expected,
            norm_sel=norm_sel,
            flags=flags,
        )
        for m in range(seq_len // SIGLIP_MMA_TILE)
        for n in range(SIGLIP_QKV_N_PAD // SIGLIP_QKV_TILE_N)
    ]
    return _distribute_siglip(work, sm_count)


def schedule_siglip_attn(
    layer_idx: int = 0,
    *,
    num_views: int | None = None,
    dep_expected: int = 0,
    dep: tuple[int, int, int] | None = None,
    flags: int = 0,
    sm_count: int | None = None,
) -> list[Instruction]:
    num_views = C.num_views if num_views is None else num_views
    dep_layer, dep_row, expected = _dependency(dep, dep_expected)
    tok_per_view = C.vision_tokens_per_view
    heads = C.vision_num_heads
    work: list[Instruction] = []
    for view in range(num_views):
        for q_tile in range(tok_per_view // SIGLIP_Q_TILE_ROWS):
            for head in range(heads):
                work.append(
                    I.SiglipAttn(
                        layer_idx=layer_idx,
                        view_idx=view,
                        q_tile=q_tile,
                        head_idx=head,
                        dep_layer=dep_layer,
                        dep_row=dep_row,
                        dep_expected=expected,
                        flags=flags,
                    )
                )
    return _distribute_siglip(work, sm_count)


def schedule_siglip_oproj(
    layer_idx: int = 0,
    *,
    seq_len: int | None = None,
    dep_expected: int = 0,
    dep: tuple[int, int, int] | None = None,
    flags: int = 0,
    sm_count: int | None = None,
) -> list[Instruction]:
    seq_len = C.siglip_seq_pad if seq_len is None else seq_len
    dep_layer, dep_row, expected = _dependency(dep, dep_expected)
    work = [
        I.SiglipOproj(
            layer_idx=layer_idx,
            tile_m=m,
            tile_n=n,
            dep_layer=dep_layer,
            dep_row=dep_row,
            dep_expected=expected,
            flags=flags,
        )
        for m in range(seq_len // SIGLIP_MMA_TILE)
        for n in range(SIGLIP_ACT_PAD // SIGLIP_MMA_TILE)
    ]
    return _distribute_siglip(work, sm_count)


def schedule_siglip_fc1(
    layer_idx: int = 0,
    *,
    seq_len: int | None = None,
    dep_expected: int = 0,
    dep: tuple[int, int, int] | None = None,
    norm_sel: int = 0,
    flags: int = 0,
    sm_count: int | None = None,
) -> list[Instruction]:
    seq_len = C.siglip_seq_pad if seq_len is None else seq_len
    dep_layer, dep_row, expected = _dependency(dep, dep_expected)
    m_tiles = seq_len // SIGLIP_MMA_TILE
    n_tiles = (C.vision_mlp_width + SIGLIP_MMA_TILE - 1) // SIGLIP_MMA_TILE
    work = [
        I.SiglipFc1(
            layer_idx=layer_idx,
            tile_m=m,
            tile_n=n,
            dep_layer=dep_layer,
            dep_row=dep_row,
            dep_expected=expected,
            norm_sel=norm_sel,
            out_row=(n // SIGLIP_SEG_TILES) * m_tiles + m,
            out_col=n % SIGLIP_SEG_TILES,
            bias_col=n,
            flags=flags,
        )
        for m in range(m_tiles)
        for n in range(n_tiles)
    ]
    return _distribute_siglip(work, sm_count)


def schedule_siglip_fc2(
    layer_idx: int = 0,
    *,
    seq_len: int | None = None,
    dep_expected: int = 0,
    dep: tuple[int, int, int] | None = None,
    flags: int = 0,
    sm_count: int | None = None,
) -> list[Instruction]:
    seq_len = C.siglip_seq_pad if seq_len is None else seq_len
    dep_layer, dep_row, expected = _dependency(dep, dep_expected)
    m_tiles = seq_len // SIGLIP_MMA_TILE
    # Fc2's last 128 output columns are storage padding.  All following consumers use
    # the logical 1152 columns with a 1280 row stride, so they need no producer tiles.
    n_tiles = C.vision_width // SIGLIP_MMA_TILE
    work = [
        I.SiglipFc2(
            layer_idx=layer_idx,
            tile_m=m,
            tile_n=n,
            dep_layer=dep_layer,
            dep_row=dep_row,
            dep_expected=expected,
            out_row=m,
            out_col=n,
            bias_col=n,
            flags=flags,
        )
        for m in range(m_tiles)
        for n in range(n_tiles)
    ]
    return _distribute_siglip(work, sm_count)


def schedule_siglip_layer(
    layer_idx: int = 0,
    *,
    num_views: int | None = None,
    prev_dep: tuple[int, int, int] | None = None,
    flags: int = 0,
    sm_count: int | None = None,
) -> list[Instruction]:
    """One SigLIP encoder block (LN→QKV→attn→oproj→fc1→fc2)."""
    num_views = _exact_siglip_views(C.num_views if num_views is None else num_views)
    seq_len = C.siglip_seq_for(num_views)
    # A row offset into ln_stats that the LnAttn producer and the Qkv consumer
    # agree on, nothing more.  LnFfn holds rows 0 and 10; ln_stats is 42 rows, so
    # 40 is the last pair that fits.  Not derived from any phase's tiling.
    ln_attn_stats_row = 40 if num_views == 1 else 20
    upstream = (0, 0, 0) if prev_dep is None else prev_dep

    ln = schedule_siglip_ln_attn(
        layer_idx=layer_idx, seq_len=seq_len, stats_row=ln_attn_stats_row,
        dep=upstream, flags=flags, sm_count=1
    )
    qkv = schedule_siglip_qkv(
        layer_idx=layer_idx,
        seq_len=seq_len,
        dep=(layer_idx, I.siglip_barrier_row(I.SiglipLnAttn.opcode()), len(ln)),
        norm_sel=ln_attn_stats_row, flags=flags,
        sm_count=1,
    )
    attn = schedule_siglip_attn(
        layer_idx=layer_idx,
        num_views=num_views,
        dep=(layer_idx, I.siglip_barrier_row(I.SiglipQkv.opcode()), len(qkv)),
        flags=flags,
        sm_count=1,
    )
    oproj = schedule_siglip_oproj(
        layer_idx=layer_idx,
        seq_len=seq_len,
        dep=(layer_idx, I.siglip_barrier_row(I.SiglipAttn.opcode()), len(attn)),
        flags=flags,
        sm_count=1,
    )
    ln_ffn = schedule_siglip_ln_ffn(
        layer_idx=layer_idx,
        seq_len=seq_len,
        dep=(layer_idx, I.siglip_barrier_row(I.SiglipOproj.opcode()), len(oproj)),
        flags=flags,
        sm_count=1,
    )
    fc1 = schedule_siglip_fc1(
        layer_idx=layer_idx,
        seq_len=seq_len,
        dep=(layer_idx, I.siglip_barrier_row(I.SiglipLnFfn.opcode()), len(ln_ffn)),
        flags=flags,
        sm_count=1,
    )
    fc2 = schedule_siglip_fc2(
        layer_idx=layer_idx,
        seq_len=seq_len,
        dep=(layer_idx, I.siglip_barrier_row(I.SiglipFc1.opcode()), len(fc1)),
        flags=flags,
        sm_count=1,
    )
    return _distribute_siglip([*ln, *qkv, *attn, *oproj, *ln_ffn, *fc1, *fc2], sm_count)


def schedule_siglip_stack(
    n_layers: int,
    *,
    num_views: int | None = None,
    initial_dep: tuple[int, int, int] | None = None,
    sm_count: int | None = None,
) -> list[Instruction]:
    work: list[Instruction] = []
    prev_dep = initial_dep
    for layer in range(n_layers):
        layer_work = schedule_siglip_layer(
            layer_idx=layer,
            num_views=num_views,
            prev_dep=prev_dep,
            sm_count=1,
        )
        work.extend(layer_work)
        fc2_count = sum(isinstance(inst, I.SiglipFc2) for inst in layer_work)
        prev_dep = (layer, I.siglip_barrier_row(I.SiglipFc2.opcode()), fc2_count)
    return _distribute_siglip(work, sm_count)


def schedule_siglip_patch(
    *,
    num_views: int,
    sm_count: int | None = None,
) -> list[Instruction]:
    """Canonical checkpoint-backed patch projection phase (opcode 8)."""
    views = _exact_siglip_views(num_views)
    sms = C.siglip_queues if sm_count is None else _exact_positive_int("sm_count", sm_count)
    seq_len = C.siglip_seq_for(views)
    work = [
        I.SiglipPatch(layer_idx=0, tile_m=m, tile_n=n)
        for m in range(seq_len // SIGLIP_TILE_M)
        for n in range(SIGLIP_ACT_PAD // SIGLIP_TILE_N)
    ]
    return _distribute_siglip(work, sms)


def schedule_siglip_final_norm(
    *,
    num_views: int,
    dep: tuple[int, int, int],
    sm_count: int | None = None,
) -> list[Instruction]:
    """Canonical post-tower LayerNorm phase (opcode 9)."""
    views = _exact_siglip_views(num_views)
    dependency = _exact_dependency("dep", dep)
    sms = C.siglip_queues if sm_count is None else _exact_positive_int("sm_count", sm_count)
    seq_len = C.siglip_seq_for(views)
    dep_layer, dep_row, dep_expected = dependency
    work = [
        I.SiglipFinalNorm(
            layer_idx=0,
            token_row=token_row,
            weight_row=SIGLIP_FINAL_NORM_WEIGHT_ROW,
            dep_layer=dep_layer,
            dep_row=dep_row,
            dep_expected=dep_expected,
            num_tokens=min(LN_BATCH, seq_len - token_row),
        )
        for token_row in range(0, seq_len, LN_BATCH)
    ]
    return _distribute_siglip(work, sms)


def schedule_siglip_projector(
    *,
    num_views: int,
    dep: tuple[int, int, int],
    sm_count: int | None = None,
) -> list[Instruction]:
    """Canonical vision-to-Gemma projector phase (opcode 10)."""
    views = _exact_siglip_views(num_views)
    dependency = _exact_dependency("dep", dep)
    sms = C.siglip_queues if sm_count is None else _exact_positive_int("sm_count", sm_count)
    seq_len = C.siglip_seq_for(views)
    dep_layer, dep_row, dep_expected = dependency
    work = [
        I.SiglipProjector(
            layer_idx=0,
            tile_m=m,
            tile_n=n,
            dep_layer=dep_layer,
            dep_row=dep_row,
            dep_expected=dep_expected,
        )
        for m in range(seq_len // SIGLIP_TILE_M)
        for n in range(C.encoder_width // SIGLIP_TILE_N)
    ]
    return _distribute_siglip(work, sms)


def schedule_siglip_runtime(
    *,
    num_views: int,
    sm_count: int | None = None,
) -> list[Instruction]:
    """Emit production opcode 8→(1..7)×27→9→10.

    Runtime flags are fixed to zero.
    Both LayerNorm phases are explicit dependency-visible instructions.
    """
    views = _exact_siglip_views(num_views)
    sms = C.siglip_queues if sm_count is None else _exact_positive_int("sm_count", sm_count)

    patch = schedule_siglip_patch(num_views=views, sm_count=1)
    patch_dep = (0, I.siglip_barrier_row(I.SiglipPatch.opcode()), len(patch))
    stack = schedule_siglip_stack(
        C.vision_layers,
        num_views=views,
        initial_dep=patch_dep,
        sm_count=1,
    )
    fc2_count = (
        (C.siglip_seq_for(views) // SIGLIP_MMA_TILE)
        * (C.vision_width // SIGLIP_MMA_TILE)
    )
    final_norm = schedule_siglip_final_norm(
        num_views=views,
        dep=(
            C.vision_layers - 1,
            I.siglip_barrier_row(I.SiglipFc2.opcode()),
            fc2_count,
        ),
        sm_count=1,
    )
    projector = schedule_siglip_projector(
        num_views=views,
        dep=(0, I.siglip_barrier_row(I.SiglipFinalNorm.opcode()), len(final_norm)),
        sm_count=1,
    )
    return _distribute_siglip([*patch, *stack, *final_norm, *projector], sms)


# ── Gemma encoder ───────────────────────────────────────────────────────────

GEMMA_LN_BATCH = 8
GEMMA_TILE_M = C.tile_m
GEMMA_TILE_N = C.tile_n
# QKV uses one GEMM tile per instruction. The scheduler and
# gemma_qkv_rope.cu must use the same N-tile width.
GEMMA_QKV_TILE_N = 64
GEMMA_Q_TILE = C.gemma_q_tile
GEMMA_KV_BLK = C.gemma_kv_block
GEMMA_KV_SPLITS = C.gemma_kv_splits
GEMMA_DOWN_SPLITS_COMPILED = 2

# Materialized attention's three phases.  The two GEMM tiles are the ones
# `gemma_attn.cu` instantiates and the two files have to agree; the transpose
# tile is 32x32 because that is what one staged shared-memory tile holds.
GEMMA_QK_TILE_M = 128
GEMMA_QK_TILE_N = 128
GEMMA_PV_TILE_M = 64
GEMMA_PV_TILE_N = 128
GEMMA_VT_TILE = 32
GEMMA_SOFTMAX_WARPS = 4  # kThreads / 32; one warp owns one row


def kv_spans(seq_len: int, n_splits: int | None = None) -> list[tuple[int, int]]:
    """Return KV block spans shared by the AE scheduler and verifier helpers."""
    n_splits = GEMMA_KV_SPLITS if n_splits is None else n_splits
    nblk = _ceil_div(seq_len, GEMMA_KV_BLK)
    n_splits = max(1, min(n_splits, nblk))
    per = _ceil_div(nblk, n_splits)
    spans, b = [], 0
    while b < nblk:
        cnt = min(per, nblk - b)
        spans.append((b, cnt))
        b += cnt
    return spans


def gemma_seq_pad(seq_len: int) -> int:
    """The padded row count every Gemma GEMM actually covers.

    `g.seq` on the device side, and the leading dimension of the score matrix.
    """
    return _ceil_div(seq_len, GEMMA_TILE_M) * GEMMA_TILE_M


def schedule_gemma_attn(
    layer_idx: int,
    seq_len: int,
    *,
    sm_count: int | None = None,
) -> tuple[list[Instruction], list[Instruction], list[Instruction]]:
    """Schedule materialized QK with V transpose, softmax, and PV phases."""
    seq_pad = gemma_seq_pad(seq_len)

    qk_m = _ceil_div(seq_pad, GEMMA_QK_TILE_M)
    qk_n = _ceil_div(seq_pad, GEMMA_QK_TILE_N)
    qk = [
        I.GemmaAttnQk(
            layer_idx=layer_idx,
            tile_m=m,
            tile_n=n,
            head_idx=h,
            seq_len=seq_len,
            kind=0,
        )
        for h in range(C.num_heads)
        for m in range(qk_m)
        for n in range(qk_n)
    ]
    vt_tiles = (C.head_dim // GEMMA_VT_TILE) * (seq_pad // GEMMA_VT_TILE)
    qk += [
        I.GemmaAttnQk(
            layer_idx=layer_idx,
            tile_m=t,
            seq_len=seq_len,
            kind=1,
        )
        for t in range(vt_tiles)
    ]

    # One warp owns a row; keep the phase within one grid round.
    rows_total = C.num_heads * seq_pad
    rows_per = GEMMA_SOFTMAX_WARPS * _ceil_div(
        rows_total, C.gemma_queues * GEMMA_SOFTMAX_WARPS
    )
    softmax = [
        I.GemmaAttnSoftmax(
            layer_idx=layer_idx,
            row0=r,
            num_rows=min(rows_per, rows_total - r),
        )
        for r in range(0, rows_total, rows_per)
    ]

    pv_m = _ceil_div(seq_pad, GEMMA_PV_TILE_M)
    pv_n = _ceil_div(C.head_dim, GEMMA_PV_TILE_N)
    pv = [
        I.GemmaAttnPv(layer_idx=layer_idx, tile_m=m, tile_n=n, head_idx=h)
        for h in range(C.num_heads)
        for m in range(pv_m)
        for n in range(pv_n)
    ]
    return (
        _distribute_gemma(qk, sm_count),
        _distribute_gemma(softmax, sm_count),
        _distribute_gemma(pv, sm_count),
    )


def _dep(inst: Instruction, dep_layer: int, dep_row: int, dep_expected: int) -> Instruction:
    """Rewrite dependency fields on a copied instruction."""
    fields = {f.name: getattr(inst, f.name) for f in inst.__dataclass_fields__.values()}
    fields["dep_layer"] = dep_layer
    fields["dep_row"] = dep_row
    fields["dep_expected"] = dep_expected
    return type(inst)(**fields)


def pick_down_splits(seq_len: int, cap: int = GEMMA_DOWN_SPLITS_COMPILED) -> int:
    n = _ceil_div(seq_len, GEMMA_TILE_M) * (C.encoder_width // GEMMA_TILE_N)
    s = max(1, min(cap, C.sm_count // n))
    while s > 1 and (256 // s) % 4:
        s -= 1
    return s


def schedule_gemma_layer(
    seq_len: int,
    layer_idx: int = 0,
    *,
    prev_down: tuple[int, int] | None = None,
    kv_only: bool = False,
    down_splits: int | None = None,
    sm_count: int | None = None,
) -> tuple[list[Instruction], dict[str, int]]:
    """Schedule one shape-specialized Gemma encoder layer."""
    m_tiles = _ceil_div(seq_len, GEMMA_TILE_M)
    if down_splits is None:
        down_splits = pick_down_splits(seq_len)
    out: list[Instruction] = []
    counts: dict[str, int] = {}

    def phase(name: str, tiles: Sequence[Instruction]) -> None:
        out.extend(tiles)
        counts[name] = len(tiles)

    def rmsn_attn(dep_layer: int, dep_row: int, dep_expected: int) -> tuple[tuple[int, int, int], int]:
        tiles = [
            I.GemmaRmsnAttn(
                layer_idx=layer_idx,
                token_row=t0,
                dep_layer=dep_layer,
                dep_row=dep_row,
                dep_expected=dep_expected,
                num_tokens=min(GEMMA_LN_BATCH, seq_len - t0),
            )
            for t0 in range(0, seq_len, GEMMA_LN_BATCH)
        ]
        phase("rmsn_attn", tiles)
        return (
            layer_idx,
            I.gemma_barrier_row(I.GemmaRmsnAttn.opcode()),
            counts["rmsn_attn"],
        ), 0

    if prev_down is None:
        qkv_dep, qkv_sel = rmsn_attn(0, 0, 0)
    else:
        # Keep RMS normalization as an explicit phase at every layer boundary.
        # A folded "next RMS" epilogue would require different Down opcode
        # semantics and scratch publication; neither the L2 graph nor this ABI
        # promises one.
        prev_row, prev_cnt = prev_down
        qkv_dep, qkv_sel = rmsn_attn(layer_idx - 1, prev_row, prev_cnt)

    qkv_tiles = [
        I.GemmaQkvRope(
            layer_idx=layer_idx,
            tile_m=m,
            tile_n=nn,
            dep_layer=qkv_dep[0],
            dep_row=qkv_dep[1],
            dep_expected=qkv_dep[2],
            norm_sel=qkv_sel,
        )
        for m in range(m_tiles)
        for nn in range(C.encoder_qkv_out // GEMMA_QKV_TILE_N)
    ]
    phase("qkv", qkv_tiles)

    if kv_only:
        return _distribute_gemma(out, sm_count), counts

    qk, softmax, pv = schedule_gemma_attn(layer_idx, seq_len, sm_count=1)
    qk = [
        _dep(
            i,
            layer_idx,
            I.gemma_barrier_row(I.GemmaQkvRope.opcode()),
            counts["qkv"],
        )
        for i in qk
    ]
    phase("attn_qk", qk)
    softmax = [
        _dep(
            i,
            layer_idx,
            I.gemma_barrier_row(I.GemmaAttnQk.opcode()),
            counts["attn_qk"],
        )
        for i in softmax
    ]
    phase("attn_sm", softmax)
    pv = [
        _dep(
            i,
            layer_idx,
            I.gemma_barrier_row(I.GemmaAttnSoftmax.opcode()),
            counts["attn_sm"],
        )
        for i in pv
    ]
    phase("attn_pv", pv)

    oproj = [
        I.GemmaOproj(
            layer_idx=layer_idx,
            tile_m=m,
            tile_n=nn,
            dep_layer=layer_idx,
            dep_row=I.gemma_barrier_row(I.GemmaAttnPv.opcode()),
            dep_expected=counts["attn_pv"],
        )
        for m in range(m_tiles)
        for nn in range(C.encoder_width // GEMMA_TILE_N)
    ]
    phase("oproj", oproj)

    rmsn_ffn = [
        I.GemmaRmsnFfn(
            layer_idx=layer_idx,
            token_row=t0,
            dep_layer=layer_idx,
            dep_row=I.gemma_barrier_row(I.GemmaOproj.opcode()),
            dep_expected=counts["oproj"],
            num_tokens=min(GEMMA_LN_BATCH, seq_len - t0),
        )
        for t0 in range(0, seq_len, GEMMA_LN_BATCH)
    ]
    phase("rmsn_ffn", rmsn_ffn)

    ffn_tiles = [
        I.GemmaGate(
            layer_idx=layer_idx,
            tile_m=m,
            tile_n=nn,
            dep_layer=layer_idx,
            dep_row=I.gemma_barrier_row(I.GemmaRmsnFfn.opcode()),
            dep_expected=counts["rmsn_ffn"],
            norm_sel=1,
        )
        for nn in range(C.encoder_mlp_width // GEMMA_TILE_N)
        for m in range(m_tiles)
    ]
    phase("ffn", ffn_tiles)
    down_ns = C.encoder_width // GEMMA_TILE_N

    if down_splits <= 1:
        down = [
            I.GemmaDown(
                layer_idx=layer_idx,
                tile_m=m,
                tile_n=nn,
                dep_layer=layer_idx,
                dep_row=I.gemma_barrier_row(I.GemmaGate.opcode()),
                dep_expected=counts["ffn"],
            )
            for m in range(m_tiles)
            for nn in range(down_ns)
        ]
        phase("down", down)
        counts["_down_row"] = I.gemma_barrier_row(I.GemmaDown.opcode())
    else:
        down = [
            I.GemmaDownSplitK(
                layer_idx=layer_idx,
                tile_m=m,
                tile_n=nn,
                k_split=sp,
                dep_layer=layer_idx,
                dep_row=I.gemma_barrier_row(I.GemmaGate.opcode()),
                dep_expected=counts["ffn"],
            )
            for m in range(m_tiles)
            for nn in range(down_ns)
            for sp in range(down_splits)
        ]
        phase("down", down)
        counts["_down_row"] = I.gemma_barrier_row(I.GemmaDownSplitK.opcode())

    return _distribute_gemma(out, sm_count), counts


def schedule_gemma_encoder(
    seq_len: int,
    n_layers: int,
    *,
    kv_only_last: bool = False,
    sm_count: int | None = None,
) -> list[Instruction]:
    out: list[Instruction] = []
    prev: tuple[int, int] | None = None
    for layer in range(n_layers):
        kv_only = kv_only_last and layer == n_layers - 1
        tiles, counts = schedule_gemma_layer(
            seq_len, layer_idx=layer, prev_down=prev, kv_only=kv_only, sm_count=1
        )
        out.extend(tiles)
        prev = None if kv_only else (counts["_down_row"], counts["down"])
    return _distribute_gemma(out, sm_count)


# ── Action expert ───────────────────────────────────────────────────────────

AE_RMSN_PER = 1
# Output columns per decoder-layer GEMM instruction; this must equal
# `AeDecoderGemm::BN` in ae_gemm.cuh.
AE_QKV_TILE_N = 32
# QKV partitions the padded row extent into two logical tiles.
AE_QKV_TILE_M = 32
AE_OPROJ_TILE_N = 32
AE_FFN_TILE_N = 32
AE_DOWN_TILE_N = 32
AE_K_SPLIT = 256
AE_OPROJ_SPLITS = C.encoder_width // AE_K_SPLIT  # 8
AE_DOWN_SPLITS = C.decoder_mlp_width // AE_K_SPLIT  # 16
AE_OPROJ_SEGMENTS = 2
AE_DOWN_SEGMENTS = 4
AE_RED_COL_BLKS = 8
AE_KV_BLKS_PER_SPAN = 2
AE_ACTION_TILE_N = 32
AE_ACT_MLP_BIAS_ROW = 10
AE_ACT_OUT_BIAS_ROW = 11
AE_ACT_OUT_NORM_SEL = 2
AE_BARRIER_SLOT = 0

AeBarrierKey = tuple[int, int, int]


def _ae_publish(
    accounting: dict[AeBarrierKey, int],
    *,
    layer: int,
    opcode: int,
    count: int,
) -> tuple[int, int, int]:
    """Advance one slot-0 AE barrier epoch and return its dependency triple."""
    row = I.ae_barrier_row(opcode)
    key = (layer, row, AE_BARRIER_SLOT)
    accounting[key] = accounting.get(key, 0) + count
    return layer, row, accounting[key]


def prefix_splits_for(prefix_len: int, per_span: int = AE_KV_BLKS_PER_SPAN) -> int:
    n_blk = _ceil_div(prefix_len, C.ae_kv_block)
    return max(1, _ceil_div(n_blk, per_span))


def ae_attn_spans(
    prefix_len: int,
    *,
    seq: int | None = None,
    prefix_splits: int | None = None,
) -> list[tuple[int, int, int]]:
    prefix_splits = prefix_splits_for(prefix_len) if prefix_splits is None else prefix_splits
    n_blk = _ceil_div(prefix_len, C.ae_kv_block)
    per = _ceil_div(n_blk, prefix_splits)
    spans, b = [], 0
    while b < n_blk:
        nb = min(per, n_blk - b)
        spans.append((b, nb, prefix_len))
        b += nb
    kv_row_blk = _ceil_div(prefix_len, C.ae_kv_block)
    seq = C.ae_seq_for(C.production_chunk) if seq is None else seq
    self_end = kv_row_blk * C.ae_kv_block + seq
    spans.append((kv_row_blk, 1, self_end))
    return spans


def schedule_ae_layer(
    layer_idx: int,
    layout: dict[str, int],
    *,
    dep: tuple[int, int, int] = (0, 0, 0),
    seen: dict[AeBarrierKey, int] | None = None,
    sm_count: int | None = None,
) -> tuple[list[Instruction], tuple[int, int, int]]:
    """Schedule one shape-specialized action-expert layer."""
    seen = {} if seen is None else seen
    prefix_splits = prefix_splits_for(layout["prefix_len"])
    spans = ae_attn_spans(layout["prefix_len"], seq=layout["seq"], prefix_splits=prefix_splits)
    out: list[Instruction] = []
    dep_layer, dep_row, dep_expected = dep

    def emit(insts: Sequence[Instruction], opcode: int) -> None:
        nonlocal dep_layer, dep_row, dep_expected
        out.extend(insts)
        dep_layer, dep_row, dep_expected = _ae_publish(
            seen,
            layer=layer_idx,
            opcode=opcode,
            count=len(insts),
        )

    emit(
        [
            I.AeRmsnAttn(
                plane=0,
                token_row=row,
                dep_layer=dep_layer,
                dep_row=dep_row,
                dep_expected=dep_expected,
                bar_layer=layer_idx,
                num_tokens=min(AE_RMSN_PER, layout["m_pad"] - row),
            )
            for row in range(0, layout["m_pad"], AE_RMSN_PER)
        ],
        I.AeRmsnAttn.opcode(),
    )
    emit(
        [
            I.AeQkvRope(
                plane=layer_idx,
                local_row=m,
                local_col=n,
                dep_layer=dep_layer,
                dep_row=dep_row,
                dep_expected=dep_expected,
                bar_layer=layer_idx,
                kv_row_blk=layout["kv_row_blk"],
                norm_sel=0,
            )
            for m in range(layout["m_pad"] // AE_QKV_TILE_M)
            for n in range((C.num_heads + 2) * C.head_dim // AE_QKV_TILE_N)
        ],
        I.AeQkvRope.opcode(),
    )
    partial = [
        I.AeAttnPartial(
            layer_idx=layer_idx,
            head_idx=h,
            partial_idx=si,
            dep_layer=dep_layer,
            dep_row=dep_row,
            dep_expected=dep_expected,
            kv_blk0=b0,
            kv_nblk=nblk,
            seq_len=span_end,
            prefix_len=layout["prefix_len"],
            state_key=layout["state_key"],
        )
        for si, (b0, nblk, span_end) in enumerate(spans)
        for h in range(C.num_heads)
    ]
    emit(partial, I.AeAttnPartial.opcode())
    emit(
        [
            I.AeAttnReduce(
                layer_idx=layer_idx,
                head_idx=h,
                col_blk=col_blk,
                dep_layer=dep_layer,
                dep_row=dep_row,
                dep_expected=dep_expected,
                n_splits=len(spans),
            )
            for h in range(C.num_heads)
            for col_blk in range(AE_RED_COL_BLKS)
        ],
        I.AeAttnReduce.opcode(),
    )
    emit(
        [
            I.AeOproj(
                plane=layer_idx * AE_OPROJ_SPLITS + split,
                local_row=split,
                local_col=n,
                dep_layer=dep_layer,
                dep_row=dep_row,
                dep_expected=dep_expected,
                bar_layer=layer_idx,
                out_row=0,
                out_col=n,
            )
            for split in range(0, AE_OPROJ_SPLITS, AE_OPROJ_SEGMENTS)
            for n in range(C.decoder_width // AE_OPROJ_TILE_N)
        ],
        I.AeOproj.opcode(),
    )
    emit(
        [
            I.AeResidualReduce(
                local_col=n,
                dep_layer=dep_layer,
                dep_row=dep_row,
                dep_expected=dep_expected,
                bar_layer=layer_idx,
            )
            for n in range(C.decoder_width // AE_OPROJ_TILE_N)
        ],
        I.AeResidualReduce.opcode(),
    )
    emit(
        [
            I.AeRmsnFfn(
                plane=0,
                token_row=row,
                dep_layer=dep_layer,
                dep_row=dep_row,
                dep_expected=dep_expected,
                bar_layer=layer_idx,
                num_tokens=min(AE_RMSN_PER, layout["m_pad"] - row),
            )
            for row in range(0, layout["m_pad"], AE_RMSN_PER)
        ],
        I.AeRmsnFfn.opcode(),
    )
    emit(
        [
            I.AeFfnDual(
                plane=layer_idx,
                local_row=0,
                local_col=n,
                dep_layer=dep_layer,
                dep_row=dep_row,
                dep_expected=dep_expected,
                bar_layer=layer_idx,
                out_row=n // (AE_K_SPLIT // AE_FFN_TILE_N),
                out_col=n % (AE_K_SPLIT // AE_FFN_TILE_N),
                norm_sel=1,
            )
            for n in range(C.decoder_mlp_width // AE_FFN_TILE_N)
        ],
        I.AeFfnDual.opcode(),
    )
    emit(
        [
            I.AeDown(
                plane=layer_idx * AE_DOWN_SPLITS + split,
                local_row=split,
                local_col=n,
                dep_layer=dep_layer,
                dep_row=dep_row,
                dep_expected=dep_expected,
                bar_layer=layer_idx,
                out_row=0,
                out_col=n,
            )
            for split in range(0, AE_DOWN_SPLITS, AE_DOWN_SEGMENTS)
            for n in range(C.decoder_width // AE_DOWN_TILE_N)
        ],
        I.AeDown.opcode(),
    )
    emit(
        [
            I.AeResidualReduce(
                local_col=n,
                dep_layer=dep_layer,
                dep_row=dep_row,
                dep_expected=dep_expected,
                bar_layer=layer_idx,
            )
            for n in range(C.decoder_width // AE_DOWN_TILE_N)
        ],
        I.AeResidualReduce.opcode(),
    )
    return _distribute_ae(out, sm_count), (dep_layer, dep_row, dep_expected)


def schedule_ae_stack(
    n_layers: int,
    layout: dict[str, int],
    *,
    n_steps: int = 1,
    sm_count: int | None = None,
) -> list[Instruction]:
    seen: dict[AeBarrierKey, int] = {}
    work: list[Instruction] = []
    dep = (0, 0, 0)
    for _ in range(n_steps):
        for layer in range(n_layers):
            insts, dep = schedule_ae_layer(layer, layout, dep=dep, seen=seen, sm_count=1)
            work.extend(insts)
    return _distribute_ae(work, sm_count)


def _schedule_ae_action_projection(
    instruction_type: type[I.AeActIn] | type[I.AeActMlp] | type[I.AeActOut],
    *,
    n_out: int,
    bias_row: int,
    dep: tuple[int, int, int],
    norm_sel: int = 0,
    n_valid_row: int = 0,
) -> list[Instruction]:
    if n_out <= 0 or n_out % AE_ACTION_TILE_N:
        raise ValueError(
            f"AE action projection width must be a positive multiple of {AE_ACTION_TILE_N}, "
            f"got {n_out}"
        )
    dep_layer, dep_row, dep_expected = dep
    return [
        instruction_type(
            plane=0,
            local_row=0,
            local_col=col,
            dep_layer=dep_layer,
            dep_row=dep_row,
            dep_expected=dep_expected,
            bar_layer=0,
            out_row=0,
            out_col=col,
            bias_row=bias_row,
            norm_sel=norm_sel,
            n_valid_row=n_valid_row,
        )
        for col in range(n_out // AE_ACTION_TILE_N)
    ]


def _validate_canonical_ae_flow_layout(layout: dict[str, int]) -> None:
    num_views = layout.get("num_views")
    chunk = layout.get("chunk")
    prompt_len = layout.get("prompt_len")
    if type(num_views) is not int or num_views not in C.VIEWS_CHOICES:
        raise ValueError(f"AE flow num_views must be one of {C.VIEWS_CHOICES}, got {num_views!r}")
    if type(chunk) is not int or chunk not in C.CHUNK_CHOICES:
        raise ValueError(f"AE flow chunk must be one of {C.CHUNK_CHOICES}, got {chunk!r}")
    if type(prompt_len) is not int or prompt_len <= 0:
        raise ValueError(f"AE flow prompt_len must be a positive integer, got {prompt_len!r}")

    expected = C.ae_kv_layout_for(
        num_views=num_views,
        chunk=chunk,
        prompt_len=prompt_len,
    )
    if set(layout) != set(expected):
        missing = sorted(set(expected).difference(layout))
        extra = sorted(set(layout).difference(expected))
        raise ValueError(
            f"AE flow layout keys must be canonical; missing={missing}, extra={extra}"
        )
    for name, expected_value in expected.items():
        if layout[name] != expected_value:
            raise ValueError(
                f"AE flow layout {name}={layout[name]!r}; expected canonical {expected_value!r}"
            )


def schedule_ae_flow(
    layout: dict[str, int],
    *,
    sm_count: int | None = None,
) -> list[Instruction]:
    """Canonical 10-step Pi0 AE flow schedule, including action-head opcodes 11--14.

    The complete instruction stream is constructed before launch. ``Bar`` must be
    zeroed exactly once before that launch; the slot-0 accounting is cumulative across
    all ten flow steps and is deliberately never reset between steps. Opcodes
    11--14 are part of the same device instruction stream.
    """
    _validate_canonical_ae_flow_layout(layout)

    accounting: dict[AeBarrierKey, int] = {}
    work: list[Instruction] = []
    dep = (0, I.ae_barrier_row(I.AeActOut.opcode()), 0)

    def emit(insts: Sequence[Instruction], *, layer: int, opcode: int) -> None:
        nonlocal dep
        work.extend(insts)
        dep = _ae_publish(accounting, layer=layer, opcode=opcode, count=len(insts))

    for step in range(C.num_flow_steps):
        emit(
            _schedule_ae_action_projection(
                I.AeActIn,
                n_out=C.decoder_width,
                bias_row=step,
                dep=dep,
            ),
            layer=0,
            opcode=I.AeActIn.opcode(),
        )
        emit(
            _schedule_ae_action_projection(
                I.AeActMlp,
                n_out=C.decoder_width,
                bias_row=AE_ACT_MLP_BIAS_ROW,
                dep=dep,
                n_valid_row=layout["seq"],
            ),
            layer=0,
            opcode=I.AeActMlp.opcode(),
        )
        for layer in range(C.decoder_layers):
            insts, dep = schedule_ae_layer(
                layer,
                layout,
                dep=dep,
                seen=accounting,
                sm_count=1,
            )
            work.extend(insts)

        emit(
            [
                I.AeRmsnOut(
                    plane=0,
                    token_row=row,
                    dep_layer=dep[0],
                    dep_row=dep[1],
                    dep_expected=dep[2],
                    bar_layer=0,
                    num_tokens=min(AE_RMSN_PER, layout["m_pad"] - row),
                )
                for row in range(0, layout["m_pad"], AE_RMSN_PER)
            ],
            layer=0,
            opcode=I.AeRmsnOut.opcode(),
        )
        emit(
            _schedule_ae_action_projection(
                I.AeActOut,
                n_out=C.action_dim,
                bias_row=AE_ACT_OUT_BIAS_ROW,
                dep=dep,
                norm_sel=AE_ACT_OUT_NORM_SEL,
            ),
            layer=0,
            opcode=I.AeActOut.opcode(),
        )
    return _distribute_ae(work, sm_count)


__all__ = [
    "pack_instructions",
    "schedule_siglip_ln_attn",
    "schedule_siglip_ln_ffn",
    "schedule_siglip_qkv",
    "schedule_siglip_attn",
    "schedule_siglip_oproj",
    "schedule_siglip_fc1",
    "schedule_siglip_fc2",
    "schedule_siglip_layer",
    "schedule_siglip_stack",
    "kv_spans",
    "schedule_gemma_attn",
    "schedule_gemma_layer",
    "schedule_gemma_encoder",
    "prefix_splits_for",
    "ae_attn_spans",
    "schedule_ae_layer",
    "schedule_ae_stack",
    "schedule_ae_flow",
]
