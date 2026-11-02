"""Barrier pre-fill so a single op can be exercised in isolation.

Every PyVM instruction group asserts that the *previous* op's barrier row is already
counted up.  A test that wants to run one op on its own (layer 0, where there is no
previous layer to have set it) therefore has to fill those counters by hand first.

The counts are not arbitrary: each is "how many tiles the producing op emits", i.e. a
function of the model's `head_dim` / `hidden_size` / `intermediate_size` and the
scheduler's `TILE_N_GRANULARITY`.  That is the whole per-model difference, which is why
this module takes a config object and a tile granularity and hands back the fills --
the fifteen tree-local `_barrier_fill.py` copies differed *only* in which config module
they imported.

Two shapes are covered:

`make_llama_six_op_fills`
    The 6-opcode decode layer (RMS_QKVProj_GEMV_RoPE, SplitKV, Reduction,
    OProj_GEMV_Residual, RMS_UpGateProj_GEMV_SiLU, DownProj_GEMV_Residual).
    Barrier row index = ``prev_opcode() - 1`` (see ``python_vm._prev_barrier_row``):

    | Op / instruction group     | prev_opcode  | row | Fill for layer 0                |
    |----------------------------|--------------|-----|---------------------------------|
    | RMS_QKVProj_GEMV_RoPE      | 6 (DownProj) | 5   | (layer_idx>0 only; L0 no check) |
    | DecodeAttentionSplitKV     | 1 (QKV)      | 0   | barriers[0,0,:] = blocks_per_head |
    | DecodeAttentionReduction   | 2 (SplitKV)  | 1   | (filled by SplitKV)             |
    | OProj_GEMV_Residual        | 3 (Reduction)| 2   | barriers[0,2,0] = num_heads     |
    | RMS_UpGateProj_GEMV_SiLU   | 4 (OProj)    | 3   | barriers[0,3,0] = hidden_blocks |
    | DownProj_GEMV_Residual     | 5 (UpGate)   | 4   | barriers[0,4,0] = intermediate_blocks |

`make_splitkv_row0_fill`
    Row 0 only, in the wider layout the Qwen schedulers use, where
    DecodeAttentionSplitKV checks three bands:
    ``[0, n_heads)``, ``[n_heads + kv]``, and ``[n_heads + n_kv + kv]``.
    Pre-filling it lets SplitKV run without first running PyVM's rms_qkv -- which is
    what makes position_id=1 style MK-vs-PyVM LSE comparisons possible.

Config attributes are read **inside** the returned closures, never captured at build
time: some suites mutate their config class from the loaded HF config during
`real_data` setup, and a captured value would silently be the pre-mutation one.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol


class _LayerConfig(Protocol):
    """The attributes a fill needs; every model's decode config class already has them."""

    head_dim: int
    num_attention_heads: int
    num_key_value_heads: int
    hidden_size: int
    intermediate_size: int


@dataclass(frozen=True)
class LlamaSixOpFills:
    """The four fills of the 6-opcode decode layer, bound to one model config."""

    after_qkv: Callable[[Any], None]
    after_attention: Callable[[Any], None]
    after_o_proj: Callable[[Any], None]
    after_upgate: Callable[[Any], None]


def make_llama_six_op_fills(
    config: Any,
    tile_n_granularity: int,
    *,
    upgate_bucket_counts: tuple[int, ...] | None = None,
) -> LlamaSixOpFills:
    """Build the layer-0 barrier fills for a 6-opcode decode layer.

    ``upgate_bucket_counts`` describes a bucketed UpGate→DownProj edge.  It
    may be ragged (for example Llama3B/H100 uses ``(192, 192, 128)``).  When it
    is omitted, legacy trees retain their single device-wide slot-0 fill.
    """

    def barrier_fill_after_qkv_layer0(globs: Any) -> None:
        """So DecodeAttentionSplitKV / Reduction can run (QKV done)."""
        # Each attention head requires blocks_per_head tiles from RMS_QKVProj_GEMV_RoPE.
        blocks_per_head = config.head_dim // tile_n_granularity
        globs.barriers[0, 0, :] = blocks_per_head

    def barrier_fill_after_attention_layer0(globs: Any) -> None:
        """So OProj_GEMV_Residual can run (DecodeAttentionReduction done)."""
        globs.barriers[0, 2, 0] = config.num_attention_heads

    def barrier_fill_after_o_proj_layer0(globs: Any) -> None:
        """So RMS_UpGateProj_GEMV_SiLU can run (OProj done)."""
        # One barrier count per hidden block produced by OProj_GEMV_Residual.
        hidden_blocks = config.hidden_size // tile_n_granularity
        globs.barriers[0, 3, 0] = hidden_blocks

    def barrier_fill_after_upgate_layer0(globs: Any) -> None:
        """So DownProj_GEMV_Residual can run (UpGate done)."""
        # One arrival per intermediate block produced by RMS_UpGateProj_GEMV_SiLU.
        intermediate_blocks = config.intermediate_size // tile_n_granularity
        if upgate_bucket_counts is None:
            globs.barriers[0, 4, 0] = intermediate_blocks
            return
        if any(count <= 0 for count in upgate_bucket_counts):
            raise ValueError("upgate bucket counts must all be positive")
        if sum(upgate_bucket_counts) != intermediate_blocks:
            raise ValueError(
                "upgate bucket counts must conserve every intermediate block: "
                f"sum={sum(upgate_bucket_counts)}, expected={intermediate_blocks}"
            )
        globs.barriers[0, 4, : len(upgate_bucket_counts)] = globs.barriers.new_tensor(
            upgate_bucket_counts
        )

    return LlamaSixOpFills(
        after_qkv=barrier_fill_after_qkv_layer0,
        after_attention=barrier_fill_after_attention_layer0,
        after_o_proj=barrier_fill_after_o_proj_layer0,
        after_upgate=barrier_fill_after_upgate_layer0,
    )


#: Barrier row 0 = prev_opcode(DecodeAttentionSplitKV) - 1 = 1 - 1 = 0.
BARRIER_ROW_AFTER_QKV = 0


def make_splitkv_row0_fill(config: Any, tile_n_granularity: int) -> Callable[..., None]:
    """Build the row-0 fill that satisfies DecodeAttentionSplitKV's three-band check."""

    def barrier_fill_after_qkv_layer0(globs: Any, layer_idx: int = 0) -> None:
        """Fill row 0 so DecodeAttentionSplitKV's assertions pass.

        Lets a test run splitkv alone, or compare MK rms_qkv against PyVM splitkv.
        """
        blocks_per_head = config.head_dim // tile_n_granularity
        n_heads = config.num_attention_heads
        n_kv = config.num_key_value_heads
        row = BARRIER_ROW_AFTER_QKV
        # [0..n_heads)
        globs.barriers[layer_idx, row, 0:n_heads] = blocks_per_head
        # [n_heads + kv_head_idx] for each kv_head
        for kv in range(n_kv):
            globs.barriers[layer_idx, row, n_heads + kv] = blocks_per_head
        # [n_heads + n_kv + kv_head_idx] for each kv_head
        for kv in range(n_kv):
            globs.barriers[layer_idx, row, n_heads + n_kv + kv] = blocks_per_head

    return barrier_fill_after_qkv_layer0
