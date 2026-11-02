"""H100 multi-batch state with a fixed B<=16 production capacity."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ...single_batch.decode_schedule.globs import Globals as _BaseGlobals

BATCH_CAPACITY = 16
MAX_BATCH_SIZE = BATCH_CAPACITY
DOWN_SPLIT_K = 4
NUM_LAYERS = 16
HIDDEN_SIZE = 2048
INTERMEDIATE_SIZE = 8192
NUM_KV_HEADS = 8
HEAD_DIM = 64
VOCAB_SIZE = 128256
MK_BARRIER_SHAPE = (NUM_LAYERS, 10, 48)


@dataclass
class Globals(_BaseGlobals):
    """Logical B views plus contiguous B16 storage for the H100 VM kernel."""

    position_ids: torch.Tensor | None = None
    kv_lengths: torch.Tensor | None = None
    active_mask: torch.Tensor | None = None
    eos_reached: torch.Tensor | None = None
    last_generated_tokens: torch.Tensor | None = None
    eos_token_id: int | None = None
    attn_partial_valid: torch.Tensor | None = None
    attn_rms_out: torch.Tensor | None = None
    mlp_rms_out: torch.Tensor | None = None
    lm_head_rms_out: torch.Tensor | None = None

    logical_batch_size: int = 0
    batch_capacity: int = BATCH_CAPACITY
    hidden_states_backing: torch.Tensor | None = None
    q_post_rope_backing: torch.Tensor | None = None
    attn_out_backing: torch.Tensor | None = None
    silu_out_backing: torch.Tensor | None = None
    logits_backing: torch.Tensor | None = None
    attn_rms_out_backing: torch.Tensor | None = None
    mlp_rms_out_backing: torch.Tensor | None = None
    lm_head_rms_out_backing: torch.Tensor | None = None
    k_cache_decode: torch.Tensor | None = None
    v_cache_decode: torch.Tensor | None = None
    down_partials: torch.Tensor | None = None
    mk_barriers: torch.Tensor | None = None

    def validate_padded_storage(self) -> None:
        """Fail closed unless logical tensors alias contiguous B16 backings."""
        batch_size = int(self.logical_batch_size)
        if not 2 <= batch_size <= MAX_BATCH_SIZE:
            raise ValueError(
                f"logical_batch_size must be in [2, {MAX_BATCH_SIZE}], got {batch_size}"
            )
        if int(self.batch_capacity) != BATCH_CAPACITY:
            raise ValueError(
                f"batch_capacity must be {BATCH_CAPACITY}, got {self.batch_capacity}"
            )
        pairs = (
            ("hidden_states", "hidden_states_backing", HIDDEN_SIZE),
            ("q_post_rope", "q_post_rope_backing", HIDDEN_SIZE),
            ("attn_out", "attn_out_backing", HIDDEN_SIZE),
            ("silu_out", "silu_out_backing", INTERMEDIATE_SIZE),
            ("logits", "logits_backing", VOCAB_SIZE),
            ("attn_rms_out", "attn_rms_out_backing", HIDDEN_SIZE),
            ("mlp_rms_out", "mlp_rms_out_backing", HIDDEN_SIZE),
            ("lm_head_rms_out", "lm_head_rms_out_backing", HIDDEN_SIZE),
        )
        for logical_name, backing_name, width in pairs:
            logical = getattr(self, logical_name)
            backing = getattr(self, backing_name)
            if logical is None or backing is None:
                raise ValueError(f"{logical_name}/{backing_name} must both be allocated")
            if tuple(backing.shape) != (BATCH_CAPACITY, width):
                raise ValueError(
                    f"{backing_name} must have shape {(BATCH_CAPACITY, width)}"
                )
            if tuple(logical.shape) != (batch_size, width):
                raise ValueError(f"{logical_name} shape does not match logical rows")
            if logical.data_ptr() != backing.data_ptr():
                raise ValueError(f"{logical_name} must alias the start of {backing_name}")
            if not logical.is_contiguous() or not backing.is_contiguous():
                raise ValueError(f"{logical_name}/{backing_name} must be contiguous")
            if logical.dtype != torch.bfloat16 or backing.dtype != torch.bfloat16:
                raise ValueError(f"{logical_name}/{backing_name} must be bfloat16")

        if self.k_cache is None or self.v_cache is None:
            raise ValueError("k_cache/v_cache must both be allocated")
        if self.k_cache.ndim != 5:
            raise ValueError("k_cache/v_cache must be rank-5")
        max_seq_len = int(self.k_cache.shape[2])
        if max_seq_len <= 0:
            raise ValueError("k_cache/v_cache sequence capacity must be positive")
        expected_primary_kv_shape = (
            NUM_LAYERS,
            batch_size,
            max_seq_len,
            NUM_KV_HEADS,
            HEAD_DIM,
        )
        for name in ("k_cache", "v_cache"):
            cache = getattr(self, name)
            if cache is None or tuple(cache.shape) != expected_primary_kv_shape:
                raise ValueError(f"{name} must have shape {expected_primary_kv_shape}")
            if cache.dtype != torch.bfloat16 or not cache.is_contiguous():
                raise ValueError(f"{name} must be contiguous bfloat16")

        if self.down_partials is None or self.down_partials.shape != (
            DOWN_SPLIT_K,
            BATCH_CAPACITY,
            HIDDEN_SIZE,
        ):
            raise ValueError(
                "down_partials must have shape "
                f"{(DOWN_SPLIT_K, BATCH_CAPACITY, HIDDEN_SIZE)}"
            )
        if (
            self.down_partials.dtype != torch.float32
            or not self.down_partials.is_contiguous()
        ):
            raise ValueError("down_partials must be contiguous float32")
        expected_kv_shape = (
            NUM_LAYERS,
            BATCH_CAPACITY,
            NUM_KV_HEADS,
            max_seq_len,
            HEAD_DIM,
        )
        for name in ("k_cache_decode", "v_cache_decode"):
            cache = getattr(self, name)
            if cache is None or cache.shape != expected_kv_shape:
                raise ValueError(f"{name} must have shape {expected_kv_shape}")
            if cache.dtype != self.k_cache.dtype or not cache.is_contiguous():
                raise ValueError(f"{name} must be contiguous and match KV dtype")
        if self.mk_barriers is None or self.mk_barriers.shape != MK_BARRIER_SHAPE:
            raise ValueError(f"mk_barriers must have shape {MK_BARRIER_SHAPE}")
        if (
            self.mk_barriers.dtype != torch.int32
            or not self.mk_barriers.is_contiguous()
        ):
            raise ValueError("mk_barriers must be contiguous int32")
