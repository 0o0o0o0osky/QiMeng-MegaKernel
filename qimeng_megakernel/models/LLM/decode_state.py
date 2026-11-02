"""Shared scalar and per-row state used by LLM decode schedules."""
from __future__ import annotations

from typing import Any

import torch


def batch_size(globs: Any) -> int:
    hidden_states = getattr(globs, "hidden_states", None)
    if hidden_states is None:
        raise ValueError("globs.hidden_states is required")
    return 1 if hidden_states.ndim == 1 else int(hidden_states.shape[0])


def decode_pos_id(globs: Any, batch_idx: int) -> int:
    position_ids = getattr(globs, "position_ids", None)
    if position_ids is not None:
        return int(position_ids[batch_idx].item())
    position_id = getattr(globs, "position_id", None)
    return int(position_id) if position_id is not None else 0


def row_active(globs: Any, batch_idx: int) -> bool:
    active_mask = getattr(globs, "active_mask", None)
    if active_mask is None:
        return True
    return bool(active_mask[batch_idx].item())


def kv_length(globs: Any, batch_idx: int) -> int:
    kv_lengths = getattr(globs, "kv_lengths", None)
    if kv_lengths is not None:
        return int(kv_lengths[batch_idx].item())
    return decode_pos_id(globs, batch_idx) + 1


def instruction_active(is_active: int, globs: Any, batch_idx: int) -> bool:
    if is_active == 0:
        return False
    if is_active == 1:
        return True
    return row_active(globs, batch_idx)


def ensure_decode_state(
    globs: Any,
    batch_size: int,
    *,
    device: torch.device,
) -> None:
    """Initialize the per-row fields used when decode has more than one row."""
    if batch_size < 2:
        return
    if (
        getattr(globs, "active_mask", None) is not None
        and getattr(globs, "kv_lengths", None) is not None
        and getattr(globs, "eos_reached", None) is not None
        and getattr(globs, "last_generated_tokens", None) is not None
        and int(globs.active_mask.shape[0]) == batch_size
    ):
        return
    if getattr(globs, "active_mask", None) is None:
        globs.active_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
    if getattr(globs, "kv_lengths", None) is None:
        position_ids = getattr(globs, "position_ids", None)
        if position_ids is not None:
            globs.kv_lengths = (position_ids.to(device=device) + 1).to(dtype=torch.long)
        else:
            position_id = getattr(globs, "position_id", None)
            position_id = 0 if position_id is None else int(position_id)
            globs.kv_lengths = torch.full(
                (batch_size,), position_id + 1, dtype=torch.long, device=device
            )
    if getattr(globs, "eos_reached", None) is None:
        globs.eos_reached = torch.zeros(batch_size, dtype=torch.bool, device=device)
    if getattr(globs, "last_generated_tokens", None) is None:
        globs.last_generated_tokens = torch.zeros(
            batch_size, dtype=torch.long, device=device
        )


def sync_kv_lengths(globs: Any) -> None:
    position_ids = getattr(globs, "position_ids", None)
    if position_ids is None:
        return
    batch_size = int(position_ids.shape[0])
    device = position_ids.device
    if (
        getattr(globs, "kv_lengths", None) is None
        or int(globs.kv_lengths.shape[0]) != batch_size
    ):
        globs.kv_lengths = torch.empty(batch_size, dtype=torch.long, device=device)
    torch.add(position_ids, 1, out=globs.kv_lengths)


def advance_decode_state(
    globs: Any,
    next_token_ids: torch.Tensor,
    eos_token_ids: list[int],
    eos_token_ids_tensor: torch.Tensor | None = None,
) -> None:
    """Advance per-row positions and EOS state after one decode step."""
    hidden_states = getattr(globs, "hidden_states", None)
    if hidden_states is None or hidden_states.ndim != 2:
        return
    batch_size = int(hidden_states.shape[0])
    if batch_size < 2 or getattr(globs, "position_ids", None) is None:
        return

    device = globs.position_ids.device
    ensure_decode_state(globs, batch_size, device=device)
    next_token_ids = next_token_ids.view(-1)
    if next_token_ids.device != device or next_token_ids.dtype != torch.long:
        next_token_ids = next_token_ids.to(device=device, dtype=torch.long)
    globs.last_generated_tokens.copy_(next_token_ids)

    active_mask = getattr(globs, "active_mask", None)
    if active_mask is None:
        return
    if eos_token_ids_tensor is not None:
        eos_ids = eos_token_ids_tensor
        if eos_ids.device != device or eos_ids.dtype != torch.long:
            eos_ids = eos_ids.to(device=device, dtype=torch.long)
        reached_eos = (next_token_ids.unsqueeze(1) == eos_ids.unsqueeze(0)).any(dim=1)
    elif eos_token_ids:
        eos_ids = torch.tensor(eos_token_ids, dtype=torch.long, device=device)
        reached_eos = (next_token_ids.unsqueeze(1) == eos_ids.unsqueeze(0)).any(dim=1)
    else:
        reached_eos = torch.zeros(batch_size, dtype=torch.bool, device=device)

    newly_finished = active_mask & reached_eos
    still_active = active_mask & ~reached_eos
    globs.eos_reached.logical_or_(newly_finished)
    globs.active_mask.copy_(still_active)
    globs.position_ids.add_(still_active.to(dtype=globs.position_ids.dtype))
    sync_kv_lengths(globs)


__all__ = [
    "advance_decode_state",
    "batch_size",
    "decode_pos_id",
    "ensure_decode_state",
    "instruction_active",
    "kv_length",
    "row_active",
    "sync_kv_lengths",
]
