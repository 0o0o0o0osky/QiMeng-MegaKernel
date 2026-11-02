"""Host-side prep for megakernel batch control tensors (position_ids, masks).

Shared by production ``mk_kernel.call_mk`` and verifier benches. Only call
``.to`` / ``.contiguous`` when device, dtype, or layout actually need fixing.
"""
from __future__ import annotations

import torch


def _move_position_ids(t: torch.Tensor, dev: torch.device) -> torch.Tensor:
    if t.device != dev or t.dtype != torch.int32:
        t = t.to(device=dev, dtype=torch.int32)
    if not t.is_contiguous():
        t = t.contiguous()
    return t


def _move_mask_like(t: torch.Tensor, dev: torch.device) -> torch.Tensor:
    if t.device != dev:
        t = t.to(device=dev)
    if t.dtype not in (torch.uint8, torch.bool):
        t = t.to(dtype=torch.uint8)
    if not t.is_contiguous():
        t = t.contiguous()
    return t


def prepare_mk_batch_control_tensors(
    dev: torch.device,
    batch_size: int,
    globs,
    *,
    always_partial_valid: bool = False,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    position_ids_arg = None
    pid = getattr(globs, "position_ids", None)
    if pid is not None and pid.numel() == batch_size:
        position_ids_arg = _move_position_ids(pid, dev)

    active_mask_arg = None
    am = getattr(globs, "active_mask", None)
    if am is not None and am.numel() == batch_size:
        active_mask_arg = _move_mask_like(am, dev)

    attn_partial_valid_arg = None
    apv = getattr(globs, "attn_partial_valid", None)
    if always_partial_valid and apv is None:
        raise AttributeError("globs.attn_partial_valid is required")
    if apv is not None:
        attn_partial_valid_arg = _move_mask_like(apv, dev)

    return position_ids_arg, active_mask_arg, attn_partial_valid_arg


def position_id_int_from_globs(globs) -> int:
    pos = getattr(globs, "position_id", None)
    if pos is None:
        return 0
    if isinstance(pos, torch.Tensor):
        if pos.numel() != 1:
            raise ValueError("globs.position_id must be scalar")
        return int(pos.item())
    return int(pos)
