"""Split-attention math shared by the Gemma and Action Expert PyVM handlers."""
from __future__ import annotations

import torch
from torch import Tensor


def split_attention_partial(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    scale: float,
    mask: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return normalized partial output plus row max and unnormalized row sum."""
    logits = q.float() @ k.float().T * scale
    if mask is not None:
        if mask.shape != logits.shape or mask.dtype != torch.bool:
            raise ValueError(
                f"attention mask must be bool {tuple(logits.shape)}, got "
                f"{mask.dtype} {tuple(mask.shape)}"
            )
        logits = logits.masked_fill(~mask, float("-inf"))
    row_max = logits.max(dim=-1).values
    if not torch.isfinite(row_max).all():
        raise ValueError("every split-attention query row must see at least one key")
    probabilities = torch.exp(logits - row_max[:, None])
    row_sum = probabilities.sum(dim=-1)
    output = (probabilities @ v.float()) / row_sum[:, None]
    return output, row_max, row_sum


def reduce_attention_partials(
    partials: Tensor,
    maxima: Tensor,
    sums: Tensor,
) -> Tensor:
    """Merge independently normalized softmax spans with the stable LSE formula."""
    if partials.ndim != 3:
        raise ValueError(f"partials must be [splits,rows,dim], got {tuple(partials.shape)}")
    expected = partials.shape[:2]
    if maxima.shape != expected or sums.shape != expected:
        raise ValueError(
            f"attention statistics must be {tuple(expected)}, got "
            f"{tuple(maxima.shape)} and {tuple(sums.shape)}"
        )
    global_max = maxima.float().max(dim=0).values
    weights = sums.float() * torch.exp(maxima.float() - global_max[None, :])
    denominator = weights.sum(dim=0)
    if not torch.isfinite(denominator).all() or not torch.all(denominator > 0):
        raise ValueError("split-attention reduction denominator must be positive and finite")
    return (weights[:, :, None] * partials.float()).sum(dim=0) / denominator[:, None]


def ae_split_attention_partial(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    key_start: int,
    prefix_len: int,
    state_key: int,
    visible_end: int,
    scale: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """One AE prefix/self attention span, including the state-row mask."""
    key_index = torch.arange(
        key_start, key_start + k.shape[0], device=q.device
    )
    visible = (key_index < prefix_len) | (
        (key_index >= state_key) & (key_index < visible_end)
    )
    mask = visible.unsqueeze(0).expand(q.shape[0], -1).clone()
    mask[0] = (key_index < prefix_len) | (key_index == state_key)
    return split_attention_partial(q, k, v, scale=scale, mask=mask)


__all__ = [
    "ae_split_attention_partial",
    "reduce_attention_partials",
    "split_attention_partial",
]
