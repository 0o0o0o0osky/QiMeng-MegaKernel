"""S0-owned Action Expert baseline used by the S0 and S1 comparisons."""
from __future__ import annotations

import torch
from torch import Tensor

from qimeng_megakernel.models.VLA.pi.pi0.A100.schedule.pi0_config import Pi0A100Config
from .gemma_ops import apply_rope_interleaved


def gelu_tanh(x: Tensor) -> Tensor:
    return torch.nn.functional.gelu(x, approximate="tanh")


def baseline_ae_attn(
    q_all: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    prefix_len: int,
    state_key: int,
    self_end: int,
    m_pad: int | None = None,
    mask_state_row: bool = True,
    num_heads: int | None = None,
    head_dim: int | None = None,
) -> Tensor:
    """AE attention reference implementing the route's prefix and self mask.

    Mask (when ``mask_state_row``):
      row 0 (state):  key < prefix_len  OR  key == state_key
      rows 1..:       key < prefix_len  OR  state_key <= key < self_end
    """
    num_heads = Pi0A100Config.num_heads if num_heads is None else num_heads
    head_dim = Pi0A100Config.head_dim if head_dim is None else head_dim
    m_pad = Pi0A100Config.ae_m if m_pad is None else m_pad
    q = q_all[:, : num_heads * head_dim].float().view(m_pad, num_heads, head_dim)
    kf, vf = k.float(), v.float()
    idx = torch.arange(kf.shape[0], device=q_all.device)
    vis = (idx < prefix_len) | ((idx >= state_key) & (idx < self_end))
    m = vis.unsqueeze(0).expand(m_pad, -1).clone()
    if mask_state_row:
        m[0] = (idx < prefix_len) | (idx == state_key)
    out = torch.zeros(m_pad, num_heads, head_dim, device=q_all.device, dtype=torch.float32)
    scale = head_dim ** -0.5
    for h in range(num_heads):
        logits = (q[:, h] @ kf.T) * scale
        out[:, h] = torch.softmax(logits.masked_fill(~m, float("-inf")), dim=-1) @ vf
    return out  # [M, H, D]


def baseline_ae_layer(
    x_acc: Tensor,
    weights: dict[str, Tensor],
    layer: int,
    rope_table: Tensor,
    k_cache: Tensor,
    v_cache: Tensor,
    *,
    layout: dict[str, int] | None = None,
    m_pad: int | None = None,
    kv_row0: int | None = None,
    prefix_len: int | None = None,
    state_key: int | None = None,
    self_end: int | None = None,
    eps: float | None = None,
    mask_state_row: bool = True,
) -> Tensor:
    """One AE layer; updates ``k_cache``/``v_cache`` in place (same as kernel).

    Prefer passing ``layout=Pi0A100Config.ae_kv_layout_for(views, chunk)``; the
    explicit kwargs override layout fields when set.
    """
    if layout is None:
        layout = Pi0A100Config.ae_kv_layout_for()
    m_pad = layout["m_pad"] if m_pad is None else m_pad
    kv_row0 = layout["state_key"] if kv_row0 is None else kv_row0
    prefix_len = layout["prefix_len"] if prefix_len is None else prefix_len
    state_key = layout["state_key"] if state_key is None else state_key
    self_end = layout["self_end"] if self_end is None else self_end
    eps = Pi0A100Config.rms_eps if eps is None else eps
    heads = Pi0A100Config.num_heads
    head_dim = Pi0A100Config.head_dim
    qw, kw = heads * head_dim, head_dim
    attn_in = heads * head_dim

    inv0 = torch.rsqrt((x_acc ** 2).mean(-1) + eps)
    xb = x_acc.bfloat16().float()
    y = inv0[:, None] * (xb @ weights["qkv"][layer].float().T)
    q = apply_rope_interleaved(y[:, :qw], rope_table, head_dim=head_dim)
    k = apply_rope_interleaved(y[:, qw:qw + kw], rope_table, head_dim=head_dim)
    v = y[:, qw + kw:]
    k_cache[kv_row0:kv_row0 + m_pad] = k.bfloat16()
    v_cache[kv_row0:kv_row0 + m_pad] = v.bfloat16()

    a = baseline_ae_attn(
        q.bfloat16(),
        k_cache,
        v_cache,
        prefix_len=prefix_len,
        state_key=state_key,
        self_end=self_end,
        m_pad=m_pad,
        mask_state_row=mask_state_row,
    )
    a = a.reshape(m_pad, attn_in).bfloat16().float()
    x_acc = x_acc + a @ weights["oproj"][layer].float().T

    inv1 = torch.rsqrt((x_acc ** 2).mean(-1) + eps)
    xb = x_acc.bfloat16().float()
    gate = gelu_tanh(inv1[:, None] * (xb @ weights["gate"][layer].float().T))
    h = (gate * (inv1[:, None] * (xb @ weights["up"][layer].float().T))).bfloat16().float()
    return x_acc + h @ weights["down"][layer].float().T


def baseline_ae_stack(
    x0: Tensor,
    weights: dict[str, list[Tensor]],
    rope_table: Tensor,
    prefix_kv: list[tuple[Tensor, Tensor]],
    *,
    layout: dict[str, int] | None = None,
    n_layers: int | None = None,
    mask_state_row: bool = True,
) -> Tensor:
    """Multi-layer AE stack; ``prefix_kv[L]`` seeds encoder prefix into each layer cache."""
    if layout is None:
        layout = Pi0A100Config.ae_kv_layout_for()
    n_layers = len(weights["qkv"]) if n_layers is None else n_layers
    x = x0.float()
    for L in range(n_layers):
        pk, pv = prefix_kv[L]
        kc = torch.zeros(layout["kv_rows"], Pi0A100Config.head_dim, device=x.device, dtype=torch.bfloat16)
        vc = torch.zeros_like(kc)
        plen = layout["prefix_len"]
        kc[:plen] = pk[:plen].to(torch.bfloat16)
        vc[:plen] = pv[:plen].to(torch.bfloat16)
        x = baseline_ae_layer(
            x,
            weights,
            L,
            rope_table,
            kc,
            vc,
            layout=layout,
            mask_state_row=mask_state_row,
        )
    return x
