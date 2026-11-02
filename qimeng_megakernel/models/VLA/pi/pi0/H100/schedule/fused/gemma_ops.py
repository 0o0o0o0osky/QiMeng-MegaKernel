"""Gemma encoder fused torch references (full layer / stack / MQA, no tiling).

Authoritative math from ``pyvm_to_megakernel/test_gemma_ops.py``:
  ``ref_mqa``, ``ref_gemma_layer``, ``_apply_rope_ref``, multi-layer stack.
"""
from __future__ import annotations

import torch
from torch import Tensor

from ..pi0_config import Pi0H100Config


def rms_inv(x: Tensor, eps: float | None = None) -> Tensor:
    eps = Pi0H100Config.rms_eps if eps is None else eps
    return torch.rsqrt((x.float() ** 2).mean(-1, keepdim=True) + eps)


def make_rope_table(
    rows: int,
    *,
    head_dim: int | None = None,
    theta: float | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> Tensor:
    """``[rows, head_dim]`` with cos on even / sin on odd (H100 / A100 layout)."""
    head_dim = Pi0H100Config.head_dim if head_dim is None else head_dim
    theta = Pi0H100Config.rope_theta if theta is None else theta
    pos = torch.arange(rows, device=device, dtype=torch.float32).unsqueeze(1)
    i = torch.arange(head_dim // 2, device=device, dtype=torch.float32).unsqueeze(0)
    ang = pos * (theta ** (-2.0 * i / head_dim))
    tbl = torch.zeros(rows, head_dim, device=device, dtype=torch.float32)
    tbl[:, 0::2] = torch.cos(ang)
    tbl[:, 1::2] = torch.sin(ang)
    return tbl.to(dtype)


def apply_rope_interleaved(
    x: Tensor,
    rope_table: Tensor,
    *,
    head_dim: int | None = None,
) -> Tensor:
    """Interleaved RoPE over every ``head_dim`` segment (``_apply_rope_ref`` / AE ``rope_ref``).

    ``rope_table`` is ``[S, head_dim]`` (or at least ``S`` rows); the same cos/sin
    pair is reused for each head along the feature axis of ``x``.
    """
    head_dim = Pi0H100Config.head_dim if head_dim is None else head_dim
    y = x.float()
    t = rope_table[: y.shape[0]].float()
    n_heads = y.shape[1] // head_dim
    c = t[:, 0::2].repeat(1, n_heads)
    s = t[:, 1::2].repeat(1, n_heads)
    even, odd = y[:, 0::2], y[:, 1::2]
    out = torch.empty_like(y)
    out[:, 0::2] = even * c - odd * s
    out[:, 1::2] = even * s + odd * c
    return out.to(dtype=x.dtype)


def fused_mqa_bidirectional(
    q_all: Tensor,
    k: Tensor,
    v: Tensor,
    seq_len: int,
    *,
    num_heads: int | None = None,
    head_dim: int | None = None,
) -> Tensor:
    """MQA 8:1, bidirectional; keys only up to ``seq_len`` (pad keys excluded)."""
    num_heads = Pi0H100Config.num_heads if num_heads is None else num_heads
    head_dim = Pi0H100Config.head_dim if head_dim is None else head_dim
    rows = q_all.shape[0]
    out = torch.zeros(rows, num_heads * head_dim, device=q_all.device, dtype=torch.float32)
    scale = head_dim ** -0.5
    kf, vf = k[:seq_len].float(), v[:seq_len].float()
    for h in range(num_heads):
        qh = q_all[:, h * head_dim:(h + 1) * head_dim].float()
        p = torch.softmax(qh @ kf.T * scale, dim=-1)
        out[:, h * head_dim:(h + 1) * head_dim] = p @ vf
    return out


def fused_qkv_rope(
    x: Tensor,
    wqkv: Tensor,
    rope_table: Tensor,
    *,
    eps: float | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Fused-side RMS → QKV → interleaved RoPE helper."""
    heads = Pi0H100Config.num_heads
    head_dim = Pi0H100Config.head_dim
    eps = Pi0H100Config.rms_eps if eps is None else eps
    xf = x.float()
    y = torch.rsqrt((xf ** 2).mean(-1, keepdim=True) + eps) * (xf @ wqkv.float().T)
    qw, kw = heads * head_dim, head_dim
    y = y.clone()
    y[:, :qw + kw] = apply_rope_interleaved(y[:, :qw + kw], rope_table, head_dim=head_dim)
    return y[:, :qw], y[:, qw : qw + kw], y[:, qw + kw :]


def fused_oproj_residual(x: Tensor, attn: Tensor, wo: Tensor) -> Tensor:
    """Fused-side attention output projection and residual."""
    return x.float() + attn.float() @ wo.float().T


def fused_ffn_gate_up_down(
    x: Tensor,
    wg: Tensor,
    wu: Tensor,
    wd: Tensor,
    *,
    eps: float | None = None,
) -> Tensor:
    """Fused-side RMS-gated FFN and residual."""
    eps = Pi0H100Config.rms_eps if eps is None else eps
    xf = x.float()
    inv = torch.rsqrt((xf ** 2).mean(-1, keepdim=True) + eps)
    gate = torch.nn.functional.gelu(inv * (xf @ wg.float().T), approximate="tanh")
    up = gate * (inv * (xf @ wu.float().T))
    return xf + up.to(torch.bfloat16).float() @ wd.float().T


def fused_gemma_layer(
    x: Tensor,
    wqkv: Tensor,
    wo: Tensor,
    wg: Tensor,
    wu: Tensor,
    wd: Tensor,
    rope_table: Tensor,
    seq_len: int,
    *,
    eps: float | None = None,
) -> Tensor:
    """One Gemma encoder layer; weights already folded as (1+w) where applicable."""
    q, k, v = fused_qkv_rope(x, wqkv, rope_table, eps=eps)
    a = fused_mqa_bidirectional(
        q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16), seq_len
    )
    return fused_ffn_gate_up_down(fused_oproj_residual(x, a, wo), wg, wu, wd, eps=eps)


def fused_gemma_stack(
    x: Tensor,
    layer_weights: list[tuple[Tensor, Tensor, Tensor, Tensor, Tensor]],
    rope_table: Tensor,
    seq_len: int,
    *,
    eps: float | None = None,
) -> Tensor:
    """Stack ``fused_gemma_layer``; each entry is ``(wqkv, wo, wg, wu, wd)``."""
    h = x
    for wqkv, wo, wg, wu, wd in layer_weights:
        h = fused_gemma_layer(
            h.to(torch.bfloat16) if h.dtype != torch.bfloat16 else h,
            wqkv,
            wo,
            wg,
            wu,
            wd,
            rope_table,
            seq_len,
            eps=eps,
        )
    return h
