"""Action Expert and flow-decoder fused PyTorch references."""
from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from ..pi0_config import Pi0A100Config
from .gemma_ops import apply_rope_interleaved


def gelu_tanh(x: Tensor) -> Tensor:
    return torch.nn.functional.gelu(x, approximate="tanh")


def fused_ae_act_in(noise: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
    """ACT_IN reference, including the CUDA epilogue's bf16 landing."""
    pre = noise.float() @ weight.float().T + bias.float()
    return (pre / (1.0 + torch.exp(-pre))).to(torch.bfloat16)


def fused_ae_act_mlp(
    act_h: Tensor,
    weight: Tensor,
    bias: Tensor,
    state_h: Tensor,
    *,
    n_valid_row: int,
) -> Tensor:
    """ACT_MLP reference with fp32 output, state row override, and pad clearing."""
    out = act_h.float() @ weight.float().T + bias.float()
    out[0] = state_h.float()
    out[n_valid_row:] = 0.0
    return out


def fused_ae_rmsn_landing(
    x_acc: Tensor,
    *,
    eps: float | None = None,
) -> tuple[Tensor, Tensor]:
    """RMSN reference returning the bf16 GEMM landing and fp32 row factors."""
    eps = Pi0A100Config.rms_eps if eps is None else eps
    inv = torch.rsqrt((x_acc.float() ** 2).mean(-1) + eps)
    return x_acc.to(torch.bfloat16), inv


def fused_ae_rms_linear(
    x_acc: Tensor,
    weight: Tensor,
    *,
    eps: float | None = None,
) -> Tensor:
    """SM80 AE projection order: BF16 residual landing, GEMM, then FP32 RMS scale.

    Scaling after the dot product is algebraically equivalent before rounding,
    but is deliberately not equivalent to materializing a BF16 normalized
    activation. This is the Ampere action-projection numerical semantics.
    """
    landed, inv = fused_ae_rmsn_landing(x_acc, eps=eps)
    return inv[:, None] * (landed.float() @ weight.float().T)


def fused_ae_linear_with_factor(x: Tensor, weight: Tensor, factor: Tensor) -> Tensor:
    return factor.float()[:, None] * (x.float() @ weight.float().T)


def pack_k_split_major(x: Tensor, *, segment: int = 256) -> Tensor:
    """Pack `[rows,K]` into `[splits*rows,segment]` physical scratch."""
    if x.ndim != 2 or x.shape[1] % segment:
        raise ValueError(f"cannot K-split activation {tuple(x.shape)} by segment={segment}")
    rows, splits = x.shape[0], x.shape[1] // segment
    return x.reshape(rows, splits, segment).permute(1, 0, 2).reshape(splits * rows, segment).contiguous()


def unpack_k_split_major(x: Tensor, *, splits: int, segment: int = 256) -> Tensor:
    """Unpack `[splits*rows,segment]` physical scratch into `[rows,K]`."""
    if x.ndim != 2 or x.shape[0] % splits or x.shape[1] != segment:
        raise ValueError(
            f"cannot unpack K-split activation {tuple(x.shape)} with splits={splits}, segment={segment}"
        )
    rows = x.shape[0] // splits
    return x.reshape(splits, rows, segment).permute(1, 0, 2).reshape(rows, splits * segment).contiguous()


def merge_k_split_weight(planes: Tensor, *, logical_k: int) -> Tensor:
    """Physical `[splits,N,Kseg]` planes to output-major `[N,logical_k]`."""
    if planes.ndim != 3:
        raise ValueError(f"split weight must be rank 3, got {tuple(planes.shape)}")
    merged = planes.permute(1, 0, 2).reshape(planes.shape[1], -1)
    if logical_k > merged.shape[1]:
        raise ValueError(f"logical K={logical_k} exceeds packed K={merged.shape[1]}")
    return merged[:, :logical_k].contiguous()


def pack_attention_head_major(attention: Tensor, *, head_dim: int) -> Tensor:
    """Dense `[rows,heads*D]` to `[heads*rows,D]`."""
    if attention.ndim != 2 or attention.shape[1] % head_dim:
        raise ValueError(f"invalid dense attention shape {tuple(attention.shape)}")
    rows, heads = attention.shape[0], attention.shape[1] // head_dim
    return attention.reshape(rows, heads, head_dim).permute(1, 0, 2).reshape(heads * rows, head_dim).contiguous()


def unpack_attention_head_major(attention: Tensor, *, rows: int) -> Tensor:
    """Head-major `[heads*rows,D]` to dense `[rows,heads*D]`."""
    if attention.ndim != 2 or attention.shape[0] % rows:
        raise ValueError(f"invalid head-major attention shape {tuple(attention.shape)} for rows={rows}")
    heads, head_dim = attention.shape[0] // rows, attention.shape[1]
    return attention.reshape(heads, rows, head_dim).permute(1, 0, 2).reshape(rows, heads * head_dim).contiguous()


def fused_ae_oproj_residual(residual: Tensor, attention: Tensor, weight: Tensor) -> Tensor:
    return residual.float() + attention.float() @ weight.float().T


def fused_ae_gate(x_acc: Tensor, weight: Tensor) -> Tensor:
    return gelu_tanh(fused_ae_rms_linear(x_acc, weight))


def fused_ae_gate_with_factor(x: Tensor, weight: Tensor, factor: Tensor) -> Tensor:
    """Gate projection consuming the RMS phase's landed input and factor."""
    return gelu_tanh(fused_ae_linear_with_factor(x, weight, factor))


def fused_ae_up_mul(x_acc: Tensor, weight: Tensor, gate: Tensor) -> Tensor:
    return (gate.float() * fused_ae_rms_linear(x_acc, weight)).to(torch.bfloat16)


def fused_ae_up_mul_with_factor(
    x: Tensor,
    weight: Tensor,
    factor: Tensor,
    gate: Tensor,
) -> Tensor:
    """Up projection consuming the RMS phase and a materialized gate."""
    return (
        gate.float() * fused_ae_linear_with_factor(x, weight, factor)
    ).to(torch.bfloat16)


def fused_ae_ffn_dual(x_acc: Tensor, gate_weight: Tensor, up_weight: Tensor) -> Tensor:
    return fused_ae_up_mul(x_acc, up_weight, fused_ae_gate(x_acc, gate_weight))


def fused_ae_ffn_dual_with_factor(
    x: Tensor,
    gate_weight: Tensor,
    up_weight: Tensor,
    factor: Tensor,
) -> Tensor:
    """Dual gate/up projection consuming one scheduler-published RMS phase."""
    gate = fused_ae_gate_with_factor(x, gate_weight, factor)
    return fused_ae_up_mul_with_factor(x, up_weight, factor, gate)


def fused_ae_down_residual(residual: Tensor, hidden: Tensor, weight: Tensor) -> Tensor:
    return residual.float() + hidden.float() @ weight.float().T


def fused_ae_act_out(
    noise: Tensor,
    x: Tensor,
    weight: Tensor,
    bias: Tensor,
    rms: Tensor,
) -> Tensor:
    """ACT_OUT reference with bf16 GEMM landing followed by bf16 store-add."""
    delta = rms.float()[:, None] * (x.float() @ weight.float().T) + bias.float()
    delta_landed = delta.to(torch.bfloat16)
    return (noise.float() + delta_landed.float()).to(torch.bfloat16)


def fused_ae_attn(
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
    """AE attention reference matching ``test_ae_ops.attn_ref`` (r-vla mask).

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


def fused_ae_layer(
    x_acc: Tensor,
    weights: dict[str, list[Tensor]],
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

    y = fused_ae_rms_linear(x_acc, weights["qkv"][layer], eps=eps)
    q = apply_rope_interleaved(y[:, :qw], rope_table, head_dim=head_dim)
    k = apply_rope_interleaved(y[:, qw:qw + kw], rope_table, head_dim=head_dim)
    v = y[:, qw + kw:]
    k_cache[kv_row0:kv_row0 + m_pad] = k.bfloat16()
    v_cache[kv_row0:kv_row0 + m_pad] = v.bfloat16()

    a = fused_ae_attn(
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

    gate = gelu_tanh(fused_ae_rms_linear(x_acc, weights["gate"][layer], eps=eps))
    up = fused_ae_rms_linear(x_acc, weights["up"][layer], eps=eps)
    h = (gate * up).bfloat16().float()
    return x_acc + h @ weights["down"][layer].float().T


def fused_ae_stack(
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
        x = fused_ae_layer(
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


def ae_layout(**kwargs: Any) -> dict[str, int]:
    """Shortcut to ``Pi0A100Config.ae_kv_layout_for``."""
    return Pi0A100Config.ae_kv_layout_for(**kwargs)


def fused_flow_step(
    x_acc: Tensor,
    weights: dict[str, list[Tensor]],
    rope_table: Tensor,
    prefix_kv: list[tuple[Tensor, Tensor]],
    *,
    layout: dict[str, int] | None = None,
    n_layers: int | None = None,
) -> Tensor:
    """One fused flow denoiser step over the action-expert stack."""
    return fused_ae_stack(
        x_acc,
        weights,
        rope_table,
        prefix_kv,
        layout=layout,
        n_layers=n_layers,
    )
