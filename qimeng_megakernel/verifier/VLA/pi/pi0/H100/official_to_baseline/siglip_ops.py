"""S0-owned SigLIP baseline used by the S0 and S1 comparisons.

Naive baseline math (independent of schedule.fused); seeded from verifier:
  ``pyvm_to_megakernel/test_siglip_{layer,attn,gemm,stack}.py`` and
  the stage comparisons.
"""
from __future__ import annotations

import torch
from torch import Tensor

from qimeng_megakernel.models.VLA.pi.pi0.H100.schedule.pi0_config import Pi0H100Config


def gelu_tanh(x: Tensor) -> Tensor:
    return torch.nn.functional.gelu(x, approximate="tanh")


def layernorm(
    x: Tensor,
    gamma: Tensor,
    beta: Tensor,
    eps: float | None = None,
) -> Tensor:
    """fp64 LN matching ``test_siglip_layer.reference_layer`` / gemm helpers."""
    eps = Pi0H100Config.ln_eps if eps is None else eps
    x64 = x.double()
    mean = x64.mean(-1, keepdim=True)
    var = ((x64 - mean) ** 2).mean(-1, keepdim=True)
    y = (x64 - mean) * torch.rsqrt(var + eps) * gamma.double() + beta.double()
    return y.to(dtype=x.dtype)

def baseline_siglip_attention(
    qkv: Tensor,
    *,
    num_views: int = 3,
    tokens_per_view: int = 256,
    num_heads: int = 16,
    head_dim: int = 72,
) -> Tensor:
    """Non-causal per-view MHA from packed dense QKV ``[S, 3*H*D]`` (no head pad)."""
    seq = qkv.shape[0]
    h_dim = num_heads * head_dim
    q, k, v = qkv[:, :h_dim], qkv[:, h_dim:2 * h_dim], qkv[:, 2 * h_dim:3 * h_dim]
    out = torch.zeros(seq, h_dim, device=qkv.device, dtype=torch.float32)
    scale = head_dim ** -0.5
    for vw in range(num_views):
        sl = slice(vw * tokens_per_view, (vw + 1) * tokens_per_view)
        for h in range(num_heads):
            hs = slice(h * head_dim, (h + 1) * head_dim)
            qh = q[sl, hs].float()
            kh = k[sl, hs].float()
            vh = v[sl, hs].float()
            out[sl, hs] = torch.softmax(qh @ kh.T * scale, dim=-1) @ vh
    return out.to(dtype=qkv.dtype)


def baseline_siglip_attention_head_padded(
    qkv: Tensor,
    *,
    num_views: int = 3,
    tokens_per_view: int = 256,
    num_heads: int = 16,
    head_dim: int = 72,
    head_dim_pad: int = 80,
) -> Tensor:
    """Same as ``test_siglip_attn.ref_attention``: QKV/out use head-padded layout."""
    seq = qkv.shape[0]
    out = torch.zeros(seq, num_heads * head_dim_pad, device=qkv.device, dtype=torch.float32)
    scale = head_dim ** -0.5
    for v in range(num_views):
        sl = slice(v * tokens_per_view, (v + 1) * tokens_per_view)
        for h in range(num_heads):
            qo = h * head_dim_pad
            ko = num_heads * head_dim_pad + h * head_dim_pad
            vo = 2 * num_heads * head_dim_pad + h * head_dim_pad
            q = qkv[sl, qo:qo + head_dim].float()
            k = qkv[sl, ko:ko + head_dim].float()
            val = qkv[sl, vo:vo + head_dim].float()
            out[sl, qo:qo + head_dim] = torch.softmax(q @ k.T * scale, dim=-1) @ val
    return out


def baseline_gemm(
    a: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    n_valid: int,
    k_valid: int,
    *,
    act: str | None = None,
    residual: Tensor | None = None,
) -> Tensor:
    """Full GEMM reference used by SigLIP GEMM opcode tests (``ref_gemm``)."""
    y = a[:n_valid, :k_valid].float() @ weight[: weight.shape[0], :k_valid].float().T
    if bias is not None:
        y = y + bias[: y.shape[1]].float()
    if act == "gelu_tanh":
        y = gelu_tanh(y)
    if residual is not None:
        y = residual[:n_valid].float() + y
    return y


def baseline_siglip_layer(
    x0: Tensor,
    weights: dict[str, Tensor],
    *,
    num_views: int = 3,
    tokens_per_view: int | None = None,
    num_heads: int | None = None,
    head_dim: int | None = None,
    ln_eps: float | None = None,
) -> Tensor:
    """One SigLIP encoder block in fp64 math (``reference_layer``).

    Expected ``weights`` keys (fp32/fp64 ok):
      g1, b1, Wqkv, bqkv, Wo, bo, g2, b2, W1, b1f, W2, b2f
    """
    tokens_per_view = (
        Pi0H100Config.vision_tokens_per_view if tokens_per_view is None else tokens_per_view
    )
    num_heads = Pi0H100Config.vision_num_heads if num_heads is None else num_heads
    head_dim = Pi0H100Config.siglip_head_dim if head_dim is None else head_dim
    ln_eps = Pi0H100Config.ln_eps if ln_eps is None else ln_eps

    x = x0.double()
    n1 = layernorm(x, weights["g1"], weights["b1"], eps=ln_eps)
    qkv = n1 @ weights["Wqkv"].double().T + weights["bqkv"].double()
    attn = baseline_siglip_attention(
        qkv.float(),
        num_views=num_views,
        tokens_per_view=tokens_per_view,
        num_heads=num_heads,
        head_dim=head_dim,
    ).double()
    x1 = x + (attn @ weights["Wo"].double().T + weights["bo"].double())
    n2 = layernorm(x1, weights["g2"], weights["b2"], eps=ln_eps)
    h1 = gelu_tanh(n2 @ weights["W1"].double().T + weights["b1f"].double())
    x2 = x1 + (h1 @ weights["W2"].double().T + weights["b2f"].double())
    return x2.float()


def baseline_siglip_stack(
    x0: Tensor,
    layer_weights: list[dict[str, Tensor]],
    *,
    num_views: int = 3,
    tokens_per_view: int | None = None,
    num_heads: int | None = None,
    head_dim: int | None = None,
    ln_eps: float | None = None,
) -> Tensor:
    """Stack ``baseline_siglip_layer`` for multilayer / tower verification."""
    x = x0
    for w in layer_weights:
        x = baseline_siglip_layer(
            x,
            w,
            num_views=num_views,
            tokens_per_view=tokens_per_view,
            num_heads=num_heads,
            head_dim=head_dim,
            ln_eps=ln_eps,
        )
    return x


# API aliases (plan frozen names)
attention_per_view = baseline_siglip_attention
encoder_layer = baseline_siglip_layer
stack = baseline_siglip_stack


def mlp_gelu_tanh(
    x: Tensor,
    w1: Tensor,
    b1: Tensor,
    w2: Tensor,
    b2: Tensor,
) -> Tensor:
    h = gelu_tanh(x.float() @ w1.float().T + b1.float())
    return (h @ w2.float().T + b2.float()).to(dtype=x.dtype)


def patch_embed_pos(
    images_nchw: Tensor,
    patch_w: Tensor,
    patch_b: Tensor,
    pos_emb: Tensor,
) -> Tensor:
    """``images`` [V,3,H,W] or [V,H,W,3]; patch_w OpenPI [out,in] flat or [C_out,3,14,14].

    Returns concatenated tokens ``[V*256, width]``.
    """
    if images_nchw.dim() == 4 and images_nchw.shape[-1] == 3:
        images_nchw = images_nchw.permute(0, 3, 1, 2).contiguous()
    v = images_nchw.shape[0]
    if patch_w.dim() == 4:
        # [out, 3, 14, 14] — non-overlapping patches (stride = kernel)
        kh, kw = int(patch_w.shape[-2]), int(patch_w.shape[-1])
        y = torch.nn.functional.conv2d(
            images_nchw.float(), patch_w.float(), patch_b.float(), stride=(kh, kw)
        )
        # [V, width, H', W'] -> [V, H'*W', width]
        y = y.flatten(2).transpose(1, 2)
    else:
        # flat [width, 14*14*3] applied per patch — rare path
        raise NotImplementedError("flat patch weight path; use 4D patch_w")
    y = y + pos_emb.float().unsqueeze(0)
    return y.reshape(v * y.shape[1], y.shape[2]).to(dtype=images_nchw.dtype)


def final_norm(x: Tensor, gamma: Tensor, beta: Tensor, eps: float | None = None) -> Tensor:
    return layernorm(x, gamma, beta, eps=eps)


def projector(x: Tensor, w: Tensor, b: Tensor | None = None) -> Tensor:
    y = x.float() @ w.float().T
    if b is not None:
        y = y + b.float()
    return y.to(dtype=x.dtype)
