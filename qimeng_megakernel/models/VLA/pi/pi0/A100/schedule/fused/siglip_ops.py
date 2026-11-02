"""SigLIP So400m fused PyTorch references without physical tiling."""
from __future__ import annotations

import torch
from torch import Tensor

from ..pi0_config import Pi0A100Config


def gelu_tanh(x: Tensor) -> Tensor:
    return torch.nn.functional.gelu(x, approximate="tanh")


def layernorm(
    x: Tensor,
    gamma: Tensor,
    beta: Tensor,
    eps: float | None = None,
) -> Tensor:
    """fp64 LN matching ``test_siglip_layer.reference_layer`` / gemm helpers."""
    eps = Pi0A100Config.ln_eps if eps is None else eps
    x64 = x.double()
    mean = x64.mean(-1, keepdim=True)
    var = ((x64 - mean) ** 2).mean(-1, keepdim=True)
    y = (x64 - mean) * torch.rsqrt(var + eps) * gamma.double() + beta.double()
    return y.to(dtype=x.dtype)


def fused_siglip_attention(
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


def fused_siglip_attention_head_padded(
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


def pad_qkv_heads(
    qkv_dense: Tensor,
    *,
    num_heads: int | None = None,
    head_dim: int | None = None,
    head_dim_pad: int | None = None,
) -> Tensor:
    """Pack dense Q/K/V heads into the physical 80-wide SigLIP head layout."""
    num_heads = Pi0A100Config.vision_num_heads if num_heads is None else num_heads
    head_dim = Pi0A100Config.siglip_head_dim if head_dim is None else head_dim
    head_dim_pad = Pi0A100Config.siglip_head_dim_pad if head_dim_pad is None else head_dim_pad
    expected = 3 * num_heads * head_dim
    if qkv_dense.ndim != 2 or qkv_dense.shape[1] != expected:
        raise ValueError(f"dense SigLIP QKV must be [rows,{expected}], got {tuple(qkv_dense.shape)}")
    out = torch.zeros(
        qkv_dense.shape[0],
        3 * num_heads * head_dim_pad,
        device=qkv_dense.device,
        dtype=qkv_dense.dtype,
    )
    for section in range(3):
        for head in range(num_heads):
            source = section * num_heads * head_dim + head * head_dim
            target = section * num_heads * head_dim_pad + head * head_dim_pad
            out[:, target : target + head_dim] = qkv_dense[:, source : source + head_dim]
    return out


def unpad_attention_heads(
    attention: Tensor,
    *,
    num_heads: int | None = None,
    head_dim: int | None = None,
    head_dim_pad: int | None = None,
) -> Tensor:
    """Remove per-head pad lanes from a physical SigLIP attention output."""
    num_heads = Pi0A100Config.vision_num_heads if num_heads is None else num_heads
    head_dim = Pi0A100Config.siglip_head_dim if head_dim is None else head_dim
    head_dim_pad = Pi0A100Config.siglip_head_dim_pad if head_dim_pad is None else head_dim_pad
    expected = num_heads * head_dim_pad
    if attention.ndim != 2 or attention.shape[1] < expected:
        raise ValueError(
            f"padded SigLIP attention must be [rows,>={expected}], got {tuple(attention.shape)}"
        )
    return torch.cat(
        [attention[:, head * head_dim_pad : head * head_dim_pad + head_dim] for head in range(num_heads)],
        dim=-1,
    )


def fused_gemm(
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


def fused_siglip_qkv(x: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
    return pad_qkv_heads((x.float() @ weight.float().T + bias.float()).to(torch.bfloat16))


def fused_siglip_oproj_residual(
    residual: Tensor,
    attention_padded: Tensor,
    weight: Tensor,
    bias: Tensor,
) -> Tensor:
    attention = unpad_attention_heads(attention_padded)
    return (
        residual.float() + attention.float() @ weight.float().T + bias.float()
    ).to(residual.dtype)


def fused_siglip_fc1(x: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
    return gelu_tanh(x.float() @ weight.float().T + bias.float()).to(torch.bfloat16)


def fused_siglip_fc2_residual(
    residual: Tensor,
    hidden: Tensor,
    weight: Tensor,
    bias: Tensor,
) -> Tensor:
    return (
        residual.float() + hidden.float() @ weight.float().T + bias.float()
    ).to(residual.dtype)


def patch_linear_residual(
    patch_rows: Tensor,
    weight: Tensor,
    bias: Tensor,
    position: Tensor,
    *,
    patch_width: int = 3 * 14 * 14,
) -> Tensor:
    """Opcode-8 math over padded HWC patch rows, including position residual."""
    y = fused_gemm(
        patch_rows,
        weight,
        bias,
        patch_rows.shape[0],
        patch_width,
        residual=position,
    )
    return y.to(dtype=position.dtype)


def final_norm(
    x: Tensor,
    gamma: Tensor,
    beta: Tensor,
    eps: float | None = None,
) -> Tensor:
    """Opcode-9 post-tower LayerNorm over the logical 1152-wide activation."""
    return layernorm(x, gamma, beta, eps=eps)


def projector(x: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
    """Opcode-10 padded vision-to-Gemma projection."""
    y = x.float() @ weight.float().T + bias.float()
    return y.to(dtype=x.dtype)


def fused_mlp_gelu_tanh(
    x: Tensor,
    w1: Tensor,
    b1: Tensor,
    w2: Tensor,
    b2: Tensor,
) -> Tensor:
    """Fused-side two-projection SigLIP MLP."""
    h = gelu_tanh(x.float() @ w1.float().T + b1.float())
    return (h @ w2.float().T + b2.float()).to(dtype=x.dtype)


def fused_siglip_layer(
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
        Pi0A100Config.vision_tokens_per_view if tokens_per_view is None else tokens_per_view
    )
    num_heads = Pi0A100Config.vision_num_heads if num_heads is None else num_heads
    head_dim = Pi0A100Config.siglip_head_dim if head_dim is None else head_dim
    ln_eps = Pi0A100Config.ln_eps if ln_eps is None else ln_eps

    x = x0.double()
    n1 = layernorm(x, weights["g1"], weights["b1"], eps=ln_eps)
    qkv = n1 @ weights["Wqkv"].double().T + weights["bqkv"].double()
    attn = fused_siglip_attention(
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


def fused_siglip_stack(
    x0: Tensor,
    layer_weights: list[dict[str, Tensor]],
    *,
    num_views: int = 3,
    tokens_per_view: int | None = None,
) -> Tensor:
    """Stack ``fused_siglip_layer`` for multilayer / tower verification."""
    x = x0
    for w in layer_weights:
        x = fused_siglip_layer(
            x,
            w,
            num_views=num_views,
            tokens_per_view=tokens_per_view,
        )
    return x
