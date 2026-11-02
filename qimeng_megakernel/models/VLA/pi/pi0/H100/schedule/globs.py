"""Python-side globals mirrors for pi0 H100 modules (field names ≈ *.cuh ORDER).

These are verification / fused / pyvm buffers — not a full pybind stand-in.
MK launches still pack tensors by the C++ bind_kernel ORDER in the verifier tests.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Optional

import torch


def diff_tensors(a: torch.Tensor, b: torch.Tensor, name: str):
    a = a.float()
    b = b.float()
    adiff = (a - b).abs()
    rdiff = 2 * adiff / (a.abs() + b.abs() + 1e-6)
    print(f"{name}: max adiff: {adiff.max()}, mean rdiff: {rdiff.mean()}")
    return adiff, rdiff


@dataclass
class _GlobalsBase:
    instructions: Optional[torch.Tensor] = None
    barriers: Optional[torch.Tensor] = None

    def diff(self, other: "_GlobalsBase", skip_weights: bool = True):
        for field in fields(self):
            name = field.name
            attr = getattr(self, name)
            other_attr = getattr(other, name)
            if not isinstance(attr, torch.Tensor):
                continue
            if skip_weights and ("_w" in name or name.endswith("_b") or name == "rope"):
                continue
            diff_tensors(attr, other_attr, name)


@dataclass
class SiglipGlobals(_GlobalsBase):
    vision_x: Optional[torch.Tensor] = None
    siglip_x_normed: Optional[torch.Tensor] = None
    qkv_buf: Optional[torch.Tensor] = None
    attn_out: Optional[torch.Tensor] = None
    fc1_buf: Optional[torch.Tensor] = None
    # Production head/tail ABI tensors used by opcodes 8--10.
    ln_gamma: Optional[torch.Tensor] = None
    ln_beta: Optional[torch.Tensor] = None
    patch_in: Optional[torch.Tensor] = None
    patch_w: Optional[torch.Tensor] = None
    patch_b: Optional[torch.Tensor] = None
    proj_w: Optional[torch.Tensor] = None
    proj_b: Optional[torch.Tensor] = None
    proj_out: Optional[torch.Tensor] = None
    # weights / biases filled by tests as needed
    ln_attn_w: Optional[torch.Tensor] = None
    ln_attn_b: Optional[torch.Tensor] = None
    ln_ffn_w: Optional[torch.Tensor] = None
    ln_ffn_b: Optional[torch.Tensor] = None
    qkv_w: Optional[torch.Tensor] = None
    qkv_b: Optional[torch.Tensor] = None
    oproj_w: Optional[torch.Tensor] = None
    oproj_b: Optional[torch.Tensor] = None
    fc1_w: Optional[torch.Tensor] = None
    fc1_b: Optional[torch.Tensor] = None
    fc2_w: Optional[torch.Tensor] = None
    fc2_b: Optional[torch.Tensor] = None


@dataclass
class GemmaGlobals(_GlobalsBase):
    encoder_x: Optional[torch.Tensor] = None
    qkv_buf: Optional[torch.Tensor] = None
    attn_out: Optional[torch.Tensor] = None
    gate_buf: Optional[torch.Tensor] = None
    up_buf: Optional[torch.Tensor] = None
    rms_factor: Optional[torch.Tensor] = None
    k_cache: Optional[torch.Tensor] = None
    v_cache: Optional[torch.Tensor] = None
    rope: Optional[torch.Tensor] = None
    qkv_w: Optional[torch.Tensor] = None
    oproj_w: Optional[torch.Tensor] = None
    gate_w: Optional[torch.Tensor] = None
    up_w: Optional[torch.Tensor] = None
    down_w: Optional[torch.Tensor] = None


@dataclass
class AeGlobals(_GlobalsBase):
    x: Optional[torch.Tensor] = None
    x_acc: Optional[torch.Tensor] = None
    qkv_buf: Optional[torch.Tensor] = None
    attn_out: Optional[torch.Tensor] = None
    gate_buf: Optional[torch.Tensor] = None
    ffn_buf: Optional[torch.Tensor] = None
    rms_factor: Optional[torch.Tensor] = None
    k_cache: Optional[torch.Tensor] = None
    v_cache: Optional[torch.Tensor] = None
    rope: Optional[torch.Tensor] = None
    qkv_w: Optional[torch.Tensor] = None
    oproj_w: Optional[torch.Tensor] = None
    gate_w: Optional[torch.Tensor] = None
    up_w: Optional[torch.Tensor] = None
    down_w: Optional[torch.Tensor] = None
    # Action-head tail, matching kernel/common/ae_globals.cuh exactly:
    #   noise     bf16 [1, 1, 64, 256]   (logical action width 32, K-padded)
    #   act_h     bf16 [1, 1, 64, 1024]
    #   act_w_in  bf16 [1, 1, 1024, 256]
    #   act_w_mlp bf16 [1, 1, 1024, 1024]
    #   act_w_out bf16 [1, 1, 32, 1024]
    #   act_bias  fp32 [1, 1, 13, 1024]
    noise: Optional[torch.Tensor] = None
    act_h: Optional[torch.Tensor] = None
    act_w_in: Optional[torch.Tensor] = None
    act_w_mlp: Optional[torch.Tensor] = None
    act_w_out: Optional[torch.Tensor] = None
    act_bias: Optional[torch.Tensor] = None
