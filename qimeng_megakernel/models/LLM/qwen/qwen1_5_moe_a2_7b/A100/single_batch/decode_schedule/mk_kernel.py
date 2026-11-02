"""Load and call the Qwen1.5-MoE-A2.7B A100 single-batch decode megakernel."""
from __future__ import annotations

import importlib.util
import math
import sysconfig
from pathlib import Path
from typing import Any

import torch

from qimeng_megakernel.models.instruction import Instruction
from .globs import Globals
from .model_config import QwenMoeConfig

_MODULE_NAME = "qimeng_qwen1_5_moe_a2_7b_a100_single_batch_kernel"


_REQUIRED_FIELDS = (
    "instructions",
    "barriers",
    "hidden_states",
    "normed_hidden_states",
    "q_post_rope",
    "k_cache",
    "v_cache",
    "attn_out",
    "attn_lse_intermediates",
    "attn_out_intermediates",
    "silu_out",
    "shared_expert_silu_out",
    "shared_out",
    "routed_silu_buffer",
    "routed_out_f32",
    "routed_out",
    "router_logits_buf",
    "expert_indices",
    "expert_weights",
    "logits",
    "attn_norm_weights",
    "qkv_weights",
    "qkv_bias",
    "rope_cos",
    "rope_sin",
    "o_proj_weights",
    "mlp_norm_weights",
    "shared_expert_gate_proj_weights",
    "shared_expert_up_proj_weights",
    "shared_expert_down_proj_weights",
    "shared_expert_gate_weights",
    "router_weights",
    "expert_gate_up_proj_weights_flat",
    "expert_down_proj_weights_flat",
    "lm_head_norm_weights",
    "lm_head_weights",
)


def _default_mk_dir() -> Path:
    """Path to the model-owned decode_kernel directory."""
    this_dir = Path(__file__).resolve().parent
    return this_dir.parent / "decode_kernel"


def get_mk(mk_dir: Path | None = None):
    """Return the prebuilt route-owned kernel."""
    root = (_default_mk_dir() if mk_dir is None else Path(mk_dir)).resolve()
    dso = root / f"{_MODULE_NAME}{sysconfig.get_config_var('EXT_SUFFIX')}"
    if not dso.is_file():
        raise ImportError(f"missing prebuilt kernel {dso}; build decode_kernel first")
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, dso)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load prebuilt kernel {dso}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "mk", None)):
        raise ImportError(f"{dso} does not export mk")
    return module.mk


def _tensor_for_mk(t: torch.Tensor | None) -> torch.Tensor | None:
    if t is None:
        return t
    return t.data if isinstance(t, torch.nn.Parameter) else t


def _assert_ready(globs: Globals) -> None:
    missing = [name for name in _REQUIRED_FIELDS if getattr(globs, name, None) is None]
    if missing:
        raise ValueError(
            "Globals is not decode-ready; these fields are None: " + ", ".join(missing)
        )


def call_mk(globs: Globals, mk: Any) -> None:
    """
    Call the compiled Qwen1.5-MoE A100 megakernel.
    KV cache: [L, 1, S, n_kv, hd] (5D).
    """
    _assert_ready(globs)

    lm_head_norm = _tensor_for_mk(globs.lm_head_norm_weights)
    lm_head_w = _tensor_for_mk(globs.lm_head_weights)

    attn_scale = 1.0 / math.sqrt(QwenMoeConfig.head_dim)
    rms_norm_eps = float(QwenMoeConfig.rms_norm_eps)
    pos_id = int(globs.position_id) if globs.position_id is not None else 0

    instructions = globs.instructions
    if instructions is not None and not instructions.is_contiguous():
        instructions = instructions.contiguous()

    k = globs.k_cache
    v = globs.v_cache
    if k.dim() == 4:
        k = k.unsqueeze(1)
    if v.dim() == 4:
        v = v.unsqueeze(1)
    k_cache_5d = k.contiguous()
    v_cache_5d = v.contiguous()

    qkv_bias = _tensor_for_mk(globs.qkv_bias)
    if qkv_bias.dim() == 2:
        qkv_bias = qkv_bias.view(qkv_bias.size(0), -1, 16).contiguous()

    rope_cos = globs.rope_cos
    rope_sin = globs.rope_sin
    if rope_cos is not None and rope_cos.dim() == 3:
        rope_cos = rope_cos[0].contiguous()
        rope_sin = rope_sin[0].contiguous()

    router_logits_buf = globs.router_logits_buf.contiguous()
    expert_weights = globs.expert_weights.float().contiguous()

    shared_expert_gate_w = globs.shared_expert_gate_weights
    # Squeeze trailing dim 1 → [L, H] for the gate weight accessor
    if shared_expert_gate_w.dim() == 3:
        shared_expert_gate_w = shared_expert_gate_w.squeeze(-1).contiguous()

    # The A100 native argument list still has three legacy dense-MLP pointer positions. MoE
    # opcodes never dereference them; reuse a valid BF16 tensor instead of allocating
    # three multi-gigabyte placeholders on every token.
    unused_dense_mlp = globs.hidden_states

    mk(
        instructions,
        globs.hidden_states,
        globs.normed_hidden_states,
        globs.q_post_rope,
        k_cache_5d,
        v_cache_5d,
        globs.attn_out,
        globs.attn_lse_intermediates,
        globs.attn_out_intermediates,
        globs.silu_out,
        globs.logits,
        globs.attn_norm_weights,
        globs.qkv_weights,
        qkv_bias,
        rope_cos,
        rope_sin,
        globs.o_proj_weights,
        globs.mlp_norm_weights,
        unused_dense_mlp,
        unused_dense_mlp,
        unused_dense_mlp,
        lm_head_norm,
        lm_head_w,
        globs.router_weights,
        router_logits_buf,
        globs.expert_indices,
        expert_weights,
        globs.shared_expert_gate_proj_weights,
        globs.shared_expert_up_proj_weights,
        globs.shared_expert_down_proj_weights,
        globs.shared_expert_silu_out,
        globs.shared_out,
        shared_expert_gate_w,
        globs.expert_gate_up_proj_weights_flat,
        globs.expert_down_proj_weights_flat,
        globs.routed_silu_buffer,
        globs.routed_out_f32,
        globs.routed_out,
        globs.barriers,
        pos_id,
        attn_scale,
        rms_norm_eps,
    )
    torch.cuda.synchronize(globs.hidden_states.device)


def run_instructions(
    globs: Globals,
    instructions: list[Instruction],
    *,
    mk: Any = None,
) -> None:
    """Execute an explicit instruction sequence through the selected A100 kernel."""
    from .instructions import scheduler

    mk_call = get_mk() if mk is None else mk
    sm_count = scheduler._get_sm_count(globs.hidden_states.device)
    queues = scheduler._round_robin_assign_to_sms(instructions, sm_count)
    scheduler.tensorize_instructions(globs, queues)
    call_mk(globs, mk_call)
