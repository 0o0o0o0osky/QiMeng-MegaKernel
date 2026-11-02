"""
Utilities for creating test globs with real prefill data (Qwen3-0.6B A100).

Builds Globals directly from HuggingFace Qwen3ForCausalLM while preserving the A100
decode layout: interleaved K cache in 5D for fused/Python verification.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import torch

from qimeng_megakernel.models.LLM.llama.preprocess import _interleave_k_cache
from qimeng_megakernel.models.LLM.preprocess import DecodePreprocessOutput, prepare_kv_cache_for_decode
from qimeng_megakernel.models.LLM.qwen.qwen3__0_6b.preprocess import (
    _load_hf_qwen3_model,
    _stack_qwen3_weights_from_hf,
)
from qimeng_megakernel.models.LLM.qwen.qwen3__0_6b.A100.single_batch.decode_schedule.globs import Globals
from qimeng_megakernel.models.LLM.qwen.qwen3__0_6b.A100.single_batch.decode_schedule.model_config import Qwen3Config

QWEN3_TEST_MODEL_ENV = "QWEN3_TEST_MODEL"
_CONFIGURED_MODEL_PATH = os.environ.get(QWEN3_TEST_MODEL_ENV, "").strip()
LOCAL_CHECKPOINT = (
    Path(_CONFIGURED_MODEL_PATH).expanduser() if _CONFIGURED_MODEL_PATH else None
)
LOCAL_CHECKPOINT_AVAILABLE = bool(
    LOCAL_CHECKPOINT is not None
    and LOCAL_CHECKPOINT.is_dir()
    and (LOCAL_CHECKPOINT / "config.json").is_file()
    and (
        (LOCAL_CHECKPOINT / "model.safetensors").is_file()
        or (LOCAL_CHECKPOINT / "model.safetensors.index.json").is_file()
    )
    and (
        (LOCAL_CHECKPOINT / "tokenizer.json").is_file()
        or (LOCAL_CHECKPOINT / "tokenizer_config.json").is_file()
    )
)

SHORT_PROMPTS = ["Hello", "Tell me", "Python is"]


def require_local_qwen3_test_model(
    model_path: str | os.PathLike[str] | None = None,
) -> str:
    """Resolve the explicit local checkpoint without any Hub-ID fallback."""
    configured = (
        os.fspath(model_path)
        if model_path is not None
        else os.environ.get(QWEN3_TEST_MODEL_ENV, "")
    )
    if not configured.strip():
        raise RuntimeError(
            f"{QWEN3_TEST_MODEL_ENV} must name a local Qwen3 checkpoint directory; "
            "Hub IDs and implicit downloads are disabled"
        )

    root = Path(configured).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(
            f"{QWEN3_TEST_MODEL_ENV} is not a local directory: {configured!r}; "
            "Hub fallback is disabled"
        )
    required_groups = (
        ("config.json",),
        ("model.safetensors", "model.safetensors.index.json"),
        ("tokenizer.json", "tokenizer_config.json"),
    )
    missing = [
        "|".join(group)
        for group in required_groups
        if not any((root / name).is_file() for name in group)
    ]
    if missing:
        raise FileNotFoundError(
            f"{QWEN3_TEST_MODEL_ENV} lacks required local assets: {', '.join(missing)}"
        )
    return str(root.resolve())


def format_prompt_with_chat_template(prompt: str, tokenizer) -> str:
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def _get_sm_count(dev: torch.device) -> int:
    if dev.type != "cuda":
        return 1
    return torch.cuda.get_device_properties(dev).multi_processor_count


def make_globs_from_real_prefill(
    prompt: str,
    model_path: str | os.PathLike[str] | None = None,
    device: str = "cuda:0",
    dtype: torch.dtype = torch.bfloat16,
    max_seq_len: int = 512,
    return_model: bool = False,
    use_chat_template: bool = True,
) -> Globals | tuple[Globals, Any]:
    from transformers import AutoTokenizer

    model_path = require_local_qwen3_test_model(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if use_chat_template:
        prompt = format_prompt_with_chat_template(prompt, tokenizer)
    dev = torch.device(device)
    model = _load_hf_qwen3_model(model_path, device=dev, dtype=dtype)

    Qwen3Config.update_from_hf(model.config)
    n_layers = Qwen3Config.num_hidden_layers
    n_kv_heads = Qwen3Config.num_key_value_heads
    head_dim = Qwen3Config.head_dim
    n_heads = Qwen3Config.num_attention_heads
    intermediate_size = Qwen3Config.intermediate_size
    vocab_size = Qwen3Config.vocab_size

    inputs = tokenizer(prompt, return_tensors="pt").to(dev)
    input_ids = inputs["input_ids"]
    prompt_len = int(input_ids.shape[1])

    with torch.no_grad():
        out = model(input_ids, use_cache=True, output_hidden_states=False)

    past_key_values = out.past_key_values
    first_gen_token_id = int(out.logits[0, -1].argmax())

    kv = prepare_kv_cache_for_decode(
        past_key_values,
        seq_len_hint=max_seq_len,
        target_seq_len=max_seq_len,
    )
    kv = DecodePreprocessOutput(
        k_cache=_interleave_k_cache(kv.k_cache, head_dim),
        v_cache=kv.v_cache,
    )

    weights = _stack_qwen3_weights_from_hf(model)
    sm_count = _get_sm_count(dev)
    num_partials = min(16, max(1, sm_count // n_kv_heads))

    def make_buf(shape, buf_dtype=dtype):
        return torch.zeros(shape, device=dev, dtype=buf_dtype)

    with torch.no_grad():
        first_gen_token = torch.tensor([[first_gen_token_id]], device=dev, dtype=torch.long)
        hidden_states_init = model.model.embed_tokens(first_gen_token).squeeze(0).squeeze(0).to(dtype)

    barriers = torch.zeros(
        (n_layers, 6, n_heads + n_kv_heads * 2),
        dtype=torch.int32,
        device=dev,
    )

    globs = Globals(
        position_id=prompt_len,
        k_cache=kv.k_cache.contiguous(),
        v_cache=kv.v_cache.contiguous(),
        qkv_weights=weights["qkv_weights"].contiguous(),
        q_norm_weights=weights["q_norm_weights"].contiguous(),
        k_norm_weights=weights["k_norm_weights"].contiguous(),
        o_proj_weights=weights["o_proj_weights"].contiguous(),
        attn_norm_weights=weights["attn_norm_weights"].contiguous(),
        mlp_norm_weights=weights["mlp_norm_weights"].contiguous(),
        up_proj_weights=weights["up_proj_weights"].contiguous(),
        gate_proj_weights=weights["gate_proj_weights"].contiguous(),
        down_proj_weights=weights["down_proj_weights"].contiguous(),
        lm_head_norm_weights=weights["lm_head_norm_weights"].contiguous(),
        lm_head_weights=weights["lm_head_weights"].contiguous(),
        rope_cos=weights["rope_cos"].contiguous(),
        rope_sin=weights["rope_sin"].contiguous(),
        hidden_states=hidden_states_init.contiguous(),
        qkv_raw=make_buf(Qwen3Config.qkv_out_dim),
        q_post_rope=make_buf(n_heads * head_dim),
        attn_out=make_buf(n_heads * head_dim),
        attn_out_intermediates=make_buf([n_heads, num_partials, head_dim], buf_dtype=torch.float32),
        attn_lse_intermediates=make_buf([n_heads, num_partials], buf_dtype=torch.float32),
        silu_out=make_buf(intermediate_size),
        logits=make_buf(vocab_size),
        barriers=barriers,
    )

    if return_model:
        return globs, model
    return globs


def backup_globs(globs: Globals) -> dict[str, Optional[torch.Tensor]]:
    def clone_required(name: str) -> torch.Tensor:
        value = getattr(globs, name, None)
        assert isinstance(value, torch.Tensor), f"missing tensor Globals.{name}"
        return value.clone()

    return {
        "hidden_states": clone_required("hidden_states"),
        "qkv_raw": clone_required("qkv_raw"),
        "q_post_rope": clone_required("q_post_rope"),
        "k_cache": clone_required("k_cache"),
        "v_cache": clone_required("v_cache"),
        "attn_out": clone_required("attn_out"),
        "attn_lse_intermediates": clone_required("attn_lse_intermediates"),
        "attn_out_intermediates": clone_required("attn_out_intermediates"),
        "silu_out": clone_required("silu_out"),
        "logits": clone_required("logits"),
        "barriers": clone_required("barriers"),
    }


def restore_globs(globs: Globals, backup: dict[str, Optional[torch.Tensor]]) -> None:
    for key, value in backup.items():
        if value is not None and hasattr(globs, key):
            dest = getattr(globs, key)
            if isinstance(dest, torch.Tensor) and dest.shape == value.shape:
                dest.copy_(value)
