from __future__ import annotations

# HF path: run_prefill (model.forward). Fused path: run_prefill_fused (tokenize + our fused ops, no HF forward).

from dataclasses import dataclass
import importlib
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class InferenceConfig:
    model: str
    device: str = "cuda:0"
    torch_dtype: str = "auto"  # "auto" | "float16" | "bfloat16" | "float32"
    max_input_len: int | None = None
    add_special_tokens: bool = True


@dataclass
class PrefillOutput:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    prompt_lens: torch.Tensor
    logits: torch.Tensor
    past_key_values: Any
    first_gen_token: torch.Tensor
    pos_id_token_1: torch.Tensor
    kv_seq_len_before: torch.Tensor
    kv_seq_len_after: torch.Tensor
    globs: Any = None  # when set (fused prefill), decode reuses and skips preprocess_decode


_MODEL_CACHE: dict[tuple[str, str, str], Any] = {}


def _dtype_from_str(s: str):
    s = s.lower().strip()
    if s == "auto":
        return "auto"
    if s in {"fp16", "float16"}:
        return torch.float16
    if s in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if s in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported torch_dtype: {s}")


def _model_cache_key(model_name: str, device: str, torch_dtype: Any) -> tuple[str, str, str]:
    return (model_name, str(device), str(torch_dtype))


@torch.inference_mode()
def get_or_load_hf_model(model_name: str, *, device: str, torch_dtype: Any) -> Any:
    """Return cached HF model if available, otherwise load and cache it."""
    key = _model_cache_key(model_name, device, torch_dtype)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
    ).to(device)
    model.eval()
    model.requires_grad_(False)
    _MODEL_CACHE[key] = model
    # When loaded with "auto", also register under resolved dtype so decode (using prefill.logits.dtype) reuses same instance
    if torch_dtype == "auto":
        resolved = str(getattr(model, "dtype", model.config.torch_dtype))
        key_resolved = _model_cache_key(model_name, str(device), resolved)
        if key_resolved not in _MODEL_CACHE:
            _MODEL_CACHE[key_resolved] = model
    return model


@torch.inference_mode()
def run_prefill(
    config: InferenceConfig,
    prompts: list[str],
    *,
    loaded_model: Any = None,
    tokenizer: Any = None,
) -> PrefillOutput:
    """Run prefill (HF forward). Optionally pass loaded_model and tokenizer to avoid loading twice (saves memory on Orin)."""
    torch.cuda.set_device(config.device)

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(config.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    tok = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=config.max_input_len is not None,
        max_length=config.max_input_len,
        add_special_tokens=config.add_special_tokens,
    )
    input_ids = tok["input_ids"].to(config.device)
    attention_mask = tok["attention_mask"].to(config.device)

    if loaded_model is not None:
        model = loaded_model
    else:
        torch_dtype = _dtype_from_str(config.torch_dtype)
        model = get_or_load_hf_model(
            config.model,
            device=config.device,
            torch_dtype=torch_dtype,
        )

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
        return_dict=True,
    )

    logits = outputs.logits
    past_key_values = outputs.past_key_values

    prompt_lens = attention_mask.sum(dim=-1)
    first_gen_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(1)
    pos_id_token_1 = prompt_lens.to(dtype=torch.long).unsqueeze(1)
    kv_seq_len_before = prompt_lens.to(dtype=torch.long)
    kv_seq_len_after = kv_seq_len_before + 1

    return PrefillOutput(
        input_ids=input_ids,
        attention_mask=attention_mask,
        prompt_lens=prompt_lens,
        logits=logits,
        past_key_values=past_key_values,
        first_gen_token=first_gen_token,
        pos_id_token_1=pos_id_token_1,
        kv_seq_len_before=kv_seq_len_before,
        kv_seq_len_after=kv_seq_len_after,
    )


def run_prefill_fused(
    config: InferenceConfig,
    prompts: list[str],
    *,
    loaded_model: Any = None,
    tokenizer: Any = None,
    hardware: str = "A100",
    max_new_tokens: int | None = None,
) -> PrefillOutput:
    """
    Run prefill with fused ops only (no HF forward). Dispatches to model/hardware-specific
    fused_prefill (A100 and H100 each have their own layout; preprocess is done at prefill).
    When max_new_tokens is set, KV cache is allocated for prompt_len + max_new_tokens
    so decode does not need to reallocate/pad.
    """
    from qimeng_megakernel.models.LLM.registry import model_module_name, resolve_llm_impl

    impl = resolve_llm_impl(config.model, hardware=hardware)
    suffix = model_module_name(impl.model_key)
    hw = impl.hardware
    family = impl.family
    try:
        mod = importlib.import_module(
            f"qimeng_megakernel.models.LLM.{family}.{suffix}.{hw}.fused_prefill"
        )
    except ModuleNotFoundError:
        raise NotImplementedError(f"fused_prefill not implemented for {impl.model_key} on {hw}")
    if not hasattr(mod, "run_fused_prefill"):
        raise NotImplementedError(f"run_fused_prefill not in {mod.__name__}")
    return mod.run_fused_prefill(
        config,
        prompts,
        loaded_model=loaded_model,
        tokenizer=tokenizer,
        hardware=hardware,
        max_new_tokens=max_new_tokens,
    )
