from __future__ import annotations

import importlib
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import torch

from qimeng_megakernel.models.LLM.decode_state import (
    advance_decode_state,
    ensure_decode_state,
    row_active,
    sync_kv_lengths,
)
from qimeng_megakernel.models.LLM.prefill import PrefillOutput, get_or_load_hf_model
from qimeng_megakernel.models.LLM.preprocess import (
    preprocess_decode as llm_preprocess_decode,
)
from qimeng_megakernel.models.LLM.registry import (
    import_decode_schedule,
    import_pyvm_instruction_modules,
    resolve_llm_impl,
)
from qimeng_megakernel.models.instruction import run_pyvm_sequence


def _tokenizer_pretrained_id_for_fused_decode(model_name: str, *, hardware: str) -> str:
    """Resolve tokenizer files from a local path, environment setting, or Hugging Face ID."""
    p = Path(model_name).expanduser()
    if p.is_dir() and (p / "config.json").exists():
        return str(p.resolve())
    try:
        impl = resolve_llm_impl(model_name, hardware=hardware)
        if impl.family == "llama" and impl.model_key == "llama-3.2-1b":
            env = os.environ.get("LLAMA_32_1B_MODEL")
            if env:
                pe = Path(env).expanduser()
                if pe.is_dir() and (pe / "config.json").exists():
                    return str(pe.resolve())
            models_home = Path.home() / "models"
            for sub in ("llama3.2_1b", "Llama-3.2-1B-Instruct"):
                hm = models_home / sub
                if hm.is_dir() and (hm / "config.json").exists():
                    return str(hm.resolve())
    except Exception:
        pass
    return model_name


def _sample_logits(
    logits: torch.Tensor,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 0,
    min_p: float = 0.0,
    presence_penalty: float = 0.0,
    generated_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply sampling (or greedy when temperature=0). Returns next token id(s): shape () or (B,) on same device.

    logits: [vocab] or [B, vocab]. If 2D with B==1 from caller, still treated as multi-batch when dim0>1.
    presence_penalty: subtract from logits for each unique token id in generated_ids (OpenAI-style).
    min_p: after softmax, zero out tokens with prob < min_p then renormalize before sampling.
    """
    if logits.dim() == 2 and logits.shape[0] > 1:
        B, V = logits.shape
        is_greedy_b = temperature <= 0 or (
            top_p >= 1.0 and (top_k <= 0 or top_k >= V) and min_p <= 0.0
        )
        if is_greedy_b and (presence_penalty == 0.0 or generated_ids is None or generated_ids.numel() == 0):
            return torch.argmax(logits, dim=-1)
        out = []
        for b in range(B):
            gen_b = None
            if generated_ids is not None and generated_ids.numel() > 0:
                gen_b = generated_ids[b : b + 1]
            out.append(
                _sample_logits(
                    logits[b],
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    min_p=min_p,
                    presence_penalty=presence_penalty,
                    generated_ids=gen_b,
                )
            )
        return torch.stack(out, dim=0)

    if logits.dim() == 2:
        logits = logits.squeeze(0)
    # Greedy fast path: skip float().clone() to reduce decode latency
    is_greedy = temperature <= 0 or (top_p >= 1.0 and (top_k <= 0 or top_k >= logits.numel()) and min_p <= 0.0)
    if is_greedy and (presence_penalty == 0.0 or generated_ids is None or generated_ids.numel() == 0):
        return torch.argmax(logits, dim=-1, keepdim=True).squeeze(-1)

    logits = logits.float().clone()
    device = logits.device

    if presence_penalty != 0.0 and generated_ids is not None and generated_ids.numel() > 0:
        seen = generated_ids.flatten().unique()
        logits[seen] = logits[seen] - presence_penalty

    if is_greedy:
        return torch.argmax(logits, dim=-1, keepdim=True).squeeze(-1)

    logits = logits / max(temperature, 1e-8)
    probs = torch.softmax(logits, dim=-1)

    if min_p > 0.0:
        probs = torch.where(probs >= min_p, probs, torch.zeros_like(probs))
        s = probs.sum()
        if s > 1e-9:
            probs = probs / s
        else:
            probs = torch.softmax(logits, dim=-1)

    if top_k > 0 and top_k < logits.numel():
        topk_probs, topk_idx = torch.topk(probs, top_k, dim=-1)
        probs = torch.zeros_like(probs)
        probs.scatter_(-1, topk_idx, topk_probs)
        s = probs.sum()
        if s > 1e-9:
            probs = probs / s

    if top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        mask = cumsum - sorted_probs > top_p
        sorted_probs = torch.where(mask, torch.zeros_like(sorted_probs), sorted_probs)
        probs = torch.zeros_like(probs)
        probs.scatter_(-1, sorted_idx, sorted_probs)
        s = probs.sum()
        if s > 1e-9:
            probs = probs / s
        else:
            probs = torch.softmax(logits, dim=-1)

    next_id = torch.multinomial(probs.clamp(min=1e-9), num_samples=1, replacement=True).squeeze(-1)
    return next_id


def _get_decode_schedule(model_name: str, hardware: str, batch_size: int = 1):
    return import_decode_schedule(model_name, hardware, batch_size=batch_size)


def _get_pyvm_instruction_modules(model_name: str, hardware: str, batch_size: int):
    return import_pyvm_instruction_modules(model_name, hardware, batch_size)


def _copy_first_gen_hidden_from_prefill(
    prefill: PrefillOutput,
    globs: Any,
    *,
    model: Any | None = None,
) -> None:
    """Fill globs.hidden_states from prefill.first_gen_token; set position_id from prefill."""
    with torch.no_grad():
        tok = prefill.first_gen_token
        emb_fn = getattr(globs, "embed_tokens", None)
        if emb_fn is not None:
            ids = tok.squeeze(-1)
            if isinstance(emb_fn, torch.Tensor):
                emb = emb_fn[ids].to(globs.hidden_states.dtype)
            else:
                emb = emb_fn(ids).to(globs.hidden_states.dtype)
        else:
            if model is None:
                raise ValueError("Need model or globs.embed_tokens to embed first_gen_token")
            emb = model.model.embed_tokens(tok).to(globs.hidden_states.dtype)
        if globs.hidden_states.ndim == 1:
            globs.hidden_states.copy_(emb.reshape(-1))
        else:
            if emb.dim() == 3:
                emb = emb.squeeze(1)
            globs.hidden_states.copy_(emb)
    B_pre = int(prefill.input_ids.shape[0])
    if B_pre > 1 and globs.hidden_states.ndim == 2:
        globs.position_ids = prefill.pos_id_token_1.squeeze(1).clone().to(
            device=globs.hidden_states.device, dtype=torch.long
        )
        ensure_decode_state(globs, B_pre, device=globs.hidden_states.device)
        sync_kv_lengths(globs)
    globs.position_id = int(prefill.pos_id_token_1.max().item())


def _token_ids_for_compare_dict(argmax_1d: torch.Tensor) -> int | list[int]:
    t = argmax_1d.flatten()
    if t.numel() == 1:
        return int(t.item())
    return [int(x) for x in t.tolist()]


def _count_generated_tokens_per_row(
    new_tokens: torch.Tensor,
    eos_token_ids: list[int] | None = None,
) -> torch.Tensor:
    """Count actual generated tokens per row, including the first EOS token when present."""
    if new_tokens.ndim == 1:
        new_tokens = new_tokens.unsqueeze(0)
    batch_size = int(new_tokens.shape[0])
    if new_tokens.numel() == 0:
        return torch.zeros(batch_size, dtype=torch.long, device=new_tokens.device)
    if not eos_token_ids:
        return torch.full(
            (batch_size,),
            int(new_tokens.shape[1]),
            dtype=torch.long,
            device=new_tokens.device,
        )

    eos_set = {int(tid) for tid in eos_token_ids}
    counts = torch.full(
        (batch_size,),
        int(new_tokens.shape[1]),
        dtype=torch.long,
        device=new_tokens.device,
    )
    for b in range(batch_size):
        row = new_tokens[b]
        for j in range(int(row.shape[0])):
            if int(row[j].item()) in eos_set:
                counts[b] = j + 1
                break
    return counts


def _pad_decode_kv_cache_to_len(globs: Any, target_seq_len: int) -> None:
    """Grow optional B-capacity decode KV storage while preserving its prefix."""
    k_cache = getattr(globs, "k_cache_decode", None)
    v_cache = getattr(globs, "v_cache_decode", None)
    if k_cache is None and v_cache is None:
        return
    if k_cache is None or v_cache is None:
        raise ValueError("k_cache_decode and v_cache_decode must be allocated together")
    if k_cache.ndim != 5 or tuple(v_cache.shape) != tuple(k_cache.shape):
        raise ValueError(
            "decode KV caches must share "
            "[layers, capacity, kv_heads, seq, head_dim] shape"
        )

    current_seq_len = int(k_cache.shape[3])
    if target_seq_len <= current_seq_len:
        return
    for name, cache in (("k_cache_decode", k_cache), ("v_cache_decode", v_cache)):
        shape = list(cache.shape)
        shape[3] = target_seq_len
        padded = cache.new_zeros(shape)
        padded[:, :, :, :current_seq_len, :].copy_(cache)
        setattr(globs, name, padded)


def _load_hf_model(model_name: str, *, device: torch.device, dtype: torch.dtype):
    return get_or_load_hf_model(
        model_name,
        device=str(device),
        torch_dtype=dtype,
    )


@torch.inference_mode()
def _decode_fused(
    model_name: str,
    *,
    prefill: PrefillOutput,
    max_new_tokens: int,
    eos_token_check_interval: int = 16,
    eos_token_ids: list[int] | None = None,
    **kwargs: Any,
):
    """Decode using fused ops only: embed -> run_fused_decode_step -> argmax, in chunks with EOS check."""
    if max_new_tokens <= 0:
        batch_size = int(prefill.input_ids.shape[0])
        return {
            "generated_ids": prefill.input_ids,
            "new_tokens": prefill.input_ids.new_empty((prefill.input_ids.shape[0], 0)),
            "per_row_new_tokens": prefill.input_ids.new_zeros(batch_size, dtype=torch.long),
        }
    ignore_eos = bool(kwargs.pop("ignore_eos", False))
    if ignore_eos:
        eos_token_ids = []
    provided_model = kwargs.pop("loaded_model", None)
    device = prefill.input_ids.device
    dtype = prefill.logits.dtype
    prefill_globs = getattr(prefill, "globs", None)
    # Disk-backed prefill may not instantiate a Hugging Face model; load it only when needed.
    if prefill_globs is not None and getattr(prefill_globs, "embed_tokens", None) is not None:
        model = None
    else:
        model = (
            provided_model
            if provided_model is not None
            else _load_hf_model(model_name, device=torch.device(device), dtype=dtype)
        )
    if prefill_globs is not None:
        from qimeng_megakernel.models.LLM.preprocess import _pad_kv_cache_to_len
        globs = prefill_globs
        prompt_len = int(prefill.input_ids.shape[1])
        target_seq_len = prompt_len + max_new_tokens
        if globs.k_cache.shape[2] < target_seq_len:
            globs.k_cache, globs.v_cache = _pad_kv_cache_to_len(
                globs.k_cache, globs.v_cache, target_seq_len=target_seq_len
            )
    else:
        globs: Any = llm_preprocess_decode(
            model_name,
            prefill=prefill,
            loaded_model=model,
            max_new_tokens=max_new_tokens,
            **{k: v for k, v in kwargs.items() if k not in ("eos_token_check_interval", "eos_token_ids", "debug")},
        )
    batch_size, prompt_len = prefill.input_ids.shape[0], int(prefill.input_ids.shape[1])
    if batch_size > 1:
        pg = getattr(prefill, "globs", None)
        if pg is None or getattr(pg, "embed_tokens", None) is None:
            raise NotImplementedError(
                "fused decode with batch_size>1 requires fused prefill (prefill.globs from run_prefill_fused)."
            )
        if int(pg.k_cache.shape[1]) != batch_size:
            raise ValueError(
                "prefill.globs.k_cache batch dim %s != prefill batch_size %s"
                % (pg.k_cache.shape[1], batch_size)
            )
    output_tokens = prefill.input_ids.new_zeros(
        batch_size, prompt_len + max_new_tokens, dtype=prefill.input_ids.dtype
    )
    output_tokens[:, :prompt_len] = prefill.input_ids
    output_tokens[:, prompt_len] = prefill.first_gen_token.squeeze(-1)

    from transformers import AutoTokenizer

    _tok_id = _tokenizer_pretrained_id_for_fused_decode(
        model_name, hardware=kwargs.get("hardware", "A100")
    )
    _tok_meta = AutoTokenizer.from_pretrained(_tok_id)
    pad_token_id = getattr(_tok_meta, "pad_token_id", None)
    pad_token_id = int(pad_token_id) if pad_token_id is not None else None
    if eos_token_ids is None:
        eos_token_ids = (
            [_tok_meta.eos_token_id]
            if _tok_meta.eos_token_id is not None
            else (getattr(_tok_meta, "additional_special_tokens_ids", []) or [])
        )
        if not eos_token_ids and model is not None and hasattr(model, "config"):
            eid = getattr(model.config, "eos_token_id", None)
            if eid is not None:
                eos_token_ids = list(eid) if isinstance(eid, (list, tuple)) else [int(eid)]

    schedule = _get_decode_schedule(
        model_name,
        kwargs.get("hardware", "A100"),
        batch_size=int(prefill.input_ids.shape[0]),
    )
    sampling_kw = {
        "temperature": kwargs.get("temperature", 0.0),
        "top_p": kwargs.get("top_p", 1.0),
        "top_k": int(kwargs.get("top_k", 0) or 0),
        "min_p": kwargs.get("min_p", 0.0),
        "presence_penalty": kwargs.get("presence_penalty", 0.0),
    }
    generator = FusedGenerator(
        globs,
        model,
        schedule.layer_scheduler.run_fused_decode_step,
        **sampling_kw,
    )
    # First token already in output_tokens[:, prompt_len]; generate remaining max_new_tokens - 1
    num_new_after_first = generator.generate_with_eos(
        output_tokens,
        prompt_len=prompt_len,
        ntok=max_new_tokens - 1,
        eos_token_check_interval=eos_token_check_interval,
        eos_token_ids=eos_token_ids,
        pad_token_id=pad_token_id,
    )
    total_new_tokens = num_new_after_first + 1
    generated_ids = output_tokens[:, : prompt_len + total_new_tokens]
    new_tokens = output_tokens[:, prompt_len : prompt_len + total_new_tokens]
    per_row_new_tokens = getattr(generator, "_last_per_row_new_tokens", None)
    if per_row_new_tokens is None:
        per_row_new_tokens = torch.full(
            (batch_size,),
            total_new_tokens,
            dtype=torch.long,
            device=output_tokens.device,
        )
    return {
        "generated_ids": generated_ids,
        "new_tokens": new_tokens,
        "per_row_new_tokens": per_row_new_tokens,
    }


@torch.inference_mode()
def _decode_python_vm(
    model_name: str,
    *,
    prefill: PrefillOutput,
    max_new_tokens: int,
    eos_token_check_interval: int = 16,
    eos_token_ids: list[int] | None = None,
    **kwargs: Any,
):
    """Decode using Python VM: embed -> run_pyvm_sequence -> argmax. Reuses prefill.globs when set (same flow as fused/mk)."""
    if max_new_tokens <= 0:
        batch_size = int(prefill.input_ids.shape[0])
        return {
            "generated_ids": prefill.input_ids,
            "new_tokens": prefill.input_ids.new_empty((prefill.input_ids.shape[0], 0)),
            "per_row_new_tokens": prefill.input_ids.new_zeros(batch_size, dtype=torch.long),
        }
    ignore_eos = bool(kwargs.pop("ignore_eos", False))
    if ignore_eos:
        eos_token_ids = []
    provided_model = kwargs.pop("loaded_model", None)
    device = prefill.input_ids.device
    dtype = prefill.logits.dtype
    prefill_globs = getattr(prefill, "globs", None)
    if prefill_globs is not None and getattr(prefill_globs, "embed_tokens", None) is not None:
        model = None
    else:
        model = (
            provided_model
            if provided_model is not None
            else _load_hf_model(model_name, device=torch.device(device), dtype=dtype)
        )
    if prefill_globs is not None:
        from qimeng_megakernel.models.LLM.preprocess import _pad_kv_cache_to_len
        globs = prefill_globs
        prompt_len = int(prefill.input_ids.shape[1])
        target_seq_len = prompt_len + max_new_tokens
        if globs.k_cache.shape[2] < target_seq_len:
            globs.k_cache, globs.v_cache = _pad_kv_cache_to_len(
                globs.k_cache, globs.v_cache, target_seq_len=target_seq_len
            )
    else:
        globs = llm_preprocess_decode(
            model_name,
            prefill=prefill,
            loaded_model=model,
            max_new_tokens=max_new_tokens,
            **{k: v for k, v in kwargs.items() if k not in ("eos_token_check_interval", "eos_token_ids", "debug")},
        )
    batch_size, prompt_len = prefill.input_ids.shape[0], int(prefill.input_ids.shape[1])
    hardware = kwargs.get("hardware", "A100")
    if batch_size > 1:
        if prefill_globs is None or globs.hidden_states is None or globs.hidden_states.ndim != 2:
            raise NotImplementedError(
                "python_vm decode with batch_size>1 requires fused prefill with prefill.globs "
                "and 2D hidden_states [B, H] (Llama 3.2 1B/3B A100)."
            )
        if int(globs.hidden_states.shape[0]) != batch_size:
            raise ValueError(
                "prefill.globs.hidden_states batch %s != prefill batch_size %s"
                % (globs.hidden_states.shape[0], batch_size)
            )
    py_sched, py_vm = _get_pyvm_instruction_modules(model_name, hardware, batch_size)
    output_tokens = prefill.input_ids.new_zeros(
        batch_size, prompt_len + max_new_tokens, dtype=prefill.input_ids.dtype
    )
    output_tokens[:, :prompt_len] = prefill.input_ids
    output_tokens[:, prompt_len] = prefill.first_gen_token.squeeze(-1)

    tok = None
    if eos_token_ids is None or (batch_size > 1 and not ignore_eos):
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_name)
    if eos_token_ids is None:
        assert tok is not None
        eos_token_ids = (
            [tok.eos_token_id]
            if tok.eos_token_id is not None
            else (getattr(tok, "additional_special_tokens_ids", []) or [])
        )
        if not eos_token_ids and model is not None and hasattr(model, "config"):
            eid = getattr(model.config, "eos_token_id", None)
            if eid is not None:
                eos_token_ids = list(eid) if isinstance(eid, (list, tuple)) else [int(eid)]
    pad_token_id = None
    if tok is not None:
        p = getattr(tok, "pad_token_id", None)
        pad_token_id = int(p) if p is not None else None

    cached_pyvm_instructions = (
        py_sched.build_instruction_list(globs) if batch_size == 1 else None
    )

    def run_pyvm_step(g: Any) -> None:
        g.barriers.zero_()
        inst = (
            cached_pyvm_instructions
            if cached_pyvm_instructions is not None
            else py_sched.build_instruction_list(g)
        )
        run_pyvm_sequence(g, inst, py_vm.INSTRUCTION_TO_HANDLER)

    sampling_kw = {
        "temperature": kwargs.get("temperature", 0.0),
        "top_p": kwargs.get("top_p", 1.0),
        "top_k": int(kwargs.get("top_k", 0) or 0),
        "min_p": kwargs.get("min_p", 0.0),
        "presence_penalty": kwargs.get("presence_penalty", 0.0),
    }
    generator = FusedGenerator(globs, model, run_pyvm_step, **sampling_kw)
    num_new_after_first = generator.generate_with_eos(
        output_tokens,
        prompt_len=prompt_len,
        ntok=max_new_tokens - 1,
        eos_token_check_interval=eos_token_check_interval,
        eos_token_ids=eos_token_ids,
        pad_token_id=pad_token_id,
    )
    total_new_tokens = num_new_after_first + 1
    generated_ids = output_tokens[:, : prompt_len + total_new_tokens]
    new_tokens = output_tokens[:, prompt_len : prompt_len + total_new_tokens]
    per_row_new_tokens = getattr(generator, "_last_per_row_new_tokens", None)
    if per_row_new_tokens is None:
        per_row_new_tokens = torch.full(
            (batch_size,),
            total_new_tokens,
            dtype=torch.long,
            device=output_tokens.device,
        )
    return {
        "generated_ids": generated_ids,
        "new_tokens": new_tokens,
        "per_row_new_tokens": per_row_new_tokens,
    }


@torch.inference_mode()
def _decode_mk(
    model_name: str,
    *,
    prefill: PrefillOutput,
    max_new_tokens: int,
    mk_dir: str | None = None,
    eos_token_check_interval: int = 16,
    eos_token_ids: list[int] | None = None,
    **kwargs: Any,
):
    """Decode using the compiled MegaKernel: embed -> MK -> argmax.

    For greedy decode (default), uses a tight ``FusedGenerator.generate_greedy_fast``
    loop with no per-step CUDA Event allocation or EOS bookkeeping overhead.
    Sampling (temperature > 0) or stream_decode uses the full ``generate_with_eos`` path.
    """
    if max_new_tokens <= 0:
        batch_size = int(prefill.input_ids.shape[0])
        return {
            "generated_ids": prefill.input_ids,
            "new_tokens": prefill.input_ids.new_empty((prefill.input_ids.shape[0], 0)),
            "per_row_new_tokens": prefill.input_ids.new_zeros(batch_size, dtype=torch.long),
        }
    ignore_eos = bool(kwargs.pop("ignore_eos", False))
    if ignore_eos:
        eos_token_ids = []
    decode_time_enabled = kwargs.pop("decode_time", False)
    stream_decode = kwargs.pop("stream_decode", False)
    greedy_fast_override = kwargs.pop("greedy_fast", None)
    provided_model = kwargs.pop("loaded_model", None)
    device = prefill.input_ids.device
    dtype = prefill.logits.dtype
    prefill_globs = getattr(prefill, "globs", None)
    # Reuse fused/MegaKernel prefill Globals without loading a Hugging Face model.
    if prefill_globs is not None:
        model = None
        if kwargs.get("debug"):
            print("[mk] Reusing prefill globs, skipping HF model load.", flush=True)
    else:
        model = (
            provided_model
            if provided_model is not None
            else _load_hf_model(model_name, device=torch.device(device), dtype=dtype)
        )
    _mk_kw = {k: v for k, v in kwargs.items() if k not in ("eos_token_check_interval", "eos_token_ids", "debug", "mk_dir", "stream_decode", "greedy_fast")}
    _mk_kw["for_mk"] = True
    if prefill_globs is not None:
        from qimeng_megakernel.models.LLM.preprocess import _pad_kv_cache_to_len
        globs = prefill_globs
        prompt_len = int(prefill.input_ids.shape[1])
        target_seq_len = prompt_len + max_new_tokens
        if globs.k_cache.shape[2] < target_seq_len:
            globs.k_cache, globs.v_cache = _pad_kv_cache_to_len(
                globs.k_cache, globs.v_cache, target_seq_len=target_seq_len
            )
        _pad_decode_kv_cache_to_len(globs, int(globs.k_cache.shape[2]))
    else:
        globs = llm_preprocess_decode(
            model_name,
            prefill=prefill,
            loaded_model=model,
            max_new_tokens=max_new_tokens,
            **_mk_kw,
        )
    schedule = _get_decode_schedule(
        model_name,
        kwargs.get("hardware", "A100"),
        batch_size=int(prefill.input_ids.shape[0]),
    )
    if schedule.get_mk is None or schedule.call_mk is None:
        raise NotImplementedError(
            "MK decode not implemented for this model (e.g. Llama 3.2 3B); use --decode-route fused or python_vm."
        )
    static_megakernel = bool(
        getattr(schedule.package, "STATIC_MEGAKERNEL", False)
    )
    batch_size, prompt_len = prefill.input_ids.shape[0], int(prefill.input_ids.shape[1])
    is_greedy = (
        float(kwargs.get("temperature", 0.0)) <= 0.0
        and float(kwargs.get("top_p", 1.0)) >= 1.0
        and int(kwargs.get("top_k", 0) or 0) <= 0
        and float(kwargs.get("min_p", 0.0)) <= 0.0
        and float(kwargs.get("presence_penalty", 0.0)) == 0.0
    )
    use_greedy_fast = is_greedy and not stream_decode
    if greedy_fast_override is not None:
        use_greedy_fast = bool(greedy_fast_override)
    queue_build_wall_s = 0.0
    tensorize_wall_s = 0.0
    instruction_setup_wall_s = 0.0
    setup_t0 = time.perf_counter()
    queue_t0 = time.perf_counter()
    _mk_dir = Path(mk_dir) if mk_dir else None
    mk = schedule.get_mk(_mk_dir)
    if not static_megakernel:
        if batch_size <= 1:
            queues = schedule.scheduler.build_instruction_queues(globs)
        else:
            queues = schedule.scheduler.build_runtime_instruction_queues(
                globs, batch_size=batch_size
            )
        queue_build_wall_s = time.perf_counter() - queue_t0
        tensorize_t0 = time.perf_counter()
        schedule.scheduler.tensorize_instructions(globs, queues)
    else:
        queue_build_wall_s = time.perf_counter() - queue_t0
        tensorize_t0 = time.perf_counter()
    if decode_time_enabled and device.type == "cuda":
        torch.cuda.synchronize(device)
    tensorize_wall_s = time.perf_counter() - tensorize_t0
    instruction_setup_wall_s = time.perf_counter() - setup_t0
    output_tokens = prefill.input_ids.new_zeros(
        batch_size, prompt_len + max_new_tokens, dtype=prefill.input_ids.dtype
    )
    output_tokens[:, :prompt_len] = prefill.input_ids
    output_tokens[:, prompt_len] = prefill.first_gen_token.squeeze(-1)

    from transformers import AutoTokenizer

    need_tokenizer_meta = (
        eos_token_ids is None or stream_decode or (batch_size > 1 and not ignore_eos)
    )
    _tokenizer = AutoTokenizer.from_pretrained(model_name) if need_tokenizer_meta else None
    pad_token_id = None
    if _tokenizer is not None:
        pad = getattr(_tokenizer, "pad_token_id", None)
        pad_token_id = int(pad) if pad is not None else None

    if eos_token_ids is None:
        eos_token_ids = (
            [_tokenizer.eos_token_id]
            if _tokenizer.eos_token_id is not None
            else (getattr(_tokenizer, "additional_special_tokens_ids", []) or [])
        )
        if not eos_token_ids and model is not None and hasattr(model, "config"):
            eid = getattr(model.config, "eos_token_id", None)
            if eid is not None:
                eos_token_ids = list(eid) if isinstance(eid, (list, tuple)) else [int(eid)]
    elif _tokenizer is None and stream_decode:
        _tokenizer = AutoTokenizer.from_pretrained(model_name)

    step_idx = [0]
    mk_step_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    step_timing = DecodeStepTiming(device) if decode_time_enabled else None

    def run_mk_step(g: Any) -> None:
        barrier_t0 = time.perf_counter()
        barrier_start = barrier_end = None
        if step_timing is not None and device.type == "cuda":
            barrier_start = torch.cuda.Event(enable_timing=True)
            barrier_end = torch.cuda.Event(enable_timing=True)
            barrier_start.record()
        if not static_megakernel:
            g.barriers.zero_()
        if barrier_start is not None and barrier_end is not None:
            barrier_end.record()
            step_timing.record_gpu_pair("barrier_zero", barrier_start, barrier_end)
        if step_timing is not None:
            step_timing.add_wall("barrier_zero", time.perf_counter() - barrier_t0)
        if decode_time_enabled and device.type == "cuda":
            start_ev = torch.cuda.Event(enable_timing=True)
            end_ev = torch.cuda.Event(enable_timing=True)
            start_ev.record()
        schedule.call_mk(g, mk)
        if decode_time_enabled and device.type == "cuda":
            end_ev.record()
            mk_step_events.append((start_ev, end_ev))
        # No per-step sync: all launches go on the same stream, so ordering already gives the correct dependency; a host sync each step would only add latency and hurt tok/s.
        step_idx[0] += 1

    sampling_kw = {
        "temperature": kwargs.get("temperature", 0.0),
        "top_p": kwargs.get("top_p", 1.0),
        "top_k": int(kwargs.get("top_k", 0) or 0),
        "min_p": kwargs.get("min_p", 0.0),
        "presence_penalty": kwargs.get("presence_penalty", 0.0),
    }
    generator = FusedGenerator(
        globs, model, run_mk_step,
        stream_decode=stream_decode, tokenizer=_tokenizer,
        **sampling_kw,
    )
    generator.step_timing = step_timing
    if decode_time_enabled:
        start_wall_s = time.perf_counter()
        if device.type == "cuda":
            start_ev = torch.cuda.Event(enable_timing=True)
            end_ev = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize(device)
            start_ev.record()
    if use_greedy_fast:
        # Pre-allocate event pairs before the loop so .record() in the loop is free (no alloc).
        fast_mk_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        if decode_time_enabled and device.type == "cuda":
            ntok_fast = max_new_tokens - 1
            fast_mk_events = [
                (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                for _ in range(ntok_fast)
            ]
        num_new_after_first = generator.generate_greedy_fast(
            output_tokens,
            prompt_len=prompt_len,
            ntok=max_new_tokens - 1,
            eos_token_ids=eos_token_ids if not ignore_eos else [],
            pad_token_id=pad_token_id,
            call_mk=lambda g: schedule.call_mk(g, mk),
            mk_step_events=fast_mk_events,
            reset_barriers=not static_megakernel,
        )
        mk_step_events.extend(fast_mk_events)
    else:
        num_new_after_first = generator.generate_with_eos(
            output_tokens,
            prompt_len=prompt_len,
            ntok=max_new_tokens - 1,
            eos_token_check_interval=eos_token_check_interval,
            eos_token_ids=[] if ignore_eos else eos_token_ids,
            pad_token_id=pad_token_id,
        )
    if decode_time_enabled:
        if device.type == "cuda":
            end_ev.record()
            torch.cuda.synchronize(device)
            decode_gpu_s = start_ev.elapsed_time(end_ev) / 1000.0
            mk_kernel_gpu_s = sum(start.elapsed_time(end) for start, end in mk_step_events) / 1000.0
        else:
            decode_gpu_s = 0.0
            mk_kernel_gpu_s = 0.0
        decode_wall_s = time.perf_counter() - start_wall_s
        per_row_new_tokens = getattr(generator, "_last_per_row_new_tokens", None)
        effective_new_tokens = int(per_row_new_tokens.sum().item()) if per_row_new_tokens is not None else batch_size * (num_new_after_first + 1)
        per_row_list = [int(x) for x in per_row_new_tokens.tolist()] if per_row_new_tokens is not None else [num_new_after_first + 1] * batch_size
        # Throughput numerator != effective_new_tokens.
        # `effective_new_tokens` preserves Hugging Face `max_new_tokens` semantics: it includes the first
        # token after the prompt, which is produced by *prefill* and therefore lies outside
        # the decode timing window (the decode loop runs max_new_tokens-1 MegaKernel steps).
        # Every row contributes exactly one such prefill token (also under early EOS), so the
        # number of tokens actually produced inside the timed decode window is
        # effective_new_tokens - batch_size. All three tok/s figures below use that numerator
        # so numerator and denominator cover the same window.
        decode_window_new_tokens = max(effective_new_tokens - batch_size, 0)
        non_kernel_gpu_s = max(decode_gpu_s - mk_kernel_gpu_s, 0.0)
        non_kernel_wall_s = max(decode_wall_s - mk_kernel_gpu_s, 0.0)
        timing = {
            "instruction_setup_wall_s": instruction_setup_wall_s,
            "queue_build_wall_s": queue_build_wall_s,
            "tensorize_wall_s": tensorize_wall_s,
            "decode_wall_s": decode_wall_s,
            "decode_gpu_s": decode_gpu_s,
            "mk_kernel_gpu_s": mk_kernel_gpu_s,
            "non_kernel_gpu_s": non_kernel_gpu_s,
            "non_kernel_wall_s": non_kernel_wall_s,
            "effective_new_tokens": effective_new_tokens,
            "decode_window_new_tokens": decode_window_new_tokens,
            "effective_tok_s_wall": (decode_window_new_tokens / decode_wall_s) if (decode_wall_s > 0 and decode_window_new_tokens > 0) else 0.0,
            "effective_tok_s_gpu": (decode_window_new_tokens / decode_gpu_s) if (decode_gpu_s > 0 and decode_window_new_tokens > 0) else 0.0,
            "kernel_tok_s": (decode_window_new_tokens / mk_kernel_gpu_s) if (mk_kernel_gpu_s > 0 and decode_window_new_tokens > 0) else 0.0,
            "kernel_steps": step_idx[0],
        }
        if step_timing is not None:
            timing["step_breakdown"] = step_timing.summarize()
        _mk_time_label = "[mk] decode_time_fast" if use_greedy_fast else "[mk] decode_time"
        print(
            "%s: wall=%.3f s, gpu=%.3f s, mk_kernel_gpu=%.3f s, non_kernel_gpu=%.3f s, "
            "setup=%.3f s (queue=%.3f s, tensorize=%.3f s), effective_new_tokens=%d, "
            "tok/s[wall=%.2f gpu=%.2f kernel=%.2f], decode_window_new_tokens=%d, "
            "per_row_new_tokens[min/avg/max]=%d/%.1f/%d%s"
            % (
                _mk_time_label,
                decode_wall_s,
                decode_gpu_s,
                mk_kernel_gpu_s,
                non_kernel_gpu_s,
                instruction_setup_wall_s,
                queue_build_wall_s,
                tensorize_wall_s,
                effective_new_tokens,
                timing["effective_tok_s_wall"],
                timing["effective_tok_s_gpu"],
                timing["kernel_tok_s"],
                decode_window_new_tokens,
                min(per_row_list) if per_row_list else 0,
                (effective_new_tokens / len(per_row_list)) if per_row_list else 0.0,
                max(per_row_list) if per_row_list else 0,
                " (greedy fast path)" if use_greedy_fast else "",
            ),
            flush=True,
        )
        if step_timing is not None:
            breakdown = timing["step_breakdown"]
            print(
                "[mk] non_kernel breakdown gpu_s=%s wall_s=%s"
                % (breakdown["gpu_s_by_section"], breakdown["wall_s_by_section"]),
                flush=True,
            )
    else:
        timing = None
    total_new_tokens = num_new_after_first + 1
    generated_ids = output_tokens[:, : prompt_len + total_new_tokens]
    new_tokens = output_tokens[:, prompt_len : prompt_len + total_new_tokens]
    per_row_new_tokens = getattr(generator, "_last_per_row_new_tokens", None)
    if per_row_new_tokens is None:
        per_row_new_tokens = torch.full(
            (batch_size,),
            total_new_tokens,
            dtype=torch.long,
            device=output_tokens.device,
        )
    return {
        "generated_ids": generated_ids,
        "new_tokens": new_tokens,
        "per_row_new_tokens": per_row_new_tokens,
        "timing": timing,
    }


@torch.inference_mode()
def _decode_hf(
    model_name: str,
    *,
    prefill: PrefillOutput,
    max_new_tokens: int,
    **gen_kwargs: Any,
):
    """Decode with HF model.generate. When temperature > 0 (or do_sample=True), use sampling; else greedy."""
    if max_new_tokens <= 0:
        batch_size = int(prefill.input_ids.shape[0])
        return {
            "generated_ids": prefill.input_ids,
            "new_tokens": prefill.input_ids.new_empty((prefill.input_ids.shape[0], 0)),
            "per_row_new_tokens": prefill.input_ids.new_zeros(batch_size, dtype=torch.long),
        }

    from transformers import GenerationConfig

    device = prefill.input_ids.device
    dtype = prefill.logits.dtype
    model = gen_kwargs.pop("loaded_model", None)
    if model is None:
        model = _load_hf_model(model_name, device=device, dtype=dtype)
    input_ids = prefill.input_ids
    attention_mask = prefill.attention_mask

    # Sampling: allow temperature, top_p, top_k, repetition_penalty, presence_penalty when temperature > 0 or do_sample=True
    do_sample = gen_kwargs.pop("do_sample", None)
    temperature = gen_kwargs.pop("temperature", 0.0)
    top_p = gen_kwargs.pop("top_p", 1.0)
    top_k = gen_kwargs.pop("top_k", 0)
    min_p = gen_kwargs.pop("min_p", None)  # HF may not support; strip so not passed as model_kwargs
    repetition_penalty = gen_kwargs.pop("repetition_penalty", 1.0)
    presence_penalty = gen_kwargs.pop("presence_penalty", 0.0)
    use_sampling = (do_sample is True) or (temperature is not None and float(temperature) > 0)

    pad_id = getattr(model.config, "pad_token_id", None)
    if isinstance(pad_id, list):
        pad_id = pad_id[0] if pad_id else None
    if pad_id is None:
        pad_id = getattr(model.config, "eos_token_id", None)
        if isinstance(pad_id, list):
            pad_id = pad_id[0] if pad_id else None
    if use_sampling:
        gen_config_kw = dict(
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=float(temperature) if temperature is not None else 0.7,
            top_p=float(top_p) if top_p is not None else 1.0,
            top_k=int(top_k) if top_k else 0,  # 0 = no top_k cutoff in HF
            pad_token_id=pad_id,
        )
        if repetition_penalty is not None and float(repetition_penalty) != 1.0:
            gen_config_kw["repetition_penalty"] = float(repetition_penalty)
        if presence_penalty is not None and float(presence_penalty) != 0.0:
            gen_config_kw["presence_penalty"] = float(presence_penalty)
        generation_config = GenerationConfig(**gen_config_kw)
    else:
        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad_id,
        )
    # Strip our impl-selection kwargs so they are not passed to model.generate().
    for key in (
        "hardware",
        "arch",
        "stream_decode",
        "decode_route",
        "decode_time",
        "memory",
    ):
        gen_kwargs.pop(key, None)
    # Avoid deprecation: pass only generation_config, not generation_config + max_new_tokens.
    generated = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        generation_config=generation_config,
        **gen_kwargs,
    )
    new_tokens = generated[:, input_ids.shape[1] :]
    eos_ids = generation_config.eos_token_id
    if eos_ids is None:
        eos_token_ids = None
    elif isinstance(eos_ids, (list, tuple)):
        eos_token_ids = [int(x) for x in eos_ids]
    else:
        eos_token_ids = [int(eos_ids)]
    per_row_new_tokens = _count_generated_tokens_per_row(new_tokens, eos_token_ids)
    return {
        "generated_ids": generated,
        "new_tokens": new_tokens,
        "per_row_new_tokens": per_row_new_tokens,
    }


def _dump_instruction_queues(path: str, queues: list[list[Any]]):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(str(queues))


class DecodeStepTiming:
    def __init__(self, device: torch.device):
        self.device = device
        self.wall_s_by_section: dict[str, float] = defaultdict(float)
        self.gpu_event_pairs: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = defaultdict(list)
        self.counts: dict[str, int] = defaultdict(int)

    def add_wall(self, section: str, elapsed_s: float) -> None:
        self.wall_s_by_section[section] += float(elapsed_s)
        self.counts[section] += 1

    def record_gpu_pair(self, section: str, start_ev: torch.cuda.Event, end_ev: torch.cuda.Event) -> None:
        self.gpu_event_pairs[section].append((start_ev, end_ev))

    def summarize(self) -> dict[str, Any]:
        gpu_s_by_section: dict[str, float] = {}
        for section, pairs in self.gpu_event_pairs.items():
            gpu_s_by_section[section] = sum(start.elapsed_time(end) for start, end in pairs) / 1000.0
        return {
            "wall_s_by_section": {k: float(v) for k, v in sorted(self.wall_s_by_section.items())},
            "gpu_s_by_section": {k: float(v) for k, v in sorted(gpu_s_by_section.items())},
            "counts": {k: int(v) for k, v in sorted(self.counts.items())},
        }


class FusedGenerator:
    """
    Decode by running fused ops per step (embed -> run_fused_decode_step -> sample or argmax).
    Supports chunked generation and per-chunk EOS check.
    When temperature > 0, uses temperature/top_p/top_k/min_p/presence_penalty sampling.
    """

    def __init__(
        self,
        globs: Any,
        model: Any,
        run_fused_step: Callable[[Any], None],
        stream_decode: bool = False,
        tokenizer: Any = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 0,
        min_p: float = 0.0,
        presence_penalty: float = 0.0,
    ):
        self.globs = globs
        self.model = model
        self.run_fused_step = run_fused_step
        self.stream_decode = stream_decode
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.presence_penalty = presence_penalty
        self.step_timing: DecodeStepTiming | None = None
        self._eos_token_ids_tensor_for_advance: torch.Tensor | None = None
        self._eos_token_ids_for_advance: list[int] = []
        # B>1: rows with active_mask False still run the step; force next_id so EOS scan / padding stay sane.
        self._pad_token_id_for_inactive: int | None = None
        self._last_end_exclusive: torch.Tensor | None = None
        self._last_per_row_new_tokens: torch.Tensor | None = None
        # stream_decode: per-row stop after first printed EOS (like megakernel generate_with_eos).
        self._stream_row_done: list[bool] | None = None
        self._stream_eos_id_set: frozenset[int] | None = None

    def run(
        self,
        input_ids: torch.Tensor,
        pos_id: int,
        generated_token_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """One step: embed -> run fused step -> sample or argmax(logits)."""
        timing = self.step_timing
        dtype = self.globs.hidden_states.dtype
        t0 = time.perf_counter()
        with torch.no_grad():
            embed_tokens = getattr(self.globs, "embed_tokens", None)
            if embed_tokens is not None:
                ids1 = input_ids.squeeze(-1)
                emb = embed_tokens[ids1]
            else:
                emb = self.model.model.embed_tokens(input_ids).squeeze(-1)
        if timing is not None:
            timing.add_wall("embed_lookup", time.perf_counter() - t0)
        if self.globs.hidden_states.ndim == 1:
            t0 = time.perf_counter()
            self.globs.hidden_states.copy_(emb.reshape(-1).to(dtype))
        else:
            t0 = time.perf_counter()
            self.globs.hidden_states.copy_(emb.to(dtype))
        if timing is not None:
            timing.add_wall("hidden_copy", time.perf_counter() - t0)
        if self.globs.hidden_states.ndim == 2 and self.globs.position_ids is not None:
            pid = self.globs.position_ids
            batch_rows = int(self.globs.hidden_states.shape[0])
            if pid.numel() == batch_rows:
                # MegaKernel reads per-row indices through position_ids_ptr; scalar pos_id is used only when the pointer is null.
                self.globs.position_id = pos_id
            else:
                self.globs.position_id = int(pid.max().item())
        else:
            self.globs.position_id = pos_id
        self.run_fused_step(self.globs)
        logits = self.globs.logits
        inactive_rows = None
        force_inactive_fill_id = None
        is_greedy_no_penalty = (
            self.temperature <= 0
            and self.top_p >= 1.0
            and self.top_k <= 0
            and self.min_p <= 0.0
            and self.presence_penalty == 0.0
            and (generated_token_ids is None or generated_token_ids.numel() == 0)
        )
        # Multi-batch: finished rows still get a logits vector; mask them so greedy/sample matches single-row EOS behavior.
        if (
            logits is not None
            and logits.dim() == 2
            and logits.shape[0] > 1
            and self.globs.active_mask is not None
            and self.globs.active_mask.shape[0] == logits.shape[0]
            and not bool(self.globs.active_mask.all().item())
        ):
            t0 = time.perf_counter()
            fill_id = self._pad_token_id_for_inactive
            if fill_id is None and self._eos_token_ids_for_advance:
                fill_id = int(self._eos_token_ids_for_advance[0])
            inactive = ~self.globs.active_mask
            if fill_id is not None and 0 <= int(fill_id) < logits.shape[1]:
                if is_greedy_no_penalty:
                    inactive_rows = inactive
                    force_inactive_fill_id = int(fill_id)
                else:
                    logits = logits.clone()
                    logits[inactive] = float("-inf")
                    logits[inactive, int(fill_id)] = 0.0
            if timing is not None:
                timing.add_wall("logits_mask", time.perf_counter() - t0)
        t0 = time.perf_counter()
        sample_start = sample_end = None
        if timing is not None and self.globs.hidden_states.device.type == "cuda":
            sample_start = torch.cuda.Event(enable_timing=True)
            sample_end = torch.cuda.Event(enable_timing=True)
            sample_start.record()
        if is_greedy_no_penalty:
            if logits.dim() == 2 and logits.shape[0] > 1:
                next_id = torch.argmax(logits, dim=-1)
            else:
                next_id = torch.argmax(logits, dim=-1, keepdim=True).squeeze(-1)
        else:
            next_id = _sample_logits(
                logits,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                min_p=self.min_p,
                presence_penalty=self.presence_penalty,
                generated_ids=generated_token_ids,
            )
        if sample_start is not None and sample_end is not None:
            sample_end.record()
            timing.record_gpu_pair("sample_logits", sample_start, sample_end)
        if timing is not None:
            timing.add_wall("sample_logits", time.perf_counter() - t0)
        next_id = next_id.unsqueeze(0) if next_id.dim() == 0 else next_id
        if inactive_rows is not None and force_inactive_fill_id is not None:
            t0 = time.perf_counter()
            next_id[inactive_rows] = force_inactive_fill_id
            if timing is not None:
                timing.add_wall("inactive_token_override", time.perf_counter() - t0)
        if self.globs.hidden_states.ndim == 2 and self.globs.position_ids is not None:
            t0 = time.perf_counter()
            advance_start = advance_end = None
            if timing is not None and self.globs.hidden_states.device.type == "cuda":
                advance_start = torch.cuda.Event(enable_timing=True)
                advance_end = torch.cuda.Event(enable_timing=True)
                advance_start.record()
            advance_decode_state(
                self.globs,
                next_id,
                self._eos_token_ids_for_advance,
                eos_token_ids_tensor=self._eos_token_ids_tensor_for_advance,
            )
            if advance_start is not None and advance_end is not None:
                advance_end.record()
                timing.record_gpu_pair("advance_decode_state", advance_start, advance_end)
            if timing is not None:
                timing.add_wall("advance_decode_state", time.perf_counter() - t0)
        return next_id

    def generate(
        self,
        output_tokens: torch.Tensor,
        prompt_len: int,
        ntok: int,
        ntok_already_generated: int = 1,
    ) -> None:
        """Run ntok decode steps; write new tokens into output_tokens [batch, prompt_len + new].
        Convention: the first step consumes the token at position prompt_len (the first generated token), with pos_id=prompt_len.
        """
        batch_size = output_tokens.shape[0]
        for i in range(ntok):
            input_token_pos = prompt_len + ntok_already_generated + i - 1
            output_token_pos = input_token_pos + 1
            input_ids = output_tokens[:, input_token_pos : input_token_pos + 1]
            pos_id = input_token_pos
            generated_so_far = output_tokens[:, :output_token_pos] if self.presence_penalty != 0.0 else None
            next_ids = self.run(input_ids, pos_id, generated_token_ids=generated_so_far)
            if next_ids.dim() == 0:
                next_ids = next_ids.unsqueeze(0)
            t0 = time.perf_counter()
            write_start = write_end = None
            if self.step_timing is not None and output_tokens.device.type == "cuda":
                write_start = torch.cuda.Event(enable_timing=True)
                write_end = torch.cuda.Event(enable_timing=True)
                write_start.record()
            output_tokens[:, output_token_pos] = next_ids
            if write_start is not None and write_end is not None:
                write_end.record()
                self.step_timing.record_gpu_pair("write_output_tokens", write_start, write_end)
            if self.step_timing is not None:
                self.step_timing.add_wall("write_output_tokens", time.perf_counter() - t0)
            if self.stream_decode:
                step = ntok_already_generated + i
                row_done = self._stream_row_done
                eos_set = self._stream_eos_id_set
                if self.tokenizer is not None:
                    for b in range(batch_size):
                        if row_done is not None and row_done[b]:
                            continue
                        tid = int(next_ids[b].item())
                        active = row_active(self.globs, b)
                        if row_done is None:
                            if not active:
                                continue
                        else:
                            show = active or (bool(eos_set) and tid in eos_set)
                            if not show:
                                continue
                        try:
                            tok_str = self.tokenizer.decode([tid], add_special_tokens=False)
                            tok_repr = repr(tok_str) if tok_str.strip() != tok_str or not tok_str.isprintable() else f'"{tok_str}"'
                        except Exception:
                            tok_repr = ""
                        print(f"[mk] b={b} step {step} pos_id={pos_id} token_id={tid} {tok_repr}".rstrip(), flush=True)
                        if row_done is not None and eos_set and tid in eos_set:
                            row_done[b] = True
                else:
                    if (
                        self.globs.active_mask is not None
                        and batch_size > 1
                        and not bool(self.globs.active_mask.all().item())
                    ):
                        parts: list[int] = []
                        for b in range(batch_size):
                            if row_done is not None and row_done[b]:
                                continue
                            tid_b = int(next_ids[b].item())
                            active_b = row_active(self.globs, b)
                            if row_done is None:
                                if active_b:
                                    parts.append(tid_b)
                            else:
                                if active_b or (bool(eos_set) and tid_b in eos_set):
                                    parts.append(tid_b)
                                    if eos_set and tid_b in eos_set:
                                        row_done[b] = True
                        print(f"[mk] step {step} pos_id={pos_id} next_ids_active={parts}", flush=True)
                    else:
                        if row_done is None:
                            print(f"[mk] step {step} pos_id={pos_id} next_ids={next_ids.tolist()}", flush=True)
                        else:
                            tids: list[int] = []
                            for b in range(batch_size):
                                if row_done[b]:
                                    continue
                                tid_b = int(next_ids[b].item())
                                active_b = row_active(self.globs, b)
                                if active_b or (bool(eos_set) and tid_b in eos_set):
                                    tids.append(tid_b)
                                    if eos_set and tid_b in eos_set:
                                        row_done[b] = True
                            print(f"[mk] step {step} pos_id={pos_id} next_ids={tids}", flush=True)

    def generate_with_eos(
        self,
        output_tokens: torch.Tensor,
        prompt_len: int,
        ntok: int,
        eos_token_check_interval: int,
        eos_token_ids: list[int],
        pad_token_id: int | None = None,
    ) -> int:
        """
        Run decode in chunks of eos_token_check_interval; after each chunk check for EOS.
        Returns num_new_tokens after the first generated token (so total new = return + 1).
        Per-row EOS: slice width uses max end over batch; shorter rows padded with pad_token_id when set.
        """
        batch_size = output_tokens.shape[0]
        device = output_tokens.device
        self._pad_token_id_for_inactive = pad_token_id
        full_end = output_tokens.shape[1]
        end_exclusive = torch.full(
            (batch_size,), full_end, dtype=torch.long, device=device
        )
        eos_ids_tensor = None
        if eos_token_ids:
            eos_ids_tensor = torch.tensor(
                eos_token_ids,
                dtype=output_tokens.dtype,
                device=device,
            )
        self._eos_token_ids_for_advance = list(eos_token_ids)
        self._eos_token_ids_tensor_for_advance = (
            eos_ids_tensor.to(dtype=torch.long) if eos_ids_tensor is not None else None
        )
        if self.stream_decode:
            self._stream_row_done = [False] * batch_size
            self._stream_eos_id_set = (
                frozenset(int(x) for x in eos_token_ids) if eos_token_ids else frozenset()
            )
        else:
            self._stream_row_done = None
            self._stream_eos_id_set = None
        for ntok_already in range(1, ntok + 1, eos_token_check_interval):
            ntok_chunk = min(eos_token_check_interval, ntok - ntok_already + 1)
            if ntok_chunk <= 0:
                break
            self.generate(
                output_tokens,
                prompt_len=prompt_len,
                ntok=ntok_chunk,
                ntok_already_generated=ntok_already,
            )
            start_out = prompt_len + ntok_already - 1
            end_out = prompt_len + ntok_already - 1 + ntok_chunk
            t0 = time.perf_counter()
            if eos_ids_tensor is not None:
                scan_start = scan_end = None
                if self.step_timing is not None and output_tokens.device.type == "cuda":
                    scan_start = torch.cuda.Event(enable_timing=True)
                    scan_end = torch.cuda.Event(enable_timing=True)
                    scan_start.record()
                chunk = output_tokens[:, start_out:end_out]
                eos_hits = (chunk.unsqueeze(-1) == eos_ids_tensor.view(1, 1, -1)).any(dim=-1)
                has_eos = eos_hits.any(dim=1)
                if bool(has_eos.any().item()):
                    first_rel = eos_hits.to(torch.int32).argmax(dim=1).to(torch.long)
                    candidate_end = end_exclusive.clone()
                    candidate_end[has_eos] = start_out + first_rel[has_eos] + 1
                    end_exclusive = torch.minimum(end_exclusive, candidate_end)
                if scan_start is not None and scan_end is not None:
                    scan_end.record()
                    self.step_timing.record_gpu_pair("eos_scan", scan_start, scan_end)
            if self.step_timing is not None:
                self.step_timing.add_wall("eos_scan", time.perf_counter() - t0)
        slice_end = int(end_exclusive.max().item())
        num_new_after_first = slice_end - prompt_len - 1
        self._last_end_exclusive = end_exclusive.clone()
        self._last_per_row_new_tokens = (end_exclusive - prompt_len).clamp(min=0).clone()
        if pad_token_id is not None:
            t0 = time.perf_counter()
            pad_start = pad_end = None
            if self.step_timing is not None and output_tokens.device.type == "cuda":
                pad_start = torch.cuda.Event(enable_timing=True)
                pad_end = torch.cuda.Event(enable_timing=True)
                pad_start.record()
            if slice_end > 0:
                pos = torch.arange(slice_end, device=device, dtype=end_exclusive.dtype)
                pad_mask = pos.unsqueeze(0) >= end_exclusive.unsqueeze(1)
                output_tokens[:, :slice_end] = torch.where(
                    pad_mask,
                    torch.full_like(output_tokens[:, :slice_end], pad_token_id),
                    output_tokens[:, :slice_end],
                )
            if pad_start is not None and pad_end is not None:
                pad_end.record()
                self.step_timing.record_gpu_pair("pad_inactive_rows", pad_start, pad_end)
            if self.step_timing is not None:
                self.step_timing.add_wall("pad_inactive_rows", time.perf_counter() - t0)
        if self.stream_decode:
            self._stream_row_done = None
            self._stream_eos_id_set = None
        return max(0, num_new_after_first)

    def generate_greedy_fast(
        self,
        output_tokens: torch.Tensor,
        prompt_len: int,
        ntok: int,
        eos_token_ids: list[int] | None = None,
        pad_token_id: int | None = None,
        call_mk: Callable[[Any], None] | None = None,
        mk_step_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] | None = None,
        reset_barriers: bool = True,
    ) -> int:
        """Tight greedy loop: no per-step timing, no per-step EOS check, no event allocation.

        Works for both B=1 and B>1. Runs all ntok steps then does a single post-hoc
        EOS scan to compute per-row lengths. Returns num_new_tokens after the first
        generated token (same convention as generate_with_eos).

        call_mk: optional direct callable(globs) that bypasses run_fused_step entirely
        (avoids per-step CUDA Event creation when decode_time is enabled).
        mk_step_events: pre-allocated list of (start, end) CUDA Event pairs (len == ntok).
        If provided, .record() is called around each call_mk (no allocation in loop).
        reset_barriers: clear the legacy instruction-VM barrier tensor before each step.
        Static kernels that own and reset their synchronization state disable this.
        """
        globs = self.globs
        embed = globs.embed_tokens
        dtype = globs.hidden_states.dtype
        is_multibatch = globs.hidden_states.ndim == 2 and globs.position_ids is not None
        device = output_tokens.device
        _step = call_mk if call_mk is not None else self.run_fused_step
        _events = mk_step_events  # pre-allocated; None means no per-step timing

        for i in range(ntok):
            pos = prompt_len + i
            ids = output_tokens[:, pos]
            # Embed + copy into hidden_states (1D for B=1, 2D for B>1)
            emb = embed[ids].to(dtype)
            globs.hidden_states.copy_(emb.reshape(-1) if globs.hidden_states.ndim == 1 else emb)
            # Update position: B=1 uses scalar; B>1 per-row tensor is already
            # set by fused prefill (or previous iteration's post-step advance).
            if not is_multibatch:
                globs.position_id = pos
            # Kernel (record pre-allocated events if provided; zero alloc in loop)
            if reset_barriers:
                globs.barriers.zero_()
            if _events is not None and i < len(_events):
                _events[i][0].record()
            _step(globs)
            if _events is not None and i < len(_events):
                _events[i][1].record()
            # Greedy argmax directly into output buffer
            logits = globs.logits
            if logits.dim() == 1:
                output_tokens[:, pos + 1] = logits.argmax()
            else:
                output_tokens[:, pos + 1] = logits.argmax(dim=-1)
            # Advance per-row position state after the step (matches
            # advance_decode_state in the generate/run path).
            if is_multibatch:
                globs.position_ids.add_(1)
                torch.add(globs.position_ids, 1, out=globs.kv_lengths)

        # Post-hoc EOS scan + per-row lengths (single GPU pass)
        num_new_after_first = ntok
        if eos_token_ids:
            eos_ids_tensor = torch.tensor(
                eos_token_ids, dtype=output_tokens.dtype, device=device
            )
            chunk = output_tokens[:, prompt_len : prompt_len + ntok + 1]
            eos_hits = (chunk.unsqueeze(-1) == eos_ids_tensor.view(1, 1, -1)).any(dim=-1)
            has_eos = eos_hits.any(dim=1)
            full_end = prompt_len + ntok + 1
            end_exclusive = torch.full(
                (output_tokens.shape[0],), full_end, dtype=torch.long, device=device
            )
            if bool(has_eos.any().item()):
                first_rel = eos_hits.to(torch.int32).argmax(dim=1).to(torch.long)
                end_exclusive[has_eos] = prompt_len + first_rel[has_eos] + 1
            slice_end = int(end_exclusive.max().item())
            num_new_after_first = slice_end - prompt_len - 1
            self._last_end_exclusive = end_exclusive
            self._last_per_row_new_tokens = (end_exclusive - prompt_len).clamp(min=0)
            if pad_token_id is not None and slice_end > 0:
                pos_idx = torch.arange(slice_end, device=device, dtype=end_exclusive.dtype)
                pad_mask = pos_idx.unsqueeze(0) >= end_exclusive.unsqueeze(1)
                output_tokens[:, :slice_end] = torch.where(
                    pad_mask,
                    torch.full_like(output_tokens[:, :slice_end], pad_token_id),
                    output_tokens[:, :slice_end],
                )
        else:
            B = output_tokens.shape[0]
            self._last_per_row_new_tokens = torch.full(
                (B,), ntok + 1, dtype=torch.long, device=device
            )
        return max(0, num_new_after_first)


def _tensor_info(t: torch.Tensor) -> str:
    return (
        f"shape={tuple(t.shape)} dtype={t.dtype} device={t.device} stride={t.stride()}"
    )


def _backup_globs_for_step(globs: Any) -> dict[str, torch.Tensor]:
    """Backup buffers that one decode step (fused or pyvm) will modify."""
    out = {
        "hidden_states": globs.hidden_states.clone(),
        "q_post_rope": globs.q_post_rope.clone(),
        "k_cache": globs.k_cache.clone(),
        "v_cache": globs.v_cache.clone(),
        "attn_out": globs.attn_out.clone(),
        "attn_out_intermediates": globs.attn_out_intermediates.clone(),
        "attn_lse_intermediates": globs.attn_lse_intermediates.clone(),
        "silu_out": globs.silu_out.clone(),
        "logits": globs.logits.clone(),
        "barriers": globs.barriers.clone(),
    }
    apv = getattr(globs, "attn_partial_valid", None)
    if isinstance(apv, torch.Tensor):
        out["attn_partial_valid"] = apv.clone()
    return out


def _restore_globs_from_backup(globs: Any, backup: dict[str, torch.Tensor]) -> None:
    """Restore globs from a backup made by _backup_globs_for_step."""
    globs.hidden_states.copy_(backup["hidden_states"])
    globs.q_post_rope.copy_(backup["q_post_rope"])
    globs.k_cache.copy_(backup["k_cache"])
    globs.v_cache.copy_(backup["v_cache"])
    globs.attn_out.copy_(backup["attn_out"])
    globs.attn_out_intermediates.copy_(backup["attn_out_intermediates"])
    globs.attn_lse_intermediates.copy_(backup["attn_lse_intermediates"])
    globs.silu_out.copy_(backup["silu_out"])
    globs.logits.copy_(backup["logits"])
    globs.barriers.copy_(backup["barriers"])
    if "attn_partial_valid" in backup and getattr(globs, "attn_partial_valid", None) is not None:
        globs.attn_partial_valid.copy_(backup["attn_partial_valid"])


@torch.inference_mode()
def compare_hf_fused_first_step(
    model_name: str,
    *,
    prefill: PrefillOutput,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Run one decode step: HF forward vs fused step (same prefill, same first token).
    Compare logits and argmax. Use to debug HF vs Fused token divergence (no pyvm needed).
    Returns dict with: argmax_match, token_id_hf, token_id_fused, logits_max_abs, logits_max_rel, top5_hf, top5_fused.
    """
    device = prefill.input_ids.device
    dtype = prefill.logits.dtype
    model = kwargs.pop("loaded_model", None)
    if model is None:
        model = _load_hf_model(model_name, device=torch.device(device), dtype=dtype)
    # HF: one forward with first_gen_token and prefill cache
    prompt_len = int(prefill.input_ids.shape[1])
    # attention_mask: [prompt_len] + 1 new = (1, prompt_len + 1)
    attn_mask = torch.cat(
        [
            prefill.attention_mask,
            torch.ones(1, 1, device=device, dtype=prefill.attention_mask.dtype),
        ],
        dim=1,
    )
    with torch.no_grad():
        hf_out = model(
            input_ids=prefill.first_gen_token,
            attention_mask=attn_mask,
            past_key_values=prefill.past_key_values,
            use_cache=True,
        )
    logits_hf = hf_out.logits[:, -1, :].float()  # (1, vocab)
    argmax_hf = logits_hf.argmax(-1).cpu()

    # Fused: preprocess -> embed first token -> one step
    globs: Any = llm_preprocess_decode(
        model_name,
        prefill=prefill,
        loaded_model=model,
        max_new_tokens=2,
        **{k: v for k, v in kwargs.items() if k not in ("eos_token_check_interval", "eos_token_ids", "debug")},
    )
    with torch.no_grad():
        emb = model.model.embed_tokens(prefill.first_gen_token)
    globs.hidden_states.copy_(emb.squeeze(0).squeeze(0).to(globs.hidden_states.dtype))
    schedule = _get_decode_schedule(model_name, kwargs.get("hardware", "A100"))
    schedule.layer_scheduler.run_fused_decode_step(globs)
    logits_fused = globs.logits.clone().float().unsqueeze(0)  # (1, vocab)
    argmax_fused = logits_fused.argmax(-1).cpu()

    argmax_match = (argmax_hf == argmax_fused).all().item()
    diff = (logits_hf - logits_fused).abs()
    denom = logits_hf.abs() + logits_fused.abs() + 1e-6
    max_rel = (2 * diff / denom).max().item()
    top5_hf = logits_hf[0].topk(5)
    top5_fused = logits_fused[0].topk(5)
    return {
        "argmax_match": argmax_match,
        "token_id_hf": int(argmax_hf.item()),
        "token_id_fused": int(argmax_fused.item()),
        "logits_max_abs": float(diff.max().item()),
        "logits_max_rel": max_rel,
        "top5_indices_hf": top5_hf.indices.tolist(),
        "top5_values_hf": top5_hf.values.tolist(),
        "top5_indices_fused": top5_fused.indices.tolist(),
        "top5_values_fused": top5_fused.values.tolist(),
    }


@torch.inference_mode()
def compare_fused_pyvm_first_step(
    model_name: str,
    *,
    prefill: PrefillOutput,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Run one decode step with the same prefill globs: fused then pyvm.
    Compare logits and argmax. Use this to debug fused vs python_vm token divergence.
    Returns dict with: argmax_match, token_id_fused, token_id_pyvm, logits_max_abs, logits_max_rel.
    For batch_size>1, token_id_* are list[int] (greedy token per batch row).
    """
    device = prefill.input_ids.device
    dtype = prefill.logits.dtype
    batch_size = int(prefill.input_ids.shape[0])
    hardware = kwargs.get("hardware", "A100")
    pre_kw = {
        k: v
        for k, v in kwargs.items()
        if k not in ("eos_token_check_interval", "eos_token_ids", "debug")
    }
    model: Any | None = None
    if batch_size > 1:
        pg = getattr(prefill, "globs", None)
        if pg is None:
            raise ValueError(
                "compare_fused_pyvm_first_step with batch_size>1 requires fused prefill (prefill.globs)."
            )
        globs = pg
        prompt_len = int(prefill.input_ids.shape[1])
        target_seq_len = prompt_len + 2
        if globs.k_cache.shape[2] < target_seq_len:
            from qimeng_megakernel.models.LLM.preprocess import _pad_kv_cache_to_len

            globs.k_cache, globs.v_cache = _pad_kv_cache_to_len(
                globs.k_cache, globs.v_cache, target_seq_len=target_seq_len
            )
    else:
        model = _load_hf_model(model_name, device=torch.device(device), dtype=dtype)
        globs = llm_preprocess_decode(
            model_name,
            prefill=prefill,
            loaded_model=model,
            max_new_tokens=2,
            **pre_kw,
        )
    _copy_first_gen_hidden_from_prefill(prefill, globs, model=model)

    schedule = _get_decode_schedule(model_name, hardware)
    py_sched, py_vm = _get_pyvm_instruction_modules(model_name, hardware, batch_size)
    instructions_flat = py_sched.build_instruction_list(globs)
    backup = _backup_globs_for_step(globs)

    schedule.layer_scheduler.run_fused_decode_step(globs)
    logits_fused = globs.logits.clone()
    argmax_fused = logits_fused.argmax(-1).cpu()

    _restore_globs_from_backup(globs, backup)
    globs.barriers.zero_()
    run_pyvm_sequence(globs, instructions_flat, py_vm.INSTRUCTION_TO_HANDLER)
    logits_pyvm = globs.logits.clone()
    argmax_pyvm = logits_pyvm.argmax(-1).cpu()

    argmax_match = (argmax_fused == argmax_pyvm).all().item()
    diff = (logits_fused.float() - logits_pyvm.float()).abs()
    denom = logits_fused.float().abs() + logits_pyvm.float().abs() + 1e-6
    max_rel = (2 * diff / denom).max().item()
    return {
        "argmax_match": argmax_match,
        "token_id_fused": _token_ids_for_compare_dict(argmax_fused),
        "token_id_pyvm": _token_ids_for_compare_dict(argmax_pyvm),
        "logits_max_abs": float(diff.max().item()),
        "logits_max_rel": max_rel,
    }


@torch.inference_mode()
def compare_mk_fused_first_step(
    model_name: str,
    *,
    prefill: PrefillOutput,
    mk_dir: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Run one decode step: MK vs fused (same prefill, same first token).
    Compare logits and argmax. If argmax differs, MK kernel likely has a bug (explains trajectory divergence).
    Returns dict with: argmax_match, token_id_mk, token_id_fused, logits_max_abs, logits_max_rel.
    """
    device = prefill.input_ids.device
    dtype = prefill.logits.dtype
    model = _load_hf_model(model_name, device=torch.device(device), dtype=dtype)
    schedule = _get_decode_schedule(model_name, kwargs.get("hardware", "A100"))
    if schedule.get_mk is None or schedule.call_mk is None:
        raise NotImplementedError("MK not available for this model")

    # Fused path: preprocess without for_mk, one step
    kw = {k: v for k, v in kwargs.items() if k not in ("eos_token_check_interval", "eos_token_ids", "debug", "mk_dir")}
    globs_fused: Any = llm_preprocess_decode(
        model_name,
        prefill=prefill,
        loaded_model=model,
        max_new_tokens=2,
        **kw,
    )
    with torch.no_grad():
        emb = model.model.embed_tokens(prefill.first_gen_token)
    globs_fused.hidden_states.copy_(emb.squeeze(0).squeeze(0).to(globs_fused.hidden_states.dtype))
    globs_fused.position_id = int(prefill.input_ids.shape[1])
    schedule.layer_scheduler.run_fused_decode_step(globs_fused)
    logits_fused = globs_fused.logits.clone().float()
    argmax_fused = logits_fused.argmax(-1).cpu()

    # MK path: preprocess with for_mk=True, tensorize, one step
    kw_mk = {**kw, "for_mk": True}
    globs_mk: Any = llm_preprocess_decode(
        model_name,
        prefill=prefill,
        loaded_model=model,
        max_new_tokens=2,
        **kw_mk,
    )
    globs_mk.hidden_states.copy_(emb.squeeze(0).squeeze(0).to(globs_mk.hidden_states.dtype))
    globs_mk.position_id = int(prefill.input_ids.shape[1])
    queues = schedule.scheduler.build_instruction_queues(globs_mk)
    schedule.scheduler.tensorize_instructions(globs_mk, queues)
    from pathlib import Path
    _mk_dir = Path(mk_dir) if mk_dir else None
    mk_call = schedule.get_mk(_mk_dir)
    schedule.call_mk(globs_mk, mk_call)
    logits_mk = globs_mk.logits.clone().float()
    argmax_mk = logits_mk.argmax(-1).cpu()

    argmax_match = (argmax_fused == argmax_mk).all().item()
    diff = (logits_fused - logits_mk).abs()
    denom = logits_fused.abs() + logits_mk.abs() + 1e-6
    max_rel = (2 * diff / denom).max().item()
    return {
        "argmax_match": argmax_match,
        "token_id_mk": int(argmax_mk.item()),
        "token_id_fused": int(argmax_fused.item()),
        "logits_max_abs": float(diff.max().item()),
        "logits_max_rel": max_rel,
    }


@torch.inference_mode()
def compare_mk_pyvm_first_step(
    model_name: str,
    *,
    prefill: PrefillOutput,
    mk_dir: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Run one decode step: MK vs pyvm (same prefill). PyVM is the reference for pyvm_to_megakernel.
    Uses for_mk=True preprocess so globs layout matches the kernel; runs pyvm then MK and compares logits.
    Returns dict with: argmax_match, token_id_mk, token_id_pyvm, logits_max_abs, logits_max_rel.
    """
    device = prefill.input_ids.device
    dtype = prefill.logits.dtype
    model = _load_hf_model(model_name, device=torch.device(device), dtype=dtype)
    schedule = _get_decode_schedule(model_name, kwargs.get("hardware", "A100"))
    if schedule.get_mk is None or schedule.call_mk is None:
        raise NotImplementedError("MK not available for this model")

    kw = {k: v for k, v in kwargs.items() if k not in ("eos_token_check_interval", "eos_token_ids", "debug", "mk_dir")}
    kw_mk = {**kw, "for_mk": True}
    globs: Any = llm_preprocess_decode(
        model_name,
        prefill=prefill,
        loaded_model=model,
        max_new_tokens=2,
        **kw_mk,
    )
    with torch.no_grad():
        emb = model.model.embed_tokens(prefill.first_gen_token)
    globs.hidden_states.copy_(emb.squeeze(0).squeeze(0).to(globs.hidden_states.dtype))
    globs.position_id = int(prefill.input_ids.shape[1])
    backup = _backup_globs_for_step(globs)

    # PyVM path (reference for pyvm_to_megakernel)
    globs.barriers.zero_()
    instructions_flat = schedule.scheduler.build_instruction_list(globs)
    run_pyvm_sequence(globs, instructions_flat, schedule.python_vm.INSTRUCTION_TO_HANDLER)
    logits_pyvm = globs.logits.clone().float()
    argmax_pyvm = logits_pyvm.argmax(-1).cpu()

    # MK path
    _restore_globs_from_backup(globs, backup)
    globs.barriers.zero_()
    queues = schedule.scheduler.build_instruction_queues(globs)
    schedule.scheduler.tensorize_instructions(globs, queues)
    from pathlib import Path
    _mk_dir = Path(mk_dir) if mk_dir else None
    mk_call = schedule.get_mk(_mk_dir)
    schedule.call_mk(globs, mk_call)
    logits_mk = globs.logits.clone().float()
    argmax_mk = logits_mk.argmax(-1).cpu()

    argmax_match = (argmax_pyvm == argmax_mk).all().item()
    diff = (logits_pyvm - logits_mk).abs()
    denom = logits_pyvm.abs() + logits_mk.abs() + 1e-6
    max_rel = (2 * diff / denom).max().item()
    return {
        "argmax_match": argmax_match,
        "token_id_mk": int(argmax_mk.item()),
        "token_id_pyvm": int(argmax_pyvm.item()),
        "logits_max_abs": float(diff.max().item()),
        "logits_max_rel": max_rel,
    }


@torch.inference_mode()
def compare_fused_pyvm_first_step_layerwise(
    model_name: str,
    *,
    prefill: PrefillOutput,
    rel_threshold: float = 0.1,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Same as compare_fused_pyvm_first_step but also run layer-by-layer and report
    the first layer where hidden_states max_rel exceeds rel_threshold (and per-layer stats).
    Fused uses interleaved RoPE (globs.rope_cos/sin from preprocess are interleaved).
    For batch_size>1, token_id_* are list[int] (greedy token per batch row).
    """
    device = prefill.input_ids.device
    dtype = prefill.logits.dtype
    batch_size = int(prefill.input_ids.shape[0])
    hardware = kwargs.get("hardware", "A100")
    pre_kw = {
        k: v
        for k, v in kwargs.items()
        if k not in ("eos_token_check_interval", "eos_token_ids", "debug")
    }
    model: Any | None = None
    if batch_size > 1:
        pg = getattr(prefill, "globs", None)
        if pg is None:
            raise ValueError(
                "compare_fused_pyvm_first_step_layerwise with batch_size>1 requires fused prefill (prefill.globs)."
            )
        globs = pg
        prompt_len = int(prefill.input_ids.shape[1])
        target_seq_len = prompt_len + 2
        if globs.k_cache.shape[2] < target_seq_len:
            from qimeng_megakernel.models.LLM.preprocess import _pad_kv_cache_to_len

            globs.k_cache, globs.v_cache = _pad_kv_cache_to_len(
                globs.k_cache, globs.v_cache, target_seq_len=target_seq_len
            )
    else:
        model = _load_hf_model(model_name, device=torch.device(device), dtype=dtype)
        globs = llm_preprocess_decode(
            model_name,
            prefill=prefill,
            loaded_model=model,
            max_new_tokens=2,
            **pre_kw,
        )
    _copy_first_gen_hidden_from_prefill(prefill, globs, model=model)
    schedule = _get_decode_schedule(model_name, hardware)
    py_sched, py_vm = _get_pyvm_instruction_modules(model_name, hardware, batch_size)
    initial_backup = _backup_globs_for_step(globs)
    num_layers = schedule.config.num_hidden_layers
    per_layer_max_rel: list[float] = []
    first_divergent_layer: int | None = None
    for L in range(1, num_layers + 1):
        _restore_globs_from_backup(globs, initial_backup)
        schedule.layer_scheduler.run_fused_decode_step(globs, num_layers=L)
        hidden_fused = globs.hidden_states.clone()
        _restore_globs_from_backup(globs, initial_backup)
        globs.barriers.zero_()
        instructions_L = py_sched.build_instruction_list(globs, num_layers=L)
        run_pyvm_sequence(globs, instructions_L, py_vm.INSTRUCTION_TO_HANDLER)
        hidden_pyvm = globs.hidden_states.clone()
        denom = hidden_fused.float().abs() + hidden_pyvm.float().abs() + 1e-6
        rel = (2 * (hidden_fused.float() - hidden_pyvm.float()).abs() / denom).max().item()
        per_layer_max_rel.append(rel)
        if first_divergent_layer is None and rel > rel_threshold:
            first_divergent_layer = L
    # If layer 1 diverges, compare q_post_rope, attn_out, hidden_states after layer 0 to pinpoint sign flip.
    stage_max_rel: dict[str, float] = {}
    if first_divergent_layer == 1:
        _restore_globs_from_backup(globs, initial_backup)
        schedule.layer_scheduler.run_fused_decode_step(globs, num_layers=1)
        q_f = globs.q_post_rope.clone()
        a_f = globs.attn_out.clone()
        h_f = globs.hidden_states.clone()
        _restore_globs_from_backup(globs, initial_backup)
        globs.barriers.zero_()
        run_pyvm_sequence(
            globs,
            py_sched.build_instruction_list(globs, num_layers=1),
            py_vm.INSTRUCTION_TO_HANDLER,
        )
        for name, fused_t, pyvm_t in [
            ("q_post_rope", q_f, globs.q_post_rope),
            ("attn_out", a_f, globs.attn_out),
            ("hidden_states", h_f, globs.hidden_states),
        ]:
            d = (fused_t.float() - pyvm_t.float()).abs()
            den = fused_t.float().abs() + pyvm_t.float().abs() + 1e-6
            stage_max_rel[name] = (2 * d / den).max().item()
    # Full step for logits
    _restore_globs_from_backup(globs, initial_backup)
    schedule.layer_scheduler.run_fused_decode_step(globs)
    logits_fused = globs.logits.clone()
    argmax_fused = logits_fused.argmax(-1).cpu()
    _restore_globs_from_backup(globs, initial_backup)
    globs.barriers.zero_()
    instructions_flat = py_sched.build_instruction_list(globs)
    run_pyvm_sequence(globs, instructions_flat, py_vm.INSTRUCTION_TO_HANDLER)
    logits_pyvm = globs.logits.clone()
    argmax_pyvm = logits_pyvm.argmax(-1).cpu()
    argmax_match = (argmax_fused == argmax_pyvm).all().item()
    diff = (logits_fused.float() - logits_pyvm.float()).abs()
    denom = logits_fused.float().abs() + logits_pyvm.float().abs() + 1e-6
    logits_max_rel = (2 * diff / denom).max().item()
    return {
        "argmax_match": argmax_match,
        "token_id_fused": _token_ids_for_compare_dict(argmax_fused),
        "token_id_pyvm": _token_ids_for_compare_dict(argmax_pyvm),
        "logits_max_abs": float(diff.max().item()),
        "logits_max_rel": logits_max_rel,
        "per_layer_hidden_max_rel": per_layer_max_rel,
        "first_divergent_layer": first_divergent_layer,
        "stage_max_rel_after_layer0": stage_max_rel,
    }


def decode(
    model: str,
    *,
    prefill: PrefillOutput,
    decode_route: str = "hf",
    max_new_tokens: int = 1,
    **kwargs: Any,
):
    decode_time_enabled = kwargs.pop("decode_time", False)

    def _timed_decode(route: str, fn, *args, **fn_kwargs):
        device = prefill.input_ids.device
        if device.type == "cuda":
            torch.cuda.set_device(device)
            torch.cuda.synchronize()
            start_ev = torch.cuda.Event(enable_timing=True)
            end_ev = torch.cuda.Event(enable_timing=True)
            start_ev.record()
        else:
            import time
            t0 = time.perf_counter()
        out = fn(*args, **fn_kwargs)
        if device.type == "cuda":
            end_ev.record()
            torch.cuda.synchronize()
            elapsed = start_ev.elapsed_time(end_ev) / 1000.0
        else:
            elapsed = time.perf_counter() - t0
        per_row_new_tokens = out.get("per_row_new_tokens")
        if isinstance(per_row_new_tokens, torch.Tensor):
            effective_new_tokens = int(per_row_new_tokens.sum().item())
            per_row_list = [int(x) for x in per_row_new_tokens.tolist()]
            avg_new_tokens = effective_new_tokens / len(per_row_list) if per_row_list else 0.0
            min_new_tokens = min(per_row_list) if per_row_list else 0
            max_new_tokens_row = max(per_row_list) if per_row_list else 0
        else:
            new_tokens = out.get("new_tokens")
            if new_tokens is not None and isinstance(new_tokens, torch.Tensor):
                effective_new_tokens = int(new_tokens.numel())
                if new_tokens.ndim >= 2:
                    avg_new_tokens = float(new_tokens.shape[1])
                    min_new_tokens = int(new_tokens.shape[1])
                    max_new_tokens_row = int(new_tokens.shape[1])
                else:
                    avg_new_tokens = float(new_tokens.numel())
                    min_new_tokens = int(new_tokens.numel())
                    max_new_tokens_row = int(new_tokens.numel())
            else:
                effective_new_tokens = 0
                avg_new_tokens = 0.0
                min_new_tokens = 0
                max_new_tokens_row = 0
        time_src = "GPU" if device.type == "cuda" else "wall"
        print(
            "[%s] decode_time: %.3f s (%s), effective_new_tokens=%d, %.2f tok/s, per_row_new_tokens[min/avg/max]=%d/%.1f/%d"
            % (
                route,
                elapsed,
                time_src,
                effective_new_tokens,
                effective_new_tokens / elapsed if elapsed > 0 else 0,
                min_new_tokens,
                avg_new_tokens,
                max_new_tokens_row,
            ),
            flush=True,
        )
        return out

    if decode_route == "hf":
        kwargs.pop("debug", None)
        if decode_time_enabled:
            return _timed_decode("hf", _decode_hf, model, prefill=prefill, max_new_tokens=max_new_tokens, **kwargs)
        return _decode_hf(
            model,
            prefill=prefill,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )
    if decode_route == "fused":
        kwargs.pop("debug", None)
        eos_token_check_interval = kwargs.pop("eos_token_check_interval", 16)
        eos_token_ids = kwargs.pop("eos_token_ids", None)
        ignore_eos = kwargs.pop("ignore_eos", False)
        fused_kw = dict(
            eos_token_check_interval=eos_token_check_interval,
            eos_token_ids=eos_token_ids,
            ignore_eos=ignore_eos,
            **kwargs,
        )
        if decode_time_enabled:
            return _timed_decode("fused", _decode_fused, model, prefill=prefill, max_new_tokens=max_new_tokens, **fused_kw)
        return _decode_fused(
            model,
            prefill=prefill,
            max_new_tokens=max_new_tokens,
            **fused_kw,
        )
    if decode_route == "python_vm":
        debug = kwargs.pop("debug", False)
        instruction_dump_path = kwargs.pop("instruction_dump_path", None)
        eos_token_check_interval = kwargs.pop("eos_token_check_interval", 16)
        eos_token_ids = kwargs.pop("eos_token_ids", None)
        ignore_eos = kwargs.pop("ignore_eos", False)
        if max_new_tokens > 0:
            pyvm_kw = dict(
                eos_token_check_interval=eos_token_check_interval,
                eos_token_ids=eos_token_ids,
                ignore_eos=ignore_eos,
                debug=debug,
                **kwargs,
            )
            if decode_time_enabled:
                return _timed_decode("python_vm", _decode_python_vm, model, prefill=prefill, max_new_tokens=max_new_tokens, **pyvm_kw)
            return _decode_python_vm(
                model,
                prefill=prefill,
                max_new_tokens=max_new_tokens,
                **pyvm_kw,
            )
        if instruction_dump_path is None and debug:
            instruction_dump_path = "instruction_queues.txt"
        globs: Any = llm_preprocess_decode(
            model,
            prefill=prefill,
            debug=debug,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )
        schedule = _get_decode_schedule(
            model,
            kwargs.get("hardware", "A100"),
            batch_size=int(prefill.input_ids.shape[0]),
        )
        queues = schedule.scheduler.build_instruction_queues(globs)
        schedule.scheduler.tensorize_instructions(globs, queues)
        if debug and isinstance(globs.instructions, torch.Tensor):
            print(f"instructions: {_tensor_info(globs.instructions)}")
        if instruction_dump_path:
            _dump_instruction_queues(instruction_dump_path, queues)
        return {"globals": globs}
    if decode_route == "mk":
        debug = kwargs.pop("debug", False)
        mk_dir = kwargs.pop("mk_dir", None)
        eos_token_check_interval = kwargs.pop("eos_token_check_interval", 16)
        eos_token_ids = kwargs.pop("eos_token_ids", None)
        ignore_eos = kwargs.pop("ignore_eos", False)
        if max_new_tokens > 0:
            return _decode_mk(
                model,
                prefill=prefill,
                max_new_tokens=max_new_tokens,
                mk_dir=mk_dir,
                eos_token_check_interval=eos_token_check_interval,
                eos_token_ids=eos_token_ids,
                ignore_eos=ignore_eos,
                debug=debug,
                decode_time=decode_time_enabled,
                **kwargs,
            )
        globs: Any = llm_preprocess_decode(
            model,
            prefill=prefill,
            debug=debug,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )
        schedule = _get_decode_schedule(
            model,
            kwargs.get("hardware", "A100"),
            batch_size=int(prefill.input_ids.shape[0]),
        )
        if not getattr(schedule.package, "STATIC_MEGAKERNEL", False):
            queues = schedule.scheduler.build_instruction_queues(globs)
            schedule.scheduler.tensorize_instructions(globs, queues)
        if debug and isinstance(globs.instructions, torch.Tensor):
            print(f"instructions: {_tensor_info(globs.instructions)}")
        return {"globals": globs}
    raise NotImplementedError(f"decode route not implemented: {decode_route}")
