"""Small helpers used by LLM end-to-end greedy generation tests."""
from __future__ import annotations

import math
import re
from numbers import Real
from typing import Any, Sequence

import torch

from .tolerance import E2E_FORK_TOPK_K


E2E_REPETITION_MIN_PERIOD = 6
E2E_REPETITION_MAX_PERIOD = 4096
E2E_REPETITION_MIN_LEN = 24


def resolve_eos_token_ids(tokenizer, model=None) -> list[int]:
    """Collect distinct EOS-like token IDs exposed by the tokenizer and model."""
    eos_ids: list[int] = []
    if getattr(tokenizer, "eos_token_id", None) is not None:
        eos_ids.append(int(tokenizer.eos_token_id))
    if model is not None and hasattr(model, "config"):
        cfg_eos = getattr(model.config, "eos_token_id", None)
        if cfg_eos is not None:
            if isinstance(cfg_eos, (list, tuple)):
                eos_ids.extend(int(token_id) for token_id in cfg_eos)
            else:
                eos_ids.append(int(cfg_eos))
    eot_token_id = getattr(tokenizer, "eot_token_id", None)
    if eot_token_id is not None:
        eos_ids.append(int(eot_token_id))
    return list(dict.fromkeys(eos_ids))


def has_gibberish(text: str) -> bool:
    """Return whether decoded text is empty, malformed, or control-byte heavy."""
    if not text.strip() or "\ufffd" in text:
        return True
    bad = sum(1 for char in text if ord(char) < 32 and char not in "\n\r\t")
    return bad / max(len(text), 1) > 0.1


def repetition_period(text: str) -> int | None:
    """Return the smallest period repeated at least three times consecutively."""
    compact = re.sub(r"\s+", " ", text.strip())
    if len(compact) < E2E_REPETITION_MIN_LEN:
        return None
    data = compact.encode("utf-8", "surrogatepass")
    length = len(data)
    max_period = min(E2E_REPETITION_MAX_PERIOD, length // 3)
    if max_period < E2E_REPETITION_MIN_PERIOD:
        return None
    whole = int.from_bytes(data, "big")
    for width in range(E2E_REPETITION_MIN_PERIOD, max_period + 1):
        head = whole >> (8 * width)
        tail = whole & ((1 << (8 * (length - width))) - 1)
        xor = (head ^ tail).to_bytes(length - width, "big")
        if b"\x00" * (2 * width) in xor:
            return width
    return None


def has_repetition(text: str) -> bool:
    """Return whether text contains a consecutive three-copy repetition."""
    return repetition_period(text) is not None


def _topk_logits_entries(logits: torch.Tensor, k: int) -> list[dict[str, Any]]:
    if k not in (5, 10):
        raise ValueError("fork top-k must be 5 or 10")
    values, indices = torch.topk(logits.float().view(-1), k)
    return [
        {"rank": rank + 1, "token_id": int(indices[rank]), "logit": float(values[rank])}
        for rank in range(k)
    ]


def _topk_distribution(entries: Sequence[dict[str, Any]]) -> dict[int, float]:
    if len(entries) not in (5, 10):
        raise ValueError("fork logits must contain exactly 5 or 10 entries")
    token_ids: list[int] = []
    logits: list[float] = []
    for expected_rank, entry in enumerate(entries, start=1):
        rank = entry.get("rank")
        token_id = entry.get("token_id")
        logit = entry.get("logit")
        if rank != expected_rank:
            raise ValueError("fork logits ranks must be consecutive and one-based")
        if isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0:
            raise ValueError("fork logits token_id must be a non-negative int")
        if token_id in token_ids:
            raise ValueError("fork logits token_id values must be unique")
        if isinstance(logit, bool) or not isinstance(logit, Real) or not math.isfinite(logit):
            raise ValueError("fork logits must be finite non-bool numbers")
        token_ids.append(token_id)
        logits.append(float(logit))
    if any(left < right for left, right in zip(logits, logits[1:])):
        raise ValueError("fork logits must be sorted by descending logit")
    probabilities = torch.softmax(torch.tensor(logits, dtype=torch.float64), dim=0)
    return {
        token_id: float(probabilities[index])
        for index, token_id in enumerate(token_ids)
    }


def fork_analysis(mk_logits: list[dict], hf_logits: list[dict]) -> dict[str, Any]:
    """Compare two top-5 or top-10 logit tables at the first decode fork."""
    if len(mk_logits) != len(hf_logits):
        raise ValueError("fork logit tables must use the same k")
    mk_probabilities = _topk_distribution(mk_logits)
    hf_probabilities = _topk_distribution(hf_logits)
    mk_ids = set(mk_probabilities)
    hf_ids = set(hf_probabilities)
    overlap = len(mk_ids & hf_ids)
    probability_overlap = sum(
        min(mk_probabilities.get(token_id, 0.0), hf_probabilities.get(token_id, 0.0))
        for token_id in mk_ids | hf_ids
    )
    return {
        "mk_top1": mk_logits[0],
        "hf_top1": hf_logits[0],
        "topk_k": len(mk_logits),
        "topk_overlap": overlap,
        "topk_probability_overlap": probability_overlap,
        "mk_top1_in_hf_topk": mk_logits[0]["token_id"] in hf_ids,
        "hf_top1_in_mk_topk": hf_logits[0]["token_id"] in mk_ids,
        "mk_margin": mk_logits[0]["logit"] - mk_logits[1]["logit"],
        "hf_margin": hf_logits[0]["logit"] - hf_logits[1]["logit"],
    }


@torch.inference_mode()
def mk_decode_via_central(
    *,
    model_path: str,
    prefill,
    max_new_tokens: int,
    hardware: str,
    decode_time: bool = False,
    ignore_eos: bool = False,
    eos_token_ids: list[int] | None = None,
) -> dict[str, Any]:
    """Run MK greedy decode through the central LLM decode entry point."""
    from qimeng_megakernel.models.LLM.decode import decode as llm_decode

    decode_kw: dict[str, Any] = {
        "ignore_eos": ignore_eos,
        "decode_time": decode_time,
    }
    if eos_token_ids is not None:
        decode_kw["eos_token_ids"] = eos_token_ids
    out = llm_decode(
        model_path,
        prefill=prefill,
        max_new_tokens=max_new_tokens,
        decode_route="mk",
        hardware=hardware,
        **decode_kw,
    )
    prompt_len = int(prefill.input_ids.shape[1])
    generated = out["generated_ids"]
    new_ids = generated[0, prompt_len : prompt_len + max_new_tokens].tolist()
    return {
        "new_token_ids": new_ids,
        "full_ids": generated[0].tolist(),
        "prompt_len": prompt_len,
        "timing": out.get("timing") or {},
        "decode_out": out,
    }


@torch.inference_mode()
def mk_step_logits_at_fork(
    *,
    model_path: str,
    prefill,
    hardware: str,
    fork_step: int,
    topk_k: int = E2E_FORK_TOPK_K,
    shared_prefix_token_ids: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    """Replay a shared prefix and return MK top-k logits at one decode step."""
    from qimeng_megakernel.models.LLM.decode import _get_decode_schedule

    if fork_step < 0:
        raise ValueError("fork_step must be non-negative")
    if fork_step == 0:
        return _topk_logits_entries(prefill.logits[0, -1], k=topk_k)

    globs = prefill.globs
    if globs is None:
        raise RuntimeError("prefill.globs required for mk_step_logits_at_fork")

    prompt_len = int(prefill.input_ids.shape[1])
    target_seq_len = prompt_len + fork_step
    if globs.k_cache.shape[2] < target_seq_len:
        from qimeng_megakernel.models.LLM.preprocess import _pad_kv_cache_to_len

        globs.k_cache, globs.v_cache = _pad_kv_cache_to_len(
            globs.k_cache, globs.v_cache, target_seq_len=target_seq_len
        )

    schedule = _get_decode_schedule(model_path, hardware, batch_size=1)
    if schedule.get_mk is None or schedule.call_mk is None or schedule.scheduler is None:
        raise NotImplementedError("MK schedule unavailable")
    mk = schedule.get_mk(None)
    queues = schedule.scheduler.build_instruction_queues(globs)
    schedule.scheduler.tensorize_instructions(globs, queues)

    def call_mk(g) -> None:
        g.barriers.zero_()
        schedule.call_mk(g, mk)

    output_tokens = prefill.input_ids.new_zeros(1, prompt_len + fork_step)
    output_tokens[0, :prompt_len] = prefill.input_ids[0]
    output_tokens[0, prompt_len] = prefill.first_gen_token.squeeze()
    if shared_prefix_token_ids is not None:
        if len(shared_prefix_token_ids) != fork_step:
            raise ValueError(
                "shared_prefix_token_ids must contain exactly fork_step tokens"
            )
        output_tokens[0, prompt_len:] = torch.as_tensor(
            shared_prefix_token_ids,
            device=output_tokens.device,
            dtype=output_tokens.dtype,
        )

    embed = globs.embed_tokens
    dtype = globs.hidden_states.dtype
    step_logits: list[dict[str, Any]] | None = None
    for index in range(fork_step):
        position = prompt_len + index
        token_ids = output_tokens[:, position]
        embedding = embed[token_ids].to(dtype)
        globs.hidden_states.copy_(
            embedding.reshape(-1) if globs.hidden_states.ndim == 1 else embedding
        )
        globs.position_id = position
        call_mk(globs)
        if index == fork_step - 1:
            step_logits = _topk_logits_entries(globs.logits, k=topk_k)
            break
        if shared_prefix_token_ids is None:
            logits = globs.logits
            next_id = logits.argmax() if logits.dim() == 1 else logits.argmax(dim=-1)
            output_tokens[0, position + 1] = next_id

    if step_logits is None:
        raise RuntimeError(f"fork_step {fork_step} out of range")
    return step_logits


@torch.inference_mode()
def hf_greedy_decode(
    *,
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    add_special_tokens: bool = False,
) -> dict[str, Any]:
    """Run a Hugging Face greedy reference for an end-to-end generation test."""
    device = next(model.parameters()).device
    tokenized = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=add_special_tokens,
    )
    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    else:
        attention_mask = attention_mask.to(device)
    prompt_len = int(input_ids.shape[1])
    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
    )
    generated = output[0].tolist()
    return {
        "tokens": generated,
        "text": tokenizer.decode(generated, skip_special_tokens=True),
        "prompt_len": prompt_len,
    }


__all__ = [
    "fork_analysis",
    "has_gibberish",
    "has_repetition",
    "hf_greedy_decode",
    "mk_decode_via_central",
    "mk_step_logits_at_fork",
    "resolve_eos_token_ids",
]
