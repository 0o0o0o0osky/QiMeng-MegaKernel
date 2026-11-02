# H100 (Hopper) fused prefill for Qwen2.5 3B: tokenize + our fused ops only (no HF forward).
# Layout matches H100 preprocess: KV interleaved, attn intermediates capped at min(sm_count, 132).
# Weights/tokenizer: from disk when possible (local dir or HF cache path); only load full HF when no disk path.
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from qimeng_megakernel.models.LLM.prefill import InferenceConfig, PrefillOutput
from qimeng_megakernel.models.LLM.llama.preprocess import _interleave_k_cache
from qimeng_megakernel.models.LLM.qwen.preprocess import _stack_qwen2_weights_from_hf
from qimeng_megakernel.models.LLM.qwen import load_weights as qwen_load_weights
from qimeng_megakernel.models.LLM.qwen.qwen2_5__3b.H100.single_batch.decode_schedule.globs import Globals
from qimeng_megakernel.models.LLM.qwen.qwen2_5__3b.H100.single_batch.decode_schedule.model_config import Qwen2Config
from qimeng_megakernel.models.LLM.qwen.qwen2_5__3b.H100.single_batch.decode_schedule.fused import layer_scheduler as fused_layer
from qimeng_megakernel.models.LLM.qwen.qwen2_5__3b.H100.single_batch.decode_schedule.fused import operations as fused_ops


def _get_sm_count(device):
    if getattr(device, "type", None) != "cuda":
        return 1
    try:
        return torch.cuda.get_device_properties(device).multi_processor_count
    except Exception:
        return 1


def _build_globs_for_prefill_from_weights(
    weights: dict[str, torch.Tensor],
    config: dict[str, Any],
    prompt_len: int,
    device: torch.device,
    dtype: torch.dtype,
    max_new_tokens: int | None = None,
) -> Globals:
    """Build H100 Globals from pre-loaded weight dict. Layout matches H100 preprocess (max_attn cap 132)."""
    Qwen2Config.update_from_dict(config)
    nlayers = config["num_hidden_layers"]
    nheads = config["num_attention_heads"]
    nkv = config.get("num_key_value_heads", nheads)
    head_dim = config.get("head_dim") or (config["hidden_size"] // nheads)
    hidden_size = config["hidden_size"]
    intermediate_size = config["intermediate_size"]
    vocab_size = config["vocab_size"]

    kv_seq_len = prompt_len + (max_new_tokens or 0)
    k_cache = torch.zeros(
        (nlayers, 1, kv_seq_len, nkv, head_dim),
        device=device,
        dtype=dtype,
    )
    k_cache = _interleave_k_cache(k_cache, head_dim).contiguous()
    v_cache = torch.zeros(
        (nlayers, 1, kv_seq_len, nkv, head_dim),
        device=device,
        dtype=dtype,
    ).contiguous()

    max_attn = min(_get_sm_count(device), 132)

    def buf(s, dt=dtype):
        return torch.zeros(s, device=device, dtype=dt)

    def buf32(s):
        return torch.zeros(s, device=device, dtype=torch.float32)

    return Globals(
        position_id=0,
        k_cache=k_cache,
        v_cache=v_cache,
        qkv_weights=weights["qkv_weights"].contiguous(),
        qkv_bias=weights.get("qkv_bias"),
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
        hidden_states=buf(hidden_size),
        q_post_rope=buf(hidden_size),
        attn_out=buf(hidden_size),
        attn_out_intermediates=buf32([nheads, max_attn, head_dim]),
        attn_lse_intermediates=buf32([nheads, max_attn]),
        silu_out=buf(intermediate_size),
        logits=buf(vocab_size),
        barriers=torch.zeros([nlayers, 10, nheads + nkv * 2], dtype=torch.int32, device=device),
        hidden_states_seq=torch.zeros((prompt_len, hidden_size), device=device, dtype=dtype),
        prefill_q_seq=torch.zeros((prompt_len, nheads, head_dim), device=device, dtype=dtype),
        attn_out_seq=torch.zeros((prompt_len, nheads * head_dim), device=device, dtype=dtype),
    )


@torch.inference_mode()
def run_fused_prefill(
    config: InferenceConfig,
    prompts: list[str],
    *,
    loaded_model: Any = None,
    tokenizer: Any = None,
    hardware: str = "H100",
    max_new_tokens: int | None = None,
    trace_memory: bool = False,
) -> PrefillOutput:
    """
    Run prefill with fused ops only (no HF forward). Prefer loading weights from disk
    (local dir or HF cache path); only load full HF model when no such path is available.
    Builds H100 globs (max_attn cap 132), runs prefill layers, returns PrefillOutput with globs.
    """
    from qimeng_megakernel.models.LLM.prefill import _dtype_from_str, get_or_load_hf_model

    torch.cuda.set_device(config.device)
    device = torch.device(config.device)

    def _can_load_from_disk(model_path: str) -> bool:
        p = Path(model_path)
        return p.exists() and (p / "config.json").exists()

    def _resolve_to_disk_path(model_id: str) -> str | None:
        """If model_id is a local dir or HF id already in cache, return path with config.json; else None."""
        if _can_load_from_disk(model_id):
            return model_id
        try:
            from huggingface_hub import snapshot_download
            return snapshot_download(model_id, local_files_only=True)
        except Exception:
            return None

    # Resolve once: use disk path for both tokenizer and weights when available
    disk_path = _resolve_to_disk_path(config.model)
    if tokenizer is None:
        from transformers import AutoTokenizer
        load_path = disk_path if disk_path is not None else config.model
        kw = {"local_files_only": True} if disk_path is not None else {}
        tokenizer = AutoTokenizer.from_pretrained(load_path, **kw)
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
    input_ids = tok["input_ids"].to(device)
    attention_mask = tok["attention_mask"].to(device)

    batch_size, prompt_len = input_ids.shape[0], input_ids.shape[1]
    if batch_size != 1:
        raise NotImplementedError("fused prefill only supports batch size 1")

    dtype = _dtype_from_str(config.torch_dtype)
    if dtype == "auto":
        dtype = torch.bfloat16

    if disk_path is not None and _can_load_from_disk(disk_path):
        config_dict, weights = qwen_load_weights.load_qwen2_weights_from_disk(
            disk_path, device=device, dtype=dtype
        )
    else:
        model = get_or_load_hf_model(config.model, device=config.device, torch_dtype=dtype)
        weights = _stack_qwen2_weights_from_hf(model)
        weights["embed_tokens"] = model.model.embed_tokens.weight.detach().clone()
        config_dict = {
            "num_hidden_layers": model.config.num_hidden_layers,
            "hidden_size": model.config.hidden_size,
            "intermediate_size": model.config.intermediate_size,
            "num_attention_heads": model.config.num_attention_heads,
            "num_key_value_heads": getattr(model.config, "num_key_value_heads", model.config.num_attention_heads),
            "vocab_size": model.config.vocab_size,
            "rms_norm_eps": getattr(model.config, "rms_norm_eps", 1e-6),
            "head_dim": getattr(model.config, "head_dim", model.config.hidden_size // model.config.num_attention_heads),
        }
        del model
        torch.cuda.empty_cache()

    globs = _build_globs_for_prefill_from_weights(
        weights, config_dict, prompt_len, device, dtype, max_new_tokens=max_new_tokens
    )
    globs.embed_tokens = weights["embed_tokens"]
    globs.hidden_states_seq.copy_(
        weights["embed_tokens"][input_ids.squeeze(0)].to(globs.hidden_states_seq.dtype)
    )

    for layer_idx in range(Qwen2Config.num_hidden_layers):
        fused_layer.run_fused_prefill_layer(globs, layer_idx, prompt_len)

    globs.hidden_states.copy_(globs.hidden_states_seq[prompt_len - 1].to(globs.hidden_states.dtype))
    fused_ops.fused_rms_lmhead_gemv(globs)

    logits = globs.logits.unsqueeze(0).unsqueeze(0)
    prompt_lens = attention_mask.sum(dim=-1)
    first_gen_token = torch.argmax(globs.logits, dim=-1).unsqueeze(0).unsqueeze(-1)
    pos_id_token_1 = prompt_lens.to(dtype=torch.long).unsqueeze(1)
    kv_seq_len_before = prompt_lens.to(dtype=torch.long)
    kv_seq_len_after = kv_seq_len_before + 1
    past_key_values = (globs.k_cache, globs.v_cache)

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
        globs=globs,
    )
