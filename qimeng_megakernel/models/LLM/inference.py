from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoTokenizer

from qimeng_megakernel.models.LLM.prefill import (
    InferenceConfig,
    PrefillOutput,
    run_prefill,
    run_prefill_fused,
    _dtype_from_str,
    get_or_load_hf_model,
)
from qimeng_megakernel.models.LLM.preprocess import (
    preprocess_decode as llm_preprocess_decode,
)
from qimeng_megakernel.models.LLM.decode import decode as llm_decode


@dataclass
class InferenceOutput:
    prefill: PrefillOutput
    decode_prep: Any | None = None
    decode_output: Any | None = None


def _tensor_info(t: torch.Tensor) -> str:
    return (
        f"shape={tuple(t.shape)} dtype={t.dtype} device={t.device} stride={t.stride()}"
    )


def _get_num_layers(past_key_values) -> int:
    if past_key_values is None:
        return 0
    if isinstance(past_key_values, (tuple, list)):
        return len(past_key_values)
    if hasattr(past_key_values, "key_cache"):
        return len(past_key_values.key_cache)
    return 0


def _get_layer_kv(past_key_values, layer_idx: int):
    if past_key_values is None:
        return None, None
    if isinstance(past_key_values, (tuple, list)):
        layer = past_key_values[layer_idx]
        if isinstance(layer, (tuple, list)) and len(layer) >= 2:
            return layer[0], layer[1]
        return None, None
    if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        return past_key_values.key_cache[layer_idx], past_key_values.value_cache[layer_idx]
    return None, None


def run_inference(
    config: InferenceConfig,
    prompts: list[str],
    *,
    enable_decode: bool = False,
    debug: bool = False,
    debug_print_layers: int = 2,
    debug_print_tokens: bool = False,
    debug_max_new_tokens: int = 0,
    **decode_kwargs,
) -> InferenceOutput:
    """
    Entry-point for LLM inference.
    - Always runs prefill.
    - Optionally runs decode preprocess + decode (when implemented).
    - When decode_route is \"hf\", decode uses greedy (do_sample=False); no sampling params
      so sample-based warnings are avoided and results are deterministic.
    """
    loaded_model = None
    non_hf_routes = {"fused", "python_vm", "mk"}
    decode_route_pre = decode_kwargs.get("decode_route", "hf")
    # For fused/python_vm/mk we use non-HF prefill flow; only HF decode route preloads HF model.
    if enable_decode and decode_route_pre not in non_hf_routes:
        torch_dtype = _dtype_from_str(config.torch_dtype)
        loaded_model = get_or_load_hf_model(
            config.model,
            device=config.device,
            torch_dtype=torch_dtype,
        )
    hardware_pre = decode_kwargs.get("hardware", "A100")
    if enable_decode and decode_route_pre in non_hf_routes:
        try:
            prefill = run_prefill_fused(
                config=config,
                prompts=prompts,
                loaded_model=loaded_model,
                hardware=hardware_pre,
                max_new_tokens=debug_max_new_tokens if debug_max_new_tokens > 0 else None,
            )
        except NotImplementedError as exc:
            raise NotImplementedError(
                "Non-HF decode routes require fused prefill and must not fallback to HF prefill. "
                f"route={decode_route_pre}, hardware={hardware_pre}, model={config.model}."
            ) from exc
    else:
        prefill = run_prefill(config=config, prompts=prompts, loaded_model=loaded_model)
    if decode_route_pre == "mk":
        print("[mk] Prefill done; entering decode path.", flush=True)
    if debug or debug_print_tokens:
        tokenizer = None
        if debug_print_tokens:
            tokenizer = AutoTokenizer.from_pretrained(config.model)
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "right"
        _debug_print_prefill(
            prefill,
            print_layers=debug_print_layers,
            print_tokens=debug_print_tokens,
            tokenizer=tokenizer,
            max_new_tokens=debug_max_new_tokens,
            full_debug=debug,
        )

    if not enable_decode:
        if debug:
            decode_prep = llm_preprocess_decode(
                config.model,
                prefill=prefill,
                debug=debug,
                max_new_tokens=debug_max_new_tokens if debug_max_new_tokens > 0 else None,
                **decode_kwargs,
            )
            return InferenceOutput(prefill=prefill, decode_prep=decode_prep)
        return InferenceOutput(prefill=prefill)

    decode_route = decode_kwargs.get("decode_route", "hf")
    if decode_route == "mk":
        print("[mk] Running decode (mk path).", flush=True)
    decode_call_kwargs = dict(decode_kwargs)
    if decode_route != "hf" and loaded_model is not None:
        decode_call_kwargs["loaded_model"] = loaded_model
    decode_out = llm_decode(
        config.model,
        prefill=prefill,
        max_new_tokens=debug_max_new_tokens,
        debug=debug,
        **decode_call_kwargs,
    )
    return InferenceOutput(
        prefill=prefill,
        decode_output=decode_out,
    )


def _debug_print_prefill(
    prefill: PrefillOutput,
    *,
    print_layers: int = 2,
    print_tokens: bool = False,
    tokenizer: Any | None = None,
    max_new_tokens: int = 0,
    full_debug: bool = True,
):
    if full_debug:
        print("=== Prefill Debug ===")
        print(f"batch_size={prefill.input_ids.shape[0]}")
        print(f"input_ids: {_tensor_info(prefill.input_ids)}")
        print(f"attention_mask: {_tensor_info(prefill.attention_mask)}")
        print(f"logits: {_tensor_info(prefill.logits)}")
        print(f"prompt_lens={prefill.prompt_lens.tolist()}")
        B = int(prefill.input_ids.shape[0])
        fg = prefill.first_gen_token.reshape(-1).tolist()
        p1 = prefill.pos_id_token_1.reshape(-1).tolist()
        kvb = prefill.kv_seq_len_before.reshape(-1).tolist()
        kva = prefill.kv_seq_len_after.reshape(-1).tolist()
        print("per-batch prefill scalars (first token / positions / KV lengths):")
        for i in range(B):
            print(
                f"  batch[{i}] first_gen_token={fg[i]} pos_id_token_1={p1[i]} "
                f"kv_seq_len_before={kvb[i]} kv_seq_len_after={kva[i]}"
            )
        if max_new_tokens > 0:
            planned_kv_seq_len_after = prefill.kv_seq_len_before + max_new_tokens
            print("=== Planned decode length ===")
            print(f"max_new_tokens={max_new_tokens}")
            print(f"planned_kv_seq_len_after={planned_kv_seq_len_after.tolist()}")
        n_layers = _get_num_layers(prefill.past_key_values)
        print(f"past_key_values layers: {n_layers}")
        k0, _ = _get_layer_kv(prefill.past_key_values, 0)
        if isinstance(k0, torch.Tensor) and k0.ndim >= 3:
            cached_seq_len = k0.shape[-2]
            print(f"cached_seq_len (from layer0.k.shape[-2]) = {cached_seq_len}")
    elif print_tokens:
        print("=== Prefill (tokens only) ===")
        print(f"batch_size={prefill.input_ids.shape[0]}  prompt_lens={prefill.prompt_lens.tolist()}")
    if print_tokens:
        if tokenizer is None:
            print("tokenizer: <unavailable>")
            return
        input_ids = prefill.input_ids.tolist()
        attention_mask = prefill.attention_mask.tolist()
        prompt_lens = prefill.prompt_lens.tolist()
        print("input_text (per batch):")
        for i, ids in enumerate(input_ids):
            prompt_len = int(prompt_lens[i])
            ids_prompt = ids[:prompt_len]
            decoded_prompt = tokenizer.decode(ids_prompt, skip_special_tokens=False)
            print(f"  batch[{i}] prompt_len={prompt_len} text={repr(decoded_prompt)}")
            if prompt_len < len(ids):
                pad_ids = ids[prompt_len:]
                pad_mask = attention_mask[i][prompt_len:]
                print(f"  batch[{i}] pad_ids={pad_ids} pad_mask={pad_mask}")
        first_tokens = prefill.first_gen_token.reshape(-1).tolist()
        print("first_gen_token (per batch):")
        for i, tok_id in enumerate(first_tokens):
            tok_text = tokenizer.decode([tok_id], skip_special_tokens=False)
            print(f"  batch[{i}] id={tok_id} text={repr(tok_text)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LLM inference debug entry")
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="HF model id",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--prompt",
        type=str,
        default="Explain KV cache shapes in one paragraph.",
    )
    parser.add_argument(
        "--max-input-len",
        type=int,
        default=None,
        help="Max prompt length (truncate if provided)",
    )
    parser.add_argument(
        "--torch-dtype",
        type=str,
        default="auto",
        help='One of: "auto", "float16", "bfloat16", "float32"',
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug prints")
    parser.add_argument(
        "--print-layers",
        type=int,
        default=2,
        help="How many layers to print KV shapes (0=none, -1=all)",
    )
    parser.add_argument(
        "--print-tokens",
        action="store_true",
        help="Print per-batch input/first-token ids and tokens",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=0,
        help="Planned max generated tokens (for KV cache sizing)",
    )
    args = parser.parse_args()

    prompts = [args.prompt for _ in range(args.batch_size)]
    cfg = InferenceConfig(
        model=args.model,
        device=args.device,
        torch_dtype=args.torch_dtype,
        max_input_len=args.max_input_len,
    )
    out = run_inference(
        cfg,
        prompts,
        enable_decode=False,
        debug=args.debug,
        debug_print_layers=args.print_layers,
        debug_print_tokens=args.print_tokens,
        debug_max_new_tokens=args.max_new_tokens,
    )
