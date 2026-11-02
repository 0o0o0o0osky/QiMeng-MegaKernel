"""Run the shared fixed-length MK quality/performance matrix.

The paper-comparable default is the raw ``Write a long story.`` prompt.  Chat
formatting is an explicit, model-neutral option.  Exact token replay and repetition
are diagnostics: unordered fast kernels and the reference model itself may diverge or
repeat.  The common local quality gate is intentionally weak: the requested length
must be complete and the decoded text must not be gibberish.  Repetition is reported,
but it does not trigger an external-framework attribution requirement or fail this
local gate.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from statistics import fmean
from typing import Any, cast

import torch
from transformers import AutoTokenizer

from qimeng_megakernel.models.LLM.decode import decode as llm_decode
from qimeng_megakernel.models.LLM.inference import InferenceConfig
from qimeng_megakernel.models.LLM.prefill import run_prefill_fused
from qimeng_megakernel.verifier.common.generation import (
    has_gibberish,
    repetition_period,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark QiMeng-MegaKernel MK over batch/output matrix")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--prompt", default="Write a long story.")
    parser.add_argument("--batch-sizes", default="1")
    parser.add_argument("--output-lengths", default="128,256,512,1024")
    parser.add_argument("--out", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hardware", default="H100")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--warmup-iters", type=int, default=5)
    parser.add_argument("--bench-iters", type=int, default=10)
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Apply the tokenizer chat template. Raw completion is the default.",
    )
    parser.add_argument(
        "--respect-eos",
        action="store_true",
        help="Stop at EOS. The fixed-length performance matrix forces continuation by default.",
    )
    return parser.parse_args()


def _parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _prepare_prompt(
    args: argparse.Namespace,
    tokenizer: Any,
) -> tuple[str, bool]:
    if not args.chat:
        return args.prompt, True
    formatted = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return str(formatted), False


def _run_once(
    args: argparse.Namespace,
    batch_size: int,
    output_len: int,
    *,
    prompt: str,
    tokenizer: Any,
    add_special_tokens: bool,
    capture_output: bool,
) -> dict[str, Any]:
    print(
        f"[mk-bench] model_path={args.model_path} batch={batch_size} max_new_tokens={output_len}",
        flush=True,
    )
    config = InferenceConfig(
        model=args.model_path,
        device=args.device,
        torch_dtype=args.torch_dtype,
        max_input_len=None,
        add_special_tokens=add_special_tokens,
    )
    prompts = [prompt] * batch_size
    print("[mk-bench] start_prefill_fused", flush=True)
    prefill = run_prefill_fused(
        config=config,
        prompts=prompts,
        hardware=args.hardware,
        max_new_tokens=output_len,
    )
    print("[mk-bench] start_decode_mk", flush=True)
    wall_t0 = time.perf_counter()
    decode_out = cast(
        dict[str, Any],
        llm_decode(
            args.model_path,
            prefill=prefill,
            max_new_tokens=output_len,
            decode_route="mk",
            hardware=args.hardware,
            decode_time=True,
            ignore_eos=not args.respect_eos,
        ),
    )
    elapsed_wall_s = time.perf_counter() - wall_t0
    per_row = [int(x) for x in decode_out["per_row_new_tokens"].tolist()]
    total_tokens = int(sum(per_row))
    timing = cast(dict[str, Any], decode_out.get("timing") or {})
    decode_wall_s = float(timing.get("decode_wall_s", elapsed_wall_s))
    decode_gpu_s = float(timing.get("decode_gpu_s", 0.0))
    result: dict[str, Any] = {
        "backend": "mk",
        "batch_size": batch_size,
        "max_new_tokens": output_len,
        "decode_time_s": decode_wall_s,
        "decode_gpu_s": decode_gpu_s,
        "effective_new_tokens": total_tokens,
        "tok_s": float(timing.get("effective_tok_s_wall", (total_tokens / decode_wall_s) if decode_wall_s > 0 else 0.0)),
        "tok_s_gpu": float(timing.get("effective_tok_s_gpu", (total_tokens / decode_gpu_s) if decode_gpu_s > 0 else 0.0)),
        "kernel_tok_s": float(timing.get("kernel_tok_s", 0.0)),
        "mk_kernel_gpu_s": float(timing.get("mk_kernel_gpu_s", 0.0)),
        "non_kernel_gpu_s": float(timing.get("non_kernel_gpu_s", 0.0)),
        "non_kernel_wall_s": float(timing.get("non_kernel_wall_s", 0.0)),
        "instruction_setup_wall_s": float(timing.get("instruction_setup_wall_s", 0.0)),
        "queue_build_wall_s": float(timing.get("queue_build_wall_s", 0.0)),
        "tensorize_wall_s": float(timing.get("tensorize_wall_s", 0.0)),
        "clock_domain": "wall",
        "per_row_new_tokens": per_row,
    }
    if capture_output:
        new_tokens = decode_out["new_tokens"].detach().cpu()
        if new_tokens.ndim == 1:
            new_tokens = new_tokens.unsqueeze(0)
        token_rows = [[int(token) for token in row.tolist()] for row in new_tokens]
        text_rows = [
            tokenizer.decode(row, skip_special_tokens=True) for row in token_rows
        ]
        result["new_token_ids"] = token_rows
        result["new_tokens_text"] = text_rows
        quality = []
        for text in text_rows:
            period = repetition_period(text)
            quality.append(
                {
                    "has_gibberish": has_gibberish(text),
                    "has_repetition": period is not None,
                    "repetition_period": period,
                }
            )
        result["quality"] = quality
    return result


def main() -> int:
    args = _parse_args()
    if args.warmup_iters < 0:
        raise ValueError("--warmup-iters must be >= 0")
    if args.bench_iters < 1:
        raise ValueError("--bench-iters must be >= 1")
    batch_sizes = _parse_int_list(args.batch_sizes)
    output_lengths = _parse_int_list(args.output_lengths)
    if not batch_sizes or any(value <= 0 for value in batch_sizes):
        raise ValueError("--batch-sizes must contain positive integers")
    if not output_lengths or any(value <= 0 for value in output_lengths):
        raise ValueError("--output-lengths must contain positive integers")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        local_files_only=Path(args.model_path).expanduser().is_dir(),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    formatted_prompt, add_special_tokens = _prepare_prompt(args, tokenizer)
    prompt_token_count = len(
        tokenizer(
            formatted_prompt,
            add_special_tokens=add_special_tokens,
        )["input_ids"]
    )
    results: dict[str, dict[str, Any]] = {}
    for batch_size in batch_sizes:
        for output_len in output_lengths:
            for _ in range(args.warmup_iters):
                _run_once(
                    args,
                    batch_size,
                    output_len,
                    prompt=formatted_prompt,
                    tokenizer=tokenizer,
                    add_special_tokens=add_special_tokens,
                    capture_output=False,
                )
            samples = [
                _run_once(
                    args,
                    batch_size,
                    output_len,
                    prompt=formatted_prompt,
                    tokenizer=tokenizer,
                    add_special_tokens=add_special_tokens,
                    capture_output=True,
                )
                for _ in range(args.bench_iters)
            ]
            captured = samples[0]
            token_samples = [
                cast(list[list[int]], sample["new_token_ids"])
                for sample in samples
            ]
            greedy_rerun_match = all(
                token_rows == token_samples[0] for token_rows in token_samples[1:]
            )
            length_complete = all(
                cast(list[int], sample["per_row_new_tokens"])
                == [output_len] * batch_size
                for sample in samples
            )
            captured_quality = cast(list[dict[str, Any]], captured["quality"])
            all_quality = [
                row
                for sample in samples
                for row in cast(list[dict[str, Any]], sample["quality"])
            ]
            has_any_gibberish = any(
                bool(row["has_gibberish"]) for row in all_quality
            )
            has_any_repetition = any(
                bool(row["has_repetition"]) for row in all_quality
            )
            timing_samples = [
                {
                    key: value
                    for key, value in sample.items()
                    if key not in {"new_token_ids", "new_tokens_text", "quality"}
                }
                for sample in samples
            ]
            mean_wall = fmean(float(sample["tok_s"]) for sample in samples)
            mean_gpu = fmean(float(sample["tok_s_gpu"]) for sample in samples)
            mean_kernel = fmean(float(sample["kernel_tok_s"]) for sample in samples)
            results[f"b{batch_size}_t{output_len}"] = {
                "backend": "mk",
                "batch_size": batch_size,
                "max_new_tokens": output_len,
                "eos_policy": "respect_eos" if args.respect_eos else "forced_continuation",
                "tok_s": mean_wall,
                "tok_s_gpu": mean_gpu,
                "kernel_tok_s": mean_kernel,
                "new_token_ids": captured["new_token_ids"],
                "new_tokens_text": captured["new_tokens_text"],
                "quality": captured_quality,
                "validation": {
                    "length_complete": length_complete,
                    "greedy_rerun_match": greedy_rerun_match,
                    "has_gibberish": has_any_gibberish,
                    "has_repetition": has_any_repetition,
                    "repetition_is_diagnostic": True,
                    "pass": length_complete and not has_any_gibberish,
                },
                "samples": timing_samples,
            }
            print(
                f"backend=mk batch={batch_size} max_new_tokens={output_len} "
                f"tok_s={mean_wall:.2f} tok_s_gpu={mean_gpu:.2f} "
                f"kernel_tok_s={mean_kernel:.2f}"
            )

    out = {
        "backend": "mk",
        "model_path": args.model_path,
        "user_prompt": args.prompt,
        "formatted_prompt": formatted_prompt,
        "prompt_token_count": prompt_token_count,
        "prompt_mode": "chat" if args.chat else "raw",
        "eos_policy": "respect_eos" if args.respect_eos else "forced_continuation",
        "decode_strategy": "greedy",
        "hardware": args.hardware,
        "torch_dtype": args.torch_dtype,
        "batch_sizes": batch_sizes,
        "output_lengths": output_lengths,
        "warmup_iters": args.warmup_iters,
        "bench_iters": args.bench_iters,
        "results": results,
    }
    if args.out:
        out_path = Path(args.out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"saved_results={out_path}")
    else:
        print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
