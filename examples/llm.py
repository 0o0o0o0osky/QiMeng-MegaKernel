"""Generate text with one registered LLM route.

Examples:
  ${QIMENG_PYTHON} -m examples.llm --model ${TARGET_MODEL_PATH} --route mk
  ${QIMENG_PYTHON} -m examples.llm --model ${TARGET_MODEL_PATH} --route fused --raw
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from transformers import AutoTokenizer

from qimeng_megakernel.models.LLM.inference import InferenceConfig, run_inference


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text with QiMeng-MegaKernel.")
    parser.add_argument("--model", default=os.environ.get("LLM_MODEL"))
    parser.add_argument("--route", choices=("hf", "fused", "python_vm", "mk"), default="mk")
    parser.add_argument("--hardware", choices=("auto", "A100", "H100"), default="auto")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--prompt", default="Write a long story.")
    parser.add_argument("--prompts-file", type=Path)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--raw", action="store_true", help="Skip the tokenizer chat template.")
    thinking = parser.add_mutually_exclusive_group()
    thinking.add_argument("--thinking", dest="thinking", action="store_true")
    thinking.add_argument("--no-thinking", dest="thinking", action="store_false")
    parser.set_defaults(thinking=None)
    parser.add_argument("--timing", action="store_true")
    args = parser.parse_args()
    if not args.model:
        parser.error("--model or LLM_MODEL is required")
    if args.batch_size < 1 or args.batch_size > 16:
        parser.error("--batch-size must be in [1, 16]")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    if args.raw and args.thinking is not None:
        parser.error("thinking flags require chat formatting")
    return args


def _hardware(value: str, device: str) -> str:
    if value != "auto":
        return value
    if not torch.cuda.is_available():
        return "A100"
    name = torch.cuda.get_device_name(device).upper()
    if "H100" in name:
        return "H100"
    if "A100" in name:
        return "A100"
    raise SystemExit(f"unsupported GPU for --hardware auto: {name}")


def _user_prompts(args: argparse.Namespace) -> list[str]:
    if args.prompts_file is None:
        return [args.prompt] * args.batch_size
    prompts = [line.strip() for line in args.prompts_file.read_text().splitlines() if line.strip()]
    if not prompts:
        raise SystemExit(f"no prompts in {args.prompts_file}")
    if len(prompts) > 16:
        raise SystemExit("at most 16 prompts are supported")
    return prompts


def _prepare_prompts(
    model: str,
    prompts: list[str],
    *,
    raw: bool,
    thinking: bool | None,
) -> tuple[AutoTokenizer, list[str], bool]:
    tokenizer = AutoTokenizer.from_pretrained(
        model,
        local_files_only=Path(model).expanduser().is_dir(),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    if raw:
        return tokenizer, prompts, True
    template_args = {"tokenize": False, "add_generation_prompt": True}
    if thinking is not None:
        template_args["enable_thinking"] = thinking
    formatted = [
        str(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                **template_args,
            )
        )
        for prompt in prompts
    ]
    return tokenizer, formatted, False


def main() -> None:
    args = _arguments()
    tokenizer, prompts, add_special_tokens = _prepare_prompts(
        args.model,
        _user_prompts(args),
        raw=args.raw,
        thinking=args.thinking,
    )
    output = run_inference(
        InferenceConfig(
            model=args.model,
            device=args.device,
            torch_dtype=args.dtype,
            add_special_tokens=add_special_tokens,
        ),
        prompts,
        enable_decode=True,
        debug_max_new_tokens=args.max_new_tokens,
        decode_route=args.route,
        hardware=_hardware(args.hardware, args.device),
        decode_time=args.timing,
    ).decode_output
    if output is None or output.get("new_tokens") is None:
        raise RuntimeError("decode returned no tokens")
    for row, token_ids in enumerate(output["new_tokens"].tolist()):
        print(f"[{row}] {tokenizer.decode(token_ids, skip_special_tokens=True)}")


if __name__ == "__main__":
    main()
