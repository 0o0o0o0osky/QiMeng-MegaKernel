"""Qwen2.5 3B H100 greedy end-to-end validation."""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from qimeng_megakernel.models.LLM.prefill import InferenceConfig, run_prefill_fused
from qimeng_megakernel.verifier.common.generation import (
    has_gibberish,
    has_repetition,
    hf_greedy_decode,
    mk_decode_via_central,
    resolve_eos_token_ids,
)
from qimeng_megakernel.verifier.common.stage_config import GenerationGate
from qimeng_megakernel.verifier.LLM.qwen.qwen2_5__3b.H100.verifier_config import CONFIG


PROMPT = "Write a long story."
DECODE_LENGTHS = (128, 256, 512, 1024)


def _chat_prompt(tokenizer) -> str:
    """The test explicitly selects Qwen's chat format and non-thinking mode."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        tokenize=False,
        add_generation_prompt=True,
    )


@pytest.fixture(scope="module")
def model_path() -> str:
    configured = os.environ.get("QWEN_TEST_MODEL", "").strip()
    path = Path(configured).expanduser() if configured else Path()
    if not configured or not path.is_dir() or not (path / "config.json").is_file():
        pytest.skip("set QWEN_TEST_MODEL to a local checkpoint directory")
    return str(path.resolve())


@pytest.fixture(scope="module")
def device(model_path: str) -> str:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    if torch.cuda.get_device_properties("cuda:0").multi_processor_count < 132:
        pytest.skip("Qwen2.5 3B H100 megakernel requires 132 SMs")
    return "cuda:0"


@pytest.fixture(scope="module")
def tokenizer(model_path: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_path, local_files_only=True)


@pytest.fixture(scope="module")
def hf_model(model_path: str, device: str):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    return model.to(device).eval()


def _prefill(model_path: str, tokenizer, device: str, max_new_tokens: int):
    config = InferenceConfig(
        model=model_path,
        device=device,
        torch_dtype="bfloat16",
        add_special_tokens=False,
    )
    return run_prefill_fused(
        config,
        [_chat_prompt(tokenizer)],
        tokenizer=tokenizer,
        hardware="H100",
        max_new_tokens=max_new_tokens,
    )


@pytest.mark.slow
@pytest.mark.parametrize("max_new_tokens", DECODE_LENGTHS)
def test_greedy_e2e_quality(
    model_path: str,
    device: str,
    tokenizer,
    hf_model,
    max_new_tokens: int,
) -> None:
    gate = CONFIG.gate("S4")
    assert isinstance(gate, GenerationGate)
    torch.cuda.set_device(device)

    hf = hf_greedy_decode(
        model=hf_model,
        tokenizer=tokenizer,
        prompt=_chat_prompt(tokenizer),
        max_new_tokens=max_new_tokens,
        add_special_tokens=False,
    )
    eos_ids = resolve_eos_token_ids(tokenizer, hf_model)
    prefill = _prefill(model_path, tokenizer, device, max_new_tokens)
    mk = mk_decode_via_central(
        model_path=model_path,
        prefill=prefill,
        max_new_tokens=max_new_tokens,
        hardware="H100",
        ignore_eos=False,
        eos_token_ids=eos_ids,
    )

    mk_ids = [int(token) for token in mk["new_token_ids"]]
    hf_ids = [int(token) for token in hf["tokens"][hf["prompt_len"] :]]
    mk_text = tokenizer.decode(mk_ids, skip_special_tokens=True)
    hf_text = tokenizer.decode(hf_ids, skip_special_tokens=True)

    assert mk_ids, "megakernel produced no tokens"
    assert not has_gibberish(mk_text), (
        f"megakernel output is corrupted: {mk_text[:200]!r}"
    )
    if gate.repetition_policy == "forbid":
        assert not (has_repetition(mk_text) and not has_repetition(hf_text)), (
            "megakernel introduced repetition absent from the same-checkpoint baseline"
        )
