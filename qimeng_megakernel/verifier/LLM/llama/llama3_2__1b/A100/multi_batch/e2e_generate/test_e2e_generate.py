"""S4 multi-batch greedy generation validation against Hugging Face."""

import os
from pathlib import Path

import pytest
import torch

from qimeng_megakernel.models.LLM.decode import decode
from qimeng_megakernel.models.LLM.prefill import InferenceConfig, run_prefill_fused
from qimeng_megakernel.verifier.common.generation import has_gibberish
from qimeng_megakernel.verifier.common.stage_config import GenerationGate
from qimeng_megakernel.verifier.LLM.llama.llama3_2__1b.A100.real_data_utils import (
    distinct_prompts_for_batch,
)
from qimeng_megakernel.verifier.LLM.llama.llama3_2__1b.A100.verifier_config import (
    MULTI_BATCH_CONFIG as CONFIG,
)

pytestmark = pytest.mark.slow
DECODE_LENGTHS = (128, 256, 512, 1024)
ADJACENT_STEPS = 5


@pytest.fixture(scope="module")
def model_path() -> str:
    configured = os.environ.get("MODEL_LLAMA3_2_1B", "").strip()
    path = Path(configured).expanduser() if configured else Path()
    if not configured or not path.is_dir() or not (path / "config.json").is_file():
        pytest.skip("set MODEL_LLAMA3_2_1B to a local checkpoint directory")
    return str(path.resolve())


@pytest.fixture(scope="module")
def tokenizer(model_path: str):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    if torch.cuda.get_device_properties("cuda:0").multi_processor_count < 108:
        pytest.skip("Llama 3.2 1B A100 megakernel requires 108 SMs")
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_path, local_files_only=True)


def _run_route(
    model_path: str,
    tokenizer,
    prompts: list[str],
    route: str,
    max_new_tokens: int,
) -> dict[str, torch.Tensor]:
    config = InferenceConfig(
        model=model_path,
        device="cuda:0",
        torch_dtype="bfloat16",
        add_special_tokens=True,
    )
    prefill = run_prefill_fused(
        config,
        prompts,
        tokenizer=tokenizer,
        hardware="A100",
        max_new_tokens=max_new_tokens,
    )
    kwargs = {
        "prefill": prefill,
        "decode_route": route,
        "max_new_tokens": max_new_tokens,
        "hardware": "A100",
    }
    if route == "hf":
        kwargs["eos_token_id"] = None
    else:
        kwargs["ignore_eos"] = True
    return decode(model_path, **kwargs)


@pytest.mark.parametrize("max_new_tokens", DECODE_LENGTHS)
@pytest.mark.parametrize("batch_size", (2, 16))
@torch.inference_mode()
def test_greedy_e2e_quality(
    model_path: str,
    tokenizer,
    batch_size: int,
    max_new_tokens: int,
) -> None:
    prompts = distinct_prompts_for_batch(batch_size)
    expected = _run_route(model_path, tokenizer, prompts, "hf", max_new_tokens)
    adjacent = _run_route(model_path, tokenizer, prompts, "fused", ADJACENT_STEPS)
    actual = _run_route(model_path, tokenizer, prompts, "mk", max_new_tokens)

    assert actual["new_tokens"].shape == (batch_size, max_new_tokens)
    assert expected["new_tokens"].shape == actual["new_tokens"].shape
    gate = CONFIG.gate("S4")
    assert isinstance(gate, GenerationGate)
    assert gate.top1_min is not None
    agreement = (
        torch.eq(
            actual["new_tokens"][:, :ADJACENT_STEPS],
            adjacent["new_tokens"],
        )
        .float()
        .mean()
    )
    assert float(agreement) >= gate.top1_min
    for row in range(batch_size):
        actual_text = tokenizer.decode(
            actual["new_tokens"][row].tolist(), skip_special_tokens=True
        )
        assert not has_gibberish(actual_text)
    assert torch.equal(
        actual["per_row_new_tokens"],
        torch.full_like(actual["per_row_new_tokens"], max_new_tokens),
    )


@torch.inference_mode()
def test_mk_grows_decode_kv_capacity(model_path: str, tokenizer) -> None:
    prompts = distinct_prompts_for_batch(2)
    config = InferenceConfig(
        model=model_path,
        device="cuda:0",
        torch_dtype="bfloat16",
        add_special_tokens=True,
    )
    prefill = run_prefill_fused(
        config,
        prompts,
        tokenizer=tokenizer,
        hardware="A100",
        max_new_tokens=1,
    )
    globs = prefill.globs
    assert globs is not None
    initial_seq_len = int(globs.k_cache.shape[2])

    result = decode(
        model_path,
        prefill=prefill,
        decode_route="mk",
        max_new_tokens=5,
        hardware="A100",
        ignore_eos=True,
    )

    target_seq_len = int(prefill.input_ids.shape[1]) + 5
    assert initial_seq_len < target_seq_len
    assert globs.k_cache.shape[2] == target_seq_len
    assert globs.v_cache.shape[2] == target_seq_len
    assert result["new_tokens"].shape == (2, 5)
