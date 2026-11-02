"""GPU-free checks for the generation helpers used by LLM S4 tests."""
from types import SimpleNamespace

import pytest

from qimeng_megakernel.verifier.common.generation import (
    fork_analysis,
    has_gibberish,
    has_repetition,
    resolve_eos_token_ids,
)


def _topk(token_ids: list[int], logits: list[float] | None = None) -> list[dict]:
    if logits is None:
        logits = [5.0, 4.9, 3.0, 2.0, 1.0]
    return [
        {"rank": rank, "token_id": token_id, "logit": logit}
        for rank, (token_id, logit) in enumerate(zip(token_ids, logits), start=1)
    ]


def test_resolve_eos_token_ids_deduplicates_sources() -> None:
    tokenizer = SimpleNamespace(eos_token_id=2, eot_token_id=3)
    model = SimpleNamespace(config=SimpleNamespace(eos_token_id=[2, 4]))
    assert resolve_eos_token_ids(tokenizer, model) == [2, 4, 3]


@pytest.mark.parametrize(
    "text, expected",
    [
        ("", True),
        ("   ", True),
        ("broken \ufffd text", True),
        ("A readable answer.\n", False),
    ],
)
def test_has_gibberish(text: str, expected: bool) -> None:
    assert has_gibberish(text) is expected


def test_has_repetition_detects_consecutive_three_copy_loop() -> None:
    assert has_repetition("The copper lantern glowed. " * 4)
    assert not has_repetition("The copper lantern glowed beside the quiet river.")


def test_fork_analysis_reports_overlap_and_probability_overlap() -> None:
    mk = _topk([10, 11, 12, 13, 14])
    hf = _topk([11, 10, 12, 15, 16])
    result = fork_analysis(mk, hf)
    assert result["topk_k"] == 5
    assert result["topk_overlap"] == 3
    assert 0.0 < result["topk_probability_overlap"] <= 1.0
    assert result["mk_top1_in_hf_topk"]
    assert result["hf_top1_in_mk_topk"]


@pytest.mark.parametrize(
    "entries, message",
    [
        (_topk([1, 1, 2, 3, 4]), "unique"),
        (_topk([1, 2, 3, 4, 5], [4.0, 5.0, 3.0, 2.0, 1.0]), "descending"),
        (_topk([1, 2, 3, 4, 5], [float("nan"), 4.0, 3.0, 2.0, 1.0]), "finite"),
    ],
)
def test_fork_analysis_rejects_invalid_tables(
    entries: list[dict],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        fork_analysis(entries, _topk([6, 7, 8, 9, 10]))
