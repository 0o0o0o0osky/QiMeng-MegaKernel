"""CPU tests for model-neutral real-prefill fixture routing."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from qimeng_megakernel.verifier.common import real_data


@pytest.mark.parametrize(
    ("hardware", "expected_present"),
    (("H100", True), (None, False)),
)
def test_real_data_spec_hardware_is_forwarded_only_when_declared(
    hardware: str | None,
    expected_present: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prefill = SimpleNamespace(
        logits=torch.zeros(1, dtype=torch.bfloat16),
        first_gen_token=torch.tensor([[1]], dtype=torch.long),
    )
    model = SimpleNamespace(
        model=SimpleNamespace(
            embed_tokens=lambda _: torch.ones(1, 1, 2, dtype=torch.bfloat16)
        )
    )
    globs = SimpleNamespace(hidden_states=torch.zeros(2, dtype=torch.bfloat16))
    received: dict[str, object] = {}

    monkeypatch.setattr(real_data, "run_prefill", lambda *_args, **_kwargs: prefill)
    monkeypatch.setattr(
        real_data,
        "_load_hf_model",
        lambda *_args, **_kwargs: model,
    )

    def fake_preprocess(*_args: object, **kwargs: object) -> object:
        received.update(kwargs)
        return globs

    monkeypatch.setattr(real_data, "llm_preprocess_decode", fake_preprocess)

    result = real_data.make_globs_from_real_prefill(
        "CPU probe",
        real_data.RealDataSpec(
            default_model_path="local-checkpoint",
            hardware=hardware,
        ),
        device="cpu",
    )

    assert result is globs
    assert ("hardware" in received) is expected_present
    if expected_present:
        assert received["hardware"] == hardware
