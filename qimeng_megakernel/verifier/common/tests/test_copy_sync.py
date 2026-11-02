"""Tests for the read-only kernel source-identity helper."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from qimeng_megakernel.verifier.common.copy_sync import make_copy_sync_test


def _identity_test(tmp_path: Path, model_text: str, verifier_text: str):
    model = tmp_path / "model"
    verifier = tmp_path / "verifier"
    model.mkdir()
    verifier.mkdir()
    (model / "kernel.cu").write_text(model_text, encoding="utf-8")
    (verifier / "kernel.cu").write_text(verifier_text, encoding="utf-8")
    generated = make_copy_sync_test(
        SimpleNamespace(model_side=model, verifier_side=verifier)
    )
    assert tuple(generated) == ("test_kernel_source_identity",)
    return generated["test_kernel_source_identity"], model, verifier


def test_identical_sources_pass(tmp_path: Path) -> None:
    check, _, _ = _identity_test(tmp_path, "same\n", "same\n")
    check()


def test_different_source_bytes_fail(tmp_path: Path) -> None:
    check, _, _ = _identity_test(tmp_path, "model\n", "verifier\n")
    with pytest.raises(AssertionError, match="kernel sources differ"):
        check()


def test_different_source_file_sets_fail(tmp_path: Path) -> None:
    check, model, _ = _identity_test(tmp_path, "same\n", "same\n")
    (model / "extra.cuh").write_text("extra\n", encoding="utf-8")
    with pytest.raises(AssertionError, match="file sets differ"):
        check()


def test_build_artifacts_and_docs_are_ignored(tmp_path: Path) -> None:
    check, model, verifier = _identity_test(tmp_path, "same\n", "same\n")
    (model / "kernel.so").write_bytes(b"model artifact")
    (verifier / "kernel.so").write_bytes(b"verifier artifact")
    (model / "README.md").write_text("model docs\n", encoding="utf-8")
    (verifier / "README.md").write_text("verifier docs\n", encoding="utf-8")
    (model / "Makefile").write_text("production build\n", encoding="utf-8")
    (verifier / "Makefile").write_text("verifier build\n", encoding="utf-8")
    check()
