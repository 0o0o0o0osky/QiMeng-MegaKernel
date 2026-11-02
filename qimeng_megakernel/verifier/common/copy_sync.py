"""Read-only identity check for verifier and production kernel sources.

S3 builds and validates the verifier-owned kernel.  Copying validated sources into
the production tree is an explicit developer action; pytest never writes either tree.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


SOURCE_SUFFIXES = {".cu", ".cuh", ".cpp", ".cc", ".h", ".hpp"}
def _source_files(root: Path) -> dict[Path, Path]:
    """Return managed source paths keyed by their location below ``root``."""
    return {
        path.relative_to(root): path
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and path.suffix in SOURCE_SUFFIXES
    }


def make_copy_sync_test(kernel_dirs: Any) -> dict[str, Any]:
    """Build one read-only source-identity test for a verifier suite."""
    if kernel_dirs is None:
        raise ValueError(
            "make_copy_sync_test called with kernel_dirs=None; suites without a "
            "verifier-owned kernel should not define a source-identity test"
        )

    model_side = Path(kernel_dirs.model_side).resolve()
    verifier_side = Path(kernel_dirs.verifier_side).resolve()

    def test_kernel_source_identity() -> None:
        assert model_side.is_dir(), f"production kernel tree missing: {model_side}"
        assert verifier_side.is_dir(), f"verifier kernel tree missing: {verifier_side}"

        model = _source_files(model_side)
        verifier = _source_files(verifier_side)
        assert model, f"no managed production kernel sources found: {model_side}"
        assert verifier, f"no managed verifier kernel sources found: {verifier_side}"

        only_model = sorted(model.keys() - verifier.keys())
        only_verifier = sorted(verifier.keys() - model.keys())
        assert not only_model and not only_verifier, (
            "kernel source file sets differ\n"
            f"  production only: {[str(path) for path in only_model]}\n"
            f"  verifier only: {[str(path) for path in only_verifier]}"
        )

        differing = [
            rel
            for rel in sorted(model)
            if model[rel].read_bytes() != verifier[rel].read_bytes()
        ]
        assert not differing, (
            "kernel sources differ between verifier and production trees: "
            + ", ".join(str(path) for path in differing)
        )

    return {"test_kernel_source_identity": test_kernel_source_identity}
