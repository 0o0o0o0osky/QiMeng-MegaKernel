"""Build and load the verifier-owned megakernel used by S3.

Inference loads the prebuilt production DSO from the model tree.  This module is only
for verifier execution: it builds the verifier source tree with a route-unique Python
extension name, loads that exact DSO, and returns its ``mk`` entrypoint.
"""
from __future__ import annotations

import importlib.machinery
import importlib.util
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType


def _validate_module_name(module_name: str) -> None:
    if not module_name.isidentifier():
        raise ValueError(
            f"verifier-kernel module name must be a Python identifier: {module_name!r}"
        )


def _resolve_dso(kernel_dir: Path, module_name: str) -> Path:
    matches = [
        kernel_dir / f"{module_name}{suffix}"
        for suffix in importlib.machinery.EXTENSION_SUFFIXES
        if (kernel_dir / f"{module_name}{suffix}").is_file()
    ]
    # The current Makefile intentionally emits the portable ``.so`` suffix.
    portable = kernel_dir / f"{module_name}.so"
    if portable.is_file() and portable not in matches:
        matches.append(portable)
    if len(matches) != 1:
        raise ImportError(
            f"expected exactly one verifier-kernel DSO named {module_name} in "
            f"{kernel_dir}, found {len(matches)}"
        )
    return matches[0].resolve()


def build_verifier_kernel(kernel_dir: Path, module_name: str) -> Path:
    """Incrementally build one verifier kernel with the active Python environment."""
    root = Path(kernel_dir).expanduser().resolve()
    _validate_module_name(module_name)
    if not (root / "Makefile").is_file():
        raise FileNotFoundError(
            f"verifier-kernel Makefile is missing: {root / 'Makefile'}"
        )
    result = subprocess.run(
        [
            "make",
            f"PYTHON={sys.executable}",
            f"MK_MODULE_NAME={module_name}",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"verifier-kernel build failed in {root}:\n{result.stdout}\n{result.stderr}"
        )
    return _resolve_dso(root, module_name)


@lru_cache(maxsize=None)
def load_verifier_kernel(kernel_dir: Path, module_name: str) -> ModuleType:
    """Build, load and identity-check the verifier kernel once per process."""
    dso = build_verifier_kernel(Path(kernel_dir), module_name)
    spec = importlib.util.spec_from_file_location(module_name, dso)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot create an import spec for verifier kernel {dso}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if Path(module.__file__ or "").resolve() != dso:
        raise ImportError(f"loaded unexpected verifier kernel: {module.__file__}")
    if not callable(getattr(module, "mk", None)):
        raise ImportError(f"verifier kernel does not export mk: {dso}")
    return module


def get_verifier_kernel(kernel_dir: Path, module_name: str):
    """Return the exact verifier-owned ``mk`` entrypoint used by S3."""
    return load_verifier_kernel(Path(kernel_dir).resolve(), module_name).mk


__all__ = [
    "build_verifier_kernel",
    "get_verifier_kernel",
    "load_verifier_kernel",
]
