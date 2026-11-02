"""Load the prebuilt Pi0 A100/A800 production extensions."""

from __future__ import annotations

import importlib.machinery
import importlib.util
from pathlib import Path
from types import ModuleType


_DSO_NAMES = {
    "siglip": "qimeng_pi0_a100_siglip_s3_kernel",
    "gemma": "qimeng_pi0_a100_gemma_s3_kernel",
    "ae": "qimeng_pi0_a100_ae_s3_kernel",
}


def kernel_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "kernel"


def load_kernel(name: str, directory: str | Path | None = None) -> ModuleType:
    if name not in _DSO_NAMES:
        raise ValueError(
            f"unknown Pi0 kernel {name!r}; expected one of {tuple(_DSO_NAMES)}"
        )
    root = kernel_dir() if directory is None else Path(directory).expanduser().resolve()
    dso_name = _DSO_NAMES[name]
    candidates = [
        root / f"{dso_name}{suffix}"
        for suffix in importlib.machinery.EXTENSION_SUFFIXES
        if (root / f"{dso_name}{suffix}").is_file()
    ]
    if len(candidates) != 1:
        raise ImportError(
            f"expected one extension for this Python ABI named {dso_name} in {root}; "
            "run `make PYTHON=/path/to/python` there"
        )
    dso = candidates[0].resolve()
    spec = importlib.util.spec_from_file_location(dso_name, dso)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot create an import spec for {dso}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if Path(module.__file__ or "").resolve() != dso or not callable(
        getattr(module, "bind", None)
    ):
        raise ImportError(f"{dso} does not expose the expected bind entrypoint")
    return module


def load_all(
    directory: str | Path | None = None,
) -> tuple[ModuleType, ModuleType, ModuleType]:
    return (
        load_kernel("siglip", directory),
        load_kernel("gemma", directory),
        load_kernel("ae", directory),
    )


__all__ = ["kernel_dir", "load_all", "load_kernel"]
