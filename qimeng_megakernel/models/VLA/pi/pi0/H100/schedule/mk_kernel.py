"""Strict loader for the prebuilt Pi0 H100 extensions.

SigLIP is compile-time specialized by view count. Gemma and Action Expert share
one DSO because their view-dependent shapes and attention spans live in the
runtime instruction schedule.
"""
from __future__ import annotations

import importlib.machinery
import importlib.util
from pathlib import Path
from types import ModuleType


_SIGLIP_MODULES = {view: f"mk_siglip_view{view}" for view in (1, 2, 3)}
_MODULES = (*_SIGLIP_MODULES.values(), "mk_gemma", "mk_ae")


def kernel_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "kernel"


def load_kernel(name: str, directory: str | Path | None = None) -> ModuleType:
    if name not in _MODULES:
        raise ValueError(f"unknown Pi0 kernel {name!r}; expected one of {_MODULES}")
    root = kernel_dir() if directory is None else Path(directory).expanduser().resolve()
    candidates = [
        root / f"{name}{suffix}"
        for suffix in importlib.machinery.EXTENSION_SUFFIXES
        if (root / f"{name}{suffix}").is_file()
    ]
    if len(candidates) != 1:
        raise ImportError(
            f"expected one extension for this Python ABI named {name} in {root}; "
            f"run `make QIMENG_PYTHON=${{QIMENG_PYTHON}}` there"
        )
    dso = candidates[0].resolve()
    spec = importlib.util.spec_from_file_location(name, dso)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot create an import spec for {dso}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if Path(module.__file__ or "").resolve() != dso or not callable(getattr(module, "mk", None)):
        raise ImportError(f"{dso} does not expose the expected mk entrypoint")
    return module


def load_all(
    directory: str | Path | None = None, *, views: int = 2
) -> tuple[ModuleType, ModuleType, ModuleType]:
    if type(views) is not int or views not in _SIGLIP_MODULES:
        raise ValueError(f"views must be one of {tuple(_SIGLIP_MODULES)}, got {views!r}")
    return (
        load_kernel(_SIGLIP_MODULES[views], directory),
        load_kernel("mk_gemma", directory),
        load_kernel("mk_ae", directory),
    )


__all__ = ["kernel_dir", "load_all", "load_kernel"]
