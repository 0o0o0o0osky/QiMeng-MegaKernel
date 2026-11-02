"""Strict loader and launch adapter for the prebuilt A100 Action Expert DSO."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

from .config import Pi0ActionExpertGlobals

_MODULE_NAME = "mk_action_expert"
_MODULE_CACHE: dict[Path, ModuleType] = {}


def _default_kernel_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "kernel" / "action_expert"


def _load_module(kernel_dir: Path) -> ModuleType:
    dso = (kernel_dir / f"{_MODULE_NAME}.so").resolve()
    if not dso.is_file():
        raise ImportError(
            f"missing prebuilt {dso}; build it explicitly with "
            "'make PYTHON=/path/to/python'"
        )
    cached = _MODULE_CACHE.get(dso)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, dso)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot create import spec for {dso}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "mk_action_expert", None)):
        raise ImportError(f"{dso} does not expose mk_action_expert")
    _MODULE_CACHE[dso] = module
    return module


def get_mk(kernel_dir: Path | None = None):
    root = _default_kernel_dir() if kernel_dir is None else Path(kernel_dir)
    return _load_module(root.expanduser().resolve()).mk_action_expert


def call_mk(
    globs: Pi0ActionExpertGlobals,
    mk=None,
    *,
    sm_count: int = 108,
) -> None:
    kernel = get_mk() if mk is None else mk
    kernel(*globs.launch_args(sm_count=sm_count))


__all__ = ["call_mk", "get_mk"]
