"""Build and load the three verifier-owned Pi0 A100 native DSOs."""

from __future__ import annotations

import importlib.machinery
import importlib.util
from pathlib import Path
import subprocess
import sys
from types import ModuleType

import pytest

from qimeng_megakernel.verifier.VLA.pi.pi0.A100.verifier_config import CONFIG
from qimeng_megakernel.verifier.VLA.pi.pi0.A100.real_data_utils import (
    LOCAL_CHECKPOINT_AVAILABLE,
    require_a100,
)


_DSO_SPECS = {
    "siglip": (
        "qimeng_pi0_a100_siglip_s3_kernel",
        ("bind", "run_phase", "run_mlp_view1", "run_mlp_view2",
         "run_mlp_view3", "run_attention"),
    ),
    "gemma": (
        "qimeng_pi0_a100_gemma_s3_kernel",
        ("bind", "run_phase", "run_down_view1", "run_qkv_view2",
         "run_pv_oproj_view1", "run_pv_oproj_view2", "run_pv_oproj_view3",
         "run_qkv_qk_view2", "run_qkv_qk_view3"),
    ),
    "ae": (
        "qimeng_pi0_a100_ae_s3_kernel",
        ("bind", "run_phase", "run_materialized", "run_logical", "run_rmsn",
         "run_residual_rmsn", "run_qkv", "run_action"),
    ),
}


def _load(
    directory: Path, name: str, required_exports: tuple[str, ...]
) -> ModuleType:
    candidates = [
        directory / f"{name}{suffix}"
        for suffix in importlib.machinery.EXTENSION_SUFFIXES
        if (directory / f"{name}{suffix}").is_file()
    ]
    if len(candidates) != 1:
        raise ImportError(f"expected one verifier kernel named {name} in {directory}")
    dso = candidates[0].resolve()
    spec = importlib.util.spec_from_file_location(name, dso)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load verifier kernel {dso}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    loaded = Path(module.__file__ or "").resolve()
    if loaded != dso:
        raise ImportError(f"verifier DSO identity mismatch: expected {dso}, loaded {loaded}")
    missing = tuple(
        export for export in required_exports
        if not callable(getattr(module, export, None))
    )
    if missing:
        raise ImportError(f"verifier DSO {dso} is missing callable exports: {missing}")
    return module


@pytest.fixture(scope="session")
def verifier_dsos() -> dict[str, ModuleType]:
    if not LOCAL_CHECKPOINT_AVAILABLE:
        pytest.skip("set PI0_CKPT to the local Pi0 safetensors checkpoint")
    require_a100()
    assert CONFIG.kernel_dirs is not None
    directory = CONFIG.kernel_dirs.verifier_side.resolve()
    result = subprocess.run(
        ["make", "-j3", f"PYTHON={sys.executable}"],
        cwd=directory,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Pi0 verifier kernel build failed:\n{result.stdout}\n{result.stderr}"
        )
    return {
        key: _load(directory, name, required_exports)
        for key, (name, required_exports) in _DSO_SPECS.items()
    }
