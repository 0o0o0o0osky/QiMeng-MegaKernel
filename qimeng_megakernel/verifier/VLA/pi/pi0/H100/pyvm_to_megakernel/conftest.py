"""Build and load the verifier-owned Pi0 H100 module kernels."""
from __future__ import annotations

import importlib.machinery
import importlib.util
from pathlib import Path
import subprocess
import sys
from types import ModuleType

import pytest

from qimeng_megakernel.verifier.VLA.pi.pi0.H100.verifier_config import CONFIG
from qimeng_megakernel.verifier.VLA.pi.pi0.H100.real_data_utils import (
    LOCAL_CHECKPOINT_AVAILABLE,
    require_h100,
)


_NAMES = {
    "siglip_view1": "qimeng_pi0_h100_siglip_view1_s3_kernel",
    "siglip_view2": "qimeng_pi0_h100_siglip_view2_s3_kernel",
    "siglip_view3": "qimeng_pi0_h100_siglip_view3_s3_kernel",
    "gemma": "qimeng_pi0_h100_gemma_s3_kernel",
    "ae": "qimeng_pi0_h100_ae_s3_kernel",
}


def _load(directory: Path, name: str) -> ModuleType:
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
    if not callable(getattr(module, "mk", None)):
        raise ImportError(f"verifier kernel has no mk entrypoint: {dso}")
    return module


@pytest.fixture(scope="session")
def verifier_kernels() -> dict[str, object]:
    if not LOCAL_CHECKPOINT_AVAILABLE:
        pytest.skip("set PI0_CKPT to the local Pi0 safetensors checkpoint")
    require_h100()
    assert CONFIG.kernel_dirs is not None
    directory = CONFIG.kernel_dirs.verifier_side.resolve()
    result = subprocess.run(
        ["make", f"PYTHON={sys.executable}"],
        cwd=directory,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Pi0 verifier kernel build failed:\n{result.stdout}\n{result.stderr}"
        )
    return {key: _load(directory, name).mk for key, name in _NAMES.items()}
