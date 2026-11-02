"""S3 fixtures shared by the independent op/layer/module test files."""
import pytest

from qimeng_megakernel.verifier.common.verifier_kernel import get_verifier_kernel
from qimeng_megakernel.verifier.LLM.qwen.qwen3__0_6b.A100.verifier_config import CONFIG


@pytest.fixture(scope="session")
def verifier_kernel():
    """Build and return the verifier-owned CUDA kernel, never production mk.so."""
    assert CONFIG.kernel_dirs is not None
    return get_verifier_kernel(
        CONFIG.kernel_dirs.verifier_side,
        "qimeng_qwen3_0_6b_a100_s3_verifier_kernel",
    )
