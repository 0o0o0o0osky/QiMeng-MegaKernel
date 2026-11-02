"""S3 fixtures shared by the independent op/layer/module test files."""
import pytest

from qimeng_megakernel.verifier.common.verifier_kernel import get_verifier_kernel
from qimeng_megakernel.verifier.LLM.qwen.qwen2_5__3b.H100.verifier_config import CONFIG


@pytest.fixture(scope="session")
def verifier_kernel():
    """Build and return the route-unique verifier-owned CUDA kernel."""
    assert CONFIG.kernel_dirs is not None
    return get_verifier_kernel(
        CONFIG.kernel_dirs.verifier_side,
        "qimeng_qwen2_5_3b_h100_s3_verifier_kernel",
    )
