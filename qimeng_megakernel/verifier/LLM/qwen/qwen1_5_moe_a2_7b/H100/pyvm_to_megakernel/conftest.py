"""S3 fixtures shared by the independent op/layer/module test files."""
import pytest

from qimeng_megakernel.verifier.common.verifier_kernel import get_verifier_kernel
from qimeng_megakernel.verifier.LLM.qwen.qwen1_5_moe_a2_7b.H100.verifier_config import CONFIG


@pytest.fixture(scope="session")
def verifier_kernel():
    """Build and return the route-unique verifier-owned CUDA kernel."""
    assert CONFIG.kernel_dirs is not None
    return get_verifier_kernel(
        CONFIG.kernel_dirs.verifier_side,
        "qimeng_qwen1_5_moe_a2_7b_h100_s3_verifier_kernel",
    )
