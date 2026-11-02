"""S3 fixture for the verifier-owned Qwen2.5 A100 kernel."""
import pytest

from qimeng_megakernel.verifier.common.verifier_kernel import get_verifier_kernel
from qimeng_megakernel.verifier.LLM.qwen.qwen2_5__3b.A100.verifier_config import (
    CONFIG,
)


@pytest.fixture(scope="session")
def verifier_kernel():
    assert CONFIG.kernel_dirs is not None
    return get_verifier_kernel(
        CONFIG.kernel_dirs.verifier_side,
        "qimeng_qwen2_5_3b_a100_s3_verifier_kernel",
    )
