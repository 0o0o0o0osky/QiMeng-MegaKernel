"""Build and load the verifier-owned multi-batch kernel for S3."""

import pytest

from qimeng_megakernel.verifier.common.verifier_kernel import get_verifier_kernel
from qimeng_megakernel.verifier.LLM.llama.llama3_2__1b.H100.verifier_config import (
    MULTI_BATCH_CONFIG,
)


@pytest.fixture(scope="session")
def verifier_kernel():
    assert MULTI_BATCH_CONFIG.kernel_dirs is not None
    return get_verifier_kernel(
        MULTI_BATCH_CONFIG.kernel_dirs.verifier_side,
        "qimeng_llama3_2_1b_h100_multi_batch_s3_kernel",
    )
