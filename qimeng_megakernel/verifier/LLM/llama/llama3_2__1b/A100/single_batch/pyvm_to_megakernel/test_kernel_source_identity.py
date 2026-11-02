"""Read-only identity check for the verifier-owned kernel source."""

from qimeng_megakernel.verifier.common.copy_sync import make_copy_sync_test
from qimeng_megakernel.verifier.LLM.llama.llama3_2__1b.A100.verifier_config import CONFIG


_generated = make_copy_sync_test(CONFIG.kernel_dirs)
globals().update(
    {name: value for name, value in _generated.items() if name.startswith("test_")}
)
