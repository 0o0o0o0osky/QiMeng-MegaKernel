"""Read-only identity check for the verifier-owned kernel source."""

from qimeng_megakernel.verifier.common.copy_sync import make_copy_sync_test
from qimeng_megakernel.verifier.LLM.qwen.qwen3__0_6b.H100.verifier_config import CONFIG


_generated = make_copy_sync_test(CONFIG.kernel_dirs)
globals().update(
    {name: value for name, value in _generated.items() if name.startswith("test_")}
)
