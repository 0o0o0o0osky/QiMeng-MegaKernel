# Qwen2.5 3B A100 decode_kernel (verifier)

Standalone build for **pyvm_to_megakernel** tests. All sources live in this directory.

- **Build**: `make` → produces `mk.so`. Tests load `mk` from here via `qwen_test_adapter`.
- **Model kernel**: A separate copy exists at
  `qimeng_megakernel/models/LLM/qwen/qwen2_5__3b/A100/single_batch/decode_kernel/`
  used by the MK inference path (`mk_kernel.py`). Verifier and model kernels are **independent** (no mutual reference); keep them in sync when changing kernel logic.
