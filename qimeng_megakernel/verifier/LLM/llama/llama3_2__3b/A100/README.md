# Llama 3.2 3B A100 verifier

This directory owns the A100 verifier routes. Its B=1 Python test structure and
test semantics match the established H100 suite; only the hardware package paths,
108-SM guards, and production kernel implementation differ.

- `baseline_to_fused`: S1 baseline to fused operator, layer, and module checks.
- `fused_to_pyvm`: S2 fused to Python VM checks.
- `pyvm_to_megakernel`: S3 Python VM to verifier-owned kernel checks.
- `e2e_generate`: S4 greedy generation correctness checks.

There is no A100 multi-batch route for this model in the current tree.

All numeric tolerances come from `verifier_config.py`; test files do not define
hardware-specific comparison formulas or fallback checkpoint downloads. Source
identity between the verifier kernel and production kernel is a read-only,
non-numeric check.

Run commands from the repository root with the project environment:

```bash
CUDA_VISIBLE_DEVICES=0 ${QIMENG_PYTHON} -m pytest -q \
  qimeng_megakernel/verifier/LLM/llama/llama3_2__3b/A100
```

S3 builds and loads the verifier-owned CUDA kernel with a route-unique module name.
After its op, layer, and module checks pass, the managed verifier and production source
sets must be byte-identical. Pytest performs this check without copying or deleting
production files. S4 and inference load only the production `mk.so`.
