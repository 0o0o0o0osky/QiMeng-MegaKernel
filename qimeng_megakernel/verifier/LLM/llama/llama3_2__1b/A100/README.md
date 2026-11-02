# Llama 3.2 1B A100 verifier

This directory owns the A100 verifier routes. Its B=1 Python test structure and
test semantics match the established H100 suite; only the hardware package paths,
108-SM guards, and production kernel implementation differ.

- `single_batch`: the B=1 route, containing S1--S4 stage folders and its
  verifier-owned `decode_kernel` source tree.
- `multi_batch`: the retained A100 multi-batch route. Its current implementation uses
  4 warps per CTA and targets 2 resident CTAs per SM. It contains the same S1--S4
  stage folders and op/layer/module test split as the H100 multi-batch verifier.

All numeric tolerances come from `verifier_config.py`; test files do not define
hardware-specific comparison formulas or fallback checkpoint downloads. Source
identity between the verifier kernel and production kernel is a read-only,
non-numeric check.

Run commands from the repository root with the project environment:

```bash
CUDA_VISIBLE_DEVICES=0 ${QIMENG_PYTHON} -m pytest -q \
  qimeng_megakernel/verifier/LLM/llama/llama3_2__1b/A100/single_batch
```

Run the multi-batch verifier separately:

```bash
CUDA_VISIBLE_DEVICES=0 ${QIMENG_PYTHON} -m pytest -q \
  qimeng_megakernel/verifier/LLM/llama/llama3_2__1b/A100/multi_batch
```

S3 builds and loads the verifier-owned CUDA kernel with a route-unique module name.
After its op, layer, and module checks pass, the managed verifier and production source
sets must be byte-identical. Pytest performs this check without copying or deleting
production files. S4 and inference load only the production `mk.so`.
