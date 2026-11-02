# Llama 3.2 1B A100 multi-batch decode kernel

This directory contains the A100 implementation for `1 <= batch_size <= 16`.
The production kernel uses 4 warps per CTA and targets 2 resident CTAs per SM.
Alternative topology experiments are not part of the main production route.

The selected instruction topology is:

1. attention RMS, then QKV GEMM and RoPE;
2. split-KV attention and reduction;
3. output projection with its selected post-processing slot;
4. MLP RMS, then dual Up/Gate GEMM and SiLU;
5. Down projection with its selected post-processing slot;
6. final RMS, then LM-head GEMM.

These placements describe this instance only. The Ampere template exposes pre- and
post-processing slots and does not prescribe which operations must occupy them.

Cross-CTA dependencies use GPU-scope release/publish and acquire/wait operations.
Every producer completes its CTA-local writes before publishing its barrier count.

Build the production module with:

```bash
make PYTHON=/path/to/python
```
