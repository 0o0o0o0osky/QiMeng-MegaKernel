# Examples

This directory contains user-facing entry points and quick-acceptance utilities. Each
script documents its own supported routes and options; examples are not verifier tests.

- `llm.py` runs the supported LLM routes and optional quick timing.
- `pi0.py` runs the production VLA action-inference entry point.

## Production inference entry points

A production-only example calls the model-owned inference API directly. It validates
checkpoint and input paths, uses the route's preprocessing and dtype, and reports the
declared output without a silent PyTorch or PyVM fallback. Route-specific observation,
view, horizon, chunk, prompt, or generation options belong here only when the public API
exposes them.

Quick-acceptance examples may expose route selection or timing when that is their stated
purpose. Formal correctness remains in `qimeng_megakernel/verifier/`; performance
experiments and external-system comparisons remain in the profiling workflow.
