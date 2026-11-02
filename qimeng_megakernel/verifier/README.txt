Verifier: how to run tests
==========================

1. Environment
   - From repo root, activate the project env (e.g. conda):
     conda activate qimeng_megakernel
   - Dependencies: torch, transformers, pytest, einops (see pyproject.toml).

2. Run all verifier tests
   cd /path/to/QiMeng-MegaKernel
   python -m pytest qimeng_megakernel/verifier/ -v --tb=short

3. Run only Llama 3.2 1B verifier tests
   python -m pytest qimeng_megakernel/verifier/LLM/llama/llama3_2__1b/ -v --tb=short

4. Run fused_to_pyvm single tests (RMSNorm+QKV+RoPE, decode attention)
   python -m pytest qimeng_megakernel/verifier/LLM/llama/llama3_2__1b/fused_to_pyvm/single/ -v --tb=short

5. Run a single test file (e.g. rms_qkv_gemv_rope)
   python -m pytest qimeng_megakernel/verifier/LLM/llama/llama3_2__1b/fused_to_pyvm/single/test_rms_qkv_gemv_rope.py -v --tb=short

6. Run one test by name
   python -m pytest qimeng_megakernel/verifier/LLM/llama/llama3_2__1b/fused_to_pyvm/single/test_rms_qkv_gemv_rope.py::test_rms_qkv_gemv_rope_layer0_hf_to_pyvm -v --tb=short

7. Show print output (stats, diffs)
   python -m pytest qimeng_megakernel/verifier/LLM/llama/llama3_2__1b/fused_to_pyvm/single/test_rms_qkv_gemv_rope.py -v -s

5b. Run baseline_to_fused tests (all fused operators)
   python -m pytest qimeng_megakernel/verifier/LLM/llama/llama3_2__1b/baseline_to_fused/ -v --tb=short

5c. Run specific baseline_to_fused test (e.g., full decode test, requires CUDA)
   python -m pytest qimeng_megakernel/verifier/LLM/llama/llama3_2__1b/baseline_to_fused/test_5_full_decode.py -v --tb=short

KV cache + weights (baseline_to_fused):
   - After prefill we have two caches (same content, different layout):
     (1) HF cache: prefill.past_key_values (per-layer B, H, S, D) — used by baseline.
     (2) Our cache: preprocess converts to [L, B, S, H, D] in globs — used by fused.
   - Weights: full/layer tests load model once and pass it to both baseline and fused so weights
     are identical; fused path stacks via _stack_llama_weights_from_hf(model) in preprocess_decode.
   - Single tests (make_globs_for_layer0_qkv): one model, weights from _stack_llama_weights_from_hf,
     cache built from scratch (empty, B,H,S,D per layer) then prepare_kv_cache_for_decode -> our layout.
     Baseline reference uses same model and same globs (our cache).
   - Layout conversion: HF (B, H, S, D) -> (B, S, H, D) in preprocess; fused ops write using
     globs.k_cache.shape[3/4] so layout is consistent.

8. Implemented tests:
   - baseline_to_fused/: test_1_qkv_rope.py (rms_qkv+rope), test_2_attention.py (attention),
     test_3_oproj.py (o_proj+residual), test_4_mlp.py (mlp), test_6_lm_head.py (lm_head).
   - baseline_to_fused/: test_5_full_decode.py (full-model: given prompt, fused path must match
     baseline on first-step logits and greedy decoded tokens; one prompt/length by default).
   - fused_to_pyvm/single: test_rms_qkv_gemv_rope.py, test_decode_attention.py.
   - pyvm_to_megakernel/single: test_rms_qkv_gemv_rope.py (pyvm vs compiled mk).
   Placeholder/skip: fused_to_pyvm full, pyvm_to_megakernel full.

   pyvm_to_megakernel tests require a built megakernel (mk.so). Build it once:
     cd qimeng_megakernel/verifier/LLM/llama/llama3_2__1b/pyvm_to_megakernel/decode_kernel
     make
   Or build in the main decode_kernel and set PYVM_TO_MK_KERNEL_DIR to that dir.
   See verifier/LLM/llama/llama3_2__1b/pyvm_to_megakernel/decode_kernel/README.md.
