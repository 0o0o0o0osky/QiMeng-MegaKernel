# Llama 3.2 1B decode megakernel (A100 single_batch)

Verifier-owned Ampere single-batch kernel. S3 builds and loads this directory;
it is not an inference route.

The generated CUDA follows the paper's per-op structure directly:

```
parse → first-tile prefetch → dependency wait → fused prologue
      → double-buffered GEMV/attention pipeline → epilogue → publish
```

There is no generated manifest or wrapper interface. Model dimensions come from
`LlamaConfig`; barrier rows use `BARRIER_ROW(opcode)`, and expected publication
counts are named beside the corresponding wait sites so the synchronization is
reviewable in the executable source.

The seven op sources instantiate the shape-driven Ampere KB
`operators/transformer` profile. Instantiation resolves `ModelConfig` /
`ModelGlobals` to the Llama types; `DecodeScheduleConfig` selects a single
2,048-wide hidden GEMV segment and four attention warps per GQA group. The same
template bodies cover this model's head64/GQA4 shape and 3B's head128/GQA3
shape. The paper-general stages and fusion slots remain documented in KB
`pipelines`.

The launch configuration is explicit: 108 CTAs, 512 threads, and
`__launch_bounds__(512, 1)`. Dynamic shared memory is 160 KB of operator
workspace plus the fixed 128-byte instruction slot (163,968 bytes total).

The cross-CTA counter uses SM80 PTX memory semantics: relaxed polling, acquire
confirmation, and release reduction publication after the CTA writers rendezvous.
UpGate → DownProj is split into four independent 2,048-element buckets:
UpGate publishes `block_idx / 128`, and DownProj split `b` waits on slot `b`
for 128 arrivals. The L3 scheduler assigns UpGate blocks strided across SMs so
these buckets can become ready progressively.
The concrete bucket values come from `DecodeScheduleConfig` and are derived through
`SplitKBucketContract`; `instruction_list.cuh` contains only the opcode ABI.
Attention runtime fanout remains an instruction field and is validated against
both the 16-partial reduction workspace and the 25-entry limit imposed by the
32-word serialized instruction before indexing SMEM. The 108-SM schedule emits 13.

## S3 build

```bash
make MK_MODULE_NAME=qimeng_llama3_2_1b_a100_s3_verifier_kernel
```

The S3 session fixture performs this incremental build automatically and identity-checks
the exact verifier DSO before returning its `mk` entrypoint.

After S3 op, layer, and module pass, the managed verifier and production source sets
must be byte-identical. The pytest source-identity check does not modify either tree.

## Handlers with prefetch-before-barrier

- `rms_qkv_gemv_rope.cu`
- `oproj_gemv_residual.cu`
- `rms_upgate_gemv_silu.cu`
- `down_proj_gemv_residual.cu`
- `rms_lm_head.cu`

The public pybind ABI and tensor layouts are unchanged. Q/K/V attention waits
are acquired together in DEPENDENCY_WAIT; the GEMV handlers keep first-tile
weight async ahead of that wait.

`utils.cuh` is the one instantiated Ampere primitive file used by this model.
Knowledge-base dataflow/pipeline directories and template metadata are not
copied into the generated CUDA tree.

Unknown nonzero opcodes trap. Opcode 0 remains the scheduler padding NoOp.
