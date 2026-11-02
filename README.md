# QiMeng-MegaKernel

QiMeng-MegaKernel provides CUDA megakernel inference implementations and layered verification against fused PyTorch and Python virtual-machine references. The Python package is `qimeng_megakernel`.

## Quick start: Llama 3.2 1B

Supported `--hardware` options are `A100` and `H100`. The example below uses `A100` for single-batch BF16 inference with fused prefill and a decode megakernel compiled for Llama 3.2 1B dimensions.

Prepare a complete local Hugging Face checkpoint with `config.json`, tokenizer files, and safetensors weights. The default example uses raw text completion and accepts Base or Instruct checkpoints. Chat examples require an Instruct checkpoint with a chat template.

### 1. Install

From the repository root:

```bash
git submodule update --init --recursive
conda create -n qimeng python=3.11 -y
conda activate qimeng

# Install CUDA-enabled PyTorch compatible with your driver and toolkit first.
python -m pip install -e '.[dev]'

export QIMENG_ROOT="$PWD"
export QIMENG_PYTHON="$(command -v python)"
export CUDA_HOME=/usr/local/cuda
export TARGET_MODEL_PATH="/path/to/Llama-3.2-1B-Instruct"
```

A C++17 compiler and the CUDA toolkit (`nvcc`) are required. Set `CUDA_HOME` to your toolkit installation, and use the same Python/PyTorch environment for compilation and inference. CUTLASS and ThunderKittens are pinned Git submodules; they do not need separate Python package installation.

### 2. Build and generate

```bash
bash examples/llama_1b.sh "$TARGET_MODEL_PATH"
```

The [example script](examples/llama_1b.sh) builds the MegaKernel for Llama 3.2 1B and runs inference using the specified local checkpoint. Subsequent runs reuse the compiled kernel.

Example output:

```text
[mk] Prefill done; entering decode path.
[mk] Running decode (mk path).
[0]  Paris. The Eiffel Tower is located in Paris. The Eiffel Tower is one of the most famous landmarks in the world.
```

### 3. Customize inference

After building the kernel and registering the checkpoint path, use the shared CLI directly:

```bash
"$QIMENG_PYTHON" -m examples.llm \
  --model "$TARGET_MODEL_PATH" --hardware A100 --route mk \
  --prompt 'The capital of France is' --raw --max-new-tokens 64
```

Use `--route mk` to run MegaKernel inference.

For an Instruct checkpoint, use the default chat formatting:

```bash
"$QIMENG_PYTHON" -m examples.llm \
  --model "$TARGET_MODEL_PATH" --hardware A100 --route mk \
  --prompt 'Explain what a megakernel is in two sentences.' --max-new-tokens 64
```

## Verification

Run the single-batch Llama 1B end-to-end generation tests with the same local checkpoint. The suite covers output lengths of 128, 256, 512, and 1024 tokens and uses Hugging Face as the reference:

```bash
export MODEL_LLAMA3_2_1B="$TARGET_MODEL_PATH"
"$QIMENG_PYTHON" -m pytest -q \
  qimeng_megakernel/verifier/LLM/llama/llama3_2__1b/A100/single_batch/e2e_generate
```

Run the shared verifier unit tests:

```bash
"$QIMENG_PYTHON" -m pytest -q qimeng_megakernel/verifier/common/tests
```

## Decode throughput

After completing the quick start, run the fixed-length decode benchmark:

```bash
"$QIMENG_PYTHON" -m benchmarks.llm_decode \
  --model-path "$TARGET_MODEL_PATH" --hardware A100 \
  --batch-sizes 1 --output-lengths 128,256,512,1024 \
  --warmup-iters 2 --bench-iters 5 --out /tmp/qimeng-llama1b-benchmark.json
```

The default prompt is raw `Write a long story.`. The benchmark forces continuation to the requested length, including past EOS. It reports decode-loop wall throughput (`tok_s`), GPU elapsed-time throughput (`tok_s_gpu`), and megakernel-only throughput (`kernel_tok_s`) separately. These measurements exclude prefill and checkpoint loading; kernel-only throughput also excludes sampling and other work outside the megakernel.

## Repository layout

| Path | Purpose |
|---|---|
| `examples/llm.py` | Shared text-generation CLI |
| `examples/llama_1b.sh` | Llama 1B build, registration, and inference example |
| `qimeng_megakernel/models/LLM/` | LLM runtime, model configurations, and CUDA kernels |
| `qimeng_megakernel/models/VLA/` | VLA implementations |
| `qimeng_megakernel/verifier/` | Layered numerical and end-to-end verification |
| `benchmarks/llm_decode.py` | LLM decode benchmark |

Supported model, hardware, and batch combinations are defined by the corresponding implementations and verifier suites.

## License

See [LICENSE](LICENSE) and [NOTICE](NOTICE) for the license, original copyright attribution, and third-party notices.
