#!/usr/bin/env bash
# Build and run the Llama 3.2 1B single-batch Ampere route.
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: bash examples/llama_1b.sh /path/to/Llama-3.2-1B-Instruct [examples.llm options]" >&2
    exit 2
fi
export TARGET_MODEL_PATH="$1"
shift
QIMENG_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
QIMENG_PYTHON="${QIMENG_PYTHON:-python}"
cd "$QIMENG_ROOT"

# Local paths need an explicit model-key mapping. Preserve existing mappings.
"$QIMENG_PYTHON" - <<'PY'
import json
import os
from pathlib import Path

model = Path(os.environ["TARGET_MODEL_PATH"]).expanduser().resolve()
config = json.loads((model / "config.json").read_text())
expected = {
    "model_type": "llama", "hidden_size": 2048, "intermediate_size": 8192,
    "num_hidden_layers": 16, "num_attention_heads": 32,
    "num_key_value_heads": 8, "vocab_size": 128256,
}
for key, value in expected.items():
    if config.get(key) != value:
        raise SystemExit(f"Expected Llama 3.2 1B: {key} must be {value!r}, got {config.get(key)!r}")
registry = Path("qimeng_megakernel/models/LLM/registry.local.json")
data = json.loads(registry.read_text()) if registry.exists() else {}
data[str(model)] = "llama-3.2-1b"
registry.write_text(json.dumps(data, indent=2) + "\n")
PY

make -C qimeng_megakernel/models/LLM/llama/llama3_2__1b/A100/single_batch/decode_kernel \
    -j2 PYTHON="$QIMENG_PYTHON" CUDA_PATH="${CUDA_HOME:-/usr/local/cuda}"

exec "$QIMENG_PYTHON" -m examples.llm \
    --model "$TARGET_MODEL_PATH" --hardware A100 --route mk \
    --prompt "The capital of France is" --raw --max-new-tokens 32 "$@"
