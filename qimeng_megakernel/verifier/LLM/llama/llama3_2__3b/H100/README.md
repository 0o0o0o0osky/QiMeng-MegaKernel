# Llama 3.2 3B H100 verifier

Set `LLAMA_32_3B_MODEL_PATH` to a local checkpoint directory. Tests never download
weights and skip when the checkpoint or H100 is unavailable.

| Stage | Required coverage | Current coverage |
|---|---|---|
| S0 | independent oracle | not applicable; HuggingFace is the baseline |
| S1 | op / layer / module | present, real checkpoint |
| S2 | op / layer / module | present, real checkpoint |
| S3 | op / layer / module | present, real checkpoint |
| S4 | greedy e2e | present, real checkpoint; 128/256/512/1024 tokens |

Each numeric stage uses only `test_op_*`, `test_layer_*`, and `test_module_*`
responsibilities. Source and ABI checks do not count as numeric coverage.
