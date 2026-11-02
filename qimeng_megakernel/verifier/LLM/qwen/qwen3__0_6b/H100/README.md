# Qwen3 0.6B H100 verifier

Set `QWEN3_TEST_MODEL` to a local checkpoint directory. Tests never download weights
and skip when the checkpoint or H100 is unavailable.

| Stage | Required coverage | Current coverage |
|---|---|---|
| S0 | independent oracle | not applicable; HuggingFace is the baseline |
| S1 | op / layer / module | present, real checkpoint |
| S2 | op / layer / module | present, real checkpoint |
| S3 | op / layer / module | present, real checkpoint |
| S4 | greedy e2e | present, chat template with thinking disabled; 128/256/512/1024 tokens |

Each numeric stage uses only `test_op_*`, `test_layer_*`, and `test_module_*`
responsibilities. Source and ABI checks do not count as numeric coverage.
