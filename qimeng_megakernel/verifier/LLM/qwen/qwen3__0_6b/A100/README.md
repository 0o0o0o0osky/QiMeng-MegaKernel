# Qwen3 0.6B A100 verifier

Set `QWEN3_TEST_MODEL` to a local checkpoint directory. Tests never download
weights and skip when the checkpoint or A100/A800 is unavailable.

| Stage | Required coverage | Current coverage |
|---|---|---|
| S0 | independent oracle | not applicable; HuggingFace is the baseline |
| S1 | op / layer / module | present, real checkpoint |
| S2 | op / layer / module | present, real checkpoint |
| S3 | op / layer / module | present, verifier-owned CUDA kernel |
| S4 | greedy e2e | present, chat template with thinking disabled; 128/256/512/1024 tokens |

Each numeric stage uses only `test_op_*`, `test_layer_*`, and
`test_module_*` responsibilities. The S3 source-identity check is read-only and
does not count as numeric coverage.

After S3 op, layer, and module checks pass, the managed verifier and production source
sets must be byte-identical. Pytest performs this check without copying or deleting
production files.
