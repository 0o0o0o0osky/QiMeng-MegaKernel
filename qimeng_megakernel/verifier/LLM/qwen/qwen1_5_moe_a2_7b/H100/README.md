# Qwen1.5 MoE A2.7B H100 verifier

Set `QWEN_MOE_A2_7B_MODEL` to a local checkpoint directory. Tests never download
weights and skip when the checkpoint or H100 is unavailable.

| Stage | Required coverage | Current coverage |
|---|---|---|
| S0 | independent oracle | not applicable; HuggingFace is the baseline |
| S1 | op / layer / module | present, real checkpoint |
| S2 | op / layer / module | present, real checkpoint |
| S3 | op / layer / module | present, real checkpoint |
| S4 | greedy e2e | present, real checkpoint |

Each numeric stage uses the dense test layout plus four MoE files: router, shared
expert, routed expert, and combine. Source and ABI checks do not count as numeric
coverage.
