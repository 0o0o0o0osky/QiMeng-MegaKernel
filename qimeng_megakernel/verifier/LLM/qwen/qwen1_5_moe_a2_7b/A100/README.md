# Qwen1.5 MoE A2.7B A100 verifier

Set `QWEN_MOE_A2_7B_MODEL` to a local checkpoint directory. Tests never download
weights and skip when the checkpoint or A100 is unavailable.

| Stage | Required coverage | Current coverage |
|---|---|---|
| S0 | independent oracle | not applicable; HuggingFace is the baseline |
| S1 | op / layer / module | present, real checkpoint |
| S2 | op / layer / module | present, real checkpoint |
| S3 | op / layer / module | present, real checkpoint |
| S4 | greedy e2e | present, real checkpoint |

Each numeric stage uses the dense test layout plus four MoE files: router, shared
expert, routed expert, and combine. Source and host/device argument-order checks do
not count as numeric coverage.

S3 validates the verifier-owned CUDA kernel first. After its op, layer, and module
checks pass, the managed verifier and production source sets must be byte-identical.
The source-identity pytest performs this check without copying or deleting production
files.
