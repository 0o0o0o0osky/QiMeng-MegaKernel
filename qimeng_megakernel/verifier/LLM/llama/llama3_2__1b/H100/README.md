# Llama 3.2 1B H100 verifier

Set `MODEL_LLAMA3_2_1B` to a local checkpoint directory. Tests never download
weights and skip when the checkpoint or H100 is unavailable.

| Stage | Required coverage | Current coverage |
|---|---|---|
| S0 | independent oracle | not applicable; HuggingFace is the baseline |
| S1 | op / layer / module | present, real checkpoint |
| S2 | op / layer / module | present, real checkpoint |
| S3 | op / layer / module | present, real checkpoint |
| S4 | greedy e2e | present, real checkpoint; 128/256/512/1024 tokens |

The two decode routes are explicit siblings:

- `single_batch/` contains S1 through S4;
- `multi_batch/` contains S1 through S4 for batch-dependent storage,
  scheduling, masks, ragged rows, replay, and generation.

Both routes keep the same six op files plus layer and module files in S1--S3.
They share the mathematical HF baseline helper, while each test executes its
own route shapes and scheduler.  Both S4 routes run their own end-to-end
generation tests.  Each route owns a verifier-side
`decode_kernel/`; S3 builds that private DSO, and the read-only source-identity
test checks it against the corresponding production source after validation.
