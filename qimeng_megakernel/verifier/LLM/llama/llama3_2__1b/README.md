# Llama 3.2 1B — Verifier

Verifier suites are organized by hardware:

| Hardware | Path | Status |
|------|------|------|
| **A100** | `A100/single_batch/`, `A100/multi_batch/` | Implemented with S1--S4 hierarchical verification for each production route. |
| **H100** | `H100/single_batch/`, `H100/multi_batch/` | Implemented with S1--S4 hierarchical verification for each production route. |

- See `A100/README.md` for A100 test commands.
- Verifier modules use the path
  `qimeng_megakernel.verifier.LLM.llama.llama3_2__1b.<hardware>.<route>.<stage>`,
  for example `A100.single_batch.baseline_to_fused`.
