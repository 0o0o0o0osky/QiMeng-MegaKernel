# Pi0 A100 verifier

Set `PI0_CKPT`, `PI0_LIBERO_HDF5`, `PI0_TOKENIZER`, and `OPENPI_ROOT` to local
real assets. Thresholds and the view/chunk matrix live only in `verifier_config.py`.

S0 checks independent oracle-backed semantic helpers. S1 and S2
compare real outputs at operator, layer, and module depth. S3 builds three
verifier-owned native DSOs in `kernel/`. Its layer and module tests drive the
same host-static SigLIP, Gemma, and Action Expert programs used during CUDA
Graph capture; those programs combine shape-specialized static subgraph MegaKernel
entrypoints with required direct native kernels. Dedicated S3 operator checks cover
the AE input projection, Gemma RMS normalization, and SigLIP projector direct leaves,
plus the SigLIP attention-chain and MLP and Gemma QKV-QK and PV-OProj static
subgraphs at every compiled view specialization. This is not a claim that every
direct leaf has dedicated operator coverage. The managed model and verifier CUDA/C++
source sets are byte-identical; their build configurations must be checked separately.

S4 contains only `test_e2e_action.py` and compares OpenPI with the native path
for views 1/2/3 and chunks 20/50 using identical explicit noise. The full 3x2
input matrix stays at S4; intermediate tests cover representative shapes and
the view/chunk boundary cases that change a kernel or padded layout.
