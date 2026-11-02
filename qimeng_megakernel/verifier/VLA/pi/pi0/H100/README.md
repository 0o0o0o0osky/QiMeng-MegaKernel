# Pi0 H100 verifier

Set `PI0_CKPT`, `PI0_LIBERO_HDF5`, `PI0_TOKENIZER`, and `OPENPI_ROOT` to local
real assets. Thresholds and the view/chunk matrix live only in `verifier_config.py`.

S0 checks independent oracle-backed semantic helpers. S1 through S3
each compare real outputs at operator, layer, and module depth. S3 builds the
verifier-owned kernels in `kernel/`; production inference continues to load only
the prebuilt kernels in the model tree.

S4 contains only `test_e2e_action.py` and compares OpenPI with the native path
for views 1/2/3 and chunks 20/50 using identical explicit noise. The full 3x2
input matrix stays at S4; intermediate tests cover representative shapes and
the view/chunk boundary cases that change a kernel or padded layout.
