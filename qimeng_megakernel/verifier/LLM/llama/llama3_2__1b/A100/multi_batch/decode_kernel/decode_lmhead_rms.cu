// Final RMS placement selected by this instance. One CTA owns one batch row.

#include "globals.cuh"
#include "llama_config.cuh"
#include "kernel_config.cuh"
#include "instruction_list.cuh"
#include "utils.cuh"

#include <cute/tensor.hpp>

namespace megakernel {

using namespace cute;

__device__ __forceinline__ void decode_lmhead_rms(
    LlamaGlobals& g,
    SmemLayout& smem,
    int inst_idx)
{
    const int32_t* inst = smem.instruction;
    int batch_size = inst[1];
    int batch_row  = inst[6];
    int tid        = threadIdx.x;
    int sm_id      = blockIdx.x;
    uint8_t* smem_base = smem.main_workspace;
    float* reduce_ws = smem_ptr_at<float>(smem_base, LmheadGemmSmemLayout::kOffsetReduceWorkspace);

    if (sm_id != batch_row || batch_row < 0 || batch_row >= batch_size) {
        return;
    }

    auto barriers = g.get_barrier();
    volatile int32_t* input_ready = &barriers(
        LlamaConfig::NUM_LAYERS - 1, OPCODE_DOWN_PROJ_RESIDUAL - 1, 0);
    if (tid == 0) {
        barrier_wait(
            input_ready,
            LlamaConfig::HIDDEN_DIM / KernelConfig::kTileN,
            sm_id,
            OPCODE_LMHEAD_RMS,
            0,
            LlamaConfig::NUM_LAYERS - 1,
            &g,
            inst_idx);
    }
    __syncthreads();

    const bfloat16_t* row_in =
        g.hidden_state_ptr + (long)batch_row * LlamaConfig::HIDDEN_DIM;

    auto out = g.get_lm_head_rms_out();
    rms_norm_one_row_to_gmem<
        KernelConfig::NUM_WARPS>(
        row_in,
        g.lm_head_norm_ptr,
        g.rms_norm_eps,
        reduce_ws,
        out,
        batch_row,
        row_active(g, batch_row));

    __syncthreads();
    if (tid == 0) {
        barrier_publish(
            &barriers(LlamaConfig::NUM_LAYERS - 1, OPCODE_LMHEAD_RMS - 1, 0),
            1);
    }
}

}  // namespace megakernel
