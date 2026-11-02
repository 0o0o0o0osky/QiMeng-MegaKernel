

#include "kittens.cuh"
#include "pyutils/pyutils.cuh"

#include "common/pi0_config.cuh"
#include "common/hardware_config.cuh"
#include "common/gemma_globals.cuh"
#include "common/gemma_instruction_list.cuh"

#include "include/megakernel.cuh"
#include "backbone/gemma/gemma_rmsn.cu"
#include "backbone/gemma/gemma_attn.cu"
#include "backbone/gemma/gemma_qkv_rope.cu"
#include "backbone/gemma/gemma_ffn_dual.cu"
#include "backbone/gemma/gemma_resid_gemm.cu"
#include "backbone/gemma/gemma_down_splitk.cu"

using config  = pi0::backbone_config;
using globals = pi0::gemma_globals_t;

using rmsn_attn_t = pi0::gemma_rmsn_attn_op<config, globals>;
using rmsn_ffn_t  = pi0::gemma_rmsn_ffn_op<config, globals>;

using qkv_t = pi0::gemma_qkv_rope_op<config, globals>;

using oproj_t = pi0::gemma_resid_stats_gemm<config, globals,
                                            &globals::attn_out, &globals::oproj_w, &globals::encoder_x,
                                            pi0::enc::WIDTH_K_ITERS, OPCODE_GEMMA_OPROJ,
                                            BARRIER_ROW(OPCODE_GEMMA_OPROJ), pi0::enc::RMS_STATS_FFN_ROW0,

                                            &globals::gate_buf, /*WRITE_SHADOW=*/false>;

using ffn_t = pi0::gemma_ffn_dual_op<config, globals>;

using down_t = pi0::gemma_resid_stats_gemm<config, globals,
                                           &globals::up_buf, &globals::down_w, &globals::encoder_x,
                                           pi0::enc::MLP_K_ITERS, OPCODE_GEMMA_DOWN,
                                           BARRIER_ROW(OPCODE_GEMMA_DOWN), pi0::enc::RMS_STATS_ATTN_ROW0>;

static constexpr int GEMMA_DOWN_SPLITS = 2;
using down_splitk_t = pi0::gemma_splitk_resid_gemm<config, globals,
                                                   &globals::up_buf, &globals::down_w, &globals::encoder_x,
                                                   pi0::enc::MLP_K_ITERS / GEMMA_DOWN_SPLITS,
                                                   OPCODE_GEMMA_DOWN_SPLITK,
                                                   BARRIER_ROW(OPCODE_GEMMA_DOWN_SPLITK)>;

using attn_p_t = pi0::gemma_attn_partial_op<config, globals>;
using attn_r_t = pi0::gemma_attn_reduce_op<config, globals>;

#ifndef PI0_GEMMA_PY_MODULE
#define PI0_GEMMA_PY_MODULE mk_gemma
#endif

PYBIND11_MODULE(PI0_GEMMA_PY_MODULE, m) {
    m.doc() = "Pi0 PaliGemma encoder MegaKernel (H100 / sm_90a)";
    kittens::py::bind_kernel<
        ::megakernel::mk<config, globals,
                         rmsn_attn_t, rmsn_ffn_t, qkv_t, attn_p_t, attn_r_t,
                         oproj_t, ffn_t, down_t, down_splitk_t>
    >(m, "mk",

      &globals::encoder_x, &globals::rms_factor, &globals::qkv_w, &globals::qkv_buf,
      &globals::rope, &globals::k_cache, &globals::v_cache,
      &globals::attn_partial, &globals::attn_stats, &globals::attn_out,
      &globals::oproj_w, &globals::gate_w, &globals::up_w,
      &globals::gate_buf, &globals::up_buf, &globals::down_w,
      &globals::Bar, &globals::instructions);
}
