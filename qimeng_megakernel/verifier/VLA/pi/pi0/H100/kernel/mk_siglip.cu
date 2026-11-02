

#include "kittens.cuh"
#include "pyutils/pyutils.cuh"

#include "common/pi0_config.cuh"
#include "common/hardware_config.cuh"
#include "common/globals.cuh"
#include "common/instruction_list.cuh"

#ifndef PI0_SIGLIP_VARIANT
#define PI0_SIGLIP_VARIANT 2
#endif
#if PI0_SIGLIP_VARIANT == 1
#include "backbone/siglip/variants/view1/config.cuh"
#elif PI0_SIGLIP_VARIANT == 2
#include "backbone/siglip/variants/view2/config.cuh"
#elif PI0_SIGLIP_VARIANT == 3
#include "backbone/siglip/variants/view3/config.cuh"
#else
#error "PI0_SIGLIP_VARIANT must be 1, 2, or 3"
#endif

#ifndef PI0_SIGLIP_PY_MODULE
#define PI0_SIGLIP_PY_MODULE mk_siglip
#endif

#include "include/megakernel.cuh"

#include "backbone/siglip/siglip_ln.cu"
#include "backbone/siglip/siglip_gemm.cu"
#include "backbone/siglip/siglip_attn.cu"

using config  = pi0::backbone_config;
using globals = pi0::siglip_globals_t;

using ln_attn_t = pi0::siglip_stats_attn_op<config, globals>;

using ln_ffn_t  = pi0::siglip_stats_ffn_op<config, globals>;
using attn_t    = pi0::siglip_attn_op<config, globals>;

using qkv_t = pi0::siglip_gemm<config, globals, pi0::EpiLnFold,
                               &globals::vision_x, &globals::qkv_w, &globals::qkv_b, &globals::qkv_buf,
                               pi0::vis::WIDTH_K_ITERS, OPCODE_SIGLIP_QKV, BARRIER_ROW(OPCODE_SIGLIP_QKV),
                               &globals::vision_x, /*_ExplicitOut=*/false, &globals::ln_fold_qkv,
                               /*_StatsTiles=*/1, pi0::siglip_variant::QKV_TILE_N>;

using oproj_t = pi0::siglip_gemm<config, globals, pi0::EpiBiasResidStats,
                                 &globals::attn_out, &globals::oproj_w, &globals::oproj_b, &globals::vision_x,
                                 pi0::vis::WIDTH_K_ITERS, OPCODE_SIGLIP_OPROJ, BARRIER_ROW(OPCODE_SIGLIP_OPROJ),
                                 &globals::vision_x, /*_ExplicitOut=*/false, &globals::oproj_b,
                                 pi0::siglip_variant::OPROJ_STATS_TILES,
                                 pi0::siglip_variant::OPROJ_TILE_N>;

using fc1_t = pi0::siglip_gemm<config, globals, pi0::EpiLnFoldGelu,
                               &globals::vision_x, &globals::fc1_w, &globals::fc1_b, &globals::fc1_buf,
                               pi0::vis::WIDTH_K_ITERS, OPCODE_SIGLIP_FC1, BARRIER_ROW(OPCODE_SIGLIP_FC1),
                               &globals::vision_x, /*_ExplicitOut=*/true, &globals::ln_fold_fc1,
                               pi0::siglip_variant::OPROJ_STATS_TILES,
                               /*_TileN=*/128>;

using fc2_t = pi0::siglip_gemm<config, globals, pi0::EpiBiasResidual,
                               &globals::fc1_buf, &globals::fc2_w, &globals::fc2_b, &globals::vision_x,
                               globals::FC2_SPLIT_ITERS, OPCODE_SIGLIP_FC2, BARRIER_ROW(OPCODE_SIGLIP_FC2),
                               &globals::fc1_buf, /*_ExplicitOut=*/true>;

using patch_t = pi0::siglip_gemm<config, globals, pi0::EpiBiasResidual,
                                 &globals::patch_in, &globals::patch_w, &globals::patch_b,
                                 &globals::vision_x,
                                 pi0::vis::PATCH_K_ITERS, OPCODE_SIGLIP_PATCH, BARRIER_ROW(OPCODE_SIGLIP_PATCH)>;

using final_norm_t = pi0::siglip_ln<config, globals,
                                    OPCODE_SIGLIP_FINAL_NORM, BARRIER_ROW(OPCODE_SIGLIP_FINAL_NORM)>;
using projector_t = pi0::siglip_gemm<config, globals, pi0::EpiBias,
                                     &globals::siglip_x_normed, &globals::proj_w, &globals::proj_b,
                                     &globals::proj_out,
                                     pi0::vis::WIDTH_K_ITERS, OPCODE_SIGLIP_PROJECTOR,
                                     BARRIER_ROW(OPCODE_SIGLIP_PROJECTOR)>;

PYBIND11_MODULE(PI0_SIGLIP_PY_MODULE, m) {
    m.doc() = "Pi0 SigLIP vision-tower MegaKernel (H100 / sm_90a)";

    kittens::py::bind_kernel<
        ::megakernel::mk<config, globals,
                         ln_attn_t,
                         ln_ffn_t,
                         qkv_t,
                         attn_t,
                         oproj_t,
                         fc1_t,
                         fc2_t,
                         patch_t,
                         final_norm_t,
                         projector_t>
    >(m, "mk",

      &globals::vision_x,
      &globals::siglip_x_normed,
      &globals::ln_gamma,
      &globals::ln_beta,
      &globals::qkv_w,
      &globals::qkv_b,
      &globals::qkv_buf,
      &globals::attn_out,
      &globals::oproj_w,
      &globals::oproj_b,
      &globals::fc1_w,
      &globals::fc1_b,
      &globals::fc1_buf,
      &globals::fc2_w,
      &globals::fc2_b,
      &globals::patch_in,
      &globals::patch_w,
      &globals::patch_b,
      &globals::proj_w,
      &globals::proj_b,
      &globals::proj_out,
      &globals::ln_stats,
      &globals::ln_fold_qkv,
      &globals::ln_fold_fc1,
      &globals::Bar,
      &globals::instructions);
}
