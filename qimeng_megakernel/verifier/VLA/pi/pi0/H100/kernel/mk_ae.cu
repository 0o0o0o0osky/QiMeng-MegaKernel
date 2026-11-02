

#include "kittens.cuh"
#include "pyutils/pyutils.cuh"

#include "common/pi0_config.cuh"
#include "common/hardware_config.cuh"
#include "common/ae_globals.cuh"

#include "include/megakernel.cuh"
#include "action_expert/ae_rmsn.cu"
#include "action_expert/ae_qkv_rope.cu"
#include "action_expert/ae_attn.cu"
#include "action_expert/ae_gemm.cu"
#include "action_expert/ae_ffn_dual.cu"
#include "action_expert/ae_action.cu"

using config  = pi0::action_expert_config;
using globals = pi0::ae_globals_t;

using rmsn_attn_t = pi0::ae_rmsn_attn_op<config, globals>;
using qkv_t       = pi0::ae_qkv_rope_op<config, globals>;
using rmsn_ffn_t  = pi0::ae_rmsn_ffn_op<config, globals>;

using attn_partial_t = pi0::ae_attn_partial_op<config, globals>;
using attn_reduce_t  = pi0::ae_attn_reduce_op<config, globals>;

using oproj_t = pi0::ae_gemm<config, globals, pi0::AeEpiStoreAdd,
                             &globals::attn_out, &globals::oproj_w, &globals::x_acc,
                             pi0::dec::SPLIT_K_ITERS, OPCODE_AE_OPROJ>;

using gate_t = pi0::ae_gemm<config, globals, pi0::AeEpiRowScaleGelu,
                            &globals::x, &globals::gate_w, &globals::gate_buf,
                            pi0::dec::WIDTH_K_ITERS, OPCODE_AE_GATE,
                            &globals::x, /*TILE_N=*/32>;

// UP_MUL: A = x, B = up_w, row_scale * gelu(gate) -> K-split-major ffn_buf.
using up_mul_t = pi0::ae_gemm<config, globals, pi0::AeEpiRowScaleGate,
                              &globals::x, &globals::up_w, &globals::ffn_buf,
                              pi0::dec::WIDTH_K_ITERS, OPCODE_AE_UP_MUL,
                              &globals::gate_buf, /*TILE_N=*/32>;

using down_t = pi0::ae_gemm<config, globals, pi0::AeEpiStoreAdd,
                            &globals::ffn_buf, &globals::down_w, &globals::x_acc,
                            pi0::dec::SPLIT_K_ITERS, OPCODE_AE_DOWN>;

using ffn_dual_t = pi0::ae_ffn_dual_op<config, globals>;

using act_in_t   = pi0::ae_act_in_op<config, globals>;
using act_mlp_t  = pi0::ae_act_mlp_op<config, globals>;
using rmsn_out_t = pi0::ae_rmsn_out_op<config, globals>;
using act_out_t  = pi0::ae_act_out_op<config, globals>;

#ifndef PI0_AE_PY_MODULE
#define PI0_AE_PY_MODULE mk_ae
#endif

PYBIND11_MODULE(PI0_AE_PY_MODULE, m) {
    m.doc() = "Pi0 Action Expert MegaKernel (H100 / sm_90a)";
    kittens::py::bind_kernel<
        ::megakernel::mk<config, globals, rmsn_attn_t, rmsn_ffn_t, qkv_t,
                         attn_partial_t, attn_reduce_t,
                         oproj_t, down_t, ffn_dual_t,
                         act_in_t, act_mlp_t, rmsn_out_t, act_out_t>
    >(m, "mk",

      &globals::x, &globals::x_acc, &globals::rms_factor, &globals::qkv_w, &globals::qkv_buf,
      &globals::rope, &globals::k_cache, &globals::v_cache,
      &globals::attn_partial, &globals::attn_stats, &globals::attn_out,
      &globals::oproj_w, &globals::gate_w, &globals::up_w,
      &globals::gate_buf, &globals::ffn_buf, &globals::down_w,
      &globals::noise, &globals::act_h, &globals::act_w_in, &globals::act_w_mlp,
      &globals::act_w_out, &globals::act_bias,
      &globals::Bar, &globals::instructions);
}
