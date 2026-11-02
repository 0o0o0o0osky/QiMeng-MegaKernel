#pragma once

#include "kittens.cuh"
#include "pi0_config.cuh"
#include "gemma_instruction_list.cuh"
#include "../include/config.cuh"

namespace pi0 {

template <typename Config>
struct gemma_globals {
    using bf16 = kittens::bf16;

    using act_t = kittens::gl<bf16, 1, 1, -1, enc::WIDTH,
                             kittens::sv_bf<enc::WIDTH>,
                             kittens::st_bf<64, 64>,
                             kittens::st_bf<64, 128>>;

    template <int K_PAD>
    using wgt_t = kittens::gl<bf16, 1, -1, -1, K_PAD, kittens::st_bf<128, 64>>;

    using qkv_t = kittens::gl<bf16, 1, 1, -1, enc::QKV_OUT,
                             kittens::st_bf<64, 128>,
                             kittens::st_bf<64, 256>>;

    using mlp_t = kittens::gl<bf16, 1, 1, -1, enc::MLP_WIDTH,
                             kittens::st_bf<64, 128>,
                             kittens::st_bf<64, 64>>;

    using kv_t = kittens::gl<bf16, 1, -1, -1, enc::HEAD_DIM,
                            kittens::st_bf<128, 256>,
                            kittens::st_bf<64, 256>,
                            kittens::st_bf<64, 128>>;

    using rms_t = kittens::gl<float, 1, 1, -1, -1>;

    using rope_t = kittens::gl<bf16, 1, 1, -1, enc::HEAD_DIM,
                              kittens::sv_bf<enc::HEAD_DIM>,
                              kittens::st_bf<64, 128>>;

    //   partial O   [splits, SEQ_PAD, 2048] bf16

    using attn_partial_t = kittens::gl<bf16, 1, -1, -1, enc::ATTN_OUT,
                                      kittens::st_bf<64, 256>,
                                      kittens::st_bf<64, 128>>;
    using attn_stats_t   = kittens::gl<float, 1, -1, -1, -1>;

    using bar_t = kittens::gl<uint, 1, -1, -1, GEMMA_BARRIER_SLOTS>;

    act_t  encoder_x;
    rms_t  rms_factor;
    wgt_t<enc::WIDTH_K_PAD> qkv_w;
    qkv_t  qkv_buf;
    rope_t rope;
    kv_t   k_cache;
    kv_t   v_cache;
    attn_partial_t attn_partial;
    attn_stats_t   attn_stats;
    act_t  attn_out;
    wgt_t<enc::WIDTH_K_PAD> oproj_w;
    wgt_t<enc::WIDTH_K_PAD> gate_w;
    wgt_t<enc::WIDTH_K_PAD> up_w;
    mlp_t  gate_buf;
    mlp_t  up_buf;
    wgt_t<enc::MLP_K_PAD>   down_w;
    bar_t  Bar;
    ::megakernel::instruction_layout<Config> instructions;

    dim3 grid() const { return dim3(H100_SM_COUNT); }
    dim3 block() const { return dim3(Config::NUM_THREADS); }
    int dynamic_shared_memory() const { return Config::DYNAMIC_SHARED_MEMORY; }
};

using gemma_globals_t = gemma_globals<backbone_config>;

}  // namespace pi0
