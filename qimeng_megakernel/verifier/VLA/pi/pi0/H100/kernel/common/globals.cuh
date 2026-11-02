#pragma once

#include "kittens.cuh"
#include "pi0_config.cuh"
#include "instruction_list.cuh"
#include "../include/config.cuh"

namespace pi0 {

template <typename Config>
struct siglip_globals {
    using bf16 = kittens::bf16;

    static constexpr int ACT_PAD = vis::WIDTH_K_PAD;  // 1280
    static_assert(ACT_PAD % 256 == 0, "activation pad must suit st_bf<*,256>");

    using act_t = kittens::gl<bf16, 1, 1, -1, ACT_PAD,
                             kittens::sv_bf<vis::WIDTH>,
                             kittens::st_bf<64, 64>,
                             kittens::st_bf<64, 80>,
                             kittens::st_bf<64, 128>>;

    using lnw_t = kittens::gl<bf16, 1, 1, -1, vis::WIDTH,
                             kittens::sv_bf<vis::WIDTH>>;

    template <int K_PAD>
    using wgt_t = kittens::gl<bf16, 1, -1, -1, K_PAD,
                              kittens::st_bf<128, 64>, kittens::st_bf<64, 64>>;

    template <int N_PAD>
    using out_t = kittens::gl<bf16, 1, 1, -1, N_PAD,
                             kittens::st_bf<64, 128>,
                             kittens::st_bf<64, 64>>;

    using qkv_t = kittens::gl<bf16, 1, 1, -1, 3840,
                             kittens::st_bf<64, 128>,
                             kittens::st_bf<64, 64>,
                             kittens::st_bf<64, 80>,
                             kittens::st_bf<128, 80>>;

    using bias_t = kittens::gl<bf16, 1, -1, -1, 128,
                               kittens::sv_bf<128>, kittens::sv_bf<64>>;

    using patch_t = kittens::gl<bf16, 1, 1, -1, vis::PATCH_K_PAD, kittens::st_bf<64, 64>>;

    using proj_t = kittens::gl<bf16, 1, 1, -1, enc::WIDTH, kittens::st_bf<64, 128>>;

    // Barrier layout: {layer, row, slot}, where row == opcode - 1.
    using bar_t = kittens::gl<uint, 1, -1, -1, BARRIER_SLOTS>;

    using attn_o_t = kittens::gl<bf16, 1, 1, -1, vis::WIDTH_K_PAD,
                                kittens::st_bf<64, 80>,
                                kittens::st_bf<64, 64>,
                                kittens::st_bf<64, 128>>;

    static constexpr int HEAD_DIM_PAD = vis::HEAD_DIM_PAD;         // 80
    static constexpr int QKV_SECTION  = vis::NUM_HEADS * HEAD_DIM_PAD;  // 1280
    static constexpr int QKV_N_PAD    = 3 * QKV_SECTION;           // 3840
    static constexpr int MLP_N_PAD  = 4608;
    static constexpr int WID_N_PAD  = 1280;

    static constexpr int MLP_K_PAD_SK   = 4608;
    static constexpr int FC2_SPLITS     = 6;                       // 4608/6 = 768
    static constexpr int FC2_K_SEG      = MLP_K_PAD_SK / FC2_SPLITS;          // 768
    static constexpr int FC2_SPLIT_ITERS = FC2_K_SEG / PIPELINE_K_DIM;        // 12
    static_assert(FC2_K_SEG % K_GRANULARITY == 0,
                  "The segment width must be a multiple of K_GRANULARITY (256)");
    static_assert(FC2_SPLIT_ITERS % PIPELINE_K_STAGES == 0,
                  "The iteration count of each stream must be a multiple of PIPELINE_K_STAGES (4)");
    static_assert(MLP_N_PAD == MLP_K_PAD_SK, "The FC1 N padding must equal the FC2 K padding");

    act_t vision_x;
    act_t siglip_x_normed;
    lnw_t ln_gamma;
    lnw_t ln_beta;

    wgt_t<vis::WIDTH_K_PAD> qkv_w;
    bias_t                  qkv_b;
    qkv_t                   qkv_buf;     // [768, 3840]

    attn_o_t                attn_out;
    wgt_t<vis::WIDTH_K_PAD> oproj_w;    // [L, 1280, 1280]
    bias_t                  oproj_b;

    wgt_t<vis::WIDTH_K_PAD> fc1_w;
    bias_t                  fc1_b;

    out_t<FC2_K_SEG>        fc1_buf;

    wgt_t<FC2_K_SEG>        fc2_w;
    bias_t                  fc2_b;

    patch_t                 patch_in;
    wgt_t<vis::PATCH_K_PAD> patch_w;     // [1, 1280, 768]
    bias_t                  patch_b;
    wgt_t<vis::WIDTH_K_PAD> proj_w;      // [1, 2048, 1280]  vision -> encoder
    bias_t                  proj_b;
    proj_t                  proj_out;    // [768, 2048]

    using stats_t = kittens::gl<float, 1, 1, -1, -1>;

    using lnfold_t = kittens::gl<bf16, 1, -1, -1, 128,
                                 kittens::sv_bf<128>, kittens::sv_bf<64>>;

    static constexpr int LN_STATS_TILES = WID_N_PAD / 128;

    static constexpr int LN_STATS_ATTN_ROW = 2 * LN_STATS_TILES;   // 20

    static constexpr int LN_STATS_FFN_ROW  = LN_STATS_ATTN_ROW + 2;  // 22
    static constexpr int LN_STATS_ROWS     = LN_STATS_FFN_ROW + 2 * LN_STATS_TILES;  // 42

    stats_t  ln_stats;
    lnfold_t ln_fold_qkv;
    lnfold_t ln_fold_fc1;

    bar_t Bar;
    ::megakernel::instruction_layout<Config> instructions;

    dim3 grid() const { return dim3(H100_SM_COUNT); }
    dim3 block() const { return dim3(Config::NUM_THREADS); }
    int dynamic_shared_memory() const { return Config::DYNAMIC_SHARED_MEMORY; }
};

using siglip_globals_t = siglip_globals<backbone_config>;

}  // namespace pi0
