#pragma once

#include "kittens.cuh"
#include "pi0_config.cuh"
#include "ae_instruction_list.cuh"
#include "../include/config.cuh"

namespace pi0 {
namespace dec {

static constexpr int K_SPLIT       = K_GRANULARITY;          // 256
static constexpr int SPLIT_K_ITERS = K_SPLIT / PIPELINE_K_DIM;  // 4

static constexpr int OPROJ_SPLITS = ATTN_IN  / K_SPLIT;      // 2048/256 =  8
static constexpr int DOWN_SPLITS  = MLP_WIDTH / K_SPLIT;     // 4096/256 = 16
static constexpr int FFN_SPLITS   = WIDTH     / K_SPLIT;     // 1024/256 =  4  (GATE/UP_MUL)

static_assert(HEAD_DIM == K_SPLIT, "The attn_out K-segment layout requires HEAD_DIM == K_SPLIT");

static constexpr int KEY_TILE = 64;
constexpr int kv_rows_needed(int prefix_len) {
    return (prefix_len + KEY_TILE - 1) / KEY_TILE * KEY_TILE + KEY_TILE;
}
static constexpr int KV_ROWS     = kv_rows_needed(enc::SEQ);
static constexpr int MAX_KV_ROWS = kv_rows_needed(enc::MAX_SEQ);  // 1088 (prompt=200)

}  // namespace dec

template <typename Config>
struct ae_globals {
    using bf16 = kittens::bf16;

    using act_t = kittens::gl<bf16, 1, 1, -1, dec::WIDTH,
                              kittens::sv_bf<dec::WIDTH>,
                              kittens::st_bf<64, 64>,
                              kittens::st_bf<64, 128>>;

    template <int K_PAD>
    using wgt_t = kittens::gl<bf16, 1, -1, -1, K_PAD,
                              kittens::st_bf<128, 64>,
                              kittens::st_bf<64, 64>,
                              kittens::st_bf<32, 64>>;

    using wgt_split_t = kittens::gl<bf16, 1, -1, -1, dec::K_SPLIT,
                                    kittens::st_bf<64, 64>,
                                    kittens::st_bf<128, 64>>;

    using qkv_t = kittens::gl<bf16, 1, 1, -1, dec::QKV_OUT,
                              kittens::st_bf<64, 128>,
                              kittens::st_bf<64, 64>,
                              kittens::st_bf<64, 32>,
                              kittens::st_bf<64, 256>>;

    using kv_t = kittens::gl<bf16, 1, -1, -1, dec::HEAD_DIM,
                             kittens::st_bf<64, 256>,
                             kittens::st_bf<64, 64>,
                             kittens::st_bf<64, 32>,
                             kittens::st_bf<64, 128>>;

    using attn_partial_t = kittens::gl<bf16, 1, -1, -1, dec::ATTN_IN,
                                       kittens::st_bf<64, 256>,
                                       kittens::st_bf<64, 64>,
                                       kittens::st_bf<64, 32>,
                                       kittens::st_bf<64, 128>>;
    using attn_stats_t   = kittens::gl<float, 1, -1, -1, -1>;

    using attn_out_t = kittens::gl<bf16, 1, 1, -1, dec::K_SPLIT,
                                   kittens::st_bf<64, 32>,
                                   kittens::st_bf<64, 64>,
                                   kittens::st_bf<64, 128>,
                                   kittens::st_bf<64, 256>>;

    using ffn_buf_t = kittens::gl<bf16, 1, 1, -1, dec::K_SPLIT,
                                  kittens::st_bf<64, 64>,
                                  kittens::st_bf<64, 32>,
                                  kittens::st_bf<64, 128>>;

    using acc_t = kittens::gl<float, 1, 1, -1, dec::WIDTH,
                              kittens::sv_fl<dec::WIDTH>,
                              kittens::st_fl<64, 32>,
                              kittens::st_fl<64, 64>,
                              kittens::st_fl<64, 128>>;

    using rms_t = kittens::gl<float, 1, 1, -1, -1>;

    using rope_t = kittens::gl<bf16, 1, 1, -1, dec::HEAD_DIM,
                               kittens::sv_bf<dec::HEAD_DIM>,
                               kittens::st_bf<64, 64>,
                               kittens::st_bf<64, 32>,
                               kittens::st_bf<64, 128>>;

    using noise_t = kittens::gl<bf16, 1, 1, -1, dec::K_SPLIT,
                                kittens::st_bf<64, 64>,
                                kittens::st_bf<64, 32>>;

    using act_h_t = kittens::gl<bf16, 1, 1, -1, dec::WIDTH,
                                kittens::st_bf<64, 32>,
                                kittens::st_bf<64, 64>>;

    using bar_t = kittens::gl<uint, 1, -1, -1, AE_BARRIER_SLOTS>;

    act_t          x;
    acc_t          x_acc;
    rms_t          rms_factor;
    wgt_t<dec::WIDTH_K_PAD> qkv_w;
    qkv_t          qkv_buf;
    rope_t         rope;
    kv_t           k_cache;
    kv_t           v_cache;
    attn_partial_t attn_partial;
    attn_stats_t   attn_stats;
    attn_out_t     attn_out;       // K-split-major
    wgt_split_t    oproj_w;
    wgt_t<dec::WIDTH_K_PAD> gate_w;
    wgt_t<dec::WIDTH_K_PAD> up_w;
    ffn_buf_t      gate_buf;
    ffn_buf_t      ffn_buf;        // K-split-major
    wgt_split_t    down_w;

    noise_t        noise;
    act_h_t        act_h;
    wgt_t<dec::K_SPLIT>     act_w_in;   // [1, 1024, 256], W_in with K padded from 32 to 256
    wgt_t<dec::WIDTH_K_PAD> act_w_mlp;  // [1, 1024, 1024]  W_mlp
    wgt_t<dec::WIDTH_K_PAD> act_w_out;  // [1,   32, 1024]  W_out

    rms_t          act_bias;
    bar_t          Bar;
    ::megakernel::instruction_layout<Config> instructions;

    dim3 grid() const { return dim3(H100_SM_COUNT); }
    dim3 block() const { return dim3(Config::NUM_THREADS); }
    int dynamic_shared_memory() const { return Config::DYNAMIC_SHARED_MEMORY; }
};

using ae_globals_t = ae_globals<action_expert_config>;

}  // namespace pi0
