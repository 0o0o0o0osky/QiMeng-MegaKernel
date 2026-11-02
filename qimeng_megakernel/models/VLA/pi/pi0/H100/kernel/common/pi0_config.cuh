#pragma once

#include "kittens.cuh"

namespace pi0 {

static constexpr float GEMMA_RMS_EPS = 1.0e-6f;

// K padding keeps the iteration count divisible by the four pipeline stages.

static constexpr int PIPELINE_K_DIM   = 64;
static constexpr int PIPELINE_K_STAGES = 4;
static constexpr int K_GRANULARITY    = PIPELINE_K_DIM * PIPELINE_K_STAGES;  // 256

constexpr int k_pad(int k)   { return (k + K_GRANULARITY - 1) / K_GRANULARITY * K_GRANULARITY; }
constexpr int k_iters(int k) { return k_pad(k) / PIPELINE_K_DIM; }

namespace vis {
static constexpr int NUM_VIEWS       = 3;
static constexpr int TOKENS_PER_VIEW = 256;      // 224x224, patch 14 -> 16x16
static constexpr int SEQ             = NUM_VIEWS * TOKENS_PER_VIEW;
static constexpr int LAYERS          = 27;
static constexpr int WIDTH           = 1152;
static constexpr int MLP_WIDTH       = 4304;
static constexpr int NUM_HEADS       = 16;
static constexpr int HEAD_DIM        = 72;       // 1152 / 16

static constexpr int HEAD_DIM_PAD    = 80;

static constexpr int QKV_OUT         = 3 * WIDTH;               // 3456
static constexpr int PATCH_K         = 3 * 14 * 14;             // 588

static constexpr int PATCH_K_PAD     = k_pad(PATCH_K);          // 588  -> 768  (1.31x)
static constexpr int PATCH_K_ITERS   = k_iters(PATCH_K);        // 12
static constexpr int WIDTH_K_PAD     = k_pad(WIDTH);            // 1152 -> 1280 (1.11x)
static constexpr int WIDTH_K_ITERS   = k_iters(WIDTH);          // 20
static constexpr int MLP_K_PAD       = k_pad(MLP_WIDTH);        // 4304 -> 4352 (1.01x)
static constexpr int MLP_K_ITERS     = k_iters(MLP_WIDTH);      // 68

static_assert(PATCH_K_ITERS % PIPELINE_K_STAGES == 0, "patch K iters must be multiple of 4");
static_assert(WIDTH_K_ITERS % PIPELINE_K_STAGES == 0, "width K iters must be multiple of 4");
static_assert(MLP_K_ITERS   % PIPELINE_K_STAGES == 0, "mlp K iters must be multiple of 4");
static_assert(HEAD_DIM_PAD % 16 == 0, "TK st tile cols must be multiple of 16");
}  // namespace vis

namespace enc {
static constexpr int LAYERS        = 18;
static constexpr int WIDTH         = 2048;
static constexpr int MLP_WIDTH     = 16384;
static constexpr int NUM_Q_HEADS   = 8;
static constexpr int NUM_KV_HEADS  = 1;    // MQA 8:1
static constexpr int HEAD_DIM      = 256;
static constexpr int QKV_OUT       = (NUM_Q_HEADS + 2 * NUM_KV_HEADS) * HEAD_DIM;  // 2560
static constexpr int ATTN_OUT      = NUM_Q_HEADS * HEAD_DIM;                       // 2048

static constexpr int PROMPT_LEN    = 20;
static constexpr int SEQ           = vis::SEQ + PROMPT_LEN;
static constexpr int MAX_SEQ       = 968;                     // 768 + max_prompt_len 200

static constexpr int TILE_M      = 128;
static constexpr int SEQ_PAD     = (SEQ + TILE_M - 1) / TILE_M * TILE_M;      // 896
static constexpr int MAX_SEQ_PAD = (MAX_SEQ + TILE_M - 1) / TILE_M * TILE_M;  // 1024

static constexpr int ACTIVE_LAYERS = LAYERS - 1;   // 17

static constexpr int Q_TILE_ROWS = TILE_M;                         // 128
static constexpr int Q_TILES     = SEQ_PAD / Q_TILE_ROWS;          // 7

static constexpr int ATTN_KV_SPLITS = 2;
static_assert(ATTN_KV_SPLITS >= 1, "");

static constexpr bool HAS_BIAS = false;

static constexpr int RMS_STATS_TILES     = ATTN_OUT / 128;
static constexpr int RMS_STATS_ATTN_ROW0 = 2;
static constexpr int RMS_STATS_FFN_ROW0  = RMS_STATS_ATTN_ROW0 + RMS_STATS_TILES;   // 18
static constexpr int RMS_ROWS            = RMS_STATS_FFN_ROW0 + RMS_STATS_TILES;    // 34

static constexpr int WIDTH_K_PAD   = k_pad(WIDTH);
static constexpr int WIDTH_K_ITERS = k_iters(WIDTH);          // 32
static constexpr int MLP_K_PAD     = k_pad(MLP_WIDTH);
static constexpr int MLP_K_ITERS   = k_iters(MLP_WIDTH);      // 256

static_assert(WIDTH_K_ITERS % PIPELINE_K_STAGES == 0, "");
static_assert(MLP_K_ITERS   % PIPELINE_K_STAGES == 0, "");
static_assert(HEAD_DIM % 64 == 0, "head_dim 256 should hit SW128 swizzle");
static_assert(SEQ_PAD % TILE_M == 0, "");

static_assert(QKV_OUT % 128 == 0 && ATTN_OUT % 128 == 0 && MLP_WIDTH % 128 == 0,
              "Gemma N must be divisible by BN=128");
}  // namespace enc

//   - pre-norm: AdaRMS -> plain RMSNorm
//   - residual: x += delta*gate -> x += delta

namespace dec {
static constexpr int LAYERS       = 18;
static constexpr int WIDTH        = 1024;
static constexpr int MLP_WIDTH    = 4096;
static constexpr int NUM_Q_HEADS  = enc::NUM_Q_HEADS;   // 8
static constexpr int NUM_KV_HEADS = enc::NUM_KV_HEADS;  // 1
static constexpr int HEAD_DIM     = enc::HEAD_DIM;      // 256
static constexpr int QKV_OUT      = enc::QKV_OUT;       // 2560
static constexpr int ATTN_IN      = enc::ATTN_OUT;      // 2048

static constexpr int ACTION_DIM       = 32;
static constexpr int PRODUCTION_CHUNK = 10;             // native DROID action horizon
static constexpr int MAX_CHUNK        = 50;             // physical ABI capacity
static constexpr int PRODUCTION_SEQ   = PRODUCTION_CHUNK + 1;  // state + 10 action
static constexpr int MAX_SEQ          = MAX_CHUNK + 1;          // state + 50 action
static constexpr int FLOW_STEPS       = 10;

static constexpr int M_PAD        = 64;

static constexpr int WIDTH_K_PAD    = k_pad(WIDTH);
static constexpr int WIDTH_K_ITERS  = k_iters(WIDTH);    // 16
static constexpr int ATTN_K_PAD     = k_pad(ATTN_IN);
static constexpr int ATTN_K_ITERS   = k_iters(ATTN_IN);  // 32
static constexpr int MLP_K_PAD      = k_pad(MLP_WIDTH);
static constexpr int MLP_K_ITERS    = k_iters(MLP_WIDTH);// 64

static_assert(WIDTH_K_ITERS % PIPELINE_K_STAGES == 0, "");
static_assert(ATTN_K_ITERS  % PIPELINE_K_STAGES == 0, "");
static_assert(MLP_K_ITERS   % PIPELINE_K_STAGES == 0, "");
static_assert(PRODUCTION_SEQ <= MAX_SEQ, "production decoder seq must fit capacity");
static_assert(MAX_SEQ <= M_PAD, "maximum decoder seq must fit one WGMMA M tile");
}  // namespace dec

}  // namespace pi0
