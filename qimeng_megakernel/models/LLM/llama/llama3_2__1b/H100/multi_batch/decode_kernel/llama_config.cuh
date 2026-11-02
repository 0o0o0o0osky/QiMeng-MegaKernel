#pragma once

namespace llama_mb {

static constexpr int NUM_LAYERS = 16;
static constexpr int HIDDEN_DIM = 2048;
static constexpr int INTERMEDIATE_DIM = 8192;
static constexpr int HEAD_DIM = 64;
static constexpr int NUM_Q_HEADS = 32;
static constexpr int NUM_KV_HEADS = 8;
static constexpr int GQA_RATIO = NUM_Q_HEADS / NUM_KV_HEADS;
static constexpr int QKV_OUT_DIM =
    (NUM_Q_HEADS + 2 * NUM_KV_HEADS) * HEAD_DIM;
static constexpr int VOCAB_SIZE = 128256;

static constexpr int BATCH_CAPACITY = 16;
static constexpr int H100_SM_COUNT = 132;
static constexpr int K_TILE = 64;
static constexpr int DENSE_TILE_M = 128;
static constexpr int DENSE_STAGES = 11;
static constexpr int DOWN_SPLIT_K = 4;
static constexpr int DOWN_STAGES = 10;

// Two words back the one startup grid barrier.  Every layer owns disjoint
// producer/consumer counters so CUDA Graph replay only needs an in-kernel
// reset, not a host memset or an extra launch.
static constexpr int PHASE_COUNTERS_PER_LAYER = 60;
static constexpr int PHASE_COUNTER_COUNT =
    NUM_LAYERS * PHASE_COUNTERS_PER_LAYER + 1;
static constexpr int SYNC_STATE_WORDS = 2 + PHASE_COUNTER_COUNT;

static_assert(GQA_RATIO == 4);
static_assert(HIDDEN_DIM % K_TILE == 0);
static_assert(INTERMEDIATE_DIM % (K_TILE * DOWN_SPLIT_K) == 0);
static_assert(QKV_OUT_DIM % DENSE_TILE_M == 0);
static_assert(HIDDEN_DIM % DENSE_TILE_M == 0);
static_assert(INTERMEDIATE_DIM % 64 == 0);
static_assert(VOCAB_SIZE % DENSE_TILE_M == 0);

}  // namespace llama_mb
