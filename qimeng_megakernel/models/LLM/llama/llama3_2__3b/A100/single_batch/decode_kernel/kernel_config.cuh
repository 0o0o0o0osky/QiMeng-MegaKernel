#pragma once

#include <cstdint>

namespace megakernel {
    struct KernelConfig {
        static constexpr int32_t NUM_WARPS = 16;                    // number of warps per thread block
        static constexpr int32_t WARP_SIZE = 32;                    //
        static constexpr int32_t NUM_THREADS = NUM_WARPS * WARP_SIZE;
        static constexpr int32_t NUM_SMS = 108;                     // number of streaming multiprocessors (A100: 108, H100: 132)
        static constexpr int32_t SHARED_MEMORY_SIZE = 160 * 1024;          // workspace; SmemLayout adds 128B instruction
        static constexpr int32_t GMEM_SPIN_LOOP_SLEEP_NANOS = 20;   // 20 nanoseconds
        static constexpr int32_t INSTRUCTION_WIDTH = 32;
    };

    // Route scheduling parameters consumed by compile-time CUDA layouts.
    // The value is mirrored by L3 instruction_defs.py and serialized into
    // every DownProj instruction; verifier invariants enforce equality.
    struct DecodeScheduleConfig {
        // Hidden 3072 uses two 1536-wide segments so two 16-row weight stages
        // remain inside the 160 KiB CTA workspace.
        static constexpr int32_t HIDDEN_GEMV_K_SEGMENT = 1536;
        // Four warps cover one 16-token SplitKV tile per Q head.  With GQA3,
        // 12 of the 16 CTA warps are active in the attention body.
        static constexpr int32_t ATTENTION_GROUP_WARPS = 4;
        static constexpr int32_t DOWN_PROJ_SPLIT_K_GRANULARITY = 2048;
        // The route emits at most 13 partials on 108 SMs.  Keep a small
        // padded CUDA workspace bound separate from the serialized-list bound:
        // [opcode, layer, head, count, list_len, list..., terminal, output].
        static constexpr int32_t ATTENTION_REDUCTION_WORKSPACE_CAPACITY = 16;
        static constexpr int32_t ATTENTION_REDUCTION_LIST_CAPACITY =
            KernelConfig::INSTRUCTION_WIDTH - 7;
    };

    template <int ProducerExtent, int ProducerTileN, int ConsumerSplitK>
    struct SplitKBucketContract {
        static_assert(ProducerExtent > 0);
        static_assert(ProducerTileN > 0);
        static_assert(ConsumerSplitK > 0);
        static_assert(ProducerExtent % ProducerTileN == 0);
        static_assert(ConsumerSplitK % ProducerTileN == 0);
        static_assert(ProducerExtent % ConsumerSplitK == 0);

        static constexpr int PRODUCER_EXTENT = ProducerExtent;
        static constexpr int PRODUCER_TILE_N = ProducerTileN;
        static constexpr int CONSUMER_SPLIT_K = ConsumerSplitK;
        static constexpr int TOTAL_PRODUCER_BLOCKS =
            PRODUCER_EXTENT / PRODUCER_TILE_N;
        static constexpr int BLOCKS_PER_BUCKET =
            CONSUMER_SPLIT_K / PRODUCER_TILE_N;
        static constexpr int NUM_BUCKETS =
            PRODUCER_EXTENT / CONSUMER_SPLIT_K;

        __host__ __device__ static constexpr int producer_bucket(int block_idx) {
            return block_idx / BLOCKS_PER_BUCKET;
        }

        __host__ __device__ static constexpr int consumer_bucket(int split_k_idx) {
            return split_k_idx;
        }

        __host__ __device__ static constexpr int expected_arrivals(
            int /* bucket_idx */) {
            return BLOCKS_PER_BUCKET;
        }
    };

    struct SmemLayout {
        alignas(128) uint8_t main_workspace[KernelConfig::SHARED_MEMORY_SIZE];
        alignas(128) int32_t instruction[KernelConfig::INSTRUCTION_WIDTH];
    };
}  // namespace megakernel
