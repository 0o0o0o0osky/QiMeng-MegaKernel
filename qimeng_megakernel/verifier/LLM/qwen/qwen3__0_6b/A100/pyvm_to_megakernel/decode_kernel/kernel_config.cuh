#pragma once

#include <cstdint>

namespace megakernel {
    struct KernelConfig {
        static constexpr int32_t NUM_WARPS = 16;
        static constexpr int32_t WARP_SIZE = 32;
        static constexpr int32_t NUM_THREADS = NUM_WARPS * WARP_SIZE;
        static constexpr int32_t NUM_SMS = 108;
        static constexpr int32_t SHARED_MEMORY_SIZE = 160 * 1024;
        static constexpr int32_t GMEM_SPIN_LOOP_SLEEP_NANOS = 20;
        static constexpr int32_t INSTRUCTION_WIDTH = 32;
    };

    struct DecodeScheduleConfig {
        static constexpr int32_t HIDDEN_GEMV_K_SEGMENT = 1024;
        static constexpr int32_t ATTN_OUT_GEMV_K_SEGMENT = 1024;
        static constexpr int32_t ATTENTION_GROUP_WARPS = 4;
        static constexpr int32_t DOWN_PROJ_SPLIT_K_GRANULARITY = 1536;
        static constexpr int32_t ATTENTION_REDUCTION_WORKSPACE_CAPACITY = 16;
        static constexpr int32_t ATTENTION_REDUCTION_LIST_CAPACITY =
            KernelConfig::INSTRUCTION_WIDTH - 7;
        enum : int32_t {
            QKV_RAW_READY_SLOT_BASE = 0,
            ATTENTION_PREPARE_PRODUCER_PARTIAL = 0,
        };
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

        __host__ __device__ static constexpr int expected_arrivals(int) {
            return BLOCKS_PER_BUCKET;
        }
    };

    // Readiness accounting for a concatenated [Q, K, V] producer whose SplitKV
    // consumer owns one GQA group plus the corresponding K/V head.  Raw
    // publications are sharded by dependency group; prepared publications
    // occupy the immediately following slot range.
    template <int NumQHeads, int NumKvHeads, int HeadDim,
              int ProducerTileN, int RawSlotBase>
    struct GroupedQkvReadinessContract {
        static_assert(NumQHeads > 0 && NumKvHeads > 0);
        static_assert(NumQHeads % NumKvHeads == 0);
        static_assert(HeadDim > 0 && ProducerTileN > 0);
        static_assert(HeadDim % ProducerTileN == 0);

        static constexpr int GQA_GROUP = NumQHeads / NumKvHeads;
        static constexpr int BLOCKS_PER_HEAD = HeadDim / ProducerTileN;
        static constexpr int Q_BLOCKS = NumQHeads * BLOCKS_PER_HEAD;
        static constexpr int KV_BLOCKS = NumKvHeads * BLOCKS_PER_HEAD;
        static constexpr int TOTAL_BLOCKS = Q_BLOCKS + 2 * KV_BLOCKS;
        static constexpr int RAW_SLOT_BASE = RawSlotBase;
        static constexpr int PREPARED_SLOT_BASE = RAW_SLOT_BASE + NumKvHeads;
        static constexpr int RAW_BLOCKS_PER_GROUP =
            (GQA_GROUP + 2) * BLOCKS_PER_HEAD;

        __host__ __device__ static constexpr int raw_group_for_block(
            int block_idx) {
            return block_idx < Q_BLOCKS
                ? (block_idx / BLOCKS_PER_HEAD) / GQA_GROUP
                : block_idx < Q_BLOCKS + KV_BLOCKS
                    ? (block_idx - Q_BLOCKS) / BLOCKS_PER_HEAD
                    : (block_idx - Q_BLOCKS - KV_BLOCKS) / BLOCKS_PER_HEAD;
        }

        __host__ __device__ static constexpr int raw_ready_slot(int group_idx) {
            return RAW_SLOT_BASE + group_idx;
        }

        __host__ __device__ static constexpr int prepared_slot(int group_idx) {
            return PREPARED_SLOT_BASE + group_idx;
        }

        __host__ __device__ static constexpr int expected_raw_arrivals(
            int /* group_idx */) {
            return RAW_BLOCKS_PER_GROUP;
        }
    };

    struct SmemLayout {
        alignas(128) uint8_t main_workspace[KernelConfig::SHARED_MEMORY_SIZE];
        alignas(128) int32_t instruction[KernelConfig::INSTRUCTION_WIDTH];
    };
}  // namespace megakernel
