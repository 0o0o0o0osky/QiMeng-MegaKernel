// Multi-batch decode attention split-KV kernel.
//
// Score: CuTe AttnScoreGemmConfig. PV: CuTe splitkv_pv_gemm_cute. Load: CuTe layouts + pipeline.
// Archives: _legacy.cu (fp32 + attn_*), _attn_pv.cu (CuTe score), _plain_load.cu (hand loops).

#include "globals.cuh"
#include "llama_config.cuh"
#include "kernel_config.cuh"
#include "instruction_list.cuh"
#include "utils.cuh"

namespace megakernel {

using namespace cute;

// ---------------------------------------------------------------------------
// Compile-time constants (from SplitkvLayoutTraits)
// ---------------------------------------------------------------------------
using Tr = SplitkvLayoutTraits;
constexpr int kKVTile    = Tr::kKVTile;
constexpr int kHeadDim   = Tr::kHeadDim;
constexpr int kGQARatio  = Tr::kGQARatio;
constexpr int kNThreads  = Tr::kNThreads;
constexpr int kNWarps    = Tr::kNWarps;
constexpr int kMathThreads = kNWarps * 32;
constexpr int kQRowsPad  = Tr::kQRowsPad;
constexpr int kMmaItersM = 1;
constexpr int kVMmaItersN = kHeadDim / 16;  // 4
// log2(e) for score → log2 domain conversion
constexpr float kLog2E = 1.44269504f;

__device__ __forceinline__ float splitkv_warp_reduce_max(float v) {
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1));
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2));
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 4));
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 8));
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 16));
    return v;
}

__device__ __forceinline__ float splitkv_warp_reduce_sum(float v) {
    v += __shfl_xor_sync(0xffffffffu, v, 1);
    v += __shfl_xor_sync(0xffffffffu, v, 2);
    v += __shfl_xor_sync(0xffffffffu, v, 4);
    v += __shfl_xor_sync(0xffffffffu, v, 8);
    v += __shfl_xor_sync(0xffffffffu, v, 16);
    return v;
}

// ---------------------------------------------------------------------------
// Shared memory layout (overlaid on SmemLayout::main_workspace)
// Layout sizes from SplitkvSmemLayoutQ, SplitkvSmemLayoutKV.
// ---------------------------------------------------------------------------
struct AttnSplitKVSmem {
    alignas(128) bfloat16_t smem_Q[SplitkvSmemLayoutQ::kCount];
    alignas(128) bfloat16_t smem_K[SplitkvSmemLayoutKV::kCount];
    alignas(128) bfloat16_t smem_V[SplitkvSmemLayoutKV::kCount];
    // Canonical softmax P tile bridge between score-C and PV-A layouts.
    alignas(128) bfloat16_t smem_P[kQRowsPad * kKVTile];
    // PV partition_C duplicates each active (row,col) across K-warps only.
    // Keep one warp-local partial tile per warp, then merge explicitly.
    alignas(128) float s_o_partial[kNWarps][kGQARatio * kHeadDim];
    // Cross-warp softmax: each warp only owns 16 K columns; must max/sum over warps per Q row.
    alignas(128) float sh_wr_max[kGQARatio][kNWarps];
    alignas(128) float sh_m_tile[kGQARatio];
    alignas(128) float sh_d_warp[kGQARatio][kNWarps];
    alignas(128) float sh_d_tile[kGQARatio];
};
static_assert(sizeof(AttnSplitKVSmem) <= KernelConfig::SHARED_MEMORY_SIZE,
              "AttnSplitKVSmem overflows shared memory budget");

// ---------------------------------------------------------------------------
// Kernel function
// ---------------------------------------------------------------------------
__device__ __forceinline__ void decode_attention_splitkv(
    LlamaGlobals& g,
    SmemLayout&   smem,
    int           inst_idx
) {
    AttnSplitKVSmem& s = *reinterpret_cast<AttnSplitKVSmem*>(smem.main_workspace);

    const int tid     = threadIdx.x;
    const int lane_id = tid & 31;
    const int warp_id = tid >> 5;
    const bool math_active = tid < kMathThreads;
    const int math_tid = math_active ? tid : 0;
#ifdef ENABLE_PROFILING
    uint64_t prof_stage_begin = 0;
#endif

    // ========================================================================
    // 1. Parse instruction
    // ========================================================================
    const int layer_idx    = smem.instruction[1];
    const int batch_idx    = smem.instruction[2];
    const int kv_head_idx  = smem.instruction[3];
    const int num_partials = smem.instruction[4];
    const int partial_idx  = smem.instruction[5];
    const int seq_len_raw  = smem.instruction[6];   // -1 → use runtime pos
    const int is_active    = smem.instruction[7];   // 0=barrier-only, 1=active, -1=check mask

    const int head_start   = kv_head_idx * kGQARatio;  // first Q head for this KV head

    // ========================================================================
    // 2. Barrier: wait for opcode-1 (RMS_QKV_GEMM_ROPE) writes to Q and KV.
    //
    // QKV barriers are GLOBAL (not per-batch): the QKV CTA processes all batch
    // rows in one sweep and writes barriers[layer, 0, head] once for all batches.
    // Use barrier_global() (no batch_idx offset) to match the QKV write address.
    // ========================================================================
    if (tid == 0) {
        const int prev_row = OPCODE_QKV_GEMM_ROPE - 1;  // = 0
        // Must match PyVM blocks_per_head = head_dim // TILE_N_GRANULARITY and
        // qkv_gemm_rope per-tile atomicAdds (one per N-tile per logical head).
        constexpr int kExpectedQkvTilesPerHead =
            LlamaConfig::HEAD_DIM / KernelConfig::kTileN;
        static_assert(LlamaConfig::HEAD_DIM % KernelConfig::kTileN == 0, "");
        const int expected = kExpectedQkvTilesPerHead;
        // Q heads
        #pragma unroll
        for (int i = 0; i < kGQARatio; ++i) {
            volatile int32_t* bar = g.barrier_global(
                layer_idx, prev_row, head_start + i);
            barrier_wait(bar, expected, blockIdx.x,
                                      OPCODE_ATTENTION_PARTIAL, head_start + i, layer_idx, &g, inst_idx);
        }
        // K head (index = NUM_Q_HEADS + kv_head_idx)
        {
            int idx = LlamaConfig::NUM_Q_HEADS + kv_head_idx;
            volatile int32_t* bar = g.barrier_global(layer_idx, prev_row, idx);
            barrier_wait(bar, expected, blockIdx.x,
                                      OPCODE_ATTENTION_PARTIAL, idx, layer_idx, &g, inst_idx);
        }
        // V head (index = NUM_Q_HEADS + NUM_KV_HEADS + kv_head_idx)
        {
            int idx = LlamaConfig::NUM_Q_HEADS + LlamaConfig::NUM_KV_HEADS + kv_head_idx;
            volatile int32_t* bar = g.barrier_global(layer_idx, prev_row, idx);
            barrier_wait(bar, expected, blockIdx.x,
                                      OPCODE_ATTENTION_PARTIAL, idx, layer_idx, &g, inst_idx);
        }
    }
    __syncthreads();

    // ========================================================================
    // 3. Compute kv_start / kv_len for this partial
    // ========================================================================
    const int seq_len = (seq_len_raw >= 0)
        ? seq_len_raw
        : (row_position(g, batch_idx) + 1);

    const bool active = (is_active == 1) ||
                        (is_active == -1 && row_active(g, batch_idx));

    // Replicate scheduler logic: total_blocks → blocks_per_partial → kv range.
    const int total_blocks      = max(1, (seq_len + kKVTile - 1) / kKVTile);
    const int blocks_per_partial = max(1, (total_blocks + num_partials - 1) / num_partials);
    const int start_block        = partial_idx * blocks_per_partial;
    const int end_block          = min(start_block + blocks_per_partial, total_blocks);
    const int kv_start           = start_block * kKVTile;
    const int kv_end             = min(end_block * kKVTile, seq_len);
    const int kv_len             = active ? max(0, kv_end - kv_start) : 0;

    // ========================================================================
    // 4. Load Q (CuTe layout + padding)
    // ========================================================================
    {
        const bfloat16_t* q_row_base =
            g.q_post_rope_ptr + (long)batch_idx * LlamaConfig::HIDDEN_DIM
            + (long)head_start * kHeadDim;
        splitkv_load_q_cute(tid, q_row_base, &s.smem_Q[0]);
    }
    __syncthreads();

    // ========================================================================
    // 5. KV cache base pointers for this (layer, batch, kv_head)
    //    Layout: [NUM_LAYERS, batch_size, max_seq_len, NUM_KV_HEADS, HEAD_DIM]
    // ========================================================================
    const long kv_stride_seq = (long)LlamaConfig::NUM_KV_HEADS * kHeadDim;
    const long kv_layer_off  = (long)layer_idx * g.batch_size * g.max_seq_len
                               * LlamaConfig::NUM_KV_HEADS * kHeadDim;
    const long kv_batch_off  = (long)batch_idx * g.max_seq_len
                               * LlamaConfig::NUM_KV_HEADS * kHeadDim;
    const long kv_head_off   = (long)kv_head_idx * kHeadDim;

    const bfloat16_t* k_base = g.k_cache_ptr + kv_layer_off + kv_batch_off + kv_head_off;
    const bfloat16_t* v_base = g.v_cache_ptr + kv_layer_off + kv_batch_off + kv_head_off;
    // Access: k_base[token * kv_stride_seq + dim]

    // ========================================================================
    // 6. Initialize MMA accumulators (CuTe partition_fragment_C for 16×64 O)
    //    m_run/d_run: per Q row (0..kGQARatio-1), correct softmax over all K columns & tiles.
    // ========================================================================
    float m_run[kGQARatio];
    float d_run[kGQARatio];
    float o_acc[kMmaItersM][kVMmaItersN][8];

    typename AttnPvGemmConfig::MMA tiled_mma_pv;
    constexpr int kPvMmaThreads = decltype(size(typename AttnPvGemmConfig::MMA{}))::value;
    auto thr_mma_pv = tiled_mma_pv.get_slice(math_tid);
    constexpr auto acc_shape_o = make_shape(Int<kQRowsPad>{}, Int<kHeadDim>{});
    auto sPStore = make_tensor(make_smem_ptr(&s.smem_P[0]), SplitkvSmemLayoutPStore{});
    auto sPLogical = make_tensor(make_smem_ptr(&s.smem_P[0]), SplitkvSmemLayoutPLogical{});
    auto gC_dummy_o = make_tensor(
        make_rmem_ptr(&o_acc[0][0][0]),
        acc_shape_o,
        LayoutRight{});
    auto acc_o = thr_mma_pv.partition_fragment_C(gC_dummy_o);
    clear(acc_o);
    #pragma unroll
    for (int r = 0; r < kGQARatio; ++r) {
        m_run[r] = -INFINITY;
        d_run[r] = 0.f;
    }

    // ========================================================================
    // 7. KV tile loop (CuTe layout + pipeline)
    // ========================================================================
    const int num_tiles = (kv_len + kKVTile - 1) / kKVTile;

    auto load_kv_async = [&](int tile_idx, int s_idx) {
        int t_start = kv_start + tile_idx * kKVTile;
        int t_count = min(kKVTile, kv_len - tile_idx * kKVTile);
        splitkv_load_kv_tile_async(
            tid,
            k_base + t_start * kv_stride_seq,
            v_base + t_start * kv_stride_seq,
            kv_stride_seq,
            t_count,
            &s.smem_K[s_idx * SplitkvSmemLayoutKV::kCountPerStage],
            &s.smem_V[s_idx * SplitkvSmemLayoutKV::kCountPerStage]);
    };

    constexpr int kKVStages = Tr::kStage;
    #pragma unroll
    for (int preload = 0; preload < kKVStages - 1; ++preload) {
        if (preload < num_tiles) {
            load_kv_async(preload, preload % kKVStages);
        }
    }

    for (int tile = 0; tile < num_tiles; ++tile) {
        constexpr int kLookahead = kKVStages - 1;
        int prefetch_tile = tile + kLookahead;
        if (prefetch_tile < num_tiles) {
            load_kv_async(prefetch_tile, prefetch_tile % kKVStages);
        }
        if constexpr (kKVStages == 2) {
            if (num_tiles - tile > 1) cp_async_wait<1>();
            else                      cp_async_wait<0>();
        } else if constexpr (kKVStages == 3) {
            if (num_tiles - tile > 2)      cp_async_wait<2>();
            else if (num_tiles - tile > 1) cp_async_wait<1>();
            else                           cp_async_wait<0>();
        } else {
            cp_async_wait<0>();
        }
        __syncthreads();

        const int curr_iter_len = min(kKVTile, kv_len - tile * kKVTile);
        const int curr_stage = tile % kKVStages;

        // ==================================================================
        // Score: Q×K^T via CuTe (AttnScoreGemmConfig), keep score fragment in regs.
        // Softmax/PV consume the same fragment via row/col layout view.
        // ==================================================================
        typename AttnScoreGemmConfig::MMA tiled_mma_s;
        auto thr_s = tiled_mma_s.get_slice(math_tid);
        float score_frag_probe[8];
        auto gC_dummy_s = make_tensor(
            make_rmem_ptr(&score_frag_probe[0]),
            make_shape(Int<kQRowsPad>{}, Int<kKVTile>{}),
            LayoutRight{});
        auto tCrScore = thr_s.partition_fragment_C(gC_dummy_s);
        Tensor cScoreCoord = make_identity_tensor(make_shape(Int<kQRowsPad>{}, Int<kKVTile>{}));
        Tensor tScCoord = thr_s.partition_C(cScoreCoord);
        clear(tCrScore);
        if (math_active) {
            splitkv_score_mma_accumulate(
                math_tid, &s.smem_Q[0], &s.smem_K[0], curr_stage, tCrScore);
        }
        constexpr int kPFragElems = decltype(size(tCrScore))::value;
        static_assert(kPFragElems == 8, "Unexpected score fragment size");
        int frag_row[kPFragElems];
        int frag_col[kPFragElems];
        float x_frag_buf[kPFragElems];
        #pragma unroll
        for (int i = 0; i < kPFragElems; ++i) {
            frag_row[i] = static_cast<int>(get<0>(tScCoord(i)));
            frag_col[i] = static_cast<int>(get<1>(tScCoord(i)));
            x_frag_buf[i] = tCrScore(i);
        }
#if DEBUG_PRINT
        if (blockIdx.x == 0 && tid == 0 && tile == 0) {
            printf("[splitkv-current] coord0=(%d,%d) score00=%f curr_iter_len=%d\n",
                frag_row[0], frag_col[0], x_frag_buf[0], curr_iter_len);
        }
#endif

        // ==================================================================
        // Online softmax (log2 domain u = score * attn_scale * log2(e)).
        // Each warp only holds 16 K columns → must max/sum across warps per Q row.
        // ==================================================================
        float u_frag_buf[kPFragElems];
        #pragma unroll
        for (int i = 0; i < kPFragElems; ++i) {
            int row = frag_row[i];
            int col = frag_col[i];
            bool valid = (row < kGQARatio) && (col < curr_iter_len);
            u_frag_buf[i] = valid
                ? (x_frag_buf[i] * g.attn_scale * kLog2E)
                : -INFINITY;
        }

        // N-parallel reduction inside this CTA tile only. Cross-partial merge is
        // handled later by decode_attention_reduction.cu.
        #pragma unroll
        for (int r = 0; r < kGQARatio; ++r) {
            float vm = -INFINITY;
            #pragma unroll
            for (int i = 0; i < kPFragElems; ++i)
                if (frag_row[i] == r)
                    vm = fmaxf(vm, u_frag_buf[i]);
            vm = splitkv_warp_reduce_max(vm);
            if (math_active && lane_id == 0)
                s.sh_wr_max[r][warp_id] = vm;
        }
        __syncthreads();
        if (tid < kGQARatio) {
            int r = tid;
            float mx = s.sh_wr_max[r][0];
            #pragma unroll
            for (int w = 1; w < kNWarps; ++w)
                mx = fmaxf(mx, s.sh_wr_max[r][w]);
            s.sh_m_tile[r] = mx;
        }
        __syncthreads();

        float rescale_r[kGQARatio];
        #pragma unroll
        for (int r = 0; r < kGQARatio; ++r) {
            float m_prev = m_run[r];
            m_run[r]       = fmaxf(m_run[r], s.sh_m_tile[r]);
            rescale_r[r] = (m_run[r] > -INFINITY || m_prev > -INFINITY)
                ? attn_ptx_exp2(m_prev - m_run[r])
                : 1.f;
        }

        #pragma unroll
        for (int i = 0; i < kPFragElems; ++i) {
            int row = frag_row[i];
            int col = frag_col[i];
            bool vcell = (row < kGQARatio) && (col < curr_iter_len);
            x_frag_buf[i] = vcell ? attn_ptx_exp2(u_frag_buf[i] - m_run[row]) : 0.f;
        }
        // Materialize the full P tile in smem so PV A-frag loading follows
        // AttnPvGemmConfig thread mapping exactly (no cross-MMA local guessing).
        splitkv_zero_p_tile(tid, &s.smem_P[0]);
        __syncthreads();
        #pragma unroll
        for (int i = 0; i < kPFragElems; ++i) {
            int pr = frag_row[i];
            int pc = frag_col[i];
            if (math_active && pr < kQRowsPad && pc < kKVTile) {
                sPStore(pr, pc) = bfloat16_t(x_frag_buf[i]);
            }
        }
        __syncthreads();

        #pragma unroll
        for (int r = 0; r < kGQARatio; ++r) {
            float pr = 0.f;
            #pragma unroll
            for (int i = 0; i < kPFragElems; ++i)
                if (frag_row[i] == r)
                    pr += x_frag_buf[i];
            pr = splitkv_warp_reduce_sum(pr);
            if (math_active && lane_id == 0)
                s.sh_d_warp[r][warp_id] = pr;
        }
        __syncthreads();
        if (tid < kGQARatio) {
            int r = tid;
            float tot = s.sh_d_warp[r][0];
            #pragma unroll
            for (int w = 1; w < kNWarps; ++w)
                tot += s.sh_d_warp[r][w];
            s.sh_d_tile[r] = tot;
        }
        __syncthreads();

        // Rescale acc_o by row (CuTe layout via identity tensor for coordinates)
        if (math_active && tid < kPvMmaThreads) {
            Tensor caccO = make_identity_tensor(acc_shape_o);
            Tensor taccOcO = thr_mma_pv.partition_C(caccO);
            #pragma unroll
            for (int i = 0; i < (int)size(acc_o); ++i) {
                int row = static_cast<int>(get<0>(taccOcO(i)));
                if (row < kGQARatio)
                    acc_o(i) *= rescale_r[row];
            }
        }

        #pragma unroll
        for (int r = 0; r < kGQARatio; ++r)
            d_run[r] = d_run[r] * rescale_r[r] + s.sh_d_tile[r];

        // PV: load A from canonical smem_P via CuTe tiled-copy and run P×V.
        if (math_active && tid < kPvMmaThreads) {
            auto tOrP = thr_mma_pv.partition_fragment_A(sPLogical);
            auto smem_tiled_copy_P = make_tiled_copy_A(
                typename AttnPvGemmConfig::S2RCopyAtomA{}, tiled_mma_pv);
            auto smem_thr_copy_P = smem_tiled_copy_P.get_slice(tid);
            Tensor tPsP = smem_thr_copy_P.partition_S(sPStore);
            Tensor tCrP_copy_view = smem_thr_copy_P.retile_D(tOrP);
            #pragma unroll
            for (int i = 0; i < (int)size<2>(tOrP); ++i) {
                cute::copy(smem_tiled_copy_P, tPsP(_, _, i), tCrP_copy_view(_, _, i));
            }
            splitkv_pv_gemm_cute(tid, acc_o, tOrP,
                &s.smem_V[curr_stage * SplitkvSmemLayoutKV::kCountPerStage], tiled_mma_pv);
        }
        __syncthreads();
    }  // end tile loop

    // ========================================================================
    // 8. Epilogue: partition_C(acc_o) repeats only across K-warps, never within
    //    a warp. Write one warp-local partial tile, then merge across warps.
    // ========================================================================
    {
        constexpr int kSOElems = kGQARatio * kHeadDim;
        constexpr int kSOPartialElems = kNWarps * kSOElems;
        for (int i = tid; i < kSOPartialElems; i += kNThreads) {
            s.s_o_partial[i / kSOElems][i % kSOElems] = 0.f;
        }
        __syncthreads();

        if (math_active && tid < kPvMmaThreads) {
            Tensor caccO = make_identity_tensor(acc_shape_o);
            Tensor taccOcO = thr_mma_pv.partition_C(caccO);
            float* warp_partial = &s.s_o_partial[warp_id][0];
            #pragma unroll
            for (int i = 0; i < (int)size(acc_o); ++i) {
                int row = static_cast<int>(get<0>(taccOcO(i)));
                int col = static_cast<int>(get<1>(taccOcO(i)));
                if (row < kGQARatio && col < kHeadDim) {
                    warp_partial[row * kHeadDim + col] = acc_o(i);
                }
            }
        }
        __syncthreads();
    }

    auto attn_lse   = g.get_attn_lse_inter_mb();
    auto attn_out_i = g.get_attn_out_inter_mb();
    const bool has_partial_valid = (g.attn_partial_valid_ptr != nullptr);
    constexpr int kMergeElems = kGQARatio * kHeadDim;
    for (int elem = tid; math_active && elem < kMergeElems; elem += kMathThreads) {
        int row = elem / kHeadDim;
        int col = elem % kHeadDim;
        float denom   = d_run[row];
        float o_sum = 0.f;
        #pragma unroll
        for (int w = 0; w < kNWarps; ++w) {
            o_sum += s.s_o_partial[w][row * kHeadDim + col];
        }
        float final_o = (denom > 0.f) ? (o_sum / denom) : 0.f;
        int abs_head  = head_start + row;
        attn_out_i(batch_idx, abs_head, partial_idx, col) = final_o;
        if (col == 0) {
            float lse = (m_run[row] == -INFINITY || denom <= 0.f)
                ? static_cast<float>(-INFINITY)
                : (m_run[row] + log2f(denom));
            attn_lse(batch_idx, abs_head, partial_idx) = lse;
            if (has_partial_valid) {
                long valid_idx = (long)batch_idx * LlamaConfig::NUM_Q_HEADS * g.max_splits
                               + (long)abs_head * g.max_splits
                               + partial_idx;
                g.attn_partial_valid_ptr[valid_idx] =
                    static_cast<uint8_t>(splitkv_lse_is_contributing(lse));
            }
        }
    }
    __syncthreads();

    // ========================================================================
    // 10. Update barriers: barrier_for_batch(layer, row=1, batch_idx, head+i)
    //     Signals to the reduction kernel that partial results are ready.
    // ========================================================================
#ifdef ENABLE_PROFILING
    if (tid == 0) {
        prof_stage_begin = prof_clock_now();
    }
#endif
    if (tid < kGQARatio) {
        int abs_head = head_start + tid;
        volatile int32_t* bar = g.barrier_for_batch(
            layer_idx, OPCODE_ATTENTION_PARTIAL - 1, batch_idx, abs_head);
        barrier_publish(bar, 1);
    }
#ifdef ENABLE_PROFILING
    if (tid == 0) {
        PROF_ADD_STAGE_CYCLES(
            g, static_cast<int>(blockIdx.x), inst_idx, PROF_FIELD_PUBLISH, prof_clock_now() - prof_stage_begin);
    }
#endif
}

}  // namespace megakernel
