#pragma once

#include "kittens.cuh"
#include "config.cuh"
#include "barrier.cuh"

// Shared VM state, opcode dispatch, page accounting and worker loops.
// The scheduler supplies fixed per-SM queues padded with opcode-0 NoOp records.

namespace megakernel {

// pid -- physical page id
// lid -- logical page id

template <typename config> struct __align__(128) instruction_state_t {
    config::instruction_t instructions;
    int pid_order[config::NUM_PAGES];
    int padding[((config::NUM_PAGES + 31) & ~31) -
                config::NUM_PAGES]; // Round up to multiple of 32
    kittens::semaphore semaphores[config::DYNAMIC_SEMAPHORES];
    int scratch[config::SCRATCH_BYTES / 4];
};

__device__ inline unsigned int get_smid() {
    unsigned int ret;
    asm volatile("mov.u32 %0, %smid;" : "=r"(ret));
    return ret;
}

__device__ inline unsigned int get_worker_id() {
    return get_smid();
}

template <template <typename> typename op_dispatcher, typename... ops>
struct dispatch_op {
    template <typename return_t, typename config, typename globals,
              typename... args>
    __device__ static inline return_t run(int opcode, const globals &g,
                                          args &...a) {
        asm volatile("trap;\n");
        return return_t{};
    }
};
template <template <typename> typename op_dispatcher, typename op,
          typename... ops>
struct dispatch_op<op_dispatcher, op, ops...> {
    template <typename return_t, typename config, typename globals,
              typename... args>
    __device__ static inline return_t run(int opcode, const globals &g,
                                          args &...a) {
        if (opcode == op::opcode)
            return op_dispatcher<op>::run(g, a...);
        else
            return dispatch_op<op_dispatcher, ops...>::template run<
                return_t, config, globals, args...>(opcode, g, a...);
    }
};

template<int N> __device__ static inline int ring_advance(int ring, int distance=1) { return (ring + distance) % N; }
template<int N> __device__ static inline int ring_retreat(int ring, int distance=1) { return (ring + 16*N - distance) % N; }

template <typename config> struct page {
    int data[config::PAGE_SIZE / sizeof(int)];
    __device__ inline void *ptr(int byte_offset = 0) {
        return (void *)(data + byte_offset / sizeof(int));
    }
    __device__ inline const void *ptr(int byte_offset = 0) const {
        return (const void *)(data + byte_offset / sizeof(int));
    }
};
template <typename config> struct mini_page {
    int data[config::MINI_PAGE_SIZE / sizeof(int)];
};

template <typename config> struct state {
    using instruction_state_array_t =
        instruction_state_t<config>[config::INSTRUCTION_PIPELINE_STAGES];
    instruction_state_array_t &all_instructions;
    using instruction_semaphore_array_t =
        kittens::semaphore[config::INSTRUCTION_PIPELINE_STAGES];
    instruction_semaphore_array_t &instruction_arrived, &instruction_finished;
    int instruction_index, instruction_ring;
    int reg_pid_order[config::NUM_PAGES];

    __device__ inline int (&instruction())[config::INSTRUCTION_WIDTH] {
        return all_instructions[instruction_ring].instructions;
    }
    __device__ inline const int (&instruction()
                                     const)[config::INSTRUCTION_WIDTH] {
        return all_instructions[instruction_ring].instructions;
    }
    __device__ inline int (&pid_order())[config::NUM_PAGES] {
        return all_instructions[instruction_ring].pid_order;
    }
    __device__ inline const int (&pid_order() const)[config::NUM_PAGES] {
        return all_instructions[instruction_ring].pid_order;
    }
    __device__ inline void *scratch() const {
        return (void *)&all_instructions[instruction_ring].scratch[0];
    }

    template <int num_bytes> __device__ inline void zero_scratch() {
        static_assert(num_bytes % 4 == 0, "num_bytes must be a multiple of 4");
        constexpr auto num_floats = num_bytes / 4;
        auto &scratch_vec = *reinterpret_cast<kittens::sv_fl<num_floats> *>(scratch());
        kittens::warp::zero(scratch_vec);
        kittens::warp::sync();
    }

    __device__ inline kittens::semaphore (
        &semaphores())[config::DYNAMIC_SEMAPHORES] {
        return all_instructions[instruction_ring].semaphores;
    }
    __device__ inline const kittens::semaphore (
        &semaphores() const)[config::DYNAMIC_SEMAPHORES] {
        return all_instructions[instruction_ring].semaphores;
    }
    __device__ inline void await_instruction() {
        mk_semaphore_wait(
            instruction_arrived[instruction_ring],
            (instruction_index / config::INSTRUCTION_PIPELINE_STAGES) & 1);
        pid_order_shared_addr =
            static_cast<uint32_t>(__cvta_generic_to_shared(&(pid_order()[0])));
    }
    __device__ inline void next_instruction() {
        __syncwarp();
        if (kittens::laneid() == 0) {
#ifdef MK_DEBUG
            if (blockIdx.x == 0)
                printf("Thread %d: arriving at instruction finished %d\n",
                       threadIdx.x, instruction_ring);
#endif
            kittens::arrive(instruction_finished[instruction_ring]);
        }
        instruction_index++;
        instruction_ring =
            ring_advance<config::INSTRUCTION_PIPELINE_STAGES>(instruction_ring);
    }

    using page_array_t = page<config>[config::NUM_PAGES];
    page_array_t &pages;

    using page_semaphore_array_t =
        kittens::semaphore[config::NUM_PAGES]
                          [config::INSTRUCTION_PIPELINE_STAGES_BITS];
    page_semaphore_array_t &page_finished;

    __device__ inline int pid(int lid) {
        int ret;
        kittens::move<int>::lds(ret,
                                pid_order_shared_addr + lid * sizeof(int));
        return ret;
    }
    __device__ inline void wait_page_ready(int pid) {
#pragma unroll
        for (int i = 0; i < config::INSTRUCTION_PIPELINE_STAGES_BITS; i++) {
            auto bit = (instruction_index >> i) & 1;
            mk_semaphore_wait(page_finished[pid][i], bit);
        }
    }

    __device__ inline void finish_page(int pid, int count) {
#pragma unroll
        for (int i = 0; i < config::INSTRUCTION_PIPELINE_STAGES_BITS; i++) {
            arrive(page_finished[pid][i], count);
        }
    }

    __device__ inline void warp_finish_page(int pid, int count) {
        if (kittens::warp::laneid() == 0) {
            finish_page(pid, count);
        }
    }

#ifdef KITTENS_BLACKWELL
    kittens::semaphore &tensor_finished;
    __device__ inline void wait_tensor_ready() {
        mk_semaphore_wait(tensor_finished, instruction_index % 2);
    }
#endif

    kittens::semaphore &semaphores_ready;
    __device__ inline void wait_semaphores_ready() {
        mk_semaphore_wait(semaphores_ready, instruction_index % 2);
    }

#ifdef KITTENS_BLACKWELL
    static constexpr int NCTA_TENSOR_ALLOC = config::CLUSTER_BLOCKS > 1 ? 2 : 1;
    using tensor_allocator_t =
        ::kittens::tensor_allocator<1, NCTA_TENSOR_ALLOC>;
    tensor_allocator_t &tensor_alloc;
#endif

    uint32_t pid_order_shared_addr;

    __device__ inline void print() {
        printf("Kittens Virtual Machine State being printed by thread %d, "
               "block %d\n  Instruction index: %d, Instruction ring: %d\n",
               threadIdx.x, blockIdx.x, instruction_index, instruction_ring);
    }
};

constexpr int FREE_SLOTS_START = 55;

} // namespace megakernel

#ifdef MK_DEBUG
#define MK_DEBUG_PRINT_START(msg)                                              \
    if (blockIdx.x == 0 && kittens::laneid() == 0)                             \
    printf("Thread %d: starting main loop for %s\n", threadIdx.x, msg);
#define MK_DEBUG_PRINT_END(msg)                                                \
    if (blockIdx.x == 0 && kittens::laneid() == 0)                             \
    printf("Thread %d: exiting main loop for %s\n", threadIdx.x, msg);
#else
#define MK_DEBUG_PRINT_START(msg)
#define MK_DEBUG_PRINT_END(msg)
#endif

// MAKE_WORKER exposes role identity and hardware cooperation width. VM instruction
// completion and page accounting remain per warp; op-private semaphores may use the
// declared group width for warpgroup-local work.
#define MAKE_WORKER(name, is_consumer, group_size)                             \
    namespace megakernel {                                                     \
    namespace name {                                                           \
                                                                               \
    static constexpr int GROUP_SIZE = (group_size);                            \
    static constexpr bool IS_CONSUMER = (is_consumer);                         \
                                                                               \
    template <typename config, typename globals> struct name##_op_dispatcher { \
        template <typename op> struct dispatcher {                             \
            __device__ static inline void                                      \
            run(const globals &g, ::megakernel::state<config> &mks) {          \
                op::name::run(g, mks);                                         \
            }                                                                  \
        };                                                                     \
    };                                                                         \
                                                                               \
    template <typename config, typename globals, typename... ops>              \
    __device__ void main_loop(const globals &g,                                \
                              ::megakernel::state<config> &mks) {              \
        MK_DEBUG_PRINT_START(#name);                                           \
        int num_iters = g.instructions.rows();                                 \
        for (mks.instruction_index = 0, mks.instruction_ring = 0;              \
             mks.instruction_index < num_iters; mks.next_instruction()) {      \
            mks.await_instruction();                                           \
            dispatch_op<name##_op_dispatcher<config, globals>::dispatcher,     \
                        ops...>::template run<void, config, globals,           \
                                              ::megakernel::state<config>>(    \
                mks.instruction()[0], g, mks);                                 \
        }                                                                      \
        __syncwarp();                                                          \
        MK_DEBUG_PRINT_END(#name);                                             \
    }                                                                          \
    }                                                                          \
    }
