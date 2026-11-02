#pragma once

#include <cstdint>

#if defined(ENABLE_PROFILING) || defined(ENABLE_BOUNDARY_PROFILING)

#define PROF_FIELDS_PER_INST 12

enum ProfFieldIndex : int {
    PROF_FIELD_OPCODE = 0,
    PROF_FIELD_START = 1,
    PROF_FIELD_WAIT = 2,
    PROF_FIELD_STAGING = 3,
    PROF_FIELD_MMA = 4,
    PROF_FIELD_EPILOGUE = 5,
    PROF_FIELD_SYNC = 6,
    PROF_FIELD_PUBLISH = 7,
    PROF_FIELD_DONE = 8,
    PROF_FIELD_EPILOGUE_SC = 9,
    PROF_FIELD_EPILOGUE_ROPE = 10,
    PROF_FIELD_BARRIER_SPIN = 11,
};

#ifdef __CUDACC__
#define PROF_INLINE __device__ __forceinline__
PROF_INLINE uint64_t prof_clock_now() {
    return static_cast<uint64_t>(clock64());
}
#else
#define PROF_INLINE inline
PROF_INLINE uint64_t prof_clock_now() { return 0; }
#endif

#define PROF_OFFSET(g, sm, inst, field) \
    (((static_cast<long>(sm) * static_cast<long>((g).prof_max_insts) + static_cast<long>(inst)) * PROF_FIELDS_PER_INST) + static_cast<long>(field))

template <typename GlobalsT>
PROF_INLINE void prof_add_field(GlobalsT& g, int sm, int inst, int field, int64_t value) {
#ifdef __CUDACC__
    if (threadIdx.x == 0 && g.prof_ptr != nullptr && inst < g.prof_max_insts) {
        g.prof_ptr[PROF_OFFSET(g, sm, inst, field)] += value;
    }
#else
    (void)g;
    (void)sm;
    (void)inst;
    (void)field;
    (void)value;
#endif
}

#if defined(ENABLE_BOUNDARY_PROFILING) && !defined(ENABLE_PROFILING)

#define PROF_RECORD_START(g, sm, inst, opcode_val)                                        \
    do {                                                                                   \
        if (threadIdx.x == 0 && (g).prof_ptr != nullptr && (inst) < (g).prof_max_insts) { \
            (g).prof_ptr[PROF_OFFSET(g, sm, inst, PROF_FIELD_OPCODE)] = static_cast<int64_t>(opcode_val); \
            (g).prof_ptr[PROF_OFFSET(g, sm, inst, PROF_FIELD_START)] = static_cast<int64_t>(prof_clock_now()); \
        }                                                                                  \
    } while (0)

#define PROF_RECORD_DONE(g, sm, inst)                                                      \
    do {                                                                                   \
        if (threadIdx.x == 0 && (g).prof_ptr != nullptr && (inst) < (g).prof_max_insts) { \
            (g).prof_ptr[PROF_OFFSET(g, sm, inst, PROF_FIELD_DONE)] = static_cast<int64_t>(prof_clock_now()); \
        }                                                                                  \
    } while (0)

#define PROF_ADD_WAIT_CYCLES(...) do {} while (0)
#define PROF_ADD_STAGE_CYCLES(...) do {} while (0)
#define PROF_ADD_EPILOGUE_SC_CYCLES(...) do {} while (0)
#define PROF_ADD_EPILOGUE_ROPE_CYCLES(...) do {} while (0)

#else

#define PROF_RECORD_START(g, sm, inst, opcode_val)                                        \
    do {                                                                                   \
        if (threadIdx.x == 0 && (g).prof_ptr != nullptr && (inst) < (g).prof_max_insts) { \
            (g).prof_ptr[PROF_OFFSET(g, sm, inst, PROF_FIELD_OPCODE)] = static_cast<int64_t>(opcode_val); \
            (g).prof_ptr[PROF_OFFSET(g, sm, inst, PROF_FIELD_START)] = static_cast<int64_t>(prof_clock_now()); \
            (g).prof_ptr[PROF_OFFSET(g, sm, inst, PROF_FIELD_WAIT)] = 0;                    \
            (g).prof_ptr[PROF_OFFSET(g, sm, inst, PROF_FIELD_STAGING)] = 0;                 \
            (g).prof_ptr[PROF_OFFSET(g, sm, inst, PROF_FIELD_MMA)] = 0;                     \
            (g).prof_ptr[PROF_OFFSET(g, sm, inst, PROF_FIELD_EPILOGUE)] = 0;                \
            (g).prof_ptr[PROF_OFFSET(g, sm, inst, PROF_FIELD_SYNC)] = 0;                    \
            (g).prof_ptr[PROF_OFFSET(g, sm, inst, PROF_FIELD_PUBLISH)] = 0;                 \
            (g).prof_ptr[PROF_OFFSET(g, sm, inst, PROF_FIELD_DONE)] = 0;                    \
            (g).prof_ptr[PROF_OFFSET(g, sm, inst, PROF_FIELD_EPILOGUE_SC)] = 0;             \
            (g).prof_ptr[PROF_OFFSET(g, sm, inst, PROF_FIELD_EPILOGUE_ROPE)] = 0;           \
            (g).prof_ptr[PROF_OFFSET(g, sm, inst, PROF_FIELD_BARRIER_SPIN)] = 0;           \
        }                                                                                  \
    } while (0)

#define PROF_ADD_EPILOGUE_SC_CYCLES(g, sm, inst, cycles)                                   \
    do {                                                                                   \
        PROF_ADD_STAGE_CYCLES((g), (sm), (inst), PROF_FIELD_EPILOGUE, (cycles));           \
        PROF_ADD_STAGE_CYCLES((g), (sm), (inst), PROF_FIELD_EPILOGUE_SC, (cycles));      \
    } while (0)

#define PROF_ADD_EPILOGUE_ROPE_CYCLES(g, sm, inst, cycles)                                 \
    do {                                                                                   \
        PROF_ADD_STAGE_CYCLES((g), (sm), (inst), PROF_FIELD_EPILOGUE, (cycles));           \
        PROF_ADD_STAGE_CYCLES((g), (sm), (inst), PROF_FIELD_EPILOGUE_ROPE, (cycles));     \
    } while (0)

#define PROF_RECORD_DONE(g, sm, inst)                                                      \
    do {                                                                                   \
        if (threadIdx.x == 0 && (g).prof_ptr != nullptr && (inst) < (g).prof_max_insts) { \
            (g).prof_ptr[PROF_OFFSET(g, sm, inst, PROF_FIELD_DONE)] = static_cast<int64_t>(prof_clock_now()); \
        }                                                                                  \
    } while (0)

#define PROF_ADD_WAIT_CYCLES(g, sm, inst, wait_cycles)                                     \
    do {                                                                                   \
        if (threadIdx.x == 0 && (g).prof_ptr != nullptr && (inst) < (g).prof_max_insts) { \
            (g).prof_ptr[PROF_OFFSET(g, sm, inst, PROF_FIELD_WAIT)] += static_cast<int64_t>(wait_cycles); \
        }                                                                                  \
    } while (0)

#define PROF_ADD_STAGE_CYCLES(g, sm, inst, field, cycles)                                  \
    do {                                                                                   \
        prof_add_field((g), (sm), (inst), (field), static_cast<int64_t>(cycles));         \
    } while (0)

#endif

#else

#define PROF_FIELDS_PER_INST 12
enum ProfFieldIndex : int {
    PROF_FIELD_OPCODE = 0,
    PROF_FIELD_START = 1,
    PROF_FIELD_WAIT = 2,
    PROF_FIELD_STAGING = 3,
    PROF_FIELD_MMA = 4,
    PROF_FIELD_EPILOGUE = 5,
    PROF_FIELD_SYNC = 6,
    PROF_FIELD_PUBLISH = 7,
    PROF_FIELD_DONE = 8,
    PROF_FIELD_EPILOGUE_SC = 9,
    PROF_FIELD_EPILOGUE_ROPE = 10,
    PROF_FIELD_BARRIER_SPIN = 11,
};
#ifdef __CUDACC__
#define PROF_INLINE __device__ __forceinline__
#else
#define PROF_INLINE inline
#endif
PROF_INLINE uint64_t prof_clock_now() { return 0; }
#define PROF_RECORD_START(...) do {} while (0)
#define PROF_RECORD_DONE(...) do {} while (0)
#define PROF_ADD_WAIT_CYCLES(...) do {} while (0)
#define PROF_ADD_STAGE_CYCLES(...) do {} while (0)
#define PROF_ADD_EPILOGUE_SC_CYCLES(...) do {} while (0)
#define PROF_ADD_EPILOGUE_ROPE_CYCLES(...) do {} while (0)

#endif
