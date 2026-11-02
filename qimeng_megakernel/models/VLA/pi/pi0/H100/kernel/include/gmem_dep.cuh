#pragma once

#include "kittens.cuh"
#include "barrier.cuh"

namespace pi0 {

// The scheduler encodes every cross-SM dependency as
// (dep_layer, dep_row, barrier expected count). A non-positive expected count
// marks an input that is already available when the production route starts.

template <typename Config, typename Globals>
__device__ inline void gmem_dep_wait(const Globals &g, int opcode,
                                     int dep_layer, int dep_row, int dep_expected) {
    (void)opcode;
    if (dep_expected <= 0) return;
    ::megakernel::mk_barrier_wait(
        (volatile int32_t *)&g.Bar[kittens::coord<>{dep_layer, dep_row, 0}],
        dep_expected);
}

}  // namespace pi0
