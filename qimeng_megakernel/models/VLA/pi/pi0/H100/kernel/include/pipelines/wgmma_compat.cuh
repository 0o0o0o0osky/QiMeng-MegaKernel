#pragma once

#include "kittens.cuh"

namespace pi0 {
namespace wgmma_compat {

template <int accumulate = 1, int fence = 1,
          kittens::ducks::rt::row_layout D,
          kittens::ducks::rt::row_layout A,
          kittens::ducks::st_descriptor::input B>
__device__ static inline void mma_AB(D &d, const A &a, const B &b) {
    kittens::group<4>::mma_AB<D, A, B, fence, accumulate>(d, a, b);
}

}  // namespace wgmma_compat
}  // namespace pi0
