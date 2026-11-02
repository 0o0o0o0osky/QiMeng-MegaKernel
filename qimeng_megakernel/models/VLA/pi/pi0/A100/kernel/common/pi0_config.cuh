#pragma once

// Pi0 A100/A800 route constants.  Model-facing names intentionally mirror the
// H100 route; only the execution mechanism is SM80-specific.
namespace pi0_a100 {

constexpr int kSmCount = 108;
constexpr int kThreads = 128;
constexpr int kInstructionWidth = 32;
constexpr int kBarrierSlots = 64;

// Queue counts also define launch grids and must not exceed resident capacity.

namespace siglip {
constexpr int kQueues = 2 * kSmCount;
constexpr int kLayers = 27;
constexpr int kWidth = 1152;
constexpr int kWidthPad = 1280;
constexpr int kHeads = 16;
constexpr int kHeadDim = 72;
constexpr int kHeadDimPad = 80;
constexpr int kQkvWidth = 3840;
constexpr int kMlp = 4304;
constexpr int kMlpWidth = 4608;
constexpr int kFc2Splits = 6;
constexpr int kFc2Segment = 768;
constexpr int kBarrierRows = 10;
constexpr float kLnEps = 1.0e-5f;
}  // namespace siglip

namespace gemma {
constexpr int kQueues = 2 * kSmCount;
constexpr int kLayers = 18;
constexpr int kWidth = 2048;
constexpr int kHeads = 8;
constexpr int kHeadDim = 256;
constexpr int kQkvWidth = 2560;
constexpr int kMlpWidth = 16384;
constexpr int kMaxSeq = 1024;
constexpr int kKvBlock = 64;
constexpr int kBarrierRows = 11;
constexpr float kRmsEps = 1.0e-6f;
}  // namespace gemma

namespace ae {
constexpr int kQueues = 2 * kSmCount;
constexpr int kLayers = 18;
constexpr int kRows = 64;
constexpr int kWidth = 1024;
constexpr int kHeads = 8;
constexpr int kHeadDim = 256;
constexpr int kQkvWidth = 2560;
constexpr int kMlpWidth = 4096;
constexpr int kActionDim = 32;
constexpr int kKSplit = 256;
constexpr int kOprojSplits = 8;
constexpr int kDownSplits = 16;
constexpr int kBarrierRows = 15;
constexpr float kRmsEps = 1.0e-6f;
}  // namespace ae

}  // namespace pi0_a100
