#pragma once

// ========== GPU / kernel config (H100 / B200) ==========
// SM counts and selection. Arch and macros (KITTENS_HOPPER, KITTENS_BLACKWELL)
// are set in Makefile; pipeline (NUM_PAGES, INSTRUCTION_WIDTH, etc.) is in
// include/config.cuh (megakernel::default_config).

#define H100_SM_COUNT 132
#define B200_SM_COUNT 148

// Active SM count: H100 when KITTENS_BLACKWELL is not set, B200 otherwise.
#ifdef KITTENS_BLACKWELL
#define KERNEL_SM_COUNT B200_SM_COUNT
#else
#define KERNEL_SM_COUNT H100_SM_COUNT
#endif
