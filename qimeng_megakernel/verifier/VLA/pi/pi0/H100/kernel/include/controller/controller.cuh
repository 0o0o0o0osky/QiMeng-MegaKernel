#pragma once

#include "kittens.cuh"

#include "../util.cuh"
#include "instruction_fetch.cuh"
#include "semaphore_constructor.cuh"
#include "page_allocator.cuh"

namespace megakernel {
namespace controller {

template <typename config, typename globals, typename... ops>
__device__ void main_loop(const globals &g, ::megakernel::state<config> &kvms) {
    auto laneid = ::kittens::laneid();
    int num_iters = g.instructions.rows();
    int num_semaphores[config::INSTRUCTION_PIPELINE_STAGES];

    static_assert(config::NUM_PAGES <= 32);

    for (kvms.instruction_index = 0, kvms.instruction_ring = 0;
         kvms.instruction_index < num_iters;
         kvms.instruction_index++,
        kvms.instruction_ring =
             ring_advance<config::INSTRUCTION_PIPELINE_STAGES>(
                 kvms.instruction_ring)) {

        // Step 0. if the slot was used in the previous iteration, wait for the
        // previous instruction to complete & invalidate its semaphores
        if (kvms.instruction_index >= config::INSTRUCTION_PIPELINE_STAGES) {
            int last_slot_instruction_index =
                kvms.instruction_index - config::INSTRUCTION_PIPELINE_STAGES;

            int phasebit = (last_slot_instruction_index /
                            config::INSTRUCTION_PIPELINE_STAGES) & 1;
            kittens::wait(kvms.instruction_finished[kvms.instruction_ring], phasebit);

            int num_to_invalidate = num_semaphores[kvms.instruction_ring];
            for (int sem_idx = laneid; sem_idx < num_to_invalidate; sem_idx += 32) {
                invalidate_semaphore(
                    kvms.all_instructions[kvms.instruction_ring]
                        .semaphores[sem_idx]);
            }

            // Complete semaphore invalidation before reusing the ring slot.
            kittens::warp::sync();
        }

        load_instructions<config, globals>(&kvms.instruction()[0],
                                           kvms.instruction_index, g);

        // Step 2. Establish physical page order
        int last_instruction_ring =
            (kvms.instruction_ring + config::INSTRUCTION_PIPELINE_STAGES - 1) %
            config::INSTRUCTION_PIPELINE_STAGES;

        if (kvms.instruction_index == 0) {
            if (laneid < config::NUM_PAGES) {
                kvms.pid_order()[laneid] = laneid;
            }
        } else {
            auto last_opcode =
                kvms.all_instructions[last_instruction_ring].instructions[0];

            if (laneid < config::NUM_PAGES) {
                int lid = dispatch_op<
                    page_allocator_op_dispatcher<config, globals>::dispatcher,
                    ops...>::template run<int, config, globals,
                                          config::instruction_t, int>(
                    last_opcode, g,
                    kvms.all_instructions[last_instruction_ring].instructions,
                    laneid);

                kvms.pid_order()[laneid] =
                    kvms.all_instructions[last_instruction_ring].pid_order[lid];
            }
        }

        // Step 3. Construct semaphores
        int opcode = kvms.instruction()[0];
        if (opcode == 0) {
            num_semaphores[kvms.instruction_ring] = 0;
        } else {
            num_semaphores[kvms.instruction_ring] = dispatch_op<
                semaphore_constructor_op_dispatcher<config,
                                                    globals>::dispatcher,
                ops...>::template run<int, config, globals,
                                        ::megakernel::state<config>>(opcode,
                                                                    g, kvms);

            // broadcast the result to all lanes
            num_semaphores[kvms.instruction_ring] = __shfl_sync(
                0xffffffff, num_semaphores[kvms.instruction_ring], 0);
        }

        if (laneid == 0) {
            // Publish the next instruction to all worker roles.
            arrive(kvms.instruction_arrived[kvms.instruction_ring], 1);
        }
    }

    // invalidate remaining semaphores
    for (int i = 0; i < config::INSTRUCTION_PIPELINE_STAGES; i++) {

        auto instruction_index = num_iters - config::INSTRUCTION_PIPELINE_STAGES + i;

        if (instruction_index < 0) {
            continue;
        }

        auto instruction_ring =
            instruction_index % config::INSTRUCTION_PIPELINE_STAGES;

        auto phasebit = (instruction_index / config::INSTRUCTION_PIPELINE_STAGES) & 1;
        kvms.instruction_index = instruction_index;
        kvms.instruction_ring = instruction_ring;

        kittens::wait(kvms.instruction_finished[instruction_ring], phasebit);

        // don't need to invalidate on teardown
        int num_to_invalidate = num_semaphores[instruction_ring];
        for (int sem_idx = laneid; sem_idx < num_to_invalidate; sem_idx += 32) {
            invalidate_semaphore(
                kvms.all_instructions[instruction_ring]
                    .semaphores[sem_idx]);
        }
    }

}

} // namespace controller
} // namespace megakernel
