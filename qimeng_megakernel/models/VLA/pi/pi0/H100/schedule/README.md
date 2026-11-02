"""pi0 / H100 schedule (LLM-style layout, VLA naming).

Layout mirrors ``models/LLM/llama/.../decode_schedule/`` but is named
``schedule`` because pi0 is not a decode model:

- ``fused/`` — untiled torch ops (baseline↔fused correctness)
- ``instructions/instruction_defs.py`` — opcode ABI twin of ``kernel/common/*_instruction_list.cuh``
- ``instructions/python_vm.py`` — tiled handlers (fused↔pyvm; per-tile still stub)
- ``instructions/scheduler.py`` — shape-specialized instruction emitters
- ``globs.py`` / ``pi0_config.py`` / ``mk_kernel.py``

Kernels remain in ``../kernel/``.
"""
