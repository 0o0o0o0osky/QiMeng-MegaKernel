"""Shape-static A100 subgraph programs used during CUDA Graph capture."""

from __future__ import annotations

from collections.abc import Iterable
from types import ModuleType
from typing import Any

import torch

from .decode import DecodeSchedule, decode_arguments
from .prefill import PrefillSchedules, gemma_arguments, siglip_arguments
from .preprocess import Pi0A100TensorPack
from .schedule.instructions import scheduler
from .schedule.pi0_config import Pi0A100Config as C


_INSTRUCTION_WIDTH = 32


def _dependency_base(module: str, opcode: int) -> int:
    return 5 if module == "siglip" and opcode == 4 else 4


def _phase_rows(
    module: str, instructions: Iterable[Any]
) -> tuple[tuple[int, list[list[int]]], ...]:
    phases: list[tuple[int, list[list[int]]]] = []
    rows: list[list[int]] = []
    previous: tuple[int, int, int, int] | None = None
    for instruction in instructions:
        words = list(instruction.serialize())
        words.extend([0] * (_INSTRUCTION_WIDTH - len(words)))
        base = _dependency_base(module, words[0])
        key = (words[0], words[base], words[base + 1], words[base + 2])
        if key != previous and rows:
            assert previous is not None
            phases.append((previous[0], rows))
            rows = []
        previous = key
        rows.append(words)
    if rows:
        assert previous is not None
        phases.append((previous[0], rows))
    return tuple(phases)


def _device_rows(rows: list[list[int]], device: torch.device) -> torch.Tensor:
    return torch.tensor(rows, dtype=torch.int32, device=device).contiguous()


class SiglipProgram:
    def __init__(
        self,
        pack: Pi0A100TensorPack,
        prefill: PrefillSchedules,
        extension: ModuleType,
        instructions: Iterable[Any] | None = None,
    ) -> None:
        self._extension = extension
        self._views = pack.num_views
        device = pack.siglip.tensors["vision_x"].device
        self._subgraph_barrier = torch.zeros(2, dtype=torch.int32, device=device)
        self._attention_ready = torch.zeros(8, dtype=torch.int32, device=device)
        schedule = (
            scheduler.schedule_siglip_runtime(num_views=self._views)
            if instructions is None
            else instructions
        )
        self._blocks = tuple(
            (opcode, int(rows[0][1]), _device_rows(rows, device))
            for opcode, rows in _phase_rows("siglip", schedule)
        )
        extension.bind(*siglip_arguments(pack, prefill))

    def run(self) -> None:
        self._attention_ready.zero_()
        for opcode, layer, block in self._blocks:
            if opcode == 3:
                continue
            if opcode == 4:
                self._extension.run_attention(
                    self._subgraph_barrier, self._attention_ready, layer, False
                )
                continue
            if opcode in (5, 6):
                continue
            if opcode == 7:
                getattr(self._extension, f"run_mlp_view{self._views}")(
                    self._subgraph_barrier, layer, True
                )
                continue
            self._extension.run_phase(block, opcode)


class GemmaProgram:
    def __init__(
        self,
        pack: Pi0A100TensorPack,
        prefill: PrefillSchedules,
        extension: ModuleType,
        instructions: Iterable[Any] | None = None,
    ) -> None:
        self._extension = extension
        self._views = pack.num_views
        self._encoder_seq = C.encoder_seq_for(self._views, pack.prompt_len)
        device = pack.gemma.tensors["encoder_x"].device
        self._pv_oproj_ready = torch.zeros(13, dtype=torch.int32, device=device)
        self._subgraph_barrier = torch.zeros(2, dtype=torch.int32, device=device)
        schedule = (
            scheduler.schedule_gemma_encoder(
                self._encoder_seq, C.encoder_layers, kv_only_last=True
            )
            if instructions is None
            else instructions
        )
        blocks = []
        for opcode, rows in _phase_rows("gemma", schedule):
            if self._views == 2 and opcode == 3:
                rows = [row for row in rows if row[3] >= 32 or row[3] % 2 == 0]
            blocks.append((opcode, int(rows[0][1]), _device_rows(rows, device)))
        self._blocks = tuple(blocks)
        extension.bind(*gemma_arguments(pack, prefill))

    def run(self) -> None:
        self._pv_oproj_ready.zero_()
        for opcode, layer, block in self._blocks:
            if self._views in (2, 3) and opcode == 3 and layer < C.encoder_layers - 1:
                continue
            if self._views == 2 and opcode == 4:
                self._extension.run_qkv_qk_view2(
                    self._subgraph_barrier, layer, self._encoder_seq
                )
            elif self._views == 3 and opcode == 4:
                self._extension.run_qkv_qk_view3(
                    self._subgraph_barrier, layer, self._encoder_seq
                )
            elif opcode == 11:
                getattr(self._extension, f"run_pv_oproj_view{self._views}")(
                    self._pv_oproj_ready, layer, self._views == 3
                )
            elif opcode == 6:
                continue
            elif self._views == 2 and opcode == 3:
                self._extension.run_qkv_view2(block)
            elif self._views == 1 and opcode == 9:
                self._extension.run_down_view1(layer)
            else:
                self._extension.run_phase(block, opcode)


class ActionExpertProgram:
    def __init__(
        self,
        pack: Pi0A100TensorPack,
        decode: DecodeSchedule,
        extension: ModuleType,
        instructions: Iterable[Any] | None = None,
    ) -> None:
        self._extension = extension
        self._query_rows = pack.chunk + 1
        self._state_key = pack.ae_layout["state_key"]
        device = pack.ae.tensors["x"].device
        schedule = (
            scheduler.schedule_ae_flow(pack.ae_layout)
            if instructions is None
            else instructions
        )
        self._require_terminal_norm = instructions is None
        self._blocks = tuple(
            (
                opcode,
                int(rows[0][1]),
                int(rows[0][10]),
                int(rows[0][11]),
                _device_rows(rows, device),
            )
            for opcode, rows in _phase_rows("ae", schedule)
        )
        extension.bind(*decode_arguments(pack, decode))

    def run(self) -> None:
        pending_residual = False
        for opcode, layer, prefix_len, state_key, block in self._blocks:
            if opcode in (1, 2, 13):
                norm_row = 2 if opcode == 13 else opcode - 1
                if pending_residual:
                    self._extension.run_residual_rmsn(norm_row, self._query_rows)
                    pending_residual = False
                elif opcode in (1, 2):
                    self._extension.run_rmsn(norm_row, self._query_rows)
                else:
                    self._extension.run_phase(block, opcode)
            elif opcode == 4:
                self._extension.run_materialized(
                    layer, self._query_rows, prefix_len, state_key
                )
            elif opcode == 5:
                continue
            elif opcode in (11, 12, 14):
                self._extension.run_action(
                    opcode, prefix_len, state_key, self._query_rows
                )
            elif opcode == 3:
                self._extension.run_qkv(layer, self._query_rows, self._state_key)
            elif opcode in (6, 9):
                self._extension.run_logical(opcode, layer, self._query_rows, True)
                pending_residual = True
            elif opcode == 15:
                continue
            elif opcode == 10:
                self._extension.run_logical(opcode, layer, self._query_rows)
            else:
                self._extension.run_phase(block, opcode)
        if pending_residual:
            if self._require_terminal_norm:
                raise RuntimeError(
                    "AE schedule ended with an unfused residual reduction"
                )
            self._extension.run_residual_rmsn(0, self._query_rows)


class ProductionPrograms:
    def __init__(
        self,
        pack: Pi0A100TensorPack,
        prefill: PrefillSchedules,
        decode: DecodeSchedule,
        extensions: tuple[ModuleType, ModuleType, ModuleType],
    ) -> None:
        siglip, gemma, ae = extensions
        self.siglip = SiglipProgram(pack, prefill, siglip)
        self.gemma = GemmaProgram(pack, prefill, gemma)
        self.ae = ActionExpertProgram(pack, decode, ae)


__all__ = [
    "ActionExpertProgram",
    "GemmaProgram",
    "ProductionPrograms",
    "SiglipProgram",
]
