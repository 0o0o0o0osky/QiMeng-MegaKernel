"""Complete ten-step Pi0 action-expert schedule and single-launch preparation."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .preprocess import (
    Pi0A100TensorPack,
    materialize_ae_noise,
    materialize_ae_state,
    materialize_ae_time_biases,
)
from .schedule.instructions import scheduler
from .schedule.pi0_config import Pi0A100Config as C


@dataclass(frozen=True)
class DecodeSchedule:
    instructions: torch.Tensor


def _rope(positions: torch.Tensor) -> torch.Tensor:
    half = torch.arange(C.head_dim // 2, device=positions.device, dtype=torch.float32)
    angles = positions.float()[:, None] * (
        C.rope_theta ** (-2.0 * half / C.head_dim)
    )[None]
    result = torch.empty(
        positions.numel(), C.head_dim,
        device=positions.device, dtype=torch.bfloat16,
    )
    result[:, 0::2] = angles.cos().bfloat16()
    result[:, 1::2] = angles.sin().bfloat16()
    return result


def build_decode_schedule(pack: Pi0A100TensorPack) -> DecodeSchedule:
    if pack.flow_steps != 10:
        raise ValueError("the Pi0 native action expert is fixed to ten flow steps")
    instructions = scheduler.schedule_ae_flow(pack.ae_layout)
    queues, _ = scheduler.pack_instructions(instructions)
    device = pack.ae.tensors["x"].device
    return DecodeSchedule(
        torch.tensor(queues, dtype=torch.int32, device=device).unsqueeze(0).contiguous()
    )


def materialize_decode(
    pack: Pi0A100TensorPack,
    state: torch.Tensor,
    noise: torch.Tensor,
) -> None:
    for name in pack.ae.scratch:
        pack.ae.tensors[name].zero_()
    materialize_ae_state(pack, state)
    materialize_ae_noise(pack, noise)
    materialize_ae_time_biases(pack)
    rope = pack.ae.tensors["rope"]
    positions = torch.arange(
        pack.ae_layout["m_pad"], device=rope.device
    ) + pack.ae_layout["prefix_len"]
    rope.copy_(_rope(positions).reshape_as(rope))


def decode_arguments(
    pack: Pi0A100TensorPack, schedule: DecodeSchedule
) -> tuple[torch.Tensor, ...]:
    return (*pack.ae.abi_arguments(), schedule.instructions)


def model_actions(pack: Pi0A100TensorPack) -> torch.Tensor:
    return pack.ae.tensors["noise"][0, 0, 1 : 1 + pack.chunk, : C.action_dim]


__all__ = [
    "DecodeSchedule",
    "build_decode_schedule",
    "decode_arguments",
    "materialize_decode",
    "model_actions",
]
