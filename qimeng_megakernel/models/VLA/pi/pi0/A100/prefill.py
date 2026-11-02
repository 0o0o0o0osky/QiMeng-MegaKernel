"""SigLIP and Gemma schedules plus one-launch preparation for each module."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .preprocess import Pi0A100TensorPack, materialize_siglip_position
from .schedule.instructions import scheduler
from .schedule.pi0_config import Pi0A100Config as C


@dataclass(frozen=True)
class PrefillSchedules:
    siglip: torch.Tensor
    gemma: torch.Tensor


def _pack(instructions: list[object], device: torch.device) -> torch.Tensor:
    # Width follows the schedule's queue count and therefore the launch grid.
    queues, _ = scheduler.pack_instructions(instructions)
    return torch.tensor(queues, dtype=torch.int32, device=device).unsqueeze(0).contiguous()


def build_prefill_schedules(pack: Pi0A100TensorPack) -> PrefillSchedules:
    device = pack.siglip.tensors["vision_x"].device
    seq = C.encoder_seq_for(pack.num_views, pack.prompt_len)
    return PrefillSchedules(
        siglip=_pack(
            scheduler.schedule_siglip_runtime(num_views=pack.num_views),
            device,
        ),
        gemma=_pack(
            scheduler.schedule_gemma_encoder(seq, C.encoder_layers, kv_only_last=True),
            device,
        ),
    )


def _rope(positions: torch.Tensor) -> torch.Tensor:
    half = torch.arange(C.head_dim // 2, device=positions.device, dtype=torch.float32)
    inv = C.rope_theta ** (-2.0 * half / C.head_dim)
    angles = positions.float()[:, None] * inv[None]
    result = torch.empty(
        positions.numel(), C.head_dim,
        device=positions.device, dtype=torch.bfloat16,
    )
    result[:, 0::2] = angles.cos().bfloat16()
    result[:, 1::2] = angles.sin().bfloat16()
    return result


def materialize_prefill(
    pack: Pi0A100TensorPack,
    images: torch.Tensor,
    prompt_tokens: torch.Tensor,
) -> None:
    """Reset request-mutated storage and pack the two prefill inputs in place."""
    device = pack.siglip.tensors["vision_x"].device
    expected_images = (pack.num_views, 224, 224, 3)
    if tuple(images.shape) != expected_images or images.device != device:
        raise ValueError(f"images must be {expected_images} on {device}")
    if tuple(prompt_tokens.shape) != (pack.prompt_len,) or prompt_tokens.device != device:
        raise ValueError(f"prompt tokens must be [{pack.prompt_len}] on {device}")
    for tensor_set in (pack.siglip, pack.gemma):
        for name in tensor_set.scratch:
            tensor_set.tensors[name].zero_()
    materialize_siglip_position(pack)

    patch = pack.siglip.tensors["patch_in"]
    patch.zero_()
    grid = images.view(pack.num_views, 16, 14, 16, 14, 3).permute(0, 1, 3, 2, 4, 5)
    destination = patch[0, 0].as_strided(
        (pack.num_views, 16, 16, 14, 14, 3),
        (256 * 768, 16 * 768, 768, 42, 3, 1),
    )
    destination.copy_(grid)

    encoder = pack.gemma.tensors["encoder_x"]
    encoder.zero_()
    vision_rows = C.siglip_seq_for(pack.num_views)
    seq = vision_rows + pack.prompt_len
    encoder[0, 0, vision_rows:seq].copy_(
        pack.runtime_aux["encoder_token_embedding"].index_select(0, prompt_tokens)
        * math.sqrt(C.encoder_width)
    )
    rope = pack.gemma.tensors["rope"]
    rope.zero_()
    rope[0, 0, :seq].copy_(_rope(torch.arange(seq, device=device)))


def siglip_arguments(
    pack: Pi0A100TensorPack, schedules: PrefillSchedules
) -> tuple[torch.Tensor, ...]:
    return (*pack.siglip.abi_arguments(), schedules.siglip)


def connect_projector(pack: Pi0A100TensorPack) -> None:
    rows = C.siglip_seq_for(pack.num_views)
    pack.gemma.tensors["encoder_x"][0, 0, :rows].copy_(
        pack.siglip.tensors["proj_out"][0, 0, :rows]
    )


def gemma_arguments(
    pack: Pi0A100TensorPack, schedules: PrefillSchedules
) -> tuple[torch.Tensor, ...]:
    return (*pack.gemma.abi_arguments(), schedules.gemma)


__all__ = [
    "PrefillSchedules",
    "build_prefill_schedules",
    "connect_projector",
    "gemma_arguments",
    "materialize_prefill",
    "siglip_arguments",
]
