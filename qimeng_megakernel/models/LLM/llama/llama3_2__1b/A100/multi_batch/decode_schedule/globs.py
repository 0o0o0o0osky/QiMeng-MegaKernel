"""Multi-batch decode Globals (extends single_batch). Ragged prompts, EOS, SplitKV validity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from ...single_batch.decode_schedule.globs import Globals as _BaseGlobals


@dataclass
class Globals(_BaseGlobals):
    """Multi-batch Globals: per-row position, active mask, SplitKV validity."""

    position_ids: Optional[torch.Tensor] = None
    kv_lengths: Optional[torch.Tensor] = None
    active_mask: Optional[torch.Tensor] = None
    eos_reached: Optional[torch.Tensor] = None
    last_generated_tokens: Optional[torch.Tensor] = None
    eos_token_id: Optional[int] = None
    attn_partial_valid: Optional[torch.Tensor] = None
    attn_rms_out: Optional[torch.Tensor] = None
    mlp_rms_out: Optional[torch.Tensor] = None
    lm_head_rms_out: Optional[torch.Tensor] = None
