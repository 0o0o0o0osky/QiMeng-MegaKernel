"""Pi0 model constants shared by the golden reference, simulator, and MK paths.

The constants define the OpenPI Pi0 flow-policy shape invariants. Pi0 differs
from Pi0.5 in the action expert: there is **no AdaRMS**.
The flow timestep is injected as an additive bias
(``decoder_action_fused_time_biases[step]``) inside the action in-projection,
and the decoder uses plain RMSNorm. The vision tower (SigLIP So400m) and the
PaliGemma encoder are structurally identical to Pi0.5.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Pi0Config:
    """Hardware-neutral shape invariants for the OpenPI Pi0 flow head."""

    # --- vision tower (SigLIP So400m / ViT-So400m-14) ---
    num_views: int = 3
    vision_tokens_per_view: int = 256
    vision_width: int = 1152
    vision_layers: int = 27
    vision_mlp_width: int = 4304
    vision_num_heads: int = 16
    vision_patch: int = 14

    # --- PaliGemma encoder (prefill, runs once) ---
    encoder_width: int = 2048
    encoder_layers: int = 18
    encoder_mlp_width: int = 16384

    # --- Pi0 action expert / flow decoder ---
    decoder_width: int = 1024
    decoder_layers: int = 18
    decoder_mlp_width: int = 4096

    # --- shared attention geometry (encoder + decoder, MQA 8:1) ---
    num_heads: int = 8
    num_kv_heads: int = 1
    head_dim: int = 256

    # --- flow / action head ---
    action_dim: int = 32
    chunk_size: int = 50
    max_chunk_size: int = 50
    num_flow_steps: int = 10
    prompt_len: int = 20
    max_prompt_len: int = 200
    rope_theta: float = 10000.0
    rms_eps: float = 1e-6
    ln_eps: float = 1e-5

    default_warps: int = 8
    small_output_warps: int = 4

    @property
    def vision_seq_len(self) -> int:
        return self.num_views * self.vision_tokens_per_view

    @property
    def vision_head_dim(self) -> int:
        return self.vision_width // self.vision_num_heads

    @property
    def encoder_seq_len(self) -> int:
        """Runtime encoder prefill rows: vision tokens + language prompt."""
        return self.vision_seq_len + self.prompt_len

    @property
    def max_encoder_seq_len(self) -> int:
        return self.vision_seq_len + self.max_prompt_len

    @property
    def decoder_seq_len(self) -> int:
        """Flow-decoder rows: one state token + ``chunk_size`` action tokens."""
        return self.chunk_size + 1

    @property
    def qkv_width(self) -> int:
        """QKV output width for encoder/decoder = (Q + K + V) * head_dim."""
        return (self.num_heads + 2 * self.num_kv_heads) * self.head_dim

    @property
    def attn_out_width(self) -> int:
        return self.num_heads * self.head_dim

    def validate_chunk_size(self) -> None:
        if not (1 <= self.chunk_size <= self.max_chunk_size):
            raise ValueError(
                f"chunk_size={self.chunk_size} out of range [1, {self.max_chunk_size}]"
            )


DEFAULT_CONFIG = Pi0Config()
