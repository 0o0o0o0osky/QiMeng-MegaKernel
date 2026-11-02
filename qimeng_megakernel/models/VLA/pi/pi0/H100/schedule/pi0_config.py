"""H100-facing Pi0 shape + hardware constants (Python mirror of pi0_config.cuh).

Shared model shapes come from ``qimeng_megakernel.models.VLA.pi.pi0.config.Pi0Config``.
H100-only pads / SM count live here so fused / pyvm / tests stay in lockstep with
``kernel/common/pi0_config.cuh``.

Views (1/2/3) and action chunks are first-class: use the helpers below instead
of hard-coding 788 / 64. The verifier covers chunks 20 and 50.
"""
from __future__ import annotations

from qimeng_megakernel.models.VLA.pi.pi0.config import DEFAULT_CONFIG, Pi0Config


def _ceil_to(n: int, tile: int) -> int:
    return -(-n // tile) * tile


class Pi0H100Config:
    """Class-attr config (same pattern as LlamaConfig) for fused/pyvm imports."""

    # --- from Pi0Config ---
    num_views = DEFAULT_CONFIG.num_views
    vision_tokens_per_view = DEFAULT_CONFIG.vision_tokens_per_view
    vision_width = DEFAULT_CONFIG.vision_width
    vision_layers = DEFAULT_CONFIG.vision_layers
    vision_mlp_width = DEFAULT_CONFIG.vision_mlp_width
    vision_num_heads = DEFAULT_CONFIG.vision_num_heads

    encoder_width = DEFAULT_CONFIG.encoder_width
    encoder_layers = DEFAULT_CONFIG.encoder_layers
    encoder_mlp_width = DEFAULT_CONFIG.encoder_mlp_width

    decoder_width = DEFAULT_CONFIG.decoder_width
    decoder_layers = DEFAULT_CONFIG.decoder_layers
    decoder_mlp_width = DEFAULT_CONFIG.decoder_mlp_width

    num_heads = DEFAULT_CONFIG.num_heads
    num_kv_heads = DEFAULT_CONFIG.num_kv_heads
    head_dim = DEFAULT_CONFIG.head_dim

    action_dim = DEFAULT_CONFIG.action_dim
    chunk_size = DEFAULT_CONFIG.chunk_size
    num_flow_steps = DEFAULT_CONFIG.num_flow_steps
    prompt_len = DEFAULT_CONFIG.prompt_len
    rope_theta = DEFAULT_CONFIG.rope_theta
    rms_eps = DEFAULT_CONFIG.rms_eps
    ln_eps = DEFAULT_CONFIG.ln_eps

    # --- H100 hardware / pad (must match pi0_config.cuh) ---
    sm_count = 132
    tile_m = 128
    tile_n = 128
    instruction_width = 32
    barrier_slots = 64

    # Gemma encoder: 3-view default seq = 3*256 + 20 = 788, pad to 896
    encoder_seq = 788
    encoder_seq_pad = 896
    encoder_qkv_out = 2560  # 8*256 + 1*256 + 1*256
    gemma_barrier_rows = 11
    gemma_kv_splits = 2
    gemma_kv_block = 64
    gemma_q_tile = 128
    gemma_max_seq_pad = 1024

    # SigLIP
    siglip_seq_pad = 768  # 3 * 256
    siglip_barrier_rows = 10
    siglip_head_dim = 72
    siglip_head_dim_pad = 80

    # Action expert
    ae_m = 64  # chunk+state padded to one WGMMA M tile
    ae_barrier_rows = 14
    ae_kv_block = 64

    # H100-local policy/capacity. The physical ABI retains 50 actions + state.
    production_chunk = 10
    max_chunk_size = 50

    # Config matrix axes used by multilayer verifiers
    VIEWS_CHOICES = (1, 2, 3)
    INFERENCE_CHUNK_CHOICES = (20, 50)
    CHUNK_CHOICES = (production_chunk, *INFERENCE_CHUNK_CHOICES)

    @classmethod
    def from_pi0(cls, cfg: Pi0Config | None = None) -> type["Pi0H100Config"]:
        """Optionally overlay a custom Pi0Config onto class attrs (in-place)."""
        c = cfg or DEFAULT_CONFIG
        cls.num_views = c.num_views
        cls.vision_tokens_per_view = c.vision_tokens_per_view
        cls.vision_width = c.vision_width
        cls.vision_layers = c.vision_layers
        cls.vision_mlp_width = c.vision_mlp_width
        cls.vision_num_heads = c.vision_num_heads
        cls.encoder_width = c.encoder_width
        cls.encoder_layers = c.encoder_layers
        cls.encoder_mlp_width = c.encoder_mlp_width
        cls.decoder_width = c.decoder_width
        cls.decoder_layers = c.decoder_layers
        cls.decoder_mlp_width = c.decoder_mlp_width
        cls.num_heads = c.num_heads
        cls.num_kv_heads = c.num_kv_heads
        cls.head_dim = c.head_dim
        cls.action_dim = c.action_dim
        cls.chunk_size = c.chunk_size
        cls.num_flow_steps = c.num_flow_steps
        cls.prompt_len = c.prompt_len
        cls.rope_theta = c.rope_theta
        cls.rms_eps = c.rms_eps
        cls.ln_eps = c.ln_eps
        cls.encoder_seq = cls.encoder_seq_for(cls.num_views, cls.prompt_len)
        cls.encoder_seq_pad = cls.encoder_seq_pad_for(cls.encoder_seq)
        cls.siglip_seq_pad = cls.siglip_seq_for(cls.num_views)
        return cls

    # ----- shape helpers (views / chunk first-class) -----

    @classmethod
    def siglip_seq_for(cls, num_views: int) -> int:
        """Vision tower sequence length for ``num_views`` cameras."""
        return int(num_views) * cls.vision_tokens_per_view

    @classmethod
    def encoder_seq_for(
        cls,
        num_views: int,
        prompt_len: int | None = None,
    ) -> int:
        """Gemma prefill rows: ``num_views * 256 + prompt``."""
        p = cls.prompt_len if prompt_len is None else int(prompt_len)
        return cls.siglip_seq_for(num_views) + p

    @classmethod
    def encoder_seq_pad_for(cls, seq_len: int, tile: int | None = None) -> int:
        """Pad encoder seq up to a multiple of ``tile_m`` (default 128)."""
        t = cls.tile_m if tile is None else int(tile)
        return _ceil_to(int(seq_len), t)

    @classmethod
    def ae_seq_for(cls, chunk: int) -> int:
        """Flow-decoder query rows: 1 state token + ``chunk`` action tokens."""
        return int(chunk) + 1

    @classmethod
    def ae_m_pad_for(cls, chunk: int) -> int:
        """AE activation rows padded to one WGMMA M tile (``ae_m`` = 64)."""
        seq = cls.ae_seq_for(chunk)
        if seq > cls.ae_m:
            raise ValueError(f"chunk+1={seq} exceeds ae_m={cls.ae_m}")
        return cls.ae_m

    @classmethod
    def ae_prefix_len_for(
        cls,
        num_views: int,
        prompt_len: int | None = None,
    ) -> int:
        """Prefix KV length written by the Gemma encoder for AE attention."""
        return cls.encoder_seq_for(num_views, prompt_len)

    @classmethod
    def ae_kv_layout_for(
        cls,
        num_views: int = 3,
        chunk: int | None = None,
        prompt_len: int | None = None,
    ) -> dict[str, int]:
        """AE KV geometry matching ``test_ae_ops.set_config`` / kernel runtime fields."""
        selected_chunk = cls.production_chunk if chunk is None else int(chunk)
        prefix_len = cls.ae_prefix_len_for(num_views, prompt_len)
        seq = cls.ae_seq_for(selected_chunk)
        m_pad = cls.ae_m_pad_for(selected_chunk)
        blk = cls.ae_kv_block
        kv_row_blk = -(-prefix_len // blk)
        state_key = kv_row_blk * blk
        self_end = state_key + seq
        kv_rows = (kv_row_blk + 1) * blk
        return {
            "num_views": int(num_views),
            "chunk": selected_chunk,
            "prompt_len": cls.prompt_len if prompt_len is None else int(prompt_len),
            "prefix_len": prefix_len,
            "seq": seq,
            "m_pad": m_pad,
            "kv_row_blk": kv_row_blk,
            "state_key": state_key,
            "self_end": self_end,
            "kv_rows": kv_rows,
        }

    @classmethod
    def shapes_for(
        cls,
        num_views: int = 3,
        chunk: int | None = None,
        prompt_len: int | None = None,
    ) -> dict[str, int]:
        """Combined SigLIP / Gemma / AE shape dict for the config matrix."""
        selected_chunk = cls.production_chunk if chunk is None else int(chunk)
        enc = cls.encoder_seq_for(num_views, prompt_len)
        ae = cls.ae_kv_layout_for(num_views, selected_chunk, prompt_len)
        return {
            "num_views": int(num_views),
            "chunk": selected_chunk,
            "prompt_len": ae["prompt_len"],
            "siglip_seq": cls.siglip_seq_for(num_views),
            "encoder_seq": enc,
            "encoder_seq_pad": cls.encoder_seq_pad_for(enc),
            **{f"ae_{k}": v for k, v in ae.items() if k not in ("num_views", "chunk", "prompt_len")},
        }
