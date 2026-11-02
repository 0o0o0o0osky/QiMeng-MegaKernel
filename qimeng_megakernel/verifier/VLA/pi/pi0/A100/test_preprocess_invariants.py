"""Pi0 A100 checkpoint, shape, and physical tensor ABI invariants."""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from qimeng_megakernel.models.VLA.pi.pi0.A100.preprocess import (
    REQUIRED_CHECKPOINT_TENSORS,
    build_checkpoint_tensor_pack,
    inspect_checkpoint_inventory,
    materialize_ae_noise,
    materialize_ae_state,
    materialize_ae_time_biases,
)
from qimeng_megakernel.models.VLA.pi.pi0.A100.schedule.pi0_config import (
    Pi0A100Config as C,
)
from qimeng_megakernel.models.VLA.pi.pi0.A100.schedule.weight_io import checkpoint_path


def _checkpoint() -> Path:
    configured = os.environ.get("PI0_CKPT")
    if not configured:
        pytest.skip("set PI0_CKPT to a converted Pi0 checkpoint directory or safetensors file")
    try:
        return checkpoint_path(configured)
    except FileNotFoundError as exc:
        pytest.skip(str(exc))


def _require_sm80_108() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    props = torch.cuda.get_device_properties(0)
    actual = (props.major, props.minor, props.multi_processor_count)
    if actual != (8, 0, C.sm_count):
        pytest.skip(f"Pi0 A100 tests require SM80 with {C.sm_count} SMs; got {actual}")


def test_six_shape_invariants() -> None:
    # `encoder_seq_pad` rounds to the 64-row Gemma tile, not 128: the encoder GEMMs
    # never mask rows, so every padded row is computed in full at ~46 us per row per
    # module.  `ae_kv_rows` still rounds to the 64-row AE KV block and is unchanged.
    expected = {
        (1, 20): (256, 276, 320, 21, 384),
        (1, 50): (256, 276, 320, 51, 384),
        (2, 20): (512, 532, 576, 21, 640),
        (2, 50): (512, 532, 576, 51, 640),
        (3, 20): (768, 788, 832, 21, 896),
        (3, 50): (768, 788, 832, 51, 896),
    }
    assert C.sm_count == 108
    assert C.CHUNK_CHOICES == (20, 50)
    for (views, chunk), geometry in expected.items():
        shapes = C.shapes_for(views, chunk)
        assert (
            shapes["siglip_seq"],
            shapes["encoder_seq"],
            shapes["encoder_seq_pad"],
            shapes["ae_seq"],
            shapes["ae_kv_rows"],
        ) == geometry
        assert shapes["ae_state_key"] >= shapes["ae_prefix_len"]
        assert shapes["ae_self_end"] <= shapes["ae_kv_rows"]


@pytest.mark.slow
def test_real_checkpoint_schema_is_exact() -> None:
    checkpoint = _checkpoint()
    inventory = inspect_checkpoint_inventory(checkpoint)
    assert inventory.path == checkpoint.resolve()
    assert inventory.required_tensor_count == len(REQUIRED_CHECKPOINT_TENSORS)
    assert inventory.checkpoint_tensor_count == inventory.required_tensor_count
    assert inventory.unused_checkpoint_keys == ()
    assert inventory.required_tensor_bytes < inventory.file_size


@pytest.mark.cuda
@pytest.mark.slow
def test_full_pack_matches_a100_abi() -> None:
    _require_sm80_108()
    pack = build_checkpoint_tensor_pack(
        _checkpoint(),
        num_views=1,
        prompt_len=20,
        chunk=20,
        flow_steps=C.num_flow_steps,
        device="cuda:0",
    )
    fp32 = {
        "siglip": {"ln_stats"},
        "gemma": {"rms_factor", "attn_stats"},
        "ae": {"x_acc", "rms_factor", "attn_stats", "act_bias"},
    }
    for tensor_set in (pack.siglip, pack.gemma, pack.ae):
        assert tuple(tensor_set.tensors) == tensor_set.abi_order
        for name, tensor in tensor_set.tensors.items():
            assert tensor.is_cuda and tensor.device.index == 0
            assert tensor.is_contiguous()
            if name == "Bar":
                assert tensor.dtype == torch.int32
                assert torch.count_nonzero(tensor) == 0
            elif name in fp32[tensor_set.module]:
                assert tensor.dtype == torch.float32
            else:
                assert tensor.dtype == torch.bfloat16

    assert pack.ae.tensors["k_cache"].data_ptr() == pack.gemma.tensors["k_cache"].data_ptr()
    assert pack.ae.tensors["v_cache"].data_ptr() == pack.gemma.tensors["v_cache"].data_ptr()
    assert pack.ae_layout == C.ae_kv_layout_for(1, 20, 20)

    materialize_ae_time_biases(pack)
    materialize_ae_state(pack, torch.zeros(C.action_dim, device="cuda:0"))
    materialize_ae_noise(pack, torch.zeros(20, C.action_dim, device="cuda:0"))
    assert torch.count_nonzero(pack.ae.tensors["noise"][:, :, 21:]) == 0

    with pytest.raises(ValueError, match="finite"):
        materialize_ae_state(pack, torch.full((C.action_dim,), float("nan"), device="cuda:0"))
    with pytest.raises(ValueError, match="finite"):
        materialize_ae_noise(
            pack,
            torch.full((20, C.action_dim), float("inf"), device="cuda:0"),
        )
