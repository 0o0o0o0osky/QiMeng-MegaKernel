"""S4 real LIBERO action comparison for every supported view/chunk pair."""
from __future__ import annotations

import math
import os
from pathlib import Path
import sys
import types

import pytest
from safetensors.torch import load_model
import torch

from qimeng_megakernel.models.VLA.pi.pi0.H100.inference import Pi0Inference
from qimeng_megakernel.models.VLA.pi.pi0.H100.preprocess import preprocess_observation
from qimeng_megakernel.verifier.VLA.pi.pi0.H100.verifier_config import CONFIG


_REQUIRED_ENV = (
    "OPENPI_ROOT",
    "PI0_CKPT",
    "PI0_LIBERO_HDF5",
    "PI0_TOKENIZER",
)
pytestmark = pytest.mark.skipif(
    any(not os.environ.get(name) for name in _REQUIRED_ENV),
    reason="set the Pi0 verifier asset environment variables",
)


def _official_embed_prefix(self, images, image_masks, tokens, token_masks):
    embeddings, masks, attention = [], [], []
    for image, image_mask in zip(images, image_masks, strict=True):
        embedded = self._apply_checkpoint(self.paligemma_with_expert.embed_image, image)
        batch, image_tokens = embedded.shape[:2]
        embeddings.append(embedded)
        masks.append(image_mask[:, None].expand(batch, image_tokens))
        attention.extend([0] * image_tokens)
    language = self._apply_checkpoint(
        self.paligemma_with_expert.embed_language_tokens, tokens
    )
    language = language * math.sqrt(language.shape[-1])
    embeddings.append(language)
    masks.append(token_masks)
    attention.extend([0] * language.shape[1])
    result = torch.cat(embeddings, dim=1)
    mask = torch.cat(masks, dim=1)
    attention_mask = torch.tensor(attention, dtype=torch.bool, device=mask.device)
    return result, mask, attention_mask[None].expand(mask.shape[0], -1)


def _openpi_actions(checkpoint: Path, observation, views: int, chunk: int, noise: torch.Tensor) -> torch.Tensor:
    root = Path(os.environ["OPENPI_ROOT"]).expanduser().resolve()
    sys.path.insert(0, str(root / "src"))
    from openpi.models.pi0_config import Pi0Config
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

    inputs = preprocess_observation(checkpoint, observation, views=views)
    model = PI0Pytorch(Pi0Config(action_horizon=chunk)).to("cuda:0").eval()
    load_model(model, str(checkpoint), strict=False)
    model.embed_prefix = types.MethodType(_official_embed_prefix, model)
    slots = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
    images = {}
    masks = {}
    for index, slot in enumerate(slots):
        source = inputs.images[min(index, inputs.images.shape[0] - 1)]
        images[slot] = source.permute(2, 0, 1)[None].to("cuda:0")
        masks[slot] = torch.tensor([index < views], device="cuda:0")
    official_observation = types.SimpleNamespace(
        images=images,
        image_masks=masks,
        state=inputs.state[None].to("cuda:0"),
        tokenized_prompt=inputs.prompt_tokens[None].to("cuda:0"),
        tokenized_prompt_mask=torch.ones(
            1, inputs.prompt_len, dtype=torch.bool, device="cuda:0"
        ),
        token_ar_mask=None,
        token_loss_mask=None,
    )
    sample = getattr(model.sample_actions, "_torchdynamo_orig_callable", model.sample_actions)
    with torch.inference_mode():
        return sample(
            torch.device("cuda:0"), official_observation,
            noise=noise[None].to("cuda:0"), num_steps=10,
        )[0].float().cpu()


@pytest.mark.parametrize("views", (1, 2, 3))
@pytest.mark.parametrize("chunk", (20, 50))
def test_e2e_action(views: int, chunk: int) -> None:
    from examples.pi0 import load_libero_observation

    checkpoint = Path(os.environ["PI0_CKPT"]).expanduser().resolve()
    observation = load_libero_observation(os.environ["PI0_LIBERO_HDF5"])
    generator = torch.Generator(device="cpu").manual_seed(20260811)
    noise = torch.randn(chunk, 32, generator=generator)
    oracle = _openpi_actions(checkpoint, observation, views, chunk, noise)
    engine = Pi0Inference(device="cuda:0")
    result = engine.infer(checkpoint, observation, views=views, chunk=chunk, noise=noise)
    actual = torch.from_numpy(result["model_actions"])
    assert actual.shape == oracle.shape == (chunk, 32)
    assert torch.isfinite(actual).all()
    assert torch.isfinite(oracle).all()
    gate = CONFIG.gate("S4")
    if gate.value_range is not None:
        low, high = gate.value_range
        assert bool((actual >= low).all() and (actual <= high).all())
    cosine = torch.nn.functional.cosine_similarity(actual.flatten(), oracle.flatten(), dim=0)
    max_abs = (actual - oracle).abs().max()
    norm_ratio = actual.norm() / oracle.norm()
    assert float(cosine) >= gate.action_cosine
    assert float(max_abs) <= gate.action_max_abs
    assert gate.norm_ratio_min <= float(norm_ratio) <= gate.norm_ratio_max
