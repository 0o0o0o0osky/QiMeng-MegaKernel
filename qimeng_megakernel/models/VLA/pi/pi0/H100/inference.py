"""The single public Pi0 H100 inference path: SigLIP -> Gemma -> Action Expert."""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
import threading
from typing import Any

import numpy as np
import torch

from .decode import (
    DecodeSchedule,
    build_decode_schedule,
    decode_arguments,
    materialize_decode,
    model_actions,
)
from .prefill import (
    PrefillSchedules,
    build_prefill_schedules,
    connect_projector,
    gemma_arguments,
    materialize_prefill,
    siglip_arguments,
)
from .preprocess import (
    Pi0H100TensorPack,
    build_checkpoint_tensor_pack,
    postprocess_actions,
    preprocess_observation,
)
from .schedule.mk_kernel import load_kernel
from .schedule.pi0_config import Pi0H100Config as C


GraphKey = tuple[int, int, int, str]


@dataclass
class _GraphEntry:
    checkpoint: Path
    pack: Pi0H100TensorPack
    prefill: PrefillSchedules
    decode: DecodeSchedule
    images: torch.Tensor
    tokens: torch.Tensor
    state: torch.Tensor
    noise: torch.Tensor
    graph: torch.cuda.CUDAGraph


class Pi0Inference:
    """Own static buffers and replay one three-launch CUDA graph per shape key."""

    def __init__(
        self,
        *,
        device: str | torch.device = "cuda:0",
        kernel_dir: str | Path | None = None,
        graph_cache_size: int = 1,
    ) -> None:
        target = torch.device(device)
        if target.type != "cuda":
            raise ValueError("Pi0Inference requires an H100 CUDA device")
        if target.index is None:
            target = torch.device("cuda", torch.cuda.current_device())
        props = torch.cuda.get_device_properties(target)
        if props.major != 9 or props.minor != 0 or props.multi_processor_count != C.sm_count:
            raise RuntimeError(
                f"Pi0 H100 kernels require compute 9.0 with {C.sm_count} SMs; "
                f"got {props.name}, {props.major}.{props.minor}, {props.multi_processor_count} SMs"
            )
        if type(graph_cache_size) is not int or graph_cache_size <= 0:
            raise ValueError("graph_cache_size must be a positive integer")
        self._device = target
        self._kernel_dir = kernel_dir
        self._cache_size = graph_cache_size
        self._graphs: OrderedDict[GraphKey, _GraphEntry] = OrderedDict()
        self._siglip_kernels = {
            views: load_kernel(f"mk_siglip_view{views}", kernel_dir)
            for views in C.VIEWS_CHOICES
        }
        self._gemma_kernel = load_kernel("mk_gemma", kernel_dir)
        self._ae_kernel = load_kernel("mk_ae", kernel_dir)
        self._lock = threading.Lock()

    def _launch(self, entry: _GraphEntry) -> None:
        stream = torch.cuda.current_stream(self._device)
        materialize_prefill(entry.pack, entry.images, entry.tokens)
        self._siglip_kernels[entry.pack.num_views].mk(
            *siglip_arguments(entry.pack, entry.prefill), stream=stream
        )
        connect_projector(entry.pack)
        self._gemma_kernel.mk(*gemma_arguments(entry.pack, entry.prefill), stream=stream)
        materialize_decode(entry.pack, entry.state, entry.noise)
        self._ae_kernel.mk(*decode_arguments(entry.pack, entry.decode), stream=stream)

    def _build_entry(
        self,
        checkpoint: Path,
        *,
        views: int,
        prompt_len: int,
        chunk: int,
    ) -> _GraphEntry:
        pack = build_checkpoint_tensor_pack(
            checkpoint,
            num_views=views,
            prompt_len=prompt_len,
            chunk=chunk,
            flow_steps=10,
            device=self._device,
        )
        entry = _GraphEntry(
            checkpoint=checkpoint,
            pack=pack,
            prefill=build_prefill_schedules(pack),
            decode=build_decode_schedule(pack),
            images=torch.zeros((views, 224, 224, 3), device=self._device),
            tokens=torch.zeros((prompt_len,), dtype=torch.int64, device=self._device),
            state=torch.zeros((C.action_dim,), device=self._device),
            noise=torch.zeros((chunk, C.action_dim), device=self._device),
            graph=torch.cuda.CUDAGraph(),
        )
        warmup = torch.cuda.Stream(device=self._device)
        warmup.wait_stream(torch.cuda.current_stream(self._device))
        with torch.cuda.stream(warmup):
            self._launch(entry)
        torch.cuda.current_stream(self._device).wait_stream(warmup)
        torch.cuda.synchronize(self._device)
        with torch.cuda.graph(entry.graph):
            self._launch(entry)
        return entry

    def infer(
        self,
        checkpoint: str | Path,
        observation: Mapping[str, Any],
        *,
        views: int,
        chunk: int,
        noise: torch.Tensor | np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Run the sole production route and return model and dataset actions."""
        if chunk not in C.INFERENCE_CHUNK_CHOICES:
            raise ValueError(f"chunk must be one of {C.INFERENCE_CHUNK_CHOICES}, got {chunk}")
        resolved = Path(checkpoint).expanduser().resolve()
        inputs = preprocess_observation(resolved, observation, views=views)
        noise_cpu = torch.as_tensor(noise, dtype=torch.float32, device="cpu")
        if tuple(noise_cpu.shape) != (chunk, C.action_dim) or not torch.isfinite(noise_cpu).all():
            raise ValueError(f"noise must be finite [{chunk}, {C.action_dim}]")
        key: GraphKey = (views, inputs.prompt_len, chunk, str(self._device))

        with self._lock, torch.inference_mode(), torch.cuda.device(self._device):
            entry = self._graphs.get(key)
            if entry is None or entry.checkpoint != resolved:
                if entry is not None:
                    del self._graphs[key]
                entry = self._build_entry(
                    resolved, views=views, prompt_len=inputs.prompt_len, chunk=chunk
                )
                self._graphs[key] = entry
                while len(self._graphs) > self._cache_size:
                    self._graphs.popitem(last=False)
            else:
                self._graphs.move_to_end(key)
            entry.images.copy_(inputs.images, non_blocking=True)
            entry.tokens.copy_(inputs.prompt_tokens, non_blocking=True)
            entry.state.copy_(inputs.state, non_blocking=True)
            entry.noise.copy_(noise_cpu, non_blocking=True)
            entry.graph.replay()
            model = model_actions(entry.pack).float().cpu().numpy().copy()
        actions = postprocess_actions(
            resolved, model, dataset=inputs.dataset, raw_state=inputs.raw_state
        )
        return {"model_actions": model, "actions": actions}


__all__ = ["Pi0Inference"]
