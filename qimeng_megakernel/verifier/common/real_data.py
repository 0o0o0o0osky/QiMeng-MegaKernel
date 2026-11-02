"""Build verifier `Globals` from a *real* prefill, not synthetic tensors.

Synthetic inputs miss the bugs that need a real prompt embedding and a non-zero KV
cache to show up -- RoPE at a real position, a cache that is partly written, logits
whose argmax is a real token.  So every suite's unit_real tests start here.

Per-model differences are three, and they are the whole reason there used to be one
copy of this file per tree:

1. **Which `Globals`** the decode schedule defines -> `spec.globals_module_globals`
   is not needed at all: `llm_preprocess_decode` returns the right one; the tree-local
   import existed only for the type annotation.
2. **Which weights** to load -> `spec.default_model_path`, resolved from the suite environment.
3. **What to snapshot** in `backup_globs` -> `spec.backup_fields`.  The field lists
   genuinely differ (a MoE schedule has no `silu_out`), so this is data, not a fork.

Additional suite setup, such as applying HF config values or tracing memory,
goes in `spec.post_preprocess`, which receives the model and globals.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import torch

from qimeng_megakernel.models.LLM.decode import _load_hf_model, llm_preprocess_decode
from qimeng_megakernel.models.LLM.prefill import InferenceConfig, run_prefill

#: Fields a plain dense-decode schedule exposes.  Suites override via `RealDataSpec`.
DEFAULT_BACKUP_FIELDS: tuple[str, ...] = (
    "hidden_states",
    "q_post_rope",
    "k_cache",
    "v_cache",
    "attn_out",
    "attn_out_intermediates",
    "attn_lse_intermediates",
    "silu_out",
    "logits",
    "barriers",
)

#: Prompts are kept SHORT on purpose: single-token decode is fast, prefill is not, and
#: prefill time scales with prompt length.  Only reach for the longer ones when the
#: test is actually about long context.
SHORT_PROMPTS: tuple[str, ...] = (
    "Hello",              # ~1-2 tokens
    "Tell me",            # ~2-3 tokens
    "Python is",          # ~2-3 tokens
)
MEDIUM_PROMPT = "Write a story about AI"
LONG_PROMPT = "Explain machine learning, neural networks, and deep learning in simple terms"


def resolve_model_path(*env_vars: str, fallback: str) -> str:
    """First non-empty value among `env_vars`, else `fallback`.

    Local snapshot dirs are preferred over hub ids not only for offline runs but
    because a hub id triggers the `trust_remote_code` prompt for some repos.
    """
    for var in env_vars:
        val = os.environ.get(var)
        if val:
            return val
    return fallback


@dataclass(frozen=True)
class RealDataSpec:
    """Everything that differs between one suite's real-data helpers and another's."""

    default_model_path: str
    backup_fields: Sequence[str] = DEFAULT_BACKUP_FIELDS
    hardware: str | None = None
    #: Runs after `llm_preprocess_decode`, before the embedding is written.
    #: Signature: (globs, model) -> None.
    post_preprocess: Callable[[Any, Any], None] | None = None
    prompts: dict[str, Any] = field(default_factory=dict)


def make_globs_from_real_prefill(
    prompt: str,
    spec: RealDataSpec,
    *,
    model_path: str | None = None,
    device: str = "cuda:0",
    torch_dtype: str = "auto",
    max_new_tokens: int = 1,
    return_model: bool = False,
) -> Any:
    """Create globs carrying real prefill state for `prompt`.

    `max_new_tokens=1` is enough for single-step decode tests; it only sizes the cache.
    Returns `globs`, or `(globs, model)` when `return_model=True`.
    """
    model_path = model_path or spec.default_model_path

    config = InferenceConfig(model=model_path, device=device, torch_dtype=torch_dtype)
    prefill = run_prefill(config, [prompt])

    model = _load_hf_model(model_path, device=torch.device(device), dtype=prefill.logits.dtype)
    globs = llm_preprocess_decode(
        model_path,
        prefill=prefill,
        loaded_model=model,
        max_new_tokens=max_new_tokens,
        **({"hardware": spec.hardware} if spec.hardware is not None else {}),
    )

    if spec.post_preprocess is not None:
        spec.post_preprocess(globs, model)

    # Initialize hidden_states with the first token embedding (as in real decode)
    with torch.no_grad():
        emb = model.model.embed_tokens(prefill.first_gen_token)
    globs.hidden_states.copy_(emb.squeeze(0).squeeze(0).to(globs.hidden_states.dtype))

    if return_model:
        return globs, model
    return globs


def backup_globs(globs: Any, fields: Sequence[str] = DEFAULT_BACKUP_FIELDS) -> dict[str, torch.Tensor]:
    """Clone the mutable buffers named in `fields`."""
    return {name: getattr(globs, name).clone() for name in fields}


def backup_all_tensors(globs: Any) -> dict[str, torch.Tensor]:
    """Clone every tensor attribute -- for schedules whose field set is not fixed."""
    return {
        name: val.clone()
        for name, val in vars(globs).items()
        if isinstance(val, torch.Tensor)
    }


def restore_globs(globs: Any, backup: dict[str, torch.Tensor]) -> None:
    """Restore in place, so anything still holding a reference sees the restored state."""
    for key, value in backup.items():
        getattr(globs, key).copy_(value)


def bind(spec: RealDataSpec) -> dict[str, Callable[..., Any]]:
    """Bind the helpers to one suite's `spec`, ready to be re-exported by that suite.

    Returns `make_globs_from_real_prefill`, `backup_globs`, and `restore_globs`
    with the spec already applied.
    """

    def _make(prompt: str, model_path: str | None = None, **kwargs: Any) -> Any:
        return make_globs_from_real_prefill(prompt, spec, model_path=model_path, **kwargs)

    def _backup(globs: Any) -> dict[str, torch.Tensor]:
        return backup_globs(globs, spec.backup_fields)

    return {
        "make_globs_from_real_prefill": _make,
        "backup_globs": _backup,
        "restore_globs": restore_globs,
    }
