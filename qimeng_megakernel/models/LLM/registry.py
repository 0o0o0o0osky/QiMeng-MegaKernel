"""
LLM backend registry: matches model identifiers to preprocess implementations.

How registry picks which code to run
------------------------------------
1. You pass model = "<HF id>" or "<local path>" (e.g. --model meta-llama/Llama-3.2-1B-Instruct
   or --model /path/to/llama3.2_1b), and hardware = A100 | H100 (or auto).

2. Local path resolution (optional):
   If registry.local.json exists and model is one of its keys, model is replaced by
   the value (model_key), e.g. "/path/to/llama3.2_1b" -> "llama-3.2-1b". So local paths
   never appear in the committed registry; they only map to a model_key.

3. Main registry (registry.json) lookup:
   For each (family, model_key) we match model against model_key and that model's hf_ids.
   - HF id "meta-llama/Llama-3.2-1B-Instruct" matches model_key "llama-3.2-1b" via hf_ids.
   - After step 2, model may already be "llama-3.2-1b", so it matches by model_key.
   The first matching (family, model_key) gives that model's implementations.

4. Implementation selection:
   From that model we take implementations[arch][hardware] (arch default first key, e.g.
   "ampere"; hardware from --hardware or first key). The registry stores only the
   model/hardware-specific preprocess module. Every LLM uses the same shared decode module;
   model differences are resolved there through the schedule registry and config.

Summary: model string (HF id or local path) -> model_key -> (family, model_key) ->
  implementations[arch][hardware] -> preprocess module + shared LLM decode module. Local
  path is only used to get model_key via registry.local.json; the rest is in
  registry.json.

How to use a local model
------------------------
1. Put your model dir somewhere (e.g. /data/models/llama3.2_1b with config.json and
   safetensors inside).
2. Create qimeng_megakernel/models/LLM/registry.local.json (copy from
   registry.local.json.example). Add one entry: your local path -> model_key.
   Example:
     {
       "/data/models/llama3.2_1b": "llama-3.2-1b",
       "/data/models/llama3.2_3b": "llama-3.2-3b"
     }
   The model_key must exist in registry.json (e.g. "llama-3.2-1b").
3. Run with that path:
   - By arg:  python -m examples.llm --model /data/models/llama3.2_1b --route fused
   - By env:  LLM_MODEL=/data/models/llama3.2_1b python -m examples.llm --route mk
   Do not commit registry.local.json (it is gitignored).

- Committed registry (registry.json): HF model ids and preprocess implementations only.
  Decode is always `qimeng_megakernel.models.LLM.decode`.
- Local registry (registry.local.json, same dir): optional, gitignored. Maps local
  paths or custom ids to model_key so backend can be resolved; do not commit.
"""
from __future__ import annotations

import importlib
from collections.abc import Iterable
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


SHARED_LLM_DECODE_MODULE = "qimeng_megakernel.models.LLM.decode"

_MODEL_KEY_TO_MODULE = {
    "llama-3.2-1b": "llama3_2__1b",
    "llama-3.2-3b": "llama3_2__3b",
    "qwen2.5-3b": "qwen2_5__3b",
    "qwen3-0.6b": "qwen3__0_6b",
    "qwen1.5-moe-a2.7b": "qwen1_5_moe_a2_7b",
}


@dataclass(frozen=True)
class LLMImplementation:
    family: str
    model_key: str
    arch: str
    hardware: str
    preprocess: str
    decode: str


@dataclass(frozen=True)
class LLMDecodeSchedule:
    package_name: str
    package: Any
    scheduler: Any | None
    layer_scheduler: Any | None
    python_vm: Any | None
    get_mk: Any | None
    call_mk: Any | None
    config: Any | None
    globs_cls: Any | None


def _load_registry() -> dict[str, Any]:
    registry_path = Path(__file__).resolve().parent / "registry.json"
    with registry_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_local_registry() -> dict[str, str] | None:
    """Load optional local path -> model_key map. Not committed."""
    local_path = Path(__file__).resolve().parent / "registry.local.json"
    if not local_path.exists():
        return None
    with local_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _match_model_key(model: str, model_key: str, hf_ids: list[str]) -> bool:
    name = model.lower()
    if model_key.lower() in name:
        return True
    for hf_id in hf_ids:
        if hf_id.lower() in name:
            return True
    return False


def resolve_llm_impl(
    model: str,
    *,
    arch: str | None = None,
    hardware: str | None = None,
) -> LLMImplementation:
    data = _load_registry()
    local = _load_local_registry()
    if local is not None:
        # Resolve local path to model_key; normalize path so "/path/to/dir" and "/path/to/dir/" match
        if model in local:
            model = local[model]
        else:
            try:
                model_path = Path(model).resolve()
                for key, value in local.items():
                    if "/" in key and Path(key).resolve() == model_path:
                        model = value
                        break
                else:
                    model_resolved = str(model_path)
                    if model_resolved in local:
                        model = local[model_resolved]
            except (OSError, RuntimeError):
                pass
    families = data.get("families", {})
    for family, family_info in families.items():
        models = family_info.get("models", {})
        for model_key, model_info in models.items():
            hf_ids = model_info.get("hf_ids", [])
            if not _match_model_key(model, model_key, hf_ids):
                continue
            impls = model_info.get("implementations", {})
            if not impls:
                raise NotImplementedError(
                    f"No implementations registered for: {model_key}"
                )
            chosen_arch = arch or next(iter(impls.keys()))
            arch_impls = impls.get(chosen_arch)
            if arch_impls is None:
                raise NotImplementedError(
                    f"No implementations for arch={chosen_arch} in {model_key}"
                )
            chosen_hw = hardware or next(iter(arch_impls.keys()))
            hw_impl = arch_impls.get(chosen_hw)
            if hw_impl is None:
                raise NotImplementedError(
                    f"No implementations for hardware={chosen_hw} in {model_key}"
                )
            return LLMImplementation(
                family=family,
                model_key=model_key,
                arch=chosen_arch,
                hardware=chosen_hw,
                preprocess=hw_impl["preprocess"],
                decode=SHARED_LLM_DECODE_MODULE,
            )
    # Help users who pass a local path but have no registry.local.json or no mapping
    hint = ""
    if "/" in model or Path(model).exists():
        example_path = Path(__file__).resolve().parent / "registry.local.json.example"
        hint = (
            f" This looks like a local path. To use it: create registry.local.json "
            f"(copy from {example_path.name}) in the same dir, add {{\"{model}\": \"<model_key>\"}} "
            f"where <model_key> is e.g. llama-3.2-1b (see registry.json). Do not commit the .local.json file."
        )
    raise NotImplementedError(f"Model not registered: {model}{hint}")


def model_module_name(model_key: str) -> str:
    return _MODEL_KEY_TO_MODULE.get(
        model_key, model_key.replace(".", "_").replace("-", "_")
    )


def _supports_multibatch(impl: LLMImplementation) -> bool:
    return (
        impl.family == "llama"
        and impl.model_key == "llama-3.2-1b"
        and impl.hardware in {"A100", "H100"}
    )


def decode_schedule_package_name(
    model_name: str,
    hardware: str,
    batch_size: int = 1,
) -> str:
    impl = resolve_llm_impl(model_name, hardware=hardware)
    batch_variant = "multi_batch" if batch_size > 1 else "single_batch"
    if batch_variant == "multi_batch" and batch_size > 16:
        raise NotImplementedError("multi-batch decode supports batch_size<=16")
    if batch_variant == "multi_batch" and not _supports_multibatch(impl):
        raise NotImplementedError(
            "multi-batch fused decode is implemented for Llama 3.2 1B on A100 "
            "and H100 "
            f"(got family={impl.family!r} model_key={impl.model_key!r} "
            f"hardware={impl.hardware!r})"
        )
    return (
        f"qimeng_megakernel.models.LLM.{impl.family}."
        f"{model_module_name(impl.model_key)}.{impl.hardware}."
        f"{batch_variant}.decode_schedule"
    )


def _decode_config(package: Any) -> Any | None:
    return (
        getattr(package, "ModelConfig", None)
        or getattr(package, "Qwen3Config", None)
        or getattr(package, "Qwen2Config", None)
        or getattr(package, "QwenMoeConfig", None)
        or getattr(package, "LlamaConfig", None)
    )


def import_decode_schedule(
    model_name: str,
    hardware: str,
    batch_size: int = 1,
) -> LLMDecodeSchedule:
    package_name = decode_schedule_package_name(model_name, hardware, batch_size)
    package = importlib.import_module(package_name)
    get_mk = call_mk = None
    try:
        mk_kernel = importlib.import_module(f"{package_name}.mk_kernel")
        get_mk = getattr(mk_kernel, "get_mk", None)
        call_mk = getattr(mk_kernel, "call_mk", None)
    except ImportError:
        pass
    return LLMDecodeSchedule(
        package_name=package_name,
        package=package,
        scheduler=getattr(package, "scheduler", None),
        layer_scheduler=getattr(package, "layer_scheduler", None),
        python_vm=getattr(package, "python_vm", None),
        get_mk=get_mk,
        call_mk=call_mk,
        config=_decode_config(package),
        globs_cls=getattr(package, "Globals", None),
    )


def import_pyvm_instruction_modules(
    model_name: str,
    hardware: str,
    batch_size: int,
) -> tuple[Any, Any]:
    package_name = decode_schedule_package_name(model_name, hardware, batch_size)
    return (
        importlib.import_module(f"{package_name}.instructions.scheduler"),
        importlib.import_module(f"{package_name}.instructions.python_vm"),
    )


def iter_decode_schedule_package_names() -> Iterable[str]:
    llm_root = Path(__file__).resolve().parent
    for init_py in sorted(llm_root.glob("**/decode_schedule/__init__.py")):
        relative = init_py.parent.relative_to(Path(__file__).resolve().parents[3])
        yield ".".join(relative.parts)
