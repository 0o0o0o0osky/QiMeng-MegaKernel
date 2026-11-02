"""S3 harness for the Python VM (PyVM) and MegaKernel (MK).

Model-agnostic; everything model-specific arrives through `MKTestConfig`.

Provides:
- kernel compilation on demand (`compile_kernel_if_needed`)
- `Globals` management (cloning, KV-cache growth)
- comparison with an abs threshold so near-zero denominators cannot blow up max_rel
- separate setup / test instruction phases
- barrier verification

Thresholds default to the loose end (rtol=atol=5e-3); suites that need a documented
gate declare it in their `verifier_config.py` instead of passing literals.
"""
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from qimeng_megakernel.models.instruction import Instruction, run_pyvm_sequence


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class MKTestConfig:
    """Model-agnostic MegaKernel test configuration."""
    name: str  # Test name.
    
    # Fusion instruction generation.
    instruction_generator: Callable  # (layer_idx, sm_count) -> List[Instruction]
    handler_map: Dict[type, Callable]  # {InstructionType: pyvm_handler}
    
    # Output comparison.
    compare_tensors: List[str]  # ["tensor1", "tensor2"]
    
    # Optional setup phase.
    setup_instruction_generators: Optional[List[Callable]] = None
    setup_handler_maps: Optional[List[Dict[type, Callable]]] = None
    
    # Optional barrier verification.
    expected_barriers: Optional[Dict[Tuple[int, int, int], int]] = None
    
    # Tolerances for the same instruction stream in PyVM and MK. The default
    # 5e-3 tolerance accommodates BF16 numerical differences.
    rtol: float = 5e-03
    atol: float = 5e-03
    
    # Execution options.
    use_megakernel_setup: bool = False  # Run setup with the MegaKernel.


@dataclass
class MKTestResult:
    """MegaKernel test result."""
    passed: bool
    stats: Dict[str, Dict[str, float]]  # {tensor_name: {max_abs, max_rel}}
    diffs: List[str]
    barrier_checks: Dict[str, bool]
    error_message: Optional[str] = None


# ============================================================================
# Compilation
# ============================================================================

def compile_kernel_if_needed(mk_dir: Path) -> Path:
    """Build the MegaKernel if no up-to-date ``mk.so`` artifact exists."""
    mk_so = mk_dir / "mk.so"
    if not mk_so.exists():
        candidates = sorted(mk_dir.glob("mk.*.so"))  # e.g. mk.cpython-312-x86_64-linux-gnu.so
        if candidates:
            mk_so = candidates[0]

    if mk_so.exists():
        source_files = list(mk_dir.glob("*.cu")) + list(mk_dir.glob("*.cuh"))
        if source_files:
            newest_source = max(f.stat().st_mtime for f in source_files)
            if mk_so.stat().st_mtime > newest_source:
                return mk_dir

    print(f"[compile] Compiling mk in {mk_dir}")
    conda_activate = os.environ.get("CONDA_ACTIVATE", "")
    cmd = f"{conda_activate} && make" if conda_activate else "make"
    result = subprocess.run(
        cmd,
        shell=True,
        cwd=mk_dir,
        capture_output=True,
        text=True,
    )
    
    if result.returncode != 0:
        raise RuntimeError(f"Compilation failed:\n{result.stdout}\n{result.stderr}")
    
    return mk_dir


# ============================================================================
# Globals Utilities
# ============================================================================

def clone_globals(globs: Any) -> Any:
    """Clone ``Globals`` from dataclass fields or instance attributes."""
    # Preserve the concrete Globals type.
    GlobalsType = type(globs)
    
    # Prefer declared dataclass fields.
    if hasattr(GlobalsType, '__dataclass_fields__'):
        cloned_dict = {}
        for field in GlobalsType.__dataclass_fields__.values():
            name = field.name
            value = getattr(globs, name)
            if isinstance(value, torch.Tensor):
                cloned_dict[name] = value.clone()
            else:
                cloned_dict[name] = value
        return GlobalsType(**cloned_dict)
    
    # Fall back to instance attributes.
    cloned_dict = {}
    for key, value in globs.__dict__.items():
        if isinstance(value, torch.Tensor):
            cloned_dict[key] = value.clone()
        else:
            cloned_dict[key] = value
    
    # Reconstruct through the public initializer when possible.
    try:
        return GlobalsType(**cloned_dict)
    except:
        # Otherwise, allocate the instance and copy its attributes.
        cloned = GlobalsType.__new__(GlobalsType)
        for key, value in cloned_dict.items():
            setattr(cloned, key, value)
        return cloned


def expand_kv_cache_if_needed(globs: Any, position_id: int, target_seq_len: int = 2048) -> None:
    """Grow K and V caches with layout [layers, batch, seq, heads, dim]."""
    if not hasattr(globs, 'k_cache') or not hasattr(globs, 'v_cache'):
        return
    
    if position_id >= globs.k_cache.shape[2]:
        n_layers, batch_size, current_seq_len, n_kv_heads, head_dim = globs.k_cache.shape
        target_seq_len = max(position_id + 1, target_seq_len)
        
        # Grow the K cache.
        new_k_cache = torch.zeros(
            n_layers, batch_size, target_seq_len, n_kv_heads, head_dim,
            device=globs.k_cache.device, dtype=globs.k_cache.dtype
        )
        new_k_cache[:, :, :current_seq_len] = globs.k_cache
        globs.k_cache = new_k_cache
        
        # Grow the V cache.
        new_v_cache = torch.zeros(
            n_layers, batch_size, target_seq_len, n_kv_heads, head_dim,
            device=globs.v_cache.device, dtype=globs.v_cache.dtype
        )
        new_v_cache[:, :, :current_seq_len] = globs.v_cache
        globs.v_cache = new_v_cache


# ============================================================================
# Comparison with Threshold Mechanism
# ============================================================================

def compare_tensors(
    name: str,
    pyvm_val: torch.Tensor,
    mk_val: torch.Tensor,
    rtol: float = 5e-03,
    atol: float = 5e-03,
    threshold: float = 0.01,
) -> Dict[str, any]:
    """Compare tensors without inflating relative error near zero.

    Matching positive or negative infinities are treated as equal.
    """
    if pyvm_val.shape != mk_val.shape:
        return {
            "match": False,
            "max_abs": float('inf'),
            "max_rel": float('inf'),
            "name": name,
            "error": f"Shape mismatch: {pyvm_val.shape} vs {mk_val.shape}"
        }

    # Exclude matching infinities from the difference.
    both_posinf = torch.isposinf(pyvm_val) & torch.isposinf(mk_val)
    both_neginf = torch.isneginf(pyvm_val) & torch.isneginf(mk_val)
    skip = both_posinf | both_neginf

    abs_diff = (pyvm_val - mk_val).abs()
    abs_diff = torch.where(skip, torch.zeros_like(abs_diff), abs_diff)
    max_abs = abs_diff.max().item()
    if torch.isnan(torch.tensor(max_abs)):
        max_abs = 0.0

    # Compute relative error only for finite reference values above the threshold.
    mask = (pyvm_val.abs() > threshold) & ~skip & ~torch.isinf(pyvm_val)
    if mask.any():
        rel_diff_masked = abs_diff[mask] / (pyvm_val[mask].abs() + 1e-8)
        max_rel = rel_diff_masked.max().item()
        if torch.isnan(torch.tensor(max_rel)):
            max_rel = 0.0
    else:
        max_rel = 0.0

    # Substitute the reference at skipped positions before applying allclose.
    compare_val = torch.where(skip, pyvm_val, mk_val)
    matches = torch.allclose(pyvm_val, compare_val, rtol=rtol, atol=atol, equal_nan=True)

    # Absolute tolerance decides the result when near-zero values inflate max_rel.
    pass_by_atol = max_abs <= atol
    match = matches or pass_by_atol

    return {
        "match": match,
        "max_abs": max_abs,
        "max_rel": max_rel,
        "name": name
    }


def _compare_globs_tensors(
    globs_a: Any,
    globs_b: Any,
    tensor_names: List[str],
    rtol: float = 5e-3,
    atol: float = 5e-3,
) -> Dict[str, Dict[str, float]]:
    """Compare tensors between two globs; return {tensor_name: {max_abs, max_rel}}."""
    out = {}
    for name in tensor_names:
        a = getattr(globs_a, name, None)
        b = getattr(globs_b, name, None)
        if a is None or b is None:
            continue
        r = compare_tensors(name, a.float(), b.float(), rtol=rtol, atol=atol)
        out[name] = {"max_abs": r["max_abs"], "max_rel": r["max_rel"]}
    return out


def run_multi_layer_per_layer_stats(
    globs: Any,
    num_layers: int,
    get_instructions_for_layers: Callable[[Any, int], List[Instruction]],
    handler_map: Dict[type, Callable],
    mk_executor: Callable,
    sm_count: int,
    compare_tensors: List[str],
    rtol: float = 1e-1,
    atol: float = 2.0,
    verbose: bool = True,
) -> List[Dict[str, Dict[str, float]]]:
    """
    Run PyVM vs MK for 1 layer, then 2, ..., num_layers; print per-layer error and return stats.
    Used to see how error grows layer by layer (MK vs PyVM).
    """
    from qimeng_megakernel.models.instruction import run_pyvm_sequence

    per_layer: List[Dict[str, Dict[str, float]]] = []
    for L in range(1, num_layers + 1):
        inst = get_instructions_for_layers(globs, L)
        g_pyvm = clone_globals(globs)
        g_mk = clone_globals(globs)
        run_pyvm_sequence(g_pyvm, inst, handler_map)
        mk_executor(g_mk, inst, sm_count)
        stats = _compare_globs_tensors(g_pyvm, g_mk, compare_tensors, rtol=rtol, atol=atol)
        per_layer.append(stats)
        if verbose:
            line = f"  layer {L:2d}:"
            for name, s in stats.items():
                line += f"  {name} max_abs={s['max_abs']:.4e} max_rel={s['max_rel']:.4e}"
            print(line)
        del g_pyvm, g_mk
        if hasattr(torch.cuda, "empty_cache"):
            torch.cuda.empty_cache()
    if verbose:
        worst = per_layer[-1]
        print(f"  [final after {num_layers} layers] " + " ".join(f"{k} max_abs={v['max_abs']:.4e} max_rel={v['max_rel']:.4e}" for k, v in worst.items()))
    return per_layer


def run_full_decode_per_layer_stats(
    globs: Any,
    get_instructions_for_layers: Callable[[Any, int | None], List[Instruction]],
    handler_map: Dict[type, Callable],
    mk_executor: Callable,
    sm_count: int,
    num_hidden_layers: int = 16,
    rtol: float = 1e-1,
    atol: float = 2.0,
    verbose: bool = True,
) -> tuple:
    """
    Run 1..num_hidden_layers layers and print hidden_states error each step;
    then run full decode (incl. lm_head) and print logits error + token match.
    Returns (per_layer_hidden_stats, logits_stats, token_match).
    """
    from qimeng_megakernel.models.instruction import run_pyvm_sequence

    per_layer: List[Dict[str, Dict[str, float]]] = []
    for L in range(1, num_hidden_layers + 1):
        inst = get_instructions_for_layers(globs, L)
        g_pyvm = clone_globals(globs)
        g_mk = clone_globals(globs)
        run_pyvm_sequence(g_pyvm, inst, handler_map)
        mk_executor(g_mk, inst, sm_count)
        stats = _compare_globs_tensors(g_pyvm, g_mk, ["hidden_states"], rtol=rtol, atol=atol)
        per_layer.append(stats)
        if verbose and stats:
            s = stats["hidden_states"]
            print(f"  layer {L:2d}: hidden_states max_abs={s['max_abs']:.4e} max_rel={s['max_rel']:.4e}")
        del g_pyvm, g_mk
        if hasattr(torch.cuda, "empty_cache"):
            torch.cuda.empty_cache()

    inst_full = get_instructions_for_layers(globs, None)
    g_pyvm = clone_globals(globs)
    g_mk = clone_globals(globs)
    run_pyvm_sequence(g_pyvm, inst_full, handler_map)
    mk_executor(g_mk, inst_full, sm_count)
    logits_stats = _compare_globs_tensors(g_pyvm, g_mk, ["logits"], rtol=rtol, atol=atol)
    pyvm_token = g_pyvm.logits.argmax().item()
    mk_token = g_mk.logits.argmax().item()
    token_match = pyvm_token == mk_token

    if verbose:
        if logits_stats.get("logits"):
            s = logits_stats["logits"]
            print(f"  [after lm_head] logits max_abs={s['max_abs']:.4e} max_rel={s['max_rel']:.4e}")
        print(f"  token: PyVM={pyvm_token} MK={mk_token}  match={token_match}")
    return per_layer, logits_stats.get("logits", {}), token_match


# ============================================================================
# Execution Runners
# ============================================================================

def run_pyvm(
    globs: Any,
    instructions: List[Instruction],
    handler_map: Dict[type, Callable],
) -> None:
    """Run the instruction stream with the Python VM (PyVM)."""
    run_pyvm_sequence(globs, instructions, handler_map)


def run_megakernel(
    globs: Any,
    instructions: List[Instruction],
    mk_executor: Callable,
    sm_count: int,
) -> None:
    """Run the instruction stream with the MegaKernel (MK).
    
    Args:
        globs: Globals object.
        instructions: Fusion instruction sequence.
        mk_executor: MegaKernel executor with signature
            ``(globs, instructions, sm_count) -> None``.
        sm_count: Number of SMs.
    """
    mk_executor(globs, instructions, sm_count)


# ============================================================================
# Main Test Function
# ============================================================================

def run_test(
    globs: Any,
    config: MKTestConfig,
    mk_executor: Callable,
    layer_idx: int,
    sm_count: int,
    verbose: bool = True,
) -> MKTestResult:
    """Run a model-agnostic Python VM (PyVM) versus MegaKernel (MK) test.
    
    Args:
        globs: Initial Globals object.
        config: Test configuration.
        mk_executor: MegaKernel executor.
        layer_idx: Layer index.
        sm_count: Number of SMs.
        verbose: Whether to print detailed results.
    
    Returns:
        Test result and comparison statistics.
    """
    if verbose:
        print(f"\n[{config.name}] layer={layer_idx}")
    
    # 1. Generate fusion instructions.
    test_instructions = config.instruction_generator(layer_idx, sm_count)
    
    setup_instructions = []
    if config.setup_instruction_generators:
        for gen in config.setup_instruction_generators:
            setup_insts = gen(layer_idx, sm_count)
            setup_instructions.extend(setup_insts)
    
    if verbose:
        print(f"  Instructions: {len(setup_instructions)} setup + {len(test_instructions)} test")
    
    # 2. Clone Globals.
    globs_pyvm = clone_globals(globs)
    globs_mk = clone_globals(globs)
    
    # 3. Run the optional setup phase.
    if setup_instructions:
        setup_handler_map = {}
        if config.setup_handler_maps:
            for hmap in config.setup_handler_maps:
                setup_handler_map.update(hmap)
        
        try:
            if verbose:
                print(f"  Running setup (PyVM)...")
            run_pyvm(globs_pyvm, setup_instructions, setup_handler_map)
            
            if config.use_megakernel_setup:
                if verbose:
                    print(f"  Running setup (MegaKernel)...")
                run_megakernel(globs_mk, setup_instructions, mk_executor, sm_count)
            else:
                # Use PyVM for setup on both copies.
                run_pyvm(globs_mk, setup_instructions, setup_handler_map)
        except Exception as e:
            import traceback
            error_msg = f"Setup failed: {str(e)}\n{traceback.format_exc()}"
            if verbose:
                print(f"\n❌ {error_msg}")
            return MKTestResult(
                passed=False,
                stats={},
                diffs=[],
                barrier_checks={},
                error_message=error_msg
            )
    
    # 4. Run the test instruction stream.
    try:
        if verbose:
            print(f"  Running test (PyVM)...")
        run_pyvm(globs_pyvm, test_instructions, config.handler_map)
        
        if verbose:
            print(f"  Running test (MegaKernel)...")
        run_megakernel(globs_mk, test_instructions, mk_executor, sm_count)
    except Exception as e:
        import traceback
        error_msg = f"Test execution failed: {str(e)}\n{traceback.format_exc()}"
        if verbose:
            print(f"\n❌ {error_msg}")
        return MKTestResult(
            passed=False,
            stats={},
            diffs=[],
            barrier_checks={},
            error_message=error_msg
        )
    
    # 5. Compare outputs.
    if verbose:
        print(f"\n  [Compare] rtol={config.rtol:.0e} atol={config.atol:.0e}")
    
    stats = {}
    diffs = []
    all_match = True
    
    for tensor_name in config.compare_tensors:
        pyvm_tensor = getattr(globs_pyvm, tensor_name).float()
        mk_tensor = getattr(globs_mk, tensor_name).float()
        
        result = compare_tensors(
            tensor_name,
            pyvm_tensor,
            mk_tensor,
            rtol=config.rtol,
            atol=config.atol,
        )
        
        stats[tensor_name] = result
        
        if verbose:
            print(f"    {tensor_name}: max_abs={result['max_abs']:.6e} max_rel={result['max_rel']:.6e}")
            print(f"      PyVM range: [{pyvm_tensor.min():.6f}, {pyvm_tensor.max():.6f}]")
            print(f"      MK range:   [{mk_tensor.min():.6f}, {mk_tensor.max():.6f}]")
        
        if not result['match']:
            all_match = False
            diffs.append(f"{tensor_name}: max_abs={result['max_abs']:.6e}")
    
    # 6. Verify barriers.
    barrier_checks = {}
    if config.expected_barriers and hasattr(globs_pyvm, 'barriers'):
        if verbose:
            print(f"\n  [Barriers]")
        
        for (layer, row, idx), expected in config.expected_barriers.items():
            pyvm_val = globs_pyvm.barriers[layer, row, idx].item()
            mk_val = globs_mk.barriers[layer, row, idx].item()
            
            check_name = f"barrier[{layer},{row},{idx}]"
            passed = (pyvm_val == expected and mk_val == expected)
            barrier_checks[check_name] = passed
            
            if verbose:
                status = "✓" if passed else "✗"
                print(f"    {check_name}: expected={expected} pyvm={pyvm_val} mk={mk_val} {status}")
            
            if not passed:
                all_match = False
                diffs.append(f"{check_name}: expected={expected} pyvm={pyvm_val} mk={mk_val}")
    
    # 7. Return the result.
    if verbose:
        if all_match:
            print(f"\n✅ All checks passed!")
            worst = max(stats.items(), key=lambda x: (x[1]["max_abs"], x[1]["max_rel"]), default=None)
            if worst:
                name, s = worst
                print(f"  [worst tensor] {name}: max_abs={s['max_abs']:.6e} max_rel={s['max_rel']:.6e}")
        else:
            print(f"\n❌ Test failed: {len(diffs)} mismatches")
    
    return MKTestResult(
        passed=all_match,
        stats=stats,
        diffs=diffs,
        barrier_checks=barrier_checks,
    )
