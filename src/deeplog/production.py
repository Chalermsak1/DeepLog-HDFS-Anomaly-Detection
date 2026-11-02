"""
DeepLog Production Optimization & TorchScript JIT Module
========================================================
Provides TorchScript JIT tracing, optimization, numerical parity verification,
and serving benchmarks across CPU and Apple Silicon MPS devices.
"""

import time
import os
from typing import Dict, Any, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import numpy as np

from deeplog.deeplog import DeepLog


def select_serving_device(preferred: Optional[str] = None) -> torch.device:
    """Select the optimal execution device for serving."""
    if preferred is not None:
        p = preferred.lower()
        if p in ("mps", "metal") and torch.backends.mps.is_available():
            return torch.device("mps")
        elif p == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        elif p == "cpu":
            return torch.device("cpu")

    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def synchronize_device(device: torch.device):
    """Ensure all asynchronous operations on device are completed for accurate timing."""
    if device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()
    elif device.type == "cuda" and hasattr(torch, "cuda") and hasattr(torch.cuda, "synchronize"):
        torch.cuda.synchronize()


def trace_deeplog(
    model: Union[DeepLog, nn.Module],
    example_input: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
    context_length: int = 10,
    optimize_for_inference: bool = True,
) -> torch.jit.ScriptModule:
    """
    Trace a DeepLog model using TorchScript JIT.

    Parameters
    ----------
    model : DeepLog or nn.Module
        Trained DeepLog model instance.
    example_input : torch.Tensor, optional
        Representative input tensor of shape (batch_size, context_length).
        If None, a tensor of shape (1, context_length) is created.
    device : torch.device, optional
        Target device for tracing. If None, uses model's device or CPU.
    context_length : int, default=10
        Context length if example_input is generated automatically.
    optimize_for_inference : bool, default=True
        Whether to apply torch.jit.optimize_for_inference to the traced graph.

    Returns
    -------
    traced_model : torch.jit.ScriptModule
        Compiled TorchScript ScriptModule ready for low-latency serving.
    """
    if device is None:
        try:
            device = next(model.parameters()).device
        except (StopIteration, AttributeError):
            device = torch.device("cpu")

    model = model.to(device)
    model.eval()

    if example_input is None:
        vocab_size = getattr(model, "input_size", 30)
        example_input = torch.randint(0, vocab_size, (1, context_length), dtype=torch.long, device=device)
    else:
        example_input = example_input.to(device)

    # Perform TorchScript JIT tracing
    with torch.no_grad():
        traced_model = torch.jit.trace(model, example_input)

    # Optionally run optimize_for_inference
    if optimize_for_inference:
        try:
            traced_model = torch.jit.optimize_for_inference(traced_model)
        except Exception:
            # Fall back to standard traced module if optimize_for_inference is not supported
            pass

    traced_model.eval()
    return traced_model


def verify_numerical_parity(
    native_model: nn.Module,
    jit_model: torch.jit.ScriptModule,
    batch_sizes: List[int] = (1, 32, 256, 1024),
    context_length: int = 10,
    atol: float = 1e-5,
    device: Optional[torch.device] = None,
) -> Tuple[bool, float, Dict[str, Any]]:
    """
    Verify numerical output parity between Native PyTorch and TorchScript JIT.

    Parameters
    ----------
    native_model : nn.Module
        Native PyTorch model.
    jit_model : torch.jit.ScriptModule
        Traced TorchScript model.
    batch_sizes : list of int
        List of batch sizes to verify.
    context_length : int, default=10
        Input sequence context length.
    atol : float, default=1e-5
        Absolute tolerance for torch.allclose.
    device : torch.device, optional
        Device to run verification on.

    Returns
    -------
    all_passed : bool
        True if all batch sizes pass torch.allclose(native, jit, atol=atol).
    overall_max_diff : float
        Maximum absolute difference observed across all tests.
    details : dict
        Per-batch verification statistics.
    """
    if device is None:
        try:
            device = next(native_model.parameters()).device
        except (StopIteration, AttributeError):
            device = torch.device("cpu")

    native_model.eval()
    jit_model.eval()

    vocab_size = getattr(native_model, "input_size", 30)
    all_passed = True
    overall_max_diff = 0.0
    details = {}

    for bs in batch_sizes:
        test_input = torch.randint(0, vocab_size, (bs, context_length), dtype=torch.long, device=device)

        with torch.no_grad():
            out_native = native_model(test_input)
            out_jit = jit_model(test_input)

        is_close = bool(torch.allclose(out_native, out_jit, atol=atol))
        max_diff = float(torch.max(torch.abs(out_native - out_jit)).item())
        mean_diff = float(torch.mean(torch.abs(out_native - out_jit)).item())

        overall_max_diff = max(overall_max_diff, max_diff)
        if not is_close:
            all_passed = False

        details[f"batch_{bs}"] = {
            "batch_size": bs,
            "allclose": is_close,
            "max_abs_diff": max_diff,
            "mean_abs_diff": mean_diff,
            "passed": is_close,
        }

    return all_passed, overall_max_diff, details


def benchmark_serving(
    model: Union[nn.Module, torch.jit.ScriptModule],
    batch_size: int,
    context_length: int = 10,
    device: Optional[torch.device] = None,
    num_warmup: int = 20,
    num_runs: int = 100,
) -> Dict[str, float]:
    """
    Benchmark serving latency distribution and throughput for a model.

    Parameters
    ----------
    model : nn.Module or torch.jit.ScriptModule
        Model to benchmark.
    batch_size : int
        Batch size.
    context_length : int, default=10
        Context length.
    device : torch.device, optional
        Target device.
    num_warmup : int, default=20
        Number of warm-up iterations.
    num_runs : int, default=100
        Number of timed measurement iterations.

    Returns
    -------
    metrics : dict
        Contains throughput (samples/sec), mean_latency_ms, p50_latency_ms,
        p95_latency_ms, p99_latency_ms, min_latency_ms, max_latency_ms.
    """
    if device is None:
        try:
            device = next(model.parameters()).device
        except (StopIteration, AttributeError):
            device = torch.device("cpu")

    model.eval()
    vocab_size = getattr(model, "input_size", 30)

    # Pre-allocate test input tensor
    X = torch.randint(0, vocab_size, (batch_size, context_length), dtype=torch.long, device=device)

    # Warm-up iterations
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(X)
        synchronize_device(device)

    # Timed runs
    latencies_ms: List[float] = []
    with torch.no_grad():
        for _ in range(num_runs):
            synchronize_device(device)
            t_start = time.perf_counter()
            _ = model(X)
            synchronize_device(device)
            t_end = time.perf_counter()
            latencies_ms.append((t_end - t_start) * 1000.0)

    total_time_sec = sum(latencies_ms) / 1000.0
    total_samples = batch_size * num_runs
    throughput = total_samples / total_time_sec if total_time_sec > 0 else 0.0

    return {
        "batch_size": batch_size,
        "num_runs": num_runs,
        "total_time_sec": float(total_time_sec),
        "throughput_samples_per_sec": float(throughput),
        "mean_latency_ms": float(np.mean(latencies_ms)),
        "p50_latency_ms": float(np.percentile(latencies_ms, 50)),
        "p95_latency_ms": float(np.percentile(latencies_ms, 95)),
        "p99_latency_ms": float(np.percentile(latencies_ms, 99)),
        "min_latency_ms": float(np.min(latencies_ms)),
        "max_latency_ms": float(np.max(latencies_ms)),
    }


def save_traced_model(traced_model: torch.jit.ScriptModule, filepath: str):
    """Save a TorchScript traced model to disk."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    traced_model.save(filepath)


def load_traced_model(filepath: str, device: Optional[torch.device] = None) -> torch.jit.ScriptModule:
    """Load a TorchScript traced model from disk."""
    if device is None:
        device = select_serving_device()
    return torch.jit.load(filepath, map_location=device)
