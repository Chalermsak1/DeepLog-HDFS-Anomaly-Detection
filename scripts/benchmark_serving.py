#!/usr/bin/env python3
"""
DeepLog Serving Benchmark & TorchScript JIT Profiler
====================================================
Profiles Native PyTorch vs TorchScript JIT serving performance across
batch sizes [1, 32, 256, 1024, 32768] on CPU and Apple Silicon MPS devices.

Measures:
- Numerical output parity verification (torch.allclose with atol=1e-5)
- Throughput (samples/sec)
- Mean, P50, P95, P99 Latency (ms)
- TorchScript JIT compilation speedup factors
- Exports metrics to results/metrics/serving_benchmark.json
"""

import os
import sys
import time
import json
import argparse
from datetime import datetime
from typing import Dict, Any, List, Optional
import torch
import numpy as np

# Ensure src/ is on python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from deeplog.deeplog import DeepLog
from deeplog.production import (
    trace_deeplog,
    verify_numerical_parity,
    synchronize_device,
    save_traced_model,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark Native PyTorch vs TorchScript JIT serving performance."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="deeplog_model_improved_v1.pt",
        help="Path to trained model checkpoint (.pt)",
    )
    parser.add_argument(
        "--batch_sizes",
        type=int,
        nargs="+",
        default=[1, 32, 256, 1024, 32768],
        help="Batch sizes to benchmark",
    )
    parser.add_argument(
        "--devices",
        type=str,
        nargs="+",
        default=["cpu", "mps"] if torch.backends.mps.is_available() else ["cpu"],
        help="Devices to benchmark ('cpu', 'mps', 'cuda')",
    )
    parser.add_argument(
        "--context_length",
        type=int,
        default=10,
        help="Sequence context length (default: 10)",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="results/metrics/serving_benchmark.json",
        help="Path to output JSON metrics file",
    )
    parser.add_argument(
        "--save_traced",
        type=str,
        default="results/deeplog_model_traced.pt",
        help="Path to save traced TorchScript model",
    )
    parser.add_argument(
        "--runs_batch_1",
        type=int,
        default=500,
        help="Number of iterations for batch size 1",
    )
    parser.add_argument(
        "--runs_batch_small",
        type=int,
        default=150,
        help="Number of iterations for small batches (32, 256)",
    )
    parser.add_argument(
        "--runs_batch_medium",
        type=int,
        default=50,
        help="Number of iterations for batch 1024",
    )
    parser.add_argument(
        "--runs_batch_large",
        type=int,
        default=15,
        help="Number of iterations for batch 32768",
    )
    return parser.parse_args()


def get_iterations_for_batch(batch_size: int, args) -> Tuple[int, int]:
    """Return (num_warmup, num_runs) scaled according to batch size."""
    if batch_size == 1:
        return 30, args.runs_batch_1
    elif batch_size <= 256:
        return 20, args.runs_batch_small
    elif batch_size <= 1024:
        return 15, args.runs_batch_medium
    else:
        return 5, args.runs_batch_large


def run_benchmark_for_model(
    model: torch.nn.Module,
    batch_size: int,
    context_length: int,
    device: torch.device,
    num_warmup: int,
    num_runs: int,
) -> Dict[str, Any]:
    """Execute high-precision latency profiling with hardware synchronization."""
    model.eval()
    vocab_size = getattr(model, "input_size", 30)

    # Pre-allocate test tensor on device
    X = torch.randint(0, vocab_size, (batch_size, context_length), dtype=torch.long, device=device)

    # Warm-up phase
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(X)
        synchronize_device(device)

    # Timing phase
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
        "num_warmup": num_warmup,
        "total_time_sec": float(total_time_sec),
        "throughput_samples_per_sec": float(throughput),
        "mean_latency_ms": float(np.mean(latencies_ms)),
        "p50_latency_ms": float(np.percentile(latencies_ms, 50)),
        "p95_latency_ms": float(np.percentile(latencies_ms, 95)),
        "p99_latency_ms": float(np.percentile(latencies_ms, 99)),
        "min_latency_ms": float(np.min(latencies_ms)),
        "max_latency_ms": float(np.max(latencies_ms)),
    }


def format_table_header(device_str: str) -> str:
    lines = [
        "",
        "=" * 105,
        f"  📊 BENCHMARK RESULTS: {device_str.upper()}",
        "=" * 105,
        f"{'Mode':<12} | {'Batch':<6} | {'Throughput (samples/s)':<24} | {'Mean (ms)':<10} | {'P50 (ms)':<9} | {'P95 (ms)':<9} | {'P99 (ms)':<9} | {'Speedup':<8}",
        "-" * 105,
    ]
    return "\n".join(lines)


def format_table_row(
    mode: str,
    batch_size: int,
    metrics: Dict[str, Any],
    speedup: Optional[float] = None,
) -> str:
    tput = f"{metrics['throughput_samples_per_sec']:>18,.1f} s/s"
    mean_ms = f"{metrics['mean_latency_ms']:>8.3f}"
    p50_ms = f"{metrics['p50_latency_ms']:>7.3f}"
    p95_ms = f"{metrics['p95_latency_ms']:>7.3f}"
    p99_ms = f"{metrics['p99_latency_ms']:>7.3f}"
    speedup_str = f"{speedup:>6.2f}x" if speedup is not None else "   -   "
    return f"{mode:<12} | {batch_size:<6} | {tput:<24} | {mean_ms:<10} | {p50_ms:<9} | {p95_ms:<9} | {p99_ms:<9} | {speedup_str:<8}"


def main():
    args = parse_args()

    print("=" * 105)
    print("  🚀 DEEPLOG PRODUCTION SERVING BENCHMARK & JIT PROFILER")
    print("=" * 105)
    print(f"Model Checkpoint   : {args.model_path}")
    print(f"Batch Sizes        : {args.batch_sizes}")
    print(f"Devices Target     : {args.devices}")
    print(f"Context Length     : {args.context_length}")
    print(f"Output Metrics File: {args.output_json}")
    print("-" * 105)

    if not os.path.exists(args.model_path):
        print(f"❌ Error: Model checkpoint '{args.model_path}' not found!")
        sys.exit(1)

    benchmark_data: Dict[str, Any] = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "model_path": args.model_path,
            "context_length": args.context_length,
            "batch_sizes": args.batch_sizes,
            "devices": args.devices,
            "torch_version": torch.__version__,
            "mps_available": torch.backends.mps.is_available(),
            "cuda_available": torch.cuda.is_available(),
        },
        "numerical_parity": {},
        "devices": {},
        "summary": {},
    }

    saved_traced_model = False

    for dev_name in args.devices:
        dev_name_clean = dev_name.lower().strip()
        if dev_name_clean == "mps" and not torch.backends.mps.is_available():
            print(f"⚠️ MPS is not available on this system. Skipping MPS.")
            continue
        elif dev_name_clean == "cuda" and not torch.cuda.is_available():
            print(f"⚠️ CUDA is not available on this system. Skipping CUDA.")
            continue

        device = torch.device(dev_name_clean)
        print(f"\n🔍 Processing Device: {dev_name_clean.upper()} ({device})")

        # 1. Load native model onto device
        native_model = DeepLog.load(args.model_path, device=device).to(device)
        native_model.eval()

        # 2. Trace model with TorchScript JIT
        print(f"   ⚙️ Tracing DeepLog with torch.jit.trace on {dev_name_clean.upper()}...")
        dummy_input = torch.randint(
            0, native_model.input_size, (1, args.context_length), dtype=torch.long, device=device
        )
        jit_model = trace_deeplog(
            native_model,
            example_input=dummy_input,
            device=device,
            context_length=args.context_length,
            optimize_for_inference=True,
        )

        # Save traced model once
        if not saved_traced_model and args.save_traced:
            save_traced_model(jit_model, args.save_traced)
            saved_traced_model = True
            print(f"   💾 Saved TorchScript model to: {args.save_traced}")

        # 3. Verify Numerical Parity across batch sizes
        print(f"   🧪 Verifying numerical parity (torch.allclose, atol=1e-5)...")
        parity_passed, max_abs_diff, parity_details = verify_numerical_parity(
            native_model=native_model,
            jit_model=jit_model,
            batch_sizes=args.batch_sizes,
            context_length=args.context_length,
            atol=1e-5,
            device=device,
        )

        benchmark_data["numerical_parity"][dev_name_clean] = {
            "allclose_verified": parity_passed,
            "max_abs_diff": max_abs_diff,
            "atol_threshold": 1e-5,
            "batch_details": parity_details,
        }

        if parity_passed:
            print(f"   ✅ Numerical parity VERIFIED on {dev_name_clean.upper()} (Max diff = {max_abs_diff:.2e} <= 1e-5)")
        else:
            print(f"   ⚠️ WARNING: Parity check failed on {dev_name_clean.upper()} (Max diff = {max_abs_diff:.2e})")

        # 4. Run Benchmarks
        print(f"   ⏱️ Profiling Native PyTorch vs TorchScript JIT across batch sizes...")
        print(format_table_header(dev_name_clean))

        device_results = {
            "native": {},
            "jit": {},
            "comparison": {},
        }

        for bs in args.batch_sizes:
            warmup, runs = get_iterations_for_batch(bs, args)

            # Benchmark Native
            native_metrics = run_benchmark_for_model(
                model=native_model,
                batch_size=bs,
                context_length=args.context_length,
                device=device,
                num_warmup=warmup,
                num_runs=runs,
            )
            device_results["native"][f"batch_{bs}"] = native_metrics

            # Benchmark TorchScript JIT
            jit_metrics = run_benchmark_for_model(
                model=jit_model,
                batch_size=bs,
                context_length=args.context_length,
                device=device,
                num_warmup=warmup,
                num_runs=runs,
            )
            device_results["jit"][f"batch_{bs}"] = jit_metrics

            # Calculate speedup
            latency_speedup = (
                native_metrics["mean_latency_ms"] / jit_metrics["mean_latency_ms"]
                if jit_metrics["mean_latency_ms"] > 0
                else 1.0
            )
            tput_speedup = (
                jit_metrics["throughput_samples_per_sec"] / native_metrics["throughput_samples_per_sec"]
                if native_metrics["throughput_samples_per_sec"] > 0
                else 1.0
            )

            device_results["comparison"][f"batch_{bs}"] = {
                "batch_size": bs,
                "latency_speedup_x": float(latency_speedup),
                "throughput_speedup_x": float(tput_speedup),
                "sub_millisecond_jit": bool(jit_metrics["p99_latency_ms"] < 1.0),
            }

            # Print table rows
            print(format_table_row("Native PyTorch", bs, native_metrics, speedup=None))
            print(format_table_row("TorchScript JIT", bs, jit_metrics, speedup=latency_speedup))
            print("-" * 105)

        benchmark_data["devices"][dev_name_clean] = device_results

    # Summary highlights
    print("\n" + "=" * 105)
    print("  🏆 BENCHMARK SUMMARY & HIGHLIGHTS")
    print("=" * 105)

    summary_records = {}
    for dev, d_res in benchmark_data["devices"].items():
        b1_jit = d_res["jit"]["batch_1"]
        b1_comp = d_res["comparison"]["batch_1"]
        b_max_bs = max(args.batch_sizes)
        b_max_jit = d_res["jit"][f"batch_{b_max_bs}"]

        sub_ms = b1_jit["mean_latency_ms"] < 1.0
        summary_records[dev] = {
            "batch_1_mean_latency_ms": b1_jit["mean_latency_ms"],
            "batch_1_p99_latency_ms": b1_jit["p99_latency_ms"],
            "batch_1_sub_millisecond": sub_ms,
            "batch_1_speedup_x": b1_comp["latency_speedup_x"],
            f"peak_throughput_batch_{b_max_bs}": b_max_jit["throughput_samples_per_sec"],
        }

        print(f"Device: {dev.upper()}")
        print(f"   • Batch 1 (Real-Time Single Event) : Mean = {b1_jit['mean_latency_ms']:.4f} ms | P99 = {b1_jit['p99_latency_ms']:.4f} ms | Sub-millisecond: {'✅ YES' if sub_ms else '❌ NO'}")
        print(f"   • Batch 1 JIT Speedup              : {b1_comp['latency_speedup_x']:.2f}x faster than Native")
        print(f"   • Peak Throughput (Batch {b_max_bs:<5})    : {b_max_jit['throughput_samples_per_sec']:,.1f} samples/sec")
        print("-" * 105)

    benchmark_data["summary"] = summary_records

    # Export to JSON
    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(benchmark_data, f, indent=2)

    print(f"\n📁 Benchmark metrics successfully exported to: {args.output_json}")
    print("=" * 105)


if __name__ == "__main__":
    main()
