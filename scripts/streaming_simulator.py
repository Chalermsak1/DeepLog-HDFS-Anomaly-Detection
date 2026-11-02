#!/usr/bin/env python3
"""
DeepLog Real-Time Streaming Ingestion & Anomaly Alerting Simulator
==================================================================
Simulates a live multi-node Hadoop cluster streaming log pipeline:
- Interleaves events across concurrent normal and abnormal blocks
- Maintains per-block sliding context buffers via SessionStreamAggregator
- Executes rolling sub-millisecond DeepLog inference (Native or TorchScript JIT)
- Flags anomalous log transitions and emits real-time alerts
- Measures throughput, queue latency, inference latency, and detection metrics
"""

import os
import sys
import time
import json
import argparse
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
import torch
import numpy as np

# Ensure src/ is on python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from deeplog.deeplog import DeepLog
from deeplog.preprocessor import Preprocessor
from deeplog.production import trace_deeplog, select_serving_device
from deeplog.streaming import SessionStreamAggregator, EventAlert, SessionSummary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simulate real-time streaming log ingestion and DeepLog anomaly detection."
    )
    parser.add_argument(
        "--speedup",
        type=float,
        default=50.0,
        help="Playback speed multiplier (default: 50.0). Set to 0 or negative for maximum throughput.",
    )
    parser.add_argument(
        "--max_events",
        type=int,
        default=3000,
        help="Maximum number of log events to ingest (default: 3000, 0 for all).",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Top-K prediction candidates threshold (parameter g, default: 5).",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="deeplog_model_improved_v1.pt",
        help="Path to DeepLog model checkpoint.",
    )
    parser.add_argument(
        "--use_jit",
        action="store_true",
        default=True,
        help="Use TorchScript JIT tracing for sub-millisecond scoring (default: True).",
    )
    parser.add_argument(
        "--no_jit",
        dest="use_jit",
        action="store_false",
        help="Disable TorchScript JIT and use Native PyTorch.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run inference ('cpu', 'mps', 'cuda', 'auto'). Default: 'cpu' for sub-millisecond latency.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="examples/data",
        help="Path to directory containing HDFS dataset files.",
    )
    parser.add_argument(
        "--num_normal_blocks",
        type=int,
        default=120,
        help="Number of normal HDFS blocks to sample for simulation (default: 120).",
    )
    parser.add_argument(
        "--num_abnormal_blocks",
        type=int,
        default=30,
        help="Number of abnormal HDFS blocks to sample for simulation (default: 30).",
    )
    parser.add_argument(
        "--session_timeout",
        type=float,
        default=15.0,
        help="Session inactivity timeout threshold in simulation seconds (default: 15.0).",
    )
    parser.add_argument(
        "--context_length",
        type=int,
        default=10,
        help="DeepLog context sequence length (default: 10).",
    )
    parser.add_argument(
        "--output_report",
        type=str,
        default="results/metrics/streaming_simulation.json",
        help="Path to save streaming evaluation results JSON.",
    )
    parser.add_argument(
        "--verbose_alerts",
        action="store_true",
        default=True,
        help="Print real-time anomaly alerts to stdout.",
    )
    return parser.parse_args()


def load_dataset_blocks(filepath: str, max_blocks: int) -> List[List[int]]:
    """Read space-separated event ID sequences from an HDFS data split."""
    blocks = []
    if not os.path.exists(filepath):
        print(f"⚠️ Warning: File '{filepath}' not found!")
        return blocks

    with open(filepath, "r") as f:
        for idx, line in enumerate(f):
            if max_blocks > 0 and idx >= max_blocks:
                break
            tokens = line.strip().split()
            if tokens:
                try:
                    events = [int(t) for t in tokens]
                    blocks.append(events)
                except ValueError:
                    pass
    return blocks


def generate_interleaved_stream(
    normal_blocks: List[List[int]],
    abnormal_blocks: List[List[int]],
    jitter_std: float = 0.005,
    seed: int = 42,
) -> Tuple[List[Tuple[float, str, int, bool]], Dict[str, bool]]:
    """
    Generate an interleaved, timestamped stream of events across multiple concurrent blocks
    to emulate distributed multi-node Hadoop cluster ingestion.

    Returns
    -------
    stream_events : list of (timestamp, block_id, raw_event, is_abnormal)
    ground_truth_blocks : dict of block_id -> bool (is_abnormal)
    """
    rng = np.random.RandomState(seed)
    stream = []
    ground_truth = {}

    all_blocks = []
    for i, events in enumerate(normal_blocks):
        block_id = f"blk_norm_{i:04d}"
        all_blocks.append((block_id, events, False))
        ground_truth[block_id] = False

    for i, events in enumerate(abnormal_blocks):
        block_id = f"blk_anom_{i:04d}"
        all_blocks.append((block_id, events, True))
        ground_truth[block_id] = True

    # Assign synthetic timeline to each block
    for block_id, events, is_abn in all_blocks:
        # Randomized block start time in [0, 8.0] seconds
        t_curr = rng.uniform(0.0, 8.0)
        for e in events:
            # Poisson/Exponential inter-arrival delay (mean 80ms)
            dt = rng.exponential(scale=0.08) + 0.01
            t_curr += dt
            # Minor network transit jitter
            t_ingest = t_curr + rng.normal(0.0, jitter_std)
            stream.append((max(0.0, t_ingest), block_id, e, is_abn))

    # Sort interleaved events by simulation ingestion timestamp
    stream.sort(key=lambda x: x[0])
    return stream, ground_truth


def main():
    args = parse_args()

    print("=" * 95)
    print("  🌊 DEEPLOG REAL-TIME STREAMING INGESTION & ALERTING SIMULATOR")
    print("=" * 95)
    print(f"Model Checkpoint     : {args.model_path}")
    print(f"Inference Engine     : {'TorchScript JIT (Optimized)' if args.use_jit else 'Native PyTorch'}")
    print(f"Top-K Threshold (g)  : {args.top_k}")
    print(f"Speedup Multiplier   : {args.speedup}x" if args.speedup > 0 else "Speedup Multiplier   : MAX (Unthrottled)")
    print(f"Max Stream Events    : {args.max_events if args.max_events > 0 else 'ALL'}")
    print(f"Normal Blocks        : {args.num_normal_blocks}")
    print(f"Abnormal Blocks      : {args.num_abnormal_blocks}")
    print(f"Session Timeout      : {args.session_timeout}s")
    print("-" * 95)

    # 1. Device Selection
    device = select_serving_device(None if args.device == "auto" else args.device)
    print(f"⚡ Hardware Target    : {device}")

    # 2. Vocabulary & Canonical Mapping
    train_file = os.path.join(args.data_dir, "hdfs_train")
    if not os.path.exists(train_file):
        print(f"❌ Error: Training data file '{train_file}' not found!")
        sys.exit(1)

    print(f"📦 Extracting Canonical Vocabulary from '{train_file}'...")
    preprocessor = Preprocessor(length=args.context_length, timeout=float("inf"))
    _, _, _, train_mapping = preprocessor.text(train_file, verbose=False)
    print(f"✅ Canonical Vocabulary Loaded ({len(train_mapping)} events mapped, NO_EVENT={train_mapping.get(len(train_mapping)-1)})")

    # 3. Model Loading & Tracing
    if not os.path.exists(args.model_path):
        print(f"❌ Error: Model checkpoint '{args.model_path}' not found!")
        sys.exit(1)

    print(f"📦 Loading Model: {args.model_path}...")
    native_model = DeepLog.load(args.model_path, device=device).to(device)
    native_model.eval()

    if args.use_jit:
        print("⚙️ Tracing DeepLog with TorchScript JIT for sub-millisecond serving...")
        dummy_input = torch.randint(
            0, native_model.input_size, (1, args.context_length), dtype=torch.long, device=device
        )
        serving_model = trace_deeplog(
            native_model,
            example_input=dummy_input,
            device=device,
            context_length=args.context_length,
            optimize_for_inference=True,
        )
        print("✅ TorchScript JIT Model ready.")
    else:
        serving_model = native_model
        print("ℹ️ Using Native PyTorch Model.")

    # 4. Load Stream Datasets
    norm_file = os.path.join(args.data_dir, "hdfs_test_normal")
    anom_file = os.path.join(args.data_dir, "hdfs_test_abnormal")

    print(f"\n📂 Sampling HDFS Log Blocks...")
    normal_blocks = load_dataset_blocks(norm_file, args.num_normal_blocks)
    abnormal_blocks = load_dataset_blocks(anom_file, args.num_abnormal_blocks)
    print(f"   • Loaded {len(normal_blocks)} normal blocks ({sum(len(b) for b in normal_blocks):,} events)")
    print(f"   • Loaded {len(abnormal_blocks)} abnormal blocks ({sum(len(b) for b in abnormal_blocks):,} events)")

    # 5. Synthesize Interleaved Real-Time Stream
    raw_stream, ground_truth = generate_interleaved_stream(normal_blocks, abnormal_blocks)
    if args.max_events > 0:
        raw_stream = raw_stream[:args.max_events]

    total_stream_events = len(raw_stream)
    print(f"🌪️ Generated Interleaved Stream: {total_stream_events:,} events spanning {len(ground_truth)} blocks")
    print(f"   Stream Time Horizon: {raw_stream[0][0]:.2f}s to {raw_stream[-1][0]:.2f}s (Span: {raw_stream[-1][0] - raw_stream[0][0]:.2f}s)")

    # 6. Initialize SessionStreamAggregator
    aggregator = SessionStreamAggregator(
        model=serving_model,
        context_length=args.context_length,
        top_k=args.top_k,
        mapping=train_mapping,
        device=device,
        session_timeout=args.session_timeout,
        vocab_size=native_model.input_size,
    )

    print("\n" + "=" * 95)
    print("  🚀 STARTING REAL-TIME STREAMING INGESTION")
    print("=" * 95)

    evicted_session_summaries: List[SessionSummary] = []
    queue_latencies_ms: List[float] = []
    inference_latencies_ms: List[float] = []

    sim_start_wall = time.time()
    last_log_time = sim_start_wall
    last_sim_time = raw_stream[0][0]

    for idx, (sim_ts, block_id, raw_event, is_abnormal) in enumerate(raw_stream):
        # Simulated stream time-pacing (if speedup > 0)
        if args.speedup > 0:
            sim_dt = sim_ts - last_sim_time
            if sim_dt > 0:
                sleep_sec = min(0.04, sim_dt / args.speedup)
                if sleep_sec > 0.0005:
                    time.sleep(sleep_sec)
        last_sim_time = sim_ts

        # Ingestion timestamp
        arrival_t = time.perf_counter()

        # Ingest and rolling scoring
        result = aggregator.process_event(
            block_id=block_id,
            raw_event=raw_event,
            timestamp=sim_ts,
            arrival_time=arrival_t,
        )

        queue_latencies_ms.append(result.queue_latency_ms)
        inference_latencies_ms.append(result.latency_ms)

        # Print alert if anomaly detected
        if result.is_anomaly and args.verbose_alerts:
            alert = result.alert
            topk_cands = [f"E{e}" for e in alert.topk_raw_predictions[:args.top_k]]
            cand_str = ", ".join(topk_cands)
            unseen_tag = " [UNSEEN]" if alert.is_unseen else ""
            print(
                f"🚨 [ALERT #{aggregator.total_anomalies_detected:>3}] "
                f"t={sim_ts:>6.2f}s | Block={block_id:<13} | Step={result.step_number:>2} | "
                f"Actual=E{raw_event:<2}{unseen_tag} ∉ [{cand_str}] | "
                f"Conf={alert.confidence * 100:>5.1f}% | Latency={result.latency_ms:.3f}ms"
            )

        # Periodic session timeout cleanup
        if (idx + 1) % 50 == 0:
            evicted = aggregator.cleanup_inactive_sessions(current_time=sim_ts)
            evicted_session_summaries.extend(evicted)

        # Periodic streaming progress log
        curr_wall = time.time()
        if (curr_wall - last_log_time >= 1.0) or ((idx + 1) % 500 == 0) or (idx + 1 == total_stream_events):
            last_log_time = curr_wall
            stats = aggregator.get_metrics()
            print(
                f"⏱️ [STREAM PROGRESS] Events: {idx + 1:>5,}/{total_stream_events:,} | "
                f"Throughput: {stats['throughput_eps']:>8,.1f} eps | "
                f"Inf Latency: {stats['mean_inference_latency_ms']:>6.3f}ms | "
                f"Active Sessions: {stats['active_sessions']:>3} | "
                f"Alerts: {stats['total_anomalies_detected']:>3}"
            )

    # 7. Final Session Flush
    remaining_evicted = aggregator.cleanup_inactive_sessions(current_time=float("inf"))
    evicted_session_summaries.extend(remaining_evicted)

    total_sim_wall_time = time.time() - sim_start_wall
    overall_throughput = total_stream_events / total_sim_wall_time if total_sim_wall_time > 0 else 0.0

    # 8. Block-Level Evaluation Metrics
    # Compile block-level classification results
    predicted_block_anomaly = {}
    for summary in evicted_session_summaries:
        predicted_block_anomaly[summary.block_id] = summary.is_anomaly_block

    # Any block still in sessions dict
    for block_id, s in aggregator.sessions.items():
        predicted_block_anomaly[block_id] = s["is_anomaly"]

    # Compute block-level confusion matrix
    tp = fp = tn = fn = 0
    for b_id, true_anom in ground_truth.items():
        if b_id in predicted_block_anomaly:
            pred_anom = predicted_block_anomaly[b_id]
            if true_anom and pred_anom:
                tp += 1
            elif not true_anom and pred_anom:
                fp += 1
            elif not true_anom and not pred_anom:
                tn += 1
            elif true_anom and not pred_anom:
                fn += 1

    total_blocks_eval = tp + fp + tn + fn
    accuracy = (tp + tn) / total_blocks_eval if total_blocks_eval > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    print("\n" + "=" * 95)
    print("  📊 REAL-TIME STREAMING SIMULATION REPORT & SUMMARY")
    print("=" * 95)
    print(f"Total Stream Events Ingested : {total_stream_events:,}")
    print(f"Total Unique Blocks Processed: {total_blocks_eval:,} (Normal: {tn + fp}, Abnormal: {tp + fn})")
    print(f"Total Wall Clock Duration    : {total_sim_wall_time:.2f} seconds")
    print(f"End-to-End Stream Throughput : {overall_throughput:,.1f} events/sec")
    print("-" * 95)
    print("⚡ LATENCY PROFILING (Real-Time Rolling Inference):")
    print(f"   • Mean Inference Latency   : {np.mean(inference_latencies_ms):.4f} ms")
    print(f"   • P50  Inference Latency   : {np.percentile(inference_latencies_ms, 50):.4f} ms")
    print(f"   • P95  Inference Latency   : {np.percentile(inference_latencies_ms, 95):.4f} ms")
    print(f"   • P99  Inference Latency   : {np.percentile(inference_latencies_ms, 99):.4f} ms")
    print(f"   • Sub-millisecond Scoring  : {'✅ YES (< 1.0 ms)' if np.percentile(inference_latencies_ms, 99) < 1.0 else '⚠️ Slightly above 1ms'}")
    print(f"   • Mean Ingestion Queue Lat : {np.mean(queue_latencies_ms):.4f} ms")
    print("-" * 95)
    print(f"🎯 BLOCK-LEVEL ANOMALY DETECTION METRICS (Top-K = {args.top_k}):")
    print(f"   • True Positives  (TP)     : {tp}")
    print(f"   • False Positives (FP)     : {fp}")
    print(f"   • True Negatives  (TN)     : {tn}")
    print(f"   • False Negatives (FN)     : {fn}")
    print(f"   • Accuracy                 : {accuracy * 100:.2f}%")
    print(f"   • Precision                : {precision * 100:.2f}%")
    print(f"   • Recall                   : {recall * 100:.2f}%")
    print(f"   • F1-Score                 : {f1 * 100:.2f}%")
    print(f"   • False Positive Rate (FPR): {fpr * 100:.2f}%")
    print("=" * 95)

    # 9. Export structured simulation metrics
    report = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "model_path": args.model_path,
            "use_jit": args.use_jit,
            "device": str(device),
            "top_k": args.top_k,
            "speedup": args.speedup,
            "max_events": args.max_events,
            "num_normal_blocks": args.num_normal_blocks,
            "num_abnormal_blocks": args.num_abnormal_blocks,
            "session_timeout": args.session_timeout,
        },
        "telemetry": {
            "total_events": total_stream_events,
            "total_blocks": total_blocks_eval,
            "wall_clock_sec": float(total_sim_wall_time),
            "throughput_eps": float(overall_throughput),
            "alerts_raised": aggregator.total_anomalies_detected,
            "inference_latency": {
                "mean_ms": float(np.mean(inference_latencies_ms)),
                "p50_ms": float(np.percentile(inference_latencies_ms, 50)),
                "p95_ms": float(np.percentile(inference_latencies_ms, 95)),
                "p99_ms": float(np.percentile(inference_latencies_ms, 99)),
                "sub_millisecond": bool(np.percentile(inference_latencies_ms, 99) < 1.0),
            },
            "queue_latency": {
                "mean_ms": float(np.mean(queue_latencies_ms)),
                "p50_ms": float(np.percentile(queue_latencies_ms, 50)),
                "p95_ms": float(np.percentile(queue_latencies_ms, 95)),
                "p99_ms": float(np.percentile(queue_latencies_ms, 99)),
            },
        },
        "detection_performance": {
            "top_k": args.top_k,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "fpr": float(fpr),
        },
        "alerts_sample": [a.to_dict() for a in aggregator.alerts[:50]],
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output_report)), exist_ok=True)
    with open(args.output_report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"📁 Simulation metrics successfully exported to: {args.output_report}\n")


if __name__ == "__main__":
    main()
