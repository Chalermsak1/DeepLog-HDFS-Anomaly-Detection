#!/usr/bin/env python3
"""
DeepLog Advanced Anomaly Scoring & Adaptive Thresholding Benchmark
==================================================================
Evaluates advanced block anomaly scoring decision algorithms across all 570,204
HDFS test blocks using trained DeepLog checkpoints.

Strategies evaluated:
1. Baseline: Any violation (g=4, g=5)
2. Consecutive Violations: min_consecutive=2 and min_consecutive=3 (g=4, g=5)
3. Anomaly Ratio Thresholds: ratio in [0.05, 0.10, 0.15] (g=4, g=5)
4. Cumulative NLL Surprise Scoring: Mean negative log-likelihood per block
"""

import os
import sys
import time
import json
import argparse
from typing import Dict, Any, List, Tuple
import numpy as np
import torch
import torch.nn.functional as F

# Ensure src/ is on python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from deeplog import DeepLog
from deeplog.preprocessor import Preprocessor
from deeplog.scoring import (
    evaluate_consecutive_violations,
    evaluate_anomaly_ratio,
    compute_anomaly_ratios,
    compute_cumulative_nll_scores,
    evaluate_nll_threshold,
    compute_classification_metrics,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark Advanced Anomaly Scoring Engine on HDFS Test Data."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="deeplog_model_improved_v1.pt",
        help="Path to trained model checkpoint (.pt)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="examples/data",
        help="Directory containing HDFS data files",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32768,
        help="Batch size for vectorized inference",
    )
    parser.add_argument(
        "--context_length",
        type=int,
        default=10,
        help="Context sequence length (default: 10)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="results/metrics/advanced_scoring_metrics.json",
        help="Path to output metrics JSON file",
    )
    parser.add_argument(
        "--max_blocks",
        type=int,
        default=None,
        help="Optional maximum number of blocks per split to evaluate (for fast testing)",
    )
    return parser.parse_args()


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def run_vectorized_inference(
    model: DeepLog,
    X: torch.Tensor,
    y: torch.Tensor,
    max_k: int,
    device: torch.device,
    batch_size: int = 32768,
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Executes batched forward inference on device, extracting top-k candidate indices
    and on-the-fly event Negative Log-Likelihood (NLL) values.
    """
    model.eval()
    total_samples = X.shape[0]
    preds_topk = []
    nll_events = []
    start_time = time.time()

    for i in range(0, total_samples, batch_size):
        xb = X[i : i + batch_size].to(device)
        yb = y[i : i + batch_size].to(device)

        log_probs = model(xb)
        _, topk_idx = log_probs.topk(max_k, dim=-1)
        preds_topk.append(topk_idx.cpu())

        # Event-level Negative Log-Likelihood (NLL)
        nll_chunk = F.nll_loss(log_probs, yb.to(torch.int64), reduction="none")
        nll_events.append(nll_chunk.cpu())

        if device.type == "mps" and (i % (batch_size * 10) == 0):
            torch.mps.empty_cache()

    elapsed = time.time() - start_time
    throughput = total_samples / elapsed if elapsed > 0 else 0
    print(
        f"   ⏱️  Inference completed: {total_samples:,} samples in {elapsed:.2f}s "
        f"({throughput:,.0f} samples/sec)"
    )

    all_preds = torch.cat(preds_topk, dim=0)
    all_nlls = torch.cat(nll_events, dim=0).numpy().astype(np.float64)
    return all_preds, all_nlls


def main():
    args = parse_args()
    device = select_device()

    print("=" * 90)
    print("  🚀 DEEPLOG ADVANCED ANOMALY SCORING & ADAPTIVE THRESHOLDING BENCHMARK")
    print("=" * 90)
    print(f"⚡ Device Accelerator : {device}")

    # Verify model checkpoint existence
    model_path = args.model_path
    if not os.path.exists(model_path):
        for fallback in ["deeplog_model_improved_v1.pt", "deeplog_model_full.pt"]:
            if os.path.exists(fallback):
                print(f"⚠️ Checkpoint '{model_path}' not found, falling back to '{fallback}'")
                model_path = fallback
                break
        else:
            print(f"❌ Error: Model checkpoint '{model_path}' not found!")
            sys.exit(1)

    print(f"📦 Loading Model: '{model_path}'...")
    deeplog = DeepLog.load(model_path, device=device).to(device)
    deeplog.eval()
    print(
        f"✅ DeepLog Architecture: num_layers={deeplog.num_layers}, "
        f"hidden_size={deeplog.hidden_size}, vocab_size={deeplog.output_size}"
    )

    # Load canonical vocabulary mapping from training set
    preprocessor = Preprocessor(length=args.context_length, timeout=float("inf"))
    train_file = os.path.join(args.data_dir, "hdfs_train")
    test_norm_file = os.path.join(args.data_dir, "hdfs_test_normal")
    test_anom_file = os.path.join(args.data_dir, "hdfs_test_abnormal")

    print("\n📦 Loading Training Vocabulary Mapping...")
    _, _, _, train_mapping = preprocessor.text(train_file, verbose=False)
    print(f"✅ Canonical Vocabulary Size: {len(train_mapping)}")

    # Extract block sequence lengths
    print("📦 Reading Block Lengths from Test Files...")
    with open(test_norm_file) as f:
        if args.max_blocks:
            lengths_norm = np.array([len(line.split()) for i, line in enumerate(f) if i < args.max_blocks])
        else:
            lengths_norm = np.array([len(line.split()) for line in f])

    with open(test_anom_file) as f:
        if args.max_blocks:
            lengths_anom = np.array([len(line.split()) for i, line in enumerate(f) if i < args.max_blocks])
        else:
            lengths_anom = np.array([len(line.split()) for line in f])

    print(f"   - Normal Blocks   : {len(lengths_norm):,}")
    print(f"   - Abnormal Blocks : {len(lengths_anom):,}")
    print(f"   - Total Blocks    : {len(lengths_norm) + len(lengths_anom):,}")

    # Load test sequences
    start_load = time.time()
    print("\n📦 Preprocessing Test Sequences...")
    X_test_norm, y_test_norm, _, _ = preprocessor.text(
        test_norm_file, nrows=args.max_blocks, mapping=train_mapping, verbose=False
    )
    X_test_anom, y_test_anom, _, _ = preprocessor.text(
        test_anom_file, nrows=args.max_blocks, mapping=train_mapping, verbose=False
    )
    print(f"✅ Loaded in {time.time() - start_load:.2f}s")
    print(f"   - Normal Events   : {X_test_norm.shape[0]:,}")
    print(f"   - Abnormal Events : {X_test_anom.shape[0]:,}")
    print(f"   - Total Events    : {X_test_norm.shape[0] + X_test_anom.shape[0]:,}")

    # Check length sum integrity
    assert lengths_norm.sum() == X_test_norm.shape[0], "Normal length mismatch"
    assert lengths_anom.sum() == X_test_anom.shape[0], "Abnormal length mismatch"

    # Forward inference
    max_k = 5
    print(f"\n⚡ Running Vectorized Inference (batch_size={args.batch_size:,}, max_k={max_k})...")
    print("   👉 Normal Test Split:")
    y_pred_norm_topk, nll_norm_events = run_vectorized_inference(
        deeplog, X_test_norm, y_test_norm, max_k=max_k, device=device, batch_size=args.batch_size
    )

    print("   👉 Abnormal Test Split:")
    y_pred_anom_topk, nll_anom_events = run_vectorized_inference(
        deeplog, X_test_anom, y_test_anom, max_k=max_k, device=device, batch_size=args.batch_size
    )

    # Cumulative NLL scores per block
    print("\n🧮 Computing Cumulative Mean NLL Surprise Scores per Block...")
    # Pre-computed event NLLs
    block_ids_norm = np.repeat(np.arange(len(lengths_norm), dtype=np.int64), lengths_norm)
    block_ids_anom = np.repeat(np.arange(len(lengths_anom), dtype=np.int64), lengths_anom)

    nll_norm_block_sums = np.bincount(block_ids_norm, weights=nll_norm_events, minlength=len(lengths_norm))
    nll_anom_block_sums = np.bincount(block_ids_anom, weights=nll_anom_events, minlength=len(lengths_anom))

    nll_scores_norm = np.zeros(len(lengths_norm), dtype=np.float64)
    valid_norm = lengths_norm > 0
    nll_scores_norm[valid_norm] = nll_norm_block_sums[valid_norm] / lengths_norm[valid_norm]

    nll_scores_anom = np.zeros(len(lengths_anom), dtype=np.float64)
    valid_anom = lengths_anom > 0
    nll_scores_anom[valid_anom] = nll_anom_block_sums[valid_anom] / lengths_anom[valid_anom]

    all_nll_scores = np.concatenate([nll_scores_norm, nll_scores_anom])
    y_true_blocks = np.concatenate([
        np.zeros(len(lengths_norm), dtype=bool),
        np.ones(len(lengths_anom), dtype=bool),
    ])

    print(f"   - Normal Blocks Mean NLL   : {np.mean(nll_scores_norm):.4f} (std: {np.std(nll_scores_norm):.4f}, median: {np.median(nll_scores_norm):.4f})")
    print(f"   - Abnormal Blocks Mean NLL : {np.mean(nll_scores_anom):.4f} (std: {np.std(nll_scores_anom):.4f}, median: {np.median(nll_scores_anom):.4f})")

    # Define Evaluation Configurations
    results = {}

    # Precompute event anomalies for g=4 and g=5
    anom_events_norm_g = {}
    anom_events_anom_g = {}
    for g in [4, 5]:
        match_norm = (y_test_norm.unsqueeze(1) == y_pred_norm_topk[:, :g]).any(dim=1).numpy()
        anom_events_norm_g[g] = ~match_norm

        match_anom = (y_test_anom.unsqueeze(1) == y_pred_anom_topk[:, :g]).any(dim=1).numpy()
        anom_events_anom_g[g] = ~match_anom

    print("\n🔍 Evaluating Decision Strategies...")

    # Strategy 1: Baseline (Any Violation / min_consecutive=1)
    for g in [5, 4]:
        pred_norm = evaluate_consecutive_violations(
            anom_events_norm_g[g], lengths_norm, min_consecutive=1
        )
        pred_anom = evaluate_consecutive_violations(
            anom_events_anom_g[g], lengths_anom, min_consecutive=1
        )
        y_pred = np.concatenate([pred_norm, pred_anom])
        metrics = compute_classification_metrics(y_true_blocks, y_pred)
        results[f"Baseline (g={g}, Any violation)"] = {
            "strategy": "Baseline (Any violation)",
            "top_k": g,
            "parameter": "min_consecutive=1",
            **metrics,
        }

    # Strategy 2: Consecutive Violations (min_consecutive=2 and 3)
    for g in [5, 4]:
        for min_c in [2, 3]:
            pred_norm = evaluate_consecutive_violations(
                anom_events_norm_g[g], lengths_norm, min_consecutive=min_c
            )
            pred_anom = evaluate_consecutive_violations(
                anom_events_anom_g[g], lengths_anom, min_consecutive=min_c
            )
            y_pred = np.concatenate([pred_norm, pred_anom])
            metrics = compute_classification_metrics(y_true_blocks, y_pred)
            results[f"Consecutive Violations (g={g}, c={min_c})"] = {
                "strategy": "Consecutive Violations",
                "top_k": g,
                "parameter": f"min_consecutive={min_c}",
                **metrics,
            }

    # Strategy 3: Anomaly Ratio Thresholds (0.05, 0.10, 0.15)
    for g in [5, 4]:
        for ratio in [0.05, 0.10, 0.15]:
            pred_norm = evaluate_anomaly_ratio(
                anom_events_norm_g[g], lengths_norm, threshold_ratio=ratio
            )
            pred_anom = evaluate_anomaly_ratio(
                anom_events_anom_g[g], lengths_anom, threshold_ratio=ratio
            )
            y_pred = np.concatenate([pred_norm, pred_anom])
            metrics = compute_classification_metrics(y_true_blocks, y_pred)
            results[f"Anomaly Ratio (g={g}, ratio={ratio:.2f})"] = {
                "strategy": "Anomaly Ratio",
                "top_k": g,
                "parameter": f"ratio={ratio:.2f}",
                **metrics,
            }

    # Strategy 4: Cumulative NLL Thresholds
    # Evaluate representative NLL thresholds
    nll_candidates = [0.30, 0.40, 0.50, 0.60, 0.75, 1.00, 1.50, 2.00]
    for thresh in nll_candidates:
        y_pred = evaluate_nll_threshold(all_nll_scores, threshold=thresh)
        metrics = compute_classification_metrics(y_true_blocks, y_pred)
        results[f"Cumulative NLL (threshold={thresh:.2f})"] = {
            "strategy": "Cumulative NLL",
            "top_k": "N/A (Softmax Prob)",
            "parameter": f"threshold={thresh:.2f}",
            **metrics,
        }

    # Print Formatted Results Benchmark Matrix
    print("\n" + "=" * 115)
    print("  📊 ADVANCED ANOMALY SCORING PERFORMANCE BENCHMARK MATRIX (570,204 BLOCKS)")
    print("=" * 115)
    header = (
        f"{'Strategy & Configuration':<38} | {'Accuracy':<9} | {'Precision':<10} | "
        f"{'Recall':<10} | {'F1-Score':<9} | {'FPR (%)':<9} | {'TP':<7} | {'FP':<7} | {'FN':<6}"
    )
    print(header)
    print("-" * 115)

    for name, m in results.items():
        acc_str = f"{m['accuracy'] * 100:6.2f}%"
        prec_str = f"{m['precision'] * 100:6.2f}%"
        rec_str = f"{m['recall'] * 100:6.2f}%"
        f1_str = f"{m['f1'] * 100:6.2f}%"
        fpr_str = f"{m['fpr'] * 100:6.4f}%"
        print(
            f"{name:<38} | {acc_str:<9} | {prec_str:<10} | {rec_str:<10} | "
            f"{f1_str:<9} | {fpr_str:<9} | {m['tp']:<7,d} | {m['fp']:<7,d} | {m['fn']:<6,d}"
        )
    print("=" * 115)

    # Save to JSON
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Saved full metrics benchmark to: {args.output_file}")


if __name__ == "__main__":
    main()
