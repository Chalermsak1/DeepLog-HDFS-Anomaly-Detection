#!/usr/bin/env python3
"""
DeepLog Evaluation & Benchmark Script
=====================================
Evaluates trained DeepLog model checkpoints on full HDFS test datasets.

Features:
- Pure PyTorch vectorization with torch.no_grad()
- High-throughput chunked batch inference (Apple Silicon MPS / CUDA / CPU)
- Canonical Event ID vocabulary mapping derived from the training set
- Multi-threshold Top-K evaluation (g in [1, 2, 3, 4, 5, 9, 15])
- Block-level aggregation: Block is Anomaly if ANY event is anomalous
- Automated metrics export to results/metrics/*.json
"""

import os
import sys
import time
import json
import argparse
import torch
import numpy as np
from sklearn.metrics import confusion_matrix

# Ensure src/ is on python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from deeplog import DeepLog
from deeplog.preprocessor import Preprocessor


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate DeepLog model performance on HDFS test datasets.")
    parser.add_argument(
        "--model_path",
        type=str,
        default="deeplog_model_improved_v1.pt",
        help="Path to trained model checkpoint (.pt)"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="examples/data",
        help="Directory containing HDFS data files"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5, 9, 15],
        help="Top-k prediction candidate thresholds (parameter g) to evaluate"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32768,
        help="Batch size for vectorized inference"
    )
    parser.add_argument(
        "--context_length",
        type=int,
        default=10,
        help="Context sequence length (default: 10)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/metrics",
        help="Directory to save metric JSON files"
    )
    return parser.parse_args()


def select_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def compute_topk_accuracies(y_true, y_pred_topk, k_list=[1, 3, 5, 9, 15], chunk_size=100000):
    """
    Computes Top-K Next-Event Accuracy efficiently using chunked batch processing
    to prevent memory overhead on large (11M+) datasets.
    """
    total_samples = y_true.shape[0]
    correct_counts = {k: 0 for k in k_list}

    for i in range(0, total_samples, chunk_size):
        y_true_chunk = y_true[i:i + chunk_size].unsqueeze(1)
        y_pred_chunk = y_pred_topk[i:i + chunk_size]

        for k in k_list:
            if k <= y_pred_chunk.shape[1]:
                is_correct = (y_true_chunk == y_pred_chunk[:, :k]).any(dim=1)
                correct_counts[k] += is_correct.sum().item()

    accuracies = {k: (correct_counts[k] / total_samples) * 100 for k in k_list}
    return accuracies


@torch.no_grad()
def fast_predict_in_batches(model, X, max_k, device, batch_size=32768):
    """Ultra-fast batch forward inference."""
    model.eval()
    preds = []
    total_samples = X.shape[0]
    start_time = time.time()

    for i in range(0, total_samples, batch_size):
        X_batch = X[i:i + batch_size].to(device)
        log_probs = model(X_batch)
        _, topk_indices = log_probs.topk(max_k, dim=-1)
        preds.append(topk_indices.cpu())

        if torch.backends.mps.is_available() and (i % (batch_size * 10) == 0):
            torch.mps.empty_cache()

    elapsed = time.time() - start_time
    print(f"   ⏱️  Inference finished: {total_samples:,} samples in {elapsed:.2f}s ({total_samples / elapsed:,.0f} samples/sec)")
    return torch.cat(preds, dim=0)


def evaluate_block_level(y_test_norm, y_test_anom, y_pred_norm_full, y_pred_anom_full, lengths_norm, lengths_anom, top_k_list=[1, 2, 3, 4, 5, 9, 15]):
    """
    Aggregates event-level predictions into block-level predictions.
    Rule: If at least ONE event in a block is anomalous (not in top-k), Block = Anomaly.
    """
    splits_norm = np.cumsum(lengths_norm)[:-1]
    splits_anom = np.cumsum(lengths_anom)[:-1]
    starts_norm = np.insert(splits_norm, 0, 0)
    starts_anom = np.insert(splits_anom, 0, 0)

    results = {}
    for g in sorted(top_k_list):
        anom_event_norm = ~torch.any(y_test_norm == y_pred_norm_full[:, :g].T, dim=0).numpy()
        anom_event_anom = ~torch.any(y_test_anom == y_pred_anom_full[:, :g].T, dim=0).numpy()

        # Block-level aggregation: Block is Anomaly if ANY event is Anomaly
        block_pred_norm = np.maximum.reduceat(anom_event_norm.astype(np.int8), starts_norm) > 0
        block_pred_anom = np.maximum.reduceat(anom_event_anom.astype(np.int8), starts_anom) > 0

        y_pred_blocks = np.concatenate((block_pred_norm, block_pred_anom))
        y_true_blocks = np.concatenate((
            np.zeros(len(block_pred_norm), dtype=bool),
            np.ones(len(block_pred_anom), dtype=bool)
        ))

        tn, fp, fn, tp = confusion_matrix(y_true_blocks, y_pred_blocks).ravel()
        acc = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        results[g] = {
            "top_k": g,
            "acc": float(acc),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "fpr": float(fpr),
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
        }
    return results


def main():
    args = parse_args()
    device = select_device()

    print("=" * 80)
    print("  🚀 DEEPLOG PERFORMANCE EVALUATION & BENCHMARK")
    print("=" * 80)
    print(f"⚡ Acceleration Engine : {device}")

    # Fallback check
    model_path = args.model_path
    if not os.path.exists(model_path):
        for fallback in ["deeplog_model_improved_v1.pt", "deeplog_model_full.pt"]:
            if os.path.exists(fallback):
                print(f"⚠️ '{model_path}' not found, falling back to '{fallback}'")
                model_path = fallback
                break
        else:
            print(f"❌ Error: Model checkpoint '{model_path}' not found!")
            sys.exit(1)

    print(f"📦 Loading Model Checkpoint: '{model_path}'...")
    deeplog = DeepLog.load(model_path, device=device).to(device)
    deeplog.eval()
    print(f"✅ Model loaded successfully (Layers: {deeplog.num_layers}, Hidden Size: {deeplog.hidden_size})")

    # 1. Load Evaluation Data & Preserve Block Boundaries
    print(f"\n📦 Loading Test Datasets (Context Length = {args.context_length})...")
    preprocessor = Preprocessor(length=args.context_length, timeout=float("inf"))

    train_file = os.path.join(args.data_dir, "hdfs_train")
    test_norm_file = os.path.join(args.data_dir, "hdfs_test_normal")
    test_anom_file = os.path.join(args.data_dir, "hdfs_test_abnormal")

    start_load = time.time()
    _, _, _, train_mapping = preprocessor.text(train_file, verbose=False)

    with open(test_norm_file) as f:
        lengths_norm = np.array([len(line.split()) for line in f])
    with open(test_anom_file) as f:
        lengths_anom = np.array([len(line.split()) for line in f])

    X_test_norm, y_test_norm, _, _ = preprocessor.text(test_norm_file, mapping=train_mapping, verbose=False)
    X_test_anom, y_test_anom, _, _ = preprocessor.text(test_anom_file, mapping=train_mapping, verbose=False)

    print(f"✅ Test Data Loaded in {time.time() - start_load:.2f}s")
    print(f"   - Normal Test Sequences   : {X_test_norm.shape[0]:,} (Blocks: {len(lengths_norm):,})")
    print(f"   - Abnormal Test Sequences : {X_test_anom.shape[0]:,} (Blocks: {len(lengths_anom):,})")
    print(f"   - Total Test Sequences    : {X_test_norm.shape[0] + X_test_anom.shape[0]:,} (Blocks: {len(lengths_norm) + len(lengths_anom):,})")

    # 2. Vectorized Forward Inference
    k_eval_list = [1, 2, 3, 4, 5, 9, 15]
    max_k = max(max(args.top_k), max(k_eval_list))
    print(f"\n⚡ Running Ultra-Fast Forward Inference (Batch Size = {args.batch_size:,}, Max Top-K = {max_k})...")
    print("   👉 Normal Dataset Inference:")
    y_pred_norm_full = fast_predict_in_batches(deeplog, X_test_norm, max_k=max_k, device=device, batch_size=args.batch_size)

    print("   👉 Abnormal Dataset Inference:")
    y_pred_anom_full = fast_predict_in_batches(deeplog, X_test_anom, max_k=max_k, device=device, batch_size=args.batch_size)

    # 3. Next-Event Top-K Accuracy
    y_test_total = torch.cat((y_test_norm, y_test_anom))
    y_pred_total = torch.cat((y_pred_norm_full, y_pred_anom_full), dim=0)
    topk_accuracies = compute_topk_accuracies(y_test_total, y_pred_total, k_list=[1, 3, 5, 9, 15])

    print("\n" + "=" * 52)
    print("NEXT-EVENT TOP-K ACCURACY")
    print("=" * 52)
    for k in [1, 3, 5, 9, 15]:
        space_padding = "  " if k < 10 else " "
        print(f"Top-{k} Accuracy{space_padding}: {topk_accuracies[k]:.2f}%")
    print("=" * 52)

    # 4. Block-Level Anomaly Detection Evaluation
    print("\n" + "=" * 52)
    print("BLOCK-LEVEL ANOMALY DETECTION")
    print("=" * 52)
    print(f"Normal Blocks   = {len(lengths_norm):,}")
    print(f"Abnormal Blocks = {len(lengths_anom):,}")
    print(f"Total Blocks    = {len(lengths_norm) + len(lengths_anom):,}")
    print("=" * 52)

    block_results = evaluate_block_level(y_test_norm, y_test_anom, y_pred_norm_full, y_pred_anom_full, lengths_norm, lengths_anom, top_k_list=k_eval_list)

    print("\n" + "=" * 95)
    print("  📊 BLOCK-LEVEL PERFORMANCE BENCHMARK MATRIX")
    print("=" * 95)
    print(f"{'Top-K (g)':<10} | {'Accuracy':<10} | {'Precision':<11} | {'Recall':<11} | {'F1-Score':<10} | {'FPR (%)':<9} | {'TP':<7} | {'FP':<8} | {'TN':<8} | {'FN':<6}")
    print("-" * 95)
    for g_val in k_eval_list:
        res = block_results[g_val]
        print(f"g = {g_val:<6} | {res['acc'] * 100:6.2f}%   | {res['precision'] * 100:8.2f}%   | {res['recall'] * 100:8.2f}%   | {res['f1']:8.4f}   | {res['fpr'] * 100:6.4f}%  | {res['tp']:<7,d} | {res['fp']:<8,d} | {res['tn']:<8,d} | {res['fn']:<6,d}")
    print("=" * 95)

    # 5. Export Metrics JSON
    os.makedirs(args.output_dir, exist_ok=True)
    next_event_path = os.path.join(args.output_dir, "next_event_metrics.json")
    block_path = os.path.join(args.output_dir, "block_anomaly_metrics.json")

    next_event_json = {f"top_{k}": round(acc, 2) for k, acc in topk_accuracies.items()}
    with open(next_event_path, "w", encoding="utf-8") as f:
        json.dump(next_event_json, f, indent=2)

    block_json = {
        str(g): {
            "top_k": g,
            "accuracy": round(res["acc"] * 100, 2),
            "precision": round(res["precision"] * 100, 2),
            "recall": round(res["recall"] * 100, 2),
            "f1": round(res["f1"] * 100, 2),
            "fpr": round(res["fpr"] * 100, 4),
            "tp": res["tp"],
            "fp": res["fp"],
            "tn": res["tn"],
            "fn": res["fn"],
        }
        for g, res in block_results.items()
    }
    with open(block_path, "w", encoding="utf-8") as f:
        json.dump(block_json, f, indent=2)

    print(f"\n💾 Saved metrics to:")
    print(f"   - {next_event_path}")
    print(f"   - {block_path}")


if __name__ == "__main__":
    main()
