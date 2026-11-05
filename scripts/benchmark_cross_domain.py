#!/usr/bin/env python3
"""
DeepLog Cross-Domain Benchmarking CLI.
=====================================
Runs comparative zero-shot transfer evaluation across heterogeneous log domains:
- HDFS (Hadoop Distributed File System distributed storage logs)
- BGL (Blue Gene/L supercomputer hardware failure logs)
- Thunderbird (Sandia HPC cluster system logs with high temporal variance)
- Kubernetes (Control plane logs: kube-apiserver, etcd, kubelet, coredns)

Evaluates:
1. Lexical and Information-Theoretic Diversity:
   - Vocabulary size and token count
   - Shannon Vocabulary Entropy (bits)
   - Temporal inter-arrival variance and burstiness
2. Cross-Domain Lexical Divergence:
   - Token-weighted and unique OOV rates
   - Jaccard vocabulary similarity
3. Zero-Shot Hypersphere Transfer (without retraining):
   - Discrete DeepLog OOV Rate (100% complete failure mode)
   - Continuous CRC32 Subword Hypersphere Projections
   - Mean / Min / Max Cosine Similarity
   - Semantic Coverage at thresholds 0.50 and 0.70
   - Sequence Next-Event Zero-Shot Prediction (Top-1, Top-3, Top-5 accuracy)
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

# Add project root and src/ to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from deeplog.cross_domain import (
    BGLParser,
    DomainDatasetSummary,
    DomainTransferEvaluator,
    DomainTransferResult,
    KubernetesLogParser,
    ThunderbirdParser,
    get_synthetic_bgl_logs,
    get_synthetic_domain_datasets,
    get_synthetic_hdfs_logs,
    get_synthetic_kubernetes_logs,
    get_synthetic_thunderbird_logs,
)
from deeplog.semantic import LogSemanticEncoder, SemanticDeepLog


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark zero-shot cross-domain log transfer for DeepLog."
    )
    parser.add_argument(
        "--source",
        type=str,
        default="HDFS",
        choices=["HDFS", "BGL", "Thunderbird", "Kubernetes"],
        help="Source domain (default: HDFS).",
    )
    parser.add_argument(
        "--targets",
        type=str,
        nargs="+",
        default=["BGL", "Thunderbird", "Kubernetes"],
        help="Target domains for cross-domain evaluation (default: BGL Thunderbird Kubernetes).",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=128,
        help="Embedding dimension for continuous hypersphere projections (default: 128).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Cosine similarity threshold for semantic coverage (default: 0.5).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top-K acceptance window for zero-shot next-event prediction (default: 5).",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["table", "json"],
        default="table",
        help="Output format: 'table' or 'json' (default: table).",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="results/metrics/cross_domain_benchmark.json",
        help="Path to output JSON file (default: results/metrics/cross_domain_benchmark.json).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress banner and verbose progress messages.",
    )
    return parser.parse_args()


def format_table(headers: List[str], rows: List[List[str]]) -> str:
    """Format an ASCII / Unicode table with aligned columns."""
    widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            widths[i] = max(widths[i], len(str(val)))

    sep = "+-" + "-+-".join(["-" * w for w in widths]) + "-+"
    header_str = "| " + " | ".join([f"{h:<{widths[i]}}" for i, h in enumerate(headers)]) + " |"

    body_lines = []
    for row in rows:
        line = "| " + " | ".join([f"{str(val):<{widths[i]}}" for i, val in enumerate(row)]) + " |"
        body_lines.append(line)

    return "\n".join([sep, header_str, sep] + body_lines + [sep])


def run_benchmark(args: argparse.Namespace) -> Dict[str, Any]:
    """Execute the full cross-domain benchmark suite."""
    start_time = time.time()
    datasets = get_synthetic_domain_datasets()

    source_domain = args.source
    target_domains = [t for t in args.targets if t != source_domain]

    if not args.quiet:
        print("=" * 80)
        print(" DeepLog Cross-Domain Zero-Shot Transfer Benchmark")
        print(" Continuous CRC32 Subword Hypersphere Projections")
        print("=" * 80)
        print(f" Source Domain : {source_domain}")
        print(f" Target Domains: {', '.join(target_domains)}")
        print(f" Embedding Dim : {args.embedding_dim}")
        print(f" Sim Threshold : {args.threshold}")
        print(f" Top-K Window  : {args.top_k}")
        print()

    # Step 1: Dataset Summaries
    summaries: Dict[str, DomainDatasetSummary] = {}
    for d_name, d_info in datasets.items():
        summaries[d_name] = d_info["summary"]

    # Table 1: Domain Dataset Profiling
    t1_headers = ["Domain", "Lines", "Tokens", "Vocab Size", "Entropy (bits)", "Mean Tok/Line", "Temporal Var (s^2)"]
    t1_rows = []
    for d_name in [source_domain] + target_domains:
        s = summaries[d_name]
        t_var = f"{s.temporal_variance:.4f}" if s.temporal_variance is not None else "N/A"
        t1_rows.append([
            d_name,
            f"{s.total_lines:,}",
            f"{s.total_tokens:,}",
            f"{s.unique_tokens:,}",
            f"{s.vocabulary_entropy:.4f}",
            f"{s.mean_tokens_per_line:.2f}",
            t_var,
        ])

    # Table 2: Lexical Divergence vs Source Domain
    source_sum = summaries[source_domain]
    t2_headers = ["Domain Pair", "Target Vocab", "Shared Tokens", "Weighted OOV %", "Unique OOV %", "Jaccard Sim"]
    t2_rows = []
    lexical_comparisons: Dict[str, Any] = {}

    for t_name in target_domains:
        t_sum = summaries[t_name]
        comp = t_sum.compare_with(source_sum)
        lexical_comparisons[t_name] = comp
        t2_rows.append([
            f"{source_domain} -> {t_name}",
            f"{t_sum.unique_tokens:,}",
            f"{comp['shared_tokens_count']:,}",
            f"{comp['oov_rate_weighted'] * 100:.2f}%",
            f"{comp['oov_rate_unique'] * 100:.2f}%",
            f"{comp['jaccard_similarity']:.4f}",
        ])

    # Step 2: Initialize DomainTransferEvaluator
    src_templates = datasets[source_domain]["templates"]
    encoder = LogSemanticEncoder(embedding_dim=args.embedding_dim)
    model = SemanticDeepLog(
        embedding_dim=args.embedding_dim,
        hidden_size=args.embedding_dim,
        encoder=encoder,
    )
    evaluator = DomainTransferEvaluator(
        source_templates=src_templates,
        model=model,
        encoder=encoder,
        embedding_dim=args.embedding_dim,
    )

    # Table 3: Zero-Shot Transfer & Continuous Hypersphere Alignment
    t3_headers = [
        "Transfer Path",
        "Discrete OOV",
        "Mean Cos Sim",
        "Coverage (>=0.5)",
        "Coverage (>=0.7)",
        "Zero-Shot Top-1",
        "Zero-Shot Top-5",
    ]
    t3_rows = []
    transfer_results: Dict[str, Any] = {}

    for t_name in target_domains:
        t_info = datasets[t_name]
        t_templates = t_info["templates"]
        t_seqs = t_info["sequences"]

        res: DomainTransferResult = evaluator.evaluate(
            source_domain=source_domain,
            target_domain=t_name,
            target_templates=t_templates,
            target_sequences=t_seqs,
            top_k=args.top_k,
            threshold=args.threshold,
        )
        transfer_results[t_name] = res.to_dict()

        seq_res = res.sequence_result
        top1_str = f"{seq_res.top_1_accuracy * 100:.1f}%" if seq_res else "N/A"
        top5_str = f"{seq_res.top_5_accuracy * 100:.1f}%" if seq_res else "N/A"

        t3_rows.append([
            f"{source_domain} -> {t_name}",
            f"{res.discrete_oov_rate * 100:.1f}% (FAIL)",
            f"{res.mean_cosine_similarity:.4f} +/- {res.std_cosine_similarity:.3f}",
            f"{res.semantic_coverage_50 * 100:.1f}%",
            f"{res.semantic_coverage_70 * 100:.1f}%",
            top1_str,
            top5_str,
        ])

    elapsed = time.time() - start_time

    # Construct final result structure
    benchmark_payload = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source_domain": source_domain,
        "target_domains": target_domains,
        "embedding_dim": args.embedding_dim,
        "threshold": args.threshold,
        "top_k": args.top_k,
        "elapsed_seconds": round(elapsed, 4),
        "dataset_summaries": {k: v.to_dict() for k, v in summaries.items()},
        "lexical_comparisons": lexical_comparisons,
        "transfer_evaluations": transfer_results,
    }

    if args.format == "json":
        print(json.dumps(benchmark_payload, indent=2))
    else:
        print("[1] Domain Dataset Characteristics & Information Entropy")
        print(format_table(t1_headers, t1_rows))
        print()
        print("[2] Cross-Domain Lexical Divergence vs Source Domain")
        print(format_table(t2_headers, t2_rows))
        print()
        print("[3] Zero-Shot Domain Transfer & Continuous Hypersphere Alignment")
        print(format_table(t3_headers, t3_rows))
        print()
        print(f"Benchmark completed successfully in {elapsed:.3f}s.")

    # Save to output JSON if requested
    if args.output_json:
        out_dir = os.path.dirname(args.output_json)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(benchmark_payload, f, indent=2)
        if not args.quiet and args.format != "json":
            print(f"Results exported to: {os.path.abspath(args.output_json)}")

    return benchmark_payload


def main():
    args = parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
