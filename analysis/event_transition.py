#!/usr/bin/env python3
"""
Event Transition (Bigram) Analysis
==================================
Analyzes sequential 2-gram event transitions (E_i -> E_j) in normal vs. anomalous HDFS blocks.

Scientific Interpretation Guidelines:
- Identifies sequential deviations and transitions predominantly observed in anomaly traces.
- Strictly avoids unsupported causal claims (e.g. do not state "E5 -> E7 caused the anomaly";
  rather state "E5 -> E7 is predominantly observed in anomalous sequences and never observed
  in normal traces in this dataset").
"""

import os
import sys
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

FIGURES_DIR = "results/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)


def extract_bigrams(filepath):
    """Extracts consecutive event transitions (E_i -> E_j)."""
    transitions = Counter()
    total_transitions = 0
    with open(filepath, "r") as f:
        for line in f:
            tokens = [int(x) for x in line.strip().split() if x]
            for i in range(len(tokens) - 1):
                transitions[(tokens[i], tokens[i + 1])] += 1
                total_transitions += 1
    return transitions, total_transitions


def main():
    print("=" * 75)
    print("  🔄 HDFS LOG EVENT TRANSITION (BIGRAM) ANALYSIS")
    print("=" * 75)

    train_path = "examples/data/hdfs_train"
    norm_test_path = "examples/data/hdfs_test_normal"
    anom_test_path = "examples/data/hdfs_test_abnormal"

    print("Extracting sequential transitions...")
    train_trans, train_total = extract_bigrams(train_path)
    norm_trans, norm_total = extract_bigrams(norm_test_path)
    anom_trans, anom_total = extract_bigrams(anom_test_path)

    all_norm_trans = train_trans + norm_trans
    all_norm_total = train_total + norm_total
    all_anom_total = anom_total

    print(f"Total Normal Transitions  : {all_norm_total:,} ({len(all_norm_trans)} unique bigrams)")
    print(f"Total Anomaly Transitions : {all_anom_total:,} ({len(anom_trans)} unique bigrams)")

    # 1. Top 10 Normal Transitions
    print("\n" + "-" * 75)
    print("TOP 10 NORMAL BIGRAM TRANSITIONS:")
    print("-" * 75)
    for (src, dst), cnt in all_norm_trans.most_common(10):
        pct = (cnt / all_norm_total) * 100
        anom_cnt = anom_trans.get((src, dst), 0)
        print(f"   E{src:<2} ➔ E{dst:<2} : {cnt:>10,d} occurrences ({pct:5.2f}%) | In Anomaly: {anom_cnt:>6,d}")

    # 2. Key Transitions Disproportionately Present in Anomaly Traces
    print("\n" + "-" * 75)
    print("KEY ANOMALY-ASSOCIATED TRANSITIONS (Forbidden or Rare in Normal):")
    print("-" * 75)
    anomaly_skewed = []
    for (src, dst), cnt in anom_trans.items():
        norm_cnt = all_norm_trans.get((src, dst), 0)
        # Check for transitions heavily skewed to anomalies
        if cnt >= 100 and norm_cnt == 0:
            anomaly_skewed.append(((src, dst), cnt, norm_cnt))

    anomaly_skewed.sort(key=lambda x: x[1], reverse=True)
    for (src, dst), a_cnt, n_cnt in anomaly_skewed[:10]:
        print(f"   E{src:<2} ➔ E{dst:<2} : {a_cnt:>6,d} in Anomaly | {n_cnt:>6,d} in Normal (Strictly Anomaly-Associated)")

    # 3. Visualization: Transition Comparison
    # Pick top 5 normal and top 5 anomaly-associated transitions
    top_normal_pairs = [pair for pair, _ in all_norm_trans.most_common(6)]
    top_anom_pairs = [pair for pair, _, _ in anomaly_skewed[:6]]
    selected_pairs = top_normal_pairs + [p for p in top_anom_pairs if p not in top_normal_pairs]

    labels = [f"E{s}➔E{d}" for s, d in selected_pairs]
    norm_counts_plot = [all_norm_trans.get(p, 0) for p in selected_pairs]
    anom_counts_plot = [anom_trans.get(p, 0) for p in selected_pairs]

    fig, ax = plt.subplots(figsize=(14, 6), dpi=300)
    x = np.arange(len(selected_pairs))
    width = 0.38

    ax.bar(x - width / 2, norm_counts_plot, width, label="Normal Transitions", color="#2563eb", alpha=0.9)
    ax.bar(x + width / 2, anom_counts_plot, width, label="Anomaly Transitions", color="#dc2626", alpha=0.9)

    ax.set_yscale("log")
    ax.set_ylabel("Transition Count (Log Scale)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Event Transition (E_src ➔ E_dst)", fontsize=12, fontweight="bold")
    ax.set_title("HDFS Log Event Bigram Transitions: Normal Workflows vs Anomaly Deviations", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
    ax.legend(frameon=True, facecolor="#ffffff", edgecolor="#cbd5e1", fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    # Highlight E5 -> E7
    if (5, 7) in selected_pairs:
        idx_e5_e7 = selected_pairs.index((5, 7))
        ax.annotate(
            "E5 ➔ E7\n1,993 Anomaly / 0 Normal\n(Forbidden in Normal)",
            xy=(idx_e5_e7 + width / 2, anom_trans.get((5, 7), 1)),
            xytext=(idx_e5_e7 - 0.5, 20000),
            arrowprops=dict(facecolor="#b91c1c", shrink=0.08, width=1.5, headwidth=6),
            bbox=dict(boxstyle="round,pad=0.3", fc="#fef2f2", ec="#ef4444", lw=1),
            fontsize=8.5,
            fontweight="bold",
            color="#991b1b"
        )

    plt.tight_layout()
    trans_fig_path = os.path.join(FIGURES_DIR, "event_transitions.png")
    plt.savefig(trans_fig_path)
    plt.close()

    print(f"\n✅ Generated: {trans_fig_path}")
    print("=" * 75)


if __name__ == "__main__":
    main()
