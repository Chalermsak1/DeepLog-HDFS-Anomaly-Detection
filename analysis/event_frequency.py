#!/usr/bin/env python3
"""
Event Frequency & Presence Analysis
===================================
Analyzes the distribution and block presence of HDFS log event IDs across normal
and anomalous execution traces.

Scientific Interpretation Guidelines:
- Event frequency alone does NOT prove causality.
- Routine system events such as E5 and E22 appear frequently in both normal and anomalous blocks.
- Events such as E7, E20, E13, E27, and E28 are strongly associated with anomaly blocks in this
  dataset, but must NOT be described as "causes" of anomalies.
- Terminology: "associated with", "more frequently observed in anomaly blocks", "predominantly appears".
"""

import os
import sys
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

# Ensure results directories exist
FIGURES_DIR = "results/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)


def parse_file_events(filepath):
    """Parses space-separated event ID sequences per line."""
    block_events = []
    total_events = Counter()
    with open(filepath, "r") as f:
        for line in f:
            tokens = [int(x) for x in line.strip().split() if x]
            if tokens:
                block_events.append(tokens)
                total_events.update(tokens)
    return block_events, total_events


def compute_presence(block_events):
    """Computes the fraction of blocks containing each event."""
    n_blocks = len(block_events)
    presence_counter = Counter()
    for seq in block_events:
        unique_in_seq = set(seq)
        presence_counter.update(unique_in_seq)
    presence_pct = {k: (v / n_blocks) * 100 for k, v in presence_counter.items()}
    return presence_pct


def main():
    print("=" * 75)
    print("  📊 HDFS EVENT FREQUENCY & PRESENCE ANALYSIS")
    print("=" * 75)

    train_path = "examples/data/hdfs_train"
    norm_test_path = "examples/data/hdfs_test_normal"
    anom_test_path = "examples/data/hdfs_test_abnormal"

    print("Loading block sequences...")
    train_blocks, train_counts = parse_file_events(train_path)
    norm_blocks, norm_counts = parse_file_events(norm_test_path)
    anom_blocks, anom_counts = parse_file_events(anom_test_path)

    # Combine all normal blocks (train + test)
    all_normal_blocks = train_blocks + norm_blocks
    all_normal_counts = train_counts + norm_counts

    total_norm_blocks = len(all_normal_blocks)
    total_anom_blocks = len(anom_blocks)

    print(f"Total Normal Blocks   : {total_norm_blocks:,} (Events: {sum(all_normal_counts.values()):,})")
    print(f"Total Anomaly Blocks  : {total_anom_blocks:,} (Events: {sum(anom_counts.values()):,})")

    # All unique event IDs
    all_event_ids = sorted(list(set(all_normal_counts.keys()).union(set(anom_counts.keys()))))
    print(f"Discovered Event IDs  : {[f'E{e}' for e in all_event_ids]}")

    # Compute block presence %
    norm_presence = compute_presence(all_normal_blocks)
    anom_presence = compute_presence(anom_blocks)

    # =========================================================================
    # 1. Figure: Event Frequency Comparison (Log Scale)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(14, 6), dpi=300)
    x = np.arange(len(all_event_ids))
    width = 0.38

    norm_vals = [all_normal_counts.get(e, 0) for e in all_event_ids]
    anom_vals = [anom_counts.get(e, 0) for e in all_event_ids]

    rects1 = ax.bar(x - width / 2, norm_vals, width, label=f"Normal Blocks (N={total_norm_blocks:,})", color="#2563eb", alpha=0.9)
    rects2 = ax.bar(x + width / 2, anom_vals, width, label=f"Anomaly Blocks (N={total_anom_blocks:,})", color="#dc2626", alpha=0.9)

    ax.set_yscale("log")
    ax.set_ylabel("Total Event Count (Log Scale)", fontsize=12, fontweight="bold")
    ax.set_xlabel("HDFS Event ID", fontsize=12, fontweight="bold")
    ax.set_title("HDFS Log Event Total Frequency Distribution by Execution Class", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([f"E{e}" for e in all_event_ids], rotation=45, ha="right", fontsize=10)
    ax.legend(frameon=True, facecolor="#ffffff", edgecolor="#cbd5e1", fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    plt.tight_layout()
    freq_path = os.path.join(FIGURES_DIR, "event_frequency.png")
    plt.savefig(freq_path)
    plt.close()
    print(f"✅ Generated: {freq_path}")

    # =========================================================================
    # 2. Figure: Event Presence Percentage Comparison
    # =========================================================================
    fig, ax = plt.subplots(figsize=(14, 6), dpi=300)

    norm_pres_vals = [norm_presence.get(e, 0.0) for e in all_event_ids]
    anom_pres_vals = [anom_presence.get(e, 0.0) for e in all_event_ids]

    rects1 = ax.bar(x - width / 2, norm_pres_vals, width, label="Normal Block Presence (%)", color="#059669", alpha=0.9)
    rects2 = ax.bar(x + width / 2, anom_pres_vals, width, label="Anomaly Block Presence (%)", color="#b91c1c", alpha=0.9)

    ax.set_ylabel("Percentage of Blocks Containing Event (%)", fontsize=12, fontweight="bold")
    ax.set_xlabel("HDFS Event ID", fontsize=12, fontweight="bold")
    ax.set_title("HDFS Log Event Presence Comparison Across Normal vs Anomaly Blocks", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([f"E{e}" for e in all_event_ids], rotation=45, ha="right", fontsize=10)
    ax.set_ylim(0, 105)
    ax.legend(frameon=True, facecolor="#ffffff", edgecolor="#cbd5e1", fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    plt.tight_layout()
    pres_path = os.path.join(FIGURES_DIR, "event_presence.png")
    plt.savefig(pres_path)
    plt.close()
    print(f"✅ Generated: {pres_path}")

    # Print Key Observations
    print("\n" + "=" * 75)
    print("KEY STATISTICAL OBSERVATIONS:")
    print("=" * 75)
    print("1. Ubiquitous Events (Observed across both normal and anomaly traces):")
    for e in [5, 22, 11, 9, 26]:
        if e in all_event_ids:
            print(f"   - E{e:<2}: Present in {norm_presence.get(e, 0):.1f}% of normal blocks | {anom_presence.get(e, 0):.1f}% of anomaly blocks")

    print("\n2. Events Predominantly Associated with Anomaly Traces:")
    anomaly_skewed = [e for e in all_event_ids if anom_presence.get(e, 0) > 10.0 and norm_presence.get(e, 0) < 1.0]
    for e in anomaly_skewed:
        print(f"   - E{e:<2}: Present in {anom_presence.get(e, 0):.1f}% of anomaly blocks | {norm_presence.get(e, 0):.2f}% of normal blocks")

    print("\n⚠️ Note: These events are statistically correlated indicators in anomalous execution traces,")
    print("   NOT isolated single causes of anomalies.")
    print("=" * 75)


if __name__ == "__main__":
    main()
