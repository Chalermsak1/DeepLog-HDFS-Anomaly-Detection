#!/usr/bin/env python3
"""
Generate DeepLog Project Workflow Diagram
=========================================
Generates the publication-quality pipeline architecture diagram saved to docs/workflow.png.
Flow:
    HDFS Logs
        ↓
    Event Extraction
        ↓
    Preprocessing
        ↓
    Canonical Event Mapping
        ↓
    Context Window
        ↓
    DeepLog LSTM
        ↓
    Top-K Next Event Prediction
        ↓
    Event-Level Anomaly Detection
        ↓
    Block-Level Aggregation
        ↓
    Ground Truth Comparison
        ↓
    Evaluation
        ↓
    Error Analysis
"""

import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

os.makedirs("docs", exist_ok=True)
OUTPUT_IMAGE = "docs/workflow.png"

STEPS = [
    ("HDFS Logs", "Unstructured system log files containing raw block execution traces", "#1e293b", "#f8fafc"),
    ("Event Extraction", "Regex template matching log lines to 29 discrete Event IDs (E1-E29)", "#0f766e", "#f0fdfa"),
    ("Preprocessing", "Grouping events by machine/block ID and sequencing by timestamp", "#0369a1", "#f0f9ff"),
    ("Canonical Event Mapping", "Deriving deterministic event vocabulary strictly from training data", "#4338ca", "#eef2ff"),
    ("Context Window", "Constructing context sequences of length h=10 with NO_EVENT padding", "#6d28d9", "#f5f3ff"),
    ("DeepLog LSTM", "2-Layer stacked LSTM network (Hidden Size = 128, LogSoftmax output)", "#b45309", "#fffbeb"),
    ("Top-K Next Event Prediction", "Generating top g prediction candidates (Top-1 through Top-15)", "#c2410c", "#fff7ed"),
    ("Event-Level Anomaly Detection", "Flagging event as anomaly if actual event is outside Top-K candidates", "#be123c", "#fff1f2"),
    ("Block-Level Aggregation", "Block = Anomaly if ANY constituent event is flagged anomalous", "#991b1b", "#fef2f2"),
    ("Ground Truth Comparison", "Aligning block predictions against true labels (Normal vs Fail)", "#15803d", "#f0fdf4"),
    ("Evaluation", "Computing Accuracy (98.53%), Precision (85.73%), Recall (60.07%), F1 (70.65%)", "#166534", "#f0fdf4"),
    ("Error Analysis", "Deep inspection of TP (10,115), FP (1,683), TN (551,683), FN (6,723)", "#1e1b4b", "#e0e7ff"),
]


def draw_workflow():
    n_steps = len(STEPS)
    fig_height = 19
    fig_width = 11

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.5, n_steps + 1.2)
    ax.axis("off")

    # Title header
    ax.text(
        5, n_steps + 0.8,
        "DeepLog HDFS Anomaly Detection Architecture",
        ha="center", va="center",
        fontsize=17, fontweight="bold", color="#0f172a"
    )
    ax.text(
        5, n_steps + 0.35,
        "End-to-End Deep Learning Pipeline from Raw Logs to Error Diagnosis",
        ha="center", va="center",
        fontsize=10.5, color="#64748b", style="italic"
    )

    box_width = 7.6
    box_height = 0.72
    x_left = 5 - box_width / 2

    for i, (title, desc, border_color, bg_color) in enumerate(STEPS):
        y_center = n_steps - i - 0.2
        y_bottom = y_center - box_height / 2

        # Draw card box
        card = FancyBboxPatch(
            (x_left, y_bottom),
            box_width,
            box_height,
            boxstyle="round,pad=0.08,rounding_size=0.15",
            ec=border_color,
            fc=bg_color,
            lw=1.8,
            mutation_scale=1,
            zorder=2,
        )
        ax.add_patch(card)

        # Step number badge
        badge = FancyBboxPatch(
            (x_left + 0.2, y_center - 0.18),
            0.45, 0.36,
            boxstyle="round,pad=0.03,rounding_size=0.08",
            ec=border_color,
            fc=border_color,
            zorder=3
        )
        ax.add_patch(badge)
        ax.text(
            x_left + 0.425, y_center,
            str(i + 1),
            ha="center", va="center",
            fontsize=10, fontweight="bold", color="#ffffff",
            zorder=4
        )

        # Title text
        ax.text(
            x_left + 0.85, y_center + 0.13,
            title,
            ha="left", va="center",
            fontsize=11.5, fontweight="bold", color=border_color,
            zorder=3
        )

        # Description text
        ax.text(
            x_left + 0.85, y_center - 0.14,
            desc,
            ha="left", va="center",
            fontsize=8.5, color="#334155",
            zorder=3
        )

        # Downward connecting arrow to next step
        if i < n_steps - 1:
            arrow_start_y = y_bottom
            arrow_end_y = (n_steps - (i + 1) - 0.2) + box_height / 2
            ax.annotate(
                "",
                xy=(5, arrow_end_y),
                xytext=(5, arrow_start_y),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color="#64748b",
                    lw=2.0,
                    mutation_scale=14,
                ),
                zorder=1,
            )

    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"✅ Workflow diagram saved to: {OUTPUT_IMAGE}")


if __name__ == "__main__":
    draw_workflow()
