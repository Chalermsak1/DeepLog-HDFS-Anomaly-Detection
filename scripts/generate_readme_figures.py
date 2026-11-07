"""Generate README figures from real benchmark outputs in results/metrics/.

Every number plotted here is read from JSON files produced by the evaluation
scripts (evaluate_advanced_scoring.py, evaluate.py, benchmark_serving.py,
streaming_simulator.py). Nothing is hard-coded.

Usage:
    python scripts/generate_readme_figures.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplcache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "results" / "metrics"
OUT = ROOT / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

BLUE = "#2563eb"
PURPLE = "#7c3aed"
TEAL = "#0d9488"
ORANGE = "#ea580c"
GRAY = "#64748b"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titleweight": "bold",
        "axes.titlesize": 14,
        "axes.labelweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    }
)


def load(name: str) -> dict:
    with open(METRICS / name, encoding="utf-8") as f:
        return json.load(f)


def fig_scoring_strategies() -> None:
    """Precision vs recall for every scoring strategy evaluated on 570,204 blocks."""
    data = load("advanced_scoring_metrics.json")
    groups = {
        "Baseline (any violation)": ([], BLUE, "o"),
        "Consecutive Violations": ([], ORANGE, "s"),
        "Anomaly Ratio": ([], GRAY, "^"),
        "Cumulative NLL": ([], PURPLE, "D"),
    }
    for name, m in data.items():
        for g in groups:
            key = g.split(" (")[0]
            if name.startswith(key):
                groups[g][0].append((m["recall"] * 100, m["precision"] * 100, m["f1"], name))
                break

    fig, ax = plt.subplots(figsize=(10, 6.2))
    for label, (pts, color, marker) in groups.items():
        if not pts:
            continue
        pts.sort()
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        if label == "Cumulative NLL":
            ax.plot(xs, ys, color=color, lw=2, alpha=0.6)
        ax.scatter(xs, ys, s=90, color=color, marker=marker, label=label, zorder=3, edgecolor="white")

    best = max(data.items(), key=lambda kv: kv[1]["f1"])
    bx, by = best[1]["recall"] * 100, best[1]["precision"] * 100
    ax.scatter([bx], [by], s=420, facecolor="none", edgecolor=PURPLE, lw=2.5, zorder=4)
    ax.annotate(
        f"Best F1 = {best[1]['f1'] * 100:.2f}%\n{best[0]}",
        xy=(bx, by), xytext=(bx - 40, by - 30),
        fontsize=10.5, fontweight="bold", color=PURPLE,
        arrowprops=dict(arrowstyle="->", color=PURPLE, lw=1.5),
    )
    cv = data.get("Consecutive Violations (g=4, c=2)")
    if cv:
        ax.annotate(
            f"Highest-precision alerting\n{cv['precision'] * 100:.1f}% precision, only {cv['fp']:,} FPs",
            xy=(cv["recall"] * 100, cv["precision"] * 100), xytext=(32, 99),
            fontsize=10, color=ORANGE, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.3),
        )

    ax.set_xlabel("Recall (%)")
    ax.set_ylabel("Precision (%)")
    ax.set_xlim(-2, 104)
    ax.set_ylim(0, 108)
    ax.set_title("Scoring Strategy Trade-off on 570,204 HDFS Test Blocks")
    ax.legend(loc="lower left", frameon=False)
    fig.savefig(OUT / "scoring_precision_recall.png")
    plt.close(fig)


def fig_nll_threshold_sweep() -> None:
    """Precision / recall / F1 as the cumulative-NLL threshold changes."""
    data = load("advanced_scoring_metrics.json")
    rows = []
    for name, m in data.items():
        if name.startswith("Cumulative NLL"):
            tau = float(name.split("threshold=")[1].rstrip(")"))
            rows.append((tau, m["precision"] * 100, m["recall"] * 100, m["f1"] * 100))
    rows.sort()
    taus = [r[0] for r in rows]

    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.plot(taus, [r[1] for r in rows], marker="o", lw=2.4, color=BLUE, label="Precision")
    ax.plot(taus, [r[2] for r in rows], marker="s", lw=2.4, color=TEAL, label="Recall")
    ax.plot(taus, [r[3] for r in rows], marker="D", lw=3, color=PURPLE, label="F1-score")
    best = max(rows, key=lambda r: r[3])
    ax.axvline(best[0], color=PURPLE, ls=":", lw=1.5)
    ax.annotate(
        f"tau = {best[0]:.2f}\nF1 = {best[3]:.2f}%",
        xy=(best[0], best[3]), xytext=(best[0] + 0.25, best[3] + 14),
        fontweight="bold", color=PURPLE,
        arrowprops=dict(arrowstyle="->", color=PURPLE),
    )
    ax.set_xlabel("Cumulative NLL threshold (tau)")
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Cumulative NLL Threshold Sweep (HDFS, 570,204 blocks)")
    ax.legend(frameon=False, loc="center right")
    fig.savefig(OUT / "nll_threshold_sweep.png")
    plt.close(fig)


def fig_topk_block_tradeoff() -> None:
    """Block-level precision/recall/F1/FPR vs the top-g candidate window."""
    data = load("block_anomaly_metrics.json")
    ks = sorted(int(k) for k in data)
    prec = [data[str(k)]["precision"] for k in ks]
    rec = [data[str(k)]["recall"] for k in ks]
    f1 = [data[str(k)]["f1"] for k in ks]
    fpr = [data[str(k)]["fpr"] for k in ks]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    x = range(len(ks))
    w = 0.27
    ax1.bar([i - w for i in x], prec, w, color=BLUE, label="Precision")
    ax1.bar(list(x), rec, w, color=TEAL, label="Recall")
    ax1.bar([i + w for i in x], f1, w, color=PURPLE, label="F1-score")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels([f"g={k}" for k in ks])
    ax1.set_ylabel("Score (%)")
    ax1.set_ylim(0, 110)
    ax1.set_title("Block Detection vs Top-g Window")
    ax1.legend(frameon=False, ncol=3, loc="upper left")

    ax2.plot(ks, fpr, marker="o", lw=2.4, color=ORANGE)
    ax2.set_yscale("log")
    ax2.set_xlabel("Top-g candidates")
    ax2.set_ylabel("False positive rate (%, log scale)")
    ax2.set_title("False Positive Rate Drops with Larger g")
    for k, v in zip(ks, fpr):
        ax2.annotate(f"{v:.2f}%", (k, v), textcoords="offset points", xytext=(6, 6), fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "topk_block_tradeoff.png")
    plt.close(fig)


def fig_serving() -> None:
    """Throughput and latency from benchmark_serving.py (CPU and Apple MPS)."""
    data = load("serving_benchmark.json")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    colors = {"cpu": BLUE, "mps": PURPLE}
    for dev, modes in data["devices"].items():
        mode = modes.get("jit") or modes.get("native")
        if not mode:
            continue
        items = sorted(mode.values(), key=lambda r: r["batch_size"])
        bs = [r["batch_size"] for r in items]
        ax1.plot(bs, [r["throughput_samples_per_sec"] for r in items], marker="o", lw=2.4,
                 color=colors.get(dev, GRAY), label=f"{dev.upper()} (TorchScript JIT)")
        ax2.plot(bs, [r["p99_latency_ms"] for r in items], marker="s", lw=2.4,
                 color=colors.get(dev, GRAY), label=f"{dev.upper()} p99")
    for ax in (ax1, ax2):
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Batch size")
        ax.legend(frameon=False)
    ax1.set_ylabel("Throughput (sequences / s)")
    ax1.set_title("Inference Throughput")
    ax2.set_yscale("log")
    ax2.set_ylabel("p99 latency (ms, log scale)")
    ax2.axhline(1.0, color=GRAY, ls=":", lw=1.3)
    ax2.text(1.2, 1.12, "1 ms", color=GRAY, fontsize=9)
    ax2.set_title("Tail Latency")
    fig.tight_layout()
    fig.savefig(OUT / "serving_benchmark.png")
    plt.close(fig)


def fig_streaming_latency() -> None:
    """Per-event inference latency from the live streaming simulator."""
    data = load("streaming_simulation.json")
    lat = data["telemetry"]["inference_latency"]
    labels = ["mean", "p50", "p95", "p99"]
    vals = [lat["mean_ms"], lat["p50_ms"], lat["p95_ms"], lat["p99_ms"]]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bars = ax.bar(labels, vals, color=[BLUE, BLUE, PURPLE, PURPLE], width=0.55)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.3f} ms", ha="center", fontweight="bold")
    ax.set_ylabel("Latency (ms)")
    ax.set_ylim(0, max(vals) * 1.3)
    tel = data["telemetry"]
    ax.set_title(f"Live Streaming Inference Latency ({tel['total_events']:,} events, "
                 f"{tel['throughput_eps']:,.0f} events/s)")
    fig.savefig(OUT / "streaming_latency.png")
    plt.close(fig)


if __name__ == "__main__":
    fig_scoring_strategies()
    fig_nll_threshold_sweep()
    fig_topk_block_tradeoff()
    fig_serving()
    fig_streaming_latency()
    print("Figures written to", OUT)
