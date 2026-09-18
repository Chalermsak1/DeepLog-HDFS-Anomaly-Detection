#!/usr/bin/env python3
"""
DeepLog Inference & Prediction CLI
==================================
Runs real-time or offline inference on an event sequence using DeepLog LSTM.

Usage Examples:
    python scripts/predict.py --sequence "5 5 5 22 11 9 11 9 11 9 26 26 26 23 23 23 21 21 21"
    python scripts/predict.py --sequence "E5 E22 E5 E11 E9 E7" --top_k 5
"""

import os
import sys
import argparse
import torch

# Ensure src/ is on python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from deeplog import DeepLog
from deeplog.preprocessor import Preprocessor


def parse_args():
    parser = argparse.ArgumentParser(description="Predict anomaly on an HDFS log event sequence.")
    parser.add_argument(
        "--sequence",
        type=str,
        default="5 5 5 22 11 9 11 9 11 9 26 26 26 23 23 23 21 21 21",
        help="Space-separated event sequence (e.g., '5 5 22 11' or 'E5 E22 E11')"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="deeplog_model_improved_v1.pt",
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--train_data",
        type=str,
        default="examples/data/hdfs_train",
        help="Path to training set to extract canonical mapping"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Top-K prediction candidates threshold (parameter g, default: 5)"
    )
    parser.add_argument(
        "--context_length",
        type=int,
        default=10,
        help="Context sequence length (default: 10)"
    )
    return parser.parse_args()


def parse_sequence_tokens(seq_str):
    tokens = seq_str.strip().split()
    events = []
    for token in tokens:
        clean = token.strip().upper().replace(",", "").replace("[", "").replace("]", "")
        if clean.startswith("E"):
            events.append(int(clean[1:]))
        else:
            events.append(int(clean))
    return events


def main():
    args = parse_args()

    device = torch.device("cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")

    # 1. Load canonical mapping
    preprocessor = Preprocessor(length=args.context_length, timeout=float("inf"))
    _, _, _, train_mapping = preprocessor.text(args.train_data, verbose=False)
    id_to_canonical = {v: k for k, v in train_mapping.items()}
    canonical_to_event = train_mapping

    # 2. Load model
    if not os.path.exists(args.model_path):
        print(f"❌ Error: Model checkpoint '{args.model_path}' not found!")
        sys.exit(1)

    deeplog = DeepLog.load(args.model_path, device=device).to(device)
    deeplog.eval()

    # 3. Parse input sequence
    events = parse_sequence_tokens(args.sequence)
    if len(events) == 0:
        print("❌ Error: Empty sequence provided.")
        sys.exit(1)

    print("=" * 70)
    print("  🔍 DEEPLOG SEQUENCE ANOMALY DIAGNOSIS")
    print("=" * 70)
    print(f"Input Sequence Length : {len(events)} events")
    print(f"Sequence Pattern      : " + " -> ".join(f"E{e}" for e in events[:30]) + ("..." if len(events) > 30 else ""))
    print(f"Top-K Threshold (g)   : {args.top_k}")
    print(f"Context Length (h)    : {args.context_length}")
    print("-" * 70)

    # 4. Step-by-step diagnosis
    # Preprocessor uses NO_EVENT padding for the initial events
    NO_EVENT = preprocessor.NO_EVENT
    no_event_canonical = id_to_canonical.get(NO_EVENT, len(train_mapping) - 1)

    mapped_events = []
    for e in events:
        if e in id_to_canonical:
            mapped_events.append(id_to_canonical[e])
        else:
            print(f"⚠️ Warning: Event E{e} not in training vocabulary! Treated as unseen anomaly.")
            mapped_events.append(-1)

    block_is_anomaly = False
    flagged_steps = []

    print(f"{'Step':<5} | {'Context':<35} | {'Actual':<8} | {'Top-K Candidates':<25} | {'Status'}")
    print("-" * 85)

    for i in range(len(mapped_events)):
        actual_mapped = mapped_events[i]
        actual_raw = events[i]

        # Context construction with NO_EVENT padding for early steps
        if i < args.context_length:
            padding_len = args.context_length - i
            raw_ctx = [no_event_canonical] * padding_len + mapped_events[:i]
        else:
            raw_ctx = mapped_events[i - args.context_length:i]

        # Replace any negative indices (unseen events) with no_event_canonical for tensor forward pass
        ctx = [c if c >= 0 else no_event_canonical for c in raw_ctx]
        ctx_tensor = torch.tensor([ctx], dtype=torch.long, device=device)

        with torch.no_grad():
            log_probs = deeplog(ctx_tensor)
            _, topk_indices = log_probs.topk(args.top_k, dim=-1)

        topk_list = topk_indices[0].cpu().tolist()
        expected_events = [f"E{canonical_to_event.get(idx, idx)}" for idx in topk_list]
        actual_name = f"E{actual_raw}"

        # Anomaly condition: actual event is not in top-k
        is_step_anomaly = (actual_mapped not in topk_list) or (actual_mapped == -1)

        status_str = "🚨 ANOMALY" if is_step_anomaly else "✅ Normal"
        if is_step_anomaly:
            block_is_anomaly = True
            flagged_steps.append((i, actual_name, expected_events))

        ctx_display = " ".join(
            "NO_EVT" if c == no_event_canonical else f"E{canonical_to_event.get(c, c)}"
            for c in ctx[-4:]  # show last 4 of context for compact display
        )
        cand_display = ", ".join(expected_events)

        print(f"{i + 1:<5} | ... {ctx_display:<29} | {actual_name:<8} | [{cand_display:<23}] | {status_str}")

    print("=" * 85)
    print("📋 DIAGNOSIS SUMMARY:")
    if block_is_anomaly:
        print(f"🚨 FINAL VERDICT : ANOMALY BLOCK DETECTED")
        print(f"   First Anomaly at Step {flagged_steps[0][0] + 1}: Actual {flagged_steps[0][1]} not in {flagged_steps[0][2]}")
        print(f"   Total Anomaly Deviations: {len(flagged_steps)} / {len(events)} events")
    else:
        print(f"✅ FINAL VERDICT : NORMAL BLOCK")
        print("   All sequential log transitions conform to normal training workflow.")
    print("=" * 85)


if __name__ == "__main__":
    main()
