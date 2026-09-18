import ast
import numpy as np
import pandas as pd
import torch

from deeplog import DeepLog
from deeplog.preprocessor import Preprocessor


MODEL_PATH = "deeplog_model_improved_v1.pt"
TRACE_PATH = "dataset/HDFS_v1/preprocessed/Event_traces.csv"

CONTEXT_LENGTH = 10
TIMEOUT = float("inf")
TOP_K = 5

# Pick examples manually after loading the dataset
ANOMALY_INDEX = 0
NORMAL_INDEX = 0


def parse_events(value):
    value = str(value).strip("[]")
    return [x.strip() for x in value.split(",") if x.strip()]


def build_contexts(events, mapping):
    """
    Convert HDFS Event IDs such as E5 -> original numeric ID 5
    -> DeepLog numeric ID using the canonical training mapping.
    """
    mapping_inverse = {v: k for k, v in mapping.items()}

    deep_ids = []
    for event in events:
        if not event.startswith("E"):
            raise ValueError(f"Unexpected Event ID format: {event}")

        original_id = int(event[1:])

        if original_id not in mapping_inverse:
            raise ValueError(
                f"Original Event ID {event} is not present in training mapping."
            )

        deep_ids.append(mapping_inverse[original_id])

    contexts = []
    targets = []

    for i in range(CONTEXT_LENGTH, len(deep_ids)):
        contexts.append(deep_ids[i - CONTEXT_LENGTH:i])
        targets.append(deep_ids[i])

    if not contexts:
        return None, None

    X = torch.tensor(contexts, dtype=torch.long)
    y = torch.tensor(targets, dtype=torch.long)

    return X, y


def main():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("=" * 80)
    print("DEEPLOG BLOCK-LEVEL PREDICTION INSPECTION")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Model : {MODEL_PATH}")

    # ------------------------------------------------------------------
    # 1. Load model
    # ------------------------------------------------------------------
    model = DeepLog.load(MODEL_PATH, device=device).to(device)
    model.eval()

    print(
        f"Model loaded: Layers={model.num_layers}, "
        f"Hidden Size={model.hidden_size}"
    )

    # ------------------------------------------------------------------
    # 2. Build canonical mapping exactly like evaluate.py
    # ------------------------------------------------------------------
    preprocessor = Preprocessor(
        length=CONTEXT_LENGTH,
        timeout=TIMEOUT,
    )

    _, _, _, train_mapping = preprocessor.text(
        "examples/data/hdfs_train",
        verbose=False,
    )

    # Invert mapping:
    # DeepLog ID -> original HDFS Event ID
    id_to_event = train_mapping

    # ------------------------------------------------------------------
    # 3. Load official HDFS traces
    # ------------------------------------------------------------------
    df = pd.read_csv(TRACE_PATH)

    normal_df = df[df["Label"] == "Success"].reset_index(drop=True)
    anomaly_df = df[df["Label"] == "Fail"].reset_index(drop=True)

    print(f"Normal blocks : {len(normal_df):,}")
    print(f"Anomaly blocks: {len(anomaly_df):,}")

    # ------------------------------------------------------------------
    # 4. Select examples
    # ------------------------------------------------------------------
    # Only inspect blocks whose every Event ID exists in the training vocabulary.
    train_event_ids = set(train_mapping.values())

    def is_compatible(features):
        events = parse_events(features)
        return all(
            event.startswith("E") and int(event[1:]) in train_event_ids
            for event in events
        )

    compatible_normal = normal_df[
        normal_df["Features"].apply(is_compatible)
    ].reset_index(drop=True)

    compatible_anomaly = anomaly_df[
        anomaly_df["Features"].apply(is_compatible)
    ].reset_index(drop=True)

    print(f"Compatible Normal blocks : {len(compatible_normal):,}")
    print(f"Compatible Anomaly blocks: {len(compatible_anomaly):,}")

    if len(compatible_normal) == 0 or len(compatible_anomaly) == 0:
        raise RuntimeError("No compatible blocks found for both classes.")

    examples = [
        ("ANOMALY", compatible_anomaly.iloc[ANOMALY_INDEX]),
        ("NORMAL", compatible_normal.iloc[NORMAL_INDEX]),
    ]

    # ------------------------------------------------------------------
    # 5. Inspect each block
    # ------------------------------------------------------------------
    for block_type, row in examples:
        block_id = row["BlockId"]
        ground_truth = "Anomaly" if block_type == "ANOMALY" else "Normal"
        events = parse_events(row["Features"])

        print("\n" + "=" * 80)
        print(f"{block_type} BLOCK")
        print("=" * 80)
        print(f"Block ID     : {block_id}")
        print(f"Ground Truth : {ground_truth}")
        print(f"Event Count  : {len(events)}")
        print(f"Event Sequence:")
        print(" -> ".join(events[:80]))
        if len(events) > 80:
            print(f"... ({len(events) - 80} more events)")

        X, y = build_contexts(events, train_mapping)

        if X is None:
            print("Block too short for Context Length = 10")
            continue

        # Only inspect the final prediction in this block.
        X_last = X[-1:].to(device)
        y_actual = y[-1].item()

        with torch.no_grad():
            log_probs = model(X_last)
            _, top_indices = log_probs.topk(TOP_K, dim=-1)

        top_indices = top_indices[0].cpu().tolist()

        actual_event = f"E{id_to_event[y_actual]}"
        predicted_events = [f"E{id_to_event[i]}" for i in top_indices]

        context_ids = X[-1].tolist()
        context_events = [f"E{id_to_event[i]}" for i in context_ids]

        print("\nLast Context:")
        print(" -> ".join(context_events))

        print(f"\nActual Next Event: {actual_event}")
        print(f"Top-{TOP_K} Predictions: {predicted_events}")

        is_in_topk = actual_event in predicted_events

        print("\nPrediction Result:")
        if is_in_topk:
            print(f"✓ Actual event {actual_event} IS in Top-{TOP_K}")
            print("  DeepLog prediction: NORMAL")
        else:
            print(f"✗ Actual event {actual_event} is NOT in Top-{TOP_K}")
            print("  DeepLog prediction: ANOMALY")

        print("\nEvent Template Mapping:")
        print(f"Actual Event: {actual_event}")

    print("\nDone.")


if __name__ == "__main__":
    main()
