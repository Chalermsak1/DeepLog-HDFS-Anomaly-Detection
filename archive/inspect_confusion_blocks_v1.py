import os
import time
import numpy as np
import torch

from deeplog import DeepLog
from deeplog.preprocessor import Preprocessor


MODEL_PATH = "deeplog_model_improved_v1.pt"
TRAIN_PATH = "examples/data/hdfs_train"
NORMAL_PATH = "examples/data/hdfs_test_normal"
ANOMALY_PATH = "examples/data/hdfs_test_abnormal"

CONTEXT_LENGTH = 10
TOP_K = 5
BATCH_SIZE = 32768

MAX_EXAMPLES_PER_CASE = 3


def parse_sequence_line(line):
    return [int(x) for x in line.strip().split() if x]


def build_block_boundaries(path):
    lengths = []
    sequences = []

    with open(path, "r") as f:
        for line in f:
            events = parse_sequence_line(line)
            lengths.append(len(events))
            sequences.append(events)

    return np.array(lengths, dtype=np.int64), sequences


def predict_topk(model, X, device, max_k):
    preds = []

    with torch.no_grad():
        for i in range(0, X.shape[0], BATCH_SIZE):
            X_batch = X[i:i + BATCH_SIZE].to(device)
            log_probs = model(X_batch)

            _, topk = log_probs.topk(max_k, dim=-1)
            preds.append(topk.cpu())

    return torch.cat(preds, dim=0)


def event_id_to_text(event_id, mapping):
    if event_id not in mapping:
        return f"E{event_id}"
    return f"E{mapping[event_id]}"


def inspect_case(
    case_name,
    block_index,
    sequences,
    predictions,
    labels,
    mapping,
):
    seq = sequences[block_index]

    start = 0

    print("\n" + "=" * 90)
    print(case_name)
    print("=" * 90)

    print(f"Block Index   : {block_index}")
    print(f"Ground Truth  : {labels}")
    print(f"Event Count   : {len(seq)}")

    readable_sequence = [
        event_id_to_text(e, mapping)
        for e in seq
    ]

    if len(readable_sequence) > 80:
        print(
            "Sequence      : "
            + " -> ".join(readable_sequence[:80])
            + f" ... ({len(readable_sequence) - 80} more)"
        )
    else:
        print("Sequence      : " + " -> ".join(readable_sequence))

    # Find first anomalous event-level prediction in this block
    found = False

    for i in range(CONTEXT_LENGTH, len(seq)):
        actual_original = seq[i]
        pred_row = predictions[start + i - CONTEXT_LENGTH]

        predicted_original = [
            mapping[int(x)]
            for x in pred_row.tolist()
        ]

        if actual_original not in predicted_original:
            context_original = seq[i - CONTEXT_LENGTH:i]

            print("\nFirst flagged event:")
            print(
                "Context       : "
                + " -> ".join(
                    event_id_to_text(e, mapping)
                    for e in context_original
                )
            )
            print(
                f"Actual Event  : {event_id_to_text(actual_original, mapping)}"
            )
            print(
                "Top-"
                + str(TOP_K)
                + " Predictions: "
                + ", ".join(
                    event_id_to_text(e, mapping)
                    for e in predicted_original
                )
            )
            print("Event Result  : ANOMALY")
            found = True
            break

        start += 1

    if not found:
        print("\nNo event-level anomaly detected in this block.")

    print()


def main():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("=" * 90)
    print("DEEPLOG STEP 6.2 - BLOCK CONFUSION ANALYSIS")
    print("=" * 90)
    print(f"Device       : {device}")
    print(f"Model        : {MODEL_PATH}")
    print(f"Context      : {CONTEXT_LENGTH}")
    print(f"Top-K        : {TOP_K}")

    # ------------------------------------------------------------
    # 1. Load model
    # ------------------------------------------------------------
    print("\n[1] Loading model...")
    model = DeepLog.load(MODEL_PATH, device=device).to(device)
    model.eval()

    print(
        f"Model loaded: Layers={model.num_layers}, "
        f"Hidden Size={model.hidden_size}"
    )

    # ------------------------------------------------------------
    # 2. Canonical mapping
    # ------------------------------------------------------------
    print("\n[2] Loading canonical training mapping...")
    preprocessor = Preprocessor(
        length=CONTEXT_LENGTH,
        timeout=float("inf"),
    )

    _, _, _, train_mapping = preprocessor.text(
        TRAIN_PATH,
        verbose=False,
    )

    print("Training mapping:")
    for k, v in sorted(train_mapping.items()):
        print(f"  DeepLog ID {k} -> E{v}")

    # ------------------------------------------------------------
    # 3. Load test data
    # ------------------------------------------------------------
    print("\n[3] Loading test blocks...")

    normal_lengths, normal_sequences = build_block_boundaries(
        NORMAL_PATH
    )

    anomaly_lengths, anomaly_sequences = build_block_boundaries(
        ANOMALY_PATH
    )

    print(f"Normal blocks  : {len(normal_sequences):,}")
    print(f"Anomaly blocks : {len(anomaly_sequences):,}")

    # ------------------------------------------------------------
    # 4. Preprocess using canonical mapping
    # ------------------------------------------------------------
    print("\n[4] Preprocessing...")

    X_normal, y_normal, _, _ = preprocessor.text(
        NORMAL_PATH,
        mapping=train_mapping,
        verbose=False,
    )

    X_anomaly, y_anomaly, _, _ = preprocessor.text(
        ANOMALY_PATH,
        mapping=train_mapping,
        verbose=False,
    )

    print(f"Normal sequences  : {X_normal.shape[0]:,}")
    print(f"Anomaly sequences : {X_anomaly.shape[0]:,}")

    # ------------------------------------------------------------
    # 5. Prediction
    # ------------------------------------------------------------
    print("\n[5] Running inference...")

    start_time = time.time()

    pred_normal = predict_topk(
        model,
        X_normal,
        device,
        TOP_K,
    )

    pred_anomaly = predict_topk(
        model,
        X_anomaly,
        device,
        TOP_K,
    )

    print(
        f"Inference time: {time.time() - start_time:.2f} sec"
    )

    # ------------------------------------------------------------
    # 6. Event-level anomaly flags
    # ------------------------------------------------------------
    print("\n[6] Computing event-level anomaly flags...")

    y_normal_np = y_normal.cpu().numpy()
    y_anomaly_np = y_anomaly.cpu().numpy()

    pred_normal_np = pred_normal.numpy()
    pred_anomaly_np = pred_anomaly.numpy()

    normal_event_anomaly = ~(
        np.any(
            pred_normal_np == y_normal_np[:, None],
            axis=1,
        )
    )

    anomaly_event_anomaly = ~(
        np.any(
            pred_anomaly_np == y_anomaly_np[:, None],
            axis=1,
        )
    )

    # ------------------------------------------------------------
    # 7. Aggregate event results -> block results
    # ------------------------------------------------------------
    print("\n[7] Aggregating to block level...")

    def aggregate(flags, lengths):
        result = []
        offset = 0

        for length in lengths:
            n_events = max(0, int(length) - CONTEXT_LENGTH)

            if n_events == 0:
                result.append(False)
            else:
                result.append(
                    bool(flags[offset:offset + n_events].any())
                )

            offset += n_events

        return np.array(result, dtype=bool)

    normal_block_pred = aggregate(
        normal_event_anomaly,
        normal_lengths,
    )

    anomaly_block_pred = aggregate(
        anomaly_event_anomaly,
        anomaly_lengths,
    )

    # ------------------------------------------------------------
    # 8. Confusion matrix
    # ------------------------------------------------------------
    tp = int(anomaly_block_pred.sum())
    fn = int((~anomaly_block_pred).sum())

    fp = int(normal_block_pred.sum())
    tn = int((~normal_block_pred).sum())

    total = tp + tn + fp + fn

    accuracy = (tp + tn) / total * 100
    precision = tp / (tp + fp) * 100 if tp + fp else 0
    recall = tp / (tp + fn) * 100 if tp + fn else 0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0
    )
    fpr = fp / (fp + tn) * 100 if fp + tn else 0

    print("\n" + "=" * 90)
    print("BLOCK-LEVEL CONFUSION MATRIX")
    print("=" * 90)

    print(f"TP (Anomaly -> Anomaly): {tp:,}")
    print(f"FP (Normal  -> Anomaly): {fp:,}")
    print(f"TN (Normal  -> Normal)  : {tn:,}")
    print(f"FN (Anomaly -> Normal)  : {fn:,}")

    print("\nMetrics:")
    print(f"Accuracy  : {accuracy:.4f}%")
    print(f"Precision : {precision:.4f}%")
    print(f"Recall    : {recall:.4f}%")
    print(f"F1-Score  : {f1:.4f}%")
    print(f"FPR       : {fpr:.4f}%")

    # ------------------------------------------------------------
    # 9. Find example indices
    # ------------------------------------------------------------
    tp_indices = np.where(anomaly_block_pred)[0]
    fn_indices = np.where(~anomaly_block_pred)[0]
    fp_indices = np.where(normal_block_pred)[0]
    tn_indices = np.where(~normal_block_pred)[0]

    print("\nExample block indices:")
    print(
        "TP:",
        tp_indices[:MAX_EXAMPLES_PER_CASE].tolist()
    )
    print(
        "FN:",
        fn_indices[:MAX_EXAMPLES_PER_CASE].tolist()
    )
    print(
        "FP:",
        fp_indices[:MAX_EXAMPLES_PER_CASE].tolist()
    )
    print(
        "TN:",
        tn_indices[:MAX_EXAMPLES_PER_CASE].tolist()
    )

    # ------------------------------------------------------------
    # 10. Inspect first example from each case
    # ------------------------------------------------------------
    if len(tp_indices):
        inspect_case(
            "TRUE POSITIVE",
            int(tp_indices[0]),
            anomaly_sequences,
            pred_anomaly,
            "Anomaly",
            train_mapping,
        )

    if len(fp_indices):
        inspect_case(
            "FALSE POSITIVE",
            int(fp_indices[0]),
            normal_sequences,
            pred_normal,
            "Normal",
            train_mapping,
        )

    if len(tn_indices):
        inspect_case(
            "TRUE NEGATIVE",
            int(tn_indices[0]),
            normal_sequences,
            pred_normal,
            "Normal",
            train_mapping,
        )

    if len(fn_indices):
        inspect_case(
            "FALSE NEGATIVE",
            int(fn_indices[0]),
            anomaly_sequences,
            pred_anomaly,
            "Anomaly",
            train_mapping,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
