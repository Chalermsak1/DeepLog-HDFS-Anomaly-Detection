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
MAX_EXAMPLES = 2


def parse_line(line):
    return [int(x) for x in line.strip().split() if x]


def load_blocks(path):
    sequences = []
    lengths = []

    with open(path, "r") as f:
        for line in f:
            seq = parse_line(line)
            sequences.append(seq)
            lengths.append(len(seq))

    return sequences, np.asarray(lengths, dtype=np.int64)


@torch.no_grad()
def predict(model, X, device):
    preds = []

    for i in range(0, X.shape[0], BATCH_SIZE):
        xb = X[i:i + BATCH_SIZE].to(device)
        log_probs = model(xb)
        _, topk = log_probs.topk(TOP_K, dim=-1)
        preds.append(topk.cpu())

    return torch.cat(preds, dim=0)


def build_event_flags(y, pred):
    """
    True = this event is anomalous:
    actual event is NOT inside Top-K predictions.
    """
    y_np = y.cpu().numpy()
    pred_np = pred.numpy()

    return ~np.any(
        pred_np == y_np[:, None],
        axis=1
    )


def aggregate_to_blocks(event_flags, lengths):
    """
    IMPORTANT:
    Preprocessor.sequence() creates ONE prediction for EVERY event,
    including the first 10 events whose context contains NO_EVENT padding.

    Therefore:
        predictions per block = number of events in that block
    """
    block_flags = []

    offset = 0

    for length in lengths:
        length = int(length)

        if length == 0:
            block_flags.append(False)
            continue

        end = offset + length

        if end > len(event_flags):
            raise RuntimeError(
                f"Block boundary exceeds predictions: "
                f"offset={offset}, length={length}, "
                f"predictions={len(event_flags)}"
            )

        block_flags.append(
            bool(event_flags[offset:end].any())
        )

        offset = end

    if offset != len(event_flags):
        raise RuntimeError(
            f"Offset mismatch: consumed {offset:,}, "
            f"predictions contain {len(event_flags):,}"
        )

    return np.asarray(block_flags, dtype=bool)


def prediction_offset(lengths, block_index):
    """
    Return the global prediction offset for a block.
    Since every event produces one prediction,
    this is simply the prefix sum of event counts.
    """
    return int(np.sum(lengths[:block_index]))


def inspect_block(
    title,
    block_idx,
    sequences,
    lengths,
    pred,
    y,
    mapping,
):
    seq = sequences[block_idx]

    start = prediction_offset(lengths, block_idx)
    end = start + len(seq)

    local_pred = pred[start:end]
    local_y = y[start:end]

    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)

    print(f"Block Index : {block_idx}")
    print(f"Event Count : {len(seq)}")

    readable_sequence = [
        f"E{mapping.get(e, e)}"
        for e in seq
    ]

    if len(readable_sequence) > 80:
        print(
            "Sequence    : "
            + " -> ".join(readable_sequence[:80])
            + f" ... ({len(readable_sequence) - 80} more)"
        )
    else:
        print(
            "Sequence    : "
            + " -> ".join(readable_sequence)
        )

    # Find first event-level anomaly
    y_np = local_y.numpy()
    p_np = local_pred.numpy()

    flags = ~np.any(
        p_np == y_np[:, None],
        axis=1
    )

    flagged = np.where(flags)[0]

    if len(flagged) == 0:
        print("\nNo event-level anomaly detected in this block.")
        return

    j = int(flagged[0])

    # Event j is predicted using the previous 10 events.
    # For first 10 events, NO_EVENT padding exists.
    context = seq[max(0, j - CONTEXT_LENGTH):j]

    # Pad displayed context when j < 10
    if j < CONTEXT_LENGTH:
        padding = [
            "NO_EVENT"
        ] * (CONTEXT_LENGTH - j)

        context_display = padding + [
            f"E{mapping.get(e, e)}"
            for e in context
        ]
    else:
        context_display = [
            f"E{mapping.get(e, e)}"
            for e in context
        ]

    actual_event = int(seq[j])

    topk_ids = local_pred[j].tolist()

    print("\nFirst flagged event:")
    print("Context  : " + " -> ".join(context_display))
    print(f"Actual   : E{mapping.get(actual_event, actual_event)}")

    print(
        "Top-5    : "
        + ", ".join(
            f"E{mapping.get(int(e), int(e))}"
            for e in topk_ids
        )
    )

    print("Result   : EVENT ANOMALY")


def main():

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("=" * 90)
    print("DEEPLOG STEP 6.2 V3 - VERIFIED BLOCK CONFUSION ANALYSIS")
    print("=" * 90)

    print(f"Device : {device}")
    print(f"Model  : {MODEL_PATH}")
    print(f"Top-K  : {TOP_K}")

    # ------------------------------------------------------------
    # 1. Load model
    # ------------------------------------------------------------

    print("\n[1] Loading model...")

    model = DeepLog.load(
        MODEL_PATH,
        device=device
    ).to(device)

    model.eval()

    print(
        f"Model loaded: "
        f"Layers={model.num_layers}, "
        f"Hidden Size={model.hidden_size}"
    )

    # ------------------------------------------------------------
    # 2. Canonical training mapping
    # ------------------------------------------------------------

    print("\n[2] Loading canonical training mapping...")

    preprocessor = Preprocessor(
        length=CONTEXT_LENGTH,
        timeout=float("inf"),
    )

    _, _, _, mapping = preprocessor.text(
        TRAIN_PATH,
        verbose=False
    )

    print("Training mapping:")

    for k, v in sorted(mapping.items()):
        print(f"  {k} -> E{v}")

    # ------------------------------------------------------------
    # 3. Load raw block boundaries
    # ------------------------------------------------------------

    print("\n[3] Loading test blocks...")

    normal_sequences, normal_lengths = load_blocks(
        NORMAL_PATH
    )

    anomaly_sequences, anomaly_lengths = load_blocks(
        ANOMALY_PATH
    )

    print(
        f"Normal blocks  : "
        f"{len(normal_sequences):,}"
    )

    print(
        f"Anomaly blocks : "
        f"{len(anomaly_sequences):,}"
    )

    # ------------------------------------------------------------
    # 4. Preprocess exactly like evaluate.py
    # ------------------------------------------------------------

    print("\n[4] Preprocessing...")

    X_normal, y_normal, _, _ = preprocessor.text(
        NORMAL_PATH,
        mapping=mapping,
        verbose=False
    )

    X_anomaly, y_anomaly, _, _ = preprocessor.text(
        ANOMALY_PATH,
        mapping=mapping,
        verbose=False
    )

    print(
        f"Normal sequences : "
        f"{X_normal.shape[0]:,}"
    )

    print(
        f"Anomaly sequences: "
        f"{X_anomaly.shape[0]:,}"
    )

    # Verify number of events exactly matches block lengths
    normal_event_count = int(normal_lengths.sum())
    anomaly_event_count = int(anomaly_lengths.sum())

    print(
        f"Normal raw events : "
        f"{normal_event_count:,}"
    )

    print(
        f"Anomaly raw events: "
        f"{anomaly_event_count:,}"
    )

    if normal_event_count != X_normal.shape[0]:
        raise RuntimeError(
            "Normal event count does not match "
            "Preprocessor output."
        )

    if anomaly_event_count != X_anomaly.shape[0]:
        raise RuntimeError(
            "Anomaly event count does not match "
            "Preprocessor output."
        )

    print("✅ Event/block alignment verified.")

    # ------------------------------------------------------------
    # 5. Inference
    # ------------------------------------------------------------

    print("\n[5] Running inference...")

    start_time = time.time()

    pred_normal = predict(
        model,
        X_normal,
        device
    )

    pred_anomaly = predict(
        model,
        X_anomaly,
        device
    )

    print(
        f"Inference time: "
        f"{time.time() - start_time:.2f}s"
    )

    # ------------------------------------------------------------
    # 6. Event-level anomaly flags
    # ------------------------------------------------------------

    print("\n[6] Computing event-level anomaly flags...")

    normal_event_flags = build_event_flags(
        y_normal,
        pred_normal
    )

    anomaly_event_flags = build_event_flags(
        y_anomaly,
        pred_anomaly
    )

    # ------------------------------------------------------------
    # 7. Block-level aggregation
    # ------------------------------------------------------------

    print("\n[7] Aggregating to block level...")

    normal_block_flags = aggregate_to_blocks(
        normal_event_flags,
        normal_lengths
    )

    anomaly_block_flags = aggregate_to_blocks(
        anomaly_event_flags,
        anomaly_lengths
    )

    # ------------------------------------------------------------
    # 8. Confusion matrix
    # ------------------------------------------------------------

    fp = int(normal_block_flags.sum())
    tn = int((~normal_block_flags).sum())

    tp = int(anomaly_block_flags.sum())
    fn = int((~anomaly_block_flags).sum())

    total = tp + tn + fp + fn

    accuracy = (
        100 * (tp + tn) / total
    )

    precision = (
        100 * tp / (tp + fp)
        if tp + fp else 0
    )

    recall = (
        100 * tp / (tp + fn)
        if tp + fn else 0
    )

    f1 = (
        100 * 2 * precision * recall /
        (precision + recall)
        if precision + recall else 0
    )

    fpr = (
        100 * fp / (fp + tn)
        if fp + tn else 0
    )

    print("\n" + "=" * 90)
    print("VERIFIED BLOCK-LEVEL RESULTS")
    print("=" * 90)

    print(f"TP : {tp:,}")
    print(f"FP : {fp:,}")
    print(f"TN : {tn:,}")
    print(f"FN : {fn:,}")

    print("\nMetrics:")

    print(f"Accuracy  : {accuracy:.4f}%")
    print(f"Precision : {precision:.4f}%")
    print(f"Recall    : {recall:.4f}%")
    print(f"F1-Score  : {f1:.4f}%")
    print(f"FPR       : {fpr:.4f}%")

    # ------------------------------------------------------------
    # 9. Find examples
    # ------------------------------------------------------------

    tp_idx = np.where(anomaly_block_flags)[0]
    fn_idx = np.where(~anomaly_block_flags)[0]
    fp_idx = np.where(normal_block_flags)[0]
    tn_idx = np.where(~normal_block_flags)[0]

    print("\nVerified example indices:")

    print(
        "TP:",
        tp_idx[:MAX_EXAMPLES].tolist()
    )

    print(
        "FP:",
        fp_idx[:MAX_EXAMPLES].tolist()
    )

    print(
        "TN:",
        tn_idx[:MAX_EXAMPLES].tolist()
    )

    print(
        "FN:",
        fn_idx[:MAX_EXAMPLES].tolist()
    )

    # ------------------------------------------------------------
    # 10. Inspect examples
    # ------------------------------------------------------------

    if len(tp_idx):

        inspect_block(
            "TRUE POSITIVE",
            int(tp_idx[0]),
            anomaly_sequences,
            anomaly_lengths,
            pred_anomaly,
            y_anomaly,
            mapping,
        )

    if len(fp_idx):

        inspect_block(
            "FALSE POSITIVE",
            int(fp_idx[0]),
            normal_sequences,
            normal_lengths,
            pred_normal,
            y_normal,
            mapping,
        )

    if len(tn_idx):

        inspect_block(
            "TRUE NEGATIVE",
            int(tn_idx[0]),
            normal_sequences,
            normal_lengths,
            pred_normal,
            y_normal,
            mapping,
        )

    if len(fn_idx):

        inspect_block(
            "FALSE NEGATIVE",
            int(fn_idx[0]),
            anomaly_sequences,
            anomaly_lengths,
            pred_anomaly,
            y_anomaly,
            mapping,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
