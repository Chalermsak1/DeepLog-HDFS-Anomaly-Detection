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
    out = []

    for i in range(0, X.shape[0], BATCH_SIZE):
        xb = X[i:i + BATCH_SIZE].to(device)
        logits = model(xb)
        _, topk = logits.topk(TOP_K, dim=-1)
        out.append(topk.cpu())

    return torch.cat(out, dim=0)


def build_event_flags(y, pred):
    y_np = y.cpu().numpy()
    pred_np = pred.numpy()

    # True = event-level anomaly
    return ~np.any(pred_np == y_np[:, None], axis=1)


def aggregate_to_blocks(event_flags, lengths):
    block_flags = []
    offset = 0

    for length in lengths:
        n = max(0, int(length) - CONTEXT_LENGTH)

        if n == 0:
            block_flags.append(False)
        else:
            block_flags.append(
                bool(event_flags[offset:offset + n].any())
            )

        offset += n

    if offset != len(event_flags):
        raise RuntimeError(
            f"Offset mismatch: consumed {offset:,}, "
            f"but predictions contain {len(event_flags):,}"
        )

    return np.asarray(block_flags, dtype=bool)


def print_sequence(seq, mapping):
    return " -> ".join(
        f"E{mapping.get(e, e)}"
        for e in seq
    )


def inspect_block(
    title,
    block_idx,
    sequences,
    X,
    y,
    pred,
    mapping,
):
    seq = sequences[block_idx]
    start = sum(
        max(0, len(s) - CONTEXT_LENGTH)
        for s in sequences[:block_idx]
    )

    n = max(0, len(seq) - CONTEXT_LENGTH)

    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)
    print(f"Block Index : {block_idx}")
    print(f"Event Count : {len(seq)}")
    print(f"Sequence    : {print_sequence(seq, mapping)}")

    if n == 0:
        print("No prediction possible: block shorter than context length.")
        return

    # Find first event-level anomaly within this block
    local_pred = pred[start:start + n]
    local_y = y[start:start + n]

    flags = ~np.any(
        local_pred.numpy() == local_y.cpu().numpy()[:, None],
        axis=1
    )

    flagged = np.where(flags)[0]

    if len(flagged) == 0:
        print("\nNo event-level anomaly detected in this block.")
        return

    j = int(flagged[0])

    context = seq[j:j + CONTEXT_LENGTH]
    actual = seq[j + CONTEXT_LENGTH]

    topk_ids = local_pred[j].tolist()

    print("\nFirst flagged event:")
    print("Context  : " + " -> ".join(
        f"E{mapping.get(e, e)}" for e in context
    ))
    print(f"Actual   : E{mapping.get(actual, actual)}")
    print("Top-5    : " + ", ".join(
        f"E{mapping.get(e, e)}" for e in topk_ids
    ))
    print("Result   : EVENT ANOMALY")


def main():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("=" * 90)
    print("DEEPLOG STEP 6.2 V2 - VERIFIED BLOCK CONFUSION ANALYSIS")
    print("=" * 90)
    print(f"Device : {device}")
    print(f"Model  : {MODEL_PATH}")
    print(f"Top-K  : {TOP_K}")

    # ------------------------------------------------------------
    # Model
    # ------------------------------------------------------------
    model = DeepLog.load(MODEL_PATH, device=device).to(device)
    model.eval()

    # ------------------------------------------------------------
    # Canonical mapping
    # ------------------------------------------------------------
    preprocessor = Preprocessor(
        length=CONTEXT_LENGTH,
        timeout=float("inf"),
    )

    _, _, _, mapping = preprocessor.text(
        TRAIN_PATH,
        verbose=False,
    )

    print("\nTraining mapping:")
    for k, v in sorted(mapping.items()):
        print(f"  {k} -> E{v}")

    # ------------------------------------------------------------
    # Raw blocks
    # ------------------------------------------------------------
    normal_seq, normal_lengths = load_blocks(NORMAL_PATH)
    anomaly_seq, anomaly_lengths = load_blocks(ANOMALY_PATH)

    # ------------------------------------------------------------
    # Preprocess
    # ------------------------------------------------------------
    Xn, yn, _, _ = preprocessor.text(
        NORMAL_PATH,
        mapping=mapping,
        verbose=False,
    )

    Xa, ya, _, _ = preprocessor.text(
        ANOMALY_PATH,
        mapping=mapping,
        verbose=False,
    )

    print(f"\nNormal blocks   : {len(normal_seq):,}")
    print(f"Anomaly blocks  : {len(anomaly_seq):,}")
    print(f"Normal sequences: {len(Xn):,}")
    print(f"Anom sequences  : {len(Xa):,}")

    # ------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------
    t0 = time.time()

    pn = predict(model, Xn, device)
    pa = predict(model, Xa, device)

    print(f"\nInference time: {time.time() - t0:.2f}s")

    # ------------------------------------------------------------
    # Event flags
    # ------------------------------------------------------------
    normal_event_flags = build_event_flags(yn, pn)
    anomaly_event_flags = build_event_flags(ya, pa)

    # ------------------------------------------------------------
    # Block flags
    # ------------------------------------------------------------
    normal_block_flags = aggregate_to_blocks(
        normal_event_flags,
        normal_lengths,
    )

    anomaly_block_flags = aggregate_to_blocks(
        anomaly_event_flags,
        anomaly_lengths,
    )

    # ------------------------------------------------------------
    # Confusion matrix
    # ------------------------------------------------------------
    fp = int(normal_block_flags.sum())
    tn = int((~normal_block_flags).sum())

    tp = int(anomaly_block_flags.sum())
    fn = int((~anomaly_block_flags).sum())

    total = tp + tn + fp + fn

    accuracy = 100 * (tp + tn) / total
    precision = 100 * tp / (tp + fp) if tp + fp else 0
    recall = 100 * tp / (tp + fn) if tp + fn else 0
    f1 = (
        100 * 2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )
    fpr = 100 * fp / (fp + tn) if fp + tn else 0

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
    # Verified examples
    # ------------------------------------------------------------
    tp_idx = np.where(anomaly_block_flags)[0]
    fn_idx = np.where(~anomaly_block_flags)[0]
    fp_idx = np.where(normal_block_flags)[0]
    tn_idx = np.where(~normal_block_flags)[0]

    print("\nVerified example indices:")
    print("TP:", tp_idx[:MAX_EXAMPLES].tolist())
    print("FP:", fp_idx[:MAX_EXAMPLES].tolist())
    print("TN:", tn_idx[:MAX_EXAMPLES].tolist())
    print("FN:", fn_idx[:MAX_EXAMPLES].tolist())

    # Pick examples only when the class actually exists
    if len(tp_idx):
        inspect_block(
            "TRUE POSITIVE",
            int(tp_idx[0]),
            anomaly_seq,
            Xa,
            ya,
            pa,
            mapping,
        )

    if len(fp_idx):
        inspect_block(
            "FALSE POSITIVE",
            int(fp_idx[0]),
            normal_seq,
            Xn,
            yn,
            pn,
            mapping,
        )

    if len(tn_idx):
        inspect_block(
            "TRUE NEGATIVE",
            int(tn_idx[0]),
            normal_seq,
            Xn,
            yn,
            pn,
            mapping,
        )

    if len(fn_idx):
        inspect_block(
            "FALSE NEGATIVE",
            int(fn_idx[0]),
            anomaly_seq,
            Xa,
            ya,
            pa,
            mapping,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
