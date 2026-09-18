import pytest
import numpy as np


def compute_metrics(tp, fp, tn, fn):
    total = tp + fp + tn + fn
    acc = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return acc, precision, recall, f1, fpr


def test_confusion_metrics_official_baseline():
    """Verify that official baseline confusion matrix numbers produce expected exact metrics."""
    tp = 10115
    fp = 1683
    tn = 551683
    fn = 6723

    acc, precision, recall, f1, fpr = compute_metrics(tp, fp, tn, fn)

    assert round(acc * 100, 2) == 98.53, f"Expected Acc 98.53%, got {acc * 100:.2f}%"
    assert round(precision * 100, 2) == 85.73, f"Expected Precision 85.73%, got {precision * 100:.2f}%"
    assert round(recall * 100, 2) == 60.07, f"Expected Recall 60.07%, got {recall * 100:.2f}%"
    assert round(f1 * 100, 2) == 70.65, f"Expected F1 70.65%, got {f1 * 100:.2f}%"
    assert round(fpr * 100, 4) == 0.3041, f"Expected FPR 0.3041%, got {fpr * 100:.4f}%"


def test_block_level_aggregation_logic():
    """Verify the block-level aggregation rule: Block is Anomaly if ANY event is anomalous."""
    # Block 1 has 3 normal events -> Block is Normal (False)
    block1_events = [False, False, False]
    # Block 2 has 2 normal events and 1 anomaly -> Block is Anomaly (True)
    block2_events = [False, True, False]
    # Block 3 has all normal events -> Block is Normal (False)
    block3_events = [False, False]

    all_event_flags = np.array(block1_events + block2_events + block3_events, dtype=bool)
    lengths = [len(block1_events), len(block2_events), len(block3_events)]

    starts = np.insert(np.cumsum(lengths)[:-1], 0, 0)
    block_preds = np.maximum.reduceat(all_event_flags.astype(np.int8), starts) > 0

    assert list(block_preds) == [False, True, False]
