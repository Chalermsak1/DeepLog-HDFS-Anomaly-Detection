"""
Unit Tests for DeepLog Advanced Anomaly Scoring Engine
======================================================
Tests consecutive violations, anomaly ratio thresholds, cumulative NLL scores,
and classification metrics across mock tensors, boundary cases, and edge cases.
"""

import pytest
import numpy as np
import torch
import torch.nn.functional as F

from deeplog.scoring import (
    evaluate_consecutive_violations,
    compute_anomaly_ratios,
    evaluate_anomaly_ratio,
    compute_cumulative_nll_scores,
    evaluate_nll_threshold,
    compute_classification_metrics,
)


# =====================================================================
# Tests: evaluate_consecutive_violations
# =====================================================================

def test_consecutive_violations_basic():
    """Verify consecutive violation logic on straightforward blocks."""
    # Block 0 (len 4): [T, F, T, F] -> max consecutive = 1 -> False
    # Block 1 (len 5): [F, T, T, F, F] -> max consecutive = 2 -> True
    # Block 2 (len 3): [F, F, F] -> max consecutive = 0 -> False
    lengths = np.array([4, 5, 3])
    events = np.array([
        True, False, True, False,
        False, True, True, False, False,
        False, False, False
    ])

    pred = evaluate_consecutive_violations(events, lengths, min_consecutive=2)
    assert np.array_equal(pred, np.array([False, True, False]))

    pred_k3 = evaluate_consecutive_violations(events, lengths, min_consecutive=3)
    assert np.array_equal(pred_k3, np.array([False, False, False]))


def test_consecutive_violations_boundary_leak_prevention():
    """Ensure consecutive violations across block boundaries are strictly rejected."""
    # Block 0 (len 3): ends with True: [F, F, T]
    # Block 1 (len 3): starts with True: [T, F, F]
    # Block 2 (len 2): [T, T]
    # Cross-boundary pair (idx 2, idx 3) must NOT trigger anomaly in Block 0 or 1.
    lengths = np.array([3, 3, 2])
    events = np.array([False, False, True, True, False, False, True, True])

    pred = evaluate_consecutive_violations(events, lengths, min_consecutive=2)
    assert np.array_equal(pred, np.array([False, False, True]))


def test_consecutive_violations_edge_cases():
    """Test edge cases with block lengths 0, 1, 10."""
    lengths = np.array([0, 1, 10, 0, 1])
    # Block 0: len 0 -> []
    # Block 1: len 1 -> [True]
    # Block 2: len 10 -> [False, True, True, True, False, False, False, False, False, False]
    # Block 3: len 0 -> []
    # Block 4: len 1 -> [False]
    events = np.array(
        [True] + [False, True, True, True, False, False, False, False, False, False] + [False]
    )

    # min_consecutive = 2
    pred_2 = evaluate_consecutive_violations(events, lengths, min_consecutive=2)
    assert np.array_equal(pred_2, np.array([False, False, True, False, False]))

    # min_consecutive = 3
    pred_3 = evaluate_consecutive_violations(events, lengths, min_consecutive=3)
    assert np.array_equal(pred_3, np.array([False, False, True, False, False]))

    # min_consecutive = 4
    pred_4 = evaluate_consecutive_violations(events, lengths, min_consecutive=4)
    assert np.array_equal(pred_4, np.array([False, False, False, False, False]))


def test_consecutive_violations_min_consecutive_1():
    """min_consecutive=1 should match any-violation baseline."""
    lengths = np.array([3, 2, 4])
    events = np.array([False, False, False, False, True, False, False, False, False])

    pred = evaluate_consecutive_violations(events, lengths, min_consecutive=1)
    assert np.array_equal(pred, np.array([False, True, False]))


def test_consecutive_violations_empty_and_extremes():
    """Test empty arrays and min_consecutive larger than data."""
    # Completely empty
    assert len(evaluate_consecutive_violations([], [])) == 0

    # Blocks with zero length
    pred = evaluate_consecutive_violations([], [0, 0], min_consecutive=2)
    assert np.array_equal(pred, np.array([False, False]))

    # min_consecutive > total_events
    pred = evaluate_consecutive_violations([True, True], [1, 1], min_consecutive=5)
    assert np.array_equal(pred, np.array([False, False]))


def test_consecutive_violations_exceptions():
    """Test validation errors for invalid inputs."""
    with pytest.raises(ValueError, match="min_consecutive must be at least 1"):
        evaluate_consecutive_violations([True], [1], min_consecutive=0)

    with pytest.raises(ValueError, match="Sum of block lengths"):
        evaluate_consecutive_violations([True, False], [1, 2], min_consecutive=2)

    with pytest.raises(ValueError, match="Block lengths must be non-negative"):
        evaluate_consecutive_violations([True], [-1], min_consecutive=1)


# =====================================================================
# Tests: compute_anomaly_ratios & evaluate_anomaly_ratio
# =====================================================================

def test_anomaly_ratio_basic():
    """Verify anomaly ratio calculation on known fractions."""
    # Block 0 (len 10): 1 anomaly -> 0.10
    # Block 1 (len 10): 2 anomalies -> 0.20
    # Block 2 (len 10): 0 anomalies -> 0.00
    lengths = np.array([10, 10, 10])
    b0 = [True] + [False] * 9
    b1 = [True, True] + [False] * 8
    b2 = [False] * 10
    events = np.array(b0 + b1 + b2)

    ratios = compute_anomaly_ratios(events, lengths)
    assert np.allclose(ratios, [0.10, 0.20, 0.00])

    # Threshold 0.10: strictly exceeds 0.10 -> only Block 1 (0.20 > 0.10)
    pred_10 = evaluate_anomaly_ratio(events, lengths, threshold_ratio=0.10)
    assert np.array_equal(pred_10, np.array([False, True, False]))

    # Threshold 0.05: Block 0 (0.10 > 0.05) and Block 1 (0.20 > 0.05)
    pred_05 = evaluate_anomaly_ratio(events, lengths, threshold_ratio=0.05)
    assert np.array_equal(pred_05, np.array([True, True, False]))

    # Threshold 0.25: None exceeds 0.25
    pred_25 = evaluate_anomaly_ratio(events, lengths, threshold_ratio=0.25)
    assert np.array_equal(pred_25, np.array([False, False, False]))


def test_anomaly_ratio_edge_cases():
    """Test edge cases with block lengths 0, 1, 10."""
    lengths = np.array([0, 1, 10])
    # Block 0: len 0 -> ratio 0.0
    # Block 1: len 1, 1 anomaly -> ratio 1.0
    # Block 2: len 10, 1 anomaly -> ratio 0.10
    events = np.array([True] + [True] + [False] * 9)

    ratios = compute_anomaly_ratios(events, lengths)
    assert np.allclose(ratios, [0.0, 1.0, 0.10])

    # Threshold 0.10
    pred = evaluate_anomaly_ratio(events, lengths, threshold_ratio=0.10)
    assert np.array_equal(pred, np.array([False, True, False]))


def test_anomaly_ratio_exceptions():
    """Test validation errors for invalid ratio inputs."""
    with pytest.raises(ValueError, match="threshold_ratio must be in"):
        evaluate_anomaly_ratio([True], [1], threshold_ratio=1.5)

    with pytest.raises(ValueError, match="threshold_ratio must be in"):
        evaluate_anomaly_ratio([True], [1], threshold_ratio=-0.1)

    with pytest.raises(ValueError, match="Sum of block lengths"):
        evaluate_anomaly_ratio([True], [2], threshold_ratio=0.1)


# =====================================================================
# Tests: compute_cumulative_nll_scores & evaluate_nll_threshold
# =====================================================================

def test_cumulative_nll_scores_mock_tensors():
    """Verify cumulative NLL score with mock log probabilities."""
    # 4 events, 3 vocabulary classes
    # Deterministic log probabilities
    log_probs = torch.tensor([
        [-0.1, -2.5, -4.0],  # event 0, target 0 -> NLL = 0.1
        [-3.0, -0.2, -1.8],  # event 1, target 1 -> NLL = 0.2
        [-1.0, -1.0, -1.0],  # event 2, target 2 -> NLL = 1.0
        [-0.5, -1.5, -2.0],  # event 3, target 0 -> NLL = 0.5
    ], dtype=torch.float32)

    targets = torch.tensor([0, 1, 2, 0], dtype=torch.long)

    # Block 0 (len 1): event 0 -> NLL = 0.1
    # Block 1 (len 0): -> NLL = 0.0
    # Block 2 (len 3): events 1, 2, 3 -> NLL = (0.2 + 1.0 + 0.5) / 3 = 1.7 / 3
    lengths = np.array([1, 0, 3])

    scores = compute_cumulative_nll_scores(log_probs, targets, lengths)
    expected = np.array([0.1, 0.0, 1.7 / 3.0])

    assert np.allclose(scores, expected, atol=1e-5)


def test_cumulative_nll_scores_edge_cases():
    """Test edge cases with block lengths 0, 1, 10."""
    num_classes = 5
    # Create 11 events: 1 for Block 1, 10 for Block 2
    targets = torch.randint(0, num_classes, (11,))
    log_probs = torch.log_softmax(torch.randn(11, num_classes), dim=-1)

    lengths = np.array([0, 1, 10])
    scores = compute_cumulative_nll_scores(log_probs, targets, lengths)

    assert len(scores) == 3
    assert scores[0] == 0.0  # len 0 block has 0 score

    # Verify Block 1: single event
    expected_b1 = -log_probs[0, targets[0]].item()
    assert np.isclose(scores[1], expected_b1, atol=1e-5)

    # Verify Block 2: mean of 10 events
    expected_b2 = -log_probs[1:11, targets[1:11]].diag().mean().item()
    # Note: gather target NLLs
    nll_b2 = [
        -log_probs[i, targets[i]].item() for i in range(1, 11)
    ]
    assert np.isclose(scores[2], np.mean(nll_b2), atol=1e-5)


def test_cumulative_nll_numpy_fallback():
    """Verify compute_cumulative_nll_scores works with pure NumPy arrays."""
    log_probs = np.array([
        [-0.5, -1.0],
        [-2.0, -0.1],
    ])
    targets = np.array([0, 1])
    lengths = np.array([2])

    scores = compute_cumulative_nll_scores(log_probs, targets, lengths)
    assert np.allclose(scores, [0.3])


def test_evaluate_nll_threshold():
    """Verify threshold evaluation."""
    scores = np.array([0.5, 1.2, 0.1, 3.4])
    pred = evaluate_nll_threshold(scores, threshold=1.0)
    assert np.array_equal(pred, np.array([False, True, False, True]))

    empty_pred = evaluate_nll_threshold(np.array([]), threshold=1.0)
    assert len(empty_pred) == 0


# =====================================================================
# Tests: compute_classification_metrics
# =====================================================================

def test_compute_classification_metrics():
    """Verify confusion matrix and metrics accuracy."""
    y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    y_pred = np.array([1, 1, 0, 0, 0, 0, 1, 0])

    # TP = 2, FN = 1, FP = 1, TN = 4
    metrics = compute_classification_metrics(y_true, y_pred)

    assert metrics["tp"] == 2
    assert metrics["fn"] == 1
    assert metrics["fp"] == 1
    assert metrics["tn"] == 4

    assert np.isclose(metrics["accuracy"], 6 / 8)
    assert np.isclose(metrics["precision"], 2 / 3)
    assert np.isclose(metrics["recall"], 2 / 3)
    assert np.isclose(metrics["f1"], 2 * (2/3 * 2/3) / (2/3 + 2/3))
    assert np.isclose(metrics["fpr"], 1 / 5)


def test_compute_classification_metrics_zero_division():
    """Zero division edge case (no predicted positives)."""
    y_true = np.array([0, 0, 0])
    y_pred = np.array([0, 0, 0])

    metrics = compute_classification_metrics(y_true, y_pred)
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0
    assert metrics["fpr"] == 0.0
    assert metrics["accuracy"] == 1.0
