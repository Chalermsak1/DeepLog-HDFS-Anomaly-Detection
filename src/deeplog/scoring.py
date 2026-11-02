"""
DeepLog Advanced Anomaly Scoring & Adaptive Thresholding Engine
=============================================================
Provides modular, vectorized decision strategies for log sequence anomaly detection:
- Consecutive violation filtering (reducing transient false alarms)
- Anomaly ratio thresholding (robustness against isolated noise)
- Cumulative Negative Log-Likelihood (NLL) surprise scoring per block
- Adaptive NLL thresholding
"""

from typing import Dict, Union
import numpy as np
import torch
import torch.nn.functional as F


def evaluate_consecutive_violations(
    event_anomalies: np.ndarray,
    lengths: np.ndarray,
    min_consecutive: int = 2,
) -> np.ndarray:
    """
    Evaluates whether each block contains at least `min_consecutive` consecutive
    anomalous events.

    This strategy effectively mitigates single-step transient false alarms by
    requiring sustained abnormal transitions within the same block boundary.

    Parameters
    ----------
    event_anomalies : np.ndarray
        1D array (boolean or 0/1) of length sum(lengths) indicating whether
        each event is flagged anomalous.
    lengths : np.ndarray
        1D array of non-negative integers containing event counts for each block.
    min_consecutive : int, default=2
        Minimum consecutive anomalous events required to trigger a block anomaly.
        Must be >= 1.

    Returns
    -------
    np.ndarray
        1D boolean array of shape (len(lengths),) indicating anomaly status.
    """
    if min_consecutive < 1:
        raise ValueError(f"min_consecutive must be at least 1, got {min_consecutive}")

    lengths = np.asarray(lengths, dtype=np.int64)
    event_anomalies = np.asarray(event_anomalies, dtype=bool)

    if np.any(lengths < 0):
        raise ValueError("Block lengths must be non-negative")

    total_events = len(event_anomalies)
    if lengths.sum() != total_events:
        raise ValueError(
            f"Sum of block lengths ({lengths.sum()}) does not match "
            f"number of event anomalies ({total_events})"
        )

    num_blocks = len(lengths)
    output = np.zeros(num_blocks, dtype=bool)

    # Edge cases: no blocks or fewer total events than min_consecutive
    if num_blocks == 0 or total_events < min_consecutive:
        return output

    # Map each event to its corresponding block ID
    block_ids = np.repeat(np.arange(num_blocks, dtype=np.int64), lengths)

    # Fast path for single violation (any violation)
    if min_consecutive == 1:
        output[block_ids[event_anomalies]] = True
        return output

    # Multi-step consecutive violation check within identical block boundaries
    k = min_consecutive
    is_consec = event_anomalies[: total_events - k + 1].copy()
    for offset in range(1, k):
        is_consec &= event_anomalies[offset : total_events - k + 1 + offset]

    # Verify all k events reside within the same block (no cross-boundary contamination)
    same_block = block_ids[: total_events - k + 1] == block_ids[k - 1 :]
    valid_violations = is_consec & same_block

    output[block_ids[: total_events - k + 1][valid_violations]] = True
    return output


def compute_anomaly_ratios(
    event_anomalies: np.ndarray,
    lengths: np.ndarray,
) -> np.ndarray:
    """
    Computes the fraction of anomalous events per block.

    Parameters
    ----------
    event_anomalies : np.ndarray
        1D array (boolean or 0/1) indicating whether each event is anomalous.
    lengths : np.ndarray
        1D array of non-negative integers containing event counts for each block.

    Returns
    -------
    np.ndarray
        1D float64 array of shape (len(lengths),) with anomaly ratios in [0.0, 1.0].
        Blocks with length 0 have ratio 0.0.
    """
    lengths = np.asarray(lengths, dtype=np.int64)
    event_anomalies = np.asarray(event_anomalies, dtype=bool)

    if np.any(lengths < 0):
        raise ValueError("Block lengths must be non-negative")

    total_events = len(event_anomalies)
    if lengths.sum() != total_events:
        raise ValueError(
            f"Sum of block lengths ({lengths.sum()}) does not match "
            f"number of event anomalies ({total_events})"
        )

    num_blocks = len(lengths)
    ratios = np.zeros(num_blocks, dtype=np.float64)

    if num_blocks == 0 or total_events == 0:
        return ratios

    block_ids = np.repeat(np.arange(num_blocks, dtype=np.int64), lengths)
    anomaly_counts = np.bincount(block_ids[event_anomalies], minlength=num_blocks)

    valid_mask = lengths > 0
    ratios[valid_mask] = anomaly_counts[valid_mask] / lengths[valid_mask]
    return ratios


def evaluate_anomaly_ratio(
    event_anomalies: np.ndarray,
    lengths: np.ndarray,
    threshold_ratio: float = 0.10,
) -> np.ndarray:
    """
    Flags a block as anomalous if the fraction of anomalous events exceeds `threshold_ratio`.

    Parameters
    ----------
    event_anomalies : np.ndarray
        1D array (boolean or 0/1) indicating whether each event is anomalous.
    lengths : np.ndarray
        1D array of non-negative integers containing event counts for each block.
    threshold_ratio : float, default=0.10
        Threshold fraction in [0.0, 1.0]. A block is an anomaly if ratio > threshold_ratio.

    Returns
    -------
    np.ndarray
        1D boolean array of shape (len(lengths),) indicating anomaly status.
    """
    if not (0.0 <= threshold_ratio <= 1.0):
        raise ValueError(
            f"threshold_ratio must be in [0.0, 1.0], got {threshold_ratio}"
        )

    ratios = compute_anomaly_ratios(event_anomalies, lengths)
    return ratios > threshold_ratio


def compute_cumulative_nll_scores(
    log_probs: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray],
    lengths: np.ndarray,
) -> np.ndarray:
    """
    Computes the mean Negative Log-Likelihood (NLL) surprise score per block:
        Score = (1 / T) * sum_{t=1}^T -log P(e_t | context_t)

    Parameters
    ----------
    log_probs : torch.Tensor or np.ndarray
        Log probabilities of shape (N, C) over vocabulary classes for each event.
    targets : torch.Tensor or np.ndarray
        Target event indices of shape (N,).
    lengths : np.ndarray
        1D array of non-negative integers containing event counts for each block.

    Returns
    -------
    np.ndarray
        1D float64 array of shape (len(lengths),) containing the mean NLL score
        per block. Blocks with length 0 have score 0.0.
    """
    lengths = np.asarray(lengths, dtype=np.int64)

    if np.any(lengths < 0):
        raise ValueError("Block lengths must be non-negative")

    num_blocks = len(lengths)
    total_events = int(lengths.sum())

    if num_blocks == 0 or total_events == 0:
        return np.zeros(num_blocks, dtype=np.float64)

    # Compute event-level NLL
    if isinstance(log_probs, torch.Tensor) and isinstance(targets, torch.Tensor):
        if log_probs.shape[0] != total_events or targets.shape[0] != total_events:
            raise ValueError(
                f"Tensor event count ({log_probs.shape[0]}, {targets.shape[0]}) "
                f"does not match sum of block lengths ({total_events})"
            )
        nll_events = (
            F.nll_loss(log_probs, targets.to(torch.int64), reduction="none")
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
    else:
        lp = np.asarray(log_probs, dtype=np.float64)
        tg = np.asarray(targets, dtype=np.int64)
        if lp.shape[0] != total_events or tg.shape[0] != total_events:
            raise ValueError(
                f"Array event count ({lp.shape[0]}, {tg.shape[0]}) "
                f"does not match sum of block lengths ({total_events})"
            )
        nll_events = -lp[np.arange(total_events), tg]

    block_ids = np.repeat(np.arange(num_blocks, dtype=np.int64), lengths)
    block_nll_sum = np.bincount(block_ids, weights=nll_events, minlength=num_blocks)

    scores = np.zeros(num_blocks, dtype=np.float64)
    valid_mask = lengths > 0
    scores[valid_mask] = block_nll_sum[valid_mask] / lengths[valid_mask]
    return scores


def evaluate_nll_threshold(
    nll_scores: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """
    Flags a block as anomalous if its mean NLL surprise score exceeds `threshold`.

    Parameters
    ----------
    nll_scores : np.ndarray
        1D array of mean NLL surprise scores per block.
    threshold : float
        Decision threshold. Score > threshold flags anomaly.

    Returns
    -------
    np.ndarray
        1D boolean array indicating anomaly status per block.
    """
    scores = np.asarray(nll_scores, dtype=np.float64)
    return scores > float(threshold)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, Union[float, int]]:
    """
    Computes confusion matrix and performance metrics:
    Accuracy, Precision, Recall, F1-Score, and False Positive Rate (FPR).

    Parameters
    ----------
    y_true : np.ndarray
        1D ground truth binary labels (0 = Normal, 1 = Anomaly).
    y_pred : np.ndarray
        1D predicted binary labels (0 = Normal, 1 = Anomaly).

    Returns
    -------
    dict
        Dictionary containing metric scores and confusion matrix counts.
    """
    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)

    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum(~y_true & y_pred))
    tn = int(np.sum(~y_true & ~y_pred))
    fn = int(np.sum(y_true & ~y_pred))

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2.0 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fpr": float(fpr),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }
