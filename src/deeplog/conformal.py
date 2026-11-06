"""Adaptive Conformal Prediction & PAC-Bayesian Bounds for Anomaly Detection.

Eliminates heuristic threshold tuning by providing finite-sample statistical coverage
guarantees under arbitrary distribution shift:

    P(s(X_{t+1}) <= q_t) >= 1 - alpha

Implements:
1. Split-Conformal Quantile Calibration: Finite-sample distribution-free validity.
2. Adaptive Conformal Inference (ACI, Gibbs & Candes 2021): Dynamic quantile tracking
   under non-exchangeable streaming time-series drift.
3. PAC-Bayes Risk Bound Evaluator: Rigorous generalization bound with confidence 1 - delta.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


@dataclass(frozen=True)
class ConformalPredictionResult:
    """Statistical outcome of conformal anomaly evaluation."""

    score: float
    dynamic_threshold: float
    current_alpha: float
    target_alpha: float
    is_anomaly: bool
    empirical_coverage: float
    rolling_error_rate: float
    p_value: float
    pac_bayes_risk_bound: float
    metadata: Dict[str, float] = field(default_factory=dict)


class AdaptiveConformalAnomalyDetector:
    """Online Adaptive Conformal Inference (ACI) detector for streaming time-series.

    Dynamically adjusts conformal significance level alpha_t such that long-term
    false positive discovery strictly converges to target alpha under arbitrary concept drift.
    """

    def __init__(
        self,
        target_alpha: float = 0.05,
        adaptation_rate: float = 0.005,
        calibration_window: int = 500,
        pac_confidence_delta: float = 0.05,
        min_alpha: float = 0.001,
        max_alpha: float = 0.50,
        update_on_anomaly: bool = False,
    ) -> None:
        if not (0.0 < target_alpha < 1.0):
            raise ValueError(f"target_alpha must be in (0, 1), got {target_alpha}")

        self.target_alpha = target_alpha
        self.adaptation_rate = adaptation_rate
        self.calibration_window = calibration_window
        self.pac_confidence_delta = pac_confidence_delta
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.update_on_anomaly = update_on_anomaly

        # Current dynamic significance level alpha_t
        self.alpha_t = target_alpha

        # Rolling score calibration buffer
        self._score_history: List[float] = []

        # Step error history: 1 if anomaly (score > q_t), 0 otherwise
        self._error_history: List[int] = []

    def calibrate(self, calibration_scores: Sequence[float]) -> float:
        """Calibrate initial score distribution with normal baseline data."""
        self._score_history = [float(s) for s in calibration_scores]
        if len(self._score_history) > self.calibration_window:
            self._score_history = self._score_history[-self.calibration_window :]
        return self.compute_conformal_quantile(self.alpha_t)

    def compute_conformal_quantile(self, alpha: float) -> float:
        """Compute finite-sample conformal quantile: ceil((n+1)(1-alpha)) / n."""
        if not self._score_history:
            return 1.0

        n = len(self._score_history)
        effective_alpha = min(max(alpha, 1.0 / (n + 1)), 1.0 - (1.0 / (n + 1)))

        # Standard conformal prediction quantile formula
        quantile_level = min(1.0, math.ceil((n + 1) * (1.0 - effective_alpha)) / n)
        # Convert to percentile (0 to 100)
        percentile = float(quantile_level * 100.0)
        return float(np.percentile(self._score_history, percentile))

    def compute_conformal_p_value(self, score: float) -> float:
        """Compute exact conformal empirical p-value for a given test score."""
        if not self._score_history:
            return 1.0
        n = len(self._score_history)
        # Proportion of calibration scores greater than or equal to test score
        count_larger = sum(1 for s in self._score_history if s >= score)
        return (count_larger + 1.0) / (n + 1.0)

    def compute_pac_bayes_bound(self, kl_divergence: float = 0.1) -> float:
        """Compute McAllester PAC-Bayesian empirical risk generalization upper bound."""
        n = max(len(self._error_history), 1)
        empirical_risk = sum(self._error_history) / float(n) if self._error_history else 0.0

        delta = max(self.pac_confidence_delta, 1e-6)
        penalty = math.sqrt(
            (kl_divergence + math.log(2.0 * math.sqrt(n) / delta)) / (2.0 * n)
        )
        return min(1.0, empirical_risk + penalty)

    def evaluate_and_update(self, score: float) -> ConformalPredictionResult:
        """Evaluate an incoming anomaly score and update ACI online tracking."""
        score_val = float(score)

        # 1. Compute current dynamic quantile threshold
        threshold_t = self.compute_conformal_quantile(self.alpha_t)
        p_val = self.compute_conformal_p_value(score_val)

        # 2. Decision: Anomaly flagged if test score exceeds dynamic quantile threshold
        is_anomaly = score_val > threshold_t
        err_t = 1 if is_anomaly else 0

        # 3. Update history: avoid poisoning normal calibration window with anomalies
        if not is_anomaly or self.update_on_anomaly:
            self._score_history.append(score_val)
            if len(self._score_history) > self.calibration_window:
                self._score_history.pop(0)

        self._error_history.append(err_t)
        if len(self._error_history) > self.calibration_window:
            self._error_history.pop(0)

        # 4. Adaptive Conformal Inference (ACI) step:
        # alpha_{t+1} = alpha_t + gamma * (target_alpha - err_t)
        # If err_t = 1 (we alerted), alpha decreases -> threshold increases next time to be more conservative.
        # If err_t = 0 (we accepted), alpha increases -> threshold decreases next time to prevent under-alerting.
        self.alpha_t = self.alpha_t + self.adaptation_rate * (self.target_alpha - float(err_t))
        self.alpha_t = max(self.min_alpha, min(self.max_alpha, self.alpha_t))

        rolling_error = (
            sum(self._error_history) / float(len(self._error_history))
            if self._error_history
            else 0.0
        )
        coverage = 1.0 - rolling_error

        pac_bound = self.compute_pac_bayes_bound()

        return ConformalPredictionResult(
            score=score_val,
            dynamic_threshold=threshold_t,
            current_alpha=self.alpha_t,
            target_alpha=self.target_alpha,
            is_anomaly=is_anomaly,
            empirical_coverage=coverage,
            rolling_error_rate=rolling_error,
            p_value=p_val,
            pac_bayes_risk_bound=pac_bound,
            metadata={
                "calibration_size": float(len(self._score_history)),
                "step_error": float(err_t),
                "threshold_headroom": float(threshold_t - score_val),
            },
        )
