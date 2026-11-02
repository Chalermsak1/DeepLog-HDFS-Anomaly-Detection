"""
DeepLog Temporal Dynamics & Time-Delta (\u0394t) Integration Module
==============================================================
Provides end-to-end temporal modeling for log anomaly detection:
1. TimeDeltaEncoder: Sinusoidal Time2Vec / logarithmic temporal feature encoder.
2. TimeAwareDeepLog: Neural architecture combining event embedding with temporal
   embeddings to model joint P(e_t, \u0394t_t | history).
3. TemporalAnomalyDetector: Evaluates joint sequence-temporal anomalies, detecting
   latency spikes, timeouts, retry storms, and unexpected event transitions.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# 1. TimeDeltaEncoder: Sinusoidal Time2Vec / Logarithmic Feature Encoder
# ==============================================================================


class TimeDeltaEncoder(nn.Module):
    """
    TimeDeltaEncoder: Encodes continuous time-deltas (\u0394t \u2265 0) into dense vectors.

    Supports:
    - "time2vec": Sinusoidal Time2Vec representation (Kazemi et al., 2019):
        t2v(\u03c4)[0] = \u03c9_0 * \u03c4 + \u03d5_0             (Linear trend / non-periodic)
        t2v(\u03c4)[i] = sin(\u03c9_i * \u03c4 + \u03d5_i)        (Periodic patterns, 1 \u2264 i < embed_dim)
      Where \u03c4 = log(1 + \u0394t) when `use_log=True` to smoothly compress wide dynamic ranges
      (microseconds to hours) into stable, well-behaved magnitudes.
    - "log": Multi-layer perceptron over log-transformed \u0394t features:
        h(\u03c4) = MLP(log(1 + \u0394t))
    - "combined": Concatenation of Time2Vec sinusoidal features and logarithmic MLP features.
    """

    def __init__(
        self,
        embed_dim: int = 16,
        mode: str = "time2vec",
        use_log: bool = True,
        learnable: bool = True,
        activation: str = "sin",
    ):
        """
        Parameters
        ----------
        embed_dim : int, default=16
            Dimensionality of the temporal embedding vector (must be \u2265 2 for time2vec).
        mode : str, default="time2vec"
            Encoding mechanism: "time2vec", "log", or "combined".
        use_log : bool, default=True
            If True, transforms input \u0394t via \u03c4 = log(1 + max(\u0394t, 0)) before encoding.
        learnable : bool, default=True
            Whether frequency (\u03c9) and phase (\u03d5) weights are learnable parameters.
        activation : str, default="sin"
            Periodic activation function for Time2Vec ("sin" or "cos").
        """
        super().__init__()

        mode = mode.lower()
        if mode not in ("time2vec", "log", "combined"):
            raise ValueError(
                f"Unsupported mode '{mode}'. Choose from 'time2vec', 'log', 'combined'."
            )
        if mode in ("time2vec", "combined") and embed_dim < 2:
            raise ValueError(
                f"embed_dim must be at least 2 for {mode} mode, got {embed_dim}."
            )

        self.embed_dim = embed_dim
        self.mode = mode
        self.use_log = use_log
        self.learnable = learnable
        self.activation_name = activation.lower()

        if self.activation_name == "sin":
            self.periodic_act = torch.sin
        elif self.activation_name == "cos":
            self.periodic_act = torch.cos
        else:
            raise ValueError(
                f"Unsupported activation '{activation}'. Choose 'sin' or 'cos'."
            )

        if self.mode == "time2vec":
            # Linear component parameters: w0, b0
            self.w0 = nn.Parameter(
                torch.empty(1), requires_grad=learnable
            )
            self.b0 = nn.Parameter(
                torch.empty(1), requires_grad=learnable
            )

            # Periodic component parameters: w, b for 1 <= i < embed_dim
            num_periodic = embed_dim - 1
            self.w = nn.Parameter(
                torch.empty(num_periodic), requires_grad=learnable
            )
            self.b = nn.Parameter(
                torch.empty(num_periodic), requires_grad=learnable
            )
            self._init_time2vec_params()

        elif self.mode == "log":
            # Direct non-linear projection of log1p(dt)
            hidden_dim = max(32, embed_dim * 2)
            self.mlp = nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, embed_dim),
            )

        elif self.mode == "combined":
            # Split embedding dimension across Time2Vec and Log MLP
            t2v_dim = max(2, embed_dim // 2)
            log_dim = embed_dim - t2v_dim

            self.w0 = nn.Parameter(torch.empty(1), requires_grad=learnable)
            self.b0 = nn.Parameter(torch.empty(1), requires_grad=learnable)
            self.w = nn.Parameter(torch.empty(t2v_dim - 1), requires_grad=learnable)
            self.b = nn.Parameter(torch.empty(t2v_dim - 1), requires_grad=learnable)
            self._init_time2vec_params()

            hidden_dim = max(16, log_dim * 2)
            self.mlp = nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, log_dim),
            )

    def _init_time2vec_params(self):
        """Initializes Time2Vec frequency and phase parameters."""
        nn.init.uniform_(self.w0, 0.1, 1.0)
        nn.init.zeros_(self.b0)

        # Initialize periodic frequencies across multiple temporal scales
        num_periodic = self.w.numel()
        if num_periodic > 1:
            # Geometrically spaced frequencies covering microsecond to multi-second intervals
            freqs = torch.exp(
                torch.linspace(math.log(0.01), math.log(10.0), num_periodic)
            )
            with torch.no_grad():
                self.w.copy_(freqs)
        else:
            nn.init.uniform_(self.w, 0.1, 2.0)

        nn.init.uniform_(self.b, -math.pi, math.pi)

    def forward(self, dt: torch.Tensor) -> torch.Tensor:
        """
        Encode time-deltas into continuous embedding vectors.

        Parameters
        ----------
        dt : torch.Tensor
            Time-delta values. Can have shape (...,) or (..., 1).
            Non-negative values (\u0394t \u2265 0). Negative values are automatically clamped.

        Returns
        -------
        torch.Tensor of shape (..., embed_dim)
            Temporal feature representations.
        """
        # Ensure float32 dtype
        dt = dt.to(torch.float32)

        # Apply logarithmic stabilization: \u03c4 = log(1 + max(\u0394t, 0))
        if self.use_log:
            tau = torch.log1p(torch.clamp(dt, min=0.0))
        else:
            tau = torch.clamp(dt, min=0.0)

        # Expand trailing singleton dimension if needed for vector multiplication
        has_trailing_dim = (tau.dim() > 0 and tau.shape[-1] == 1)
        if not has_trailing_dim:
            tau_expanded = tau.unsqueeze(-1)  # (..., 1)
        else:
            tau_expanded = tau

        if self.mode == "time2vec":
            # Linear component: \u03c4 * w0 + b0 -> shape (..., 1)
            lin = tau_expanded * self.w0 + self.b0
            # Periodic components: sin(\u03c4 * w + b) -> shape (..., embed_dim - 1)
            per = self.periodic_act(tau_expanded * self.w + self.b)
            # Concatenate along feature dimension
            out = torch.cat([lin, per], dim=-1)

        elif self.mode == "log":
            out = self.mlp(tau_expanded)

        elif self.mode == "combined":
            lin = tau_expanded * self.w0 + self.b0
            per = self.periodic_act(tau_expanded * self.w + self.b)
            t2v_feat = torch.cat([lin, per], dim=-1)
            log_feat = self.mlp(tau_expanded)
            out = torch.cat([t2v_feat, log_feat], dim=-1)

        return out


# ==============================================================================
# 2. TimeAwareDeepLog: Joint Sequence & Temporal DeepLog Architecture
# ==============================================================================


class TimeAwareDeepLog(nn.Module):
    """
    TimeAwareDeepLog: Neural architecture modeling joint log sequence and temporal dynamics.

    Models the joint probability factorized via:
        P(e_t, \u0394t_t | history_{<t}) = P(e_t | history_{<t}) * P(\u0394t_t | history_{<t})

    Architecture:
    1. Event Embedding: Maps discrete log event IDs e_i into dense vector v_i.
    2. Temporal Encoder: Maps continuous time-deltas \u0394t_i into dense vector t_i (Time2Vec / log).
    3. Input Fusion: Projects concatenated [v_i; t_i] through a normalization layer.
    4. Recurrent Backbone: Multi-layer stacked LSTM tracking temporal state transitions.
    5. Dual Prediction Heads:
       - Event Head: Categorical distribution P(e_t | context) via LogSoftmax.
       - Temporal Head: Continuous log-normal distribution (\u03bc_t, \u03c3_t) for \u03c4_t = log(1 + \u0394t_t).
    """

    def __init__(
        self,
        vocab_size: Optional[int] = None,
        input_size: Optional[int] = None,
        embed_dim: int = 64,
        time_dim: int = 16,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        time_mode: str = "time2vec",
        use_log_time: bool = True,
        output_size: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        vocab_size : int, optional
            Number of distinct log event IDs. Can also be passed as `input_size`.
        input_size : int, optional
            DeepLog API compatibility alias for `vocab_size`.
        embed_dim : int, default=64
            Dimension of learned discrete event token embeddings.
        time_dim : int, default=16
            Dimension of continuous time-delta embeddings from TimeDeltaEncoder.
        hidden_size : int, default=128
            Dimension of LSTM hidden state.
        num_layers : int, default=2
            Number of stacked LSTM layers.
        dropout : float, default=0.1
            Dropout probability between LSTM layers and in prediction heads.
        time_mode : str, default="time2vec"
            Temporal encoding mechanism: "time2vec", "log", or "combined".
        use_log_time : bool, default=True
            Whether to apply logarithmic compression to time deltas in TimeDeltaEncoder.
        output_size : int, optional
            DeepLog API compatibility alias for vocabulary prediction dimension.
        """
        super().__init__()

        effective_vocab = vocab_size if vocab_size is not None else input_size
        if effective_vocab is None:
            raise ValueError("Either vocab_size or input_size must be specified.")

        self.vocab_size = effective_vocab
        self.input_size = effective_vocab
        self.output_size = output_size if output_size is not None else effective_vocab
        self.embed_dim = embed_dim
        self.time_dim = time_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.time_mode = time_mode
        self.use_log_time = use_log_time

        # 1. Embeddings
        self.event_embedding = nn.Embedding(self.vocab_size, self.embed_dim)
        self.time_encoder = TimeDeltaEncoder(
            embed_dim=self.time_dim,
            mode=self.time_mode,
            use_log=self.use_log_time,
        )

        # 2. Input Fusion & Projection
        fused_dim = self.embed_dim + self.time_dim
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 3. Stacked LSTM Backbone
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # 4. Event Prediction Head (Categorical next-event log-probabilities)
        self.event_head = nn.Linear(hidden_size, self.output_size)
        self.event_log_softmax = nn.LogSoftmax(dim=-1)

        # 5. Temporal Prediction Head (Log-Normal distribution: \u03bc, \u03c3)
        time_mlp_dim = max(32, hidden_size // 2)
        self.time_head = nn.Sequential(
            nn.Linear(hidden_size, time_mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(time_mlp_dim, 2),  # Produces (\u03bc, raw_\u03c3)
        )

    def forward(
        self,
        events: torch.Tensor,
        time_deltas: torch.Tensor,
        return_all_steps: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through TimeAwareDeepLog.

        Parameters
        ----------
        events : torch.Tensor
            Sequence of integer event IDs of shape (batch_size, seq_len) or (seq_len,).
        time_deltas : torch.Tensor
            Sequence of float time deltas of shape (batch_size, seq_len) or (seq_len,).
        return_all_steps : bool, default=False
            If True, returns predictions for all sequence time-steps.
            If False, returns predictions for the last time-step (next event/delta).

        Returns
        -------
        event_log_probs : torch.Tensor
            Log probabilities over the event vocabulary.
            Shape: (batch_size, output_size) if not return_all_steps else (batch_size, seq_len, output_size).
        time_mu : torch.Tensor
            Predicted mean \u03bc for log(1 + \u0394t).
            Shape: (batch_size, 1) if not return_all_steps else (batch_size, seq_len, 1).
        time_sigma : torch.Tensor
            Predicted standard deviation \u03c3 > 0 for log(1 + \u0394t).
            Shape: (batch_size, 1) if not return_all_steps else (batch_size, seq_len, 1).
        """
        if events.dim() == 1:
            events = events.unsqueeze(0)
        if time_deltas.dim() == 1:
            time_deltas = time_deltas.unsqueeze(0)

        # Embed discrete events and continuous time-deltas
        e_emb = self.event_embedding(events.to(torch.int64))  # (B, L, embed_dim)
        t_emb = self.time_encoder(time_deltas)                # (B, L, time_dim)

        # Fuse representations
        fused = torch.cat([e_emb, t_emb], dim=-1)             # (B, L, embed_dim + time_dim)
        x = self.fusion(fused)                                # (B, L, hidden_size)

        # LSTM state transition
        lstm_out, _ = self.lstm(x)                            # (B, L, hidden_size)

        # Select target time steps
        if return_all_steps:
            h = lstm_out                                      # (B, L, hidden_size)
        else:
            h = lstm_out[:, -1, :]                            # (B, hidden_size)

        # 1. Event Log-Probabilities
        event_logits = self.event_head(h)
        event_log_probs = self.event_log_softmax(event_logits)

        # 2. Temporal Distribution Parameters (\u03bc, \u03c3)
        time_params = self.time_head(h)
        time_mu = time_params[..., 0:1]
        # Ensure \u03c3 is strictly positive via softplus and numerical epsilon
        time_sigma = F.softplus(time_params[..., 1:2]) + 1e-4

        return event_log_probs, time_mu, time_sigma

    def compute_loss(
        self,
        events: torch.Tensor,
        time_deltas: torch.Tensor,
        target_events: torch.Tensor,
        target_deltas: torch.Tensor,
        alpha: float = 1.0,
        beta: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Computes the joint sequence-temporal negative log-likelihood loss:
            L_total = \u03b1 * L_event + \u03b2 * L_temporal

        Where:
            L_event = NLLLoss(event_log_probs, target_events)
            L_temporal = 0.5 * [ (y* - \u03bc)^2 / \u03c3^2 + 2 * log(\u03c3) + log(2\u03c0) ]
            with y* = log(1 + max(target_deltas, 0)).

        Parameters
        ----------
        events : torch.Tensor of shape (B, L)
            Context event ID history.
        time_deltas : torch.Tensor of shape (B, L)
            Context time-delta history.
        target_events : torch.Tensor of shape (B,)
            Ground truth next event IDs.
        target_deltas : torch.Tensor of shape (B,) or (B, 1)
            Ground truth next time-deltas.
        alpha : float, default=1.0
            Weight for event prediction loss.
        beta : float, default=1.0
            Weight for temporal prediction loss.

        Returns
        -------
        dict with keys: 'total_loss', 'event_loss', 'temporal_loss'
        """
        event_log_probs, time_mu, time_sigma = self.forward(events, time_deltas)

        # 1. Event NLL Loss
        target_events = target_events.to(torch.int64)
        event_loss = F.nll_loss(event_log_probs, target_events)

        # 2. Continuous Temporal Log-Normal NLL Loss
        target_deltas = target_deltas.to(torch.float32)
        if target_deltas.dim() == 1:
            target_deltas = target_deltas.unsqueeze(-1)

        y_target = torch.log1p(torch.clamp(target_deltas, min=0.0))
        variance = time_sigma.pow(2)
        temporal_nll = 0.5 * (
            ((y_target - time_mu).pow(2) / variance)
            + 2.0 * torch.log(time_sigma)
            + math.log(2.0 * math.pi)
        )
        temporal_loss = temporal_nll.mean()

        total_loss = alpha * event_loss + beta * temporal_loss

        return {
            "total_loss": total_loss,
            "event_loss": event_loss,
            "temporal_loss": temporal_loss,
        }

    def predict(
        self,
        events: torch.Tensor,
        time_deltas: torch.Tensor,
        k: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict Top-K next events and expected next time-delta.

        Parameters
        ----------
        events : torch.Tensor of shape (B, L)
            Context event sequence.
        time_deltas : torch.Tensor of shape (B, L)
            Context time-delta sequence.
        k : int, default=1
            Number of top event candidates to return.

        Returns
        -------
        topk_events : torch.Tensor of shape (B, k)
            Top-K predicted event IDs.
        topk_confidences : torch.Tensor of shape (B, k)
            Softmax probabilities corresponding to Top-K predictions.
        predicted_dt : torch.Tensor of shape (B, 1)
            Median predicted next time delta in raw seconds/scale: exp(\u03bc) - 1.
        predicted_sigma : torch.Tensor of shape (B, 1)
            Standard deviation \u03c3 of predicted log-time distribution.
        """
        self.eval()
        with torch.no_grad():
            event_log_probs, time_mu, time_sigma = self.forward(events, time_deltas)

        probs = event_log_probs.exp()
        topk_confidences, topk_events = probs.topk(k, dim=-1)

        # Invert log1p transform to produce expected time-delta: \u0394t = exp(\u03bc) - 1
        predicted_dt = torch.clamp(torch.expm1(time_mu), min=0.0)

        return topk_events, topk_confidences, predicted_dt, time_sigma

    def save(self, outfile: str):
        """Save model checkpoint to file."""
        checkpoint = {
            "state_dict": self.state_dict(),
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
            "time_dim": self.time_dim,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout_rate,
            "time_mode": self.time_mode,
            "use_log_time": self.use_log_time,
            "output_size": self.output_size,
        }
        torch.save(checkpoint, outfile)

    @classmethod
    def load(cls, infile: str, device=None, **kwargs) -> "TimeAwareDeepLog":
        """Load model checkpoint from file."""
        checkpoint = torch.load(infile, map_location=device)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            params = {
                "vocab_size": checkpoint.get("vocab_size", 30),
                "embed_dim": checkpoint.get("embed_dim", 64),
                "time_dim": checkpoint.get("time_dim", 16),
                "hidden_size": checkpoint.get("hidden_size", 128),
                "num_layers": checkpoint.get("num_layers", 2),
                "dropout": checkpoint.get("dropout", 0.1),
                "time_mode": checkpoint.get("time_mode", "time2vec"),
                "use_log_time": checkpoint.get("use_log_time", True),
                "output_size": checkpoint.get("output_size", None),
            }
        else:
            state_dict = checkpoint
            params = {
                "vocab_size": state_dict["event_embedding.weight"].shape[0],
                "embed_dim": state_dict["event_embedding.weight"].shape[1],
                "hidden_size": state_dict["lstm.weight_hh_l0"].shape[1],
                "output_size": state_dict["event_head.weight"].shape[0],
            }

        params.update(kwargs)
        model = cls(**params)
        model.load_state_dict(state_dict)

        if device is not None:
            model = model.to(device)

        return model


# ==============================================================================
# 3. TemporalAnomalyDetector: Joint Sequence & Temporal Anomaly Diagnostics
# ==============================================================================


@dataclass
class TemporalAnomalyResult:
    """
    Structured diagnostic result for a single event transition in a log stream.
    """
    is_anomaly: bool
    primary_type: str  # "NORMAL", "TIMEOUT", "LATENCY_SPIKE", "RETRY_STORM", "SEQUENCE_ANOMALY", "JOINT_ANOMALY"
    anomaly_types: List[str]  # e.g. ["TIMEOUT", "SEQUENCE_ANOMALY"]
    is_sequence_anomaly: bool
    is_temporal_anomaly: bool
    is_timeout: bool
    is_latency_spike: bool
    is_retry_storm: bool
    observed_event: int
    observed_delta: float
    expected_delta_mean: float
    expected_delta_std: float
    z_score: float
    sequence_nll: float
    event_confidence: float
    topk_predictions: List[int]
    joint_score: float
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert diagnostic result to serializable dictionary."""
        return {
            "is_anomaly": self.is_anomaly,
            "primary_type": self.primary_type,
            "anomaly_types": self.anomaly_types,
            "is_sequence_anomaly": self.is_sequence_anomaly,
            "is_temporal_anomaly": self.is_temporal_anomaly,
            "is_timeout": self.is_timeout,
            "is_latency_spike": self.is_latency_spike,
            "is_retry_storm": self.is_retry_storm,
            "observed_event": self.observed_event,
            "observed_delta": self.observed_delta,
            "expected_delta_mean": self.expected_delta_mean,
            "expected_delta_std": self.expected_delta_std,
            "z_score": self.z_score,
            "sequence_nll": self.sequence_nll,
            "event_confidence": self.event_confidence,
            "topk_predictions": self.topk_predictions,
            "joint_score": self.joint_score,
            "details": self.details,
        }


class TemporalAnomalyDetector:
    """
    TemporalAnomalyDetector: Evaluates joint sequence-temporal anomalies for DeepLog.

    Diagnoses:
    1. Sequence Anomalies:
       - Observed next event key e_{obs} \u2209 Top-K prediction candidates.
       - Sequence negative log-likelihood surprisal exceeds threshold.
    2. Temporal Anomalies:
       Given predicted log-normal distribution (\u03bc, \u03c3) on y = log(1 + \u0394t),
       the standardized residual Z-score is computed as:
           z = (log(1 + \u0394t_{obs}) - \u03bc) / \u03c3
       - Latency Spike: z \u2265 z_spike_threshold (e.g. z \u2265 3.0).
       - Timeout: z \u2265 z_timeout_threshold (e.g. z \u2265 5.0) OR \u0394t_{obs} \u2265 hard_timeout_seconds.
       - Retry Storm: z \u2264 z_retry_storm_threshold (e.g. z \u2264 -3.0) OR \u0394t_{obs} \u2264 min_retry_delta_seconds.
    3. Joint Anomaly Score:
       S_joint = w_seq * S_seq + w_time * |z|
    """

    def __init__(
        self,
        model: Optional[TimeAwareDeepLog] = None,
        top_k: int = 9,
        z_spike_threshold: float = 3.0,
        z_timeout_threshold: float = 5.0,
        hard_timeout_seconds: Optional[float] = None,
        z_retry_storm_threshold: float = -3.0,
        min_retry_delta_seconds: float = 0.005,
        min_retry_storm_count: int = 3,
        nll_threshold: Optional[float] = None,
        joint_weight_seq: float = 0.5,
        joint_weight_time: float = 0.5,
    ):
        """
        Parameters
        ----------
        model : TimeAwareDeepLog, optional
            Trained TimeAwareDeepLog model.
        top_k : int, default=9
            DeepLog top-k prediction candidate set size.
        z_spike_threshold : float, default=3.0
            Z-score threshold for classifying latency spikes.
        z_timeout_threshold : float, default=5.0
            Z-score threshold for classifying statistical timeouts.
        hard_timeout_seconds : float, optional
            Deterministic maximum time duration (e.g. 30.0s). Any \u0394t exceeding
            this is immediately flagged as a TIMEOUT regardless of model prediction.
        z_retry_storm_threshold : float, default=-3.0
            Z-score threshold for classifying abnormally fast/burst transitions.
        min_retry_delta_seconds : float, default=0.005
            Minimum time delta threshold (e.g. 5ms) representing rapid spinlock/retry bursts.
        min_retry_storm_count : int, default=3
            Consecutive sub-millisecond transitions needed to confirm a retry storm in traces.
        nll_threshold : float, optional
            Surprisal NLL threshold for sequence anomalies.
        joint_weight_seq : float, default=0.5
            Weight for sequence surprisal in continuous joint score.
        joint_weight_time : float, default=0.5
            Weight for temporal deviation in continuous joint score.
        """
        self.model = model
        self.top_k = top_k
        self.z_spike_threshold = float(z_spike_threshold)
        self.z_timeout_threshold = float(z_timeout_threshold)
        self.hard_timeout_seconds = (
            float(hard_timeout_seconds) if hard_timeout_seconds is not None else None
        )
        self.z_retry_storm_threshold = float(z_retry_storm_threshold)
        self.min_retry_delta_seconds = float(min_retry_delta_seconds)
        self.min_retry_storm_count = int(min_retry_storm_count)
        self.nll_threshold = (
            float(nll_threshold) if nll_threshold is not None else None
        )
        self.joint_weight_seq = float(joint_weight_seq)
        self.joint_weight_time = float(joint_weight_time)

    def diagnose_from_predictions(
        self,
        event_log_probs: Union[torch.Tensor, np.ndarray],
        time_mu: Union[torch.Tensor, float],
        time_sigma: Union[torch.Tensor, float],
        next_event: int,
        next_delta: float,
    ) -> TemporalAnomalyResult:
        """
        Diagnoses a single event transition using pre-computed model outputs.

        Parameters
        ----------
        event_log_probs : torch.Tensor or np.ndarray of shape (vocab_size,)
            Log-probabilities over event vocabulary.
        time_mu : torch.Tensor or float
            Predicted log-time mean \u03bc.
        time_sigma : torch.Tensor or float
            Predicted log-time standard deviation \u03c3.
        next_event : int
            Observed ground-truth event ID.
        next_delta : float
            Observed ground-truth time-delta in seconds.

        Returns
        -------
        TemporalAnomalyResult
        """
        if isinstance(event_log_probs, torch.Tensor):
            log_probs = event_log_probs.squeeze().detach().cpu().numpy()
        else:
            log_probs = np.asarray(event_log_probs, dtype=np.float32).squeeze()

        if isinstance(time_mu, torch.Tensor):
            mu_val = float(time_mu.squeeze().detach().cpu().item())
        else:
            mu_val = float(time_mu)

        if isinstance(time_sigma, torch.Tensor):
            sigma_val = float(time_sigma.squeeze().detach().cpu().item())
        else:
            sigma_val = float(time_sigma)

        # Ensure sigma is positive
        sigma_val = max(1e-4, sigma_val)
        next_event = int(next_event)
        next_delta = max(0.0, float(next_delta))

        # 1. Sequence Anomaly Analysis
        vocab_size = len(log_probs)
        k = min(self.top_k, vocab_size)
        topk_indices = np.argsort(log_probs)[::-1][:k].tolist()

        if 0 <= next_event < vocab_size:
            seq_nll = float(-log_probs[next_event])
            event_conf = float(np.exp(log_probs[next_event]))
        else:
            seq_nll = float("inf")
            event_conf = 0.0

        is_seq_anomaly = (next_event not in topk_indices)
        if self.nll_threshold is not None and seq_nll > self.nll_threshold:
            is_seq_anomaly = True

        # 2. Temporal Anomaly Analysis
        y_obs = math.log1p(next_delta)
        z_score = (y_obs - mu_val) / sigma_val
        expected_delta_mean = max(0.0, math.expm1(mu_val))

        # Check Timeout: hard threshold or high Z-score
        is_hard_timeout = (
            self.hard_timeout_seconds is not None
            and next_delta >= self.hard_timeout_seconds
        )
        is_stat_timeout = z_score >= self.z_timeout_threshold
        is_timeout = bool(is_hard_timeout or is_stat_timeout)

        # Check Latency Spike: statistically significant delay, but below timeout
        is_latency_spike = bool(
            z_score >= self.z_spike_threshold and not is_timeout
        )

        # Check Retry Storm: abnormally rapid arrival (negative Z-score or sub-millisecond burst)
        is_sub_delta_burst = bool(
            next_delta <= self.min_retry_delta_seconds
            and expected_delta_mean >= (self.min_retry_delta_seconds * 2.0)
        )
        is_retry_storm = bool(
            (z_score <= self.z_retry_storm_threshold) or is_sub_delta_burst
        )

        is_temporal_anomaly = bool(is_timeout or is_latency_spike or is_retry_storm)

        # 3. Anomaly Types Aggregation
        anomaly_types: List[str] = []
        if is_timeout:
            anomaly_types.append("TIMEOUT")
        if is_latency_spike:
            anomaly_types.append("LATENCY_SPIKE")
        if is_retry_storm:
            anomaly_types.append("RETRY_STORM")
        if is_seq_anomaly:
            anomaly_types.append("SEQUENCE_ANOMALY")

        is_joint_anomaly = bool(is_seq_anomaly and is_temporal_anomaly)
        is_overall_anomaly = bool(is_seq_anomaly or is_temporal_anomaly)

        # 4. Determine Primary Anomaly Type
        if is_timeout:
            primary_type = "TIMEOUT"
        elif is_retry_storm:
            primary_type = "RETRY_STORM"
        elif is_joint_anomaly:
            primary_type = "JOINT_ANOMALY"
        elif is_latency_spike:
            primary_type = "LATENCY_SPIKE"
        elif is_seq_anomaly:
            primary_type = "SEQUENCE_ANOMALY"
        else:
            primary_type = "NORMAL"

        # Continuous Joint Anomaly Score
        capped_nll = min(20.0, seq_nll) if not math.isinf(seq_nll) else 20.0
        joint_score = float(
            self.joint_weight_seq * capped_nll
            + self.joint_weight_time * max(0.0, abs(z_score))
        )

        details = {
            "hard_timeout_triggered": is_hard_timeout,
            "statistical_timeout_triggered": is_stat_timeout,
            "mu_log": mu_val,
            "sigma_log": sigma_val,
            "y_obs_log": y_obs,
        }

        return TemporalAnomalyResult(
            is_anomaly=is_overall_anomaly,
            primary_type=primary_type,
            anomaly_types=anomaly_types,
            is_sequence_anomaly=is_seq_anomaly,
            is_temporal_anomaly=is_temporal_anomaly,
            is_timeout=is_timeout,
            is_latency_spike=is_latency_spike,
            is_retry_storm=is_retry_storm,
            observed_event=next_event,
            observed_delta=next_delta,
            expected_delta_mean=expected_delta_mean,
            expected_delta_std=sigma_val,
            z_score=float(z_score),
            sequence_nll=seq_nll,
            event_confidence=event_conf,
            topk_predictions=topk_indices,
            joint_score=joint_score,
            details=details,
        )

    def diagnose_event(
        self,
        context_events: torch.Tensor,
        context_deltas: torch.Tensor,
        next_event: int,
        next_delta: float,
    ) -> TemporalAnomalyResult:
        """
        Diagnoses a single event transition given context history and trained model.

        Parameters
        ----------
        context_events : torch.Tensor of shape (seq_len,) or (1, seq_len)
            Historical event sequence.
        context_deltas : torch.Tensor of shape (seq_len,) or (1, seq_len)
            Historical time-delta sequence.
        next_event : int
            Observed candidate next event.
        next_delta : float
            Observed elapsed time-delta to next event.

        Returns
        -------
        TemporalAnomalyResult
        """
        if self.model is None:
            raise RuntimeError(
                "A trained TimeAwareDeepLog model must be provided to diagnose_event."
            )

        self.model.eval()
        if context_events.dim() == 1:
            context_events = context_events.unsqueeze(0)
        if context_deltas.dim() == 1:
            context_deltas = context_deltas.unsqueeze(0)

        # Move to model device
        device = next(self.model.parameters()).device
        context_events = context_events.to(device)
        context_deltas = context_deltas.to(device)

        with torch.no_grad():
            event_log_probs, time_mu, time_sigma = self.model.forward(
                context_events, context_deltas
            )

        return self.diagnose_from_predictions(
            event_log_probs=event_log_probs[0],
            time_mu=time_mu[0],
            time_sigma=time_sigma[0],
            next_event=next_event,
            next_delta=next_delta,
        )

    def evaluate_batch(
        self,
        events: torch.Tensor,
        time_deltas: torch.Tensor,
        target_events: torch.Tensor,
        target_deltas: torch.Tensor,
    ) -> List[TemporalAnomalyResult]:
        """
        Vectorized evaluation over a batch of transitions.

        Parameters
        ----------
        events : torch.Tensor of shape (batch_size, seq_len)
        time_deltas : torch.Tensor of shape (batch_size, seq_len)
        target_events : torch.Tensor of shape (batch_size,)
        target_deltas : torch.Tensor of shape (batch_size,)

        Returns
        -------
        List[TemporalAnomalyResult] of length batch_size
        """
        if self.model is None:
            raise RuntimeError("Model required for evaluate_batch.")

        self.model.eval()
        device = next(self.model.parameters()).device
        events = events.to(device)
        time_deltas = time_deltas.to(device)

        with torch.no_grad():
            event_log_probs, time_mu, time_sigma = self.model.forward(
                events, time_deltas
            )

        batch_size = events.size(0)
        results: List[TemporalAnomalyResult] = []

        target_events_cpu = target_events.cpu().numpy()
        target_deltas_cpu = target_deltas.cpu().numpy()

        for i in range(batch_size):
            res = self.diagnose_from_predictions(
                event_log_probs=event_log_probs[i],
                time_mu=time_mu[i],
                time_sigma=time_sigma[i],
                next_event=int(target_events_cpu[i]),
                next_delta=float(target_deltas_cpu[i]),
            )
            results.append(res)

        return results

    def evaluate_trace(
        self,
        trace_events: List[int],
        trace_deltas: List[float],
        window_size: int = 10,
    ) -> List[TemporalAnomalyResult]:
        """
        Sliding-window evaluation across a full log session or execution trace.

        Tracks consecutive micro-deltas to detect sustained retry storm cascades.

        Parameters
        ----------
        trace_events : List[int]
            Chronological sequence of log event IDs.
        trace_deltas : List[float]
            Chronological sequence of elapsed time-deltas.
        window_size : int, default=10
            Context window size.

        Returns
        -------
        List[TemporalAnomalyResult] for all steps >= window_size.
        """
        if len(trace_events) != len(trace_deltas):
            raise ValueError("trace_events and trace_deltas must have identical lengths.")
        if len(trace_events) <= window_size:
            return []

        results: List[TemporalAnomalyResult] = []
        consecutive_micro_deltas = 0

        for t in range(window_size, len(trace_events)):
            ctx_ev = torch.tensor(
                trace_events[t - window_size : t], dtype=torch.long
            )
            ctx_dt = torch.tensor(
                trace_deltas[t - window_size : t], dtype=torch.float32
            )
            next_ev = trace_events[t]
            next_dt = trace_deltas[t]

            res = self.diagnose_event(
                context_events=ctx_ev,
                context_deltas=ctx_dt,
                next_event=next_ev,
                next_delta=next_dt,
            )

            # Track multi-step consecutive retry storm cascades
            if next_dt <= self.min_retry_delta_seconds:
                consecutive_micro_deltas += 1
            else:
                consecutive_micro_deltas = 0

            if consecutive_micro_deltas >= self.min_retry_storm_count:
                res.is_retry_storm = True
                res.is_temporal_anomaly = True
                res.is_anomaly = True
                if "RETRY_STORM" not in res.anomaly_types:
                    res.anomaly_types.append("RETRY_STORM")
                res.primary_type = "RETRY_STORM"
                res.details["consecutive_micro_deltas"] = consecutive_micro_deltas

            results.append(res)

        return results


__all__ = [
    "TimeDeltaEncoder",
    "TimeAwareDeepLog",
    "TemporalAnomalyDetector",
    "TemporalAnomalyResult",
]
