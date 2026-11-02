"""
DeepLog Root Cause Analysis (RCA) & Explainable AI (XAI) Engine
===============================================================
Provides attention rollout, gradient-based saliency (Gradient \times Input),
and Integrated Gradients attribution for LogTransformer and attention-based sequence models.
Pinpoints specific log events and transitions triggering anomaly alerts with
interpretable, normalized diagnostic explanations.
"""

import math
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer import LogTransformer
from .dual_head import DeepLogDualHead
from .deeplog import DeepLog


@dataclass
class RootCauseExplanation:
    """
    Structured Root Cause Analysis (RCA) Explanation for DeepLog.

    Represents token-level and transition-level attribution metrics explaining
    why a sequence or transition was flagged as anomalous.

    Attributes
    ----------
    sequence : List[int]
        Input log event ID sequence.
    token_indices : List[int]
        Positional indices [0, 1, ..., T-1] in the sequence.
    event_ids : List[int]
        List of event IDs corresponding to each token.
    attributions : np.ndarray
        Normalized attribution scores in [0.0, 1.0] for each token, summing to 1.0.
    attribution_percentages : np.ndarray
        Attribution scores expressed as percentages in [0.0, 100.0], summing to 100.0%.
    anomaly_score : float
        Surprise, NLL, or anomaly probability score triggering the RCA.
    is_anomalous : bool
        Flag indicating if the sequence or target transition violated anomaly criteria.
    anomaly_trigger_points : List[int]
        Positional indices identified as anomaly triggers or anomalous events.
    ranked_events : List[Dict[str, Any]]
        List of event attributions ranked descending by attribution score.
        Each entry has: rank, token_index, event_id, event_name, attribution, attribution_pct, is_trigger.
    ranked_transitions : List[Dict[str, Any]]
        List of adjacent transitions ranked descending by combined attribution score.
        Each entry has: rank, from_index, to_index, from_event, to_event, transition, attribution, attribution_pct.
    method : str
        Attribution method utilized ("rollout", "grad_input", "integrated_gradients", "pooling", "combined").
    target_event : Optional[int]
        Target next event evaluated, if applicable.
    event_names : Optional[List[str]]
        Human-readable template strings or names for each event.
    metadata : Dict[str, Any]
        Additional diagnostic information (e.g., top predictions, completeness delta).
    """

    sequence: List[int]
    token_indices: List[int]
    event_ids: List[int]
    attributions: np.ndarray
    attribution_percentages: np.ndarray
    anomaly_score: float
    is_anomalous: bool
    anomaly_trigger_points: List[int]
    ranked_events: List[Dict[str, Any]]
    ranked_transitions: List[Dict[str, Any]]
    method: str
    target_event: Optional[int] = None
    event_names: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def top_k(self, k: int = 3) -> List[Dict[str, Any]]:
        """Return the top-k highest attributed tokens."""
        return self.ranked_events[:k]

    def top_transitions(self, k: int = 3) -> List[Dict[str, Any]]:
        """Return the top-k highest attributed transitions."""
        return self.ranked_transitions[:k]

    def to_dict(self) -> Dict[str, Any]:
        """Convert explanation to a JSON-serializable dictionary."""
        data = asdict(self)
        data["attributions"] = self.attributions.tolist()
        data["attribution_percentages"] = self.attribution_percentages.tolist()
        return data

    def summary(self, ascii_bars: bool = True, max_width: int = 25) -> str:
        """
        Format a human-readable text explanation for SRE and DevOps diagnostic reports.

        Parameters
        ----------
        ascii_bars : bool, default=True
            Whether to display an ASCII distribution bar chart.
        max_width : int, default=25
            Maximum character width of ASCII bars.

        Returns
        -------
        str
            Formatted summary report.
        """
        lines = []
        status_str = "ANOMALOUS" if self.is_anomalous else "NORMAL"
        lines.append("=" * 70)
        lines.append("               DEEPLOG ROOT CAUSE EXPLANATION (XAI)               ")
        lines.append("=" * 70)
        lines.append(
            f"Status: {status_str} | Anomaly Score: {self.anomaly_score:.4f} | Method: {self.method}"
        )
        lines.append(
            f"Sequence Length: {len(self.sequence)} tokens | Trigger Positions: {self.anomaly_trigger_points}"
        )
        if self.target_event is not None:
            t_name = (
                self.metadata.get("target_event_name", f"E{self.target_event}")
                if self.metadata
                else f"E{self.target_event}"
            )
            lines.append(f"Target Next Event: ID {self.target_event} ({t_name})")

        lines.append("\n--- Top Contributing Log Events (Root Cause Candidates) ---")
        for ev in self.top_k(k=min(5, len(self.ranked_events))):
            trig_mark = " [TRIGGER]" if ev["is_trigger"] else ""
            lines.append(
                f"Rank {ev['rank']}: Index {ev['token_index']} | Event {ev['event_id']} "
                f"(\"{ev['event_name']}\") -> {ev['attribution_pct']:.2f}% attribution{trig_mark}"
            )

        if self.ranked_transitions:
            lines.append("\n--- Top Contributing Transitions ---")
            for tr in self.top_transitions(k=min(3, len(self.ranked_transitions))):
                lines.append(
                    f"Rank {tr['rank']}: Step ({tr['from_index']} -> {tr['to_index']}) | "
                    f"({tr['from_event']} -> {tr['to_event']}) -> {tr['attribution_pct']:.2f}%"
                )

        if ascii_bars and len(self.sequence) > 0:
            lines.append("\n--- Sequence Attribution Distribution ---")
            max_pct = max(float(np.max(self.attribution_percentages)), 1e-6)
            for idx, (eid, pct) in enumerate(zip(self.sequence, self.attribution_percentages)):
                ename = self.event_names[idx] if self.event_names else f"E{eid}"
                bar_len = int(round((pct / max_pct) * max_width))
                bar = "#" * bar_len
                is_trig = idx in self.anomaly_trigger_points
                trig_tag = " <-- [TRIGGER]" if is_trig else ""
                lines.append(
                    f"[{idx:2d}] Event {eid:2d} ({ename:<15}): {bar:<{max_width}} ({pct:5.1f}%){trig_tag}"
                )

        lines.append("=" * 70)
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"<RootCauseExplanation status={'ANOMALY' if self.is_anomalous else 'NORMAL'} "
            f"score={self.anomaly_score:.3f} method={self.method} "
            f"triggers={self.anomaly_trigger_points}>"
        )

    def __str__(self) -> str:
        return self.summary(ascii_bars=False)


class AttentionAttributor:
    """
    Attention-based and Gradient-based Attribution Engine for DeepLog.

    Computes:
    1. Multi-head Attention Rollout across transformer blocks (Abnar & Zuidema, 2020),
       accounting for residual connections and causal attention masking.
    2. Gradient-based Saliency (Gradient * Input) on continuous token embeddings.
    3. Integrated Gradients (Sundararajan et al., 2017) satisfying Completeness and Monotonicity.
    4. Additive Self-Attention Pooling weights (for dual-head models).
    5. Combined hybrid attributions (rollout + gradient saliency).
    """

    def __init__(
        self,
        model: nn.Module,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ):
        """
        Initialize AttentionAttributor.

        Parameters
        ----------
        model : nn.Module
            LogTransformer, DeepLogDualHead, or attention-based sequence model.
        device : torch.device or str, optional
            Target device for computations.
        """
        self.model = model
        self.device = (
            torch.device(device)
            if device is not None
            else next(model.parameters()).device
            if list(model.parameters())
            else torch.device("cpu")
        )
        self.model.to(self.device)
        self.model.eval()

    def _to_tensor(self, sequence: Union[torch.Tensor, List[int], np.ndarray]) -> torch.Tensor:
        """Helper to ensure input sequence is a 2D LongTensor on device."""
        if isinstance(sequence, torch.Tensor):
            t = sequence.to(device=self.device, dtype=torch.long)
        else:
            t = torch.tensor(sequence, dtype=torch.long, device=self.device)
        if t.dim() == 1:
            t = t.unsqueeze(0)
        return t

    def _get_embeddings(self, seq_t: torch.Tensor) -> torch.Tensor:
        """Extract continuous token embeddings for sequence."""
        if hasattr(self.model, "get_input_embeddings"):
            emb_layer = self.model.get_input_embeddings()
            if emb_layer is not None:
                return emb_layer(seq_t)

        if hasattr(self.model, "token_embedding") and isinstance(
            self.model.token_embedding, nn.Embedding
        ):
            return self.model.token_embedding(seq_t)
        elif hasattr(self.model, "embedding") and isinstance(
            self.model.embedding, nn.Embedding
        ):
            return self.model.embedding(seq_t)

        # Fallback to one-hot embedding for models without dense embedding layer
        vocab_size = getattr(self.model, "vocab_size", getattr(self.model, "input_size", None))
        if vocab_size is not None:
            return F.one_hot(seq_t, vocab_size).to(torch.float32)

        raise AttributeError("Model has no detectable embedding layer or vocabulary size.")

    def compute_attention_rollout(
        self,
        sequence: Union[torch.Tensor, List[int], np.ndarray],
        alpha: float = 0.5,
        head_fusion: str = "mean",
        discard_ratio: float = 0.0,
        query_idx: int = -1,
        return_full_matrix: bool = False,
    ) -> np.ndarray:
        """
        Compute Attention Attribution Rollout across all self-attention layers.

        Implements Abnar & Zuidema (2020) attention rollout:
            A_tilde_l = alpha * A_l + (1 - alpha) * I
            R_0 = A_tilde_0
            R_l = A_tilde_l @ R_{l-1}

        Parameters
        ----------
        sequence : Union[torch.Tensor, List[int], np.ndarray]
            Input log event sequence of length T.
        alpha : float, default=0.5
            Weight of attention vs. residual identity connection (alpha=1.0 is pure attention,
            alpha=0.0 is pure identity).
        head_fusion : str, default="mean"
            Strategy for aggregating multi-head attention weights ("mean", "max", or "min").
        discard_ratio : float, default=0.0
            Fraction of lowest attention weights to prune for noise suppression (0.0 to 1.0).
        query_idx : int, default=-1
            Index of query token whose incoming attention flow is evaluated (default: last token).
        return_full_matrix : bool, default=False
            If True, return the full (T, T) rollout matrix.
            If False, return the 1D attribution vector to query_idx (shape: (T,)).

        Returns
        -------
        np.ndarray
            Rollout attribution vector (summing to 1.0) or full rollout matrix.
        """
        seq_t = self._to_tensor(sequence)
        seq_len = seq_t.size(1)

        if seq_len == 1:
            if return_full_matrix:
                return np.ones((1, 1), dtype=np.float64)
            return np.ones(1, dtype=np.float64)

        # Extract layer-wise attention matrices
        if isinstance(self.model, LogTransformer):
            with torch.no_grad():
                _, attentions = self.model(
                    seq_t, return_attention=True, average_attn_weights=False
                )
        elif hasattr(self.model, "forward") and "return_attention" in self.model.forward.__code__.co_varnames:
            with torch.no_grad():
                out = self.model(seq_t, return_attention=True)
                attentions = out[-1]
                if isinstance(attentions, torch.Tensor):
                    attentions = [attentions]
        else:
            raise NotImplementedError(
                f"Attention rollout is only supported for LogTransformer or attention models, got {type(self.model).__name__}"
            )

        if not attentions:
            raise ValueError("No attention weights captured from model forward pass.")

        device = attentions[0].device
        identity = torch.eye(seq_len, device=device, dtype=torch.float32)
        rollout: Optional[torch.Tensor] = None

        for layer_attn in attentions:
            # Expected shape: (1, nhead, seq_len, seq_len) or (seq_len, seq_len)
            if layer_attn.dim() == 4:
                layer_attn = layer_attn.squeeze(0)  # (nhead, seq_len, seq_len)

            if layer_attn.dim() == 3:
                if head_fusion == "mean":
                    fused = layer_attn.mean(dim=0)
                elif head_fusion == "max":
                    fused = layer_attn.max(dim=0).values
                elif head_fusion == "min":
                    fused = layer_attn.min(dim=0).values
                else:
                    raise ValueError(f"Unknown head_fusion strategy: {head_fusion}")
            else:
                fused = layer_attn

            # Discard lowest percentile if requested
            if discard_ratio > 0.0:
                flat_k = int(fused.size(-1) * discard_ratio)
                if flat_k > 0:
                    threshold = torch.quantile(fused, discard_ratio)
                    fused = torch.where(fused < threshold, torch.zeros_like(fused), fused)

            # Fuse with residual identity connection
            a_tilde = alpha * fused + (1.0 - alpha) * identity

            # Row normalization: ensure each row sums to 1.0
            row_sums = a_tilde.sum(dim=-1, keepdim=True)
            a_tilde = a_tilde / torch.clamp(row_sums, min=1e-12)

            # Rollout accumulation
            if rollout is None:
                rollout = a_tilde
            else:
                rollout = torch.matmul(a_tilde, rollout)

        rollout_mat = rollout.detach().cpu().numpy().astype(np.float64)

        if return_full_matrix:
            return rollout_mat

        # Resolve query token index
        q = query_idx if query_idx >= 0 else seq_len + query_idx
        if not (0 <= q < seq_len):
            raise IndexError(f"query_idx {query_idx} out of bounds for sequence length {seq_len}")

        attribution = rollout_mat[q, :].copy()
        # Guarantee non-negativity and sum-to-one
        attribution = np.maximum(attribution, 0.0)
        total = attribution.sum()
        if total > 0:
            attribution = attribution / total
        else:
            attribution = np.ones(seq_len) / seq_len

        return attribution

    def compute_gradient_saliency(
        self,
        sequence: Union[torch.Tensor, List[int], np.ndarray],
        target_class: Optional[int] = None,
        norm: str = "dot",
        normalize: bool = True,
        abs_value: bool = True,
    ) -> np.ndarray:
        """
        Compute Gradient * Input saliency on token continuous embeddings.

        Parameters
        ----------
        sequence : Union[torch.Tensor, List[int], np.ndarray]
            Input log sequence.
        target_class : int, optional
            Target next event ID or class index whose logit/loss is differentiated.
            If None, evaluates the anomaly output or top predicted token.
        norm : str, default="dot"
            Reduction norm across embedding dimension ("dot", "abs_dot", "l1", "l2", "grad_norm").
        normalize : bool, default=True
            Whether to normalize token attributions to sum to 1.0.
        abs_value : bool, default=True
            Whether to take absolute value of attributions before normalization.

        Returns
        -------
        np.ndarray
            Token-level saliency attribution scores of shape (T,).
        """
        seq_t = self._to_tensor(sequence)
        seq_len = seq_t.size(1)

        embeddings = self._get_embeddings(seq_t)
        emb = embeddings.clone().detach().requires_grad_(True)

        scalar_target = self._evaluate_model_target(emb, target_class=target_class)

        self.model.zero_grad()
        scalar_target.backward()

        if emb.grad is None:
            raise RuntimeError("Gradient computation failed; embedding grad is None.")

        e_val = emb.detach().squeeze(0)  # (T, D)
        g_val = emb.grad.detach().squeeze(0)  # (T, D)

        if norm == "dot":
            raw_scores = (e_val * g_val).sum(dim=-1)
        elif norm == "abs_dot":
            raw_scores = torch.abs((e_val * g_val).sum(dim=-1))
        elif norm == "l1":
            raw_scores = torch.abs(e_val * g_val).sum(dim=-1)
        elif norm == "l2":
            raw_scores = torch.sqrt(torch.sum((e_val * g_val) ** 2, dim=-1) + 1e-12)
        elif norm == "grad_norm":
            raw_scores = torch.sqrt(torch.sum(g_val ** 2, dim=-1) + 1e-12)
        else:
            raise ValueError(f"Unknown reduction norm: {norm}")

        scores = raw_scores.cpu().numpy().astype(np.float64)

        if abs_value:
            scores = np.abs(scores)
        else:
            scores = np.maximum(scores, 0.0)

        if normalize:
            total = scores.sum()
            if total > 0:
                scores = scores / total
            else:
                scores = np.ones(seq_len, dtype=np.float64) / seq_len

        return scores

    def compute_integrated_gradients(
        self,
        sequence: Union[torch.Tensor, List[int], np.ndarray],
        target_class: Optional[int] = None,
        steps: int = 50,
        baseline: Optional[torch.Tensor] = None,
        normalize: bool = True,
        return_convergence_delta: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, float]]:
        """
        Compute Integrated Gradients (Sundararajan et al., 2017) on continuous embeddings.

        Satisfies the Completeness Axiom: sum(IG) ~ F(x) - F(x').

        Parameters
        ----------
        sequence : Union[torch.Tensor, List[int], np.ndarray]
            Input log sequence.
        target_class : int, optional
            Target class index whose scalar output is differentiated.
        steps : int, default=50
            Number of Riemann path interpolation steps.
        baseline : torch.Tensor, optional
            Reference baseline embedding. Defaults to all-zeros embedding.
        normalize : bool, default=True
            Whether to normalize token attributions to sum to 1.0.
        return_convergence_delta : bool, default=False
            Whether to return the completeness approximation error |sum(IG) - (F(x) - F(x'))|.

        Returns
        -------
        np.ndarray or Tuple[np.ndarray, float]
            Token attributions (and optionally convergence error delta).
        """
        seq_t = self._to_tensor(sequence)
        seq_len = seq_t.size(1)

        input_emb = self._get_embeddings(seq_t).clone().detach()

        if baseline is None:
            base_emb = torch.zeros_like(input_emb)
        else:
            base_emb = baseline.to(self.device)
            if base_emb.dim() == 2:
                base_emb = base_emb.unsqueeze(0)

        # Evaluate target scalar at input and baseline
        with torch.no_grad():
            f_input = self._evaluate_model_target(input_emb, target_class=target_class).item()
            f_base = self._evaluate_model_target(base_emb, target_class=target_class).item()
        delta_f = f_input - f_base

        total_grads = torch.zeros_like(input_emb)

        for step in range(1, steps + 1):
            alpha = step / steps
            interpolated = (base_emb + alpha * (input_emb - base_emb)).clone().detach().requires_grad_(True)
            scalar = self._evaluate_model_target(interpolated, target_class=target_class)
            self.model.zero_grad()
            scalar.backward()
            total_grads += interpolated.grad.detach()

        avg_grads = total_grads / steps
        ig = (input_emb - base_emb) * avg_grads  # (1, T, D)
        unnorm_scores = ig.sum(dim=-1).squeeze(0).cpu().numpy().astype(np.float64)

        completeness_delta = float(abs(unnorm_scores.sum() - delta_f))

        if normalize:
            scores = np.abs(unnorm_scores)
            total = scores.sum()
            if total > 0:
                scores = scores / total
            else:
                scores = np.ones(seq_len, dtype=np.float64) / seq_len
        else:
            scores = unnorm_scores

        if return_convergence_delta:
            return scores, completeness_delta
        return scores

    def compute_pooling_attention(
        self,
        sequence: Union[torch.Tensor, List[int], np.ndarray],
    ) -> np.ndarray:
        """
        Extract direct sequence self-attention pooling weights for DeepLogDualHead.

        Parameters
        ----------
        sequence : Union[torch.Tensor, List[int], np.ndarray]
            Input log sequence.

        Returns
        -------
        np.ndarray
            Attention pooling weights over sequence positions (summing to 1.0).
        """
        seq_t = self._to_tensor(sequence)
        seq_len = seq_t.size(1)

        if isinstance(self.model, DeepLogDualHead):
            with torch.no_grad():
                out = self.model(seq_t, return_attention=True)
                attn_weights = out[2]  # shape: (1, seq_len)
                if attn_weights is not None:
                    weights = attn_weights.squeeze(0).cpu().numpy().astype(np.float64)
                    return weights / np.maximum(weights.sum(), 1e-12)

        # Fallback to uniform if model does not feature attention pooling
        return np.ones(seq_len, dtype=np.float64) / seq_len

    def attribute(
        self,
        sequence: Union[torch.Tensor, List[int], np.ndarray],
        method: str = "rollout",
        target_event: Optional[int] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Unified attribution interface supporting rollout, grad_input, integrated_gradients,
        pooling, and combined strategies.

        Parameters
        ----------
        sequence : Union[torch.Tensor, List[int], np.ndarray]
            Input sequence of log event IDs.
        method : str, default="rollout"
            Attribution method: "rollout", "grad_input", "integrated_gradients", "pooling", or "combined".
        target_event : int, optional
            Target next event ID or prediction class.
        **kwargs : optional
            Method-specific hyper-parameters (alpha, head_fusion, steps, lambda_weight, etc.).

        Returns
        -------
        np.ndarray
            1D array of normalized token attributions summing to 1.0.
        """
        m = method.lower()
        if m in ("rollout", "attention_rollout"):
            return self.compute_attention_rollout(
                sequence,
                alpha=kwargs.get("alpha", 0.5),
                head_fusion=kwargs.get("head_fusion", "mean"),
                discard_ratio=kwargs.get("discard_ratio", 0.0),
                query_idx=kwargs.get("query_idx", -1),
            )
        elif m in ("grad_input", "gradient_saliency", "saliency"):
            return self.compute_gradient_saliency(
                sequence,
                target_class=target_event,
                norm=kwargs.get("norm", "dot"),
                normalize=True,
                abs_value=kwargs.get("abs_value", True),
            )
        elif m in ("integrated_gradients", "ig"):
            return self.compute_integrated_gradients(
                sequence,
                target_class=target_event,
                steps=kwargs.get("steps", 50),
                baseline=kwargs.get("baseline", None),
                normalize=True,
            )
        elif m in ("pooling", "attention_pooling"):
            return self.compute_pooling_attention(sequence)
        elif m in ("combined", "hybrid"):
            lambda_weight = kwargs.get("lambda_weight", 0.5)
            # Rollout + Gradient
            s_roll = self.compute_attention_rollout(
                sequence,
                alpha=kwargs.get("alpha", 0.5),
                head_fusion=kwargs.get("head_fusion", "mean"),
            )
            s_grad = self.compute_gradient_saliency(
                sequence,
                target_class=target_event,
                norm=kwargs.get("norm", "dot"),
            )
            combined = lambda_weight * s_roll + (1.0 - lambda_weight) * s_grad
            total = combined.sum()
            return combined / total if total > 0 else np.ones_like(combined) / len(combined)
        else:
            raise ValueError(f"Unknown attribution method: {method}")

    def _evaluate_model_target(
        self,
        inputs_embeds: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> torch.Tensor:
        """Helper to run model forward with continuous embeddings and return scalar target."""
        if isinstance(self.model, DeepLogDualHead):
            token_logits, seq_prob = self.model(inputs_embeds=inputs_embeds)
            if target_class is not None:
                return token_logits[0, target_class]
            else:
                return seq_prob[0, 0]

        elif isinstance(self.model, LogTransformer):
            logits = self.model(inputs_embeds=inputs_embeds)
            if target_class is not None:
                return logits[0, target_class]
            else:
                return logits.max(dim=-1).values[0]

        elif isinstance(self.model, DeepLog):
            logits = self.model(inputs_embeds=inputs_embeds)
            if target_class is not None:
                return logits[0, target_class]
            else:
                return logits.max(dim=-1).values[0]

        else:
            # Generic model
            out = self.model(inputs_embeds=inputs_embeds)
            if isinstance(out, tuple):
                out = out[0]
            if out.dim() == 2 and target_class is not None:
                return out[0, target_class]
            elif out.dim() == 2:
                return out.max(dim=-1).values[0]
            elif out.dim() == 1:
                return out[0]
            return out.sum()


class RootCauseAnalyzer:
    """
    Root Cause Analysis (RCA) Engine for DeepLog Anomalies.

    Identifies and ranks the specific log events and transitions that contributed
    most heavily to an anomaly detection alert, producing a structured,
    interpretable RootCauseExplanation.
    """

    def __init__(
        self,
        model: nn.Module,
        attributor: Optional[AttentionAttributor] = None,
        vocab_mapping: Optional[Dict[int, str]] = None,
        device: Optional[Union[str, torch.device]] = None,
        default_method: str = "rollout",
        anomaly_threshold: float = 0.5,
        trigger_threshold_factor: float = 1.5,
    ):
        """
        Initialize RootCauseAnalyzer.

        Parameters
        ----------
        model : nn.Module
            Trained DeepLog, DeepLogDualHead, or LogTransformer model.
        attributor : AttentionAttributor, optional
            Pre-configured attributor instance. Created automatically if omitted.
        vocab_mapping : Dict[int, str], optional
            Dictionary mapping integer event IDs to log template strings or descriptive names.
        device : torch.device or str, optional
            Target device.
        default_method : str, default="rollout"
            Default attribution algorithm ("rollout", "grad_input", "integrated_gradients", "combined").
        anomaly_threshold : float, default=0.5
            Decision boundary threshold for flagging sequences as anomalous.
        trigger_threshold_factor : float, default=1.5
            Multiplicative factor above uniform attribution (1/T) to designate a token
            as an anomaly trigger point.
        """
        self.model = model
        self.device = (
            torch.device(device)
            if device is not None
            else next(model.parameters()).device
            if list(model.parameters())
            else torch.device("cpu")
        )
        self.attributor = attributor or AttentionAttributor(model=model, device=self.device)
        self.vocab_mapping = vocab_mapping or {}
        self.default_method = default_method
        self.anomaly_threshold = anomaly_threshold
        self.trigger_threshold_factor = trigger_threshold_factor

    def analyze(
        self,
        sequence: Union[List[int], np.ndarray, torch.Tensor],
        next_event: Optional[int] = None,
        method: Optional[str] = None,
        top_k: int = 5,
        anomaly_threshold: Optional[float] = None,
        **kwargs,
    ) -> RootCauseExplanation:
        """
        Analyze an anomalous sequence and produce a structured root cause explanation.

        Parameters
        ----------
        sequence : Union[List[int], np.ndarray, torch.Tensor]
            Input log event sequence [e_0, e_1, ..., e_{T-1}].
        next_event : int, optional
            The actual next event observed that triggered the anomaly detection alert.
        method : str, optional
            Attribution method override (default: self.default_method).
        top_k : int, default=5
            Number of top contributing events to evaluate.
        anomaly_threshold : float, optional
            Override for anomaly threshold.
        **kwargs : optional
            Additional parameters passed to Attributor.attribute.

        Returns
        -------
        RootCauseExplanation
            Structured diagnostic explanation object with ranked events, transitions,
            and attribution percentages.
        """
        if isinstance(sequence, torch.Tensor):
            seq_list = [int(x) for x in sequence.detach().cpu().flatten().tolist()]
        elif isinstance(sequence, np.ndarray):
            seq_list = [int(x) for x in sequence.flatten().tolist()]
        else:
            seq_list = [int(x) for x in sequence]

        seq_len = len(seq_list)
        if seq_len == 0:
            raise ValueError("Sequence cannot be empty.")

        threshold = anomaly_threshold if anomaly_threshold is not None else self.anomaly_threshold
        seq_t = torch.tensor([seq_list], dtype=torch.long, device=self.device)

        # 1. Forward inference to compute anomaly score and anomaly status
        self.model.eval()
        metadata: Dict[str, Any] = {"model_type": type(self.model).__name__}

        if isinstance(self.model, DeepLogDualHead):
            with torch.no_grad():
                token_log_probs, seq_prob = self.model(seq_t)
                anomaly_prob = float(seq_prob.item())

            metadata["sequence_anomaly_prob"] = anomaly_prob
            anomaly_score = anomaly_prob
            is_anomalous = anomaly_score >= threshold

            if next_event is not None:
                token_prob = float(token_log_probs[0, next_event].exp().item())
                metadata["next_event_prob"] = token_prob
                top_preds = token_log_probs.topk(
                    k=min(top_k, self.model.output_size), dim=-1
                ).indices[0].tolist()
                metadata["top_predicted_events"] = top_preds
                if next_event not in top_preds or anomaly_prob >= threshold:
                    is_anomalous = True

        elif isinstance(self.model, (LogTransformer, DeepLog)):
            with torch.no_grad():
                token_log_probs = self.model(seq_t)

            top_k_indices = token_log_probs.topk(
                k=min(top_k, getattr(self.model, "output_size", 10)), dim=-1
            ).indices[0].tolist()
            metadata["top_predicted_events"] = top_k_indices

            if next_event is not None:
                token_prob = float(token_log_probs[0, next_event].exp().item())
                metadata["next_event_prob"] = token_prob
                anomaly_score = float(1.0 - token_prob)
                is_anomalous = (next_event not in top_k_indices) or (token_prob < (1.0 - threshold))
            else:
                top_prob = float(token_log_probs.exp().max(dim=-1).values[0].item())
                anomaly_score = float(1.0 - top_prob)
                is_anomalous = anomaly_score >= threshold

        else:
            anomaly_score = 0.5
            is_anomalous = True

        # 2. Compute token attributions
        attr_method = method or self.default_method
        attributions = self.attributor.attribute(
            seq_t,
            method=attr_method,
            target_event=next_event,
            **kwargs,
        )
        attributions = np.asarray(attributions, dtype=np.float64)
        attribution_percentages = attributions * 100.0

        # 3. Identify anomaly trigger points
        uniform_baseline = 1.0 / seq_len
        trigger_cutoff = self.trigger_threshold_factor * uniform_baseline
        trigger_points = [i for i, attr in enumerate(attributions) if attr >= trigger_cutoff]

        # If no single token exceeds cutoff, select maximum attribution point
        if not trigger_points and seq_len > 0:
            trigger_points = [int(np.argmax(attributions))]

        # 4. Rank events by attribution descending
        ranked_indices = np.argsort(-attributions)
        ranked_events = []
        for rank, idx in enumerate(ranked_indices, start=1):
            eid = seq_list[idx]
            ename = self.vocab_mapping.get(eid, f"E{eid}")
            ranked_events.append(
                {
                    "rank": rank,
                    "token_index": int(idx),
                    "event_id": eid,
                    "event_name": ename,
                    "attribution": float(attributions[idx]),
                    "attribution_pct": float(attribution_percentages[idx]),
                    "is_trigger": int(idx) in trigger_points,
                }
            )

        # 5. Rank adjacent transitions
        ranked_transitions = []
        transitions_raw = []

        # Internal sequence transitions
        for i in range(1, seq_len):
            from_idx, to_idx = i - 1, i
            from_eid, to_eid = seq_list[from_idx], seq_list[to_idx]
            trans_score = float((attributions[from_idx] + attributions[to_idx]) / 2.0)
            transitions_raw.append(
                {
                    "from_index": from_idx,
                    "to_index": to_idx,
                    "from_event": from_eid,
                    "to_event": to_eid,
                    "transition": (from_eid, to_eid),
                    "attribution": trans_score,
                    "attribution_pct": trans_score * 100.0,
                }
            )

        # Final transition into next_event if available
        if next_event is not None:
            last_idx = seq_len - 1
            last_eid = seq_list[last_idx]
            trans_score = float(attributions[last_idx])
            transitions_raw.append(
                {
                    "from_index": last_idx,
                    "to_index": seq_len,
                    "from_event": last_eid,
                    "to_event": next_event,
                    "transition": (last_eid, next_event),
                    "attribution": trans_score,
                    "attribution_pct": trans_score * 100.0,
                }
            )

        # Sort transitions by attribution descending
        transitions_raw.sort(key=lambda t: t["attribution"], reverse=True)
        for rank, tr in enumerate(transitions_raw, start=1):
            tr["rank"] = rank
            ranked_transitions.append(tr)

        event_names = [self.vocab_mapping.get(eid, f"E{eid}") for eid in seq_list]

        if next_event is not None:
            metadata["target_event_name"] = self.vocab_mapping.get(next_event, f"E{next_event}")

        return RootCauseExplanation(
            sequence=seq_list,
            token_indices=list(range(seq_len)),
            event_ids=seq_list,
            attributions=attributions,
            attribution_percentages=attribution_percentages,
            anomaly_score=anomaly_score,
            is_anomalous=is_anomalous,
            anomaly_trigger_points=trigger_points,
            ranked_events=ranked_events,
            ranked_transitions=ranked_transitions,
            method=attr_method,
            target_event=next_event,
            event_names=event_names,
            metadata=metadata,
        )

    def explain_batch(
        self,
        sequences: List[Union[List[int], np.ndarray, torch.Tensor]],
        next_events: Optional[List[Optional[int]]] = None,
        method: Optional[str] = None,
        top_k: int = 5,
        **kwargs,
    ) -> List[RootCauseExplanation]:
        """
        Explain a batch of anomalous sequences.

        Parameters
        ----------
        sequences : List of sequences
            Sequences to diagnose.
        next_events : List of int or None, optional
            Next events corresponding to each sequence.
        method : str, optional
            Attribution method override.
        top_k : int, default=5
            Number of top contributing events to extract.

        Returns
        -------
        List[RootCauseExplanation]
            List of structured RCA explanations.
        """
        explanations = []
        if next_events is None:
            next_events = [None] * len(sequences)

        for seq, ne in zip(sequences, next_events):
            exp = self.analyze(
                sequence=seq,
                next_event=ne,
                method=method,
                top_k=top_k,
                **kwargs,
            )
            explanations.append(exp)

        return explanations
