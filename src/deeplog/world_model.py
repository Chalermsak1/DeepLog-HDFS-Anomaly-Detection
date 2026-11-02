"""
DeepLog-Omni Pillar IV: Latent World Model & Pre-Emptive Self-Healing Control
=============================================================================
Causal Dreamer Architecture inspired by DreamerV3 and World Models for
distributed systems and microservice topologies.

Components:
1. RecurrentStateSpaceModel (RSSM):
   - Deterministic recurrent state (h_t) via GRU.
   - Stochastic variational state (z_t) via continuous Gaussian with reparameterization.
   - Transition prior model p(z_t | h_t).
   - Representation posterior model q(z_t | h_t, x_t).
   - Observation decoder p(x_t | h_t, z_t).
   - Anomaly / failure cascade risk predictor p(fail_t | h_t, z_t).
   - KL-balancing regularization between prior and posterior.

2. CounterfactualDreamer:
   - Imagines multi-step forward trajectories K steps into the future in latent space.
   - Simulates hypothetical control interventions (Counterfactual reasoning).
   - Evaluates risk trajectories across candidate actions without impacting production.

3. AutonomousMitigator:
   - Evaluates imagined failure probabilities and lead time to failure.
   - Selects optimal pre-emptive control directives:
     * Live Container / Replica Migration
     * Adaptive eBPF Backpressure / TCP Window Throttling
     * Dynamic Circuit Breaking
   - Generates actionable mitigation logs and automated remediation event streams.
"""

from __future__ import annotations

import enum
import json
import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# Enums and Structured Data Containers
# ==============================================================================


class MitigationAction(enum.IntEnum):
    """
    Control actions available for self-healing distributed systems.
    """
    NO_ACTION = 0
    MIGRATE_REPLICA = 1
    EBPF_THROTTLE = 2
    CIRCUIT_BREAKER_OPEN = 3

    @property
    def identifier(self) -> str:
        return self.name.lower()

    @property
    def human_name(self) -> str:
        mapping = {
            MitigationAction.NO_ACTION: "No Action (Steady State)",
            MitigationAction.MIGRATE_REPLICA: "Live Container / Replica Migration",
            MitigationAction.EBPF_THROTTLE: "Adaptive eBPF Backpressure / TCP Clamp",
            MitigationAction.CIRCUIT_BREAKER_OPEN: "Dynamic Circuit Breaking",
        }
        return mapping[self]


class MitigationUrgency(enum.Enum):
    """Urgency level for automated remediation."""
    NONE = "NONE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class RSSMState:
    """
    State container for the Recurrent State Space Model.

    Attributes
    ----------
    h : torch.Tensor
        Deterministic recurrent hidden state of shape (batch_size, hidden_dim).
    z : torch.Tensor
        Stochastic variational latent state of shape (batch_size, latent_dim).
    mean_prior : Optional[torch.Tensor]
        Prior distribution mean of shape (batch_size, latent_dim).
    std_prior : Optional[torch.Tensor]
        Prior distribution standard deviation of shape (batch_size, latent_dim).
    mean_post : Optional[torch.Tensor]
        Posterior distribution mean of shape (batch_size, latent_dim).
    std_post : Optional[torch.Tensor]
        Posterior distribution standard deviation of shape (batch_size, latent_dim).
    """

    h: torch.Tensor
    z: torch.Tensor
    mean_prior: Optional[torch.Tensor] = None
    std_prior: Optional[torch.Tensor] = None
    mean_post: Optional[torch.Tensor] = None
    std_post: Optional[torch.Tensor] = None

    def detach(self) -> "RSSMState":
        """Detach all tensors in the state from the computation graph."""
        return RSSMState(
            h=self.h.detach(),
            z=self.z.detach(),
            mean_prior=self.mean_prior.detach() if self.mean_prior is not None else None,
            std_prior=self.std_prior.detach() if self.std_prior is not None else None,
            mean_post=self.mean_post.detach() if self.mean_post is not None else None,
            std_post=self.std_post.detach() if self.std_post is not None else None,
        )

    def clone(self) -> "RSSMState":
        """Clone all tensors in the state."""
        return RSSMState(
            h=self.h.clone(),
            z=self.z.clone(),
            mean_prior=self.mean_prior.clone() if self.mean_prior is not None else None,
            std_prior=self.std_prior.clone() if self.std_prior is not None else None,
            mean_post=self.mean_post.clone() if self.mean_post is not None else None,
            std_post=self.std_post.clone() if self.std_post is not None else None,
        )

    def to(self, device: torch.device) -> "RSSMState":
        """Move all state tensors to specified device."""
        return RSSMState(
            h=self.h.to(device),
            z=self.z.to(device),
            mean_prior=self.mean_prior.to(device) if self.mean_prior is not None else None,
            std_prior=self.std_prior.to(device) if self.std_prior is not None else None,
            mean_post=self.mean_post.to(device) if self.mean_post is not None else None,
            std_post=self.std_post.to(device) if self.std_post is not None else None,
        )


@dataclass
class CounterfactualTrajectory:
    """
    Imagined latent trajectory generated by the CounterfactualDreamer.

    Attributes
    ----------
    action : MitigationAction
        Hypothetical action evaluated across this trajectory.
    action_name : str
        Human-readable name of the evaluated action.
    horizon : int
        Number of forward imagination steps K.
    failure_risks : np.ndarray
        Predicted failure probability at each step tau in [1, K], shape (K,).
    predicted_observations : np.ndarray
        Decoded observation feature vectors across the horizon, shape (K, obs_dim).
    peak_risk : float
        Maximum failure probability along the trajectory.
    mean_risk : float
        Average failure probability along the trajectory.
    time_to_failure : Optional[int]
        First step tau at which failure risk crosses critical threshold, or None.
    risk_reduction : float
        Absolute risk reduction compared to NO_ACTION baseline: R_base - R_action.
    causal_effect : float
        Relative percentage reduction compared to NO_ACTION baseline in [0.0, 100.0].
    states : List[RSSMState]
        Sequence of imagined latent RSSM states.
    """

    action: MitigationAction
    action_name: str
    horizon: int
    failure_risks: np.ndarray
    predicted_observations: np.ndarray
    peak_risk: float
    mean_risk: float
    time_to_failure: Optional[int]
    risk_reduction: float = 0.0
    causal_effect: float = 0.0
    states: List[RSSMState] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert trajectory metrics to serializable dictionary."""
        return {
            "action": self.action.name,
            "action_id": int(self.action),
            "action_name": self.action_name,
            "horizon": self.horizon,
            "failure_risks": self.failure_risks.tolist(),
            "predicted_observations": self.predicted_observations.tolist(),
            "peak_risk": float(self.peak_risk),
            "mean_risk": float(self.mean_risk),
            "time_to_failure": self.time_to_failure,
            "risk_reduction": float(self.risk_reduction),
            "causal_effect": float(self.causal_effect),
        }


@dataclass
class MitigationDecision:
    """
    Pre-emptive mitigation directive issued by AutonomousMitigator.

    Attributes
    ----------
    action : MitigationAction
        Recommended mitigation action.
    action_name : str
        Human-readable action description.
    target_resource : str
        Identifier of target microservice, pod, or network entity.
    baseline_risk : float
        Predicted peak failure risk if no action is taken.
    mitigated_risk : float
        Predicted peak failure risk under the recommended action.
    risk_reduction : float
        Absolute reduction in failure risk (baseline_risk - mitigated_risk).
    lead_time_steps : Optional[int]
        Estimated forward steps until baseline failure cascade triggers.
    urgency : MitigationUrgency
        Assessed urgency level.
    confidence : float
        Confidence score of the world model's prediction in [0.0, 1.0].
    rationale : str
        Causal explanation detailing why this mitigation was chosen.
    parameters : Dict[str, Any]
        Action-specific operational execution parameters.
    action_plan : List[str]
        Ordered step-by-step remediation execution commands.
    timestamp : float
        Unix epoch timestamp of decision creation.
    all_trajectories : Dict[str, CounterfactualTrajectory]
        Counterfactual trajectories evaluated across all candidate actions.
    """

    action: MitigationAction
    action_name: str
    target_resource: str
    baseline_risk: float
    mitigated_risk: float
    risk_reduction: float
    lead_time_steps: Optional[int]
    urgency: MitigationUrgency
    confidence: float
    rationale: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    action_plan: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    all_trajectories: Dict[str, CounterfactualTrajectory] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert decision to a JSON-serializable dictionary."""
        return {
            "action": self.action.name,
            "action_id": int(self.action),
            "action_name": self.action_name,
            "target_resource": self.target_resource,
            "baseline_risk": round(float(self.baseline_risk), 4),
            "mitigated_risk": round(float(self.mitigated_risk), 4),
            "risk_reduction": round(float(self.risk_reduction), 4),
            "lead_time_steps": self.lead_time_steps,
            "urgency": self.urgency.value,
            "confidence": round(float(self.confidence), 4),
            "rationale": self.rationale,
            "parameters": self.parameters,
            "action_plan": self.action_plan,
            "timestamp": self.timestamp,
            "trajectories": {
                k: traj.to_dict() for k, traj in self.all_trajectories.items()
            },
        }

    def format_log(self) -> str:
        """Format an actionable SRE mitigation event log."""
        lead_str = f"T+{self.lead_time_steps} steps" if self.lead_time_steps is not None else "N/A"
        return (
            f"[PRE-EMPTIVE-CONTROL] [{self.urgency.value}] "
            f"Resource: {self.target_resource} | "
            f"Action: {self.action.name} ({self.action_name}) | "
            f"Lead Time: {lead_str} | "
            f"Baseline Risk: {self.baseline_risk * 100:.1f}% -> Mitigated: {self.mitigated_risk * 100:.1f}% "
            f"(Delta: -{self.risk_reduction * 100:.1f}%) | "
            f"Rationale: {self.rationale}"
        )


class RemediationEventStream:
    """
    Append-only buffer and event stream for self-healing remediation events.
    """

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._events: List[MitigationDecision] = []

    def record_decision(self, decision: MitigationDecision) -> None:
        """Record a newly issued mitigation decision."""
        self._events.append(decision)
        if len(self._events) > self.max_size:
            self._events.pop(0)

    def get_recent_events(self, limit: int = 50) -> List[MitigationDecision]:
        """Return the most recent decisions up to limit."""
        return self._events[-limit:]

    def filter_by_action(self, action: MitigationAction) -> List[MitigationDecision]:
        """Return decisions matching a given action."""
        return [d for d in self._events if d.action == action]

    def filter_by_urgency(self, urgency: MitigationUrgency) -> List[MitigationDecision]:
        """Return decisions matching a given urgency."""
        return [d for d in self._events if d.urgency == urgency]

    def export_json(self, indent: int = 2) -> str:
        """Export all events as formatted JSON string."""
        return json.dumps([d.to_dict() for d in self._events], indent=indent)

    def clear(self) -> None:
        """Clear the event buffer."""
        self._events.clear()

    def __len__(self) -> int:
        return len(self._events)


# ==============================================================================
# 1. RecurrentStateSpaceModel (RSSM)
# ==============================================================================


class RecurrentStateSpaceModel(nn.Module):
    """
    Recurrent State Space Model (RSSM) for distributed microservice dynamics.

    Inspired by DreamerV3 (Hafner et al.), the RSSM splits latent state into:
    1. Deterministic recurrent hidden state h_t in R^{hidden_dim}.
    2. Stochastic variational latent state z_t in R^{latent_dim}.

    Mathematical Equations:
    -----------------------
    - Deterministic transition:
        h_t = GRUCell(W_in [z_{t-1}, a_{t-1}], h_{t-1})
    - Stochastic transition prior:
        p(z_t | h_t) = N(mu_prior(h_t), diag(sigma_prior^2(h_t)))
    - Stochastic representation posterior:
        q(z_t | h_t, x_t) = N(mu_post(h_t, Enc(x_t)), diag(sigma_post^2(h_t, Enc(x_t))))
    - Observation reconstruction decoder:
        x_hat_t = Dec(h_t, z_t)
    - Failure cascade risk predictor:
        p_fail_t = sigmoid(RiskNet(h_t, z_t))
    - Variational KL divergence with KL balancing:
        L_kl = alpha * D_KL(sg[q] || p) + (1 - alpha) * D_KL(q || sg[p])
    """

    def __init__(
        self,
        obs_dim: int = 16,
        action_dim: int = 4,
        action_embed_dim: int = 16,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        encoder_hidden_dim: int = 64,
        decoder_hidden_dim: int = 64,
        min_std: float = 0.1,
        max_std: float = 2.0,
        kl_balance: float = 0.8,
        kl_free_nats: float = 0.1,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_embed_dim = action_embed_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.min_std = min_std
        self.max_std = max_std
        self.kl_balance = kl_balance
        self.kl_free_nats = kl_free_nats

        # 1. Action Embedding
        self.action_embedding = nn.Embedding(action_dim, action_embed_dim)

        # 2. Observation Encoder: Enc(x_t)
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, encoder_hidden_dim),
            nn.ELU(),
            nn.Linear(encoder_hidden_dim, encoder_hidden_dim),
            nn.ELU(),
        )

        # 3. Recurrent Transition Core
        # Projects [z_{t-1}, a_{t-1}] to input size of GRUCell
        self.recurrent_input_proj = nn.Sequential(
            nn.Linear(latent_dim + action_embed_dim, hidden_dim),
            nn.ELU(),
        )
        self.recurrent_cell = nn.GRUCell(hidden_dim, hidden_dim)

        # 4. Stochastic Transition Prior Net: p(z_t | h_t)
        self.prior_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 2 * latent_dim),
        )

        # 5. Stochastic Representation Posterior Net: q(z_t | h_t, e_t)
        self.posterior_net = nn.Sequential(
            nn.Linear(hidden_dim + encoder_hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 2 * latent_dim),
        )

        # 6. Observation Reconstruction Decoder: Dec(h_t, z_t)
        self.obs_decoder = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim, decoder_hidden_dim),
            nn.ELU(),
            nn.Linear(decoder_hidden_dim, decoder_hidden_dim),
            nn.ELU(),
            nn.Linear(decoder_hidden_dim, obs_dim),
        )

        # 7. Failure Cascade Risk Predictor: p(fail_t | h_t, z_t)
        self.risk_predictor = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim, decoder_hidden_dim),
            nn.ELU(),
            nn.Linear(decoder_hidden_dim, 32),
            nn.ELU(),
            nn.Linear(32, 1),
        )

    def init_state(self, batch_size: int = 1, device: Optional[torch.device] = None) -> RSSMState:
        """
        Initialize an empty zeroed RSSM state.
        """
        if device is None:
            device = next(self.parameters()).device
        h = torch.zeros(batch_size, self.hidden_dim, device=device)
        z = torch.zeros(batch_size, self.latent_dim, device=device)
        return RSSMState(h=h, z=z)

    def _get_gaussian_params(
        self, raw_params: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Split raw network outputs into mean and standard deviation.
        sigma = clamp(softplus(raw_std), min_std, max_std)
        """
        mean, raw_std = torch.chunk(raw_params, 2, dim=-1)
        std = F.softplus(raw_std) + self.min_std
        std = torch.clamp(std, min=self.min_std, max=self.max_std)
        return mean, std

    def _sample_gaussian(
        self, mean: torch.Tensor, std: torch.Tensor, sample: bool = True
    ) -> torch.Tensor:
        """
        Sample from Gaussian via reparameterization trick: z = mu + sigma * eps.
        """
        if sample:
            eps = torch.randn_like(std)
            return mean + std * eps
        return mean

    def step_prior(
        self,
        action: torch.Tensor,
        prev_state: RSSMState,
        sample: bool = True,
    ) -> RSSMState:
        """
        Forward step in imagination / simulation mode without current observation.

        Computes:
        1. h_t = f(h_{t-1}, z_{t-1}, a_{t-1})
        2. p(z_t | h_t) ~ N(mu_prior, sigma_prior^2)
        3. z_t ~ p(z_t | h_t)

        Parameters
        ----------
        action : torch.Tensor of shape (batch_size,) or (batch_size, action_embed_dim)
            Action index or action vector.
        prev_state : RSSMState
            Previous state (h_{t-1}, z_{t-1}).
        sample : bool, default=True
            Whether to sample stochastically or take the mean.

        Returns
        -------
        RSSMState
            Updated state containing deterministic h_t and sampled z_t.
        """
        # Embed action if integer indices provided
        if action.dtype in (torch.int32, torch.int64):
            act_emb = self.action_embedding(action)
        else:
            act_emb = action

        # Recurrent deterministic update
        gru_in = self.recurrent_input_proj(torch.cat([prev_state.z, act_emb], dim=-1))
        h = self.recurrent_cell(gru_in, prev_state.h)

        # Transition prior
        prior_raw = self.prior_net(h)
        mean_prior, std_prior = self._get_gaussian_params(prior_raw)
        z = self._sample_gaussian(mean_prior, std_prior, sample=sample)

        return RSSMState(
            h=h,
            z=z,
            mean_prior=mean_prior,
            std_prior=std_prior,
            mean_post=None,
            std_post=None,
        )

    def step_posterior(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        prev_state: RSSMState,
        sample: bool = True,
    ) -> RSSMState:
        """
        Forward step in filtering / observation mode incorporating current observation x_t.

        Computes:
        1. h_t = f(h_{t-1}, z_{t-1}, a_{t-1})
        2. Prior p(z_t | h_t) ~ N(mu_prior, sigma_prior)
        3. Observation embedding e_t = Enc(x_t)
        4. Posterior q(z_t | h_t, e_t) ~ N(mu_post, sigma_post)
        5. z_t ~ q(z_t | h_t, e_t)

        Parameters
        ----------
        obs : torch.Tensor of shape (batch_size, obs_dim)
            Current observation vector.
        action : torch.Tensor of shape (batch_size,)
            Previous control action taken leading to obs.
        prev_state : RSSMState
            Previous state (h_{t-1}, z_{t-1}).
        sample : bool, default=True
            Whether to sample stochastically.

        Returns
        -------
        RSSMState
            Updated state containing deterministic h_t, posterior z_t, and distribution stats.
        """
        if action.dtype in (torch.int32, torch.int64):
            act_emb = self.action_embedding(action)
        else:
            act_emb = action

        # 1. Deterministic recurrent update
        gru_in = self.recurrent_input_proj(torch.cat([prev_state.z, act_emb], dim=-1))
        h = self.recurrent_cell(gru_in, prev_state.h)

        # 2. Prior distribution
        prior_raw = self.prior_net(h)
        mean_prior, std_prior = self._get_gaussian_params(prior_raw)

        # 3. Observation encoding
        e_t = self.obs_encoder(obs)

        # 4. Posterior distribution
        post_raw = self.posterior_net(torch.cat([h, e_t], dim=-1))
        mean_post, std_post = self._get_gaussian_params(post_raw)

        # 5. Stochastic sampling
        z = self._sample_gaussian(mean_post, std_post, sample=sample)

        return RSSMState(
            h=h,
            z=z,
            mean_prior=mean_prior,
            std_prior=std_prior,
            mean_post=mean_post,
            std_post=std_post,
        )

    def decode_observation(self, state: RSSMState) -> torch.Tensor:
        """
        Reconstruct observation x_hat_t from latent state (h_t, z_t).
        """
        latent_features = torch.cat([state.h, state.z], dim=-1)
        return self.obs_decoder(latent_features)

    def predict_risk(self, state: RSSMState) -> torch.Tensor:
        """
        Predict failure cascade probability p_fail in [0, 1] from latent state (h_t, z_t).
        """
        latent_features = torch.cat([state.h, state.z], dim=-1)
        logits = self.risk_predictor(latent_features)
        return torch.sigmoid(logits)

    @staticmethod
    def kl_divergence(
        mu_q: torch.Tensor,
        std_q: torch.Tensor,
        mu_p: torch.Tensor,
        std_p: torch.Tensor,
    ) -> torch.Tensor:
        """
        Closed-form analytical KL divergence D_KL(N(mu_q, diag(std_q^2)) || N(mu_p, diag(std_p^2))).

        Equation:
        D_KL(q || p) = sum_i [ log(std_p / std_q) + (std_q^2 + (mu_q - mu_p)^2)/(2 * std_p^2) - 0.5 ]
        """
        var_q = std_q.pow(2)
        var_p = std_p.pow(2)
        term1 = torch.log(std_p / std_q)
        term2 = (var_q + (mu_q - mu_p).pow(2)) / (2.0 * var_p)
        kl = term1 + term2 - 0.5
        return torch.sum(kl, dim=-1)

    def compute_balanced_kl_loss(
        self,
        mean_post: torch.Tensor,
        std_post: torch.Tensor,
        mean_prior: torch.Tensor,
        std_prior: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute KL loss with Dreamer-style KL balancing and free nats.

        L_kl = alpha * D_KL(sg[q] || p) + (1 - alpha) * D_KL(q || sg[p])
        """
        # Prior loss: trains prior network to match posterior
        kl_prior = self.kl_divergence(
            mean_post.detach(), std_post.detach(), mean_prior, std_prior
        )
        # Posterior loss: prevents posterior from collapsing
        kl_post = self.kl_divergence(
            mean_post, std_post, mean_prior.detach(), std_prior.detach()
        )

        balanced_kl = self.kl_balance * kl_prior + (1.0 - self.kl_balance) * kl_post
        if self.kl_free_nats > 0.0:
            balanced_kl = torch.clamp(balanced_kl, min=self.kl_free_nats)

        return torch.mean(balanced_kl)

    def forward(
        self,
        obs_seq: torch.Tensor,
        act_seq: torch.Tensor,
        initial_state: Optional[RSSMState] = None,
        target_risks: Optional[torch.Tensor] = None,
        lambda_risk: float = 1.0,
        beta_kl: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Process a temporal sequence of observations and actions.

        Parameters
        ----------
        obs_seq : torch.Tensor of shape (batch_size, seq_len, obs_dim)
            Sequential observation vectors.
        act_seq : torch.Tensor of shape (batch_size, seq_len)
            Sequential control action indices.
        initial_state : RSSMState, optional
            Initial state. Defaults to zero init_state.
        target_risks : torch.Tensor, optional of shape (batch_size, seq_len, 1)
            Ground truth failure labels in {0, 1}.
        lambda_risk : float, default=1.0
            Loss scaling factor for failure risk prediction.
        beta_kl : float, default=0.5
            Loss scaling factor for KL divergence regularization.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing reconstructed observations, predicted risks,
            individual loss terms, and final hidden state.
        """
        batch_size, seq_len, _ = obs_seq.shape
        device = obs_seq.device

        if initial_state is None:
            state = self.init_state(batch_size=batch_size, device=device)
        else:
            state = initial_state

        recon_obs_list = []
        pred_risk_list = []
        mean_prior_list = []
        std_prior_list = []
        mean_post_list = []
        std_post_list = []
        states_list = []

        for t in range(seq_len):
            obs_t = obs_seq[:, t, :]
            act_t = act_seq[:, t]

            # Update state with observation
            state = self.step_posterior(obs=obs_t, action=act_t, prev_state=state, sample=True)

            # Decode and predict
            recon_obs_t = self.decode_observation(state)
            pred_risk_t = self.predict_risk(state)

            recon_obs_list.append(recon_obs_t)
            pred_risk_list.append(pred_risk_t)
            mean_prior_list.append(state.mean_prior)
            std_prior_list.append(state.std_prior)
            mean_post_list.append(state.mean_post)
            std_post_list.append(state.std_post)
            states_list.append(state)

        recon_obs = torch.stack(recon_obs_list, dim=1)  # (B, T, obs_dim)
        pred_risks = torch.stack(pred_risk_list, dim=1)  # (B, T, 1)

        mean_prior = torch.stack(mean_prior_list, dim=1)  # (B, T, latent_dim)
        std_prior = torch.stack(std_prior_list, dim=1)
        mean_post = torch.stack(mean_post_list, dim=1)
        std_post = torch.stack(std_post_list, dim=1)

        # 1. Reconstruction Loss (MSE)
        recon_loss = F.mse_loss(recon_obs, obs_seq)

        # 2. Risk Prediction Loss (BCE)
        if target_risks is not None:
            risk_loss = F.binary_cross_entropy(pred_risks, target_risks)
        else:
            risk_loss = torch.tensor(0.0, device=device)

        # 3. KL Divergence Loss with balancing
        kl_loss = self.compute_balanced_kl_loss(
            mean_post=mean_post,
            std_post=std_post,
            mean_prior=mean_prior,
            std_prior=std_prior,
        )

        total_loss = recon_loss + lambda_risk * risk_loss + beta_kl * kl_loss

        return {
            "total_loss": total_loss,
            "recon_loss": recon_loss,
            "risk_loss": risk_loss,
            "kl_loss": kl_loss,
            "reconstructed_obs": recon_obs,
            "predicted_risks": pred_risks,
            "final_state": state,
            "states": states_list,
        }


# ==============================================================================
# 2. CounterfactualDreamer
# ==============================================================================


class CounterfactualDreamer:
    """
    Imagines forward multi-step trajectories in latent space under hypothetical control actions.

    Enables counterfactual 'What-If' reasoning without perturbing the live distributed system.
    Evaluates:
    - Baseline: What happens if we do nothing (NO_ACTION)?
    - Mitigation A: What happens under Live Container Migration?
    - Mitigation B: What happens under Adaptive eBPF Backpressure?
    - Mitigation C: What happens under Dynamic Circuit Breaking?
    """

    def __init__(
        self,
        world_model: RecurrentStateSpaceModel,
        default_horizon: int = 10,
        critical_risk_threshold: float = 0.70,
    ):
        self.world_model = world_model
        self.default_horizon = default_horizon
        self.critical_risk_threshold = critical_risk_threshold

    def imagine_trajectory(
        self,
        initial_state: RSSMState,
        action: Union[int, MitigationAction],
        horizon: Optional[int] = None,
        sample: bool = False,
    ) -> CounterfactualTrajectory:
        """
        Simulate a forward trajectory K steps into the future conditioned on a single policy action.

        Parameters
        ----------
        initial_state : RSSMState
            Current filtered latent state (h_0, z_0).
        action : Union[int, MitigationAction]
            Action applied along the imagined trajectory.
        horizon : int, optional
            Number of forward steps K. Defaults to self.default_horizon.
        sample : bool, default=False
            If False, uses distribution mean for deterministic rollouts.

        Returns
        -------
        CounterfactualTrajectory
            Detailed imagined trajectory with failure probabilities and decoded telemetry.
        """
        k = horizon if horizon is not None else self.default_horizon
        device = initial_state.h.device
        act_enum = MitigationAction(int(action))

        # Action tensor: shape (batch_size,)
        batch_size = initial_state.h.shape[0]
        act_tensor = torch.full(
            (batch_size,), int(act_enum), dtype=torch.long, device=device
        )

        state = initial_state
        risks: List[float] = []
        decoded_obs: List[np.ndarray] = []
        states: List[RSSMState] = []
        time_to_fail: Optional[int] = None

        with torch.no_grad():
            for tau in range(1, k + 1):
                # Transition prior step without observation
                state = self.world_model.step_prior(
                    action=act_tensor,
                    prev_state=state,
                    sample=sample,
                )
                states.append(state)

                # Decode telemetry and predict risk
                obs_pred = self.world_model.decode_observation(state)
                risk_pred = self.world_model.predict_risk(state)

                risk_val = float(risk_pred[0].item())
                risks.append(risk_val)
                decoded_obs.append(obs_pred[0].cpu().numpy())

                if time_to_fail is None and risk_val >= self.critical_risk_threshold:
                    time_to_fail = tau

        risks_arr = np.array(risks, dtype=np.float32)
        obs_arr = np.array(decoded_obs, dtype=np.float32)

        return CounterfactualTrajectory(
            action=act_enum,
            action_name=act_enum.human_name,
            horizon=k,
            failure_risks=risks_arr,
            predicted_observations=obs_arr,
            peak_risk=float(np.max(risks_arr)) if len(risks_arr) > 0 else 0.0,
            mean_risk=float(np.mean(risks_arr)) if len(risks_arr) > 0 else 0.0,
            time_to_failure=time_to_fail,
            states=states,
        )

    def dream_counterfactuals(
        self,
        initial_state: RSSMState,
        candidate_actions: Optional[Sequence[Union[int, MitigationAction]]] = None,
        horizon: Optional[int] = None,
        baseline_action: MitigationAction = MitigationAction.NO_ACTION,
        sample: bool = False,
    ) -> Dict[str, CounterfactualTrajectory]:
        """
        Simulate and causally contrast counterfactual trajectories across all candidate actions.

        Parameters
        ----------
        initial_state : RSSMState
            Current latent state.
        candidate_actions : Sequence[MitigationAction], optional
            Set of actions to compare. Defaults to all 4 mitigation actions.
        horizon : int, optional
            Imagination horizon K.
        baseline_action : MitigationAction, default=NO_ACTION
            The reference baseline action representing inaction.
        sample : bool, default=False
            Whether to sample stochastically.

        Returns
        -------
        Dict[str, CounterfactualTrajectory]
            Mapping from action name to its counterfactual trajectory.
        """
        if candidate_actions is None:
            actions_to_eval = [
                MitigationAction.NO_ACTION,
                MitigationAction.MIGRATE_REPLICA,
                MitigationAction.EBPF_THROTTLE,
                MitigationAction.CIRCUIT_BREAKER_OPEN,
            ]
        else:
            actions_to_eval = [MitigationAction(int(a)) for a in candidate_actions]

        # 1. First evaluate baseline
        trajectories: Dict[str, CounterfactualTrajectory] = {}
        baseline_traj = self.imagine_trajectory(
            initial_state=initial_state,
            action=baseline_action,
            horizon=horizon,
            sample=sample,
        )
        trajectories[baseline_action.name] = baseline_traj
        base_peak_risk = baseline_traj.peak_risk

        # 2. Evaluate counterfactual interventions
        for act in actions_to_eval:
            if act == baseline_action:
                continue
            traj = self.imagine_trajectory(
                initial_state=initial_state,
                action=act,
                horizon=horizon,
                sample=sample,
            )
            # Causal comparison against baseline
            traj.risk_reduction = max(0.0, base_peak_risk - traj.peak_risk)
            if base_peak_risk > 1e-4:
                traj.causal_effect = (traj.risk_reduction / base_peak_risk) * 100.0
            else:
                traj.causal_effect = 0.0

            trajectories[act.name] = traj

        return trajectories


# ==============================================================================
# 3. AutonomousMitigator
# ==============================================================================


class AutonomousMitigator:
    """
    Autonomous self-healing controller for distributed microservice systems.

    Evaluates counterfactual forward trajectories imagined by CounterfactualDreamer
    and triggers pre-emptive mitigation directives before failure cascades materialize:
    - Live Container / Replica Migration
    - Adaptive eBPF Backpressure / TCP Window Throttling
    - Dynamic Circuit Breaking
    """

    def __init__(
        self,
        dreamer: CounterfactualDreamer,
        warning_threshold: float = 0.40,
        critical_threshold: float = 0.70,
        safe_target_risk: float = 0.25,
        horizon: int = 10,
        action_costs: Optional[Dict[MitigationAction, float]] = None,
        event_stream: Optional[RemediationEventStream] = None,
    ):
        self.dreamer = dreamer
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.safe_target_risk = safe_target_risk
        self.horizon = horizon

        # Disruption / operational cost penalties for actions:
        # NO_ACTION = 0.0 (cheapest)
        # EBPF_THROTTLE = 0.08 (low overhead, gentle backpressure)
        # CIRCUIT_BREAKER_OPEN = 0.18 (cuts traffic, some error responses)
        # MIGRATE_REPLICA = 0.25 (cold/warm migration network & reschedule overhead)
        self.action_costs = action_costs if action_costs is not None else {
            MitigationAction.NO_ACTION: 0.0,
            MitigationAction.EBPF_THROTTLE: 0.08,
            MitigationAction.CIRCUIT_BREAKER_OPEN: 0.18,
            MitigationAction.MIGRATE_REPLICA: 0.25,
        }

        self.event_stream = event_stream if event_stream is not None else RemediationEventStream()

    def evaluate_and_mitigate(
        self,
        initial_state: RSSMState,
        target_resource: str = "service/checkout-api",
        candidate_actions: Optional[Sequence[MitigationAction]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MitigationDecision:
        """
        Evaluate imagined trajectories and synthesize an optimal pre-emptive mitigation decision.

        Decision Logic:
        1. Dream trajectories for all candidate actions over horizon K.
        2. Inspect baseline (NO_ACTION) peak risk and lead time to failure.
        3. If baseline risk < warning_threshold:
           Maintain steady-state (NO_ACTION).
        4. If baseline risk >= warning_threshold:
           Identify candidate actions that reduce risk below safe target.
           Compute net utility: Utility(a) = RiskReduction(a) - Cost(a).
           Select action maximizing utility, or highest risk reduction if all above threshold.
        5. Formulate actionable parameters, low-level directives, and rationale.

        Parameters
        ----------
        initial_state : RSSMState
            Current state.
        target_resource : str, default="service/checkout-api"
            Name of service/pod.
        candidate_actions : Sequence[MitigationAction], optional
            Subset of actions to consider.
        metadata : Dict[str, Any], optional
            Extra cluster context (e.g. node names, IP addresses).

        Returns
        -------
        MitigationDecision
            Complete pre-emptive self-healing directive.
        """
        trajectories = self.dreamer.dream_counterfactuals(
            initial_state=initial_state,
            candidate_actions=candidate_actions,
            horizon=self.horizon,
        )

        base_traj = trajectories[MitigationAction.NO_ACTION.name]
        baseline_risk = base_traj.peak_risk
        lead_time = base_traj.time_to_failure

        meta = metadata or {}

        # Safe steady state: No failure predicted
        if baseline_risk < self.warning_threshold:
            decision = MitigationDecision(
                action=MitigationAction.NO_ACTION,
                action_name=MitigationAction.NO_ACTION.human_name,
                target_resource=target_resource,
                baseline_risk=baseline_risk,
                mitigated_risk=baseline_risk,
                risk_reduction=0.0,
                lead_time_steps=None,
                urgency=MitigationUrgency.NONE,
                confidence=0.95,
                rationale=(
                    f"Latent world model predicts system stability for {target_resource} "
                    f"over {self.horizon} steps. Peak risk {baseline_risk * 100:.1f}% "
                    f"remains below warning threshold ({self.warning_threshold * 100:.1f}%)."
                ),
                parameters={"monitoring_mode": "steady_state", "interval_sec": 5},
                action_plan=[
                    f"system.monitor/telemetry-steady-state: resource={target_resource}",
                    "audit/emit-remediation-log: status=NOMINAL",
                ],
                all_trajectories=trajectories,
            )
            self.event_stream.record_decision(decision)
            return decision

        # Failure impending: assess urgency
        if baseline_risk >= self.critical_threshold and (lead_time is not None and lead_time <= 2):
            urgency = MitigationUrgency.CRITICAL
        elif baseline_risk >= self.critical_threshold:
            urgency = MitigationUrgency.HIGH
        elif lead_time is not None and lead_time <= 3:
            urgency = MitigationUrgency.MEDIUM
        else:
            urgency = MitigationUrgency.LOW

        # Select optimal intervention
        best_action = MitigationAction.NO_ACTION
        best_utility = -1e9
        best_traj = base_traj

        for act_name, traj in trajectories.items():
            act_enum = traj.action
            if act_enum == MitigationAction.NO_ACTION:
                continue

            cost = self.action_costs.get(act_enum, 0.1)
            risk_reduction = traj.risk_reduction

            # Utility: reward risk reduction, penalize operational disruption cost
            utility = risk_reduction - cost
            # Bonus if action reduces risk strictly below safe target
            if traj.peak_risk <= self.safe_target_risk:
                utility += 0.25

            if utility > best_utility:
                best_utility = utility
                best_action = act_enum
                best_traj = traj

        # If no intervention offered meaningful improvement, fallback to NO_ACTION with alert
        if best_traj.risk_reduction <= 0.01:
            best_action = MitigationAction.NO_ACTION
            best_traj = base_traj

        # Synthesize actionable parameters and low-level execution plan
        params, plan, rationale = self._build_action_directives(
            action=best_action,
            target_resource=target_resource,
            traj=best_traj,
            base_traj=base_traj,
            metadata=meta,
        )

        decision = MitigationDecision(
            action=best_action,
            action_name=best_action.human_name,
            target_resource=target_resource,
            baseline_risk=baseline_risk,
            mitigated_risk=best_traj.peak_risk,
            risk_reduction=best_traj.risk_reduction,
            lead_time_steps=lead_time,
            urgency=urgency,
            confidence=0.92,
            rationale=rationale,
            parameters=params,
            action_plan=plan,
            all_trajectories=trajectories,
        )

        self.event_stream.record_decision(decision)
        return decision

    def _build_action_directives(
        self,
        action: MitigationAction,
        target_resource: str,
        traj: CounterfactualTrajectory,
        base_traj: CounterfactualTrajectory,
        metadata: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], List[str], str]:
        """
        Build specialized operational execution parameters and command plans.
        """
        delta_pct = traj.risk_reduction * 100.0
        lead_steps = base_traj.time_to_failure or self.horizon

        if action == MitigationAction.MIGRATE_REPLICA:
            dest_node = metadata.get("destination_node", "worker-node-04-uswest")
            grace_period = metadata.get("grace_period_sec", 15)
            params = {
                "destination_node": dest_node,
                "grace_period_sec": grace_period,
                "strategy": "live_drain_and_reschedule",
                "eviction_priority": "preemptive_high",
            }
            plan = [
                f"kubectl cordon {target_resource} --namespace production",
                f"kubernetes.io/pre-eviction-drain: pod/{target_resource} destination={dest_node}",
                f"scheduler.k8s/reschedule: target_node={dest_node}, priority=preemptive_high",
                f"sdn.calico/traffic-shift: drain_weight=0.0, target_node={dest_node}",
                f"audit/emit-remediation-log: action=MIGRATE_REPLICA, risk_reduction={delta_pct:.1f}%",
            ]
            rationale = (
                f"Triggering pre-emptive live container migration of {target_resource} to {dest_node}. "
                f"Failure cascade predicted in {lead_steps} steps with {base_traj.peak_risk * 100:.1f}% risk. "
                f"Migration mitigates risk to {traj.peak_risk * 100:.1f}% (reduction: {delta_pct:.1f}%)."
            )

        elif action == MitigationAction.EBPF_THROTTLE:
            clamp_bytes = metadata.get("tcp_window_clamp_bytes", 32768)
            bw_limit = metadata.get("max_bandwidth_mbps", 50)
            iface = metadata.get("interface", "eth0")
            params = {
                "interface": iface,
                "tcp_window_clamp_bytes": clamp_bytes,
                "max_bandwidth_mbps": bw_limit,
                "backpressure_mode": "adaptive_ebpf_tc",
                "burst_tolerance_pkts": 500,
            }
            plan = [
                f"bpf_prog_attach: hook=tc_ingress, dev={iface}, map=rate_limit_map",
                f"bpf_map_update: key={target_resource}, window_clamp_bytes={clamp_bytes}, max_bw_mbps={bw_limit}",
                "kernel.sysctl/tcp_ecn: set=1 (explicit congestion notification enabled)",
                f"audit/emit-remediation-log: action=EBPF_THROTTLE, risk_reduction={delta_pct:.1f}%",
            ]
            rationale = (
                f"Applying adaptive eBPF kernel backpressure and TCP window clamp ({clamp_bytes} B) on {target_resource}. "
                f"Attenuates ingress traffic surge before queue saturation. "
                f"Reduces failure cascade risk from {base_traj.peak_risk * 100:.1f}% to {traj.peak_risk * 100:.1f}%."
            )

        elif action == MitigationAction.CIRCUIT_BREAKER_OPEN:
            cooldown = metadata.get("cooldown_seconds", 30)
            fallback = metadata.get("fallback_route", "cached_degraded_response")
            params = {
                "circuit_name": f"circuit-{target_resource.replace('/', '-')}",
                "cooldown_seconds": cooldown,
                "fallback_route": fallback,
                "half_open_probe_rate": 0.05,
            }
            plan = [
                f"envoy.filters.http/circuit_breaker: cluster={target_resource}, max_connections=0",
                f"service_mesh/route_fallback: status=503, fallback_body='{{\"status\":\"{fallback}\"}}'",
                f"telemetry/timer: cooldown_seconds={cooldown}, half_open_probe_rate=0.05",
                f"audit/emit-remediation-log: action=CIRCUIT_BREAKER_OPEN, risk_reduction={delta_pct:.1f}%",
            ]
            rationale = (
                f"Opening dynamic circuit breaker on {target_resource} for {cooldown}s. "
                f"Halts cascading failure propagation to upstream caller microservices. "
                f"Mitigates failure risk from {base_traj.peak_risk * 100:.1f}% to {traj.peak_risk * 100:.1f}%."
            )

        else:
            params = {"monitoring": "steady_state"}
            plan = [f"system.monitor/telemetry-steady-state: resource={target_resource}"]
            rationale = (
                f"No intervention required for {target_resource}. "
                f"Baseline risk {base_traj.peak_risk * 100:.1f}%."
            )

        return params, plan, rationale

    def step_and_mitigate(
        self,
        current_obs: Union[torch.Tensor, np.ndarray],
        prev_action: Union[int, MitigationAction],
        prev_state: RSSMState,
        target_resource: str = "service/checkout-api",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[MitigationDecision, RSSMState]:
        """
        Online filtering and mitigation update:
        1. Ingests new telemetry observation x_t.
        2. Updates latent state via RSSM posterior step.
        3. Invokes counterfactual evaluation and produces pre-emptive mitigation directive.

        Parameters
        ----------
        current_obs : Union[torch.Tensor, np.ndarray]
            Latest observation vector.
        prev_action : Union[int, MitigationAction]
            Last action executed.
        prev_state : RSSMState
            Previous state.
        target_resource : str
            Identifier of target service.
        metadata : Dict[str, Any], optional
            Extra cluster context.

        Returns
        -------
        Tuple[MitigationDecision, RSSMState]
            Decision directive and updated posterior state.
        """
        device = prev_state.h.device
        if isinstance(current_obs, np.ndarray):
            obs_tensor = torch.from_numpy(current_obs).float().to(device)
        else:
            obs_tensor = current_obs.to(device)

        if obs_tensor.dim() == 1:
            obs_tensor = obs_tensor.unsqueeze(0)

        act_tensor = torch.tensor([int(prev_action)], dtype=torch.long, device=device)

        # Update posterior state
        current_state = self.dreamer.world_model.step_posterior(
            obs=obs_tensor,
            action=act_tensor,
            prev_state=prev_state,
            sample=False,
        )

        decision = self.evaluate_and_mitigate(
            initial_state=current_state,
            target_resource=target_resource,
            metadata=metadata,
        )

        return decision, current_state
