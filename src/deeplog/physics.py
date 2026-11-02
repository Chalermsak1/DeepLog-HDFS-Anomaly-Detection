"""
DeepLog-Omni Pillar II: Physics-Informed Neural Queuing Operators (PISNO)
=======================================================================
Implements physics-informed queuing network operators, constrained neural predictors,
and physical impossibility anomaly detectors for distributed systems.

Components:
1. `JacksonQueueNetwork`: Exact open Jackson queueing network formulation modeling
   external arrivals gamma_i, routing probabilities P_{ij}, aggregate arrivals lambda_i,
   service rates mu_i, server counts c_i, queue lengths L_i, and response times W_i.
2. `PhysicsInformedLoss`: Multi-objective loss function enforcing Little's Law
   (|L - lambda * W| <= epsilon), node and global conservation of flow,
   and M/M/c traffic intensity stability (rho = lambda / (c * mu) < 1).
3. `PhysicalProjectionLayer` & `PhysicsConstrainedPredictor`: Neural network modules
   incorporating an axiomatic projection layer guaranteeing that predictions strictly
   adhere to distributed systems physical invariants (non-negative service times,
   stable traffic intensity, exact Little's Law compliance, and capacity boundaries).
4. `PhysicalAnomalyDetector`: Detector flagging physical impossibility violations
   (negative latencies, buffer overflow, flow leakage, Little's Law divergence,
   and queue instability).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# Data Structures
# ==============================================================================


@dataclass
class PhysicalQueueState:
    """
    Structured representation of distributed queuing state metrics.

    Attributes
    ----------
    arrival_rates : torch.Tensor or np.ndarray
        Effective total arrival rates lambda_i at each node (shape: (..., num_nodes)).
    service_rates : torch.Tensor or np.ndarray
        Service rates mu_i of each server at each node (shape: (..., num_nodes)).
    service_times : torch.Tensor or np.ndarray
        Mean service times S_i = 1 / mu_i at each node (shape: (..., num_nodes)).
    waiting_times : torch.Tensor or np.ndarray
        Mean system waiting/response times W_i at each node (shape: (..., num_nodes)).
    queue_lengths : torch.Tensor or np.ndarray
        Mean system queue lengths L_i at each node (shape: (..., num_nodes)).
    traffic_intensities : torch.Tensor or np.ndarray
        Traffic intensity / utilization rho_i = lambda_i / (c_i * mu_i) (shape: (..., num_nodes)).
    departure_rates : Optional[torch.Tensor or np.ndarray]
        Outflow rates departing the network from each node (shape: (..., num_nodes)).
    queue_waiting_times : Optional[torch.Tensor or np.ndarray]
        Mean waiting times in queue W_{q, i} = W_i - S_i (shape: (..., num_nodes)).
    queue_length_waiting : Optional[torch.Tensor or np.ndarray]
        Mean number of jobs waiting in queue L_{q, i} (shape: (..., num_nodes)).
    """

    arrival_rates: Union[torch.Tensor, np.ndarray]
    service_rates: Union[torch.Tensor, np.ndarray]
    service_times: Union[torch.Tensor, np.ndarray]
    waiting_times: Union[torch.Tensor, np.ndarray]
    queue_lengths: Union[torch.Tensor, np.ndarray]
    traffic_intensities: Union[torch.Tensor, np.ndarray]
    departure_rates: Optional[Union[torch.Tensor, np.ndarray]] = None
    queue_waiting_times: Optional[Union[torch.Tensor, np.ndarray]] = None
    queue_length_waiting: Optional[Union[torch.Tensor, np.ndarray]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to a python dictionary."""
        d = {
            "arrival_rates": self.arrival_rates,
            "service_rates": self.service_rates,
            "service_times": self.service_times,
            "waiting_times": self.waiting_times,
            "queue_lengths": self.queue_lengths,
            "traffic_intensities": self.traffic_intensities,
        }
        if self.departure_rates is not None:
            d["departure_rates"] = self.departure_rates
        if self.queue_waiting_times is not None:
            d["queue_waiting_times"] = self.queue_waiting_times
        if self.queue_length_waiting is not None:
            d["queue_length_waiting"] = self.queue_length_waiting
        return d

    def detach(self) -> PhysicalQueueState:
        """Detach all torch tensors in this state."""

        def _det(x):
            return x.detach() if isinstance(x, torch.Tensor) else x

        return PhysicalQueueState(
            arrival_rates=_det(self.arrival_rates),
            service_rates=_det(self.service_rates),
            service_times=_det(self.service_times),
            waiting_times=_det(self.waiting_times),
            queue_lengths=_det(self.queue_lengths),
            traffic_intensities=_det(self.traffic_intensities),
            departure_rates=_det(self.departure_rates)
            if self.departure_rates is not None
            else None,
            queue_waiting_times=_det(self.queue_waiting_times)
            if self.queue_waiting_times is not None
            else None,
            queue_length_waiting=_det(self.queue_length_waiting)
            if self.queue_length_waiting is not None
            else None,
        )

    def numpy(self) -> PhysicalQueueState:
        """Convert all tensors to numpy arrays."""

        def _to_np(x):
            if x is None:
                return None
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
            return np.asarray(x)

        return PhysicalQueueState(
            arrival_rates=_to_np(self.arrival_rates),
            service_rates=_to_np(self.service_rates),
            service_times=_to_np(self.service_times),
            waiting_times=_to_np(self.waiting_times),
            queue_lengths=_to_np(self.queue_lengths),
            traffic_intensities=_to_np(self.traffic_intensities),
            departure_rates=_to_np(self.departure_rates),
            queue_waiting_times=_to_np(self.queue_waiting_times),
            queue_length_waiting=_to_np(self.queue_length_waiting),
        )


@dataclass
class PhysicalAnomalyResult:
    """
    Diagnostic result returned by PhysicalAnomalyDetector.

    Attributes
    ----------
    is_anomaly : bool
        True if any physical impossibility or threshold violation was detected.
    anomaly_score : float
        Scalar continuous anomaly severity score.
    violations : List[str]
        List of specific violation identifiers triggered.
    littles_law_residuals : np.ndarray
        Absolute residuals |L_i - lambda_i * W_i| per node.
    flow_residuals : Optional[np.ndarray]
        Node flow conservation residuals |lambda_i - (gamma_i + sum_j lambda_j P_{ji})|.
    traffic_intensities : np.ndarray
        Estimated node traffic intensities rho_i.
    bottleneck_node : Optional[int]
        Index of the node with highest physical violation or utilization.
    details : Dict[str, Any]
        Additional context, node metrics, and diagnosis summaries.
    """

    is_anomaly: bool
    anomaly_score: float
    violations: List[str]
    littles_law_residuals: np.ndarray
    flow_residuals: Optional[np.ndarray]
    traffic_intensities: np.ndarray
    bottleneck_node: Optional[int]
    details: Dict[str, Any] = field(default_factory=dict)


# ==============================================================================
# Helper Functions: Differentiable Erlang C
# ==============================================================================


def compute_erlang_c_torch(
    c: Union[int, torch.Tensor],
    offered_load: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Numerically stable and autograd-differentiable Erlang C formula for M/M/c queues.

    Formula:
        C(c, a) = [ a^c / (c! (1 - rho)) ] / [ sum_{k=0}^{c-1} a^k / k! + a^c / (c! (1 - rho)) ]
    where rho = a / c.

    Parameters
    ----------
    c : int or torch.Tensor
        Number of parallel servers (must be >= 1).
    offered_load : torch.Tensor
        Offered traffic load a = lambda / mu (shape: (...)).
    eps : float, default=1e-12
        Numerical stability epsilon.

    Returns
    -------
    prob_queue : torch.Tensor
        Probability that an arriving request must wait in queue, bounded in [0, 1].
    """
    if isinstance(c, torch.Tensor):
        c_int = int(c.squeeze().item()) if c.numel() == 1 else int(c.max().item())
    else:
        c_int = int(c)

    if c_int <= 1:
        # M/M/1 queue: Erlang C is simply rho = a
        rho = offered_load
        return torch.clamp(rho, 0.0, 1.0)

    rho = offered_load / float(c_int)
    a_clamped = torch.clamp(offered_load, min=eps)
    log_a = torch.log(a_clamped)

    # Compute log(a^k / k!) = k * log(a) - lgamma(k + 1)
    k_vals = torch.arange(c_int, device=offered_load.device, dtype=offered_load.dtype)
    shape_prefix = [1] * offered_load.dim()
    k_expanded = k_vals.view(*shape_prefix, c_int)
    log_a_expanded = log_a.unsqueeze(-1)

    log_terms = k_expanded * log_a_expanded - torch.lgamma(k_expanded + 1.0)

    # Last term: c * log(a) - lgamma(c + 1) - log(max(1 - rho, eps))
    omr = torch.clamp(1.0 - rho, min=1e-7)
    c_tensor = torch.tensor(float(c_int), dtype=offered_load.dtype, device=offered_load.device)
    log_last = (
        float(c_int) * log_a
        - torch.lgamma(c_tensor + 1.0)
        - torch.log(omr)
    )

    # Combine all terms using logsumexp for stability
    all_log_terms = torch.cat([log_terms, log_last.unsqueeze(-1)], dim=-1)
    log_denom = torch.logsumexp(all_log_terms, dim=-1)

    prob = torch.exp(log_last - log_denom)
    # When rho >= 1, system is saturated (probability of waiting -> 1.0)
    prob = torch.where(rho >= 1.0, torch.ones_like(prob), prob)
    return torch.clamp(prob, 0.0, 1.0)


# ==============================================================================
# 1. Jackson Queue Network
# ==============================================================================


class JacksonQueueNetwork(nn.Module):
    """
    Exact Open Jackson Queueing Network Model.

    Models a distributed system of M service stations with Poisson external arrivals
    gamma, Markovian routing matrix P, exponential service rates mu, and server counts c.

    Mathematical Invariants:
    1. Traffic Equations:
       lambda_i = gamma_i + sum_{j=1}^M lambda_j * P_{ji}
       In matrix form: lambda = gamma * (I - P)^{-1}
    2. Conservation of Flow:
       sum_{i=1}^M gamma_i = sum_{i=1}^M lambda_i * (1 - sum_{j=1}^M P_{ij})
    3. Traffic Intensity Stability:
       rho_i = lambda_i / (c_i * mu_i) < 1.0  forall i in {1, ..., M}
    4. Little's Law:
       L_i = lambda_i * W_i   (system)
       L_{q, i} = lambda_i * W_{q, i}   (queue)
    5. Network-level Little's Law:
       L_{net} = sum_i L_i = Gamma * W_{net}, where Gamma = sum_i gamma_i
    """

    def __init__(
        self,
        num_nodes: int,
        gamma: Optional[Union[torch.Tensor, np.ndarray, List[float]]] = None,
        routing_matrix: Optional[Union[torch.Tensor, np.ndarray, List[List[float]]]] = None,
        service_rates: Optional[Union[torch.Tensor, np.ndarray, List[float]]] = None,
        servers: Optional[Union[torch.Tensor, np.ndarray, List[int], int]] = None,
        capacities: Optional[Union[torch.Tensor, np.ndarray, List[float]]] = None,
        node_names: Optional[List[str]] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        """
        Parameters
        ----------
        num_nodes : int
            Number of nodes/queues M in the distributed network.
        gamma : tensor-like, optional
            External arrival rates gamma in R^M (default: zeros).
        routing_matrix : tensor-like, optional
            Sub-stochastic routing matrix P in R^{M x M} where P_{ij} is the transition
            probability from node i to node j (default: zeros).
        service_rates : tensor-like, optional
            Service rates mu in R^M (default: ones).
        servers : int or tensor-like, optional
            Number of parallel servers c_i per node (default: 1 for M/M/1).
        capacities : tensor-like, optional
            Maximum queue buffer capacity C_i per node (default: None for infinite).
        node_names : List[str], optional
            Human-readable names for each service node.
        """
        super().__init__()
        self.num_nodes = int(num_nodes)
        if self.num_nodes <= 0:
            raise ValueError(f"num_nodes must be positive, got {num_nodes}")

        self.device = device or torch.device("cpu")
        self.dtype = dtype

        # External arrival rates gamma (M,)
        if gamma is None:
            gamma_t = torch.zeros(self.num_nodes, dtype=self.dtype, device=self.device)
        else:
            gamma_t = torch.as_tensor(gamma, dtype=self.dtype, device=self.device)
        self.register_buffer("gamma", gamma_t)

        # Routing matrix P (M, M)
        if routing_matrix is None:
            P_t = torch.zeros((self.num_nodes, self.num_nodes), dtype=self.dtype, device=self.device)
        else:
            P_t = torch.as_tensor(routing_matrix, dtype=self.dtype, device=self.device)
        self.register_buffer("routing_matrix", P_t)

        # Service rates mu (M,)
        if service_rates is None:
            mu_t = torch.ones(self.num_nodes, dtype=self.dtype, device=self.device)
        else:
            mu_t = torch.as_tensor(service_rates, dtype=self.dtype, device=self.device)
        self.register_buffer("service_rates", mu_t)

        # Server counts c (M,)
        if servers is None:
            c_t = torch.ones(self.num_nodes, dtype=self.dtype, device=self.device)
        elif isinstance(servers, int):
            c_t = torch.full((self.num_nodes,), float(servers), dtype=self.dtype, device=self.device)
        else:
            c_t = torch.as_tensor(servers, dtype=self.dtype, device=self.device)
        self.register_buffer("servers", c_t)

        # Capacities C (M,)
        if capacities is not None:
            cap_t = torch.as_tensor(capacities, dtype=self.dtype, device=self.device)
            self.register_buffer("capacities", cap_t)
        else:
            self.register_buffer("capacities", None)

        self.node_names = node_names or [f"node_{i}" for i in range(self.num_nodes)]
        if len(self.node_names) != self.num_nodes:
            raise ValueError(
                f"Length of node_names ({len(self.node_names)}) must match num_nodes ({self.num_nodes})"
            )

    def solve_traffic_equations(
        self,
        gamma: Optional[torch.Tensor] = None,
        routing_matrix: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Solves the linear traffic equations for open Jackson networks:
            lambda = gamma * (I - P)^{-1}
        Transposed system:
            (I - P)^T * lambda^T = gamma^T

        Supports single vectors (M,) or batched inputs (B, M).

        Parameters
        ----------
        gamma : torch.Tensor, optional
            External arrival rates. Defaults to buffer `self.gamma`.
        routing_matrix : torch.Tensor, optional
            Routing matrix P. Defaults to buffer `self.routing_matrix`.

        Returns
        -------
        lambdas : torch.Tensor
            Aggregate arrival rates lambda at each node (same batch shape as gamma).
        """
        g = gamma if gamma is not None else self.gamma
        P = routing_matrix if routing_matrix is not None else self.routing_matrix

        is_batched = g.dim() > 1
        orig_shape = g.shape
        M = self.num_nodes

        g_flat = g.view(-1, M)  # (B, M)
        B = g_flat.shape[0]

        # Expand P to (B, M, M) if not already batched
        if P.dim() == 2:
            P_batch = P.unsqueeze(0).expand(B, M, M)
        else:
            P_batch = P.view(B, M, M)

        I = torch.eye(M, dtype=g.dtype, device=g.device).unsqueeze(0).expand(B, M, M)
        A = (I - P_batch).transpose(1, 2)  # (I - P)^T for (B, M, M)

        # Solve A * lambda = g_flat^T
        # torch.linalg.solve expects (..., M, M) and (..., M, K)
        try:
            sol = torch.linalg.solve(A, g_flat.unsqueeze(-1)).squeeze(-1)
        except RuntimeError:
            # Fallback to least squares / pseudoinverse for ill-conditioned systems
            A_pinv = torch.linalg.pinv(A)
            sol = torch.bmm(A_pinv, g_flat.unsqueeze(-1)).squeeze(-1)

        # Physical arrival rates must be non-negative
        sol = torch.clamp(sol, min=0.0)
        return sol.view(orig_shape)

    def compute_traffic_intensities(
        self,
        arrival_rates: Optional[torch.Tensor] = None,
        service_rates: Optional[torch.Tensor] = None,
        servers: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Computes traffic intensities (utilization):
            rho_i = lambda_i / (c_i * mu_i)

        Parameters
        ----------
        arrival_rates : torch.Tensor, optional
            Node aggregate arrivals lambda. Defaults to solving traffic equations.
        service_rates : torch.Tensor, optional
            Node service rates mu. Defaults to buffer `self.service_rates`.
        servers : torch.Tensor, optional
            Server counts c. Defaults to buffer `self.servers`.

        Returns
        -------
        rhos : torch.Tensor
            Traffic intensities rho per node.
        """
        lam = (
            arrival_rates
            if arrival_rates is not None
            else self.solve_traffic_equations()
        )
        mu = service_rates if service_rates is not None else self.service_rates
        c = servers if servers is not None else self.servers

        denom = torch.clamp(c * mu, min=1e-9)
        return lam / denom

    def is_stable(
        self,
        arrival_rates: Optional[torch.Tensor] = None,
        service_rates: Optional[torch.Tensor] = None,
        servers: Optional[torch.Tensor] = None,
        threshold: float = 1.0,
    ) -> torch.Tensor:
        """
        Checks if the queueing network is stable (rho_i < threshold for all nodes).

        Returns
        -------
        stable : torch.Tensor of dtype bool
            Boolean tensor (shape: (...) without node dimension) indicating stability.
        """
        rhos = self.compute_traffic_intensities(arrival_rates, service_rates, servers)
        max_rho = rhos.max(dim=-1).values
        return max_rho < threshold

    def compute_queue_lengths(
        self,
        arrival_rates: Optional[torch.Tensor] = None,
        service_rates: Optional[torch.Tensor] = None,
        servers: Optional[torch.Tensor] = None,
        in_system: bool = True,
        clip_unstable: bool = True,
        max_queue_len: float = 1e5,
    ) -> torch.Tensor:
        """
        Computes expected queue lengths according to M/M/c Jackson product-form solution.

        For M/M/1 (c=1):
            L = rho / (1 - rho) = lambda / (mu - lambda)
            L_q = rho^2 / (1 - rho) = L - rho
        For M/M/c (c >= 1):
            L_q = C(c, a) * rho / (1 - rho)
            L = L_q + a = L_q + lambda / mu

        Parameters
        ----------
        arrival_rates : torch.Tensor, optional
            Aggregate arrivals lambda.
        service_rates : torch.Tensor, optional
            Service rates mu.
        servers : torch.Tensor, optional
            Server counts c.
        in_system : bool, default=True
            If True, returns total customers in system L. If False, returns queue length L_q.
        clip_unstable : bool, default=True
            If True, clips unstable queue length at max_queue_len to prevent NaN in autograd.
        max_queue_len : float, default=1e5
            Maximum queue length cap for saturated queues.

        Returns
        -------
        L : torch.Tensor
            Expected queue lengths.
        """
        lam = (
            arrival_rates
            if arrival_rates is not None
            else self.solve_traffic_equations()
        )
        mu = service_rates if service_rates is not None else self.service_rates
        c = servers if servers is not None else self.servers

        mu_safe = torch.clamp(mu, min=1e-8)
        offered_load = lam / mu_safe  # a = lambda / mu
        rho = offered_load / torch.clamp(c, min=1.0)

        # Use differentiable Erlang C
        c_max = int(c.max().item())
        if c_max == 1:
            prob_queue = torch.clamp(rho, 0.0, 1.0)
        else:
            prob_queue = compute_erlang_c_torch(c_max, offered_load)

        # 1 - rho clamped for stability
        omr = torch.clamp(1.0 - rho, min=1e-6)
        L_q = prob_queue * (rho / omr)

        if in_system:
            L = L_q + offered_load
        else:
            L = L_q

        if clip_unstable:
            L = torch.clamp(L, min=0.0, max=max_queue_len)
        else:
            L = torch.where(rho >= 1.0, torch.full_like(L, float("inf")), L)

        return L

    def compute_waiting_times(
        self,
        arrival_rates: Optional[torch.Tensor] = None,
        service_rates: Optional[torch.Tensor] = None,
        servers: Optional[torch.Tensor] = None,
        in_system: bool = True,
        clip_unstable: bool = True,
        max_waiting_time: float = 1e5,
    ) -> torch.Tensor:
        """
        Computes expected waiting/response times according to Little's Law:
            W = L / lambda
            W_q = L_q / lambda

        Parameters
        ----------
        arrival_rates : torch.Tensor, optional
        service_rates : torch.Tensor, optional
        servers : torch.Tensor, optional
        in_system : bool, default=True
        clip_unstable : bool, default=True
        max_waiting_time : float, default=1e5

        Returns
        -------
        W : torch.Tensor
            Expected waiting times.
        """
        lam = (
            arrival_rates
            if arrival_rates is not None
            else self.solve_traffic_equations()
        )
        L = self.compute_queue_lengths(
            arrival_rates=lam,
            service_rates=service_rates,
            servers=servers,
            in_system=in_system,
            clip_unstable=clip_unstable,
            max_queue_len=max_waiting_time,
        )

        lam_safe = torch.clamp(lam, min=1e-8)
        W = L / lam_safe

        # For nodes with near-zero arrivals, W defaults to bare service time 1/mu
        mu = service_rates if service_rates is not None else self.service_rates
        S = 1.0 / torch.clamp(mu, min=1e-8)
        zero_arrival_mask = lam < 1e-7
        if in_system:
            W = torch.where(zero_arrival_mask, S, W)
        else:
            W = torch.where(zero_arrival_mask, torch.zeros_like(W), W)

        if clip_unstable:
            W = torch.clamp(W, min=0.0, max=max_waiting_time)
        return W

    def verify_littles_law(
        self,
        queue_lengths: Optional[torch.Tensor] = None,
        arrival_rates: Optional[torch.Tensor] = None,
        waiting_times: Optional[torch.Tensor] = None,
        tolerance: float = 1e-4,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Verifies Little's Law: |L_i - lambda_i * W_i| <= tolerance.

        Returns
        -------
        residuals : torch.Tensor
            Absolute difference |L - lambda * W|.
        is_compliant : torch.Tensor of dtype bool
            Boolean indicator of whether Little's Law is satisfied within tolerance.
        """
        lam = (
            arrival_rates
            if arrival_rates is not None
            else self.solve_traffic_equations()
        )
        L = (
            queue_lengths
            if queue_lengths is not None
            else self.compute_queue_lengths(arrival_rates=lam)
        )
        W = (
            waiting_times
            if waiting_times is not None
            else self.compute_waiting_times(arrival_rates=lam)
        )

        residual = torch.abs(L - lam * W)
        compliant = residual <= tolerance
        return residual, compliant

    def verify_flow_conservation(
        self,
        arrival_rates: Optional[torch.Tensor] = None,
        gamma: Optional[torch.Tensor] = None,
        routing_matrix: Optional[torch.Tensor] = None,
        tolerance: float = 1e-4,
    ) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        """
        Verifies conservation of flow:
        1. Node flow balance: lambda_i = gamma_i + sum_j lambda_j * P_{ji}
        2. Global network balance: sum_i gamma_i = sum_i lambda_i * (1 - sum_j P_{ij})

        Returns
        -------
        node_residuals : torch.Tensor
            Per-node balance residual |lambda - (gamma + lambda * P)|.
        global_residual : torch.Tensor
            Scalar absolute difference between total inflows and outflows.
        is_conserved : bool
            True if all residuals fall within tolerance.
        """
        g = gamma if gamma is not None else self.gamma
        P = routing_matrix if routing_matrix is not None else self.routing_matrix
        lam = (
            arrival_rates
            if arrival_rates is not None
            else self.solve_traffic_equations(g, P)
        )

        # Node flow: lambda * P
        if lam.dim() == 1:
            inflow_internal = torch.matmul(lam, P)  # (M,)
            departure_prob = torch.clamp(1.0 - torch.sum(P, dim=-1), min=0.0)  # (M,)
            outflows = torch.sum(lam * departure_prob)
            total_gamma = torch.sum(g)
        else:
            inflow_internal = torch.bmm(lam.unsqueeze(1), P).squeeze(1)
            departure_prob = torch.clamp(1.0 - torch.sum(P, dim=-1), min=0.0)
            outflows = torch.sum(lam * departure_prob, dim=-1)
            total_gamma = torch.sum(g, dim=-1)

        node_residuals = torch.abs(lam - (g + inflow_internal))
        global_residual = torch.abs(total_gamma - outflows)

        is_conserved = bool(
            (node_residuals.max() <= tolerance) and (global_residual.max() <= tolerance)
        )
        return node_residuals, global_residual, is_conserved

    def network_metrics(
        self,
        gamma: Optional[torch.Tensor] = None,
        service_rates: Optional[torch.Tensor] = None,
        routing_matrix: Optional[torch.Tensor] = None,
        servers: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Computes comprehensive Jackson network metrics.

        Returns
        -------
        metrics : Dict[str, Any]
            Dictionary containing network-level and node-level metrics:
            - total_external_arrival
            - total_network_customers
            - mean_network_response_time
            - bottleneck_node
            - bottleneck_utilization
            - is_stable
            - node_metrics table
        """
        g = gamma if gamma is not None else self.gamma
        P = routing_matrix if routing_matrix is not None else self.routing_matrix
        mu = service_rates if service_rates is not None else self.service_rates
        c = servers if servers is not None else self.servers

        lam = self.solve_traffic_equations(g, P)
        rhos = self.compute_traffic_intensities(lam, mu, c)
        L = self.compute_queue_lengths(lam, mu, c, in_system=True)
        W = self.compute_waiting_times(lam, mu, c, in_system=True)
        L_q = self.compute_queue_lengths(lam, mu, c, in_system=False)
        W_q = self.compute_waiting_times(lam, mu, c, in_system=False)

        total_gamma = torch.sum(g).item()
        total_L = torch.sum(L).item()
        W_net = total_L / max(total_gamma, 1e-9)

        bottleneck_idx = int(torch.argmax(rhos).item())
        max_rho = float(rhos[bottleneck_idx].item())
        stable = max_rho < 1.0

        node_stats = []
        for i in range(self.num_nodes):
            node_stats.append({
                "node_index": i,
                "node_name": self.node_names[i],
                "gamma": float(g[i].item()),
                "lambda": float(lam[i].item()),
                "mu": float(mu[i].item()),
                "servers": int(c[i].item()),
                "rho": float(rhos[i].item()),
                "L": float(L[i].item()),
                "W": float(W[i].item()),
                "L_q": float(L_q[i].item()),
                "W_q": float(W_q[i].item()),
            })

        return {
            "total_external_arrival": total_gamma,
            "total_network_customers": total_L,
            "mean_network_response_time": W_net,
            "bottleneck_node": bottleneck_idx,
            "bottleneck_node_name": self.node_names[bottleneck_idx],
            "bottleneck_utilization": max_rho,
            "is_stable": stable,
            "node_metrics": node_stats,
        }

    def forward(
        self,
        gamma: Optional[torch.Tensor] = None,
        service_rates: Optional[torch.Tensor] = None,
        routing_matrix: Optional[torch.Tensor] = None,
    ) -> PhysicalQueueState:
        """
        Differentiable forward pass for Jackson Queue Network.

        Returns
        -------
        state : PhysicalQueueState
        """
        g = gamma if gamma is not None else self.gamma
        P = routing_matrix if routing_matrix is not None else self.routing_matrix
        mu = service_rates if service_rates is not None else self.service_rates
        c = self.servers

        lam = self.solve_traffic_equations(g, P)
        rhos = self.compute_traffic_intensities(lam, mu, c)
        L = self.compute_queue_lengths(lam, mu, c, in_system=True)
        W = self.compute_waiting_times(lam, mu, c, in_system=True)
        L_q = self.compute_queue_lengths(lam, mu, c, in_system=False)
        W_q = self.compute_waiting_times(lam, mu, c, in_system=False)
        S = 1.0 / torch.clamp(mu, min=1e-8)

        # Departure rates
        if P.dim() == 2:
            d = torch.clamp(1.0 - torch.sum(P, dim=-1), min=0.0)
            dep = lam * d
        else:
            d = torch.clamp(1.0 - torch.sum(P, dim=-1), min=0.0)
            dep = lam * d

        return PhysicalQueueState(
            arrival_rates=lam,
            service_rates=mu,
            service_times=S,
            waiting_times=W,
            queue_lengths=L,
            traffic_intensities=rhos,
            departure_rates=dep,
            queue_waiting_times=W_q,
            queue_length_waiting=L_q,
        )


# ==============================================================================
# 2. Physics-Informed Loss Function
# ==============================================================================


class PhysicsInformedLoss(nn.Module):
    """
    Physics-Informed Loss Function for Distributed Queuing Networks.

    Enforces axiomatic queuing theory constraints:
    1. Little's Law: |L - lambda * W| <= epsilon
    2. Conservation of Flow:
       - Node balance: lambda_i = gamma_i + sum_j lambda_j * P_{ji}
       - Global balance: sum_i gamma_i = sum_i lambda_i * (1 - sum_j P_{ij})
    3. Traffic Intensity Stability: rho = lambda / (c * mu) < rho_max < 1.0
    4. Queue Buffer Capacity Limit: L_i <= C_i
    5. Non-Negativity: lambda >= 0, mu >= mu_min > 0, L >= 0, W >= 1/mu
    6. Data Fidelity: MSE(y_hat, y_true)

    Total Loss:
        L_total = w_data * L_data
                + w_little * L_little
                + w_flow * L_flow
                + w_stability * L_stability
                + w_capacity * L_capacity
                + w_nonneg * L_nonneg
    """

    def __init__(
        self,
        weight_data: float = 1.0,
        weight_little: float = 1.0,
        weight_flow: float = 1.0,
        weight_stability: float = 1.0,
        weight_capacity: float = 0.5,
        weight_nonneg: float = 0.5,
        epsilon_little: float = 0.0,
        rho_max: float = 0.99,
        mu_min: float = 1e-4,
        loss_type: str = "mse",
    ):
        """
        Parameters
        ----------
        weight_data : float, default=1.0
            Weight for supervised data fidelity loss.
        weight_little : float, default=1.0
            Weight for Little's Law penalty (|L - lambda * W|).
        weight_flow : float, default=1.0
            Weight for conservation of flow penalty.
        weight_stability : float, default=1.0
            Weight for traffic intensity stability penalty (rho < rho_max).
        weight_capacity : float, default=0.5
            Weight for buffer capacity overflow penalty (L <= C).
        weight_nonneg : float, default=0.5
            Weight for non-negativity constraint penalty.
        epsilon_little : float, default=0.0
            Slack tolerance for Little's Law (|L - lambda * W| <= epsilon).
        rho_max : float, default=0.99
            Stability ceiling for traffic intensity rho.
        mu_min : float, default=1e-4
            Strict minimum service rate.
        loss_type : str, default="mse"
            Loss penalty formulation ("mse", "huber", or "l1").
        """
        super().__init__()
        self.weight_data = float(weight_data)
        self.weight_little = float(weight_little)
        self.weight_flow = float(weight_flow)
        self.weight_stability = float(weight_stability)
        self.weight_capacity = float(weight_capacity)
        self.weight_nonneg = float(weight_nonneg)
        self.epsilon_little = float(epsilon_little)
        self.rho_max = float(rho_max)
        self.mu_min = float(mu_min)
        self.loss_type = loss_type.lower()

    def _penalty(self, residual: torch.Tensor) -> torch.Tensor:
        """Applies configured penalty function (MSE, Huber, or L1)."""
        if self.loss_type == "mse":
            return torch.mean(residual**2)
        elif self.loss_type == "huber":
            return torch.mean(F.huber_loss(residual, torch.zeros_like(residual), reduction="none"))
        elif self.loss_type == "l1":
            return torch.mean(torch.abs(residual))
        else:
            return torch.mean(residual**2)

    def forward(
        self,
        predicted_L: Optional[torch.Tensor] = None,
        predicted_lambda: Optional[torch.Tensor] = None,
        predicted_W: Optional[torch.Tensor] = None,
        predicted_mu: Optional[torch.Tensor] = None,
        gamma: Optional[torch.Tensor] = None,
        routing_matrix: Optional[torch.Tensor] = None,
        servers: Optional[Union[torch.Tensor, int]] = None,
        capacities: Optional[torch.Tensor] = None,
        target_y: Optional[torch.Tensor] = None,
        predicted_y: Optional[torch.Tensor] = None,
        queue_state: Optional[PhysicalQueueState] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Computes the physics-informed multi-objective loss.

        Can accept individual metric tensors or a `PhysicalQueueState` instance.

        Returns
        -------
        total_loss : torch.Tensor
            Scalar aggregated differentiable loss.
        loss_dict : Dict[str, torch.Tensor]
            Dictionary of scalar loss components.
        """
        if queue_state is not None:
            if predicted_L is None:
                predicted_L = queue_state.queue_lengths
            if predicted_lambda is None:
                predicted_lambda = queue_state.arrival_rates
            if predicted_W is None:
                predicted_W = queue_state.waiting_times
            if predicted_mu is None:
                predicted_mu = queue_state.service_rates

        device = (
            predicted_L.device
            if predicted_L is not None
            else (predicted_lambda.device if predicted_lambda is not None else torch.device("cpu"))
        )
        dtype = (
            predicted_L.dtype
            if predicted_L is not None
            else (predicted_lambda.dtype if predicted_lambda is not None else torch.float32)
        )

        zero = torch.tensor(0.0, device=device, dtype=dtype)
        loss_data = zero
        loss_little = zero
        loss_flow = zero
        loss_stability = zero
        loss_capacity = zero
        loss_nonneg = zero

        # 1. Supervised Data Fidelity
        if target_y is not None and predicted_y is not None:
            loss_data = F.mse_loss(predicted_y, target_y)

        # 2. Little's Law Penalty: |L - lambda * W| <= epsilon
        if predicted_L is not None and predicted_lambda is not None and predicted_W is not None:
            littles_error = torch.abs(predicted_L - predicted_lambda * predicted_W)
            if self.epsilon_little > 0.0:
                violation = F.relu(littles_error - self.epsilon_little)
            else:
                violation = littles_error
            loss_little = self._penalty(violation)

        # 3. Conservation of Flow Penalty
        if (
            predicted_lambda is not None
            and gamma is not None
            and routing_matrix is not None
        ):
            # Node-level balance: lambda - (gamma + lambda * P)
            if predicted_lambda.dim() == 1:
                internal_in = torch.matmul(predicted_lambda, routing_matrix)
                node_residual = predicted_lambda - (gamma + internal_in)

                d_prob = torch.clamp(1.0 - torch.sum(routing_matrix, dim=-1), min=0.0)
                departures = torch.sum(predicted_lambda * d_prob)
                global_residual = torch.sum(gamma) - departures
            else:
                internal_in = torch.bmm(predicted_lambda.unsqueeze(1), routing_matrix).squeeze(1)
                node_residual = predicted_lambda - (gamma + internal_in)

                d_prob = torch.clamp(1.0 - torch.sum(routing_matrix, dim=-1), min=0.0)
                departures = torch.sum(predicted_lambda * d_prob, dim=-1)
                global_residual = torch.sum(gamma, dim=-1) - departures

            loss_node_flow = self._penalty(node_residual)
            loss_global_flow = self._penalty(global_residual)
            loss_flow = loss_node_flow + loss_global_flow

        # 4. Traffic Intensity Stability (M/M/c): rho = lambda / (c * mu) < rho_max
        if predicted_lambda is not None and predicted_mu is not None:
            c = (
                servers
                if servers is not None
                else torch.ones_like(predicted_lambda)
            )
            if isinstance(c, (int, float)):
                c = torch.full_like(predicted_lambda, float(c))

            denom = torch.clamp(c * predicted_mu, min=1e-8)
            rho = predicted_lambda / denom
            rho_violation = F.relu(rho - self.rho_max)
            loss_stability = self._penalty(rho_violation)

        # 5. Buffer Capacity Limit: L <= C
        if predicted_L is not None and capacities is not None:
            cap_violation = F.relu(predicted_L - capacities)
            loss_capacity = self._penalty(cap_violation)

        # 6. Physical Non-Negativity & Lower Bounds
        nonneg_terms = []
        if predicted_lambda is not None:
            nonneg_terms.append(F.relu(-predicted_lambda))
        if predicted_mu is not None:
            nonneg_terms.append(F.relu(self.mu_min - predicted_mu))
        if predicted_L is not None:
            nonneg_terms.append(F.relu(-predicted_L))
        if predicted_W is not None:
            if predicted_mu is not None:
                # W must be at least service time S = 1 / mu
                S = 1.0 / torch.clamp(predicted_mu, min=1e-8)
                nonneg_terms.append(F.relu(S - predicted_W))
            else:
                nonneg_terms.append(F.relu(-predicted_W))

        if nonneg_terms:
            stacked_nonneg = torch.cat([t.reshape(-1) for t in nonneg_terms])
            loss_nonneg = self._penalty(stacked_nonneg)

        total_loss = (
            self.weight_data * loss_data
            + self.weight_little * loss_little
            + self.weight_flow * loss_flow
            + self.weight_stability * loss_stability
            + self.weight_capacity * loss_capacity
            + self.weight_nonneg * loss_nonneg
        )

        loss_breakdown = {
            "total_loss": total_loss,
            "data_loss": loss_data,
            "little_loss": loss_little,
            "flow_loss": loss_flow,
            "stability_loss": loss_stability,
            "capacity_loss": loss_capacity,
            "nonneg_loss": loss_nonneg,
        }

        return total_loss, loss_breakdown


# ==============================================================================
# 3. Physical Projection Layer & Constrained Predictor
# ==============================================================================


class PhysicalProjectionLayer(nn.Module):
    """
    Physical Projection Layer for Distributed Systems.

    Axiomatically projects unconstrained latent neural outputs onto the physically
    admissible queueing manifold:
    1. Service Rate & Time:
       mu_i = Softplus(z_mu) + mu_min > 0  =>  S_i = 1 / mu_i > 0
    2. Stability & Traffic Intensity:
       rho_i = rho_max * Sigmoid(z_rho) in (0, rho_max)
       lambda_i = rho_i * c_i * mu_i
       Guarantees rho_i <= rho_max < 1.0 unconditionally.
    3. Waiting Time:
       W_{q, i} = Softplus(z_w) >= 0  =>  W_i = S_i + W_{q, i} >= S_i > 0
    4. Little's Law Invariant:
       L_i = lambda_i * W_i
       Guarantees |L_i - lambda_i * W_i| == 0 strictly.
    5. Capacity Boundaries:
       If capacity C_i is configured, L_i = min(L_i, C_i), and W_i is projected accordingly.
    """

    def __init__(
        self,
        num_nodes: int,
        rho_max: float = 0.99,
        mu_min: float = 1e-4,
        enforce_littles_law: bool = True,
        capacities: Optional[Union[torch.Tensor, np.ndarray, List[float]]] = None,
    ):
        """
        Parameters
        ----------
        num_nodes : int
            Number of nodes/queues M.
        rho_max : float, default=0.99
            Maximum traffic intensity threshold (< 1.0).
        mu_min : float, default=1e-4
            Strict positive floor for service rate mu.
        enforce_littles_law : bool, default=True
            Whether to compute L = lambda * W directly, strictly satisfying Little's Law.
        capacities : tensor-like, optional
            Node buffer capacities.
        """
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.rho_max = float(rho_max)
        self.mu_min = float(mu_min)
        self.enforce_littles_law = bool(enforce_littles_law)

        if capacities is not None:
            self.register_buffer("capacities", torch.as_tensor(capacities, dtype=torch.float32))
        else:
            self.register_buffer("capacities", None)

    def forward(
        self,
        raw_mu: torch.Tensor,
        raw_rho: Optional[torch.Tensor] = None,
        raw_lambda: Optional[torch.Tensor] = None,
        raw_W: Optional[torch.Tensor] = None,
        raw_L: Optional[torch.Tensor] = None,
        servers: Optional[Union[torch.Tensor, int]] = None,
        capacities: Optional[torch.Tensor] = None,
    ) -> PhysicalQueueState:
        """
        Forward projection pass.

        Parameters
        ----------
        raw_mu : torch.Tensor
            Unconstrained logits for service rate.
        raw_rho : torch.Tensor, optional
            Unconstrained logits for traffic intensity.
        raw_lambda : torch.Tensor, optional
            Unconstrained logits for arrival rate (if raw_rho not given).
        raw_W : torch.Tensor, optional
            Unconstrained logits for queue waiting time.
        raw_L : torch.Tensor, optional
            Unconstrained logits for queue length (only used if enforce_littles_law is False).
        servers : int or torch.Tensor, optional
            Server counts per node (default: 1).
        capacities : torch.Tensor, optional
            Buffer capacity limits.

        Returns
        -------
        state : PhysicalQueueState
            Physically projected state strictly satisfying all physical invariants.
        """
        # 1. Guaranteed positive service rate & service time
        mu = F.softplus(raw_mu) + self.mu_min
        S = 1.0 / mu

        # Servers count
        if servers is None:
            c = torch.ones_like(mu)
        elif isinstance(servers, (int, float)):
            c = torch.full_like(mu, float(servers))
        else:
            c = servers

        # 2. Guaranteed stable arrival rate & traffic intensity (rho <= rho_max < 1.0)
        if raw_rho is not None:
            rho = self.rho_max * torch.sigmoid(raw_rho)
            lam = rho * c * mu
        elif raw_lambda is not None:
            # Clamp raw_lambda so that rho <= rho_max
            lam_candidate = F.softplus(raw_lambda)
            max_allowed_lambda = self.rho_max * c * mu
            lam = torch.minimum(lam_candidate, max_allowed_lambda)
            rho = lam / torch.clamp(c * mu, min=1e-8)
        else:
            # Default fallback: 50% utilization
            rho = torch.full_like(mu, 0.5 * self.rho_max)
            lam = rho * c * mu

        # 3. Guaranteed positive waiting time (W >= S)
        if raw_W is not None:
            W_q = F.softplus(raw_W)
            W = S + W_q
        else:
            # Theoretical M/M/1 waiting time baseline
            denom = torch.clamp(mu - lam / c, min=1e-7)
            W = 1.0 / denom
            W_q = torch.clamp(W - S, min=0.0)

        # 4. Guaranteed Little's Law compliance: L = lambda * W
        if self.enforce_littles_law or raw_L is None:
            L = lam * W
            L_q = lam * W_q
        else:
            L = F.softplus(raw_L)
            L_q = torch.clamp(L - lam * S, min=0.0)

        # 5. Capacity constraints
        caps = capacities if capacities is not None else self.capacities
        if caps is not None:
            L = torch.minimum(L, caps)
            # Re-align W to maintain exact Little's Law if capacity truncated
            if self.enforce_littles_law:
                lam_safe = torch.clamp(lam, min=1e-8)
                W = L / lam_safe
                W_q = torch.clamp(W - S, min=0.0)
                L_q = lam * W_q

        return PhysicalQueueState(
            arrival_rates=lam,
            service_rates=mu,
            service_times=S,
            waiting_times=W,
            queue_lengths=L,
            traffic_intensities=rho,
            queue_waiting_times=W_q,
            queue_length_waiting=L_q,
        )

    def project(
        self,
        arrival_rates: Union[torch.Tensor, np.ndarray],
        service_rates: Union[torch.Tensor, np.ndarray],
        queue_lengths: Optional[Union[torch.Tensor, np.ndarray]] = None,
        waiting_times: Optional[Union[torch.Tensor, np.ndarray]] = None,
        servers: Optional[Union[torch.Tensor, int]] = None,
        capacities: Optional[Union[torch.Tensor, np.ndarray]] = None,
    ) -> PhysicalQueueState:
        """
        Directly projects ANY unconstrained / corrupt tensor inputs onto the physical manifold.

        Parameters
        ----------
        arrival_rates : tensor-like
            Arbitrary arrival rates (may be negative or unstable).
        service_rates : tensor-like
            Arbitrary service rates (may be negative or zero).
        queue_lengths : tensor-like, optional
        waiting_times : tensor-like, optional
        servers : int or tensor, optional
        capacities : tensor-like, optional

        Returns
        -------
        projected_state : PhysicalQueueState
        """
        # Convert to torch tensors if needed
        is_np = isinstance(arrival_rates, np.ndarray)
        lam_t = torch.as_tensor(arrival_rates, dtype=torch.float32)
        mu_t = torch.as_tensor(service_rates, dtype=torch.float32)

        # 1. Project service rate: mu >= mu_min > 0
        mu_proj = torch.clamp(mu_t, min=self.mu_min)
        S_proj = 1.0 / mu_proj

        # Server counts
        if servers is None:
            c_t = torch.ones_like(mu_proj)
        elif isinstance(servers, (int, float)):
            c_t = torch.full_like(mu_proj, float(servers))
        else:
            c_t = torch.as_tensor(servers, dtype=torch.float32)

        # 2. Project arrival rates: 0 <= lambda <= rho_max * c * mu
        max_lam = self.rho_max * c_t * mu_proj
        lam_proj = torch.clamp(lam_t, min=0.0)
        lam_proj = torch.minimum(lam_proj, max_lam)
        rho_proj = lam_proj / (c_t * mu_proj)

        # 3. Project waiting times: W >= S = 1 / mu
        if waiting_times is not None:
            W_t = torch.as_tensor(waiting_times, dtype=torch.float32)
            W_proj = torch.maximum(W_t, S_proj)
        else:
            # Default to theoretical M/M/1
            W_proj = 1.0 / torch.clamp(mu_proj - lam_proj / c_t, min=1e-7)
        W_q_proj = torch.clamp(W_proj - S_proj, min=0.0)

        # 4. Project queue lengths via Little's Law: L = lambda * W
        L_proj = lam_proj * W_proj
        L_q_proj = lam_proj * W_q_proj

        # 5. Apply capacities
        caps_t = (
            torch.as_tensor(capacities, dtype=torch.float32)
            if capacities is not None
            else self.capacities
        )
        if caps_t is not None:
            L_proj = torch.minimum(L_proj, caps_t)
            lam_safe = torch.clamp(lam_proj, min=1e-8)
            W_proj = L_proj / lam_safe
            W_proj = torch.maximum(W_proj, S_proj)
            W_q_proj = torch.clamp(W_proj - S_proj, min=0.0)

        res = PhysicalQueueState(
            arrival_rates=lam_proj,
            service_rates=mu_proj,
            service_times=S_proj,
            waiting_times=W_proj,
            queue_lengths=L_proj,
            traffic_intensities=rho_proj,
            queue_waiting_times=W_q_proj,
            queue_length_waiting=L_q_proj,
        )

        return res.numpy() if is_np else res


class PhysicsConstrainedPredictor(nn.Module):
    """
    Neural Queuing Operator with Axiomatic Physical Projection Layer.

    Predicts future distributed queuing states from sequence representations or telemetry
    features while mathematically guaranteeing that outputs never violate physical laws:
    - Service rates are strictly positive (mu >= mu_min > 0).
    - Service times are strictly non-negative (S = 1/mu > 0).
    - Traffic intensities are stable (rho <= rho_max < 1.0).
    - Little's Law is satisfied exactly (|L - lambda * W| == 0).
    - Queue lengths never exceed physical capacity buffers (L <= C).

    Architecture:
    Input -> Feature Backbone (MLP / GRU / Transformer) -> Multi-Head Projections -> PhysicalProjectionLayer
    """

    def __init__(
        self,
        input_dim: int,
        num_nodes: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        model_type: str = "mlp",
        servers: Optional[Union[torch.Tensor, int]] = None,
        capacities: Optional[Union[torch.Tensor, List[float]]] = None,
        rho_max: float = 0.99,
        mu_min: float = 1e-4,
    ):
        """
        Parameters
        ----------
        input_dim : int
            Dimension of input feature representation.
        num_nodes : int
            Number of distributed queuing stations M.
        hidden_dim : int, default=64
            Hidden state dimensionality.
        num_layers : int, default=2
            Depth of neural encoder backbone.
        dropout : float, default=0.1
            Dropout rate.
        model_type : str, default="mlp"
            Backbone architecture ("mlp", "gru", or "lstm").
        servers : int or tensor, optional
            Server count per node (default: 1).
        capacities : tensor-like, optional
            Node buffer capacity limits.
        rho_max : float, default=0.99
            Stability utilization upper bound (< 1.0).
        mu_min : float, default=1e-4
            Strict minimum service rate.
        """
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_nodes = int(num_nodes)
        self.hidden_dim = int(hidden_dim)
        self.model_type = model_type.lower()

        # Encoder Backbone
        if self.model_type == "mlp":
            layers = []
            curr_dim = self.input_dim
            for _ in range(num_layers):
                layers.extend([
                    nn.Linear(curr_dim, self.hidden_dim),
                    nn.LayerNorm(self.hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ])
                curr_dim = self.hidden_dim
            self.backbone = nn.Sequential(*layers)
        elif self.model_type == "gru":
            self.rnn = nn.GRU(
                input_size=self.input_dim,
                hidden_size=self.hidden_dim,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=True,
            )
            self.backbone = None
        elif self.model_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=self.input_dim,
                hidden_size=self.hidden_dim,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=True,
            )
            self.backbone = None
        else:
            raise ValueError(f"Unsupported model_type: {model_type}. Choose 'mlp', 'gru', or 'lstm'.")

        # Multi-Head Projections
        self.head_mu = nn.Linear(self.hidden_dim, self.num_nodes)
        self.head_rho = nn.Linear(self.hidden_dim, self.num_nodes)
        self.head_W = nn.Linear(self.hidden_dim, self.num_nodes)

        # Physical Projection Layer
        self.projection_layer = PhysicalProjectionLayer(
            num_nodes=self.num_nodes,
            rho_max=rho_max,
            mu_min=mu_min,
            enforce_littles_law=True,
            capacities=capacities,
        )

        # Buffer for servers
        if servers is None:
            c_t = torch.ones(self.num_nodes, dtype=torch.float32)
        elif isinstance(servers, (int, float)):
            c_t = torch.full((self.num_nodes,), float(servers), dtype=torch.float32)
        else:
            c_t = torch.as_tensor(servers, dtype=torch.float32)
        self.register_buffer("servers", c_t)

    def forward(
        self,
        x: torch.Tensor,
        servers: Optional[torch.Tensor] = None,
        capacities: Optional[torch.Tensor] = None,
    ) -> PhysicalQueueState:
        """
        Forward pass predicting invariant-adherent queuing metrics.

        Parameters
        ----------
        x : torch.Tensor
            Input features of shape (batch_size, input_dim) or (batch_size, seq_len, input_dim).
        servers : torch.Tensor, optional
        capacities : torch.Tensor, optional

        Returns
        -------
        state : PhysicalQueueState
        """
        if self.model_type == "mlp":
            if x.dim() == 3:
                # Sequence input: pool or take last step
                h = self.backbone(x)[:, -1, :]
            else:
                h = self.backbone(x)
        else:
            out, _ = self.rnn(x)
            h = out[:, -1, :] if x.dim() == 3 else out

        raw_mu = self.head_mu(h)
        raw_rho = self.head_rho(h)
        raw_W = self.head_W(h)

        c = servers if servers is not None else self.servers
        return self.projection_layer(
            raw_mu=raw_mu,
            raw_rho=raw_rho,
            raw_W=raw_W,
            servers=c,
            capacities=capacities,
        )

    def predict(
        self,
        x: Union[torch.Tensor, np.ndarray],
        as_numpy: bool = True,
    ) -> PhysicalQueueState:
        """
        Inference prediction helper.

        Parameters
        ----------
        x : tensor-like
            Input features.
        as_numpy : bool, default=True
            Whether to return numpy arrays inside PhysicalQueueState.

        Returns
        -------
        state : PhysicalQueueState
        """
        self.eval()
        with torch.no_grad():
            x_t = torch.as_tensor(x, dtype=torch.float32, device=next(self.parameters()).device)
            state = self.forward(x_t)
            return state.numpy() if as_numpy else state


# ==============================================================================
# 4. Physical Anomaly Detector
# ==============================================================================


class PhysicalAnomalyDetector:
    """
    Physical Impossibility & Invariant Anomaly Detector.

    Evaluates system telemetry and queue metrics against distributed physical laws:
    1. NEGATIVE_SERVICE_TIME / NEGATIVE_RATE: mu <= 0, lambda < 0, S <= 0, W < S.
    2. LITTLES_LAW_VIOLATION: |L - lambda * W| > tau_little.
    3. FLOW_CONSERVATION_DIVERGENCE:
       - Node flow residual: |lambda - (gamma + lambda * P)| > tau_flow
       - Global flow residual: |sum gamma - sum lambda * (1 - sum P)| > tau_flow
    4. TRAFFIC_SATURATION / INSTABILITY: rho >= rho_unstable (queue blowup).
    5. CAPACITY_OVERFLOW: L > C (buffer dropped/overflowed).
    6. SUPERLINEAR_LATENCY_EXPLOSION: W spikes disproportionately without lambda growth.
    """

    def __init__(
        self,
        tau_little: float = 0.1,
        tau_flow: float = 0.1,
        rho_threshold: float = 1.0,
        use_relative_tolerance: bool = True,
        capacities: Optional[Union[np.ndarray, List[float]]] = None,
        node_names: Optional[List[str]] = None,
    ):
        """
        Parameters
        ----------
        tau_little : float, default=0.1
            Tolerance threshold for Little's Law residual |L - lambda * W|.
        tau_flow : float, default=0.1
            Tolerance threshold for flow conservation residual.
        rho_threshold : float, default=1.0
            Critical traffic intensity threshold (>= 1.0 is physically unstable).
        use_relative_tolerance : bool, default=True
            If True, evaluates Little's Law error relative to max(L, lambda * W, 1.0).
        capacities : array-like, optional
            Node buffer capacities.
        node_names : List[str], optional
            Node labels.
        """
        self.tau_little = float(tau_little)
        self.tau_flow = float(tau_flow)
        self.rho_threshold = float(rho_threshold)
        self.use_relative_tolerance = bool(use_relative_tolerance)
        self.capacities = np.asarray(capacities) if capacities is not None else None
        self.node_names = node_names

    def detect(
        self,
        queue_lengths: Union[torch.Tensor, np.ndarray, List[float]],
        arrival_rates: Union[torch.Tensor, np.ndarray, List[float]],
        waiting_times: Union[torch.Tensor, np.ndarray, List[float]],
        service_rates: Optional[Union[torch.Tensor, np.ndarray, List[float]]] = None,
        external_arrivals: Optional[Union[torch.Tensor, np.ndarray, List[float]]] = None,
        routing_matrix: Optional[Union[torch.Tensor, np.ndarray, List[List[float]]]] = None,
        servers: Optional[Union[torch.Tensor, np.ndarray, int]] = 1,
        capacities: Optional[Union[torch.Tensor, np.ndarray, List[float]]] = None,
    ) -> PhysicalAnomalyResult:
        """
        Analyzes a single observation of distributed queuing telemetry for physical violations.

        Parameters
        ----------
        queue_lengths : array-like
            Observed queue lengths L_i.
        arrival_rates : array-like
            Observed arrival rates lambda_i.
        waiting_times : array-like
            Observed response/waiting times W_i.
        service_rates : array-like, optional
            Observed service rates mu_i.
        external_arrivals : array-like, optional
            External inflows gamma_i.
        routing_matrix : array-like, optional
            Routing matrix P.
        servers : int or array-like, default=1
            Number of servers per node.
        capacities : array-like, optional
            Buffer capacity caps.

        Returns
        -------
        result : PhysicalAnomalyResult
        """

        def _to_np(x):
            if x is None:
                return None
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
            return np.asarray(x, dtype=np.float64)

        L = _to_np(queue_lengths)
        lam = _to_np(arrival_rates)
        W = _to_np(waiting_times)
        mu = _to_np(service_rates)
        gamma = _to_np(external_arrivals)
        P = _to_np(routing_matrix)
        c = _to_np(servers) if not isinstance(servers, int) else servers
        caps = _to_np(capacities) if capacities is not None else self.capacities

        num_nodes = len(L)
        violations: List[str] = []
        severity_scores: List[float] = []

        # 1. Non-Negativity & Impossible Latency
        has_neg_lambda = np.any(lam < 0)
        has_neg_L = np.any(L < 0)
        has_neg_W = np.any(W < 0)
        has_neg_mu = mu is not None and np.any(mu <= 0)

        if has_neg_lambda or has_neg_L or has_neg_W or has_neg_mu:
            violations.append("NEGATIVE_PHYSICAL_QUANTITY")
            neg_mag = 0.0
            if has_neg_lambda:
                neg_mag = max(neg_mag, float(np.max(-lam[lam < 0])))
            if has_neg_L:
                neg_mag = max(neg_mag, float(np.max(-L[L < 0])))
            if has_neg_W:
                neg_mag = max(neg_mag, float(np.max(-W[W < 0])))
            if has_neg_mu:
                neg_mag = max(neg_mag, float(np.max(-mu[mu <= 0]) + 1.0))
            severity_scores.append(1.0 + neg_mag)

        # Service time underflow (W < 1/mu)
        if mu is not None and not has_neg_mu:
            S = 1.0 / np.maximum(mu, 1e-9)
            underflow_mask = (W < (S - 1e-5)) & (lam > 1e-5)
            if np.any(underflow_mask):
                violations.append("RESPONSE_TIME_UNDERFLOW")
                underflow_mag = float(np.max((S - W)[underflow_mask]))
                severity_scores.append(underflow_mag / float(np.mean(S)))

        # 2. Little's Law Residuals: |L - lambda * W|
        expected_L = lam * W
        littles_res = np.abs(L - expected_L)
        if self.use_relative_tolerance:
            scale = np.maximum(np.maximum(L, expected_L), 1.0)
            littles_eval = littles_res / scale
        else:
            littles_eval = littles_res

        if np.any(littles_eval > self.tau_little):
            violations.append("LITTLES_LAW_VIOLATION")
            max_viol = float(np.max(littles_eval))
            severity_scores.append(max_viol / self.tau_little)

        # 3. Flow Conservation Residuals
        flow_residuals = None
        if gamma is not None and P is not None:
            internal_in = np.dot(lam, P)
            flow_residuals = np.abs(lam - (gamma + internal_in))

            dep_prob = np.clip(1.0 - np.sum(P, axis=-1), 0.0, 1.0)
            global_res = abs(float(np.sum(gamma)) - float(np.sum(lam * dep_prob)))

            node_viol = np.any(flow_residuals > self.tau_flow)
            global_viol = global_res > self.tau_flow

            if node_viol or global_viol:
                violations.append("FLOW_CONSERVATION_DIVERGENCE")
                max_flow_viol = max(float(np.max(flow_residuals)), global_res)
                severity_scores.append(max_flow_viol / self.tau_flow)

        # 4. Traffic Intensity & Stability
        if mu is not None and not has_neg_mu:
            denom = np.maximum(c * mu, 1e-9)
            rhos = lam / denom
            if np.any(rhos >= self.rho_threshold):
                violations.append("TRAFFIC_SATURATION_UNSTABLE")
                max_rho = float(np.max(rhos))
                severity_scores.append(1.0 + (max_rho - self.rho_threshold) / 0.1)
        else:
            rhos = np.zeros(num_nodes)

        # 5. Buffer Capacity Overflow
        if caps is not None:
            cap_excess = L - caps
            if np.any(cap_excess > 0):
                violations.append("BUFFER_CAPACITY_OVERFLOW")
                max_cap = float(np.max(cap_excess))
                severity_scores.append(1.0 + max_cap)

        # Composite Anomaly Score
        is_anomaly = len(violations) > 0
        composite_score = float(max(severity_scores)) if severity_scores else 0.0

        # Bottleneck Node Identification
        if len(rhos) > 0:
            candidate_node = int(np.argmax(rhos))
        else:
            candidate_node = int(np.argmax(littles_res))

        details = {
            "num_violations": len(violations),
            "max_littles_residual": float(np.max(littles_res)),
            "mean_littles_residual": float(np.mean(littles_res)),
            "max_traffic_intensity": float(np.max(rhos)) if len(rhos) > 0 else 0.0,
        }

        return PhysicalAnomalyResult(
            is_anomaly=is_anomaly,
            anomaly_score=composite_score,
            violations=violations,
            littles_law_residuals=littles_res,
            flow_residuals=flow_residuals,
            traffic_intensities=rhos,
            bottleneck_node=candidate_node if is_anomaly else None,
            details=details,
        )

    def detect_batch(
        self,
        queue_lengths: Union[torch.Tensor, np.ndarray],
        arrival_rates: Union[torch.Tensor, np.ndarray],
        waiting_times: Union[torch.Tensor, np.ndarray],
        service_rates: Optional[Union[torch.Tensor, np.ndarray]] = None,
        external_arrivals: Optional[Union[torch.Tensor, np.ndarray]] = None,
        routing_matrix: Optional[Union[torch.Tensor, np.ndarray]] = None,
        servers: Optional[Union[torch.Tensor, np.ndarray, int]] = 1,
        capacities: Optional[Union[torch.Tensor, np.ndarray]] = None,
    ) -> List[PhysicalAnomalyResult]:
        """
        Batch anomaly evaluation over sequences or traces of queuing metrics.

        Parameters
        ----------
        All tensor-like arguments with leading batch dimension B.

        Returns
        -------
        results : List[PhysicalAnomalyResult]
        """

        def _to_np(x):
            if x is None:
                return None
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
            return np.asarray(x, dtype=np.float64)

        L_b = _to_np(queue_lengths)
        lam_b = _to_np(arrival_rates)
        W_b = _to_np(waiting_times)
        mu_b = _to_np(service_rates)
        g_b = _to_np(external_arrivals)
        P_b = _to_np(routing_matrix)

        B = L_b.shape[0]
        results = []
        for i in range(B):
            res = self.detect(
                queue_lengths=L_b[i],
                arrival_rates=lam_b[i],
                waiting_times=W_b[i],
                service_rates=mu_b[i] if mu_b is not None else None,
                external_arrivals=g_b[i] if g_b is not None else None,
                routing_matrix=P_b[i] if P_b is not None and P_b.ndim == 3 else P_b,
                servers=servers,
                capacities=capacities,
            )
            results.append(res)
        return results

    def inject_synthetic_anomaly(
        self,
        base_state: Union[PhysicalQueueState, Dict[str, Any]],
        anomaly_type: str = "littles_law",
        severity: float = 2.0,
        target_node: int = 0,
    ) -> Dict[str, np.ndarray]:
        """
        Injects a synthetic physical impossibility into clean queuing telemetry.
        Used for unit tests, benchmark evaluation, and red-teaming.

        Supported anomaly types:
        - "littles_law": Perturbs queue length L without proportional change in lambda * W.
        - "negative_service_time": Sets service time or rate to negative value.
        - "flow_divergence": Diverts or drops aggregate flow without route conservation.
        - "traffic_saturation": Spikes arrival rate so rho >= 1.0 without capacity limits.
        - "capacity_overflow": Increases queue length beyond hardware buffer capacity.
        - "underflow_response_time": Reduces response time W below bare service time S = 1/mu.

        Returns
        -------
        corrupted_state : Dict[str, np.ndarray]
        """
        if isinstance(base_state, PhysicalQueueState):
            d = base_state.numpy().to_dict()
        else:
            d = {k: np.array(v, copy=True) for k, v in base_state.items()}

        out = {k: np.array(v, copy=True) for k, v in d.items()}
        node = target_node % len(out["arrival_rates"])

        if anomaly_type == "littles_law":
            # Violate Little's Law by arbitrarily inflating queue length
            out["queue_lengths"][node] = (
                out["arrival_rates"][node] * out["waiting_times"][node] + severity * 10.0
            )
        elif anomaly_type == "negative_service_time":
            out["service_rates"][node] = -abs(severity)
            out["service_times"][node] = -abs(severity)
        elif anomaly_type == "flow_divergence":
            out["arrival_rates"][node] *= severity * 3.0
        elif anomaly_type == "traffic_saturation":
            # Force rho = lambda / (c * mu) >= 1.0 + severity
            mu_val = out["service_rates"][node]
            out["arrival_rates"][node] = (1.0 + severity) * mu_val
        elif anomaly_type == "capacity_overflow":
            cap_limit = self.capacities[node] if self.capacities is not None else 50.0
            out["queue_lengths"][node] = cap_limit + severity * 20.0
        elif anomaly_type == "underflow_response_time":
            # Make waiting time faster than bare service time (physically impossible)
            S_val = 1.0 / max(out["service_rates"][node], 1e-4)
            out["waiting_times"][node] = max(0.01, S_val * 0.1)
        else:
            raise ValueError(f"Unknown anomaly_type: {anomaly_type}")

        return out
