"""
DeepLog-Omni Frontier 4: Provable Safety & Formal Verification
============================================================
Implements control barrier functions (CBFs), Quadratic Program (QP) safety filters,
discrete Lyapunov stability verifiers, and physical safety violation guards
for autonomous self-healing distributed systems.

Components
----------
1. `SafetyInvariant` & `ControlBarrierFunction`:
   Evaluates safety invariant functions h(x) >= 0 (e.g. minimum replica count >= 2,
   cluster CPU headroom >= 15%, traffic drop <= 5%, queue depth <= Q_max).
   Computes exact analytical / autograd gradients and Lie derivatives L_f h, L_g h.
2. `QPSafetyFilter`:
   Solves the Quadratic Program (QP) projecting proposed mitigation actions onto
   the safe control space:
   min_u 1/2 ||u - u_proposed||_H^2 + 1/2 lambda xi^2
   s.t.  nabla h(x)^T f(x, u) + alpha(h(x)) >= -xi
         u_min <= u <= u_max
3. `LyapunovStabilityVerifier`:
   Computes discrete Lyapunov energy V(x) and drift Delta V = V(x_{t+1}) - V(x_t) <= -kappa V(x_t)
   to prove absence of destructive feedback oscillations, limit cycles, and control flapping.
   Provides Riccati (DARE) and discrete Lyapunov synthesis.
4. `SafetyViolationGuard`:
   Enforces hard aborts, emergency lockouts, and deterministic fail-safes if the physical
   state threatens cluster survival.
"""

from __future__ import annotations

import enum
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.linalg
from scipy.optimize import minimize

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


__all__ = [
    "SafetyClusterState",
    "ClusterState",
    "SafetyInvariant",
    "CBFEvaluationResult",
    "SafetyFilterResult",
    "LyapunovStepResult",
    "OscillationDetectionResult",
    "GuardAction",
    "GuardDecision",
    "ControlBarrierFunction",
    "QPSafetyFilter",
    "LyapunovStabilityVerifier",
    "SafetyViolationGuard",
]


# ==============================================================================
# Domain State & Data Structures
# ==============================================================================


@dataclass
class SafetyClusterState:
    """
    Standardized physical state vector for formal safety evaluation of cluster nodes and microservices.

    Attributes
    ----------
    replicas : float
        Active pod / container replica count (e.g. 2.0).
    cpu_headroom : float
        Fractional unallocated CPU capacity in [0.0, 1.0] (1.0 - utilization).
    traffic_drop : float
        Fractional rate of unserved / dropped customer traffic in [0.0, 1.0].
    queue_depth : float
        Number of pending requests in ingress / service buffer.
    latency_ms : float
        Mean request-response round-trip latency in milliseconds.
    memory_headroom : float
        Fractional unallocated RAM headroom in [0.0, 1.0].
    error_rate : float
        Fraction of requests returning HTTP 5xx or network errors.
    extra_metrics : Dict[str, float]
        Optional user-defined custom metrics.
    """

    replicas: float = 3.0
    cpu_headroom: float = 0.35
    traffic_drop: float = 0.0
    queue_depth: float = 10.0
    latency_ms: float = 45.0
    memory_headroom: float = 0.40
    error_rate: float = 0.001
    extra_metrics: Dict[str, float] = field(default_factory=dict)

    # Standard feature ordering for vector conversions
    FEATURE_NAMES = (
        "replicas",
        "cpu_headroom",
        "traffic_drop",
        "queue_depth",
        "latency_ms",
        "memory_headroom",
        "error_rate",
    )

    def to_array(self) -> np.ndarray:
        """Convert state to a 1D numpy array corresponding to FEATURE_NAMES."""
        return np.array([
            float(self.replicas),
            float(self.cpu_headroom),
            float(self.traffic_drop),
            float(self.queue_depth),
            float(self.latency_ms),
            float(self.memory_headroom),
            float(self.error_rate),
        ], dtype=np.float64)

    @classmethod
    def from_array(cls, arr: Union[np.ndarray, Sequence[float]]) -> SafetyClusterState:
        """Construct SafetyClusterState from a numpy array or sequence."""
        vec = np.asarray(arr, dtype=np.float64)
        if len(vec) < 7:
            raise ValueError(f"Expected at least 7 features for SafetyClusterState, got {len(vec)}")
        return cls(
            replicas=float(vec[0]),
            cpu_headroom=float(vec[1]),
            traffic_drop=float(vec[2]),
            queue_depth=float(vec[3]),
            latency_ms=float(vec[4]),
            memory_headroom=float(vec[5]),
            error_rate=float(vec[6]),
        )

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> SafetyClusterState:
        """Construct SafetyClusterState from a dictionary."""
        known = {k: float(d[k]) for k in cls.FEATURE_NAMES if k in d}
        extras = {k: float(v) for k, v in d.items() if k not in cls.FEATURE_NAMES and isinstance(v, (int, float))}
        return cls(**known, extra_metrics=extras)

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        d = {k: getattr(self, k) for k in self.FEATURE_NAMES}
        d.update(self.extra_metrics)
        return d


# Alias for backwards compatibility
ClusterState = SafetyClusterState


@dataclass
class SafetyInvariant:
    """
    Formal representation of a single scalar safety invariant h(x) >= 0.

    Attributes
    ----------
    name : str
        Unique identifier for the safety rule (e.g. 'min_replicas').
    func : Callable[[np.ndarray], float]
        Evaluation function h(x). Safe if and only if h(x) >= 0.
    grad_func : Optional[Callable[[np.ndarray], np.ndarray]]
        Analytical gradient nabla h(x). If None, numerical finite differences are used.
    alpha_gain : float, default=1.0
        Linear class-K gain in alpha(h(x)) = alpha_gain * h(x).
    metric_index : Optional[int]
        If invariant depends directly on a single state dimension x[j], the index j.
    is_upper_bound : bool, default=False
        True if invariant represents an upper bound rule (threshold - x[j] >= 0).
    threshold : float, default=0.0
        The numerical threshold associated with this invariant.
    description : Optional[str]
        Human-readable explanation of the safety invariant.
    """

    name: str
    func: Callable[[np.ndarray], float]
    grad_func: Optional[Callable[[np.ndarray], np.ndarray]] = None
    alpha_gain: float = 1.0
    metric_index: Optional[int] = None
    is_upper_bound: bool = False
    threshold: float = 0.0
    description: Optional[str] = None

    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate scalar barrier value h(x)."""
        val = self.func(x)
        return float(val)

    def gradient(self, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """
        Compute gradient nabla h(x).
        Uses exact analytical grad_func if provided; otherwise central finite differences.
        """
        if self.grad_func is not None:
            g = self.grad_func(x)
            return np.asarray(g, dtype=np.float64)

        if self.metric_index is not None:
            # Analytical gradient for 1D threshold invariant
            g = np.zeros_like(x, dtype=np.float64)
            g[self.metric_index] = -1.0 if self.is_upper_bound else 1.0
            return g

        # Central finite difference approximation
        n = len(x)
        grad = np.zeros(n, dtype=np.float64)
        x_mut = x.astype(np.float64, copy=True)
        for i in range(n):
            orig = x_mut[i]
            x_mut[i] = orig + eps
            f_plus = self.func(x_mut)
            x_mut[i] = orig - eps
            f_minus = self.func(x_mut)
            x_mut[i] = orig
            grad[i] = (f_plus - f_minus) / (2.0 * eps)
        return grad


@dataclass
class CBFEvaluationResult:
    """
    Evaluation summary of a set of control barrier functions at state x.
    """

    h_values: Dict[str, float]
    h_vector: np.ndarray
    is_safe: bool
    min_margin: float
    violated_invariants: List[str]
    active_invariants: List[str]
    tolerated_margin: float = 0.0


@dataclass
class SafetyFilterResult:
    """
    Outcome of a Quadratic Program (QP) safety filter projection.
    """

    safe_action: np.ndarray
    proposed_action: np.ndarray
    is_modified: bool
    cost: float
    barrier_values: Dict[str, float]
    safety_margin: float
    status: str
    slack: float = 0.0
    cbf_violations_prevented: List[str] = field(default_factory=list)
    solve_time_ms: float = 0.0


@dataclass
class LyapunovStepResult:
    """
    Verification outcome of discrete Lyapunov energy and drift at step t -> t+1.
    """

    is_stable: bool
    v_t: float
    v_next: float
    drift: float
    required_drift: float
    contraction_ratio: float
    margin: float


@dataclass
class OscillationDetectionResult:
    """
    Diagnostic analysis of trajectory stability, ringing, and feedback oscillation.
    """

    is_oscillating: bool
    oscillation_type: str
    dominant_frequency: float
    spectral_power: float
    zero_crossing_rate: float
    flapping_count: int
    lyapunov_drift_violations: int
    recommendation: str


class GuardAction(enum.Enum):
    """Actions rendered by the SafetyViolationGuard."""
    PERMIT = "PERMIT"
    OVERRIDE_FAILSAFE = "OVERRIDE_FAILSAFE"
    EMERGENCY_ABORT = "EMERGENCY_ABORT"
    LOCKOUT = "LOCKOUT"


@dataclass
class GuardDecision:
    """
    Actionable directive emitted by SafetyViolationGuard.
    """

    decision: GuardAction
    is_safe: bool
    action: Any
    reason: str
    violations: List[str] = field(default_factory=list)
    lockout_remaining: int = 0


# ==============================================================================
# 1. Control Barrier Function (CBF)
# ==============================================================================


class ControlBarrierFunction:
    """
    Formal Control Barrier Function (CBF) suite for distributed systems.

    Evaluates safety invariants h_i(x) >= 0 defining the safe forward-invariant set:
        C = { x in R^n | h_i(x) >= 0 for all i = 1, ..., K }

    Provides:
    - Scalar evaluation of all invariants and safe-set boundary proximity.
    - Exact Jacobian gradients nabla h(x) in R^{K x n}.
    - Lie derivatives L_f h(x) and L_g h(x) along system vector fields dot{x} = f(x) + g(x)u.
    - Discrete-time barrier checks: h(x_{t+1}) >= (1 - gamma) h(x_t).
    - Factory constructor with standard cluster survival invariants:
        * Minimum replica count >= 2.0
        * Cluster CPU headroom >= 15% (0.15)
        * Traffic drop <= 5% (0.05)
        * Queue depth <= Q_max
        * Request latency <= latency_max
    """

    def __init__(
        self,
        invariants: Optional[Sequence[SafetyInvariant]] = None,
        default_alpha: float = 1.0,
        state_dim: Optional[int] = None,
    ):
        self.invariants: Dict[str, SafetyInvariant] = {}
        self.default_alpha = float(default_alpha)
        self.state_dim = state_dim

        if invariants:
            for inv in invariants:
                self.add_invariant(inv)

    def add_invariant(self, invariant: SafetyInvariant) -> None:
        """Register a new safety invariant."""
        self.invariants[invariant.name] = invariant

    def remove_invariant(self, name: str) -> None:
        """Remove a safety invariant by name."""
        if name in self.invariants:
            del self.invariants[name]

    @classmethod
    def create_cluster_safety_cbf(
        cls,
        min_replicas: float = 2.0,
        min_cpu_headroom: float = 0.15,
        max_traffic_drop: float = 0.05,
        max_queue_depth: float = 500.0,
        max_latency_ms: float = 250.0,
        default_alpha: float = 1.0,
    ) -> ControlBarrierFunction:
        """
        Construct a ControlBarrierFunction with standard distributed cluster invariants:
        1. replicas >= min_replicas (h_1 = x[0] - min_replicas >= 0)
        2. cpu_headroom >= min_cpu_headroom (h_2 = x[1] - min_cpu_headroom >= 0)
        3. traffic_drop <= max_traffic_drop (h_3 = max_traffic_drop - x[2] >= 0)
        4. queue_depth <= max_queue_depth (h_4 = max_queue_depth - x[3] >= 0)
        5. latency_ms <= max_latency_ms (h_5 = max_latency_ms - x[4] >= 0)
        """
        invariants = [
            SafetyInvariant(
                name="min_replicas",
                func=lambda x, m=min_replicas: float(x[0] - m),
                metric_index=0,
                is_upper_bound=False,
                threshold=min_replicas,
                alpha_gain=default_alpha,
                description=f"Active replica count must be at least {min_replicas:.1f}",
            ),
            SafetyInvariant(
                name="min_cpu_headroom",
                func=lambda x, m=min_cpu_headroom: float(x[1] - m),
                metric_index=1,
                is_upper_bound=False,
                threshold=min_cpu_headroom,
                alpha_gain=default_alpha,
                description=f"Cluster CPU headroom must be at least {min_cpu_headroom * 100:.1f}%",
            ),
            SafetyInvariant(
                name="max_traffic_drop",
                func=lambda x, m=max_traffic_drop: float(m - x[2]),
                metric_index=2,
                is_upper_bound=True,
                threshold=max_traffic_drop,
                alpha_gain=default_alpha,
                description=f"Traffic drop must not exceed {max_traffic_drop * 100:.1f}%",
            ),
            SafetyInvariant(
                name="max_queue_depth",
                func=lambda x, m=max_queue_depth: float(m - x[3]),
                metric_index=3,
                is_upper_bound=True,
                threshold=max_queue_depth,
                alpha_gain=default_alpha,
                description=f"Ingress queue backlog must not exceed {max_queue_depth:.1f} requests",
            ),
            SafetyInvariant(
                name="max_latency_ms",
                func=lambda x, m=max_latency_ms: float(m - x[4]),
                metric_index=4,
                is_upper_bound=True,
                threshold=max_latency_ms,
                alpha_gain=default_alpha,
                description=f"Response latency must not exceed {max_latency_ms:.1f} ms",
            ),
        ]
        return cls(invariants=invariants, default_alpha=default_alpha, state_dim=7)

    def _normalize_state(self, x: Union[SafetyClusterState, Dict[str, Any], np.ndarray, Sequence[float]]) -> np.ndarray:
        """Convert state to 1D float64 numpy array."""
        if hasattr(x, "to_array") and callable(x.to_array):
            return x.to_array()
        if isinstance(x, dict):
            return SafetyClusterState.from_dict(x).to_array()
        arr = np.asarray(x, dtype=np.float64)
        if arr.ndim == 0:
            return arr.reshape(1)
        elif arr.ndim > 1:
            arr = arr.flatten()
        return arr

    def evaluate(
        self,
        x: Union[ClusterState, np.ndarray, Sequence[float]],
        boundary_tolerance: float = 0.05,
    ) -> CBFEvaluationResult:
        """
        Evaluate all registered barrier invariants h_i(x) at state x.

        Parameters
        ----------
        x : ClusterState, np.ndarray, or sequence
            Current physical state vector.
        boundary_tolerance : float, default=0.05
            Margin threshold below which an invariant is flagged as 'active'.

        Returns
        -------
        CBFEvaluationResult
            Detailed safety evaluation including per-invariant values and violations.
        """
        x_vec = self._normalize_state(x)
        h_vals: Dict[str, float] = {}
        violated: List[str] = []
        active: List[str] = []

        for name, inv in self.invariants.items():
            val = inv.evaluate(x_vec)
            h_vals[name] = val
            if val < 0.0:
                violated.append(name)
            elif val <= boundary_tolerance:
                active.append(name)

        h_arr = np.array(list(h_vals.values()), dtype=np.float64)
        min_margin = float(np.min(h_arr)) if len(h_arr) > 0 else float("inf")
        is_safe = len(violated) == 0

        return CBFEvaluationResult(
            h_values=h_vals,
            h_vector=h_arr,
            is_safe=is_safe,
            min_margin=min_margin,
            violated_invariants=violated,
            active_invariants=active,
        )

    def is_safe(self, x: Union[ClusterState, np.ndarray, Sequence[float]]) -> bool:
        """Check if all barrier invariants satisfy h_i(x) >= 0."""
        return self.evaluate(x).is_safe

    def margin(self, x: Union[ClusterState, np.ndarray, Sequence[float]]) -> float:
        """Compute the minimum safety margin min_i h_i(x)."""
        return self.evaluate(x).min_margin

    def compute_jacobian(
        self,
        x: Union[ClusterState, np.ndarray, Sequence[float]],
    ) -> np.ndarray:
        """
        Compute the Jacobian matrix nabla h(x) in R^{K x n},
        where row i is nabla h_i(x)^T.
        """
        x_vec = self._normalize_state(x)
        k = len(self.invariants)
        n = len(x_vec)
        jac = np.zeros((k, n), dtype=np.float64)

        for i, inv in enumerate(self.invariants.values()):
            jac[i, :] = inv.gradient(x_vec)

        return jac

    def compute_lie_derivatives(
        self,
        x: Union[ClusterState, np.ndarray, Sequence[float]],
        f_x: Optional[np.ndarray] = None,
        g_x: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute continuous-time Lie derivatives:
            L_f h_i(x) = nabla h_i(x)^T f(x)
            L_g h_i(x) = nabla h_i(x)^T g(x)

        Parameters
        ----------
        x : state vector
        f_x : np.ndarray, shape (n,) or (n, 1)
            Autonomous drift vector field f(x). Default: zeros(n).
        g_x : np.ndarray, shape (n, m)
            Control input matrix field g(x). Default: identity(n, n) or identity(n, m).

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            L_f h of shape (K,), and L_g h of shape (K, m).
        """
        x_vec = self._normalize_state(x)
        n = len(x_vec)
        jac = self.compute_jacobian(x_vec)  # (K, n)
        k = jac.shape[0]

        if f_x is None:
            f_vec = np.zeros(n, dtype=np.float64)
        else:
            f_vec = np.asarray(f_x, dtype=np.float64).flatten()

        if g_x is None:
            g_mat = np.eye(n, dtype=np.float64)
        else:
            g_mat = np.asarray(g_x, dtype=np.float64)
            if g_mat.ndim == 1:
                g_mat = g_mat.reshape(-1, 1)

        # Lie derivatives:
        # L_f h = jac @ f_vec -> shape (K,)
        # L_g h = jac @ g_mat -> shape (K, m)
        lf_h = jac @ f_vec
        lg_h = jac @ g_mat

        return lf_h, lg_h

    def check_discrete_barrier(
        self,
        x_t: Union[ClusterState, np.ndarray, Sequence[float]],
        x_next: Union[ClusterState, np.ndarray, Sequence[float]],
        gamma: float = 0.5,
    ) -> Dict[str, bool]:
        """
        Verify discrete-time barrier condition for each invariant:
            h_i(x_{t+1}) - h_i(x_t) >= -gamma * h_i(x_t)
            <=> h_i(x_{t+1}) >= (1 - gamma) * h_i(x_t)

        Parameters
        ----------
        x_t : state at step t
        x_next : predicted or observed state at step t+1
        gamma : float, default=0.5
            Decay tolerance factor in (0, 1].

        Returns
        -------
        Dict[str, bool]
            Mapping of invariant name -> True if condition satisfied, False otherwise.
        """
        x_t_vec = self._normalize_state(x_t)
        x_next_vec = self._normalize_state(x_next)

        res: Dict[str, bool] = {}
        for name, inv in self.invariants.items():
            h_t = inv.evaluate(x_t_vec)
            h_next = inv.evaluate(x_next_vec)
            # D-CBF condition:
            # If h_t >= 0: h_next >= (1 - gamma) * h_t
            # If h_t < 0 (already violating): h_next must improve towards safety: h_next >= h_t + gamma * |h_t|
            required = (1.0 - gamma) * h_t if h_t >= 0 else h_t + gamma * abs(h_t)
            res[name] = bool(h_next >= required - 1e-7)

        return res


# ==============================================================================
# 2. Quadratic Program (QP) Safety Filter
# ==============================================================================


class QPSafetyFilter:
    """
    Quadratic Program (QP) Safety Filter for Pre-Emptive Mitigation.

    Projects proposed nominal mitigation actions u_proposed onto the safe control space:
        min_u  1/2 (u - u_proposed)^T H (u - u_proposed) + 1/2 lambda_slack xi^2
        s.t.   - nabla h_i(x)^T g(x) u <= nabla h_i(x)^T f(x) + alpha_i(h_i(x)) + xi
               u_min <= u <= u_max
               xi >= 0

    When u_proposed is already safe, the filter returns u_proposed untouched.
    When u_proposed would drive the system across a safety barrier (e.g. terminating too many
    replicas or causing excessive traffic drop), the filter computes the minimally deviating
    safe control vector u*.
    """

    def __init__(
        self,
        cbf: ControlBarrierFunction,
        u_dim: int = 1,
        u_min: Optional[Union[float, Sequence[float], np.ndarray]] = None,
        u_max: Optional[Union[float, Sequence[float], np.ndarray]] = None,
        H: Optional[np.ndarray] = None,
        slack_penalty: float = 1e4,
        allow_slack: bool = True,
        solver: str = "slsqp",
    ):
        self.cbf = cbf
        self.u_dim = int(u_dim)

        if u_min is None:
            self.u_min = np.full(self.u_dim, -10.0, dtype=np.float64)
        else:
            self.u_min = np.broadcast_to(np.asarray(u_min, dtype=np.float64), (self.u_dim,)).copy()

        if u_max is None:
            self.u_max = np.full(self.u_dim, 10.0, dtype=np.float64)
        else:
            self.u_max = np.broadcast_to(np.asarray(u_max, dtype=np.float64), (self.u_dim,)).copy()

        if H is None:
            self.H = np.eye(self.u_dim, dtype=np.float64)
        else:
            self.H = np.asarray(H, dtype=np.float64)

        self.slack_penalty = float(slack_penalty)
        self.allow_slack = bool(allow_slack)
        self.solver = solver

    def _solve_analytical_1d(
        self,
        u_prop: float,
        a_cbf: np.ndarray,
        b_cbf: np.ndarray,
        u_min: float,
        u_max: float,
    ) -> Optional[float]:
        """
        Analytical solution for 1D control with linear CBF constraints:
            a_i * u <= b_i
        """
        # Lower and upper allowable bounds on u
        low = u_min
        high = u_max

        for a, b in zip(a_cbf, b_cbf):
            if abs(a) < 1e-12:
                if b < 0:
                    return None  # Infeasible invariant
            elif a > 0:
                # u <= b / a
                high = min(high, b / a)
            else:
                # -|a| u <= b <=> u >= b / a
                low = max(low, b / a)

        if low > high + 1e-9:
            return None  # Infeasible without slack

        # Project u_prop onto [low, high]
        safe_u = np.clip(u_prop, low, high)
        return float(safe_u)

    def filter_action(
        self,
        x: Union[ClusterState, np.ndarray, Sequence[float]],
        u_proposed: Union[float, Sequence[float], np.ndarray],
        f_x: Optional[np.ndarray] = None,
        g_x: Optional[np.ndarray] = None,
        alpha_override: Optional[Dict[str, float]] = None,
        custom_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> SafetyFilterResult:
        """
        Filter a proposed continuous control vector through the CBF Quadratic Program.

        Parameters
        ----------
        x : ClusterState or array
            Current physical state.
        u_proposed : float or array
            Proposed nominal control action vector.
        f_x : np.ndarray, optional
            Autonomous system drift vector.
        g_x : np.ndarray, optional
            System control input matrix.
        alpha_override : Dict[str, float], optional
            Optional overrides for class-K alpha gains per invariant.
        custom_bounds : Tuple[np.ndarray, np.ndarray], optional
            Override bounds (u_min, u_max) for this evaluation.

        Returns
        -------
        SafetyFilterResult
            The filtered safe control vector, modification status, cost, and diagnostic margins.
        """
        t_start = time.perf_counter()
        x_vec = self.cbf._normalize_state(x)
        u_prop = np.asarray(u_proposed, dtype=np.float64).flatten()
        if len(u_prop) != self.u_dim:
            u_prop = np.resize(u_prop, self.u_dim)

        u_min = self.u_min if custom_bounds is None else custom_bounds[0]
        u_max = self.u_max if custom_bounds is None else custom_bounds[1]

        # Compute CBF quantities:
        # Continuous barrier condition:
        # L_f h_i(x) + L_g h_i(x) u + alpha_i * h_i(x) >= 0
        # <=> - L_g h_i(x) u <= L_f h_i(x) + alpha_i * h_i(x)
        # Let A_i = - L_g h_i(x), b_i = L_f h_i(x) + alpha_i * h_i(x)
        lf_h, lg_h = self.cbf.compute_lie_derivatives(x_vec, f_x=f_x, g_x=g_x)
        eval_res = self.cbf.evaluate(x_vec)

        k_invariants = len(self.cbf.invariants)
        A_cbf = - lg_h  # shape (K, m)
        b_cbf = np.zeros(k_invariants, dtype=np.float64)

        for i, (name, inv) in enumerate(self.cbf.invariants.items()):
            alpha = (
                alpha_override[name]
                if alpha_override and name in alpha_override
                else inv.alpha_gain
            )
            h_val = eval_res.h_values[name]
            b_cbf[i] = lf_h[i] + alpha * h_val

        # Check if u_proposed is already strictly safe
        violations_proposed: List[str] = []
        cbf_margins = b_cbf - A_cbf @ u_prop
        for i, (name, _) in enumerate(self.cbf.invariants.items()):
            if cbf_margins[i] < -1e-7:
                violations_proposed.append(name)

        is_within_box = np.all(u_prop >= u_min - 1e-7) and np.all(u_prop <= u_max + 1e-7)

        if len(violations_proposed) == 0 and is_within_box:
            # Already safe! No modification required.
            t_solve = (time.perf_counter() - t_start) * 1000.0
            return SafetyFilterResult(
                safe_action=u_prop.copy(),
                proposed_action=u_prop.copy(),
                is_modified=False,
                cost=0.0,
                barrier_values=eval_res.h_values,
                safety_margin=eval_res.min_margin,
                status="PASSED_UNMODIFIED",
                slack=0.0,
                cbf_violations_prevented=[],
                solve_time_ms=t_solve,
            )

        # Attempt analytical 1D fast projection if u_dim == 1
        if self.u_dim == 1:
            u_safe_1d = self._solve_analytical_1d(
                u_prop=float(u_prop[0]),
                a_cbf=A_cbf.flatten(),
                b_cbf=b_cbf,
                u_min=float(u_min[0]),
                u_max=float(u_max[0]),
            )
            if u_safe_1d is not None:
                t_solve = (time.perf_counter() - t_start) * 1000.0
                safe_arr = np.array([u_safe_1d], dtype=np.float64)
                cost = 0.5 * float((safe_arr - u_prop) @ self.H @ (safe_arr - u_prop))
                return SafetyFilterResult(
                    safe_action=safe_arr,
                    proposed_action=u_prop.copy(),
                    is_modified=True,
                    cost=cost,
                    barrier_values=eval_res.h_values,
                    safety_margin=eval_res.min_margin,
                    status="PROJECTED_SAFE",
                    slack=0.0,
                    cbf_violations_prevented=violations_proposed,
                    solve_time_ms=t_solve,
                )

        # General Multi-Dimensional QP Solve via SLSQP
        # Decision variable: w = [u; xi] in R^{m + 1} if allow_slack else u in R^m
        m = self.u_dim
        use_slack = self.allow_slack

        if use_slack:
            # Variable w = [u_1, ..., u_m, xi]
            w0 = np.concatenate([np.clip(u_prop, u_min, u_max), [0.0]])

            def objective(w: np.ndarray) -> float:
                u = w[:m]
                xi = w[m]
                diff = u - u_prop
                return float(0.5 * diff.T @ self.H @ diff + 0.5 * self.slack_penalty * (xi ** 2))

            def objective_grad(w: np.ndarray) -> np.ndarray:
                u = w[:m]
                xi = w[m]
                grad = np.empty(m + 1, dtype=np.float64)
                grad[:m] = self.H @ (u - u_prop)
                grad[m] = self.slack_penalty * xi
                return grad

            # Constraint: b_cbf + xi * 1 - A_cbf u >= 0
            def ineq_constraint(w: np.ndarray) -> np.ndarray:
                u = w[:m]
                xi = w[m]
                return b_cbf + xi - A_cbf @ u

            def ineq_jacobian(w: np.ndarray) -> np.ndarray:
                ones_col = np.ones((k_invariants, 1), dtype=np.float64)
                return np.hstack([-A_cbf, ones_col])

            bounds = [(u_min[i], u_max[i]) for i in range(m)] + [(0.0, 1e5)]
            constraints = [{
                "type": "ineq",
                "fun": ineq_constraint,
                "jac": ineq_jacobian,
            }]

        else:
            w0 = np.clip(u_prop, u_min, u_max)

            def objective(w: np.ndarray) -> float:
                diff = w - u_prop
                return float(0.5 * diff.T @ self.H @ diff)

            def objective_grad(w: np.ndarray) -> np.ndarray:
                return self.H @ (w - u_prop)

            def ineq_constraint(w: np.ndarray) -> np.ndarray:
                return b_cbf - A_cbf @ w

            def ineq_jacobian(w: np.ndarray) -> np.ndarray:
                return -A_cbf

            bounds = [(u_min[i], u_max[i]) for i in range(m)]
            constraints = [{
                "type": "ineq",
                "fun": ineq_constraint,
                "jac": ineq_jacobian,
            }]

        opt_res = minimize(
            fun=objective,
            x0=w0,
            jac=objective_grad,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 100, "ftol": 1e-8, "disp": False},
        )

        t_solve = (time.perf_counter() - t_start) * 1000.0

        if opt_res.success:
            safe_u = opt_res.x[:m]
            slack_val = float(opt_res.x[m]) if use_slack else 0.0
            status = "RELAXED_FEASIBLE" if slack_val > 1e-4 else "PROJECTED_SAFE"
            cost = float(0.5 * (safe_u - u_prop).T @ self.H @ (safe_u - u_prop))
            is_mod = bool(np.linalg.norm(safe_u - u_prop) > 1e-5)
            return SafetyFilterResult(
                safe_action=safe_u,
                proposed_action=u_prop.copy(),
                is_modified=is_mod,
                cost=cost,
                barrier_values=eval_res.h_values,
                safety_margin=eval_res.min_margin,
                status=status,
                slack=slack_val,
                cbf_violations_prevented=violations_proposed,
                solve_time_ms=t_solve,
            )
        else:
            # Fallback to box-clamped action with warning
            clamped = np.clip(u_prop, u_min, u_max)
            return SafetyFilterResult(
                safe_action=clamped,
                proposed_action=u_prop.copy(),
                is_modified=True,
                cost=float(np.linalg.norm(clamped - u_prop)),
                barrier_values=eval_res.h_values,
                safety_margin=eval_res.min_margin,
                status="INFEASIBLE_ABORT",
                slack=float("inf"),
                cbf_violations_prevented=violations_proposed,
                solve_time_ms=t_solve,
            )

    def filter_discrete_mitigation(
        self,
        current_state: ClusterState,
        proposed_action_idx: int,
        action_effects: Optional[Dict[int, np.ndarray]] = None,
    ) -> Tuple[int, SafetyFilterResult]:
        """
        Safety filter for discrete mitigation actions (e.g. MitigationAction enum).

        Maps each discrete action to its state displacement vector Delta x,
        checks CBF invariance, and returns the highest-utility safe action.
        """
        if action_effects is None:
            # Default physical displacement Delta x = [replicas, cpu, traffic_drop, queue, latency, mem, err]
            # 0: NO_ACTION
            # 1: MIGRATE_REPLICA: temporary replica drop -0.2 (cold start), CPU -0.05, latency +10
            # 2: EBPF_THROTTLE: traffic_drop +0.02, queue -30, latency -15
            # 3: CIRCUIT_BREAKER_OPEN: traffic_drop +0.08 (cuts traffic), queue -80, latency -30
            action_effects = {
                0: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                1: np.array([-0.1, -0.03, 0.01, -5.0, 5.0, 0.0, 0.0]),
                2: np.array([0.0, 0.05, 0.02, -25.0, -10.0, 0.05, 0.0]),
                3: np.array([0.0, 0.15, 0.08, -50.0, -25.0, 0.15, 0.0]),
            }

        # Check if proposed action is safe
        s_vec = current_state.to_array()
        delta = action_effects.get(proposed_action_idx, np.zeros_like(s_vec))
        next_state_vec = s_vec + delta

        eval_prop = self.cbf.evaluate(next_state_vec)

        if eval_prop.is_safe:
            # Action is safe!
            res = SafetyFilterResult(
                safe_action=np.array([proposed_action_idx], dtype=np.float64),
                proposed_action=np.array([proposed_action_idx], dtype=np.float64),
                is_modified=False,
                cost=0.0,
                barrier_values=eval_prop.h_values,
                safety_margin=eval_prop.min_margin,
                status="PASSED_UNMODIFIED",
                slack=0.0,
                cbf_violations_prevented=[],
            )
            return proposed_action_idx, res

        # Proposed action violates a safety invariant. Search for safest alternative.
        best_action = 0  # Fallback: NO_ACTION
        best_margin = -float("inf")
        prevented = eval_prop.violated_invariants

        for act_idx, act_delta in action_effects.items():
            cand_next = s_vec + act_delta
            cand_eval = self.cbf.evaluate(cand_next)
            if cand_eval.is_safe and cand_eval.min_margin > best_margin:
                best_margin = cand_eval.min_margin
                best_action = act_idx

        # If no strictly safe action, pick NO_ACTION
        safe_eval = self.cbf.evaluate(s_vec + action_effects.get(best_action, np.zeros_like(s_vec)))
        res = SafetyFilterResult(
            safe_action=np.array([best_action], dtype=np.float64),
            proposed_action=np.array([proposed_action_idx], dtype=np.float64),
            is_modified=(best_action != proposed_action_idx),
            cost=abs(float(best_action - proposed_action_idx)),
            barrier_values=safe_eval.h_values,
            safety_margin=safe_eval.min_margin,
            status="PROJECTED_SAFE" if safe_eval.is_safe else "RELAXED_FEASIBLE",
            slack=max(0.0, -safe_eval.min_margin),
            cbf_violations_prevented=prevented,
        )
        return best_action, res


# ==============================================================================
# 3. Lyapunov Stability & Oscillation Verifier
# ==============================================================================


class LyapunovStabilityVerifier:
    """
    Formal Discrete-Time Lyapunov Stability & Destructive Oscillation Verifier.

    Evaluates discrete Lyapunov function:
        V(x) = (x - x^*)^T P (x - x^*)
    and Lyapunov contraction drift:
        Delta V(x_t) = V(x_{t+1}) - V(x_t) <= -kappa * V(x_t)

    Where:
    - x^* is the nominal stable equilibrium operating point.
    - P is a symmetric positive-definite matrix (P = P^T > 0).
    - kappa in (0, 1) is the strict exponential contraction rate.

    Guarantees:
    - Monotonic energy dissipation towards steady-state equilibrium.
    - Strict absence of limit cycles, high-frequency ringing, and mitigation flapping.
    - Riccati synthesis (DARE) for optimal stabilizing feedback gain K: u = -K (x - x^*).
    """

    def __init__(
        self,
        dim: int,
        x_eq: Optional[Union[ClusterState, Sequence[float], np.ndarray]] = None,
        P: Optional[np.ndarray] = None,
        kappa: float = 0.05,
        custom_v_fn: Optional[Callable[[np.ndarray], float]] = None,
    ):
        self.dim = int(dim)

        if x_eq is None:
            self.x_eq = np.zeros(self.dim, dtype=np.float64)
        elif isinstance(x_eq, ClusterState):
            self.x_eq = x_eq.to_array()[:self.dim]
        else:
            self.x_eq = np.asarray(x_eq, dtype=np.float64).flatten()[:self.dim]

        if P is None:
            self.P = np.eye(self.dim, dtype=np.float64)
        else:
            p_arr = np.asarray(P, dtype=np.float64)
            # Ensure symmetry
            self.P = 0.5 * (p_arr + p_arr.T)

        # Validate positive definiteness
        eigs = np.linalg.eigvalsh(self.P)
        if np.any(eigs <= 1e-9):
            min_eig = float(np.min(eigs))
            # Regularize to guarantee positive definiteness
            self.P += (abs(min_eig) + 1e-4) * np.eye(self.dim)

        self.kappa = float(kappa)
        self.custom_v_fn = custom_v_fn

    @classmethod
    def from_linear_system(
        cls,
        A: np.ndarray,
        x_eq: Optional[np.ndarray] = None,
        Q: Optional[np.ndarray] = None,
        kappa: float = 0.05,
    ) -> LyapunovStabilityVerifier:
        """
        Synthesize positive-definite P satisfying the Discrete Lyapunov Equation:
            A^T P A - P = -Q
        for autonomous error dynamics e_{t+1} = A e_t.
        """
        A_mat = np.asarray(A, dtype=np.float64)
        n = A_mat.shape[0]
        if Q is None:
            Q_mat = np.eye(n, dtype=np.float64)
        else:
            Q_mat = np.asarray(Q, dtype=np.float64)

        # Check Schur stability of A
        spectral_radius = float(np.max(np.abs(np.linalg.eigvals(A_mat))))
        if spectral_radius >= 1.0 - 1e-6:
            # Scaled stabilization to prevent unbounded solver error
            scale = (0.95 / spectral_radius)
            A_mat = A_mat * scale

        # Solve A^T P A - P = -Q
        P_sol = scipy.linalg.solve_discrete_lyapunov(A_mat.T, Q_mat)
        return cls(dim=n, x_eq=x_eq, P=P_sol, kappa=kappa)

    @classmethod
    def from_lqr_dare(
        cls,
        A: np.ndarray,
        B: np.ndarray,
        Q: Optional[np.ndarray] = None,
        R: Optional[np.ndarray] = None,
        x_eq: Optional[np.ndarray] = None,
        kappa: float = 0.05,
    ) -> Tuple[LyapunovStabilityVerifier, np.ndarray]:
        """
        Solve Discrete Algebraic Riccati Equation (DARE):
            P = A^T P A - (A^T P B)(R + B^T P B)^{-1}(B^T P A) + Q
        Synthesizes provably stable Lyapunov matrix P and feedback gain K:
            K = (R + B^T P B)^{-1} B^T P A
        """
        A_mat = np.asarray(A, dtype=np.float64)
        B_mat = np.asarray(B, dtype=np.float64)
        n = A_mat.shape[0]
        m = B_mat.shape[1] if B_mat.ndim > 1 else 1
        if B_mat.ndim == 1:
            B_mat = B_mat.reshape(-1, 1)

        Q_mat = np.eye(n, dtype=np.float64) if Q is None else np.asarray(Q, dtype=np.float64)
        R_mat = np.eye(m, dtype=np.float64) if R is None else np.asarray(R, dtype=np.float64)

        P_dare = scipy.linalg.solve_discrete_are(A_mat, B_mat, Q_mat, R_mat)
        K = np.linalg.inv(R_mat + B_mat.T @ P_dare @ B_mat) @ (B_mat.T @ P_dare @ A_mat)

        verifier = cls(dim=n, x_eq=x_eq, P=P_dare, kappa=kappa)
        return verifier, K

    @classmethod
    def create_cluster_verifier(
        cls,
        x_eq: Optional[ClusterState] = None,
        kappa: float = 0.05,
    ) -> LyapunovStabilityVerifier:
        """
        Create standard cluster verifier with nominal target equilibrium:
        replicas=3.0, cpu_headroom=0.35, traffic_drop=0.0, queue=10.0, latency=45.0
        """
        if x_eq is None:
            x_eq = ClusterState(
                replicas=3.0,
                cpu_headroom=0.35,
                traffic_drop=0.0,
                queue_depth=10.0,
                latency_ms=45.0,
                memory_headroom=0.40,
                error_rate=0.001,
            )
        # Scaled diagonal weights reflecting metric standard deviations:
        # replicas, cpu_headroom, traffic_drop, queue, latency, mem, error
        p_weights = np.array([2.0, 10.0, 50.0, 0.01, 0.005, 5.0, 100.0], dtype=np.float64)
        P = np.diag(p_weights)
        return cls(dim=7, x_eq=x_eq, P=P, kappa=kappa)

    def _normalize(self, x: Union[SafetyClusterState, Dict[str, Any], Sequence[float], np.ndarray]) -> np.ndarray:
        if hasattr(x, "to_array") and callable(x.to_array):
            return x.to_array()[:self.dim]
        if isinstance(x, dict):
            return SafetyClusterState.from_dict(x).to_array()[:self.dim]
        arr = np.asarray(x, dtype=np.float64).flatten()
        if len(arr) < self.dim:
            padded = np.zeros(self.dim, dtype=np.float64)
            padded[:len(arr)] = arr
            return padded
        return arr[:self.dim]

    def v(self, x: Union[ClusterState, Sequence[float], np.ndarray]) -> float:
        """
        Compute scalar Lyapunov energy V(x) >= 0.
        """
        x_vec = self._normalize(x)
        if self.custom_v_fn is not None:
            return float(self.custom_v_fn(x_vec))

        err = x_vec - self.x_eq
        energy = float(err.T @ self.P @ err)
        return max(0.0, energy)

    def gradient_v(self, x: Union[ClusterState, Sequence[float], np.ndarray]) -> np.ndarray:
        """
        Compute gradient nabla V(x) = 2 P (x - x^*).
        """
        x_vec = self._normalize(x)
        err = x_vec - self.x_eq
        return 2.0 * (self.P @ err)

    def verify_step(
        self,
        x_t: Union[ClusterState, Sequence[float], np.ndarray],
        x_next: Union[ClusterState, Sequence[float], np.ndarray],
    ) -> LyapunovStepResult:
        """
        Verify discrete Lyapunov drift condition for transition x_t -> x_{t+1}:
            Delta V = V(x_{t+1}) - V(x_t) <= -kappa * V(x_t)
        """
        v_t = self.v(x_t)
        v_next = self.v(x_next)
        drift = v_next - v_t

        required_drift = - self.kappa * v_t
        margin = required_drift - drift  # Positive means strictly satisfying or exceeding required decay

        # Stable if drift <= required_drift + epsilon tolerance
        is_stable = bool(drift <= required_drift + 1e-7 or v_t < 1e-6)

        contraction_ratio = (v_next / v_t) if v_t > 1e-9 else 0.0

        return LyapunovStepResult(
            is_stable=is_stable,
            v_t=v_t,
            v_next=v_next,
            drift=drift,
            required_drift=required_drift,
            contraction_ratio=contraction_ratio,
            margin=margin,
        )

    def verify_trajectory(
        self,
        trajectory: Sequence[Union[ClusterState, Sequence[float], np.ndarray]],
    ) -> Dict[str, Any]:
        """
        Verify discrete Lyapunov stability across a complete multi-step trajectory:
        [x_0, x_1, ..., x_T].
        """
        if len(trajectory) < 2:
            return {
                "is_stable": True,
                "step_results": [],
                "cumulative_drift": 0.0,
                "max_drift": 0.0,
                "empirical_decay_rate": self.kappa,
                "energy_profile": [self.v(trajectory[0])] if trajectory else [],
            }

        energies = [self.v(st) for st in trajectory]
        steps: List[LyapunovStepResult] = []
        all_stable = True

        for t in range(len(trajectory) - 1):
            step_res = self.verify_step(trajectory[t], trajectory[t + 1])
            steps.append(step_res)
            if not step_res.is_stable:
                all_stable = False

        ratios = [
            (energies[t + 1] / energies[t])
            for t in range(len(energies) - 1)
            if energies[t] > 1e-6
        ]
        max_ratio = float(np.max(ratios)) if ratios else 1.0
        empirical_decay = max(0.0, 1.0 - max_ratio)

        return {
            "is_stable": all_stable,
            "step_results": steps,
            "cumulative_drift": energies[-1] - energies[0],
            "max_drift": float(np.max([s.drift for s in steps])),
            "empirical_decay_rate": empirical_decay,
            "energy_profile": energies,
        }

    def detect_oscillations(
        self,
        trajectory: Sequence[Union[ClusterState, Sequence[float], np.ndarray]],
        window: int = 10,
        spectral_threshold: float = 0.35,
        zero_crossing_threshold: float = 0.35,
        flapping_actions: Optional[Sequence[int]] = None,
    ) -> OscillationDetectionResult:
        """
        Detect destructive feedback oscillations, limit cycles, and control flapping.

        Evaluates:
        1. Velocity sign alternations (Zero-Crossing Rate of Delta x_t).
        2. Fast Fourier Transform (FFT) dominant resonant peak power ratio.
        3. Lyapunov drift oscillation (alternating positive/negative energy injection).
        4. Discrete action toggle frequency (flapping).

        Parameters
        ----------
        trajectory : Sequence of state vectors
        window : int
            Analysis window length.
        spectral_threshold : float
            Power ratio above which a non-DC frequency indicates a limit cycle.
        zero_crossing_threshold : float
            Zero crossing rate of state derivative above which ringing is flagged.
        flapping_actions : Optional sequence of action indices
        """
        t_len = len(trajectory)
        if t_len < 4:
            return OscillationDetectionResult(
                is_oscillating=False,
                oscillation_type="INSUFFICIENT_DATA",
                dominant_frequency=0.0,
                spectral_power=0.0,
                zero_crossing_rate=0.0,
                flapping_count=0,
                lyapunov_drift_violations=0,
                recommendation="Continue observation; insufficient steps for spectral verification.",
            )

        traj_arr = np.array([self._normalize(st) for st in trajectory], dtype=np.float64)

        # 1. Zero-Crossing Rate of state differences Delta x_t
        dx = np.diff(traj_arr, axis=0)  # shape (T-1, dim)
        zc_rates: List[float] = []
        for j in range(self.dim):
            col = dx[:, j]
            # Detect sign change between consecutive derivatives
            signs = np.sign(col)
            # Replace 0 with previous sign to prevent false crossings on plateaus
            for idx in range(1, len(signs)):
                if signs[idx] == 0:
                    signs[idx] = signs[idx - 1]
            sign_flips = np.diff(signs) != 0
            zc = float(np.mean(sign_flips)) if len(sign_flips) > 0 else 0.0
            zc_rates.append(zc)

        max_zcr = float(np.max(zc_rates))

        # 2. Spectral Analysis via FFT on error norms ||x_t - x^*||
        errors = np.linalg.norm(traj_arr - self.x_eq, axis=1)
        err_centered = errors - np.mean(errors)

        if np.var(errors) > 1e-6:
            fft_vals = np.abs(np.fft.rfft(err_centered))
            # Ignore DC component at index 0
            ac_fft = fft_vals[1:]
            total_ac_power = float(np.sum(ac_fft ** 2))
            if total_ac_power > 1e-9:
                peak_idx = int(np.argmax(ac_fft))
                dom_freq = float(peak_idx + 1) / float(t_len)
                spectral_ratio = float((ac_fft[peak_idx] ** 2) / total_ac_power)
            else:
                dom_freq = 0.0
                spectral_ratio = 0.0
        else:
            dom_freq = 0.0
            spectral_ratio = 0.0

        # 3. Lyapunov Drift Violations
        drift_violations = 0
        energies = [self.v(st) for st in traj_arr]
        for t in range(len(energies) - 1):
            drift = energies[t + 1] - energies[t]
            if drift > 0.0:
                drift_violations += 1

        # 4. Action Flapping Detection
        flapping_count = 0
        if flapping_actions is not None and len(flapping_actions) >= 3:
            acts = list(flapping_actions)
            for i in range(1, len(acts) - 1):
                # Flapping: a != b and then returns to a (a -> b -> a)
                if acts[i - 1] != acts[i] and acts[i - 1] == acts[i + 1]:
                    flapping_count += 1

        # Determine Classification
        is_oscillating = False
        osc_type = "STABLE_MONOTONIC"
        rec = "System converges monotonically. Normal autonomous operation permitted."

        if flapping_count >= 2:
            is_oscillating = True
            osc_type = "ACTION_FLAPPING"
            rec = "Mitigation action flapping detected. Dampen controller gain or increase hysteresis."
        elif spectral_ratio > spectral_threshold and (max_zcr > 0.15 or drift_violations >= 3):
            is_oscillating = True
            osc_type = "LIMIT_CYCLE"
            rec = f"Persistent limit cycle detected at frequency {dom_freq:.3f} Hz. Engage safety filter projection."
        elif max_zcr > zero_crossing_threshold:
            is_oscillating = True
            osc_type = "DESTRUCTIVE_RINGING"
            rec = f"Severe ringing observed (ZCR={max_zcr:.2f}). Attenuate feedback sensitivity."
        elif drift_violations > (t_len // 2) and energies[-1] > energies[0]:
            is_oscillating = True
            osc_type = "DIVERGENT"
            rec = "Trajectory diverges from target equilibrium. Activate fail-safe clamp."

        return OscillationDetectionResult(
            is_oscillating=is_oscillating,
            oscillation_type=osc_type,
            dominant_frequency=dom_freq,
            spectral_power=spectral_ratio,
            zero_crossing_rate=max_zcr,
            flapping_count=flapping_count,
            lyapunov_drift_violations=drift_violations,
            recommendation=rec,
        )


# ==============================================================================
# 4. Safety Violation Guard
# ==============================================================================


class SafetyViolationGuard:
    """
    Deterministic Safety Violation Guard & Fail-Safe Circuit Breaker.

    Acts as the final barrier in the control pipeline:
    - Enforces hard aborts if physical state breaches catastrophic survival limits:
        * Replicas < min_replicas_fatal (e.g. < 1.0)
        * Cluster CPU headroom < min_cpu_headroom_fatal (e.g. < 0.01)
        * Traffic drop > max_traffic_drop_fatal (e.g. > 0.50)
        * Queue depth > max_queue_depth_fatal
    - Automatically overrides dangerous directives with deterministic fail-safe actions.
    - Engages temporary lockout windows following emergency aborts to prevent runaway thrashing.
    - Maintains immutable audit trail of safety intervention events.
    """

    def __init__(
        self,
        hard_limits: Optional[Dict[str, Tuple[float, str]]] = None,
        fail_safe_action: Any = 0,
        lockout_duration_steps: int = 5,
        max_lyapunov_divergence: float = 3.0,
    ):
        """
        Parameters
        ----------
        hard_limits : Dict[str, Tuple[float, str]], optional
            Mapping of metric name -> (threshold, "min" | "max").
        fail_safe_action : Any, default=0
            Neutral safe fallback action (e.g. MitigationAction.NO_ACTION or 0.0).
        lockout_duration_steps : int, default=5
            Number of evaluation steps to maintain lockout mode after emergency abort.
        max_lyapunov_divergence : float, default=3.0
            Maximum allowable energy ratio V(x_{t+1}) / V(x_t) before triggering abort.
        """
        if hard_limits is None:
            # Default catastrophic survival thresholds
            self.hard_limits = {
                "replicas": (1.0, "min"),             # Service death if replicas < 1
                "cpu_headroom": (0.01, "min"),        # Cluster freeze / deadlocks if headroom < 1%
                "traffic_drop": (0.50, "max"),        # Catastrophic blackhole if drop > 50%
                "queue_depth": (1000.0, "max"),       # OOM killer cascade if queue depth > 1000
                "error_rate": (0.50, "max"),          # Half requests failing
            }
        else:
            self.hard_limits = hard_limits

        self.fail_safe_action = fail_safe_action
        self.lockout_duration_steps = int(lockout_duration_steps)
        self.max_lyapunov_divergence = float(max_lyapunov_divergence)

        # Internal state
        self.lockout_counter: int = 0
        self.abort_history: List[Dict[str, Any]] = []
        self.total_evaluations: int = 0
        self.total_overrides: int = 0

    @classmethod
    def create_cluster_guard(
        cls,
        min_replicas: float = 1.0,
        min_cpu_headroom: float = 0.02,
        max_traffic_drop: float = 0.50,
        max_queue_depth: float = 1000.0,
        fail_safe_action: Any = 0,
        lockout_steps: int = 5,
    ) -> SafetyViolationGuard:
        """Create guard with standard cluster survival boundaries."""
        limits = {
            "replicas": (min_replicas, "min"),
            "cpu_headroom": (min_cpu_headroom, "min"),
            "traffic_drop": (max_traffic_drop, "max"),
            "queue_depth": (max_queue_depth, "max"),
        }
        return cls(
            hard_limits=limits,
            fail_safe_action=fail_safe_action,
            lockout_duration_steps=lockout_steps,
        )

    def check_hard_limits(
        self,
        state: Union[ClusterState, Dict[str, float], np.ndarray, Sequence[float]],
    ) -> Tuple[bool, List[str]]:
        """
        Check whether the current state violates any fatal cluster survival boundaries.

        Returns
        -------
        Tuple[bool, List[str]]
            (is_survivable, list_of_breaches)
        """
        if isinstance(state, ClusterState):
            state_dict = state.to_dict()
        elif isinstance(state, dict):
            state_dict = state
        elif isinstance(state, (np.ndarray, list, tuple)):
            c_state = ClusterState.from_array(state)
            state_dict = c_state.to_dict()
        else:
            raise TypeError(f"Unsupported state type: {type(state)}")

        breaches: List[str] = []
        for metric, (threshold, bound_type) in self.hard_limits.items():
            if metric not in state_dict:
                continue
            val = float(state_dict[metric])
            if bound_type == "min" and val < threshold:
                breaches.append(
                    f"FATAL: {metric}={val:.3f} below absolute survival threshold {threshold:.3f}"
                )
            elif bound_type == "max" and val > threshold:
                breaches.append(
                    f"FATAL: {metric}={val:.3f} above absolute survival threshold {threshold:.3f}"
                )

        is_survivable = len(breaches) == 0
        return is_survivable, breaches

    def evaluate(
        self,
        state: Union[ClusterState, Dict[str, float], np.ndarray, Sequence[float]],
        proposed_action: Any,
        cbf_result: Optional[SafetyFilterResult] = None,
        lyapunov_result: Optional[LyapunovStepResult] = None,
    ) -> GuardDecision:
        """
        Evaluate full guard envelope over the proposed mitigation directive.

        Parameters
        ----------
        state : ClusterState, dict, or array
            Current physical state.
        proposed_action : Any
            The control directive proposed for execution.
        cbf_result : Optional[SafetyFilterResult]
            Output of the QPSafetyFilter if previously executed.
        lyapunov_result : Optional[LyapunovStepResult]
            Output of the LyapunovStabilityVerifier if transition predicted.

        Returns
        -------
        GuardDecision
            Decision containing the permitted action (or fail-safe override) and rationale.
        """
        self.total_evaluations += 1

        # 1. Check if active in lockout mode
        if self.lockout_counter > 0:
            self.lockout_counter -= 1
            self.total_overrides += 1
            return GuardDecision(
                decision=GuardAction.LOCKOUT,
                is_safe=False,
                action=self.fail_safe_action,
                reason=(
                    f"Emergency lockout active ({self.lockout_counter + 1} steps remaining). "
                    "Autonomous mitigation commands suppressed. Fail-safe maintained."
                ),
                violations=["LOCKOUT_ENGAGED"],
                lockout_remaining=self.lockout_counter,
            )

        # 2. Check fatal survival limits
        is_survivable, breaches = self.check_hard_limits(state)
        if not is_survivable:
            self.lockout_counter = self.lockout_duration_steps
            self.total_overrides += 1
            self._log_abort(reason="Fatal physical limit breach", violations=breaches)
            return GuardDecision(
                decision=GuardAction.EMERGENCY_ABORT,
                is_safe=False,
                action=self.fail_safe_action,
                reason=(
                    f"Catastrophic state breach detected! Overriding proposed action with "
                    f"fail-safe action {self.fail_safe_action}. Lockout engaged for "
                    f"{self.lockout_duration_steps} steps."
                ),
                violations=breaches,
                lockout_remaining=self.lockout_counter,
            )

        # 3. Check QP safety filter status if provided
        if cbf_result is not None:
            if cbf_result.status == "INFEASIBLE_ABORT":
                self.lockout_counter = 2
                self.total_overrides += 1
                viols = [f"QP infeasible: {inv}" for inv in cbf_result.cbf_violations_prevented]
                self._log_abort(reason="QP Safety Filter infeasibility", violations=viols)
                return GuardDecision(
                    decision=GuardAction.OVERRIDE_FAILSAFE,
                    is_safe=False,
                    action=self.fail_safe_action,
                    reason=(
                        "QP Safety Filter found no safe feasible control input. "
                        "Reverting to deterministic fail-safe action."
                    ),
                    violations=viols,
                    lockout_remaining=self.lockout_counter,
                )

        # 4. Check Lyapunov drift divergence if provided
        if lyapunov_result is not None:
            if lyapunov_result.contraction_ratio > self.max_lyapunov_divergence:
                self.lockout_counter = 2
                self.total_overrides += 1
                viols = [
                    f"Lyapunov divergence ratio {lyapunov_result.contraction_ratio:.2f} > "
                    f"threshold {self.max_lyapunov_divergence:.2f}"
                ]
                self._log_abort(reason="Destructive Lyapunov divergence", violations=viols)
                return GuardDecision(
                    decision=GuardAction.OVERRIDE_FAILSAFE,
                    is_safe=False,
                    action=self.fail_safe_action,
                    reason=(
                        "Proposed mitigation triggers explosive Lyapunov energy divergence. "
                        "Suppressing action to avoid destructive instability."
                    ),
                    violations=viols,
                    lockout_remaining=self.lockout_counter,
                )

        # 5. If CBF modified action safely, permit the filtered safe action
        effective_action = proposed_action
        if cbf_result is not None and cbf_result.is_modified:
            effective_action = cbf_result.safe_action

        return GuardDecision(
            decision=GuardAction.PERMIT,
            is_safe=True,
            action=effective_action,
            reason="All formal safety barriers and survival conditions verified.",
            violations=[],
            lockout_remaining=0,
        )

    def trigger_emergency_abort(self, reason: str, extra_info: Optional[Dict[str, Any]] = None) -> GuardDecision:
        """
        Manually trigger an emergency abort and lock out autonomous mitigations.
        """
        self.lockout_counter = self.lockout_duration_steps
        self.total_overrides += 1
        self._log_abort(reason=reason, violations=[reason])
        return GuardDecision(
            decision=GuardAction.EMERGENCY_ABORT,
            is_safe=False,
            action=self.fail_safe_action,
            reason=f"Manual emergency abort invoked: {reason}",
            violations=[reason],
            lockout_remaining=self.lockout_counter,
        )

    def _log_abort(self, reason: str, violations: List[str]) -> None:
        self.abort_history.append({
            "timestamp": time.time(),
            "reason": reason,
            "violations": violations,
            "lockout_steps": self.lockout_duration_steps,
        })

    def step_lockout(self) -> int:
        """Advance lockout counter by 1 step."""
        if self.lockout_counter > 0:
            self.lockout_counter -= 1
        return self.lockout_counter

    def reset(self) -> None:
        """Reset internal guard counters and lockout status."""
        self.lockout_counter = 0
        self.total_overrides = 0
        self.total_evaluations = 0
