"""
Unit tests for Frontier 4: Provable Safety & Formal Verification
(Control Barrier Functions & Lyapunov Stability).

Covers:
1. `ControlBarrierFunction` & `SafetyInvariant`: Invariant evaluation, gradients, Lie derivatives, D-CBF.
2. `QPSafetyFilter`: Minimum-intervention QP projection, slack relaxation, discrete action mapping.
3. `LyapunovStabilityVerifier`: Discrete Lyapunov drift Delta V <= -kappa V, DARE synthesis, oscillation detection.
4. `SafetyViolationGuard`: Hard survival aborts, emergency lockouts, fail-safe overrides.
5. End-to-End Closed-Loop Safety Verification.
"""

import math
import numpy as np
import pytest
import scipy.linalg

from deeplog.formal_safety import (
    ControlBarrierFunction,
    QPSafetyFilter,
    LyapunovStabilityVerifier,
    SafetyViolationGuard,
    SafetyInvariant,
    CBFEvaluationResult,
    SafetyFilterResult,
    LyapunovStepResult,
    OscillationDetectionResult,
    GuardAction,
    GuardDecision,
    SafetyClusterState,
    ClusterState,
)


# ==============================================================================
# 1. SafetyInvariant & ControlBarrierFunction Unit Tests
# ==============================================================================


class TestControlBarrierFunction:
    """Tests for Control Barrier Functions and safety invariant evaluations."""

    @pytest.fixture
    def cluster_cbf(self):
        return ControlBarrierFunction.create_cluster_safety_cbf(
            min_replicas=2.0,
            min_cpu_headroom=0.15,
            max_traffic_drop=0.05,
            max_queue_depth=500.0,
            max_latency_ms=250.0,
            default_alpha=1.0,
        )

    def test_safety_invariant_scalar_evaluation(self):
        """Verify individual SafetyInvariant evaluation and bounds."""
        inv_rep = SafetyInvariant(
            name="min_replicas",
            func=lambda x: float(x[0] - 2.0),
            metric_index=0,
            is_upper_bound=False,
            threshold=2.0,
        )

        assert inv_rep.evaluate(np.array([3.0, 0.5, 0.0])) == 1.0
        assert inv_rep.evaluate(np.array([2.0, 0.5, 0.0])) == 0.0
        assert inv_rep.evaluate(np.array([1.5, 0.5, 0.0])) == -0.5

        # Analytical gradient vs finite difference
        grad = inv_rep.gradient(np.array([3.0, 0.5, 0.0]))
        assert np.allclose(grad, [1.0, 0.0, 0.0])

    def test_cbf_cluster_evaluation_safe_state(self, cluster_cbf):
        """Verify evaluation on a nominal safe cluster state."""
        state = SafetyClusterState(
            replicas=4.0,
            cpu_headroom=0.30,
            traffic_drop=0.01,
            queue_depth=50.0,
            latency_ms=60.0,
        )
        res = cluster_cbf.evaluate(state)

        assert res.is_safe is True
        assert len(res.violated_invariants) == 0
        assert res.min_margin > 0.0
        assert res.h_values["min_replicas"] == 2.0  # 4 - 2
        assert np.isclose(res.h_values["min_cpu_headroom"], 0.15)  # 0.30 - 0.15
        assert np.isclose(res.h_values["max_traffic_drop"], 0.04)  # 0.05 - 0.01

    def test_cbf_cluster_evaluation_unsafe_state(self, cluster_cbf):
        """Verify evaluation detects invariant violations in unsafe states."""
        unsafe_state = SafetyClusterState(
            replicas=1.0,          # Violates min_replicas (1 < 2)
            cpu_headroom=0.05,     # Violates min_cpu_headroom (0.05 < 0.15)
            traffic_drop=0.12,     # Violates max_traffic_drop (0.12 > 0.05)
            queue_depth=600.0,     # Violates max_queue_depth (600 > 500)
            latency_ms=300.0,      # Violates max_latency_ms (300 > 250)
        )
        res = cluster_cbf.evaluate(unsafe_state)

        assert res.is_safe is False
        assert len(res.violated_invariants) == 5
        assert "min_replicas" in res.violated_invariants
        assert "min_cpu_headroom" in res.violated_invariants
        assert "max_traffic_drop" in res.violated_invariants
        assert "max_queue_depth" in res.violated_invariants
        assert "max_latency_ms" in res.violated_invariants
        assert res.min_margin < 0.0

    def test_active_invariants_near_boundary(self, cluster_cbf):
        """Verify active invariants are flagged when within boundary tolerance."""
        state = SafetyClusterState(
            replicas=2.02,         # Within 0.05 of boundary 2.0
            cpu_headroom=0.40,
            traffic_drop=0.00,
            queue_depth=10.0,
            latency_ms=50.0,
        )
        res = cluster_cbf.evaluate(state, boundary_tolerance=0.05)
        assert res.is_safe is True
        assert "min_replicas" in res.active_invariants

    def test_jacobian_and_lie_derivatives(self, cluster_cbf):
        """Verify CBF Jacobian and Lie derivatives L_f h, L_g h."""
        state = SafetyClusterState(replicas=3.0, cpu_headroom=0.25, traffic_drop=0.02)
        jac = cluster_cbf.compute_jacobian(state)

        assert jac.shape == (5, 7)
        # min_replicas gradient is e_0
        assert jac[0, 0] == 1.0
        # min_cpu_headroom gradient is e_1
        assert jac[1, 1] == 1.0
        # max_traffic_drop gradient is -e_2
        assert jac[2, 2] == -1.0

        # Vector field f(x) and control matrix g(x)
        f_x = np.zeros(7)
        f_x[1] = -0.01  # Autonomous CPU drift -1%/sec
        g_x = np.eye(7)

        lf_h, lg_h = cluster_cbf.compute_lie_derivatives(state, f_x=f_x, g_x=g_x)
        assert lf_h.shape == (5,)
        assert lg_h.shape == (5, 7)
        assert np.isclose(lf_h[1], -0.01)
        assert lg_h[0, 0] == 1.0

    def test_discrete_time_barrier_condition(self, cluster_cbf):
        """Verify discrete-time barrier condition h(x_{t+1}) >= (1 - gamma) h(x_t)."""
        x_t = SafetyClusterState(replicas=4.0, cpu_headroom=0.30, traffic_drop=0.01)
        # Safe transition: replica goes from 4 -> 3.5 (still >= 2, allowable approach)
        x_safe_next = SafetyClusterState(replicas=3.5, cpu_headroom=0.28, traffic_drop=0.01)
        res_safe = cluster_cbf.check_discrete_barrier(x_t, x_safe_next, gamma=0.5)
        assert all(res_safe.values())

        # Unsafe transition: sudden plunge to replicas=1.0
        x_unsafe_next = SafetyClusterState(replicas=1.0, cpu_headroom=0.30, traffic_drop=0.01)
        res_unsafe = cluster_cbf.check_discrete_barrier(x_t, x_unsafe_next, gamma=0.5)
        assert res_unsafe["min_replicas"] is False

    def test_custom_nonlinear_safety_invariant(self):
        """Verify support for custom user-defined nonlinear invariant functions."""
        cbf = ControlBarrierFunction()

        # Invariant: x[0]^2 + x[1]^2 <= 25 (safe inside radius 5) => 25 - (x0^2 + x1^2) >= 0
        circle_inv = SafetyInvariant(
            name="safe_circle",
            func=lambda x: float(25.0 - (x[0]**2 + x[1]**2)),
            grad_func=lambda x: np.array([-2.0 * x[0], -2.0 * x[1]], dtype=np.float64),
            threshold=0.0,
        )
        cbf.add_invariant(circle_inv)

        assert cbf.is_safe(np.array([3.0, 3.0])) is True   # 9+9 = 18 <= 25
        assert cbf.is_safe(np.array([4.0, 4.0])) is False  # 16+16 = 32 > 25

        jac = cbf.compute_jacobian(np.array([3.0, 4.0]))
        assert np.allclose(jac[0], [-6.0, -8.0])


# ==============================================================================
# 2. QPSafetyFilter Unit Tests
# ==============================================================================


class TestQPSafetyFilter:
    """Tests for the Quadratic Program (QP) Safety Filter."""

    @pytest.fixture
    def cbf(self):
        return ControlBarrierFunction.create_cluster_safety_cbf(
            min_replicas=2.0,
            min_cpu_headroom=0.15,
            max_traffic_drop=0.05,
        )

    def test_qp_passes_safe_control_unmodified(self, cbf):
        """When proposed action satisfies safety invariants, it must pass untouched."""
        qp_filter = QPSafetyFilter(cbf=cbf, u_dim=1, u_min=-5.0, u_max=5.0)

        # State has 4 replicas (h_rep = 2.0). Safe proposed delta = +1.0
        state = SafetyClusterState(replicas=4.0, cpu_headroom=0.35, traffic_drop=0.01)
        g_x = np.zeros((7, 1))
        g_x[0, 0] = 1.0  # u controls rate of replica change dot{replicas} = u

        res = qp_filter.filter_action(state, u_proposed=1.0, g_x=g_x)

        assert res.is_modified is False
        assert res.status == "PASSED_UNMODIFIED"
        assert np.isclose(res.safe_action[0], 1.0)
        assert res.cost == 0.0

    def test_qp_projects_dangerous_scaling_to_boundary(self, cbf):
        """Proposed scale-down u=-5 would crash replicas to 0. Filter must project onto boundary."""
        qp_filter = QPSafetyFilter(cbf=cbf, u_dim=1, u_min=-10.0, u_max=10.0)

        # State has 3 replicas (h_rep = 3 - 2 = 1.0).
        # Continuous barrier condition: L_g h * u + alpha * h >= 0
        # 1.0 * u + 1.0 * (3.0 - 2.0) >= 0 <=> u >= -1.0.
        state = SafetyClusterState(replicas=3.0, cpu_headroom=0.35, traffic_drop=0.01)
        g_x = np.zeros((7, 1))
        g_x[0, 0] = 1.0

        # Controller proposes dangerous u = -4.0 (terminating 4 replicas)
        res = qp_filter.filter_action(state, u_proposed=-4.0, g_x=g_x)

        assert res.is_modified is True
        assert res.status == "PROJECTED_SAFE"
        # Projected safe action must be >= -1.0
        assert res.safe_action[0] >= -1.0 - 1e-4
        assert np.isclose(res.safe_action[0], -1.0, atol=1e-3)
        assert res.cost > 0.0
        assert "min_replicas" in res.cbf_violations_prevented

    def test_qp_multidimensional_coupling(self, cbf):
        """Test multi-dimensional control with coupled physical dynamics."""
        # u = [delta_replicas, delta_throttling] in R^2
        qp_filter = QPSafetyFilter(
            cbf=cbf,
            u_dim=2,
            u_min=np.array([-5.0, 0.0]),
            u_max=np.array([5.0, 1.0]),
        )
        state = SafetyClusterState(
            replicas=2.5,          # margin: 0.5
            cpu_headroom=0.20,     # margin: 0.05
            traffic_drop=0.03,     # margin: 0.02
        )
        # g_x matrix mapping u to state derivatives:
        # dot{replicas} = u_0
        # dot{cpu_headroom} = 0.1 * u_0 + 0.2 * u_1
        # dot{traffic_drop} = 0.05 * u_1
        g_x = np.zeros((7, 2))
        g_x[0, 0] = 1.0
        g_x[1, 0] = 0.1
        g_x[1, 1] = 0.2
        g_x[2, 1] = 0.05

        # Propose extreme throttle u_1 = 0.9 (which would induce 0.05 * 0.9 = 0.045 traffic drop,
        # breaching 0.03 + 0.045 = 0.075 > 0.05)
        u_prop = np.array([0.0, 0.9])
        res = qp_filter.filter_action(state, u_proposed=u_prop, g_x=g_x)

        assert res.is_modified is True
        # Throttling must be clamped so that 0.05 * u_1 <= 0.02 + alpha*h
        assert res.safe_action[1] < 0.9

    def test_qp_box_constraints_enforcement(self, cbf):
        """Verify strict adherence to control box constraints u_min <= u <= u_max."""
        qp_filter = QPSafetyFilter(
            cbf=cbf,
            u_dim=1,
            u_min=-1.5,
            u_max=2.0,
        )
        state = SafetyClusterState(replicas=5.0, cpu_headroom=0.50, traffic_drop=0.0)
        g_x = np.zeros((7, 1))
        g_x[0, 0] = 1.0

        # Propose u = 10.0 (violates u_max=2.0)
        res = qp_filter.filter_action(state, u_proposed=10.0, g_x=g_x)
        assert np.isclose(res.safe_action[0], 2.0)

    def test_qp_filter_discrete_mitigation_actions(self, cbf):
        """Test filtering of discrete mitigation directives."""
        qp_filter = QPSafetyFilter(cbf=cbf, u_dim=1)
        # State at traffic drop limit: drop = 0.048 (limit is 0.05)
        state = SafetyClusterState(
            replicas=3.0,
            cpu_headroom=0.30,
            traffic_drop=0.048,
        )
        # Action 3: CIRCUIT_BREAKER_OPEN (causes +0.08 traffic drop, pushing drop to 0.128 > 0.05)
        safe_act, res = qp_filter.filter_discrete_mitigation(
            current_state=state,
            proposed_action_idx=3,
        )

        assert safe_act != 3  # Circuit breaker prevented from opening
        assert res.is_modified is True
        assert "max_traffic_drop" in res.cbf_violations_prevented


# ==============================================================================
# 3. LyapunovStabilityVerifier Unit Tests
# ==============================================================================


class TestLyapunovStabilityVerifier:
    """Tests for discrete Lyapunov stability, Riccati synthesis, and oscillation detection."""

    @pytest.fixture
    def cluster_verifier(self):
        return LyapunovStabilityVerifier.create_cluster_verifier(kappa=0.10)

    def test_lyapunov_energy_and_gradient(self, cluster_verifier):
        """Verify V(x^*) = 0, V(x) > 0, and nabla V(x)."""
        x_eq = cluster_verifier.x_eq
        assert cluster_verifier.v(x_eq) == 0.0

        # Displaced state
        x_disp = x_eq.copy()
        x_disp[0] += 2.0  # +2 replicas
        v_val = cluster_verifier.v(x_disp)
        assert v_val > 0.0

        grad = cluster_verifier.gradient_v(x_disp)
        assert grad.shape == (7,)
        assert grad[0] > 0.0

    def test_discrete_drift_step_stable(self, cluster_verifier):
        """Verify Delta V <= -kappa V condition on a dissipative stabilizing transition."""
        x_eq = cluster_verifier.x_eq
        x_t = x_eq + np.array([2.0, -0.1, 0.02, 30.0, 20.0, -0.1, 0.005])

        # Step t+1 contracts by 30% towards equilibrium
        x_next = x_eq + 0.70 * (x_t - x_eq)

        res = cluster_verifier.verify_step(x_t, x_next)
        assert res.is_stable is True
        assert res.drift < 0.0
        assert res.drift <= res.required_drift
        assert res.contraction_ratio < 1.0

    def test_discrete_drift_step_unstable(self, cluster_verifier):
        """Verify detection of destabilizing energy injection."""
        x_eq = cluster_verifier.x_eq
        x_t = x_eq + np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # Divergent step: doubles distance from equilibrium
        x_next = x_eq + np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        res = cluster_verifier.verify_step(x_t, x_next)
        assert res.is_stable is False
        assert res.drift > 0.0
        assert res.contraction_ratio > 1.0

    def test_lyapunov_from_linear_system_synthesis(self):
        """Verify Discrete Lyapunov Equation synthesis: A^T P A - P = -Q."""
        A = np.array([[0.8, 0.1], [0.0, 0.7]])
        Q = np.eye(2)

        verifier = LyapunovStabilityVerifier.from_linear_system(A=A, Q=Q, kappa=0.05)
        P = verifier.P

        # Check Lyapunov residual: A^T P A - P + Q == 0
        residual = A.T @ P @ A - P + Q
        assert np.allclose(residual, 0.0, atol=1e-10)
        # Eigenvalues of P must be strictly positive
        assert np.all(np.linalg.eigvalsh(P) > 0.0)

    def test_lqr_dare_controller_synthesis(self):
        """Verify Discrete Algebraic Riccati Equation (DARE) controller synthesis."""
        A = np.array([[1.1, 0.2], [0.0, 0.9]])  # Open-loop unstable eigenvalue 1.1
        B = np.array([[0.0], [1.0]])

        verifier, K = LyapunovStabilityVerifier.from_lqr_dare(A=A, B=B)

        # Closed-loop matrix A_cl = A - B @ K
        A_cl = A - B @ K
        cl_eigs = np.abs(np.linalg.eigvals(A_cl))

        # Closed-loop must be strictly Schur stable (all eigenvalues < 1.0)
        assert np.all(cl_eigs < 1.0)

    def test_trajectory_exponential_convergence(self, cluster_verifier):
        """Verify full multi-step trajectory evaluation."""
        x_eq = cluster_verifier.x_eq
        trajectory = []
        x_curr = x_eq + np.array([3.0, 0.2, 0.04, 50.0, 30.0, 0.2, 0.01])

        for _ in range(15):
            trajectory.append(x_curr.copy())
            x_curr = x_eq + 0.8 * (x_curr - x_eq)

        res = cluster_verifier.verify_trajectory(trajectory)
        assert res["is_stable"] is True
        assert res["cumulative_drift"] < 0.0
        assert res["empirical_decay_rate"] > 0.15

    def test_oscillation_detection_destructive_ringing(self, cluster_verifier):
        """Verify detection of high-frequency limit cycles / ringing."""
        x_eq = cluster_verifier.x_eq
        t = np.arange(40)
        # Sinusoidal oscillation around equilibrium with high amplitude
        oscillatory_traj = []
        for step in t:
            perturb = np.zeros(7)
            perturb[0] = 2.0 * math.sin(0.8 * step)       # Replicas swinging +/- 2
            perturb[3] = 100.0 * math.sin(0.8 * step)     # Queue swinging +/- 100
            oscillatory_traj.append(x_eq + perturb)

        diag = cluster_verifier.detect_oscillations(oscillatory_traj)
        assert diag.is_oscillating is True
        assert diag.oscillation_type in ("LIMIT_CYCLE", "DESTRUCTIVE_RINGING")
        assert diag.spectral_power > 0.35
        assert diag.zero_crossing_rate > 0.20

    def test_oscillation_detection_action_flapping(self, cluster_verifier):
        """Verify detection of discrete action flapping (e.g. circuit breaker toggling)."""
        x_eq = cluster_verifier.x_eq
        flat_traj = [x_eq.copy() for _ in range(10)]
        flapping_actions = [0, 3, 0, 3, 0, 3, 0, 3]

        diag = cluster_verifier.detect_oscillations(
            trajectory=flat_traj,
            flapping_actions=flapping_actions,
        )
        assert diag.is_oscillating is True
        assert diag.oscillation_type == "ACTION_FLAPPING"
        assert diag.flapping_count >= 2


# ==============================================================================
# 4. SafetyViolationGuard Unit Tests
# ==============================================================================


class TestSafetyViolationGuard:
    """Tests for the hard survival guard, lockout windows, and emergency aborts."""

    @pytest.fixture
    def guard(self):
        return SafetyViolationGuard.create_cluster_guard(
            min_replicas=1.0,
            min_cpu_headroom=0.01,
            max_traffic_drop=0.50,
            max_queue_depth=1000.0,
            fail_safe_action=0,
            lockout_steps=3,
        )

    def test_guard_permits_healthy_state(self, guard):
        """Normal operational state is permitted without modification."""
        state = SafetyClusterState(replicas=3.0, cpu_headroom=0.30, traffic_drop=0.01)
        dec = guard.evaluate(state, proposed_action=2)

        assert dec.decision == GuardAction.PERMIT
        assert dec.is_safe is True
        assert dec.action == 2
        assert len(dec.violations) == 0

    def test_guard_triggers_emergency_abort_on_fatal_state(self, guard):
        """Fatal physical state breaches trigger hard emergency abort and lockout."""
        fatal_state = SafetyClusterState(
            replicas=0.0,          # Fatal: zero replicas means dead service
            cpu_headroom=0.005,    # Fatal: < 1% CPU headroom
            traffic_drop=0.75,     # Fatal: 75% traffic dropped
        )
        dec = guard.evaluate(fatal_state, proposed_action=1)

        assert dec.decision == GuardAction.EMERGENCY_ABORT
        assert dec.is_safe is False
        assert dec.action == 0  # Reverted to fail_safe_action
        assert len(dec.violations) >= 3
        assert dec.lockout_remaining == 3
        assert len(guard.abort_history) == 1

    def test_guard_lockout_window_suppression(self, guard):
        """During lockout mode, all subsequent directives are suppressed."""
        # Trigger lockout
        fatal_state = SafetyClusterState(replicas=0.5)
        guard.evaluate(fatal_state, proposed_action=1)

        # Immediate next attempt on recovered state
        healthy_state = SafetyClusterState(replicas=3.0, cpu_headroom=0.30, traffic_drop=0.0)
        dec_lockout = guard.evaluate(healthy_state, proposed_action=2)

        assert dec_lockout.decision == GuardAction.LOCKOUT
        assert dec_lockout.is_safe is False
        assert dec_lockout.action == 0
        assert dec_lockout.lockout_remaining == 2

    def test_guard_overrides_qp_infeasible(self, guard):
        """When QP solver declares INFEASIBLE_ABORT, guard triggers fail-safe override."""
        healthy_state = SafetyClusterState(replicas=3.0, cpu_headroom=0.30)
        cbf_infeasible = SafetyFilterResult(
            safe_action=np.array([0.0]),
            proposed_action=np.array([5.0]),
            is_modified=True,
            cost=10.0,
            barrier_values={},
            safety_margin=-1.0,
            status="INFEASIBLE_ABORT",
            cbf_violations_prevented=["min_replicas"],
        )

        dec = guard.evaluate(healthy_state, proposed_action=5.0, cbf_result=cbf_infeasible)
        assert dec.decision == GuardAction.OVERRIDE_FAILSAFE
        assert dec.is_safe is False
        assert dec.action == 0

    def test_guard_overrides_lyapunov_divergence(self, guard):
        """When Lyapunov drift diverges exponentially, guard triggers fail-safe override."""
        healthy_state = SafetyClusterState(replicas=3.0, cpu_headroom=0.30)
        lyap_divergent = LyapunovStepResult(
            is_stable=False,
            v_t=10.0,
            v_next=45.0,  # 4.5x energy increase > max_lyapunov_divergence=3.0
            drift=35.0,
            required_drift=-1.0,
            contraction_ratio=4.5,
            margin=-36.0,
        )

        dec = guard.evaluate(healthy_state, proposed_action=2, lyapunov_result=lyap_divergent)
        assert dec.decision == GuardAction.OVERRIDE_FAILSAFE
        assert dec.is_safe is False
        assert dec.action == 0


# ==============================================================================
# 5. End-to-End Closed-Loop Safety Verification
# ==============================================================================


class TestEndToEndFormalSafetyLoop:
    """Verifies complete integration of CBF + QP Filter + Lyapunov + Guard in a simulation loop."""

    def test_closed_loop_prevention_of_replica_underflow(self):
        """
        Simulates an aggressive controller trying to terminate replicas.
        Verifies that formal safety components guarantee replicas NEVER drop below 2.0.
        """
        cbf = ControlBarrierFunction.create_cluster_safety_cbf(
            min_replicas=2.0,
            min_cpu_headroom=0.15,
            max_traffic_drop=0.05,
        )
        qp_filter = QPSafetyFilter(cbf=cbf, u_dim=1, u_min=-3.0, u_max=3.0)
        verifier = LyapunovStabilityVerifier.create_cluster_verifier(kappa=0.05)
        guard = SafetyViolationGuard.create_cluster_guard(min_replicas=1.0)

        # Start cluster with 3.0 replicas
        state = SafetyClusterState(
            replicas=3.0,
            cpu_headroom=0.35,
            traffic_drop=0.01,
            queue_depth=20.0,
            latency_ms=45.0,
        )

        dt = 0.5
        g_x = np.zeros((7, 1))
        g_x[0, 0] = 1.0  # dot{replicas} = u

        # Run 10 control cycles where an aggressive controller insists on u = -2.0
        min_replicas_observed = 3.0

        for step in range(10):
            # 1. Proposed aggressive control
            u_prop = -2.0

            # 2. QP Safety Filter projection
            cbf_res = qp_filter.filter_action(state, u_proposed=u_prop, g_x=g_x)

            # 3. Predict next state under filtered action
            next_vec = state.to_array() + (g_x @ cbf_res.safe_action) * dt
            next_state = SafetyClusterState.from_array(next_vec)

            # 4. Verify Lyapunov stability
            lyap_res = verifier.verify_step(state, next_state)

            # 5. Safety Guard evaluation
            guard_dec = guard.evaluate(
                state=state,
                proposed_action=cbf_res.safe_action,
                cbf_result=cbf_res,
                lyapunov_result=lyap_res,
            )

            # Execute permitted action
            actual_u = guard_dec.action
            if isinstance(actual_u, np.ndarray):
                actual_u = float(actual_u[0])

            state.replicas = max(0.0, state.replicas + actual_u * dt)
            min_replicas_observed = min(min_replicas_observed, state.replicas)

        # Formal verification check: Replicas must strictly respect the CBF boundary >= 2.0
        assert min_replicas_observed >= 2.0 - 1e-4
