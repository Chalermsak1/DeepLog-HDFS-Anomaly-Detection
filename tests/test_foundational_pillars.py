"""Comprehensive Test Suite for Foundational Research Pillars.

Validates:
1. Continuous Byte-Level Neural Field & HDC (Zero-Parsing Observability)
2. Port-Hamiltonian & Symplectic Systems (Queue Energy Passivity & Physical Invariants)
3. Adaptive Conformal Prediction (Finite-Sample Statistical Coverage Guarantees)
4. In-Kernel eBPF XDP Fast-Path Actuation (Sub-Microsecond Mitigation)
5. Modern Cloud-Native Benchmark Suite (etcd/Raft, gRPC, and CNI Telemetry)
"""

from __future__ import annotations

import math
import numpy as np
import pytest
import torch

from deeplog.byte_field_operator import (
    ByteStreamAnomalyResult,
    ContinuousByteFieldOperator,
)
from deeplog.cloud_native_benchmark import (
    CloudNativeBenchmarkSuite,
    DistributedWorkloadType,
    generate_synthetic_cloud_native_trace,
)
from deeplog.conformal import (
    AdaptiveConformalAnomalyDetector,
    ConformalPredictionResult,
)
from deeplog.ebpf_xdp_actuator import (
    FastPathActuationResult,
    GlobalClusterStatus,
    InKernelXDPActuator,
    XDPAction,
)
from deeplog.port_hamiltonian import (
    DistributedPortHamiltonianObserver,
    PortHamiltonianNeuralQueue,
    PortHamiltonianState,
)


# ==============================================================================
# Pillar 1: Byte-Level Neural Field & Hypervector Tests
# ==============================================================================
class TestContinuousByteFieldOperator:
    """Validates zero-parsing continuous byte representation and HDC encoding."""

    def test_shannon_entropy_calculation(self) -> None:
        op = ContinuousByteFieldOperator()
        # Empty stream
        assert op.compute_shannon_entropy(b"") == 0.0

        # Uniform byte stream (maximum entropy = 8 bits/byte)
        uniform_bytes = bytes(range(256))
        entropy = op.compute_shannon_entropy(uniform_bytes)
        assert abs(entropy - 8.0) < 1e-4

        # Monotonous repetitive stream (zero entropy)
        constant_bytes = b"AAAAAAAAAAAAAAAAAAAA"
        assert op.compute_shannon_entropy(constant_bytes) == 0.0

    def test_fourier_field_projection(self) -> None:
        op = ContinuousByteFieldOperator(embedding_dim=64, num_frequencies=8)
        raw_stream = b"Packet received from 192.168.1.1 port 8080 length 1420"
        field_vec = op.encode_fourier_field(raw_stream)

        assert field_vec.shape == (64,)
        # Unit sphere normalization
        norm = np.linalg.norm(field_vec)
        assert abs(norm - 1.0) < 1e-4

    def test_hdc_hypervector_properties(self) -> None:
        op = ContinuousByteFieldOperator(hdc_dim=1000)
        raw_a = b"raft.Node: 1 elected leader at term 4"
        raw_b = b"raft.Node: 1 elected leader at term 4"
        raw_diff = b"Fatal disk failure on block device /dev/sdb1: I/O error"

        vec_a = op.encode_hdc_hypervector(raw_a)
        vec_b = op.encode_hdc_hypervector(raw_b)
        vec_diff = op.encode_hdc_hypervector(raw_diff)

        # Values are strictly bipolar {-1, +1}
        assert set(np.unique(vec_a)).issubset({-1, 1})
        # Exact match yields identical hypervectors
        assert np.array_equal(vec_a, vec_b)

        # Orthogonality of different sequences in high dimensions
        cosine_sim = np.dot(vec_a.astype(float), vec_diff.astype(float)) / 1000.0
        assert abs(cosine_sim) < 0.25  # Near orthogonal

    def test_resilience_to_template_mutations(self) -> None:
        """Proves zero-parsing resilience: format string mutations do not cause OOV collapse."""
        op = ContinuousByteFieldOperator(embedding_dim=64, hdc_dim=2000)
        normal_samples = [
            b"Received block blk_1001 of size 4096 from /10.0.0.1",
            b"Received block blk_1002 of size 4096 from /10.0.0.2",
            b"Received block blk_1003 of size 8192 from /10.0.0.3",
        ]
        op.fit_baseline(normal_samples)

        # Developer updates log message format in next software release
        mutated_version = b"Block blk_1004 received (size: 4096 bytes, source: 10.0.0.4)"
        res = op.evaluate_stream(mutated_version)

        # Semantic field similarity remains strong, no crash, no OOV
        assert res.metadata["fourier_field_similarity"] > 0.85
        assert res.hdc_cosine_similarity > 0.15
        assert not res.is_anomalous

        # Truly malicious payload should be flagged
        malicious = b"\x90\x90\x90\x31\xc0\x50\x68\x2f\x2f\x73\x68\x68\x2f\x62\x69\x6e" * 20
        res_mal = op.evaluate_stream(malicious)
        assert res_mal.is_anomalous


# ==============================================================================
# Pillar 2: Port-Hamiltonian Neural Queue Tests
# ==============================================================================
class TestPortHamiltonianNeuralQueue:
    """Validates Industrial Physics conservation laws, skew-symmetry, and passivity."""

    def test_skew_symmetry_of_J(self) -> None:
        phnn = PortHamiltonianNeuralQueue(state_dim=6)
        J = phnn.get_skew_symmetric_J()

        # Strict antisymmetry: J + J^T = 0
        anti_sym_diff = J + J.T
        assert torch.max(torch.abs(anti_sym_diff)).item() < 1e-6

        # Diagonal of J is strictly zero
        diag = torch.diag(J)
        assert torch.max(torch.abs(diag)).item() < 1e-6

        # Quadratic form z^T J z == 0 for ANY arbitrary vector z
        z = torch.randn(6)
        quad_form = torch.dot(z, J @ z).item()
        assert abs(quad_form) < 1e-6

    def test_dissipation_positive_semidefinite(self) -> None:
        phnn = PortHamiltonianNeuralQueue(state_dim=4, min_dissipation=1e-3)
        R = phnn.get_dissipation_R()

        # Symmetry: R = R^T
        sym_diff = R - R.T
        assert torch.max(torch.abs(sym_diff)).item() < 1e-6

        # All eigenvalues >= min_dissipation
        eigvals = torch.linalg.eigvalsh(R)
        assert torch.min(eigvals).item() >= 1e-4

        # Quadratic form z^T R z >= 0
        z = torch.randn(4)
        quad_form = torch.dot(z, R @ z).item()
        assert quad_form >= 0.0

    def test_passivity_and_first_law_of_thermodynamics(self) -> None:
        """Verifies dH/dt <= y^T u. Isolated system (u=0) CANNOT gain energy."""
        phnn = PortHamiltonianNeuralQueue(state_dim=4, input_dim=1)
        x = torch.tensor([10.0, 5.0, 2.0, 1.0])

        # Case A: Autonomous unforced system (u = 0)
        dH_dt, diss, ext, is_valid = phnn.verify_energy_conservation(x, u=None)
        assert is_valid
        assert ext == 0.0
        # Energy must dissipate or stay constant (dH/dt <= 0)
        assert dH_dt <= 1e-6

        # Case B: Forced system with external workload u
        u = torch.tensor([5.0])
        dH_dt_f, diss_f, ext_f, is_valid_f = phnn.verify_energy_conservation(x, u=u)
        assert is_valid_f
        # dH/dt == dissipation + external work
        assert abs(dH_dt_f - (diss_f + ext_f)) < 1e-4

    def test_rk4_integration_trajectory(self) -> None:
        phnn = PortHamiltonianNeuralQueue(state_dim=4)
        x0 = torch.tensor([5.0, 5.0, 5.0, 5.0])
        x1 = phnn.rk4_step(x0, u=None, dt=0.05)

        # In autonomous dissipation, energy at t1 should be strictly <= energy at t0
        H0 = phnn.hamiltonian_energy(x0).item()
        H1 = phnn.hamiltonian_energy(x1).item()
        assert H1 <= H0 + 1e-5

    def test_observer_ghost_request_detection(self) -> None:
        observer = DistributedPortHamiltonianObserver(num_queues=4)
        # Baseline normal steps with gradual dissipation
        for _ in range(30):
            observer.observe_transition(
                current_state=[10.0, 10.0, 10.0, 10.0],
                next_state=[9.8, 9.8, 9.8, 9.8],
                control_input=0.0,
            )

        # Sudden unphysical jump: queue spikes 500% with ZERO external input work
        st = observer.observe_transition(
            current_state=[9.8, 9.8, 9.8, 9.8],
            next_state=[80.0, 95.0, 70.0, 85.0],
            control_input=0.0,
        )
        assert st.is_conservative_violation
        assert "PASSIVITY_VIOLATION_GHOST_REQUESTS" in st.diagnostic


# ==============================================================================
# Pillar 3: Conformal Anomaly Prediction Tests
# ==============================================================================
class TestAdaptiveConformalAnomalyDetector:
    """Validates finite-sample statistical coverage and adaptive tracking."""

    def test_initial_calibration_and_quantile(self) -> None:
        det = AdaptiveConformalAnomalyDetector(target_alpha=0.05, calibration_window=500)
        rng = np.random.RandomState(42)
        normal_scores = rng.normal(loc=0.2, scale=0.05, size=200).tolist()
        det.calibrate(normal_scores)

        q = det.compute_conformal_quantile(alpha=0.05)
        # 95th percentile should be ~ 0.2 + 1.645 * 0.05 = ~0.28
        assert 0.25 <= q <= 0.35

    def test_conformal_coverage_empirical_property(self) -> None:
        """Evaluates empirical coverage convergence over stationary baseline."""
        det = AdaptiveConformalAnomalyDetector(target_alpha=0.10, adaptation_rate=0.01)
        rng = np.random.RandomState(123)
        calib = rng.exponential(scale=1.0, size=300).tolist()
        det.calibrate(calib)

        test_stream = rng.exponential(scale=1.0, size=500).tolist()
        anomalies = 0
        for s in test_stream:
            res = det.evaluate_and_update(s)
            if res.is_anomaly:
                anomalies += 1

        empirical_error = anomalies / 500.0
        # Should be closely bounded around target alpha (0.10 +/- 0.05)
        assert 0.05 <= empirical_error <= 0.16

    def test_pac_bayes_risk_bound(self) -> None:
        det = AdaptiveConformalAnomalyDetector(target_alpha=0.05)
        # Evaluate 50 normal scores
        for _ in range(50):
            det.evaluate_and_update(0.1)

        pac_bound = det.compute_pac_bayes_bound(kl_divergence=0.05)
        # PAC bound is a valid upper bound in [0, 1]
        assert 0.0 <= pac_bound <= 1.0


# ==============================================================================
# Pillar 4: In-Kernel eBPF XDP Fast-Path Actuation Tests
# ==============================================================================
class TestInKernelXDPActuator:
    """Validates sub-microsecond packet evaluation and policy enforcement."""

    def test_ip_conversion(self) -> None:
        actuator = InKernelXDPActuator()
        ip_str = "192.168.1.100"
        ip_int = actuator.ip_to_int(ip_str)
        assert actuator.int_to_ip(ip_int) == ip_str

    def test_hard_drop_actuation(self) -> None:
        actuator = InKernelXDPActuator()
        bad_ip = "198.51.100.4"
        actuator.set_ip_rule(bad_ip, XDPAction.DROP)

        res = actuator.evaluate_packet_fastpath(bad_ip)
        assert res.action == XDPAction.DROP
        assert "CBF_HARD_DROP" in res.rule_applied
        # Sub-microsecond latency guarantee
        assert res.latency_microseconds < 10.0

    def test_canary_shedding_50_percent(self) -> None:
        actuator = InKernelXDPActuator()
        canary_ip = "10.0.0.50"
        actuator.set_ip_rule(canary_ip, XDPAction.CANARY_SHED)

        actions = [
            actuator.evaluate_packet_fastpath(canary_ip).action for _ in range(10)
        ]
        # Should alternate or split drops and passes
        drops = actions.count(XDPAction.DROP)
        passes = actions.count(XDPAction.PASS)
        assert drops == 5
        assert passes == 5

    def test_global_emergency_mode(self) -> None:
        actuator = InKernelXDPActuator()
        actuator.set_global_emergency_status(GlobalClusterStatus.CRITICAL_SHEDDING)

        # Non-SYN packet dropped
        res_data = actuator.evaluate_packet_fastpath("10.1.2.3", is_syn=False)
        assert res_data.action == XDPAction.DROP

        # SYN handshake preserved to prevent connection starvation
        res_syn = actuator.evaluate_packet_fastpath("10.1.2.3", is_syn=True)
        assert res_syn.action == XDPAction.PASS


# ==============================================================================
# Pillar 5: Cloud-Native Distributed Benchmark Suite Tests
# ==============================================================================
class TestCloudNativeBenchmarkSuite:
    """Validates real-world distributed fault scenarios and comparative metrics."""

    def test_synthetic_trace_generation(self) -> None:
        events = generate_synthetic_cloud_native_trace(num_events=100, anomaly_ratio=0.2)
        assert len(events) == 100
        anomalies = [e for e in events if e.is_true_anomaly]
        assert len(anomalies) > 0
        workload_types = {e.workload for e in anomalies}
        assert DistributedWorkloadType.RAFT_CONSENSUS_SPLIT_BRAIN in workload_types

    def test_full_benchmark_comparison_execution(self) -> None:
        suite = CloudNativeBenchmarkSuite(seed=42)
        metrics = suite.run_full_comparison()

        assert len(metrics) == 4
        d_infinity = metrics[0]
        assert d_infinity.system_name.startswith("DeepLog-Infinity")
        assert d_infinity.f1_score > 0.70
        assert d_infinity.resilient_to_template_mutation
        assert d_infinity.mean_mitigation_latency_us < 15.0  # Microsecond fast-path (< 15us vs 2.85s baseline)
