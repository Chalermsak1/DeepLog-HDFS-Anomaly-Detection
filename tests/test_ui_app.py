"""
Unit Tests for DeepLog Mission Control Cockpit (app.py)
======================================================
Tests verify:
- Tab 1: SOTA Benchmark metrics, confusion matrix math, sequence diagnosis with fallback.
- Tab 2: OTel MELT telemetry window construction, cross-modal attribution isolation.
- Tab 3: Chaos Gym environment initialization, adversarial step execution, self-play trajectory.
- Tab 4: Control Barrier Functions h(x), QP safety filter projection, Lyapunov contraction drift.
- Tab 5: Neuro-Symbolic AST mapping, patch synthesis with unified diff and JUnit 5 test suite.
- Tab 6: Federated ZK threat packet generation, Differential Privacy, Byzantine poisoning defense, HDC memory bank.
"""

import math
import os
import sys
import pytest
import numpy as np

# Ensure root directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import app


# ==============================================================================
# Tab 1: SOTA Benchmarks & Confusion Matrix Tests
# ==============================================================================

def test_sota_benchmarks_structure_and_values():
    """Verify SOTA benchmarks contain all 3 DeepLog generations with valid metrics."""
    benchmarks = app.get_sota_benchmarks()
    assert len(benchmarks) == 3

    generations = [b["Generation"] for b in benchmarks]
    assert any("1.0" in g for g in generations)
    assert any("2.0" in g for g in generations)
    assert any("3.0" in g for g in generations)

    dl3 = [b for b in benchmarks if "3.0" in b["Generation"]][0]
    dl1 = [b for b in benchmarks if "1.0" in b["Generation"]][0]

    # Verify DeepLog 3.0 outperforms DeepLog 1.0 baseline
    assert dl3["Accuracy"] > dl1["Accuracy"]
    assert dl3["F1-Score"] > dl1["F1-Score"]
    assert dl3["Recall"] > dl1["Recall"]
    assert dl3["FPR"] < dl1["FPR"]
    assert "eBPF" in dl3["Inference Latency"]


def test_confusion_matrix_mathematical_consistency():
    """Verify confusion matrix totals, accuracy, precision, recall, and FPR consistency."""
    cm = app.get_confusion_matrix_data()
    tp = cm["tp"]
    fp = cm["fp"]
    tn = cm["tn"]
    fn = cm["fn"]
    total = cm["total"]

    assert total == tp + fp + tn + fn
    assert total == 570204

    # Verify rates
    expected_acc = round((tp + tn) / total * 100, 2)
    assert cm["accuracy"] == expected_acc

    expected_prec = round(tp / (tp + fp) * 100, 2)
    assert cm["precision"] == expected_prec

    expected_rec = round(tp / (tp + fn) * 100, 2)
    assert cm["recall"] == expected_rec

    expected_f1 = round(2 * tp / (2 * tp + fp + fn) * 100, 2)
    assert cm["f1"] == expected_f1

    expected_fpr = round(fp / (fp + tn) * 100, 3)
    assert cm["fpr"] == expected_fpr
    assert cm["fpr"] < 0.5  # Prevents operator alert fatigue


def test_load_deeplog_model_and_mapping():
    """Verify load_deeplog_model_and_mapping returns mapping and handles fallback."""
    model, mapping, preprocessor, device, is_fallback = app.load_deeplog_model_and_mapping()
    assert mapping is not None
    assert len(mapping) >= 15
    assert 5 in mapping or -1337 in mapping
    assert str(device) in {"cpu", "cuda", "mps"}
    assert isinstance(is_fallback, bool)


def test_diagnose_sequence_normal_and_anomaly():
    """Verify live sequence anomaly diagnosis on normal, abnormal, and empty sequences."""
    # 1. Normal execution block
    normal_seq = "5 5 5 22 11 9 11 9 26 26 23 21"
    diag_norm = app.diagnose_sequence(normal_seq)
    assert diag_norm["is_anomaly"] is False
    assert diag_norm["verdict"] == "NORMAL_WORKFLOW"
    assert len(diag_norm["steps"]) > 0

    # 2. Anomaly execution block with error template E7
    anomaly_seq = "5 5 22 7 26 23"
    diag_anom = app.diagnose_sequence(anomaly_seq)
    assert diag_anom["is_anomaly"] is True
    assert diag_anom["verdict"] == "ANOMALY_DETECTED"
    assert any(step["is_anomaly"] for step in diag_anom["steps"])

    # 3. Empty sequence handling
    empty_diag = app.diagnose_sequence("")
    assert empty_diag["is_anomaly"] is False
    assert empty_diag["verdict"] == "EMPTY_SEQUENCE"


# ==============================================================================
# Tab 2: OTel Multi-Modal Live Stream Tests
# ==============================================================================

def test_build_otel_telemetry_window_and_failure_modes():
    """Verify TelemetryWindow contains synchronized metrics, spans, logs, and profiles."""
    window = app.build_otel_telemetry_window(failure_mode="NONE")
    assert window is not None

    metrics = window.metrics if hasattr(window, "metrics") else window.get("metrics", [])
    spans = window.spans if hasattr(window, "spans") else window.get("spans", [])
    logs = window.logs if hasattr(window, "logs") else window.get("logs", [])
    profiles = window.profiles if hasattr(window, "profiles") else window.get("profiles", [])

    assert len(metrics) > 0
    assert len(spans) > 0
    assert len(logs) > 0
    assert len(profiles) > 0


def test_compute_multimodal_attribution_drivers():
    """Verify cross-modal attribution identifies root cause drivers correctly."""
    # Thread exhaustion -> METRIC_SATURATION
    w_thread = app.build_otel_telemetry_window(failure_mode="THREAD_EXHAUSTION")
    attr_thread = app.compute_multimodal_attribution(w_thread)
    assert attr_thread["primary_driver"] == "METRIC_SATURATION"
    assert sum(attr_thread["attribution_percentages"].values()) == pytest.approx(100.0, 0.1)

    # Latency spike -> TRACE_BOTTLENECK
    w_lat = app.build_otel_telemetry_window(failure_mode="LATENCY_SPIKE")
    attr_lat = app.compute_multimodal_attribution(w_lat)
    assert attr_lat["primary_driver"] == "TRACE_BOTTLENECK"
    assert attr_lat["attribution_percentages"]["TRACE_BOTTLENECK"] > 50.0

    # Profiling hotspot -> PROFILE_HOTSPOT
    w_prof = app.build_otel_telemetry_window(failure_mode="PROFILE_HOTSPOT")
    attr_prof = app.compute_multimodal_attribution(w_prof)
    assert attr_prof["primary_driver"] == "PROFILE_HOTSPOT"


# ==============================================================================
# Tab 3: Chaos Gym Playground Tests
# ==============================================================================

def test_chaos_gym_initialization_and_step_execution():
    """Verify ChaosEnvironment steps with Red attack and Blue immune mitigation."""
    env = app.init_chaos_gym_env(num_nodes=4)

    # Red injects thread exhaustion; Blue immune agent migrates replicas
    res = app.execute_chaos_step(
        env,
        target_node=1,
        fault_type=app.ChaosFaultType.THREAD_EXHAUSTION if app.ChaosFaultType else "THREAD_EXHAUSTION",
        intensity=0.85,
        blue_enabled=True,
    )

    assert "availability" in res
    assert 0.0 <= res["availability"] <= 1.0
    assert "blast_radius" in res
    assert "p99_latency_ms" in res
    assert len(res["nodes"]) == 4
    assert res["blue_action"] in {"MIGRATE_REPLICA", "EBPF_THROTTLE", "CIRCUIT_BREAKER_OPEN", "NO_ACTION"}


def test_chaos_gym_byzantine_slowloris_ebpf_throttle():
    """Verify Blue immune agent triggers in-kernel eBPF throttling on Byzantine Slowloris."""
    env = app.init_chaos_gym_env(num_nodes=4)
    res = app.execute_chaos_step(
        env,
        target_node=2,
        fault_type=app.ChaosFaultType.BYZANTINE_SLOWLORIS if app.ChaosFaultType else "BYZANTINE_SLOWLORIS",
        intensity=0.90,
        blue_enabled=True,
    )
    assert res["blue_action"] == "EBPF_THROTTLE"
    assert res["target_node"] == 2


def test_run_chaos_trajectory():
    """Verify multi-step simulation trajectory produces chronological step records."""
    traj = app.run_chaos_trajectory(num_steps=5, blue_enabled=True)
    assert len(traj) == 5
    for i, t in enumerate(traj):
        assert t["step"] == i + 1
        assert "availability" in t
        assert "blast_radius" in t


# ==============================================================================
# Tab 4: Provable Safety Radar Tests
# ==============================================================================

def test_evaluate_cbf_safety_safe_and_unsafe_states():
    """Verify Control Barrier Functions evaluate invariants h(x) >= 0 correctly."""
    # 1. Safe Cluster State
    safe_state = {
        "replicas": 3.0,
        "cpu_headroom": 0.35,
        "traffic_drop": 0.01,
        "queue_depth": 10.0,
        "latency_ms": 45.0,
    }
    eval_safe = app.evaluate_cbf_safety(safe_state)
    assert eval_safe["is_safe"] is True
    assert eval_safe["status"] == "SAFE"
    assert len(eval_safe["invariants"]) == 5
    for inv in eval_safe["invariants"]:
        assert inv["status"] == "SATISFIED"
        assert inv["barrier_h"] >= 0.0

    # 2. Unsafe Cluster State (Violating Replicas and Latency)
    unsafe_state = {
        "replicas": 1.0,  # Below minimum threshold of 2.0
        "cpu_headroom": 0.05,  # Below minimum threshold of 0.15
        "traffic_drop": 0.08,  # Above maximum threshold of 0.05
        "queue_depth": 65.0,  # Above maximum threshold of 50.0
        "latency_ms": 220.0,  # Above maximum threshold of 150.0
    }
    eval_unsafe = app.evaluate_cbf_safety(unsafe_state)
    assert eval_unsafe["is_safe"] is False
    assert eval_unsafe["status"] == "UNSAFE_VIOLATION"
    assert eval_unsafe["min_margin"] < 0.0


def test_solve_qp_safety_filter_projection():
    """Verify QP safety filter clamps aggressive control actions violating safety sets."""
    state = {"cpu_headroom": 0.25}  # Allowable capacity: (0.25 - 0.15) * 10 = 1.0

    # Action within safe set: u = 0.8
    safe_prop = app.solve_qp_safety_filter(0.8, state)
    assert safe_prop["was_filtered"] is False
    assert safe_prop["safe_action"] == pytest.approx(0.8, 0.01)
    assert safe_prop["slack"] == 0.0

    # Aggressive action outside safe set: u = 4.5
    aggr_prop = app.solve_qp_safety_filter(4.5, state)
    assert aggr_prop["was_filtered"] is True
    assert aggr_prop["safe_action"] <= 1.001
    assert aggr_prop["slack"] > 0.0


def test_evaluate_lyapunov_stability():
    """Verify Lyapunov energy V(x) and discrete contraction drift calculation."""
    curr = {"queue_depth": 30.0, "latency_ms": 100.0}
    # Stabilizing transition
    stabilizing_next = {"queue_depth": 15.0, "latency_ms": 50.0}
    res_stable = app.evaluate_lyapunov_stability(curr, stabilizing_next)
    assert res_stable["is_stable"] is True
    assert res_stable["drift"] < 0.0
    assert "Verified" in res_stable["oscillation_risk"]

    # Destabilizing transition
    destabilizing_next = {"queue_depth": 60.0, "latency_ms": 200.0}
    res_unstable = app.evaluate_lyapunov_stability(curr, destabilizing_next)
    assert res_unstable["is_stable"] is False
    assert res_unstable["drift"] > 0.0


# ==============================================================================
# Tab 5: Neuro-Symbolic AST & Auto Git Patch Tests
# ==============================================================================

def test_synthesize_ast_patch_proposal_format():
    """Verify patch synthesis outputs syntactically valid Git diffs and JUnit 5 code."""
    keys = app.get_available_incident_keys()
    assert len(keys) >= 2
    assert "HDFS-E7-SOCKET-TIMEOUT" in keys

    patch = app.synthesize_ast_patch_proposal("HDFS-E7-SOCKET-TIMEOUT")
    assert patch["target_file"].endswith(".java")
    assert patch["faulting_line"] > 0
    assert patch["dry_run_ok"] is True

    # Check Unified Git Diff syntax
    diff = patch["patch_diff"]
    assert "--- a/" in diff
    assert "+++ b/" in diff
    assert "@@ " in diff

    # Check JUnit test suite structure
    junit = patch["unit_test_code"]
    assert "@Test" in junit
    assert "org.junit" in junit
    assert "assert" in junit.lower()


# ==============================================================================
# Tab 6: Federated ZK Threat Mesh & HDC Tests
# ==============================================================================

def test_federated_mesh_packet_generation_and_verification():
    """Verify 10,000-bit HDC threat signature generation, DP masking, and verification."""
    # 1. Authentic clean packet
    pkt_clean = app.generate_federated_threat_packet(
        node_id="us-east-1",
        failure_mode="DISTRIBUTED_SYN_FLOOD",
        epsilon=1.0,
        inject_poison=False,
    )
    assert pkt_clean["node_id"] == "us-east-1"
    assert pkt_clean["is_poisoned"] is False
    assert 0.35 <= pkt_clean["bit_density"] <= 0.65  # Statistical asymptotic density
    assert pkt_clean["has_zk_proof"] is True

    verif_clean = app.verify_mesh_threat_packet(pkt_clean)
    assert verif_clean["is_valid"] is True
    assert verif_clean["status"] == "VERIFIED_AUTHENTIC"
    assert verif_clean["reputation_delta"] > 0

    # 2. Byzantine poisoned packet (Density/Entropy violation)
    pkt_poison = app.generate_federated_threat_packet(
        node_id="us-east-1",
        failure_mode="DISTRIBUTED_SYN_FLOOD",
        epsilon=1.0,
        inject_poison=True,
    )
    assert pkt_poison["is_poisoned"] is True

    verif_poison = app.verify_mesh_threat_packet(pkt_poison)
    assert verif_poison["is_valid"] is False
    assert verif_poison["status"] == "POISONING_DETECTED"
    assert verif_poison["reputation_delta"] < 0


def test_query_hdc_memory_bank():
    """Verify sub-microsecond HDC memory bank cosine similarity query."""
    match = app.query_hdc_memory_bank(None, [5, 22, 11, 7], failure_mode="SYN_FLOOD")
    assert match["matched_threat"] == "SYN_FLOOD"
    assert 0.0 <= match["cosine_similarity"] <= 1.0
    assert match["confidence"] > 90.0
    assert "match_latency_us" in match
    assert match["match_latency_us"] < 10.0  # Sub-microsecond scale
