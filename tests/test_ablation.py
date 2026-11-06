"""
Rigorous Unit Tests for DeepLog-Infinity Ablation Study Matrix & Modern LLM Benchmarking Suite.
==============================================================================================
Tests:
1. AblationConfig initialization, factory constructors, parameter validation, and serialization.
2. Workload generation across HDFS, BGL, and CHAOS domains with statistical properties.
3. Full Architecture evaluation: Precision, Recall, F1, MTTR, zero safety violations, Lyapunov stability.
4. Ablation w/o PISNO: Demonstrates high false alarm surge on sudden benign bursts.
5. Ablation w/o CBF: Demonstrates unconstrained remediation, safety envelope violations, and MTTR inflation.
6. Ablation w/o Consecutive Violations: Demonstrates false positive rate explosion on transient blips.
7. Ablation w/o eBPF Telemetry: Demonstrates kernel blindness and sharp recall drop.
8. Ablation w/o HDC: Demonstrates 10x latency inflation and higher memory footprint.
9. Vanilla Baseline: Demonstrates compound failure modes.
10. ModernLLMComparisonSimulator: Verifies latency, financial cost, hallucination rates, and stability guarantees.
11. Markdown table and JSON metrics exporters.
12. Benchmark CLI execution and output files.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import pytest
import numpy as np

# Ensure project root and src/ are on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from deeplog.ablation import (
    AblationConfig,
    AblationMetricResult,
    AblationPillar,
    AblationRunner,
    BenchmarkEvent,
    LLMComparisonResult,
    LLMModelProfile,
    ModernLLMComparisonSimulator,
    WorkloadBatch,
    WorkloadType,
    export_ablation_matrix_json,
    format_ablation_markdown_table,
    format_llm_comparison_markdown_table,
    generate_ablation_workload,
)


# ==============================================================================
# 1. AblationConfig Tests
# ==============================================================================


def test_ablation_config_defaults():
    """Verify default Full Architecture config has all pillars enabled."""
    cfg = AblationConfig()
    assert cfg.name == "DeepLog-Infinity (Full)"
    assert cfg.pillar_mode == AblationPillar.FULL
    assert cfg.use_pisno is True
    assert cfg.use_cbf is True
    assert cfg.use_consecutive is True
    assert cfg.use_ebpf is True
    assert cfg.use_hdc is True
    assert cfg.min_consecutive == 3
    assert cfg.static_threshold == 0.50
    assert cfg.cbf_alpha == 1.0


def test_ablation_config_validation():
    """Verify validation raises errors on invalid parameters."""
    with pytest.raises(ValueError, match="min_consecutive must be >= 1"):
        AblationConfig(min_consecutive=0)

    with pytest.raises(ValueError, match="static_threshold must be in"):
        AblationConfig(static_threshold=0.0)

    with pytest.raises(ValueError, match="static_threshold must be in"):
        AblationConfig(static_threshold=1.5)

    with pytest.raises(ValueError, match="cbf_alpha must be positive"):
        AblationConfig(cbf_alpha=-0.5)


def test_ablation_config_factories():
    """Verify factory methods toggle respective pillars correctly."""
    full = AblationConfig.full()
    assert full.pillar_mode == AblationPillar.FULL
    assert full.use_pisno and full.use_cbf and full.use_consecutive and full.use_ebpf and full.use_hdc

    wo_pisno = AblationConfig.without_pisno()
    assert wo_pisno.pillar_mode == AblationPillar.WO_PISNO
    assert not wo_pisno.use_pisno
    assert wo_pisno.use_cbf and wo_pisno.use_consecutive

    wo_cbf = AblationConfig.without_cbf()
    assert wo_cbf.pillar_mode == AblationPillar.WO_CBF
    assert not wo_cbf.use_cbf
    assert wo_cbf.use_pisno and wo_cbf.use_consecutive

    wo_consec = AblationConfig.without_consecutive()
    assert wo_consec.pillar_mode == AblationPillar.WO_CONSECUTIVE
    assert not wo_consec.use_consecutive
    assert wo_consec.min_consecutive == 1

    wo_ebpf = AblationConfig.without_ebpf()
    assert wo_ebpf.pillar_mode == AblationPillar.WO_EBPF
    assert not wo_ebpf.use_ebpf

    wo_hdc = AblationConfig.without_hdc()
    assert wo_hdc.pillar_mode == AblationPillar.WO_HDC
    assert not wo_hdc.use_hdc

    vanilla = AblationConfig.vanilla()
    assert vanilla.pillar_mode == AblationPillar.VANILLA_BASELINE
    assert not vanilla.use_pisno and not vanilla.use_cbf and not vanilla.use_consecutive
    assert not vanilla.use_ebpf and not vanilla.use_hdc


def test_ablation_config_serialization():
    """Verify dictionary conversion round-trip."""
    cfg = AblationConfig.without_pisno(seed=99)
    d = cfg.to_dict()
    assert d["pillar_mode"] == "wo_pisno"
    assert d["use_pisno"] is False
    assert d["random_seed"] == 99

    loaded = AblationConfig.from_dict(d)
    assert loaded.pillar_mode == AblationPillar.WO_PISNO
    assert loaded.use_pisno is False
    assert loaded.random_seed == 99


# ==============================================================================
# 2. Workload Generation Tests
# ==============================================================================


@pytest.mark.parametrize("w_type", [WorkloadType.HDFS, WorkloadType.BGL, WorkloadType.CHAOS])
def test_generate_ablation_workload_domains(w_type: WorkloadType):
    """Verify workload generation across all supported domains."""
    batch = generate_ablation_workload(w_type, num_events=300, seed=42)
    assert batch.num_events >= 300
    assert batch.domain == w_type.value
    assert batch.num_anomalies > 0
    assert batch.num_burst_events > 0
    assert batch.num_transient_blips > 0
    assert batch.num_kernel_events > 0
    assert len(batch.block_lengths) > 0
    assert int(np.sum(batch.block_lengths)) == batch.num_events
    assert len(batch.ground_truth) == batch.num_events


def test_generate_ablation_workload_determinism():
    """Verify identical random seeds produce identical workloads."""
    batch1 = generate_ablation_workload(WorkloadType.HDFS, num_events=200, seed=123)
    batch2 = generate_ablation_workload(WorkloadType.HDFS, num_events=200, seed=123)

    assert batch1.num_events == batch2.num_events
    assert np.array_equal(batch1.ground_truth, batch2.ground_truth)
    for e1, e2 in zip(batch1.events, batch2.events):
        assert e1.raw_message == e2.raw_message
        assert e1.is_genuine_anomaly == e2.is_genuine_anomaly
        assert e1.is_burst == e2.is_burst
        assert e1.is_kernel_level == e2.is_kernel_level


# ==============================================================================
# 3. Ablation Empirical Pillar Tests
# ==============================================================================


@pytest.fixture
def calibrated_workload() -> WorkloadBatch:
    """Fixture providing a deterministic test workload."""
    return generate_ablation_workload(WorkloadType.CHAOS, num_events=600, seed=42)


def test_full_architecture_guarantees(calibrated_workload: WorkloadBatch):
    """
    Verify Full Architecture:
    - High Precision (>= 0.90)
    - High Recall (>= 0.90)
    - High F1 (>= 0.90)
    - Fast MTTR (< 2.0s)
    - Zero Safety Violations (100% CBF guarantee)
    - Provably Lyapunov Stable
    - Low Latency (~2.1ms)
    - Zero Monthly Cost ($0.00)
    """
    runner = AblationRunner(random_seed=42)
    full_cfg = AblationConfig.full(seed=42)
    res = runner.run_variant(full_cfg, calibrated_workload)

    assert res.precision >= 0.90, f"Expected Precision >= 0.90, got {res.precision}"
    assert res.recall >= 0.90, f"Expected Recall >= 0.90, got {res.recall}"
    assert res.f1_score >= 0.90, f"Expected F1 >= 0.90, got {res.f1_score}"
    assert res.safety_violations == 0, f"Expected 0 violations, got {res.safety_violations}"
    assert res.safety_violation_rate == 0.0
    assert res.lyapunov_stable is True
    assert res.mttr_seconds < 2.0
    assert res.latency_ms < 5.0
    assert res.monthly_cost_usd == 0.0


def test_ablation_without_pisno_exhibits_burst_false_alarms(calibrated_workload: WorkloadBatch):
    """
    Verify w/o PISNO:
    - Falls back to static threshold.
    - Sudden benign bursts trigger high false alarms.
    - Precision drops significantly compared to Full Architecture.
    """
    runner = AblationRunner(random_seed=42)
    full_res = runner.run_variant(AblationConfig.full(seed=42), calibrated_workload)
    wo_pisno_res = runner.run_variant(AblationConfig.without_pisno(seed=42), calibrated_workload)

    # Precision of w/o PISNO should drop significantly due to false alarms on bursts
    assert wo_pisno_res.precision < full_res.precision - 0.20, (
        f"Expected w/o PISNO precision ({wo_pisno_res.precision:.3f}) to be significantly lower "
        f"than Full ({full_res.precision:.3f})"
    )
    assert wo_pisno_res.false_positives > full_res.false_positives
    assert wo_pisno_res.f1_score < full_res.f1_score


def test_ablation_without_cbf_causes_safety_violations_and_thrashing(calibrated_workload: WorkloadBatch):
    """
    Verify w/o CBF:
    - Unconstrained remediation causes safety envelope breaches (replicas < 2, CPU < 15%, traffic drop > 5%).
    - Safety violations rate > 0 (typically high).
    - MTTR is significantly longer due to recovery from self-induced outages and flapping.
    """
    runner = AblationRunner(random_seed=42)
    full_res = runner.run_variant(AblationConfig.full(seed=42), calibrated_workload)
    wo_cbf_res = runner.run_variant(AblationConfig.without_cbf(seed=42), calibrated_workload)

    assert full_res.safety_violations == 0
    assert wo_cbf_res.safety_violations > 0, "w/o CBF must exhibit safety violations"
    assert wo_cbf_res.safety_violation_rate > 0.50
    assert wo_cbf_res.mttr_seconds > full_res.mttr_seconds * 3.0, (
        f"Expected w/o CBF MTTR ({wo_cbf_res.mttr_seconds:.2f}s) to be at least 3x Full MTTR ({full_res.mttr_seconds:.2f}s)"
    )


def test_ablation_without_consecutive_explodes_on_blips(calibrated_workload: WorkloadBatch):
    """
    Verify w/o Consecutive Violations:
    - Triggering on single spike causes false positive rate to explode under transient blips.
    - Precision is lower than Full Architecture.
    """
    runner = AblationRunner(random_seed=42)
    full_res = runner.run_variant(AblationConfig.full(seed=42), calibrated_workload)
    wo_consec_res = runner.run_variant(AblationConfig.without_consecutive(seed=42), calibrated_workload)

    assert wo_consec_res.false_positives > full_res.false_positives
    assert wo_consec_res.precision < full_res.precision - 0.10, (
        f"Expected w/o Consecutive precision ({wo_consec_res.precision:.3f}) to be lower than Full ({full_res.precision:.3f})"
    )
    assert wo_consec_res.false_positive_rate > full_res.false_positive_rate


def test_ablation_without_ebpf_misses_kernel_events(calibrated_workload: WorkloadBatch):
    """
    Verify w/o eBPF Telemetry:
    - Userspace logs only; completely blind to in-kernel drops and socket stalls.
    - Recall plummets compared to Full Architecture.
    """
    runner = AblationRunner(random_seed=42)
    full_res = runner.run_variant(AblationConfig.full(seed=42), calibrated_workload)
    wo_ebpf_res = runner.run_variant(AblationConfig.without_ebpf(seed=42), calibrated_workload)

    assert wo_ebpf_res.recall < full_res.recall - 0.20, (
        f"Expected w/o eBPF recall ({wo_ebpf_res.recall:.3f}) to be significantly lower than Full ({full_res.recall:.3f})"
    )
    assert wo_ebpf_res.false_negatives > full_res.false_negatives


def test_ablation_without_hdc_increases_latency_and_memory(calibrated_workload: WorkloadBatch):
    """
    Verify w/o HDC:
    - Replaces bitwise HDC with dense floating-point representations.
    - Latency increases significantly (e.g. ~22ms vs ~2.1ms).
    - Memory footprint is substantially higher (e.g. 84MB vs 18.5MB).
    """
    runner = AblationRunner(random_seed=42)
    full_res = runner.run_variant(AblationConfig.full(seed=42), calibrated_workload)
    wo_hdc_res = runner.run_variant(AblationConfig.without_hdc(seed=42), calibrated_workload)

    assert wo_hdc_res.latency_ms > full_res.latency_ms * 5.0, (
        f"Expected w/o HDC latency ({wo_hdc_res.latency_ms:.2f}ms) to be > 5x Full ({full_res.latency_ms:.2f}ms)"
    )
    assert wo_hdc_res.memory_mb > full_res.memory_mb * 2.0


def test_ablation_vanilla_baseline(calibrated_workload: WorkloadBatch):
    """
    Verify Vanilla Baseline (all modern pillars disabled):
    - Demonstrates compound failure: lowest precision, lowest recall, lowest F1,
      severe safety violations, and slowest latency.
    """
    runner = AblationRunner(random_seed=42)
    vanilla_res = runner.run_variant(AblationConfig.vanilla(seed=42), calibrated_workload)

    assert vanilla_res.precision < 0.50
    assert vanilla_res.recall < 0.75
    assert vanilla_res.f1_score < 0.55
    assert vanilla_res.safety_violations > 0
    assert vanilla_res.mttr_seconds > 7.0
    assert vanilla_res.latency_ms > 20.0


# ==============================================================================
# 4. Modern LLM Comparison Simulator Tests
# ==============================================================================


def test_modern_llm_comparison_canonical_profiles():
    """
    Verify Modern LLM Comparison against GPT-4o, Claude 3.5 Sonnet, Llama-3-70B:
    1. Latency: LLM is 800 - 2500ms vs DeepLog-Infinity 2.1ms (400x - 1200x speedup).
    2. Cost: LLM is $0.015 - $0.05 per incident vs DeepLog-Infinity $0.00.
    3. Safety / Drift: LLM prompt drift 4% - 12% vs CBF 0.000% mathematical guarantee.
    4. Determinism: LLM is non-deterministic vs DeepLog provably stable Lyapunov certificate.
    """
    sim = ModernLLMComparisonSimulator(seed=42)
    results = sim.run_full_comparison(num_incidents=150)

    # 1. DeepLog-Infinity
    deeplog = results["DeepLog-Infinity"]
    assert deeplog.mean_latency_ms < 3.0
    assert deeplog.cost_per_incident_usd == 0.0
    assert deeplog.monthly_cost_100k_usd == 0.0
    assert deeplog.hallucination_rate == 0.000
    assert deeplog.safety_violation_rate == 0.000
    assert deeplog.is_deterministic is True
    assert deeplog.has_lyapunov_certificate is True
    assert deeplog.typical_mttr_seconds < 2.0

    # 2. GPT-4o
    gpt4o = results["GPT-4o"]
    assert 700.0 <= gpt4o.mean_latency_ms <= 1200.0
    assert 0.015 <= gpt4o.cost_per_incident_usd <= 0.05
    assert gpt4o.monthly_cost_100k_usd > 1000.0
    assert 0.02 <= gpt4o.hallucination_rate <= 0.10
    assert gpt4o.is_deterministic is False
    assert gpt4o.has_lyapunov_certificate is False
    assert gpt4o.speedup_vs_baseline > 300.0

    # 3. Claude-3.5-Sonnet
    claude = results["Claude-3.5-Sonnet"]
    assert 900.0 <= claude.mean_latency_ms <= 1500.0
    assert 0.010 <= claude.cost_per_incident_usd <= 0.05
    assert 0.01 <= claude.hallucination_rate <= 0.08
    assert claude.is_deterministic is False
    assert claude.has_lyapunov_certificate is False

    # 4. Llama-3-70B
    llama = results["Llama-3-70B"]
    assert 1200.0 <= llama.mean_latency_ms <= 2200.0
    assert 0.010 <= llama.cost_per_incident_usd <= 0.03
    assert 0.05 <= llama.hallucination_rate <= 0.20
    assert llama.is_deterministic is False
    assert llama.has_lyapunov_certificate is False


def test_modern_llm_comparison_custom_profile():
    """Verify simulator with custom model profile."""
    custom_profile = LLMModelProfile(
        name="Custom-SLM",
        vendor="On-Prem",
        mean_latency_ms=250.0,
        latency_std_ms=30.0,
        p95_latency_ms=320.0,
        p99_latency_ms=400.0,
        input_cost_per_million=1.00,
        output_cost_per_million=2.00,
        avg_input_tokens=1000,
        avg_output_tokens=200,
        prompt_drift_hallucination_rate=0.03,
        is_deterministic=False,
        safety_guarantee="None",
        has_lyapunov_certificate=False,
        typical_mttr_seconds=15.0,
    )
    sim = ModernLLMComparisonSimulator(
        profiles={
            "DeepLog-Infinity": ModernLLMComparisonSimulator().profiles["DeepLog-Infinity"],
            "Custom-SLM": custom_profile,
        },
        seed=123,
    )
    res = sim.simulate_model("Custom-SLM", num_incidents=50)
    assert res.model_name == "Custom-SLM"
    assert 200.0 <= res.mean_latency_ms <= 300.0
    assert res.cost_per_incident_usd > 0.0
    assert res.cost_savings_pct > 99.0


# ==============================================================================
# 5. Reporting & Serialization Tests
# ==============================================================================


def test_formatting_markdown_tables(calibrated_workload: WorkloadBatch):
    """Verify markdown table generators return valid markdown with expected columns."""
    runner = AblationRunner(random_seed=42)
    matrix = {
        "TEST_WORKLOAD": {
            "Full": runner.run_variant(AblationConfig.full(42), calibrated_workload),
            "w/o CBF": runner.run_variant(AblationConfig.without_cbf(42), calibrated_workload),
        }
    }
    md_table = format_ablation_markdown_table(matrix)
    assert "# DeepLog-Infinity: Architectural Ablation Matrix" in md_table
    assert "TEST_WORKLOAD" in md_table
    assert "DeepLog-Infinity (Full)" in md_table
    assert "w/o CBF" in md_table

    sim = ModernLLMComparisonSimulator(seed=42)
    llm_res = sim.run_full_comparison(num_incidents=10)
    llm_md = format_llm_comparison_markdown_table(llm_res)
    assert "# DeepLog-Infinity vs Modern LLMs: Empirical Comparison" in llm_md
    assert "GPT-4o" in llm_md
    assert "DeepLog-Infinity" in llm_md
    assert "PROVED (Lyapunov)" in llm_md


def test_export_ablation_matrix_json(calibrated_workload: WorkloadBatch):
    """Verify JSON export contains all required fields and parses properly."""
    runner = AblationRunner(random_seed=42)
    matrix = {
        "HDFS": {
            "Full": runner.run_variant(AblationConfig.full(42), calibrated_workload),
        }
    }
    sim = ModernLLMComparisonSimulator(seed=42)
    llm_res = sim.run_full_comparison(num_incidents=10)

    json_str = export_ablation_matrix_json(matrix, llm_res)
    parsed = json.loads(json_str)

    assert "timestamp" in parsed
    assert "ablation_matrix" in parsed
    assert "HDFS" in parsed["ablation_matrix"]
    assert "Full" in parsed["ablation_matrix"]["HDFS"]
    assert "llm_comparison" in parsed
    assert "DeepLog-Infinity" in parsed["llm_comparison"]
    assert "GPT-4o" in parsed["llm_comparison"]


def test_ablation_runner_run_full_matrix():
    """Verify run_full_matrix executes all canonical variants across default workloads."""
    runner = AblationRunner(random_seed=42)
    # Use smaller event count for fast test execution
    matrix = runner.run_full_matrix(num_events_per_workload=200)

    assert "HDFS" in matrix
    assert "BGL" in matrix
    assert "CHAOS" in matrix

    for w_name, var_dict in matrix.items():
        assert "DeepLog-Infinity (Full)" in var_dict
        assert "w/o PISNO" in var_dict
        assert "w/o CBF" in var_dict
        assert "w/o Consecutive Violations" in var_dict
        assert "w/o eBPF Telemetry" in var_dict
        assert "w/o HDC" in var_dict
        assert "Vanilla Baseline" in var_dict


# ==============================================================================
# 6. Benchmark CLI Integration Test
# ==============================================================================


def test_benchmark_cli_invocation():
    """Verify scripts/benchmark_ablation.py runs to completion via CLI."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_json = os.path.join(tmpdir, "test_out.json")
        out_md = os.path.join(tmpdir, "test_out.md")

        cmd = [
            sys.executable,
            "scripts/benchmark_ablation.py",
            "--workloads", "CHAOS",
            "--num-events", "250",
            "--llm-incidents", "20",
            "--output-json", out_json,
            "--output-md", out_md,
            "--seed", "99",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        assert proc.returncode == 0, f"Benchmark script failed: {proc.stderr}"
        assert "DeepLog-Infinity: Ablation Study Matrix" in proc.stdout
        assert "Empirical Benchmark & Ablation Study Completed Successfully" in proc.stdout

        assert os.path.exists(out_json)
        assert os.path.exists(out_md)

        with open(out_json, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert "CHAOS" in data["ablation_matrix"]
            assert "llm_comparison" in data
