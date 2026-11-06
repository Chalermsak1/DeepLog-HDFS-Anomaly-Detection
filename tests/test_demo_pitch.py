"""
Unit and Integration Tests for DeepLog 3-Minute Live Exhibition Pitch & Streamlit UI
=====================================================================================
Validates:
1. CLI Argument Parsing and Flags (--auto, --duration, --step-delay, --headless, --export-json, --no-color, --stage).
2. PitchStyler ANSI engine, ASCII banner, stage boxes, narration prompts, and stat bar clamping.
3. Execution of individual stages 1 through 7:
   - Stage 1: Healthy cluster baseline (all green, nominal telemetry).
   - Stage 2: Chaos fault injection (packet drop, Byzantine delay, slowloris).
   - Stage 3: In-Kernel eBPF + HDC screening at 0.293 µs & PISNO detection <5ms.
   - Stage 4: QP-CBF formal safety filtering and Lyapunov contraction proof (<350ms).
   - Stage 5: Neuro-symbolic AST resolution & Git .patch unified diff synthesis.
   - Stage 6: ArgoCD GitOps deployment & automated health rollback.
   - Stage 7: Cluster restored to healthy status with quantified business ROI scorecard.
4. End-to-end `run_all()` pipeline execution in headless mode.
5. Selective stage execution filtering via `--stage`.
6. JSON telemetry export, schema integrity, and scorecard fields.
7. Subprocess CLI invocation and zero-error termination.
8. Streamlit IPITEx 2027 live tour helper functions in app.py:
   - `get_ipitex_tour_stages()`
   - `get_ipitex_stage_data()`
   - `get_ipitex_scorecard()`
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest

# Ensure scripts, src, and repo root are on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
SRC_DIR = REPO_ROOT / "src"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo_pitch_3min import (
    DeepLogPitchPresenter,
    PitchStyler,
    main,
    parse_args,
)
import app


# ==============================================================================
# 1. CLI Parsing and Terminal Styler Tests
# ==============================================================================


class TestPitchCLIAndStyler:
    """Verifies command line interface argument parsing and ANSI styling engine."""

    def test_default_cli_arguments(self):
        """Verify default CLI arguments."""
        args = parse_args([])
        assert args.headless is False
        assert args.auto is False
        assert args.duration == 180.0
        assert args.step_delay is None
        assert args.export_json is None
        assert args.no_color is False
        assert args.stage == 0

    def test_custom_cli_arguments(self):
        """Verify custom CLI flags are correctly parsed."""
        args = parse_args([
            "--auto",
            "--duration", "30.0",
            "--step-delay", "0.5",
            "--headless",
            "--export-json", "/tmp/pitch_test.json",
            "--no-color",
            "--stage", "3",
        ])
        assert args.auto is True
        assert args.duration == 30.0
        assert args.step_delay == 0.5
        assert args.headless is True
        assert args.export_json == "/tmp/pitch_test.json"
        assert args.no_color is True
        assert args.stage == 3

    def test_json_alias_argument(self):
        """Verify --json alias works the same as --export-json."""
        args = parse_args(["--json", "/tmp/alias.json"])
        assert args.export_json == "/tmp/alias.json"

    def test_invalid_stage_rejected(self):
        """Verify out-of-bounds stage raises SystemExit."""
        with pytest.raises(SystemExit):
            parse_args(["--stage", "99"])

    def test_styler_with_color_enabled(self):
        """Verify styler returns ANSI sequences when enabled."""
        styler = PitchStyler(enabled=True)
        assert "\033[31m" in styler.red("alert")
        assert "\033[32m" in styler.green("safe")
        assert "\033[33m" in styler.yellow("warn")
        assert "\033[36m" in styler.cyan("ebpf")
        assert "\033[1m" in styler.bold("headline")
        assert "[" in styler.badge("OK", styler.green)

    def test_styler_with_color_disabled(self):
        """Verify styler returns clean plaintext when color is disabled."""
        styler = PitchStyler(enabled=False)
        assert styler.red("alert") == "alert"
        assert styler.green("safe") == "safe"
        assert styler.yellow("warn") == "warn"
        assert styler.bold("headline") == "headline"
        assert styler.badge("OK", styler.green) == "[OK]"

    def test_banner_and_stage_box_generation(self):
        """Verify ASCII banners and stage boxes render cleanly."""
        styler = PitchStyler(enabled=False)
        banner = styler.banner()
        assert "IPITEx 2027 OFFICIAL 3-MINUTE LIVE EXHIBITION PITCH" in banner
        assert "Patent Pending DIP 2027" in banner

        box = styler.stage_box(1, 7, "TEST STAGE", "Test Subtitle", styler.green)
        assert "STAGE 1/7: TEST STAGE" in box
        assert "Test Subtitle" in box

    def test_narration_box_rendering(self):
        """Verify narration script box renders timing and text."""
        styler = PitchStyler(enabled=False)
        narr = styler.narration_box("0:00 - 0:25", "Welcome honorable judges.")
        assert "0:00 - 0:25" in narr
        assert "Welcome honorable judges." in narr

    def test_stat_bar_clamping(self):
        """Verify stat bar percentage calculations and ratio clamping."""
        styler = PitchStyler(enabled=False)
        bar_50 = styler.stat_bar("CPU", 50.0, max_val=100.0, width=10, unit="%")
        assert "50.00%" in bar_50
        assert "█████░░░░░" in bar_50

        # Clamping above max_val
        bar_over = styler.stat_bar("CPU", 150.0, max_val=100.0, width=10, unit="%")
        assert "██████████" in bar_over


# ==============================================================================
# 2. Individual Stage Execution Tests
# ==============================================================================


class TestIndividualStages:
    """Verifies that each of the 7 stages executes cleanly with deterministic telemetry."""

    @pytest.fixture
    def presenter(self) -> DeepLogPitchPresenter:
        """Create a headless presenter instance for instant execution."""
        return DeepLogPitchPresenter(headless=True, no_color=True)

    def test_stage_1_healthy_baseline(self, presenter: DeepLogPitchPresenter):
        """Stage 1: Healthy cluster baseline."""
        res = presenter.run_stage_1_healthy_baseline()
        assert res["stage_id"] == 1
        assert res["status"] == "NOMINAL_GREEN"
        assert len(res["nodes"]) == 4
        assert all(n["status"] == "HEALTHY" for n in res["nodes"])
        assert res["metrics"]["cluster_cpu_pct"] < 30.0
        assert res["metrics"]["p99_latency_ms"] < 2.0
        assert res["metrics"]["anomaly_score"] < 0.10
        assert "5 / 5" in res["metrics"]["invariants_satisfied"]
        assert "stage_1" in presenter.results["stages"]

    def test_stage_2_chaos_injection(self, presenter: DeepLogPitchPresenter):
        """Stage 2: Chaos fault injection (packet drop, Byzantine delay, slowloris)."""
        res = presenter.run_stage_2_chaos_injection()
        assert res["stage_id"] == 2
        assert res["status"] == "WARNING_AMBER"
        assert len(res["faults_injected"]) == 3
        fault_types = [f["type"] for f in res["faults_injected"]]
        assert "BYZANTINE_RPC_DELAY" in fault_types
        assert "PACKET_DROP" in fault_types
        assert "SLOWLORIS_STARVATION" in fault_types
        assert res["metrics"]["p99_latency_ms"] > 400.0  # Spiked from 1.15ms
        assert res["metrics"]["affected_node"] == "node-charlie"
        assert "stage_2" in presenter.results["stages"]

    def test_stage_3_ebpf_hdc_screening(self, presenter: DeepLogPitchPresenter):
        """Stage 3: In-Kernel eBPF + HDC screening at 0.293 µs & PISNO detection <5ms."""
        res = presenter.run_stage_3_ebpf_hdc_screening()
        assert res["stage_id"] == 3
        assert res["status"] == "INCIDENT_RED"
        assert res["ebpf_telemetry"]["screening_latency_us"] == 0.293
        assert res["ebpf_telemetry"]["hdc_vector_dimension"] == 10000
        assert res["ebpf_telemetry"]["filter_rate_pct"] > 99.0
        assert res["pisno_detection"]["detection_latency_ms"] < 5.0
        assert res["pisno_detection"]["anomaly_score"] > 0.85
        assert res["pisno_detection"]["consecutive_breaches"] >= 3
        assert "stage_3" in presenter.results["stages"]

    def test_stage_4_qp_cbf_safety(self, presenter: DeepLogPitchPresenter):
        """Stage 4: QP-CBF formal safety filtering and Lyapunov contraction proof (<350ms)."""
        res = presenter.run_stage_4_qp_cbf_safety()
        assert res["stage_id"] == 4
        assert res["status"] == "SAFETY_PROVEN"
        assert "KILL_NODE" in res["proposed_raw_action"]
        assert "REJECTED_UNSAFE" in res["cbf_verdict"]
        assert res["qp_filter"]["solve_time_ms"] < 350.0  # 118.4 ms
        assert res["qp_filter"]["satisfied"] is True
        assert res["qp_filter"]["h_x_barrier_margin"] > 0.0  # h(x) >= 0 verified
        assert res["lyapunov_proof"]["is_stable"] is True
        assert res["lyapunov_proof"]["contraction_delta_v"] < 0.0  # ΔV <= -κV
        assert "0x" in res["lyapunov_proof"]["proof_hash"]
        assert "Zero-Hallucination" in res["safety_badge"]
        assert "stage_4" in presenter.results["stages"]

    def test_stage_5_neuro_symbolic_patch(self, presenter: DeepLogPitchPresenter):
        """Stage 5: Neuro-symbolic AST resolution & Git .patch unified diff synthesis."""
        res = presenter.run_stage_5_neuro_symbolic_patch()
        assert res["stage_id"] == 5
        assert res["status"] == "PATCH_SYNTHESIZED"
        assert "DataXceiver.java" in res["root_cause_analysis"]["target_file"]
        assert res["root_cause_analysis"]["target_line"] == 142
        assert res["patch_synthesis"]["synthesis_latency_ms"] < 300.0  # 210.5 ms
        assert res["patch_synthesis"]["ast_syntax_valid"] is True
        assert "--- a/DataXceiver.java" in res["patch_synthesis"]["unified_diff"]
        assert "+++ b/DataXceiver.java" in res["patch_synthesis"]["unified_diff"]
        assert "JUnit 5" in res["patch_synthesis"]["junit5_test"]
        assert "PASSED" in res["patch_synthesis"]["dry_run_validation"]
        assert "stage_5" in presenter.results["stages"]

    def test_stage_6_argocd_gitops_deploy(self, presenter: DeepLogPitchPresenter):
        """Stage 6: ArgoCD GitOps deployment & automated health rollback."""
        res = presenter.run_stage_6_argocd_gitops_deploy()
        assert res["stage_id"] == 6
        assert res["status"] == "DEPLOYED_SYNCED"
        assert res["gitops"]["pull_request"] == "#142"
        assert res["gitops"]["sync_state"] == "Synced"
        assert res["gitops"]["health_state"] == "Healthy"
        assert res["gitops"]["sync_duration_sec"] < 2.0  # 1.42s
        assert res["ebpf_enforcement"]["active_enforcement"] is True
        assert res["ebpf_enforcement"]["quarantined_conns"] > 0
        assert "stage_6" in presenter.results["stages"]

    def test_stage_7_restoration_roi(self, presenter: DeepLogPitchPresenter):
        """Stage 7: Cluster restored to healthy status with quantified business ROI scorecard."""
        res = presenter.run_stage_7_restoration_roi()
        assert res["stage_id"] == 7
        assert res["status"] == "ALL_GREEN_RESTORED"
        scorecard = res["scorecard"]
        assert scorecard["mttd_detection_time_ms"] < 5.0
        assert scorecard["mttp_safety_policy_time_ms"] < 350.0
        assert scorecard["mttr_total_remediation_time_sec"] < 3.0  # 2.85s
        assert scorecard["mttr_reduction_pct"] > 99.0
        assert scorecard["annual_roi_per_100_nodes_usd"] >= 400000
        assert scorecard["uptime_achieved_pct"] >= 99.9
        assert any("Patent Pending DIP" in b for b in scorecard["badges"])
        assert any("Zero-Hallucination" in b for b in scorecard["badges"])
        assert scorecard["final_telemetry"]["cluster_status"] == "STABLE_GREEN"
        assert "stage_7" in presenter.results["stages"]


# ==============================================================================
# 3. Pipeline Integration & JSON Export Tests
# ==============================================================================


class TestPitchIntegration:
    """Verifies end-to-end presentation flow, stage filtering, and JSON export."""

    def test_run_all_end_to_end_headless(self):
        """Verify full 7-stage presentation executes end-to-end in headless mode."""
        presenter = DeepLogPitchPresenter(headless=True, no_color=True)
        results = presenter.run_all()

        assert results["status"] == "COMPLETED"
        assert results["actual_duration_sec"] >= 0.0
        assert len(results["stages"]) == 7
        for stage_i in range(1, 8):
            assert f"stage_{stage_i}" in results["stages"]
        assert "scorecard" in results
        assert results["scorecard"]["mttr_total_remediation_time_sec"] < 3.0

    def test_stage_filtering(self):
        """Verify running a single filtered stage only runs that stage."""
        presenter = DeepLogPitchPresenter(headless=True, stage_filter=4, no_color=True)
        results = presenter.run_all()

        assert results["status"] == "COMPLETED"
        assert len(results["stages"]) == 1
        assert "stage_4" in results["stages"]
        assert "stage_1" not in results["stages"]

    def test_json_telemetry_export(self):
        """Verify JSON export creates a valid, parseable telemetry file with all stages."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
            json_path = tf.name

        try:
            presenter = DeepLogPitchPresenter(
                headless=True,
                export_json=json_path,
                no_color=True,
            )
            presenter.run_all()

            assert os.path.exists(json_path)
            assert os.path.getsize(json_path) > 100

            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            assert data["demo_name"] == "DeepLog-Omni IPITEx 2027 3-Minute Pitch"
            assert data["status"] == "COMPLETED"
            assert len(data["stages"]) == 7
            assert "scorecard" in data
            assert data["scorecard"]["annual_roi_per_100_nodes_usd"] == 480000
            assert "Patent Pending DIP 2027" in data["scorecard"]["badges"]
        finally:
            if os.path.exists(json_path):
                os.remove(json_path)

    def test_main_cli_function(self):
        """Verify main() function entrypoint returns exit code 0."""
        rc = main(["--headless", "--no-color", "--duration", "30.0", "--auto"])
        assert rc == 0

    def test_subprocess_cli_invocation(self):
        """Verify running scripts/demo_pitch_3min.py via python subprocess."""
        cmd = [
            sys.executable,
            str(SCRIPTS_DIR / "demo_pitch_3min.py"),
            "--headless",
            "--no-color",
            "--step-delay", "0.0",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        assert proc.returncode == 0
        assert "IPITEx 2027 OFFICIAL 3-MINUTE LIVE EXHIBITION PITCH" in proc.stdout
        assert "STAGE 1/7" in proc.stdout
        assert "STAGE 7/7" in proc.stdout
        assert "QUANTIFIED BUSINESS ROI SCORECARD" in proc.stdout


# ==============================================================================
# 4. Streamlit UI App IPITEx Integration Tests
# ==============================================================================


class TestStreamlitIPITExIntegration:
    """Verifies that app.py exposes required IPITEx 2027 tour data structures and helpers."""

    def test_ipitex_tour_stages_structure(self):
        """Verify get_ipitex_tour_stages returns 4 synchronized stages with valid keys."""
        stages = app.get_ipitex_tour_stages()
        assert len(stages) == 4

        # Verify Stage 1
        st1 = stages[0]
        assert st1["stage_id"] == 1
        assert st1["status_label"] == "NOMINAL GREEN"
        assert st1["color_hex"] == "#10b981"
        assert len(st1["nodes"]) == 4
        assert st1["metrics"]["cpu_pct"] < 30.0
        assert st1["metrics"]["latency_p99_ms"] < 5.0
        assert st1["metrics"]["anomaly_score"] < 0.10

        # Verify Stage 2
        st2 = stages[1]
        assert st2["stage_id"] == 2
        assert st2["status_label"] == "WARNING AMBER"
        assert st2["color_hex"] == "#f59e0b"
        assert st2["metrics"]["latency_p99_ms"] > 400.0
        assert any(n["status"] == "DEGRADED" for n in st2["nodes"])

        # Verify Stage 3
        st3 = stages[2]
        assert st3["stage_id"] == 3
        assert st3["status_label"] == "INCIDENT RED"
        assert st3["color_hex"] == "#ef4444"
        assert st3["metrics"]["anomaly_score"] > 0.85
        assert st3["metrics"]["ebpf_screening_us"] == 0.293
        assert any(n["status"] == "QUARANTINED" for n in st3["nodes"])

        # Verify Stage 4
        st4 = stages[3]
        assert st4["stage_id"] == 4
        assert st4["status_label"] == "STABLE GREEN"
        assert st4["color_hex"] == "#10b981"
        assert st4["metrics"]["anomaly_score"] < 0.10
        assert st4["metrics"]["latency_p99_ms"] < 5.0
        assert all(n["status"] in ("HEALTHY", "RESTORED") for n in st4["nodes"])

    def test_ipitex_stage_data_bounds(self):
        """Verify get_ipitex_stage_data clamps boundary indices correctly."""
        st_first = app.get_ipitex_stage_data(1)
        assert st_first["stage_id"] == 1

        st_last = app.get_ipitex_stage_data(4)
        assert st_last["stage_id"] == 4

        # Out-of-bounds clamped to valid range
        st_under = app.get_ipitex_stage_data(-5)
        assert st_under["stage_id"] == 1

        st_over = app.get_ipitex_stage_data(99)
        assert st_over["stage_id"] == 4

    def test_ipitex_scorecard_metrics(self):
        """Verify get_ipitex_scorecard returns complete business ROI metrics."""
        scorecard = app.get_ipitex_scorecard()
        assert scorecard["mttd_ms"] == 4.82
        assert scorecard["mttp_ms"] == 118.4
        assert scorecard["mttpatch_ms"] == 210.5
        assert scorecard["mttr_sec"] == 2.85
        assert scorecard["mttr_reduction_pct"] > 99.0
        assert scorecard["uptime_pct"] >= 99.9
        assert scorecard["annual_savings_usd"] == 480000
        assert scorecard["false_alert_reduction_pct"] == 99.4
        assert "0x" in scorecard["proof_hash"]

        badges = scorecard["badges"]
        assert "Patent Pending DIP 2027" in badges
        assert "99.9% Uptime" in badges
        assert "Zero-Hallucination Safe Remediator" in badges
        assert "IPITEx 2027 Gold Medal Showcase" in badges
