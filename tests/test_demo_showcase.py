"""
Unit and Integration Tests for DeepLog Infinity Grand Showcase Demonstrator
==========================================================================
Validates:
1. CLI Argument Parsing and Configuration (`--headless`, `--steps`, `--mode`, `--speed`, `--no-color`, `--json`).
2. TerminalStyler ANSI formatting, banner generation, and fallback without color.
3. Programmatic Execution of each individual pillar (Pillars 1 through 7).
4. End-to-end `run_all()` pipeline execution in headless mode across all 7 pillars.
5. Selective pillar execution modes (`quick`, `ebpf`, `otel`, `chaos`, `world_model`, `safety`, `neuro`, `mesh`).
6. JSON telemetry export and schema integrity.
7. Subprocess CLI invocation and zero-error termination.
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

# Ensure scripts and src are on path
REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
SRC_DIR = REPO_ROOT / "src"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from demo_infinity_showcase import (
    DeepLogInfinityShowcase,
    TerminalStyler,
    main,
    parse_args,
)


# ==============================================================================
# 1. CLI Parsing & Styler Unit Tests
# ==============================================================================


class TestShowcaseCLIAndStyler:
    """Tests for CLI arguments parsing and terminal styling engine."""

    def test_default_args(self):
        """Verify default CLI arguments."""
        args = parse_args([])
        assert args.headless is False
        assert args.steps == 5
        assert args.mode == "full"
        assert args.speed == 1.0
        assert args.no_color is False
        assert args.json_output is None

    def test_custom_cli_arguments(self):
        """Verify custom CLI flags are correctly parsed."""
        args = parse_args([
            "--headless",
            "--steps", "8",
            "--mode", "quick",
            "--speed", "2.5",
            "--no-color",
            "--json", "/tmp/test_showcase.json",
        ])
        assert args.headless is True
        assert args.steps == 8
        assert args.mode == "quick"
        assert args.speed == 2.5
        assert args.no_color is True
        assert args.json_output == "/tmp/test_showcase.json"

    def test_invalid_mode_rejected(self):
        """Verify invalid mode choices raise SystemExit."""
        with pytest.raises(SystemExit):
            parse_args(["--mode", "invalid_future_mode"])

    def test_styler_with_color_enabled(self):
        """Verify styler returns ANSI escape sequences when enabled."""
        styler = TerminalStyler(enabled=True)
        assert "\033[31m" in styler.red("alert")
        assert "\033[32m" in styler.green("safe")
        assert "\033[36m" in styler.cyan("ebpf")
        assert "\033[1m" in styler.bold("header")
        assert "[" in styler.badge("OK", styler.green)

    def test_styler_with_color_disabled(self):
        """Verify styler strips ANSI formatting when disabled."""
        styler = TerminalStyler(enabled=False)
        assert styler.red("alert") == "alert"
        assert styler.green("safe") == "safe"
        assert styler.bold("header") == "header"
        assert styler.badge("OK", styler.green) == "[OK]"

    def test_banner_and_box_rendering(self):
        """Verify ASCII banner and pillar box generation."""
        styler = TerminalStyler(enabled=False)
        banner = styler.banner()
        assert "INFINITY GRAND SHOWCASE" in banner

        box = styler.pillar_box(1, "TEST TITLE", "Test Subtitle")
        assert "PILLAR 1/7: TEST TITLE" in box
        assert "Test Subtitle" in box

    def test_stat_bar_clamping(self):
        """Verify stat bar percentage formatting and ratio clamping."""
        styler = TerminalStyler(enabled=False)
        bar_50 = styler.stat_bar("Metric", 0.5, max_val=1.0, width=10)
        assert "50.0%" in bar_50
        assert "█████░░░░░" in bar_50

        # Clamping beyond 1.0
        bar_over = styler.stat_bar("Metric", 1.5, max_val=1.0, width=10)
        assert "100.0%" in bar_over


# ==============================================================================
# 2. Individual Pillar Execution Tests
# ==============================================================================


class TestIndividualPillars:
    """Verifies that each DeepLog pillar executes cleanly and returns valid telemetry."""

    @pytest.fixture
    def showcase(self) -> DeepLogInfinityShowcase:
        """Create a headless showcase instance for fast testing."""
        return DeepLogInfinityShowcase(headless=True, steps=3, no_color=True)

    def test_pillar_1_ebpf_hdc(self, showcase: DeepLogInfinityShowcase):
        """Pillar 1: In-kernel eBPF screening and 10,000-bit HDC filter."""
        res = showcase.run_pillar_1_ebpf_hdc()

        assert res["status"] == "SUCCESS"
        assert res["events_filtered"] >= 50
        assert res["filter_rate"] == 1.0  # Normal events 100% filtered
        assert res["avg_latency_us"] > 0.0
        assert res["anomalies_detected"] >= 1
        assert res["alert_score"] >= 0.35
        assert "HDC" in res["alert_reason"]
        assert "pillar_1_ebpf_hdc" in showcase.results["pillars_executed"]

    def test_pillar_2_otel_melt(self, showcase: DeepLogInfinityShowcase):
        """Pillar 2: Multi-modal OTel MELT fusion (Logs, Metrics, Traces, Profiles)."""
        res = showcase.run_pillar_2_otel_melt()

        assert res["status"] == "SUCCESS"
        assert 0.0 <= res["anomaly_score"] <= 1.0
        assert isinstance(res["primary_driver"], str)
        assert len(res["primary_driver"]) > 0

        # Contributions must sum to 1.0
        contribs = res["modal_contributions"]
        assert set(contribs.keys()) == {"logs", "metrics", "traces", "profiles"}
        assert sum(contribs.values()) == pytest.approx(1.0, abs=1e-4)
        assert "pillar_2_otel_melt" in showcase.results["pillars_executed"]

    def test_pillar_3_chaos_gym(self, showcase: DeepLogInfinityShowcase):
        """Pillar 3: Chaos Gym adversarial simulation (Red Team vs Blue Agent)."""
        res = showcase.run_pillar_3_chaos_gym()

        assert res["status"] == "SUCCESS"
        assert res["steps"] == showcase.steps
        assert 0.0 <= res["final_availability"] <= 1.0
        assert 0.0 <= res["final_blast_radius"] <= 1.0
        assert len(res["step_records"]) == showcase.steps
        assert "pillar_3_chaos_gym" in showcase.results["pillars_executed"]

    def test_pillar_4_world_model(self, showcase: DeepLogInfinityShowcase):
        """Pillar 4: Latent World Model imagination and pre-emptive mitigation."""
        res = showcase.run_pillar_4_world_model()

        assert res["status"] == "SUCCESS"
        assert res["action"] in [
            "MIGRATE_REPLICA",
            "EBPF_THROTTLE",
            "CIRCUIT_BREAKER_OPEN",
            "NO_ACTION",
        ]
        assert res["lead_time_seconds"] >= 30.0
        assert res["chosen_trajectory_risk"] < res["baseline_risk"]
        assert "pillar_4_world_model" in showcase.results["pillars_executed"]

    def test_pillar_5_formal_safety(self, showcase: DeepLogInfinityShowcase):
        """Pillar 5: CBF QP projection and Lyapunov asymptotic stability proof."""
        res = showcase.run_pillar_5_formal_safety()

        assert res["status"] == "SUCCESS"
        assert res["qp_status"] == "PROJECTED_SAFE"
        # Dangerous action u=-5.0 projected to safe boundary u*=-1.0
        assert res["u_proposed"] == -5.0
        assert res["u_projected"] >= -1.0 - 1e-4
        assert res["lyapunov_stable"] is True
        assert res["contraction_ratio"] < 1.0  # Asymptotic contraction
        assert "pillar_5_formal_safety" in showcase.results["pillars_executed"]

    def test_pillar_6_neuro_symbolic(self, showcase: DeepLogInfinityShowcase):
        """Pillar 6: Neuro-symbolic AST resolution and automated Git patch generation."""
        res = showcase.run_pillar_6_neuro_symbolic()

        assert res["status"] == "SUCCESS"
        assert "DataXceiver" in res["file_path"] or "DataXceiver" in res["target_class"]
        assert res["root_cause_category"] == "TIMEOUT_EXHAUSTION"
        assert res["syntax_valid"] is True
        assert res["patch_lines"] > 5
        assert "pillar_6_neuro_symbolic" in showcase.results["pillars_executed"]

    def test_pillar_7_federated_mesh(self, showcase: DeepLogInfinityShowcase):
        """Pillar 7: Zero-Knowledge federated threat sharing with differential privacy."""
        res = showcase.run_pillar_7_federated_mesh()

        assert res["status"] == "SUCCESS"
        assert res["dp_epsilon"] == 1.5
        assert 0.0 < res["dp_flip_prob"] < 0.5
        assert res["zk_proof_valid"] is True
        assert res["gossip_transfers"] >= 1
        assert "socket_exhaustion" in res["memory_bank_match_mode"]
        assert res["memory_bank_similarity"] > 0.35
        assert "pillar_7_federated_mesh" in showcase.results["pillars_executed"]

    def test_executive_summary(self, showcase: DeepLogInfinityShowcase):
        """Executive summary compilation and mission scorecard."""
        summary = showcase.run_executive_summary()

        assert summary["total_pillars"] == 7
        assert summary["mission_status"] == "SECURED"
        assert summary["resilience_score"] == 1.0
        assert showcase.results["status"] == "COMPLETED"


# ==============================================================================
# 3. End-to-End Showcase Modes & Integration Tests
# ==============================================================================


class TestShowcaseIntegration:
    """Tests end-to-end execution, different modes, JSON export, and CLI subprocess."""

    def test_run_all_full_mode(self):
        """Verify full mode executes all 7 pillars end-to-end."""
        showcase = DeepLogInfinityShowcase(headless=True, steps=3, mode="full", no_color=True)
        results = showcase.run_all()

        assert results["status"] == "COMPLETED"
        assert len(results["pillars_executed"]) == 7
        assert results["pillars_executed"] == [
            "pillar_1_ebpf_hdc",
            "pillar_2_otel_melt",
            "pillar_3_chaos_gym",
            "pillar_4_world_model",
            "pillar_5_formal_safety",
            "pillar_6_neuro_symbolic",
            "pillar_7_federated_mesh",
        ]
        assert "pillar_1" in results["metrics"]
        assert "pillar_2" in results["metrics"]
        assert "pillar_3" in results["metrics"]
        assert "pillar_4" in results["metrics"]
        assert "pillar_5" in results["metrics"]
        assert "pillar_6" in results["metrics"]
        assert "pillar_7" in results["metrics"]
        assert "executive_summary" in results["metrics"]
        assert results["total_elapsed_seconds"] > 0.0

    def test_run_quick_mode(self):
        """Verify quick mode runs all pillars and summary rapidly."""
        showcase = DeepLogInfinityShowcase(headless=True, steps=2, mode="quick", no_color=True)
        results = showcase.run_all()

        assert results["status"] == "COMPLETED"
        assert len(results["pillars_executed"]) == 7
        assert "executive_summary" in results["metrics"]

    @pytest.mark.parametrize(
        "mode_name,expected_pillar",
        [
            ("ebpf", "pillar_1_ebpf_hdc"),
            ("otel", "pillar_2_otel_melt"),
            ("chaos", "pillar_3_chaos_gym"),
            ("world_model", "pillar_4_world_model"),
            ("safety", "pillar_5_formal_safety"),
            ("neuro", "pillar_6_neuro_symbolic"),
            ("mesh", "pillar_7_federated_mesh"),
        ],
    )
    def test_single_pillar_modes(self, mode_name: str, expected_pillar: str):
        """Verify selective execution mode executes only the requested pillar."""
        showcase = DeepLogInfinityShowcase(headless=True, steps=2, mode=mode_name, no_color=True)
        results = showcase.run_all()

        assert len(results["pillars_executed"]) == 1
        assert results["pillars_executed"][0] == expected_pillar

    def test_json_telemetry_export(self, tmp_path: Path):
        """Verify JSON export creates a valid, parseable schema document."""
        out_json = tmp_path / "deep_showcase_results.json"
        showcase = DeepLogInfinityShowcase(
            headless=True,
            steps=2,
            mode="quick",
            no_color=True,
            json_output=str(out_json),
        )
        results = showcase.run_all()

        assert out_json.exists()
        content = json.loads(out_json.read_text(encoding="utf-8"))
        assert content["version"] == "1.0.0-infinity"
        assert content["status"] == "COMPLETED"
        assert len(content["pillars_executed"]) == 7
        assert "metrics" in content

    def test_main_cli_function(self):
        """Verify Python main() entry point returns exit code 0."""
        exit_code = main(["--headless", "--steps", "2", "--mode", "quick", "--no-color"])
        assert exit_code == 0

    def test_subprocess_cli_invocation(self):
        """Verify terminal execution of script via python executable returns exit code 0."""
        script_path = str(SCRIPTS_DIR / "demo_infinity_showcase.py")
        cmd = [
            sys.executable,
            script_path,
            "--headless",
            "--steps", "2",
            "--mode", "quick",
            "--no-color",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        assert proc.returncode == 0, f"Script failed with stderr:\n{proc.stderr}"
        assert "INFINITY GRAND SHOWCASE" in proc.stdout
        assert "100% END-TO-END AUTONOMOUS DEEP DEFENSE PIPELINE FULLY OPERATIONAL" in proc.stdout
