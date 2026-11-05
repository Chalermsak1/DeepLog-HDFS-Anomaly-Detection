"""
DeepLog GitOps & SLM Dataset Curator Test Suite
================================================
Comprehensive test suite verifying:
- GitOpsManager branch lifecycle, staging, commits, and unified diff patch application.
- Invalid diff handling, context mismatch detection, and dry-run patch verification.
- Pre-merge validation test harness (Python AST syntax, custom validators).
- GitPRSpec markdown rendering, Mermaid sequence diagram syntax, and CBF certificates.
- GitHubWebhookEmulator HMAC-SHA256 signature verification and PR merge flows.
- ArgoCDSyncEngine continuous deployment synchronization, health checks, and automated rollbacks.
- SLMDatasetCurator instruction dataset formatting (Alpaca, ShareGPT, HuggingFace),
  schema validation, dataset splitting, JSONL export, and CLI script execution.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import pytest

from deeplog.gitops import (
    GitOpsManager,
    GitPRSpec,
    GitHubWebhookEmulator,
    ArgoCDSyncEngine,
    SLMDatasetCurator,
    PRStatus,
    SyncStatus,
    HealthStatus,
    SLMDatasetFormat,
    GitCommit,
    GitStatus,
    PatchResult,
    ValidationResult,
    ASTBlameProvenance,
    CBFSafetyCertificate,
    WebhookEvent,
    parse_unified_diff,
    apply_hunks_to_content,
)


# ==============================================================================
# 1. Tests for GitOpsManager & Unified Patch Engine
# ==============================================================================


class TestGitOpsManager:
    """Test suite for GitOpsManager core Git operations and unified diff patching."""

    @pytest.fixture
    def manager(self):
        gm = GitOpsManager()
        yield gm
        gm.cleanup()

    def test_repo_initialization_and_branching(self, manager: GitOpsManager):
        assert manager.get_current_branch() == "main"
        assert manager.list_branches() == ["main"]

        # Create new branch
        b = manager.create_branch("feature/socket-timeout-fix", checkout=True)
        assert b.name == "feature/socket-timeout-fix"
        assert manager.get_current_branch() == "feature/socket-timeout-fix"
        assert "feature/socket-timeout-fix" in manager.list_branches()

        # Switch back to main
        manager.checkout("main")
        assert manager.get_current_branch() == "main"

        # Duplicate branch creation raises ValueError
        with pytest.raises(ValueError, match="already exists"):
            manager.create_branch("feature/socket-timeout-fix")

        # Checkout non-existent branch raises ValueError
        with pytest.raises(ValueError, match="does not exist"):
            manager.checkout("non-existent-branch")

    def test_file_status_staging_and_commits(self, manager: GitOpsManager):
        # Initial status is clean
        status = manager.status()
        assert status.is_clean
        assert len(status.untracked) == 0

        # Create a file on disk
        test_file = manager.repo_dir / "service.py"
        test_file.write_text("def serve():\n    return 200\n", encoding="utf-8")

        status = manager.status()
        assert not status.is_clean
        assert "service.py" in status.untracked

        # Stage the file
        manager.stage("service.py")
        status = manager.status()
        assert "service.py" in status.staged

        # Unstage
        manager.unstage("service.py")
        status = manager.status()
        assert "service.py" not in status.staged

        # Commit
        manager.stage_all()
        commit = manager.commit(message="feat: initial service implementation")
        assert isinstance(commit, GitCommit)
        assert commit.commit_sha is not None
        assert commit.parent_sha is None
        assert "service.py" in commit.files_changed

        # Post-commit status should be clean
        status = manager.status()
        assert status.is_clean

        # Modify file
        test_file.write_text("def serve():\n    return 204\n", encoding="utf-8")
        status = manager.status()
        assert "service.py" in status.modified

        # Second commit
        manager.stage_all()
        commit2 = manager.commit(message="fix: return 204 No Content")
        assert commit2.parent_sha == commit.commit_sha

        # Inspect log history
        log = manager.get_log()
        assert len(log) == 2
        assert log[0].commit_sha == commit2.commit_sha
        assert log[1].commit_sha == commit.commit_sha

    def test_revert_last_commit(self, manager: GitOpsManager):
        test_file = manager.repo_dir / "config.json"
        test_file.write_text('{"v": 1}', encoding="utf-8")
        manager.commit("Initial v1")

        test_file.write_text('{"v": 2}', encoding="utf-8")
        manager.commit("Update to v2")
        assert test_file.read_text(encoding="utf-8") == '{"v": 2}'

        # Revert to parent
        reverted = manager.revert_last_commit()
        assert reverted is not None
        assert test_file.read_text(encoding="utf-8") == '{"v": 1}'

    def test_branch_deletion_rules(self, manager: GitOpsManager):
        manager.create_branch("temp-branch", checkout=False)
        assert "temp-branch" in manager.list_branches()

        # Cannot delete active branch
        manager.checkout("temp-branch")
        with pytest.raises(ValueError, match="Cannot delete active branch"):
            manager.delete_branch("temp-branch")

        manager.checkout("main")
        assert manager.delete_branch("temp-branch") is True
        assert "temp-branch" not in manager.list_branches()

    def test_unified_diff_patch_application(self, manager: GitOpsManager):
        # Create target file
        orig_code = (
            "import time\n"
            "\n"
            "def handle_connection(sock):\n"
            "    sock.settimeout(1.0)\n"
            "    return sock.recv(1024)\n"
        )
        file_path = manager.repo_dir / "network.py"
        file_path.write_text(orig_code, encoding="utf-8")
        manager.commit("Initial network implementation")

        patch_diff = (
            "--- a/network.py\n"
            "+++ b/network.py\n"
            "@@ -3,3 +3,5 @@\n"
            " def handle_connection(sock):\n"
            "-    sock.settimeout(1.0)\n"
            "+    # DeepLog patch: exponential backoff\n"
            "+    sock.settimeout(15.0)\n"
            "     return sock.recv(1024)\n"
        )

        res = manager.apply_patch(patch_diff)
        assert res.success is True
        assert res.target_file == "network.py"
        assert res.applied_hunks == 1

        patched = file_path.read_text(encoding="utf-8")
        assert "sock.settimeout(15.0)" in patched
        assert "sock.settimeout(1.0)" not in patched

        # Test reverse patch application (rollback patch)
        rev_res = manager.apply_patch(patch_diff, reverse=True)
        assert rev_res.success is True
        restored = file_path.read_text(encoding="utf-8")
        assert "sock.settimeout(1.0)" in restored

    def test_dry_run_patch_verification(self, manager: GitOpsManager):
        orig_code = "line1\nline2\nline3\n"
        file_path = manager.repo_dir / "target.txt"
        file_path.write_text(orig_code, encoding="utf-8")

        patch_diff = (
            "--- a/target.txt\n"
            "+++ b/target.txt\n"
            "@@ -1,3 +1,3 @@\n"
            " line1\n"
            "-line2\n"
            "+line2_modified\n"
            " line3\n"
        )

        # Dry run must succeed without modifying disk
        res = manager.dry_run_patch(patch_diff)
        assert res.success is True
        assert res.dry_run is True
        assert file_path.read_text(encoding="utf-8") == orig_code

    def test_invalid_diff_edge_cases(self, manager: GitOpsManager):
        # 1. Empty patch
        res = manager.apply_patch("")
        assert res.success is False
        assert "Empty patch" in str(res.error)

        # 2. Corrupt hunk header
        corrupt_diff = (
            "--- a/file.py\n"
            "+++ b/file.py\n"
            "@@ not_a_valid_hunk @@\n"
            "+bad line\n"
        )
        res = manager.apply_patch(corrupt_diff)
        assert res.success is False
        assert "Malformed" in str(res.error)

        # 3. Context mismatch
        file_path = manager.repo_dir / "app.py"
        file_path.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
        mismatched_diff = (
            "--- a/app.py\n"
            "+++ b/app.py\n"
            "@@ -1,3 +1,3 @@\n"
            " non_matching_line_one\n"
            "-beta\n"
            "+beta_new\n"
            " gamma\n"
        )
        res = manager.apply_patch(mismatched_diff)
        assert res.success is False
        assert "mismatch" in str(res.error).lower()
        # Ensure file was not corrupted
        assert file_path.read_text(encoding="utf-8") == "alpha\nbeta\ngamma\n"

    def test_validation_test_harness(self, manager: GitOpsManager):
        # Valid Python code
        (manager.repo_dir / "valid.py").write_text("x = 42\ndef get_x():\n    return x\n", encoding="utf-8")
        val = manager.run_validation_suite(python_syntax_check=True)
        assert val.passed is True
        assert val.failed_count == 0

        # Syntax error
        (manager.repo_dir / "broken.py").write_text("def broken_syntax(:\n", encoding="utf-8")
        val2 = manager.run_validation_suite(python_syntax_check=True)
        assert val2.passed is False
        assert val2.failed_count >= 1
        assert any("SyntaxError" in issue for issue in val2.issues)

        # Remove broken file
        (manager.repo_dir / "broken.py").unlink()

        # Custom callable validator
        val3 = manager.run_validation_suite(
            python_syntax_check=True,
            custom_validator=lambda: True,
        )
        assert val3.passed is True

        val4 = manager.run_validation_suite(
            python_syntax_check=True,
            custom_validator=lambda: False,
        )
        assert val4.passed is False

    def test_branch_conflict_detection(self, manager: GitOpsManager):
        shared_file = manager.repo_dir / "config.yaml"
        shared_file.write_text("timeout: 10\nretries: 3\n", encoding="utf-8")
        manager.commit("Initial shared config")

        # Branch A
        manager.create_branch("branch-a", checkout=True)
        shared_file.write_text("timeout: 99\nretries: 3\n", encoding="utf-8")
        manager.commit("Branch A timeout edit")

        # Branch B
        manager.checkout("main")
        manager.create_branch("branch-b", checkout=True)
        shared_file.write_text("timeout: 5\nretries: 3\n", encoding="utf-8")
        manager.commit("Branch B timeout edit")

        conflicts = manager.detect_branch_conflict("branch-a", "branch-b")
        assert "config.yaml" in conflicts


# ==============================================================================
# 2. Tests for GitPRSpec & Provenance Descriptors
# ==============================================================================


class TestGitPRSpec:
    """Test suite for GitPRSpec rich Pull Request synthesis and formatting."""

    @pytest.fixture
    def sample_pr_spec(self) -> GitPRSpec:
        blame = ASTBlameProvenance(
            commit_sha="e5f1b2c3",
            author="Sarah Connor <sconnor@cyberdyne.org>",
            commit_date="2026-11-04T10:00:00Z",
            commit_message="Decrease socket timeout to 500ms",
            file_path="org/apache/hadoop/hdfs/server/datanode/DataXceiver.java",
            line_number=248,
            class_name="DataXceiver",
            method_name="writeBlock",
            exception_type="java.net.SocketTimeoutException",
        )
        safety_cert = CBFSafetyCertificate(
            is_safe=True,
            margin=0.2850,
            invariants_evaluated=[
                {
                    "name": "cluster_cpu_headroom >= 0.15",
                    "threshold": 0.15,
                    "actual": 0.435,
                    "margin": 0.285,
                    "satisfied": True,
                }
            ],
            lie_derivatives={"L_f_h": 0.08, "L_g_h": 0.04},
            lyapunov_drift=-0.145,
            lyapunov_stable=True,
            proof_summary="CBF barrier condition dot{h}(x) + alpha(h(x)) >= 0 verified analytically.",
        )
        patch = (
            "--- a/DataXceiver.java\n"
            "+++ b/DataXceiver.java\n"
            "@@ -248,1 +248,2 @@\n"
            "-    socket.setSoTimeout(500);\n"
            "+    socket.setSoTimeout(15000);\n"
        )
        return GitPRSpec(
            pr_id="PR-2026-1042",
            title="fix(datanode): increase socket timeout to mitigate premature termination",
            branch_name="deeplog/remediation-E14",
            target_file="org/apache/hadoop/hdfs/server/datanode/DataXceiver.java",
            summary="Autonomous remediation for recurrent socket timeout spikes during block replication.",
            patch_diff=patch,
            ast_blame=blame,
            safety_certificate=safety_cert,
        )

    def test_mermaid_sequence_diagram_generation(self, sample_pr_spec: GitPRSpec):
        diagram = sample_pr_spec.generate_mermaid_sequence_diagram()
        assert diagram.startswith("sequenceDiagram")
        assert "actor SRE as SRE Operator" in diagram
        assert "participant D as DeepLog Detector" in diagram
        assert "participant CBF as CBF Safety Verifier" in diagram
        assert "participant Argo as ArgoCD Sync Engine" in diagram
        assert "alt Invariant Violation / Degraded" in diagram
        assert "end" in diagram

    def test_suggested_reviewers_derivation(self, sample_pr_spec: GitPRSpec):
        reviewers = sample_pr_spec.suggested_reviewers
        assert any("sarah-connor" in r for r in reviewers)
        assert "@hdfs-core-team" in reviewers
        assert "@deeplog-ai-guard" in reviewers

    def test_markdown_rendering_and_saving(self, sample_pr_spec: GitPRSpec, tmp_path: Path):
        md = sample_pr_spec.to_markdown()
        assert f"# {sample_pr_spec.title}" in md
        assert "```mermaid" in md
        assert "## 3. AST Blame & Provenance" in md
        assert "Sarah Connor" in md
        assert "## 4. Formal CBF Safety Certificate" in md
        assert "0.2850" in md
        assert "```diff" in md
        assert "+    socket.setSoTimeout(15000);" in md

        # Save to disk
        out_file = tmp_path / "PR_SUMMARY.md"
        saved_path = sample_pr_spec.save_markdown(out_file)
        assert saved_path.exists()
        assert saved_path.read_text(encoding="utf-8") == md

    def test_serialization_round_trip(self, sample_pr_spec: GitPRSpec):
        d = sample_pr_spec.to_dict()
        assert d["pr_id"] == "PR-2026-1042"
        assert d["ast_blame"]["author"] == "Sarah Connor <sconnor@cyberdyne.org>"
        assert d["safety_certificate"]["margin"] == 0.2850

        spec2 = GitPRSpec.from_dict(d)
        assert spec2.pr_id == sample_pr_spec.pr_id
        assert spec2.title == sample_pr_spec.title
        assert spec2.ast_blame.line_number == 248
        assert spec2.safety_certificate.is_safe is True


# ==============================================================================
# 3. Tests for GitHubWebhookEmulator
# ==============================================================================


class TestGitHubWebhookEmulator:
    """Test suite for GitHubWebhookEmulator webhook dispatching and mock API client."""

    @pytest.fixture
    def emulator(self):
        return GitHubWebhookEmulator(default_secret="test-secret-123")

    def test_webhook_subscription_and_hmac_signature(self, emulator: GitHubWebhookEmulator):
        received_events: List[WebhookEvent] = []

        sub_id = emulator.register_webhook(
            events=["pull_request", "push"],
            callback=lambda e: received_events.append(e),
            secret="my-custom-secret",
        )
        assert sub_id is not None

        # Dispatch push event
        deliveries = emulator.dispatch_event(
            event_type="push",
            payload={"ref": "refs/heads/main", "commit": "123456"},
        )
        assert len(deliveries) == 1
        assert deliveries[0].success is True
        assert len(received_events) == 1

        ev = received_events[0]
        assert ev.event_type == "push"
        # Validate HMAC signature
        payload_bytes = json.dumps(ev.payload, sort_keys=True).encode("utf-8")
        assert emulator.verify_signature(payload_bytes, ev.signature, "my-custom-secret") is True
        assert emulator.verify_signature(payload_bytes, ev.signature, "wrong-secret") is False

        # Unregister
        assert emulator.unregister_webhook(sub_id) is True
        emulator.dispatch_event("push", {"ref": "refs/heads/main"})
        assert len(received_events) == 1  # No new event received

    def test_mock_pr_lifecycle_and_merge(self, emulator: GitHubWebhookEmulator):
        spec = GitPRSpec(
            pr_id="PR-99",
            title="Autonomous hotfix for connection pool leak",
            branch_name="deeplog/hotfix-leak",
            target_file="src/pool.py",
            summary="Hotfix connection pool leak",
            patch_diff="--- a/pool.py\n+++ b/pool.py\n@@ -1 +1 @@\n-old\n+new\n",
        )

        # 1. Create PR
        pr = emulator.create_pull_request(spec)
        assert pr["number"] == 1
        assert pr["state"] == "open"
        assert pr["merged"] is False

        # 2. Add review comments
        c1 = emulator.add_review_comment(pr_number=1, comment="Looks great, verified CBF safety margin.")
        c2 = emulator.add_review_comment(
            pr_number=1,
            comment="Inline fix verified",
            file_path="src/pool.py",
            line=1,
        )
        comments = emulator.get_review_comments(pr_number=1)
        assert len(comments) == 2

        # 3. Add labels
        labels = emulator.add_labels(pr_number=1, labels=["auto-merge", "production-approved"])
        assert "auto-merge" in labels

        # 4. Set commit statuses
        emulator.set_commit_status(
            commit_sha="abcd1234",
            state="success",
            context="deeplog/formal-safety-verifier",
            description="CBF margin +0.32 certified",
        )
        statuses = emulator.get_commit_statuses("abcd1234")
        assert len(statuses) == 1
        assert statuses[0]["state"] == "success"

        # 5. Merge PR
        merged_pr = emulator.merge_pull_request(pr_number=1, merged_by="gitops-bot")
        assert merged_pr["state"] == "closed"
        assert merged_pr["merged"] is True
        assert merged_pr["merge_commit_sha"] is not None

        # Verify audit history
        assert len(emulator.delivery_history) > 0
        emulator.clear_history()
        assert len(emulator.delivery_history) == 0


# ==============================================================================
# 4. Tests for ArgoCDSyncEngine & Automated Rollback
# ==============================================================================


class TestArgoCDSyncEngine:
    """Test suite for ArgoCD deployment sync, health evaluation, and automated rollback."""

    @pytest.fixture
    def engine(self):
        eng = ArgoCDSyncEngine(initial_revision="v1.0.0-baseline")
        eng.register_application(
            app_name="hdfs-datanode-cluster",
            repo_url="https://github.com/deeplog/k8s-apps.git",
            target_revision="main",
            auto_sync=True,
            auto_rollback=True,
        )
        return eng

    def test_initial_sync_and_state_transition(self, engine: ArgoCDSyncEngine):
        app = engine.get_application("hdfs-datanode-cluster")
        assert app is not None
        assert app.sync_status == SyncStatus.OUT_OF_SYNC
        assert app.active_revision == "v1.0.0-baseline"

        # Sync to new revision
        res = engine.sync(app_name="hdfs-datanode-cluster", revision="commit-sha-7a8b9c")
        assert res.success is True
        assert res.sync_status == SyncStatus.SYNCED
        assert res.health_status == HealthStatus.HEALTHY
        assert app.active_revision == "commit-sha-7a8b9c"
        assert len(app.history) == 1

    def test_dry_run_sync(self, engine: ArgoCDSyncEngine):
        app = engine.get_application("hdfs-datanode-cluster")
        res = engine.sync(app_name="hdfs-datanode-cluster", revision="preview-sha", dry_run=True)
        assert res.success is True
        assert app.active_revision == "v1.0.0-baseline"  # Did not mutate

    def test_health_evaluation_and_automated_rollback_on_error_rate(self, engine: ArgoCDSyncEngine):
        # Step 1: Sync to release v1.1.0
        engine.sync("hdfs-datanode-cluster", revision="v1.1.0")

        # Step 2: Healthy evaluation (error rate 1%)
        h1 = engine.evaluate_health("hdfs-datanode-cluster", error_rate=0.01)
        assert h1.is_healthy is True
        assert h1.health_status == HealthStatus.HEALTHY

        # Step 3: Degraded evaluation (error rate 8.5% > 5%)
        # Should automatically trigger rollback!
        h2 = engine.evaluate_health("hdfs-datanode-cluster", error_rate=0.085)
        assert h2.is_healthy is False
        assert h2.health_status == HealthStatus.DEGRADED
        assert len(h2.degraded_reasons) >= 1

        app = engine.get_application("hdfs-datanode-cluster")
        # App should have rolled back to previous baseline
        assert app.active_revision == "v1.0.0-baseline"
        assert len(engine.rollback_history) == 1
        assert engine.rollback_history[0].to_revision == "v1.0.0-baseline"

    def test_automated_rollback_on_cbf_safety_violation(self, engine: ArgoCDSyncEngine):
        engine.sync("hdfs-datanode-cluster", revision="experimental-patch")

        # Negative CBF margin indicates provable safety violation
        h = engine.evaluate_health("hdfs-datanode-cluster", cbf_margin=-0.05)
        assert h.is_healthy is False
        assert any("safety violation" in reason.lower() for reason in h.degraded_reasons)

        app = engine.get_application("hdfs-datanode-cluster")
        assert app.active_revision == "v1.0.0-baseline"

    def test_webhook_emulator_auto_sync_integration(self, engine: ArgoCDSyncEngine):
        emulator = GitHubWebhookEmulator()
        engine.connect_webhook_emulator(emulator, app_name="hdfs-datanode-cluster")

        spec = GitPRSpec(
            pr_id="PR-100",
            title="Apply validated timeout fix",
            branch_name="hotfix/timeout",
            target_file="DataXceiver.java",
            summary="Timeout fix",
            patch_diff="--- a/file\n+++ b/file\n",
        )
        emulator.create_pull_request(spec)

        # Merge PR - this automatically dispatches push & PR closed webhooks
        emulator.merge_pull_request(pr_number=1)

        app = engine.get_application("hdfs-datanode-cluster")
        # App active revision should have updated via webhook!
        assert app.sync_status == SyncStatus.SYNCED
        assert app.active_revision != "v1.0.0-baseline"


# ==============================================================================
# 5. Tests for SLMDatasetCurator & Schema Validation
# ==============================================================================


class TestSLMDatasetCurator:
    """Test suite for SLM instruction-tuning dataset curation and schemas."""

    @pytest.fixture
    def curator(self):
        return SLMDatasetCurator()

    def test_synthetic_incident_generation_and_diversity(self, curator: SLMDatasetCurator):
        samples = curator.generate_synthetic_incidents(count=25, seed=42)
        assert len(samples) == 25

        categories = {s.metadata.get("category") for s in samples}
        assert len(categories) >= 4
        assert "TIMEOUT_EXHAUSTION" in categories
        assert "RESOURCE_LEAK" in categories
        assert "DEADLOCK_CONCURRENCY" in categories

        for s in samples:
            assert s.system_prompt
            assert s.incident_telemetry
            assert s.ast_cause
            assert s.formal_safety_proof
            assert s.git_patch.startswith("--- ")

    def test_alpaca_schema_formatting_and_validation(self, curator: SLMDatasetCurator):
        samples = curator.generate_synthetic_incidents(count=5, seed=1)
        for s in samples:
            alp = s.to_alpaca()
            assert curator.validate_schema(alp, SLMDatasetFormat.ALPACA) is True
            assert "instruction" in alp
            assert "input" in alp
            assert "output" in alp
            assert "AST Root Cause Diagnosis" in alp["output"]
            assert "```diff" in alp["output"]

    def test_sharegpt_schema_formatting_and_validation(self, curator: SLMDatasetCurator):
        samples = curator.generate_synthetic_incidents(count=5, seed=2)
        for s in samples:
            sg = s.to_sharegpt()
            assert curator.validate_schema(sg, SLMDatasetFormat.SHAREGPT) is True
            convs = sg["conversations"]
            assert len(convs) == 3
            assert convs[0]["from"] == "system"
            assert convs[1]["from"] == "human"
            assert convs[2]["from"] == "gpt"

    def test_huggingface_schema_formatting_and_validation(self, curator: SLMDatasetCurator):
        samples = curator.generate_synthetic_incidents(count=5, seed=3)
        for s in samples:
            hf = s.to_huggingface()
            assert curator.validate_schema(hf, SLMDatasetFormat.HUGGINGFACE) is True
            msgs = hf["messages"]
            assert len(msgs) == 3
            assert msgs[0]["role"] == "system"
            assert msgs[1]["role"] == "user"
            assert msgs[2]["role"] == "assistant"

    def test_dataset_filtering_and_deduplication(self, curator: SLMDatasetCurator):
        curator.generate_synthetic_incidents(count=20, seed=10)
        initial_count = len(curator.samples)

        # Filter by category
        timeouts = curator.filter_samples(category="TIMEOUT_EXHAUSTION")
        assert len(timeouts) > 0
        assert all(t.metadata.get("category") == "TIMEOUT_EXHAUSTION" for t in timeouts)

        # Deduplication
        # Add duplicate sample
        curator.add_sample(curator.samples[0])
        assert len(curator.samples) == initial_count + 1
        removed = curator.deduplicate()
        assert removed == 1
        assert len(curator.samples) == initial_count

    def test_dataset_split_and_jsonl_export(self, curator: SLMDatasetCurator, tmp_path: Path):
        curator.generate_synthetic_incidents(count=30, seed=99)
        out_base = tmp_path / "finetune.jsonl"

        # Export with 80/10/10 split
        results = curator.export_dataset(
            output_path=out_base,
            format_type=SLMDatasetFormat.ALPACA,
            split_ratios=(0.8, 0.1, 0.1),
        )

        assert len(results) == 3
        train_path = tmp_path / "finetune_train.jsonl"
        val_path = tmp_path / "finetune_val.jsonl"
        test_path = tmp_path / "finetune_test.jsonl"

        assert train_path.exists()
        assert val_path.exists()
        assert test_path.exists()

        train_lines = train_path.read_text(encoding="utf-8").strip().splitlines()
        val_lines = val_path.read_text(encoding="utf-8").strip().splitlines()
        test_lines = test_path.read_text(encoding="utf-8").strip().splitlines()

        assert len(train_lines) == 24
        assert len(val_lines) == 3
        assert len(test_lines) == 3

        # Verify JSON record integrity
        rec = json.loads(train_lines[0])
        assert "instruction" in rec
        assert "input" in rec
        assert "output" in rec

    def test_dataset_statistics(self, curator: SLMDatasetCurator):
        curator.generate_synthetic_incidents(count=15, seed=7)
        stats = curator.get_statistics()
        assert stats["total_samples"] == 15
        assert stats["estimated_tokens"] > 1000
        assert stats["average_tokens_per_sample"] > 50
        assert len(stats["categories"]) >= 3


# ==============================================================================
# 6. CLI Script Integration Test
# ==============================================================================


class TestCLIScript:
    """Test suite for scripts/generate_slm_dataset.py CLI tool."""

    def test_cli_execution_with_flags(self, tmp_path: Path):
        cli_path = Path(__file__).resolve().parent.parent / "scripts" / "generate_slm_dataset.py"
        out_file = tmp_path / "cli_slm.jsonl"

        cmd = [
            sys.executable,
            str(cli_path),
            "--output", str(out_file),
            "--format", "sharegpt",
            "--num-samples", "12",
            "--seed", "123",
            "--validate",
            "--stats",
        ]

        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        assert res.returncode == 0, f"CLI failed: {res.stderr}"
        assert out_file.exists()

        lines = out_file.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 12

        sample = json.loads(lines[0])
        assert "conversations" in sample

    def test_cli_split_execution(self, tmp_path: Path):
        cli_path = Path(__file__).resolve().parent.parent / "scripts" / "generate_slm_dataset.py"
        out_file = tmp_path / "split_dataset.jsonl"

        cmd = [
            sys.executable,
            str(cli_path),
            "--output", str(out_file),
            "--format", "huggingface",
            "--num-samples", "20",
            "--split", "0.7,0.2,0.1",
            "--validate",
        ]

        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        assert res.returncode == 0, f"CLI split failed: {res.stderr}"

        train_f = tmp_path / "split_dataset_train.jsonl"
        val_f = tmp_path / "split_dataset_val.jsonl"
        test_f = tmp_path / "split_dataset_test.jsonl"

        assert train_f.exists()
        assert val_f.exists()
        assert test_f.exists()
