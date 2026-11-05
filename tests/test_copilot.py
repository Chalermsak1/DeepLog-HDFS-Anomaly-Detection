"""
Unit Tests for DeepLog AI SRE Copilot & Autonomous Verification Reasoning Engine
================================================================================
Validates:
1. `CopilotIntent` & Intent Detection across operational SRE queries.
2. `CodeSnippet`, `CLICommand`, `SafetyProofReport`, `CopilotContext`, `CopilotResponse` data structures.
3. "Why was this mitigation selected?" causal counterfactual reasoning.
4. "Prove safety of action" Control Barrier Function (CBF) & Lyapunov stability verification.
5. "Explain root cause" AST call stack, attention attribution, and git lineage correlation.
6. "Refactor patch" autonomous patch hardening, exponential backoff, and unit test generation.
7. Multi-modal Telemetry & Incident investigation triage.
8. Local/Mock LLM endpoint integration, custom callable backends, and zero-dependency offline fallback.
9. End-to-end multi-turn conversational session context management.
"""

import json
import pytest
import numpy as np

from deeplog.copilot import (
    SRECopilot,
    CopilotResponse,
    CopilotIntent,
    CommandSafetyLevel,
    CodeSnippet,
    CLICommand,
    SafetyProofReport,
    CopilotContext,
    BaseLLMBackend,
    RuleBasedExpertBackend,
    MockLLMBackend,
    CallableLLMBackend,
    HTTPLLMBackend,
)
from deeplog.formal_safety import (
    ControlBarrierFunction,
    SafetyClusterState,
    ClusterState,
    SafetyInvariant,
    QPSafetyFilter,
)
from deeplog.world_model import (
    MitigationAction,
    MitigationDecision,
    MitigationUrgency,
)
from deeplog.neuro_symbolic import (
    PatchProposal,
    CodeLocation,
    GitBlameInfo,
)
from deeplog.summarizer import (
    IncidentReport,
    TimelineItem,
)


# ==============================================================================
# 1. Data Structures & Serialization Tests
# ==============================================================================


class TestCopilotDataStructures:
    """Test SRE Copilot data containers, markdown generation, and serializations."""

    def test_code_snippet_markdown_and_dict(self):
        snip = CodeSnippet(
            language="python",
            code="x = 42\nprint(x)",
            file_path="src/sample.py",
            description="Sample calculation snippet",
            line_start=10,
            line_end=11,
        )
        md = snip.to_markdown()
        assert "### Sample calculation snippet" in md
        assert "**File:** `src/sample.py`" in md
        assert "```python\nx = 42\nprint(x)\n```" in md

        d = snip.to_dict()
        assert d["language"] == "python"
        assert d["file_path"] == "src/sample.py"
        assert d["line_start"] == 10

    def test_cli_command_markdown_and_safety_levels(self):
        cmd = CLICommand(
            command="kubectl scale deployment/datanode --replicas=4",
            description="Scale datanode replicas",
            safety_level=CommandSafetyLevel.CAUTION,
            target_system="kubernetes",
            dry_run="kubectl scale deployment/datanode --replicas=4 --dry-run=client",
            rollback="kubectl scale deployment/datanode --replicas=2",
        )
        md = cmd.to_markdown()
        assert "[CAUTION]" in md
        assert "(kubernetes)" in md
        assert "```bash\n  kubectl scale deployment/datanode --replicas=4\n  ```" in md
        assert "*Dry Run:*" in md
        assert "*Rollback:*" in md

        d = cmd.to_dict()
        assert d["safety_level"] == "CAUTION"
        assert d["target_system"] == "kubernetes"

    def test_safety_proof_report_markdown_and_dict(self):
        proof = SafetyProofReport(
            is_safe=True,
            action="eBPF Throttling",
            margin=0.085,
            invariants_evaluated=[
                {"name": "min_replicas", "threshold": 2.0, "actual": 3.0, "h_val": 1.0, "satisfied": True},
                {"name": "min_cpu_headroom", "threshold": 0.15, "actual": 0.35, "h_val": 0.2, "satisfied": True},
            ],
            violated_invariants=[],
            active_invariants=["max_traffic_drop"],
            lie_derivatives={"min_replicas": 0.0},
            lyapunov_drift=-0.12,
            lyapunov_stable=True,
            mathematical_proof="Nagumo CBF forward invariance holds.",
        )
        md = proof.to_markdown()
        assert "PASSED (PROVABLY SAFE)" in md
        assert "+0.0850" in md
        assert "Stable (No Oscillation)" in md
        assert "`min_replicas`" in md

        d = proof.to_dict()
        assert d["is_safe"] is True
        assert len(d["invariants_evaluated"]) == 2

    def test_copilot_context_management(self):
        ctx = CopilotContext()
        assert ctx.incident is None
        assert len(ctx.conversation_history) == 0

        # Update
        ctx.update(
            incident={"id": "INC-01"},
            extra_metric=99.5,
        )
        assert ctx.incident == {"id": "INC-01"}
        assert ctx.extra_data["extra_metric"] == 99.5

        # Summary
        summary = ctx.to_summary_dict()
        assert summary["has_incident"] is True
        assert summary["has_telemetry"] is False

        # Clear
        ctx.clear()
        assert ctx.incident is None
        assert len(ctx.extra_data) == 0

    def test_copilot_response_json_and_markdown(self):
        resp = CopilotResponse(
            query="Test query",
            intent=CopilotIntent.PROVE_SAFETY,
            answer="This action is formally safe.",
            confidence=0.99,
            code_snippets=[
                CodeSnippet(language="bash", code="echo 'ok'", description="Test check")
            ],
            cli_commands=[
                CLICommand(command="kubectl get pods", description="List pods", safety_level=CommandSafetyLevel.SAFE),
                CLICommand(command="kubectl delete pod foo", description="Delete pod", safety_level=CommandSafetyLevel.DESTRUCTIVE),
            ],
        )
        j = resp.to_json()
        data = json.loads(j)
        assert data["query"] == "Test query"
        assert data["intent"] == "PROVE_SAFETY"
        assert data["confidence"] == 0.99

        md = resp.to_markdown()
        assert "DeepLog AI SRE Copilot Response" in md
        assert "99.0%" in md

        # Filter safe commands
        safe = resp.get_safe_commands()
        assert len(safe) == 1
        assert safe[0].command == "kubectl get pods"

        # Code snippet lookup
        snip = resp.get_first_code_snippet("bash")
        assert snip is not None
        assert snip.code == "echo 'ok'"


# ==============================================================================
# 2. Intent Classification Tests
# ==============================================================================


class TestIntentClassification:
    """Validate query intent detection across typical SRE prompts."""

    @pytest.fixture
    def copilot(self):
        return SRECopilot()

    def test_detect_why_mitigation_intents(self, copilot):
        queries = [
            "Why was this mitigation selected?",
            "Why did the system choose this action?",
            "Explain the rationale for mitigation",
            "Why migrate the replica instead of throttling?",
            "What is the counterfactual reason for selecting eBPF backpressure?",
        ]
        for q in queries:
            assert copilot.detect_intent(q) == CopilotIntent.WHY_MITIGATION, f"Failed for query: {q}"

    def test_detect_prove_safety_intents(self, copilot):
        queries = [
            "Prove safety of action",
            "Verify safety of proposed control",
            "Prove that the control barrier function is satisfied",
            "Generate Nagumo Lyapunov safety proof",
            "Is this mitigation action provably safe?",
        ]
        for q in queries:
            assert copilot.detect_intent(q) == CopilotIntent.PROVE_SAFETY, f"Failed for query: {q}"

    def test_detect_explain_root_cause_intents(self, copilot):
        queries = [
            "Explain root cause",
            "What is the root cause of the error?",
            "Why did it fail at line 247?",
            "Who to blame for the exception writing block to mirror?",
            "Perform root cause analysis on this incident",
        ]
        for q in queries:
            assert copilot.detect_intent(q) == CopilotIntent.EXPLAIN_ROOT_CAUSE, f"Failed for query: {q}"

    def test_detect_refactor_patch_intents(self, copilot):
        queries = [
            "Refactor patch",
            "Improve patch to add exponential backoff",
            "Generate safer patch diff with try-with-resources",
            "Refactor this fix to prevent resource leakage",
            "Review and harden the patch proposal",
        ]
        for q in queries:
            assert copilot.detect_intent(q) == CopilotIntent.REFACTOR_PATCH, f"Failed for query: {q}"

    def test_detect_telemetry_and_incident_intents(self, copilot):
        assert copilot.detect_intent("Analyze telemetry and PromQL metrics") == CopilotIntent.ANALYZE_TELEMETRY
        assert copilot.detect_intent("Investigate incident INC-9942 severity") == CopilotIntent.INVESTIGATE_INCIDENT
        assert copilot.detect_intent("What is the recommended SLA target?") == CopilotIntent.GENERAL_SRE


# ==============================================================================
# 3. Question 1: "Why was this mitigation selected?"
# ==============================================================================


class TestExplainMitigation:
    """Tests for causal counterfactual mitigation explanation."""

    @pytest.fixture
    def copilot(self):
        return SRECopilot()

    def test_default_mitigation_explanation(self, copilot):
        resp = copilot.ask("Why was this mitigation selected?")
        assert resp.intent == CopilotIntent.WHY_MITIGATION
        assert resp.confidence >= 0.85
        assert "eBPF" in resp.answer or "TCP" in resp.answer
        assert "Baseline Risk" in resp.answer
        assert "Net Risk Reduction" in resp.answer
        assert len(resp.cli_commands) > 0
        assert len(resp.code_snippets) > 0

    def test_mitigation_decision_with_ebpf_throttle(self, copilot):
        decision = MitigationDecision(
            action=MitigationAction.EBPF_THROTTLE,
            action_name="Adaptive eBPF Backpressure",
            target_resource="datanode-pod-03",
            baseline_risk=0.91,
            mitigated_risk=0.09,
            risk_reduction=0.82,
            lead_time_steps=5,
            urgency=MitigationUrgency.CRITICAL,
            confidence=0.95,
            rationale="Socket queue filling rapidly; TCP clamp prevents buffer drop.",
            parameters={"tcp_window_clamp": 1460},
            action_plan=["Attach eBPF program", "Monitor queue drop"],
            timestamp=1690000000.0,
            all_trajectories={},
        )
        resp = copilot.explain_mitigation(decision=decision)
        assert resp.intent == CopilotIntent.WHY_MITIGATION
        assert resp.confidence >= 0.90
        assert "91.0%" in resp.answer  # Baseline
        assert "9.0%" in resp.answer   # Mitigated
        assert "82.0%" in resp.answer  # Reduction
        assert "5 steps" in resp.answer

        # Verify CLI commands include eBPF tooling
        commands = [c.command for c in resp.cli_commands]
        assert any("bpftool" in cmd for cmd in commands)

    def test_mitigation_decision_with_replica_migration(self, copilot):
        decision = MitigationDecision(
            action=MitigationAction.MIGRATE_REPLICA,
            action_name="Live Container Migration",
            target_resource="datanode-worker-08",
            baseline_risk=0.85,
            mitigated_risk=0.15,
            risk_reduction=0.70,
            lead_time_steps=3,
            urgency=MitigationUrgency.HIGH,
            confidence=0.92,
            rationale="Hardware degradation detected on node.",
            parameters={},
            action_plan=["Evict pod"],
            timestamp=1690000000.0,
            all_trajectories={},
        )
        resp = copilot.explain_mitigation(decision=decision)
        assert resp.intent == CopilotIntent.WHY_MITIGATION
        assert "Live Container Migration" in resp.answer
        commands = [c.command for c in resp.cli_commands]
        assert any("kubectl drain" in cmd for cmd in commands)

    def test_mitigation_decision_from_dict(self, copilot):
        decision_dict = {
            "action_name": "Dynamic Circuit Breaking",
            "action": MitigationAction.CIRCUIT_BREAKER_OPEN,
            "baseline_risk": 0.95,
            "mitigated_risk": 0.20,
            "risk_reduction": 0.75,
            "lead_time_steps": 2,
            "rationale": "Cascading failure isolation.",
        }
        resp = copilot.explain_mitigation(decision=decision_dict)
        assert resp.intent == CopilotIntent.WHY_MITIGATION
        assert "Dynamic Circuit Breaking" in resp.answer
        commands = [c.command for c in resp.cli_commands]
        assert any("circuit_breakers" in cmd for cmd in commands)


# ==============================================================================
# 4. Question 2: "Prove safety of action"
# ==============================================================================


class TestProveSafety:
    """Tests for formal Control Barrier Function (CBF) & Lyapunov verification."""

    @pytest.fixture
    def copilot(self):
        return SRECopilot()

    def test_prove_safety_default_state(self, copilot):
        resp = copilot.ask("Prove safety of action")
        assert resp.intent == CopilotIntent.PROVE_SAFETY
        assert resp.confidence == 1.0  # Formal proof confidence
        assert resp.safety_proof is not None
        assert resp.safety_proof.is_safe is True
        assert resp.safety_proof.margin > 0.0
        assert len(resp.safety_proof.invariants_evaluated) >= 5
        assert "Nagumo / Ames CBF" in resp.safety_proof.mathematical_proof

    def test_prove_safety_with_custom_healthy_state(self, copilot):
        state = SafetyClusterState(
            replicas=4.0,
            cpu_headroom=0.50,
            traffic_drop=0.01,
            queue_depth=15.0,
            latency_ms=30.0,
            memory_headroom=0.55,
            error_rate=0.0,
        )
        resp = copilot.prove_safety(action=MitigationAction.EBPF_THROTTLE, state=state)
        assert resp.safety_proof is not None
        assert resp.safety_proof.is_safe is True
        assert resp.safety_proof.margin >= 0.04  # max traffic drop bound
        assert len(resp.safety_proof.violated_invariants) == 0

    def test_prove_safety_unsafe_violation(self, copilot):
        # Replica count 1.0 violates min_replicas >= 2.0
        state = SafetyClusterState(
            replicas=1.0,
            cpu_headroom=0.05,  # Violates min_cpu_headroom >= 0.15
            traffic_drop=0.20,  # Violates max_traffic_drop <= 0.05
            queue_depth=800.0,  # Violates max_queue_depth <= 500
            latency_ms=400.0,   # Violates max_latency_ms <= 250
        )
        resp = copilot.prove_safety(action="High Risk Action", state=state)
        assert resp.safety_proof is not None
        assert resp.safety_proof.is_safe is False
        assert resp.safety_proof.margin < 0.0
        assert "min_replicas" in resp.safety_proof.violated_invariants
        assert "min_cpu_headroom" in resp.safety_proof.violated_invariants
        assert resp.safety_proof.recommended_adjustments is not None

    def test_prove_safety_lie_derivatives_and_lyapunov(self, copilot):
        resp = copilot.prove_safety(action="Test Action")
        proof = resp.safety_proof
        assert len(proof.lie_derivatives) > 0
        assert proof.lyapunov_stable is not None
        assert proof.lyapunov_drift is not None
        assert "Delta V" in proof.mathematical_proof


# ==============================================================================
# 5. Question 3: "Explain root cause"
# ==============================================================================


class TestExplainRootCause:
    """Tests for neuro-symbolic AST call stack and root cause analysis."""

    @pytest.fixture
    def copilot(self):
        return SRECopilot()

    def test_explain_root_cause_default_flow(self, copilot):
        resp = copilot.ask("Explain root cause")
        assert resp.intent == CopilotIntent.EXPLAIN_ROOT_CAUSE
        assert resp.confidence >= 0.90
        assert "Root Cause Diagnosis" in resp.answer
        assert "Antecedent Trigger" in resp.answer
        assert "Git Lineage" in resp.answer
        assert len(resp.code_snippets) > 0
        assert len(resp.cli_commands) > 0

        # Check code snippet details
        snip = resp.code_snippets[0]
        assert snip.language == "java"
        assert snip.file_path is not None
        assert "public void" in snip.code

    def test_explain_root_cause_with_incident_report(self, copilot):
        incident = IncidentReport(
            incident_id="INC-HDFS-773",
            block_id="blk_1092837",
            severity="CRITICAL",
            status="TRIGGERED",
            timestamp="2026-11-05T12:00:00Z",
            executive_summary="Packet timeout on DataNode mirror",
            root_cause_analysis={"primary_trigger": "E12"},
            anomaly_breakdown={},
            timeline=[
                TimelineItem(step=1, timestamp=1.0, event_id="E5", template="Receiving block", is_anomaly=False),
                TimelineItem(step=2, timestamp=2.0, event_id="E6", template="Received block", is_anomaly=False),
                TimelineItem(step=3, timestamp=3.0, event_id="E12", template="Exception writing block to mirror", is_anomaly=True),
            ],
            impact_analysis={},
            remediation_suggestions=[],
        )
        resp = copilot.explain_root_cause(incident=incident)
        assert resp.intent == CopilotIntent.EXPLAIN_ROOT_CAUSE
        assert "INC-HDFS-773" in resp.answer
        assert "CRITICAL" in resp.answer
        assert resp.root_cause_summary is not None
        assert resp.root_cause_summary["commit"] is not None
        assert resp.root_cause_summary["author"] is not None

    def test_explain_root_cause_with_sequence(self, copilot):
        resp = copilot.explain_root_cause(incident=["E5", "E6", "E7"])
        assert resp.intent == CopilotIntent.EXPLAIN_ROOT_CAUSE
        assert "DataXceiver" in resp.answer
        assert resp.root_cause_summary["file_path"].endswith(".java")


# ==============================================================================
# 6. Question 4: "Refactor patch"
# ==============================================================================


class TestRefactorPatch:
    """Tests for patch refactoring, unified diff generation, and unit testing."""

    @pytest.fixture
    def copilot(self):
        return SRECopilot()

    def test_refactor_patch_default_flow(self, copilot):
        resp = copilot.ask("Refactor patch")
        assert resp.intent == CopilotIntent.REFACTOR_PATCH
        assert resp.confidence >= 0.90
        assert "Exponential Backoff" in resp.answer
        assert "Resource Leak Prevention" in resp.answer

        # Check code snippets include unified diff and unit test
        diff_snip = resp.get_first_code_snippet("diff")
        assert diff_snip is not None
        assert "--- a/" in diff_snip.code
        assert "+++ b/" in diff_snip.code
        assert "@@" in diff_snip.code

        test_snip = resp.get_first_code_snippet("java")
        assert test_snip is not None
        assert "@Test" in test_snip.code
        assert "assertDoesNotThrow" in test_snip.code

        # Check CLI commands
        cmds = [c.command for c in resp.cli_commands]
        assert any("git apply --check" in cmd for cmd in cmds)

    def test_refactor_patch_from_existing_proposal(self, copilot):
        diff_text = (
            "--- a/DataXceiver.java\n"
            "+++ b/DataXceiver.java\n"
            "@@ -10,3 +10,4 @@\n"
            " void writeBlock() {\n"
            "+    socket.setSoTimeout(5000);\n"
            " }\n"
        )
        proposal = PatchProposal(
            patch_id="PATCH-CUSTOM-01",
            title="Add Socket Timeout",
            target_file="org/apache/hadoop/hdfs/server/datanode/DataXceiver.java",
            target_class="DataXceiver",
            target_method="writeBlock",
            faulting_line=247,
            category="TIMEOUT_EXHAUSTION",
            rationale="Socket read stall",
            patch_diff=diff_text,
            unit_test_code="class CustomTest {}",
            unit_test_description="Verify timeout",
            pr_description="PR body",
            dry_run_verified=True,
        )
        resp = copilot.refactor_patch(patch=proposal)
        assert resp.intent == CopilotIntent.REFACTOR_PATCH
        assert "Add Socket Timeout" in resp.answer
        diff_snip = resp.get_first_code_snippet("diff")
        assert diff_snip is not None
        assert "socket.setSoTimeout" in diff_snip.code

    def test_refactor_patch_with_string_diff(self, copilot):
        raw_diff = (
            "--- a/DataNode.java\n"
            "+++ b/DataNode.java\n"
            "@@ -50,2 +50,3 @@\n"
            "+// retry guard\n"
        )
        resp = copilot.refactor_patch(patch=raw_diff)
        assert resp.intent == CopilotIntent.REFACTOR_PATCH
        snip = resp.get_first_code_snippet("diff")
        assert snip is not None
        assert "retry guard" in snip.code


# ==============================================================================
# 7. Telemetry & Incident Analysis Tests
# ==============================================================================


class TestTelemetryAndIncidentAnalysis:
    """Validate telemetry inspection and incident triage."""

    @pytest.fixture
    def copilot(self):
        return SRECopilot()

    def test_analyze_telemetry(self, copilot):
        resp = copilot.analyze_telemetry()
        assert resp.intent == CopilotIntent.ANALYZE_TELEMETRY
        assert "METRIC_SATURATION" in resp.answer
        assert "LOG_ERRORS" in resp.answer
        assert len(resp.cli_commands) > 0

    def test_investigate_incident(self, copilot):
        resp = copilot.investigate_incident(incident={"incident_id": "INC-099", "severity": "CRITICAL"})
        assert resp.intent == CopilotIntent.INVESTIGATE_INCIDENT
        assert "INC-099" in resp.answer
        assert "CRITICAL" in resp.answer

    def test_general_sre_guidance(self, copilot):
        resp = copilot.ask("What is the difference between SLI and SLO?")
        assert resp.intent == CopilotIntent.GENERAL_SRE
        assert "SRE Copilot Operational Guidance" in resp.answer


# ==============================================================================
# 8. Local / Mock LLM Integration & Offline Fallback Tests
# ==============================================================================


class TestLLMIntegrationAndFallback:
    """Validate mock LLM backends, callable adapters, and offline fallback."""

    def test_mock_llm_backend_success(self):
        mock = MockLLMBackend()
        mock.set_response(
            "mitigation",
            json.dumps({
                "answer": "Mock LLM: Throttling chosen to prevent packet queue spill.",
                "confidence": 0.97,
                "code_snippets": [{"language": "python", "code": "apply_throttle()", "description": "Apply"}],
                "cli_commands": [{"command": "bpftool prog show", "description": "Check eBPF", "safety_level": "SAFE"}],
            })
        )
        copilot = SRECopilot(llm_backend=mock, mode="llm")
        resp = copilot.ask("Why was this mitigation selected?")
        assert resp.intent == CopilotIntent.WHY_MITIGATION
        assert resp.confidence == 0.97
        assert "Mock LLM" in resp.answer
        assert len(resp.cli_commands) == 1
        assert len(resp.code_snippets) == 1
        assert len(mock.call_log) == 1

    def test_callable_llm_backend(self):
        def my_llm(prompt: str) -> str:
            assert "User Query: Explain root cause" in prompt
            return json.dumps({
                "answer": "Custom callable: Socket closed unexpectedly.",
                "confidence": 0.91,
            })

        copilot = SRECopilot(llm_backend=my_llm, mode="llm")
        resp = copilot.ask("Explain root cause")
        assert "Custom callable" in resp.answer
        assert resp.confidence == 0.91

    def test_hybrid_mode_resilient_fallback_on_llm_exception(self):
        # When LLM raises an unexpected timeout / connection error in hybrid mode
        failing_mock = MockLLMBackend(raise_exception=ConnectionError("LLM API endpoint offline"))
        copilot = SRECopilot(llm_backend=failing_mock, mode="hybrid")

        # Must not raise an exception; must fall back to rule-based expert engine
        resp = copilot.ask("Why was this mitigation selected?")
        assert resp.intent == CopilotIntent.WHY_MITIGATION
        assert resp.confidence >= 0.85
        assert "Latent World Model" in resp.answer

    def test_llm_mode_raises_when_failing(self):
        failing_mock = MockLLMBackend(raise_exception=ConnectionError("Hard failure in LLM mode"))
        copilot = SRECopilot(llm_backend=failing_mock, mode="llm")
        with pytest.raises(ConnectionError):
            copilot.ask("Why was this mitigation selected?")

    def test_rule_based_expert_backend_zero_dependencies(self):
        expert_backend = RuleBasedExpertBackend()
        copilot = SRECopilot(llm_backend=expert_backend, mode="expert")
        resp = copilot.ask("Prove safety of action")
        assert resp.intent == CopilotIntent.PROVE_SAFETY
        assert resp.safety_proof.is_safe is True


# ==============================================================================
# 9. End-to-End Multi-Turn Conversational Session Flow
# ==============================================================================


class TestEndToEndConversationalFlow:
    """Test full multi-turn conversational SRE investigation lifecycle."""

    def test_full_incident_triage_conversation(self):
        copilot = SRECopilot()

        # Step 1: Ingest active incident and ask for root cause
        copilot.update_context(
            incident={
                "incident_id": "INC-PROD-404",
                "severity": "CRITICAL",
                "event_sequence": ["E5", "E6", "E12", "E17"],
            }
        )
        resp1 = copilot.ask("Explain root cause")
        assert resp1.intent == CopilotIntent.EXPLAIN_ROOT_CAUSE
        assert "INC-PROD-404" in resp1.answer
        assert len(copilot.context.conversation_history) == 2

        # Step 2: Inquire why mitigation was chosen
        copilot.update_context(
            mitigation_decision=MitigationDecision(
                action=MitigationAction.EBPF_THROTTLE,
                action_name="Adaptive eBPF Backpressure",
                target_resource="datanode-01",
                baseline_risk=0.89,
                mitigated_risk=0.11,
                risk_reduction=0.78,
                lead_time_steps=4,
                urgency=MitigationUrgency.HIGH,
                confidence=0.94,
                rationale="Draining socket queue safely",
                parameters={},
                action_plan=[],
                timestamp=1690000000.0,
                all_trajectories={},
            )
        )
        resp2 = copilot.ask("Why was this mitigation selected?")
        assert resp2.intent == CopilotIntent.WHY_MITIGATION
        assert "78.0%" in resp2.answer
        assert len(copilot.context.conversation_history) == 4

        # Step 3: Prove safety of the chosen mitigation
        resp3 = copilot.ask("Prove safety of action")
        assert resp3.intent == CopilotIntent.PROVE_SAFETY
        assert resp3.safety_proof.is_safe is True
        assert len(copilot.context.conversation_history) == 6

        # Step 4: Ask copilot to refactor code patch
        resp4 = copilot.ask("Refactor patch")
        assert resp4.intent == CopilotIntent.REFACTOR_PATCH
        diff_snip = resp4.get_first_code_snippet("diff")
        assert diff_snip is not None
        assert "--- a/" in diff_snip.code
        assert len(copilot.context.conversation_history) == 8

        # Step 5: Reset conversation
        copilot.clear_context()
        assert len(copilot.context.conversation_history) == 0
        assert copilot.context.incident is None
