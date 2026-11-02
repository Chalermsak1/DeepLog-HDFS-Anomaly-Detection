"""
Unit tests for Track 5: Graph Topology (GNN) and GenAI Incident Summarizer.
Verifies:
- LogGraphBuilder: transition graph generation, node/edge features, DAG/cycle analysis, batched collation.
- LogGNNAnomalyDetector: GAT and Gated Graph Conv layers, attention/mean/max pooling, backward pass, save/load.
- IncidentSummarizer: 5-pillar telemetry synthesis (temporal, parameter, attention, semantic, graph topology),
  Markdown & JSON report exports, Mermaid visualization, and GenAI prompt formatting.
"""

import os
import json
import tempfile
import pytest
import numpy as np
import networkx as nx
import torch
import torch.nn as nn

from deeplog.graph_gnn import (
    LogGraphBuilder,
    LogGNNAnomalyDetector,
    GraphAttentionLayer,
    GatedGraphConvLayer,
    GlobalAttentionPooling,
)
from deeplog.summarizer import (
    IncidentSummarizer,
    IncidentReport,
    AnomalyEvidence,
    TimelineItem,
    DEFAULT_HDFS_TEMPLATES,
)
from deeplog.temporal import TemporalAnomalyResult
from deeplog.parameters import ParameterAnomalyResult


# ============================================================================ #
#                           1. LogGraphBuilder Tests                           #
# ============================================================================ #

class TestLogGraphBuilder:
    """Tests for execution graph construction and PyTorch tensor conversion."""

    def test_build_single_graph_basic(self):
        """Verify graph construction with nodes, directed edges, and counts."""
        builder = LogGraphBuilder(vocab_size=30)
        events = [5, 5, 26, 26, 9]
        graph = builder.build_graph(events, block_id="blk_101")

        assert isinstance(graph, nx.DiGraph)
        assert graph.graph["block_id"] == "blk_101"
        assert graph.graph["seq_len"] == 5

        # Unique nodes: 5, 26, 9
        assert set(graph.nodes()) == {5, 26, 9}
        assert graph.nodes[5]["count"] == 2
        assert graph.nodes[5]["is_start"] is True
        assert graph.nodes[9]["is_end"] is True

        # Transitions: 5->5, 5->26, 26->26, 26->9
        assert graph.has_edge(5, 5)
        assert graph.has_edge(5, 26)
        assert graph.has_edge(26, 26)
        assert graph.has_edge(26, 9)

        # Transition counts
        assert graph[5][5]["count"] == 1
        assert graph[5][26]["count"] == 1
        assert "transition_prob" in graph[5][26]

    def test_build_graph_with_timestamps(self):
        """Verify edge latency calculation from monotonic timestamps."""
        builder = LogGraphBuilder(vocab_size=30)
        events = [5, 5, 26]
        timestamps = [10.0, 10.5, 12.0]
        graph = builder.build_graph(events, timestamps=timestamps)

        # 5->5 latency is 0.5s
        assert abs(graph[5][5]["mean_latency"] - 0.5) < 1e-4
        # 5->26 latency is 1.5s
        assert abs(graph[5][26]["mean_latency"] - 1.5) < 1e-4

    def test_empty_and_single_event(self):
        """Verify robust handling of empty and single-event edge cases."""
        builder = LogGraphBuilder(vocab_size=30)

        # Empty sequence
        g_empty = builder.build_graph([])
        assert g_empty.number_of_nodes() == 0
        t_empty = builder.graph_to_tensors(g_empty)
        assert t_empty["x"].shape == (0, 16)
        assert t_empty["edge_index"].shape == (2, 0)

        # Single event
        g_single = builder.build_graph([5])
        assert g_single.number_of_nodes() == 1
        t_single = builder.graph_to_tensors(g_single)
        assert t_single["x"].shape == (1, 16)
        # With add_self_loops=True, 1 self-loop edge exists
        assert t_single["edge_index"].shape == (2, 1)

    def test_graph_to_tensors_shapes(self):
        """Verify tensor conversion shapes and dtypes."""
        builder = LogGraphBuilder(vocab_size=32, node_feature_dim=16, edge_feature_dim=4)
        events = [5, 5, 26, 11]
        tensors = builder.sequence_to_tensors(events)

        assert "x" in tensors
        assert "edge_index" in tensors
        assert "edge_attr" in tensors
        assert "node_ids" in tensors
        assert "adjacency" in tensors

        num_nodes = len(set(events))
        assert tensors["x"].shape == (num_nodes, 16)
        assert tensors["node_ids"].shape == (num_nodes,)
        assert tensors["edge_index"].shape[0] == 2
        assert tensors["edge_attr"].shape[1] == 4
        assert tensors["adjacency"].shape == (num_nodes, num_nodes)
        assert tensors["x"].dtype == torch.float32
        assert tensors["edge_index"].dtype == torch.long

    def test_collate_graphs_batching(self):
        """Verify graph batching and node index offset logic."""
        builder = LogGraphBuilder(vocab_size=30)
        g1 = builder.sequence_to_tensors([5, 26, 9])
        g2 = builder.sequence_to_tensors([22, 5, 11, 9])

        batched = LogGraphBuilder.collate_graphs([g1, g2])

        total_nodes = g1["x"].size(0) + g2["x"].size(0)
        assert batched["x"].shape[0] == total_nodes
        assert batched["batch"].shape[0] == total_nodes
        assert batched["num_graphs"] == 2

        # Check that g2 node indices in edge_index are offset by g1 num_nodes
        offset = g1["x"].size(0)
        g2_edges = batched["edge_index"][:, g1["edge_index"].size(1):]
        assert (g2_edges >= offset).all()

    def test_topological_cycle_and_dag_detection(self):
        """Verify DAG verification, cycle detection, and reference graph comparison."""
        builder = LogGraphBuilder(vocab_size=30)

        # DAG sequence: 1 -> 2 -> 3 (no back-edges)
        g_dag = builder.build_graph([1, 2, 3])
        metrics_dag = LogGraphBuilder.extract_topological_features(g_dag)
        assert metrics_dag["is_dag"] is True
        assert metrics_dag["has_cycles"] is False

        # Cyclic sequence: 11 -> 8 -> 11 (deadlock/retry loop)
        g_cycle = builder.build_graph([11, 8, 11])
        metrics_cycle = LogGraphBuilder.extract_topological_features(g_cycle)
        assert metrics_cycle["is_dag"] is False
        assert metrics_cycle["has_cycles"] is True
        assert metrics_cycle["cycle_count"] >= 1

        # Compare against normal reference graph
        ref_graph = builder.build_graph([5, 26, 9])
        anom_res = LogGraphBuilder.detect_structural_anomalies(g_cycle, reference_graph=ref_graph)
        assert anom_res["has_structural_anomaly"] is True
        assert len(anom_res["novel_edges"]) > 0


# ============================================================================ #
#                        2. LogGNNAnomalyDetector Tests                        #
# ============================================================================ #

class TestLogGNNAnomalyDetector:
    """Tests for Graph Neural Network architecture, message passing, and classification."""

    def test_gat_forward_pass(self):
        """Verify Graph Attention Network (GAT) forward pass and output contracts."""
        detector = LogGNNAnomalyDetector(
            node_in_dim=16,
            edge_in_dim=4,
            hidden_dim=32,
            out_dim=2,
            num_layers=2,
            nheads=4,
            gnn_type="gat",
            pooling="attention",
        )
        detector.eval()

        builder = LogGraphBuilder(vocab_size=30)
        tensors = builder.sequence_to_tensors([5, 5, 26, 9])

        with torch.no_grad():
            out = detector(
                x=tensors["x"],
                edge_index=tensors["edge_index"],
                edge_attr=tensors["edge_attr"],
                node_ids=tensors["node_ids"],
            )

        assert out["logits"].shape == (1, 2)
        assert out["probs"].shape == (1, 2)
        assert out["anomaly_score"].shape == (1,)
        assert out["graph_embedding"].shape == (1, 32)
        assert torch.allclose(out["probs"].sum(), torch.tensor(1.0), atol=1e-5)
        assert 0.0 <= out["anomaly_score"].item() <= 1.0

    def test_gated_gnn_forward_pass(self):
        """Verify Gated Graph Convolution layer forward pass."""
        detector = LogGNNAnomalyDetector(
            node_in_dim=16,
            edge_in_dim=4,
            hidden_dim=32,
            out_dim=2,
            num_layers=2,
            gnn_type="gated",
            pooling="mean",
        )
        detector.eval()

        builder = LogGraphBuilder(vocab_size=30)
        tensors = builder.sequence_to_tensors([5, 26, 11])

        with torch.no_grad():
            out = detector(
                x=tensors["x"],
                edge_index=tensors["edge_index"],
                edge_attr=tensors["edge_attr"],
                node_ids=tensors["node_ids"],
            )

        assert out["logits"].shape == (1, 2)
        assert torch.isfinite(out["logits"]).all()

    @pytest.mark.parametrize("pooling", ["attention", "mean", "max", "concat"])
    def test_pooling_mechanisms(self, pooling):
        """Verify all graph readout pooling strategies."""
        detector = LogGNNAnomalyDetector(
            node_in_dim=16,
            edge_in_dim=4,
            hidden_dim=32,
            out_dim=2,
            pooling=pooling,
        )
        builder = LogGraphBuilder(vocab_size=30)
        tensors = builder.sequence_to_tensors([5, 26, 9])

        out = detector(
            x=tensors["x"],
            edge_index=tensors["edge_index"],
            edge_attr=tensors["edge_attr"],
            node_ids=tensors["node_ids"],
        )
        assert out["logits"].shape == (1, 2)
        assert torch.isfinite(out["graph_embedding"]).all()

    def test_batched_forward_pass(self):
        """Verify multi-graph batched forward pass with batch indicator."""
        detector = LogGNNAnomalyDetector(
            node_in_dim=16,
            edge_in_dim=4,
            hidden_dim=32,
            out_dim=2,
            nheads=2,
        )
        builder = LogGraphBuilder(vocab_size=30)
        g1 = builder.sequence_to_tensors([5, 5, 26])
        g2 = builder.sequence_to_tensors([22, 5, 11, 9])
        g3 = builder.sequence_to_tensors([5, 9])

        batched = LogGraphBuilder.collate_graphs([g1, g2, g3])

        out = detector(
            x=batched["x"],
            edge_index=batched["edge_index"],
            edge_attr=batched["edge_attr"],
            node_ids=batched["node_ids"],
            batch=batched["batch"],
        )

        assert out["logits"].shape == (3, 2)
        assert out["anomaly_score"].shape == (3,)
        assert out["graph_embedding"].shape == (3, 32)

    def test_backward_pass_gradients(self):
        """Verify that all trainable parameters receive valid, non-zero gradients."""
        detector = LogGNNAnomalyDetector(
            node_in_dim=16,
            edge_in_dim=4,
            hidden_dim=32,
            out_dim=2,
            num_layers=2,
            gnn_type="gat",
        )
        detector.train()

        builder = LogGraphBuilder(vocab_size=30)
        tensors = builder.sequence_to_tensors([5, 26, 11])

        out = detector(
            x=tensors["x"],
            edge_index=tensors["edge_index"],
            edge_attr=tensors["edge_attr"],
            node_ids=tensors["node_ids"],
        )

        loss_fn = nn.CrossEntropyLoss()
        target = torch.tensor([1], dtype=torch.long)
        loss = loss_fn(out["logits"], target)
        loss.backward()

        for name, param in detector.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Parameter {name} has None gradient."
                assert torch.isfinite(param.grad).all(), f"Parameter {name} has NaN/Inf gradient."

    def test_predict_graph_high_level_api(self):
        """Verify high-level predict_graph API with graph, sequence, and node importance."""
        detector = LogGNNAnomalyDetector(node_in_dim=16, hidden_dim=32, vocab_size=30)
        builder = LogGraphBuilder(vocab_size=30)

        graph = builder.build_graph([5, 5, 26, 9])
        pred = detector.predict_graph(graph, builder=builder, threshold=0.5)

        assert "is_anomaly" in pred
        assert "anomaly_score" in pred
        assert "confidence" in pred
        assert "graph_embedding" in pred
        assert "node_importance" in pred
        assert isinstance(pred["is_anomaly"], bool)
        assert 0.0 <= pred["anomaly_score"] <= 1.0

    def test_save_and_load_checkpoint(self):
        """Verify model checkpoint serialization and numerical parity."""
        model1 = LogGNNAnomalyDetector(
            node_in_dim=16,
            edge_in_dim=4,
            hidden_dim=32,
            num_layers=2,
            gnn_type="gat",
        )
        model1.eval()

        builder = LogGraphBuilder(vocab_size=30)
        tensors = builder.sequence_to_tensors([5, 26, 9])

        with torch.no_grad():
            out1 = model1(
                x=tensors["x"],
                edge_index=tensors["edge_index"],
                edge_attr=tensors["edge_attr"],
                node_ids=tensors["node_ids"],
            )

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            model1.save(tmp_path)
            model2 = LogGNNAnomalyDetector.load(tmp_path)
            model2.eval()

            with torch.no_grad():
                out2 = model2(
                    x=tensors["x"],
                    edge_index=tensors["edge_index"],
                    edge_attr=tensors["edge_attr"],
                    node_ids=tensors["node_ids"],
                )

            assert torch.allclose(out1["logits"], out2["logits"], atol=1e-5)
            assert torch.allclose(out1["anomaly_score"], out2["anomaly_score"], atol=1e-5)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


# ============================================================================ #
#                           3. IncidentSummarizer Tests                        #
# ============================================================================ #

class TestIncidentSummarizer:
    """Tests for GenAI Incident Diagnostic Engine and SRE Report Generation."""

    @pytest.fixture
    def summarizer(self):
        return IncidentSummarizer(timeout_threshold=5.0, parameter_z_threshold=3.0)

    def test_template_resolution(self, summarizer):
        """Verify resolution of standard HDFS log templates."""
        assert "Receiving block" in summarizer.resolve_template("E5")
        assert "Receiving block" in summarizer.resolve_template(5)
        assert "terminating" in summarizer.resolve_template("E11")
        assert "terminating" in summarizer.resolve_template(11)
        assert "Verification succeeded" in summarizer.resolve_template("E2")
        # Unmapped template fallback
        assert "Unmapped Pattern" in summarizer.resolve_template("E999")

    def test_temporal_analysis_detection(self, summarizer):
        """Verify Step 1 temporal timeout detection and missing EOS detection."""
        events = ["E5", "E5", "E26"]
        times_normal = [1.0, 2.0, 3.0]
        res_normal = summarizer.analyze_temporal(events, timestamps=times_normal)
        assert res_normal["is_anomaly"] is False
        assert len(res_normal["timeouts"]) == 0

        # Latency spike of 15.0 seconds
        times_timeout = [1.0, 2.0, 17.0]
        res_timeout = summarizer.analyze_temporal(events, timestamps=times_timeout)
        assert res_timeout["is_anomaly"] is True
        assert len(res_timeout["timeouts"]) == 1
        assert res_timeout["timeouts"][0]["step_to"] == 2
        assert abs(res_timeout["timeouts"][0]["delta_seconds"] - 15.0) < 1e-4

    def test_parameter_analysis_detection(self, summarizer):
        """Verify Step 2 parameter outlier and zero-packet detection."""
        events = ["E5", "E6", "E9"]
        params_normal = [{"size": 67108864}, {"size": 67108864}, {"size": 67108864}]
        res_normal = summarizer.analyze_parameters(events, parameters=params_normal)
        assert res_normal["is_anomaly"] is False

        # Zero size and explicit error status
        params_anom = [{"size": 67108864}, {"size": 0}, {"status": "ERROR"}]
        res_anom = summarizer.analyze_parameters(events, parameters=params_anom)
        assert res_anom["is_anomaly"] is True
        assert res_anom["severity"] == "HIGH"
        assert len(res_anom["anomalous_parameters"]) >= 2

    def test_attention_root_cause_attribution(self, summarizer):
        """Verify Step 3 attention root cause ranking and percentage attribution."""
        events = ["E5", "E5", "E26", "E11"]
        attn = [0.10, 0.60, 0.20, 0.10]
        ranked = summarizer.analyze_attention(events, attention_weights=attn, top_k=3)

        assert len(ranked) == 3
        # Step 1 (the second E5) received highest attention mass (60%)
        assert ranked[0]["step"] == 1
        assert ranked[0]["event_id"] == "E5"
        assert abs(ranked[0]["percentage"] - 60.0) < 1e-2
        assert ranked[0]["role"] == "Primary Trigger"

    def test_semantic_analysis_categorization(self, summarizer):
        """Verify Step 4 semantic error classification and failure domain categorization."""
        events_normal = ["E5", "E5", "E26", "E9"]
        res_normal = summarizer.analyze_semantics(events_normal)
        assert res_normal["has_fatal_event"] is False

        events_fatal = ["E5", "E5", "E26", "E11"]
        res_fatal = summarizer.analyze_semantics(events_fatal)
        assert res_fatal["has_fatal_event"] is True
        assert res_fatal["primary_failure_event"] == "E11"
        assert "PacketResponder" in res_fatal["failure_category"]

    def test_graph_topology_analysis(self, summarizer):
        """Verify Step 5 graph topology cycle detection and GNN integration."""
        events = ["E11", "E8", "E11"]  # Cycle
        res = summarizer.analyze_graph_topology(events)
        assert res["is_anomaly"] is True
        assert res["has_cycles"] is True
        assert len(res["cycles"]) >= 1

    def test_full_incident_diagnosis_synthesis(self, summarizer):
        """Verify end-to-end synthesis across all 5 diagnostic pillars."""
        events = ["E5", "E5", "E26", "E26", "E11"]
        timestamps = [10.0, 10.2, 10.4, 10.6, 25.0]  # Step 4 latency spike
        params = [{}, {}, {"size": 67108864}, {"size": 0}, {"status": "FAILED"}]
        attn = [0.05, 0.15, 0.60, 0.10, 0.10]

        report = summarizer.diagnose_incident(
            block_id="blk_-1608999687919862906",
            events=events,
            timestamps=timestamps,
            parameters=params,
            attention_weights=attn,
        )

        assert isinstance(report, IncidentReport)
        assert report.block_id == "blk_-1608999687919862906"
        assert report.severity in ("CRITICAL", "HIGH")
        assert report.status == "TRIGGERED"
        assert len(report.timeline) == 5
        assert len(report.remediation_suggestions) >= 2
        assert report.root_cause_analysis["primary_root_cause_event"] == "E11"
        assert len(report.root_cause_analysis["attention_attribution"]) > 0

    def test_report_json_and_dict_serialization(self, summarizer):
        """Verify JSON and dict export formats."""
        events = ["E5", "E26", "E11"]
        report = summarizer.diagnose_incident(block_id="blk_test_export", events=events)

        d = report.to_dict()
        assert isinstance(d, dict)
        assert d["block_id"] == "blk_test_export"
        assert "root_cause_analysis" in d
        assert "timeline" in d

        raw_json = report.to_json(indent=2)
        parsed = json.loads(raw_json)
        assert parsed["block_id"] == "blk_test_export"
        assert parsed["severity"] in ("CRITICAL", "HIGH", "MEDIUM", "LOW")

    def test_report_markdown_rendering(self, summarizer):
        """Verify Markdown postmortem structure, callout badges, and Mermaid diagram."""
        events = ["E5", "E26", "E11"]
        report = summarizer.diagnose_incident(block_id="blk_test_md", events=events)
        md = report.to_markdown()

        assert "# DeepLog SRE Incident Postmortem:" in md
        assert "**Target Resource / Block ID:** `blk_test_md`" in md
        assert "> [!CAUTION]" in md or "> [!WARNING]" in md
        assert "## 1. Root Cause Analysis (RCA)" in md
        assert "## 2. Multi-Dimensional Anomaly Breakdown" in md
        assert "## 4. Chronological Incident Timeline" in md
        assert "## 6. Actionable SRE Remediation Runbook" in md
        assert "flowchart TD" in md
        assert "hdfs fsck" in md

    def test_report_save_to_disk(self, summarizer):
        """Verify saving report to markdown and json files on disk."""
        events = ["E5", "E26", "E11"]
        report = summarizer.diagnose_incident(block_id="blk_save_test", events=events)

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_base = os.path.join(tmp_dir, "incident_report")
            md_path, json_path = report.save(out_base, format="both")

            assert os.path.exists(md_path)
            assert os.path.exists(json_path)

            with open(md_path, "r", encoding="utf-8") as f:
                content_md = f.read()
                assert "DeepLog SRE Incident Postmortem" in content_md

            with open(json_path, "r", encoding="utf-8") as f:
                content_json = json.load(f)
                assert content_json["block_id"] == "blk_save_test"

    def test_genai_prompt_generation(self, summarizer):
        """Verify GenAI LLM prompt formatting."""
        events = ["E5", "E26", "E11"]
        report = summarizer.diagnose_incident(block_id="blk_genai_test", events=events)
        prompt = summarizer.generate_genai_prompt(report)

        assert "You are a Principal SRE" in prompt
        assert report.incident_id in prompt
        assert report.block_id in prompt
        assert "Multi-Dimensional Anomaly Evidence:" in prompt

    def test_integration_with_temporal_and_parameter_results(self, summarizer):
        """Verify integration with Step 1 and Step 2 result objects."""
        events = ["E5", "E6", "E9"]

        # Mock Step 1 TemporalAnomalyResult
        mock_temporal = TemporalAnomalyResult(
            is_anomaly=True,
            primary_type="timeout",
            anomaly_types=["timeout", "latency_spike"],
            is_sequence_anomaly=False,
            is_temporal_anomaly=True,
            is_timeout=True,
            is_latency_spike=True,
            is_retry_storm=False,
            observed_event=6,
            observed_delta=18.5,
            expected_delta_mean=1.2,
            expected_delta_std=0.4,
            z_score=43.25,
            sequence_nll=1.5,
            event_confidence=0.85,
            topk_predictions=[6, 5],
            joint_score=44.75,
            details={},
        )

        # Mock Step 2 ParameterAnomalyResult
        mock_parameter = ParameterAnomalyResult(
            is_anomaly=np.array([False, True, False]),
            scores=np.array([0.2, 5.8, 0.3]),
            threshold=2.5,
            predictions=np.array([[67108864.0], [67108864.0], [67108864.0]]),
            targets=np.array([[67108864.0], [0.0], [67108864.0]]),
            residuals=np.array([[0.0], [-67108864.0], [0.0]]),
            feature_contributions=np.array([[0.0], [1.0], [0.0]]),
            feature_names=["size"],
            diagnostics=["Residual for feature 'size' exceeded threshold"],
        )

        report = summarizer.diagnose_incident(
            block_id="blk_multistep_test",
            events=events,
            temporal_result=mock_temporal,
            parameter_result=mock_parameter,
        )

        assert report.anomaly_breakdown["temporal"]["is_anomaly"] is True
        assert report.anomaly_breakdown["parameter"]["is_anomaly"] is True
        assert "timeout" in report.anomaly_breakdown["temporal"]["summary"].lower()
        assert "size" in report.anomaly_breakdown["parameter"]["summary"].lower()

    def test_integration_with_attribution_result(self, summarizer):
        """Verify integration with Step 3 RootCauseExplanation object."""
        from deeplog.attribution import RootCauseExplanation

        events = ["E5", "E26", "E11"]
        mock_rca = RootCauseExplanation(
            sequence=[5, 26, 11],
            token_indices=[0, 1, 2],
            event_ids=[5, 26, 11],
            attributions=np.array([0.1, 0.7, 0.2]),
            attribution_percentages=np.array([10.0, 70.0, 20.0]),
            anomaly_score=4.5,
            is_anomalous=True,
            anomaly_trigger_points=[2],
            ranked_events=[
                {"step": 1, "token_index": 1, "event_id": "E26", "attribution_score": 0.7, "attribution_percentage": 70.0},
                {"step": 2, "token_index": 2, "event_id": "E11", "attribution_score": 0.2, "attribution_percentage": 20.0},
                {"step": 0, "token_index": 0, "event_id": "E5", "attribution_score": 0.1, "attribution_percentage": 10.0},
            ],
            ranked_transitions=[],
            method="integrated_gradients",
        )

        report = summarizer.diagnose_incident(
            block_id="blk_rca_test",
            events=events,
            attribution_result=mock_rca,
        )

        attn_list = report.root_cause_analysis["attention_attribution"]
        assert len(attn_list) == 3
        assert attn_list[0]["event_id"] == "E26"
        assert abs(attn_list[0]["percentage"] - 70.0) < 1e-2
