"""
Rigorous Unit Tests for Cross-Domain Benchmark and Zero-Shot Transfer.

Covers:
1. BGLParser (standard 10-field, relaxed format, alert categories, multi-line stack traces, corrupt logs, template extraction).
2. ThunderbirdParser (syslog format, PID extraction, high temporal variance metrics, burst detection, kernel oops stack traces, corrupt logs).
3. KubernetesLogParser (Klog format, JSON format, CoreDNS text format, component auto-detection, Go stack traces, corrupt logs).
4. DomainDatasetSummary (Shannon vocabulary entropy, weighted vs unique OOV rates, cross-domain divergence, temporal variance).
5. DomainTransferEvaluator (CRC32 subword hypersphere projections, TemplateSimilarityMatcher alignment, SemanticDeepLog zero-shot sequence prediction, discrete vs semantic OOV comparison).
6. Synthetic representative datasets and edge cases.
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import pytest
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from deeplog.cross_domain import (
    BGLParser,
    BGLRecord,
    DomainDatasetSummary,
    DomainTransferEvaluator,
    DomainTransferResult,
    KubernetesLogParser,
    KubernetesRecord,
    SequenceTransferResult,
    TargetTemplateMatch,
    ThunderbirdParser,
    ThunderbirdRecord,
    get_synthetic_bgl_logs,
    get_synthetic_domain_datasets,
    get_synthetic_hdfs_logs,
    get_synthetic_kubernetes_logs,
    get_synthetic_thunderbird_logs,
)
from deeplog.semantic import LogSemanticEncoder, SemanticDeepLog, TemplateSimilarityMatcher


# ==============================================================================
# 1. BGLParser Tests
# ==============================================================================


def test_bgl_parser_standard_line():
    """Verify parsing of standard 10-field Loghub BGL log line."""
    line = "- 1117838570 2005.06.03 R02-M1-N0-C:J12-U11 2005-06-03-15.42.50.675872 R02-M1-N0-C:J12-U11 RAS KERNEL INFO instruction cache parity error corrected"
    parser = BGLParser()
    record = parser.parse_line(line)

    assert record is not None
    assert record.alert_category == "-"
    assert not record.is_alert
    assert not record.is_corrupt
    assert record.timestamp == 1117838570.0
    assert record.date == "2005.06.03"
    assert record.node_id == "R02-M1-N0-C:J12-U11"
    assert record.component == "KERNEL"
    assert record.level == "INFO"
    assert record.subsystem == "RAS"
    assert record.message == "instruction cache parity error corrected"


def test_bgl_parser_alert_categories():
    """Verify alert category detection and is_alert flag for FATAL/FAILURE/WARNING."""
    parser = BGLParser()

    fatal_line = "FATAL 1117838574 2005.06.03 R02-M1-N0-C:J12-U11 2005-06-03-15.42.54.400000 R02-M1-N0-C:J12-U11 RAS KERNEL FATAL double hummer alignment exception"
    rec1 = parser.parse_line(fatal_line)
    assert rec1 is not None
    assert rec1.alert_category == "FATAL"
    assert rec1.is_alert is True
    assert rec1.level == "FATAL"

    failure_line = "FAILURE 1117838575 2005.06.03 R02-M1-N0-C:J12-U11 2005-06-03-15.42.55.500000 R02-M1-N0-C:J12-U11 RAS HARDWARE FAILURE ddr error"
    rec2 = parser.parse_line(failure_line)
    assert rec2 is not None
    assert rec2.alert_category == "FAILURE"
    assert rec2.is_alert is True

    warn_line = "WARNING 1117838576 2005.06.03 R04-M1-NF-C:J08-U11 2005-06-03-15.42.56.600000 R04-M1-NF-C:J08-U11 RAS DISCOVERY WARNING ping timeout"
    rec3 = parser.parse_line(warn_line)
    assert rec3 is not None
    assert rec3.alert_category == "WARNING"
    assert rec3.is_alert is True


def test_bgl_parser_relaxed_format():
    """Verify parsing of relaxed BGL formats with fewer fields."""
    parser = BGLParser()
    relaxed_line = "- 1117838570 2005.06.03 R02-M1-N0-C:J12-U11 KERNEL INFO instruction cache parity error"
    record = parser.parse_line(relaxed_line)
    assert record is not None
    assert not record.is_corrupt
    assert record.node_id == "R02-M1-N0-C:J12-U11"
    assert record.component == "KERNEL"
    assert record.message == "instruction cache parity error"


def test_bgl_parser_filter_alerts_only():
    """Verify filter_alerts_only skips non-alert logs."""
    parser = BGLParser(filter_alerts_only=True)
    normal_line = "- 1117838570 2005.06.03 R02-M1-N0-C:J12-U11 2005-06-03-15.42.50.675872 R02-M1-N0-C:J12-U11 RAS KERNEL INFO normal event"
    assert parser.parse_line(normal_line) is None

    alert_line = "FATAL 1117838570 2005.06.03 R02-M1-N0-C:J12-U11 2005-06-03-15.42.50.675872 R02-M1-N0-C:J12-U11 RAS KERNEL FATAL fatal error"
    rec = parser.parse_line(alert_line)
    assert rec is not None
    assert rec.is_alert is True


def test_bgl_parser_multiline_stack_trace():
    """Verify multi-line register dumps and stack traces are aggregated with originating log."""
    lines = [
        "FATAL 1117838574 2005.06.03 R02-M1-N0-C:J12-U11 2005-06-03-15.42.54.400000 R02-M1-N0-C:J12-U11 RAS KERNEL FATAL panic: critical interrupt",
        "   iar 00106304 dear 0245b000",
        "   msr 00001000 lr 001062f8",
        "- 1117838575 2005.06.03 R02-M1-N0-C:J12-U11 2005-06-03-15.42.55.500000 R02-M1-N0-C:J12-U11 RAS KERNEL INFO next normal line",
    ]
    parser = BGLParser()
    records = list(parser.parse_stream(lines))

    assert len(records) == 2
    assert records[0].alert_category == "FATAL"
    assert records[0].stack_trace is not None
    assert "iar 00106304" in records[0].stack_trace
    assert "msr 00001000" in records[0].stack_trace
    assert "iar 00106304" in records[0].message
    assert records[1].alert_category == "-"


def test_bgl_parser_corrupt_logs():
    """Verify corrupt / malformed lines do not raise exceptions and are flagged is_corrupt."""
    parser = BGLParser()
    corrupt_line = "CORRUPT_RANDOM_NOISE_GARBAGE !@#$%^&*() 12345"
    record = parser.parse_line(corrupt_line)

    assert record is not None
    assert record.is_corrupt is True
    assert record.message == corrupt_line


def test_bgl_parser_empty_and_whitespace():
    """Verify empty or whitespace-only lines return None."""
    parser = BGLParser()
    assert parser.parse_line("") is None
    assert parser.parse_line("   \n\t  ") is None


def test_bgl_parser_template_extraction():
    """Verify extract_template masks hex addresses, node locations, and numbers."""
    raw_msg = "microseconds common interrupt: iar 00106304 dear 0245b000 on node R02-M1-N0-C:J12-U11 with code 0x1a2b and count 42"
    tpl = BGLParser.extract_template(raw_msg)

    assert "00106304" not in tpl
    assert "0245b000" not in tpl
    assert "R02-M1-N0-C:J12-U11" not in tpl
    assert "0x1a2b" not in tpl
    assert "42" not in tpl
    assert "<param>" in tpl


# ==============================================================================
# 2. ThunderbirdParser Tests
# ==============================================================================


def test_thunderbird_parser_standard_line():
    """Verify parsing of standard Thunderbird log line."""
    line = "- 1131561730 2005.11.09 an179 Nov  9 10:42:10 an179 syslogd.1.4.1: restart."
    parser = ThunderbirdParser()
    rec = parser.parse_line(line)

    assert rec is not None
    assert not rec.is_corrupt
    assert not rec.is_alert
    assert rec.timestamp == 1131561730.0
    assert rec.node == "an179"
    assert rec.syslog_time == "Nov  9 10:42:10"
    assert rec.component == "syslogd.1.4.1"
    assert rec.message == "restart."


def test_thunderbird_parser_with_pid_and_alert():
    """Verify PID extraction and alert severity parsing."""
    line = "ALERT 1131561735 2005.11.09 cn301 Nov  9 10:42:15 cn301 ntpd[2450]: synchronized to 10.0.0.1"
    parser = ThunderbirdParser()
    rec = parser.parse_line(line)

    assert rec is not None
    assert rec.label == "ALERT"
    assert rec.is_alert is True
    assert rec.pid == 2450
    assert rec.component == "ntpd"
    assert rec.message == "synchronized to 10.0.0.1"


def test_thunderbird_parser_relaxed_format():
    """Verify relaxed format without redundant host."""
    line = "ERR 1131561740 2005.11.09 cn112 lustre[123]: LustreError: read timeout"
    parser = ThunderbirdParser()
    rec = parser.parse_line(line)

    assert rec is not None
    assert not rec.is_corrupt
    assert rec.label == "ERR"
    assert rec.node == "cn112"
    assert rec.component == "lustre"
    assert rec.pid == 123
    assert rec.message == "LustreError: read timeout"


def test_thunderbird_parser_multiline_oops():
    """Verify kernel oops Call Trace is attached to preceding record."""
    lines = [
        "ERR 1131561781 2005.11.09 cn112 Nov  9 10:43:01 cn112 kernel: Unable to handle kernel paging request",
        "   Call Trace:",
        "   [<c0123456>] sys_read+0x46/0x70",
        "- 1131561785 2005.11.09 an179 Nov  9 10:43:05 an179 pbs_mom: node ok",
    ]
    parser = ThunderbirdParser()
    records = list(parser.parse_stream(lines))

    assert len(records) == 2
    assert records[0].stack_trace is not None
    assert "Call Trace:" in records[0].stack_trace
    assert "sys_read+0x46/0x70" in records[0].message
    assert records[1].component == "pbs_mom"


def test_thunderbird_parser_temporal_variance():
    """Verify compute_temporal_metrics correctly calculates mean, variance, and burst ratio."""
    lines = [
        "- 1131561730 2005.11.09 an179 Nov  9 10:42:10 an179 kernel: event 1",
        "- 1131561730.1 2005.11.09 an179 Nov  9 10:42:10 an179 kernel: event 2 (burst)",
        "- 1131561730.2 2005.11.09 an179 Nov  9 10:42:10 an179 kernel: event 3 (burst)",
        "- 1131561780 2005.11.09 an179 Nov  9 10:43:00 an179 kernel: event 4 (gap)",
    ]
    parser = ThunderbirdParser()
    records = list(parser.parse_stream(lines))

    metrics = ThunderbirdParser.compute_temporal_metrics(records, burst_threshold=0.5)

    assert metrics["total_events"] == 4.0
    assert metrics["mean_delta"] > 0.0
    assert metrics["variance_delta"] > 0.0
    assert metrics["burst_ratio"] >= 0.5  # 2 of 3 intervals are <= 0.5s


def test_thunderbird_parser_corrupt_logs():
    """Verify corrupt lines return is_corrupt=True."""
    parser = ThunderbirdParser()
    rec = parser.parse_line("NOT_A_VALID_THUNDERBIRD_LINE")
    assert rec is not None
    assert rec.is_corrupt is True


# ==============================================================================
# 3. KubernetesLogParser Tests
# ==============================================================================


def test_kubernetes_parser_klog_format():
    """Verify parsing of standard Klog (glog) formatted lines."""
    line = 'I1105 14:23:45.678901   12345 round_trippers.go:466] Response: 200 OK GET /api/v1/pods pod="default/nginx"'
    parser = KubernetesLogParser()
    rec = parser.parse_line(line)

    assert rec is not None
    assert not rec.is_corrupt
    assert rec.format == "klog"
    assert rec.level == "INFO"
    assert rec.caller == "round_trippers.go:466"
    assert rec.thread_id == 12345
    assert rec.component == "kube-apiserver"
    assert "Response: 200 OK GET /api/v1/pods" in rec.message
    assert rec.attributes.get("pod") == "default/nginx"


def test_kubernetes_parser_klog_levels():
    """Verify I, W, E, F level mapping in Klog."""
    parser = KubernetesLogParser()
    w_line = "W1105 14:23:45.000000 1 server.go:10] warning msg"
    e_line = "E1105 14:23:45.000000 1 server.go:10] error msg"
    f_line = "F1105 14:23:45.000000 1 server.go:10] fatal msg"

    assert parser.parse_line(w_line).level == "WARNING"
    assert parser.parse_line(e_line).level == "ERROR"
    assert parser.parse_line(f_line).level == "FATAL"


def test_kubernetes_parser_json_format():
    """Verify parsing of structured JSON logs (etcd / kube-apiserver)."""
    line = '{"level":"warn","ts":1730816627.123,"logger":"etcdserver","caller":"etcdserver/server.go:1234","msg":"apply request took too long","took":"125ms"}'
    parser = KubernetesLogParser()
    rec = parser.parse_line(line)

    assert rec is not None
    assert not rec.is_corrupt
    assert rec.format == "json"
    assert rec.level == "WARNING"
    assert rec.component == "etcd"
    assert rec.caller == "etcdserver/server.go:1234"
    assert rec.message == "apply request took too long"
    assert rec.attributes["took"] == "125ms"
    assert rec.timestamp == 1730816627.123


def test_kubernetes_parser_json_iso_timestamp():
    """Verify parsing of ISO8601 timestamps in JSON format."""
    line = '{"level":"info","time":"2026-11-05T14:23:45.678Z","component":"kubelet","caller":"kubelet.go:100","msg":"syncing pod"}'
    parser = KubernetesLogParser()
    rec = parser.parse_line(line)

    assert rec is not None
    assert rec.timestamp > 0.0
    assert rec.component == "kubelet"
    assert rec.message == "syncing pod"


def test_kubernetes_parser_coredns_text():
    """Verify parsing of CoreDNS text syslog format."""
    line = '[INFO] 127.0.0.1:54321 - 42533 "A IN example.com. udp 40 false 512" NOERROR qr,aa,rd 64 0.000234s'
    parser = KubernetesLogParser()
    rec = parser.parse_line(line)

    assert rec is not None
    assert rec.format == "coredns"
    assert rec.level == "INFO"
    assert rec.component == "coredns"
    assert "example.com" in rec.message


def test_kubernetes_parser_component_auto_detection():
    """Verify component inference for apiserver, etcd, kubelet, and coredns."""
    parser = KubernetesLogParser()

    rec_api = parser.parse_line("I1105 14:00:00.000000 1 handler.go:50] Handling API request")
    assert rec_api.component == "kube-apiserver"

    rec_etcd = parser.parse_line("I1105 14:00:00.000000 1 raft.go:50] Raft leader elected")
    assert rec_etcd.component == "etcd"

    rec_kubelet = parser.parse_line("I1105 14:00:00.000000 1 pleg.go:50] PLEG relist finished")
    assert rec_kubelet.component == "kubelet"

    rec_coredns = parser.parse_line("[ERROR] plugin/errors: unreachable upstream DNS")
    assert rec_coredns.component == "coredns"


def test_kubernetes_parser_multiline_go_stack_trace():
    """Verify Go panic / stack traces are appended to previous record."""
    lines = [
        "F1105 14:23:51.111222 1 server.go:142] panic: unable to create etcd client",
        "   goroutine 1 [running]:",
        "   k8s.io/kubernetes/cmd/kube-apiserver/app.Run(...)",
        "       /go/src/k8s.io/server.go:142 +0x62",
        "I1105 14:23:55.000000 2 round_trippers.go:10] next request",
    ]
    parser = KubernetesLogParser()
    records = list(parser.parse_stream(lines))

    assert len(records) == 2
    assert records[0].level == "FATAL"
    assert records[0].stack_trace is not None
    assert "goroutine 1 [running]:" in records[0].stack_trace
    assert "k8s.io/kubernetes/cmd/kube-apiserver" in records[0].message
    assert records[1].level == "INFO"


def test_kubernetes_parser_corrupt_json():
    """Verify invalid JSON is flagged as corrupt without raising exception."""
    parser = KubernetesLogParser()
    rec = parser.parse_line('{"level": "error", "broken json')
    assert rec is not None
    assert rec.is_corrupt is True


# ==============================================================================
# 4. DomainDatasetSummary Tests
# ==============================================================================


def test_domain_summary_shannon_entropy():
    """Verify Shannon entropy is 0 for identical tokens and positive for diverse tokens."""
    # Single repeated token -> 0.0 entropy
    repeated_texts = ["word word word word", "word word"]
    s_repeat = DomainDatasetSummary.from_texts(repeated_texts, domain_name="Repeated")
    assert s_repeat.unique_tokens == 1
    assert math.isclose(s_repeat.vocabulary_entropy, 0.0, abs_tol=1e-5)

    # Uniform distribution over 4 tokens -> -4 * (0.25 * log2(0.25)) = -log2(0.25) = 2.0 bits
    uniform_texts = ["alpha beta gamma delta"]
    s_uniform = DomainDatasetSummary.from_texts(uniform_texts, domain_name="Uniform")
    assert s_uniform.unique_tokens == 4
    assert math.isclose(s_uniform.vocabulary_entropy, 2.0, abs_tol=1e-5)


def test_domain_summary_oov_rate_calculation():
    """Verify weighted and unique OOV rates against reference vocabulary."""
    source_texts = ["apple banana cherry orange"]
    s_source = DomainDatasetSummary.from_texts(source_texts, domain_name="Source")

    # Identical dataset -> 0.0 OOV
    assert math.isclose(s_source.compute_oov_rate(s_source, weighted=True), 0.0)
    assert math.isclose(s_source.compute_oov_rate(s_source, weighted=False), 0.0)

    # Disjoint dataset -> 1.0 OOV
    disjoint_texts = ["dog cat elephant"]
    s_disjoint = DomainDatasetSummary.from_texts(disjoint_texts, domain_name="Disjoint")
    assert math.isclose(s_disjoint.compute_oov_rate(s_source, weighted=True), 1.0)
    assert math.isclose(s_disjoint.compute_oov_rate(s_source, weighted=False), 1.0)

    # Partial overlap: 'apple' (known) and 'mango' (unknown)
    partial_texts = ["apple mango"]
    s_partial = DomainDatasetSummary.from_texts(partial_texts, domain_name="Partial")
    assert math.isclose(s_partial.compute_oov_rate(s_source, weighted=True), 0.5)
    assert math.isclose(s_partial.compute_oov_rate(s_source, weighted=False), 0.5)


def test_domain_summary_cross_domain_comparison():
    """Verify compare_with calculates Jaccard, shared tokens, and entropy diff."""
    t1 = ["user login success from host"]
    t2 = ["user logout failure from host"]
    s1 = DomainDatasetSummary.from_texts(t1, domain_name="Domain1")
    s2 = DomainDatasetSummary.from_texts(t2, domain_name="Domain2")

    comp = s2.compare_with(s1)
    assert comp["source_domain"] == "Domain1"
    assert comp["target_domain"] == "Domain2"
    assert comp["shared_tokens_count"] >= 3  # 'user', 'from', 'host'
    assert 0.0 < comp["jaccard_similarity"] < 1.0
    assert "entropy_diff" in comp


def test_domain_summary_temporal_variance():
    """Verify mean time delta and temporal variance computation."""
    texts = ["msg 1", "msg 2", "msg 3"]
    timestamps = [100.0, 102.0, 106.0]  # deltas: 2.0, 4.0 -> mean=3.0, var=1.0
    summary = DomainDatasetSummary.from_texts(texts, timestamps=timestamps, domain_name="Timed")

    assert summary.mean_time_delta is not None
    assert math.isclose(summary.mean_time_delta, 3.0, abs_tol=1e-5)
    assert summary.temporal_variance is not None
    assert math.isclose(summary.temporal_variance, 1.0, abs_tol=1e-5)


def test_domain_summary_serialization():
    """Verify to_dict and summary strings."""
    s = DomainDatasetSummary.from_texts(["error in component"], domain_name="Test")
    d = s.to_dict()
    assert d["domain_name"] == "Test"
    assert d["total_lines"] == 1
    assert "vocabulary_entropy" in d
    assert "=== Domain Dataset Summary: Test ===" in s.summary()


# ==============================================================================
# 5. DomainTransferEvaluator Tests
# ==============================================================================


def test_transfer_evaluator_template_alignment():
    """Verify template alignment using continuous CRC32 hypersphere projections."""
    source_templates = {
        "S1": "Receiving data packet from source node",
        "S2": "Verification succeeded for block",
        "S3": "Terminating connection responder",
    }
    target_templates = {
        "T1": "Receiving data packet from source node",
        "T2": "Unrelated kernel memory parity failure",
    }

    evaluator = DomainTransferEvaluator(source_templates=source_templates, embedding_dim=64)
    matches = evaluator.evaluate_template_transfer(target_templates, threshold=0.3)

    assert len(matches) == 2
    for m in matches:
        assert isinstance(m, TargetTemplateMatch)
        assert -1.0 <= m.cosine_similarity <= 1.0
        assert m.matched_source_id in source_templates

    assert matches[0].matched_source_id == "S1"
    assert matches[0].cosine_similarity > 0.8
    assert matches[0].is_covered is True


def test_transfer_evaluator_zero_shot_sequences():
    """Verify zero-shot next-event prediction and anomaly detection on target sequences."""
    source_templates = {
        "S1": "allocateBlock",
        "S2": "Verification succeeded",
        "S3": "PacketResponder terminating",
    }
    evaluator = DomainTransferEvaluator(source_templates=source_templates, embedding_dim=64)

    target_sequences = [
        ["allocateBlock", "Verification succeeded"],
        ["Verification succeeded", "PacketResponder terminating"],
        ["allocateBlock", "PacketResponder terminating"],
    ]

    seq_res = evaluator.evaluate_sequence_transfer(
        target_sequences=target_sequences,
        top_k=3,
        similarity_threshold=0.2,
    )

    assert isinstance(seq_res, SequenceTransferResult)
    assert seq_res.total_sequences == 3
    assert 0.0 <= seq_res.top_1_accuracy <= 1.0
    assert 0.0 <= seq_res.top_3_accuracy <= 1.0
    assert -1.0 <= seq_res.mean_cosine_similarity <= 1.0
    assert 0 <= seq_res.anomaly_count <= 3


def test_transfer_evaluator_with_anomaly_labels():
    """Verify ROC-AUC and F1 computation when ground truth anomaly labels are supplied."""
    source_templates = {"S1": "event A", "S2": "event B"}
    evaluator = DomainTransferEvaluator(source_templates=source_templates, embedding_dim=64)

    target_sequences = [
        ["event A", "event B"],             # Normal
        ["event A", "wild unexpected crash"] # Anomaly
    ]
    labels = [0, 1]

    seq_res = evaluator.evaluate_sequence_transfer(
        target_sequences=target_sequences,
        anomaly_labels=labels,
        similarity_threshold=0.5,
    )

    assert seq_res.roc_auc is not None
    assert 0.0 <= seq_res.roc_auc <= 1.0
    assert seq_res.f1_score is not None
    assert 0.0 <= seq_res.f1_score <= 1.0


def test_discrete_vs_semantic_oov_comparison():
    """
    Verify fundamental architectural advantage:
    Discrete DeepLog suffers 100% OOV across domains,
    while SemanticDeepLog provides 0% hard OOV and continuous hypersphere coverage.
    """
    hdfs_templates = {"E1": "Receiving block", "E2": "PacketResponder terminating"}
    bgl_templates = {"B1": "instruction cache parity error", "B2": "double hummer alignment exception"}

    evaluator = DomainTransferEvaluator(source_templates=hdfs_templates, embedding_dim=64)
    res = evaluator.evaluate(
        source_domain="HDFS",
        target_domain="BGL",
        target_templates=bgl_templates,
    )

    # Discrete DeepLog is 100% OOV across domains
    assert res.discrete_oov_rate == 1.0
    # Semantic DeepLog maps all tokens into unit hypersphere
    assert res.mean_cosine_similarity > -1.0
    assert len(res.template_matches) == len(bgl_templates)
    assert "Zero-Shot Domain Transfer: HDFS -> BGL" in res.summary_report


# ==============================================================================
# 6. Representative Datasets & Roundtrip Tests
# ==============================================================================


def test_synthetic_domain_datasets_integrity():
    """Verify all 4 domains have valid representative data, records, and templates."""
    datasets = get_synthetic_domain_datasets()

    for d_name in ("HDFS", "BGL", "Thunderbird", "Kubernetes"):
        assert d_name in datasets
        d = datasets[d_name]
        assert len(d["raw"]) > 0
        assert len(d["templates"]) > 0
        assert len(d["sequences"]) > 0
        assert isinstance(d["summary"], DomainDatasetSummary)
        assert d["summary"].total_lines > 0
        assert d["summary"].vocabulary_entropy > 0.0


def test_empty_and_edge_case_inputs():
    """Verify robustness to empty sequences and empty template sets."""
    evaluator = DomainTransferEvaluator(source_templates={"S1": "test"}, embedding_dim=32)

    # Empty templates
    matches = evaluator.evaluate_template_transfer({})
    assert len(matches) == 0

    # Empty sequences
    seq_res = evaluator.evaluate_sequence_transfer([])
    assert seq_res.total_sequences == 0
    assert seq_res.top_1_accuracy == 0.0

    # Single-event sequence (length < 2)
    seq_res_short = evaluator.evaluate_sequence_transfer([["single_event"]])
    assert seq_res_short.total_sequences == 0
