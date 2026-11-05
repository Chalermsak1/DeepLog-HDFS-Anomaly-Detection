"""
Unit Tests for DeepLog-Omni Frontier 2: Multi-Modal OTel Convergence (MELT + Profiling)
======================================================================================
Verifies:
1. Telemetry Data Models: MetricPoint, TraceSpan, LogRecord, ProfileSample, TelemetryWindow.
2. Stream Synchronization: UnifiedTelemetryStream temporal alignment, trace correlation, and tensor batching.
3. Multimodal Fusion: TelemetryFusionModel cross-attention network, unified state z_system, gating & heads.
4. Anomaly Attribution: CrossModalAttributor isolation across Log Errors, Metric Saturation, Trace Bottlenecks, and Profile Hotspots.
5. Invariants: Attribution sum-to-one (100%), gradient flow, and JSON serializability.
"""

import json
import math
import pytest
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from deeplog.otel_melt import (
    MetricPoint,
    MetricType,
    TraceSpan,
    LogRecord,
    ProfileSample,
    TelemetryWindow,
    UnifiedTelemetryStream,
    TelemetryFusionModel,
    TelemetryFusionOutput,
    CrossModalAttributor,
    CrossModalAttributionResult,
    AnomalyDriver,
)


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def small_fusion_model():
    """Small deterministic TelemetryFusionModel for unit testing."""
    torch.manual_seed(42)
    model = TelemetryFusionModel(
        d_model=32,
        nhead=2,
        num_fusion_layers=2,
        log_vocab_size=100,
        num_metrics=8,
        trace_feat_dim=8,
        profile_feat_dim=16,
        dim_feedforward=64,
        dropout=0.0,
    )
    model.eval()
    return model


@pytest.fixture
def empty_stream():
    """Fresh empty UnifiedTelemetryStream."""
    return UnifiedTelemetryStream(
        default_window_size=10.0,
        default_step_size=5.0,
        metric_steps=5,
        num_metrics=8,
        max_logs=16,
        max_spans=8,
        max_profiles=8,
        trace_feat_dim=8,
        profile_feat_dim=16,
    )


@pytest.fixture
def sample_telemetry_data():
    """Diverse synthetic MELT + continuous profiling records."""
    t0 = 1700000000.0

    metrics = [
        MetricPoint(name="cpu_util", timestamp=t0 + 1.0, value=45.2, labels={"service": "cart", "host": "srv-1"}),
        MetricPoint(name="cpu_util", timestamp=t0 + 3.0, value=88.7, labels={"service": "cart", "host": "srv-1"}),
        MetricPoint(name="mem_bytes", timestamp=t0 + 2.0, value=1024.0 * 1024.0 * 500.0, labels={"service": "cart"}),
    ]

    spans = [
        TraceSpan(
            span_id="span-001",
            trace_id="trace-abc",
            parent_id=None,
            service="gateway",
            name="GET /cart",
            start_time=t0 + 0.5,
            end_time=t0 + 1.2,
            duration=0.7,
            status_code="OK",
        ),
        TraceSpan(
            span_id="span-002",
            trace_id="trace-abc",
            parent_id="span-001",
            service="cart",
            name="RPC GetItems",
            start_time=t0 + 0.6,
            end_time=t0 + 1.1,
            duration=0.5,
            status_code="ERROR",
            attributes={"error": True, "http.status_code": 500},
        ),
    ]

    logs = [
        LogRecord(
            event_id=12,
            timestamp=t0 + 0.7,
            template="Incoming request to <*> from <*>",
            level="INFO",
            service="gateway",
            trace_id="trace-abc",
            span_id="span-001",
        ),
        LogRecord(
            event_id=45,
            timestamp=t0 + 0.9,
            template="Database connection timed out after <*> ms",
            level="ERROR",
            service="cart",
            trace_id="trace-abc",
            span_id="span-002",
        ),
    ]

    profiles = [
        ProfileSample(
            call_stack=["main", "ServeHTTP", "runtime.mallocgc"],
            weight=1250.0,
            timestamp=t0 + 1.5,
            sample_type="cpu",
            service="cart",
        ),
        ProfileSample(
            call_stack=["main", "ServeHTTP", "cart.QueryDB", "sync.runtime_Semacquire"],
            weight=4500.0,
            timestamp=t0 + 2.0,
            sample_type="cpu",
            service="cart",
        ),
    ]

    return {"metrics": metrics, "spans": spans, "logs": logs, "profiles": profiles, "t0": t0}


# ==============================================================================
# 1. Telemetry Data Models Tests
# ==============================================================================


class TestTelemetryDataModels:
    """Tests for MetricPoint, TraceSpan, LogRecord, ProfileSample, and TelemetryWindow."""

    def test_metric_point_properties_and_serialization(self):
        m = MetricPoint(
            name="http_requests_total",
            timestamp=100.0,
            value=42.0,
            labels={"method": "GET", "status": "200"},
            metric_type="counter",
            unit="requests",
        )
        assert m.name == "http_requests_total"
        assert m.value == 42.0
        assert m.metric_type == "counter"
        assert m.identifier() == 'http_requests_total{method="GET",status="200"}'

        d = m.to_dict()
        assert d["value"] == 42.0
        m_restored = MetricPoint.from_dict(d)
        assert m_restored.name == m.name
        assert m_restored.value == m.value
        assert m_restored.labels == m.labels

    def test_trace_span_duration_and_error_detection(self):
        # Auto duration calculation
        span_ok = TraceSpan(
            span_id="sp1",
            trace_id="tr1",
            start_time=10.0,
            end_time=12.5,
            status_code="OK",
        )
        assert span_ok.duration == pytest.approx(2.5)
        assert not span_ok.is_error

        # Status code error
        span_err1 = TraceSpan(
            span_id="sp2",
            trace_id="tr1",
            status_code="ERROR",
        )
        assert span_err1.is_error

        # HTTP 500 error in attributes
        span_err2 = TraceSpan(
            span_id="sp3",
            trace_id="tr1",
            attributes={"http.status_code": 503},
        )
        assert span_err2.is_error

        # Serialization
        d = span_ok.to_dict()
        s_restored = TraceSpan.from_dict(d)
        assert s_restored.span_id == "sp1"
        assert s_restored.duration == pytest.approx(2.5)

    def test_log_record_levels_and_error_classification(self):
        log_info = LogRecord(
            event_id=5,
            timestamp=100.0,
            template="Service initialized successfully",
            level="INFO",
        )
        assert not log_info.is_error

        log_err = LogRecord(
            event_id=18,
            timestamp=101.0,
            template="Connection to redis failed",
            level="ERROR",
        )
        assert log_err.is_error

        log_keyword = LogRecord(
            event_id=19,
            timestamp=102.0,
            template="Unhandled panic during deserialization",
            level="WARN",
        )
        assert log_keyword.is_error  # 'panic' triggers error classification

        d = log_info.to_dict()
        l_restored = LogRecord.from_dict(d)
        assert l_restored.event_id == 5
        assert l_restored.template == log_info.template

    def test_profile_sample_flamegraph_frames(self):
        sample = ProfileSample(
            call_stack=["main", "http.Handle", "db.Query", "pg.parse"],
            weight=8200.0,
            timestamp=50.0,
            sample_type="cpu",
            service="orders",
        )
        assert sample.leaf_frame == "pg.parse"
        assert sample.stack_str() == "main;http.Handle;db.Query;pg.parse"
        assert sample.stack_str("/") == "main/http.Handle/db.Query/pg.parse"

        empty_sample = ProfileSample(call_stack=[], weight=0.0)
        assert empty_sample.leaf_frame == "<empty>"

        d = sample.to_dict()
        p_restored = ProfileSample.from_dict(d)
        assert p_restored.leaf_frame == "pg.parse"
        assert p_restored.weight == 8200.0

    def test_telemetry_window_properties(self):
        w = TelemetryWindow(
            window_id="w1",
            start_time=10.0,
            end_time=25.0,
        )
        assert w.duration == pytest.approx(15.0)
        assert w.is_empty
        assert w.counts() == {"metrics": 0, "spans": 0, "logs": 0, "profiles": 0}


# ==============================================================================
# 2. Unified Telemetry Stream Tests
# ==============================================================================


class TestUnifiedTelemetryStream:
    """Tests for stream ingestion, trace-correlation, windowing, and tensor compilation."""

    def test_ingestion_and_time_bounds(self, empty_stream, sample_telemetry_data):
        stream = empty_stream
        assert stream.get_time_bounds() == (0.0, 0.0)

        stream.add_batch(
            metrics=sample_telemetry_data["metrics"],
            spans=sample_telemetry_data["spans"],
            logs=sample_telemetry_data["logs"],
            profiles=sample_telemetry_data["profiles"],
        )

        min_t, max_t = stream.get_time_bounds()
        t0 = sample_telemetry_data["t0"]
        assert min_t == pytest.approx(t0 + 0.5)
        assert max_t == pytest.approx(t0 + 3.0)

    def test_trace_correlation_linking(self, empty_stream, sample_telemetry_data):
        stream = empty_stream
        stream.add_batch(
            spans=sample_telemetry_data["spans"],
            logs=sample_telemetry_data["logs"],
        )

        corr = stream.get_trace_correlation("trace-abc")
        assert corr["num_spans"] == 2
        assert corr["num_logs"] == 2
        assert corr["has_error"] is True

        corr_empty = stream.get_trace_correlation("trace-unknown")
        assert corr_empty["num_spans"] == 0
        assert corr_empty["has_error"] is False

    def test_sliding_window_partitioning(self, empty_stream, sample_telemetry_data):
        stream = empty_stream
        stream.add_batch(
            metrics=sample_telemetry_data["metrics"],
            spans=sample_telemetry_data["spans"],
            logs=sample_telemetry_data["logs"],
            profiles=sample_telemetry_data["profiles"],
        )

        # 1.0 second windows with 0.5 second step
        windows = stream.get_windows(window_size_sec=1.0, step_sec=0.5)
        assert len(windows) > 0
        for w in windows:
            assert w.duration == pytest.approx(1.0)

        # Slicing specific window
        t0 = sample_telemetry_data["t0"]
        win_sub = stream.get_window(start_time=t0, end_time=t0 + 1.5)
        assert len(win_sub.spans) >= 1
        assert len(win_sub.logs) >= 1

    def test_build_tensor_batch_shapes_and_types(self, empty_stream, sample_telemetry_data):
        stream = empty_stream
        stream.add_batch(
            metrics=sample_telemetry_data["metrics"],
            spans=sample_telemetry_data["spans"],
            logs=sample_telemetry_data["logs"],
            profiles=sample_telemetry_data["profiles"],
        )

        t0 = sample_telemetry_data["t0"]
        w1 = stream.get_window(start_time=t0, end_time=t0 + 2.0)
        w2 = stream.get_window(start_time=t0 + 1.0, end_time=t0 + 3.5)

        batch = stream.build_tensor_batch([w1, w2])
        assert batch["logs"].shape == (2, stream.max_logs)
        assert batch["log_mask"].shape == (2, stream.max_logs)
        assert batch["metrics"].shape == (2, stream.metric_steps, stream.num_metrics)
        assert batch["metric_mask"].shape == (2, stream.metric_steps)
        assert batch["traces"].shape == (2, stream.max_spans, stream.trace_feat_dim)
        assert batch["trace_mask"].shape == (2, stream.max_spans)
        assert batch["profiles"].shape == (2, stream.max_profiles, stream.profile_feat_dim)
        assert batch["profile_mask"].shape == (2, stream.max_profiles)

        # Data types
        assert batch["logs"].dtype == torch.int64
        assert batch["log_mask"].dtype == torch.bool
        assert batch["metrics"].dtype == torch.float32

    def test_build_tensor_batch_empty_stream(self, empty_stream):
        # Empty windows list should return batch size 0 without crash
        batch = empty_stream.build_tensor_batch([])
        assert batch["logs"].shape == (0, empty_stream.max_logs)
        assert batch["metrics"].shape == (0, empty_stream.metric_steps, empty_stream.num_metrics)


# ==============================================================================
# 3. Telemetry Fusion Model Tests
# ==============================================================================


class TestTelemetryFusionModel:
    """Tests for TelemetryFusionModel forward pass, attention extraction, and gradients."""

    def test_forward_pass_all_modalities(self, small_fusion_model, empty_stream, sample_telemetry_data):
        model = small_fusion_model
        empty_stream.add_batch(
            metrics=sample_telemetry_data["metrics"],
            spans=sample_telemetry_data["spans"],
            logs=sample_telemetry_data["logs"],
            profiles=sample_telemetry_data["profiles"],
        )

        t0 = sample_telemetry_data["t0"]
        win = empty_stream.get_window(t0, t0 + 5.0)
        batch = empty_stream.build_tensor_batch([win])

        out = model(batch=batch, return_attention=True)
        assert isinstance(out, TelemetryFusionOutput)

        # Unified system state z_system
        assert out.z_system.shape == (1, model.d_model)
        assert not torch.isnan(out.z_system).any()

        # Anomaly outputs
        assert out.anomaly_logits.shape == (1, 1)
        assert out.anomaly_probs.shape == (1, 1)
        assert 0.0 <= out.anomaly_probs.item() <= 1.0

        # Modality gating weights
        assert out.modality_weights.shape == (1, 4)
        assert out.modality_weights.sum().item() == pytest.approx(1.0, abs=1e-5)

        # Modal token slices
        assert "logs" in out.modal_token_slices
        assert "metrics" in out.modal_token_slices
        assert "traces" in out.modal_token_slices
        assert "profiles" in out.modal_token_slices

        # Attention weights
        assert out.attention_weights is not None
        assert out.attention_weights.shape[0] == 1
        assert out.attention_weights.sum(dim=-1).item() == pytest.approx(1.0, abs=1e-4)

    def test_forward_pass_missing_modalities(self, small_fusion_model):
        model = small_fusion_model
        B = 2
        # Provide only logs and metrics
        logs = torch.randint(1, 50, (B, 10))
        metrics = torch.randn(B, 5, model.num_metrics)

        out = model(logs=logs, metrics=metrics, return_attention=True)
        assert out.z_system.shape == (B, model.d_model)
        assert "logs" in out.modal_token_slices
        assert "metrics" in out.modal_token_slices
        assert "traces" not in out.modal_token_slices
        assert "profiles" not in out.modal_token_slices

    def test_gradient_flow_end_to_end(self, small_fusion_model, empty_stream, sample_telemetry_data):
        model = small_fusion_model
        model.train()
        empty_stream.add_batch(
            metrics=sample_telemetry_data["metrics"],
            spans=sample_telemetry_data["spans"],
            logs=sample_telemetry_data["logs"],
            profiles=sample_telemetry_data["profiles"],
        )

        win = empty_stream.get_window(0.0, 1e10)
        batch = empty_stream.build_tensor_batch([win])

        out = model(batch=batch)
        loss = F.binary_cross_entropy(out.anomaly_probs, torch.tensor([[1.0]]))
        loss.backward()

        # Check gradients exist on modal projection layers
        assert model.metric_proj[0].weight.grad is not None
        assert model.trace_proj[0].weight.grad is not None
        assert model.profile_proj[0].weight.grad is not None
        assert model.log_embedding.weight.grad is not None
        assert not torch.isnan(model.q_sys.grad).any()


# ==============================================================================
# 4. Cross-Modal Attributor Tests
# ==============================================================================


class TestCrossModalAttributor:
    """Tests for root-cause attribution, sum-to-one invariants, and synthetic failure localization."""

    def test_attribution_sum_normalization_across_all_methods(self, small_fusion_model, empty_stream, sample_telemetry_data):
        model = small_fusion_model
        attributor = CrossModalAttributor(model=model, anomaly_threshold=0.5)

        empty_stream.add_batch(
            metrics=sample_telemetry_data["metrics"],
            spans=sample_telemetry_data["spans"],
            logs=sample_telemetry_data["logs"],
            profiles=sample_telemetry_data["profiles"],
        )
        win = empty_stream.get_window(0.0, 1e10)
        batch = empty_stream.build_tensor_batch([win])

        methods = ["attention", "gradient", "integrated_gradients", "ablation", "combined"]
        for method in methods:
            result = attributor.attribute(batch=batch, method=method, window=win)
            assert isinstance(result, CrossModalAttributionResult)

            # Sum of contributions must be strictly 1.0
            total_contrib = sum(result.modal_contributions.values())
            assert total_contrib == pytest.approx(1.0, abs=1e-5), f"Failed for {method}"

            # Sum of percentages must be 100%
            total_pct = sum(result.modal_percentages.values())
            assert total_pct == pytest.approx(100.0, abs=0.1), f"Failed for {method}"

            # Keys must cover MELT + Profiling
            for k in ["logs", "metrics", "traces", "profiles"]:
                assert k in result.modal_contributions
                assert 0.0 <= result.modal_contributions[k] <= 1.0

    def test_synthetic_localization_log_errors(self, small_fusion_model, empty_stream):
        """Case A: Inject extreme log error spike -> primary driver must be LOG_ERRORS."""
        model = small_fusion_model
        attributor = CrossModalAttributor(model=model, anomaly_threshold=0.0)

        # Baseline metrics and traces, but inject 10 severe error logs
        t0 = 100.0
        logs = [
            LogRecord(event_id=99, timestamp=t0 + i * 0.1, template="Fatal database crash", level="CRITICAL")
            for i in range(10)
        ]
        empty_stream.add_batch(logs=logs)
        win = empty_stream.get_window(t0, t0 + 2.0)
        batch = empty_stream.build_tensor_batch([win])

        # Test attribution
        result = attributor.attribute(batch=batch, method="attention", window=win)
        assert result.is_anomalous
        # Logs should dominate attention since only logs are present in window
        assert result.primary_driver == AnomalyDriver.LOG_ERRORS.value
        assert result.modal_contributions["logs"] > 0.5
        assert len(result.evidence["top_log_templates"]) > 0

    def test_synthetic_localization_metric_saturation(self, small_fusion_model, empty_stream):
        """Case B: Massive metric spike -> primary driver must be METRIC_SATURATION."""
        model = small_fusion_model
        attributor = CrossModalAttributor(model=model, anomaly_threshold=0.0)

        t0 = 200.0
        # Inject massive metric saturation (e.g., 500.0 on all channels)
        metrics = [
            MetricPoint(name=f"saturation_chan_{i}", timestamp=t0 + 0.5, value=999.0)
            for i in range(empty_stream.num_metrics)
        ]
        empty_stream.add_batch(metrics=metrics)
        win = empty_stream.get_window(t0, t0 + 2.0)
        batch = empty_stream.build_tensor_batch([win])

        result = attributor.attribute(batch=batch, method="attention", window=win)
        assert result.is_anomalous
        assert result.primary_driver == AnomalyDriver.METRIC_SATURATION.value
        assert result.modal_contributions["metrics"] > 0.5
        assert len(result.evidence["saturated_metrics"]) > 0

    def test_synthetic_localization_trace_bottleneck(self, small_fusion_model, empty_stream):
        """Case C: Massive span latencies & errors -> primary driver must be TRACE_BOTTLENECK."""
        model = small_fusion_model
        attributor = CrossModalAttributor(model=model, anomaly_threshold=0.0)

        t0 = 300.0
        spans = [
            TraceSpan(
                span_id=f"sp-{i}",
                trace_id="tr-spike",
                start_time=t0,
                end_time=t0 + 30.0,
                duration=30.0,
                status_code="ERROR",
                attributes={"error": True, "http.status_code": 504},
            )
            for i in range(5)
        ]
        empty_stream.add_batch(spans=spans)
        win = empty_stream.get_window(t0, t0 + 35.0)
        batch = empty_stream.build_tensor_batch([win])

        result = attributor.attribute(batch=batch, method="attention", window=win)
        assert result.is_anomalous
        assert result.primary_driver == AnomalyDriver.TRACE_BOTTLENECK.value
        assert result.modal_contributions["traces"] > 0.5
        assert len(result.evidence["bottleneck_spans"]) > 0

    def test_synthetic_localization_profile_hotspot(self, small_fusion_model, empty_stream):
        """Case D: Heavy flamegraph hotspot stack -> primary driver must be PROFILE_HOTSPOT."""
        model = small_fusion_model
        attributor = CrossModalAttributor(model=model, anomaly_threshold=0.0)

        t0 = 400.0
        profiles = [
            ProfileSample(
                call_stack=["main", "crypto.ComputeHash", "runtime.lockSlow"],
                weight=100000.0,
                timestamp=t0 + 0.5,
                sample_type="cpu",
            )
            for _ in range(5)
        ]
        empty_stream.add_batch(profiles=profiles)
        win = empty_stream.get_window(t0, t0 + 2.0)
        batch = empty_stream.build_tensor_batch([win])

        result = attributor.attribute(batch=batch, method="attention", window=win)
        assert result.is_anomalous
        assert result.primary_driver == AnomalyDriver.PROFILE_HOTSPOT.value
        assert result.modal_contributions["profiles"] > 0.5
        assert len(result.evidence["hotspot_stacks"]) > 0

    def test_attribute_window_end_to_end(self, small_fusion_model, empty_stream, sample_telemetry_data):
        model = small_fusion_model
        attributor = CrossModalAttributor(model=model, anomaly_threshold=0.5)

        empty_stream.add_batch(
            metrics=sample_telemetry_data["metrics"],
            spans=sample_telemetry_data["spans"],
            logs=sample_telemetry_data["logs"],
            profiles=sample_telemetry_data["profiles"],
        )
        t0 = sample_telemetry_data["t0"]
        win = empty_stream.get_window(t0, t0 + 5.0)

        result = attributor.attribute_window(window=win, stream=empty_stream, method="combined")
        assert isinstance(result, CrossModalAttributionResult)
        assert result.method == "combined"
        assert result.primary_driver in [d.value for d in AnomalyDriver]

    def test_summary_and_serialization(self, small_fusion_model, empty_stream, sample_telemetry_data):
        model = small_fusion_model
        attributor = CrossModalAttributor(model=model, anomaly_threshold=0.0)

        empty_stream.add_batch(
            metrics=sample_telemetry_data["metrics"],
            spans=sample_telemetry_data["spans"],
            logs=sample_telemetry_data["logs"],
            profiles=sample_telemetry_data["profiles"],
        )
        win = empty_stream.get_window(0.0, 1e10)
        result = attributor.attribute_window(window=win, stream=empty_stream)

        # ASCII summary formatting
        summary_text = result.summary(ascii_bars=True)
        assert "DeepLog-Omni Multi-Modal OTel Attribution" in summary_text
        assert "Primary Driver:" in summary_text
        assert "Modality Attribution Breakdown" in summary_text
        assert "█" in summary_text  # Contains ASCII filled bar characters

        # Plain summary formatting
        plain_summary = result.summary(ascii_bars=False)
        assert "█" not in plain_summary

        # JSON serialization
        d = result.to_dict()
        json_str = json.dumps(d)
        assert "primary_driver" in json_str
        assert "modal_percentages" in json_str
