"""
Unit tests for Track 4: Production Optimization, TorchScript JIT, and Streaming Aggregator.
Verifies:
- TorchScript JIT tracing and numerical parity (torch.allclose atol=1e-5) across CPU and MPS.
- SessionStreamAggregator buffer management (sliding window, NO_EVENT padding, LRU, timeout eviction).
"""

import os
import pytest
import torch
import numpy as np

from deeplog.deeplog import DeepLog
from deeplog.production import (
    trace_deeplog,
    verify_numerical_parity,
    benchmark_serving,
    save_traced_model,
    load_traced_model,
)
from deeplog.streaming import (
    SessionStreamAggregator,
    EventResult,
    SessionSummary,
)


@pytest.fixture(scope="module")
def model_path():
    path = "deeplog_model_improved_v1.pt"
    if not os.path.exists(path):
        pytest.skip(f"Model checkpoint '{path}' not found.")
    return path


@pytest.fixture(scope="module")
def native_model(model_path):
    model = DeepLog.load(model_path, device=torch.device("cpu")).to("cpu")
    model.eval()
    return model


class TestTorchScriptJIT:
    """Tests for TorchScript JIT tracing and numerical parity."""

    def test_jit_tracing_cpu(self, native_model):
        """Verify model can be traced using torch.jit.trace on CPU."""
        dummy_input = torch.randint(0, native_model.input_size, (1, 10), dtype=torch.long)
        traced = trace_deeplog(native_model, example_input=dummy_input, device=torch.device("cpu"))
        assert isinstance(traced, torch.jit.ScriptModule)

        # Forward pass should produce expected tensor shape
        out = traced(dummy_input)
        assert out.shape == (1, native_model.output_size)

    def test_numerical_parity_cpu(self, native_model):
        """Verify strict numerical parity on CPU (torch.allclose atol=1e-5)."""
        traced = trace_deeplog(native_model, device=torch.device("cpu"), optimize_for_inference=True)

        batch_sizes = [1, 16, 64, 256]
        all_passed, max_diff, details = verify_numerical_parity(
            native_model=native_model,
            jit_model=traced,
            batch_sizes=batch_sizes,
            atol=1e-5,
            device=torch.device("cpu"),
        )

        assert all_passed, f"Parity failed for some batch sizes. Details: {details}"
        assert max_diff <= 1e-5, f"Max diff {max_diff} exceeded 1e-5 tolerance."
        for bs in batch_sizes:
            assert details[f"batch_{bs}"]["passed"] is True

    def test_numerical_parity_mps(self, model_path):
        """Verify strict numerical parity on MPS if Apple Silicon GPU is available."""
        if not torch.backends.mps.is_available():
            pytest.skip("Apple Silicon MPS not available on this platform.")

        mps_device = torch.device("mps")
        model_mps = DeepLog.load(model_path, device=mps_device).to(mps_device)
        model_mps.eval()

        traced_mps = trace_deeplog(model_mps, device=mps_device, optimize_for_inference=True)

        batch_sizes = [1, 32, 256]
        all_passed, max_diff, details = verify_numerical_parity(
            native_model=model_mps,
            jit_model=traced_mps,
            batch_sizes=batch_sizes,
            atol=1e-5,
            device=mps_device,
        )

        assert all_passed, f"MPS Parity failed. Details: {details}"
        assert max_diff <= 1e-5, f"MPS Max diff {max_diff} exceeded tolerance."

    def test_jit_save_and_load(self, native_model, tmp_path):
        """Verify traced TorchScript model can be saved and loaded accurately."""
        traced = trace_deeplog(native_model, device=torch.device("cpu"))
        save_path = str(tmp_path / "traced_deeplog.pt")
        save_traced_model(traced, save_path)

        assert os.path.exists(save_path)
        loaded = load_traced_model(save_path, device=torch.device("cpu"))

        test_x = torch.randint(0, native_model.input_size, (5, 10), dtype=torch.long)
        out_traced = traced(test_x)
        out_loaded = loaded(test_x)

        assert torch.allclose(out_traced, out_loaded, atol=1e-5)

    def test_submillisecond_serving_latency(self, native_model):
        """Verify TorchScript JIT achieves sub-millisecond (< 1.0 ms) latency for batch size 1."""
        traced = trace_deeplog(native_model, device=torch.device("cpu"), optimize_for_inference=True)
        metrics = benchmark_serving(
            model=traced,
            batch_size=1,
            device=torch.device("cpu"),
            num_warmup=15,
            num_runs=100,
        )

        assert metrics["mean_latency_ms"] < 1.0, f"Mean latency {metrics['mean_latency_ms']} ms not sub-millisecond"
        assert metrics["p99_latency_ms"] < 1.0, f"P99 latency {metrics['p99_latency_ms']} ms exceeded 1.0 ms"


class TestSessionStreamAggregator:
    """Tests for SessionStreamAggregator buffer management, sliding context, and timeout cleanup."""

    @pytest.fixture
    def mock_mapping(self):
        # Maps canonical indices 0..4 to raw event IDs, NO_EVENT mapped to index 5
        return {
            0: 2,
            1: 5,
            2: 11,
            3: 22,
            4: 26,
            5: -1337,
        }

    def test_cold_start_padding(self, mock_mapping):
        """Verify uninitialized session context is fully padded with NO_EVENT canonical index."""
        agg = SessionStreamAggregator(
            model=None,
            context_length=10,
            mapping=mock_mapping,
            no_event_canonical=5,
        )

        # Before any events are processed, context should be 10 NO_EVENTs
        ctx = agg.get_context("block_1")
        assert len(ctx) == 10
        assert ctx == [5] * 10
        assert agg.active_session_count == 0

    def test_sliding_window_progression(self, mock_mapping):
        """Verify context buffer properly progresses and maintains length = context_length."""
        agg = SessionStreamAggregator(
            model=None,
            context_length=6,
            mapping=mock_mapping,
            no_event_canonical=5,
        )

        # Step 1: Ingest first event E5 (canonical=1)
        res1 = agg.process_event("block_1", raw_event=5, timestamp=1.0)
        assert res1.step_number == 1
        assert res1.canonical_event == 1
        # Context should now be 5 NO_EVENTs followed by event 1
        assert agg.get_context("block_1") == [5, 5, 5, 5, 5, 1]

        # Step 2: Ingest second event E22 (canonical=3)
        res2 = agg.process_event("block_1", raw_event=22, timestamp=2.0)
        assert res2.step_number == 2
        assert res2.canonical_event == 3
        assert agg.get_context("block_1") == [5, 5, 5, 5, 1, 3]

        # Ingest more events to exceed context_length (6)
        agg.process_event("block_1", raw_event=11, timestamp=3.0)  # canonical=2
        agg.process_event("block_1", raw_event=26, timestamp=4.0)  # canonical=4
        agg.process_event("block_1", raw_event=2, timestamp=5.0)   # canonical=0
        agg.process_event("block_1", raw_event=5, timestamp=6.0)   # canonical=1
        # Context is now completely full (no NO_EVENT): [1, 3, 2, 4, 0, 1]
        assert agg.get_context("block_1") == [1, 3, 2, 4, 0, 1]

        # Step 7: Push one more event E22 (canonical=3), oldest (1) should slide out
        agg.process_event("block_1", raw_event=22, timestamp=7.0)
        assert agg.get_context("block_1") == [3, 2, 4, 0, 1, 3]
        assert len(agg.get_context("block_1")) == 6

    def test_unseen_event_handling(self, mock_mapping):
        """Verify unseen raw event is flagged as anomaly and safely padded in context."""
        agg = SessionStreamAggregator(
            model=None,
            context_length=5,
            mapping=mock_mapping,
            no_event_canonical=5,
        )

        # Event 999 is not in mock_mapping
        res = agg.process_event("block_x", raw_event=999, timestamp=1.0)
        assert res.canonical_event == -1
        # In context buffer, unseen event is padded with no_event_canonical to avoid out-of-range indexing
        assert agg.get_context("block_x") == [5, 5, 5, 5, 5]

    def test_session_isolation_multiple_blocks(self, mock_mapping):
        """Verify interleaved events from different blocks do not cross-contaminate buffers."""
        agg = SessionStreamAggregator(
            model=None,
            context_length=4,
            mapping=mock_mapping,
            no_event_canonical=5,
        )

        # Interleave events between block A and block B
        agg.process_event("block_A", raw_event=5, timestamp=1.0)   # canon=1
        agg.process_event("block_B", raw_event=22, timestamp=1.1)  # canon=3
        agg.process_event("block_A", raw_event=11, timestamp=2.0)  # canon=2
        agg.process_event("block_B", raw_event=26, timestamp=2.1)  # canon=4

        assert agg.get_context("block_A") == [5, 5, 1, 2]
        assert agg.get_context("block_B") == [5, 5, 3, 4]
        assert agg.active_session_count == 2

    def test_session_timeout_and_eviction(self, mock_mapping):
        """Verify inactive sessions are evicted based on session_timeout threshold."""
        agg = SessionStreamAggregator(
            model=None,
            context_length=5,
            mapping=mock_mapping,
            session_timeout=10.0,
        )

        # block_old is active at t=2.0
        agg.process_event("block_old", raw_event=5, timestamp=2.0)
        # block_active is active at t=14.0
        agg.process_event("block_active", raw_event=22, timestamp=14.0)

        assert agg.active_session_count == 2

        # Advance current time to t=15.0 with timeout=10.0
        # block_old: 15.0 - 2.0 = 13.0 > 10.0 -> evicted
        # block_active: 15.0 - 14.0 = 1.0 <= 10.0 -> retained
        evicted = agg.cleanup_inactive_sessions(current_time=15.0)

        assert len(evicted) == 1
        assert evicted[0].block_id == "block_old"
        assert evicted[0].total_events == 1
        assert agg.active_session_count == 1
        assert "block_active" in agg.sessions
        assert "block_old" not in agg.sessions

    def test_manual_session_close(self, mock_mapping):
        """Verify close_session explicitly finalizes and evicts a session."""
        agg = SessionStreamAggregator(
            model=None,
            context_length=5,
            mapping=mock_mapping,
        )

        agg.process_event("block_close_me", raw_event=5, timestamp=1.0)
        agg.process_event("block_close_me", raw_event=11, timestamp=3.0)

        summary = agg.close_session("block_close_me")
        assert summary is not None
        assert summary.block_id == "block_close_me"
        assert summary.total_events == 2
        assert summary.duration_seconds == 2.0
        assert agg.active_session_count == 0

    def test_lru_capacity_eviction(self, mock_mapping):
        """Verify max_active_sessions enforces LRU eviction when capacity is exceeded."""
        agg = SessionStreamAggregator(
            model=None,
            context_length=4,
            mapping=mock_mapping,
            max_active_sessions=2,
        )

        agg.process_event("block_1", raw_event=5, timestamp=1.0)
        agg.process_event("block_2", raw_event=22, timestamp=2.0)
        assert agg.active_session_count == 2

        # Ingesting block_3 must evict the least recently used (block_1)
        agg.process_event("block_3", raw_event=11, timestamp=3.0)
        assert agg.active_session_count == 2
        assert "block_1" not in agg.sessions
        assert "block_2" in agg.sessions
        assert "block_3" in agg.sessions

    def test_realtime_anomaly_alerting_with_model(self, native_model):
        """Verify end-to-end rolling inference with model triggers alerts on anomalous events."""
        traced_model = trace_deeplog(native_model, device=torch.device("cpu"), optimize_for_inference=True)

        # Dummy mapping matching model's vocabulary
        vocab_size = native_model.input_size
        mapping = {i: i for i in range(vocab_size)}

        agg = SessionStreamAggregator(
            model=traced_model,
            context_length=10,
            top_k=3,
            mapping=mapping,
            no_event_canonical=vocab_size - 1,
            device=torch.device("cpu"),
        )

        # Ingest events
        res1 = agg.process_event("block_stream", raw_event=5, timestamp=1.0)
        assert isinstance(res1, EventResult)
        assert res1.latency_ms > 0
        assert len(res1.topk_predictions) == 3

        # Unseen event out of vocabulary (e.g. 9999) must trigger anomaly alert
        res_anom = agg.process_event("block_stream", raw_event=9999, timestamp=2.0)
        assert res_anom.is_anomaly is True
        assert res_anom.alert is not None
        assert res_anom.alert.is_unseen is True
        assert res_anom.alert.raw_event == 9999
        assert agg.total_anomalies_detected >= 1
