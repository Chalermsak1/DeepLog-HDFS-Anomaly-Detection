"""
Comprehensive Unit Tests for Temporal Dynamics & Time-Delta (\u0394t) Integration
=============================================================================
Tests:
1. TimeDeltaEncoder:
   - Initialization, mode validation, parameter assertions
   - Mathematical correctness of Sinusoidal Time2Vec (linear + periodic terms)
   - Dynamic range compression via log1p
   - Input shape flexibility: 1D, 2D, 3D
   - Backward gradient flow across linear and periodic parameters
2. TimeAwareDeepLog:
   - Architecture initialization, parameter aliases
   - Forward pass shapes (next-step and all-steps)
   - Probabilistic validity: sum(P(e)) == 1, \u03c3 > 0 strictly
   - Top-K prediction API and predicted time-delta bounds
   - Joint loss computation (categorical NLL + log-normal NLL)
   - End-to-end gradient flow without NaN/Inf across all submodules
   - Model checkpoint save, load, and numerical parity
3. TemporalAnomalyDetector:
   - Normal event transitions (no false positives)
   - Latency spike detection (z >= 3.0)
   - Statistical timeout detection (z >= 5.0)
   - Deterministic hard timeout thresholding (\u0394t >= hard_timeout)
   - Retry storm detection (sub-millisecond intervals & z <= -3.0)
   - Sequence anomaly detection (unseen / out-of-topk events)
   - Joint anomaly detection (simultaneous sequence + temporal violation)
   - Batch evaluation and sequential trace cascade detection
"""

import math
import os
import sys
import tempfile
import pytest
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from deeplog.temporal import (
    TimeDeltaEncoder,
    TimeAwareDeepLog,
    TemporalAnomalyDetector,
    TemporalAnomalyResult,
)


# ==============================================================================
# Track 1: TimeDeltaEncoder Tests
# ==============================================================================


class TestTimeDeltaEncoder:
    """Tests for Sinusoidal Time2Vec and Logarithmic Temporal Feature Encoders."""

    def test_init_and_parameter_validation(self):
        """Verify initialization and validation of embedding dimensions and modes."""
        # Valid Time2Vec
        enc = TimeDeltaEncoder(embed_dim=16, mode="time2vec", use_log=True)
        assert enc.embed_dim == 16
        assert enc.mode == "time2vec"
        assert enc.use_log is True
        assert enc.w0.numel() == 1
        assert enc.b0.numel() == 1
        assert enc.w.numel() == 15
        assert enc.b.numel() == 15

        # Invalid modes
        with pytest.raises(ValueError, match="Unsupported mode"):
            TimeDeltaEncoder(embed_dim=16, mode="invalid_mode")

        # Invalid embed_dim for Time2Vec (requires at least 2 for linear + periodic)
        with pytest.raises(ValueError, match="must be at least 2"):
            TimeDeltaEncoder(embed_dim=1, mode="time2vec")

        # Invalid activation
        with pytest.raises(ValueError, match="Unsupported activation"):
            TimeDeltaEncoder(embed_dim=16, activation="tanh")

    def test_time2vec_mathematical_properties(self):
        """
        Verify Time2Vec decomposition:
            t2v(\u03c4)[0] = \u03c9_0 * \u03c4 + \u03d5_0
            t2v(\u03c4)[i] = sin(\u03c9_i * \u03c4 + \u03d5_i), 1 <= i < d
        """
        torch.manual_seed(42)
        embed_dim = 8
        encoder = TimeDeltaEncoder(
            embed_dim=embed_dim, mode="time2vec", use_log=False, learnable=False
        )

        # Set deterministic weights for exact mathematical verification
        with torch.no_grad():
            encoder.w0.fill_(2.5)
            encoder.b0.fill_(0.5)
            # Periodic weights: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
            for i in range(embed_dim - 1):
                encoder.w[i] = float(i + 1)
                encoder.b[i] = 0.0

        test_dt = torch.tensor([0.0, 1.0, 2.5, 10.0], dtype=torch.float32)
        out = encoder(test_dt)

        assert out.shape == (4, embed_dim)

        # Verify linear component: \u03c9_0 * \u03c4 + \u03d5_0
        expected_linear = test_dt * 2.5 + 0.5
        assert torch.allclose(out[:, 0], expected_linear, atol=1e-5), "Linear component mismatch"

        # Verify periodic components: sin(\u03c9_i * \u03c4 + \u03d5_i)
        for i in range(1, embed_dim):
            freq = float(i)
            expected_periodic = torch.sin(test_dt * freq)
            assert torch.allclose(out[:, i], expected_periodic, atol=1e-5), (
                f"Periodic component {i} mismatch"
            )

        # Verify periodic outputs are strictly bounded in [-1.0, 1.0]
        assert (out[:, 1:] >= -1.0 - 1e-6).all() and (out[:, 1:] <= 1.0 + 1e-6).all(), (
            "Sinusoidal components must be bounded in [-1, 1]"
        )

    def test_logarithmic_compression_handling(self):
        """Verify log1p compression prevents numerical overflow on extreme time deltas."""
        encoder = TimeDeltaEncoder(embed_dim=16, mode="time2vec", use_log=True)

        # Diverse range: microsecond to 1 day (86400 seconds) + zero + negative clamp
        extreme_deltas = torch.tensor(
            [-5.0, 0.0, 0.0001, 1.0, 60.0, 3600.0, 86400.0, 1e7], dtype=torch.float32
        )
        out = encoder(extreme_deltas)

        # Check shapes and no NaN/Inf
        assert out.shape == (len(extreme_deltas), 16)
        assert not torch.isnan(out).any(), "NaN detected in TimeDeltaEncoder output"
        assert not torch.isinf(out).any(), "Inf detected in TimeDeltaEncoder output"

        # Negative input (-5.0) should be clamped to 0.0, yielding identical output to 0.0
        assert torch.allclose(out[0], out[1], atol=1e-6), (
            "Negative time deltas must be clamped to 0.0"
        )

    def test_alternative_modes(self):
        """Verify 'log' MLP mode and 'combined' mode."""
        batch_size, seq_len = 4, 10
        dt = torch.rand(batch_size, seq_len) * 5.0

        # Log mode
        log_enc = TimeDeltaEncoder(embed_dim=16, mode="log")
        out_log = log_enc(dt)
        assert out_log.shape == (batch_size, seq_len, 16)
        assert not torch.isnan(out_log).any()

        # Combined mode
        comb_enc = TimeDeltaEncoder(embed_dim=16, mode="combined")
        out_comb = comb_enc(dt)
        assert out_comb.shape == (batch_size, seq_len, 16)
        assert not torch.isnan(out_comb).any()

    def test_input_shape_flexibility(self):
        """Verify encoder supports 1D, 2D, and 3D tensor shapes."""
        encoder = TimeDeltaEncoder(embed_dim=16, mode="time2vec")

        # 1D: (seq_len,)
        dt_1d = torch.tensor([0.1, 0.5, 1.2])
        out_1d = encoder(dt_1d)
        assert out_1d.shape == (3, 16)

        # 2D: (batch_size, seq_len)
        dt_2d = torch.rand(8, 12)
        out_2d = encoder(dt_2d)
        assert out_2d.shape == (8, 12, 16)

        # 3D: (batch_size, seq_len, 1)
        dt_3d = torch.rand(4, 10, 1)
        out_3d = encoder(dt_3d)
        assert out_3d.shape == (4, 10, 16)

    def test_gradient_flow_time2vec(self):
        """Verify gradient flow through all Time2Vec parameters."""
        encoder = TimeDeltaEncoder(embed_dim=16, mode="time2vec", learnable=True)
        dt = torch.tensor([0.2, 1.5, 4.0, 10.0], requires_grad=False)

        out = encoder(dt)
        loss = out.sum()
        loss.backward()

        assert encoder.w0.grad is not None and not torch.isnan(encoder.w0.grad).any()
        assert encoder.b0.grad is not None and not torch.isnan(encoder.b0.grad).any()
        assert encoder.w.grad is not None and not torch.isnan(encoder.w.grad).any()
        assert encoder.b.grad is not None and not torch.isnan(encoder.b.grad).any()
        assert (encoder.w.grad.abs() > 0).any(), "Weights must receive non-zero gradients"


# ==============================================================================
# Track 2: TimeAwareDeepLog Architecture Tests
# ==============================================================================


class TestTimeAwareDeepLog:
    """Tests for the TimeAwareDeepLog neural network architecture."""

    def test_model_initialization(self):
        """Verify architectural parameters, submodules, and API aliases."""
        model = TimeAwareDeepLog(
            vocab_size=32,
            embed_dim=64,
            time_dim=16,
            hidden_size=128,
            num_layers=2,
            dropout=0.1,
            time_mode="time2vec",
        )
        assert model.vocab_size == 32
        assert model.embed_dim == 64
        assert model.time_dim == 16
        assert model.hidden_size == 128
        assert model.num_layers == 2
        assert isinstance(model.event_embedding, nn.Embedding)
        assert isinstance(model.time_encoder, TimeDeltaEncoder)
        assert isinstance(model.lstm, nn.LSTM)

        # Verify input_size alias
        model_alias = TimeAwareDeepLog(input_size=40, hidden_size=64)
        assert model_alias.vocab_size == 40
        assert model_alias.output_size == 40

    def test_forward_pass_shapes_and_values(self):
        """Verify forward pass output shapes and mathematical validity."""
        batch_size = 8
        seq_len = 10
        vocab_size = 25
        model = TimeAwareDeepLog(vocab_size=vocab_size, hidden_size=64, num_layers=2)
        model.eval()

        events = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)
        deltas = torch.rand(batch_size, seq_len, dtype=torch.float32) * 5.0

        with torch.no_grad():
            event_log_probs, time_mu, time_sigma = model(events, deltas)

        # Output shapes
        assert event_log_probs.shape == (batch_size, vocab_size)
        assert time_mu.shape == (batch_size, 1)
        assert time_sigma.shape == (batch_size, 1)

        # Probabilities sum to 1.0 per sample
        probs = event_log_probs.exp()
        sums = probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(batch_size), atol=1e-4), (
            "Event probabilities must sum to 1.0"
        )

        # Standard deviation must be strictly positive
        assert (time_sigma > 0).all(), "Time sigma must be strictly positive"

    def test_forward_return_all_steps(self):
        """Verify return_all_steps=True yields predictions for every sequence step."""
        batch_size = 4
        seq_len = 12
        vocab_size = 20
        model = TimeAwareDeepLog(vocab_size=vocab_size, hidden_size=64, num_layers=1)
        model.eval()

        events = torch.randint(0, vocab_size, (batch_size, seq_len))
        deltas = torch.rand(batch_size, seq_len)

        with torch.no_grad():
            ev_lp, mu, sigma = model(events, deltas, return_all_steps=True)

        assert ev_lp.shape == (batch_size, seq_len, vocab_size)
        assert mu.shape == (batch_size, seq_len, 1)
        assert sigma.shape == (batch_size, seq_len, 1)

    def test_single_1d_input_handling(self):
        """Verify model handles unbatched 1D inputs seamlessly."""
        vocab_size = 15
        model = TimeAwareDeepLog(vocab_size=vocab_size, hidden_size=32)
        model.eval()

        ev = torch.tensor([1, 4, 2, 7])
        dt = torch.tensor([0.1, 0.3, 0.05, 1.2])

        with torch.no_grad():
            ev_lp, mu, sigma = model(ev, dt)

        assert ev_lp.shape == (1, vocab_size)
        assert mu.shape == (1, 1)
        assert sigma.shape == (1, 1)

    def test_predict_topk(self):
        """Verify predict API returns valid Top-K predictions and positive time-deltas."""
        vocab_size = 30
        model = TimeAwareDeepLog(vocab_size=vocab_size, hidden_size=64)
        model.eval()

        batch_size = 6
        seq_len = 8
        k = 5
        events = torch.randint(0, vocab_size, (batch_size, seq_len))
        deltas = torch.rand(batch_size, seq_len) * 2.0

        topk_events, topk_conf, pred_dt, pred_sigma = model.predict(events, deltas, k=k)

        # Shapes
        assert topk_events.shape == (batch_size, k)
        assert topk_conf.shape == (batch_size, k)
        assert pred_dt.shape == (batch_size, 1)
        assert pred_sigma.shape == (batch_size, 1)

        # Bounds
        assert (topk_events >= 0).all() and (topk_events < vocab_size).all()
        assert (topk_conf >= 0.0).all() and (topk_conf <= 1.0).all()
        assert (pred_dt >= 0.0).all(), "Predicted raw time deltas must be >= 0"
        assert (pred_sigma > 0.0).all(), "Predicted sigma must be strictly positive"

        # Confidence monotonicity: sorted descending
        diffs = topk_conf[:, :-1] - topk_conf[:, 1:]
        assert (diffs >= -1e-6).all(), "Confidences must be sorted descending"

    def test_compute_loss_and_gradient_flow(self):
        """Verify joint loss calculation and backward gradient flow through all layers."""
        torch.manual_seed(42)
        vocab_size = 20
        model = TimeAwareDeepLog(
            vocab_size=vocab_size,
            embed_dim=32,
            time_dim=16,
            hidden_size=64,
            num_layers=2,
            dropout=0.0,
        )

        batch_size = 4
        seq_len = 6
        events = torch.randint(0, vocab_size, (batch_size, seq_len))
        deltas = torch.rand(batch_size, seq_len) * 3.0
        target_events = torch.randint(0, vocab_size, (batch_size,))
        target_deltas = torch.tensor([0.05, 1.2, 5.0, 0.0], dtype=torch.float32)

        loss_dict = model.compute_loss(
            events, deltas, target_events, target_deltas, alpha=1.0, beta=1.0
        )

        assert "total_loss" in loss_dict
        assert "event_loss" in loss_dict
        assert "temporal_loss" in loss_dict

        total_loss = loss_dict["total_loss"]
        assert torch.isfinite(total_loss), "Loss must be finite"
        assert total_loss.item() > 0.0

        total_loss.backward()

        # Check gradients for every named parameter in the model
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Parameter {name} has no gradient"
                assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN gradient"
                assert not torch.isinf(param.grad).any(), f"Parameter {name} has Inf gradient"

    def test_checkpoint_save_and_load(self):
        """Verify checkpoint serialization and numerical parity between original and loaded models."""
        vocab_size = 25
        model = TimeAwareDeepLog(vocab_size=vocab_size, hidden_size=64, num_layers=2)
        model.eval()

        events = torch.randint(0, vocab_size, (2, 8))
        deltas = torch.rand(2, 8)

        with torch.no_grad():
            orig_ev, orig_mu, orig_sigma = model(events, deltas)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            model.save(tmp_path)
            loaded_model = TimeAwareDeepLog.load(tmp_path)
            loaded_model.eval()

            with torch.no_grad():
                load_ev, load_mu, load_sigma = loaded_model(events, deltas)

            assert torch.allclose(orig_ev, load_ev, atol=1e-6), "Event logits mismatch after load"
            assert torch.allclose(orig_mu, load_mu, atol=1e-6), "Time mu mismatch after load"
            assert torch.allclose(orig_sigma, load_sigma, atol=1e-6), "Time sigma mismatch after load"
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


# ==============================================================================
# Track 3: TemporalAnomalyDetector Tests
# ==============================================================================


class TestTemporalAnomalyDetector:
    """Tests for joint sequence-temporal anomaly diagnostics."""

    @pytest.fixture
    def detector_setup(self):
        """Creates a mock prediction setup for controlled testing."""
        vocab_size = 10
        # Simulated log probabilities: event 0 has highest prob (0.7), event 1 has (0.2), rest tiny
        logits = torch.tensor([5.0, 3.5, 1.0, 0.0, 0.0, -1.0, -2.0, -3.0, -4.0, -5.0])
        log_probs = F.log_softmax(logits, dim=-1)

        # Mean predicted log1p(dt): mu = log1p(0.1) ~ 0.0953 (expected ~100ms)
        # Std predicted: sigma = 0.1
        mu = math.log1p(0.1)
        sigma = 0.1

        detector = TemporalAnomalyDetector(
            top_k=2,
            z_spike_threshold=3.0,
            z_timeout_threshold=5.0,
            hard_timeout_seconds=30.0,
            z_retry_storm_threshold=-3.0,
            min_retry_delta_seconds=0.005,
        )
        return detector, log_probs, mu, sigma

    def test_normal_transition(self, detector_setup):
        """Verify normal event transition is not flagged as an anomaly."""
        detector, log_probs, mu, sigma = detector_setup

        # Expected event (0 \in Top-2) and expected latency ~ 100ms (\u0394t = 0.1)
        result = detector.diagnose_from_predictions(
            event_log_probs=log_probs,
            time_mu=mu,
            time_sigma=sigma,
            next_event=0,
            next_delta=0.1,
        )

        assert result.is_anomaly is False
        assert result.primary_type == "NORMAL"
        assert result.is_sequence_anomaly is False
        assert result.is_temporal_anomaly is False
        assert result.is_timeout is False
        assert result.is_latency_spike is False
        assert result.is_retry_storm is False
        assert abs(result.z_score) < 0.1

    def test_latency_spike_detection(self, detector_setup):
        """Verify latency spike is correctly diagnosed (3.0 <= z < 5.0)."""
        detector, log_probs, mu, sigma = detector_setup

        # For mu=0.0953, sigma=0.1:
        # To get z ~ 3.5: y_obs = 0.0953 + 0.35 = 0.4453 -> dt = exp(0.4453) - 1 ~ 0.56s (560ms)
        dt_spike = math.expm1(mu + 3.5 * sigma)

        result = detector.diagnose_from_predictions(
            event_log_probs=log_probs,
            time_mu=mu,
            time_sigma=sigma,
            next_event=0,  # Valid sequence event
            next_delta=dt_spike,
        )

        assert result.is_anomaly is True
        assert result.primary_type == "LATENCY_SPIKE"
        assert result.is_latency_spike is True
        assert result.is_timeout is False
        assert result.is_sequence_anomaly is False
        assert result.is_temporal_anomaly is True
        assert 3.0 <= result.z_score < 5.0

    def test_statistical_timeout_detection(self, detector_setup):
        """Verify statistical timeout is diagnosed when z >= 5.0."""
        detector, log_probs, mu, sigma = detector_setup

        # To get z ~ 6.0: dt = exp(mu + 6.0 * sigma) - 1
        dt_timeout = math.expm1(mu + 6.0 * sigma)

        result = detector.diagnose_from_predictions(
            event_log_probs=log_probs,
            time_mu=mu,
            time_sigma=sigma,
            next_event=0,
            next_delta=dt_timeout,
        )

        assert result.is_anomaly is True
        assert result.primary_type == "TIMEOUT"
        assert result.is_timeout is True
        assert result.is_latency_spike is False
        assert result.z_score >= 5.0

    def test_hard_timeout_threshold_detection(self, detector_setup):
        """Verify deterministic hard timeout triggers even if z-score is below threshold."""
        detector, log_probs, _, _ = detector_setup
        # High mu and sigma where z-score might be small, but duration exceeds hard limit (30.0s)
        mu_large = math.log1p(25.0)
        sigma_large = 2.0

        result = detector.diagnose_from_predictions(
            event_log_probs=log_probs,
            time_mu=mu_large,
            time_sigma=sigma_large,
            next_event=0,
            next_delta=35.0,  # > hard_timeout_seconds (30.0)
        )

        assert result.is_anomaly is True
        assert result.is_timeout is True
        assert result.primary_type == "TIMEOUT"
        assert result.details["hard_timeout_triggered"] is True

    def test_retry_storm_detection(self, detector_setup):
        """Verify retry storm / rapid burst anomaly detection via both micro-delta and negative Z-score."""
        detector, log_probs, mu, sigma = detector_setup

        # Case 1: Sub-millisecond micro-burst when expected is 100ms (0.1s)
        dt_rapid = 0.0001
        result_burst = detector.diagnose_from_predictions(
            event_log_probs=log_probs,
            time_mu=mu,
            time_sigma=sigma,
            next_event=0,
            next_delta=dt_rapid,
        )

        assert result_burst.is_anomaly is True
        assert result_burst.is_retry_storm is True
        assert result_burst.primary_type == "RETRY_STORM"

        # Case 2: Statistical negative Z-score (z <= -3.0) for long-duration baseline
        mu_long = math.log1p(5.0)  # expected 5.0 seconds
        sigma_long = 0.3
        result_stat = detector.diagnose_from_predictions(
            event_log_probs=log_probs,
            time_mu=mu_long,
            time_sigma=sigma_long,
            next_event=0,
            next_delta=0.001,  # 1ms when expected is 5.0s
        )

        assert result_stat.is_anomaly is True
        assert result_stat.is_retry_storm is True
        assert result_stat.primary_type == "RETRY_STORM"
        assert result_stat.z_score <= -3.0

    def test_sequence_anomaly_detection(self, detector_setup):
        """Verify unexpected event outside Top-K is flagged as SEQUENCE_ANOMALY."""
        detector, log_probs, mu, sigma = detector_setup

        # Event 9 is lowest probability (outside top_k=2), normal latency
        result = detector.diagnose_from_predictions(
            event_log_probs=log_probs,
            time_mu=mu,
            time_sigma=sigma,
            next_event=9,
            next_delta=0.1,  # Normal latency
        )

        assert result.is_anomaly is True
        assert result.is_sequence_anomaly is True
        assert result.is_temporal_anomaly is False
        assert result.primary_type == "SEQUENCE_ANOMALY"

    def test_joint_anomaly_detection(self, detector_setup):
        """Verify concurrent sequence and temporal violations trigger JOINT_ANOMALY."""
        detector, log_probs, mu, sigma = detector_setup

        # Event 8 (out of top-k) + latency spike (z ~ 3.5)
        dt_spike = math.expm1(mu + 3.5 * sigma)
        result = detector.diagnose_from_predictions(
            event_log_probs=log_probs,
            time_mu=mu,
            time_sigma=sigma,
            next_event=8,
            next_delta=dt_spike,
        )

        assert result.is_anomaly is True
        assert result.is_sequence_anomaly is True
        assert result.is_temporal_anomaly is True
        assert result.primary_type == "JOINT_ANOMALY"
        assert "LATENCY_SPIKE" in result.anomaly_types
        assert "SEQUENCE_ANOMALY" in result.anomaly_types

    def test_detector_with_trained_model(self):
        """Verify end-to-end integration of TemporalAnomalyDetector with TimeAwareDeepLog."""
        vocab_size = 12
        model = TimeAwareDeepLog(vocab_size=vocab_size, hidden_size=32, num_layers=1)
        detector = TemporalAnomalyDetector(model=model, top_k=3, hard_timeout_seconds=20.0)

        context_events = torch.tensor([1, 2, 3, 4, 5])
        context_deltas = torch.tensor([0.1, 0.15, 0.12, 0.11, 0.13])

        # Normal prediction query
        result = detector.diagnose_event(
            context_events=context_events,
            context_deltas=context_deltas,
            next_event=6,
            next_delta=0.12,
        )
        assert isinstance(result, TemporalAnomalyResult)
        assert isinstance(result.to_dict(), dict)

        # Timeout query
        timeout_res = detector.diagnose_event(
            context_events=context_events,
            context_deltas=context_deltas,
            next_event=6,
            next_delta=25.0,  # Exceeds 20.0s hard timeout
        )
        assert timeout_res.is_timeout is True
        assert timeout_res.primary_type == "TIMEOUT"

    def test_evaluate_batch(self):
        """Verify batch evaluation over multiple concurrent transitions."""
        vocab_size = 16
        model = TimeAwareDeepLog(vocab_size=vocab_size, hidden_size=32)
        detector = TemporalAnomalyDetector(model=model, top_k=3, hard_timeout_seconds=15.0)

        batch_size = 5
        seq_len = 8
        events = torch.randint(0, vocab_size, (batch_size, seq_len))
        deltas = torch.rand(batch_size, seq_len) * 0.5
        target_events = torch.tensor([0, 1, 2, 3, 4])
        target_deltas = torch.tensor([0.2, 20.0, 0.0001, 0.5, 0.1])  # 20.0 is timeout

        results = detector.evaluate_batch(events, deltas, target_events, target_deltas)

        assert len(results) == batch_size
        assert results[1].is_timeout is True  # Index 1 has \u0394t = 20.0 > 15.0s

    def test_trace_cascade_retry_storm(self):
        """Verify trace evaluation detects multi-step retry storm cascades."""
        vocab_size = 10
        model = TimeAwareDeepLog(vocab_size=vocab_size, hidden_size=32)
        detector = TemporalAnomalyDetector(
            model=model,
            top_k=5,
            min_retry_delta_seconds=0.005,
            min_retry_storm_count=3,
        )

        # Trace with 10 normal steps, followed by 4 rapid retry steps (0.001s)
        trace_events = [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 2, 2, 2]
        trace_deltas = [
            0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
            0.001, 0.001, 0.001, 0.001
        ]

        results = detector.evaluate_trace(trace_events, trace_deltas, window_size=5)

        assert len(results) == len(trace_events) - 5
        # The cascade should trigger retry storm on step 3 of micro-deltas
        retry_storm_detected = any(r.is_retry_storm for r in results)
        assert retry_storm_detected is True, "Cascade of micro-deltas must trigger RETRY_STORM"
