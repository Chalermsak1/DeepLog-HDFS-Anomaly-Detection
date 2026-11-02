"""
Unit Tests for DeepLog Parameter Value Anomaly Model (Pillar 2)
================================================================
Comprehensive test suite covering:
1. ParameterValuePredictor:
   - LSTM multivariate regression forward pass (shapes, hidden state, aliases)
   - Training convergence with MSE loss
   - Autoregressive multi-step rollout forecasting
   - Model checkpoint save and load
2. MultivariateGaussianEstimator:
   - Mean, covariance, and precision calculation
   - Mahalanobis distance vs scipy.spatial.distance.mahalanobis
   - Gaussian NLL vs scipy.stats.multivariate_normal.logpdf
   - Additive feature decomposition and z-score attribution
   - Singularity handling (collinear features, constant zero-variance, N < d, N = 1)
   - Regularization modes: shrinkage, Ledoit-Wolf, epsilon-identity, diagonal
   - Decision threshold strategies (chi2, quantile, sigma, manual)
   - Estimator serialization (save/load)
3. ParameterAnomalyDetector:
   - Sliding window generator
   - End-to-end fit, scoring, and detection on domain parameters (byte size, thread count, queue latency)
   - Injected real-world log parameter anomalies (byte surge, thread leak, latency spike)
   - Diagnostic explanation and parameter attribution
   - Full pipeline save and load round-trip
"""

import os
import tempfile
import pytest
import numpy as np
import scipy.spatial.distance
import scipy.stats
import torch

from deeplog.parameters import (
    ParameterValuePredictor,
    MultivariateGaussianEstimator,
    ParameterAnomalyDetector,
    ParameterAnomalyResult,
)


# ==============================================================================
# 1. ParameterValuePredictor Tests
# ==============================================================================

class TestParameterValuePredictor:
    """Tests for the LSTM parameter value regressor."""

    def test_init_and_aliases(self):
        """Verify model initialization and DeepLog parameter aliases."""
        model = ParameterValuePredictor(input_dim=4, hidden_dim=32, num_layers=2, output_dim=4)
        assert model.input_dim == 4
        assert model.hidden_dim == 32
        assert model.output_dim == 4
        assert model.num_layers == 2

        # Check aliases
        model_alias = ParameterValuePredictor(input_size=5, hidden_size=64, output_size=5)
        assert model_alias.input_dim == 5
        assert model_alias.hidden_dim == 64
        assert model_alias.output_dim == 5

    def test_forward_shapes(self):
        """Verify forward pass output shapes for single step and all steps."""
        batch_size = 8
        seq_len = 10
        d = 3
        model = ParameterValuePredictor(input_dim=d, hidden_dim=16, num_layers=1)
        x = torch.randn(batch_size, seq_len, d)

        # Next-step prediction: shape (batch_size, d)
        out_next = model(x, return_all_steps=False)
        assert out_next.shape == (batch_size, d)

        # All-step prediction: shape (batch_size, seq_len, d)
        out_all = model(x, return_all_steps=True)
        assert out_all.shape == (batch_size, seq_len, d)

        # 2D single sequence input: shape (seq_len, d) -> (1, d)
        x_2d = torch.randn(seq_len, d)
        out_2d = model(x_2d, return_all_steps=False)
        assert out_2d.shape == (1, d)

    def test_forward_with_hidden_state(self):
        """Verify forward pass passing and receiving hidden states."""
        model = ParameterValuePredictor(input_dim=2, hidden_dim=8, num_layers=2)
        x = torch.randn(4, 5, 2)
        h0 = torch.zeros(2, 4, 8)
        c0 = torch.zeros(2, 4, 8)

        out, (hn, cn) = model(x, hidden=(h0, c0))
        assert out.shape == (4, 2)
        assert hn.shape == (2, 4, 8)
        assert cn.shape == (2, 4, 8)

    def test_predict_numpy(self):
        """Verify predict() accepts NumPy array and returns NumPy array."""
        model = ParameterValuePredictor(input_dim=3, hidden_dim=16, num_layers=1)
        x_np = np.random.randn(5, 7, 3).astype(np.float32)

        preds = model.predict(x_np)
        assert isinstance(preds, np.ndarray)
        assert preds.shape == (5, 3)

    def test_multistep_prediction(self):
        """Verify autoregressive multi-step sequence rollout."""
        model = ParameterValuePredictor(input_dim=3, hidden_dim=16, num_layers=1)
        x = torch.randn(2, 10, 3)

        # Rollout 5 steps into future
        forecast = model.predict_multistep(x, steps=5, keep_window=True)
        assert isinstance(forecast, np.ndarray)
        assert forecast.shape == (2, 5, 3)

        # Single step matches standard predict
        forecast_1 = model.predict_multistep(x, steps=1)
        pred_1 = model.predict(x)
        np.testing.assert_allclose(forecast_1[:, 0, :], pred_1, atol=1e-5)

        # Exception on invalid steps
        with pytest.raises(ValueError):
            model.predict_multistep(x, steps=0)

    def test_fit_convergence(self):
        """Verify that training decreases MSE loss on a smooth autoregressive signal."""
        torch.manual_seed(42)
        np.random.seed(42)

        # Create synthetic 2D harmonic series: v_t = [sin(0.1*t), cos(0.1*t)]
        t = np.linspace(0, 30, 200)
        series = np.column_stack([np.sin(t), np.cos(t)])  # (200, 2)

        # Window size 5
        X_list, y_list = [], []
        for i in range(len(series) - 5):
            X_list.append(series[i : i + 5])
            y_list.append(series[i + 5])
        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.float32)

        model = ParameterValuePredictor(input_dim=2, hidden_dim=32, num_layers=1)
        history = model.fit(X, y, epochs=35, batch_size=16, lr=0.01)

        # Loss should decrease significantly
        assert history["train_loss"][-1] < history["train_loss"][0]
        assert history["train_loss"][-1] < 0.05

    def test_save_and_load(self):
        """Verify model checkpoint save and load fidelity."""
        model = ParameterValuePredictor(input_dim=4, hidden_dim=20, num_layers=2, output_dim=4, dropout=0.1)
        x = torch.randn(3, 8, 4)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "predictor.pt")
            model.save(ckpt_path)

            loaded = ParameterValuePredictor.load(ckpt_path)
            assert loaded.input_dim == model.input_dim
            assert loaded.hidden_dim == model.hidden_dim
            assert loaded.num_layers == model.num_layers
            assert loaded.output_dim == model.output_dim

            # Verify identical predictions
            orig_out = model.predict(x)
            loaded_out = loaded.predict(x)
            np.testing.assert_allclose(orig_out, loaded_out, atol=1e-6)


# ==============================================================================
# 2. MultivariateGaussianEstimator Tests
# ==============================================================================

class TestMultivariateGaussianEstimator:
    """Tests for the Gaussian error residual estimator and Mahalanobis engine."""

    def test_mean_and_covariance_estimation(self):
        """Verify sample mean and covariance estimation on isotropic Gaussian data."""
        np.random.seed(42)
        N = 500
        d = 3
        true_mean = np.array([1.5, -2.0, 0.5])
        true_cov = np.array([
            [2.0, 0.4, 0.1],
            [0.4, 1.5, 0.2],
            [0.1, 0.2, 1.0],
        ])
        samples = np.random.multivariate_normal(true_mean, true_cov, size=N)

        # Fit estimator with epsilon regularization
        estimator = MultivariateGaussianEstimator(regularization="epsilon", epsilon=1e-7)
        estimator.fit(samples)

        # Mean should match sample mean closely
        np.testing.assert_allclose(estimator.mean, np.mean(samples, axis=0), atol=1e-6)
        # Covariance should match empirical sample covariance closely
        sample_cov = np.cov(samples, rowvar=False)
        np.testing.assert_allclose(estimator.covariance, sample_cov, atol=1e-4)

    def test_mahalanobis_vs_scipy(self):
        """Verify that Mahalanobis distance calculation exactly matches scipy."""
        np.random.seed(123)
        N = 100
        d = 4
        residuals = np.random.randn(N, d) * 2.0 + 1.0

        estimator = MultivariateGaussianEstimator(regularization="none", epsilon=1e-6)
        estimator.fit(residuals)

        test_points = np.random.randn(10, d)
        estimator_dists = estimator.mahalanobis(test_points)

        # Compute scipy distances point-by-point
        # Scipy formula: sqrt( (u - v) * VI * (u - v).T )
        scipy_dists = []
        for i in range(len(test_points)):
            d_i = scipy.spatial.distance.mahalanobis(
                test_points[i],
                estimator.mean,
                estimator.precision,
            )
            scipy_dists.append(d_i)

        np.testing.assert_allclose(estimator_dists, scipy_dists, atol=1e-6)

    def test_mahalanobis_zero_at_mean(self):
        """Residuals exactly at the mean must have Mahalanobis distance 0."""
        estimator = MultivariateGaussianEstimator(regularization="epsilon")
        res = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        estimator.fit(res)

        mean_res = estimator.mean[np.newaxis, :]
        dist = estimator.mahalanobis(mean_res)
        assert np.isclose(dist[0], 0.0, atol=1e-7)

    def test_nll_vs_scipy(self):
        """Verify Gaussian Negative Log-Likelihood matches -scipy.stats.multivariate_normal.logpdf."""
        np.random.seed(99)
        N = 150
        d = 3
        residuals = np.random.randn(N, d) + 0.5

        estimator = MultivariateGaussianEstimator(regularization="epsilon", epsilon=1e-6)
        estimator.fit(residuals)

        test_points = np.random.randn(5, d)
        est_nll = estimator.nll(test_points)

        scipy_nll = -scipy.stats.multivariate_normal.logpdf(
            test_points,
            mean=estimator.mean,
            cov=estimator.covariance,
        )

        np.testing.assert_allclose(est_nll, scipy_nll, atol=1e-5)

    def test_additive_feature_attribution(self):
        """Verify exact additive decomposition sum(c_j) == D_M^2."""
        np.random.seed(42)
        residuals = np.random.randn(50, 3)
        estimator = MultivariateGaussianEstimator(regularization="shrinkage", shrinkage_factor=0.2)
        estimator.fit(residuals)

        test_residuals = np.array([
            [1.0, 2.0, -1.5],
            [0.2, -0.4, 0.1],
            [10.0, 0.0, 0.0],  # Heavy anomaly on feature 0
        ])

        explanation = estimator.explain(test_residuals, feature_names=["bytes", "threads", "latency"])
        contributions = explanation["contributions"]
        sq_dists = estimator.mahalanobis(test_residuals, squared=True)

        # Sum of contributions across features must equal D_M^2
        sum_contribs = np.sum(contributions, axis=-1)
        np.testing.assert_allclose(sum_contribs, sq_dists, atol=1e-6)

        # For the third point [10, 0, 0], feature "bytes" must be top contributor
        assert explanation["top_features"][2] == "bytes"

    @pytest.mark.parametrize("reg_mode", ["shrinkage", "ledoit_wolf", "epsilon", "diagonal"])
    def test_singularity_handling_collinear(self, reg_mode):
        """Verify that estimator handles perfectly collinear features without singular matrix error."""
        np.random.seed(42)
        N = 100
        feat0 = np.random.randn(N)
        feat1 = 2.0 * feat0            # Perfectly collinear with feat0
        feat2 = -3.0 * feat0 + 1.0     # Perfectly collinear with feat0
        residuals = np.column_stack([feat0, feat1, feat2])

        estimator = MultivariateGaussianEstimator(regularization=reg_mode, epsilon=1e-4)
        estimator.fit(residuals)

        # Check positive definiteness: all eigenvalues must be >= epsilon
        eigvals = np.linalg.eigvalsh(estimator.covariance)
        assert np.all(eigvals >= 1e-4 - 1e-8)

        # Inversion and Mahalanobis computation must produce finite numbers
        test_pt = np.array([[1.0, 2.0, -2.0]])
        dist = estimator.mahalanobis(test_pt)
        assert np.isfinite(dist[0])
        assert dist[0] >= 0.0

    def test_singularity_handling_zero_variance_constant(self):
        """Verify handling of completely constant features (zero variance)."""
        N = 50
        feat0 = np.random.randn(N)
        feat1 = np.ones(N) * 42.0     # Constant, zero variance!
        residuals = np.column_stack([feat0, feat1])

        estimator = MultivariateGaussianEstimator(regularization="shrinkage", epsilon=1e-5)
        estimator.fit(residuals)

        assert np.isfinite(estimator.precision).all()
        assert np.all(np.linalg.eigvalsh(estimator.covariance) > 0)

        dist = estimator.mahalanobis(residuals[:5])
        assert np.all(np.isfinite(dist))

    def test_singularity_underdetermined_sample_size(self):
        """Verify handling of underdetermined N < d cases (e.g. 3 samples in 10-dimensional space)."""
        residuals = np.random.randn(3, 10)
        estimator = MultivariateGaussianEstimator(regularization="shrinkage", epsilon=1e-4)
        estimator.fit(residuals)

        assert estimator.covariance.shape == (10, 10)
        assert np.all(np.linalg.eigvalsh(estimator.covariance) >= 1e-4 - 1e-8)
        assert np.all(np.isfinite(estimator.precision))

        dists = estimator.mahalanobis(residuals)
        assert np.all(np.isfinite(dists))

    def test_single_sample_fit(self):
        """Verify estimator does not crash when fitted on a single residual sample (N=1)."""
        single_res = np.array([[1.0, 2.0, 3.0]])
        estimator = MultivariateGaussianEstimator(regularization="epsilon", epsilon=1e-3)
        estimator.fit(single_res)

        assert estimator.mean.shape == (3,)
        assert np.all(np.isfinite(estimator.precision))
        dist = estimator.mahalanobis(single_res)
        assert np.isclose(dist[0], 0.0, atol=1e-6)

    def test_thresholding_methods(self):
        """Verify chi2, quantile, sigma, and manual thresholding behavior."""
        np.random.seed(42)
        residuals = np.random.randn(300, 3)
        estimator = MultivariateGaussianEstimator(regularization="shrinkage")
        estimator.fit(residuals)

        # 1. Chi2 threshold at alpha=0.01
        t_chi2 = estimator.set_threshold(method="chi2", alpha=0.01)
        expected_chi2 = np.sqrt(scipy.stats.chi2.ppf(0.99, df=3))
        assert np.isclose(t_chi2, expected_chi2, atol=1e-4)

        # 2. Quantile threshold at alpha=0.05
        t_quant = estimator.set_threshold(method="quantile", alpha=0.05, residuals=residuals)
        assert t_quant > 0.0

        # 3. Sigma threshold (mean + 3*std)
        t_sigma = estimator.set_threshold(method="sigma", k_sigma=3.0, residuals=residuals)
        assert t_sigma > 0.0

        # 4. Manual threshold
        t_manual = estimator.set_threshold(method="manual", threshold=4.5)
        assert t_manual == 4.5

    def test_save_and_load(self):
        """Verify MultivariateGaussianEstimator serialization round-trip."""
        residuals = np.random.randn(100, 3) * 1.5 + 0.2
        estimator = MultivariateGaussianEstimator(regularization="shrinkage", shrinkage_factor=0.15)
        estimator.fit(residuals)

        test_points = np.random.randn(10, 3)
        orig_dists = estimator.mahalanobis(test_points)
        orig_nll = estimator.nll(test_points)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "estimator.pt")
            estimator.save(save_path)

            loaded = MultivariateGaussianEstimator.load(save_path)
            loaded_dists = loaded.mahalanobis(test_points)
            loaded_nll = loaded.nll(test_points)

            np.testing.assert_allclose(orig_dists, loaded_dists, atol=1e-7)
            np.testing.assert_allclose(orig_nll, loaded_nll, atol=1e-7)
            assert loaded.threshold == estimator.threshold


# ==============================================================================
# 3. ParameterAnomalyDetector End-to-End Tests
# ==============================================================================

class TestParameterAnomalyDetector:
    """Tests for the end-to-end ParameterAnomalyDetector system."""

    def test_sliding_window_generator(self):
        """Verify continuous stream to sliding window converter."""
        # 100 timesteps of 3 parameters
        stream = np.arange(300).reshape(100, 3).astype(np.float64)
        X, y = ParameterAnomalyDetector.create_sliding_windows(stream, window_size=5)

        assert X.shape == (95, 5, 3)
        assert y.shape == (95, 3)
        # Check window alignment: y[0] should equal stream[5]
        np.testing.assert_array_equal(y[0], stream[5])
        np.testing.assert_array_equal(X[0, -1], stream[4])

    def test_end_to_end_log_parameter_anomalies(self):
        """
        Verify end-to-end detection on realistic log parameters:
        [byte_size, thread_count, queue_latency]
        Injecting:
        1. Byte size anomaly
        2. Abnormal thread count surge
        3. Queue latency surge
        """
        np.random.seed(42)
        torch.manual_seed(42)

        # Generate normal background parameter sequence (N=300)
        # byte_size ~ N(5000, 200), thread_count ~ N(16, 2), queue_latency ~ N(20, 3)
        N = 300
        bytes_normal = 5000.0 + 200.0 * np.sin(np.linspace(0, 20, N)) + np.random.randn(N) * 50.0
        threads_normal = 16.0 + 2.0 * np.cos(np.linspace(0, 10, N)) + np.random.randn(N) * 0.5
        latency_normal = 20.0 + 1.0 * np.sin(np.linspace(0, 15, N)) + np.random.randn(N) * 1.0

        stream = np.column_stack([bytes_normal, threads_normal, latency_normal])
        X_train, y_train = ParameterAnomalyDetector.create_sliding_windows(stream, window_size=6)

        detector = ParameterAnomalyDetector(
            input_dim=3,
            hidden_dim=32,
            num_layers=1,
            feature_names=["byte_size", "thread_count", "queue_latency"],
            threshold_method="chi2",
            alpha=0.001,
        )

        detector.fit(X_train, y_train, epochs=25, batch_size=16, lr=0.01)

        # Create test evaluation set with 3 normal points and 3 distinct injected anomalies
        X_test = X_train[:6].copy()
        y_test = y_train[:6].copy()

        # Injected anomalies:
        # Sample 3: Massive byte size surge (50,000 bytes vs ~5,000)
        y_test[3, 0] = 50000.0
        # Sample 4: Abnormal thread count surge (150 threads vs ~16)
        y_test[4, 1] = 150.0
        # Sample 5: Severe queue latency surge (2,000 ms vs ~20 ms)
        y_test[5, 2] = 2000.0

        result = detector.detect(X_test, y_test)
        assert isinstance(result, ParameterAnomalyResult)

        # Verify normal samples (indices 0, 1, 2) are not anomalous
        assert result.is_anomaly[0] == False
        assert result.is_anomaly[1] == False
        assert result.is_anomaly[2] == False

        # Verify injected anomalies (indices 3, 4, 5) are flagged
        assert result.is_anomaly[3] == True
        assert result.is_anomaly[4] == True
        assert result.is_anomaly[5] == True

        # Check diagnostics pinpoint correct primary parameters
        diag_sample_3 = [d for d in result.diagnostics if "Sample 3" in d][0]
        assert "byte_size" in diag_sample_3

        diag_sample_4 = [d for d in result.diagnostics if "Sample 4" in d][0]
        assert "thread_count" in diag_sample_4

        diag_sample_5 = [d for d in result.diagnostics if "Sample 5" in d][0]
        assert "queue_latency" in diag_sample_5

        # Check summary string output
        summary = result.summary()
        assert "DeepLog Parameter Value Anomaly Detection Summary" in summary
        assert "Anomalies Flagged        : 3" in summary

    def test_save_and_load_detector_pipeline(self):
        """Verify complete ParameterAnomalyDetector pipeline save/load round-trip."""
        np.random.seed(42)
        stream = np.random.randn(80, 2) * 2.0 + 5.0
        X, y = ParameterAnomalyDetector.create_sliding_windows(stream, window_size=4)

        detector = ParameterAnomalyDetector(
            input_dim=2,
            hidden_dim=16,
            num_layers=1,
            feature_names=["f0", "f1"],
            threshold_method="chi2",
            alpha=0.01,
        )
        detector.fit(X, y, epochs=10, batch_size=16)

        test_X = X[:5]
        test_y = y[:5]

        orig_res = detector.detect(test_X, test_y)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "detector.pt")
            detector.save(ckpt_path)

            loaded = ParameterAnomalyDetector.load(ckpt_path)
            loaded_res = loaded.detect(test_X, test_y)

            np.testing.assert_allclose(orig_res.scores, loaded_res.scores, atol=1e-5)
            np.testing.assert_allclose(orig_res.residuals, loaded_res.residuals, atol=1e-5)
            np.testing.assert_array_equal(orig_res.is_anomaly, loaded_res.is_anomaly)
            assert orig_res.threshold == loaded_res.threshold
            assert loaded.feature_names == detector.feature_names

    def test_detector_without_normalization(self):
        """Verify detector functions correctly with normalize=False."""
        np.random.seed(42)
        stream = np.random.randn(60, 2)
        X, y = ParameterAnomalyDetector.create_sliding_windows(stream, window_size=4)

        detector = ParameterAnomalyDetector(
            input_dim=2,
            hidden_dim=8,
            num_layers=1,
            normalize=False,
            alpha=0.01,
        )
        detector.fit(X, y, epochs=5)

        res = detector.detect(X[:3], y[:3])
        assert len(res.scores) == 3
        assert np.all(np.isfinite(res.scores))

    def test_nll_metric_detection(self):
        """Verify detector scoring with metric='nll'."""
        np.random.seed(42)
        stream = np.random.randn(50, 2)
        X, y = ParameterAnomalyDetector.create_sliding_windows(stream, window_size=4)

        detector = ParameterAnomalyDetector(input_dim=2, hidden_dim=8, num_layers=1)
        detector.fit(X, y, epochs=5)

        nll_scores = detector.score(X[:4], y[:4], metric="nll")
        assert len(nll_scores) == 4
        assert np.all(np.isfinite(nll_scores))

    def test_unfitted_exceptions(self):
        """Verify that accessing predict/detect on unfitted components raises RuntimeError."""
        unfitted_det = ParameterAnomalyDetector(input_dim=3)
        dummy_X = np.zeros((2, 4, 3))
        dummy_y = np.zeros((2, 3))

        with pytest.raises(RuntimeError):
            unfitted_det.predict(dummy_X, dummy_y)

        with pytest.raises(RuntimeError):
            unfitted_det.detect(dummy_X, dummy_y)

        unfitted_est = MultivariateGaussianEstimator()
        with pytest.raises(RuntimeError):
            unfitted_est.mahalanobis(np.zeros((2, 3)))

    def test_result_properties_and_diagnostics(self):
        """Verify ParameterAnomalyResult properties and formatting."""
        res = ParameterAnomalyResult(
            is_anomaly=np.array([False, True, False, True]),
            scores=np.array([1.2, 8.5, 0.8, 12.0]),
            threshold=4.0,
            predictions=np.zeros((4, 2)),
            targets=np.zeros((4, 2)),
            residuals=np.zeros((4, 2)),
            feature_contributions=np.zeros((4, 2)),
            feature_names=["bytes", "threads"],
            diagnostics=["Sample 1: ANOMALY", "Sample 3: ANOMALY"],
        )

        assert res.num_anomalies == 2
        assert res.anomaly_ratio == 0.5
        np.testing.assert_array_equal(res.anomalous_indices, np.array([1, 3]))

        summary_text = res.summary()
        assert "Total Sequences Analyzed : 4" in summary_text
        assert "Anomalies Flagged        : 2 (50.00%)" in summary_text

