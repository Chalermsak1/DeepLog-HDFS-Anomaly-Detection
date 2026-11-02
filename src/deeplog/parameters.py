"""
DeepLog Parameter Value Anomaly Model (Pillar 2)
=================================================
Implements the continuous parameter anomaly detection component of DeepLog:
1. `ParameterValuePredictor`: Stacked LSTM regressor modeling the temporal
   dynamics of continuous parameter vectors v_t in R^d over time.
2. `MultivariateGaussianEstimator`: Computes error residuals r_t = v_t - hat{v}_t,
   fits N(mu_r, Sigma_r) with robust covariance regularization (shrinkage, Ledoit-Wolf,
   and epsilon-identity), and computes Mahalanobis distance / Gaussian NLL.
3. `ParameterAnomalyDetector`: End-to-end detector unifying the predictor,
   residual covariance estimator, adaptive thresholding, and diagnostic attribution
   for log parameters (e.g. byte sizes, thread counts, queue surges).
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import os

import numpy as np
import scipy.stats
import torch
import torch.nn as nn


# ==============================================================================
# Pillar 2: Parameter Value Predictor (LSTM Regressor)
# ==============================================================================

class ParameterValuePredictor(nn.Module):
    """
    Multivariate LSTM regressor for predicting continuous log parameter vectors.

    Given a historical sequence of parameter vectors [v_{t-w+1}, ..., v_t] in R^{w x d},
    predicts the next parameter value vector hat{v}_{t+1} in R^d.

    Parameters
    ----------
    input_dim : int
        Dimensionality d of the parameter vector at each time step.
    hidden_dim : int, default=64
        Number of hidden units in each LSTM layer.
    num_layers : int, default=2
        Number of stacked LSTM layers.
    output_dim : int, optional
        Dimensionality of prediction target. Defaults to input_dim.
    dropout : float, default=0.0
        Dropout probability between LSTM layers (active if num_layers > 1).
    bidirectional : bool, default=False
        Whether to use a bidirectional LSTM (default False for causal time-series).
    input_size : int, optional
        Alias for input_dim (DeepLog convention compatibility).
    hidden_size : int, optional
        Alias for hidden_dim.
    output_size : int, optional
        Alias for output_dim.
    """

    def __init__(
        self,
        input_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        num_layers: int = 2,
        output_dim: Optional[int] = None,
        dropout: float = 0.0,
        bidirectional: bool = False,
        *,
        input_size: Optional[int] = None,
        hidden_size: Optional[int] = None,
        output_size: Optional[int] = None,
    ):
        super().__init__()

        # Resolve aliases
        self.input_dim = input_dim if input_dim is not None else input_size
        if self.input_dim is None:
            raise ValueError("Must specify 'input_dim' (or 'input_size').")

        self.hidden_dim = hidden_dim if hidden_dim is not None else (hidden_size if hidden_size is not None else 64)
        self.output_dim = output_dim if output_dim is not None else (output_size if output_size is not None else self.input_dim)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.bidirectional = bool(bidirectional)

        # Aliases for DeepLog API compatibility
        self.input_size = self.input_dim
        self.hidden_size = self.hidden_dim
        self.output_size = self.output_dim

        # Stacked LSTM backbone
        lstm_dropout = self.dropout if self.num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=self.bidirectional,
        )

        # Projection head to R^d
        lstm_out_dim = self.hidden_dim * 2 if self.bidirectional else self.hidden_dim
        self.fc = nn.Linear(lstm_out_dim, self.output_dim)

    def forward(
        self,
        x: torch.Tensor,
        return_all_steps: bool = False,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass through LSTM regressor.

        Parameters
        ----------
        x : torch.Tensor of shape (batch_size, seq_len, input_dim)
            Continuous parameter sequence.
        return_all_steps : bool, default=False
            If True, returns predictions at all time steps (batch_size, seq_len, output_dim).
            If False, returns prediction for the next step (batch_size, output_dim).
        hidden : tuple of (h_0, c_0), optional
            Initial hidden states for LSTM.

        Returns
        -------
        torch.Tensor or tuple
            If hidden is None and return_all_steps is False:
                pred : torch.Tensor of shape (batch_size, output_dim)
            If return_all_steps is True:
                pred : torch.Tensor of shape (batch_size, seq_len, output_dim)
            If hidden is provided:
                (pred, (h_n, c_n))
        """
        if x.dim() == 2:
            # (seq_len, input_dim) -> (1, seq_len, input_dim)
            x = x.unsqueeze(0)

        lstm_out, next_hidden = self.lstm(x, hidden)

        if return_all_steps:
            out = self.fc(lstm_out)
        else:
            out = self.fc(lstm_out[:, -1, :])

        if hidden is not None:
            return out, next_hidden
        return out

    def predict(
        self,
        x: Union[torch.Tensor, np.ndarray],
        return_all_steps: bool = False,
    ) -> np.ndarray:
        """
        Perform inference and return predictions as a NumPy array.

        Parameters
        ----------
        x : torch.Tensor or np.ndarray of shape (batch_size, seq_len, input_dim)
            Input parameter sequences.
        return_all_steps : bool, default=False
            Whether to return predictions for all timesteps.

        Returns
        -------
        np.ndarray
            Predictions of shape (batch_size, output_dim) or (batch_size, seq_len, output_dim).
        """
        self.eval()
        device = next(self.parameters()).device

        if isinstance(x, np.ndarray):
            tensor_x = torch.from_numpy(x).to(dtype=torch.float32, device=device)
        else:
            tensor_x = x.to(dtype=torch.float32, device=device)

        if tensor_x.dim() == 2:
            tensor_x = tensor_x.unsqueeze(0)

        with torch.no_grad():
            preds = self.forward(tensor_x, return_all_steps=return_all_steps)
            if isinstance(preds, tuple):
                preds = preds[0]

        return preds.detach().cpu().numpy()

    def predict_multistep(
        self,
        x: Union[torch.Tensor, np.ndarray],
        steps: int = 1,
        keep_window: bool = True,
    ) -> np.ndarray:
        """
        Autoregressive multi-step parameter sequence forecasting.

        Rolls predictions forward by feeding predicted hat{v}_{t+1} back into
        the context window to forecast hat{v}_{t+2}, ..., hat{v}_{t+steps}.

        Parameters
        ----------
        x : torch.Tensor or np.ndarray of shape (batch_size, seq_len, input_dim)
            Seed context parameter sequence.
        steps : int, default=1
            Number of future timesteps to forecast (H >= 1).
        keep_window : bool, default=True
            If True, slides context window preserving seq_len.
            If False, expands context window.

        Returns
        -------
        np.ndarray of shape (batch_size, steps, output_dim)
            Multi-step forecast trajectory.
        """
        if steps < 1:
            raise ValueError(f"steps must be >= 1, got {steps}")

        self.eval()
        device = next(self.parameters()).device

        if isinstance(x, np.ndarray):
            curr_x = torch.from_numpy(x).to(dtype=torch.float32, device=device)
        else:
            curr_x = x.clone().to(dtype=torch.float32, device=device)

        if curr_x.dim() == 2:
            curr_x = curr_x.unsqueeze(0)

        batch_size, seq_len, _ = curr_x.shape
        predictions: List[torch.Tensor] = []

        with torch.no_grad():
            for _ in range(steps):
                # Predict next step from current context
                next_pred = self.forward(curr_x, return_all_steps=False)  # (batch_size, output_dim)
                predictions.append(next_pred.unsqueeze(1))  # (batch_size, 1, output_dim)

                next_pred_token = next_pred.unsqueeze(1)
                if keep_window and curr_x.size(1) >= seq_len:
                    curr_x = torch.cat([curr_x[:, 1:, :], next_pred_token], dim=1)
                else:
                    curr_x = torch.cat([curr_x, next_pred_token], dim=1)

        stacked = torch.cat(predictions, dim=1)  # (batch_size, steps, output_dim)
        return stacked.detach().cpu().numpy()

    def fit(
        self,
        X: Union[torch.Tensor, np.ndarray],
        y: Optional[Union[torch.Tensor, np.ndarray]] = None,
        epochs: int = 20,
        batch_size: int = 32,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        val_data: Optional[Tuple[Union[torch.Tensor, np.ndarray], Union[torch.Tensor, np.ndarray]]] = None,
        verbose: bool = False,
    ) -> Dict[str, List[float]]:
        """
        Train the parameter predictor using Mean Squared Error (MSE) loss.

        Parameters
        ----------
        X : torch.Tensor or np.ndarray of shape (N, seq_len, input_dim)
            Context sequences. If y is None and X has seq_len > 1, the last
            timestep is used as target y, and X[:, :-1, :] as inputs.
        y : torch.Tensor or np.ndarray of shape (N, output_dim), optional
            Ground-truth parameter targets for next step.
        epochs : int, default=20
            Number of training epochs.
        batch_size : int, default=32
            Mini-batch size.
        lr : float, default=1e-3
            Learning rate for Adam optimizer.
        weight_decay : float, default=1e-5
            L2 regularization penalty.
        val_data : tuple of (X_val, y_val), optional
            Validation data split for tracking validation loss.
        verbose : bool, default=False
            Whether to print per-epoch loss metrics.

        Returns
        -------
        Dict[str, List[float]]
            Training history dictionary containing 'train_loss' and 'val_loss'.
        """
        device = next(self.parameters()).device

        # Format inputs and targets
        if isinstance(X, np.ndarray):
            X_tensor = torch.from_numpy(X).float()
        else:
            X_tensor = X.float()

        if y is None:
            if X_tensor.size(1) < 2:
                raise ValueError("When y is None, X must have seq_len >= 2 to form next-step target.")
            y_tensor = X_tensor[:, -1, :]
            X_tensor = X_tensor[:, :-1, :]
        else:
            if isinstance(y, np.ndarray):
                y_tensor = torch.from_numpy(y).float()
            else:
                y_tensor = y.float()

        X_tensor = X_tensor.to(device)
        y_tensor = y_tensor.to(device)

        # Validation split formatting
        val_X_tensor, val_y_tensor = None, None
        if val_data is not None:
            vx, vy = val_data
            val_X_tensor = torch.from_numpy(vx).float().to(device) if isinstance(vx, np.ndarray) else vx.float().to(device)
            val_y_tensor = torch.from_numpy(vy).float().to(device) if isinstance(vy, np.ndarray) else vy.float().to(device)

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.MSELoss()

        history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}

        for epoch in range(epochs):
            self.train()
            running_loss = 0.0
            total_samples = 0

            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                pred = self.forward(batch_x, return_all_steps=False)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * batch_x.size(0)
                total_samples += batch_x.size(0)

            epoch_loss = running_loss / max(1, total_samples)
            history["train_loss"].append(epoch_loss)

            if val_X_tensor is not None and val_y_tensor is not None:
                self.eval()
                with torch.no_grad():
                    val_pred = self.forward(val_X_tensor, return_all_steps=False)
                    vloss = criterion(val_pred, val_y_tensor).item()
                    history["val_loss"].append(vloss)

            if verbose and ((epoch + 1) % max(1, epochs // 5) == 0 or epoch == epochs - 1):
                msg = f"Epoch [{epoch + 1}/{epochs}] - Train MSE: {epoch_loss:.6f}"
                if history["val_loss"]:
                    msg += f" - Val MSE: {history['val_loss'][-1]:.6f}"
                print(msg)

        self.eval()
        return history

    def save(self, filepath: str) -> None:
        """
        Save model architecture configuration and weights to file.

        Parameters
        ----------
        filepath : str
            Target path for saving model checkpoint (.pt).
        """
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        checkpoint = {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "output_dim": self.output_dim,
            "dropout": self.dropout,
            "bidirectional": self.bidirectional,
            "state_dict": self.state_dict(),
        }
        torch.save(checkpoint, filepath)

    @classmethod
    def load(cls, filepath: str, device: Optional[torch.device] = None) -> "ParameterValuePredictor":
        """
        Load a trained ParameterValuePredictor from checkpoint.

        Parameters
        ----------
        filepath : str
            Path to model checkpoint.
        device : torch.device, optional
            Device onto which the model is loaded.

        Returns
        -------
        ParameterValuePredictor
            Initialized model loaded with trained weights.
        """
        checkpoint = torch.load(filepath, map_location=device or "cpu")
        model = cls(
            input_dim=checkpoint["input_dim"],
            hidden_dim=checkpoint["hidden_dim"],
            num_layers=checkpoint["num_layers"],
            output_dim=checkpoint["output_dim"],
            dropout=checkpoint.get("dropout", 0.0),
            bidirectional=checkpoint.get("bidirectional", False),
        )
        model.load_state_dict(checkpoint["state_dict"])
        if device is not None:
            model = model.to(device)
        model.eval()
        return model


# ==============================================================================
# Pillar 2: Multivariate Gaussian Residual Estimator & Mahalanobis Engine
# ==============================================================================

class MultivariateGaussianEstimator:
    """
    Fits N(mu_r, Sigma_r) to parameter error residuals r_t = v_t - hat{v}_t in R^d.

    Computes Mahalanobis distance and Gaussian Negative Log-Likelihood (NLL).
    Handles potential singular or ill-conditioned covariance matrices through:
    1. Shrinkage towards structured targets (diagonal, spherical, or Ledoit-Wolf).
    2. Epsilon-identity diagonal regularization (jitter): Sigma_reg = Sigma + eps * I.
    3. Spectral eigenvalue floor clipping: lambda_i >= eps.

    Parameters
    ----------
    regularization : str, default="shrinkage"
        Covariance regularization method:
        - "shrinkage": convex combination (1 - lambda) * Sigma + lambda * Target + eps * I.
        - "ledoit_wolf": analytical Ledoit-Wolf optimal shrinkage estimator.
        - "epsilon": diagonal jitter Sigma + eps * I.
        - "diagonal": retains only diagonal variance elements (independent assumption).
        - "none": unregularized sample covariance with spectral floor.
    epsilon : float, default=1e-5
        Diagonal jitter and spectral eigenvalue floor for numerical stability.
    shrinkage_factor : float, optional
        Convex shrinkage weight lambda in [0, 1]. If None and regularization="shrinkage",
        defaults to 0.1.
    threshold_method : str, default="chi2"
        Threshold decision strategy:
        - "chi2": theoretical chi-squared quantile at (1 - alpha) with d degrees of freedom.
        - "quantile": empirical quantile of training Mahalanobis distances at (1 - alpha).
        - "sigma": mean + k_sigma * std deviation threshold.
        - "manual": explicit user-provided threshold.
    alpha : float, default=0.01
        Significance level for chi-squared or empirical quantile thresholding.
    k_sigma : float, default=3.0
        Standard deviation multiplier for "sigma" thresholding.
    manual_threshold : float, optional
        Pre-set manual threshold value.
    """

    def __init__(
        self,
        regularization: str = "shrinkage",
        epsilon: float = 1e-5,
        shrinkage_factor: Optional[float] = None,
        threshold_method: str = "chi2",
        alpha: float = 0.01,
        k_sigma: float = 3.0,
        manual_threshold: Optional[float] = None,
    ):
        valid_regs = {"shrinkage", "ledoit_wolf", "epsilon", "diagonal", "none"}
        if regularization not in valid_regs:
            raise ValueError(f"Unknown regularization '{regularization}', must be one of {valid_regs}")

        valid_thresh = {"chi2", "quantile", "sigma", "manual"}
        if threshold_method not in valid_thresh:
            raise ValueError(f"Unknown threshold_method '{threshold_method}', must be one of {valid_thresh}")

        self.regularization = regularization
        self.epsilon = float(epsilon)
        self.shrinkage_factor = float(shrinkage_factor) if shrinkage_factor is not None else 0.1
        self.threshold_method = threshold_method
        self.alpha = float(alpha)
        self.k_sigma = float(k_sigma)
        self.manual_threshold = float(manual_threshold) if manual_threshold is not None else None

        # Fitted parameters
        self.mean: Optional[np.ndarray] = None          # (d,)
        self.covariance: Optional[np.ndarray] = None    # (d, d)
        self.precision: Optional[np.ndarray] = None     # (d, d)
        self.log_det: Optional[float] = None            # ln |Sigma|
        self.threshold: Optional[float] = None          # Decision boundary
        self.fitted_dim: Optional[int] = None           # d

    def fit(
        self,
        residuals: Union[np.ndarray, torch.Tensor],
        threshold_method: Optional[str] = None,
        alpha: Optional[float] = None,
        k_sigma: Optional[float] = None,
    ) -> "MultivariateGaussianEstimator":
        """
        Fit mean vector and regularized covariance matrix on residual errors.

        Parameters
        ----------
        residuals : np.ndarray or torch.Tensor of shape (N, d) or (N,)
            Residual vectors r = v - hat{v}.
        threshold_method : str, optional
            Override default thresholding method.
        alpha : float, optional
            Override significance level.
        k_sigma : float, optional
            Override sigma multiplier.

        Returns
        -------
        self : MultivariateGaussianEstimator
            Fitted estimator instance.
        """
        R = self._to_numpy_2d(residuals)
        N, d = R.shape

        if N < 1:
            raise ValueError("Cannot fit MultivariateGaussianEstimator on empty residuals array.")

        self.fitted_dim = d

        # 1. Sample mean: mu = (1/N) * sum(r_i)
        self.mean = np.mean(R, axis=0)  # (d,)

        # 2. Empirical covariance: Sigma_emp
        centered = R - self.mean  # (N, d)
        if N > 1:
            cov_emp = (centered.T @ centered) / (N - 1)
        else:
            cov_emp = np.zeros((d, d), dtype=np.float64)

        # 3. Apply Regularization
        cov_reg = self._regularize_covariance(cov_emp, R)

        # 4. Spectral floor & Positive Definiteness enforcement
        # Eigendecomposition of symmetric covariance: Sigma = V * diag(lambda) * V^T
        cov_sym = 0.5 * (cov_reg + cov_reg.T)
        eigvals, eigvecs = np.linalg.eigh(cov_sym)

        # Floor eigenvalues at epsilon to strictly guarantee positive definiteness
        clipped_eigvals = np.maximum(eigvals, self.epsilon)
        cov_spd = eigvecs @ np.diag(clipped_eigvals) @ eigvecs.T
        self.covariance = 0.5 * (cov_spd + cov_spd.T)

        # 5. Stable Precision Matrix: Sigma^{-1} = V * diag(1 / lambda) * V^T
        inv_eigvals = 1.0 / clipped_eigvals
        self.precision = eigvecs @ np.diag(inv_eigvals) @ eigvecs.T

        # 6. Log-determinant: ln |Sigma| = sum(ln(clipped_eigvals))
        self.log_det = float(np.sum(np.log(clipped_eigvals)))

        # 7. Compute decision threshold
        t_method = threshold_method or self.threshold_method
        t_alpha = alpha if alpha is not None else self.alpha
        t_k = k_sigma if k_sigma is not None else self.k_sigma

        self.set_threshold(
            method=t_method,
            alpha=t_alpha,
            k_sigma=t_k,
            threshold=self.manual_threshold,
            residuals=R,
        )

        return self

    def _regularize_covariance(self, cov_emp: np.ndarray, residuals: np.ndarray) -> np.ndarray:
        """Apply selected regularization strategy to empirical covariance."""
        d = cov_emp.shape[0]
        N = residuals.shape[0]

        if self.regularization == "none":
            return cov_emp + self.epsilon * np.eye(d)

        elif self.regularization == "epsilon":
            return cov_emp + self.epsilon * np.eye(d)

        elif self.regularization == "diagonal":
            diag_elements = np.diag(cov_emp).copy()
            diag_elements = np.maximum(diag_elements, self.epsilon)
            return np.diag(diag_elements)

        elif self.regularization == "ledoit_wolf":
            # Attempt Ledoit-Wolf shrinkage from scikit-learn
            if N >= 3:
                try:
                    import sklearn.covariance
                    lw_cov, _ = sklearn.covariance.ledoit_wolf(residuals, assume_centered=False)
                    return lw_cov + self.epsilon * np.eye(d)
                except Exception:
                    pass
            # Fallback to analytical shrinkage towards identity
            lam = min(1.0, max(0.01, self.shrinkage_factor))
            target = (np.trace(cov_emp) / max(1, d)) * np.eye(d)
            return (1.0 - lam) * cov_emp + lam * target + self.epsilon * np.eye(d)

        elif self.regularization == "shrinkage":
            lam = min(1.0, max(0.0, self.shrinkage_factor))
            # Diagonal shrinkage target preserving individual feature variances
            diag_target = np.diag(np.diag(cov_emp))
            cov_shrunk = (1.0 - lam) * cov_emp + lam * diag_target
            return cov_shrunk + self.epsilon * np.eye(d)

        return cov_emp + self.epsilon * np.eye(d)

    def mahalanobis(
        self,
        residuals: Union[np.ndarray, torch.Tensor],
        squared: bool = False,
    ) -> np.ndarray:
        """
        Compute Mahalanobis distance D_M(r) for residual vectors.

        D_M(r) = sqrt( (r - mu)^T Sigma^{-1} (r - mu) )

        Parameters
        ----------
        residuals : np.ndarray or torch.Tensor of shape (N, d) or (d,)
            Residual vectors.
        squared : bool, default=False
            If True, returns squared Mahalanobis distance D_M^2(r).

        Returns
        -------
        np.ndarray of shape (N,)
            Mahalanobis distance for each sample.
        """
        self._check_is_fitted()
        R = self._to_numpy_2d(residuals)

        # Delta = (r - mu)
        diff = R - self.mean  # (N, d)

        # D_M^2 = sum_j (diff @ precision)_j * diff_j
        left = diff @ self.precision  # (N, d)
        sq_dist = np.sum(left * diff, axis=-1)  # (N,)
        sq_dist = np.maximum(sq_dist, 0.0)

        if squared:
            return sq_dist
        return np.sqrt(sq_dist)

    def nll(
        self,
        residuals: Union[np.ndarray, torch.Tensor],
    ) -> np.ndarray:
        """
        Compute Gaussian Negative Log-Likelihood (NLL) for residual vectors.

        NLL(r) = 0.5 * D_M^2(r) + 0.5 * ln|Sigma| + 0.5 * d * ln(2*pi)

        Parameters
        ----------
        residuals : np.ndarray or torch.Tensor of shape (N, d) or (d,)
            Residual vectors.

        Returns
        -------
        np.ndarray of shape (N,)
            NLL values for each sample.
        """
        self._check_is_fitted()
        sq_dist = self.mahalanobis(residuals, squared=True)
        d = self.fitted_dim
        nll_val = 0.5 * sq_dist + 0.5 * self.log_det + 0.5 * d * np.log(2.0 * np.pi)
        return nll_val

    def score(
        self,
        residuals: Union[np.ndarray, torch.Tensor],
        metric: str = "mahalanobis",
    ) -> np.ndarray:
        """
        Compute anomaly score for residual vectors.

        Parameters
        ----------
        residuals : np.ndarray or torch.Tensor of shape (N, d)
            Residual vectors.
        metric : str, default="mahalanobis"
            Scoring metric: "mahalanobis", "mahalanobis_sq", or "nll".

        Returns
        -------
        np.ndarray of shape (N,)
            Anomaly score values.
        """
        if metric == "mahalanobis":
            return self.mahalanobis(residuals, squared=False)
        elif metric == "mahalanobis_sq":
            return self.mahalanobis(residuals, squared=True)
        elif metric == "nll":
            return self.nll(residuals)
        else:
            raise ValueError(f"Unknown metric '{metric}'. Expected 'mahalanobis', 'mahalanobis_sq', or 'nll'.")

    def predict(
        self,
        residuals: Union[np.ndarray, torch.Tensor],
        metric: str = "mahalanobis",
    ) -> np.ndarray:
        """
        Predict binary anomaly status for residual vectors.

        Returns True if score > threshold, else False.

        Parameters
        ----------
        residuals : np.ndarray or torch.Tensor of shape (N, d)
            Residual vectors.
        metric : str, default="mahalanobis"
            Scoring metric matching the fitted threshold.

        Returns
        -------
        np.ndarray of shape (N,), dtype=bool
            Boolean anomaly flags.
        """
        scores = self.score(residuals, metric=metric)
        return scores > self.threshold

    def explain(
        self,
        residuals: Union[np.ndarray, torch.Tensor],
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Explain parameter anomalies via exact additive decomposition of D_M^2
        and feature-level z-scores.

        For each parameter j in {1, ..., d}:
        - Component contribution: c_j = (r_j - mu_j) * [Sigma^{-1}(r - mu)]_j
          Note that sum_{j=1}^d c_j = D_M^2(r).
        - Normalized z-score: z_j = (r_j - mu_j) / sqrt(Sigma_{jj})

        Parameters
        ----------
        residuals : np.ndarray or torch.Tensor of shape (N, d)
            Residual vectors.
        feature_names : list of str, optional
            Human-readable names for each parameter. Defaults to ['param_0', ...].

        Returns
        -------
        dict with keys:
            - 'contributions': np.ndarray of shape (N, d)
            - 'contribution_ratios': np.ndarray of shape (N, d) in [0, 1]
            - 'z_scores': np.ndarray of shape (N, d)
            - 'top_features': list of str or int indicating highest anomalous feature per sample
        """
        self._check_is_fitted()
        R = self._to_numpy_2d(residuals)
        N, d = R.shape

        if feature_names is None:
            names = [f"param_{j}" for j in range(d)]
        else:
            if len(feature_names) != d:
                raise ValueError(f"Length of feature_names ({len(feature_names)}) != dimension ({d})")
            names = list(feature_names)

        diff = R - self.mean  # (N, d)
        prec_diff = diff @ self.precision  # (N, d)

        # Additive decomposition of squared Mahalanobis distance
        contributions = diff * prec_diff  # (N, d)
        total_sq = np.sum(contributions, axis=-1, keepdims=True)  # (N, 1)
        total_sq_safe = np.maximum(total_sq, 1e-9)
        contribution_ratios = np.maximum(contributions, 0.0) / total_sq_safe

        # Standardized z-scores
        std_devs = np.sqrt(np.maximum(np.diag(self.covariance), 1e-12))  # (d,)
        z_scores = diff / std_devs  # (N, d)

        # Identify dominant anomalous feature per sample
        top_indices = np.argmax(np.abs(contributions), axis=-1)
        top_features = [names[idx] for idx in top_indices]

        return {
            "contributions": contributions,
            "contribution_ratios": contribution_ratios,
            "z_scores": z_scores,
            "top_features": top_features,
            "feature_names": names,
        }

    def set_threshold(
        self,
        method: str = "chi2",
        alpha: float = 0.01,
        k_sigma: float = 3.0,
        threshold: Optional[float] = None,
        residuals: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> float:
        """
        Configure the anomaly decision boundary threshold.

        Parameters
        ----------
        method : str, default="chi2"
            "chi2", "quantile", "sigma", or "manual".
        alpha : float, default=0.01
            Significance level.
        k_sigma : float, default=3.0
            Sigma multiplier.
        threshold : float, optional
            Direct manual threshold.
        residuals : np.ndarray, optional
            Calibration residuals (required for "quantile" and "sigma").

        Returns
        -------
        float
            Computed threshold value for Mahalanobis distance D_M.
        """
        self._check_is_fitted()
        d = self.fitted_dim

        if threshold is not None or method == "manual":
            if threshold is None:
                raise ValueError("Must provide 'threshold' when method='manual'.")
            self.threshold = float(threshold)
            self.threshold_method = "manual"
            return self.threshold

        if method == "chi2":
            # Theoretical chi-squared quantile with d degrees of freedom
            # Under Gaussian assumption: D_M^2 ~ Chi2(d)
            chi2_crit = scipy.stats.chi2.ppf(1.0 - alpha, df=d)
            self.threshold = float(np.sqrt(max(0.0, chi2_crit)))
            self.threshold_method = "chi2"
            self.alpha = alpha

        elif method in {"quantile", "sigma"}:
            if residuals is None:
                raise ValueError(f"Calibration residuals are required for threshold method '{method}'.")
            dists = self.mahalanobis(residuals, squared=False)

            if method == "quantile":
                self.threshold = float(np.percentile(dists, 100.0 * (1.0 - alpha)))
                self.threshold_method = "quantile"
                self.alpha = alpha
            else:  # sigma
                mean_d = float(np.mean(dists))
                std_d = float(np.std(dists))
                self.threshold = mean_d + k_sigma * std_d
                self.threshold_method = "sigma"
                self.k_sigma = k_sigma
        else:
            raise ValueError(f"Unknown threshold method: {method}")

        return self.threshold

    def _to_numpy_2d(self, arr: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Convert input tensor or array to 2D float64 NumPy array."""
        if isinstance(arr, torch.Tensor):
            np_arr = arr.detach().cpu().numpy().astype(np.float64)
        else:
            np_arr = np.asarray(arr, dtype=np.float64)

        if np_arr.ndim == 1:
            np_arr = np_arr[:, np.newaxis]
        elif np_arr.ndim > 2:
            raise ValueError(f"Expected 1D or 2D array, got {np_arr.ndim}D.")
        return np_arr

    def _check_is_fitted(self) -> None:
        """Verify that the estimator has been fitted."""
        if self.mean is None or self.covariance is None or self.precision is None:
            raise RuntimeError("MultivariateGaussianEstimator is not fitted. Call fit() first.")

    def save(self, filepath: str) -> None:
        """Save estimator parameters and threshold to file."""
        self._check_is_fitted()
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        data = {
            "mean": self.mean,
            "covariance": self.covariance,
            "precision": self.precision,
            "log_det": self.log_det,
            "threshold": self.threshold,
            "fitted_dim": self.fitted_dim,
            "regularization": self.regularization,
            "epsilon": self.epsilon,
            "shrinkage_factor": self.shrinkage_factor,
            "threshold_method": self.threshold_method,
            "alpha": self.alpha,
            "k_sigma": self.k_sigma,
        }
        torch.save(data, filepath)

    @classmethod
    def load(cls, filepath: str) -> "MultivariateGaussianEstimator":
        """Load estimator parameters and threshold from file."""
        data = torch.load(filepath, map_location="cpu", weights_only=False)
        estimator = cls(
            regularization=data["regularization"],
            epsilon=data["epsilon"],
            shrinkage_factor=data["shrinkage_factor"],
            threshold_method=data["threshold_method"],
            alpha=data["alpha"],
            k_sigma=data["k_sigma"],
        )
        estimator.mean = data["mean"]
        estimator.covariance = data["covariance"]
        estimator.precision = data["precision"]
        estimator.log_det = data["log_det"]
        estimator.threshold = data["threshold"]
        estimator.fitted_dim = data["fitted_dim"]
        return estimator


# ==============================================================================
# Pillar 2: Parameter Anomaly Result Dataclass
# ==============================================================================

@dataclass
class ParameterAnomalyResult:
    """
    Structured outcome container for ParameterAnomalyDetector detection passes.

    Attributes
    ----------
    is_anomaly : np.ndarray of shape (N,), dtype=bool
        Flags indicating whether each sample violates the decision threshold.
    scores : np.ndarray of shape (N,)
        Mahalanobis distance (or NLL) scores.
    threshold : float
        Decision threshold utilized.
    predictions : np.ndarray of shape (N, d)
        Model predicted parameter vectors hat{v}.
    targets : np.ndarray of shape (N, d)
        Observed ground truth parameter vectors v.
    residuals : np.ndarray of shape (N, d)
        Residual error vectors r = v - hat{v}.
    feature_contributions : np.ndarray of shape (N, d)
        Additive feature contributions to Mahalanobis distance squared.
    feature_names : List[str]
        Descriptive names of parameters.
    diagnostics : List[str]
        Per-sample explanatory diagnostic strings for anomalies.
    """
    is_anomaly: np.ndarray
    scores: np.ndarray
    threshold: float
    predictions: np.ndarray
    targets: np.ndarray
    residuals: np.ndarray
    feature_contributions: np.ndarray
    feature_names: List[str]
    diagnostics: List[str]

    @property
    def num_anomalies(self) -> int:
        """Total number of anomalous samples detected."""
        return int(np.sum(self.is_anomaly))

    @property
    def anomaly_ratio(self) -> float:
        """Ratio of anomalous samples to total samples."""
        return float(np.mean(self.is_anomaly)) if len(self.is_anomaly) > 0 else 0.0

    @property
    def anomalous_indices(self) -> np.ndarray:
        """1D array of indices corresponding to flagged anomalies."""
        return np.where(self.is_anomaly)[0]

    def summary(self) -> str:
        """Format an executive diagnostic report of detection results."""
        lines = [
            "===========================================================",
            " DeepLog Parameter Value Anomaly Detection Summary",
            "===========================================================",
            f" Total Sequences Analyzed : {len(self.is_anomaly)}",
            f" Anomalies Flagged        : {self.num_anomalies} ({self.anomaly_ratio * 100:.2f}%)",
            f" Decision Threshold       : {self.threshold:.4f}",
            f" Min / Mean / Max Score   : {np.min(self.scores):.4f} / {np.mean(self.scores):.4f} / {np.max(self.scores):.4f}",
            "-----------------------------------------------------------",
        ]
        if self.num_anomalies > 0:
            lines.append("Flagged Sample Diagnostics (First 5):")
            for diag in self.diagnostics[:5]:
                lines.append(f"  * {diag}")
        else:
            lines.append("All parameter vectors are within normal confidence bounds.")
        lines.append("===========================================================")
        return "\n".join(lines)


# ==============================================================================
# Pillar 2: Parameter Anomaly Detector (End-to-End Orchestrator)
# ==============================================================================

class ParameterAnomalyDetector:
    """
    Unified Parameter Value Anomaly Detector (DeepLog Pillar 2).

    Orchestrates:
    - Sequence windowing & optional feature standardization.
    - LSTM-based multivariate regression forecasting (ParameterValuePredictor).
    - Error residual modeling with regularized covariance (MultivariateGaussianEstimator).
    - Mahalanobis / Gaussian NLL scoring and automated thresholding.
    - Detailed parameter-level diagnostic attribution for alerts (e.g. byte size anomalies,
      thread pool exhaustion, queue surges).

    Parameters
    ----------
    input_dim : int, optional
        Dimensionality d of parameter vectors. Inferred automatically if not provided.
    hidden_dim : int, default=64
        Hidden units in LSTM regressor.
    num_layers : int, default=2
        Number of LSTM layers.
    dropout : float, default=0.0
        Dropout probability in LSTM.
    regularization : str, default="shrinkage"
        Covariance regularization method ("shrinkage", "ledoit_wolf", "epsilon", "diagonal").
    epsilon : float, default=1e-5
        Diagonal regularizer jitter and spectral floor.
    threshold_method : str, default="chi2"
        Threshold determination strategy ("chi2", "quantile", "sigma", "manual").
    alpha : float, default=0.01
        Significance level for anomaly boundary.
    k_sigma : float, default=3.0
        Sigma multiplier for "sigma" thresholding.
    normalize : bool, default=True
        Whether to standardize features to zero-mean unit-variance prior to LSTM fitting.
    feature_names : list of str, optional
        Descriptive names for parameters (e.g. ['byte_size', 'thread_count', 'queue_latency']).
    predictor : ParameterValuePredictor, optional
        Pre-instantiated predictor instance.
    estimator : MultivariateGaussianEstimator, optional
        Pre-instantiated estimator instance.
    """

    def __init__(
        self,
        input_dim: Optional[int] = None,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.0,
        regularization: str = "shrinkage",
        epsilon: float = 1e-5,
        threshold_method: str = "chi2",
        alpha: float = 0.01,
        k_sigma: float = 3.0,
        normalize: bool = True,
        feature_names: Optional[List[str]] = None,
        predictor: Optional[ParameterValuePredictor] = None,
        estimator: Optional[MultivariateGaussianEstimator] = None,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.regularization = regularization
        self.epsilon = epsilon
        self.threshold_method = threshold_method
        self.alpha = alpha
        self.k_sigma = k_sigma
        self.normalize = normalize
        self.feature_names = feature_names

        # Subcomponents
        self.predictor = predictor
        self.estimator = estimator

        # Normalization statistics
        self.scaler_mean: Optional[np.ndarray] = None  # (d,)
        self.scaler_std: Optional[np.ndarray] = None   # (d,)

    @staticmethod
    def create_sliding_windows(
        series: Union[np.ndarray, torch.Tensor],
        window_size: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform a continuous multivariate parameter stream of shape (T, d)
        into sliding context windows X and next-step targets y.

        Parameters
        ----------
        series : np.ndarray or torch.Tensor of shape (T, d) or (T,)
            Continuous parameter stream.
        window_size : int
            Length of context window w.

        Returns
        -------
        X : np.ndarray of shape (T - window_size, window_size, d)
        y : np.ndarray of shape (T - window_size, d)
        """
        if isinstance(series, torch.Tensor):
            arr = series.detach().cpu().numpy().astype(np.float64)
        else:
            arr = np.asarray(series, dtype=np.float64)

        if arr.ndim == 1:
            arr = arr[:, np.newaxis]

        T, d = arr.shape
        if T <= window_size:
            raise ValueError(f"Sequence length ({T}) must be strictly greater than window_size ({window_size})")

        num_windows = T - window_size
        X = np.empty((num_windows, window_size, d), dtype=np.float64)
        y = np.empty((num_windows, d), dtype=np.float64)

        for i in range(num_windows):
            X[i] = arr[i : i + window_size]
            y[i] = arr[i + window_size]

        return X, y

    def _fit_scaler(self, data: np.ndarray) -> None:
        """Compute standardization statistics across sequence timesteps."""
        # data shape: (N, seq_len, d) or (N, d)
        if data.ndim == 3:
            flattened = data.reshape(-1, data.shape[-1])
        else:
            flattened = data

        self.scaler_mean = np.mean(flattened, axis=0)
        stds = np.std(flattened, axis=0)
        self.scaler_std = np.where(stds < 1e-7, 1.0, stds)

    def _scale(self, data: np.ndarray) -> np.ndarray:
        """Standardize data using fitted mean and std."""
        if not self.normalize or self.scaler_mean is None or self.scaler_std is None:
            return data
        return (data - self.scaler_mean) / self.scaler_std

    def _unscale(self, data: np.ndarray) -> np.ndarray:
        """Restore standardized data to original measurement scale."""
        if not self.normalize or self.scaler_mean is None or self.scaler_std is None:
            return data
        return data * self.scaler_std + self.scaler_mean

    def fit(
        self,
        X: Union[np.ndarray, torch.Tensor],
        y: Optional[Union[np.ndarray, torch.Tensor]] = None,
        epochs: int = 20,
        batch_size: int = 32,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        verbose: bool = False,
    ) -> "ParameterAnomalyDetector":
        """
        Train the parameter value predictor and fit residual covariance model.

        Parameters
        ----------
        X : np.ndarray or torch.Tensor of shape (N, seq_len, d)
            Input parameter sequences.
        y : np.ndarray or torch.Tensor of shape (N, d), optional
            Ground truth next-step parameter vectors. If None and seq_len >= 2,
            target is taken from the last timestep.
        epochs : int, default=20
            Number of training epochs.
        batch_size : int, default=32
            Batch size.
        lr : float, default=1e-3
            Learning rate.
        weight_decay : float, default=1e-5
            Weight decay.
        verbose : bool, default=False
            Whether to print progress.

        Returns
        -------
        self : ParameterAnomalyDetector
            Fitted detector instance.
        """
        if isinstance(X, torch.Tensor):
            X_np = X.detach().cpu().numpy().astype(np.float64)
        else:
            X_np = np.asarray(X, dtype=np.float64)

        if y is not None:
            if isinstance(y, torch.Tensor):
                y_np = y.detach().cpu().numpy().astype(np.float64)
            else:
                y_np = np.asarray(y, dtype=np.float64)
            if y_np.ndim == 1:
                y_np = y_np[:, np.newaxis]
        else:
            if X_np.shape[1] < 2:
                raise ValueError("When y is None, X seq_len must be >= 2.")
            y_np = X_np[:, -1, :]
            X_np = X_np[:, :-1, :]

        N, seq_len, d = X_np.shape
        self.input_dim = d

        # Setup feature names
        if self.feature_names is None:
            self.feature_names = [f"param_{j}" for j in range(d)]
        elif len(self.feature_names) != d:
            raise ValueError(f"feature_names length ({len(self.feature_names)}) != d ({d})")

        # Fit feature scaler
        if self.normalize:
            self._fit_scaler(X_np)
            X_train = self._scale(X_np)
            y_train = self._scale(y_np)
        else:
            X_train = X_np
            y_train = y_np

        # Initialize predictor if needed
        if self.predictor is None:
            self.predictor = ParameterValuePredictor(
                input_dim=d,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                output_dim=d,
                dropout=self.dropout,
            )

        # Train predictor
        self.predictor.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            verbose=verbose,
        )

        # Predict on training data to compute normal calibration residuals
        preds_scaled = self.predictor.predict(X_train, return_all_steps=False)
        residuals_train = y_train - preds_scaled  # Computed in standardized space

        # Fit residual Gaussian estimator
        if self.estimator is None:
            self.estimator = MultivariateGaussianEstimator(
                regularization=self.regularization,
                epsilon=self.epsilon,
                threshold_method=self.threshold_method,
                alpha=self.alpha,
                k_sigma=self.k_sigma,
            )

        self.estimator.fit(
            residuals_train,
            threshold_method=self.threshold_method,
            alpha=self.alpha,
            k_sigma=self.k_sigma,
        )

        return self

    def predict_residuals(
        self,
        X: Union[np.ndarray, torch.Tensor],
        y: Union[np.ndarray, torch.Tensor],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Forecast parameters and compute residual errors.

        Parameters
        ----------
        X : np.ndarray or torch.Tensor of shape (N, seq_len, d)
        y : np.ndarray or torch.Tensor of shape (N, d)

        Returns
        -------
        residuals : np.ndarray of shape (N, d)
            Residual errors r in normalized estimator space.
        preds_orig : np.ndarray of shape (N, d)
            Predictions in original domain measurement scale.
        targets_orig : np.ndarray of shape (N, d)
            Targets in original domain measurement scale.
        """
        self._check_is_fitted()

        if isinstance(X, torch.Tensor):
            X_np = X.detach().cpu().numpy().astype(np.float64)
        else:
            X_np = np.asarray(X, dtype=np.float64)

        if isinstance(y, torch.Tensor):
            y_np = y.detach().cpu().numpy().astype(np.float64)
        else:
            y_np = np.asarray(y, dtype=np.float64)
        if y_np.ndim == 1:
            y_np = y_np[:, np.newaxis]

        X_scaled = self._scale(X_np)
        y_scaled = self._scale(y_np)

        preds_scaled = self.predictor.predict(X_scaled, return_all_steps=False)
        residuals = y_scaled - preds_scaled

        preds_orig = self._unscale(preds_scaled)
        targets_orig = y_np

        return residuals, preds_orig, targets_orig

    def score(
        self,
        X: Union[np.ndarray, torch.Tensor],
        y: Union[np.ndarray, torch.Tensor],
        metric: str = "mahalanobis",
    ) -> np.ndarray:
        """
        Compute anomaly score for parameter sequences.

        Parameters
        ----------
        X : np.ndarray or torch.Tensor of shape (N, seq_len, d)
        y : np.ndarray or torch.Tensor of shape (N, d)
        metric : str, default="mahalanobis"

        Returns
        -------
        np.ndarray of shape (N,)
            Anomaly scores.
        """
        residuals, _, _ = self.predict_residuals(X, y)
        return self.estimator.score(residuals, metric=metric)

    def predict(
        self,
        X: Union[np.ndarray, torch.Tensor],
        y: Union[np.ndarray, torch.Tensor],
        metric: str = "mahalanobis",
    ) -> np.ndarray:
        """
        Classify parameter sequences as normal (False) or anomalous (True).

        Parameters
        ----------
        X : np.ndarray or torch.Tensor of shape (N, seq_len, d)
        y : np.ndarray or torch.Tensor of shape (N, d)
        metric : str, default="mahalanobis"

        Returns
        -------
        np.ndarray of shape (N,), dtype=bool
        """
        residuals, _, _ = self.predict_residuals(X, y)
        return self.estimator.predict(residuals, metric=metric)

    def detect(
        self,
        X: Union[np.ndarray, torch.Tensor],
        y: Union[np.ndarray, torch.Tensor],
        metric: str = "mahalanobis",
    ) -> ParameterAnomalyResult:
        """
        Perform complete diagnostic anomaly detection on parameter vectors.

        Parameters
        ----------
        X : np.ndarray or torch.Tensor of shape (N, seq_len, d)
        y : np.ndarray or torch.Tensor of shape (N, d)
        metric : str, default="mahalanobis"

        Returns
        -------
        ParameterAnomalyResult
            Comprehensive detection artifact with scores, predictions, residuals,
            feature-level attributions, and human-readable diagnostic messages.
        """
        residuals, preds_orig, targets_orig = self.predict_residuals(X, y)
        scores = self.estimator.score(residuals, metric=metric)
        is_anomaly = scores > self.estimator.threshold

        # Feature explanation and diagnostic messages
        explanation = self.estimator.explain(residuals, feature_names=self.feature_names)
        contributions = explanation["contributions"]
        contribution_ratios = explanation["contribution_ratios"]
        z_scores = explanation["z_scores"]
        top_features = explanation["top_features"]

        diagnostics: List[str] = []
        for idx in range(len(scores)):
            if is_anomaly[idx]:
                feat = top_features[idx]
                feat_idx = self.feature_names.index(feat)
                ratio_pct = contribution_ratios[idx, feat_idx] * 100.0
                z_val = z_scores[idx, feat_idx]
                actual_val = targets_orig[idx, feat_idx]
                pred_val = preds_orig[idx, feat_idx]

                msg = (
                    f"Sample {idx}: ANOMALY (score={scores[idx]:.2f} > thresh={self.estimator.threshold:.2f}). "
                    f"Primary parameter '{feat}': actual={actual_val:.2f}, pred={pred_val:.2f} "
                    f"(deviation={z_val:+.2f} sigma, contribution={ratio_pct:.1f}%)"
                )
                diagnostics.append(msg)

        return ParameterAnomalyResult(
            is_anomaly=is_anomaly,
            scores=scores,
            threshold=float(self.estimator.threshold),
            predictions=preds_orig,
            targets=targets_orig,
            residuals=residuals,
            feature_contributions=contributions,
            feature_names=self.feature_names,
            diagnostics=diagnostics,
        )

    def _check_is_fitted(self) -> None:
        """Ensure detector is fitted before inference."""
        if self.predictor is None or self.estimator is None:
            raise RuntimeError("ParameterAnomalyDetector is not fitted. Call fit() first.")

    def save(self, filepath: str) -> None:
        """
        Save complete ParameterAnomalyDetector pipeline to file.

        Parameters
        ----------
        filepath : str
            Destination checkpoint file (.pt).
        """
        self._check_is_fitted()
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        checkpoint = {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "regularization": self.regularization,
            "epsilon": self.epsilon,
            "threshold_method": self.threshold_method,
            "alpha": self.alpha,
            "k_sigma": self.k_sigma,
            "normalize": self.normalize,
            "feature_names": self.feature_names,
            "scaler_mean": self.scaler_mean,
            "scaler_std": self.scaler_std,
            "predictor_state": {
                "input_dim": self.predictor.input_dim,
                "hidden_dim": self.predictor.hidden_dim,
                "num_layers": self.predictor.num_layers,
                "output_dim": self.predictor.output_dim,
                "dropout": self.predictor.dropout,
                "bidirectional": self.predictor.bidirectional,
                "state_dict": self.predictor.state_dict(),
            },
            "estimator_state": {
                "mean": self.estimator.mean,
                "covariance": self.estimator.covariance,
                "precision": self.estimator.precision,
                "log_det": self.estimator.log_det,
                "threshold": self.estimator.threshold,
                "fitted_dim": self.estimator.fitted_dim,
                "regularization": self.estimator.regularization,
                "epsilon": self.estimator.epsilon,
                "shrinkage_factor": self.estimator.shrinkage_factor,
                "threshold_method": self.estimator.threshold_method,
                "alpha": self.estimator.alpha,
                "k_sigma": self.estimator.k_sigma,
            },
        }
        torch.save(checkpoint, filepath)

    @classmethod
    def load(cls, filepath: str, device: Optional[torch.device] = None) -> "ParameterAnomalyDetector":
        """
        Load complete ParameterAnomalyDetector pipeline from file.

        Parameters
        ----------
        filepath : str
            Path to saved detector checkpoint.
        device : torch.device, optional
            Target device for neural network predictor.

        Returns
        -------
        ParameterAnomalyDetector
            Restored, ready-to-serve detector instance.
        """
        checkpoint = torch.load(filepath, map_location=device or "cpu", weights_only=False)

        # Restore predictor
        p_info = checkpoint["predictor_state"]
        predictor = ParameterValuePredictor(
            input_dim=p_info["input_dim"],
            hidden_dim=p_info["hidden_dim"],
            num_layers=p_info["num_layers"],
            output_dim=p_info["output_dim"],
            dropout=p_info.get("dropout", 0.0),
            bidirectional=p_info.get("bidirectional", False),
        )
        predictor.load_state_dict(p_info["state_dict"])
        if device is not None:
            predictor = predictor.to(device)
        predictor.eval()

        # Restore estimator
        e_info = checkpoint["estimator_state"]
        estimator = MultivariateGaussianEstimator(
            regularization=e_info["regularization"],
            epsilon=e_info["epsilon"],
            shrinkage_factor=e_info["shrinkage_factor"],
            threshold_method=e_info["threshold_method"],
            alpha=e_info["alpha"],
            k_sigma=e_info["k_sigma"],
        )
        estimator.mean = e_info["mean"]
        estimator.covariance = e_info["covariance"]
        estimator.precision = e_info["precision"]
        estimator.log_det = e_info["log_det"]
        estimator.threshold = e_info["threshold"]
        estimator.fitted_dim = e_info["fitted_dim"]

        detector = cls(
            input_dim=checkpoint["input_dim"],
            hidden_dim=checkpoint["hidden_dim"],
            num_layers=checkpoint["num_layers"],
            dropout=checkpoint.get("dropout", 0.0),
            regularization=checkpoint["regularization"],
            epsilon=checkpoint["epsilon"],
            threshold_method=checkpoint["threshold_method"],
            alpha=checkpoint["alpha"],
            k_sigma=checkpoint["k_sigma"],
            normalize=checkpoint.get("normalize", True),
            feature_names=checkpoint.get("feature_names"),
            predictor=predictor,
            estimator=estimator,
        )
        detector.scaler_mean = checkpoint.get("scaler_mean")
        detector.scaler_std = checkpoint.get("scaler_std")

        return detector
