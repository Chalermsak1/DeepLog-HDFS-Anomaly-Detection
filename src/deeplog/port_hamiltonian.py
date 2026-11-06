"""Port-Hamiltonian Neural Network (PHNN) for Distributed Queues.

Grounds cloud observability directly in Industrial Physics and Non-linear Control Theory.
Models distributed microservice queues, thread pools, and network buffers as a
continuous Port-Hamiltonian System (PHS):

    dx/dt = (J(x) - R(x)) * grad_H(x) + G(x) * u
    y     = G(x)^T * grad_H(x)

where:
    J(x) = -J(x)^T  (Skew-symmetric interconnection matrix: lossless energy routing)
    R(x) >= 0        (Symmetric positive semi-definite: physical dissipation & friction)
    H(x) >= 0        (Total Hamiltonian storage energy: queue depth + processing inertia)
    u               (Port control/workload input)
    y               (Conjugate output throughput)

Guarantees strict energy passivity: dH/dt <= y^T u.
Spontaneous creation of requests or phantom state jumps is mathematically impossible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn


@dataclass(frozen=True)
class PortHamiltonianState:
    """State snapshot of a distributed queue dynamical system."""

    state_vector: np.ndarray
    hamiltonian_energy: float
    dissipation_rate: float
    external_work_rate: float
    energy_derivative: float
    is_conservative_violation: bool
    diagnostic: str


class PortHamiltonianNeuralQueue(nn.Module):
    """PyTorch module implementing an energy-conserving Port-Hamiltonian queue operator."""

    def __init__(
        self,
        state_dim: int = 4,
        input_dim: int = 2,
        hidden_dim: int = 32,
        min_dissipation: float = 1e-4,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.min_dissipation = min_dissipation

        # Skew-symmetric generator: J = W_j - W_j^T
        self.W_j = nn.Parameter(torch.randn(state_dim, state_dim) * 0.1)

        # Dissipation generator: R = L_r @ L_r^T + min_dissipation * I
        self.L_r = nn.Parameter(torch.eye(state_dim) * 0.1)

        # Hamiltonian quadratic metric matrix: M = C^T @ C + 1e-3 * I
        self.C_m = nn.Parameter(torch.eye(state_dim) * 0.5)

        # Input coupling matrix: G in R^{state_dim x input_dim}
        self.G = nn.Parameter(torch.randn(state_dim, input_dim) * 0.1)

    def get_skew_symmetric_J(self) -> torch.Tensor:
        """Compute exact antisymmetric matrix J = W - W^T."""
        return self.W_j - self.W_j.T

    def get_dissipation_R(self) -> torch.Tensor:
        """Compute exact positive semi-definite dissipation matrix R = L L^T + eps * I."""
        tril = torch.tril(self.L_r)
        R = tril @ tril.T + self.min_dissipation * torch.eye(
            self.state_dim, device=self.L_r.device
        )
        return R

    def get_metric_M(self) -> torch.Tensor:
        """Compute positive-definite Hamiltonian energy metric matrix M."""
        M = self.C_m.T @ self.C_m + 1e-3 * torch.eye(
            self.state_dim, device=self.C_m.device
        )
        return M

    def hamiltonian_energy(self, x: torch.Tensor) -> torch.Tensor:
        """Compute total stored Hamiltonian energy H(x) = 0.5 * x^T M x."""
        M = self.get_metric_M()
        if x.dim() == 1:
            return 0.5 * torch.dot(x, M @ x)
        # Batch mode: (B, state_dim)
        quad = torch.bmm(x.unsqueeze(1), (M @ x.T).T.unsqueeze(-1)).squeeze()
        return 0.5 * quad

    def hamiltonian_gradient(self, x: torch.Tensor) -> torch.Tensor:
        """Compute exact gradient of Hamiltonian grad_H(x) = M @ x."""
        M = self.get_metric_M()
        if x.dim() == 1:
            return M @ x
        return (M @ x.T).T

    def vector_field(self, x: torch.Tensor, u: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Evaluate continuous vector field dx/dt = (J - R) grad_H + G u."""
        J = self.get_skew_symmetric_J()
        R = self.get_dissipation_R()
        grad_H = self.hamiltonian_gradient(x)

        if x.dim() == 1:
            drift = (J - R) @ grad_H
            if u is not None:
                drift = drift + self.G @ u
            return drift
        else:
            # Batch mode
            drift = grad_H @ (J - R).T
            if u is not None:
                drift = drift + u @ self.G.T
            return drift

    def rk4_step(
        self, x: torch.Tensor, u: Optional[torch.Tensor] = None, dt: float = 0.05
    ) -> torch.Tensor:
        """Perform 4th-order Runge-Kutta numerical integration step."""
        k1 = self.vector_field(x, u)
        k2 = self.vector_field(x + 0.5 * dt * k1, u)
        k3 = self.vector_field(x + 0.5 * dt * k2, u)
        k4 = self.vector_field(x + dt * k3, u)
        return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def verify_energy_conservation(
        self, x: torch.Tensor, u: Optional[torch.Tensor] = None
    ) -> Tuple[float, float, float, bool]:
        """Verify the first law of thermodynamics and passivity condition dH/dt <= y^T u.

        Returns:
            dH_dt (float): Actual rate of energy change.
            dissipation_power (float): Energy lost to dissipation -grad_H^T R grad_H <= 0.
            external_power (float): Work done by external port y^T u.
            is_valid (bool): True if passivity holds within numerical tolerance.
        """
        with torch.no_grad():
            if x.dim() > 1:
                x = x[0]
            if u is not None and u.dim() > 1:
                u = u[0]

            R = self.get_dissipation_R()
            grad_H = self.hamiltonian_gradient(x)
            dx_dt = self.vector_field(x, u)

            # dH/dt = grad_H^T * dx/dt
            dH_dt = float(torch.dot(grad_H, dx_dt).item())

            # Dissipation power: -grad_H^T * R * grad_H (always <= 0)
            dissipation_power = float(-torch.dot(grad_H, R @ grad_H).item())

            external_power = 0.0
            if u is not None:
                # y = G^T * grad_H
                y = self.G.T @ grad_H
                external_power = float(torch.dot(y, u).item())

            # Numerical tolerance check: dH/dt == dissipation_power + external_power
            expected_dH_dt = dissipation_power + external_power
            residual = abs(dH_dt - expected_dH_dt)
            is_valid = residual < 1e-4 and (dH_dt <= external_power + 1e-5)

            return dH_dt, dissipation_power, external_power, is_valid


class DistributedPortHamiltonianObserver:
    """Production observer applying Port-Hamiltonian physics to node telemetry.

    Detects ghost requests, memory leaks, and Byzantine packet injections when observed
    cluster state transitions violate physical passivity bounds.
    """

    def __init__(
        self,
        num_queues: int = 4,
        energy_tolerance_sigma: float = 3.0,
    ) -> None:
        self.num_queues = num_queues
        self.state_dim = num_queues
        self.phnn = PortHamiltonianNeuralQueue(state_dim=self.state_dim, input_dim=1)
        self.energy_tolerance_sigma = energy_tolerance_sigma
        self._history: list[float] = []

    def observe_transition(
        self,
        current_state: Sequence[float],
        next_state: Sequence[float],
        control_input: float = 0.0,
        dt: float = 0.1,
    ) -> PortHamiltonianState:
        """Observe state transition [x_t -> x_{t+1}] and verify physical passivity."""
        x_curr = torch.tensor(current_state, dtype=torch.float32)
        x_next = torch.tensor(next_state, dtype=torch.float32)
        u_ten = torch.tensor([control_input], dtype=torch.float32)

        H_curr = float(self.phnn.hamiltonian_energy(x_curr).item())
        H_next = float(self.phnn.hamiltonian_energy(x_next).item())

        dH_dt, diss_pow, ext_pow, passivity_ok = self.phnn.verify_energy_conservation(
            x_curr, u_ten
        )

        # Empirical finite difference vs theoretical bound
        empirical_dH = (H_next - H_curr) / max(dt, 1e-5)
        unaccounted_energy = empirical_dH - ext_pow

        self._history.append(unaccounted_energy)
        if len(self._history) > 200:
            self._history.pop(0)

        # Detect violation when unaccounted energy exceeds physical bounds
        mean_unacc = float(np.mean(self._history))
        std_unacc = float(np.std(self._history)) if len(self._history) > 5 else 1.0

        z_score = (unaccounted_energy - mean_unacc) / max(std_unacc, 1e-4)
        is_violation = (z_score > self.energy_tolerance_sigma) and (unaccounted_energy > 0.05)

        diagnostic = "CONSERVATIVE_NORMAL"
        if is_violation:
            diagnostic = (
                f"PASSIVITY_VIOLATION_GHOST_REQUESTS: unaccounted_energy={unaccounted_energy:.4f}, "
                f"z_score={z_score:.2f}sigma"
            )

        return PortHamiltonianState(
            state_vector=np.array(next_state, dtype=np.float64),
            hamiltonian_energy=H_next,
            dissipation_rate=abs(diss_pow),
            external_work_rate=ext_pow,
            energy_derivative=empirical_dH,
            is_conservative_violation=is_violation,
            diagnostic=diagnostic,
        )
