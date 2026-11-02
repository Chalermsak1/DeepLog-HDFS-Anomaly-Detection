"""
Comprehensive Unit Tests for Pillar II: Physics-Informed Neural Queuing Operators (PISNO)
========================================================================================
Validates:
1. JacksonQueueNetwork:
   - Traffic equation solving: lambda = gamma * (I - P)^{-1}
   - Flow conservation: sum gamma = sum lambda * (1 - sum P)
   - Single-server (M/M/1) queue lengths & waiting times
   - Multi-server (M/M/c) queue lengths, waiting times, and differentiable Erlang C
   - Little's Law adherence: L = lambda * W
   - Network Little's Law: L_net = Lambda_total * W_net
   - Stability condition: rho < 1.0
   - Autograd differentiability across all equations
   - Batched inputs (B, M)
2. PhysicsInformedLoss:
   - Little's Law gradient penalty (|L - lambda * W| <= epsilon)
   - Node and global flow conservation gradient penalty
   - Traffic intensity stability penalty (rho < rho_max)
   - Capacity overflow penalty (L <= C)
   - Non-negativity lower bound penalties
   - Zero penalty on physically exact states
   - Differentiable loss minimization driving unconstrained tensors to physical manifold
3. PhysicalProjectionLayer & PhysicsConstrainedPredictor:
   - Guaranteed non-negative service times: S > 0, mu >= mu_min > 0
   - Guaranteed traffic stability: rho <= rho_max < 1.0 unconditionally
   - Guaranteed exact Little's Law compliance: |L - lambda * W| == 0
   - Guaranteed buffer capacity compliance: L <= C
   - Arbitrary tensor projection method handling corrupted inputs
   - Neural network training, backpropagation, and state dict serialization
4. PhysicalAnomalyDetector:
   - Zero false positives on physically valid queue states
   - Detection of Little's Law violations
   - Detection of Jackson flow conservation divergence
   - Detection of negative service times / impossible latency
   - Detection of traffic saturation (rho >= 1.0)
   - Detection of buffer capacity overflows
   - Detection of response time underflow (W < S = 1/mu)
   - Synthetic anomaly injection helpers
   - Batch evaluation and bottleneck identification
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

from deeplog.physics import (
    JacksonQueueNetwork,
    PhysicsInformedLoss,
    PhysicalProjectionLayer,
    PhysicsConstrainedPredictor,
    PhysicalAnomalyDetector,
    PhysicalAnomalyResult,
    PhysicalQueueState,
    compute_erlang_c_torch,
)


# ==============================================================================
# Track 1: JacksonQueueNetwork Tests
# ==============================================================================


class TestJacksonQueueNetwork:
    """Tests for exact Jackson Queueing Network analytical modeling."""

    def test_init_and_parameter_validation(self):
        """Verify initialization and validation of network dimensions."""
        net = JacksonQueueNetwork(num_nodes=3)
        assert net.num_nodes == 3
        assert net.gamma.shape == (3,)
        assert net.routing_matrix.shape == (3, 3)
        assert net.service_rates.shape == (3,)
        assert net.servers.shape == (3,)

        with pytest.raises(ValueError, match="num_nodes must be positive"):
            JacksonQueueNetwork(num_nodes=0)

        with pytest.raises(ValueError, match="Length of node_names"):
            JacksonQueueNetwork(num_nodes=2, node_names=["a", "b", "c"])

    def test_traffic_equations_single_node(self):
        """Verify traffic equation for a single node with no feedback."""
        net = JacksonQueueNetwork(num_nodes=1, gamma=[10.0], routing_matrix=[[0.0]], service_rates=[20.0])
        lam = net.solve_traffic_equations()
        assert math.isclose(lam.item(), 10.0, rel_tol=1e-5)

    def test_traffic_equations_multi_node_cascade(self):
        """
        Verify traffic equations for a 3-node linear cascade:
        External: gamma = [10.0, 0.0, 0.0]
        Routing: 0 -> 1 (p=0.8), 1 -> 2 (p=0.5)
        Expected arrivals:
        lambda_0 = 10.0
        lambda_1 = 10.0 * 0.8 = 8.0
        lambda_2 = 8.0 * 0.5 = 4.0
        """
        gamma = [10.0, 0.0, 0.0]
        P = [
            [0.0, 0.8, 0.0],
            [0.0, 0.0, 0.5],
            [0.0, 0.0, 0.0],
        ]
        net = JacksonQueueNetwork(num_nodes=3, gamma=gamma, routing_matrix=P)
        lam = net.solve_traffic_equations()

        assert math.isclose(lam[0].item(), 10.0, rel_tol=1e-5)
        assert math.isclose(lam[1].item(), 8.0, rel_tol=1e-5)
        assert math.isclose(lam[2].item(), 4.0, rel_tol=1e-5)

    def test_traffic_equations_with_feedback_loop(self):
        """
        Verify traffic equations with cyclic feedback:
        External: gamma = [10.0, 5.0]
        P = [[0.0, 0.5],
             [0.2, 0.0]]
        Equations:
        lambda_0 = 10 + 0.2 * lambda_1
        lambda_1 = 5 + 0.5 * lambda_0
        lambda_0 = 10 + 0.2 * (5 + 0.5 * lambda_0) = 11 + 0.1 * lambda_0
        0.9 * lambda_0 = 11 => lambda_0 = 11 / 0.9 = 12.2222...
        lambda_1 = 5 + 0.5 * (12.2222...) = 11.1111...
        """
        gamma = [10.0, 5.0]
        P = [[0.0, 0.5], [0.2, 0.0]]
        net = JacksonQueueNetwork(num_nodes=2, gamma=gamma, routing_matrix=P)
        lam = net.solve_traffic_equations()

        expected_lam0 = 11.0 / 0.9
        expected_lam1 = 5.0 + 0.5 * expected_lam0
        assert math.isclose(lam[0].item(), expected_lam0, rel_tol=1e-4)
        assert math.isclose(lam[1].item(), expected_lam1, rel_tol=1e-4)

    def test_conservation_of_flow_verification(self):
        """Verify conservation of flow: Total external inflows == Total departures."""
        gamma = [10.0, 5.0]
        P = [[0.0, 0.5], [0.2, 0.0]]
        net = JacksonQueueNetwork(num_nodes=2, gamma=gamma, routing_matrix=P)
        node_res, global_res, is_conserved = net.verify_flow_conservation()

        assert is_conserved
        assert global_res.item() < 1e-4
        assert node_res.max().item() < 1e-4

    def test_littles_law_single_server_mm1(self):
        """
        Verify Little's Law for M/M/1 queue:
        L = lambda / (mu - lambda)
        W = 1 / (mu - lambda)
        Verify: L == lambda * W identically.
        """
        net = JacksonQueueNetwork(
            num_nodes=1,
            gamma=[15.0],
            routing_matrix=[[0.0]],
            service_rates=[20.0],
            servers=1,
        )
        lam = net.solve_traffic_equations()
        L = net.compute_queue_lengths(lam)
        W = net.compute_waiting_times(lam)

        expected_L = 15.0 / (20.0 - 15.0)  # 3.0
        expected_W = 1.0 / (20.0 - 15.0)   # 0.2
        assert math.isclose(L.item(), expected_L, rel_tol=1e-4)
        assert math.isclose(W.item(), expected_W, rel_tol=1e-4)

        # Little's law verification
        res, compliant = net.verify_littles_law(L, lam, W)
        assert compliant.all()
        assert res.max().item() < 1e-5

    def test_littles_law_multi_server_mmc(self):
        """
        Verify Little's Law for multi-server M/M/c queue (c = 3):
        Check that L = lambda * W and L_q = lambda * W_q hold exactly.
        """
        net = JacksonQueueNetwork(
            num_nodes=1,
            gamma=[20.0],
            routing_matrix=[[0.0]],
            service_rates=[10.0],
            servers=3,  # c * mu = 30 > 20 => rho = 2/3 < 1
        )
        lam = net.solve_traffic_equations()
        L = net.compute_queue_lengths(lam, in_system=True)
        W = net.compute_waiting_times(lam, in_system=True)
        L_q = net.compute_queue_lengths(lam, in_system=False)
        W_q = net.compute_waiting_times(lam, in_system=False)

        # Little's Law in system: L = lambda * W
        res_sys, comp_sys = net.verify_littles_law(L, lam, W)
        assert comp_sys.all()
        assert res_sys.max().item() < 1e-4

        # Little's Law in queue: L_q = lambda * W_q
        res_q, comp_q = net.verify_littles_law(L_q, lam, W_q)
        assert comp_q.all()
        assert res_q.max().item() < 1e-4

        # W = W_q + 1 / mu
        assert math.isclose(W.item(), (W_q + 1.0 / 10.0).item(), rel_tol=1e-4)
        # L = L_q + lambda / mu
        assert math.isclose(L.item(), (L_q + 20.0 / 10.0).item(), rel_tol=1e-4)

    def test_network_level_littles_law(self):
        """
        Verify Network-level Little's Law across an arbitrary 4-node network:
        L_net = Lambda_total * W_net
        """
        gamma = [12.0, 8.0, 0.0, 0.0]
        P = [
            [0.0, 0.4, 0.3, 0.0],
            [0.0, 0.0, 0.5, 0.2],
            [0.1, 0.0, 0.0, 0.4],
            [0.0, 0.0, 0.0, 0.0],
        ]
        mu = [25.0, 20.0, 18.0, 15.0]
        servers = [2, 1, 2, 1]

        net = JacksonQueueNetwork(
            num_nodes=4,
            gamma=gamma,
            routing_matrix=P,
            service_rates=mu,
            servers=servers,
        )
        metrics = net.network_metrics()

        assert metrics["is_stable"]
        total_arrival = metrics["total_external_arrival"]  # 12 + 8 = 20.0
        total_customers = metrics["total_network_customers"]
        mean_resp_time = metrics["mean_network_response_time"]

        # L_net = Lambda_total * W_net
        assert math.isclose(total_customers, total_arrival * mean_resp_time, rel_tol=1e-5)

    def test_stability_checking(self):
        """Verify stability detection for stable and saturated queue networks."""
        # Stable
        net_stable = JacksonQueueNetwork(
            num_nodes=2,
            gamma=[5.0, 5.0],
            routing_matrix=[[0.0, 0.0], [0.0, 0.0]],
            service_rates=[10.0, 10.0],
            servers=1,
        )
        assert net_stable.is_stable().item()

        # Unstable: arrival rate >= service rate
        net_unstable = JacksonQueueNetwork(
            num_nodes=2,
            gamma=[15.0, 5.0],
            routing_matrix=[[0.0, 0.0], [0.0, 0.0]],
            service_rates=[10.0, 10.0],
            servers=1,
        )
        assert not net_unstable.is_stable().item()

    def test_autograd_differentiability(self):
        """Verify backward gradients flow cleanly through the Jackson Queue equations."""
        gamma = torch.tensor([8.0, 4.0], requires_grad=True)
        P = torch.tensor([[0.0, 0.5], [0.1, 0.0]], requires_grad=True)
        mu = torch.tensor([15.0, 12.0], requires_grad=True)

        net = JacksonQueueNetwork(num_nodes=2)
        lam = net.solve_traffic_equations(gamma, P)
        L = net.compute_queue_lengths(lam, mu, servers=torch.tensor([1.0, 1.0]))

        loss = torch.sum(L)
        loss.backward()

        assert gamma.grad is not None and not torch.isnan(gamma.grad).any()
        assert P.grad is not None and not torch.isnan(P.grad).any()
        assert mu.grad is not None and not torch.isnan(mu.grad).any()
        # Increasing service rate mu must decrease queue length L (negative gradient)
        assert (mu.grad < 0.0).all()
        # Increasing arrival rate gamma must increase queue length L (positive gradient)
        assert (gamma.grad > 0.0).all()

    def test_batched_traffic_solving(self):
        """Verify batched solving of traffic equations (B, M)."""
        B, M = 8, 3
        gamma = torch.rand(B, M) * 5.0 + 1.0
        P = torch.zeros(M, M)
        P[0, 1] = 0.5
        P[1, 2] = 0.5

        net = JacksonQueueNetwork(num_nodes=M, routing_matrix=P)
        lam_batch = net.solve_traffic_equations(gamma)
        assert lam_batch.shape == (B, M)

        # Check each batch row matches individual solve
        for b in range(B):
            lam_single = net.solve_traffic_equations(gamma[b])
            assert torch.allclose(lam_batch[b], lam_single, atol=1e-5)

    def test_forward_pass_state(self):
        """Verify JacksonQueueNetwork forward() returns a valid PhysicalQueueState."""
        net = JacksonQueueNetwork(
            num_nodes=2,
            gamma=[6.0, 4.0],
            routing_matrix=[[0.0, 0.3], [0.2, 0.0]],
            service_rates=[12.0, 10.0],
        )
        state = net()
        assert isinstance(state, PhysicalQueueState)
        assert state.arrival_rates.shape == (2,)
        assert state.service_rates.shape == (2,)
        assert state.service_times.shape == (2,)
        assert state.waiting_times.shape == (2,)
        assert state.queue_lengths.shape == (2,)
        assert state.traffic_intensities.shape == (2,)


# ==============================================================================
# Track 2: PhysicsInformedLoss Tests
# ==============================================================================


class TestPhysicsInformedLoss:
    """Tests for PhysicsInformedLoss multi-objective queuing loss."""

    def test_zero_loss_on_perfect_physical_state(self):
        """Verify that a state perfectly satisfying all invariants yields near-zero physics loss."""
        # Perfect M/M/1 node: lambda = 10, mu = 20 => W = 0.1, L = 1.0
        lam = torch.tensor([10.0])
        mu = torch.tensor([20.0])
        W = torch.tensor([0.1])
        L = torch.tensor([1.0])  # L == lam * W exactly
        gamma = torch.tensor([10.0])
        P = torch.tensor([[0.0]])

        loss_fn = PhysicsInformedLoss(
            weight_data=0.0,
            weight_little=1.0,
            weight_flow=1.0,
            weight_stability=1.0,
            weight_capacity=1.0,
            weight_nonneg=1.0,
            rho_max=0.99,
        )

        total_loss, breakdown = loss_fn(
            predicted_L=L,
            predicted_lambda=lam,
            predicted_W=W,
            predicted_mu=mu,
            gamma=gamma,
            routing_matrix=P,
            capacities=torch.tensor([10.0]),
        )

        assert total_loss.item() < 1e-6
        assert breakdown["little_loss"].item() < 1e-6
        assert breakdown["flow_loss"].item() < 1e-6
        assert breakdown["stability_loss"].item() < 1e-6
        assert breakdown["capacity_loss"].item() < 1e-6

    def test_littles_law_penalty_and_gradient(self):
        """Verify penalty when Little's Law is violated and gradient drives recovery."""
        # Predicted: lambda = 10, W = 0.5 => True Little's L should be 5.0
        # Corrupt L = 25.0
        lam = torch.tensor([10.0])
        W = torch.tensor([0.5])
        corrupt_L = torch.tensor([25.0], requires_grad=True)

        loss_fn = PhysicsInformedLoss(weight_little=1.0, weight_data=0.0, weight_flow=0.0, weight_stability=0.0)
        total_loss, breakdown = loss_fn(
            predicted_L=corrupt_L,
            predicted_lambda=lam,
            predicted_W=W,
        )

        # Expected error: |25.0 - 5.0| = 20.0 => MSE = 400.0
        assert math.isclose(breakdown["little_loss"].item(), 400.0, rel_tol=1e-4)

        # Gradient should point toward reducing L
        total_loss.backward()
        assert corrupt_L.grad is not None
        assert corrupt_L.grad.item() > 0.0  # Reducing L will decrease loss

    def test_flow_conservation_penalty(self):
        """Verify penalty when conservation of flow is violated."""
        # gamma = [10.0], P = [[0.0]], but lambda is predicted as 20.0 (flow created from nowhere)
        lam = torch.tensor([20.0], requires_grad=True)
        gamma = torch.tensor([10.0])
        P = torch.tensor([[0.0]])

        loss_fn = PhysicsInformedLoss(weight_flow=1.0, weight_data=0.0, weight_little=0.0, weight_stability=0.0)
        total_loss, breakdown = loss_fn(
            predicted_lambda=lam,
            gamma=gamma,
            routing_matrix=P,
        )

        assert breakdown["flow_loss"].item() > 0.0
        total_loss.backward()
        assert lam.grad is not None
        assert lam.grad.item() > 0.0  # Decreasing lambda brings it toward gamma = 10.0

    def test_stability_penalty_for_saturation(self):
        """Verify penalty when traffic intensity rho >= rho_max."""
        # mu = 10.0, lambda = 15.0 => rho = 1.5 > rho_max = 0.99
        lam = torch.tensor([15.0], requires_grad=True)
        mu = torch.tensor([10.0])

        loss_fn = PhysicsInformedLoss(
            weight_stability=1.0,
            weight_data=0.0,
            weight_little=0.0,
            weight_flow=0.0,
            rho_max=0.95,
        )
        total_loss, breakdown = loss_fn(
            predicted_lambda=lam,
            predicted_mu=mu,
            servers=1,
        )

        assert breakdown["stability_loss"].item() > 0.0
        # rho violation: (1.5 - 0.95)^2 = 0.55^2 = 0.3025
        assert math.isclose(breakdown["stability_loss"].item(), 0.55**2, rel_tol=1e-4)

    def test_capacity_penalty(self):
        """Verify penalty when predicted queue length exceeds buffer capacity."""
        L = torch.tensor([150.0], requires_grad=True)
        cap = torch.tensor([100.0])

        loss_fn = PhysicsInformedLoss(weight_capacity=1.0, weight_data=0.0, weight_little=0.0)
        total_loss, breakdown = loss_fn(
            predicted_L=L,
            capacities=cap,
        )

        # Excess: 150 - 100 = 50 => MSE = 2500.0
        assert math.isclose(breakdown["capacity_loss"].item(), 2500.0, rel_tol=1e-4)

    def test_non_negativity_penalty(self):
        """Verify penalty when physical quantities are negative."""
        lam = torch.tensor([-5.0], requires_grad=True)
        mu = torch.tensor([-1.0], requires_grad=True)
        L = torch.tensor([-10.0], requires_grad=True)
        W = torch.tensor([-2.0], requires_grad=True)

        loss_fn = PhysicsInformedLoss(weight_nonneg=1.0, weight_data=0.0, weight_little=0.0)
        total_loss, breakdown = loss_fn(
            predicted_lambda=lam,
            predicted_mu=mu,
            predicted_L=L,
            predicted_W=W,
        )

        assert breakdown["nonneg_loss"].item() > 0.0


# ==============================================================================
# Track 3: PhysicalProjectionLayer & PhysicsConstrainedPredictor Tests
# ==============================================================================


class TestPhysicalProjectionLayerAndPredictor:
    """Tests for physical projection layers and invariant-constrained predictors."""

    def test_physical_projection_layer_invariants(self):
        """Verify projection layer guarantees all distributed systems physical invariants."""
        proj_layer = PhysicalProjectionLayer(num_nodes=3, rho_max=0.95, mu_min=1e-3)

        # Feed completely arbitrary unconstrained logits (including negative numbers)
        raw_mu = torch.tensor([-5.0, 0.0, 10.0])
        raw_rho = torch.tensor([-10.0, 0.0, 20.0])
        raw_W = torch.tensor([-2.0, 5.0, -10.0])

        state = proj_layer(raw_mu=raw_mu, raw_rho=raw_rho, raw_W=raw_W)

        # Invariant 1: Strictly positive service rates and service times
        assert (state.service_rates >= 1e-3).all()
        assert (state.service_times > 0.0).all()
        assert torch.allclose(state.service_times, 1.0 / state.service_rates)

        # Invariant 2: Traffic intensity strictly bounded by rho_max < 1.0
        assert (state.traffic_intensities >= 0.0).all()
        assert (state.traffic_intensities <= 0.95).all()

        # Invariant 3: Waiting time is strictly non-negative and >= bare service time
        assert (state.waiting_times >= state.service_times).all()

        # Invariant 4: Little's Law is satisfied exactly: |L - lambda * W| == 0
        diff = torch.abs(state.queue_lengths - state.arrival_rates * state.waiting_times)
        assert (diff < 1e-6).all()

    def test_physical_projection_layer_capacity_bounds(self):
        """Verify projection layer caps queue length at configured capacity while preserving Little's Law."""
        cap = torch.tensor([5.0, 10.0])
        proj_layer = PhysicalProjectionLayer(num_nodes=2, capacities=cap)

        raw_mu = torch.tensor([1.0, 1.0])
        raw_rho = torch.tensor([5.0, 5.0])
        raw_W = torch.tensor([20.0, 20.0])

        state = proj_layer(raw_mu=raw_mu, raw_rho=raw_rho, raw_W=raw_W)
        assert (state.queue_lengths <= cap + 1e-5).all()
        # Little's Law remains aligned
        diff = torch.abs(state.queue_lengths - state.arrival_rates * state.waiting_times)
        assert (diff < 1e-5).all()

    def test_direct_projection_method_arbitrary_inputs(self):
        """Verify .project() method repairs corrupted / negative / unstable inputs."""
        proj_layer = PhysicalProjectionLayer(num_nodes=2, rho_max=0.90)

        # Corrupted inputs with negative rates, saturation, negative waiting time
        corrupted_lam = np.array([-10.0, 100.0])
        corrupted_mu = np.array([-5.0, 2.0])  # Node 1 would have rho = 100 / 2 = 50.0!
        corrupted_W = np.array([-1.0, 0.01])  # Faster than 1/mu = 0.5

        repaired = proj_layer.project(
            arrival_rates=corrupted_lam,
            service_rates=corrupted_mu,
            waiting_times=corrupted_W,
        )

        assert (repaired.arrival_rates >= 0.0).all()
        assert (repaired.service_rates >= 1e-4).all()
        assert (repaired.traffic_intensities <= 0.90).all()
        assert (repaired.waiting_times >= repaired.service_times).all()
        # Little's Law verified
        littles_diff = np.abs(repaired.queue_lengths - repaired.arrival_rates * repaired.waiting_times)
        assert (littles_diff < 1e-5).all()

    def test_physics_constrained_predictor_forward_and_shapes(self):
        """Verify forward pass of PhysicsConstrainedPredictor with flat and sequential inputs."""
        B, D, M = 4, 16, 3
        model = PhysicsConstrainedPredictor(input_dim=D, num_nodes=M, model_type="mlp")

        x_flat = torch.randn(B, D)
        state_flat = model(x_flat)
        assert state_flat.queue_lengths.shape == (B, M)
        assert state_flat.arrival_rates.shape == (B, M)
        assert state_flat.service_rates.shape == (B, M)
        assert (state_flat.traffic_intensities <= 0.99).all()

        # Sequence input (B, T, D)
        T = 10
        x_seq = torch.randn(B, T, D)
        state_seq = model(x_seq)
        assert state_seq.queue_lengths.shape == (B, M)

    def test_physics_constrained_predictor_recurrent_backbone(self):
        """Verify predictor works with GRU and LSTM backbones."""
        B, T, D, M = 2, 5, 8, 2
        gru_model = PhysicsConstrainedPredictor(input_dim=D, num_nodes=M, model_type="gru")
        state_gru = gru_model(torch.randn(B, T, D))
        assert state_gru.queue_lengths.shape == (B, M)

        lstm_model = PhysicsConstrainedPredictor(input_dim=D, num_nodes=M, model_type="lstm")
        state_lstm = lstm_model(torch.randn(B, T, D))
        assert state_lstm.queue_lengths.shape == (B, M)

    def test_end_to_end_training_step(self):
        """Verify PhysicsConstrainedPredictor trains cleanly with PhysicsInformedLoss."""
        model = PhysicsConstrainedPredictor(input_dim=8, num_nodes=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        loss_fn = PhysicsInformedLoss(weight_data=1.0, weight_little=1.0, weight_flow=0.0)

        x = torch.randn(16, 8)
        target_L = torch.ones(16, 2) * 2.0

        for _ in range(5):
            optimizer.zero_grad()
            state = model(x)
            loss, _ = loss_fn(
                predicted_L=state.queue_lengths,
                predicted_lambda=state.arrival_rates,
                predicted_W=state.waiting_times,
                target_y=target_L,
                predicted_y=state.queue_lengths,
            )
            loss.backward()
            optimizer.step()

        # Ensure Little's Law holds even after parameter updates
        with torch.no_grad():
            eval_state = model(x)
            diff = torch.abs(eval_state.queue_lengths - eval_state.arrival_rates * eval_state.waiting_times)
            assert (diff < 1e-5).all()

    def test_model_serialization_and_state_parity(self):
        """Verify model checkpoint save, load, and numerical output parity."""
        model = PhysicsConstrainedPredictor(input_dim=8, num_nodes=2)
        model.eval()
        x = torch.randn(4, 8)
        with torch.no_grad():
            state_before = model(x)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            torch.save(model.state_dict(), tmp_path)
            loaded_model = PhysicsConstrainedPredictor(input_dim=8, num_nodes=2)
            loaded_model.load_state_dict(torch.load(tmp_path, weights_only=True))
            loaded_model.eval()
            with torch.no_grad():
                state_after = loaded_model(x)

            assert torch.allclose(state_before.queue_lengths, state_after.queue_lengths)
            assert torch.allclose(state_before.arrival_rates, state_after.arrival_rates)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


# ==============================================================================
# Track 4: PhysicalAnomalyDetector Tests
# ==============================================================================


class TestPhysicalAnomalyDetector:
    """Tests for PhysicalAnomalyDetector detecting impossible states and violations."""

    def test_clean_baseline_no_anomalies(self):
        """Verify normal physically valid queuing metrics raise no false alarms."""
        detector = PhysicalAnomalyDetector(tau_little=0.1, tau_flow=0.1, rho_threshold=1.0)

        # Stable M/M/1 node: lambda = 10, mu = 20 => W = 0.1, L = 1.0
        res = detector.detect(
            queue_lengths=[1.0, 2.0],
            arrival_rates=[10.0, 10.0],
            waiting_times=[0.1, 0.2],
            service_rates=[20.0, 15.0],
            external_arrivals=[10.0, 10.0],
            routing_matrix=[[0.0, 0.0], [0.0, 0.0]],
        )

        assert not res.is_anomaly
        assert res.anomaly_score == 0.0
        assert len(res.violations) == 0

    def test_detect_littles_law_violation(self):
        """Verify detection of severe Little's Law discrepancies."""
        detector = PhysicalAnomalyDetector(tau_little=0.1)

        # lambda = 10, W = 0.1 => expected L = 1.0, but observed L = 10.0
        res = detector.detect(
            queue_lengths=[10.0],
            arrival_rates=[10.0],
            waiting_times=[0.1],
            service_rates=[20.0],
        )

        assert res.is_anomaly
        assert "LITTLES_LAW_VIOLATION" in res.violations
        assert res.anomaly_score > 0.0

    def test_detect_negative_service_time(self):
        """Verify detection of impossible negative service rate or negative waiting time."""
        detector = PhysicalAnomalyDetector()

        res = detector.detect(
            queue_lengths=[1.0],
            arrival_rates=[10.0],
            waiting_times=[-0.5],
            service_rates=[-20.0],
        )

        assert res.is_anomaly
        assert "NEGATIVE_PHYSICAL_QUANTITY" in res.violations

    def test_detect_response_time_underflow(self):
        """Verify detection of response time W faster than bare service time S = 1/mu."""
        detector = PhysicalAnomalyDetector()

        # mu = 10 => bare service time S = 0.1s. Observed W = 0.01s (physically impossible)
        res = detector.detect(
            queue_lengths=[0.1],
            arrival_rates=[10.0],
            waiting_times=[0.01],
            service_rates=[10.0],
        )

        assert res.is_anomaly
        assert "RESPONSE_TIME_UNDERFLOW" in res.violations

    def test_detect_flow_conservation_divergence(self):
        """Verify detection when traffic equations or network flow balance diverges."""
        detector = PhysicalAnomalyDetector(tau_flow=0.05)

        # External arrival = 10.0, no internal routing, but arrival_rate is observed as 30.0
        res = detector.detect(
            queue_lengths=[3.0],
            arrival_rates=[30.0],
            waiting_times=[0.1],
            service_rates=[50.0],
            external_arrivals=[10.0],
            routing_matrix=[[0.0]],
        )

        assert res.is_anomaly
        assert "FLOW_CONSERVATION_DIVERGENCE" in res.violations

    def test_detect_traffic_saturation(self):
        """Verify detection when traffic intensity rho >= 1.0."""
        detector = PhysicalAnomalyDetector(rho_threshold=1.0)

        # lambda = 25.0, mu = 20.0 => rho = 1.25 >= 1.0
        res = detector.detect(
            queue_lengths=[100.0],
            arrival_rates=[25.0],
            waiting_times=[4.0],
            service_rates=[20.0],
            servers=1,
        )

        assert res.is_anomaly
        assert "TRAFFIC_SATURATION_UNSTABLE" in res.violations
        assert res.bottleneck_node == 0

    def test_detect_buffer_capacity_overflow(self):
        """Verify detection when queue length exceeds physical buffer limit."""
        detector = PhysicalAnomalyDetector(capacities=[50.0])

        res = detector.detect(
            queue_lengths=[75.0],  # 75 > 50
            arrival_rates=[10.0],
            waiting_times=[7.5],
            service_rates=[20.0],
        )

        assert res.is_anomaly
        assert "BUFFER_CAPACITY_OVERFLOW" in res.violations

    def test_synthetic_anomaly_injection(self):
        """Verify all synthetic anomaly injection types can be generated and detected."""
        detector = PhysicalAnomalyDetector(capacities=[50.0])

        base_state = {
            "arrival_rates": np.array([10.0, 10.0]),
            "service_rates": np.array([20.0, 20.0]),
            "service_times": np.array([0.05, 0.05]),
            "waiting_times": np.array([0.1, 0.1]),
            "queue_lengths": np.array([1.0, 1.0]),
            "traffic_intensities": np.array([0.5, 0.5]),
        }

        # 1. Little's law injection
        corrupt_ll = detector.inject_synthetic_anomaly(base_state, "littles_law", severity=3.0)
        res_ll = detector.detect(
            queue_lengths=corrupt_ll["queue_lengths"],
            arrival_rates=corrupt_ll["arrival_rates"],
            waiting_times=corrupt_ll["waiting_times"],
        )
        assert "LITTLES_LAW_VIOLATION" in res_ll.violations

        # 2. Negative service time injection
        corrupt_neg = detector.inject_synthetic_anomaly(base_state, "negative_service_time", severity=2.0)
        res_neg = detector.detect(
            queue_lengths=corrupt_neg["queue_lengths"],
            arrival_rates=corrupt_neg["arrival_rates"],
            waiting_times=corrupt_neg["waiting_times"],
            service_rates=corrupt_neg["service_rates"],
        )
        assert "NEGATIVE_PHYSICAL_QUANTITY" in res_neg.violations

        # 3. Traffic saturation injection
        corrupt_sat = detector.inject_synthetic_anomaly(base_state, "traffic_saturation", severity=0.5)
        res_sat = detector.detect(
            queue_lengths=corrupt_sat["queue_lengths"],
            arrival_rates=corrupt_sat["arrival_rates"],
            waiting_times=corrupt_sat["waiting_times"],
            service_rates=corrupt_sat["service_rates"],
        )
        assert "TRAFFIC_SATURATION_UNSTABLE" in res_sat.violations

        # 4. Underflow response time injection
        corrupt_und = detector.inject_synthetic_anomaly(base_state, "underflow_response_time")
        res_und = detector.detect(
            queue_lengths=corrupt_und["queue_lengths"],
            arrival_rates=corrupt_und["arrival_rates"],
            waiting_times=corrupt_und["waiting_times"],
            service_rates=corrupt_und["service_rates"],
        )
        assert "RESPONSE_TIME_UNDERFLOW" in res_und.violations

        # 5. Capacity overflow injection
        corrupt_cap = detector.inject_synthetic_anomaly(base_state, "capacity_overflow")
        res_cap = detector.detect(
            queue_lengths=corrupt_cap["queue_lengths"],
            arrival_rates=corrupt_cap["arrival_rates"],
            waiting_times=corrupt_cap["waiting_times"],
            service_rates=corrupt_cap["service_rates"],
        )
        assert "BUFFER_CAPACITY_OVERFLOW" in res_cap.violations

    def test_batch_detection(self):
        """Verify batch detection runs across multidimensional time series traces."""
        detector = PhysicalAnomalyDetector()

        # Batch of 4 traces: trace 0 normal, trace 1 Little's violation, trace 2 saturation, trace 3 normal
        L_b = torch.tensor([
            [1.0, 1.0],
            [50.0, 1.0],  # Little's violation
            [10.0, 10.0], # Saturation
            [2.0, 2.0],
        ])
        lam_b = torch.tensor([
            [10.0, 10.0],
            [10.0, 10.0],
            [25.0, 25.0], # lambda > mu
            [10.0, 10.0],
        ])
        W_b = torch.tensor([
            [0.1, 0.1],
            [0.1, 0.1],
            [0.4, 0.4],
            [0.2, 0.2],
        ])
        mu_b = torch.tensor([
            [20.0, 20.0],
            [20.0, 20.0],
            [20.0, 20.0],
            [20.0, 20.0],
        ])

        results = detector.detect_batch(
            queue_lengths=L_b,
            arrival_rates=lam_b,
            waiting_times=W_b,
            service_rates=mu_b,
        )

        assert len(results) == 4
        assert not results[0].is_anomaly
        assert results[1].is_anomaly
        assert "LITTLES_LAW_VIOLATION" in results[1].violations
        assert results[2].is_anomaly
        assert "TRAFFIC_SATURATION_UNSTABLE" in results[2].violations
        assert not results[3].is_anomaly
