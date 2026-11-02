"""
Unit tests for Pillar IV: Latent World Model & Pre-Emptive Self-Healing Control (Causal Dreamer).
Tests RSSM, CounterfactualDreamer, and AutonomousMitigator.
"""

import json
import math
import pytest
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from deeplog.world_model import (
    RecurrentStateSpaceModel,
    RSSMState,
    CounterfactualDreamer,
    CounterfactualTrajectory,
    AutonomousMitigator,
    MitigationAction,
    MitigationDecision,
    MitigationUrgency,
    RemediationEventStream,
)


# ==============================================================================
# 1. RSSM Core Unit Tests
# ==============================================================================


class TestRecurrentStateSpaceModel:
    """Tests for the Recurrent State Space Model architecture and mechanics."""

    @pytest.fixture
    def rssm(self):
        torch.manual_seed(42)
        return RecurrentStateSpaceModel(
            obs_dim=8,
            action_dim=4,
            action_embed_dim=8,
            hidden_dim=32,
            latent_dim=16,
            encoder_hidden_dim=32,
            decoder_hidden_dim=32,
            kl_balance=0.8,
            kl_free_nats=0.05,
        )

    def test_rssm_initialization(self, rssm):
        """Verify layer dimensions and initialization state."""
        state = rssm.init_state(batch_size=2)
        assert isinstance(state, RSSMState)
        assert state.h.shape == (2, 32)
        assert state.z.shape == (2, 16)
        assert torch.all(state.h == 0.0)
        assert torch.all(state.z == 0.0)

    def test_rssm_state_detach_clone_to(self, rssm):
        """Verify RSSMState utility methods."""
        h = torch.randn(2, 32, requires_grad=True)
        z = torch.randn(2, 16, requires_grad=True)
        state = RSSMState(h=h, z=z)

        detached = state.detach()
        assert not detached.h.requires_grad
        assert not detached.z.requires_grad

        cloned = state.clone()
        assert torch.allclose(cloned.h, state.h)
        assert cloned.h.data_ptr() != state.h.data_ptr()

    def test_deterministic_recurrent_update(self, rssm):
        """Verify that deterministic state h_t is strictly reproducible given inputs."""
        state = rssm.init_state(batch_size=1)
        action = torch.tensor([1], dtype=torch.long)

        # Execute step_prior twice with same inputs
        out1 = rssm.step_prior(action=action, prev_state=state, sample=False)
        out2 = rssm.step_prior(action=action, prev_state=state, sample=False)

        # h should be identical
        assert torch.allclose(out1.h, out2.h)
        # With sample=False, z should also be identical to mean_prior
        assert torch.allclose(out1.z, out2.z)
        assert torch.allclose(out1.z, out1.mean_prior)

    def test_stochastic_variational_update(self, rssm):
        """Verify stochastic sampling generates valid distributions."""
        state = rssm.init_state(batch_size=1)
        action = torch.tensor([2], dtype=torch.long)

        # Sampling with sample=True should produce stochastic variation
        out1 = rssm.step_prior(action=action, prev_state=state, sample=True)
        out2 = rssm.step_prior(action=action, prev_state=state, sample=True)

        assert out1.mean_prior.shape == (1, 16)
        assert out1.std_prior.shape == (1, 16)
        assert torch.all(out1.std_prior >= rssm.min_std)
        # Two random samples with non-zero std should differ
        assert not torch.allclose(out1.z, out2.z)

    def test_step_posterior_with_observation(self, rssm):
        """Verify posterior integrates observation and outputs both prior and posterior stats."""
        state = rssm.init_state(batch_size=2)
        obs = torch.randn(2, 8)
        action = torch.tensor([0, 1], dtype=torch.long)

        post_state = rssm.step_posterior(obs=obs, action=action, prev_state=state, sample=False)

        assert post_state.h.shape == (2, 32)
        assert post_state.z.shape == (2, 16)
        assert post_state.mean_prior.shape == (2, 16)
        assert post_state.std_prior.shape == (2, 16)
        assert post_state.mean_post.shape == (2, 16)
        assert post_state.std_post.shape == (2, 16)

        # Observation decoding and risk prediction
        recon_obs = rssm.decode_observation(post_state)
        risk = rssm.predict_risk(post_state)

        assert recon_obs.shape == (2, 8)
        assert risk.shape == (2, 1)
        assert torch.all((risk >= 0.0) & (risk <= 1.0))

    def test_reparameterization_gradient_flow(self, rssm):
        """Verify gradient flow through stochastic latent state z_t via reparameterization."""
        state = rssm.init_state(batch_size=2)
        obs = torch.randn(2, 8)
        action = torch.tensor([1, 2], dtype=torch.long)

        post_state = rssm.step_posterior(obs=obs, action=action, prev_state=state, sample=True)
        risk = rssm.predict_risk(post_state)
        loss = risk.sum()
        loss.backward()

        # Check gradients exist on encoder, prior, posterior, and GRU
        assert rssm.obs_encoder[0].weight.grad is not None
        assert torch.norm(rssm.obs_encoder[0].weight.grad) > 0.0
        assert rssm.posterior_net[0].weight.grad is not None
        assert torch.norm(rssm.posterior_net[0].weight.grad) > 0.0
        assert rssm.recurrent_cell.weight_hh.grad is not None

    def test_kl_divergence_analytic_correctness(self, rssm):
        """Verify analytical Gaussian KL divergence calculation against exact values."""
        # 1. Identical distributions -> KL == 0
        mu = torch.zeros(1, 4)
        std = torch.ones(1, 4)
        kl_zero = rssm.kl_divergence(mu, std, mu, std)
        assert torch.allclose(kl_zero, torch.tensor([0.0]), atol=1e-6)

        # 2. Known 1D Gaussian KL:
        # q ~ N(1, 2^2), p ~ N(0, 1)
        # KL = log(1/2) + (4 + 1)/(2*1) - 0.5 = -0.693147 + 2.5 - 0.5 = 1.30685
        mu_q = torch.tensor([[1.0]])
        std_q = torch.tensor([[2.0]])
        mu_p = torch.tensor([[0.0]])
        std_p = torch.tensor([[1.0]])
        kl_val = rssm.kl_divergence(mu_q, std_q, mu_p, std_p)
        expected = -math.log(2.0) + (4.0 + 1.0) / 2.0 - 0.5
        assert torch.allclose(kl_val, torch.tensor([expected]), atol=1e-5)

    def test_kl_balancing_regularization(self, rssm):
        """Verify KL balancing loss correctly balances prior and posterior gradients."""
        mean_post = torch.randn(2, 16, requires_grad=True)
        std_post = torch.ones(2, 16, requires_grad=True)
        mean_prior = torch.randn(2, 16, requires_grad=True)
        std_prior = torch.ones(2, 16, requires_grad=True)

        kl_loss = rssm.compute_balanced_kl_loss(
            mean_post=mean_post,
            std_post=std_post,
            mean_prior=mean_prior,
            std_prior=std_prior,
        )

        assert kl_loss.item() >= rssm.kl_free_nats
        kl_loss.backward()

        # Both prior and posterior should receive gradients
        assert mean_post.grad is not None
        assert mean_prior.grad is not None
        assert std_post.grad is not None
        assert std_prior.grad is not None

    def test_forward_sequence_and_training_step(self, rssm):
        """Verify sequential forward pass and loss reduction during optimization."""
        batch_size = 4
        seq_len = 6
        obs_seq = torch.randn(batch_size, seq_len, 8)
        act_seq = torch.randint(0, 4, (batch_size, seq_len))
        target_risks = torch.rand(batch_size, seq_len, 1)

        optimizer = optim.Adam(rssm.parameters(), lr=0.01)

        # Initial forward pass
        initial_out = rssm.forward(obs_seq=obs_seq, act_seq=act_seq, target_risks=target_risks)
        initial_loss = initial_out["total_loss"].item()

        # 3 training steps
        for _ in range(3):
            optimizer.zero_grad()
            out = rssm.forward(obs_seq=obs_seq, act_seq=act_seq, target_risks=target_risks)
            out["total_loss"].backward()
            optimizer.step()

        final_out = rssm.forward(obs_seq=obs_seq, act_seq=act_seq, target_risks=target_risks)
        final_loss = final_out["total_loss"].item()

        assert final_loss < initial_loss
        assert final_out["reconstructed_obs"].shape == (batch_size, seq_len, 8)
        assert final_out["predicted_risks"].shape == (batch_size, seq_len, 1)
        assert len(final_out["states"]) == seq_len


# ==============================================================================
# 2. CounterfactualDreamer Tests
# ==============================================================================


class TestCounterfactualDreamer:
    """Tests for multi-step latent imagination and causal counterfactual rollouts."""

    @pytest.fixture
    def trained_env(self):
        """Set up world model and dreamer."""
        torch.manual_seed(123)
        model = RecurrentStateSpaceModel(
            obs_dim=6,
            action_dim=4,
            hidden_dim=32,
            latent_dim=16,
            encoder_hidden_dim=32,
            decoder_hidden_dim=32,
        )
        dreamer = CounterfactualDreamer(
            world_model=model,
            default_horizon=8,
            critical_risk_threshold=0.60,
        )
        return model, dreamer

    def test_imagine_trajectory_shapes_and_metrics(self, trained_env):
        """Verify single trajectory rollout over K steps."""
        model, dreamer = trained_env
        state = model.init_state(batch_size=1)

        traj = dreamer.imagine_trajectory(
            initial_state=state,
            action=MitigationAction.EBPF_THROTTLE,
            horizon=5,
            sample=False,
        )

        assert isinstance(traj, CounterfactualTrajectory)
        assert traj.action == MitigationAction.EBPF_THROTTLE
        assert traj.horizon == 5
        assert len(traj.failure_risks) == 5
        assert traj.predicted_observations.shape == (5, 6)
        assert len(traj.states) == 5
        assert 0.0 <= traj.peak_risk <= 1.0
        assert 0.0 <= traj.mean_risk <= 1.0

        # Verify serialization
        d = traj.to_dict()
        assert d["action"] == "EBPF_THROTTLE"
        assert d["horizon"] == 5

    def test_counterfactual_comparison_against_baseline(self, trained_env):
        """Verify dreaming compares all actions against baseline NO_ACTION."""
        model, dreamer = trained_env
        state = model.init_state(batch_size=1)

        trajectories = dreamer.dream_counterfactuals(
            initial_state=state,
            horizon=6,
            baseline_action=MitigationAction.NO_ACTION,
        )

        assert "NO_ACTION" in trajectories
        assert "MIGRATE_REPLICA" in trajectories
        assert "EBPF_THROTTLE" in trajectories
        assert "CIRCUIT_BREAKER_OPEN" in trajectories

        base_traj = trajectories["NO_ACTION"]
        assert base_traj.risk_reduction == 0.0
        assert base_traj.causal_effect == 0.0

        for name, traj in trajectories.items():
            if name == "NO_ACTION":
                continue
            assert traj.risk_reduction >= 0.0
            assert traj.causal_effect >= 0.0
            assert traj.horizon == 6

    def test_time_to_failure_detection(self, trained_env):
        """Verify time-to-failure step is accurately marked when threshold is crossed."""
        model, dreamer = trained_env
        dreamer.critical_risk_threshold = 0.50

        # Mock predict_risk to return escalating failure risks
        step_counter = {"count": 0}
        orig_predict = model.predict_risk

        def mock_escalating_risk(s):
            step_counter["count"] += 1
            # Step 1: 0.1, Step 2: 0.3, Step 3: 0.7 (crosses 0.50 at step 3)
            val = min(1.0, step_counter["count"] * 0.25)
            return torch.tensor([[val]])

        model.predict_risk = mock_escalating_risk
        state = model.init_state(batch_size=1)

        traj = dreamer.imagine_trajectory(
            initial_state=state,
            action=MitigationAction.NO_ACTION,
            horizon=5,
        )

        assert traj.time_to_failure == 2 or traj.time_to_failure == 3
        assert traj.failure_risks[traj.time_to_failure - 1] >= 0.50


# ==============================================================================
# 3. AutonomousMitigator Tests
# ==============================================================================


class TestAutonomousMitigator:
    """Tests for the pre-emptive self-healing controller."""

    @pytest.fixture
    def mitigator_setup(self):
        torch.manual_seed(42)
        model = RecurrentStateSpaceModel(
            obs_dim=4,
            action_dim=4,
            hidden_dim=24,
            latent_dim=12,
        )
        dreamer = CounterfactualDreamer(
            world_model=model,
            default_horizon=6,
            critical_risk_threshold=0.70,
        )
        stream = RemediationEventStream(max_size=20)
        mitigator = AutonomousMitigator(
            dreamer=dreamer,
            warning_threshold=0.40,
            critical_threshold=0.70,
            safe_target_risk=0.25,
            horizon=6,
            event_stream=stream,
        )
        return model, dreamer, mitigator, stream

    def test_steady_state_no_action_trigger(self, mitigator_setup):
        """When baseline failure risk is low, mitigator maintains NO_ACTION."""
        model, dreamer, mitigator, stream = mitigator_setup

        # Force predict_risk to return low healthy risk (0.10)
        model.predict_risk = lambda s: torch.tensor([[0.10]])
        state = model.init_state(batch_size=1)

        decision = mitigator.evaluate_and_mitigate(
            initial_state=state,
            target_resource="pod/payment-service-84a",
        )

        assert isinstance(decision, MitigationDecision)
        assert decision.action == MitigationAction.NO_ACTION
        assert decision.urgency == MitigationUrgency.NONE
        assert decision.lead_time_steps is None
        assert decision.target_resource == "pod/payment-service-84a"
        assert "remains below warning threshold" in decision.rationale
        assert len(stream) == 1

    def test_impending_failure_triggers_ebpf_backpressure(self, mitigator_setup):
        """When high risk is mitigated best by eBPF throttle, trigger EBPF_THROTTLE."""
        model, dreamer, mitigator, stream = mitigator_setup

        # Simulate scenario where:
        # NO_ACTION -> failure risk 0.85 at step 3
        # EBPF_THROTTLE -> failure risk 0.15 (safe)
        # MIGRATE_REPLICA -> failure risk 0.20
        # CIRCUIT_BREAKER_OPEN -> failure risk 0.35
        def mock_scenario_risk(state):
            # Check last applied action via state
            return torch.tensor([[0.5]])

        # We can directly mock dreamer.dream_counterfactuals to simulate exact scenario
        fake_trajectories = {
            "NO_ACTION": CounterfactualTrajectory(
                action=MitigationAction.NO_ACTION,
                action_name=MitigationAction.NO_ACTION.human_name,
                horizon=6,
                failure_risks=np.array([0.2, 0.5, 0.85, 0.9, 0.9, 0.9]),
                predicted_observations=np.zeros((6, 4)),
                peak_risk=0.90,
                mean_risk=0.70,
                time_to_failure=3,
                risk_reduction=0.0,
                causal_effect=0.0,
            ),
            "EBPF_THROTTLE": CounterfactualTrajectory(
                action=MitigationAction.EBPF_THROTTLE,
                action_name=MitigationAction.EBPF_THROTTLE.human_name,
                horizon=6,
                failure_risks=np.array([0.2, 0.25, 0.20, 0.15, 0.15, 0.15]),
                predicted_observations=np.zeros((6, 4)),
                peak_risk=0.25,
                mean_risk=0.18,
                time_to_failure=None,
                risk_reduction=0.65,
                causal_effect=72.2,
            ),
            "MIGRATE_REPLICA": CounterfactualTrajectory(
                action=MitigationAction.MIGRATE_REPLICA,
                action_name=MitigationAction.MIGRATE_REPLICA.human_name,
                horizon=6,
                failure_risks=np.array([0.2, 0.3, 0.25, 0.20, 0.10, 0.05]),
                predicted_observations=np.zeros((6, 4)),
                peak_risk=0.30,
                mean_risk=0.18,
                time_to_failure=None,
                risk_reduction=0.60,
                causal_effect=66.7,
            ),
            "CIRCUIT_BREAKER_OPEN": CounterfactualTrajectory(
                action=MitigationAction.CIRCUIT_BREAKER_OPEN,
                action_name=MitigationAction.CIRCUIT_BREAKER_OPEN.human_name,
                horizon=6,
                failure_risks=np.array([0.2, 0.35, 0.35, 0.35, 0.35, 0.35]),
                predicted_observations=np.zeros((6, 4)),
                peak_risk=0.35,
                mean_risk=0.32,
                time_to_failure=None,
                risk_reduction=0.55,
                causal_effect=61.1,
            ),
        }

        dreamer.dream_counterfactuals = lambda **kwargs: fake_trajectories
        state = model.init_state(batch_size=1)

        decision = mitigator.evaluate_and_mitigate(
            initial_state=state,
            target_resource="service/checkout-api",
            metadata={"tcp_window_clamp_bytes": 16384, "max_bandwidth_mbps": 40},
        )

        assert decision.action == MitigationAction.EBPF_THROTTLE
        assert decision.urgency in (MitigationUrgency.HIGH, MitigationUrgency.CRITICAL)
        assert decision.lead_time_steps == 3
        assert decision.baseline_risk == 0.90
        assert decision.mitigated_risk == 0.25
        assert decision.risk_reduction == 0.65
        assert decision.parameters["tcp_window_clamp_bytes"] == 16384
        assert any("bpf_prog_attach" in cmd for cmd in decision.action_plan)

        # Check log line format
        log_line = decision.format_log()
        assert "[PRE-EMPTIVE-CONTROL]" in log_line
        assert "service/checkout-api" in log_line
        assert "EBPF_THROTTLE" in log_line
        assert "T+3 steps" in log_line

    def test_replica_migration_directive_generation(self, mitigator_setup):
        """Verify live container migration generates valid Kubernetes/SDN plan."""
        model, dreamer, mitigator, stream = mitigator_setup

        fake_trajectories = {
            "NO_ACTION": CounterfactualTrajectory(
                action=MitigationAction.NO_ACTION,
                action_name=MitigationAction.NO_ACTION.human_name,
                horizon=6,
                failure_risks=np.array([0.3, 0.6, 0.88, 0.95]),
                predicted_observations=np.zeros((4, 4)),
                peak_risk=0.95,
                mean_risk=0.68,
                time_to_failure=2,
                risk_reduction=0.0,
                causal_effect=0.0,
            ),
            "MIGRATE_REPLICA": CounterfactualTrajectory(
                action=MitigationAction.MIGRATE_REPLICA,
                action_name=MitigationAction.MIGRATE_REPLICA.human_name,
                horizon=6,
                failure_risks=np.array([0.3, 0.35, 0.15, 0.05]),
                predicted_observations=np.zeros((4, 4)),
                peak_risk=0.35,
                mean_risk=0.21,
                time_to_failure=None,
                risk_reduction=0.60,
                causal_effect=63.2,
            ),
        }
        dreamer.dream_counterfactuals = lambda **kwargs: fake_trajectories
        state = model.init_state(batch_size=1)

        decision = mitigator.evaluate_and_mitigate(
            initial_state=state,
            target_resource="pod/stateful-db-0",
            candidate_actions=[MitigationAction.NO_ACTION, MitigationAction.MIGRATE_REPLICA],
            metadata={"destination_node": "worker-node-backup-09", "grace_period_sec": 20},
        )

        assert decision.action == MitigationAction.MIGRATE_REPLICA
        assert decision.parameters["destination_node"] == "worker-node-backup-09"
        assert decision.parameters["grace_period_sec"] == 20
        assert any("kubectl cordon" in cmd for cmd in decision.action_plan)
        assert any("worker-node-backup-09" in cmd for cmd in decision.action_plan)

    def test_dynamic_circuit_breaking_directive(self, mitigator_setup):
        """Verify circuit breaker directive opens circuit and sets fallback routes."""
        model, dreamer, mitigator, stream = mitigator_setup

        fake_trajectories = {
            "NO_ACTION": CounterfactualTrajectory(
                action=MitigationAction.NO_ACTION,
                action_name=MitigationAction.NO_ACTION.human_name,
                horizon=6,
                failure_risks=np.array([0.4, 0.8, 0.99]),
                predicted_observations=np.zeros((3, 4)),
                peak_risk=0.99,
                mean_risk=0.73,
                time_to_failure=2,
                risk_reduction=0.0,
                causal_effect=0.0,
            ),
            "CIRCUIT_BREAKER_OPEN": CounterfactualTrajectory(
                action=MitigationAction.CIRCUIT_BREAKER_OPEN,
                action_name=MitigationAction.CIRCUIT_BREAKER_OPEN.human_name,
                horizon=6,
                failure_risks=np.array([0.4, 0.2, 0.05]),
                predicted_observations=np.zeros((3, 4)),
                peak_risk=0.40,
                mean_risk=0.21,
                time_to_failure=None,
                risk_reduction=0.59,
                causal_effect=59.6,
            ),
        }
        dreamer.dream_counterfactuals = lambda **kwargs: fake_trajectories
        state = model.init_state(batch_size=1)

        decision = mitigator.evaluate_and_mitigate(
            initial_state=state,
            target_resource="service/inventory-upstream",
            candidate_actions=[MitigationAction.NO_ACTION, MitigationAction.CIRCUIT_BREAKER_OPEN],
            metadata={"cooldown_seconds": 45, "fallback_route": "cached_inventory_snapshot"},
        )

        assert decision.action == MitigationAction.CIRCUIT_BREAKER_OPEN
        assert decision.parameters["cooldown_seconds"] == 45
        assert decision.parameters["fallback_route"] == "cached_inventory_snapshot"
        assert any("circuit_breaker" in cmd for cmd in decision.action_plan)

    def test_step_and_mitigate_online_filtering(self, mitigator_setup):
        """Verify online streaming step integrates new observation and yields decision."""
        model, dreamer, mitigator, stream = mitigator_setup
        state = model.init_state(batch_size=1)
        obs = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)

        decision, next_state = mitigator.step_and_mitigate(
            current_obs=obs,
            prev_action=MitigationAction.NO_ACTION,
            prev_state=state,
            target_resource="service/ingress-router",
        )

        assert isinstance(decision, MitigationDecision)
        assert isinstance(next_state, RSSMState)
        assert next_state.h.shape == state.h.shape


# ==============================================================================
# 4. RemediationEventStream Tests
# ==============================================================================


class TestRemediationEventStream:
    """Tests for remediation audit trail and event streaming."""

    def test_event_stream_buffering_and_filtering(self):
        stream = RemediationEventStream(max_size=3)

        d1 = MitigationDecision(
            action=MitigationAction.NO_ACTION,
            action_name="No Action",
            target_resource="res-1",
            baseline_risk=0.1,
            mitigated_risk=0.1,
            risk_reduction=0.0,
            lead_time_steps=None,
            urgency=MitigationUrgency.NONE,
            confidence=0.9,
            rationale="OK",
        )
        d2 = MitigationDecision(
            action=MitigationAction.EBPF_THROTTLE,
            action_name="eBPF Throttle",
            target_resource="res-2",
            baseline_risk=0.8,
            mitigated_risk=0.2,
            risk_reduction=0.6,
            lead_time_steps=2,
            urgency=MitigationUrgency.HIGH,
            confidence=0.9,
            rationale="Throttled",
        )
        d3 = MitigationDecision(
            action=MitigationAction.MIGRATE_REPLICA,
            action_name="Migrate",
            target_resource="res-3",
            baseline_risk=0.9,
            mitigated_risk=0.1,
            risk_reduction=0.8,
            lead_time_steps=1,
            urgency=MitigationUrgency.CRITICAL,
            confidence=0.9,
            rationale="Migrated",
        )

        stream.record_decision(d1)
        stream.record_decision(d2)
        stream.record_decision(d3)
        assert len(stream) == 3

        # Test filtering
        ebpf_events = stream.filter_by_action(MitigationAction.EBPF_THROTTLE)
        assert len(ebpf_events) == 1
        assert ebpf_events[0].target_resource == "res-2"

        crit_events = stream.filter_by_urgency(MitigationUrgency.CRITICAL)
        assert len(crit_events) == 1
        assert crit_events[0].target_resource == "res-3"

        # Test overflow eviction
        d4 = MitigationDecision(
            action=MitigationAction.NO_ACTION,
            action_name="No Action",
            target_resource="res-4",
            baseline_risk=0.1,
            mitigated_risk=0.1,
            risk_reduction=0.0,
            lead_time_steps=None,
            urgency=MitigationUrgency.NONE,
            confidence=0.9,
            rationale="OK",
        )
        stream.record_decision(d4)
        assert len(stream) == 3
        # First event d1 evicted
        assert stream.get_recent_events()[0].target_resource == "res-2"

        # Test JSON serialization
        exported = stream.export_json()
        parsed = json.loads(exported)
        assert len(parsed) == 3
        assert parsed[-1]["target_resource"] == "res-4"

        # Test clear
        stream.clear()
        assert len(stream) == 0


# ==============================================================================
# 5. Distributed System Chaos & End-to-End Self-Healing Simulation Tests
# ==============================================================================


class TestDistributedSystemChaosSimulation:
    """
    End-to-end chaos engineering simulation testing pre-emptive self-healing
    under simulated microservice cascading failure scenarios.
    """

    @pytest.fixture
    def chaos_environment(self):
        torch.manual_seed(101)
        # Telemetry: [cpu_util, mem_util, queue_depth, p99_latency]
        rssm = RecurrentStateSpaceModel(
            obs_dim=4,
            action_dim=4,
            action_embed_dim=8,
            hidden_dim=32,
            latent_dim=16,
            encoder_hidden_dim=32,
            decoder_hidden_dim=32,
        )
        dreamer = CounterfactualDreamer(
            world_model=rssm,
            default_horizon=6,
            critical_risk_threshold=0.75,
        )
        stream = RemediationEventStream(max_size=50)
        mitigator = AutonomousMitigator(
            dreamer=dreamer,
            warning_threshold=0.45,
            critical_threshold=0.75,
            safe_target_risk=0.25,
            horizon=6,
            event_stream=stream,
        )
        return rssm, dreamer, mitigator, stream

    def test_continuous_action_vector_support(self, chaos_environment):
        """Verify RSSM supports continuous action embeddings in addition to discrete indices."""
        rssm, _, _, _ = chaos_environment
        state = rssm.init_state(batch_size=2)
        # Continuous action vector of shape (batch_size, action_embed_dim)
        continuous_action = torch.randn(2, rssm.action_embed_dim)

        out_prior = rssm.step_prior(action=continuous_action, prev_state=state, sample=False)
        assert out_prior.h.shape == (2, rssm.hidden_dim)

        obs = torch.randn(2, 4)
        out_post = rssm.step_posterior(
            obs=obs, action=continuous_action, prev_state=state, sample=False
        )
        assert out_post.h.shape == (2, rssm.hidden_dim)

    def test_cost_penalty_steers_mitigation_choice(self, chaos_environment):
        """Verify operational cost penalties steer mitigator between high-impact and low-overhead interventions."""
        rssm, dreamer, mitigator, _ = chaos_environment

        # Simulate scenario where both actions mitigate risk equally well:
        # Peak risk under NO_ACTION = 0.90
        # Peak risk under EBPF_THROTTLE = 0.15 (risk reduction 0.75)
        # Peak risk under MIGRATE_REPLICA = 0.12 (risk reduction 0.78)
        fake_trajectories = {
            "NO_ACTION": CounterfactualTrajectory(
                action=MitigationAction.NO_ACTION,
                action_name=MitigationAction.NO_ACTION.human_name,
                horizon=6,
                failure_risks=np.array([0.2, 0.5, 0.9, 0.9]),
                predicted_observations=np.zeros((4, 4)),
                peak_risk=0.90,
                mean_risk=0.62,
                time_to_failure=3,
                risk_reduction=0.0,
                causal_effect=0.0,
            ),
            "EBPF_THROTTLE": CounterfactualTrajectory(
                action=MitigationAction.EBPF_THROTTLE,
                action_name=MitigationAction.EBPF_THROTTLE.human_name,
                horizon=6,
                failure_risks=np.array([0.2, 0.2, 0.15, 0.15]),
                predicted_observations=np.zeros((4, 4)),
                peak_risk=0.20,
                mean_risk=0.17,
                time_to_failure=None,
                risk_reduction=0.70,
                causal_effect=77.8,
            ),
            "MIGRATE_REPLICA": CounterfactualTrajectory(
                action=MitigationAction.MIGRATE_REPLICA,
                action_name=MitigationAction.MIGRATE_REPLICA.human_name,
                horizon=6,
                failure_risks=np.array([0.2, 0.25, 0.15, 0.10]),
                predicted_observations=np.zeros((4, 4)),
                peak_risk=0.25,
                mean_risk=0.17,
                time_to_failure=None,
                risk_reduction=0.65,
                causal_effect=72.2,
            ),
        }
        dreamer.dream_counterfactuals = lambda **kwargs: fake_trajectories
        state = rssm.init_state(batch_size=1)

        # Standard costs: EBPF_THROTTLE cost (0.08) is much lower than MIGRATE_REPLICA (0.25)
        decision = mitigator.evaluate_and_mitigate(
            initial_state=state,
            target_resource="service/ingress-api",
            candidate_actions=[
                MitigationAction.NO_ACTION,
                MitigationAction.EBPF_THROTTLE,
                MitigationAction.MIGRATE_REPLICA,
            ],
        )
        assert decision.action == MitigationAction.EBPF_THROTTLE

        # If we drastically penalize eBPF cost and make migration free:
        mitigator.action_costs[MitigationAction.EBPF_THROTTLE] = 0.80
        mitigator.action_costs[MitigationAction.MIGRATE_REPLICA] = 0.01

        decision2 = mitigator.evaluate_and_mitigate(
            initial_state=state,
            target_resource="service/ingress-api",
            candidate_actions=[
                MitigationAction.NO_ACTION,
                MitigationAction.EBPF_THROTTLE,
                MitigationAction.MIGRATE_REPLICA,
            ],
        )
        assert decision2.action == MitigationAction.MIGRATE_REPLICA

    def test_multi_step_streaming_incident_lifecycle(self, chaos_environment):
        """
        Simulate an entire streaming timeline:
        - Steps 0..2: Healthy traffic -> NO_ACTION
        - Step 3: Ingress surge detected -> Pre-emptive mitigation triggered before cascade
        - Step 4: System stabilized -> Healthy telemetry restored
        """
        rssm, dreamer, mitigator, stream = chaos_environment
        state = rssm.init_state(batch_size=1)

        # 1. Healthy phases (steps 0..2)
        healthy_obs = np.array([0.25, 0.30, 0.10, 0.15], dtype=np.float32)
        healthy_decision, state = mitigator.step_and_mitigate(
            current_obs=healthy_obs,
            prev_action=MitigationAction.NO_ACTION,
            prev_state=state,
            target_resource="service/payment-gateway",
        )
        assert isinstance(healthy_decision, MitigationDecision)

        # 2. Mock impending cascade at step 3
        escalating_trajectories = {
            "NO_ACTION": CounterfactualTrajectory(
                action=MitigationAction.NO_ACTION,
                action_name=MitigationAction.NO_ACTION.human_name,
                horizon=6,
                failure_risks=np.array([0.3, 0.55, 0.85, 0.98]),
                predicted_observations=np.zeros((4, 4)),
                peak_risk=0.98,
                mean_risk=0.67,
                time_to_failure=3,
                risk_reduction=0.0,
                causal_effect=0.0,
            ),
            "CIRCUIT_BREAKER_OPEN": CounterfactualTrajectory(
                action=MitigationAction.CIRCUIT_BREAKER_OPEN,
                action_name=MitigationAction.CIRCUIT_BREAKER_OPEN.human_name,
                horizon=6,
                failure_risks=np.array([0.3, 0.25, 0.15, 0.10]),
                predicted_observations=np.zeros((4, 4)),
                peak_risk=0.30,
                mean_risk=0.20,
                time_to_failure=None,
                risk_reduction=0.68,
                causal_effect=69.4,
            ),
        }
        dreamer.dream_counterfactuals = lambda **kwargs: escalating_trajectories

        surge_obs = np.array([0.85, 0.90, 0.95, 0.92], dtype=np.float32)
        incident_decision, state = mitigator.step_and_mitigate(
            current_obs=surge_obs,
            prev_action=MitigationAction.NO_ACTION,
            prev_state=state,
            target_resource="service/payment-gateway",
        )

        assert incident_decision.action == MitigationAction.CIRCUIT_BREAKER_OPEN
        assert incident_decision.urgency in (MitigationUrgency.HIGH, MitigationUrgency.CRITICAL)
        assert incident_decision.lead_time_steps == 3
        assert incident_decision.risk_reduction > 0.60
        assert len(incident_decision.action_plan) >= 3

        # Verify event stream logged this incident
        assert len(stream) >= 2
        recent = stream.get_recent_events(limit=10)
        assert recent[-1].action == MitigationAction.CIRCUIT_BREAKER_OPEN
        assert "service/payment-gateway" in recent[-1].target_resource
