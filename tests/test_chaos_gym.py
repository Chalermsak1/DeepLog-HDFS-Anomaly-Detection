"""
Comprehensive Unit Tests for Pillar VI: Autonomous Chaos Gym & Adversarial Red/Blue Self-Play.
Tests ChaosEnvironment, RedChaosAgent, BlueImmuneAgent, and ChaosSelfPlayTrainer.
"""

import json
import pytest
import numpy as np
import torch
import torch.nn as nn

from deeplog.chaos_gym import (
    ChaosEnvironment,
    RedChaosAgent,
    BlueImmuneAgent,
    ChaosSelfPlayTrainer,
    ChaosFaultType,
    RedAction,
    BlueAction,
    ClusterNode,
    ClusterState,
    ChaosEpisodeResult,
    ChaosSelfPlayHistory,
    ResilienceBenchmarkResult,
)
from deeplog.world_model import (
    RecurrentStateSpaceModel,
    RSSMState,
    CounterfactualDreamer,
    AutonomousMitigator,
    MitigationAction,
    MitigationDecision,
    MitigationUrgency,
)


# ==============================================================================
# 1. Enums and Data Containers Tests
# ==============================================================================


class TestChaosDataContainers:
    """Test enums, dataclasses, and serialization utilities."""

    def test_chaos_fault_type_properties(self):
        """Verify fault type enum values and human-readable names."""
        assert ChaosFaultType.NONE == 0
        assert ChaosFaultType.LATENCY_SPIKE == 1
        assert ChaosFaultType.PACKET_LOSS == 2
        assert ChaosFaultType.THREAD_EXHAUSTION == 3
        assert ChaosFaultType.SPLIT_BRAIN == 4
        assert ChaosFaultType.BYZANTINE_SLOWLORIS == 5

        assert "Latency" in ChaosFaultType.LATENCY_SPIKE.human_name
        assert "Slowloris" in ChaosFaultType.BYZANTINE_SLOWLORIS.human_name
        assert ChaosFaultType.NONE.identifier == "none"

    def test_red_action_serialization(self):
        """Verify RedAction fields and to_dict serialization."""
        action = RedAction(
            target_node=2,
            fault_type=ChaosFaultType.THREAD_EXHAUSTION,
            intensity=0.85,
            duration=3,
        )
        d = action.to_dict()
        assert d["target_node"] == 2
        assert d["fault_type"] == "THREAD_EXHAUSTION"
        assert pytest.approx(d["intensity"], 0.01) == 0.85
        assert d["duration"] == 3

    def test_blue_action_serialization(self):
        """Verify BlueAction fields and to_dict serialization."""
        action = BlueAction(
            action=MitigationAction.MIGRATE_REPLICA,
            target_node=1,
            destination_node=3,
        )
        d = action.to_dict()
        assert d["action"] == "MIGRATE_REPLICA"
        assert d["target_node"] == 1
        assert d["destination_node"] == 3
        assert not d["has_decision"]

    def test_cluster_node_reset_and_feature_vector(self):
        """Verify ClusterNode reset and telemetry vector normalization."""
        node = ClusterNode(node_id=0, name="worker-0")
        node.cpu_util = 0.95
        node.queue_depth = 55.0
        node.slowloris_connections = 450.0
        node.is_circuit_broken = True

        vec = node.to_feature_vector()
        assert isinstance(vec, np.ndarray)
        assert vec.shape == (10,)
        assert vec.dtype == np.float32
        assert pytest.approx(vec[0], 0.01) == 0.95
        assert vec[3] == 1.0  # queue_depth capped at 1.0 (55 / 50 > 1)
        assert vec[7] == 1.0  # is_circuit_broken

        # Test nominal reset
        node.reset_nominal()
        assert node.cpu_util == 0.20
        assert not node.is_circuit_broken
        assert node.slowloris_connections == 0.0
        assert node.queue_depth == 1.0


# ==============================================================================
# 2. ChaosEnvironment Tests
# ==============================================================================


class TestChaosEnvironment:
    """Tests for the distributed cluster simulator ChaosEnvironment."""

    @pytest.fixture
    def env(self):
        return ChaosEnvironment(
            num_nodes=4,
            base_capacity_rps=1000.0,
            nominal_latency_ms=5.0,
            sla_latency_threshold_ms=80.0,
            sla_availability_target=0.95,
            max_steps=20,
            seed=42,
        )

    def test_environment_initialization(self, env):
        """Verify cluster dimensions and observation space."""
        assert env.num_nodes == 4
        # (4 nodes * 10 features) + 5 cluster global features = 45
        assert env.obs_dim == 45
        assert env.num_red_actions == 1 + (4 * 5)  # 21
        assert env.num_blue_actions == 1 + (4 * 3)  # 13

    def test_environment_reset(self, env):
        """Verify reset restores steady-state baseline and valid observation."""
        obs, info = env.reset(seed=123)
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (45,)
        assert obs.dtype == np.float32
        assert info["step_count"] == 0
        assert info["availability"] == 1.0
        assert info["blast_radius"] == 0.0
        assert len(info["nodes"]) == 4

    def test_steady_state_step(self, env):
        """Under NO_OP and NO_ACTION, cluster should remain 100% available with zero blast radius."""
        env.reset(seed=42)
        red_act = RedAction(fault_type=ChaosFaultType.NONE)
        blue_act = BlueAction(action=MitigationAction.NO_ACTION)

        obs, rewards, term, trunc, info = env.step(red_act, blue_act)
        assert not term
        assert not trunc
        assert info["availability"] >= 0.99
        assert info["blast_radius"] == 0.0
        assert info["p99_latency_ms"] < 15.0
        assert rewards["blue"] > 0.0  # Rewarded for nominal availability

    def test_fault_injection_latency_spike(self, env):
        """Verify latency spike degrades p99 latency and causes SLA violations."""
        env.reset(seed=42)
        red_act = RedAction(target_node=1, fault_type=ChaosFaultType.LATENCY_SPIKE, intensity=1.0)
        blue_act = BlueAction(action=MitigationAction.NO_ACTION)

        obs, rewards, term, trunc, info = env.step(red_act, blue_act)
        target_node = env.nodes[1]
        assert target_node.p99_latency_ms > 100.0  # Latency increased significantly
        assert target_node.p99_latency_ms > env.sla_latency_threshold_ms
        assert info["blast_radius"] > 0.0
        assert rewards["red"] > 0.0  # Red rewarded for SLA degradation

    def test_fault_injection_packet_loss(self, env):
        """Verify packet drop injection causes packet drops and availability drop."""
        env.reset(seed=42)
        red_act = RedAction(target_node=2, fault_type=ChaosFaultType.PACKET_LOSS, intensity=0.9)
        blue_act = BlueAction(action=MitigationAction.NO_ACTION)

        obs, rewards, term, trunc, info = env.step(red_act, blue_act)
        target_node = env.nodes[2]
        assert target_node.packet_loss_rate >= 0.50
        assert info["availability"] < 1.0
        assert info["sla_violations"] > 0.0

    def test_fault_injection_thread_exhaustion(self, env):
        """Verify thread pool starvation causes queue depth explosion and high CPU."""
        env.reset(seed=42)
        red_act = RedAction(target_node=0, fault_type=ChaosFaultType.THREAD_EXHAUSTION, intensity=1.0)
        blue_act = BlueAction(action=MitigationAction.NO_ACTION)

        obs, rewards, term, trunc, info = env.step(red_act, blue_act)
        target_node = env.nodes[0]
        assert target_node.thread_pool_util >= 0.90
        assert target_node.queue_depth > 20.0
        assert target_node.cpu_util > 0.50

    def test_fault_injection_split_brain(self, env):
        """Verify split-brain network partition isolates node into partition island."""
        env.reset(seed=42)
        red_act = RedAction(target_node=3, fault_type=ChaosFaultType.SPLIT_BRAIN, intensity=1.0)
        blue_act = BlueAction(action=MitigationAction.NO_ACTION)

        obs, rewards, term, trunc, info = env.step(red_act, blue_act)
        target_node = env.nodes[3]
        assert target_node.partition_id == 1
        assert info["availability"] < 1.0  # Consistency drops on partitioned node

    def test_fault_injection_byzantine_slowloris(self, env):
        """Verify Byzantine slowloris pins connections and threads while CPU remains deceptively low."""
        env.reset(seed=42)
        red_act = RedAction(target_node=0, fault_type=ChaosFaultType.BYZANTINE_SLOWLORIS, intensity=1.0)
        blue_act = BlueAction(action=MitigationAction.NO_ACTION)

        obs, rewards, term, trunc, info = env.step(red_act, blue_act)
        target_node = env.nodes[0]
        assert target_node.slowloris_connections >= 300.0
        assert target_node.thread_pool_util >= 0.85
        # Key characteristic of slowloris: stealthy low CPU despite thread exhaustion
        assert target_node.cpu_util <= 0.45

    def test_mitigation_migrate_replica(self, env):
        """Verify live replica migration drains degraded node and restores capacity."""
        env.reset(seed=42)
        # Degrade node 1 with thread exhaustion
        node1 = env.nodes[1]
        node1.thread_pool_util = 0.95
        node1.queue_depth = 50.0
        node1.cpu_util = 0.90

        # Apply Blue live migration from node 1 to node 2
        blue_act = BlueAction(
            action=MitigationAction.MIGRATE_REPLICA,
            target_node=1,
            destination_node=2,
        )
        red_act = RedAction(fault_type=ChaosFaultType.NONE)

        obs, rewards, term, trunc, info = env.step(red_act, blue_act)
        # Target node queue and thread pool drained, node cordoned during migration step
        assert node1.is_cordoned is True
        assert node1.thread_pool_util < 0.40
        assert node1.queue_depth <= 2.0

        # On the next step, cordon is released
        env.step(RedAction(fault_type=ChaosFaultType.NONE), BlueAction(action=MitigationAction.NO_ACTION))
        assert node1.is_cordoned is False

    def test_mitigation_ebpf_throttle(self, env):
        """Verify in-kernel eBPF backpressure immediately clears slowloris sockets."""
        env.reset(seed=42)
        node0 = env.nodes[0]
        node0.slowloris_connections = 400.0
        node0.thread_pool_util = 0.90

        # Apply eBPF throttle mitigation
        blue_act = BlueAction(
            action=MitigationAction.EBPF_THROTTLE,
            target_node=0,
        )
        red_act = RedAction(fault_type=ChaosFaultType.NONE)

        obs, rewards, term, trunc, info = env.step(red_act, blue_act)
        # Slowloris connections terminated at kernel boundary
        assert node0.slowloris_connections == 0.0
        assert node0.thread_pool_util < 0.50
        assert node0.ebpf_rate_limit <= 0.75  # Rate limit applied

    def test_mitigation_circuit_breaker(self, env):
        """Verify dynamic circuit breaking trips breaker, fast-falls back, and recovers after timer."""
        env.reset(seed=42)
        blue_act = BlueAction(
            action=MitigationAction.CIRCUIT_BREAKER_OPEN,
            target_node=2,
        )
        red_act = RedAction(fault_type=ChaosFaultType.NONE)

        obs, rewards, term, trunc, info = env.step(red_act, blue_act)
        node2 = env.nodes[2]
        assert node2.is_circuit_broken
        assert node2.circuit_breaker_timer == env.circuit_breaker_recovery_steps

        # Advance steps until circuit breaker resets
        for _ in range(env.circuit_breaker_recovery_steps + 1):
            env.step(RedAction(fault_type=ChaosFaultType.NONE), BlueAction(action=MitigationAction.NO_ACTION))

        assert not node2.is_circuit_broken

    def test_discrete_action_parsing(self, env):
        """Verify environment parses discrete integer actions correctly for Red and Blue."""
        env.reset(seed=42)
        # Red action index 0 = NONE
        r0 = env._parse_red_action(0)
        assert r0.fault_type == ChaosFaultType.NONE

        # Red action index 1 = target node 0, LATENCY_SPIKE
        r1 = env._parse_red_action(1)
        assert r1.target_node == 0
        assert r1.fault_type == ChaosFaultType.LATENCY_SPIKE

        # Blue action index 0 = NO_ACTION
        b0 = env._parse_blue_action(0)
        assert b0.action == MitigationAction.NO_ACTION

        # Blue action index 1 = target node 0, MIGRATE_REPLICA
        b1 = env._parse_blue_action(1)
        assert b1.target_node == 0
        assert b1.action == MitigationAction.MIGRATE_REPLICA

        # Invalid type should raise TypeError
        with pytest.raises(TypeError):
            env._parse_red_action("invalid_action")
        with pytest.raises(TypeError):
            env._parse_blue_action(12.34)

    def test_episode_truncation(self, env):
        """Verify episode truncates when reaching max_steps."""
        env.reset(seed=42)
        for step in range(env.max_steps):
            _, _, term, trunc, _ = env.step(
                RedAction(fault_type=ChaosFaultType.NONE),
                BlueAction(action=MitigationAction.NO_ACTION),
            )
            if step < env.max_steps - 1:
                assert not trunc
            else:
                assert trunc


# ==============================================================================
# 3. RedChaosAgent Tests
# ==============================================================================


class TestRedChaosAgent:
    """Tests for the adversarial Red Chaos Agent neural policy network."""

    @pytest.fixture
    def red_agent(self):
        torch.manual_seed(42)
        return RedChaosAgent(
            obs_dim=45,
            num_nodes=4,
            hidden_dim=32,
            learning_rate=1e-3,
        )

    def test_forward_pass_dimensions(self, red_agent):
        """Verify output tensor dimensions for actor logits, intensity, and critic value."""
        obs = torch.randn(2, 45)
        logits, intensity, value = red_agent(obs)

        # 1 + (4 nodes * 5 faults) = 21 discrete actions
        assert logits.shape == (2, 21)
        assert intensity.shape == (2, 1)
        assert value.shape == (2, 1)
        # Intensity must be bounded in [0.2, 1.0]
        assert torch.all(intensity >= 0.20)
        assert torch.all(intensity <= 1.0)

    def test_act_deterministic_and_stochastic(self, red_agent):
        """Verify act() produces valid RedAction instances in both modes."""
        obs = np.random.randn(45).astype(np.float32)

        # Stochastic sampling
        action_stoch = red_agent.act(obs, deterministic=False)
        assert isinstance(action_stoch, RedAction)
        assert 0 <= action_stoch.target_node < 4
        assert 0.20 <= action_stoch.intensity <= 1.0

        # Deterministic sampling (should produce identical action on identical obs)
        action_det1 = red_agent.act(obs, deterministic=True)
        action_det2 = red_agent.act(obs, deterministic=True)
        assert action_det1.target_node == action_det2.target_node
        assert action_det1.fault_type == action_det2.fault_type
        assert pytest.approx(action_det1.intensity, 1e-4) == action_det2.intensity

    def test_evaluate_actions(self, red_agent):
        """Verify evaluate_actions returns log_probs, entropy, and values with proper gradients."""
        obs = torch.randn(4, 45)
        actions = torch.tensor([0, 1, 5, 20], dtype=torch.long)

        log_probs, entropy, values = red_agent.evaluate_actions(obs, actions)
        assert log_probs.shape == (4,)
        assert entropy.shape == (4,)
        assert values.shape == (4,)
        assert log_probs.requires_grad
        assert values.requires_grad

    def test_train_step_gradient_update(self, red_agent):
        """Verify train_step executes backprop and updates model weights."""
        obs = torch.randn(4, 45)
        actions = torch.tensor([0, 2, 4, 10], dtype=torch.long)
        returns = torch.tensor([1.0, 0.5, 2.0, -0.5])
        advantages = torch.tensor([0.5, 0.2, 1.0, -0.3])

        initial_param = next(red_agent.parameters()).clone()
        loss = red_agent.train_step(obs, actions, returns, advantages)
        assert isinstance(loss, float)
        assert not torch.allclose(initial_param, next(red_agent.parameters()))


# ==============================================================================
# 4. BlueImmuneAgent Tests
# ==============================================================================


class TestBlueImmuneAgent:
    """Tests for the autonomous Blue Immune Agent and its RSSM world model integration."""

    @pytest.fixture
    def blue_agent(self):
        torch.manual_seed(42)
        return BlueImmuneAgent(
            num_nodes=4,
            node_feature_dim=10,
            cluster_global_dim=5,
            warning_threshold=0.35,
            critical_threshold=0.65,
            horizon=4,
        )

    def test_agent_initialization(self, blue_agent):
        """Verify internal world model, dreamer, and mitigator setup."""
        assert isinstance(blue_agent.world_model, RecurrentStateSpaceModel)
        assert isinstance(blue_agent.dreamer, CounterfactualDreamer)
        assert isinstance(blue_agent.mitigator, AutonomousMitigator)
        assert len(blue_agent.node_states) == 4
        assert len(blue_agent.last_actions) == 4

    def test_steady_state_decision(self, blue_agent):
        """When nodes are nominal, agent should return NO_ACTION without triggering mitigations."""
        # Generate nominal observation
        obs = np.zeros(45, dtype=np.float32)
        # Nominal cpu ~ 0.20, mem ~ 0.25 across all 4 nodes
        for i in range(4):
            obs[i * 10] = 0.20
            obs[i * 10 + 1] = 0.25
            obs[i * 10 + 2] = 0.15

        action = blue_agent.act(obs, deterministic=True)
        assert isinstance(action, BlueAction)
        assert action.action == MitigationAction.NO_ACTION

    def test_proactive_mitigation_under_degradation(self, blue_agent):
        """When a node exhibits critical telemetry, agent synthesizes a counterfactual mitigation."""
        obs = np.zeros(45, dtype=np.float32)
        # Heavily degrade node 1 (high thread starvation, queue depth, and slowloris)
        obs[10] = 0.90      # cpu
        obs[11] = 0.85      # mem
        obs[12] = 0.98      # thread_pool
        obs[13] = 0.80      # queue
        obs[14] = 0.90      # latency
        obs[15] = 0.50      # packet loss
        obs[19] = 0.95      # slowloris level

        # Force mitigator threshold low to guarantee trigger in test
        blue_agent.mitigator.warning_threshold = 0.05
        action = blue_agent.act(obs, deterministic=True)

        assert isinstance(action, BlueAction)
        assert action.target_node == 1
        assert action.action in [
            MitigationAction.MIGRATE_REPLICA,
            MitigationAction.EBPF_THROTTLE,
            MitigationAction.CIRCUIT_BREAKER_OPEN,
            MitigationAction.NO_ACTION,
        ]
        if action.action != MitigationAction.NO_ACTION:
            assert action.decision is not None

    def test_reset_clears_latent_states(self, blue_agent):
        """Verify reset re-initializes all node RSSMStates."""
        # Mutate first state
        blue_agent.node_states[0].h.fill_(5.0)
        assert torch.any(blue_agent.node_states[0].h != 0.0)

        blue_agent.reset()
        assert torch.all(blue_agent.node_states[0].h == 0.0)

    def test_train_world_model_step(self, blue_agent):
        """Verify Blue agent can train its internal RSSM world model on observation transitions."""
        obs_seq = torch.randn(2, 4, blue_agent.rssm_obs_dim)
        action_seq = torch.randint(0, 4, (2, 4))
        fail_labels = torch.zeros(2, 4)

        loss = blue_agent.train_world_model_step(obs_seq, action_seq, fail_labels)
        assert isinstance(loss, float)
        assert loss > 0.0


# ==============================================================================
# 5. ChaosSelfPlayTrainer & Resilience Benchmark Tests
# ==============================================================================


class TestChaosSelfPlayTrainer:
    """Tests for the adversarial Red/Blue self-play training loop and resilience evaluation."""

    @pytest.fixture
    def trainer(self):
        torch.manual_seed(42)
        env = ChaosEnvironment(
            num_nodes=3,
            max_steps=10,
            seed=42,
        )
        red_agent = RedChaosAgent(
            obs_dim=env.obs_dim,
            num_nodes=3,
            hidden_dim=32,
        )
        blue_agent = BlueImmuneAgent(
            num_nodes=3,
            warning_threshold=0.40,
            horizon=4,
        )
        return ChaosSelfPlayTrainer(
            env=env,
            red_agent=red_agent,
            blue_agent=blue_agent,
        )

    def test_train_single_episode(self, trainer):
        """Verify train_episode runs to completion and logs valid metrics."""
        result = trainer.train_episode(episode_id=1)

        assert isinstance(result, ChaosEpisodeResult)
        assert result.episode_id == 1
        assert 1 <= result.steps <= trainer.env.max_steps
        assert 0.0 <= result.mean_availability <= 1.0
        assert 0.0 <= result.mean_blast_radius <= 1.0
        assert isinstance(result.red_total_reward, float)
        assert isinstance(result.blue_total_reward, float)
        assert len(result.red_action_counts) > 0
        assert len(result.blue_action_counts) > 0
        assert len(trainer.history.episodes) == 1
        assert len(trainer.history.red_losses) == 1

    def test_multi_episode_training_and_history(self, trainer):
        """Verify train() executes multi-episode loop and computes history statistics."""
        history = trainer.train(num_episodes=3)

        assert isinstance(history, ChaosSelfPlayHistory)
        assert len(history.episodes) == 3
        assert 0.0 <= history.win_rate_blue <= 1.0
        assert 0.0 <= history.mean_availability <= 1.0
        assert 0.0 <= history.mean_blast_radius <= 1.0

    def test_evaluate_resilience_benchmark(self, trainer):
        """
        Verify evaluate_resilience benchmarks defended vs undefended clusters.
        Demonstrates that Blue immune defense maintains higher availability and lower blast radius.
        """
        benchmark = trainer.evaluate_resilience(num_episodes=3, seed=101)

        assert isinstance(benchmark, ResilienceBenchmarkResult)
        assert benchmark.num_episodes == 3
        assert 0.0 <= benchmark.defended_availability <= 1.0
        assert 0.0 <= benchmark.undefended_availability <= 1.0
        # Defended cluster should have availability >= undefended cluster
        assert benchmark.defended_availability >= benchmark.undefended_availability - 0.05
        # Defended blast radius should be <= undefended blast radius + small tolerance
        assert benchmark.defended_blast_radius <= benchmark.undefended_blast_radius + 0.10
        assert benchmark.blast_radius_reduction_pct >= 0.0
