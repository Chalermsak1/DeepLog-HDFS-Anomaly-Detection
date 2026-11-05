"""
DeepLog-Omni Pillar VI: Autonomous Chaos Gym & Adversarial Red/Blue Self-Play
=============================================================================
Reinforcement Learning Environment and Adversarial Self-Play for Distributed Systems.

Components:
1. ChaosFaultType & Action Containers:
   - ChaosFaultType: Latency spikes, packet loss, thread exhaustion, split-brain, slowloris.
   - RedAction & BlueAction: Structured action containers for adversarial failure injection
     and autonomous mitigation.

2. ChaosEnvironment:
   - High-fidelity discrete-time simulator of a distributed microservice cluster with N nodes.
   - Models physical queue dynamics, cascading failure propagation, network partitions,
     and Byzantine slowloris resource pinning.
   - Computes cluster availability, blast radius, p99 latency, and SLA compliance.
   - Standard Gym-style step/reset interfaces with dual-agent rewards.

3. RedChaosAgent:
   - Adversarial policy network (Actor-Critic) targeting vulnerable nodes and injecting
     calibrated failure modes to maximize disruption and cascade blast radius.

4. BlueImmuneAgent:
   - Autonomous defense agent integrated with Pillar IV's Recurrent State Space Model (RSSM),
     CounterfactualDreamer, and AutonomousMitigator.
   - Executes live container migrations, in-kernel eBPF backpressure/TCP clamps,
     and dynamic circuit breaking.

5. ChaosSelfPlayTrainer:
   - Self-play training and evaluation loop optimizing Red and Blue policies.
   - Evaluates resilience frontiers, Pareto trade-offs, and cluster availability under chaos.
"""

from __future__ import annotations

import copy
import enum
import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .world_model import (
    AutonomousMitigator,
    CounterfactualDreamer,
    CounterfactualTrajectory,
    MitigationAction,
    MitigationDecision,
    MitigationUrgency,
    RecurrentStateSpaceModel,
    RemediationEventStream,
    RSSMState,
)


# ==============================================================================
# 1. Enums and Structured Data Containers
# ==============================================================================


class ChaosFaultType(enum.IntEnum):
    """
    Failure injection modes available to the adversarial Red Chaos Agent.
    """
    NONE = 0
    LATENCY_SPIKE = 1
    PACKET_LOSS = 2
    THREAD_EXHAUSTION = 3
    SPLIT_BRAIN = 4
    BYZANTINE_SLOWLORIS = 5

    @property
    def identifier(self) -> str:
        return self.name.lower()

    @property
    def human_name(self) -> str:
        mapping = {
            ChaosFaultType.NONE: "Nominal (No Attack)",
            ChaosFaultType.LATENCY_SPIKE: "Network / Processing Latency Injection",
            ChaosFaultType.PACKET_LOSS: "Network Packet Drop Injection",
            ChaosFaultType.THREAD_EXHAUSTION: "Worker Thread Pool Starvation",
            ChaosFaultType.SPLIT_BRAIN: "Network Partition / Split-Brain",
            ChaosFaultType.BYZANTINE_SLOWLORIS: "Byzantine Slowloris Connection Pinning",
        }
        return mapping[self]


@dataclass
class RedAction:
    """
    Adversarial failure injection directive emitted by RedChaosAgent.

    Attributes
    ----------
    target_node : int
        Index of the target cluster node (0 to num_nodes - 1).
    fault_type : ChaosFaultType
        Type of failure to inject.
    intensity : float
        Severity/magnitude of the fault in [0.0, 1.0].
    duration : int
        Duration in environment steps for which the fault remains active.
    """
    target_node: int = 0
    fault_type: ChaosFaultType = ChaosFaultType.NONE
    intensity: float = 1.0
    duration: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_node": self.target_node,
            "fault_type": self.fault_type.name,
            "intensity": float(self.intensity),
            "duration": int(self.duration),
        }


@dataclass
class BlueAction:
    """
    Pre-emptive defense or remediation directive emitted by BlueImmuneAgent.

    Attributes
    ----------
    action : MitigationAction
        High-level mitigation action (NO_ACTION, MIGRATE_REPLICA, EBPF_THROTTLE, CIRCUIT_BREAKER_OPEN).
    target_node : int
        Target cluster node to apply the mitigation to.
    destination_node : Optional[int]
        Destination node for live replica migration.
    decision : Optional[MitigationDecision]
        Structured counterfactual decision produced by AutonomousMitigator.
    """
    action: MitigationAction = MitigationAction.NO_ACTION
    target_node: int = 0
    destination_node: Optional[int] = None
    decision: Optional[MitigationDecision] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action.name,
            "target_node": self.target_node,
            "destination_node": self.destination_node,
            "has_decision": self.decision is not None,
        }


@dataclass
class ClusterNode:
    """
    State container for an individual node / microservice instance in the cluster.

    Attributes
    ----------
    node_id : int
        Zero-based index of the node.
    name : str
        Human-readable hostname / service identifier.
    cpu_util : float
        CPU utilization fraction in [0.0, 1.0].
    mem_util : float
        Memory utilization fraction in [0.0, 1.0].
    thread_pool_util : float
        Worker thread pool saturation in [0.0, 1.0].
    queue_depth : float
        Pending request queue length normalized (nominal ~0.0 to 10.0, max 100.0).
    p99_latency_ms : float
        99th percentile response latency in milliseconds.
    packet_loss_rate : float
        Inbound/outbound packet drop probability in [0.0, 1.0].
    partition_id : int
        Partition island identifier (0 = primary cluster partition, >0 = isolated island).
    is_circuit_broken : bool
        Whether dynamic circuit breaker is currently open (fast-falling back).
    circuit_breaker_timer : int
        Steps remaining before circuit breaker transitions to half-open / closed.
    is_cordoned : bool
        Whether node is cordoned due to replica migration.
    ebpf_rate_limit : float
        Traffic throttling factor enforced by in-kernel eBPF [0.0, 1.0].
    slowloris_connections : float
        Number of active connections held by slowloris sockets.
    active_injected_fault : ChaosFaultType
        Active failure injected into this node.
    fault_timer : int
        Steps remaining for active injected fault.
    is_healthy : bool
        Derived health status.
    """
    node_id: int
    name: str = "node-0"
    cpu_util: float = 0.20
    mem_util: float = 0.25
    thread_pool_util: float = 0.15
    queue_depth: float = 1.0
    p99_latency_ms: float = 5.0
    packet_loss_rate: float = 0.0
    partition_id: int = 0
    is_circuit_broken: bool = False
    circuit_breaker_timer: int = 0
    is_cordoned: bool = False
    ebpf_rate_limit: float = 1.0
    slowloris_connections: float = 0.0
    active_injected_fault: ChaosFaultType = ChaosFaultType.NONE
    fault_timer: int = 0
    is_healthy: bool = True

    def reset_nominal(self) -> None:
        """Reset node to healthy steady-state baselines."""
        self.cpu_util = 0.20
        self.mem_util = 0.25
        self.thread_pool_util = 0.15
        self.queue_depth = 1.0
        self.p99_latency_ms = 5.0
        self.packet_loss_rate = 0.0
        self.partition_id = 0
        self.is_circuit_broken = False
        self.circuit_breaker_timer = 0
        self.is_cordoned = False
        self.ebpf_rate_limit = 1.0
        self.slowloris_connections = 0.0
        self.active_injected_fault = ChaosFaultType.NONE
        self.fault_timer = 0
        self.is_healthy = True

    def to_feature_vector(self) -> np.ndarray:
        """
        Encode node telemetry into a normalized feature vector for neural policies.
        Length: 10 dimensions.
        """
        return np.array([
            float(self.cpu_util),
            float(self.mem_util),
            float(self.thread_pool_util),
            float(min(1.0, self.queue_depth / 50.0)),
            float(min(1.0, self.p99_latency_ms / 200.0)),
            float(self.packet_loss_rate),
            1.0 if self.partition_id > 0 else 0.0,
            1.0 if self.is_circuit_broken else 0.0,
            1.0 if self.is_cordoned else 0.0,
            float(min(1.0, self.slowloris_connections / 500.0)),
        ], dtype=np.float32)


@dataclass
class ClusterState:
    """
    Snapshot of cluster-wide telemetry and health metrics.
    """
    step_count: int
    num_nodes: int
    availability: float
    blast_radius: float
    cluster_p99_latency_ms: float
    total_requests: float
    successful_requests: float
    sla_violations: float
    cascade_risk: float
    nodes: List[ClusterNode]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_count": self.step_count,
            "num_nodes": self.num_nodes,
            "availability": float(self.availability),
            "blast_radius": float(self.blast_radius),
            "cluster_p99_latency_ms": float(self.cluster_p99_latency_ms),
            "total_requests": float(self.total_requests),
            "successful_requests": float(self.successful_requests),
            "sla_violations": float(self.sla_violations),
            "cascade_risk": float(self.cascade_risk),
            "nodes": [asdict(n) for n in self.nodes],
        }


@dataclass
class ChaosEpisodeResult:
    """Summary metrics of a completed self-play or evaluation episode."""
    episode_id: int
    steps: int
    mean_availability: float
    min_availability: float
    mean_blast_radius: float
    max_blast_radius: float
    mean_p99_latency_ms: float
    sla_violation_rate: float
    red_total_reward: float
    blue_total_reward: float
    red_action_counts: Dict[str, int]
    blue_action_counts: Dict[str, int]
    is_resilient: bool


@dataclass
class ChaosSelfPlayHistory:
    """Historical tracking of self-play training dynamics."""
    episodes: List[ChaosEpisodeResult] = field(default_factory=list)
    red_losses: List[float] = field(default_factory=list)
    blue_losses: List[float] = field(default_factory=list)

    def record_episode(self, ep: ChaosEpisodeResult) -> None:
        self.episodes.append(ep)

    @property
    def win_rate_blue(self) -> float:
        """Percentage of episodes where cluster maintained SLA availability >= 0.95."""
        if not self.episodes:
            return 0.0
        return sum(1 for ep in self.episodes if ep.is_resilient) / len(self.episodes)

    @property
    def mean_availability(self) -> float:
        if not self.episodes:
            return 0.0
        return float(np.mean([ep.mean_availability for ep in self.episodes]))

    @property
    def mean_blast_radius(self) -> float:
        if not self.episodes:
            return 0.0
        return float(np.mean([ep.mean_blast_radius for ep in self.episodes]))


@dataclass
class ResilienceBenchmarkResult:
    """Comparison benchmarks between defended cluster vs undefended cluster under attack."""
    num_episodes: int
    defended_availability: float
    undefended_availability: float
    availability_improvement_pct: float
    defended_blast_radius: float
    undefended_blast_radius: float
    blast_radius_reduction_pct: float
    defended_p99_latency_ms: float
    undefended_p99_latency_ms: float
    defended_resilience_rate: float
    undefended_resilience_rate: float


# ==============================================================================
# 2. ChaosEnvironment
# ==============================================================================


class ChaosEnvironment:
    """
    Autonomous Chaos Gym: High-Fidelity Distributed Cluster State Simulator.

    Simulates an N-node distributed microservice cluster subject to adversarial
    chaos injections:
    - Latency spikes & network delays
    - Packet loss & transmission drops
    - Worker thread pool exhaustion
    - Split-brain network partitions
    - Byzantine slowloris socket pinning

    Supports pre-emptive mitigation interventions:
    - Live replica/container migration
    - In-kernel eBPF backpressure / TCP connection clamp
    - Dynamic circuit breaking

    Follows standard Gym-style reset() and step() interfaces.
    """

    NODE_FEATURE_DIM = 10
    CLUSTER_GLOBAL_FEATURE_DIM = 5

    def __init__(
        self,
        num_nodes: int = 4,
        base_capacity_rps: float = 1000.0,
        nominal_latency_ms: float = 5.0,
        sla_latency_threshold_ms: float = 80.0,
        sla_availability_target: float = 0.95,
        max_steps: int = 30,
        circuit_breaker_recovery_steps: int = 3,
        seed: Optional[int] = None,
    ):
        self.num_nodes = max(2, num_nodes)
        self.base_capacity_rps = float(base_capacity_rps)
        self.nominal_latency_ms = float(nominal_latency_ms)
        self.sla_latency_threshold_ms = float(sla_latency_threshold_ms)
        self.sla_availability_target = float(sla_availability_target)
        self.max_steps = int(max_steps)
        self.circuit_breaker_recovery_steps = int(circuit_breaker_recovery_steps)

        self.rng = np.random.default_rng(seed)

        # Operational costs for Blue mitigation actions
        self.blue_action_costs: Dict[MitigationAction, float] = {
            MitigationAction.NO_ACTION: 0.0,
            MitigationAction.EBPF_THROTTLE: 0.05,
            MitigationAction.CIRCUIT_BREAKER_OPEN: 0.15,
            MitigationAction.MIGRATE_REPLICA: 0.22,
        }

        # Attack budget/cost penalties for Red fault injection
        self.red_fault_costs: Dict[ChaosFaultType, float] = {
            ChaosFaultType.NONE: 0.0,
            ChaosFaultType.LATENCY_SPIKE: 0.05,
            ChaosFaultType.PACKET_LOSS: 0.06,
            ChaosFaultType.THREAD_EXHAUSTION: 0.08,
            ChaosFaultType.SPLIT_BRAIN: 0.12,
            ChaosFaultType.BYZANTINE_SLOWLORIS: 0.10,
        }

        # Observation dimension: (num_nodes * 10) + 5 cluster features
        self.obs_dim = (self.num_nodes * self.NODE_FEATURE_DIM) + self.CLUSTER_GLOBAL_FEATURE_DIM

        # Discrete action dimensions
        # Red: NONE (0) + (num_nodes * 5 fault types)
        self.num_red_actions = 1 + (self.num_nodes * 5)
        # Blue: NO_ACTION (0) + (num_nodes * 3 mitigation types)
        self.num_blue_actions = 1 + (self.num_nodes * 3)

        self.nodes: List[ClusterNode] = []
        self.step_count = 0
        self.last_availability = 1.0
        self.last_blast_radius = 0.0
        self.last_p99_latency_ms = nominal_latency_ms
        self.last_cascade_risk = 0.0

        self.reset(seed=seed)

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset cluster environment to healthy baseline.

        Returns
        -------
        obs : np.ndarray
            Observation feature vector.
        info : Dict[str, Any]
            Cluster diagnostics info.
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.step_count = 0
        self.nodes = [
            ClusterNode(
                node_id=i,
                name=f"worker-node-{i:02d}",
                cpu_util=0.20 + float(self.rng.uniform(-0.02, 0.02)),
                mem_util=0.25 + float(self.rng.uniform(-0.02, 0.02)),
                thread_pool_util=0.15 + float(self.rng.uniform(-0.02, 0.02)),
                queue_depth=1.0,
                p99_latency_ms=self.nominal_latency_ms + float(self.rng.uniform(-0.5, 0.5)),
            )
            for i in range(self.num_nodes)
        ]

        self.last_availability = 1.0
        self.last_blast_radius = 0.0
        self.last_p99_latency_ms = self.nominal_latency_ms
        self.last_cascade_risk = 0.0

        obs = self._build_observation()
        info = self.get_state().to_dict()
        return obs, info

    def step(
        self,
        red_action: Union[RedAction, int, None] = None,
        blue_action: Union[BlueAction, int, None] = None,
    ) -> Tuple[np.ndarray, Dict[str, float], bool, bool, Dict[str, Any]]:
        """
        Execute one discrete simulation step with adversarial Red injection and Blue mitigation.

        Parameters
        ----------
        red_action : RedAction, int, or None
            Adversarial fault action.
        blue_action : BlueAction, int, or None
            Pre-emptive defense action.

        Returns
        -------
        obs : np.ndarray
            Next observation state vector of shape (obs_dim,).
        rewards : Dict[str, float]
            Rewards: {"red": r_red, "blue": r_blue}.
        terminated : bool
            Whether episode ended due to cluster collapse.
        truncated : bool
            Whether max_steps horizon was reached.
        info : Dict[str, Any]
            Cluster diagnostics.
        """
        self.step_count += 1

        parsed_red = self._parse_red_action(red_action)
        parsed_blue = self._parse_blue_action(blue_action)

        # 1. Update active timers and circuit breaker states
        self._step_timers()

        # 2. Execute Blue Mitigations first (pre-emptive mitigation)
        self._apply_blue_mitigation(parsed_blue)

        # 3. Apply Red Chaos Fault Injection
        self._apply_red_fault(parsed_red)

        # 4. Simulate Cluster Traffic, Queuing, and Cascading Propagation
        cluster_metrics = self._simulate_traffic_and_cascades()

        # Update cached cluster-level metrics
        self.last_availability = cluster_metrics["availability"]
        self.last_blast_radius = cluster_metrics["blast_radius"]
        self.last_p99_latency_ms = cluster_metrics["p99_latency"]
        self.last_cascade_risk = cluster_metrics["cascade_risk"]

        # 5. Compute Dual Rewards
        r_red, r_blue = self._compute_rewards(
            availability=self.last_availability,
            blast_radius=self.last_blast_radius,
            cascade_risk=self.last_cascade_risk,
            red_action=parsed_red,
            blue_action=parsed_blue,
        )

        terminated = self.last_availability < 0.10  # Catastrophic cluster collapse
        truncated = self.step_count >= self.max_steps

        obs = self._build_observation()
        info = {
            "step": self.step_count,
            "availability": self.last_availability,
            "blast_radius": self.last_blast_radius,
            "p99_latency_ms": self.last_p99_latency_ms,
            "cascade_risk": self.last_cascade_risk,
            "sla_violations": cluster_metrics["sla_violations"],
            "red_action": parsed_red.to_dict(),
            "blue_action": parsed_blue.to_dict(),
            "terminated": terminated,
            "truncated": truncated,
        }

        return obs, {"red": r_red, "blue": r_blue}, terminated, truncated, info

    def get_state(self) -> ClusterState:
        """Return full snapshot of current cluster state."""
        total_req = self.num_nodes * self.base_capacity_rps
        succ_req = total_req * self.last_availability
        sla_viols = total_req - succ_req

        return ClusterState(
            step_count=self.step_count,
            num_nodes=self.num_nodes,
            availability=self.last_availability,
            blast_radius=self.last_blast_radius,
            cluster_p99_latency_ms=self.last_p99_latency_ms,
            total_requests=total_req,
            successful_requests=succ_req,
            sla_violations=sla_viols,
            cascade_risk=self.last_cascade_risk,
            nodes=copy.deepcopy(self.nodes),
        )

    # --------------------------------------------------------------------------
    # Internal Simulation Mechanics
    # --------------------------------------------------------------------------

    def _step_timers(self) -> None:
        """Advance timers for circuit breakers, migrations, and faults."""
        for node in self.nodes:
            # Advance fault timer
            if node.fault_timer > 0:
                node.fault_timer -= 1
                if node.fault_timer == 0:
                    node.active_injected_fault = ChaosFaultType.NONE
                    node.partition_id = 0

            # Advance circuit breaker timer
            if node.is_circuit_broken:
                if node.circuit_breaker_timer > 0:
                    node.circuit_breaker_timer -= 1
                if node.circuit_breaker_timer <= 0:
                    # Transition back to closed / healthy state
                    node.is_circuit_broken = False
                    node.circuit_breaker_timer = 0

            # Cordone relaxation (restores migration after 1 step)
            if node.is_cordoned:
                node.is_cordoned = False

            # Restore eBPF rate limit smoothly if backpressure relaxed
            if node.ebpf_rate_limit < 1.0:
                node.ebpf_rate_limit = min(1.0, node.ebpf_rate_limit + 0.25)

    def _apply_blue_mitigation(self, action: BlueAction) -> None:
        """Apply pre-emptive mitigation directive."""
        if action.action == MitigationAction.NO_ACTION:
            return

        target_idx = action.target_node % self.num_nodes
        target_node = self.nodes[target_idx]

        if action.action == MitigationAction.MIGRATE_REPLICA:
            # Drain target node and migrate workload to destination
            target_node.is_cordoned = True
            # Destination node defaults to another node
            dest_idx = action.destination_node
            if dest_idx is None or dest_idx == target_idx:
                dest_idx = (target_idx + 1) % self.num_nodes
            dest_node = self.nodes[dest_idx]

            # Relieve queue and thread pool on target node
            target_node.queue_depth = 1.0
            target_node.thread_pool_util = max(0.15, target_node.thread_pool_util * 0.2)
            target_node.cpu_util = max(0.20, target_node.cpu_util * 0.3)
            target_node.slowloris_connections = 0.0

            # Destination handles temporary migration overhead
            dest_node.cpu_util = min(0.80, dest_node.cpu_util + 0.15)
            dest_node.mem_util = min(0.80, dest_node.mem_util + 0.10)

        elif action.action == MitigationAction.EBPF_THROTTLE:
            # Enforce in-kernel eBPF backpressure: terminates slowloris sockets & clamps abusive load
            target_node.ebpf_rate_limit = 0.50
            target_node.slowloris_connections = 0.0
            target_node.thread_pool_util = max(0.20, target_node.thread_pool_util * 0.4)
            target_node.packet_loss_rate = max(0.0, target_node.packet_loss_rate * 0.5)

        elif action.action == MitigationAction.CIRCUIT_BREAKER_OPEN:
            # Trip circuit breaker immediately: sheds failing traffic, isolates cascade
            target_node.is_circuit_broken = True
            target_node.circuit_breaker_timer = self.circuit_breaker_recovery_steps
            target_node.queue_depth = max(1.0, target_node.queue_depth * 0.3)

    def _apply_red_fault(self, action: RedAction) -> None:
        """Inject adversarial chaos fault into target node."""
        if action.fault_type == ChaosFaultType.NONE:
            return

        target_idx = action.target_node % self.num_nodes
        node = self.nodes[target_idx]
        intensity = float(np.clip(action.intensity, 0.1, 1.0))
        duration = max(1, action.duration)

        node.active_injected_fault = action.fault_type
        node.fault_timer = duration

        if action.fault_type == ChaosFaultType.LATENCY_SPIKE:
            # Latency spike: increases p99 latency significantly
            added_delay = 50.0 + (150.0 * intensity)
            node.p99_latency_ms = min(500.0, node.p99_latency_ms + added_delay)
            node.thread_pool_util = min(1.0, node.thread_pool_util + (0.35 * intensity))

        elif action.fault_type == ChaosFaultType.PACKET_LOSS:
            # Network packet drop
            node.packet_loss_rate = min(0.95, 0.25 + (0.65 * intensity))

        elif action.fault_type == ChaosFaultType.THREAD_EXHAUSTION:
            # Thread pool starvation
            node.thread_pool_util = min(1.0, 0.80 + (0.20 * intensity))
            node.queue_depth = min(80.0, node.queue_depth + (35.0 * intensity))
            node.cpu_util = min(1.0, node.cpu_util + (0.40 * intensity))

        elif action.fault_type == ChaosFaultType.SPLIT_BRAIN:
            # Network partition: node partitioned into isolated partition group 1
            node.partition_id = 1

        elif action.fault_type == ChaosFaultType.BYZANTINE_SLOWLORIS:
            # Byzantine slowloris: pins sockets and threads while CPU remains deceptively low
            node.slowloris_connections = 400.0 * intensity
            node.thread_pool_util = min(1.0, 0.85 + (0.15 * intensity))
            node.p99_latency_ms = min(400.0, node.p99_latency_ms + (80.0 * intensity))
            # Notice CPU does not spike (stealthy attack)
            node.cpu_util = min(0.40, node.cpu_util + 0.05)

    def _simulate_traffic_and_cascades(self) -> Dict[str, float]:
        """
        Simulate request routing, queuing theory dynamics, and cascading failure propagation.
        """
        total_cluster_requests = self.num_nodes * self.base_capacity_rps
        inbound_per_node = total_cluster_requests / self.num_nodes

        # Determine available / active capacity
        active_nodes = [n for n in self.nodes if not n.is_circuit_broken and not n.is_cordoned]
        num_active = len(active_nodes)

        # Quorum calculation for split-brain
        # Partition 0 is primary. If partition 0 has strictly fewer than (N // 2 + 1) nodes, quorum lost
        quorum_threshold = (self.num_nodes // 2) + 1
        partition_0_count = sum(1 for n in self.nodes if n.partition_id == 0)
        has_quorum = partition_0_count >= quorum_threshold

        successful_requests = 0.0
        sla_violations = 0.0
        node_latencies: List[float] = []
        degraded_node_count = 0

        # Traffic redistribution: if nodes are circuit broken or cordoned, load spills over
        if num_active > 0:
            assigned_rps_per_active = total_cluster_requests / num_active
        else:
            assigned_rps_per_active = 0.0

        for node in self.nodes:
            # 1. Natural drift / recovery if no active fault
            if node.active_injected_fault == ChaosFaultType.NONE and not node.is_circuit_broken:
                node.p99_latency_ms = max(
                    self.nominal_latency_ms,
                    (node.p99_latency_ms * 0.70) + (self.nominal_latency_ms * 0.30),
                )
                node.packet_loss_rate = max(0.0, node.packet_loss_rate * 0.50)
                node.queue_depth = max(1.0, (node.queue_depth * 0.70) + 0.30)
                node.cpu_util = max(0.20, (node.cpu_util * 0.75) + 0.05)
                node.thread_pool_util = max(0.15, (node.thread_pool_util * 0.75) + 0.04)
                node.slowloris_connections = max(0.0, node.slowloris_connections * 0.50)

            # 2. If node is circuit-broken
            if node.is_circuit_broken:
                # Fast fallback: 70% of requests shed or fail fast, 30% cached/graceful
                successful_requests += inbound_per_node * 0.30
                sla_violations += inbound_per_node * 0.70
                node_latencies.append(self.nominal_latency_ms * 1.5)
                degraded_node_count += 1
                node.is_healthy = False
                continue

            # 3. If node is cordoned (migrating)
            if node.is_cordoned:
                # Workload migrated away; near 100% success handled elsewhere
                successful_requests += inbound_per_node * 0.95
                sla_violations += inbound_per_node * 0.05
                node_latencies.append(self.nominal_latency_ms * 2.0)
                node.is_healthy = True
                continue

            # 4. Active node receiving load
            effective_rps = assigned_rps_per_active * node.ebpf_rate_limit
            load_factor = effective_rps / self.base_capacity_rps

            # Cascade dynamics: when load_factor exceeds 1.0, queuing explosion occurs
            if load_factor > 1.0:
                overload = load_factor - 1.0
                node.queue_depth = min(100.0, node.queue_depth + (15.0 * overload))
                node.cpu_util = min(1.0, node.cpu_util + (0.25 * overload))
                node.thread_pool_util = min(1.0, node.thread_pool_util + (0.30 * overload))
                # Non-linear latency expansion (M/M/1 queuing approximation: rho / (1 - rho))
                rho = min(0.95, node.thread_pool_util)
                node.p99_latency_ms += 10.0 * (rho / max(0.05, 1.0 - rho))

            # 5. Split-brain partition handling
            if node.partition_id > 0 or not has_quorum:
                # Node in isolated partition or cluster lost quorum: cross-node consistency fails
                partition_drop_rate = 0.85 if node.partition_id > 0 else 0.50
            else:
                partition_drop_rate = 0.0

            # 6. Slowloris impact on concurrency
            if node.slowloris_connections > 50:
                slowloris_drop_factor = min(0.80, node.slowloris_connections / 500.0)
            else:
                slowloris_drop_factor = 0.0

            # 7. Overall success probability for requests arriving at this node
            queue_drop_rate = max(0.0, (node.queue_depth - 40.0) / 60.0) if node.queue_depth > 40.0 else 0.0
            total_drop_rate = min(
                1.0,
                node.packet_loss_rate + partition_drop_rate + queue_drop_rate + slowloris_drop_factor,
            )

            # SLA compliance: latency must not exceed SLA threshold
            meets_sla = node.p99_latency_ms <= self.sla_latency_threshold_ms

            node_success = effective_rps * (1.0 - total_drop_rate)
            if not meets_sla:
                # Latency SLA violation penalty
                latency_degradation = min(
                    1.0,
                    (node.p99_latency_ms - self.sla_latency_threshold_ms) / 100.0,
                )
                node_success *= max(0.0, 1.0 - (0.75 * latency_degradation))

            node_viol = effective_rps - node_success
            successful_requests += node_success
            sla_violations += node_viol

            node_latencies.append(node.p99_latency_ms)

            # Node health evaluation
            is_degraded = (
                (node_success / max(1.0, effective_rps) < 0.85)
                or (node.p99_latency_ms > self.sla_latency_threshold_ms)
                or (node.thread_pool_util > 0.85)
                or (node.queue_depth > 30.0)
            )
            node.is_healthy = not is_degraded
            if is_degraded:
                degraded_node_count += 1

        # Aggregate cluster metrics
        cluster_availability = float(np.clip(successful_requests / max(1.0, total_cluster_requests), 0.0, 1.0))
        cluster_blast_radius = float(degraded_node_count / float(self.num_nodes))
        cluster_p99 = float(np.max(node_latencies)) if node_latencies else self.nominal_latency_ms

        # Cascade risk metric [0.0, 1.0]
        mean_thread_util = float(np.mean([n.thread_pool_util for n in self.nodes]))
        cascade_risk = float(np.clip(
            (0.50 * cluster_blast_radius) + (0.50 * mean_thread_util),
            0.0,
            1.0,
        ))

        return {
            "availability": cluster_availability,
            "blast_radius": cluster_blast_radius,
            "p99_latency": cluster_p99,
            "sla_violations": sla_violations,
            "cascade_risk": cascade_risk,
        }

    def _compute_rewards(
        self,
        availability: float,
        blast_radius: float,
        cascade_risk: float,
        red_action: RedAction,
        blue_action: BlueAction,
    ) -> Tuple[float, float]:
        """
        Compute dual reinforcement learning rewards for Red and Blue.

        Red Objective: Maximize SLA violations & blast radius, minimize attack budget cost.
        Blue Objective: Maximize cluster availability, minimize blast radius and operational disruption cost.
        """
        # Blue Reward:
        # Reward maintaining availability near 1.0
        # Penalize blast radius and cascade risk
        # Penalize mitigation operational disruption costs
        blue_cost = self.blue_action_costs.get(blue_action.action, 0.0)
        availability_reward = (availability - 0.70) * 3.0  # Positive above 70%, strong positive at 99%+
        blast_penalty = -2.0 * blast_radius
        cascade_penalty = -1.0 * cascade_risk
        r_blue = availability_reward + blast_penalty + cascade_penalty - blue_cost

        # Red Reward:
        # Reward pushing availability down (SLA degradation)
        # Reward expanding blast radius and triggering cascades
        # Penalize attack budget cost
        red_cost = self.red_fault_costs.get(red_action.fault_type, 0.0) * red_action.intensity
        impact_reward = (1.0 - availability) * 3.5
        red_blast_reward = 2.0 * blast_radius
        r_red = impact_reward + red_blast_reward - red_cost

        return float(r_red), float(r_blue)

    def _build_observation(self) -> np.ndarray:
        """
        Construct flat numpy array observation encoding:
        - Per-node features: shape (num_nodes * 10)
        - Cluster-global features: shape (5,)
        Total shape: (num_nodes * 10 + 5,)
        """
        node_features = [node.to_feature_vector() for node in self.nodes]
        flat_nodes = np.concatenate(node_features, axis=0)

        # Cluster-global features
        global_features = np.array([
            float(self.last_availability),
            float(self.last_blast_radius),
            float(min(1.0, self.last_p99_latency_ms / 200.0)),
            float(self.last_cascade_risk),
            float(self.step_count / float(self.max_steps)),
        ], dtype=np.float32)

        return np.concatenate([flat_nodes, global_features], axis=0)

    def _parse_red_action(self, action: Union[RedAction, int, None]) -> RedAction:
        """Parse discrete action index or RedAction instance."""
        if action is None:
            return RedAction(target_node=0, fault_type=ChaosFaultType.NONE, intensity=0.0)
        if isinstance(action, RedAction):
            return action
        if isinstance(action, (int, np.integer)):
            act_idx = int(action)
            if act_idx == 0:
                return RedAction(target_node=0, fault_type=ChaosFaultType.NONE, intensity=0.0)
            # Map index 1..num_red_actions-1 to (target_node, fault_type)
            adjusted = act_idx - 1
            num_faults = 5  # LATENCY_SPIKE(1) to BYZANTINE_SLOWLORIS(5)
            target_node = (adjusted // num_faults) % self.num_nodes
            fault_type_val = (adjusted % num_faults) + 1
            return RedAction(
                target_node=target_node,
                fault_type=ChaosFaultType(fault_type_val),
                intensity=1.0,
            )
        raise TypeError(f"Unsupported red_action type: {type(action)}")

    def _parse_blue_action(self, action: Union[BlueAction, int, None]) -> BlueAction:
        """Parse discrete action index or BlueAction instance."""
        if action is None:
            return BlueAction(action=MitigationAction.NO_ACTION, target_node=0)
        if isinstance(action, BlueAction):
            return action
        if isinstance(action, (int, np.integer)):
            act_idx = int(action)
            if act_idx == 0:
                return BlueAction(action=MitigationAction.NO_ACTION, target_node=0)
            # Map index 1..num_blue_actions-1 to (target_node, mitigation_action)
            adjusted = act_idx - 1
            num_mitigations = 3  # MIGRATE(1), EBPF(2), CIRCUIT_BREAKER(3)
            target_node = (adjusted // num_mitigations) % self.num_nodes
            action_val = (adjusted % num_mitigations) + 1
            return BlueAction(
                action=MitigationAction(action_val),
                target_node=target_node,
            )
        raise TypeError(f"Unsupported blue_action type: {type(action)}")


# ==============================================================================
# 3. RedChaosAgent
# ==============================================================================


class RedChaosAgent(nn.Module):
    """
    Adversarial Red Chaos Agent.

    Neural Actor-Critic policy targeting cluster nodes and selecting optimal
    failure injection vectors to maximize cascade blast radius and SLA violations.
    """

    def __init__(
        self,
        obs_dim: int,
        num_nodes: int = 4,
        hidden_dim: int = 64,
        learning_rate: float = 1e-3,
        entropy_coef: float = 0.01,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.entropy_coef = entropy_coef

        # Action space: 0 = NO_OP, 1..(num_nodes * 5) = target_node + fault_type
        self.num_actions = 1 + (self.num_nodes * 5)

        # Shared representation torso
        self.torso = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        # Policy head (Actor)
        self.actor = nn.Linear(hidden_dim, self.num_actions)

        # Intensity head (Continuous severity [0.2, 1.0])
        self.intensity_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        # Value head (Critic)
        self.critic = nn.Linear(hidden_dim, 1)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute action logits, intensity, and state value.

        Parameters
        ----------
        obs : torch.Tensor
            Observation tensor of shape (batch_size, obs_dim).

        Returns
        -------
        logits : torch.Tensor
            Action logits of shape (batch_size, num_actions).
        intensity : torch.Tensor
            Fault severity in [0.2, 1.0] of shape (batch_size, 1).
        value : torch.Tensor
            Estimated state value of shape (batch_size, 1).
        """
        features = self.torso(obs)
        logits = self.actor(features)
        raw_intensity = self.intensity_head(features)
        intensity = 0.2 + (0.8 * raw_intensity)
        value = self.critic(features)
        return logits, intensity, value

    def act(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        deterministic: bool = False,
    ) -> RedAction:
        """
        Sample an adversarial failure injection action given the current observation.
        """
        self.eval()
        with torch.no_grad():
            if isinstance(obs, np.ndarray):
                obs_t = torch.from_numpy(obs).float().unsqueeze(0)
            else:
                obs_t = obs if obs.ndim == 2 else obs.unsqueeze(0)

            logits, intensity_t, _ = self.forward(obs_t)
            probs = F.softmax(logits, dim=-1)

            if deterministic:
                action_idx = int(torch.argmax(probs, dim=-1).item())
            else:
                dist = torch.distributions.Categorical(probs=probs)
                action_idx = int(dist.sample().item())

            intensity_val = float(intensity_t.squeeze().item())

        if action_idx == 0:
            return RedAction(target_node=0, fault_type=ChaosFaultType.NONE, intensity=0.0)

        adjusted = action_idx - 1
        num_faults = 5
        target_node = (adjusted // num_faults) % self.num_nodes
        fault_val = (adjusted % num_faults) + 1

        return RedAction(
            target_node=target_node,
            fault_type=ChaosFaultType(fault_val),
            intensity=intensity_val,
        )

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate log probabilities, entropy, and state values for PPO/A2C updates.
        """
        logits, _, values = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy, values.squeeze(-1)

    def train_step(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        returns: torch.Tensor,
        advantages: torch.Tensor,
    ) -> float:
        """
        Execute one policy gradient update step.
        """
        self.train()
        self.optimizer.zero_grad()

        log_probs, entropy, values = self.evaluate_actions(obs, actions)

        # Policy gradient loss
        policy_loss = -(log_probs * advantages.detach()).mean()

        # Value MSE loss
        value_loss = F.mse_loss(values, returns)

        # Entropy regularization bonus
        entropy_loss = -self.entropy_coef * entropy.mean()

        total_loss = policy_loss + (0.5 * value_loss) + entropy_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.optimizer.step()

        return float(total_loss.item())


# ==============================================================================
# 4. BlueImmuneAgent
# ==============================================================================


class BlueImmuneAgent:
    """
    Autonomous Blue Immune Agent.

    Pre-emptive defense policy integrated with Pillar IV's Recurrent State Space
    Model (RSSM), CounterfactualDreamer, and AutonomousMitigator.

    Features:
    - Encodes cluster telemetry into continuous latent representations.
    - Imagines counterfactual futures under live migration, eBPF backpressure,
      and dynamic circuit breaking.
    - Selects mitigations maximizing net utility before cascades materialize.
    """

    def __init__(
        self,
        num_nodes: int = 4,
        node_feature_dim: int = 10,
        cluster_global_dim: int = 5,
        world_model: Optional[RecurrentStateSpaceModel] = None,
        dreamer: Optional[CounterfactualDreamer] = None,
        mitigator: Optional[AutonomousMitigator] = None,
        warning_threshold: float = 0.35,
        critical_threshold: float = 0.65,
        horizon: int = 6,
    ):
        self.num_nodes = num_nodes
        self.node_feature_dim = node_feature_dim
        self.cluster_global_dim = cluster_global_dim
        self.obs_dim = (num_nodes * node_feature_dim) + cluster_global_dim

        # Per-node observation fed to RSSM: [cpu, mem, thread_pool, queue, latency, packet_loss, partition, slowloris]
        self.rssm_obs_dim = 8

        # Initialize RSSM World Model if not provided
        if world_model is None:
            self.world_model = RecurrentStateSpaceModel(
                obs_dim=self.rssm_obs_dim,
                action_dim=4,  # 4 MitigationAction types
                action_embed_dim=8,
                hidden_dim=32,
                latent_dim=16,
                encoder_hidden_dim=32,
                decoder_hidden_dim=32,
            )
        else:
            self.world_model = world_model

        # Counterfactual Dreamer
        if dreamer is None:
            self.dreamer = CounterfactualDreamer(
                world_model=self.world_model,
                default_horizon=horizon,
                critical_risk_threshold=critical_threshold,
            )
        else:
            self.dreamer = dreamer

        # Autonomous Mitigator
        if mitigator is None:
            self.mitigator = AutonomousMitigator(
                dreamer=self.dreamer,
                warning_threshold=warning_threshold,
                critical_threshold=critical_threshold,
                safe_target_risk=0.20,
                horizon=horizon,
                event_stream=RemediationEventStream(max_size=100),
            )
        else:
            self.mitigator = mitigator

        # Track latent states per node
        self.node_states: List[RSSMState] = [
            self.world_model.init_state(batch_size=1) for _ in range(self.num_nodes)
        ]
        self.last_actions: List[MitigationAction] = [
            MitigationAction.NO_ACTION for _ in range(self.num_nodes)
        ]

        self.optimizer = torch.optim.Adam(self.world_model.parameters(), lr=1e-3)

    def reset(self) -> None:
        """Reset internal world model latent states across all nodes."""
        self.node_states = [
            self.world_model.init_state(batch_size=1) for _ in range(self.num_nodes)
        ]
        self.last_actions = [
            MitigationAction.NO_ACTION for _ in range(self.num_nodes)
        ]

    def act(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        deterministic: bool = True,
    ) -> BlueAction:
        """
        Evaluate cluster observation and synthesize optimal pre-emptive mitigation.

        Parameters
        ----------
        obs : np.ndarray or torch.Tensor
            Cluster observation vector.
        deterministic : bool
            Deterministic selection.

        Returns
        -------
        BlueAction
            Synthesized defense directive.
        """
        if isinstance(obs, torch.Tensor):
            obs_np = obs.detach().cpu().numpy().flatten()
        else:
            obs_np = obs.flatten()

        # 1. Update RSSM posterior states for each node
        per_node_risks: List[float] = []
        for i in range(self.num_nodes):
            start_idx = i * self.node_feature_dim
            node_feat = obs_np[start_idx : start_idx + self.node_feature_dim]
            # Extract 8-dim subset for RSSM: [cpu, mem, thread, queue, latency, loss, partition, slowloris]
            rssm_feat = np.array([
                node_feat[0],  # cpu
                node_feat[1],  # mem
                node_feat[2],  # thread_pool
                node_feat[3],  # queue
                node_feat[4],  # latency
                node_feat[5],  # loss
                node_feat[6],  # partition
                node_feat[9],  # slowloris
            ], dtype=np.float32)

            obs_t = torch.from_numpy(rssm_feat).unsqueeze(0)
            act_t = torch.tensor([self.last_actions[i].value], dtype=torch.long)

            with torch.no_grad():
                next_state = self.world_model.step_posterior(
                    obs=obs_t,
                    action=act_t,
                    prev_state=self.node_states[i],
                    sample=not deterministic,
                )
                self.node_states[i] = next_state
                risk = float(self.world_model.predict_risk(next_state).squeeze().item())

            # Real-time observable telemetry degradation indicator [0.0, 1.0]
            telemetry_degradation = float(np.mean([
                node_feat[2],  # thread_pool
                node_feat[3],  # queue
                node_feat[4],  # latency
                node_feat[5],  # loss
                node_feat[9],  # slowloris
            ]))
            # Combine latent world model risk with observable telemetry degradation
            node_risk = (0.5 * risk) + (0.5 * telemetry_degradation)
            per_node_risks.append(node_risk)

        # 2. Identify highest risk node in cluster
        most_vulnerable_node = int(np.argmax(per_node_risks))
        max_risk = per_node_risks[most_vulnerable_node]

        # 3. If highest risk is nominal, maintain steady state (NO_ACTION)
        if max_risk < self.mitigator.warning_threshold:
            self.last_actions[most_vulnerable_node] = MitigationAction.NO_ACTION
            return BlueAction(action=MitigationAction.NO_ACTION, target_node=most_vulnerable_node)

        # 4. Use AutonomousMitigator to dream counterfactuals and decide intervention
        target_state = self.node_states[most_vulnerable_node]
        target_resource = f"worker-node-{most_vulnerable_node:02d}"

        # Heuristic context metadata for mitigator
        dest_node_idx = (most_vulnerable_node + 1) % self.num_nodes
        decision = self.mitigator.evaluate_and_mitigate(
            initial_state=target_state,
            target_resource=target_resource,
            candidate_actions=[
                MitigationAction.NO_ACTION,
                MitigationAction.EBPF_THROTTLE,
                MitigationAction.CIRCUIT_BREAKER_OPEN,
                MitigationAction.MIGRATE_REPLICA,
            ],
            metadata={
                "destination_node": f"worker-node-{dest_node_idx:02d}",
                "grace_period_sec": 5,
            },
        )

        chosen_action = decision.action
        # If the untrained world model selected NO_ACTION despite high risk, apply informed heuristic
        if chosen_action == MitigationAction.NO_ACTION and max_risk >= self.mitigator.warning_threshold:
            start_idx = most_vulnerable_node * self.node_feature_dim
            target_feat = obs_np[start_idx : start_idx + self.node_feature_dim]
            if target_feat[9] > 0.3:  # slowloris connections
                chosen_action = MitigationAction.EBPF_THROTTLE
            elif target_feat[2] > 0.7 or target_feat[3] > 0.4:  # thread exhaustion or queue explosion
                chosen_action = MitigationAction.MIGRATE_REPLICA
            elif target_feat[5] > 0.4 or target_feat[6] > 0.0:  # packet loss or partition
                chosen_action = MitigationAction.CIRCUIT_BREAKER_OPEN
            else:
                chosen_action = MitigationAction.EBPF_THROTTLE

        self.last_actions[most_vulnerable_node] = chosen_action
        return BlueAction(
            action=chosen_action,
            target_node=most_vulnerable_node,
            destination_node=dest_node_idx if chosen_action == MitigationAction.MIGRATE_REPLICA else None,
            decision=decision,
        )

    def train_world_model_step(
        self,
        obs_seq: torch.Tensor,
        action_seq: torch.Tensor,
        fail_labels: Optional[torch.Tensor] = None,
    ) -> float:
        """
        Train RSSM world model on observed transitions.
        """
        self.world_model.train()
        self.optimizer.zero_grad()

        if fail_labels is not None and fail_labels.ndim == 2:
            target_risks = fail_labels.unsqueeze(-1)
        else:
            target_risks = fail_labels

        losses = self.world_model(
            obs_seq=obs_seq,
            act_seq=action_seq,
            target_risks=target_risks,
        )

        total_loss = losses["total_loss"]
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return float(total_loss.item())


# ==============================================================================
# 5. ChaosSelfPlayTrainer
# ==============================================================================


class ChaosSelfPlayTrainer:
    """
    Self-Play Training and Evaluation Loop for Autonomous Chaos Engineering.

    Coordinates adversarial self-play between:
    - RedChaosAgent (maximizing cluster degradation & cascade blast radius)
    - BlueImmuneAgent (minimizing downtime, SLA violations, & operational disruption)

    Computes availability rewards, blast radius penalties, and evaluates cluster resilience.
    """

    def __init__(
        self,
        env: ChaosEnvironment,
        red_agent: RedChaosAgent,
        blue_agent: BlueImmuneAgent,
        gamma: float = 0.95,
        gae_lambda: float = 0.95,
    ):
        self.env = env
        self.red_agent = red_agent
        self.blue_agent = blue_agent
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.history = ChaosSelfPlayHistory()

    def train_episode(self, episode_id: int = 0) -> ChaosEpisodeResult:
        """
        Execute one full training episode of adversarial self-play.
        """
        obs, _ = self.env.reset()
        self.blue_agent.reset()

        obs_list: List[np.ndarray] = []
        red_actions_list: List[int] = []
        red_rewards_list: List[float] = []
        red_values_list: List[float] = []
        blue_rewards_list: List[float] = []

        availabilities: List[float] = []
        blast_radii: List[float] = []
        p99_latencies: List[float] = []
        sla_violations: List[float] = []

        red_action_counts: Dict[str, int] = {}
        blue_action_counts: Dict[str, int] = {}

        done = False
        step = 0

        while not done:
            step += 1
            obs_list.append(obs)

            # Red Agent acts
            red_act = self.red_agent.act(obs, deterministic=False)
            fault_name = red_act.fault_type.name
            red_action_counts[fault_name] = red_action_counts.get(fault_name, 0) + 1

            # Convert to discrete index for policy update
            if red_act.fault_type == ChaosFaultType.NONE:
                red_action_idx = 0
            else:
                red_action_idx = 1 + (red_act.target_node * 5) + (red_act.fault_type.value - 1)
            red_actions_list.append(red_action_idx)

            # Compute value estimate for Red
            with torch.no_grad():
                obs_t = torch.from_numpy(obs).float().unsqueeze(0)
                _, _, val_t = self.red_agent(obs_t)
                red_values_list.append(float(val_t.squeeze().item()))

            # Blue Agent acts
            blue_act = self.blue_agent.act(obs, deterministic=False)
            mit_name = blue_act.action.name
            blue_action_counts[mit_name] = blue_action_counts.get(mit_name, 0) + 1

            # Step environment
            next_obs, rewards, terminated, truncated, info = self.env.step(red_act, blue_act)

            red_rewards_list.append(rewards["red"])
            blue_rewards_list.append(rewards["blue"])

            availabilities.append(info["availability"])
            blast_radii.append(info["blast_radius"])
            p99_latencies.append(info["p99_latency_ms"])
            sla_violations.append(info["sla_violations"])

            obs = next_obs
            done = terminated or truncated

        # Update Red policy via policy gradient
        red_loss = self._update_red_policy(
            obs_list=obs_list,
            actions_list=red_actions_list,
            rewards_list=red_rewards_list,
            values_list=red_values_list,
        )
        self.history.red_losses.append(red_loss)

        # Compile episode metrics
        mean_avail = float(np.mean(availabilities))
        min_avail = float(np.min(availabilities))
        mean_blast = float(np.mean(blast_radii))
        max_blast = float(np.max(blast_radii))
        mean_lat = float(np.mean(p99_latencies))
        total_sla_viols = float(np.sum(sla_violations))
        total_reqs = float(len(availabilities) * self.env.num_nodes * self.env.base_capacity_rps)
        sla_violation_rate = total_sla_viols / max(1.0, total_reqs)

        is_resilient = mean_avail >= self.env.sla_availability_target

        result = ChaosEpisodeResult(
            episode_id=episode_id,
            steps=step,
            mean_availability=mean_avail,
            min_availability=min_avail,
            mean_blast_radius=mean_blast,
            max_blast_radius=max_blast,
            mean_p99_latency_ms=mean_lat,
            sla_violation_rate=sla_violation_rate,
            red_total_reward=float(np.sum(red_rewards_list)),
            blue_total_reward=float(np.sum(blue_rewards_list)),
            red_action_counts=red_action_counts,
            blue_action_counts=blue_action_counts,
            is_resilient=is_resilient,
        )

        self.history.record_episode(result)
        return result

    def train(self, num_episodes: int = 10) -> ChaosSelfPlayHistory:
        """
        Run multi-episode adversarial self-play training loop.
        """
        for ep in range(num_episodes):
            self.train_episode(episode_id=ep)
        return self.history

    def evaluate_resilience(
        self,
        num_episodes: int = 5,
        seed: int = 42,
    ) -> ResilienceBenchmarkResult:
        """
        Benchmark cluster resilience under adversarial Red attack:
        Compares Defended Cluster (with BlueImmuneAgent) vs Undefended Cluster (NO_ACTION baseline).
        """
        defended_avails: List[float] = []
        defended_blasts: List[float] = []
        defended_lats: List[float] = []
        defended_resilient_count = 0

        undefended_avails: List[float] = []
        undefended_blasts: List[float] = []
        undefended_lats: List[float] = []
        undefended_resilient_count = 0

        # 1. Defended Evaluation Runs
        for ep in range(num_episodes):
            obs, _ = self.env.reset(seed=seed + ep)
            self.blue_agent.reset()
            ep_avails: List[float] = []
            ep_blasts: List[float] = []
            ep_lats: List[float] = []
            done = False

            while not done:
                red_act = self.red_agent.act(obs, deterministic=True)
                blue_act = self.blue_agent.act(obs, deterministic=True)
                obs, _, term, trunc, info = self.env.step(red_act, blue_act)

                ep_avails.append(info["availability"])
                ep_blasts.append(info["blast_radius"])
                ep_lats.append(info["p99_latency_ms"])
                done = term or trunc

            mean_av = float(np.mean(ep_avails))
            defended_avails.append(mean_av)
            defended_blasts.append(float(np.mean(ep_blasts)))
            defended_lats.append(float(np.mean(ep_lats)))
            if mean_av >= self.env.sla_availability_target:
                defended_resilient_count += 1

        # 2. Undefended Evaluation Runs (Blue does NO_ACTION)
        no_action = BlueAction(action=MitigationAction.NO_ACTION, target_node=0)
        for ep in range(num_episodes):
            obs, _ = self.env.reset(seed=seed + ep)
            ep_avails = []
            ep_blasts = []
            ep_lats = []
            done = False

            while not done:
                red_act = self.red_agent.act(obs, deterministic=True)
                obs, _, term, trunc, info = self.env.step(red_act, no_action)

                ep_avails.append(info["availability"])
                ep_blasts.append(info["blast_radius"])
                ep_lats.append(info["p99_latency_ms"])
                done = term or trunc

            mean_av = float(np.mean(ep_avails))
            undefended_avails.append(mean_av)
            undefended_blasts.append(float(np.mean(ep_blasts)))
            undefended_lats.append(float(np.mean(ep_lats)))
            if mean_av >= self.env.sla_availability_target:
                undefended_resilient_count += 1

        def_mean_av = float(np.mean(defended_avails))
        undef_mean_av = float(np.mean(undefended_avails))
        av_improvement = max(0.0, (def_mean_av - undef_mean_av) / max(1e-4, undef_mean_av)) * 100.0

        def_mean_blast = float(np.mean(defended_blasts))
        undef_mean_blast = float(np.mean(undefended_blasts))
        blast_reduction = max(0.0, (undef_mean_blast - def_mean_blast) / max(1e-4, undef_mean_blast)) * 100.0

        return ResilienceBenchmarkResult(
            num_episodes=num_episodes,
            defended_availability=def_mean_av,
            undefended_availability=undef_mean_av,
            availability_improvement_pct=av_improvement,
            defended_blast_radius=def_mean_blast,
            undefended_blast_radius=undef_mean_blast,
            blast_radius_reduction_pct=blast_reduction,
            defended_p99_latency_ms=float(np.mean(defended_lats)),
            undefended_p99_latency_ms=float(np.mean(undefended_lats)),
            defended_resilience_rate=defended_resilient_count / float(num_episodes),
            undefended_resilience_rate=undefended_resilient_count / float(num_episodes),
        )

    # --------------------------------------------------------------------------
    # Policy Optimization Helpers
    # --------------------------------------------------------------------------

    def _update_red_policy(
        self,
        obs_list: List[np.ndarray],
        actions_list: List[int],
        rewards_list: List[float],
        values_list: List[float],
    ) -> float:
        """
        Compute Generalized Advantage Estimation (GAE) and update RedChaosAgent policy.
        """
        if not rewards_list:
            return 0.0

        T = len(rewards_list)
        returns = np.zeros(T, dtype=np.float32)
        advantages = np.zeros(T, dtype=np.float32)

        # Discounted Monte Carlo / GAE computation
        last_gae = 0.0
        for t in reversed(range(T)):
            next_val = values_list[t + 1] if t + 1 < T else 0.0
            delta = rewards_list[t] + (self.gamma * next_val) - values_list[t]
            last_gae = delta + (self.gamma * self.gae_lambda * last_gae)
            advantages[t] = last_gae
            returns[t] = advantages[t] + values_list[t]

        # Standardize advantages
        if T > 1 and np.std(advantages) > 1e-6:
            advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        obs_t = torch.from_numpy(np.array(obs_list, dtype=np.float32))
        act_t = torch.tensor(actions_list, dtype=torch.long)
        ret_t = torch.from_numpy(returns)
        adv_t = torch.from_numpy(advantages)

        return self.red_agent.train_step(
            obs=obs_t,
            actions=act_t,
            returns=ret_t,
            advantages=adv_t,
        )
