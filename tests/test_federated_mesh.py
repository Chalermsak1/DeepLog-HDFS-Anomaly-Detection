"""Unit tests for Frontier 5: Federated Zero-Knowledge Threat Sharing Mesh.

Tests cover:
1. HDCThreatSignature: 10,000-bit cryptographic threat vectors with Ed25519 and SHA256 signatures.
2. DifferentialPrivacyMasker: Calibrated Bernoulli bit-flipping noise, theoretical bounds,
   unbiased consensus reconstruction, and privacy budget tracking.
3. ZeroKnowledgeVerifier: Schnorr NIZK proof of cluster telemetry witness,
   statistical density/entropy validation, anti-consensus poisoning detection,
   and node reputation / quarantine tracking.
4. HolographicDefenseMemoryBank: Sub-microsecond associative matching, bundling, and mode querying.
5. FederatedThreatMeshNode: P2P broadcast, verification, gossip sync, live log scanning,
   and Byzantine mesh resilience against poisoned adversarial vectors.
"""

from __future__ import annotations

import copy
import json
import secrets
import time
import numpy as np
import pytest

from deeplog.hdc import HyperVector, HyperdimensionalEncoder
from deeplog.federated_mesh import (
    HDCThreatSignature,
    DifferentialPrivacyMasker,
    FederatedThreatMeshNode,
    ZeroKnowledgeVerifier,
    HolographicDefenseMemoryBank,
    FederatedMeshError,
    InvalidSignatureError,
    PoisoningDetectedError,
    PrivacyBudgetExceededError,
    ed25519_keypair,
    ed25519_sign,
    ed25519_verify,
)


class TestEd25519CryptoEngine:
    """Tests for the pure-Python RFC 8032 Ed25519 engine."""

    def test_keypair_generation(self):
        seed, pk = ed25519_keypair()
        assert len(seed) == 32
        assert len(pk) == 32
        assert isinstance(seed, bytes)
        assert isinstance(pk, bytes)

    def test_deterministic_keypair_from_seed(self):
        seed = bytes(range(32))
        _, pk1 = ed25519_keypair(seed)
        _, pk2 = ed25519_keypair(seed)
        assert pk1 == pk2

    def test_sign_and_verify_valid(self):
        seed, pk = ed25519_keypair()
        msg = b"DeepLog-Omni Threat Hypervector 10000-bit payload"
        sig = ed25519_sign(msg, seed, pk)
        assert len(sig) == 64
        assert ed25519_verify(sig, msg, pk) is True

    def test_verify_rejects_corrupted_message(self):
        seed, pk = ed25519_keypair()
        msg = b"Original threat payload"
        sig = ed25519_sign(msg, seed, pk)
        assert ed25519_verify(sig, msg + b"tampered", pk) is False

    def test_verify_rejects_wrong_public_key(self):
        seed1, pk1 = ed25519_keypair()
        _, pk2 = ed25519_keypair()
        msg = b"Original threat payload"
        sig = ed25519_sign(msg, seed1, pk1)
        assert ed25519_verify(sig, msg, pk2) is False

    def test_verify_rejects_corrupted_signature(self):
        seed, pk = ed25519_keypair()
        msg = b"Original threat payload"
        sig = bytearray(ed25519_sign(msg, seed, pk))
        sig[10] ^= 0xFF  # Flip bits
        assert ed25519_verify(bytes(sig), msg, pk) is False


class TestHDCThreatSignature:
    """Tests for HDCThreatSignature creation, serialization, and signing."""

    @pytest.fixture
    def sample_hypervector(self) -> HyperVector:
        return HyperVector.random(dim=10000, mode="binary", seed=42)

    def test_creation_from_hypervector(self, sample_hypervector):
        sig = HDCThreatSignature.from_hypervector(
            hv=sample_hypervector,
            failure_mode="ransomware_burst",
            node_id="node-us-east-1",
            metadata={"severity": "critical", "incident_id": "INC-991"},
        )
        assert sig.hypervector.dim == 10000
        assert sig.failure_mode == "ransomware_burst"
        assert sig.node_id == "node-us-east-1"
        assert sig.metadata["severity"] == "critical"
        assert len(sig.threat_id) > 0
        assert sig.signature is None

    def test_creation_from_sequence(self):
        encoder = HyperdimensionalEncoder(dim=10000, seed=42)
        events = ["open_file", "write_file", "encrypt_block", "delete_vss_shadow"]
        sig = HDCThreatSignature.from_sequence(
            encoder=encoder,
            sequence=events,
            failure_mode="ransomware_flow",
            node_id="node-k8s-pod-7",
        )
        assert sig.hypervector.dim == 10000
        assert sig.metadata["sequence_length"] == 4
        assert sig.metadata["encoding_method"] == "positional"

    def test_canonical_payload_and_digest(self, sample_hypervector):
        sig1 = HDCThreatSignature(
            threat_id="threat-123",
            failure_mode="ddos_syn",
            hypervector=sample_hypervector,
            node_id="node-1",
            timestamp=1700000000.0,
            metadata={"port": 80, "protocol": "tcp"},
        )
        sig2 = HDCThreatSignature(
            threat_id="threat-123",
            failure_mode="ddos_syn",
            hypervector=sample_hypervector,
            node_id="node-1",
            timestamp=1700000000.0,
            metadata={"protocol": "tcp", "port": 80},  # Different key order
        )
        # Should be identical due to canonical sorting
        assert sig1.canonical_payload() == sig2.canonical_payload()
        assert sig1.digest() == sig2.digest()

    def test_ed25519_sign_and_verify(self, sample_hypervector):
        seed, pk = ed25519_keypair()
        sig = HDCThreatSignature.from_hypervector(
            hv=sample_hypervector,
            failure_mode="oom_kill",
            node_id="node-prod-1",
        )
        assert sig.verify() is False  # Not signed yet
        sig.sign(seed, scheme="ed25519")
        assert sig.signature is not None
        assert sig.public_key == pk.hex()
        assert sig.verify() is True

        # Tampering with metadata invalidates signature
        sig.metadata["tampered"] = True
        assert sig.verify() is False

    def test_sha256_hmac_sign_and_verify(self, sample_hypervector):
        hmac_key = secrets.token_bytes(32)
        sig = HDCThreatSignature.from_hypervector(
            hv=sample_hypervector,
            failure_mode="privilege_escalation",
            node_id="node-test",
        )
        sig.sign(hmac_key, scheme="sha256_hmac")
        assert sig.signature_scheme == "sha256_hmac"
        assert sig.verify(public_key=hmac_key) is True

    def test_serialization_dict_and_json(self, sample_hypervector):
        seed, _ = ed25519_keypair()
        sig = HDCThreatSignature.from_hypervector(
            hv=sample_hypervector,
            failure_mode="brute_force",
            node_id="node-edge-1",
            metadata={"attempts": 500},
        )
        sig.sign(seed)
        d = sig.to_dict()
        restored_from_dict = HDCThreatSignature.from_dict(d)
        assert restored_from_dict.threat_id == sig.threat_id
        assert restored_from_dict.failure_mode == sig.failure_mode
        assert restored_from_dict.hypervector.dim == 10000
        assert restored_from_dict.verify() is True

        json_str = sig.to_json()
        restored_from_json = HDCThreatSignature.from_json(json_str)
        assert restored_from_json.verify() is True
        assert restored_from_json.similarity(sig) == pytest.approx(1.0, abs=1e-5)

    def test_similarity_and_hamming_distance(self):
        hv1 = HyperVector.random(dim=10000, seed=10)
        hv2 = HyperVector.random(dim=10000, seed=20)
        sig1 = HDCThreatSignature.from_hypervector(hv1, "mode1", "node1")
        sig2 = HDCThreatSignature.from_hypervector(hv2, "mode2", "node2")
        # Asymptotically orthogonal random vectors
        assert abs(sig1.similarity(sig2)) < 0.05
        assert 0.48 < sig1.hamming_distance(sig2) < 0.52
        # Self similarity
        assert sig1.similarity(sig1) == pytest.approx(1.0)
        assert sig1.hamming_distance(sig1) == pytest.approx(0.0)


class TestDifferentialPrivacyMasker:
    """Tests for Bernoulli DP bit-flipping and privacy guarantees."""

    def test_flip_probability_calibration(self):
        masker_eps1 = DifferentialPrivacyMasker(epsilon=1.0, dim=10000, mode="ldp")
        masker_eps3 = DifferentialPrivacyMasker(epsilon=3.0, dim=10000, mode="ldp")
        masker_eps10 = DifferentialPrivacyMasker(epsilon=10.0, dim=10000, mode="ldp")

        # Higher epsilon -> less noise -> smaller flip probability
        assert masker_eps1.flip_prob > masker_eps3.flip_prob > masker_eps10.flip_prob
        # q = 1 / (1 + e^1) approx 0.2689
        assert pytest.approx(masker_eps1.flip_prob, abs=1e-3) == 0.2689
        # q = 1 / (1 + e^3) approx 0.0474
        assert pytest.approx(masker_eps3.flip_prob, abs=1e-3) == 0.0474

    def test_invalid_parameters_raise(self):
        with pytest.raises(ValueError, match="Epsilon must be positive"):
            DifferentialPrivacyMasker(epsilon=-0.5)
        with pytest.raises(ValueError, match="Delta must be in"):
            DifferentialPrivacyMasker(delta=1.5)

    def test_mask_hypervector_preserves_dimension(self):
        masker = DifferentialPrivacyMasker(epsilon=1.5, dim=10000)
        hv = HyperVector.random(dim=10000, seed=42)
        masked_hv = masker.mask_hypervector(hv)
        assert masked_hv.dim == 10000
        # Hamming distance between masked and original should be close to flip_prob
        dist = hv.hamming_distance(masked_hv, normalized=True)
        assert pytest.approx(dist, abs=0.03) == masker.flip_prob

    def test_mask_threat_signature(self):
        masker = DifferentialPrivacyMasker(epsilon=2.0, dim=10000)
        hv = HyperVector.random(dim=10000, seed=123)
        sig = HDCThreatSignature.from_hypervector(hv, "exfiltration", "node-9")
        masked_sig = masker.mask_threat_signature(sig)
        assert masked_sig.privacy_metadata is not None
        assert masked_sig.privacy_metadata["mechanism"] == "bernoulli_randomized_response"
        assert masked_sig.privacy_metadata["epsilon"] == 2.0
        assert masked_sig.hypervector.dim == 10000

    def test_privacy_guarantee_bounds(self):
        masker = DifferentialPrivacyMasker(epsilon=1.0, delta=1e-5, dim=10000)
        bounds = masker.get_privacy_guarantee()
        assert bounds["epsilon"] == 1.0
        assert bounds["delta"] == 1e-5
        # Single bit reconstruction is bounded by retention prob p = 1 - q
        assert bounds["single_bit_reconstruction_bound"] < 0.75
        # Mutual information bound is strictly positive and finite
        assert bounds["mutual_information_bound_nats"] > 0.0
        assert bounds["privacy_loss_ratio"] == pytest.approx(np.exp(1.0), abs=1e-2)

    def test_unbiased_consensus_reconstruction(self):
        """Verify that aggregating noisy DP vectors reconstructs the true consensus pattern."""
        true_hv = HyperVector.random(dim=10000, seed=999)
        masker = DifferentialPrivacyMasker(epsilon=1.0, dim=10000)

        # 30 peer nodes report noisy masked versions of the same failure pattern
        peer_masked_vectors = []
        rng = np.random.default_rng(42)
        for _ in range(30):
            masked = masker.mask_hypervector(true_hv, rng=rng)
            peer_masked_vectors.append(masked)

        # Reconstruct consensus defense vector
        reconstructed_hv = masker.reconstruct_defense_hypervector(peer_masked_vectors)

        # Over 99% of bits should match the true vector despite 27% noise per peer!
        overlap = 1.0 - true_hv.hamming_distance(reconstructed_hv)
        assert overlap > 0.98

    def test_budget_exhaustion_raises(self):
        masker = DifferentialPrivacyMasker(epsilon=1.0, max_budget_epsilon=2.5, dim=1000)
        hv = HyperVector.random(dim=1000)
        masker.mask_hypervector(hv)  # spent 1.0
        masker.mask_hypervector(hv)  # spent 2.0
        with pytest.raises(PrivacyBudgetExceededError, match="Epsilon budget depleted"):
            masker.mask_hypervector(hv)  # spent 3.0 > 2.5


class TestZeroKnowledgeVerifier:
    """Tests for Schnorr NIZK proof verification, poisoning detection, and reputation."""

    @pytest.fixture
    def verifier(self) -> ZeroKnowledgeVerifier:
        return ZeroKnowledgeVerifier()

    def test_schnorr_nizk_proof_valid(self, verifier):
        witness = b"cluster_private_telemetry_trace_node_10_4_1_2"
        hv = HyperVector.random(dim=10000, seed=42)
        proof = verifier.generate_zk_proof(
            telemetry_witness=witness,
            hypervector=hv,
            failure_mode="oom_killer",
            node_id="node-prod-k8s",
        )
        assert proof["protocol"] == "schnorr_nizk_curve25519"
        valid = verifier.verify_zk_proof(
            proof=proof,
            hypervector=hv,
            failure_mode="oom_killer",
            node_id="node-prod-k8s",
        )
        assert valid is True

    def test_schnorr_nizk_tampered_fails(self, verifier):
        witness = b"cluster_telemetry"
        hv = HyperVector.random(dim=10000, seed=42)
        proof = verifier.generate_zk_proof(witness, hv, "ransomware", "node-1")

        # Tampered hypervector
        hv_other = HyperVector.random(dim=10000, seed=99)
        assert verifier.verify_zk_proof(proof, hv_other, "ransomware", "node-1") is False

        # Tampered failure mode
        assert verifier.verify_zk_proof(proof, hv, "other_mode", "node-1") is False

        # Tampered node id
        assert verifier.verify_zk_proof(proof, hv, "ransomware", "node-2") is False

    def test_density_validation(self, verifier):
        # Valid random vector ~50% density
        normal_hv = HyperVector.random(dim=10000, seed=1)
        valid, msg = verifier.validate_density(normal_hv)
        assert valid is True

        # Degenerate all-zeros
        zero_hv = HyperVector.zeros(dim=10000)
        valid, msg = verifier.validate_density(zero_hv)
        assert valid is False
        assert "all-zeros" in msg

        # Degenerate all-ones
        ones_hv = HyperVector.ones(dim=10000)
        valid, msg = verifier.validate_density(ones_hv)
        assert valid is False
        assert "all-ones" in msg

        # Skewed density (e.g. 10% density)
        arr = np.zeros(10000, dtype=np.uint8)
        arr[:1000] = 1
        skewed_hv = HyperVector.from_binary_array(arr)
        valid, msg = verifier.validate_density(skewed_hv)
        assert valid is False
        assert "Density violation" in msg

    def test_entropy_validation_detects_repeating_tiles(self, verifier):
        # Normal high-entropy vector
        normal_hv = HyperVector.random(dim=10000, seed=5)
        valid, _ = verifier.validate_entropy(normal_hv)
        assert valid is True

        # Synthetic poisoned vector with repeated 64-bit word tile
        tile = np.uint64(0xAAAAAAAAAAAAAAAA)
        poisoned_words = np.full(10000 // 64 + 1, tile, dtype=np.uint64)
        raw_bytes = poisoned_words.tobytes()[: 10000 // 8]
        poisoned_hv = HyperVector.from_bytes(raw_bytes, dim=10000)
        valid, msg = verifier.validate_entropy(poisoned_hv)
        assert valid is False
        assert "synthetic poisoning attack" in msg or "entropy" in msg.lower()

    def test_anti_consensus_poisoning_detection(self, verifier):
        memory_bank = HolographicDefenseMemoryBank(dim=10000)
        base_hv = HyperVector.random(dim=10000, seed=77)

        # Populate memory bank with 3 legitimate signatures
        for i in range(3):
            # Minor variations
            noisy_hv = base_hv.bind(HyperVector.random(dim=10000, seed=100 + i))
            # Keep high similarity
            sig = HDCThreatSignature.from_hypervector(base_hv, "syn_flood", f"node-{i}")
            memory_bank.add(sig)

        # Adversary constructs an exact inverted/anti-vector
        inverted_hv = base_hv.invert()
        assert base_hv.cosine_similarity(inverted_hv) == pytest.approx(-1.0)
        poison_sig = HDCThreatSignature.from_hypervector(inverted_hv, "syn_flood", "adversary-node")

        is_poisoned, reason = verifier.detect_poisoning(poison_sig, memory_bank=memory_bank)
        assert is_poisoned is True
        assert "Anti-consensus attack detected" in reason

    def test_reputation_and_quarantine(self, verifier):
        node = "suspicious-node-66"
        assert verifier.get_reputation(node) == 0.8
        assert verifier.is_blacklisted(node) is False

        # Multiple failures decrement reputation
        verifier.update_reputation(node, success=False, delta=0.3)
        assert verifier.get_reputation(node) == pytest.approx(0.5)

        verifier.update_reputation(node, success=False, delta=0.3)
        assert verifier.get_reputation(node) == pytest.approx(0.2)
        # Dropped below quarantine threshold (0.25)
        assert verifier.is_blacklisted(node) is True

        # Unblacklist restores
        verifier.unblacklist(node)
        assert verifier.is_blacklisted(node) is False


class TestHolographicDefenseMemoryBank:
    """Tests for holographic memory bank storage and sub-microsecond retrieval."""

    def test_bundle_and_query_modes(self):
        bank = HolographicDefenseMemoryBank(dim=10000)
        encoder = HyperdimensionalEncoder(dim=10000, seed=42)

        # Create two distinct failure mode trajectories
        ransomware_events = ["encrypt_file", "delete_backup", "ransom_note"]
        syn_flood_events = ["syn_rcvd", "half_open_conn", "backlog_overflow"]

        sig_rw1 = HDCThreatSignature.from_sequence(encoder, ransomware_events, "ransomware", "node-1")
        sig_rw2 = HDCThreatSignature.from_sequence(encoder, ransomware_events, "ransomware", "node-2")
        sig_syn = HDCThreatSignature.from_sequence(encoder, syn_flood_events, "syn_flood", "node-3")

        bank.add(sig_rw1)
        bank.add(sig_rw2)
        bank.add(sig_syn)

        assert bank.total_signatures() == 3
        assert set(bank.failure_modes()) == {"ransomware", "syn_flood"}

        # Query with incoming ransomware trace
        query_rw = encoder.encode_sequence(ransomware_events)
        matches = bank.query(query_rw, threshold=0.7)
        assert len(matches) > 0
        assert matches[0][0] == "ransomware"
        assert matches[0][1] > 0.85

        # Query with incoming SYN flood trace
        query_syn = encoder.encode_sequence(syn_flood_events)
        matches_syn = bank.query(query_syn, threshold=0.7)
        assert matches_syn[0][0] == "syn_flood"


class TestFederatedThreatMeshNode:
    """Tests for peer-to-peer threat sharing, verification, and mesh resilience."""

    def test_two_node_peer_exchange(self):
        node_a = FederatedThreatMeshNode(node_id="cluster-aws-us-east")
        node_b = FederatedThreatMeshNode(node_id="cluster-gcp-europe-west")

        # Establish mutual mesh peering
        node_a.connect_peer(node_b)
        assert "cluster-gcp-europe-west" in node_a.peers
        assert "cluster-aws-us-east" in node_b.peers

        # Node A observes an incident and publishes a threat signature
        events = ["excessive_fork", "ptrace_inject", "priv_escalate"]
        sig_a = node_a.create_threat_signature(
            failure_mode="privilege_escalation",
            log_events=events,
            metadata={"cve": "CVE-2026-9999"},
        )
        assert sig_a.verify() is True
        assert sig_a.zk_proof is not None

        # Publish to mesh
        pub_results = node_a.publish_threat(sig_a)
        assert pub_results["cluster-gcp-europe-west"] is True

        # Node B should have received and verified the signature
        assert len(node_b.received_threats) == 1
        assert "privilege_escalation" in node_b.memory_bank.failure_modes()

        # Node B now detects live incident matching the shared threat pattern
        detected = node_b.scan_log_sequence(events, threshold=0.40)
        assert detected is not None
        matched_mode, sim = detected
        assert matched_mode == "privilege_escalation"
        assert sim > 0.40

    def test_multi_node_gossip_sync(self):
        node1 = FederatedThreatMeshNode(node_id="node-1")
        node2 = FederatedThreatMeshNode(node_id="node-2")
        node3 = FederatedThreatMeshNode(node_id="node-3")

        # Linear mesh: 1 <-> 2 <-> 3
        node1.connect_peer(node2)
        node2.connect_peer(node3)

        # Node 1 publishes a threat (delivered only to peer node 2)
        sig = node1.create_threat_signature("disk_thrashing", ["read_storm", "io_wait_high"])
        node1.publish_threat(sig)

        assert len(node2.received_threats) == 1
        assert len(node3.received_threats) == 0

        # Run gossip sync between node 2 and node 3
        synced = node3.gossip_sync()
        assert synced == 1
        assert len(node3.received_threats) == 1
        assert "disk_thrashing" in node3.memory_bank.failure_modes()

    def test_rejection_of_poisoned_threat_from_rogue_peer(self):
        honest_node = FederatedThreatMeshNode(node_id="honest-node")
        rogue_node = FederatedThreatMeshNode(node_id="rogue-node")
        honest_node.connect_peer(rogue_node)

        # Rogue node creates an all-zeros poisoned hypervector
        zero_hv = HyperVector.zeros(dim=10000)
        poison_sig = HDCThreatSignature.from_hypervector(
            hv=zero_hv,
            failure_mode="poison_attack",
            node_id=rogue_node.node_id,
        )
        poison_sig.sign(rogue_node.private_seed)

        # Honest node must reject the poisoned signature
        accepted, reason = honest_node.receive_threat(poison_sig, sender_node_id=rogue_node.node_id)
        assert accepted is False
        assert "Poisoning detected" in reason
        assert "all-zeros" in reason

        # Honest node's defense memory remains uncontaminated
        assert "poison_attack" not in honest_node.memory_bank.failure_modes()

    def test_rejection_of_forged_signature(self):
        receiver = FederatedThreatMeshNode(node_id="receiver-node")
        attacker = FederatedThreatMeshNode(node_id="attacker-node")

        hv = HyperVector.random(dim=10000, seed=12)
        sig = HDCThreatSignature.from_hypervector(hv, "attack", attacker.node_id)
        # Sign with attacker key
        sig.sign(attacker.private_seed)
        # Forge sender ID to impersonate a trusted authority
        sig.node_id = "trusted-authority"

        accepted, reason = receiver.receive_threat(sig)
        assert accepted is False
        assert "signature verification failed" in reason.lower()

    def test_get_mesh_status(self):
        node = FederatedThreatMeshNode(node_id="telemetry-node")
        status = node.get_mesh_status()
        assert status["node_id"] == "telemetry-node"
        assert status["peer_count"] == 0
        assert status["published_threats_count"] == 0
        assert status["reputation_score"] == 0.8
        assert "public_key" in status
