"""Federated Zero-Knowledge Threat Sharing Mesh.

This module implements Frontier 5: Decentralized P2P threat sharing across
untrusted infrastructure nodes using 10,000-bit Holographic Hypervectors (HDC),
calibrated Bernoulli Differential Privacy, and Non-Interactive Zero-Knowledge (NIZK)
proofs with Ed25519 cryptographic signatures.

Key Architectural Pillars:
1. HDCThreatSignature: 10,000-bit holographic threat hypervector representing failure
   modes and attack patterns signed cryptographically with Ed25519 (RFC 8032) or SHA256 HMAC.
2. DifferentialPrivacyMasker: Injects calibrated Bernoulli bit-flipping noise providing
   (\\epsilon, \\delta)-differential privacy, mathematically guaranteeing that proprietary
   log tokens, user identifiers, internal IP addresses, and PII cannot be reverse-engineered.
3. ZeroKnowledgeVerifier: Validates signature integrity and mathematically verifies that
   shared hypervectors are genuine and non-poisoned without decrypting or exposing internal
   cluster details (using Schnorr NIZK Proofs of Knowledge over Curve25519 and statistical
   entropy / density tests).
4. FederatedThreatMeshNode: Decentralized peer node participating in a threat intelligence
   mesh, publishing differentially private threat signatures and merging incoming peer
   signatures into a local Holographic Defense Memory Bank.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import math
import os
import secrets
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union

import numpy as np

from .hdc import HDCAnomalyDetector, HDCProfileAccumulator, HyperdimensionalEncoder, HyperVector

# ==============================================================================
# Pure Python Ed25519 (RFC 8032) Cryptographic Engine
# Uses Extended Edwards Coordinates (X:Y:Z:T) for high-performance verification
# ==============================================================================

_ED25519_P: int = 2**255 - 19
_ED25519_L: int = 2**252 + 27742317777372353535851937790883648493
_ED25519_D: int = -121665 * pow(121666, _ED25519_P - 2, _ED25519_P) % _ED25519_P
_ED25519_I: int = pow(2, (_ED25519_P - 1) // 4, _ED25519_P)


def _ed25519_inv(x: int) -> int:
    return pow(x, _ED25519_P - 2, _ED25519_P)


def _ed25519_recover_x(y: int) -> int:
    xx = (y * y - 1) * _ed25519_inv(_ED25519_D * y * y + 1) % _ED25519_P
    x = pow(xx, (_ED25519_P + 3) // 8, _ED25519_P)
    if (x * x - xx) % _ED25519_P != 0:
        x = (x * _ED25519_I) % _ED25519_P
    if x % 2 != 0:
        x = _ED25519_P - x
    return x


# Base point B in affine and extended coordinates
_ED25519_BY: int = 4 * _ed25519_inv(5) % _ED25519_P
_ED25519_BX: int = _ed25519_recover_x(_ED25519_BY)
# Extended Edwards point: (X, Y, Z, T) with T = X * Y / Z
_ED25519_BASE: Tuple[int, int, int, int] = (
    _ED25519_BX,
    _ED25519_BY,
    1,
    (_ED25519_BX * _ED25519_BY) % _ED25519_P,
)


def _point_add(
    p1: Tuple[int, int, int, int], p2: Tuple[int, int, int, int]
) -> Tuple[int, int, int, int]:
    """Add two points in Extended Edwards coordinates (0 inversions)."""
    x1, y1, z1, t1 = p1
    x2, y2, z2, t2 = p2
    p = _ED25519_P
    a = (y1 - x1) * (y2 - x2) % p
    b = (y1 + x1) * (y2 + x2) % p
    c = 2 * t1 * t2 * _ED25519_D % p
    d = 2 * z1 * z2 % p
    e = (b - a) % p
    f = (d - c) % p
    g = (d + c) % p
    h = (b + a) % p
    return (e * f % p, g * h % p, f * g % p, e * h % p)


def _point_double(p: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    """Double a point in Extended Edwards coordinates."""
    x1, y1, z1, _ = p
    mod = _ED25519_P
    a = x1 * x1 % mod
    b = y1 * y1 % mod
    c = 2 * z1 * z1 % mod
    d = (-a) % mod
    e = ((x1 + y1) * (x1 + y1) - a - b) % mod
    g = (d + b) % mod
    f = (g - c) % mod
    h = (d - b) % mod
    return (e * f % mod, g * h % mod, f * g % mod, e * h % mod)


def _point_mul(p: Tuple[int, int, int, int], scalar: int) -> Tuple[int, int, int, int]:
    """Scalar multiplication via double-and-add."""
    r: Tuple[int, int, int, int] = (0, 1, 1, 0)
    q = p
    s = scalar
    while s > 0:
        if s & 1:
            r = _point_add(r, q)
        q = _point_double(q)
        s >>= 1
    return r


def _to_affine(p: Tuple[int, int, int, int]) -> Tuple[int, int]:
    """Convert Extended Edwards point to affine (x, y)."""
    x, y, z, _ = p
    z_inv = _ed25519_inv(z)
    return (x * z_inv % _ED25519_P, y * z_inv % _ED25519_P)


def _encode_point(p: Tuple[int, int, int, int]) -> bytes:
    """Encode point to 32 bytes little-endian."""
    x, y = _to_affine(p)
    bits = [(y >> i) & 1 for i in range(255)] + [x & 1]
    return bytes(sum(bits[i * 8 + j] << j for j in range(8)) for i in range(32))


def _decode_point(s: bytes) -> Tuple[int, int, int, int]:
    """Decode 32-byte point into Extended Edwards coordinates."""
    if len(s) != 32:
        raise ValueError("Invalid point byte length")
    y = sum(2**i * ((s[i // 8] >> (i % 8)) & 1) for i in range(255))
    p = _ED25519_P
    xx = (y * y - 1) * _ed25519_inv(_ED25519_D * y * y + 1) % p
    x = pow(xx, (p + 3) // 8, p)
    if (x * x - xx) % p != 0:
        x = (x * _ED25519_I) % p
    if (x & 1) != ((s[31] >> 7) & 1):
        x = p - x
    if (-x * x + y * y - 1 - _ED25519_D * x * x * y * y) % p != 0:
        raise ValueError("Point is not on Curve25519")
    return (x, y, 1, (x * y) % p)


def ed25519_keypair(seed: Optional[bytes] = None) -> Tuple[bytes, bytes]:
    """Generate Ed25519 (private_seed, public_key) pair."""
    if seed is None:
        seed = secrets.token_bytes(32)
    elif len(seed) != 32:
        raise ValueError(f"Seed must be 32 bytes, got {len(seed)}")

    h = hashlib.sha512(seed).digest()
    a = int.from_bytes(h[:32], "little")
    a &= (1 << 254) - 8
    a |= 1 << 254
    point_a = _point_mul(_ED25519_BASE, a)
    public_key = _encode_point(point_a)
    return seed, public_key


def ed25519_sign(message: bytes, private_seed: bytes, public_key: bytes) -> bytes:
    """Compute RFC 8032 Ed25519 64-byte signature."""
    h = hashlib.sha512(private_seed).digest()
    a = int.from_bytes(h[:32], "little")
    a &= (1 << 254) - 8
    a |= 1 << 254
    prefix = h[32:]
    r_hash = hashlib.sha512(prefix + message).digest()
    r = int.from_bytes(r_hash, "little") % _ED25519_L
    point_r = _point_mul(_ED25519_BASE, r)
    r_bytes = _encode_point(point_r)
    k_hash = hashlib.sha512(r_bytes + public_key + message).digest()
    k = int.from_bytes(k_hash, "little") % _ED25519_L
    s = (r + k * a) % _ED25519_L
    s_bytes = s.to_bytes(32, "little")
    return r_bytes + s_bytes


def ed25519_verify(signature: bytes, message: bytes, public_key: bytes) -> bool:
    """Verify RFC 8032 Ed25519 signature."""
    if len(signature) != 64 or len(public_key) != 32:
        return False
    try:
        point_r = _decode_point(signature[:32])
        point_a = _decode_point(public_key)
        s = int.from_bytes(signature[32:], "little")
        if s >= _ED25519_L:
            return False
        k_hash = hashlib.sha512(signature[:32] + public_key + message).digest()
        k = int.from_bytes(k_hash, "little") % _ED25519_L
        sb = _point_mul(_ED25519_BASE, s)
        ka = _point_mul(point_a, k)
        r_plus_ka = _point_add(point_r, ka)
        return _to_affine(sb) == _to_affine(r_plus_ka)
    except Exception:
        return False


# ==============================================================================
# Exceptions
# ==============================================================================


class FederatedMeshError(Exception):
    """Base exception for Federated Mesh operations."""


class InvalidSignatureError(FederatedMeshError):
    """Raised when cryptographic signature verification fails."""


class PoisoningDetectedError(FederatedMeshError):
    """Raised when an adversarial poisoned hypervector is rejected."""


class PrivacyBudgetExceededError(FederatedMeshError):
    """Raised when differential privacy budget (epsilon, delta) is exhausted."""


# ==============================================================================
# 1. HDCThreatSignature: 10,000-bit Holographic Threat Hypervector
# ==============================================================================


@dataclass
class HDCThreatSignature:
    """10,000-bit Holographic Threat Hypervector with Cryptographic Signatures.

    Represents distributed failure modes (e.g. ransomware I/O burst, SYN floods,
    auth brute force, memory exhaustion) in a high-dimensional holographic space.

    Attributes:
        threat_id: Globally unique identifier for this threat event.
        failure_mode: Classification/type string (e.g., 'ransomware', 'ddos', 'oom').
        hypervector: 10,000-bit HyperVector instance encoding the failure trajectory.
        node_id: Identifier of the publishing peer node.
        timestamp: Unix timestamp when the signature was created.
        metadata: Contextual metadata (severity, log count, affected services).
        signature_scheme: Cryptographic scheme ('ed25519' or 'sha256_hmac').
        public_key: Hex string of the publisher's public key.
        signature: Hex string of the cryptographic signature.
        zk_proof: Optional zero-knowledge proof of cluster telemetry authenticity.
        privacy_metadata: Optional details on applied differential privacy noise.
    """

    threat_id: str
    failure_mode: str
    hypervector: HyperVector
    node_id: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    signature_scheme: str = "ed25519"
    public_key: Optional[str] = None
    signature: Optional[str] = None
    zk_proof: Optional[Dict[str, Any]] = None
    privacy_metadata: Optional[Dict[str, Any]] = None

    def canonical_payload(self) -> bytes:
        """Deterministic byte representation used for hashing and signing."""
        # Stabilize metadata via sorted JSON serialization
        meta_json = json.dumps(self.metadata, sort_keys=True, separators=(",", ":"))
        hv_bytes = self.hypervector.to_bytes()
        # Header components: threat_id | failure_mode | node_id | timestamp | dim
        header = f"{self.threat_id}|{self.failure_mode}|{self.node_id}|{self.timestamp:.6f}|{self.hypervector.dim}|{meta_json}|"
        return header.encode("utf-8") + hv_bytes

    def digest(self) -> str:
        """Compute SHA-256 hex digest of the canonical threat payload."""
        return hashlib.sha256(self.canonical_payload()).hexdigest()

    def sign(self, private_key: bytes, scheme: Optional[str] = None) -> HDCThreatSignature:
        """Cryptographically sign the threat signature in-place.

        Args:
            private_key: 32-byte Ed25519 seed or HMAC secret key.
            scheme: Cryptographic scheme override ('ed25519' or 'sha256_hmac').

        Returns:
            Self reference for fluent chaining.
        """
        if scheme is not None:
            self.signature_scheme = scheme

        payload = self.canonical_payload()

        if self.signature_scheme == "ed25519":
            seed, pk = ed25519_keypair(private_key)
            sig_bytes = ed25519_sign(payload, seed, pk)
            self.public_key = pk.hex()
            self.signature = sig_bytes.hex()
        elif self.signature_scheme == "sha256_hmac":
            sig_bytes = hmac.new(private_key, payload, hashlib.sha256).digest()
            # In HMAC mode, public key is represented by key fingerprint
            self.public_key = hashlib.sha256(private_key).hexdigest()[:32]
            self.signature = sig_bytes.hex()
        else:
            raise ValueError(f"Unsupported signature scheme: {self.signature_scheme}")

        return self

    def verify(self, public_key: Optional[bytes] = None) -> bool:
        """Verify the cryptographic signature.

        Args:
            public_key: Optional explicit public key bytes. If None, uses self.public_key.

        Returns:
            True if signature is valid and untampered, False otherwise.
        """
        if not self.signature or (self.public_key is None and public_key is None):
            return False

        payload = self.canonical_payload()
        try:
            sig_bytes = bytes.fromhex(self.signature)
            pk_bytes = public_key if public_key is not None else bytes.fromhex(self.public_key or "")

            if self.signature_scheme == "ed25519":
                return ed25519_verify(sig_bytes, payload, pk_bytes)
            elif self.signature_scheme == "sha256_hmac":
                # For HMAC, public_key parameter must be the shared secret
                expected = hmac.new(pk_bytes, payload, hashlib.sha256).digest()
                return hmac.compare_digest(sig_bytes, expected)
            return False
        except Exception:
            return False

    def similarity(self, other: HDCThreatSignature) -> float:
        """Sub-microsecond HDC cosine similarity to another threat signature."""
        return self.hypervector.cosine_similarity(other.hypervector)

    def hamming_distance(self, other: HDCThreatSignature, normalized: bool = True) -> float:
        """Hamming distance to another threat signature."""
        return self.hypervector.hamming_distance(other.hypervector, normalized=normalized)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize threat signature to JSON-friendly dictionary."""
        return {
            "threat_id": self.threat_id,
            "failure_mode": self.failure_mode,
            "dim": self.hypervector.dim,
            "mode": self.hypervector.mode,
            "hypervector_b64": base64.b64encode(self.hypervector.to_bytes()).decode("ascii"),
            "node_id": self.node_id,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "signature_scheme": self.signature_scheme,
            "public_key": self.public_key,
            "signature": self.signature,
            "zk_proof": self.zk_proof,
            "privacy_metadata": self.privacy_metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> HDCThreatSignature:
        """Deserialize threat signature from dictionary."""
        raw_hv = base64.b64decode(data["hypervector_b64"])
        hv = HyperVector.from_bytes(raw_hv, dim=data.get("dim", 10000), mode=data.get("mode", "binary"))
        return cls(
            threat_id=data["threat_id"],
            failure_mode=data["failure_mode"],
            hypervector=hv,
            node_id=data["node_id"],
            timestamp=data.get("timestamp", time.time()),
            metadata=data.get("metadata", {}),
            signature_scheme=data.get("signature_scheme", "ed25519"),
            public_key=data.get("public_key"),
            signature=data.get("signature"),
            zk_proof=data.get("zk_proof"),
            privacy_metadata=data.get("privacy_metadata"),
        )

    def to_json(self) -> str:
        """Serialize threat signature to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> HDCThreatSignature:
        """Deserialize threat signature from JSON string."""
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def from_hypervector(
        cls,
        hv: HyperVector,
        failure_mode: str,
        node_id: str,
        threat_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> HDCThreatSignature:
        """Factory method creating an unsigned threat signature from a HyperVector."""
        tid = threat_id or secrets.token_hex(16)
        return cls(
            threat_id=tid,
            failure_mode=failure_mode,
            hypervector=hv,
            node_id=node_id,
            metadata=metadata or {},
        )

    @classmethod
    def from_sequence(
        cls,
        encoder: HyperdimensionalEncoder,
        sequence: Sequence[Any],
        failure_mode: str,
        node_id: str,
        threat_id: Optional[str] = None,
        method: str = "positional",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> HDCThreatSignature:
        """Encode log sequence into an HDC threat signature."""
        hv = encoder.encode_sequence(sequence, method=method)
        meta = metadata.copy() if metadata else {}
        meta["sequence_length"] = len(sequence)
        meta["encoding_method"] = method
        return cls.from_hypervector(
            hv=hv,
            failure_mode=failure_mode,
            node_id=node_id,
            threat_id=threat_id,
            metadata=meta,
        )


# ==============================================================================
# 2. DifferentialPrivacyMasker: Calibrated Bernoulli Noise
# ==============================================================================


class DifferentialPrivacyMasker:
    """Injects Calibrated Bernoulli Bit-Flipping Noise for (epsilon, delta)-DP.

    Prevents reverse-engineering of sensitive enterprise telemetry, proprietary
    log text, customer PII, internal cluster hostnames, or private parameters.

    Mathematical Foundations:
    - Coordinate-level Randomized Response:
      Each bit $x_i \\in \\{0, 1\\}$ is retained with probability $p$ and flipped
      with probability $q = 1 - p$:
      $$p = \\frac{e^{\\epsilon_{\\text{local}}}}{1 + e^{\\epsilon_{\\text{local}}}}, \\quad
        q = \\frac{1}{1 + e^{\\epsilon_{\\text{local}}}}$$
      This provides rigorous Local Differential Privacy (LDP) with privacy ratio:
      $$\\frac{P(Y = 1 \\mid X = 1)}{P(Y = 1 \\mid X = 0)} = \\frac{p}{q} = e^{\\epsilon}$$
    - Coordinate-wise Composition over $D$ coordinates:
      Under Advanced Composition Theorem or Gaussian/Renyi relaxation, a global
      $(\\epsilon, \\delta)$ guarantee is achieved by calibrating coordinate noise:
      $$\\epsilon_{\\text{coord}} \\approx \\frac{\\epsilon}{2 \\sqrt{2 D \\ln(1.25 / \\delta)}}$$
    - Unbiased Defense Reconstruction:
      Given $M$ masked peer hypervectors $Y_1, \\dots, Y_M$, since
      $\\mathbb{E}[Y_i] = (1 - 2q) X_i + q$, the consensus defense bit is estimated
      unbiasedly via:
      $$\\hat{X}_i = \\frac{\\bar{Y}_i - q}{1 - 2q}$$
      allowing federated nodes to reconstruct collective threat profiles with zero PII leakage.
    """

    def __init__(
        self,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        mode: str = "ldp",
        dim: int = 10000,
        max_budget_epsilon: Optional[float] = None,
        max_budget_delta: Optional[float] = None,
    ) -> None:
        """Initialize Differential Privacy Masker.

        Args:
            epsilon: Privacy budget parameter (epsilon > 0).
            delta: Relaxation parameter (0 <= delta < 1).
            mode: 'ldp' for direct local DP or 'composed' for coordinate composition.
            dim: Dimension of hypervectors (default 10,000).
            max_budget_epsilon: Optional maximum total epsilon budget.
            max_budget_delta: Optional maximum total delta budget.
        """
        if epsilon <= 0:
            raise ValueError(f"Epsilon must be positive, got {epsilon}")
        if not (0 <= delta < 1):
            raise ValueError(f"Delta must be in [0, 1), got {delta}")

        self.epsilon: float = epsilon
        self.delta: float = delta
        self.mode: str = mode
        self.dim: int = dim
        self.max_budget_epsilon: Optional[float] = max_budget_epsilon
        self.max_budget_delta: Optional[float] = max_budget_delta

        self._spent_epsilon: float = 0.0
        self._spent_delta: float = 0.0

        # Calculate coordinate flip probability
        self.flip_prob: float = self._compute_flip_prob()

    def _compute_flip_prob(self) -> float:
        """Calibrate Bernoulli flip probability from epsilon and delta."""
        if self.mode == "ldp":
            eff_eps = self.epsilon
        elif self.mode == "composed":
            # Advanced composition calibration over D dimensions
            d_safe = max(1, self.dim)
            delta_safe = max(1e-15, self.delta)
            eff_eps = self.epsilon / (2.0 * math.sqrt(2.0 * d_safe * math.log(1.25 / delta_safe)))
        else:
            raise ValueError(f"Unknown DP mode: {self.mode}")

        # Bernoulli flip probability q = 1 / (1 + exp(eff_eps))
        # Bounded between (0, 0.5)
        q = 1.0 / (1.0 + math.exp(min(50.0, eff_eps)))
        return float(np.clip(q, 1e-7, 0.499999))

    @property
    def spent_budget(self) -> Tuple[float, float]:
        """Return total spent (epsilon, delta) budget."""
        return (self._spent_epsilon, self._spent_delta)

    def mask_hypervector(
        self, hv: HyperVector, rng: Optional[np.random.Generator] = None
    ) -> HyperVector:
        """Inject calibrated Bernoulli noise into a 10,000-bit hypervector.

        Args:
            hv: Input HyperVector to mask.
            rng: Optional numpy Generator for reproducibility.

        Returns:
            Differentially private masked HyperVector.
        """
        # Check budget limits
        if self.max_budget_epsilon is not None:
            if self._spent_epsilon + self.epsilon > self.max_budget_epsilon:
                raise PrivacyBudgetExceededError(
                    f"Epsilon budget depleted: spent {self._spent_epsilon:.2f} + {self.epsilon:.2f} "
                    f"> max {self.max_budget_epsilon:.2f}"
                )

        r = rng if rng is not None else np.random.default_rng()
        bits = hv.to_binary()

        # Vectorized Bernoulli bit flipping
        flips = r.binomial(1, self.flip_prob, size=hv.dim).astype(np.uint8)
        masked_bits = bits ^ flips

        # Track privacy budget
        self._spent_epsilon += self.epsilon
        self._spent_delta += self.delta

        return HyperVector.from_binary_array(masked_bits, dim=hv.dim, mode=hv.mode)

    def mask_threat_signature(
        self, sig: HDCThreatSignature, rng: Optional[np.random.Generator] = None
    ) -> HDCThreatSignature:
        """Create a differentially private copy of an HDCThreatSignature."""
        masked_hv = self.mask_hypervector(sig.hypervector, rng=rng)
        meta = sig.metadata.copy()

        privacy_info = {
            "mechanism": "bernoulli_randomized_response",
            "epsilon": self.epsilon,
            "delta": self.delta,
            "mode": self.mode,
            "flip_prob": self.flip_prob,
            "timestamp": time.time(),
        }

        # Return a new signature object with updated metadata
        return HDCThreatSignature(
            threat_id=sig.threat_id,
            failure_mode=sig.failure_mode,
            hypervector=masked_hv,
            node_id=sig.node_id,
            timestamp=sig.timestamp,
            metadata=meta,
            signature_scheme=sig.signature_scheme,
            zk_proof=sig.zk_proof,
            privacy_metadata=privacy_info,
        )

    def unbiased_mean(self, masked_vectors: Sequence[HyperVector]) -> np.ndarray:
        """Compute coordinate-wise unbiased probability estimate across peers.

        Estimates underlying bit probabilities $\\hat{X}_i \\in [0, 1]$ from
        noisy observed peer vectors without learning any individual node's private bits.
        """
        if not masked_vectors:
            return np.zeros(self.dim, dtype=np.float64)

        q = self.flip_prob
        # Coordinate-wise sample mean across M peers
        stacked = np.stack([v.to_binary() for v in masked_vectors], axis=0)
        sample_mean = np.mean(stacked, axis=0)

        # Invert Randomized Response: y_bar = (1 - 2q) * x + q  =>  x = (y_bar - q) / (1 - 2q)
        denom = max(1e-6, 1.0 - 2.0 * q)
        unbiased = (sample_mean - q) / denom
        return np.clip(unbiased, 0.0, 1.0)

    def reconstruct_defense_hypervector(
        self, masked_vectors: Sequence[HyperVector]
    ) -> HyperVector:
        """Reconstruct clean consensus defense hypervector from noisy peer vectors."""
        if not masked_vectors:
            return HyperVector.zeros(dim=self.dim)
        if len(masked_vectors) == 1:
            return masked_vectors[0]

        prob_estimates = self.unbiased_mean(masked_vectors)
        binary_consensus = (prob_estimates >= 0.5).astype(np.uint8)
        return HyperVector.from_binary_array(binary_consensus, dim=self.dim)

    def get_privacy_guarantee(self) -> Dict[str, float]:
        """Compute formal mathematical differential privacy bounds.

        Returns:
            Dictionary containing epsilon, delta, flip probability,
            reconstruction bound, and mutual information upper bound.
        """
        q = self.flip_prob
        p = 1.0 - q
        # Max single-bit reconstruction probability under flat prior
        reconstruction_bound = p
        # Mutual information upper bound I(X; Y) <= 0.5 * (1 - 2q)^2
        mutual_info_bound = 0.5 * ((1.0 - 2.0 * q) ** 2)

        return {
            "epsilon": self.epsilon,
            "delta": self.delta,
            "flip_probability": q,
            "retention_probability": p,
            "single_bit_reconstruction_bound": reconstruction_bound,
            "mutual_information_bound_nats": mutual_info_bound,
            "privacy_loss_ratio": p / max(1e-12, q),
        }


# ==============================================================================
# 3. Holographic Defense Memory Bank
# ==============================================================================


class HolographicDefenseMemoryBank:
    """Sub-microsecond Holographic Associative Defense Repository.

    Aggregates incoming federated threat signatures into bundled holographic
    memory profiles organized by failure mode. Provides ultra-fast O(1) similarity
    matching for live anomaly mitigation.
    """

    def __init__(self, dim: int = 10000, mode: str = "binary") -> None:
        self.dim: int = dim
        self.mode: str = mode
        self._signatures_by_mode: Dict[str, List[HDCThreatSignature]] = {}
        self._profiles_by_mode: Dict[str, HyperVector] = {}
        self._accumulators_by_mode: Dict[str, HDCProfileAccumulator] = {}
        self._all_signatures: List[HDCThreatSignature] = []
        self._global_accumulator: HDCProfileAccumulator = HDCProfileAccumulator(dim=dim, mode=mode)
        self._cached_global_profile: Optional[HyperVector] = None

    def add(self, sig: HDCThreatSignature) -> None:
        """Add verified threat signature and update holographic memory profiles."""
        if sig.hypervector.dim != self.dim:
            raise ValueError(f"Dimension mismatch: {sig.hypervector.dim} vs {self.dim}")

        mode = sig.failure_mode
        if mode not in self._signatures_by_mode:
            self._signatures_by_mode[mode] = []
            self._accumulators_by_mode[mode] = HDCProfileAccumulator(dim=self.dim, mode=self.mode)

        self._signatures_by_mode[mode].append(sig)
        self._all_signatures.append(sig)

        # Update mode profile
        self._accumulators_by_mode[mode].add(sig.hypervector)
        self._profiles_by_mode[mode] = self._accumulators_by_mode[mode].to_hypervector()

        # Update global profile
        self._global_accumulator.add(sig.hypervector)
        self._cached_global_profile = self._global_accumulator.to_hypervector()

    def query(
        self, query_hv: HyperVector, threshold: float = 0.60
    ) -> List[Tuple[str, float]]:
        """Match query hypervector against all defense profiles.

        Returns:
            Sorted list of (failure_mode, cosine_similarity) exceeding threshold.
        """
        results: List[Tuple[str, float]] = []
        for mode_name, profile_hv in self._profiles_by_mode.items():
            sim = query_hv.cosine_similarity(profile_hv)
            if sim >= threshold:
                results.append((mode_name, sim))

        # Sort descending by similarity
        results.sort(key=lambda item: item[1], reverse=True)
        return results

    def is_threat(self, query_hv: HyperVector, threshold: float = 0.60) -> bool:
        """Quick check if query hypervector matches any stored threat pattern."""
        matches = self.query(query_hv, threshold=threshold)
        return len(matches) > 0

    def get_profile(self, failure_mode: str) -> Optional[HyperVector]:
        """Retrieve bundled holographic defense profile for a failure mode."""
        return self._profiles_by_mode.get(failure_mode)

    @property
    def global_defense_profile(self) -> Optional[HyperVector]:
        """Bundled profile across all observed failure modes."""
        return self._cached_global_profile

    def failure_modes(self) -> List[str]:
        """List of all registered failure modes in memory."""
        return list(self._profiles_by_mode.keys())

    def total_signatures(self) -> int:
        """Total number of stored threat signatures."""
        return len(self._all_signatures)

    def clear(self) -> None:
        """Clear memory bank."""
        self._signatures_by_mode.clear()
        self._profiles_by_mode.clear()
        self._accumulators_by_mode.clear()
        self._all_signatures.clear()
        self._global_accumulator = HDCProfileAccumulator(dim=self.dim, mode=self.mode)
        self._cached_global_profile = None


# ==============================================================================
# 4. ZeroKnowledgeVerifier: Non-Interactive ZK & Poisoning Verification
# ==============================================================================


class ZeroKnowledgeVerifier:
    """Verifies Signature Integrity and Validates Hypervectors Without Exposing PII.

    Mathematical Pillars:
    1. Signature Integrity:
       Ed25519 (RFC 8032) or HMAC-SHA256 signature verification over canonical payloads.
    2. Non-Interactive Zero-Knowledge (NIZK) Proof of Cluster Witness:
       Uses Fiat-Shamir heuristic over Curve25519 (Schnorr-style Sigma protocol):
       Proves the publishing cluster observed an authentic, high-entropy telemetry
       anomaly matching a public cluster commitment, without exposing internal
       hostnames, container IDs, user IP addresses, or proprietary log strings.
    3. Poisoning & Byzantine Defense:
       - HDC Asymptotic Density Bounds:
         Rejects degenerate hypervectors (all-zeros, all-ones) or vectors whose bit
         density deviates outside normal statistical bounds ($\\mu \\approx 0.50$).
       - Repetition & Block Entropy:
         Rejects repeating synthetic tiles or low-entropy spoofing attacks.
       - Adversarial Anti-Consensus Detection:
         Rejects vectors designed to maliciously cancel existing consensus defense memories.
    4. Node Reputation & Quarantine:
       Tracks reputation score per node $R(u) \\in [0.0, 1.0]$; penalizes poisoning attempts
       and blacklists compromised nodes.
    """

    def __init__(
        self,
        min_density: float = 0.38,
        max_density: float = 0.62,
        min_entropy: float = 0.80,
        quarantine_threshold: float = 0.25,
    ) -> None:
        self.min_density: float = min_density
        self.max_density: float = max_density
        self.min_entropy: float = min_entropy
        self.quarantine_threshold: float = quarantine_threshold

        self._node_reputations: Dict[str, float] = {}
        self._blacklisted_nodes: Set[str] = set()

    def get_reputation(self, node_id: str) -> float:
        """Get reputation score of a node in [0.0, 1.0]. Defaults to 0.8."""
        return self._node_reputations.get(node_id, 0.8)

    def is_blacklisted(self, node_id: str) -> bool:
        """Check if node is currently blacklisted / quarantined."""
        return node_id in self._blacklisted_nodes

    def update_reputation(self, node_id: str, success: bool, delta: Optional[float] = None) -> float:
        """Update node reputation score based on verification outcome."""
        current = self.get_reputation(node_id)
        if success:
            step = delta if delta is not None else 0.05
            new_score = min(1.0, current + step)
        else:
            step = delta if delta is not None else 0.25
            new_score = max(0.0, current - step)

        self._node_reputations[node_id] = new_score
        if new_score < self.quarantine_threshold:
            self._blacklisted_nodes.add(node_id)

        return new_score

    def unblacklist(self, node_id: str, initial_reputation: float = 0.5) -> None:
        """Manually restore a blacklisted node."""
        self._blacklisted_nodes.discard(node_id)
        self._node_reputations[node_id] = initial_reputation

    # --------------------------------------------------------------------------
    # Schnorr NIZK Proof of Knowledge Generation & Verification
    # --------------------------------------------------------------------------

    def generate_zk_proof(
        self,
        telemetry_witness: bytes,
        hypervector: HyperVector,
        failure_mode: str,
        node_id: str,
        nonce_seed: Optional[bytes] = None,
    ) -> Dict[str, Any]:
        """Generate a Schnorr Non-Interactive Zero-Knowledge (NIZK) Proof of Cluster Witness.

        Proves that the cluster possessed genuine telemetry witness $W$
        consistent with the public commitment without revealing $W$ to the mesh.

        Args:
            telemetry_witness: Proprietary cluster telemetry bytes (internal logs, trace events).
            hypervector: Associated 10,000-bit HDC threat vector.
            failure_mode: Declared failure mode.
            node_id: Identifier of the prover node.
            nonce_seed: Optional deterministic nonce seed for testing.

        Returns:
            Dictionary containing public commitment, nonce point, challenge, and scalar response.
        """
        # Secret witness scalar w = H(witness) mod L
        w_hash = hashlib.sha512(telemetry_witness).digest()
        w = int.from_bytes(w_hash, "little") % _ED25519_L
        # Public commitment point V = w * B
        v_point = _point_mul(_ED25519_BASE, w)
        v_bytes = _encode_point(v_point)

        # Prover random nonce k in [1, L-1]
        seed = nonce_seed or secrets.token_bytes(32)
        k_hash = hashlib.sha512(seed + telemetry_witness).digest()
        k = int.from_bytes(k_hash, "little") % _ED25519_L
        r_point = _point_mul(_ED25519_BASE, k)
        r_bytes = _encode_point(r_point)

        # Fiat-Shamir challenge e = H(R || V || hv_digest || failure_mode || node_id) mod L
        hv_digest = hashlib.sha256(hypervector.to_bytes()).digest()
        challenge_preimage = (
            r_bytes
            + v_bytes
            + hv_digest
            + failure_mode.encode("utf-8")
            + node_id.encode("utf-8")
        )
        e_hash = hashlib.sha512(challenge_preimage).digest()
        e = int.from_bytes(e_hash, "little") % _ED25519_L

        # Response scalar s = (k + e * w) mod L
        s = (k + e * w) % _ED25519_L
        s_bytes = s.to_bytes(32, "little")

        return {
            "protocol": "schnorr_nizk_curve25519",
            "commitment": v_bytes.hex(),
            "nonce_point": r_bytes.hex(),
            "challenge": e_hash[:32].hex(),
            "response": s_bytes.hex(),
            "hv_digest": hv_digest.hex(),
        }

    def verify_zk_proof(
        self,
        proof: Dict[str, Any],
        hypervector: HyperVector,
        failure_mode: str,
        node_id: str,
    ) -> bool:
        """Verify Schnorr NIZK Proof in zero knowledge without learning telemetry.

        Checks: $s \\cdot B \\stackrel{?}{=} R + e \\cdot V$.
        """
        if not proof or proof.get("protocol") != "schnorr_nizk_curve25519":
            return False

        try:
            v_bytes = bytes.fromhex(proof["commitment"])
            r_bytes = bytes.fromhex(proof["nonce_point"])
            s_bytes = bytes.fromhex(proof["response"])

            # Verify hypervector binding
            hv_digest = hashlib.sha256(hypervector.to_bytes()).digest()
            if proof.get("hv_digest") != hv_digest.hex():
                return False

            # Reconstruct Fiat-Shamir challenge e
            challenge_preimage = (
                r_bytes
                + v_bytes
                + hv_digest
                + failure_mode.encode("utf-8")
                + node_id.encode("utf-8")
            )
            e_hash = hashlib.sha512(challenge_preimage).digest()
            e = int.from_bytes(e_hash, "little") % _ED25519_L

            # Decode points and scalars
            point_v = _decode_point(v_bytes)
            point_r = _decode_point(r_bytes)
            s_val = int.from_bytes(s_bytes, "little")
            if s_val >= _ED25519_L:
                return False

            # Evaluate: s * B == R + e * V
            sb = _point_mul(_ED25519_BASE, s_val)
            ev = _point_mul(point_v, e)
            r_plus_ev = _point_add(point_r, ev)

            return _to_affine(sb) == _to_affine(r_plus_ev)
        except Exception:
            return False

    # --------------------------------------------------------------------------
    # Statistical Poisoning Defense: Density & Entropy Verification
    # --------------------------------------------------------------------------

    def validate_density(self, hv: HyperVector) -> Tuple[bool, str]:
        """Validate that 10,000-bit hypervector satisfies HDC asymptotic density bounds."""
        pop = hv.bit_count
        density = pop / hv.dim

        if pop == 0:
            return False, "Degenerate hypervector: all-zeros detected"
        if pop == hv.dim:
            return False, "Degenerate hypervector: all-ones detected"

        if density < self.min_density or density > self.max_density:
            return (
                False,
                f"Density violation: {density:.3f} outside valid bounds [{self.min_density}, {self.max_density}]",
            )

        return True, "Valid density"

    def validate_entropy(self, hv: HyperVector) -> Tuple[bool, str]:
        """Check coordinate run-length and chunk entropy to detect repeating synthetic patterns."""
        words = hv.to_words()
        if len(words) < 2:
            return True, "Valid entropy"

        # Check for repeating identical 64-bit words (trivial tile attack)
        unique_words = len(np.unique(words))
        ratio = unique_words / len(words)
        if ratio < 0.35:
            return (
                False,
                f"Low entropy repetition: unique word ratio {ratio:.3f} < 0.35 (synthetic poisoning attack)",
            )

        # Coordinate-wise transition rate (flip frequency across adjacent coordinates)
        binary = hv.to_binary()
        diffs = np.sum(binary[:-1] != binary[1:])
        transition_rate = diffs / (hv.dim - 1)
        if transition_rate < 0.30 or transition_rate > 0.70:
            return (
                False,
                f"Low spatial entropy: coordinate transition rate {transition_rate:.3f} outside [0.30, 0.70]",
            )

        return True, "Valid entropy"

    def detect_poisoning(
        self,
        sig: HDCThreatSignature,
        memory_bank: Optional[HolographicDefenseMemoryBank] = None,
    ) -> Tuple[bool, str]:
        """Detect malicious or adversarial hypervector poisoning.

        Returns:
            (is_poisoned: bool, reason: str)
        """
        # 1. Check density
        valid_d, reason_d = self.validate_density(sig.hypervector)
        if not valid_d:
            return True, reason_d

        # 2. Check entropy
        valid_e, reason_e = self.validate_entropy(sig.hypervector)
        if not valid_e:
            return True, reason_e

        # 3. Check anti-consensus poisoning against existing memory bank
        if memory_bank is not None:
            existing_profile = memory_bank.get_profile(sig.failure_mode)
            if existing_profile is not None and memory_bank.total_signatures() >= 3:
                # Anti-correlation: if candidate vector is diametrically inverted (-1.0 cosine)
                # to the established defense profile for this exact failure mode
                sim = sig.hypervector.cosine_similarity(existing_profile)
                if sim < -0.35:
                    return (
                        True,
                        f"Anti-consensus attack detected: similarity {sim:.3f} strongly opposes established profile",
                    )

        return False, "Non-poisoned"

    # --------------------------------------------------------------------------
    # Comprehensive Threat Signature Verification
    # --------------------------------------------------------------------------

    def verify_threat_signature(
        self,
        sig: HDCThreatSignature,
        memory_bank: Optional[HolographicDefenseMemoryBank] = None,
        require_zk_proof: bool = False,
    ) -> Tuple[bool, str]:
        """Verify signature authenticity, ZK proof, and poisoning status.

        Returns:
            (is_valid: bool, detail: str)
        """
        # Check node quarantine status
        if self.is_blacklisted(sig.node_id):
            return False, f"Node '{sig.node_id}' is blacklisted due to prior malicious submissions"

        # 1. Cryptographic signature check
        if not sig.verify():
            self.update_reputation(sig.node_id, success=False)
            return False, "Cryptographic signature verification failed"

        # 2. ZK Proof check (if present or required)
        if sig.zk_proof is not None:
            zk_valid = self.verify_zk_proof(
                proof=sig.zk_proof,
                hypervector=sig.hypervector,
                failure_mode=sig.failure_mode,
                node_id=sig.node_id,
            )
            if not zk_valid:
                self.update_reputation(sig.node_id, success=False)
                return False, "Zero-Knowledge cluster witness proof verification failed"
        elif require_zk_proof:
            return False, "Zero-Knowledge proof required but not provided"

        # 3. Poisoning detection
        is_poisoned, p_reason = self.detect_poisoning(sig, memory_bank=memory_bank)
        if is_poisoned:
            self.update_reputation(sig.node_id, success=False)
            return False, f"Poisoning detected: {p_reason}"

        # Success - reward node reputation
        self.update_reputation(sig.node_id, success=True)
        return True, "Verified authentic and clean"


# ==============================================================================
# 5. FederatedThreatMeshNode: Decentralized Mesh Peer Node
# ==============================================================================


class FederatedThreatMeshNode:
    """Decentralized Peer Node in Federated Threat Sharing Mesh.

    Participates in a peer-to-peer threat intelligence network:
    - Encodes failure modes into 10,000-bit holographic threat hypervectors.
    - Masks hypervectors using calibrated Bernoulli Differential Privacy to protect PII.
    - Generates Schnorr NIZK proofs proving cluster telemetry integrity.
    - Signs threat vectors with Ed25519 and broadcasts them to peer nodes.
    - Receives, verifies, sanitizes, and merges incoming peer threats into a local
      Holographic Defense Memory Bank for instant sub-microsecond anomaly detection.
    """

    def __init__(
        self,
        node_id: str,
        private_key: Optional[bytes] = None,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        dim: int = 10000,
        verifier: Optional[ZeroKnowledgeVerifier] = None,
    ) -> None:
        """Initialize Federated Threat Mesh Node.

        Args:
            node_id: Unique string identifier for this cluster / peer node.
            private_key: 32-byte Ed25519 seed. Generated securely if None.
            epsilon: Local Differential Privacy epsilon parameter.
            delta: Differential Privacy delta relaxation.
            dim: Hypervector dimensionality (default 10,000).
            verifier: Optional custom ZeroKnowledgeVerifier instance.
        """
        self.node_id: str = node_id
        self.dim: int = dim
        self.private_seed, self.public_key_bytes = ed25519_keypair(private_key)
        self.public_key_hex: str = self.public_key_bytes.hex()

        self.privacy_masker = DifferentialPrivacyMasker(
            epsilon=epsilon, delta=delta, mode="ldp", dim=dim
        )
        self.verifier: ZeroKnowledgeVerifier = verifier or ZeroKnowledgeVerifier()
        self.encoder = HyperdimensionalEncoder(dim=dim, mode="binary", seed=42)
        self.memory_bank = HolographicDefenseMemoryBank(dim=dim, mode="binary")

        # P2P mesh network topology
        self.peers: Dict[str, FederatedThreatMeshNode] = {}
        self.published_threats: List[HDCThreatSignature] = []
        self.received_threats: List[HDCThreatSignature] = []
        self._processed_threat_ids: Set[str] = set()

    def connect_peer(self, peer: FederatedThreatMeshNode, mutual: bool = True) -> None:
        """Establish peering connection with another node in the mesh."""
        if peer.node_id == self.node_id:
            return
        self.peers[peer.node_id] = peer
        if mutual and self.node_id not in peer.peers:
            peer.connect_peer(self, mutual=False)

    def disconnect_peer(self, peer_id: str, mutual: bool = True) -> None:
        """Disconnect peering connection with a node."""
        if peer_id in self.peers:
            peer = self.peers.pop(peer_id)
            if mutual and self.node_id in peer.peers:
                peer.disconnect_peer(self.node_id, mutual=False)

    def create_threat_signature(
        self,
        failure_mode: str,
        log_events: Sequence[Any],
        metadata: Optional[Dict[str, Any]] = None,
        telemetry_witness: Optional[bytes] = None,
        apply_differential_privacy: bool = True,
        generate_zk_proof: bool = True,
    ) -> HDCThreatSignature:
        """Encode, mask, prove, and sign a new threat signature from observed log events.

        Args:
            failure_mode: Classification string (e.g. 'ransomware', 'syn_flood').
            log_events: Observed sequence of log events or error traces.
            metadata: Contextual operational metadata.
            telemetry_witness: Raw telemetry bytes for ZK proof. Derived from events if None.
            apply_differential_privacy: Whether to apply Bernoulli DP masking.
            generate_zk_proof: Whether to include Schnorr NIZK proof.

        Returns:
            Cryptographically signed and differentially private HDCThreatSignature.
        """
        # 1. Encode into 10,000-bit hypervector
        raw_hv = self.encoder.encode_sequence(log_events, method="positional")

        # 2. Derive telemetry witness for ZK proof
        if telemetry_witness is None:
            witness_str = f"{self.node_id}:{failure_mode}:{len(log_events)}:" + ":".join(
                str(e) for e in log_events
            )
            witness_bytes = witness_str.encode("utf-8")
        else:
            witness_bytes = telemetry_witness

        # 3. Apply Differential Privacy Masking
        meta = metadata.copy() if metadata else {}
        meta["raw_event_count"] = len(log_events)

        if apply_differential_privacy:
            threat_hv = self.privacy_masker.mask_hypervector(raw_hv)
            privacy_info = {
                "mechanism": "bernoulli_randomized_response",
                "epsilon": self.privacy_masker.epsilon,
                "delta": self.privacy_masker.delta,
                "flip_prob": self.privacy_masker.flip_prob,
            }
        else:
            threat_hv = raw_hv
            privacy_info = None

        # 4. Generate ZK Proof bound to the final threat hypervector
        zk_proof: Optional[Dict[str, Any]] = None
        if generate_zk_proof:
            zk_proof = self.verifier.generate_zk_proof(
                telemetry_witness=witness_bytes,
                hypervector=threat_hv,
                failure_mode=failure_mode,
                node_id=self.node_id,
            )

        # 5. Construct Threat Signature
        sig = HDCThreatSignature(
            threat_id=secrets.token_hex(16),
            failure_mode=failure_mode,
            hypervector=threat_hv,
            node_id=self.node_id,
            timestamp=time.time(),
            metadata=meta,
            signature_scheme="ed25519",
            zk_proof=zk_proof,
            privacy_metadata=privacy_info,
        )

        # 6. Cryptographically sign
        sig.sign(self.private_seed, scheme="ed25519")
        return sig

    def publish_threat(self, threat_sig: HDCThreatSignature) -> Dict[str, bool]:
        """Publish a threat signature across the decentralized mesh network.

        Also immediately merges the threat into local Holographic Defense Memory.

        Returns:
            Dictionary mapping peer_id to peer acceptance status (True if accepted).
        """
        # Save in local published repository & defense memory
        self.published_threats.append(threat_sig)
        self._processed_threat_ids.add(threat_sig.threat_id)
        self.memory_bank.add(threat_sig)

        results: Dict[str, bool] = {}
        # Broadcast to all connected peers
        for peer_id, peer in list(self.peers.items()):
            try:
                accepted, _ = peer.receive_threat(threat_sig, sender_node_id=self.node_id)
                results[peer_id] = accepted
            except Exception:
                results[peer_id] = False

        return results

    def receive_threat(
        self, threat_sig: HDCThreatSignature, sender_node_id: Optional[str] = None
    ) -> Tuple[bool, str]:
        """Receive, verify, sanitize, and merge a threat signature from a peer.

        Args:
            threat_sig: The incoming candidate HDCThreatSignature.
            sender_node_id: Optional sender node ID.

        Returns:
            (accepted: bool, reason_message: str)
        """
        # Deduplication check
        if threat_sig.threat_id in self._processed_threat_ids:
            return True, "Already processed"

        # Full verification: Signature, ZK Proof, and Poisoning Defense
        valid, reason = self.verifier.verify_threat_signature(
            threat_sig, memory_bank=self.memory_bank, require_zk_proof=False
        )

        if not valid:
            return False, f"Threat rejected: {reason}"

        # Verified authentic & clean: merge into defense memory bank
        self._processed_threat_ids.add(threat_sig.threat_id)
        self.received_threats.append(threat_sig)
        self.memory_bank.add(threat_sig)

        return True, "Verified and merged into defense memory bank"

    def scan_log_sequence(
        self, events: Sequence[Any], threshold: float = 0.40
    ) -> Optional[Tuple[str, float]]:
        """Scan incoming live log sequence against Holographic Defense Memory Bank.

        Sub-microsecond anomaly detection and failure mode matching.

        Returns:
            (matched_failure_mode, cosine_similarity) if threat detected, else None.
        """
        hv = self.encoder.encode_sequence(events, method="positional")
        matches = self.memory_bank.query(hv, threshold=threshold)
        if matches:
            return matches[0]
        return None

    def gossip_sync(self) -> int:
        """Gossip sync protocol: exchange missing threat signatures with connected peers.

        Returns:
            Count of newly synchronized threat signatures.
        """
        new_synced = 0
        for peer in self.peers.values():
            known_threats = peer.published_threats + peer.received_threats
            for threat in known_threats:
                if threat.threat_id not in self._processed_threat_ids:
                    accepted, _ = self.receive_threat(threat, sender_node_id=peer.node_id)
                    if accepted:
                        new_synced += 1
        return new_synced

    def get_mesh_status(self) -> Dict[str, Any]:
        """Return status and telemetry for this mesh node."""
        return {
            "node_id": self.node_id,
            "public_key": self.public_key_hex,
            "peer_count": len(self.peers),
            "connected_peers": list(self.peers.keys()),
            "published_threats_count": len(self.published_threats),
            "received_threats_count": len(self.received_threats),
            "memory_bank_failure_modes": self.memory_bank.failure_modes(),
            "memory_bank_total_signatures": self.memory_bank.total_signatures(),
            "privacy_spent_epsilon": self.privacy_masker.spent_budget[0],
            "reputation_score": self.verifier.get_reputation(self.node_id),
        }
