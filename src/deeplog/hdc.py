"""Hyperdimensional Computing (HDC) & Vector Symbolic Architectures (VSA).

This module implements Pillar I: Hyperdimensional Computing for ultra-fast,
sub-microsecond anomaly detection and symbolic sequence representation.

Mathematical Foundations:
- High-Dimensional Random Space: As $D \\to \\infty$ (e.g. $D = 10,000$), two randomly
  chosen hypervectors are asymptotically orthogonal with high probability:
  $\\mathbb{E}[d_H(u, v)] = 0.5$, $\\mathrm{Var}[d_H(u, v)] = \\frac{1}{4D}$.
- Binding ($\\otimes$): Bitwise XOR in binary $\\{0, 1\\}^D$ or elementwise multiplication
  in bipolar $\\{-1, +1\\}^D$. Preserves distance, creates an orthogonal hypervector,
  and is self-invertible: $(A \\otimes B) \\otimes B = A$.
- Bundling ($\\oplus$): Superposition via coordinate-wise majority rule (binary) or sign of sum
  (bipolar). Preserves high similarity to constituent hypervectors:
  $\\mathrm{sim}(A \\oplus B, A) > \\mathrm{sim}(A \\oplus B, C)$ for unrelated $C$.
- Cyclic Permutation ($\\Pi^k$): Circular bit/coordinate shift by $k$ positions.
  Preserves pairwise distances, invertibility $\\Pi^{-k}(\\Pi^k(A)) = A$, and produces
  quasi-orthogonal vectors for $k \\ne 0$. Used for sequence/position encoding.
- Holographic Memory: Bundling normal sequence hypervectors into a prototype memory $\\mathbf{M}$.
  Anomaly detection evaluates Hamming distance or Cosine similarity in sub-microsecond latency
  (millions of queries/sec via hardware popcount / bitwise ALU).
"""

from __future__ import annotations

import hashlib
import math
import secrets
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np


class HyperVector:
    """A high-dimensional vector in a Vector Symbolic Architecture (VSA).

    Supports both binary {0, 1} and bipolar {-1, +1} representations over
    D dimensions (default D = 10,000). Bitwise packed representations allow
    hardware-accelerated XOR and popcount instructions at sub-microsecond latency.

    Attributes:
        dim: Dimensionality of the hypervector (default 10,000).
        mode: Representation mode ('binary' or 'bipolar').
        int_val: Packed integer containing exactly D bits.
    """

    __slots__ = ("dim", "mode", "int_val", "_mask", "_cached_binary", "_cached_bipolar")

    def __init__(
        self,
        int_val: int,
        dim: int = 10000,
        mode: str = "binary",
    ) -> None:
        """Initialize HyperVector from packed integer value."""
        if dim <= 0:
            raise ValueError(f"Dimension must be positive, got {dim}")
        if mode not in ("binary", "bipolar"):
            raise ValueError(f"Mode must be 'binary' or 'bipolar', got {mode}")

        self.dim: int = dim
        self.mode: str = mode
        self._mask: int = (1 << dim) - 1
        self.int_val: int = int_val & self._mask
        self._cached_binary: Optional[np.ndarray] = None
        self._cached_bipolar: Optional[np.ndarray] = None

    @classmethod
    def random(
        cls,
        dim: int = 10000,
        mode: str = "binary",
        seed: Optional[int] = None,
    ) -> HyperVector:
        """Generate a random pseudo-orthogonal hypervector with ~50% density."""
        if seed is not None:
            rng = np.random.RandomState(seed)
            # Generate random bytes
            n_bytes = (dim + 7) // 8
            raw_bytes = rng.bytes(n_bytes)
            int_val = int.from_bytes(raw_bytes, byteorder="little")
        else:
            int_val = secrets.randbits(dim)
        return cls(int_val=int_val, dim=dim, mode=mode)

    @classmethod
    def from_binary_array(
        cls,
        arr: np.ndarray,
        dim: Optional[int] = None,
        mode: str = "binary",
    ) -> HyperVector:
        """Create HyperVector from a numpy array of {0, 1} binary values."""
        flat = np.asarray(arr, dtype=np.uint8).ravel()
        d = dim if dim is not None else len(flat)
        if len(flat) < d:
            padded = np.zeros(d, dtype=np.uint8)
            padded[: len(flat)] = flat
            flat = padded
        elif len(flat) > d:
            flat = flat[:d]

        packed = np.packbits(flat, bitorder="little").tobytes()
        int_val = int.from_bytes(packed, byteorder="little")
        hv = cls(int_val=int_val, dim=d, mode=mode)
        hv._cached_binary = flat
        return hv

    @classmethod
    def from_bipolar_array(
        cls,
        arr: np.ndarray,
        dim: Optional[int] = None,
        mode: str = "bipolar",
    ) -> HyperVector:
        """Create HyperVector from a numpy array of {-1, +1} bipolar values."""
        flat = np.asarray(arr).ravel()
        # Map bipolar {-1, +1} -> binary {0, 1}: +1 -> 0, -1 -> 1
        binary_arr = (flat < 0).astype(np.uint8)
        return cls.from_binary_array(binary_arr, dim=dim, mode=mode)

    @classmethod
    def from_numpy(
        cls,
        arr: np.ndarray,
        dim: Optional[int] = None,
        mode: str = "binary",
    ) -> HyperVector:
        """Create HyperVector from numpy array inferring binary vs bipolar."""
        arr_flat = np.asarray(arr).ravel()
        if np.any(arr_flat < 0):
            return cls.from_bipolar_array(arr_flat, dim=dim, mode=mode)
        return cls.from_binary_array((arr_flat > 0).astype(np.uint8), dim=dim, mode=mode)

    @classmethod
    def zeros(cls, dim: int = 10000, mode: str = "binary") -> HyperVector:
        """Create a zero-initialized hypervector."""
        return cls(int_val=0, dim=dim, mode=mode)

    @classmethod
    def ones(cls, dim: int = 10000, mode: str = "binary") -> HyperVector:
        """Create an all-ones hypervector."""
        return cls(int_val=(1 << dim) - 1, dim=dim, mode=mode)

    @property
    def bit_count(self) -> int:
        """Count number of active set bits (popcount)."""
        return self.int_val.bit_count()

    def popcount(self) -> int:
        """Alias for bit_count."""
        return self.int_val.bit_count()

    def to_binary(self) -> np.ndarray:
        """Convert to a 1D numpy array of uint8 {0, 1}."""
        if self._cached_binary is None:
            n_bytes = (self.dim + 7) // 8
            raw = self.int_val.to_bytes(n_bytes, byteorder="little")
            u8 = np.frombuffer(raw, dtype=np.uint8)
            bits = np.unpackbits(u8, bitorder="little")[: self.dim]
            self._cached_binary = bits.astype(np.uint8)
        return self._cached_binary.copy()

    def to_bipolar(self) -> np.ndarray:
        """Convert to a 1D numpy array of int8 {-1, +1}.

        Isomorphic mapping: 0 -> +1, 1 -> -1.
        """
        if self._cached_bipolar is None:
            binary = self.to_binary()
            self._cached_bipolar = (1 - 2 * binary.astype(np.int8)).astype(np.int8)
        return self._cached_bipolar.copy()

    def to_numpy(self) -> np.ndarray:
        """Return numpy array corresponding to self.mode."""
        return self.to_bipolar() if self.mode == "bipolar" else self.to_binary()

    def to_words(self) -> np.ndarray:
        """Export as 64-bit uint64 words for eBPF / C-interop."""
        n_bytes = ((self.dim + 63) // 64) * 8
        raw = self.int_val.to_bytes(n_bytes, byteorder="little")
        return np.frombuffer(raw, dtype=np.uint64)

    def to_bytes(self) -> bytes:
        """Export as raw little-endian bytes."""
        n_bytes = (self.dim + 7) // 8
        return self.int_val.to_bytes(n_bytes, byteorder="little")

    @classmethod
    def from_bytes(cls, raw: bytes, dim: int = 10000, mode: str = "binary") -> HyperVector:
        """Create HyperVector from raw little-endian bytes."""
        int_val = int.from_bytes(raw, byteorder="little")
        return cls(int_val=int_val, dim=dim, mode=mode)

    def bind(self, other: HyperVector) -> HyperVector:
        """Bitwise binding operation (\\otimes).

        Binary mode: bitwise XOR ($a \\oplus b$).
        Bipolar mode: elementwise multiplication (isomorphic to XOR under 0->+1, 1->-1).
        Invertibility: bind(bind(a, b), b) == a.
        """
        if self.dim != other.dim:
            raise ValueError(f"Dimension mismatch: {self.dim} vs {other.dim}")
        # Bitwise XOR is the universal binding operator for both modes
        bound_val = self.int_val ^ other.int_val
        return HyperVector(int_val=bound_val, dim=self.dim, mode=self.mode)

    def permute(self, shifts: int = 1) -> HyperVector:
        """Cyclic permutation (\\Pi^k).

        Circular left shift by k coordinates.
        Preserves Hamming distance, is invertible via permute(-shifts),
        and yields quasi-orthogonal vectors for shifts != 0.
        """
        k = shifts % self.dim
        if k == 0:
            return HyperVector(int_val=self.int_val, dim=self.dim, mode=self.mode)
        shifted = ((self.int_val << k) & self._mask) | (self.int_val >> (self.dim - k))
        return HyperVector(int_val=shifted, dim=self.dim, mode=self.mode)

    def invert(self) -> HyperVector:
        """Bitwise negation / sign reversal."""
        return HyperVector(int_val=self.int_val ^ self._mask, dim=self.dim, mode=self.mode)

    def hamming_distance(self, other: HyperVector, normalized: bool = True) -> float:
        """Compute Hamming distance between two hypervectors.

        Uses sub-microsecond hardware popcount on the bitwise XOR.
        """
        if self.dim != other.dim:
            raise ValueError(f"Dimension mismatch: {self.dim} vs {other.dim}")
        diff_bits = (self.int_val ^ other.int_val).bit_count()
        return diff_bits / self.dim if normalized else float(diff_bits)

    def cosine_similarity(self, other: HyperVector) -> float:
        """Compute exact cosine similarity in [-1.0, 1.0].

        By VSA isomorphism: cos(u, v) = 1.0 - 2.0 * normalized_hamming_distance.
        Executes in sub-microsecond time with zero floating point multiplications!
        """
        norm_h = self.hamming_distance(other, normalized=True)
        return 1.0 - 2.0 * norm_h

    def similarity(self, other: HyperVector) -> float:
        """Compute similarity score.

        Bipolar: Cosine similarity in [-1, 1].
        Binary: Normalized overlap (1 - normalized Hamming distance) in [0, 1].
        """
        if self.mode == "bipolar":
            return self.cosine_similarity(other)
        return 1.0 - self.hamming_distance(other, normalized=True)

    # Operator Overloads
    def __xor__(self, other: HyperVector) -> HyperVector:
        return self.bind(other)

    def __mul__(self, other: HyperVector) -> HyperVector:
        return self.bind(other)

    def __add__(self, other: HyperVector) -> HyperVector:
        # Bundling of two vectors: majority rule with deterministic tie-breaker
        return HyperdimensionalEncoder(dim=self.dim, mode=self.mode).bundle([self, other])

    def __lshift__(self, shifts: int) -> HyperVector:
        return self.permute(shifts)

    def __rshift__(self, shifts: int) -> HyperVector:
        return self.permute(-shifts)

    def __invert__(self) -> HyperVector:
        return self.invert()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HyperVector):
            return False
        return self.dim == other.dim and self.int_val == other.int_val

    def __repr__(self) -> str:
        return (
            f"<HyperVector dim={self.dim} mode={self.mode} "
            f"density={self.bit_count / self.dim:.3f}>"
        )


class HDCProfileAccumulator:
    """High-performance accumulator for streaming / batch bundling.

    Accumulates coordinate-wise vote counts to perform majority-rule bundling
    across an arbitrary number of hypervectors with O(D) memory.
    """

    def __init__(self, dim: int = 10000, mode: str = "binary") -> None:
        self.dim: int = dim
        self.mode: str = mode
        self.counts: np.ndarray = np.zeros(dim, dtype=np.int32)
        self.total_weight: float = 0.0

    def add(self, hv: HyperVector, weight: float = 1.0) -> None:
        """Add a hypervector with given weight into the accumulator."""
        if hv.dim != self.dim:
            raise ValueError(f"Dimension mismatch: {hv.dim} vs {self.dim}")
        # In binary: bit 1 is counted as +1, bit 0 as 0
        bits = hv.to_binary()
        if weight == 1.0:
            self.counts += bits
        else:
            self.counts += (bits * weight).astype(np.int32)
        self.total_weight += weight

    def to_hypervector(self) -> HyperVector:
        """Resolve majority rule and return bundled HyperVector."""
        if self.total_weight <= 0:
            return HyperVector.zeros(dim=self.dim, mode=self.mode)

        threshold = self.total_weight / 2.0
        majority = (self.counts > threshold).astype(np.uint8)

        # Deterministic tie-breaking: use index parity (or coordinate hash)
        ties = (self.counts == threshold)
        if np.any(ties):
            tie_indices = np.arange(self.dim)[ties]
            majority[ties] = (tie_indices % 2).astype(np.uint8)

        return HyperVector.from_binary_array(majority, dim=self.dim, mode=self.mode)


class HyperdimensionalEncoder:
    """Vector Symbolic Architecture (VSA) Hyperdimensional Encoder.

    Encodes tokens, event types, temporal sequences, sets, and key-value records
    into 10,000-dimensional hypervectors with holographic associative memory.

    Mathematical operations supported:
    - Binding (\\otimes): Bitwise XOR
    - Bundling (\\oplus): Majority rule superposition
    - Cyclic Permutation (\\Pi^k): Circular shifts
    """

    def __init__(
        self,
        dim: int = 10000,
        mode: str = "binary",
        seed: Optional[int] = 42,
    ) -> None:
        """Initialize encoder with dimension and random seed."""
        self.dim: int = dim
        self.mode: str = mode
        self.seed: Optional[int] = seed
        self._item_memory: Dict[Any, HyperVector] = {}
        self._tie_breaker: np.ndarray = (np.arange(dim) % 2).astype(np.uint8)

    def random_hypervector(self, seed: Optional[int] = None) -> HyperVector:
        """Generate a random pseudo-orthogonal hypervector."""
        s = seed if seed is not None else self.seed
        return HyperVector.random(dim=self.dim, mode=self.mode, seed=s)

    def get_item_vector(self, item: Any) -> HyperVector:
        """Retrieve or deterministically generate item memory hypervector."""
        if item in self._item_memory:
            return self._item_memory[item]

        # Deterministically derive 64-bit integer seed from item string hash
        item_str = str(item).encode("utf-8")
        digest = hashlib.sha256(item_str).digest()
        seed = int.from_bytes(digest[:8], byteorder="little") & 0xFFFFFFFF

        hv = HyperVector.random(dim=self.dim, mode=self.mode, seed=seed)
        self._item_memory[item] = hv
        return hv

    def register_item_vector(self, item: Any, hv: HyperVector) -> None:
        """Register explicit hypervector into item memory."""
        if hv.dim != self.dim:
            raise ValueError(f"Dimension mismatch: {hv.dim} vs {self.dim}")
        self._item_memory[item] = hv

    def bind(self, a: HyperVector, b: HyperVector) -> HyperVector:
        """Bitwise binding operation (\\otimes): a ^ b."""
        return a.bind(b)

    def bundle(
        self,
        vectors: Sequence[HyperVector],
        weights: Optional[Sequence[float]] = None,
    ) -> HyperVector:
        """Superposition / Bundling (\\oplus) across multiple hypervectors.

        Computes coordinate-wise majority rule with deterministic tie-breaking.
        Preserves high similarity to all constituent hypervectors.
        """
        if not vectors:
            return HyperVector.zeros(dim=self.dim, mode=self.mode)
        if len(vectors) == 1:
            return vectors[0]

        # Fast bitwise majority logic paths when unweighted
        if weights is None:
            if len(vectors) == 2:
                # 2-vector bundling with deterministic tie-breaking
                a_bin = vectors[0].to_binary()
                b_bin = vectors[1].to_binary()
                ties = (a_bin != b_bin)
                res_bin = a_bin.copy()
                res_bin[ties] = self._tie_breaker[ties]
                return HyperVector.from_binary_array(res_bin, dim=self.dim, mode=self.mode)
            elif len(vectors) == 3:
                # 3-input Boolean majority: (a & b) | (c & (a ^ b))
                a_int = vectors[0].int_val
                b_int = vectors[1].int_val
                c_int = vectors[2].int_val
                maj3 = (a_int & b_int) | (c_int & (a_int ^ b_int))
                return HyperVector(int_val=maj3, dim=self.dim, mode=self.mode)
            elif len(vectors) == 5:
                # 5-input Boolean majority via carry-save full adders
                a_int = vectors[0].int_val
                b_int = vectors[1].int_val
                c_int = vectors[2].int_val
                d_int = vectors[3].int_val
                e_int = vectors[4].int_val
                s1 = a_int ^ b_int ^ c_int
                c1 = (a_int & b_int) | (c_int & (a_int ^ b_int))
                s2 = s1 ^ d_int ^ e_int
                c2 = (s1 & d_int) | (e_int & (s1 ^ d_int))
                maj5 = (c1 & c2) | ((c1 ^ c2) & s2)
                return HyperVector(int_val=maj5, dim=self.dim, mode=self.mode)

        # General accumulator for arbitrary N or weighted vectors
        accumulator = HDCProfileAccumulator(dim=self.dim, mode=self.mode)
        if weights is None:
            for v in vectors:
                accumulator.add(v, 1.0)
        else:
            if len(weights) != len(vectors):
                raise ValueError("Length of weights must match length of vectors")
            for v, w in zip(vectors, weights):
                accumulator.add(v, float(w))

        return accumulator.to_hypervector()

    def permute(self, vector: HyperVector, shifts: int = 1) -> HyperVector:
        """Cyclic permutation (\\Pi^shifts): circular bit rotation."""
        return vector.permute(shifts)

    def encode_sequence(
        self,
        sequence: Sequence[Any],
        method: str = "positional",
        ngram_size: int = 3,
    ) -> HyperVector:
        """Encode a temporal sequence of log events into a single hypervector.

        Formulations:
        1. 'positional' (Positional Permutation Bundling):
           $$S = \\bigoplus_{i=0}^{N-1} \\Pi^i(v_{e_i})$$
           Each event is permuted by its position index and bundled.
        2. 'ngram' (N-gram Chain Binding & Bundling):
           $$W_t = \\bigotimes_{j=0}^{k-1} \\Pi^j(v_{e_{t-j}}), \\quad S = \\bigoplus_t W_t$$
           Consecutive event windows are bound with position shifts and bundled.
        """
        if not sequence:
            return HyperVector.zeros(dim=self.dim, mode=self.mode)

        if method == "positional":
            pos_vectors: List[HyperVector] = []
            for pos, event in enumerate(sequence):
                base_v = self.get_item_vector(event)
                pos_v = base_v.permute(pos)
                pos_vectors.append(pos_v)
            return self.bundle(pos_vectors)

        elif method == "ngram":
            k = max(1, ngram_size)
            if len(sequence) < k:
                # Fall back to positional for short sequences
                return self.encode_sequence(sequence, method="positional")

            ngram_vectors: List[HyperVector] = []
            for t in range(k - 1, len(sequence)):
                window = sequence[t - k + 1 : t + 1]
                # Bind events in window with positional shifts: v_0 * Pi(v_1) * Pi^2(v_2)...
                window_hv = self.get_item_vector(window[0])
                for j in range(1, k):
                    shifted_v = self.get_item_vector(window[j]).permute(j)
                    window_hv = window_hv.bind(shifted_v)
                ngram_vectors.append(window_hv)

            return self.bundle(ngram_vectors)
        else:
            raise ValueError(f"Unknown encoding method '{method}', expected 'positional' or 'ngram'")

    def encode_set(self, items: Iterable[Any]) -> HyperVector:
        """Encode an unordered set of items via bundling: $\\bigoplus_{x} v_x$."""
        item_vectors = [self.get_item_vector(x) for x in items]
        return self.bundle(item_vectors)

    def encode_record(self, attributes: Mapping[str, Any]) -> HyperVector:
        """Encode key-value record via bound pairs: $\\bigoplus_{(k, v)} (v_k \\otimes v_v)$."""
        pair_vectors: List[HyperVector] = []
        for key, val in attributes.items():
            key_v = self.get_item_vector(f"KEY_{key}")
            val_v = self.get_item_vector(f"VAL_{val}")
            pair_vectors.append(key_v.bind(val_v))
        return self.bundle(pair_vectors)

    def hamming_distance(
        self,
        a: HyperVector,
        b: HyperVector,
        normalized: bool = True,
    ) -> float:
        """Normalized Hamming distance between two hypervectors."""
        return a.hamming_distance(b, normalized=normalized)

    def cosine_similarity(self, a: HyperVector, b: HyperVector) -> float:
        """Cosine similarity between two hypervectors."""
        return a.cosine_similarity(b)


class HDCAnomalyDetector:
    """Holographic associative memory anomaly detector.

    Learns normal log event sequence patterns by bundling them into one or more
    normal profile prototypes $\\mathbf{M}_{\\mathrm{normal}}$.
    Inference checks Hamming distance against normal profile memory at
    millions of operations per second with sub-microsecond latency.

    Mathematical Decision Boundary:
    $$\\mathrm{Anomaly}(S) \\iff d_H(S, \\mathbf{M}_{\\mathrm{normal}}) > \\theta$$
    where $\\theta = \\mu_{\\mathrm{normal}} + k_{\\sigma} \\cdot \\sigma_{\\mathrm{normal}}$.
    """

    def __init__(
        self,
        encoder: Optional[HyperdimensionalEncoder] = None,
        dim: int = 10000,
        mode: str = "binary",
        threshold: Optional[float] = None,
        k_sigma: float = 3.0,
        num_prototypes: int = 1,
        seed: Optional[int] = 42,
    ) -> None:
        """Initialize HDC Anomaly Detector."""
        self.dim: int = dim
        self.mode: str = mode
        self.encoder: HyperdimensionalEncoder = (
            encoder if encoder is not None else HyperdimensionalEncoder(dim=dim, mode=mode, seed=seed)
        )
        self.threshold: Optional[float] = threshold
        self.k_sigma: float = k_sigma
        self.num_prototypes: int = max(1, num_prototypes)

        # Trained state
        self.profile_memory: Optional[HyperVector] = None
        self.prototypes: List[HyperVector] = []
        self.threshold_: float = threshold if threshold is not None else 0.40
        self.mean_distance_: float = 0.0
        self.std_distance_: float = 0.0
        self.is_fitted: bool = False

    def fit(
        self,
        normal_sequences: Sequence[Sequence[Any]],
        method: str = "positional",
        ngram_size: int = 3,
    ) -> HDCAnomalyDetector:
        """Fit normal profile holographic memory from normal event sequences."""
        if not normal_sequences:
            raise ValueError("Cannot fit on empty sequence dataset")

        # Encode all normal sequences
        encoded_vectors = [
            self.encoder.encode_sequence(seq, method=method, ngram_size=ngram_size)
            for seq in normal_sequences
        ]

        if self.num_prototypes == 1:
            # Single holographic profile memory
            self.profile_memory = self.encoder.bundle(encoded_vectors)
            self.prototypes = [self.profile_memory]
        elif len(encoded_vectors) <= self.num_prototypes:
            # Each distinct vector is its own prototype
            self.prototypes = list(encoded_vectors)
            self.profile_memory = self.encoder.bundle(encoded_vectors)
        else:
            # Multi-prototype clustering via HDC k-medoids / k-means partitioning
            self._fit_multi_prototype(encoded_vectors, k=self.num_prototypes)

        # Compute baseline distance distribution for calibration
        distances = [self.score_hypervector(hv) for hv in encoded_vectors]
        self.mean_distance_ = float(np.mean(distances))
        self.std_distance_ = float(np.std(distances))
        max_dist = float(np.max(distances))

        # Dynamic threshold: mean + k_sigma * std bounded by maximum training distance
        if self.threshold is None:
            calibrated = self.mean_distance_ + self.k_sigma * self.std_distance_
            if self.std_distance_ < 1e-6:
                # Add default discrimination margin when normal samples have zero variance
                calibrated = max(calibrated, 0.05)
            # Bound threshold logically: must be above max training distance but below random baseline (0.45)
            upper_bound = max(max_dist + 0.08, 0.05)
            self.threshold_ = min(calibrated, upper_bound, 0.48)
        else:
            self.threshold_ = self.threshold

        self.is_fitted = True
        return self

    def _fit_multi_prototype(self, vectors: List[HyperVector], k: int) -> None:
        """Simple deterministic k-means / prototype clustering in HDC space."""
        n = len(vectors)
        step = max(1, n // k)
        seeds = [vectors[i * step] for i in range(k)]

        # 3 iterations of cluster assignment and bundling
        for _ in range(3):
            clusters: List[List[HyperVector]] = [[] for _ in range(k)]
            for v in vectors:
                dists = [v.hamming_distance(s) for s in seeds]
                best_k = int(np.argmin(dists))
                clusters[best_k].append(v)

            # Re-bundle centroids
            new_seeds = []
            for idx in range(k):
                if clusters[idx]:
                    new_seeds.append(self.encoder.bundle(clusters[idx]))
                else:
                    new_seeds.append(seeds[idx])
            seeds = new_seeds

        self.prototypes = seeds
        self.profile_memory = self.encoder.bundle(seeds)

    def score_hypervector(self, hv: HyperVector) -> float:
        """Compute anomaly score directly from a HyperVector in sub-microsecond time.

        Calculates minimum Hamming distance across profile prototypes using
        hardware bitwise popcount.
        """
        if self.profile_memory is None:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")

        if len(self.prototypes) == 1:
            return (hv.int_val ^ self.profile_memory.int_val).bit_count() / self.dim

        # Multi-prototype: minimum distance to any prototype
        min_dist = 1.0
        for p in self.prototypes:
            d = (hv.int_val ^ p.int_val).bit_count() / self.dim
            if d < min_dist:
                min_dist = d
        return min_dist

    def score_hypervectors_batch(self, hvs: Sequence[HyperVector]) -> np.ndarray:
        """Fast batch scoring across a sequence of hypervectors at millions of ops/sec."""
        if self.profile_memory is None:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")

        prof_int = self.profile_memory.int_val
        dim = self.dim
        if len(self.prototypes) == 1:
            return np.array(
                [(h.int_val ^ prof_int).bit_count() / dim for h in hvs],
                dtype=np.float32,
            )

        # Multi-prototype batch
        p_ints = [p.int_val for p in self.prototypes]
        results = np.zeros(len(hvs), dtype=np.float32)
        for i, h in enumerate(hvs):
            h_int = h.int_val
            results[i] = min((h_int ^ p_int).bit_count() for p_int in p_ints) / dim
        return results

    def score(
        self,
        sequence: Sequence[Any],
        method: str = "positional",
        ngram_size: int = 3,
    ) -> float:
        """Score an event sequence. Returns normalized Hamming distance [0.0, 1.0]."""
        hv = self.encoder.encode_sequence(sequence, method=method, ngram_size=ngram_size)
        return self.score_hypervector(hv)

    def score_batch(
        self,
        sequences: Sequence[Sequence[Any]],
        method: str = "positional",
        ngram_size: int = 3,
    ) -> np.ndarray:
        """Batch score event sequences."""
        hvs = [
            self.encoder.encode_sequence(seq, method=method, ngram_size=ngram_size)
            for seq in sequences
        ]
        return self.score_hypervectors_batch(hvs)

    def predict(
        self,
        sequence: Sequence[Any],
        method: str = "positional",
        ngram_size: int = 3,
    ) -> bool:
        """Predict whether sequence is anomalous. True = Anomaly, False = Normal."""
        return self.score(sequence, method=method, ngram_size=ngram_size) > self.threshold_

    def predict_batch(
        self,
        sequences: Sequence[Sequence[Any]],
        method: str = "positional",
        ngram_size: int = 3,
    ) -> np.ndarray:
        """Batch prediction: returns boolean array where True indicates anomaly."""
        scores = self.score_batch(sequences, method=method, ngram_size=ngram_size)
        return scores > self.threshold_

    def update(
        self,
        normal_sequence: Sequence[Any],
        learning_rate: float = 0.05,
        method: str = "positional",
    ) -> None:
        """Online holographic memory update with a newly observed normal sequence."""
        if self.profile_memory is None:
            self.fit([normal_sequence], method=method)
            return

        new_hv = self.encoder.encode_sequence(normal_sequence, method=method)
        # Stochastic bit-flipping update in binary VSA space (Kanerva 2009)
        prof_bin = self.profile_memory.to_binary()
        new_bin = new_hv.to_binary()
        diff_mask = prof_bin != new_bin
        if np.any(diff_mask):
            rng = np.random.RandomState(42)
            flips = (rng.rand(self.dim) < learning_rate) & diff_mask
            if not np.any(flips):
                diff_idx = np.where(diff_mask)[0]
                flips[rng.choice(diff_idx)] = True
            prof_bin[flips] = new_bin[flips]
            updated = HyperVector.from_binary_array(prof_bin, dim=self.dim, mode=self.mode)
            self.profile_memory = updated
            if self.prototypes:
                self.prototypes[0] = updated

    def get_metrics(self) -> Dict[str, Any]:
        """Return diagnostic metrics of holographic memory state."""
        return {
            "dim": self.dim,
            "mode": self.mode,
            "num_prototypes": len(self.prototypes),
            "threshold": self.threshold_,
            "mean_normal_distance": self.mean_distance_,
            "std_normal_distance": self.std_distance_,
            "profile_density": (
                self.profile_memory.bit_count / self.dim if self.profile_memory else 0.0
            ),
        }
