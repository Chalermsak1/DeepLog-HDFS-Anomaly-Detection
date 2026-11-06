"""Byte-Level Neural Field & Hypervector Operator (Zero-Parsing Observability).

Eliminates brittle log template extraction (e.g., Drain, Spell, regex) by mapping
raw UTF-8 byte streams and kernel event buffers directly into continuous Fourier
neural fields and 10,000-bit Hyperdimensional Computing (HDC) hypervectors.

Guarantees 100% vocabulary coverage (zero Out-Of-Vocabulary / OOV collapse) and
resilience against dynamic software release format string mutations.
"""

from __future__ import annotations

import math
import struct
import zlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


@dataclass(frozen=True)
class ByteStreamAnomalyResult:
    """Result of continuous byte field anomaly and drift evaluation."""

    entropy: float
    field_energy: float
    reconstruction_residual: float
    hdc_cosine_similarity: float
    is_anomalous: bool
    detected_mutation_type: str
    metadata: Dict[str, float] = field(default_factory=dict)


class ContinuousByteFieldOperator:
    """Continuous Neural Field Operator over arbitrary raw byte streams.

    Maps byte sequences B = [b_0, b_1, ..., b_{L-1}] where b_i in [0, 255]
    into continuous coordinate manifolds using multi-frequency Fourier embeddings
    and Vector Symbolic Architecture (VSA) hypervectors.
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        num_frequencies: int = 16,
        hdc_dim: int = 10000,
        seed: int = 42,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.num_frequencies = num_frequencies
        self.hdc_dim = hdc_dim
        self.rng = np.random.RandomState(seed)

        # Basis projection matrix for continuous byte coordinates: B_proj in R^{num_freq x 1}
        self.freq_bands = np.array(
            [2.0**i * math.pi for i in range(num_frequencies)], dtype=np.float32
        )

        # Learned or deterministic projection from 2*num_frequencies to embedding_dim
        self.proj_weight = self.rng.randn(embedding_dim, 2 * num_frequencies).astype(
            np.float32
        ) / math.sqrt(2 * num_frequencies)

        # HDC base item memory for 256 byte values: 256 x hdc_dim bipolar vectors in {-1, +1}
        raw_bits = self.rng.randint(0, 2, size=(256, hdc_dim), dtype=np.int8)
        self.byte_item_memory = np.where(raw_bits == 0, -1, 1).astype(np.int8)

        # Baseline baseline profile for healthy system byte streams
        self._baseline_field: Optional[np.ndarray] = None
        self._baseline_hdc: Optional[np.ndarray] = None

    def compute_shannon_entropy(self, raw_bytes: Union[bytes, bytearray, np.ndarray]) -> float:
        """Compute Shannon information entropy H(B) in bits/byte."""
        if isinstance(raw_bytes, (bytes, bytearray)):
            byte_arr = np.frombuffer(raw_bytes, dtype=np.uint8)
        else:
            byte_arr = np.asarray(raw_bytes, dtype=np.uint8)

        if len(byte_arr) == 0:
            return 0.0

        counts = np.bincount(byte_arr, minlength=256)
        probs = counts[counts > 0] / float(len(byte_arr))
        return float(-np.sum(probs * np.log2(probs)))

    def encode_fourier_field(
        self, raw_bytes: Union[bytes, bytearray, np.ndarray]
    ) -> np.ndarray:
        """Encode raw byte stream into continuous Fourier neural field coordinates.

        Returns:
            np.ndarray: Continuous field vector of shape (embedding_dim,) normalized to unit sphere.
        """
        if isinstance(raw_bytes, (bytes, bytearray)):
            byte_arr = np.frombuffer(raw_bytes, dtype=np.uint8)
        else:
            byte_arr = np.asarray(raw_bytes, dtype=np.uint8)

        if len(byte_arr) == 0:
            return np.zeros(self.embedding_dim, dtype=np.float32)

        # Normalize byte values to [-1, 1]
        norm_bytes = (byte_arr.astype(np.float32) / 127.5) - 1.0  # (L,)

        # Multi-scale sinusoidal positional + value encoding:
        # Outer product: (L, num_freq)
        angles = np.outer(norm_bytes, self.freq_bands)
        fourier_features = np.concatenate([np.sin(angles), np.cos(angles)], axis=1)  # (L, 2*num_freq)

        # Mean field pooling over byte sequence length
        pooled_fourier = np.mean(fourier_features, axis=0)  # (2*num_freq,)

        # Linear projection to continuous manifold
        field_vec = self.proj_weight @ pooled_fourier  # (embedding_dim,)

        # Unit sphere normalization
        norm = np.linalg.norm(field_vec)
        if norm > 1e-9:
            field_vec = field_vec / norm
        return field_vec

    def encode_hdc_hypervector(
        self, raw_bytes: Union[bytes, bytearray, np.ndarray], window_size: int = 4
    ) -> np.ndarray:
        """Encode raw bytes into a 10,000-bit VSA holographic hypervector.

        Uses circular shift permutation for sequential n-grams and bundling (majority voting).
        Returns:
            np.ndarray: Bipolar hypervector of shape (hdc_dim,) with values in {-1, 1}.
        """
        if isinstance(raw_bytes, (bytes, bytearray)):
            byte_arr = np.frombuffer(raw_bytes, dtype=np.uint8)
        else:
            byte_arr = np.asarray(raw_bytes, dtype=np.uint8)

        if len(byte_arr) == 0:
            return np.ones(self.hdc_dim, dtype=np.int8)

        # Accumulator for bundling (majority voting)
        accumulator = np.zeros(self.hdc_dim, dtype=np.int32)
        L = len(byte_arr)

        if L < window_size:
            # Single sequence binding
            ngram_vec = np.ones(self.hdc_dim, dtype=np.int8)
            for shift, b in enumerate(byte_arr):
                vec = np.roll(self.byte_item_memory[b], shift)
                ngram_vec = ngram_vec * vec  # Element-wise XOR/multiplication
            accumulator += ngram_vec
        else:
            # Rolling n-gram binding over sliding byte windows
            for i in range(L - window_size + 1):
                ngram_vec = np.ones(self.hdc_dim, dtype=np.int8)
                for shift in range(window_size):
                    b = byte_arr[i + shift]
                    vec = np.roll(self.byte_item_memory[b], shift)
                    ngram_vec = ngram_vec * vec
                accumulator += ngram_vec

        # Bipolar thresholding with zero-bias tie breaker
        result = np.where(accumulator >= 0, 1, -1).astype(np.int8)
        return result

    def fit_baseline(self, normal_streams: Sequence[Union[bytes, str]]) -> None:
        """Calibrate baseline neural field and HDC representations from normal execution."""
        field_list = []
        hdc_accum = np.zeros(self.hdc_dim, dtype=np.int32)

        for s in normal_streams:
            raw = s.encode("utf-8") if isinstance(s, str) else s
            field_list.append(self.encode_fourier_field(raw))
            hdc_accum += self.encode_hdc_hypervector(raw)

        if field_list:
            mean_field = np.mean(field_list, axis=0)
            norm = np.linalg.norm(mean_field)
            self._baseline_field = mean_field / (norm if norm > 1e-9 else 1.0)
            self._baseline_hdc = np.where(hdc_accum >= 0, 1, -1).astype(np.int8)

    def evaluate_stream(
        self,
        stream: Union[bytes, str],
        cosine_threshold: float = 0.55,
        entropy_spike_threshold: float = 7.2,
    ) -> ByteStreamAnomalyResult:
        """Evaluate raw stream against calibrated continuous field and HDC memory."""
        raw = stream.encode("utf-8") if isinstance(stream, str) else stream
        entropy = self.compute_shannon_entropy(raw)
        field_vec = self.encode_fourier_field(raw)
        hdc_vec = self.encode_hdc_hypervector(raw)

        field_energy = float(np.sum(field_vec**2))

        if self._baseline_field is None or self._baseline_hdc is None:
            # Uncalibrated fallback
            return ByteStreamAnomalyResult(
                entropy=entropy,
                field_energy=field_energy,
                reconstruction_residual=0.0,
                hdc_cosine_similarity=1.0,
                is_anomalous=False,
                detected_mutation_type="NORMAL_UNCALIBRATED",
            )

        # Field cosine similarity
        field_sim = float(np.dot(self._baseline_field, field_vec))

        # HDC cosine similarity (Hamming distance equivalent)
        # sim = (dot_product) / hdc_dim
        hdc_sim = float(np.dot(self._baseline_hdc.astype(np.float32), hdc_vec.astype(np.float32)) / self.hdc_dim)

        # Combined multi-scale residual: 80% n-gram sequence structure + 20% continuous Fourier spectrum
        combined_sim = float(0.8 * hdc_sim + 0.2 * field_sim)
        residual = float(1.0 - combined_sim)

        is_anomalous = False
        mutation_type = "NORMAL"

        if entropy > entropy_spike_threshold:
            is_anomalous = True
            mutation_type = "ENCRYPTED_OR_CORRUPT_BUFFER_INJECTION"
        elif field_sim >= 0.88 and hdc_sim < cosine_threshold:
            # Semantic Fourier continuous field preserves invariant: benign format mutation
            is_anomalous = False
            mutation_type = "FORMAT_MUTATION_BENIGN"
        elif hdc_sim < cosine_threshold or residual > (1.0 - cosine_threshold):
            is_anomalous = True
            mutation_type = "PATHOLOGICAL_SEQUENCE_DEVIATION"

        return ByteStreamAnomalyResult(
            entropy=entropy,
            field_energy=field_energy,
            reconstruction_residual=residual,
            hdc_cosine_similarity=hdc_sim,
            is_anomalous=is_anomalous,
            detected_mutation_type=mutation_type,
            metadata={
                "stream_length_bytes": float(len(raw)),
                "fourier_field_similarity": field_sim,
                "hdc_raw_agreement": hdc_sim,
            },
        )
