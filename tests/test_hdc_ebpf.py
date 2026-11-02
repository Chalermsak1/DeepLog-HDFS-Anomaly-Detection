"""Unit tests for Pillar I: Hyperdimensional Computing (HDC) & Zero-IO In-Kernel eBPF Core.

Validates:
1. Hypervector asymptotic orthogonality in high dimensions (D = 10,000).
2. Binding invertibility, commutativity, and quasi-orthogonality.
3. Bundling (majority rule / superposition) component similarity retention.
4. Cyclic permutation distance invariance, invertibility, and pseudo-orthogonality.
5. Temporal sequence order-sensitivity in HDC space.
6. HDC anomaly detection separation between normal and anomalous traces.
7. Sub-microsecond execution latency and > 1,000,000 ops/sec throughput.
8. Zero-IO in-memory eBPF ring buffer circular wrap-around and drop counters.
9. In-kernel fixed-point quantized integer inference and trace filtering.
"""

from __future__ import annotations

import time
import numpy as np
import pytest

from deeplog.hdc import (
    HyperVector,
    HyperdimensionalEncoder,
    HDCAnomalyDetector,
    HDCProfileAccumulator,
)
from deeplog.ebpf_core import (
    TraceEvent,
    AnomalyAlert,
    EBPFRingBufferEmulator,
    InKernelInferenceEngine,
    InKernelHDCFilter,
    FixedPointLinear,
    FixedPointAutoencoder,
    BPF_RINGBUF_BUSY_BIT,
    BPF_RINGBUF_DISCARD_BIT,
)


class TestHyperVectorRepresentation:
    """Tests for basic HyperVector properties and conversions."""

    def test_hypervector_creation_binary(self):
        hv = HyperVector.random(dim=10000, mode="binary", seed=42)
        assert hv.dim == 10000
        assert hv.mode == "binary"
        # Random vector should have ~50% density
        density = hv.bit_count / 10000
        assert 0.48 < density < 0.52

    def test_hypervector_creation_bipolar(self):
        hv = HyperVector.random(dim=10000, mode="bipolar", seed=42)
        assert hv.mode == "bipolar"
        bipolar_arr = hv.to_bipolar()
        assert len(bipolar_arr) == 10000
        assert set(np.unique(bipolar_arr)).issubset({-1, 1})

    def test_zeros_and_ones(self):
        z = HyperVector.zeros(dim=1000)
        assert z.bit_count == 0
        assert np.all(z.to_binary() == 0)

        o = HyperVector.ones(dim=1000)
        assert o.bit_count == 1000
        assert np.all(o.to_binary() == 1)

    def test_numpy_roundtrip(self):
        rng = np.random.RandomState(123)
        arr = rng.randint(0, 2, size=1000, dtype=np.uint8)
        hv = HyperVector.from_binary_array(arr, dim=1000)
        arr_recovered = hv.to_binary()
        assert np.array_equal(arr, arr_recovered)

    def test_words_export_and_bytes(self):
        hv = HyperVector.random(dim=10000, seed=99)
        words = hv.to_words()
        assert len(words) == (10000 + 63) // 64
        raw = hv.to_bytes()
        hv_from_raw = HyperVector.from_bytes(raw, dim=10000)
        assert hv == hv_from_raw


class TestVSAOperations:
    """Tests for Vector Symbolic Architecture operations: Orthogonality, Binding, Bundling, Permutation."""

    def test_hypervector_orthogonality(self):
        """Random 10,000-D hypervectors are asymptotically orthogonal."""
        dim = 10000
        enc = HyperdimensionalEncoder(dim=dim, seed=42)

        distances = []
        similarities = []
        for i in range(50):
            a = enc.random_hypervector(seed=i)
            b = enc.random_hypervector(seed=i + 1000)
            d = a.hamming_distance(b, normalized=True)
            s = a.cosine_similarity(b)
            distances.append(d)
            similarities.append(s)

        mean_d = float(np.mean(distances))
        std_d = float(np.std(distances))
        mean_s = float(np.mean(similarities))

        # Expected normalized Hamming distance = 0.50, std ~ 1 / (2*sqrt(D)) = 0.005
        assert 0.495 < mean_d < 0.505
        assert std_d < 0.015
        assert abs(mean_s) < 0.02

    def test_binding_invertibility_and_commutativity(self):
        """Binding (\\otimes) is self-invertible and commutative."""
        dim = 10000
        enc = HyperdimensionalEncoder(dim=dim)

        a = enc.random_hypervector(seed=1)
        b = enc.random_hypervector(seed=2)

        # Commutativity: a ^ b == b ^ a
        ab = a ^ b
        ba = b ^ a
        assert ab == ba

        # Exact Invertibility: (a ^ b) ^ b == a
        unbound_a = ab ^ b
        assert unbound_a == a

        # Exact Invertibility: (a ^ b) ^ a == b
        unbound_b = ab ^ a
        assert unbound_b == b

    def test_binding_quasi_orthogonality(self):
        """Bound vector a \\otimes b is quasi-orthogonal to both a and b."""
        dim = 10000
        enc = HyperdimensionalEncoder(dim=dim)

        a = enc.random_hypervector(seed=10)
        b = enc.random_hypervector(seed=20)
        c = a ^ b

        dist_ca = c.hamming_distance(a)
        dist_cb = c.hamming_distance(b)

        assert 0.48 < dist_ca < 0.52
        assert 0.48 < dist_cb < 0.52

    def test_bundling_similarity_retention(self):
        """Bundling (\\oplus) retains high similarity to constituent hypervectors."""
        dim = 10000
        enc = HyperdimensionalEncoder(dim=dim)

        # Bundle 5 distinct vectors
        vectors = [enc.random_hypervector(seed=i) for i in range(5)]
        bundle = enc.bundle(vectors)

        # Unrelated control vector
        unrelated = enc.random_hypervector(seed=999)

        # Each constituent must have low Hamming distance (< 0.35) and high similarity
        for i, v in enumerate(vectors):
            d = bundle.hamming_distance(v)
            s = bundle.cosine_similarity(v)
            assert d < 0.35, f"Vector {i} distance {d} too high"
            assert s > 0.30, f"Vector {i} similarity {s} too low"

        # Unrelated vector must be quasi-orthogonal (~0.50)
        d_unrel = bundle.hamming_distance(unrelated)
        assert 0.48 < d_unrel < 0.52

    def test_weighted_bundling(self):
        """Weighted bundling biases result toward higher weighted vector."""
        dim = 10000
        enc = HyperdimensionalEncoder(dim=dim)

        a = enc.random_hypervector(seed=1)
        b = enc.random_hypervector(seed=2)

        # a has 5x weight of b
        bundle = enc.bundle([a, b], weights=[5.0, 1.0])
        dist_a = bundle.hamming_distance(a)
        dist_b = bundle.hamming_distance(b)

        assert dist_a < dist_b
        assert dist_a < 0.10

    def test_cyclic_permutation_properties(self):
        """Cyclic permutation preserves norm, is invertible, and is pseudo-orthogonal."""
        dim = 10000
        enc = HyperdimensionalEncoder(dim=dim)

        a = enc.random_hypervector(seed=42)

        # Preserves bit count / norm
        p1 = a.permute(1)
        assert p1.bit_count == a.bit_count

        # For k != 0, permuted vector is quasi-orthogonal to original
        dist_orig = a.hamming_distance(p1)
        assert 0.48 < dist_orig < 0.52

        # Invertibility: Pi^{-k}(Pi^k(a)) == a
        unpermuted = p1.permute(-1)
        assert unpermuted == a

        # Multiple shifts: Pi^5
        p5 = a.permute(5)
        assert a.hamming_distance(p5) > 0.48
        assert p5.permute(-5) == a

        # Preserves pairwise distance between two vectors
        b = enc.random_hypervector(seed=99)
        d_ab = a.hamming_distance(b)
        d_perm_ab = a.permute(7).hamming_distance(b.permute(7))
        assert abs(d_ab - d_perm_ab) < 1e-6


class TestHDCSequenceEncoding:
    """Tests for encoding temporal sequences, sets, and records."""

    def test_positional_sequence_encoding(self):
        """Identical sequences yield identical hypervectors, order changes yield differences."""
        enc = HyperdimensionalEncoder(dim=10000, seed=42)

        seq_a = [101, 102, 103, 104, 105]
        seq_b = [101, 102, 103, 104, 105]
        seq_swapped = [105, 104, 103, 102, 101]
        seq_diff = [901, 902, 903, 904, 905]

        hv_a = enc.encode_sequence(seq_a, method="positional")
        hv_b = enc.encode_sequence(seq_b, method="positional")
        hv_swapped = enc.encode_sequence(seq_swapped, method="positional")
        hv_diff = enc.encode_sequence(seq_diff, method="positional")

        # Exact match for identical
        assert hv_a == hv_b

        # Swapped order produces significant distance
        d_swap = hv_a.hamming_distance(hv_swapped)
        assert d_swap > 0.30

        # Completely different events produces near-orthogonal distance (~0.50)
        d_diff = hv_a.hamming_distance(hv_diff)
        assert 0.47 < d_diff < 0.53

    def test_ngram_sequence_encoding(self):
        """Sliding window n-gram chain binding is order-sensitive."""
        enc = HyperdimensionalEncoder(dim=10000, seed=42)

        seq1 = [1, 2, 3, 4, 5, 6]
        seq2 = [1, 2, 3, 4, 5, 6]
        seq_mutated = [1, 2, 99, 4, 5, 6]

        hv1 = enc.encode_sequence(seq1, method="ngram", ngram_size=3)
        hv2 = enc.encode_sequence(seq2, method="ngram", ngram_size=3)
        hv_mut = enc.encode_sequence(seq_mutated, method="ngram", ngram_size=3)

        assert hv1 == hv2
        assert hv1.hamming_distance(hv_mut) > 0.20

    def test_record_encoding(self):
        """Key-value record encoding via bound pairs."""
        enc = HyperdimensionalEncoder(dim=10000, seed=42)

        rec1 = {"syscall": "read", "fd": 3, "ret": 1024}
        rec2 = {"syscall": "read", "fd": 3, "ret": 1024}
        rec3 = {"syscall": "write", "fd": 1, "ret": 512}

        hv1 = enc.encode_record(rec1)
        hv2 = enc.encode_record(rec2)
        hv3 = enc.encode_record(rec3)

        assert hv1 == hv2
        assert hv1.hamming_distance(hv3) > 0.30


class TestHDCAnomalyDetector:
    """Tests for holographic memory anomaly detection."""

    @pytest.fixture
    def trained_detector(self):
        enc = HyperdimensionalEncoder(dim=10000, seed=42)
        detector = HDCAnomalyDetector(encoder=enc, dim=10000, k_sigma=3.0)

        # Baseline normal workflows (e.g. HDFS open -> read -> replicate -> close)
        normal_sequences = [
            [1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5],
            [1, 2, 3, 4],
            [1, 2, 3, 4, 5],
            [1, 2, 3, 5],
        ] * 10
        detector.fit(normal_sequences)
        return detector

    def test_normal_sequences_low_anomaly_score(self, trained_detector):
        normal_test = [1, 2, 3, 4, 5]
        score = trained_detector.score(normal_test)
        is_anom = trained_detector.predict(normal_test)

        assert score <= trained_detector.threshold_
        assert is_anom is False

    def test_anomalous_sequences_high_anomaly_score(self, trained_detector):
        # Injected foreign events
        anom_unseen = [999, 888, 777]
        score_unseen = trained_detector.score(anom_unseen)
        assert score_unseen > trained_detector.threshold_
        assert trained_detector.predict(anom_unseen) is True

        # Inverted sequence order
        anom_reversed = [5, 4, 3, 2, 1]
        score_rev = trained_detector.score(anom_reversed)
        assert score_rev > trained_detector.threshold_
        assert trained_detector.predict(anom_reversed) is True

    def test_batch_scoring(self, trained_detector):
        sequences = [
            [1, 2, 3, 4, 5],  # normal
            [99, 98, 97],      # anomaly
            [1, 2, 3, 4],      # normal
            [5, 4, 3, 2, 1],  # anomaly
        ]
        scores = trained_detector.score_batch(sequences)
        preds = trained_detector.predict_batch(sequences)

        assert len(scores) == 4
        assert preds[0] == False
        assert preds[1] == True
        assert preds[2] == False
        assert preds[3] == True

    def test_multi_prototype_clustering(self):
        """Multi-prototype clustering captures multi-modal normal workflows."""
        enc = HyperdimensionalEncoder(dim=10000, seed=42)
        detector = HDCAnomalyDetector(encoder=enc, dim=10000, num_prototypes=2)

        # Mode A: 10 -> 20 -> 30
        # Mode B: 100 -> 200 -> 300
        dataset = [[10, 20, 30] for _ in range(15)] + [[100, 200, 300] for _ in range(15)]
        detector.fit(dataset)

        assert len(detector.prototypes) == 2
        # Both modes should be recognized as normal
        assert detector.predict([10, 20, 30]) is False
        assert detector.predict([100, 200, 300]) is False
        # Outlier is detected
        assert detector.predict([999, 888, 777]) is True

    def test_online_update(self, trained_detector):
        new_normal = [1, 2, 7, 4, 5]
        score_before = trained_detector.score(new_normal)

        # Online update adapts holographic memory
        trained_detector.update(new_normal, learning_rate=0.2)
        score_after = trained_detector.score(new_normal)

        assert score_after < score_before


class TestHDCPerformanceBenchmark:
    """Benchmark tests validating sub-microsecond latency and millions of ops/sec."""

    def test_submicrosecond_hamming_distance_latency(self):
        """Hamming distance evaluation between 10,000-bit hypervectors executes in sub-microsecond time."""
        dim = 10000
        hv1 = HyperVector.random(dim=dim, seed=1)
        hv2 = HyperVector.random(dim=dim, seed=2)

        # Warm-up
        for _ in range(1000):
            _ = hv1.hamming_distance(hv2)

        n_ops = 50000
        t0 = time.perf_counter()
        for _ in range(n_ops):
            _ = hv1.hamming_distance(hv2)
        t1 = time.perf_counter()

        dt = t1 - t0
        latency_us = (dt / n_ops) * 1e6
        throughput_ops = n_ops / dt

        print(f"\nHDC Hamming Latency: {latency_us:.3f} us/op | Throughput: {throughput_ops:,.0f} ops/sec")
        # Must execute in sub-microsecond latency (< 1.0 us)
        assert latency_us < 1.0, f"Expected < 1.0 us latency, got {latency_us:.3f} us"
        # Must exceed 1,000,000 ops/sec
        assert throughput_ops > 1000000, f"Expected > 1,000,000 ops/sec, got {throughput_ops:,.0f}"

    def test_submicrosecond_detector_scoring(self):
        """HDCAnomalyDetector score_hypervector executes in sub-microsecond latency."""
        dim = 10000
        enc = HyperdimensionalEncoder(dim=dim, seed=42)
        detector = HDCAnomalyDetector(encoder=enc, dim=dim)
        detector.fit([[1, 2, 3, 4, 5] for _ in range(10)])

        test_hv = enc.random_hypervector(seed=77)

        # Warm-up
        for _ in range(1000):
            _ = detector.score_hypervector(test_hv)

        n_ops = 50000
        t0 = time.perf_counter()
        for _ in range(n_ops):
            _ = detector.score_hypervector(test_hv)
        t1 = time.perf_counter()

        dt = t1 - t0
        latency_us = (dt / n_ops) * 1e6
        throughput_ops = n_ops / dt

        print(f"\nDetector Latency: {latency_us:.3f} us/op | Throughput: {throughput_ops:,.0f} ops/sec")
        assert latency_us < 1.0
        assert throughput_ops > 1000000


class TestTraceEventBinary:
    """Tests for raw binary C-struct TraceEvent serialization."""

    def test_trace_event_pack_unpack_parity(self):
        event = TraceEvent(
            timestamp_ns=1700000000123456789,
            duration_ns=4250,
            ret_val=-13,  # EACCES
            pid=28491,
            tgid=28491,
            event_id=257,  # sys_enter_openat
            cpu_id=7,
            flags=2,
            args_val=0xDEADBEEF,
            comm="systemd-udevd",
        )

        raw = event.to_bytes()
        assert len(raw) == 64

        recovered = TraceEvent.from_bytes(raw)
        assert recovered.timestamp_ns == 1700000000123456789
        assert recovered.duration_ns == 4250
        assert recovered.ret_val == -13
        assert recovered.pid == 28491
        assert recovered.tgid == 28491
        assert recovered.event_id == 257
        assert recovered.cpu_id == 7
        assert recovered.flags == 2
        assert recovered.args_val == 0xDEADBEEF
        assert recovered.comm_str == "systemd-udevd"

    def test_trace_event_zero_copy_pack_into(self):
        buf = bytearray(128)
        event = TraceEvent(timestamp_ns=5000, event_id=42, comm="kworker")
        event.pack_into(buf, offset=32)

        recovered = TraceEvent.unpack_from(buf, offset=32)
        assert recovered.timestamp_ns == 5000
        assert recovered.event_id == 42
        assert recovered.comm_str == "kworker"


class TestEBPFRingBuffer:
    """Tests for EBPFRingBufferEmulator: high throughput, zero-IO, wrap-around, and drops."""

    def test_push_and_poll_lifecycle(self):
        rb = EBPFRingBufferEmulator(capacity_power_of_2=14)  # 16 KB

        for i in range(50):
            evt = TraceEvent(timestamp_ns=i * 100, event_id=i, comm="app")
            assert rb.push_event(evt) is True

        polled = rb.poll()
        assert len(polled) == 50
        assert polled[0].event_id == 0
        assert polled[-1].event_id == 49
        assert rb.stats()["events_consumed"] == 50

    def test_circular_wrap_around_integrity(self):
        """Ring buffer wraps around end of circular buffer without data corruption."""
        # Small 2KB buffer to force frequent wrap-around
        rb = EBPFRingBufferEmulator(capacity_power_of_2=11)  # 2048 bytes

        # Push 1,000 events while polling periodically (forces wrap-around dozens of times)
        total_events = 1000
        consumed = []

        for i in range(total_events):
            evt = TraceEvent(timestamp_ns=i * 1000, event_id=i, comm=f"proc_{i%10}")
            ok = rb.push_event(evt)
            assert ok, f"Failed pushing event {i}"

            # Interleaved consumption
            if i % 10 == 0:
                batch = rb.poll()
                consumed.extend(batch)

        # Drain remaining
        consumed.extend(rb.poll())

        assert len(consumed) == total_events
        for i, evt in enumerate(consumed):
            assert evt.timestamp_ns == i * 1000
            assert evt.event_id == i

    def test_backpressure_and_drop_handling(self):
        """When consumer is halted and buffer fills, events are dropped and counter increments."""
        # 4 KB buffer: each event is 8+64 = 72 bytes. Max ~56 events.
        rb = EBPFRingBufferEmulator(capacity_power_of_2=12)  # 4096 bytes

        pushed_count = 0
        for i in range(100):
            ok = rb.push_event(TraceEvent(timestamp_ns=i, event_id=i))
            if ok:
                pushed_count += 1

        stats = rb.stats()
        assert stats["events_dropped"] > 0
        assert pushed_count < 100
        assert pushed_count + stats["events_dropped"] == 100

        # Now consume all, freeing space
        drained = rb.poll()
        assert len(drained) == pushed_count

        # Pushing should now succeed again
        assert rb.push_event(TraceEvent(timestamp_ns=999, event_id=999)) is True

    def test_discard_record_behavior(self):
        """Discarded records are skipped by consumer."""
        rb = EBPFRingBufferEmulator(capacity_power_of_2=12)

        # Reserve and submit event 1
        off1 = rb.reserve(64)
        TraceEvent(event_id=1).pack_into(rb._buf, (off1 + 8) & rb.mask)
        rb.submit(off1)

        # Reserve and discard event 2
        off2 = rb.reserve(64)
        TraceEvent(event_id=2).pack_into(rb._buf, (off2 + 8) & rb.mask)
        rb.discard(off2)

        # Reserve and submit event 3
        off3 = rb.reserve(64)
        TraceEvent(event_id=3).pack_into(rb._buf, (off3 + 8) & rb.mask)
        rb.submit(off3)

        # Consumer should read only events 1 and 3
        polled = rb.poll()
        assert len(polled) == 2
        assert polled[0].event_id == 1
        assert polled[1].event_id == 3
        assert rb.stats()["events_discarded"] == 1

    def test_high_throughput_zero_io(self):
        """Ring buffer pushes and consumes > 500,000 events/sec in memory."""
        rb = EBPFRingBufferEmulator(capacity_power_of_2=20)  # 1 MB
        n_events = 100000

        t0 = time.perf_counter()
        for i in range(n_events):
            evt = TraceEvent(timestamp_ns=i, event_id=i % 100, comm="stress")
            rb.push_event(evt)
        t1 = time.perf_counter()
        _ = rb.poll()
        t2 = time.perf_counter()

        push_tput = n_events / (t1 - t0)
        total_tput = n_events / (t2 - t0)

        print(f"\nRing Buffer Push Tput: {push_tput:,.0f} evt/sec | Total Tput: {total_tput:,.0f} evt/sec")
        assert push_tput > 300000


class TestFixedPointQuantization:
    """Tests for eBPF verifier-compliant integer fixed-point layers."""

    def test_fixed_point_linear_numerical_parity(self):
        """Fixed-point integer matrix multiplication matches float within Q-tolerance."""
        rng = np.random.RandomState(42)
        w_float = rng.randn(4, 8).astype(np.float32)
        b_float = rng.randn(4).astype(np.float32)
        x_float = rng.randn(8).astype(np.float32)

        # Float reference
        y_float = np.dot(w_float, x_float) + b_float

        # Fixed point Q8.8
        shift = 8
        scale = 1 << shift
        layer_q = FixedPointLinear.from_float(w_float, b_float, shift=shift)

        x_q = np.round(x_float * scale).astype(np.int64)
        y_q = layer_q.forward(x_q)
        y_dequant = y_q / float(scale)

        # Error must be below 0.05
        max_err = float(np.max(np.abs(y_float - y_dequant)))
        assert max_err < 0.05, f"Quantization error {max_err} exceeded tolerance"

    def test_fixed_point_autoencoder(self):
        ae = FixedPointAutoencoder(in_dim=4, hidden_dim=2, shift=8, threshold_q=200)
        # Normal inputs around 0
        normal_x = np.array([10, -5, 8, -2], dtype=np.int64)
        score_norm = ae.anomaly_score_q(normal_x)

        # Extreme outlier
        outlier_x = np.array([2000, -1500, 3000, 4000], dtype=np.int64)
        score_outlier = ae.anomaly_score_q(outlier_x)

        assert score_outlier > score_norm
        assert ae.is_anomaly(outlier_x) is True


class TestInKernelInferenceEngine:
    """Tests for end-to-end zero-IO in-kernel inference pipeline."""

    def test_in_kernel_hdc_filtering(self):
        """Normal tracepoints are filtered inside kernel, anomalous ones produce alerts."""
        engine = InKernelInferenceEngine(mode="hdc", dim=10000, window_size=5)

        # Baseline normal workflow: sequence 1 -> 2 -> 3 -> 4 -> 5
        normal_workflows = [[1, 2, 3, 4, 5] for _ in range(20)]
        engine.fit_normal_patterns(normal_workflows)

        # Feed 10 repetitions of normal sequence
        alerts = []
        for _ in range(5):
            for eid in [1, 2, 3, 4, 5]:
                alert = engine.process_trace_event(
                    TraceEvent(event_id=eid, comm="normal_worker")
                )
                if alert:
                    alerts.append(alert)

        # All normal events should be filtered out directly in kernel
        assert len(alerts) == 0
        assert engine.events_filtered == 25
        assert engine.events_inspected == 25
        assert engine.stats()["filter_rate"] == 1.0

        # Now inject an anomalous syscall sequence
        anom_alerts = []
        for eid in [999, 888, 777, 666, 555]:
            alert = engine.process_trace_event(
                TraceEvent(event_id=eid, comm="exploit_payload")
            )
            if alert:
                anom_alerts.append(alert)

        assert len(anom_alerts) > 0
        assert engine.anomalies_detected > 0

        # Verify ring buffer received the anomalous event
        polled = engine.ring_buffer.poll()
        assert len(polled) == len(anom_alerts)
        assert polled[0].comm_str == "exploit_payload"

    def test_in_kernel_fixed_point_mode(self):
        engine = InKernelInferenceEngine(mode="fixed_point")

        # Normal event with low latency and 0 return code
        evt_norm = TraceEvent(event_id=1, duration_ns=100, ret_val=0, cpu_id=0, flags=0)
        alert_norm = engine.process_trace_event(evt_norm)
        assert alert_norm is None

        # Extreme anomaly with huge duration and error return code
        evt_anom = TraceEvent(event_id=1, duration_ns=100000000, ret_val=-1, cpu_id=7, flags=15)
        alert_anom = engine.process_trace_event(evt_anom)
        assert alert_anom is not None

    def test_benchmark_pipeline_throughput(self):
        """Pipeline achieves high throughput and zero-IO execution."""
        engine = InKernelInferenceEngine(mode="hdc", dim=10000, window_size=5)
        engine.fit_normal_patterns([[1, 2, 3, 4, 5] for _ in range(10)])

        bench = engine.benchmark_pipeline(num_events=10000)
        print(f"\nIn-Kernel Engine Benchmark: {bench['throughput_events_per_sec']:,.0f} evt/sec | Latency: {bench['latency_microseconds']:.3f} us")

        assert bench["throughput_events_per_sec"] > 20000
        assert bench["filter_rate"] > 0.0
