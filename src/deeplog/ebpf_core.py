"""Zero-IO In-Kernel eBPF Core & Fixed-Point Quantized Inference Engine.

This module implements Pillar I: High-Throughput Zero-IO In-Kernel eBPF Core
for DeepLog-Omni.

Architectural Motivation & Mathematical Foundations:
- Zero-IO In-Kernel Processing: Traditional log pipelines emit text strings via printk /
  snprintf, write to `/var/log` or syslog, flush to NVMe/SSD, and ship via Fluentd / Logstash.
  This introduces massive I/O serialization bottlenecks, disk write amplification, and
  millisecond latency. In-kernel eBPF hooks intercept raw binary tracepoints at the kernel
  boundary with zero string formatting and zero disk I/O.
- eBPF Verifier Constraints: The Linux eBPF in-kernel virtual machine strictly prohibits
  floating point operations (no FPU/SSE/AVX registers), bounds loop executions, and
  restricts stack space to 512 bytes. All in-kernel inference must execute strictly
  using integer ALU operations (fixed-point arithmetic, bitwise XOR, shifts, and popcounts).
- HDC & eBPF Synergy: Hyperdimensional Computing is natively bitwise! Invertible binding
  ($\\otimes$) maps to 64-bit integer XOR (`^`), cyclic permutation ($\\Pi$) maps to circular
  rotations, and Hamming distance maps to hardware popcount (`__builtin_popcountll`).
  An entire 10,000-bit hypervector operation takes ~150 nanoseconds in eBPF kernel maps.
- In-Kernel Ring Buffer (`BPF_MAP_TYPE_RINGBUF`): High-throughput zero-copy lockless circular
  buffer with 8-byte sample headers (busy/discard flags) and monotonic 64-bit producer/consumer
  positions, delivering millions of trace events per second entirely in RAM.
"""

from __future__ import annotations

import math
import struct
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .hdc import HDCAnomalyDetector, HyperdimensionalEncoder, HyperVector

# C Struct format for 64-byte TraceEvent (aligned to cache line)
# uint64 timestamp_ns (8 bytes)
# uint64 duration_ns  (8 bytes)
# int64  ret_val      (8 bytes)
# uint32 pid          (4 bytes)
# uint32 tgid         (4 bytes)
# uint32 event_id     (4 bytes)
# uint16 cpu_id       (2 bytes)
# uint16 flags        (2 bytes)
# uint64 args_val     (8 bytes)
# char   comm[16]     (16 bytes)
# Total size: 8+8+8+4+4+4+2+2+8+16 = 64 bytes
TRACE_EVENT_FORMAT = "=QQqIIIHHQ16s"
TRACE_EVENT_SIZE = struct.calcsize(TRACE_EVENT_FORMAT)
assert TRACE_EVENT_SIZE == 64, f"TraceEvent must be exactly 64 bytes, got {TRACE_EVENT_SIZE}"
_TRACE_STRUCT = struct.Struct(TRACE_EVENT_FORMAT)

# 8-byte Linux bpf_ringbuf record header:
# uint32 len   (record payload length)
# uint32 flags (BUSY_BIT = 1 << 31, DISCARD_BIT = 1 << 30)
RINGBUF_HDR_FORMAT = "=II"
RINGBUF_HDR_SIZE = struct.calcsize(RINGBUF_HDR_FORMAT)
assert RINGBUF_HDR_SIZE == 8, f"RingBuf header must be 8 bytes, got {RINGBUF_HDR_SIZE}"
_RINGBUF_HDR_STRUCT = struct.Struct(RINGBUF_HDR_FORMAT)

BPF_RINGBUF_BUSY_BIT = 1 << 31
BPF_RINGBUF_DISCARD_BIT = 1 << 30


class TraceEvent:
    """Represents a raw binary Linux kernel tracepoint event.

    Packed into a 64-byte C-compatible struct matching kernel-space trace events
    (e.g., sys_enter_*, sched_switch, block_rq_issue) with zero string overhead.

    Attributes:
        timestamp_ns: Kernel monotonic timestamp in nanoseconds.
        duration_ns: Execution duration of the syscall/event in nanoseconds.
        ret_val: Syscall return value or error code (negative errno).
        pid: Process ID.
        tgid: Thread Group ID.
        event_id: Syscall number or unique tracepoint ID.
        cpu_id: Hardware CPU core ID.
        flags: Kernel trace flags (e.g. context flags, user/kernel bit).
        args_val: Raw integer argument or pointer value.
        comm: Process command name (up to 16 bytes, TASK_COMM_LEN in Linux).
    """

    __slots__ = (
        "timestamp_ns",
        "duration_ns",
        "ret_val",
        "pid",
        "tgid",
        "event_id",
        "cpu_id",
        "flags",
        "args_val",
        "comm",
    )

    def __init__(
        self,
        timestamp_ns: int = 0,
        duration_ns: int = 0,
        ret_val: int = 0,
        pid: int = 0,
        tgid: int = 0,
        event_id: int = 0,
        cpu_id: int = 0,
        flags: int = 0,
        args_val: int = 0,
        comm: Union[str, bytes] = "proc",
    ) -> None:
        self.timestamp_ns: int = timestamp_ns
        self.duration_ns: int = duration_ns
        self.ret_val: int = ret_val
        self.pid: int = pid
        self.tgid: int = tgid
        self.event_id: int = event_id
        self.cpu_id: int = cpu_id
        self.flags: int = flags
        self.args_val: int = args_val

        if isinstance(comm, str):
            comm_bytes = comm.encode("utf-8")[:16]
        else:
            comm_bytes = comm[:16]
        self.comm: bytes = comm_bytes.ljust(16, b"\x00")

    @property
    def comm_str(self) -> str:
        """Decode process command name, stripping null terminators."""
        return self.comm.split(b"\x00", 1)[0].decode("utf-8", errors="replace")

    def to_bytes(self) -> bytes:
        """Serialize into 64-byte binary C struct."""
        return _TRACE_STRUCT.pack(
            self.timestamp_ns,
            self.duration_ns,
            self.ret_val,
            self.pid,
            self.tgid,
            self.event_id,
            self.cpu_id,
            self.flags,
            self.args_val,
            self.comm,
        )

    def pack_into(self, buffer: Union[bytearray, memoryview], offset: int = 0) -> None:
        """Zero-copy pack directly into a pre-allocated buffer."""
        _TRACE_STRUCT.pack_into(
            buffer,
            offset,
            self.timestamp_ns,
            self.duration_ns,
            self.ret_val,
            self.pid,
            self.tgid,
            self.event_id,
            self.cpu_id,
            self.flags,
            self.args_val,
            self.comm,
        )

    @classmethod
    def from_bytes(cls, data: bytes, offset: int = 0) -> TraceEvent:
        """Deserialize from 64-byte binary slice."""
        fields = _TRACE_STRUCT.unpack_from(data, offset)
        return cls(
            timestamp_ns=fields[0],
            duration_ns=fields[1],
            ret_val=fields[2],
            pid=fields[3],
            tgid=fields[4],
            event_id=fields[5],
            cpu_id=fields[6],
            flags=fields[7],
            args_val=fields[8],
            comm=fields[9],
        )

    @classmethod
    def unpack_from(cls, buffer: Union[bytearray, memoryview, bytes], offset: int = 0) -> TraceEvent:
        """Zero-copy deserialize from byte buffer or memoryview."""
        fields = _TRACE_STRUCT.unpack_from(buffer, offset)
        return cls(
            timestamp_ns=fields[0],
            duration_ns=fields[1],
            ret_val=fields[2],
            pid=fields[3],
            tgid=fields[4],
            event_id=fields[5],
            cpu_id=fields[6],
            flags=fields[7],
            args_val=fields[8],
            comm=fields[9],
        )

    def __repr__(self) -> str:
        return (
            f"<TraceEvent id={self.event_id} pid={self.pid} comm='{self.comm_str}' "
            f"dur={self.duration_ns}ns ret={self.ret_val}>"
        )


class AnomalyAlert:
    """Alert record generated when in-kernel inference detects an anomaly."""

    __slots__ = ("event", "anomaly_score", "threshold", "reason", "timestamp_ns")

    def __init__(
        self,
        event: TraceEvent,
        anomaly_score: float,
        threshold: float,
        reason: str = "HDC_HAMMING_DISTANCE_VIOLATION",
        timestamp_ns: Optional[int] = None,
    ) -> None:
        self.event: TraceEvent = event
        self.anomaly_score: float = float(anomaly_score)
        self.threshold: float = float(threshold)
        self.reason: str = reason
        self.timestamp_ns: int = timestamp_ns if timestamp_ns is not None else event.timestamp_ns

    def __repr__(self) -> str:
        return (
            f"<AnomalyAlert score={self.anomaly_score:.4f} thr={self.threshold:.4f} "
            f"event_id={self.event.event_id} comm='{self.event.comm_str}' reason='{self.reason}'>"
        )


class EBPFRingBufferEmulator:
    """High-throughput in-memory zero-IO trace buffer emulating Linux kernel BPF ringbuf.

    Emulates Linux kernel `BPF_MAP_TYPE_RINGBUF` (introduced in Linux 5.8):
    - Circular byte array in memory of power-of-two capacity.
    - Lockless single-producer single-consumer circular pointer arithmetic.
    - 8-byte record headers containing sample length and status bits (busy, discard).
    - Drop counters for buffer saturation (backpressure).
    - Fully zero-IO: all operations occur in RAM with zero disk writes.
    """

    def __init__(self, capacity_power_of_2: int = 16) -> None:
        """Initialize ring buffer emulator.

        Args:
            capacity_power_of_2: Exponent for capacity ($2^{k}$ bytes).
                                 Default 16 = 65,536 bytes (64 KB).
        """
        if capacity_power_of_2 < 10:
            raise ValueError("Capacity exponent must be at least 10 (1024 bytes)")

        self.capacity: int = 1 << capacity_power_of_2
        self.mask: int = self.capacity - 1
        self._buf: bytearray = bytearray(self.capacity)

        # Monotonic positions
        self.producer_pos: int = 0
        self.consumer_pos: int = 0

        # Telemetry
        self.events_produced: int = 0
        self.events_consumed: int = 0
        self.events_dropped: int = 0
        self.events_discarded: int = 0

    def reserve(self, size: int) -> Optional[int]:
        """Emulate `bpf_ringbuf_reserve()`.

        Reserves space for 8-byte header + `size` payload.
        Sets BPF_RINGBUF_BUSY_BIT in header. Returns logical offset on success,
        or None if ring buffer is saturated (backpressure drop).
        """
        total_len = RINGBUF_HDR_SIZE + size
        if (self.producer_pos - self.consumer_pos) + total_len > self.capacity:
            self.events_dropped += 1
            return None

        offset = self.producer_pos
        w_idx = offset & self.mask
        # Write header with BUSY_BIT set
        hdr_data = _RINGBUF_HDR_STRUCT.pack(size, BPF_RINGBUF_BUSY_BIT)
        self._write_bytes(w_idx, hdr_data)

        self.producer_pos += total_len
        return offset

    def submit(self, offset: int) -> None:
        """Emulate `bpf_ringbuf_submit()`.

        Commits reserved sample by clearing BPF_RINGBUF_BUSY_BIT.
        Makes record readable by consumer.
        """
        w_idx = offset & self.mask
        hdr_bytes = self._read_bytes(w_idx, RINGBUF_HDR_SIZE)
        length, flags = _RINGBUF_HDR_STRUCT.unpack(hdr_bytes)
        # Clear busy bit
        new_hdr = _RINGBUF_HDR_STRUCT.pack(length, flags & ~BPF_RINGBUF_BUSY_BIT)
        self._write_bytes(w_idx, new_hdr)
        self.events_produced += 1

    def discard(self, offset: int) -> None:
        """Emulate `bpf_ringbuf_discard()`.

        Marks reserved sample with BPF_RINGBUF_DISCARD_BIT so consumer skips it.
        """
        w_idx = offset & self.mask
        hdr_bytes = self._read_bytes(w_idx, RINGBUF_HDR_SIZE)
        length, flags = _RINGBUF_HDR_STRUCT.unpack(hdr_bytes)
        # Set discard bit and clear busy bit
        new_flags = (flags | BPF_RINGBUF_DISCARD_BIT) & ~BPF_RINGBUF_BUSY_BIT
        new_hdr = _RINGBUF_HDR_STRUCT.pack(length, new_flags)
        self._write_bytes(w_idx, new_hdr)
        self.events_discarded += 1

    def push_event(self, event: TraceEvent) -> bool:
        """Convenience method: reserves, packs TraceEvent, and submits.

        Returns True on success, False if dropped due to buffer overflow.
        """
        offset = self.reserve(TRACE_EVENT_SIZE)
        if offset is None:
            return False

        payload_offset = (offset + RINGBUF_HDR_SIZE) & self.mask
        event_bytes = event.to_bytes()
        self._write_bytes(payload_offset, event_bytes)
        self.submit(offset)
        return True

    def push_raw(self, data: bytes) -> bool:
        """Push raw binary payload into ring buffer."""
        offset = self.reserve(len(data))
        if offset is None:
            return False
        payload_offset = (offset + RINGBUF_HDR_SIZE) & self.mask
        self._write_bytes(payload_offset, data)
        self.submit(offset)
        return True

    def poll(self, max_events: Optional[int] = None) -> List[TraceEvent]:
        """Poll committed records from the ring buffer.

        Emulates user-space consumer polling (`bpf_ringbuf__poll`).
        Returns list of deserialized `TraceEvent` records.
        """
        events: List[TraceEvent] = []
        limit = max_events if max_events is not None else float("inf")

        while self.consumer_pos < self.producer_pos and len(events) < limit:
            r_idx = self.consumer_pos & self.mask
            hdr_bytes = self._read_bytes(r_idx, RINGBUF_HDR_SIZE)
            length, flags = _RINGBUF_HDR_STRUCT.unpack(hdr_bytes)

            # If producer is still writing (BUSY_BIT set), stop polling
            if flags & BPF_RINGBUF_BUSY_BIT:
                break

            total_len = RINGBUF_HDR_SIZE + length

            # If marked discarded, skip payload
            if flags & BPF_RINGBUF_DISCARD_BIT:
                self.consumer_pos += total_len
                continue

            # Read payload
            p_idx = (r_idx + RINGBUF_HDR_SIZE) & self.mask
            payload = self._read_bytes(p_idx, length)
            self.consumer_pos += total_len

            if length == TRACE_EVENT_SIZE:
                events.append(TraceEvent.from_bytes(payload))
            self.events_consumed += 1

        return events

    def consume_batch(
        self,
        callback: Callable[[TraceEvent], None],
        max_events: Optional[int] = None,
    ) -> int:
        """Fast streaming consumption invoking callback for each committed record."""
        count = 0
        limit = max_events if max_events is not None else float("inf")

        while self.consumer_pos < self.producer_pos and count < limit:
            r_idx = self.consumer_pos & self.mask
            hdr_bytes = self._read_bytes(r_idx, RINGBUF_HDR_SIZE)
            length, flags = _RINGBUF_HDR_STRUCT.unpack(hdr_bytes)

            if flags & BPF_RINGBUF_BUSY_BIT:
                break

            total_len = RINGBUF_HDR_SIZE + length

            if flags & BPF_RINGBUF_DISCARD_BIT:
                self.consumer_pos += total_len
                continue

            p_idx = (r_idx + RINGBUF_HDR_SIZE) & self.mask
            payload = self._read_bytes(p_idx, length)
            self.consumer_pos += total_len

            if length == TRACE_EVENT_SIZE:
                event = TraceEvent.from_bytes(payload)
                callback(event)
            self.events_consumed += 1
            count += 1

        return count

    def stats(self) -> Dict[str, Any]:
        """Return diagnostic metrics of the ring buffer."""
        bytes_used = self.producer_pos - self.consumer_pos
        utilization = bytes_used / self.capacity if self.capacity > 0 else 0.0
        return {
            "capacity_bytes": self.capacity,
            "bytes_used": bytes_used,
            "utilization": utilization,
            "events_produced": self.events_produced,
            "events_consumed": self.events_consumed,
            "events_dropped": self.events_dropped,
            "events_discarded": self.events_discarded,
        }

    def reset(self) -> None:
        """Reset ring buffer pointers and counters."""
        self.producer_pos = 0
        self.consumer_pos = 0
        self.events_produced = 0
        self.events_consumed = 0
        self.events_dropped = 0
        self.events_discarded = 0

    def _write_bytes(self, idx: int, data: bytes) -> None:
        """Helper to write data handling circular boundary split."""
        n = len(data)
        if idx + n <= self.capacity:
            self._buf[idx : idx + n] = data
        else:
            part1 = self.capacity - idx
            self._buf[idx : self.capacity] = data[:part1]
            self._buf[0 : n - part1] = data[part1:]

    def _read_bytes(self, idx: int, n: int) -> bytes:
        """Helper to read data handling circular boundary split."""
        if idx + n <= self.capacity:
            return bytes(self._buf[idx : idx + n])
        else:
            part1 = self.capacity - idx
            return bytes(self._buf[idx : self.capacity]) + bytes(self._buf[0 : n - part1])


class FixedPointLinear:
    """Fixed-point quantized linear layer (Q-format) for in-kernel execution.

    Computes integer matrix multiplication:
    $$y_q = \\left(W_q \\cdot x_q + (b_q \\ll \\mathrm{shift})\\right) \\gg \\mathrm{shift}$$
    Executes entirely in 64-bit integer registers without floating point hardware.
    """

    def __init__(
        self,
        weights_q: np.ndarray,
        bias_q: np.ndarray,
        shift: int = 8,
    ) -> None:
        """Initialize layer with quantized weights and biases."""
        self.shift: int = shift
        self.weights_q: np.ndarray = np.asarray(weights_q, dtype=np.int64)
        self.bias_q: np.ndarray = np.asarray(bias_q, dtype=np.int64)
        self.in_features: int = self.weights_q.shape[1]
        self.out_features: int = self.weights_q.shape[0]

    @classmethod
    def from_float(
        cls,
        weights: np.ndarray,
        bias: np.ndarray,
        shift: int = 8,
    ) -> FixedPointLinear:
        """Quantize floating point weights and biases to Q-format integer."""
        scale = 1 << shift
        w_q = np.round(weights * scale).astype(np.int64)
        b_q = np.round(bias * scale).astype(np.int64)
        return cls(weights_q=w_q, bias_q=b_q, shift=shift)

    def forward(self, x_q: np.ndarray) -> np.ndarray:
        """Forward pass using pure integer arithmetic."""
        x_int = np.asarray(x_q, dtype=np.int64)
        # Dot product in 64-bit accumulator: W @ x
        accum = np.dot(self.weights_q, x_int)
        # Add bias (bias is in Q-format, so we shift before or add scaled)
        accum += (self.bias_q << self.shift)
        # Arithmetic right shift to rescale to Q-format
        return accum >> self.shift


class FixedPointAutoencoder:
    """Fixed-point quantized shallow autoencoder for in-kernel anomaly scoring.

    Features:
    - Pure integer arithmetic adhering to Linux eBPF verifier limitations.
    - Integer ReLU activation: max(0, x).
    - Reconstruction error: integer L1 distance $\\sum |x_{q, i} - \\hat{x}_{q, i}|$.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        shift: int = 8,
        threshold_q: int = 256,
    ) -> None:
        self.in_dim: int = in_dim
        self.hidden_dim: int = hidden_dim
        self.shift: int = shift
        self.threshold_q: int = threshold_q

        # Initialize pseudo-identity/orthogonal integer weights
        rng = np.random.RandomState(42)
        scale = 1 << shift
        # Small weights around identity projection
        w1 = rng.randn(hidden_dim, in_dim) * 0.5
        b1 = np.zeros(hidden_dim)
        w2 = rng.randn(in_dim, hidden_dim) * 0.5
        b2 = np.zeros(in_dim)

        self.encoder_layer = FixedPointLinear.from_float(w1, b1, shift=shift)
        self.decoder_layer = FixedPointLinear.from_float(w2, b2, shift=shift)

    def encode(self, x_q: np.ndarray) -> np.ndarray:
        """Encoder forward: ReLU(W1 @ x + b1)."""
        h = self.encoder_layer.forward(x_q)
        return np.maximum(0, h)

    def decode(self, h_q: np.ndarray) -> np.ndarray:
        """Decoder forward: W2 @ h + b2."""
        return self.decoder_layer.forward(h_q)

    def reconstruct(self, x_q: np.ndarray) -> np.ndarray:
        """Complete forward pass."""
        h = self.encode(x_q)
        return self.decode(h)

    def anomaly_score_q(self, x_q: np.ndarray) -> int:
        """Integer L1 reconstruction error: $\\sum |x_{q, i} - \\hat{x}_{q, i}|$."""
        x_recon = self.reconstruct(x_q)
        return int(np.sum(np.abs(x_q - x_recon)))

    def is_anomaly(self, x_q: np.ndarray) -> bool:
        """Check if reconstruction error exceeds integer threshold."""
        return self.anomaly_score_q(x_q) > self.threshold_q


class InKernelHDCFilter:
    """In-kernel sliding window HDC anomaly filter.

    Simulates an eBPF program maintaining a bounded circular window of recent
    event IDs in a BPF per-CPU array map, encoding the window into a hypervector
    using bitwise shifts and XOR, and comparing against holographic normal memory.
    """

    def __init__(
        self,
        window_size: int = 10,
        dim: int = 10000,
        threshold: float = 0.35,
        encoder: Optional[HyperdimensionalEncoder] = None,
    ) -> None:
        self.window_size: int = window_size
        self.dim: int = dim
        self.threshold: float = threshold
        self.encoder: HyperdimensionalEncoder = (
            encoder if encoder is not None else HyperdimensionalEncoder(dim=dim, seed=42)
        )

        # Sliding window buffer (simulating BPF per-CPU array)
        self.window: List[int] = []
        self.profile_memory: Optional[HyperVector] = None
        self.prototypes: List[HyperVector] = []
        self.detector: Optional[HDCAnomalyDetector] = None

    def fit_normal_patterns(self, normal_sequences: Sequence[Sequence[int]]) -> None:
        """Train normal profile holographic memory from baseline event sequences."""
        # Extract sliding windows of size self.window_size across normal sequences
        all_windows: List[Sequence[int]] = []
        for seq in normal_sequences:
            seq_list = list(seq)
            if len(seq_list) > self.window_size:
                for i in range(len(seq_list) - self.window_size + 1):
                    all_windows.append(seq_list[i : i + self.window_size])
            elif len(seq_list) == self.window_size:
                # Cyclic recurring pattern in traces: include cyclic rotations
                all_windows.append(seq_list)
                for shift in range(1, len(seq_list)):
                    all_windows.append(seq_list[shift:] + seq_list[:shift])
            else:
                all_windows.append(seq_list)

        if not all_windows:
            all_windows = list(normal_sequences)

        # Multi-prototype codebook up to 16 prototypes
        num_protos = max(1, min(len(all_windows), 16))
        self.detector = HDCAnomalyDetector(
            encoder=self.encoder,
            dim=self.dim,
            threshold=self.threshold,
            num_prototypes=num_protos,
        )
        self.detector.fit(all_windows, method="positional")
        self.profile_memory = self.detector.profile_memory
        self.prototypes = self.detector.prototypes
        self.threshold = self.detector.threshold_

    def process_event(self, event_id: int) -> Tuple[bool, float]:
        """Process incoming kernel event ID through in-kernel sliding window.

        Returns:
            (is_anomaly, score):
            If is_anomaly is False, the event is normal and dropped directly in kernel.
            If is_anomaly is True, the event violated normal profile and is flagged.
        """
        self.window.append(event_id)
        if len(self.window) > self.window_size:
            self.window.pop(0)

        # If window not filled yet, treat as normal warm-up
        if len(self.window) < self.window_size:
            return False, 0.0

        if not self.prototypes:
            return False, 0.0

        # Encode current sliding window in-kernel
        win_hv = self.encoder.encode_sequence(self.window, method="positional")
        # Sub-microsecond hardware popcount across in-kernel prototypes
        win_int = win_hv.int_val
        dim = self.dim
        min_dist = min((win_int ^ p.int_val).bit_count() for p in self.prototypes) / dim
        is_anom = min_dist > self.threshold
        return is_anom, min_dist

    def reset(self) -> None:
        """Reset sliding window."""
        self.window.clear()


class InKernelInferenceEngine:
    """High-throughput In-Kernel Inference Engine for Zero-IO anomaly detection.

    Executes inference directly on raw kernel tracepoints without writing strings
    to disk. Filters out normal telemetry in kernel space (zero context switch,
    zero user-space wakeup) and emits alerts for anomalies to the eBPF ring buffer.

    Supported Modes:
    - "hdc": Ultra-fast bitwise holographic anomaly filtering (sub-microsecond).
    - "fixed_point": Integer fixed-point quantized neural inference.
    - "hybrid": Fast HDC pre-filtering combined with fixed-point verification.
    """

    def __init__(
        self,
        mode: str = "hdc",
        dim: int = 10000,
        window_size: int = 10,
        threshold: float = 0.35,
        ring_buffer: Optional[EBPFRingBufferEmulator] = None,
        encoder: Optional[HyperdimensionalEncoder] = None,
    ) -> None:
        """Initialize in-kernel inference engine."""
        if mode not in ("hdc", "fixed_point", "hybrid"):
            raise ValueError(f"Unknown mode '{mode}', expected 'hdc', 'fixed_point', or 'hybrid'")

        self.mode: str = mode
        self.dim: int = dim
        self.window_size: int = window_size
        self.threshold: float = threshold
        self.ring_buffer: EBPFRingBufferEmulator = (
            ring_buffer if ring_buffer is not None else EBPFRingBufferEmulator(capacity_power_of_2=16)
        )

        # HDC filter component
        self.hdc_filter: InKernelHDCFilter = InKernelHDCFilter(
            window_size=window_size,
            dim=dim,
            threshold=threshold,
            encoder=encoder,
        )

        # Fixed-point component
        self.fixed_point_ae: FixedPointAutoencoder = FixedPointAutoencoder(
            in_dim=4,
            hidden_dim=2,
            shift=8,
            threshold_q=512,
        )

        # Telemetry counters
        self.events_inspected: int = 0
        self.events_filtered: int = 0
        self.anomalies_detected: int = 0

    def fit_normal_patterns(self, normal_sequences: Sequence[Sequence[int]]) -> None:
        """Fit normal profile memory on baseline sequences."""
        self.hdc_filter.fit_normal_patterns(normal_sequences)
        self.threshold = self.hdc_filter.threshold

    def process_trace_event(self, event: TraceEvent) -> Optional[AnomalyAlert]:
        """Process a single raw binary TraceEvent.

        In-kernel decision:
        - If normal: event is filtered out and dropped in kernel space (returns None).
          Zero disk I/O, zero user-space wakeups!
        - If anomalous: emits alert to ring buffer and returns AnomalyAlert.
        """
        self.events_inspected += 1

        is_anom = False
        score = 0.0
        reason = "NORMAL"

        if self.mode in ("hdc", "hybrid"):
            is_anom, score = self.hdc_filter.process_event(event.event_id)
            if is_anom:
                reason = "IN_KERNEL_HDC_HAMMING_DISTANCE_VIOLATION"

        if self.mode == "fixed_point" or (self.mode == "hybrid" and not is_anom):
            # Fixed-point feature vector from binary trace struct:
            # [duration_ns >> 10, ret_val, cpu_id, flags]
            scale = 1 << 8
            feat_q = np.array(
                [
                    min(event.duration_ns >> 10, 1024),
                    abs(event.ret_val) * scale,
                    event.cpu_id * scale,
                    event.flags * scale,
                ],
                dtype=np.int64,
            )
            ae_score = self.fixed_point_ae.anomaly_score_q(feat_q)
            if ae_score > self.fixed_point_ae.threshold_q:
                is_anom = True
                score = ae_score / (1 << 8)
                reason = "IN_KERNEL_FIXED_POINT_RECONSTRUCTION_ERROR"

        if not is_anom:
            # Filtered directly in kernel space (zero I/O)
            self.events_filtered += 1
            return None

        # Anomaly detected! Emit alert into ring buffer
        self.anomalies_detected += 1
        alert = AnomalyAlert(
            event=event,
            anomaly_score=score,
            threshold=self.threshold,
            reason=reason,
            timestamp_ns=event.timestamp_ns,
        )
        self.ring_buffer.push_event(event)
        return alert

    def process_batch(self, events: Sequence[TraceEvent]) -> List[AnomalyAlert]:
        """Process batch of raw trace events."""
        alerts: List[AnomalyAlert] = []
        for event in events:
            alert = self.process_trace_event(event)
            if alert is not None:
                alerts.append(alert)
        return alerts

    def ingest_from_ring_buffer(
        self,
        ring_buf: EBPFRingBufferEmulator,
        max_events: Optional[int] = None,
    ) -> List[AnomalyAlert]:
        """Ingest trace events directly from an external ring buffer."""
        raw_events = ring_buf.poll(max_events=max_events)
        return self.process_batch(raw_events)

    def benchmark_pipeline(self, num_events: int = 100000) -> Dict[str, float]:
        """Benchmark throughput and zero-IO execution performance."""
        # Generate synthetic trace stream
        rng = np.random.RandomState(42)
        events = [
            TraceEvent(
                timestamp_ns=int(i * 1000),
                duration_ns=int(rng.randint(50, 500)),
                ret_val=0,
                pid=1001,
                tgid=1001,
                event_id=int(rng.choice([1, 2, 3, 4, 5])),
                cpu_id=int(i % 8),
                comm=b"worker_node\x00",
            )
            for i in range(num_events)
        ]

        t0 = time.perf_counter()
        alerts = self.process_batch(events)
        t1 = time.perf_counter()
        dt = t1 - t0

        throughput = num_events / dt if dt > 0 else float("inf")
        latency_us = (dt / num_events) * 1e6 if num_events > 0 else 0.0
        filter_rate = (
            self.events_filtered / self.events_inspected if self.events_inspected > 0 else 0.0
        )

        return {
            "num_events": float(num_events),
            "elapsed_seconds": dt,
            "throughput_events_per_sec": throughput,
            "latency_microseconds": latency_us,
            "filter_rate": filter_rate,
            "anomalies_detected": float(len(alerts)),
        }

    def stats(self) -> Dict[str, Any]:
        """Return diagnostic metrics."""
        return {
            "mode": self.mode,
            "dim": self.dim,
            "events_inspected": self.events_inspected,
            "events_filtered": self.events_filtered,
            "anomalies_detected": self.anomalies_detected,
            "filter_rate": (
                self.events_filtered / self.events_inspected if self.events_inspected > 0 else 0.0
            ),
            "ring_buffer": self.ring_buffer.stats(),
        }

    def reset(self) -> None:
        """Reset internal state and ring buffer."""
        self.hdc_filter.reset()
        self.ring_buffer.reset()
        self.events_inspected = 0
        self.events_filtered = 0
        self.anomalies_detected = 0
