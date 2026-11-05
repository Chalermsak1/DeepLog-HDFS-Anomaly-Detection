"""
DeepLog-Omni Frontier 2: Multi-Modal OpenTelemetry (OTel) Full-Stack Convergence (MELT + Profiling)
=================================================================================================
Unifies Prometheus Metrics, OpenTelemetry Traces, Distributed Log Records, and Continuous Code
Profiling (flamegraphs / call stacks) into a single synchronized representation.

Components:
1. Data Models:
   - `MetricPoint`: Prometheus gauge/counter metrics with multi-dimensional labels.
   - `TraceSpan`: Distributed trace span with span_id, parent_id, service, duration, status.
   - `LogRecord`: Structured log event with template, level, event_id, and trace correlation.
   - `ProfileSample`: Continuous flamegraph call stack frames with CPU/allocation weights.
   - `TelemetryWindow`: Correlated multimodal container spanning a synchronized time window.

2. `UnifiedTelemetryStream`:
   - High-throughput stream buffer ingesting heterogeneous MELT + Profiling signals.
   - Synchronizes multi-modal streams by timestamp intervals and distributed trace IDs.
   - Resamples and tokenizes heterogeneous modalities into aligned tensor batches.

3. `TelemetryFusionModel`:
   - PyTorch multimodal cross-attention network with system query token q_sys.
   - Fuses log tokens, metric time-series tensors, trace span latencies, and flamegraph stack
     embeddings into a unified system state z_system in R^{d_model}.
   - Modality-type encodings, dynamic gating heads, and anomaly prediction heads.

4. `CrossModalAttributor`:
   - Isolates whether an anomaly is primarily driven by:
     * "LOG_ERRORS" (unseen error templates, invalid state transitions)
     * "METRIC_SATURATION" (CPU/memory saturation spikes, I/O exhaustion)
     * "TRACE_BOTTLENECK" (distributed latency spikes, downstream RPC timeouts)
     * "PROFILE_HOTSPOT" (excessive CPU flamegraph stack weight in specific functions)
   - Multi-method attribution: Cross-Attention Saliency, Gradient x Input, Integrated Gradients,
     and Perturbation / Ablation.
   - Outputs normalized diagnostic attribution percentages (summing to 100%) and fine-grained
     evidence summaries for SRE & DevOps root-cause triage.
"""

from __future__ import annotations

import copy
import enum
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# 1. Telemetry Data Models (MELT + Continuous Profiling)
# ==============================================================================


class MetricType(str, enum.Enum):
    """Metric types aligned with Prometheus and OpenTelemetry metrics."""
    GAUGE = "gauge"
    COUNTER = "counter"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MetricPoint:
    """
    Structured Metric data point representing a Prometheus or OTel metric observation.

    Parameters
    ----------
    name : str
        Metric identifier (e.g., 'cpu_utilization', 'http_requests_total', 'memory_bytes').
    timestamp : float
        Observation timestamp in epoch seconds.
    value : float
        Numerical measurement value.
    labels : Dict[str, str], optional
        Multi-dimensional metric dimensions/labels (e.g., {'service': 'checkout', 'host': 'node-1'}).
    metric_type : str or MetricType, default 'gauge'
        Metric classification ('gauge', 'counter', 'histogram', 'summary').
    unit : Optional[str], optional
        Physical unit of measurement (e.g., 'seconds', 'bytes', 'percent').
    """
    name: str
    timestamp: float
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    metric_type: str = "gauge"
    unit: Optional[str] = None

    def identifier(self) -> str:
        """Unique series identifier based on name and sorted labels."""
        if not self.labels:
            return self.name
        sorted_labels = ",".join(f'{k}="{v}"' for k, v in sorted(self.labels.items()))
        return f"{self.name}{{{sorted_labels}}}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert metric point to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MetricPoint:
        """Instantiate MetricPoint from dictionary."""
        return cls(
            name=data["name"],
            timestamp=float(data["timestamp"]),
            value=float(data["value"]),
            labels=dict(data.get("labels", {})),
            metric_type=str(data.get("metric_type", "gauge")),
            unit=data.get("unit"),
        )


@dataclass
class TraceSpan:
    """
    OpenTelemetry distributed trace span representation.

    Parameters
    ----------
    span_id : str
        Unique hexadecimal identifier for the span.
    trace_id : str
        Hexadecimal identifier for the end-to-end distributed trace.
    parent_id : Optional[str], optional
        Span ID of the parent caller span, or None if root span.
    service : str, default 'default'
        Microservice name executing the span operation.
    name : str, default 'operation'
        Operation or RPC name (e.g., 'POST /api/v1/checkout').
    start_time : float, default 0.0
        Start timestamp in epoch seconds.
    end_time : float, default 0.0
        End timestamp in epoch seconds.
    duration : float, default 0.0
        Span duration in seconds. If 0.0 and end_time > start_time, automatically computed.
    status_code : str, default 'OK'
        Span status ('OK', 'ERROR', 'UNSET').
    attributes : Dict[str, Any], optional
        OTel semantic attributes (e.g., {'http.status_code': 500, 'rpc.method': 'Pay'}).
    events : List[Dict[str, Any]], optional
        In-span events or exception logs.
    """
    span_id: str
    trace_id: str
    parent_id: Optional[str] = None
    service: str = "default"
    name: str = "operation"
    start_time: float = 0.0
    end_time: float = 0.0
    duration: float = 0.0
    status_code: str = "OK"
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.duration <= 0.0 and self.end_time > self.start_time:
            self.duration = float(self.end_time - self.start_time)

    @property
    def is_error(self) -> bool:
        """Check if span indicates an execution or RPC failure."""
        if self.status_code.upper() == "ERROR":
            return True
        if self.attributes.get("error", False):
            return True
        http_code = self.attributes.get("http.status_code")
        if http_code is not None and int(http_code) >= 500:
            return True
        return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert span to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TraceSpan:
        """Instantiate TraceSpan from dictionary."""
        return cls(
            span_id=data["span_id"],
            trace_id=data["trace_id"],
            parent_id=data.get("parent_id"),
            service=str(data.get("service", "default")),
            name=str(data.get("name", "operation")),
            start_time=float(data.get("start_time", 0.0)),
            end_time=float(data.get("end_time", 0.0)),
            duration=float(data.get("duration", 0.0)),
            status_code=str(data.get("status_code", "OK")),
            attributes=dict(data.get("attributes", {})),
            events=list(data.get("events", [])),
        )


@dataclass
class LogRecord:
    """
    Structured Log Record mapped to DeepLog event template IDs.

    Parameters
    ----------
    event_id : int
        Categorical template integer index (vocabulary token ID).
    timestamp : float
        Epoch seconds timestamp of the log event.
    template : str
        Parsed template string (e.g., 'Connection refused to host <*> on port <*>').
    level : str, default 'INFO'
        Logging severity level ('DEBUG', 'INFO', 'WARN', 'ERROR', 'FATAL').
    service : Optional[str], optional
        Service name generating the log message.
    trace_id : Optional[str], optional
        Correlated OTel trace ID.
    span_id : Optional[str], optional
        Correlated OTel span ID.
    attributes : Dict[str, Any], optional
        Extracted variable parameters or structured payload fields.
    raw_message : Optional[str], optional
        Original unparsed raw log line.
    """
    event_id: int
    timestamp: float
    template: str
    level: str = "INFO"
    service: Optional[str] = None
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    raw_message: Optional[str] = None

    @property
    def is_error(self) -> bool:
        """Check if log represents an error or critical failure."""
        lvl = self.level.upper()
        if lvl in {"ERROR", "FATAL", "CRITICAL"}:
            return True
        tmpl_lower = self.template.lower()
        if "exception" in tmpl_lower or "failed" in tmpl_lower or "panic" in tmpl_lower:
            return True
        return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert log record to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> LogRecord:
        """Instantiate LogRecord from dictionary."""
        return cls(
            event_id=int(data["event_id"]),
            timestamp=float(data["timestamp"]),
            template=str(data["template"]),
            level=str(data.get("level", "INFO")),
            service=data.get("service"),
            trace_id=data.get("trace_id"),
            span_id=data.get("span_id"),
            attributes=dict(data.get("attributes", {})),
            raw_message=data.get("raw_message"),
        )


@dataclass
class ProfileSample:
    """
    Continuous Profiling sample (pprof / Pyroscope / Parca / eBPF flamegraph stack).

    Parameters
    ----------
    call_stack : List[str]
        Call stack frames ordered from root to leaf
        (e.g., ['main', 'net/http.HandlerFunc.ServeHTTP', 'runtime.mallocgc']).
    weight : float, default 1.0
        Cumulative profiling weight (e.g., CPU sample counts, CPU nanoseconds, allocation bytes).
    timestamp : float, default 0.0
        Observation timestamp in epoch seconds.
    sample_type : str, default 'cpu'
        Profiling dimension ('cpu', 'alloc_bytes', 'alloc_objects', 'lock_nanoseconds').
    service : Optional[str], optional
        Service producing the continuous profiling sample.
    thread_id : Optional[int], optional
        OS thread or goroutine ID.
    labels : Dict[str, str], optional
        Stack-level metadata and tags.
    """
    call_stack: List[str]
    weight: float = 1.0
    timestamp: float = 0.0
    sample_type: str = "cpu"
    service: Optional[str] = None
    thread_id: Optional[int] = None
    labels: Dict[str, str] = field(default_factory=dict)

    @property
    def leaf_frame(self) -> str:
        """Return the leaf (innermost) active execution frame."""
        return self.call_stack[-1] if self.call_stack else "<empty>"

    def stack_str(self, delimiter: str = ";") -> str:
        """Render folded stack string compatible with flamegraph tools."""
        return delimiter.join(self.call_stack)

    def to_dict(self) -> Dict[str, Any]:
        """Convert profile sample to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ProfileSample:
        """Instantiate ProfileSample from dictionary."""
        return cls(
            call_stack=list(data.get("call_stack", [])),
            weight=float(data.get("weight", 1.0)),
            timestamp=float(data.get("timestamp", 0.0)),
            sample_type=str(data.get("sample_type", "cpu")),
            service=data.get("service"),
            thread_id=data.get("thread_id"),
            labels=dict(data.get("labels", {})),
        )


@dataclass
class TelemetryWindow:
    """
    Synchronized multimodal window frame holding time-aligned telemetry records.

    Parameters
    ----------
    window_id : str
        Unique identifier for the telemetry time window.
    start_time : float
        Window start time in epoch seconds (inclusive).
    end_time : float
        Window end time in epoch seconds (exclusive).
    metrics : List[MetricPoint]
        Metrics observed during this window.
    spans : List[TraceSpan]
        Distributed trace spans active during this window.
    logs : List[LogRecord]
        Structured log events emitted during this window.
    profiles : List[ProfileSample]
        Profiling stack samples captured during this window.
    metadata : Dict[str, Any]
        Optional operational metadata.
    """
    window_id: str
    start_time: float
    end_time: float
    metrics: List[MetricPoint] = field(default_factory=list)
    spans: List[TraceSpan] = field(default_factory=list)
    logs: List[LogRecord] = field(default_factory=list)
    profiles: List[ProfileSample] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        """Duration of the time window in seconds."""
        return max(0.0, self.end_time - self.start_time)

    @property
    def is_empty(self) -> bool:
        """Check if window contains no telemetry data across all 4 modalities."""
        return (
            len(self.metrics) == 0
            and len(self.spans) == 0
            and len(self.logs) == 0
            and len(self.profiles) == 0
        )

    def counts(self) -> Dict[str, int]:
        """Return counts per modality."""
        return {
            "metrics": len(self.metrics),
            "spans": len(self.spans),
            "logs": len(self.logs),
            "profiles": len(self.profiles),
        }


# ==============================================================================
# 2. Unified Telemetry Stream (Multi-Modal Stream Synchronizer)
# ==============================================================================


class UnifiedTelemetryStream:
    """
    Ingests and synchronizes multi-modal OTel streams (MELT + Continuous Profiling).

    Maintains in-memory temporal index structures and trace correlation graphs:
    - Time-slice alignment: partitions asynchronous streams into fixed or sliding windows.
    - Trace ID correlation: links distributed spans with log records.
    - Tensor compilation: formats synchronized telemetry into tensor batches ready for
      `TelemetryFusionModel`.
    """

    def __init__(
        self,
        default_window_size: float = 10.0,
        default_step_size: float = 5.0,
        metric_steps: int = 10,
        num_metrics: int = 16,
        max_logs: int = 64,
        max_spans: int = 32,
        max_profiles: int = 16,
        trace_feat_dim: int = 8,
        profile_feat_dim: int = 16,
    ):
        self.default_window_size = float(default_window_size)
        self.default_step_size = float(default_step_size)
        self.metric_steps = int(metric_steps)
        self.num_metrics = int(num_metrics)
        self.max_logs = int(max_logs)
        self.max_spans = int(max_spans)
        self.max_profiles = int(max_profiles)
        self.trace_feat_dim = int(trace_feat_dim)
        self.profile_feat_dim = int(profile_feat_dim)

        # Telemetry storage
        self._metrics: List[MetricPoint] = []
        self._spans: List[TraceSpan] = []
        self._logs: List[LogRecord] = []
        self._profiles: List[ProfileSample] = []

        # Index structures
        self._trace_spans: Dict[str, List[TraceSpan]] = {}
        self._trace_logs: Dict[str, List[LogRecord]] = {}
        self._service_registry: Dict[str, int] = {}
        self._metric_registry: Dict[str, int] = {}
        self._frame_registry: Dict[str, int] = {}

    def clear(self) -> None:
        """Clear all stored telemetry and correlation indices."""
        self._metrics.clear()
        self._spans.clear()
        self._logs.clear()
        self._profiles.clear()
        self._trace_spans.clear()
        self._trace_logs.clear()

    # --- Ingestion APIs ---

    def add_metric(self, metric: MetricPoint) -> None:
        """Ingest a single MetricPoint."""
        self._metrics.append(metric)
        m_id = metric.name
        if m_id not in self._metric_registry:
            self._metric_registry[m_id] = len(self._metric_registry) % self.num_metrics

    def add_span(self, span: TraceSpan) -> None:
        """Ingest a single TraceSpan and update trace correlation index."""
        self._spans.append(span)
        if span.trace_id:
            self._trace_spans.setdefault(span.trace_id, []).append(span)
        if span.service and span.service not in self._service_registry:
            self._service_registry[span.service] = len(self._service_registry)

    def add_log(self, log: LogRecord) -> None:
        """Ingest a single LogRecord and update trace correlation index."""
        self._logs.append(log)
        if log.trace_id:
            self._trace_logs.setdefault(log.trace_id, []).append(log)
        if log.service and log.service not in self._service_registry:
            self._service_registry[log.service] = len(self._service_registry)

    def add_profile(self, profile: ProfileSample) -> None:
        """Ingest a continuous flamegraph ProfileSample."""
        self._profiles.append(profile)
        for frame in profile.call_stack:
            if frame not in self._frame_registry:
                self._frame_registry[frame] = len(self._frame_registry)
        if profile.service and profile.service not in self._service_registry:
            self._service_registry[profile.service] = len(self._service_registry)

    def add_batch(
        self,
        metrics: Optional[Sequence[MetricPoint]] = None,
        spans: Optional[Sequence[TraceSpan]] = None,
        logs: Optional[Sequence[LogRecord]] = None,
        profiles: Optional[Sequence[ProfileSample]] = None,
    ) -> None:
        """Batch ingestion helper for all modalities."""
        if metrics:
            for m in metrics:
                self.add_metric(m)
        if spans:
            for s in spans:
                self.add_span(s)
        if logs:
            for l in logs:
                self.add_log(l)
        if profiles:
            for p in profiles:
                self.add_profile(p)

    # --- Retrieval & Synchronization ---

    def get_time_bounds(self) -> Tuple[float, float]:
        """Compute the global minimum and maximum timestamps across all modalities."""
        timestamps: List[float] = []
        if self._metrics:
            timestamps.extend([m.timestamp for m in self._metrics])
        if self._spans:
            timestamps.extend([s.start_time for s in self._spans])
            timestamps.extend([s.end_time for s in self._spans if s.end_time > 0])
        if self._logs:
            timestamps.extend([l.timestamp for l in self._logs])
        if self._profiles:
            timestamps.extend([p.timestamp for p in self._profiles])

        if not timestamps:
            return (0.0, 0.0)
        return (float(min(timestamps)), float(max(timestamps)))

    def get_window(
        self,
        start_time: float,
        end_time: float,
        service: Optional[str] = None,
    ) -> TelemetryWindow:
        """
        Extract all multimodal records falling strictly within [start_time, end_time).
        """
        w_id = f"win_{start_time:.3f}_{end_time:.3f}"

        # Filter metrics
        filtered_metrics = [
            m for m in self._metrics
            if start_time <= m.timestamp < end_time
            and (service is None or m.labels.get("service") == service)
        ]

        # Filter spans (active if overlapping with [start_time, end_time))
        filtered_spans = [
            s for s in self._spans
            if (s.start_time < end_time and (s.end_time >= start_time or s.start_time >= start_time))
            and (service is None or s.service == service)
        ]

        # Filter logs
        filtered_logs = [
            l for l in self._logs
            if start_time <= l.timestamp < end_time
            and (service is None or l.service == service)
        ]

        # Filter profiles
        filtered_profiles = [
            p for p in self._profiles
            if start_time <= p.timestamp < end_time
            and (service is None or p.service == service)
        ]

        return TelemetryWindow(
            window_id=w_id,
            start_time=start_time,
            end_time=end_time,
            metrics=filtered_metrics,
            spans=filtered_spans,
            logs=filtered_logs,
            profiles=filtered_profiles,
        )

    def get_windows(
        self,
        window_size_sec: Optional[float] = None,
        step_sec: Optional[float] = None,
        service: Optional[str] = None,
    ) -> List[TelemetryWindow]:
        """
        Partition full telemetry stream into sliding time windows.
        """
        w_size = float(window_size_sec if window_size_sec is not None else self.default_window_size)
        s_step = float(step_sec if step_sec is not None else self.default_step_size)

        min_t, max_t = self.get_time_bounds()
        if min_t >= max_t:
            return []

        windows: List[TelemetryWindow] = []
        curr_start = min_t
        while curr_start < max_t:
            curr_end = curr_start + w_size
            win = self.get_window(curr_start, curr_end, service=service)
            windows.append(win)
            curr_start += s_step

        return windows

    def get_trace_correlation(self, trace_id: str) -> Dict[str, Any]:
        """
        Retrieve all spans and correlated log records matching a given trace_id.
        """
        spans = self._trace_spans.get(trace_id, [])
        logs = self._trace_logs.get(trace_id, [])
        return {
            "trace_id": trace_id,
            "spans": spans,
            "logs": logs,
            "num_spans": len(spans),
            "num_logs": len(logs),
            "has_error": any(s.is_error for s in spans) or any(l.is_error for l in logs),
        }

    # --- Multimodal Tensor Batch Compilation ---

    def _encode_metrics(self, window: TelemetryWindow) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Resample and bucket metric values into (metric_steps, num_metrics) tensor.
        """
        T = self.metric_steps
        K = self.num_metrics
        metric_mat = np.zeros((T, K), dtype=np.float32)
        mask = np.zeros((T,), dtype=bool)

        w_dur = max(1e-6, window.duration)
        step_dur = w_dur / T

        if window.metrics:
            counts = np.zeros((T, K), dtype=np.float32)
            for m in window.metrics:
                idx_t = int(min(T - 1, max(0, (m.timestamp - window.start_time) / step_dur)))
                idx_k = self._metric_registry.get(m.name, abs(hash(m.name)) % K)
                metric_mat[idx_t, idx_k] += m.value
                counts[idx_t, idx_k] += 1.0

            # Average buckets with multiple points
            mask_nz = counts > 0
            metric_mat[mask_nz] /= counts[mask_nz]
            mask = np.any(mask_nz, axis=1)

        return torch.from_numpy(metric_mat), torch.from_numpy(mask)

    def _encode_logs(self, window: TelemetryWindow) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode log event IDs into padded sequence (max_logs,).
        """
        L = self.max_logs
        log_ids = np.zeros((L,), dtype=np.int64)
        mask = np.zeros((L,), dtype=bool)

        for i, log in enumerate(window.logs[:L]):
            log_ids[i] = max(1, log.event_id)  # 0 reserved for padding
            mask[i] = True

        return torch.from_numpy(log_ids), torch.from_numpy(mask)

    def _encode_traces(self, window: TelemetryWindow) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode span latencies, errors, and service IDs into (max_spans, trace_feat_dim).
        """
        S = self.max_spans
        D = self.trace_feat_dim
        trace_mat = np.zeros((S, D), dtype=np.float32)
        mask = np.zeros((S,), dtype=bool)

        for i, span in enumerate(window.spans[:S]):
            # Feature 0: Duration (log-scaled)
            trace_mat[i, 0] = math.log1p(max(0.0, span.duration * 1000.0))  # ms scale
            # Feature 1: Error flag
            trace_mat[i, 1] = 1.0 if span.is_error else 0.0
            # Feature 2: Service categorical hash
            svc_hash = (self._service_registry.get(span.service, abs(hash(span.service)))) % 32
            trace_mat[i, 2] = float(svc_hash) / 32.0
            # Feature 3: Has parent
            trace_mat[i, 3] = 1.0 if span.parent_id is not None else 0.0
            # Feature 4: Timestamp relative offset
            offset = (span.start_time - window.start_time) / max(1e-6, window.duration)
            trace_mat[i, 4] = np.clip(offset, 0.0, 1.0)
            mask[i] = True

        return torch.from_numpy(trace_mat), torch.from_numpy(mask)

    def _encode_profiles(self, window: TelemetryWindow) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode flamegraph stacks and weights into (max_profiles, profile_feat_dim).
        """
        P = self.max_profiles
        D = self.profile_feat_dim
        profile_mat = np.zeros((P, D), dtype=np.float32)
        mask = np.zeros((P,), dtype=bool)

        for i, prof in enumerate(window.profiles[:P]):
            # Feature 0: Weight (log-scaled)
            profile_mat[i, 0] = math.log1p(max(0.0, prof.weight))
            # Feature 1: Call stack depth
            profile_mat[i, 1] = float(min(128, len(prof.call_stack))) / 32.0
            # Feature 2: Sample type indicator (cpu=1.0, alloc=2.0, other=0.0)
            st_type = prof.sample_type.lower()
            profile_mat[i, 2] = 1.0 if "cpu" in st_type else (2.0 if "alloc" in st_type else 0.5)
            # Features 3..D: Frame hash embeddings across top and leaf frames
            if prof.call_stack:
                leaf = prof.leaf_frame
                leaf_idx = (self._frame_registry.get(leaf, abs(hash(leaf)))) % max(1, D - 3)
                profile_mat[i, 3 + leaf_idx] += 1.0
            mask[i] = True

        return torch.from_numpy(profile_mat), torch.from_numpy(mask)

    def build_tensor_batch(
        self,
        windows: List[TelemetryWindow],
        device: Optional[Union[str, torch.device]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compile a list of TelemetryWindows into a synchronized PyTorch tensor batch.

        Returns
        -------
        Dict[str, torch.Tensor] containing:
          - 'logs': (B, max_logs) int64
          - 'log_mask': (B, max_logs) bool
          - 'metrics': (B, metric_steps, num_metrics) float32
          - 'metric_mask': (B, metric_steps) bool
          - 'traces': (B, max_spans, trace_feat_dim) float32
          - 'trace_mask': (B, max_spans) bool
          - 'profiles': (B, max_profiles, profile_feat_dim) float32
          - 'profile_mask': (B, max_profiles) bool
        """
        if not windows:
            # Return empty batch with 0 batch size
            batch = {
                "logs": torch.zeros((0, self.max_logs), dtype=torch.int64),
                "log_mask": torch.zeros((0, self.max_logs), dtype=torch.bool),
                "metrics": torch.zeros((0, self.metric_steps, self.num_metrics), dtype=torch.float32),
                "metric_mask": torch.zeros((0, self.metric_steps), dtype=torch.bool),
                "traces": torch.zeros((0, self.max_spans, self.trace_feat_dim), dtype=torch.float32),
                "trace_mask": torch.zeros((0, self.max_spans), dtype=torch.bool),
                "profiles": torch.zeros((0, self.max_profiles, self.profile_feat_dim), dtype=torch.float32),
                "profile_mask": torch.zeros((0, self.max_profiles), dtype=torch.bool),
            }
            if device is not None:
                batch = {k: v.to(device) for k, v in batch.items()}
            return batch

        B = len(windows)
        logs_list, log_mask_list = [], []
        metrics_list, metric_mask_list = [], []
        traces_list, trace_mask_list = [], []
        profiles_list, profile_mask_list = [], []

        for win in windows:
            l, lm = self._encode_logs(win)
            m, mm = self._encode_metrics(win)
            t, tm = self._encode_traces(win)
            p, pm = self._encode_profiles(win)

            logs_list.append(l)
            log_mask_list.append(lm)
            metrics_list.append(m)
            metric_mask_list.append(mm)
            traces_list.append(t)
            trace_mask_list.append(tm)
            profiles_list.append(p)
            profile_mask_list.append(pm)

        batch = {
            "logs": torch.stack(logs_list, dim=0),
            "log_mask": torch.stack(log_mask_list, dim=0),
            "metrics": torch.stack(metrics_list, dim=0),
            "metric_mask": torch.stack(metric_mask_list, dim=0),
            "traces": torch.stack(traces_list, dim=0),
            "trace_mask": torch.stack(trace_mask_list, dim=0),
            "profiles": torch.stack(profiles_list, dim=0),
            "profile_mask": torch.stack(profile_mask_list, dim=0),
        }

        if device is not None:
            batch = {k: v.to(device) for k, v in batch.items()}
        return batch


# ==============================================================================
# 3. Telemetry Fusion Model (PyTorch Multi-Modal Cross-Attention Network)
# ==============================================================================


@dataclass
class TelemetryFusionOutput:
    """
    Output of TelemetryFusionModel forward pass.

    Parameters
    ----------
    z_system : torch.Tensor
        Unified latent system state vector in R^{B x d_model}.
    anomaly_logits : torch.Tensor
        Raw classification logits for system anomaly (B, 1).
    anomaly_probs : torch.Tensor
        Probability of system anomaly in [0, 1] (B, 1).
    modality_weights : torch.Tensor
        Predicted dynamic modality importance weights in R^{B x 4} summing to 1.
        Ordering: [logs, metrics, traces, profiles].
    modal_embeddings : Dict[str, torch.Tensor]
        Pool/summary embedding per modality prior to final cross-attention fusion.
    attention_weights : Optional[torch.Tensor]
        Attention matrix from the system query token to all modal tokens.
    modal_token_slices : Optional[Dict[str, Tuple[int, int]]]
        Token index ranges [start, end) for each modality in the concatenated sequence.
    """
    z_system: torch.Tensor
    anomaly_logits: torch.Tensor
    anomaly_probs: torch.Tensor
    modality_weights: torch.Tensor
    modal_embeddings: Dict[str, torch.Tensor] = field(default_factory=dict)
    attention_weights: Optional[torch.Tensor] = None
    modal_token_slices: Optional[Dict[str, Tuple[int, int]]] = None


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to sequence (B, T, d_model)."""
        T = x.size(1)
        return x + self.pe[:, :T, :]


class TelemetryFusionModel(nn.Module):
    """
    PyTorch multi-modal cross-attention network for full-stack OTel convergence (MELT + Profiling).

    Architecture:
    -------------
    1. Modal Projection & Embeddings:
       - Logs: Vocab embedding + Positional Encoding -> R^{B x T_L x d_model}
       - Metrics: 1D Temporal Convolution / MLP + Positional Encoding -> R^{B x T_M x d_model}
       - Traces: Span Feature MLP + Positional Encoding -> R^{B x T_T x d_model}
       - Profiles: Flamegraph Stack MLP + Positional Encoding -> R^{B x T_P x d_model}
    2. Modality Type Encodings:
       - Distinct learnable embeddings E_{mod} added to modal tokens to preserve origin.
    3. Global System Query Token (q_sys):
       - Learnable query vector prepended to the multimodal sequence:
         Z_in = [q_sys, H_logs, H_metrics, H_traces, H_profiles] in R^{B x (1 + N_total) x d_model}
    4. Multi-Modal Self & Cross Attention:
       - Deep multi-head cross-attention Transformer blocks with key padding masks.
       - q_sys attends globally across all modalities, producing unified system state z_system = Z_out[0].
    5. Output Prediction Heads:
       - Anomaly Classification Head: p(anomaly | z_system).
       - Modality Gating Head: alpha in Delta^3 (weights over Logs, Metrics, Traces, Profiles).
       - Reconstruction / Prediction Heads for regularized self-supervised representations.
    """

    def __init__(
        self,
        d_model: int = 64,
        nhead: int = 4,
        num_fusion_layers: int = 2,
        log_vocab_size: int = 500,
        num_metrics: int = 16,
        trace_feat_dim: int = 8,
        profile_feat_dim: int = 16,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_fusion_layers = num_fusion_layers
        self.log_vocab_size = log_vocab_size
        self.num_metrics = num_metrics
        self.trace_feat_dim = trace_feat_dim
        self.profile_feat_dim = profile_feat_dim

        # 1. Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=max_seq_len)

        # 2. Modality Type Embeddings (0: SYS, 1: LOGS, 2: METRICS, 3: TRACES, 4: PROFILES)
        self.modality_type_embedding = nn.Embedding(5, d_model)

        # 3. Learnable System Query Token q_sys
        self.q_sys = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # 4. Modal Encoders
        # Logs
        self.log_embedding = nn.Embedding(log_vocab_size, d_model, padding_idx=0)
        self.log_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )

        # Metrics: 1D projection + residual conv
        self.metric_proj = nn.Sequential(
            nn.Linear(num_metrics, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )

        # Traces
        self.trace_proj = nn.Sequential(
            nn.Linear(trace_feat_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )

        # Profiles
        self.profile_proj = nn.Sequential(
            nn.Linear(profile_feat_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )

        # 5. Multi-Modal Transformer Encoder Layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_fusion_layers,
            enable_nested_tensor=False,
        )

        # 6. Prediction Heads
        self.anomaly_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

        self.modality_gate_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 4),  # logs, metrics, traces, profiles
        )

        # 7. Reconstruction Heads (Self-supervision)
        self.metric_recon = nn.Linear(d_model, num_metrics)
        self.log_recon = nn.Linear(d_model, log_vocab_size)

    def forward(
        self,
        batch: Optional[Dict[str, torch.Tensor]] = None,
        logs: Optional[torch.Tensor] = None,
        log_mask: Optional[torch.Tensor] = None,
        metrics: Optional[torch.Tensor] = None,
        metric_mask: Optional[torch.Tensor] = None,
        traces: Optional[torch.Tensor] = None,
        trace_mask: Optional[torch.Tensor] = None,
        profiles: Optional[torch.Tensor] = None,
        profile_mask: Optional[torch.Tensor] = None,
        return_attention: bool = True,
    ) -> TelemetryFusionOutput:
        """
        Forward pass fusing all 4 modalities into unified system state z_system.
        """
        if batch is not None:
            logs = batch.get("logs") if logs is None else logs
            log_mask = batch.get("log_mask") if log_mask is None else log_mask
            metrics = batch.get("metrics") if metrics is None else metrics
            metric_mask = batch.get("metric_mask") if metric_mask is None else metric_mask
            traces = batch.get("traces") if traces is None else traces
            trace_mask = batch.get("trace_mask") if trace_mask is None else trace_mask
            profiles = batch.get("profiles") if profiles is None else profiles
            profile_mask = batch.get("profile_mask") if profile_mask is None else profile_mask

        # Determine batch size and target device
        candidates = [t for t in (logs, metrics, traces, profiles) if t is not None]
        if not candidates:
            raise ValueError("TelemetryFusionModel received no modal input tensors.")
        B = candidates[0].size(0)
        device = candidates[0].device

        # 1. Process System Query Token
        # Shape: (B, 1, d_model)
        sys_tokens = self.q_sys.expand(B, -1, -1) + self.modality_type_embedding(
            torch.zeros(1, dtype=torch.long, device=device)
        )
        sys_mask = torch.ones((B, 1), dtype=torch.bool, device=device)

        seq_tokens = [sys_tokens]
        seq_masks = [sys_mask]
        modal_slices: Dict[str, Tuple[int, int]] = {}
        modal_embeddings: Dict[str, torch.Tensor] = {}
        current_idx = 1

        # 2. Process Logs
        if logs is not None:
            # logs: (B, T_L)
            T_L = logs.size(1)
            # Clip indices to vocab bounds
            safe_logs = torch.clamp(logs, min=0, max=self.log_vocab_size - 1)
            h_logs = self.log_embedding(safe_logs)
            h_logs = self.pos_encoder(self.log_proj(h_logs))
            h_logs = h_logs + self.modality_type_embedding(torch.tensor(1, device=device))
            seq_tokens.append(h_logs)

            if log_mask is not None:
                seq_masks.append(log_mask.to(device))
            else:
                seq_masks.append(torch.ones((B, T_L), dtype=torch.bool, device=device))

            modal_slices["logs"] = (current_idx, current_idx + T_L)
            # Mean pool for modal embedding
            modal_embeddings["logs"] = h_logs.mean(dim=1)
            current_idx += T_L

        # 3. Process Metrics
        if metrics is not None:
            # metrics: (B, T_M, K)
            T_M = metrics.size(1)
            h_metrics = self.metric_proj(metrics)
            h_metrics = self.pos_encoder(h_metrics)
            h_metrics = h_metrics + self.modality_type_embedding(torch.tensor(2, device=device))
            seq_tokens.append(h_metrics)

            if metric_mask is not None:
                seq_masks.append(metric_mask.to(device))
            else:
                seq_masks.append(torch.ones((B, T_M), dtype=torch.bool, device=device))

            modal_slices["metrics"] = (current_idx, current_idx + T_M)
            modal_embeddings["metrics"] = h_metrics.mean(dim=1)
            current_idx += T_M

        # 4. Process Traces
        if traces is not None:
            # traces: (B, T_T, D_T)
            T_T = traces.size(1)
            h_traces = self.trace_proj(traces)
            h_traces = self.pos_encoder(h_traces)
            h_traces = h_traces + self.modality_type_embedding(torch.tensor(3, device=device))
            seq_tokens.append(h_traces)

            if trace_mask is not None:
                seq_masks.append(trace_mask.to(device))
            else:
                seq_masks.append(torch.ones((B, T_T), dtype=torch.bool, device=device))

            modal_slices["traces"] = (current_idx, current_idx + T_T)
            modal_embeddings["traces"] = h_traces.mean(dim=1)
            current_idx += T_T

        # 5. Process Profiles
        if profiles is not None:
            # profiles: (B, T_P, D_P)
            T_P = profiles.size(1)
            h_profiles = self.profile_proj(profiles)
            h_profiles = self.pos_encoder(h_profiles)
            h_profiles = h_profiles + self.modality_type_embedding(torch.tensor(4, device=device))
            seq_tokens.append(h_profiles)

            if profile_mask is not None:
                seq_masks.append(profile_mask.to(device))
            else:
                seq_masks.append(torch.ones((B, T_P), dtype=torch.bool, device=device))

            modal_slices["profiles"] = (current_idx, current_idx + T_P)
            modal_embeddings["profiles"] = h_profiles.mean(dim=1)
            current_idx += T_P

        # 6. Concatenate All Modalities
        # full_tokens: (B, Total_Tokens, d_model)
        full_tokens = torch.cat(seq_tokens, dim=1)
        full_mask = torch.cat(seq_masks, dim=1)  # (B, Total_Tokens) True for valid, False for pad

        # PyTorch MultiheadAttention uses key_padding_mask where True indicates PADDED (to ignore)
        key_padding_mask = ~full_mask
        # Guarantee at least the sys token (col 0) is never masked
        key_padding_mask[:, 0] = False

        # 7. Multimodal Transformer Encoder
        # Transformer forward pass
        fused_tokens = self.transformer_encoder(
            src=full_tokens,
            src_key_padding_mask=key_padding_mask,
        )

        # Extract unified system state z_system from position 0 (q_sys)
        z_system = fused_tokens[:, 0, :]  # (B, d_model)

        # 8. Compute cross-modal attention from system token to all tokens
        attention_weights: Optional[torch.Tensor] = None
        if return_attention:
            # Approximate cross-modal attention via query-key dot product:
            # sys_token (query) vs fused_tokens (keys)
            d_k = math.sqrt(self.d_model)
            q = z_system.unsqueeze(1)  # (B, 1, d_model)
            k = fused_tokens           # (B, Total_Tokens, d_model)
            scores = torch.bmm(q, k.transpose(1, 2)) / d_k  # (B, 1, Total_Tokens)
            # Mask out padded positions
            scores = scores.masked_fill(key_padding_mask.unsqueeze(1), -1e9)
            attention_weights = F.softmax(scores, dim=-1).squeeze(1)  # (B, Total_Tokens)

        # 9. Prediction Heads
        anomaly_logits = self.anomaly_head(z_system)  # (B, 1)
        anomaly_probs = torch.sigmoid(anomaly_logits)  # (B, 1)

        mod_logits = self.modality_gate_head(z_system)  # (B, 4)
        modality_weights = F.softmax(mod_logits, dim=-1)  # (B, 4)

        return TelemetryFusionOutput(
            z_system=z_system,
            anomaly_logits=anomaly_logits,
            anomaly_probs=anomaly_probs,
            modality_weights=modality_weights,
            modal_embeddings=modal_embeddings,
            attention_weights=attention_weights,
            modal_token_slices=modal_slices,
        )


# ==============================================================================
# 4. Cross-Modal Attributor (MELT + Continuous Profiling Attribution)
# ==============================================================================


class AnomalyDriver(str, enum.Enum):
    """Categorical classification of root cause modality."""
    LOG_ERRORS = "LOG_ERRORS"
    METRIC_SATURATION = "METRIC_SATURATION"
    TRACE_BOTTLENECK = "TRACE_BOTTLENECK"
    PROFILE_HOTSPOT = "PROFILE_HOTSPOT"
    NORMAL = "NORMAL_STEADY_STATE"


@dataclass
class CrossModalAttributionResult:
    """
    Diagnostic root-cause attribution result across MELT + Continuous Profiling.

    Attributes
    ----------
    primary_driver : str
        The primary operational modality triggering the alert
        ('LOG_ERRORS', 'METRIC_SATURATION', 'TRACE_BOTTLENECK', 'PROFILE_HOTSPOT').
    modal_contributions : Dict[str, float]
        Normalized attribution scores in [0.0, 1.0] summing strictly to 1.0.
    modal_percentages : Dict[str, float]
        Attribution scores expressed as percentages in [0.0, 100.0] summing to 100.0%.
    anomaly_score : float
        Scalar anomaly confidence score in [0.0, 1.0].
    is_anomalous : bool
        Whether anomaly_score exceeds detection threshold.
    method : str
        Attribution algorithm utilized ('attention', 'gradient', 'integrated_gradients', 'ablation', 'combined').
    evidence : Dict[str, Any]
        Concrete evidence tokens, span IDs, saturated metrics, or hotspot frames.
    metadata : Dict[str, Any]
        Additional execution context.
    """
    primary_driver: str
    modal_contributions: Dict[str, float]
    modal_percentages: Dict[str, float]
    anomaly_score: float
    is_anomalous: bool
    method: str
    evidence: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert attribution result to dictionary."""
        return asdict(self)

    def summary(self, ascii_bars: bool = True, max_bar_width: int = 25) -> str:
        """
        Render a human-readable diagnostic SRE report with ASCII contribution bars.
        """
        lines = [
            "=======================================================================",
            " DeepLog-Omni Multi-Modal OTel Attribution (MELT + Profiling)",
            "=======================================================================",
            f" Status        : {'🚨 ANOMALOUS' if self.is_anomalous else '✅ NORMAL'}",
            f" Anomaly Score : {self.anomaly_score:.4f}",
            f" Primary Driver: {self.primary_driver}",
            f" Method        : {self.method}",
            "-----------------------------------------------------------------------",
            " Modality Attribution Breakdown (MELT + Profiling):",
        ]

        # Order: logs, metrics, traces, profiles
        keys = ["logs", "metrics", "traces", "profiles"]
        labels = {
            "logs": "Events / Logs      ",
            "metrics": "Metrics Saturation ",
            "traces": "Trace Latencies    ",
            "profiles": "Profiling Hotspots ",
        }

        for k in keys:
            pct = self.modal_percentages.get(k, 0.0)
            if ascii_bars:
                filled = int(round((pct / 100.0) * max_bar_width))
                bar = "█" * filled + "░" * (max_bar_width - filled)
                lines.append(f"  [{k.upper():<8}] {labels[k]} : {pct:6.2f}% |{bar}|")
            else:
                lines.append(f"  [{k.upper():<8}] {labels[k]} : {pct:6.2f}%")

        lines.append("-----------------------------------------------------------------------")
        lines.append(" Diagnostic Evidence:")
        ev = self.evidence
        if "top_log_templates" in ev and ev["top_log_templates"]:
            lines.append(f"  * Logs     : {ev['top_log_templates']}")
        if "saturated_metrics" in ev and ev["saturated_metrics"]:
            lines.append(f"  * Metrics  : {ev['saturated_metrics']}")
        if "bottleneck_spans" in ev and ev["bottleneck_spans"]:
            lines.append(f"  * Traces   : {ev['bottleneck_spans']}")
        if "hotspot_stacks" in ev and ev["hotspot_stacks"]:
            lines.append(f"  * Profiling: {ev['hotspot_stacks']}")

        lines.append("=======================================================================")
        return "\n".join(lines)


class CrossModalAttributor:
    """
    Multi-modal root cause attributor across MELT + Continuous Profiling telemetry.

    Isolates whether an anomaly is primarily driven by:
    - Log errors (L)
    - Metric saturation spikes (M)
    - Trace network bottlenecks (T)
    - Code profile hotspots (P)

    Algorithms supported:
    - 'attention': Cross-modal attention rollout from system query token.
    - 'gradient': Saliency Gradient x Input backpropagated to each modality.
    - 'integrated_gradients': Path integral from baseline to input for each modality.
    - 'ablation': Counterfactual zero-out perturbation.
    - 'combined': Multi-objective ensemble blending attention, saliency, and gating weights.
    """

    def __init__(
        self,
        model: TelemetryFusionModel,
        anomaly_threshold: float = 0.5,
        default_method: str = "combined",
        ig_steps: int = 10,
    ):
        self.model = model
        self.anomaly_threshold = float(anomaly_threshold)
        self.default_method = str(default_method).lower()
        self.ig_steps = int(ig_steps)

    def _normalize_scores(
        self,
        raw_scores: Dict[str, float],
        eps: float = 1e-8,
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Normalize raw scores across modalities to sum to 1.0 (contributions) and 100%."""
        total = sum(max(0.0, s) for s in raw_scores.values())
        if total < eps:
            uniform = 1.0 / max(1, len(raw_scores))
            contributions = {k: uniform for k in raw_scores}
            percentages = {k: uniform * 100.0 for k in raw_scores}
            return contributions, percentages

        contributions = {k: float(max(0.0, s) / total) for k, s in raw_scores.items()}
        # Enforce exact sum to 1.0
        sum_c = sum(contributions.values())
        if abs(sum_c - 1.0) > 1e-6:
            contributions = {k: v / sum_c for k, v in contributions.items()}

        percentages = {k: round(v * 100.0, 4) for k, v in contributions.items()}
        return contributions, percentages

    def _determine_primary_driver(
        self,
        contributions: Dict[str, float],
        is_anomalous: bool,
    ) -> str:
        """Map highest scoring modality to AnomalyDriver classification."""
        if not is_anomalous:
            return AnomalyDriver.NORMAL.value

        mapping = {
            "logs": AnomalyDriver.LOG_ERRORS.value,
            "metrics": AnomalyDriver.METRIC_SATURATION.value,
            "traces": AnomalyDriver.TRACE_BOTTLENECK.value,
            "profiles": AnomalyDriver.PROFILE_HOTSPOT.value,
        }
        best_modality = max(contributions.items(), key=lambda kv: kv[1])[0]
        return mapping.get(best_modality, AnomalyDriver.NORMAL.value)

    # --- Attribution Strategies ---

    def _attribute_attention(
        self,
        out: TelemetryFusionOutput,
    ) -> Dict[str, float]:
        """
        Attention-based attribution: sum attention mass from system token to each modality.
        """
        raw_scores = {"logs": 0.0, "metrics": 0.0, "traces": 0.0, "profiles": 0.0}
        if out.attention_weights is None or out.modal_token_slices is None:
            return raw_scores

        attn = out.attention_weights[0]  # Take first sample in batch: (Total_Tokens,)
        for mod, (st, end) in out.modal_token_slices.items():
            if mod in raw_scores and end > st:
                raw_scores[mod] = float(attn[st:end].sum().item())

        return raw_scores

    def _attribute_gradient(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """
        Gradient x Input saliency attribution: computes norm of grad(logits) * input.
        """
        raw_scores = {"logs": 0.0, "metrics": 0.0, "traces": 0.0, "profiles": 0.0}

        metrics = batch.get("metrics")
        traces = batch.get("traces")
        profiles = batch.get("profiles")
        logs = batch.get("logs")

        # For continuous tensors, enable requires_grad
        tensors_to_grad: Dict[str, torch.Tensor] = {}
        if metrics is not None and metrics.numel() > 0:
            m_in = metrics.clone().detach().requires_grad_(True)
            tensors_to_grad["metrics"] = m_in
        if traces is not None and traces.numel() > 0:
            t_in = traces.clone().detach().requires_grad_(True)
            tensors_to_grad["traces"] = t_in
        if profiles is not None and profiles.numel() > 0:
            p_in = profiles.clone().detach().requires_grad_(True)
            tensors_to_grad["profiles"] = p_in

        # Forward pass
        out = self.model(
            logs=logs,
            log_mask=batch.get("log_mask"),
            metrics=tensors_to_grad.get("metrics"),
            metric_mask=batch.get("metric_mask"),
            traces=tensors_to_grad.get("traces"),
            trace_mask=batch.get("trace_mask"),
            profiles=tensors_to_grad.get("profiles"),
            profile_mask=batch.get("profile_mask"),
            return_attention=False,
        )

        logit = out.anomaly_logits[0, 0]

        # Compute gradients
        grads = torch.autograd.grad(
            logit,
            list(tensors_to_grad.values()),
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )

        for (mod, tensor), grad in zip(tensors_to_grad.items(), grads):
            if grad is not None:
                # Saliency = sum(|grad * tensor|)
                saliency = torch.sum(torch.abs(grad * tensor)).item()
                raw_scores[mod] = float(saliency)

        # For logs (discrete tokens), approximate using model's learned gating or attention
        raw_scores["logs"] = float(out.modality_weights[0, 0].item())

        return raw_scores

    def _attribute_integrated_gradients(
        self,
        batch: Dict[str, torch.Tensor],
        steps: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Integrated Gradients (IG) path integral from zeros baseline.
        """
        k_steps = steps if steps is not None else self.ig_steps
        raw_scores = {"logs": 0.0, "metrics": 0.0, "traces": 0.0, "profiles": 0.0}

        metrics = batch.get("metrics")
        traces = batch.get("traces")
        profiles = batch.get("profiles")
        logs = batch.get("logs")

        # Accumulators
        grad_acc: Dict[str, float] = {"metrics": 0.0, "traces": 0.0, "profiles": 0.0}

        for step in range(1, k_steps + 1):
            alpha = float(step) / k_steps
            tensors_step: Dict[str, torch.Tensor] = {}

            if metrics is not None and metrics.numel() > 0:
                tensors_step["metrics"] = (alpha * metrics).clone().detach().requires_grad_(True)
            if traces is not None and traces.numel() > 0:
                tensors_step["traces"] = (alpha * traces).clone().detach().requires_grad_(True)
            if profiles is not None and profiles.numel() > 0:
                tensors_step["profiles"] = (alpha * profiles).clone().detach().requires_grad_(True)

            out = self.model(
                logs=logs,
                log_mask=batch.get("log_mask"),
                metrics=tensors_step.get("metrics"),
                metric_mask=batch.get("metric_mask"),
                traces=tensors_step.get("traces"),
                trace_mask=batch.get("trace_mask"),
                profiles=tensors_step.get("profiles"),
                profile_mask=batch.get("profile_mask"),
                return_attention=False,
            )

            logit = out.anomaly_logits[0, 0]
            grads = torch.autograd.grad(
                logit,
                list(tensors_step.values()),
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )

            for (mod, _), g in zip(tensors_step.items(), grads):
                if g is not None:
                    grad_acc[mod] += float(torch.sum(torch.abs(g)).item())

        # Average gradients * input magnitude
        if metrics is not None:
            raw_scores["metrics"] = (grad_acc["metrics"] / k_steps) * float(torch.norm(metrics).item())
        if traces is not None:
            raw_scores["traces"] = (grad_acc["traces"] / k_steps) * float(torch.norm(traces).item())
        if profiles is not None:
            raw_scores["profiles"] = (grad_acc["profiles"] / k_steps) * float(torch.norm(profiles).item())

        # Forward full to get logs gating
        with torch.no_grad():
            full_out = self.model(batch=batch, return_attention=False)
            raw_scores["logs"] = float(full_out.modality_weights[0, 0].item())

        return raw_scores

    def _attribute_ablation(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """
        Perturbation / Ablation attribution: measure anomaly score drop when zeroing modality.
        """
        raw_scores = {"logs": 0.0, "metrics": 0.0, "traces": 0.0, "profiles": 0.0}

        with torch.no_grad():
            full_out = self.model(batch=batch, return_attention=False)
            p_full = float(full_out.anomaly_probs[0, 0].item())

            # 1. Ablate logs
            batch_no_logs = dict(batch)
            if "logs" in batch_no_logs:
                batch_no_logs["logs"] = torch.zeros_like(batch_no_logs["logs"])
            p_no_logs = float(self.model(batch=batch_no_logs, return_attention=False).anomaly_probs[0, 0].item())
            raw_scores["logs"] = max(0.0, p_full - p_no_logs)

            # 2. Ablate metrics
            batch_no_metrics = dict(batch)
            if "metrics" in batch_no_metrics:
                batch_no_metrics["metrics"] = torch.zeros_like(batch_no_metrics["metrics"])
            p_no_metrics = float(self.model(batch=batch_no_metrics, return_attention=False).anomaly_probs[0, 0].item())
            raw_scores["metrics"] = max(0.0, p_full - p_no_metrics)

            # 3. Ablate traces
            batch_no_traces = dict(batch)
            if "traces" in batch_no_traces:
                batch_no_traces["traces"] = torch.zeros_like(batch_no_traces["traces"])
            p_no_traces = float(self.model(batch=batch_no_traces, return_attention=False).anomaly_probs[0, 0].item())
            raw_scores["traces"] = max(0.0, p_full - p_no_traces)

            # 4. Ablate profiles
            batch_no_profiles = dict(batch)
            if "profiles" in batch_no_profiles:
                batch_no_profiles["profiles"] = torch.zeros_like(batch_no_profiles["profiles"])
            p_no_profiles = float(self.model(batch=batch_no_profiles, return_attention=False).anomaly_probs[0, 0].item())
            raw_scores["profiles"] = max(0.0, p_full - p_no_profiles)

        return raw_scores

    # --- Evidence Extraction ---

    def _extract_evidence(
        self,
        window: Optional[TelemetryWindow],
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, Any]:
        """
        Extract fine-grained evidence items from the telemetry window or tensor batch.
        """
        evidence: Dict[str, Any] = {
            "top_log_templates": [],
            "saturated_metrics": [],
            "bottleneck_spans": [],
            "hotspot_stacks": [],
        }

        if window is not None:
            # 1. Error logs
            err_logs = [l for l in window.logs if l.is_error]
            if err_logs:
                evidence["top_log_templates"] = [
                    f"[{l.level}] template_id={l.event_id}: {l.template}"
                    for l in err_logs[:3]
                ]
            elif window.logs:
                evidence["top_log_templates"] = [
                    f"template_id={l.event_id}: {l.template}"
                    for l in window.logs[:2]
                ]

            # 2. Saturated metrics
            if window.metrics:
                sorted_metrics = sorted(window.metrics, key=lambda m: abs(m.value), reverse=True)
                evidence["saturated_metrics"] = [
                    f"{m.name}={m.value:.2f} ({m.metric_type})"
                    for m in sorted_metrics[:3]
                ]

            # 3. Bottleneck spans
            if window.spans:
                sorted_spans = sorted(window.spans, key=lambda s: (s.is_error, s.duration), reverse=True)
                evidence["bottleneck_spans"] = [
                    f"span_id={s.span_id} ({s.service}:{s.name}) dur={s.duration*1000.0:.1f}ms status={s.status_code}"
                    for s in sorted_spans[:3]
                ]

            # 4. Hotspot stacks
            if window.profiles:
                sorted_profiles = sorted(window.profiles, key=lambda p: p.weight, reverse=True)
                evidence["hotspot_stacks"] = [
                    f"leaf='{p.leaf_frame}' weight={p.weight:.1f} type={p.sample_type}"
                    for p in sorted_profiles[:3]
                ]
        else:
            # Fallback from tensors
            if "metrics" in batch and batch["metrics"].numel() > 0:
                m_vals = batch["metrics"][0].mean(dim=0).cpu().numpy()
                top_k = np.argsort(m_vals)[-3:][::-1]
                evidence["saturated_metrics"] = [f"metric_idx_{idx}={m_vals[idx]:.2f}" for idx in top_k]

            if "traces" in batch and batch["traces"].numel() > 0:
                t_dur = batch["traces"][0, :, 0].cpu().numpy()
                top_s = np.argsort(t_dur)[-3:][::-1]
                evidence["bottleneck_spans"] = [f"span_slot_{idx}_dur_idx={t_dur[idx]:.2f}" for idx in top_s]

        return evidence

    # --- Public Main Attribution API ---

    def attribute(
        self,
        batch: Dict[str, torch.Tensor],
        method: Optional[str] = None,
        window: Optional[TelemetryWindow] = None,
    ) -> CrossModalAttributionResult:
        """
        Execute cross-modal attribution on an aligned telemetry tensor batch.

        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            Multimodal batch dictionary prepared by UnifiedTelemetryStream.
        method : Optional[str]
            Attribution method ('attention', 'gradient', 'integrated_gradients', 'ablation', 'combined').
        window : Optional[TelemetryWindow], optional
            Source TelemetryWindow for fine-grained domain evidence extraction.

        Returns
        -------
        CrossModalAttributionResult
            Structured attribution report with primary driver and percentage contributions.
        """
        attr_method = (method or self.default_method).lower()

        # Execute forward pass to obtain anomaly score and gating weights
        self.model.eval()
        with torch.no_grad():
            out = self.model(batch=batch, return_attention=True)
            anomaly_score = float(out.anomaly_probs[0, 0].item())
            is_anomalous = anomaly_score >= self.anomaly_threshold

        # Compute raw scores based on requested method
        if attr_method == "attention":
            raw_scores = self._attribute_attention(out)
        elif attr_method == "gradient":
            raw_scores = self._attribute_gradient(batch)
        elif attr_method == "integrated_gradients":
            raw_scores = self._attribute_integrated_gradients(batch)
        elif attr_method == "ablation":
            raw_scores = self._attribute_ablation(batch)
        elif attr_method == "combined":
            attn_scores = self._attribute_attention(out)
            ablation_scores = self._attribute_ablation(batch)
            gate_scores = {
                "logs": float(out.modality_weights[0, 0].item()),
                "metrics": float(out.modality_weights[0, 1].item()),
                "traces": float(out.modality_weights[0, 2].item()),
                "profiles": float(out.modality_weights[0, 3].item()),
            }
            # Weighted ensemble: 0.4 Attention + 0.3 Ablation + 0.3 Gating
            raw_scores = {}
            for k in ["logs", "metrics", "traces", "profiles"]:
                raw_scores[k] = (
                    0.4 * attn_scores.get(k, 0.0)
                    + 0.3 * ablation_scores.get(k, 0.0)
                    + 0.3 * gate_scores.get(k, 0.0)
                )
        else:
            raise ValueError(
                f"Unknown attribution method '{attr_method}'. "
                f"Must be one of: 'attention', 'gradient', 'integrated_gradients', 'ablation', 'combined'."
            )

        # Normalize contributions to sum strictly to 1.0 (and 100%)
        contributions, percentages = self._normalize_scores(raw_scores)

        # Determine primary operational driver
        primary_driver = self._determine_primary_driver(contributions, is_anomalous)

        # Extract diagnostic evidence
        evidence = self._extract_evidence(window, batch)

        return CrossModalAttributionResult(
            primary_driver=primary_driver,
            modal_contributions=contributions,
            modal_percentages=percentages,
            anomaly_score=anomaly_score,
            is_anomalous=is_anomalous,
            method=attr_method,
            evidence=evidence,
            metadata={"batch_size": batch["logs"].size(0) if "logs" in batch else 1},
        )

    def attribute_window(
        self,
        window: TelemetryWindow,
        stream: UnifiedTelemetryStream,
        method: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> CrossModalAttributionResult:
        """
        End-to-end attribution directly from a TelemetryWindow instance.
        """
        batch = stream.build_tensor_batch([window], device=device)
        return self.attribute(batch, method=method, window=window)
