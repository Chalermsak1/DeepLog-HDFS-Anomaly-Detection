"""
DeepLog Real-Time Streaming Log Aggregator & Pipeline Simulator
==============================================================
Implements real-time sessionization and sliding context aggregation per block_id,
rolling DeepLog inference (Native or TorchScript JIT), anomaly alerting, and
session lifecycle timeout management.
"""

import time
from collections import deque, OrderedDict
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import numpy as np


@dataclass
class EventAlert:
    """Represents an anomaly detection alert raised for an anomalous log event."""
    block_id: str
    timestamp: float
    step_number: int
    raw_event: int
    canonical_event: int
    topk_predictions: List[int]
    topk_raw_predictions: List[int]
    is_unseen: bool
    confidence: float
    latency_ms: float
    context: List[int]
    context_raw: List[int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "block_id": self.block_id,
            "timestamp": self.timestamp,
            "step_number": self.step_number,
            "raw_event": self.raw_event,
            "canonical_event": self.canonical_event,
            "topk_predictions": self.topk_predictions,
            "topk_raw_predictions": self.topk_raw_predictions,
            "is_unseen": self.is_unseen,
            "confidence": self.confidence,
            "latency_ms": self.latency_ms,
            "context": self.context,
            "context_raw": self.context_raw,
        }


@dataclass
class EventResult:
    """Result of ingesting and diagnosing a single streaming log event."""
    block_id: str
    timestamp: float
    step_number: int
    raw_event: int
    canonical_event: int
    is_anomaly: bool
    topk_predictions: List[int]
    topk_raw_predictions: List[int]
    confidence: float
    latency_ms: float
    queue_latency_ms: float
    alert: Optional[EventAlert] = None


@dataclass
class SessionSummary:
    """Summary of a completed or evicted log session / block."""
    block_id: str
    total_events: int
    total_anomalies: int
    is_anomaly_block: bool
    start_time: float
    last_seen_time: float
    duration_seconds: float
    first_anomaly_step: Optional[int] = None
    first_anomaly_event: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "block_id": self.block_id,
            "total_events": self.total_events,
            "total_anomalies": self.total_anomalies,
            "is_anomaly_block": self.is_anomaly_block,
            "start_time": self.start_time,
            "last_seen_time": self.last_seen_time,
            "duration_seconds": self.duration_seconds,
            "first_anomaly_step": self.first_anomaly_step,
            "first_anomaly_event": self.first_anomaly_event,
        }


class SessionStreamAggregator:
    """
    SessionStreamAggregator: Real-time sliding context buffer aggregator.
    
    Mimics distributed stream ingestion (e.g. Apache Flink / Kafka block sessionization)
    for HDFS log events. Manages sliding context windows per block_id, performs
    rolling DeepLog inference, and handles session timeout lifecycle.
    """

    def __init__(
        self,
        model: Optional[Union[nn.Module, torch.jit.ScriptModule]] = None,
        context_length: int = 10,
        top_k: int = 5,
        mapping: Optional[Dict[int, int]] = None,
        id_to_canonical: Optional[Dict[int, int]] = None,
        canonical_to_event: Optional[Dict[int, int]] = None,
        no_event_canonical: Optional[int] = None,
        device: Optional[torch.device] = None,
        session_timeout: float = 60.0,
        max_active_sessions: Optional[int] = None,
        vocab_size: Optional[int] = None,
    ):
        """
        Initialize the SessionStreamAggregator.

        Parameters
        ----------
        model : nn.Module or torch.jit.ScriptModule, optional
            Trained or JIT-traced DeepLog model for rolling inference.
        context_length : int, default=10
            Context window length (h).
        top_k : int, default=5
            Top-K candidate threshold (g).
        mapping : dict, optional
            Mapping dictionary {canonical_id: raw_event_id} from Preprocessor.
        id_to_canonical : dict, optional
            Dictionary mapping raw event ID -> canonical index.
        canonical_to_event : dict, optional
            Dictionary mapping canonical index -> raw event ID.
        no_event_canonical : int, optional
            Canonical index for NO_EVENT padding.
        device : torch.device, optional
            Target device for model inference.
        session_timeout : float, default=60.0
            Inactivity threshold (seconds or timestamp units) for session cleanup.
        max_active_sessions : int, optional
            Maximum concurrent active sessions before LRU eviction.
        vocab_size : int, optional
            Vocabulary size of the model. Defaults to model.input_size or 30.
        """
        self.model = model
        self.context_length = context_length
        self.top_k = top_k
        self.session_timeout = session_timeout
        self.max_active_sessions = max_active_sessions

        # Resolve vocabulary mappings
        if mapping is not None:
            self.canonical_to_event = {int(k): int(v) for k, v in mapping.items()}
            self.id_to_canonical = {int(v): int(k) for k, v in mapping.items()}
        else:
            self.id_to_canonical = {int(k): int(v) for k, v in (id_to_canonical or {}).items()}
            self.canonical_to_event = {int(k): int(v) for k, v in (canonical_to_event or {}).items()}

        # Resolve NO_EVENT canonical index
        if no_event_canonical is not None:
            self.no_event_canonical = int(no_event_canonical)
        elif -1337 in self.id_to_canonical:
            self.no_event_canonical = self.id_to_canonical[-1337]
        elif len(self.id_to_canonical) > 0:
            self.no_event_canonical = max(self.id_to_canonical.values())
        else:
            self.no_event_canonical = 0

        # Resolve vocabulary size
        if vocab_size is not None:
            self.vocab_size = vocab_size
        elif self.model is not None and hasattr(self.model, "input_size"):
            self.vocab_size = self.model.input_size
        elif len(self.canonical_to_event) > 0:
            self.vocab_size = max(self.canonical_to_event.keys()) + 1
        else:
            self.vocab_size = 30

        # Resolve device
        if device is not None:
            self.device = device
        elif self.model is not None:
            try:
                self.device = next(self.model.parameters()).device
            except (StopIteration, AttributeError):
                self.device = torch.device("cpu")
        else:
            self.device = torch.device("cpu")

        if self.model is not None:
            self.model.eval()

        # Session state store: block_id -> session dict (OrderedDict for LRU)
        self.sessions: OrderedDict[str, Dict[str, Any]] = OrderedDict()

        # Telemetry & metrics
        self.total_events_processed: int = 0
        self.total_anomalies_detected: int = 0
        self.total_sessions_created: int = 0
        self.total_sessions_completed: int = 0
        self.total_inference_time_ms: float = 0.0
        self.total_queue_time_ms: float = 0.0
        self.start_wall_time: float = time.time()
        self.alerts: List[EventAlert] = []

    def get_context(self, block_id: str) -> List[int]:
        """
        Get the current context buffer for a block_id padded with NO_EVENT.

        Returns
        -------
        context : list of int of length `context_length`
            Canonical event indices.
        """
        if block_id not in self.sessions:
            return [self.no_event_canonical] * self.context_length

        ctx_deque = self.sessions[block_id]["context"]
        curr = list(ctx_deque)
        if len(curr) < self.context_length:
            padding = [self.no_event_canonical] * (self.context_length - len(curr))
            return padding + curr
        return curr[-self.context_length:]

    def get_raw_context(self, block_id: str) -> List[int]:
        """Get the current context buffer mapped back to raw event IDs."""
        canonical_ctx = self.get_context(block_id)
        return [self.canonical_to_event.get(c, c) for c in canonical_ctx]

    def process_event(
        self,
        block_id: str,
        raw_event: int,
        timestamp: Optional[float] = None,
        arrival_time: Optional[float] = None,
    ) -> EventResult:
        """
        Ingest a streaming log event, perform rolling DeepLog inference, and update session context.

        Parameters
        ----------
        block_id : str
            Identifier of the HDFS block / machine session.
        raw_event : int
            Raw log event ID (e.g. 5, 22, 11).
        timestamp : float, optional
            Event occurrence timestamp. Defaults to current wall clock.
        arrival_time : float, optional
            System ingestion timestamp for measuring queue latency.

        Returns
        -------
        result : EventResult
            Inspection result with anomaly status, top-k candidates, and latencies.
        """
        curr_wall = time.time()
        event_time = timestamp if timestamp is not None else curr_wall
        queue_latency_ms = (time.perf_counter() - arrival_time) * 1000.0 if arrival_time is not None else 0.0

        # Retrieve or initialize session state
        if block_id in self.sessions:
            # Move to end for LRU tracking
            self.sessions.move_to_end(block_id)
            session = self.sessions[block_id]
        else:
            # Check capacity limit and evict LRU session if necessary
            if self.max_active_sessions is not None and len(self.sessions) >= self.max_active_sessions:
                oldest_block_id, _ = self.sessions.popitem(last=False)
                self.total_sessions_completed += 1

            session = {
                "block_id": block_id,
                "context": deque(maxlen=self.context_length),
                "event_count": 0,
                "anomaly_count": 0,
                "is_anomaly": False,
                "start_time": event_time,
                "last_seen_time": event_time,
                "first_anomaly_step": None,
                "first_anomaly_event": None,
                "total_inference_ms": 0.0,
            }
            self.sessions[block_id] = session
            self.total_sessions_created += 1

        step_num = session["event_count"] + 1

        # Current context before ingesting this event
        current_canonical_ctx = self.get_context(block_id)

        # Map raw event to canonical index
        raw_int = int(raw_event)
        if len(self.id_to_canonical) > 0:
            canonical_event = self.id_to_canonical.get(raw_int, -1)
        else:
            canonical_event = raw_int

        is_unseen = (canonical_event < 0 or canonical_event >= self.vocab_size)

        # Rolling inference
        inference_latency_ms = 0.0
        topk_canonical: List[int] = []
        topk_raw: List[int] = []
        confidence: float = 0.0
        is_anomaly = False

        if self.model is not None:
            ctx_tensor = torch.tensor([current_canonical_ctx], dtype=torch.long, device=self.device)

            # Accurate device-synchronized timing
            if self.device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
                torch.mps.synchronize()
            elif self.device.type == "cuda" and hasattr(torch, "cuda") and hasattr(torch.cuda, "synchronize"):
                torch.cuda.synchronize()

            t0 = time.perf_counter()
            with torch.no_grad():
                logits = self.model(ctx_tensor)
                probs = logits.exp() if hasattr(logits, "exp") else torch.softmax(logits, dim=-1)
                top_conf, top_idx = probs.topk(self.top_k, dim=-1)

            if self.device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
                torch.mps.synchronize()
            elif self.device.type == "cuda" and hasattr(torch, "cuda") and hasattr(torch.cuda, "synchronize"):
                torch.cuda.synchronize()

            t1 = time.perf_counter()
            inference_latency_ms = (t1 - t0) * 1000.0

            topk_canonical = top_idx[0].cpu().tolist()
            topk_raw = [self.canonical_to_event.get(c, c) for c in topk_canonical]
            confidence = float(top_conf[0, 0].item()) if top_conf.shape[-1] > 0 else 0.0

            # Anomaly rule: event is unseen or not in top-k predicted candidates
            is_anomaly = is_unseen or (canonical_event not in topk_canonical)

        # Update session state
        session["event_count"] = step_num
        session["last_seen_time"] = event_time
        session["total_inference_ms"] += inference_latency_ms

        alert: Optional[EventAlert] = None
        if is_anomaly:
            session["anomaly_count"] += 1
            if not session["is_anomaly"]:
                session["is_anomaly"] = True
                session["first_anomaly_step"] = step_num
                session["first_anomaly_event"] = raw_int

            context_raw = [self.canonical_to_event.get(c, c) for c in current_canonical_ctx]
            alert = EventAlert(
                block_id=block_id,
                timestamp=event_time,
                step_number=step_num,
                raw_event=raw_int,
                canonical_event=canonical_event,
                topk_predictions=topk_canonical,
                topk_raw_predictions=topk_raw,
                is_unseen=is_unseen,
                confidence=confidence,
                latency_ms=inference_latency_ms,
                context=current_canonical_ctx,
                context_raw=context_raw,
            )
            self.alerts.append(alert)
            self.total_anomalies_detected += 1

        # Advance sliding context window
        ctx_event = canonical_event if not is_unseen else self.no_event_canonical
        session["context"].append(ctx_event)

        # Aggregate telemetry
        self.total_events_processed += 1
        self.total_inference_time_ms += inference_latency_ms
        self.total_queue_time_ms += queue_latency_ms

        return EventResult(
            block_id=block_id,
            timestamp=event_time,
            step_number=step_num,
            raw_event=raw_int,
            canonical_event=canonical_event,
            is_anomaly=is_anomaly,
            topk_predictions=topk_canonical,
            topk_raw_predictions=topk_raw,
            confidence=confidence,
            latency_ms=inference_latency_ms,
            queue_latency_ms=queue_latency_ms,
            alert=alert,
        )

    def cleanup_inactive_sessions(
        self,
        current_time: float,
        timeout: Optional[float] = None,
    ) -> List[SessionSummary]:
        """
        Scan and evict sessions that have exceeded the inactivity timeout threshold.

        Parameters
        ----------
        current_time : float
            Current stream timestamp or wall clock time.
        timeout : float, optional
            Timeout duration. Defaults to self.session_timeout.

        Returns
        -------
        evicted_summaries : list of SessionSummary
            Summary of all evicted sessions.
        """
        thresh = timeout if timeout is not None else self.session_timeout
        evicted: List[SessionSummary] = []
        expired_block_ids: List[str] = []

        for block_id, s in self.sessions.items():
            if (current_time - s["last_seen_time"]) > thresh:
                expired_block_ids.append(block_id)

        for block_id in expired_block_ids:
            s = self.sessions.pop(block_id)
            self.total_sessions_completed += 1
            duration = max(0.0, s["last_seen_time"] - s["start_time"])
            summary = SessionSummary(
                block_id=block_id,
                total_events=s["event_count"],
                total_anomalies=s["anomaly_count"],
                is_anomaly_block=s["is_anomaly"],
                start_time=s["start_time"],
                last_seen_time=s["last_seen_time"],
                duration_seconds=duration,
                first_anomaly_step=s["first_anomaly_step"],
                first_anomaly_event=s["first_anomaly_event"],
            )
            evicted.append(summary)

        return evicted

    def close_session(self, block_id: str) -> Optional[SessionSummary]:
        """
        Explicitly close and evict a session (e.g. upon EOS / block completion).

        Parameters
        ----------
        block_id : str
            Identifier of block to close.

        Returns
        -------
        summary : SessionSummary or None
        """
        if block_id not in self.sessions:
            return None

        s = self.sessions.pop(block_id)
        self.total_sessions_completed += 1
        duration = max(0.0, s["last_seen_time"] - s["start_time"])
        return SessionSummary(
            block_id=block_id,
            total_events=s["event_count"],
            total_anomalies=s["anomaly_count"],
            is_anomaly_block=s["is_anomaly"],
            start_time=s["start_time"],
            last_seen_time=s["last_seen_time"],
            duration_seconds=duration,
            first_anomaly_step=s["first_anomaly_step"],
            first_anomaly_event=s["first_anomaly_event"],
        )

    @property
    def active_session_count(self) -> int:
        """Return the number of currently active sessions in buffer."""
        return len(self.sessions)

    def get_metrics(self) -> Dict[str, Any]:
        """Return real-time stream aggregation metrics and statistics."""
        elapsed_sec = max(1e-6, time.time() - self.start_wall_time)
        throughput = self.total_events_processed / elapsed_sec
        avg_inf_ms = (
            self.total_inference_time_ms / self.total_events_processed
            if self.total_events_processed > 0
            else 0.0
        )
        avg_queue_ms = (
            self.total_queue_time_ms / self.total_events_processed
            if self.total_events_processed > 0
            else 0.0
        )

        return {
            "total_events_processed": self.total_events_processed,
            "total_anomalies_detected": self.total_anomalies_detected,
            "total_sessions_created": self.total_sessions_created,
            "total_sessions_completed": self.total_sessions_completed,
            "active_sessions": len(self.sessions),
            "throughput_eps": float(throughput),
            "mean_inference_latency_ms": float(avg_inf_ms),
            "mean_queue_latency_ms": float(avg_queue_ms),
            "total_alerts": len(self.alerts),
        }

    def reset(self):
        """Reset internal buffers and telemetry."""
        self.sessions.clear()
        self.alerts.clear()
        self.total_events_processed = 0
        self.total_anomalies_detected = 0
        self.total_sessions_created = 0
        self.total_sessions_completed = 0
        self.total_inference_time_ms = 0.0
        self.total_queue_time_ms = 0.0
        self.start_wall_time = time.time()
