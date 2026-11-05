"""
Cross-Domain Benchmark and Zero-Shot Transfer Learning Module for DeepLog.

This module provides parsers, dataset profilers, and zero-shot transfer evaluators
across heterogeneous log domains:
1. `BGLParser`: Parses Blue Gene/L supercomputer failure logs (Alert category,
   Node ID, Component, Message, Timestamp).
2. `ThunderbirdParser`: Parses HPC Thunderbird system logs with high temporal variance.
3. `KubernetesLogParser`: Parses K8s control plane logs (`kube-apiserver`, `etcd`,
   `kubelet`, `coredns`) with JSON and Klog formats.
4. `DomainDatasetSummary`: Summarizes vocabulary entropy, OOV rate, and token distribution.
5. `DomainTransferEvaluator`: Uses `TemplateSimilarityMatcher` and `SemanticDeepLog`
   (CRC32 subword hypersphere projections) to evaluate zero-shot domain transfer
   without retraining.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field
import datetime
import json
import math
import os
import re
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from .semantic import (
    LogSemanticEncoder,
    SemanticDeepLog,
    TemplateSimilarityMatcher,
    extract_subword_features,
    load_hdfs_templates,
    normalize_log_text,
)


# ==============================================================================
# 1. Blue Gene/L (BGL) Supercomputer Log Parser
# ==============================================================================


@dataclass
class BGLRecord:
    """Structured representation of a Blue Gene/L supercomputer log line."""

    alert_category: str
    timestamp: float
    node_id: str
    component: str
    message: str
    date: str = ""
    level: str = "INFO"
    subsystem: str = "RAS"
    full_timestamp: str = ""
    node_repeat: str = ""
    raw: str = ""
    is_alert: bool = False
    is_corrupt: bool = False
    stack_trace: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to JSON-serializable dictionary."""
        return {
            "alert_category": self.alert_category,
            "timestamp": self.timestamp,
            "node_id": self.node_id,
            "component": self.component,
            "message": self.message,
            "date": self.date,
            "level": self.level,
            "subsystem": self.subsystem,
            "is_alert": self.is_alert,
            "is_corrupt": self.is_corrupt,
            "stack_trace": self.stack_trace,
        }


class BGLParser:
    """
    Parser for Blue Gene/L (BGL) failure logs.

    Handles standard Loghub BGL 10-field formats, relaxed format variations,
    corrupt records, and multi-line stack trace / register dump continuations.
    """

    BGL_LEVELS = r"(?:INFO|WARNING|WARN|FATAL|ERROR|SEVERE|DEBUG|FAILURE)"

    # Standard Loghub BGL format regex:
    # Label Timestamp Date Node FullTime NodeRep Subsystem Component Level Message
    # Example:
    # - 1117838570 2005.06.03 R02-M1-N0-C:J12-U11 2005-06-03-15.42.50.675872 R02-M1-N0-C:J12-U11 RAS KERNEL INFO instruction cache parity error corrected
    STANDARD_PATTERN = re.compile(
        r"^(\S+)\s+(\d+(?:\.\d+)?)\s+(\d{4}\.\d{2}\.\d{2})\s+(\S+)\s+(\d{4}-\d{2}-\d{2}-\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+("
        + BGL_LEVELS
        + r")\s*(.*)$",
        re.IGNORECASE,
    )

    # Relaxed BGL format regex (7-8 fields):
    RELAXED_PATTERN = re.compile(
        r"^(\S+)\s+(\d+(?:\.\d+)?)\s+(\d{4}\.\d{2}\.\d{2})\s+(\S+)(?:\s+([A-Za-z0-9_:\-]+))?\s+("
        + BGL_LEVELS
        + r")\s*(.*)$",
        re.IGNORECASE,
    )

    # Multi-line stack trace / register dump indicators
    TRACE_INDICATORS = (
        "iar ",
        "dear ",
        "msr ",
        "lr ",
        "backtrace:",
        "traceback",
        "call trace",
        "stack:",
        "exception:",
        "interrupt:",
    )

    def __init__(self, filter_alerts_only: bool = False):
        """
        Initialize BGLParser.

        Parameters
        ----------
        filter_alerts_only : bool, default=False
            If True, only yields lines where alert_category != '-'.
        """
        self.filter_alerts_only = filter_alerts_only

    def parse_line(self, line: str) -> Optional[BGLRecord]:
        """
        Parse a single BGL log line.

        Parameters
        ----------
        line : str
            Raw log line.

        Returns
        -------
        record : BGLRecord or None
        """
        raw = line.rstrip("\r\n")
        stripped = raw.strip()
        if not stripped:
            return None

        # Check standard format
        m = self.STANDARD_PATTERN.match(stripped)
        if m:
            alert, ts_str, date_str, node1, full_ts, node2, subsys, comp, level, msg = m.groups()
            try:
                ts = float(ts_str)
            except ValueError:
                ts = 0.0

            is_alert = alert != "-" and bool(alert)
            if self.filter_alerts_only and not is_alert:
                return None

            return BGLRecord(
                alert_category=alert,
                timestamp=ts,
                node_id=node1,
                component=comp,
                message=msg.strip(),
                date=date_str,
                level=level.upper(),
                subsystem=subsys,
                full_timestamp=full_ts,
                node_repeat=node2,
                raw=raw,
                is_alert=is_alert,
                is_corrupt=False,
            )

        # Check relaxed format
        m_rel = self.RELAXED_PATTERN.match(stripped)
        if m_rel:
            alert, ts_str, date_str, node1, comp_or_none, level, msg = m_rel.groups()
            try:
                ts = float(ts_str)
            except ValueError:
                ts = 0.0

            comp = comp_or_none or "KERNEL"
            is_alert = alert != "-" and bool(alert)
            if self.filter_alerts_only and not is_alert:
                return None

            return BGLRecord(
                alert_category=alert,
                timestamp=ts,
                node_id=node1,
                component=comp,
                message=msg.strip(),
                date=date_str,
                level=level.upper(),
                raw=raw,
                is_alert=is_alert,
                is_corrupt=False,
            )

        # Fallback: Corrupt or unparseable record
        return BGLRecord(
            alert_category="-",
            timestamp=0.0,
            node_id="UNKNOWN",
            component="UNKNOWN",
            message=stripped,
            level="INFO",
            raw=raw,
            is_alert=False,
            is_corrupt=True,
        )

    def is_continuation_line(self, line: str) -> bool:
        """Determine if a line is a stack trace continuation or register dump."""
        stripped = line.strip()
        if not stripped:
            return False

        # Lines indented with tabs or multiple spaces
        if line.startswith("   ") or line.startswith("\t"):
            return True

        lower = stripped.lower()
        if any(lower.startswith(ind) for ind in self.TRACE_INDICATORS):
            return True

        if self.STANDARD_PATTERN.match(stripped) or self.RELAXED_PATTERN.match(stripped):
            return False

        # Check if line does NOT start with alert label + timestamp
        tokens = stripped.split(maxsplit=2)
        if len(tokens) >= 2:
            try:
                float(tokens[1])
                return False
            except ValueError:
                pass

        return True

    def parse_stream(self, lines: Iterable[str]) -> Iterator[BGLRecord]:
        """
        Parse an iterable stream of log lines, aggregating multi-line stack traces.

        Parameters
        ----------
        lines : Iterable[str]
            Stream of raw log lines.

        Yields
        ------
        record : BGLRecord
        """
        current_record: Optional[BGLRecord] = None
        current_trace: List[str] = []

        for raw_line in lines:
            line = raw_line.rstrip("\r\n")
            if not line.strip():
                continue

            if current_record is not None and self.is_continuation_line(line):
                # Continuation line belongs to current record
                current_trace.append(line.strip())
                continue

            # We reached a new record header: yield pending record
            if current_record is not None:
                if current_trace:
                    trace_str = "\n".join(current_trace)
                    current_record.stack_trace = trace_str
                    current_record.message = f"{current_record.message}\n{trace_str}"
                    current_trace = []
                yield current_record

            current_record = self.parse_line(line)

        # Flush final record
        if current_record is not None:
            if current_trace:
                trace_str = "\n".join(current_trace)
                current_record.stack_trace = trace_str
                current_record.message = f"{current_record.message}\n{trace_str}"
            yield current_record

    def parse_file(self, filepath: str) -> List[BGLRecord]:
        """Parse an entire BGL log file from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"BGL log file not found at '{filepath}'")
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            return list(self.parse_stream(f))

    @staticmethod
    def extract_template(message: str) -> str:
        """
        Extract generalized template from BGL message by masking parameters.

        Masks hex addresses, node locations, core/socket IDs, and numbers.
        """
        t = message
        # Hex values: 0x... or 0b...
        t = re.sub(r"0x[0-9a-fA-F]+", "<param>", t)
        t = re.sub(r"0b[01]+", "<param>", t)
        # BGL Node locations like R02-M1-N0-C:J12-U11
        t = re.sub(r"R\d{2}-M\d-N\d(?:-C:J\d{2}-U\d{2})?", "<param>", t)
        # Hardware register pairs like iar 00106304
        t = re.sub(r"\b[0-9a-fA-F]{8}\b", "<param>", t)
        # Pure numeric literals
        t = re.sub(r"\b\d+\b", "<param>", t)
        # Normalize whitespace
        t = re.sub(r"\s+", " ", t).strip()
        return t


# ==============================================================================
# 2. HPC Thunderbird System Log Parser
# ==============================================================================


@dataclass
class ThunderbirdRecord:
    """Structured representation of an HPC Thunderbird system log line."""

    label: str
    timestamp: float
    node: str
    component: str
    message: str
    date: str = ""
    syslog_time: str = ""
    host: str = ""
    pid: Optional[int] = None
    level: str = "INFO"
    raw: str = ""
    is_alert: bool = False
    is_corrupt: bool = False
    time_delta: Optional[float] = None
    stack_trace: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to JSON-serializable dictionary."""
        return {
            "label": self.label,
            "timestamp": self.timestamp,
            "node": self.node,
            "component": self.component,
            "message": self.message,
            "date": self.date,
            "pid": self.pid,
            "level": self.level,
            "is_alert": self.is_alert,
            "is_corrupt": self.is_corrupt,
            "time_delta": self.time_delta,
            "stack_trace": self.stack_trace,
        }


class ThunderbirdParser:
    """
    Parser for HPC Thunderbird system logs.

    Handles high temporal variance, burst detection, PID extraction,
    corrupt lines, and multi-line kernel oops / call trace continuations.
    """

    # Standard Thunderbird format:
    # Label Timestamp Date User Month Day Time Host Component[PID]: Message
    # Example:
    # - 1131561730 2005.11.09 an179 Nov  9 10:42:10 an179 syslogd.1.4.1: restart.
    # ALERT 1131561735 2005.11.09 cn301 Nov  9 10:42:15 cn301 ntpd[2450]: synchronized to 10.0.0.1
    STANDARD_PATTERN = re.compile(
        r"^(\S+)\s+(\d+(?:\.\d+)?)\s+(\d{4}\.\d{2}\.\d{2})\s+(\S+)\s+([A-Za-z]{3}\s+\d+\s+\d{2}:\d{2}:\d{2})\s+(\S+)\s+([^:\[]+)(?:\[(\d+)\])?:\s*(.*)$"
    )

    # Relaxed Thunderbird format (omits redundant host or syslog timestamp):
    RELAXED_PATTERN = re.compile(
        r"^(\S+)\s+(\d+(?:\.\d+)?)\s+(\d{4}\.\d{2}\.\d{2})\s+(\S+)\s+([^:\[]+)(?:\[(\d+)\])?:\s*(.*)$"
    )

    ALERT_LEVELS = {"ALERT", "CRIT", "CRITICAL", "ERR", "ERROR", "EMERG", "EMERGENCY"}

    def __init__(self, filter_alerts_only: bool = False):
        self.filter_alerts_only = filter_alerts_only

    def parse_line(self, line: str) -> Optional[ThunderbirdRecord]:
        """
        Parse a single Thunderbird log line.

        Parameters
        ----------
        line : str
            Raw log line.

        Returns
        -------
        record : ThunderbirdRecord or None
        """
        raw = line.rstrip("\r\n")
        stripped = raw.strip()
        if not stripped:
            return None

        # Try standard pattern
        m = self.STANDARD_PATTERN.match(stripped)
        if m:
            label, ts_str, date_str, user_node, syslog_time, host, comp, pid_str, msg = m.groups()
            try:
                ts = float(ts_str)
            except ValueError:
                ts = 0.0

            pid = int(pid_str) if pid_str else None
            is_alert = label.upper() in self.ALERT_LEVELS or (label != "-" and bool(label))
            level = label.upper() if label != "-" else "INFO"

            if self.filter_alerts_only and not is_alert:
                return None

            return ThunderbirdRecord(
                label=label,
                timestamp=ts,
                node=user_node,
                component=comp.strip(),
                message=msg.strip(),
                date=date_str,
                syslog_time=syslog_time,
                host=host,
                pid=pid,
                level=level,
                raw=raw,
                is_alert=is_alert,
                is_corrupt=False,
            )

        # Try relaxed pattern
        m_rel = self.RELAXED_PATTERN.match(stripped)
        if m_rel:
            label, ts_str, date_str, node, comp, pid_str, msg = m_rel.groups()
            try:
                ts = float(ts_str)
            except ValueError:
                ts = 0.0

            pid = int(pid_str) if pid_str else None
            is_alert = label.upper() in self.ALERT_LEVELS or (label != "-" and bool(label))
            level = label.upper() if label != "-" else "INFO"

            if self.filter_alerts_only and not is_alert:
                return None

            return ThunderbirdRecord(
                label=label,
                timestamp=ts,
                node=node,
                component=comp.strip(),
                message=msg.strip(),
                date=date_str,
                pid=pid,
                level=level,
                raw=raw,
                is_alert=is_alert,
                is_corrupt=False,
            )

        # Fallback: Corrupt record
        return ThunderbirdRecord(
            label="-",
            timestamp=0.0,
            node="UNKNOWN",
            component="UNKNOWN",
            message=stripped,
            level="INFO",
            raw=raw,
            is_alert=False,
            is_corrupt=True,
        )

    def is_continuation_line(self, line: str) -> bool:
        """Check if line is a continuation trace (e.g., Call Trace:, [<c0123456>])."""
        stripped = line.strip()
        if not stripped:
            return False

        if line.startswith(" ") or line.startswith("\t"):
            return True

        lower = stripped.lower()
        if lower.startswith("call trace:") or lower.startswith("[<") or lower.startswith("traceback"):
            return True

        if self.STANDARD_PATTERN.match(stripped) or self.RELAXED_PATTERN.match(stripped):
            return False

        tokens = stripped.split(maxsplit=2)
        if len(tokens) >= 2:
            try:
                float(tokens[1])
                return False
            except ValueError:
                pass

        return True

    def parse_stream(self, lines: Iterable[str]) -> Iterator[ThunderbirdRecord]:
        """
        Parse stream of Thunderbird logs, computing time deltas and aggregating stack traces.
        """
        current_record: Optional[ThunderbirdRecord] = None
        current_trace: List[str] = []
        last_timestamp: Optional[float] = None

        for raw_line in lines:
            line = raw_line.rstrip("\r\n")
            if not line.strip():
                continue

            if current_record is not None and self.is_continuation_line(line):
                current_trace.append(line.strip())
                continue

            if current_record is not None:
                if current_trace:
                    trace_str = "\n".join(current_trace)
                    current_record.stack_trace = trace_str
                    current_record.message = f"{current_record.message}\n{trace_str}"
                    current_trace = []
                yield current_record

            current_record = self.parse_line(line)
            if current_record is not None and not current_record.is_corrupt:
                if last_timestamp is not None:
                    current_record.time_delta = max(0.0, current_record.timestamp - last_timestamp)
                last_timestamp = current_record.timestamp

        if current_record is not None:
            if current_trace:
                trace_str = "\n".join(current_trace)
                current_record.stack_trace = trace_str
                current_record.message = f"{current_record.message}\n{trace_str}"
            yield current_record

    def parse_file(self, filepath: str) -> List[ThunderbirdRecord]:
        """Parse Thunderbird log file from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Thunderbird log file not found at '{filepath}'")
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            return list(self.parse_stream(f))

    @staticmethod
    def compute_temporal_metrics(
        records: Sequence[ThunderbirdRecord],
        burst_threshold: float = 0.5,
    ) -> Dict[str, float]:
        """
        Compute high temporal variance metrics across a sequence of records.

        Parameters
        ----------
        records : Sequence[ThunderbirdRecord]
            Parsed Thunderbird log records.
        burst_threshold : float, default=0.5
            Maximum seconds delta between records considered part of a burst.

        Returns
        -------
        metrics : Dict[str, float]
            Dictionary containing mean_delta, variance_delta, std_delta, burst_ratio,
            and fano_factor (dispersion ratio).
        """
        valid_records = [r for r in records if not r.is_corrupt and r.timestamp > 0]
        if len(valid_records) < 2:
            return {
                "total_events": float(len(valid_records)),
                "mean_delta": 0.0,
                "variance_delta": 0.0,
                "std_delta": 0.0,
                "min_delta": 0.0,
                "max_delta": 0.0,
                "burst_ratio": 0.0,
                "fano_factor": 1.0,
            }

        deltas: List[float] = []
        burst_count = 0

        for i in range(1, len(valid_records)):
            delta = max(0.0, valid_records[i].timestamp - valid_records[i - 1].timestamp)
            deltas.append(delta)
            if delta <= burst_threshold:
                burst_count += 1

        delta_arr = np.array(deltas, dtype=np.float64)
        mean_d = float(np.mean(delta_arr))
        var_d = float(np.var(delta_arr))
        std_d = float(np.std(delta_arr))
        min_d = float(np.min(delta_arr))
        max_d = float(np.max(delta_arr))
        burst_ratio = float(burst_count / max(1, len(deltas)))

        # Compute Fano factor across 10-second temporal windows
        start_ts = valid_records[0].timestamp
        end_ts = valid_records[-1].timestamp
        window_size = 10.0
        num_windows = max(1, int(math.ceil((end_ts - start_ts) / window_size)))
        counts = [0] * num_windows

        for r in valid_records:
            idx = min(num_windows - 1, max(0, int((r.timestamp - start_ts) // window_size)))
            counts[idx] += 1

        c_arr = np.array(counts, dtype=np.float64)
        mean_c = float(np.mean(c_arr))
        var_c = float(np.var(c_arr))
        fano = float(var_c / mean_c) if mean_c > 0 else 1.0

        return {
            "total_events": float(len(valid_records)),
            "mean_delta": mean_d,
            "variance_delta": var_d,
            "std_delta": std_d,
            "min_delta": min_d,
            "max_delta": max_d,
            "burst_ratio": burst_ratio,
            "fano_factor": fano,
        }


# ==============================================================================
# 3. Kubernetes Control Plane Log Parser (JSON + Klog)
# ==============================================================================


@dataclass
class KubernetesRecord:
    """Structured representation of a Kubernetes control plane log record."""

    timestamp: float
    level: str
    component: str
    caller: str
    message: str
    format: str  # "klog", "json", "coredns", or "raw"
    thread_id: Optional[int] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    stack_trace: Optional[str] = None
    is_corrupt: bool = False
    raw: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to JSON-serializable dictionary."""
        return {
            "timestamp": self.timestamp,
            "level": self.level,
            "component": self.component,
            "caller": self.caller,
            "message": self.message,
            "format": self.format,
            "thread_id": self.thread_id,
            "attributes": self.attributes,
            "stack_trace": self.stack_trace,
            "is_corrupt": self.is_corrupt,
        }


class KubernetesLogParser:
    """
    Parser for Kubernetes control plane logs (`kube-apiserver`, `etcd`, `kubelet`, `coredns`).

    Supports:
    - Klog format: `Lmmdd hh:mm:ss.uuuuuu threadid file:line] msg...`
    - JSON format: `{"ts": ..., "level": ..., "caller": ..., "msg": ...}`
    - CoreDNS syslog format: `[INFO] 127.0.0.1:54321 - ...`
    - Multi-line Go stack traces and panic dumps.
    """

    # Klog regex:
    # I1105 14:23:45.678901   12345 round_trippers.go:466] Response: 200 OK
    KLOG_PATTERN = re.compile(
        r"^([IWEF])(\d{4}\s+\d{2}:\d{2}:\d{2}(?:\.\d+)?)\s+(\d+)\s+([^:\]]+:\d+)\]\s*(.*)$"
    )

    # CoreDNS syslog regex:
    # [INFO] 127.0.0.1:54321 - 42533 "A IN example.com. udp 40 false 512" NOERROR qr,aa,rd 64 0.000234s
    COREDNS_PATTERN = re.compile(
        r"^\[(INFO|WARNING|ERROR|FATAL|DEBUG)\]\s*(.*)$"
    )

    LEVEL_MAP = {
        "I": "INFO",
        "W": "WARNING",
        "E": "ERROR",
        "F": "FATAL",
    }

    def __init__(self, default_component: str = "k8s-control-plane", reference_year: int = 2026):
        """
        Initialize KubernetesLogParser.

        Parameters
        ----------
        default_component : str
            Fallback component when component cannot be deduced.
        reference_year : int
            Year used to resolve MMDD timestamps from Klog logs.
        """
        self.default_component = default_component
        self.reference_year = reference_year

    def parse_line(self, line: str) -> Optional[KubernetesRecord]:
        """
        Parse a single Kubernetes log line in Klog, JSON, or CoreDNS format.

        Parameters
        ----------
        line : str
            Raw log line.

        Returns
        -------
        record : KubernetesRecord or None
        """
        raw = line.rstrip("\r\n")
        stripped = raw.strip()
        if not stripped:
            return None

        # 1. Detect JSON format
        if stripped.startswith("{") or stripped.endswith("}"):
            try:
                data = json.loads(stripped)
                if isinstance(data, dict):
                    return self._parse_json_record(data, raw)
            except Exception:
                # If invalid JSON, treat as corrupt record
                return KubernetesRecord(
                    timestamp=0.0,
                    level="ERROR",
                    component=self.default_component,
                    caller="json_parser",
                    message=stripped,
                    format="json",
                    is_corrupt=True,
                    raw=raw,
                )

        # 2. Detect Klog format
        m_klog = self.KLOG_PATTERN.match(stripped)
        if m_klog:
            lvl_char, time_str, tid_str, caller_str, msg_str = m_klog.groups()
            level = self.LEVEL_MAP.get(lvl_char, "INFO")
            ts = self._parse_klog_timestamp(time_str)
            thread_id = int(tid_str) if tid_str.isdigit() else None
            comp = self._detect_component(caller_str, msg_str)

            # Extract structured key-values from klog msg (e.g. key="val")
            msg_clean, attrs = self._extract_klog_attributes(msg_str)

            return KubernetesRecord(
                timestamp=ts,
                level=level,
                component=comp,
                caller=caller_str,
                message=msg_clean,
                format="klog",
                thread_id=thread_id,
                attributes=attrs,
                is_corrupt=False,
                raw=raw,
            )

        # 3. Detect CoreDNS format
        m_dns = self.COREDNS_PATTERN.match(stripped)
        if m_dns:
            level_str, msg_str = m_dns.groups()
            return KubernetesRecord(
                timestamp=0.0,
                level=level_str.upper(),
                component="coredns",
                caller="plugin/coredns",
                message=msg_str.strip(),
                format="coredns",
                is_corrupt=False,
                raw=raw,
            )

        # 4. Fallback: unformatted raw log line
        return KubernetesRecord(
            timestamp=0.0,
            level="INFO",
            component=self.default_component,
            caller="unknown",
            message=stripped,
            format="raw",
            is_corrupt=False,
            raw=raw,
        )

    def _parse_json_record(self, data: Dict[str, Any], raw: str) -> KubernetesRecord:
        """Parse structured JSON log dictionary from etcd / k8s / coredns."""
        # Level extraction
        raw_level = str(data.get("level") or data.get("severity") or "info").upper()
        if "WARN" in raw_level:
            level = "WARNING"
        elif "ERR" in raw_level:
            level = "ERROR"
        elif "FATAL" in raw_level:
            level = "FATAL"
        else:
            level = "INFO"

        # Timestamp extraction
        ts_val = data.get("ts") or data.get("time") or data.get("timestamp") or 0.0
        ts = self._normalize_timestamp(ts_val)

        # Caller / caller extraction
        caller = str(data.get("caller") or data.get("source") or data.get("file") or "unknown")

        # Message extraction
        msg = str(data.get("msg") or data.get("message") or "")

        # Component extraction
        comp_candidate = data.get("component") or data.get("logger") or data.get("plugin")
        if comp_candidate:
            comp = self._canonical_component(str(comp_candidate))
        else:
            comp = self._detect_component(caller, msg)

        # Remaining fields into attributes
        reserved_keys = {"level", "severity", "ts", "time", "timestamp", "caller", "source", "file", "msg", "message", "component", "logger", "plugin"}
        attrs = {k: v for k, v in data.items() if k not in reserved_keys}

        return KubernetesRecord(
            timestamp=ts,
            level=level,
            component=comp,
            caller=caller,
            message=msg,
            format="json",
            attributes=attrs,
            is_corrupt=False,
            raw=raw,
        )

    def _normalize_timestamp(self, ts_val: Any) -> float:
        """Convert arbitrary timestamp value (epoch float, ms int, or ISO string) to float seconds."""
        if isinstance(ts_val, (int, float)):
            val = float(ts_val)
            # If in milliseconds (> 1e11), convert to seconds
            if val > 1e11:
                return val / 1000.0
            return val
        if isinstance(ts_val, str):
            try:
                # Try float string
                return float(ts_val)
            except ValueError:
                pass
            # Try ISO8601 parsing
            clean_str = ts_val.rstrip("Z")
            for fmt in (
                "%Y-%m-%dT%H:%M:%S.%f",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d %H:%M:%S.%f",
                "%Y-%m-%d %H:%M:%S",
            ):
                try:
                    dt = datetime.datetime.strptime(clean_str, fmt)
                    return dt.replace(tzinfo=datetime.timezone.utc).timestamp()
                except ValueError:
                    continue
        return 0.0

    def _parse_klog_timestamp(self, time_str: str) -> float:
        """Parse MMDD HH:MM:SS.uuuuuu Klog timestamp format."""
        try:
            # Example: '1105 14:23:45.678901'
            tokens = time_str.split()
            if len(tokens) == 2:
                mmdd, hms = tokens
                month = int(mmdd[:2])
                day = int(mmdd[2:])
                time_parts = hms.split(":")
                hour = int(time_parts[0])
                minute = int(time_parts[1])
                sec_parts = time_parts[2].split(".")
                sec = int(sec_parts[0])
                usec = int(sec_parts[1][:6].ljust(6, "0")) if len(sec_parts) > 1 else 0

                dt = datetime.datetime(
                    year=self.reference_year,
                    month=month,
                    day=day,
                    hour=hour,
                    minute=minute,
                    second=sec,
                    microsecond=usec,
                    tzinfo=datetime.timezone.utc,
                )
                return dt.timestamp()
        except Exception:
            pass
        return 0.0

    def _detect_component(self, caller: str, msg: str) -> str:
        """Infer Kubernetes component from caller filepath and message semantics."""
        caller_l = caller.lower()
        msg_l = msg.lower()

        if any(k in caller_l or k in msg_l for k in ("apiserver", "handler.go", "round_tripper", "admission", "audit", "watchcache", "storage_etcd")):
            return "kube-apiserver"
        if any(k in caller_l or k in msg_l for k in ("etcd", "etcdserver", "raft", "wal", "lease", "mvcc")):
            return "etcd"
        if any(k in caller_l or k in msg_l for k in ("kubelet", "pleg", "dockershim", "kuberuntime", "container_manager", "volume_manager", "pod_workers")):
            return "kubelet"
        if any(k in caller_l or k in msg_l for k in ("coredns", "dnsserver", "plugin/errors", "plugin/dns")):
            return "coredns"

        return self.default_component

    def _canonical_component(self, comp: str) -> str:
        """Normalize component names to standard control plane names."""
        c = comp.lower()
        if "apiserver" in c:
            return "kube-apiserver"
        if "etcd" in c:
            return "etcd"
        if "kubelet" in c:
            return "kubelet"
        if "coredns" in c or "dns" in c:
            return "coredns"
        return comp

    def _extract_klog_attributes(self, msg_str: str) -> Tuple[str, Dict[str, Any]]:
        """Extract structured key=val or key="val" from Klog message."""
        attrs: Dict[str, Any] = {}
        # Match pattern key="value" or key=value
        kv_pattern = re.compile(r'(\b[a-zA-Z0-9_\-\.]+)=("(?:[^"\\]|\\.)*"|\S+)')
        cleaned_msg = msg_str
        for match in kv_pattern.finditer(msg_str):
            k, v = match.groups()
            if v.startswith('"') and v.endswith('"'):
                v = v[1:-1]
            attrs[k] = v

        # If quotes exist, extract core message
        m_msg = re.match(r'^\s*"([^"]+)"\s*(.*)$', msg_str)
        if m_msg:
            cleaned_msg = m_msg.group(1)

        return cleaned_msg, attrs

    def is_continuation_line(self, line: str) -> bool:
        """Check if line is a Go stack trace continuation or panic dump."""
        stripped = line.strip()
        if not stripped:
            return False

        if line.startswith("\t") or line.startswith("    "):
            return True

        lower = stripped.lower()
        if (
            lower.startswith("goroutine ")
            or lower.startswith("panic:")
            or lower.startswith("created by ")
            or lower.startswith("runtime.")
            or lower.startswith("k8s.io/")
            or lower.startswith("github.com/")
        ):
            return True

        # Check if line starts with Klog or JSON pattern
        if self.KLOG_PATTERN.match(stripped) or (stripped.startswith("{") and stripped.endswith("}")):
            return False

        return False

    def parse_stream(self, lines: Iterable[str]) -> Iterator[KubernetesRecord]:
        """Parse stream of Kubernetes logs, aggregating multi-line stack traces."""
        current_record: Optional[KubernetesRecord] = None
        current_trace: List[str] = []

        for raw_line in lines:
            line = raw_line.rstrip("\r\n")
            if not line.strip():
                continue

            if current_record is not None and self.is_continuation_line(line):
                current_trace.append(line.strip())
                continue

            if current_record is not None:
                if current_trace:
                    trace_str = "\n".join(current_trace)
                    current_record.stack_trace = trace_str
                    current_record.message = f"{current_record.message}\n{trace_str}"
                    current_trace = []
                yield current_record

            current_record = self.parse_line(line)

        if current_record is not None:
            if current_trace:
                trace_str = "\n".join(current_trace)
                current_record.stack_trace = trace_str
                current_record.message = f"{current_record.message}\n{trace_str}"
            yield current_record

    def parse_file(self, filepath: str) -> List[KubernetesRecord]:
        """Parse Kubernetes log file from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Kubernetes log file not found at '{filepath}'")
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            return list(self.parse_stream(f))


# ==============================================================================
# 4. Domain Dataset Summary (Entropy, OOV Rate, Token Distribution)
# ==============================================================================


@dataclass
class DomainDatasetSummary:
    """Statistical summary profiling vocabulary entropy, OOV rates, and token distribution."""

    domain_name: str
    total_lines: int
    total_tokens: int
    unique_tokens: int
    vocabulary_entropy: float
    token_counts: Dict[str, int]
    vocabulary: Set[str]
    mean_tokens_per_line: float
    temporal_variance: Optional[float] = None
    mean_time_delta: Optional[float] = None
    templates: List[str] = field(default_factory=list)
    unique_templates_count: int = 0

    @classmethod
    def from_texts(
        cls,
        texts: Sequence[str],
        domain_name: str = "unknown",
        timestamps: Optional[Sequence[float]] = None,
        templates: Optional[Sequence[str]] = None,
    ) -> DomainDatasetSummary:
        """
        Build DomainDatasetSummary from raw log texts or template strings.

        Parameters
        ----------
        texts : Sequence[str]
            List of log lines or messages.
        domain_name : str
            Identifier for the domain (e.g. 'HDFS', 'BGL', 'Thunderbird', 'Kubernetes').
        timestamps : Sequence[float], optional
            Timestamps associated with the logs to compute temporal variance.
        templates : Sequence[str], optional
            Known template strings in the domain.
        """
        token_counter: collections.Counter[str] = collections.Counter()
        total_tokens = 0

        for line in texts:
            norm = normalize_log_text(line)
            tokens = re.findall(r"[a-zA-Z0-9_<>]+", norm.lower())
            token_counter.update(tokens)
            total_tokens += len(tokens)

        unique_tokens = len(token_counter)
        vocab = set(token_counter.keys())

        # Compute Shannon entropy in bits: H = -sum(p * log2(p))
        if total_tokens > 0 and unique_tokens > 1:
            probs = np.array(list(token_counter.values()), dtype=np.float64) / total_tokens
            entropy = float(-np.sum(probs * np.log2(probs)))
        else:
            entropy = 0.0

        mean_tokens = total_tokens / max(1, len(texts))

        # Temporal metrics
        temp_var = None
        mean_delta = None
        if timestamps is not None and len(timestamps) > 1:
            deltas = [max(0.0, timestamps[i] - timestamps[i - 1]) for i in range(1, len(timestamps))]
            if deltas:
                d_arr = np.array(deltas, dtype=np.float64)
                mean_delta = float(np.mean(d_arr))
                temp_var = float(np.var(d_arr))

        t_list = list(templates) if templates is not None else []
        u_templates = len(set(t_list)) if t_list else 0

        return cls(
            domain_name=domain_name,
            total_lines=len(texts),
            total_tokens=total_tokens,
            unique_tokens=unique_tokens,
            vocabulary_entropy=entropy,
            token_counts=dict(token_counter.most_common()),
            vocabulary=vocab,
            mean_tokens_per_line=mean_tokens,
            temporal_variance=temp_var,
            mean_time_delta=mean_delta,
            templates=t_list,
            unique_templates_count=u_templates,
        )

    @classmethod
    def from_records(
        cls,
        records: Sequence[Any],
        domain_name: str = "unknown",
        templates: Optional[Sequence[str]] = None,
    ) -> DomainDatasetSummary:
        """Build summary from sequence of parsed records (BGLRecord, ThunderbirdRecord, KubernetesRecord)."""
        texts = [getattr(r, "message", str(r)) for r in records]
        timestamps = [getattr(r, "timestamp", 0.0) for r in records if hasattr(r, "timestamp") and getattr(r, "timestamp") > 0]
        ts_seq = timestamps if len(timestamps) == len(records) else None
        return cls.from_texts(texts=texts, domain_name=domain_name, timestamps=ts_seq, templates=templates)

    def compute_oov_rate(
        self,
        reference: Union[Set[str], DomainDatasetSummary],
        weighted: bool = True,
    ) -> float:
        """
        Compute Out-of-Vocabulary (OOV) rate against a reference vocabulary.

        Parameters
        ----------
        reference : Set[str] or DomainDatasetSummary
            Reference vocabulary or domain summary.
        weighted : bool, default=True
            If True, weights each OOV token by its occurrence frequency.
            If False, computes unique token OOV fraction.

        Returns
        -------
        oov_rate : float in [0.0, 1.0]
        """
        ref_vocab = reference.vocabulary if isinstance(reference, DomainDatasetSummary) else reference

        if not self.vocabulary:
            return 0.0

        if not ref_vocab:
            return 1.0

        if weighted:
            if self.total_tokens == 0:
                return 0.0
            oov_tokens_count = sum(count for tok, count in self.token_counts.items() if tok not in ref_vocab)
            return float(oov_tokens_count / self.total_tokens)
        else:
            unique_oov = len(self.vocabulary - ref_vocab)
            return float(unique_oov / max(1, self.unique_tokens))

    def compare_with(self, reference: DomainDatasetSummary) -> Dict[str, Any]:
        """
        Compare this domain dataset summary against a reference domain summary.

        Returns
        -------
        comparison : Dict[str, Any]
            Comparative metrics including weighted and unique OOV rates,
            Jaccard similarity, shared tokens count, and entropy difference.
        """
        shared_tokens = self.vocabulary & reference.vocabulary
        union_tokens = self.vocabulary | reference.vocabulary
        jaccard = float(len(shared_tokens) / max(1, len(union_tokens)))

        weighted_oov = self.compute_oov_rate(reference, weighted=True)
        unique_oov = self.compute_oov_rate(reference, weighted=False)
        entropy_diff = self.vocabulary_entropy - reference.vocabulary_entropy

        return {
            "source_domain": reference.domain_name,
            "target_domain": self.domain_name,
            "source_vocab_size": reference.unique_tokens,
            "target_vocab_size": self.unique_tokens,
            "shared_tokens_count": len(shared_tokens),
            "jaccard_similarity": jaccard,
            "oov_rate_weighted": weighted_oov,
            "oov_rate_unique": unique_oov,
            "source_entropy": reference.vocabulary_entropy,
            "target_entropy": self.vocabulary_entropy,
            "entropy_diff": entropy_diff,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert summary to JSON-serializable dictionary."""
        top_tokens = dict(list(self.token_counts.items())[:20])
        return {
            "domain_name": self.domain_name,
            "total_lines": self.total_lines,
            "total_tokens": self.total_tokens,
            "unique_tokens": self.unique_tokens,
            "vocabulary_entropy": self.vocabulary_entropy,
            "mean_tokens_per_line": self.mean_tokens_per_line,
            "temporal_variance": self.temporal_variance,
            "mean_time_delta": self.mean_time_delta,
            "unique_templates_count": self.unique_templates_count,
            "top_tokens": top_tokens,
        }

    def summary(self) -> str:
        """Generate human-readable multi-line summary string."""
        lines = [
            f"=== Domain Dataset Summary: {self.domain_name} ===",
            f"  Total Lines          : {self.total_lines:,}",
            f"  Total Tokens         : {self.total_tokens:,}",
            f"  Unique Vocabulary    : {self.unique_tokens:,}",
            f"  Vocabulary Entropy   : {self.vocabulary_entropy:.4f} bits",
            f"  Mean Tokens / Line   : {self.mean_tokens_per_line:.2f}",
        ]
        if self.temporal_variance is not None:
            lines.append(f"  Mean Time Delta (s)  : {self.mean_time_delta:.4f}")
            lines.append(f"  Temporal Variance    : {self.temporal_variance:.4f}")
        if self.unique_templates_count > 0:
            lines.append(f"  Unique Templates     : {self.unique_templates_count:,}")
        top_toks = ", ".join([f"{k} ({v})" for k, v in list(self.token_counts.items())[:5]])
        lines.append(f"  Top Tokens           : {top_toks}")
        return "\n".join(lines)


# ==============================================================================
# 5. Domain Transfer Evaluator (Zero-Shot Hypersphere Projection)
# ==============================================================================


@dataclass
class TargetTemplateMatch:
    """Detailed matching diagnostics for an individual target template."""

    target_id: str
    target_text: str
    matched_source_id: str
    matched_source_text: str
    cosine_similarity: float
    is_covered: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_id": self.target_id,
            "target_text": self.target_text,
            "matched_source_id": self.matched_source_id,
            "matched_source_text": self.matched_source_text,
            "cosine_similarity": self.cosine_similarity,
            "is_covered": self.is_covered,
        }


@dataclass
class SequenceTransferResult:
    """Metrics from zero-shot sequence prediction on target domain."""

    total_sequences: int
    top_1_hits: int
    top_3_hits: int
    top_5_hits: int
    top_1_accuracy: float
    top_3_accuracy: float
    top_5_accuracy: float
    mean_cosine_similarity: float
    anomaly_count: int
    anomaly_rate: float
    roc_auc: Optional[float] = None
    f1_score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_sequences": self.total_sequences,
            "top_1_hits": self.top_1_hits,
            "top_3_hits": self.top_3_hits,
            "top_5_hits": self.top_5_hits,
            "top_1_accuracy": self.top_1_accuracy,
            "top_3_accuracy": self.top_3_accuracy,
            "top_5_accuracy": self.top_5_accuracy,
            "mean_cosine_similarity": self.mean_cosine_similarity,
            "anomaly_count": self.anomaly_count,
            "anomaly_rate": self.anomaly_rate,
            "roc_auc": self.roc_auc,
            "f1_score": self.f1_score,
        }


@dataclass
class DomainTransferResult:
    """Full cross-domain zero-shot evaluation result."""

    source_domain: str
    target_domain: str
    source_templates_count: int
    target_templates_count: int
    mean_cosine_similarity: float
    std_cosine_similarity: float
    min_cosine_similarity: float
    max_cosine_similarity: float
    semantic_coverage_50: float  # Fraction with cosine sim >= 0.5
    semantic_coverage_70: float  # Fraction with cosine sim >= 0.7
    discrete_oov_rate: float     # Discrete DeepLog hard OOV rate (typically 1.0)
    semantic_oov_rate_50: float  # Semantic OOV rate at threshold 0.5 (1 - coverage_50)
    template_matches: List[TargetTemplateMatch]
    sequence_result: Optional[SequenceTransferResult] = None
    lexical_comparison: Optional[Dict[str, Any]] = None
    summary_report: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_domain": self.source_domain,
            "target_domain": self.target_domain,
            "source_templates_count": self.source_templates_count,
            "target_templates_count": self.target_templates_count,
            "mean_cosine_similarity": self.mean_cosine_similarity,
            "std_cosine_similarity": self.std_cosine_similarity,
            "min_cosine_similarity": self.min_cosine_similarity,
            "max_cosine_similarity": self.max_cosine_similarity,
            "semantic_coverage_50": self.semantic_coverage_50,
            "semantic_coverage_70": self.semantic_coverage_70,
            "discrete_oov_rate": self.discrete_oov_rate,
            "semantic_oov_rate_50": self.semantic_oov_rate_50,
            "template_matches": [m.to_dict() for m in self.template_matches],
            "sequence_result": self.sequence_result.to_dict() if self.sequence_result else None,
            "lexical_comparison": self.lexical_comparison,
        }


class DomainTransferEvaluator:
    """
    Zero-Shot Cross-Domain Transfer Evaluator.

    Utilizes `TemplateSimilarityMatcher` and `SemanticDeepLog` continuous CRC32
    subword hypersphere projections to evaluate transfer from a source domain (e.g. HDFS)
    to target domains (BGL, Thunderbird, Kubernetes) without retraining.
    """

    def __init__(
        self,
        source_templates: Optional[Dict[str, str]] = None,
        model: Optional[SemanticDeepLog] = None,
        encoder: Optional[LogSemanticEncoder] = None,
        matcher: Optional[TemplateSimilarityMatcher] = None,
        embedding_dim: int = 128,
    ):
        """
        Initialize DomainTransferEvaluator.

        Parameters
        ----------
        source_templates : Dict[str, str], optional
            Source domain log templates (e.g. canonical 29 HDFS templates).
            If None, attempts to load default HDFS templates.
        model : SemanticDeepLog, optional
            Pretrained or initialized SemanticDeepLog instance.
        encoder : LogSemanticEncoder, optional
            Semantic encoder mapping log templates to continuous unit hypersphere.
        matcher : TemplateSimilarityMatcher, optional
            Cosine similarity matcher fitted on source templates.
        embedding_dim : int, default=128
            Continuous embedding dimension.
        """
        self.embedding_dim = embedding_dim
        self.encoder = encoder or LogSemanticEncoder(embedding_dim=embedding_dim)

        # Resolve source templates
        if source_templates is not None:
            self.source_templates = dict(source_templates)
        else:
            try:
                self.source_templates = load_hdfs_templates()
            except Exception:
                self.source_templates = {
                    f"S{i}": f"Generic source template {i}" for i in range(1, 10)
                }

        # Initialize matcher
        if matcher is not None:
            self.matcher = matcher
        else:
            self.matcher = TemplateSimilarityMatcher(
                encoder=self.encoder,
                templates=self.source_templates,
            )

        # Initialize sequence model
        if model is not None:
            self.model = model
        else:
            self.model = SemanticDeepLog(
                embedding_dim=embedding_dim,
                hidden_size=embedding_dim,
                encoder=self.encoder,
            )
            # Register source templates into template bank
            if self.source_templates:
                self.model.register_templates(self.source_templates)

    def evaluate_template_transfer(
        self,
        target_templates: Dict[str, str],
        threshold: float = 0.5,
    ) -> List[TargetTemplateMatch]:
        """
        Evaluate zero-shot semantic alignment between target templates and source templates.

        Parameters
        ----------
        target_templates : Dict[str, str]
            Target domain templates {id: text}.
        threshold : float, default=0.5
            Cosine similarity threshold for semantic coverage.

        Returns
        -------
        matches : List[TargetTemplateMatch]
        """
        matches: List[TargetTemplateMatch] = []
        for t_id, t_text in target_templates.items():
            best_candidates = self.matcher.match(t_text, top_k=1)
            if best_candidates:
                src_id, sim_raw, src_text = best_candidates[0]
                sim = max(-1.0, min(1.0, float(sim_raw)))
                is_covered = sim >= threshold
                matches.append(
                    TargetTemplateMatch(
                        target_id=t_id,
                        target_text=t_text,
                        matched_source_id=src_id,
                        matched_source_text=src_text,
                        cosine_similarity=sim,
                        is_covered=is_covered,
                    )
                )
            else:
                matches.append(
                    TargetTemplateMatch(
                        target_id=t_id,
                        target_text=t_text,
                        matched_source_id="NONE",
                        matched_source_text="",
                        cosine_similarity=0.0,
                        is_covered=False,
                    )
                )
        return matches

    def evaluate_sequence_transfer(
        self,
        target_sequences: Sequence[Sequence[str]],
        candidate_templates: Optional[Dict[str, str]] = None,
        anomaly_labels: Optional[Sequence[int]] = None,
        top_k: int = 5,
        similarity_threshold: float = 0.5,
    ) -> SequenceTransferResult:
        """
        Evaluate zero-shot next-event prediction and anomaly detection on target sequences.

        Parameters
        ----------
        target_sequences : Sequence[Sequence[str]]
            Sequences of log events from the target domain. Each sequence must have length >= 2.
        candidate_templates : Dict[str, str], optional
            Target candidate templates. If None, uses target templates extracted from sequences.
        anomaly_labels : Sequence[int], optional
            Ground truth binary labels (0 = normal, 1 = anomaly) for each sequence.
        top_k : int, default=5
            Top-K window for next-event prediction accuracy.
        similarity_threshold : float, default=0.5
            Cosine similarity threshold below which predicted vs observed is flagged anomalous.

        Returns
        -------
        result : SequenceTransferResult
        """
        valid_seqs = [seq for seq in target_sequences if len(seq) >= 2]
        if not valid_seqs:
            return SequenceTransferResult(
                total_sequences=0,
                top_1_hits=0,
                top_3_hits=0,
                top_5_hits=0,
                top_1_accuracy=0.0,
                top_3_accuracy=0.0,
                top_5_accuracy=0.0,
                mean_cosine_similarity=0.0,
                anomaly_count=0,
                anomaly_rate=0.0,
            )

        # Prepare candidates for ranking
        if candidate_templates is not None and len(candidate_templates) > 0:
            c_ids = list(candidate_templates.keys())
            c_texts = [candidate_templates[k] for k in c_ids]
        else:
            # Aggregate all unique observed events
            unique_events = sorted(list({ev for seq in valid_seqs for ev in seq}))
            c_ids = [f"T{i}" for i in range(len(unique_events))]
            c_texts = unique_events

        self.encoder.eval()
        with torch.no_grad():
            candidate_embs = self.encoder.encode(c_texts, normalize=True)

        top_1_hits = 0
        top_3_hits = 0
        top_5_hits = 0
        sims: List[float] = []
        anomalies = 0
        anomaly_scores: List[float] = []

        self.model.eval()
        with torch.no_grad():
            for i, seq in enumerate(valid_seqs):
                prefix = list(seq[:-1])
                target_ev = seq[-1]

                # Encode observed next event
                target_emb = self.encoder.encode(target_ev, normalize=True)

                # Model predicts next embedding vector in continuous space
                pred_emb = self.model.predict_next_embedding(prefix)

                # Compute cosine similarity between prediction and actual next event
                cos_sim_raw = float(torch.sum(pred_emb * target_emb, dim=-1).item())
                cos_sim = max(-1.0, min(1.0, cos_sim_raw))
                sims.append(cos_sim)

                # Anomaly score: 1.0 - cos_sim
                score = 1.0 - cos_sim
                anomaly_scores.append(score)
                if cos_sim < similarity_threshold:
                    anomalies += 1

                # Rank target candidate against candidate pool
                pool_sims = torch.matmul(pred_emb, candidate_embs.T).squeeze(0)  # (num_candidates,)
                _, ranked_indices = pool_sims.sort(descending=True)
                ranked_idx_list = ranked_indices.tolist()

                # Find rank of target event
                # Either exact text match in candidates or closest cosine candidate
                target_cand_sims = torch.matmul(target_emb, candidate_embs.T).squeeze(0)
                closest_cand_idx = int(torch.argmax(target_cand_sims).item())

                if closest_cand_idx in ranked_idx_list[:1]:
                    top_1_hits += 1
                if closest_cand_idx in ranked_idx_list[: min(3, len(ranked_idx_list))]:
                    top_3_hits += 1
                if closest_cand_idx in ranked_idx_list[: min(top_k, len(ranked_idx_list))]:
                    top_5_hits += 1

        total = len(valid_seqs)
        top_1_acc = float(top_1_hits / total)
        top_3_acc = float(top_3_hits / total)
        top_5_acc = float(top_5_hits / total)
        mean_sim = float(np.mean(sims)) if sims else 0.0
        anom_rate = float(anomalies / total)

        roc_auc = None
        f1 = None
        if anomaly_labels is not None and len(anomaly_labels) == total:
            roc_auc = self._compute_roc_auc(anomaly_labels, anomaly_scores)
            pred_labels = [1 if s >= (1.0 - similarity_threshold) else 0 for s in anomaly_scores]
            f1 = self._compute_f1(anomaly_labels, pred_labels)

        return SequenceTransferResult(
            total_sequences=total,
            top_1_hits=top_1_hits,
            top_3_hits=top_3_hits,
            top_5_hits=top_5_hits,
            top_1_accuracy=top_1_acc,
            top_3_accuracy=top_3_acc,
            top_5_accuracy=top_5_acc,
            mean_cosine_similarity=mean_sim,
            anomaly_count=anomalies,
            anomaly_rate=anom_rate,
            roc_auc=roc_auc,
            f1_score=f1,
        )

    def evaluate(
        self,
        source_domain: str,
        target_domain: str,
        target_templates: Dict[str, str],
        target_sequences: Optional[Sequence[Sequence[str]]] = None,
        anomaly_labels: Optional[Sequence[int]] = None,
        source_texts: Optional[Sequence[str]] = None,
        target_texts: Optional[Sequence[str]] = None,
        top_k: int = 5,
        threshold: float = 0.5,
    ) -> DomainTransferResult:
        """
        Perform comprehensive cross-domain zero-shot transfer evaluation.

        Parameters
        ----------
        source_domain : str
            Source domain name (e.g. 'HDFS').
        target_domain : str
            Target domain name (e.g. 'BGL', 'Thunderbird', 'Kubernetes').
        target_templates : Dict[str, str]
            Target templates {id: text}.
        target_sequences : Sequence[Sequence[str]], optional
            Log event sequences from target domain.
        anomaly_labels : Sequence[int], optional
            Binary labels for anomaly sequences.
        source_texts : Sequence[str], optional
            Raw source texts for vocabulary profiling.
        target_texts : Sequence[str], optional
            Raw target texts for vocabulary profiling.
        top_k : int, default=5
            Top-K candidate acceptance window.
        threshold : float, default=0.5
            Cosine similarity threshold.

        Returns
        -------
        transfer_result : DomainTransferResult
        """
        # 1. Template alignment
        matches = self.evaluate_template_transfer(target_templates, threshold=threshold)
        sims = [m.cosine_similarity for m in matches] if matches else [0.0]

        mean_sim = float(np.mean(sims))
        std_sim = float(np.std(sims))
        min_sim = float(np.min(sims))
        max_sim = float(np.max(sims))

        cov_50 = float(sum(1 for s in sims if s >= 0.5) / max(1, len(sims)))
        cov_70 = float(sum(1 for s in sims if s >= 0.7) / max(1, len(sims)))
        sem_oov_50 = 1.0 - cov_50

        # Discrete DeepLog has 100% OOV when crossing domains without retraining
        discrete_oov = 1.0

        # 2. Sequence evaluation
        seq_res: Optional[SequenceTransferResult] = None
        if target_sequences is not None and len(target_sequences) > 0:
            seq_res = self.evaluate_sequence_transfer(
                target_sequences=target_sequences,
                candidate_templates=target_templates,
                anomaly_labels=anomaly_labels,
                top_k=top_k,
                similarity_threshold=threshold,
            )

        # 3. Lexical comparison
        lex_comp: Optional[Dict[str, Any]] = None
        if source_texts is not None and target_texts is not None:
            s_sum = DomainDatasetSummary.from_texts(source_texts, domain_name=source_domain)
            t_sum = DomainDatasetSummary.from_texts(target_texts, domain_name=target_domain)
            lex_comp = t_sum.compare_with(s_sum)

        # 4. Generate report
        report_lines = [
            f"=== Zero-Shot Domain Transfer: {source_domain} -> {target_domain} ===",
            f"  Source Templates     : {len(self.source_templates)}",
            f"  Target Templates     : {len(target_templates)}",
            f"  Discrete DeepLog OOV : {discrete_oov * 100:.1f}% (Total failure without retraining)",
            f"  Semantic OOV (>=0.5) : {sem_oov_50 * 100:.1f}%",
            f"  Semantic Coverage    : {cov_50 * 100:.1f}% (sim >= 0.50), {cov_70 * 100:.1f}% (sim >= 0.70)",
            f"  Mean Cosine Sim      : {mean_sim:.4f} +/- {std_sim:.4f} (range: [{min_sim:.4f}, {max_sim:.4f}])",
        ]
        if seq_res is not None:
            report_lines.extend([
                f"  Sequence Evaluation  : {seq_res.total_sequences} sequences",
                f"    Zero-Shot Top-1 Acc: {seq_res.top_1_accuracy * 100:.2f}%",
                f"    Zero-Shot Top-3 Acc: {seq_res.top_3_accuracy * 100:.2f}%",
                f"    Zero-Shot Top-5 Acc: {seq_res.top_5_accuracy * 100:.2f}%",
                f"    Mean Seq Cosine Sim: {seq_res.mean_cosine_similarity:.4f}",
            ])
            if seq_res.roc_auc is not None:
                report_lines.append(f"    Anomaly ROC-AUC    : {seq_res.roc_auc:.4f}")
            if seq_res.f1_score is not None:
                report_lines.append(f"    Anomaly F1 Score   : {seq_res.f1_score:.4f}")

        report = "\n".join(report_lines)

        return DomainTransferResult(
            source_domain=source_domain,
            target_domain=target_domain,
            source_templates_count=len(self.source_templates),
            target_templates_count=len(target_templates),
            mean_cosine_similarity=mean_sim,
            std_cosine_similarity=std_sim,
            min_cosine_similarity=min_sim,
            max_cosine_similarity=max_sim,
            semantic_coverage_50=cov_50,
            semantic_coverage_70=cov_70,
            discrete_oov_rate=discrete_oov,
            semantic_oov_rate_50=sem_oov_50,
            template_matches=matches,
            sequence_result=seq_res,
            lexical_comparison=lex_comp,
            summary_report=report,
        )

    @staticmethod
    def _compute_roc_auc(y_true: Sequence[int], y_scores: Sequence[float]) -> float:
        """Compute ROC-AUC from binary labels and continuous scores."""
        pos = [score for label, score in zip(y_true, y_scores) if label == 1]
        neg = [score for label, score in zip(y_true, y_scores) if label == 0]
        if not pos or not neg:
            return 0.5

        # Mann-Whitney U test statistic for AUC
        u_stat = 0
        for p in pos:
            for n in neg:
                if p > n:
                    u_stat += 1.0
                elif p == n:
                    u_stat += 0.5
        return float(u_stat / (len(pos) * len(neg)))

    @staticmethod
    def _compute_f1(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
        """Compute F1 score for binary predictions."""
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)

        precision = tp / max(1, tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / max(1, tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall == 0:
            return 0.0
        return float(2 * precision * recall / (precision + recall))


# ==============================================================================
# 6. Representative Synthetic Datasets for Benchmark Reproducibility
# ==============================================================================


def get_synthetic_hdfs_logs() -> List[str]:
    """Canonical representative HDFS log lines."""
    return [
        "081109 203518 143 INFO dfs.DataNode$DataXceiver: Receiving block blk_-1070887964 src: /10.250.19.102:54106 dest: /10.250.19.102:50010",
        "081109 203519 143 INFO dfs.DataNode$BlockReceiver: Receiving block blk_-1070887964 src: /10.250.19.102:54106 dest: /10.250.19.102:50010",
        "081109 203520 142 INFO dfs.DataBlockScanner: Verification succeeded for blk_-1070887964",
        "081109 203521 145 INFO dfs.DataNode$DataXceiver: Received block blk_-1070887964 of size 67108864 from /10.250.19.102",
        "081109 203521 145 INFO dfs.DataNode$PacketResponder: PacketResponder 1 for block blk_-1070887964 terminating",
        "081109 203522 23 INFO dfs.FSNamesystem: BLOCK* NameSystem.addStoredBlock: blockMap updated: 10.250.19.102:50010 is added to blk_-1070887964",
        "081109 203523 23 INFO dfs.FSNamesystem: BLOCK* NameSystem.allocateBlock: /user/root/rand/_temporary/_task_0001_m_000001/part-00001. blk_4429188",
        "081109 203524 142 INFO dfs.DataBlockScanner: Verification succeeded for blk_4429188",
        "081109 203525 24 INFO dfs.FSNamesystem: BLOCK* ask 10.250.19.102:50010 to delete blk_-1070887964",
        "081109 203526 24 INFO dfs.FSNamesystem: Deleting block blk_-1070887964 file /mnt/hadoop/dfs/data/current/blk_-1070887964",
    ]


def get_synthetic_bgl_logs() -> List[str]:
    """Representative Blue Gene/L supercomputer failure logs."""
    return [
        "- 1117838570 2005.06.03 R02-M1-N0-C:J12-U11 2005-06-03-15.42.50.675872 R02-M1-N0-C:J12-U11 RAS KERNEL INFO instruction cache parity error corrected",
        "- 1117838571 2005.06.03 R02-M1-N0-C:J12-U11 2005-06-03-15.42.51.100000 R02-M1-N0-C:J12-U11 RAS KERNEL INFO microseconds common interrupt: iar 00106304 dear 0245b000",
        "- 1117838572 2005.06.03 R00-M0-N1-C:J02-U01 2005-06-03-15.42.52.200000 R00-M0-N1-C:J02-U01 RAS APP INFO tree auxiliary header format 0, 1 packets",
        "- 1117838573 2005.06.03 R00-M0-N1-C:J02-U01 2005-06-03-15.42.53.300000 R00-M0-N1-C:J02-U01 RAS APP INFO 1 floating point alignment exceptions",
        "FATAL 1117838574 2005.06.03 R02-M1-N0-C:J12-U11 2005-06-03-15.42.54.400000 R02-M1-N0-C:J12-U11 RAS KERNEL FATAL double hummer alignment exception",
        "   iar 00106304 dear 0245b000",
        "   msr 00001000 lr 001062f8",
        "FAILURE 1117838575 2005.06.03 R02-M1-N0-C:J12-U11 2005-06-03-15.42.55.500000 R02-M1-N0-C:J12-U11 RAS HARDWARE FAILURE ddr error(s) detected and corrected on rank 0, symbol 1",
        "WARNING 1117838576 2005.06.03 R04-M1-NF-C:J08-U11 2005-06-03-15.42.56.600000 R04-M1-NF-C:J08-U11 RAS DISCOVERY WARNING link card not responding to ping",
        "FATAL 1117838577 2005.06.03 R04-M1-NF-C:J08-U11 2005-06-03-15.42.57.700000 R04-M1-NF-C:J08-U11 RAS KERNEL FATAL panic: critical interrupt trap handler failure",
    ]


def get_synthetic_thunderbird_logs() -> List[str]:
    """Representative HPC Thunderbird logs exhibiting high temporal burstiness."""
    return [
        "- 1131561730 2005.11.09 an179 Nov  9 10:42:10 an179 syslogd.1.4.1: restart.",
        "- 1131561730 2005.11.09 an179 Nov  9 10:42:10 an179 kernel: Linux version 2.6.9-22.ELsmp",
        "- 1131561730 2005.11.09 an179 Nov  9 10:42:10 an179 kernel: BIOS-provided physical RAM map:",
        "- 1131561731 2005.11.09 an179 Nov  9 10:42:11 an179 pbs_mom: node an179 started",
        "ALERT 1131561731 2005.11.09 cn301 Nov  9 10:42:11 cn301 ntpd[2450]: synchronized to 10.0.0.1, stratum 2",
        "ALERT 1131561731 2005.11.09 cn302 Nov  9 10:42:11 cn302 ntpd[2451]: synchronized to 10.0.0.1, stratum 2",
        "CRIT 1131561780 2005.11.09 cn112 Nov  9 10:43:00 cn112 kernel: EDAC MC0: CE row 2, channel 0, label '': ECC error",
        "ERR 1131561781 2005.11.09 cn112 Nov  9 10:43:01 cn112 lustre: LustreError: 13-0: an error occurred during read",
        "ERR 1131561781 2005.11.09 cn112 Nov  9 10:43:01 cn112 kernel: Unable to handle kernel paging request",
        "   Call Trace:",
        "   [<c0123456>] sys_read+0x46/0x70",
    ]


def get_synthetic_kubernetes_logs() -> List[str]:
    """Representative Kubernetes control plane logs (Klog, JSON, CoreDNS)."""
    return [
        'I1105 14:23:45.678901   12345 round_trippers.go:466] Response: 200 OK GET /api/v1/namespaces/default/pods/nginx',
        'I1105 14:23:46.123456   12345 handler.go:142] "Authentication successful" user="system:serviceaccount:default:default"',
        '{"level":"info","ts":1730816626.789,"logger":"etcdserver","caller":"etcdserver/server.go:123","msg":"ready to serve client requests"}',
        '{"level":"warn","ts":1730816627.123,"logger":"etcdserver","caller":"etcdserver/server.go:1234","msg":"apply request took too long","took":"125ms","expected":"100ms"}',
        'W1105 14:23:48.345678    5678 kubelet.go:412] "Failed to start container" pod="default/nginx" err="context deadline exceeded"',
        'E1105 14:23:49.987654    5678 reflector.go:138] Failed to watch *v1.Pod: failed to list *v1.Pod: connection refused',
        '[INFO] 127.0.0.1:54321 - 42533 "A IN kubernetes.default.svc.cluster.local. udp 40 false 512" NOERROR qr,aa,rd 64 0.000234s',
        '[ERROR] plugin/errors: 2 kubernetes.default.svc.cluster.local. A: unreachable backend',
        '{"level":"error","ts":1730816630.555,"logger":"kube-apiserver","caller":"handler.go:45","msg":"etcd cluster unavailable","error":"rpc error: code = Unavailable"}',
        'F1105 14:23:51.111222       1 server.go:142] panic: unable to create etcd client',
        '   goroutine 1 [running]:',
        '   k8s.io/kubernetes/cmd/kube-apiserver/app.Run(...)',
    ]


def get_synthetic_domain_datasets() -> Dict[str, Dict[str, Any]]:
    """
    Generate fully populated parsed records, templates, sequences, and summaries
    for HDFS, BGL, Thunderbird, and Kubernetes domains for cross-domain benchmarking.
    """
    # 1. HDFS
    hdfs_raw = get_synthetic_hdfs_logs()
    hdfs_templates = {
        "E1": "Receiving block <*> src: <*> dest: <*>",
        "E2": "Verification succeeded for <*>",
        "E3": "Received block <*> of size <*> from <*>",
        "E4": "PacketResponder <*> for block <*> terminating",
        "E5": "BLOCK* NameSystem.addStoredBlock: blockMap updated: <*> is added to <*>",
        "E6": "BLOCK* NameSystem.allocateBlock: <*> blk_<*>",
        "E7": "Deleting block <*> file <*>",
    }
    hdfs_sequences = [
        ["Receiving block", "Verification succeeded", "PacketResponder terminating"],
        ["Receiving block", "Received block", "blockMap updated"],
        ["BLOCK* NameSystem.allocateBlock", "Verification succeeded", "Deleting block"],
    ]
    hdfs_summary = DomainDatasetSummary.from_texts(
        texts=hdfs_raw,
        domain_name="HDFS",
        templates=list(hdfs_templates.values()),
    )

    # 2. BGL
    bgl_parser = BGLParser()
    bgl_records = list(bgl_parser.parse_stream(get_synthetic_bgl_logs()))
    bgl_templates = {
        "BGL_1": "instruction cache parity error corrected",
        "BGL_2": "microseconds common interrupt: iar <*> dear <*>",
        "BGL_3": "tree auxiliary header format <*>, <*> packets",
        "BGL_4": "<*> floating point alignment exceptions",
        "BGL_5": "double hummer alignment exception",
        "BGL_6": "ddr error(s) detected and corrected on rank <*>, symbol <*>",
        "BGL_7": "link card not responding to ping",
        "BGL_8": "panic: critical interrupt trap handler failure",
    }
    bgl_sequences = [
        ["instruction cache parity error corrected", "microseconds common interrupt", "double hummer alignment exception"],
        ["tree auxiliary header format", "floating point alignment exceptions"],
        ["link card not responding to ping", "panic: critical interrupt trap handler failure"],
    ]
    bgl_summary = DomainDatasetSummary.from_records(
        records=bgl_records,
        domain_name="BGL",
        templates=list(bgl_templates.values()),
    )

    # 3. Thunderbird
    tb_parser = ThunderbirdParser()
    tb_records = list(tb_parser.parse_stream(get_synthetic_thunderbird_logs()))
    tb_templates = {
        "TB_1": "syslogd.<*>: restart.",
        "TB_2": "Linux version <*> ELsmp",
        "TB_3": "BIOS-provided physical RAM map:",
        "TB_4": "node <*> started",
        "TB_5": "synchronized to <*>, stratum <*>",
        "TB_6": "EDAC MC0: CE row <*>, channel <*>, label: ECC error",
        "TB_7": "LustreError: an error occurred during read",
        "TB_8": "Unable to handle kernel paging request",
    }
    tb_sequences = [
        ["syslogd restart", "Linux version", "BIOS-provided physical RAM map"],
        ["synchronized to stratum", "node started"],
        ["EDAC MC0: ECC error", "LustreError during read", "Unable to handle kernel paging request"],
    ]
    tb_summary = DomainDatasetSummary.from_records(
        records=tb_records,
        domain_name="Thunderbird",
        templates=list(tb_templates.values()),
    )

    # 4. Kubernetes
    k8s_parser = KubernetesLogParser()
    k8s_records = list(k8s_parser.parse_stream(get_synthetic_kubernetes_logs()))
    k8s_templates = {
        "K8S_1": "Response: 200 OK GET <*> pods",
        "K8S_2": "Authentication successful user=<*>",
        "K8S_3": "ready to serve client requests",
        "K8S_4": "apply request took too long took=<*>",
        "K8S_5": "Failed to start container pod=<*> err=<*>",
        "K8S_6": "Failed to watch *v1.Pod: connection refused",
        "K8S_7": "A IN kubernetes.default.svc.cluster.local NOERROR",
        "K8S_8": "plugin/errors: unreachable backend",
        "K8S_9": "etcd cluster unavailable error=<*>",
        "K8S_10": "panic: unable to create etcd client",
    }
    k8s_sequences = [
        ["Response: 200 OK GET pods", "Authentication successful"],
        ["ready to serve client requests", "apply request took too long"],
        ["Failed to start container", "Failed to watch *v1.Pod: connection refused", "etcd cluster unavailable"],
    ]
    k8s_summary = DomainDatasetSummary.from_records(
        records=k8s_records,
        domain_name="Kubernetes",
        templates=list(k8s_templates.values()),
    )

    return {
        "HDFS": {
            "raw": hdfs_raw,
            "records": hdfs_raw,
            "templates": hdfs_templates,
            "sequences": hdfs_sequences,
            "summary": hdfs_summary,
        },
        "BGL": {
            "raw": get_synthetic_bgl_logs(),
            "records": bgl_records,
            "templates": bgl_templates,
            "sequences": bgl_sequences,
            "summary": bgl_summary,
        },
        "Thunderbird": {
            "raw": get_synthetic_thunderbird_logs(),
            "records": tb_records,
            "templates": tb_templates,
            "sequences": tb_sequences,
            "summary": tb_summary,
        },
        "Kubernetes": {
            "raw": get_synthetic_kubernetes_logs(),
            "records": k8s_records,
            "templates": k8s_templates,
            "sequences": k8s_sequences,
            "summary": k8s_summary,
        },
    }
