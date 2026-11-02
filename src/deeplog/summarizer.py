"""
DeepLog GenAI Incident Diagnostic & SRE Postmortem Engine
=========================================================
Synthesizes multi-dimensional telemetry across DeepLog components:
- Step 1: Temporal & Timeout Anomalies (Time delta intervals, execution durations, EOS token completion)
- Step 2: Parameter Anomalies (Packet sizes, transmission latencies, statistical z-scores)
- Step 3: Attention Root Cause Attribution (Multi-head / Self-Attention weights isolating antecedent triggers)
- Step 4: Semantic Meanings (HDFS Log Templates, exception categorization, natural language descriptors)
- Step 5: Graph Topology & GNN Anomalies (Execution DAG structural validity, cycles, GNN anomaly scores)

Produces production-grade SRE Incident Reports in both Markdown and structured JSON.
"""

from typing import Dict, Any, List, Optional, Tuple, Union, Sequence, Set
from dataclasses import dataclass, field, asdict
import os
import json
import math
import time
from datetime import datetime
import numpy as np
import networkx as nx
import torch

from .graph_gnn import LogGraphBuilder, LogGNNAnomalyDetector
from .streaming import EventAlert, SessionSummary


# ============================================================================ #
#                         Default HDFS Template Registry                       #
# ============================================================================ #

DEFAULT_HDFS_TEMPLATES: Dict[str, str] = {
    "E1": "Adding an already existing block",
    "E2": "Verification succeeded for block",
    "E3": "Served block to client",
    "E4": "Got exception while serving block to client",
    "E5": "Receiving block src dest",
    "E6": "Received block src dest of size",
    "E7": "writeBlock received exception",
    "E8": "PacketResponder for block Interrupted",
    "E9": "Received block of size from src",
    "E10": "PacketResponder Exception",
    "E11": "PacketResponder for block terminating",
    "E12": "Exception writing block to mirror",
    "E13": "Receiving empty packet for block",
    "E14": "Exception in receiveBlock for block",
    "E15": "Changing block file offset of block",
    "E16": "Transmitted block to dest",
    "E17": "Failed to transfer block to dest",
    "E18": "Starting thread to transfer block to dest",
    "E19": "Reopen Block",
    "E20": "Unexpected error trying to delete block: BlockInfo not found in volumeMap",
    "E21": "Deleting block file",
    "E22": "BLOCK* NameSystem allocateBlock",
    "E23": "BLOCK* NameSystem delete is added to invalidSet",
    "E24": "BLOCK* Removing block from neededReplications as it does not belong to any file",
    "E25": "BLOCK* ask DataNode to replicate to target",
    "E26": "BLOCK* NameSystem addStoredBlock: blockMap updated",
    "E27": "BLOCK* NameSystem addStoredBlock: Redundant addStoredBlock request received",
    "E28": "BLOCK* NameSystem addStoredBlock: addStoredBlock request received but does not belong to any file",
    "E29": "PendingReplicationMonitor timed out block",
    "<EOS>": "End of Sequence (Session Terminated Normally)",
    "-1337": "NO_EVENT (Context Timeout / Padding)",
    "-1338": "EOS_ID (End of Session Token)",
}

# Semantic exception keywords indicating potential failure severity
FATAL_EVENTS: Set[str] = {"E4", "E7", "E8", "E10", "E11", "E12", "E14", "E17", "E20", "E29"}
NORMAL_IO_EVENTS: Set[str] = {"E2", "E3", "E5", "E6", "E9", "E16", "E22", "E26"}


# ============================================================================ #
#                           Incident Data Structures                           #
# ============================================================================ #

@dataclass
class AnomalyEvidence:
    """Detailed evidence supporting an identified anomaly across the 5 diagnostic pillars."""
    dimension: str  # "temporal", "parameter", "sequence", "attention", "graph_topology"
    severity: str   # "CRITICAL", "HIGH", "MEDIUM", "LOW"
    title: str
    description: str
    score: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TimelineItem:
    """Chronological event entry in an incident timeline."""
    step: int
    timestamp: Optional[float]
    event_id: str
    template: str
    is_anomaly: bool
    anomaly_dimension: Optional[str] = None
    attention_weight: float = 0.0
    latency_ms: Optional[float] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class IncidentReport:
    """
    Comprehensive, production-grade SRE Incident Report.
    Synthesizes temporal, parameter, attention, semantic, and graph topology findings.
    """
    incident_id: str
    block_id: str
    severity: str  # "CRITICAL", "HIGH", "MEDIUM", "LOW"
    status: str    # "TRIGGERED", "INVESTIGATING", "RESOLVED"
    timestamp: str
    executive_summary: str
    root_cause_analysis: Dict[str, Any]
    anomaly_breakdown: Dict[str, Any]
    timeline: List[TimelineItem]
    impact_analysis: Dict[str, Any]
    remediation_suggestions: List[Dict[str, str]]
    evidence_list: List[AnomalyEvidence] = field(default_factory=list)
    mermaid_diagram: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to JSON-serializable Python dictionary."""
        return {
            "incident_id": self.incident_id,
            "block_id": self.block_id,
            "severity": self.severity,
            "status": self.status,
            "timestamp": self.timestamp,
            "executive_summary": self.executive_summary,
            "root_cause_analysis": self.root_cause_analysis,
            "anomaly_breakdown": self.anomaly_breakdown,
            "timeline": [item.to_dict() for item in self.timeline],
            "impact_analysis": self.impact_analysis,
            "remediation_suggestions": self.remediation_suggestions,
            "evidence_list": [e.to_dict() for e in self.evidence_list],
            "mermaid_diagram": self.mermaid_diagram,
            "metadata": self.metadata,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert report to formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_markdown(self) -> str:
        """
        Render production-grade GitHub-flavored Markdown SRE Incident Report
        with callouts, summary tables, Mermaid execution graph, and runbook.
        """
        sev_color = {
            "CRITICAL": "CRITICAL",
            "HIGH": "HIGH",
            "MEDIUM": "MEDIUM",
            "LOW": "LOW",
        }.get(self.severity, "INFO")

        callout_type = "CAUTION" if self.severity in ("CRITICAL", "HIGH") else "WARNING"

        md = []
        md.append(f"# DeepLog SRE Incident Postmortem: {self.incident_id}\n")
        md.append(f"**Target Resource / Block ID:** `{self.block_id}`  ")
        md.append(f"**Severity:** `{self.severity}` | **Status:** `{self.status}` | **Generated At:** `{self.timestamp}`\n")
        md.append("---\n")

        # Executive Summary Callout
        md.append(f"> [!{callout_type}]")
        md.append(f"> **Executive Summary:**  ")
        md.append(f"> {self.executive_summary}\n")

        # Root Cause Analysis Section
        rca = self.root_cause_analysis
        md.append("## 1. Root Cause Analysis (RCA)")
        md.append(f"- **Primary Root Cause Event:** `{rca.get('primary_root_cause_event', 'N/A')}`: {rca.get('primary_root_cause_template', 'N/A')}")
        md.append(f"- **Failure Domain:** {rca.get('failure_category', 'Distributed Execution Anomaly')}")
        md.append(f"- **Diagnostic Confidence:** `{float(rca.get('confidence_score', 0.95)):.1%}`")
        md.append(f"- **Causal Narrative:** {rca.get('causal_chain', 'Detailed sequence analysis below.')}\n")

        # Attention Root Cause Attribution Table
        attn_list = rca.get("attention_attribution", [])
        if attn_list:
            md.append("### Attention-Weighted Antecedent Attribution (Step 3)")
            md.append("| Step | Event ID | Log Template | Attention Weight | Attribution % | Role |")
            md.append("|:---:|:---:|:---|:---:|:---:|:---|")
            for item in attn_list:
                step = item.get("step", "-")
                eid = item.get("event_id", "-")
                tmpl = item.get("template", "-")
                w = float(item.get("weight", 0.0))
                pct = float(item.get("percentage", 0.0))
                role = item.get("role", "Antecedent Context")
                md.append(f"| {step} | `{eid}` | {tmpl} | {w:.4f} | {pct:.1f}% | {role} |")
            md.append("")

        # Multi-Dimensional Anomaly Breakdown Table
        md.append("## 2. Multi-Dimensional Anomaly Breakdown")
        md.append("| Diagnostic Dimension | Status | Severity | Evidence Summary |")
        md.append("|:---|:---:|:---:|:---|")

        ab = self.anomaly_breakdown
        t_stat = "ANOMALY" if ab.get("temporal", {}).get("is_anomaly") else "NORMAL"
        t_sev = ab.get("temporal", {}).get("severity", "LOW")
        t_desc = ab.get("temporal", {}).get("summary", "No temporal delay violations observed.")
        md.append(f"| **Step 1: Temporal (Timeouts/EOS)** | `{t_stat}` | `{t_sev}` | {t_desc} |")

        p_stat = "ANOMALY" if ab.get("parameter", {}).get("is_anomaly") else "NORMAL"
        p_sev = ab.get("parameter", {}).get("severity", "LOW")
        p_desc = ab.get("parameter", {}).get("summary", "All parameters within standard bounds.")
        md.append(f"| **Step 2: Parameter Outliers** | `{p_stat}` | `{p_sev}` | {p_desc} |")

        s_stat = "ANOMALY" if ab.get("sequence", {}).get("is_anomaly") else "NORMAL"
        s_sev = ab.get("sequence", {}).get("severity", "LOW")
        s_desc = ab.get("sequence", {}).get("summary", "Sequence transitions matched expected path.")
        md.append(f"| **Step 3: Execution Path (Next-Event)** | `{s_stat}` | `{s_sev}` | {s_desc} |")

        g_stat = "ANOMALY" if ab.get("graph_topology", {}).get("is_anomaly") else "NORMAL"
        g_sev = ab.get("graph_topology", {}).get("severity", "LOW")
        g_desc = ab.get("graph_topology", {}).get("summary", "Graph topology matches normal execution DAG.")
        md.append(f"| **Step 5: Graph Topology (GNN)** | `{g_stat}` | `{g_sev}` | {g_desc} |\n")

        # Mermaid Graph Diagram
        if self.mermaid_diagram:
            md.append("## 3. Execution Graph & Transition Topology")
            md.append("```mermaid")
            md.append(self.mermaid_diagram.strip())
            md.append("```\n")

        # Incident Timeline
        md.append("## 4. Chronological Incident Timeline")
        md.append("| Step | Timestamp | Event ID | Log Message Template | Interval (ms) | Anomaly? | Notes |")
        md.append("|:---:|:---:|:---:|:---|:---:|:---:|:---|")
        for item in self.timeline:
            st = item.step
            ts_str = f"{item.timestamp:.3f}" if item.timestamp is not None else "-"
            lat_str = f"{item.latency_ms:.1f}" if item.latency_ms is not None else "-"
            anom_badge = f"`YES ({item.anomaly_dimension})`" if item.is_anomaly else "No"
            notes = item.notes or "-"
            md.append(f"| {st} | {ts_str} | `{item.event_id}` | {item.template} | {lat_str} | {anom_badge} | {notes} |")
        md.append("")

        # Impact Analysis
        impact = self.impact_analysis
        md.append("## 5. Blast Radius & Impact Analysis")
        md.append(f"- **Affected Services:** {', '.join(impact.get('affected_components', ['HDFS DataNode']))}")
        md.append(f"- **Data Durability Risk:** `{impact.get('data_durability_risk', 'Low')}`")
        md.append(f"- **Replica Integrity:** {impact.get('replica_integrity', 'Replication pipeline affected')}")
        md.append(f"- **Service Level Indicator (SLI):** {impact.get('sli_impact', 'Minor tail latency increase')}\n")

        # Remediation Runbook
        md.append("## 6. Actionable SRE Remediation Runbook")
        for idx, rem in enumerate(self.remediation_suggestions, 1):
            prio = rem.get("priority", f"P{idx}")
            action = rem.get("action", "")
            cmd = rem.get("command", "")
            rationale = rem.get("rationale", "")
            md.append(f"### {prio}: {action}")
            if rationale:
                md.append(f"*{rationale}*")
            if cmd:
                md.append("```bash")
                md.append(cmd)
                md.append("```")
            md.append("")

        return "\n".join(md)

    def save(self, filepath: str, format: str = "both") -> Tuple[Optional[str], Optional[str]]:
        """
        Save the incident report to file.

        Parameters
        ----------
        filepath : str
            Base filepath (e.g. "reports/INC_123").
        format : str, default="both"
            "both", "markdown", or "json".
        """
        base, ext = os.path.splitext(filepath)
        md_path, json_path = None, None

        if format in ("both", "markdown"):
            md_path = base + ".md" if ext != ".md" else filepath
            os.makedirs(os.path.dirname(os.path.abspath(md_path)), exist_ok=True)
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(self.to_markdown())

        if format in ("both", "json"):
            json_path = base + ".json" if ext != ".json" else filepath
            os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)
            with open(json_path, "w", encoding="utf-8") as f:
                f.write(self.to_json(indent=2))

        return md_path, json_path


# ============================================================================ #
#                           Incident Diagnostic Engine                         #
# ============================================================================ #

class IncidentSummarizer:
    """
    IncidentSummarizer: Automated diagnostic engine synthesizing DeepLog telemetry.

    Synthesizes:
    - Step 1: Temporal anomalies (inter-event timeouts, session durations, EOS)
    - Step 2: Parameter anomalies (packet sizes, z-scores, buffer errors)
    - Step 3: Attention root cause attribution (multi-head attention weights)
    - Step 4: Semantic log meanings (HDFS template mapping, failure domain)
    - Step 5: Graph topology anomalies (DAG validation, cycles, GNN scores)
    """

    def __init__(
        self,
        templates: Optional[Dict[str, str]] = None,
        template_file: Optional[str] = None,
        timeout_threshold: float = 5.0,
        parameter_z_threshold: float = 3.0,
        graph_anomaly_threshold: float = 0.5,
        default_block_size: int = 67108864,  # Standard 64MB HDFS block
    ):
        """
        Parameters
        ----------
        templates : Dict[str, str], optional
            Custom dictionary mapping event IDs to template strings.
        template_file : str, optional
            Path to CSV file with EventId,EventTemplate columns.
        timeout_threshold : float, default=5.0
            Maximum expected time delta between consecutive events in seconds.
        parameter_z_threshold : float, default=3.0
            Z-score cutoff for statistical parameter outliers.
        graph_anomaly_threshold : float, default=0.5
            Decision threshold for GNN anomaly detector.
        default_block_size : int, default=67108864
            Expected HDFS block size (64MB).
        """
        self.timeout_threshold = timeout_threshold
        self.parameter_z_threshold = parameter_z_threshold
        self.graph_anomaly_threshold = graph_anomaly_threshold
        self.default_block_size = default_block_size

        # Initialize template registry
        self.templates = dict(DEFAULT_HDFS_TEMPLATES)
        if template_file is not None and os.path.exists(template_file):
            self._load_templates_from_file(template_file)
        elif os.path.exists("dataset/HDFS_v1/HDFS.log_templates.csv"):
            self._load_templates_from_file("dataset/HDFS_v1/HDFS.log_templates.csv")

        if templates:
            self.templates.update(templates)

        self.graph_builder = LogGraphBuilder(vocab_size=64)

    def _load_templates_from_file(self, filepath: str):
        """Load templates from CSV file."""
        try:
            import csv
            with open(filepath, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                for row in reader:
                    if len(row) >= 2:
                        eid = row[0].strip()
                        tmpl = row[1].strip()
                        self.templates[eid] = tmpl
        except Exception:
            pass

    def resolve_template(self, event_id: Union[int, str]) -> str:
        """Resolve an event ID to its human-readable template string."""
        s_id = str(event_id)
        if s_id in self.templates:
            return self.templates[s_id]

        # Check prefixed format "E{id}"
        if isinstance(event_id, int) or s_id.isdigit():
            e_prefix = f"E{int(s_id)}"
            if e_prefix in self.templates:
                return self.templates[e_prefix]

        return f"Log Event {event_id} (Unmapped Pattern)"

    # ------------------------------------------------------------------------ #
    #                       Diagnostic Analysis Sub-Engines                    #
    # ------------------------------------------------------------------------ #

    def analyze_temporal(
        self,
        events: List[Union[int, str]],
        timestamps: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """Step 1: Analyze temporal timeout and EOS sequence anomalies."""
        if timestamps is None or len(timestamps) < 2:
            return {
                "is_anomaly": False,
                "severity": "LOW",
                "summary": "Insufficient timestamp data to assess temporal anomalies.",
                "timeouts": [],
                "total_duration": 0.0,
                "missing_eos": False,
            }

        timeouts = []
        for i in range(1, len(timestamps)):
            delta = timestamps[i] - timestamps[i - 1]
            if delta > self.timeout_threshold:
                timeouts.append({
                    "step_from": i - 1,
                    "step_to": i,
                    "delta_seconds": float(delta),
                    "event_from": str(events[i - 1]),
                    "event_to": str(events[i]),
                })

        total_duration = timestamps[-1] - timestamps[0]
        last_event = str(events[-1])
        has_eos = last_event in ("<EOS>", "-1338", "EOS")
        missing_eos = (len(events) > 1 and not has_eos and any(str(e) in FATAL_EVENTS for e in events))

        is_anomaly = (len(timeouts) > 0 or missing_eos)
        severity = "HIGH" if (len(timeouts) > 1 or (len(timeouts) > 0 and timeouts[0]["delta_seconds"] > 30.0)) else ("MEDIUM" if is_anomaly else "LOW")

        summary_parts = []
        if timeouts:
            max_to = max(timeouts, key=lambda x: x["delta_seconds"])
            summary_parts.append(f"Detected {len(timeouts)} inter-event timeout(s) (Peak latency {max_to['delta_seconds']:.1f}s at step {max_to['step_to']}).")
        if missing_eos:
            summary_parts.append("Session terminated abnormally without expected <EOS> completion token.")

        summary = " ".join(summary_parts) if summary_parts else "Normal temporal cadence observed."

        return {
            "is_anomaly": is_anomaly,
            "severity": severity,
            "summary": summary,
            "timeouts": timeouts,
            "total_duration": float(total_duration),
            "missing_eos": missing_eos,
        }

    def analyze_parameters(
        self,
        events: List[Union[int, str]],
        parameters: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Step 2: Analyze parameter values (sizes, latencies, error codes)."""
        if not parameters or len(parameters) == 0:
            return {
                "is_anomaly": False,
                "severity": "LOW",
                "summary": "No parameter payloads extracted for analysis.",
                "anomalous_parameters": [],
            }

        anomalies = []
        sizes = []

        for idx, p in enumerate(parameters):
            if not isinstance(p, dict):
                continue

            # Check packet/block size
            if "size" in p:
                try:
                    sz = float(p["size"])
                    sizes.append((idx, sz))
                    if sz <= 0:
                        anomalies.append({
                            "step": idx,
                            "parameter": "size",
                            "value": sz,
                            "reason": "Invalid or zero packet size transmitted.",
                            "severity": "HIGH",
                        })
                    elif sz > self.default_block_size * 2:
                        anomalies.append({
                            "step": idx,
                            "parameter": "size",
                            "value": sz,
                            "reason": f"Transmitted size exceeds 2x default block size ({sz / (1024*1024):.1f} MB).",
                            "severity": "MEDIUM",
                        })
                except (ValueError, TypeError):
                    pass

            # Check status codes
            if "status" in p and str(p["status"]).upper() in ("ERROR", "FAILED", "EXCEPTION", "500", "503"):
                anomalies.append({
                    "step": idx,
                    "parameter": "status",
                    "value": p["status"],
                    "reason": f"Explicit error status detected in event parameters: {p['status']}",
                    "severity": "HIGH",
                })

        # Statistical z-score outlier check on sizes
        if len(sizes) >= 4:
            vals = [s[1] for s in sizes]
            mean_sz = np.mean(vals)
            std_sz = np.std(vals)
            if std_sz > 1e-4:
                for step, sz in sizes:
                    z = abs(sz - mean_sz) / std_sz
                    if z > self.parameter_z_threshold:
                        anomalies.append({
                            "step": step,
                            "parameter": "size",
                            "value": sz,
                            "reason": f"Statistical parameter outlier (z-score: {z:.2f}).",
                            "severity": "MEDIUM",
                        })

        is_anomaly = len(anomalies) > 0
        severity = "HIGH" if any(a["severity"] == "HIGH" for a in anomalies) else ("MEDIUM" if is_anomaly else "LOW")
        summary = (
            f"Detected {len(anomalies)} parameter outlier(s) (e.g. {anomalies[0]['reason']})"
            if anomalies
            else "All monitored numeric and status parameters were within nominal distributions."
        )

        return {
            "is_anomaly": is_anomaly,
            "severity": severity,
            "summary": summary,
            "anomalous_parameters": anomalies,
        }

    def analyze_attention(
        self,
        events: List[Union[int, str]],
        attention_weights: Optional[Union[List[float], torch.Tensor, np.ndarray]] = None,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """Step 3: Attention-weighted Root Cause Attribution."""
        if attention_weights is None:
            # Default uniform attention attribution over antecedent context
            seq_len = len(events)
            if seq_len == 0:
                return []
            weights = np.ones(seq_len) / float(seq_len)
        else:
            if isinstance(attention_weights, torch.Tensor):
                weights = attention_weights.detach().cpu().numpy()
            else:
                weights = np.array(attention_weights)

            if weights.ndim > 1:
                weights = weights.flatten()

            if len(weights) != len(events):
                # Align lengths
                min_len = min(len(weights), len(events))
                weights = weights[:min_len]

            # Normalize weights
            sum_w = float(np.sum(weights))
            if sum_w > 1e-8:
                weights = weights / sum_w
            else:
                weights = np.ones(len(weights)) / float(len(weights))

        # Rank indices by attention mass
        ranked_indices = np.argsort(weights)[::-1]
        results = []
        for rank, idx in enumerate(ranked_indices[:top_k], 1):
            eid = str(events[idx])
            tmpl = self.resolve_template(eid)
            w = float(weights[idx])
            role = "Primary Trigger" if rank == 1 else "Context Trigger"
            results.append({
                "step": int(idx),
                "event_id": eid,
                "template": tmpl,
                "weight": w,
                "percentage": float(w * 100.0),
                "role": role,
            })

        return results

    def analyze_semantics(
        self,
        events: List[Union[int, str]],
    ) -> Dict[str, Any]:
        """Step 4: Semantic log analysis and failure domain categorization."""
        fatal_found = []
        for idx, e in enumerate(events):
            s_e = str(e)
            if s_e in FATAL_EVENTS or (isinstance(e, int) and f"E{e}" in FATAL_EVENTS):
                fatal_found.append((idx, s_e))

        if not fatal_found:
            return {
                "has_fatal_event": False,
                "primary_failure_event": None,
                "primary_failure_step": None,
                "failure_category": "Normal Operation",
                "severity": "LOW",
            }

        first_fatal_step, first_fatal_event = fatal_found[0]
        tmpl = self.resolve_template(first_fatal_event)

        # Categorize failure domain
        tmpl_lower = tmpl.lower()
        if "packetresponder" in tmpl_lower or "terminating" in tmpl_lower or "interrupted" in tmpl_lower:
            category = "DataNode PacketResponder Thread Termination"
        elif "exception in receiveblock" in tmpl_lower or "writeblock received exception" in tmpl_lower:
            category = "Block Data Transmission Exception"
        elif "mirror" in tmpl_lower:
            category = "Pipeline Mirroring Failure"
        elif "timed out" in tmpl_lower:
            category = "Pending Replication Timeout"
        elif "blockinfo not found" in tmpl_lower or "does not belong to any file" in tmpl_lower:
            category = "NameNode Metadata Inconsistency"
        else:
            category = "Unclassified Distributed Execution Exception"

        return {
            "has_fatal_event": True,
            "primary_failure_event": first_fatal_event,
            "primary_failure_step": first_fatal_step,
            "failure_category": category,
            "severity": "HIGH",
        }

    def analyze_graph_topology(
        self,
        events: List[Union[int, str]],
        timestamps: Optional[List[float]] = None,
        graph: Optional[nx.DiGraph] = None,
        gnn_result: Optional[Dict[str, Any]] = None,
        reference_graph: Optional[nx.DiGraph] = None,
    ) -> Dict[str, Any]:
        """Step 5: Execution Graph Topology and GNN anomaly detection."""
        g = graph or self.graph_builder.build_graph(events, timestamps=timestamps)
        topo_metrics = LogGraphBuilder.extract_topological_features(g)
        structural_anom = LogGraphBuilder.detect_structural_anomalies(g, reference_graph=reference_graph)

        gnn_score = 0.0
        gnn_is_anomaly = False
        if gnn_result is not None:
            gnn_score = float(gnn_result.get("anomaly_score", 0.0))
            gnn_is_anomaly = bool(gnn_result.get("is_anomaly", gnn_score >= self.graph_anomaly_threshold))
        elif structural_anom["has_structural_anomaly"]:
            gnn_score = float(structural_anom["structural_score"])
            gnn_is_anomaly = bool(gnn_score >= self.graph_anomaly_threshold)

        is_anomaly = bool(structural_anom["has_structural_anomaly"] or gnn_is_anomaly)
        severity = "CRITICAL" if (topo_metrics["has_cycles"] and gnn_score > 0.8) else ("HIGH" if is_anomaly else "LOW")

        summary_parts = []
        if topo_metrics["has_cycles"]:
            summary_parts.append(f"Topological cycle detected ({topo_metrics['cycle_count']} cycle(s) found).")
        if structural_anom["novel_edges"]:
            summary_parts.append(f"Encountered {len(structural_anom['novel_edges'])} novel transition edge(s) outside reference execution topology.")
        if gnn_is_anomaly:
            summary_parts.append(f"GNN structural anomaly detector flagged high anomaly probability ({gnn_score:.1%}).")

        summary = " ".join(summary_parts) if summary_parts else "DAG topology consistent with expected distributed execution flow."

        return {
            "is_anomaly": is_anomaly,
            "severity": severity,
            "gnn_anomaly_score": gnn_score,
            "is_gnn_anomaly": gnn_is_anomaly,
            "has_cycles": topo_metrics["has_cycles"],
            "cycles": topo_metrics["cycles"],
            "novel_edges": structural_anom["novel_edges"],
            "summary": summary,
            "num_nodes": topo_metrics["num_nodes"],
            "num_edges": topo_metrics["num_edges"],
            "density": topo_metrics["density"],
        }

    # ------------------------------------------------------------------------ #
    #                       Mermaid Diagram & Remediation                      #
    # ------------------------------------------------------------------------ #

    def generate_mermaid_diagram(
        self,
        events: List[Union[int, str]],
        anomalous_steps: Set[int],
        max_nodes: int = 12,
    ) -> str:
        """Generate a Mermaid sequence flowchart with anomalous steps highlighted."""
        lines = ["flowchart TD"]
        sub_events = events[:max_nodes]
        for idx, ev in enumerate(sub_events):
            eid = str(ev)
            tmpl = self.resolve_template(eid)
            # Truncate template for clean Mermaid display
            short_tmpl = tmpl.split(":")[0].split("for")[0].strip()[:24]
            lines.append(f'    S{idx}["Step {idx}: {eid}<br/>{short_tmpl}"]')

        for idx in range(1, len(sub_events)):
            lines.append(f"    S{idx - 1} --> S{idx}")

        if len(events) > max_nodes:
            lines.append(f'    S{max_nodes - 1} -.-> S_more["... +{len(events) - max_nodes} more steps"]')

        # Color-code anomalous steps in red/orange
        for step in anomalous_steps:
            if step < max_nodes:
                lines.append(f"    style S{step} fill:#ffdddd,stroke:#d9534f,stroke-width:2px")

        return "\n".join(lines)

    def generate_remediation_runbook(
        self,
        semantic_cat: str,
        temporal_anom: bool,
        parameter_anom: bool,
        graph_anom: bool,
    ) -> List[Dict[str, str]]:
        """Generate actionable SRE remediation runbook items."""
        actions = []

        # Priority 1: Immediate triage & state verification
        actions.append({
            "priority": "P1 - Critical Verification",
            "action": "Inspect HDFS File & Block Integrity",
            "command": "hdfs fsck / -files -blocks -locations",
            "rationale": "Verifies whether under-replicated, corrupted, or missing block replicas exist in the filesystem.",
        })

        # Priority 2: DataNode health check
        if "packetresponder" in semantic_cat.lower() or "transmission" in semantic_cat.lower() or graph_anom:
            actions.append({
                "priority": "P2 - Pipeline Recovery",
                "action": "Diagnose DataNode Thread Health & Tail Daemon Logs",
                "command": "hdfs dfsadmin -report\njournalctl -u hadoop-datanode -n 100 --no-pager",
                "rationale": "Examines if socket thread exhaustion or OOM caused PacketResponder threads to terminate.",
            })

        # Priority 3: Network & Parameter remediation
        if temporal_anom or parameter_anom:
            actions.append({
                "priority": "P3 - Network Latency & MTU Check",
                "action": "Verify Network Connectivity and Packet Loss on DataNode Storage Interfaces",
                "command": "ip -s link show eth0\nping -c 5 -s 65507 <target_datanode_ip>",
                "rationale": "Inter-event timeouts and truncated packet sizes typically stem from jumbo frame MTU mismatches or silent packet loss.",
            })

        # Priority 4: Rebalancing & Maintenance
        actions.append({
            "priority": "P4 - Cluster Rebalancing",
            "action": "Trigger Block Replication and Disk Balancer",
            "command": "hdfs balancer -threshold 5",
            "rationale": "Redistributes block replicas evenly away from strained or failing storage nodes.",
        })

        return actions

    # ------------------------------------------------------------------------ #
    #                         Master Diagnostic Method                         #
    # ------------------------------------------------------------------------ #

    def diagnose_incident(
        self,
        block_id: str,
        events: List[Union[int, str]],
        timestamps: Optional[List[float]] = None,
        parameters: Optional[List[Dict[str, Any]]] = None,
        attention_weights: Optional[Union[List[float], torch.Tensor, np.ndarray]] = None,
        prediction_scores: Optional[Dict[str, Any]] = None,
        graph: Optional[nx.DiGraph] = None,
        gnn_result: Optional[Dict[str, Any]] = None,
        session_summary: Optional[SessionSummary] = None,
        alerts: Optional[List[EventAlert]] = None,
        reference_graph: Optional[nx.DiGraph] = None,
        temporal_result: Optional[Any] = None,
        parameter_result: Optional[Any] = None,
        attribution_result: Optional[Any] = None,
        semantic_result: Optional[Any] = None,
    ) -> IncidentReport:
        """
        Synthesize all 5 diagnostic dimensions into a comprehensive IncidentReport.

        Parameters
        ----------
        block_id : str
            Unique session / block ID.
        events : List[Union[int, str]]
            Event ID sequence.
        timestamps : List[float], optional
            Timestamps per step.
        parameters : List[Dict[str, Any]], optional
            Extracted parameters per event step.
        attention_weights : Tensor or array, optional
            Transformer attention weights over the sequence.
        prediction_scores : Dict[str, Any], optional
            DeepLog next-event prediction surprises.
        graph : nx.DiGraph, optional
            Pre-constructed execution graph.
        gnn_result : Dict[str, Any], optional
            Output from `LogGNNAnomalyDetector.predict_graph()`.
        session_summary : SessionSummary, optional
            Real-time streaming session summary.
        alerts : List[EventAlert], optional
            Raised streaming alerts.
        reference_graph : nx.DiGraph, optional
            Reference graph for topological comparison.
        temporal_result : Optional[Any]
            Output from TemporalAnomalyDetector (Step 1).
        parameter_result : Optional[Any]
            Output from ParameterAnomalyDetector (Step 2).
        attribution_result : Optional[Any]
            Output from RootCauseAnalyzer / AttentionAttributor (Step 3).
        semantic_result : Optional[Any]
            Output from TemplateSimilarityMatcher / SemanticDeepLog (Step 4).

        Returns
        -------
        report : IncidentReport
            Structured, production-grade SRE Incident Report.
        """
        incident_id = f"INC-{datetime.now().strftime('%Y%m%d%H%M%S')}-{str(block_id)[:12]}"
        now_iso = datetime.now().isoformat()

        # 1. Temporal Analysis (Step 1)
        temporal_res = self.analyze_temporal(events, timestamps=timestamps)
        if temporal_result is not None:
            t_items = temporal_result if isinstance(temporal_result, list) else [temporal_result]
            for tr in t_items:
                if getattr(tr, "is_anomaly", False):
                    temporal_res["is_anomaly"] = True
                    if getattr(tr, "is_timeout", False) or getattr(tr, "is_latency_spike", False):
                        temporal_res["severity"] = "HIGH"
                    anom_types = getattr(tr, "anomaly_types", [])
                    obs_d = getattr(tr, "observed_delta", None)
                    z_sc = getattr(tr, "z_score", None)
                    extra_msg = f" Temporal detector identified: {', '.join(anom_types)}"
                    if obs_d is not None and z_sc is not None:
                        extra_msg += f" (delta={obs_d:.2f}s, z={z_sc:.2f})"
                    temporal_res["summary"] += extra_msg

        # 2. Parameter Analysis (Step 2)
        parameter_res = self.analyze_parameters(events, parameters=parameters)
        if parameter_result is not None:
            is_anom = getattr(parameter_result, "is_anomaly", None)
            if is_anom is not None and np.any(is_anom):
                parameter_res["is_anomaly"] = True
                parameter_res["severity"] = "HIGH"
                diags = getattr(parameter_result, "diagnostics", [])
                if diags:
                    parameter_res["summary"] += f" Parameter detector flagged: {'; '.join(diags[:3])}"

        # 3. Attention Attribution (Step 3)
        if attribution_result is not None:
            ranked = getattr(attribution_result, "ranked_events", None)
            if ranked:
                attention_res = []
                for rank_idx, r in enumerate(ranked[:3], 1):
                    attention_res.append({
                        "step": r.get("token_index", r.get("step", 0)),
                        "event_id": str(r.get("event_id", "")),
                        "template": self.resolve_template(r.get("event_id", "")),
                        "weight": float(r.get("attribution_score", r.get("weight", 0.0))),
                        "percentage": float(r.get("attribution_percentage", r.get("percentage", 0.0))),
                        "role": "Primary Trigger" if rank_idx == 1 else "Context Trigger",
                    })
            else:
                raw_weights = getattr(attribution_result, "attributions", None)
                attention_res = self.analyze_attention(events, attention_weights=raw_weights)
        else:
            attention_res = self.analyze_attention(events, attention_weights=attention_weights)

        # 4. Semantic Analysis (Step 4)
        semantic_res = self.analyze_semantics(events)
        if semantic_result is not None:
            matched_tmpl = getattr(semantic_result, "matched_template", None)
            if matched_tmpl:
                semantic_res["matched_template"] = matched_tmpl

        # 5. Graph Topology Analysis (Step 5)
        graph_res = self.analyze_graph_topology(
            events,
            timestamps=timestamps,
            graph=graph,
            gnn_result=gnn_result,
            reference_graph=reference_graph,
        )

        # Collect Evidence and identify anomalous steps
        evidence_list: List[AnomalyEvidence] = []
        anomalous_steps: Set[int] = set()

        if temporal_res["is_anomaly"]:
            evidence_list.append(AnomalyEvidence(
                dimension="temporal",
                severity=temporal_res["severity"],
                title="Inter-Event Latency Timeout Violation",
                description=temporal_res["summary"],
                score=min(1.0, float(temporal_res.get("total_duration", 0.0)) / 60.0),
                details=temporal_res,
            ))
            for to in temporal_res.get("timeouts", []):
                anomalous_steps.add(to["step_to"])

        if parameter_res["is_anomaly"]:
            evidence_list.append(AnomalyEvidence(
                dimension="parameter",
                severity=parameter_res["severity"],
                title="Parameter Distribution Outlier",
                description=parameter_res["summary"],
                score=0.8,
                details=parameter_res,
            ))
            for po in parameter_res.get("anomalous_parameters", []):
                anomalous_steps.add(po["step"])

        if semantic_res["has_fatal_event"]:
            evidence_list.append(AnomalyEvidence(
                dimension="semantic",
                severity=semantic_res["severity"],
                title=f"Fatal Log Signature: {semantic_res['primary_failure_event']}",
                description=f"Observed fatal event: {self.resolve_template(semantic_res['primary_failure_event'])}",
                score=1.0,
                details=semantic_res,
            ))
            anomalous_steps.add(semantic_res["primary_failure_step"])

        if graph_res["is_anomaly"]:
            evidence_list.append(AnomalyEvidence(
                dimension="graph_topology",
                severity=graph_res["severity"],
                title="Topological Execution Graph Anomaly",
                description=graph_res["summary"],
                score=float(graph_res.get("gnn_anomaly_score", 0.8)),
                details=graph_res,
            ))

        if alerts:
            for al in alerts:
                anomalous_steps.add(al.step_number)

        # Compute Composite Incident Severity
        has_critical = any(e.severity == "CRITICAL" for e in evidence_list) or graph_res["severity"] == "CRITICAL"
        has_high = any(e.severity == "HIGH" for e in evidence_list) or semantic_res["has_fatal_event"]
        has_medium = any(e.severity == "MEDIUM" for e in evidence_list) or temporal_res["is_anomaly"] or parameter_res["is_anomaly"]

        if has_critical:
            overall_severity = "CRITICAL"
        elif has_high:
            overall_severity = "HIGH"
        elif has_medium:
            overall_severity = "MEDIUM"
        else:
            overall_severity = "LOW"

        # Determine Primary Root Cause
        if semantic_res["has_fatal_event"]:
            primary_rc_event = semantic_res["primary_failure_event"]
            primary_rc_step = semantic_res["primary_failure_step"]
            category = semantic_res["failure_category"]
        elif graph_res["is_anomaly"] and graph_res["cycles"]:
            primary_rc_event = str(graph_res["cycles"][0][0])
            primary_rc_step = events.index(primary_rc_event) if primary_rc_event in events else 0
            category = "Cyclic Execution Graph Deadlock"
        elif temporal_res["timeouts"]:
            to = temporal_res["timeouts"][0]
            primary_rc_event = to["event_to"]
            primary_rc_step = to["step_to"]
            category = "Pipeline Network Timeout"
        elif parameter_res["anomalous_parameters"]:
            po = parameter_res["anomalous_parameters"][0]
            primary_rc_step = po["step"]
            primary_rc_event = str(events[primary_rc_step]) if primary_rc_step < len(events) else "UNKNOWN"
            category = "Parameter Value Corruption"
        else:
            primary_rc_event = str(events[-1]) if events else "UNKNOWN"
            primary_rc_step = len(events) - 1 if events else 0
            category = "Nominal Sequence Pattern"

        primary_rc_template = self.resolve_template(primary_rc_event)

        # Build Executive Summary
        exec_summary = (
            f"Incident {incident_id} detected for storage block {block_id}. "
            f"The primary failure domain was identified as '{category}', "
            f"initiated at step {primary_rc_step} by event {primary_rc_event} ('{primary_rc_template}'). "
            f"Multi-head attention attribution identified antecedent triggers contributing "
            f"{attention_res[0]['percentage']:.1f}% attention weight. "
            f"Overall system severity is rated {overall_severity}."
        )

        # Build Root Cause Analysis dict
        rca_dict = {
            "primary_root_cause_event": primary_rc_event,
            "primary_root_cause_step": primary_rc_step,
            "primary_root_cause_template": primary_rc_template,
            "failure_category": category,
            "confidence_score": 0.96 if overall_severity in ("CRITICAL", "HIGH") else 0.82,
            "causal_chain": (
                f"Execution proceeded normally until step {primary_rc_step}, where {primary_rc_event} triggered an abnormal termination. "
                f"Graph analysis reported GNN anomaly score of {graph_res.get('gnn_anomaly_score', 0.0):.2f} "
                f"with {len(graph_res.get('cycles', []))} cycle(s)."
            ),
            "attention_attribution": attention_res,
        }

        # Build Anomaly Breakdown
        seq_is_anomaly = bool(alerts or (prediction_scores and prediction_scores.get("is_anomaly", False)))
        anomaly_breakdown = {
            "temporal": temporal_res,
            "parameter": parameter_res,
            "sequence": {
                "is_anomaly": seq_is_anomaly,
                "severity": "HIGH" if seq_is_anomaly else "LOW",
                "summary": (
                    f"Sequence model flagged {len(alerts)} transition violation(s)."
                    if alerts
                    else "Sequential transitions conform to DeepLog language model predictions."
                ),
            },
            "graph_topology": graph_res,
        }

        # Build Chronological Timeline
        timeline: List[TimelineItem] = []
        for step, ev in enumerate(events):
            eid = str(ev)
            tmpl = self.resolve_template(eid)
            is_anom = step in anomalous_steps
            ts = timestamps[step] if timestamps and step < len(timestamps) else None
            lat_ms = (
                (timestamps[step] - timestamps[step - 1]) * 1000.0
                if timestamps and step > 0 and step < len(timestamps)
                else None
            )
            param_dict = parameters[step] if parameters and step < len(parameters) else {}
            attn_val = (
                attention_res[0]["weight"] if attention_res and attention_res[0]["step"] == step else 0.0
            )

            anom_dim = None
            if is_anom:
                if step == primary_rc_step:
                    anom_dim = "ROOT_CAUSE"
                elif any(to["step_to"] == step for to in temporal_res.get("timeouts", [])):
                    anom_dim = "TEMPORAL"
                elif any(po["step"] == step for po in parameter_res.get("anomalous_parameters", [])):
                    anom_dim = "PARAMETER"
                else:
                    anom_dim = "SEQUENCE"

            timeline.append(TimelineItem(
                step=step,
                timestamp=ts,
                event_id=eid,
                template=tmpl,
                is_anomaly=is_anom,
                anomaly_dimension=anom_dim,
                attention_weight=attn_val,
                latency_ms=lat_ms,
                parameters=param_dict,
                notes="Anomalous transition" if is_anom else "",
            ))

        # Build Impact Analysis
        impact_analysis = {
            "affected_components": ["HDFS DataNode", "HDFS NameNode", "Client I/O Pipeline"],
            "data_durability_risk": "High (Block under-replicated)" if overall_severity in ("CRITICAL", "HIGH") else "None / Minor",
            "replica_integrity": "Block replication interrupted; mirror pipeline aborted." if overall_severity in ("CRITICAL", "HIGH") else "Nominal replica count.",
            "sli_impact": "Write operation failed, triggering client-side retry escalation." if overall_severity in ("CRITICAL", "HIGH") else "P99 within nominal bounds.",
        }

        # Build Remediation Runbook
        remediations = self.generate_remediation_runbook(
            semantic_cat=category,
            temporal_anom=temporal_res["is_anomaly"],
            parameter_anom=parameter_res["is_anomaly"],
            graph_anom=graph_res["is_anomaly"],
        )

        # Build Mermaid Diagram
        mermaid = self.generate_mermaid_diagram(events, anomalous_steps)

        return IncidentReport(
            incident_id=incident_id,
            block_id=str(block_id),
            severity=overall_severity,
            status="TRIGGERED" if overall_severity in ("CRITICAL", "HIGH") else "RESOLVED",
            timestamp=now_iso,
            executive_summary=exec_summary,
            root_cause_analysis=rca_dict,
            anomaly_breakdown=anomaly_breakdown,
            timeline=timeline,
            impact_analysis=impact_analysis,
            remediation_suggestions=remediations,
            evidence_list=evidence_list,
            mermaid_diagram=mermaid,
            metadata={
                "total_events": len(events),
                "total_anomalies": len(anomalous_steps),
                "diagnostic_engine": "DeepLog-GenAI-Summarizer-v1.0",
            },
        )

    def generate_genai_prompt(self, report: IncidentReport) -> str:
        """
        Format a comprehensive GenAI prompt containing all telemetry evidence
        for downstream LLM automated incident triage or multi-agent orchestration.
        """
        prompt = f"""You are a Principal SRE and Distributed Systems Incident Commander.
Analyze the following DeepLog Next-Gen automated telemetry report and provide:
1. Executive Incident Summary
2. Detailed Causal Chain of Events
3. Immediate Mitigation Steps
4. Long-Term Preventative Engineering Recommendations

### Incident Telemetry Context:
- Incident ID: {report.incident_id}
- Target Block / Session: {report.block_id}
- Severity: {report.severity}
- Primary Root Cause Event: {report.root_cause_analysis.get('primary_root_cause_event')} ({report.root_cause_analysis.get('primary_root_cause_template')})
- Failure Domain: {report.root_cause_analysis.get('failure_category')}
- Diagnostic Confidence: {report.root_cause_analysis.get('confidence_score')}

### Multi-Dimensional Anomaly Evidence:
{json.dumps(report.anomaly_breakdown, indent=2)}

### Top Attention-Weighted Antecedents:
{json.dumps(report.root_cause_analysis.get('attention_attribution'), indent=2)}

Please generate a professional, actionable incident diagnosis.
"""
        return prompt
