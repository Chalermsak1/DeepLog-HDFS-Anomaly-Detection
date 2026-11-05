"""Kubernetes Manifests, CRD, and Cluster Management for DeepLog-Omni.

This module provides production-grade Kubernetes tooling for DeepLog:
- CustomResourceDefinition (CRD) validation and programmatic generation for
  DeepLogCluster and DeepLogPolicy.
- eBPF DaemonSet inspection and validation (hostPID, privileged securityContext,
  bpf mounts).
- Helm chart parameter verification and template rendering helper.
- Status inspection, health evaluation, and autonomous reconciliation logic.
- Pure-Python zero-dependency YAML parser and serializer with optional PyYAML
  acceleration.
"""

from __future__ import annotations

import copy
import datetime
import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

API_GROUP = "deeplog.ai"
API_VERSION = "v1alpha1"
FULL_API_VERSION = f"{API_GROUP}/{API_VERSION}"

CLUSTER_KIND = "DeepLogCluster"
POLICY_KIND = "DeepLogPolicy"

VALID_DETECTION_MODES: Set[str] = {
    "native-ebpf",
    "semantic",
    "physics",
    "neuro-symbolic",
    "ensemble",
}

VALID_ACTIONS: Set[str] = {
    "Alert",
    "Quarantine",
    "Remediate",
    "Block",
    "Evict",
}

VALID_PHASES: Set[str] = {
    "Pending",
    "Initializing",
    "Running",
    "Degraded",
    "Failed",
}


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class ManifestValidationResult:
    """Result of validating a Kubernetes manifest."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    resource_kind: Optional[str] = None
    resource_name: Optional[str] = None
    api_version: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ClusterStatusInspection:
    """Status evaluation and health metrics for a DeepLogCluster instance."""
    phase: str
    health_percentage: float
    ready_replicas: int
    desired_replicas: int
    ebpf_nodes_active: int
    anomalies_detected: int
    is_healthy: bool
    conditions_summary: Dict[str, str] = field(default_factory=dict)
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PolicyEvaluationResult:
    """Evaluation result for DeepLogPolicy threshold checking."""
    triggered: bool
    action: str
    anomaly_score: float
    threshold: float
    consecutive_violations: int
    required_consecutive: int
    message: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Pure-Python Robust YAML Parser & Serializer
# ---------------------------------------------------------------------------

def _strip_yaml_comment(line: str) -> str:
    """Strip comment while preserving strings containing hash symbols."""
    in_quote = False
    quote_char = ""
    for i, ch in enumerate(line):
        if ch in ('"', "'") and (i == 0 or line[i - 1] != "\\"):
            if not in_quote:
                in_quote = True
                quote_char = ch
            elif quote_char == ch:
                in_quote = False
        elif ch == "#" and not in_quote:
            return line[:i]
    return line


def _parse_flow_sequence(s: str) -> Optional[List[Any]]:
    s = s.strip()
    if not (s.startswith("[") and s.endswith("]")):
        return None
    inner = s[1:-1].strip()
    if not inner:
        return []
    items: List[str] = []
    current: List[str] = []
    in_quote = False
    quote_char = ""
    bracket_depth = 0
    brace_depth = 0
    for ch in inner:
        if ch in ('"', "'") and (not in_quote or quote_char == ch):
            in_quote = not in_quote
            quote_char = ch if in_quote else ""
            current.append(ch)
        elif not in_quote:
            if ch == "[":
                bracket_depth += 1
                current.append(ch)
            elif ch == "]":
                bracket_depth -= 1
                current.append(ch)
            elif ch == "{":
                brace_depth += 1
                current.append(ch)
            elif ch == "}":
                brace_depth -= 1
                current.append(ch)
            elif ch == "," and bracket_depth == 0 and brace_depth == 0:
                items.append("".join(current).strip())
                current = []
            else:
                current.append(ch)
        else:
            current.append(ch)
    if current:
        items.append("".join(current).strip())
    return [_parse_yaml_scalar(x) for x in items]


def _parse_flow_mapping(s: str) -> Optional[Dict[str, Any]]:
    s = s.strip()
    if not (s.startswith("{") and s.endswith("}")):
        return None
    inner = s[1:-1].strip()
    if not inner:
        return {}
    parts: List[str] = []
    current: List[str] = []
    in_quote = False
    quote_char = ""
    bracket_depth = 0
    brace_depth = 0
    for ch in inner:
        if ch in ('"', "'") and (not in_quote or quote_char == ch):
            in_quote = not in_quote
            quote_char = ch if in_quote else ""
            current.append(ch)
        elif not in_quote:
            if ch == "{":
                brace_depth += 1
                current.append(ch)
            elif ch == "}":
                brace_depth -= 1
                current.append(ch)
            elif ch == "[":
                bracket_depth += 1
                current.append(ch)
            elif ch == "]":
                bracket_depth -= 1
                current.append(ch)
            elif ch == "," and bracket_depth == 0 and brace_depth == 0:
                parts.append("".join(current).strip())
                current = []
            else:
                current.append(ch)
        else:
            current.append(ch)
    if current:
        parts.append("".join(current).strip())

    result: Dict[str, Any] = {}
    for part in parts:
        if ":" in part:
            idx = part.index(":")
            k = part[:idx].strip()
            v = part[idx + 1:].strip()
            result[str(_parse_yaml_scalar(k))] = _parse_yaml_scalar(v)
    return result


def _parse_yaml_scalar(val: str) -> Any:
    """Parse a single YAML scalar value into appropriate Python primitive."""
    val = val.strip()
    if val == "" or val == "~" or val.lower() == "null":
        return None
    if val.lower() == "true":
        return True
    if val.lower() == "false":
        return False
    if val.startswith("[") and val.endswith("]"):
        seq = _parse_flow_sequence(val)
        if seq is not None:
            return seq
    if val.startswith("{") and val.endswith("}"):
        mp = _parse_flow_mapping(val)
        if mp is not None:
            return mp
    if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
        inner = val[1:-1]
        # Basic unescape
        inner = inner.replace('\\"', '"').replace("\\'", "'").replace("\\n", "\n").replace("\\t", "\t")
        return inner
    if re.match(r"^-?[0-9]+$", val):
        try:
            return int(val)
        except ValueError:
            pass
    if re.match(r"^-?[0-9]+\.[0-9]+([eE][-+]?[0-9]+)?$", val):
        try:
            return float(val)
        except ValueError:
            pass
    return val


def _find_colon_separator(s: str) -> int:
    """Find position of key: value separator colon, ignoring colons in quotes."""
    in_quote = False
    quote_char = ""
    for i, ch in enumerate(s):
        if ch in ('"', "'") and (i == 0 or s[i - 1] != "\\"):
            if not in_quote:
                in_quote = True
                quote_char = ch
            elif quote_char == ch:
                in_quote = False
        elif ch == ":" and not in_quote:
            if i + 1 == len(s) or s[i + 1] in (" ", "\t", "\r", "\n"):
                return i
    return -1


class _InternalYamlParser:
    """Pure-Python YAML parser supporting multi-doc Kubernetes manifests."""

    def __init__(self, text: str) -> None:
        self.raw_lines = text.splitlines()

    def parse_documents(self) -> List[Any]:
        docs: List[Any] = []
        doc_lines: List[str] = []
        for line in self.raw_lines:
            stripped = line.strip()
            if stripped in ("---", "..."):
                if doc_lines:
                    doc = self._parse_block(doc_lines)
                    if doc is not None:
                        docs.append(doc)
                    doc_lines = []
            else:
                doc_lines.append(line)
        if doc_lines:
            doc = self._parse_block(doc_lines)
            if doc is not None:
                docs.append(doc)
        return docs

    def _parse_block(self, lines: List[str]) -> Any:
        filtered: List[Tuple[int, str, str]] = []
        for ln in lines:
            clean = _strip_yaml_comment(ln)
            if clean.strip() != "":
                indent = len(clean) - len(clean.lstrip(" "))
                filtered.append((indent, clean.strip(), ln))
        if not filtered:
            return None
        parsed, _ = self._parse_nodes(filtered, 0, filtered[0][0])
        return parsed

    def _parse_nodes(
        self, lines: List[Tuple[int, str, str]], idx: int, current_indent: int
    ) -> Tuple[Any, int]:
        if idx >= len(lines):
            return None, idx
        indent, text, raw = lines[idx]
        if text.startswith("- ") or text == "-":
            return self._parse_list(lines, idx, current_indent)
        else:
            return self._parse_dict(lines, idx, current_indent)

    def _parse_list(
        self, lines: List[Tuple[int, str, str]], idx: int, list_indent: int
    ) -> Tuple[List[Any], int]:
        res: List[Any] = []
        while idx < len(lines):
            indent, text, raw = lines[idx]
            if indent < list_indent:
                break
            if indent > list_indent and not (text.startswith("- ") or text == "-"):
                break
            if text.startswith("- ") or text == "-":
                rem = text[2:].strip() if text.startswith("- ") else ""
                idx += 1
                if not rem:
                    if idx < len(lines) and lines[idx][0] > indent:
                        child, idx = self._parse_nodes(lines, idx, lines[idx][0])
                        res.append(child)
                    else:
                        res.append(None)
                elif ":" in rem and not rem.startswith("{") and not (
                    (rem.startswith('"') and rem.endswith('"'))
                    or (rem.startswith("'") and rem.endswith("'"))
                ):
                    colon_pos = _find_colon_separator(rem)
                    if colon_pos != -1:
                        k = rem[:colon_pos].strip()
                        v = rem[colon_pos + 1:].strip()
                        dict_item: Dict[str, Any] = {}
                        content_indent = indent + 2
                        if v in ("|", "|-", ">", ">-"):
                            block_str, idx = self._read_multiline(lines, idx, content_indent)
                            dict_item[str(_parse_yaml_scalar(k))] = block_str
                        elif v != "":
                            dict_item[str(_parse_yaml_scalar(k))] = _parse_yaml_scalar(v)
                        else:
                            if idx < len(lines) and lines[idx][0] > indent:
                                child, idx = self._parse_nodes(lines, idx, lines[idx][0])
                                dict_item[str(_parse_yaml_scalar(k))] = child
                            else:
                                dict_item[str(_parse_yaml_scalar(k))] = None
                        while idx < len(lines):
                            next_indent, next_text, next_raw = lines[idx]
                            if next_indent < content_indent or next_text.startswith("- ") or next_text == "-":
                                break
                            c_pos = _find_colon_separator(next_text)
                            if c_pos != -1:
                                sub_k = next_text[:c_pos].strip()
                                sub_v = next_text[c_pos + 1:].strip()
                                idx += 1
                                if sub_v in ("|", "|-", ">", ">-"):
                                    b_str, idx = self._read_multiline(lines, idx, next_indent + 1)
                                    dict_item[str(_parse_yaml_scalar(sub_k))] = b_str
                                elif sub_v != "":
                                    dict_item[str(_parse_yaml_scalar(sub_k))] = _parse_yaml_scalar(sub_v)
                                else:
                                    if idx < len(lines) and lines[idx][0] > next_indent:
                                        child, idx = self._parse_nodes(lines, idx, lines[idx][0])
                                        dict_item[str(_parse_yaml_scalar(sub_k))] = child
                                    else:
                                        dict_item[str(_parse_yaml_scalar(sub_k))] = None
                            else:
                                break
                        res.append(dict_item)
                    else:
                        res.append(_parse_yaml_scalar(rem))
                else:
                    res.append(_parse_yaml_scalar(rem))
            else:
                break
        return res, idx

    def _parse_dict(
        self, lines: List[Tuple[int, str, str]], idx: int, dict_indent: int
    ) -> Tuple[Dict[str, Any], int]:
        res: Dict[str, Any] = {}
        while idx < len(lines):
            indent, text, raw = lines[idx]
            if indent < dict_indent:
                break
            if text.startswith("- ") or text == "-":
                break
            colon_pos = _find_colon_separator(text)
            if colon_pos == -1:
                idx += 1
                continue
            k = text[:colon_pos].strip()
            v = text[colon_pos + 1:].strip()
            idx += 1
            if v in ("|", "|-", ">", ">-"):
                b_str, idx = self._read_multiline(lines, idx, indent + 1)
                res[str(_parse_yaml_scalar(k))] = b_str
            elif v != "":
                res[str(_parse_yaml_scalar(k))] = _parse_yaml_scalar(v)
            else:
                if idx < len(lines) and lines[idx][0] > indent:
                    child, idx = self._parse_nodes(lines, idx, lines[idx][0])
                    res[str(_parse_yaml_scalar(k))] = child
                else:
                    res[str(_parse_yaml_scalar(k))] = None
        return res, idx

    def _read_multiline(
        self, lines: List[Tuple[int, str, str]], idx: int, min_indent: int
    ) -> Tuple[str, int]:
        collected: List[str] = []
        base_ind: Optional[int] = None
        while idx < len(lines):
            indent, text, raw = lines[idx]
            if indent < min_indent:
                break
            if base_ind is None:
                base_ind = indent
            strip_len = min(base_ind, len(raw) - len(raw.lstrip(" ")))
            collected.append(raw[strip_len:].rstrip("\r\n"))
            idx += 1
        return "\n".join(collected), idx


def _format_yaml_scalar(v: Any) -> str:
    """Format scalar for YAML output."""
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    s = str(v)
    if "\n" in s:
        sub = "\n  ".join(s.splitlines())
        return f"|\n  {sub}"
    special_chars = {":", "{", "}", "[", "]", ",", "&", "*", "#", "?", "|", "-", "<", ">", "=", "!", "%", "@", "`"}
    if any(c in s for c in special_chars) or s.lower() in ("true", "false", "null", "yes", "no"):
        escaped = s.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return s


def _dump_yaml_node(data: Any, indent: int = 0) -> str:
    """Recursively dump Python structure to YAML."""
    lines: List[str] = []
    prefix = " " * indent
    if isinstance(data, dict):
        if not data:
            return f"{prefix}{{}}"
        for k, v in data.items():
            if isinstance(v, dict):
                if not v:
                    lines.append(f"{prefix}{k}: {{}}")
                else:
                    lines.append(f"{prefix}{k}:")
                    lines.append(_dump_yaml_node(v, indent + 2))
            elif isinstance(v, list):
                if not v:
                    lines.append(f"{prefix}{k}: []")
                else:
                    lines.append(f"{prefix}{k}:")
                    lines.append(_dump_yaml_node(v, indent + 2))
            else:
                lines.append(f"{prefix}{k}: {_format_yaml_scalar(v)}")
    elif isinstance(data, list):
        if not data:
            return f"{prefix}[]"
        for item in data:
            if isinstance(item, dict):
                if not item:
                    lines.append(f"{prefix}- {{}}")
                else:
                    first = True
                    for k, v in item.items():
                        if first:
                            if isinstance(v, dict):
                                lines.append(f"{prefix}- {k}:")
                                lines.append(_dump_yaml_node(v, indent + 4))
                            elif isinstance(v, list):
                                lines.append(f"{prefix}- {k}:")
                                lines.append(_dump_yaml_node(v, indent + 4))
                            else:
                                lines.append(f"{prefix}- {k}: {_format_yaml_scalar(v)}")
                            first = False
                        else:
                            if isinstance(v, dict):
                                lines.append(f"{prefix}  {k}:")
                                lines.append(_dump_yaml_node(v, indent + 4))
                            elif isinstance(v, list):
                                lines.append(f"{prefix}  {k}:")
                                lines.append(_dump_yaml_node(v, indent + 4))
                            else:
                                lines.append(f"{prefix}  {k}: {_format_yaml_scalar(v)}")
            elif isinstance(item, list):
                lines.append(f"{prefix}-")
                lines.append(_dump_yaml_node(item, indent + 2))
            else:
                lines.append(f"{prefix}- {_format_yaml_scalar(item)}")
    else:
        lines.append(f"{prefix}{_format_yaml_scalar(data)}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# K8sManager Implementation
# ---------------------------------------------------------------------------

class K8sManager:
    """Enterprise Kubernetes & Platform Manager for DeepLog.

    Provides end-to-end tooling to inspect, validate, programmatically synthesize,
    and reconcile DeepLog Kubernetes deployment manifests, Custom Resource Definitions,
    Helm templates, and eBPF kernel telemetry DaemonSets.
    """

    def __init__(self, workspace_root: Optional[Union[str, Path]] = None) -> None:
        if workspace_root is not None:
            self.root = Path(workspace_root)
        else:
            self.root = Path(os.getcwd())

        self.crd_path = self.root / "deploy" / "k8s" / "crd" / "deeplog_crd.yaml"
        self.daemonset_path = self.root / "deploy" / "k8s" / "daemonset" / "deeplog_ebpf_daemonset.yaml"
        self.helm_dir = self.root / "deploy" / "k8s" / "helm"

    # -----------------------------------------------------------------------
    # YAML I/O
    # -----------------------------------------------------------------------

    def parse_yaml(self, content: str) -> List[Dict[str, Any]]:
        """Parse a YAML string (potentially containing multiple documents)."""
        # Try PyYAML if installed, otherwise use pure-Python implementation
        try:
            import yaml  # type: ignore
            docs = list(yaml.safe_load_all(content))
            return [d for d in docs if d is not None]
        except (ImportError, Exception):
            parser = _InternalYamlParser(content)
            docs = parser.parse_documents()
            return [d for d in docs if isinstance(d, dict)]

    def dump_yaml(self, obj: Union[Dict[str, Any], Sequence[Dict[str, Any]]]) -> str:
        """Serialize a dictionary or list of dictionaries to YAML."""
        try:
            import yaml  # type: ignore
            if isinstance(obj, (list, tuple)):
                return yaml.safe_dump_all(obj, sort_keys=False)
            return yaml.safe_dump(obj, sort_keys=False)
        except (ImportError, Exception):
            if isinstance(obj, (list, tuple)):
                docs = [_dump_yaml_node(d) for d in obj if d is not None]
                return "\n---\n".join(docs) + "\n"
            return _dump_yaml_node(obj) + "\n"

    def load_manifest(self, file_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """Load and parse all Kubernetes documents from a file."""
        path = Path(file_path)
        if not path.is_absolute():
            path = self.root / path
        if not path.exists():
            raise FileNotFoundError(f"Manifest file not found: {path}")
        content = path.read_text(encoding="utf-8")
        return self.parse_yaml(content)

    def export_manifest(
        self,
        manifest: Union[Dict[str, Any], Sequence[Dict[str, Any]]],
        output_path: Union[str, Path],
    ) -> Path:
        """Export manifest(s) to a file on disk."""
        target = Path(output_path)
        if not target.is_absolute():
            target = self.root / target
        target.parent.mkdir(parents=True, exist_ok=True)
        content = self.dump_yaml(manifest)
        target.write_text(content, encoding="utf-8")
        return target

    # -----------------------------------------------------------------------
    # Manifest Validation
    # -----------------------------------------------------------------------

    def validate_manifest(self, doc: Dict[str, Any]) -> ManifestValidationResult:
        """Validate a single Kubernetes resource manifest dictionary."""
        errors: List[str] = []
        warnings: List[str] = []

        if not isinstance(doc, dict):
            return ManifestValidationResult(
                is_valid=False,
                errors=["Manifest is not a valid dictionary object."],
            )

        api_version = doc.get("apiVersion")
        kind = doc.get("kind")
        metadata = doc.get("metadata", {})
        name = metadata.get("name") if isinstance(metadata, dict) else None

        # Basic Kubernetes checks
        if not api_version or not isinstance(api_version, str):
            errors.append("Missing or invalid 'apiVersion'.")
        if not kind or not isinstance(kind, str):
            errors.append("Missing or invalid 'kind'.")
        if not isinstance(metadata, dict):
            errors.append("'metadata' must be a dictionary.")
        elif not name or not isinstance(name, str):
            errors.append("Missing or invalid 'metadata.name'.")

        # Specific Kind Validations
        if kind == "CustomResourceDefinition":
            self._validate_crd(doc, errors, warnings)
        elif kind == CLUSTER_KIND:
            self._validate_deeplog_cluster(doc, errors, warnings)
        elif kind == POLICY_KIND:
            self._validate_deeplog_policy(doc, errors, warnings)
        elif kind == "DaemonSet":
            self._validate_daemonset(doc, errors, warnings)
        elif kind == "Deployment":
            self._validate_deployment(doc, errors, warnings)

        return ManifestValidationResult(
            is_valid=(len(errors) == 0),
            errors=errors,
            warnings=warnings,
            resource_kind=kind,
            resource_name=name,
            api_version=api_version,
        )

    def _validate_crd(self, doc: Dict[str, Any], errors: List[str], warnings: List[str]) -> None:
        spec = doc.get("spec", {})
        if not isinstance(spec, dict):
            errors.append("CRD 'spec' must be a dictionary.")
            return

        group = spec.get("group")
        if not group or not isinstance(group, str):
            errors.append("CRD spec missing 'group'.")
        elif group != API_GROUP:
            warnings.append(f"CRD group '{group}' differs from default '{API_GROUP}'.")

        names = spec.get("names", {})
        if not isinstance(names, dict) or not names.get("kind") or not names.get("plural"):
            errors.append("CRD spec.names must specify 'kind' and 'plural'.")

        versions = spec.get("versions")
        if not isinstance(versions, list) or len(versions) == 0:
            errors.append("CRD spec.versions must be a non-empty list.")
        else:
            has_storage = any(v.get("storage") is True for v in versions if isinstance(v, dict))
            if not has_storage:
                errors.append("CRD must define at least one version with 'storage: true'.")

    def _validate_deeplog_cluster(self, doc: Dict[str, Any], errors: List[str], warnings: List[str]) -> None:
        api_version = doc.get("apiVersion", "")
        if not api_version.startswith(API_GROUP):
            warnings.append(f"DeepLogCluster apiVersion '{api_version}' does not match expected group '{API_GROUP}'.")

        spec = doc.get("spec", {})
        if not isinstance(spec, dict):
            errors.append("DeepLogCluster 'spec' must be a dictionary.")
            return

        replicas = spec.get("replicas")
        if replicas is None:
            errors.append("DeepLogCluster spec missing required field 'replicas'.")
        elif not isinstance(replicas, int) or replicas < 1:
            errors.append(f"DeepLogCluster replicas must be positive integer, got: {replicas}")

        ebpf_probe = spec.get("ebpfProbe")
        if ebpf_probe is not None:
            if not isinstance(ebpf_probe, dict):
                errors.append("DeepLogCluster spec.ebpfProbe must be a dictionary.")
            else:
                sampling = ebpf_probe.get("samplingRate")
                if sampling is not None and (not isinstance(sampling, (int, float)) or sampling < 0.0 or sampling > 1.0):
                    errors.append(f"DeepLogCluster ebpfProbe.samplingRate must be between 0.0 and 1.0, got: {sampling}")

    def _validate_deeplog_policy(self, doc: Dict[str, Any], errors: List[str], warnings: List[str]) -> None:
        spec = doc.get("spec", {})
        if not isinstance(spec, dict):
            errors.append("DeepLogPolicy 'spec' must be a dictionary.")
            return

        mode = spec.get("detectionMode")
        if not mode:
            errors.append("DeepLogPolicy spec missing required field 'detectionMode'.")
        elif mode not in VALID_DETECTION_MODES:
            errors.append(f"Invalid detectionMode '{mode}'. Valid modes: {sorted(VALID_DETECTION_MODES)}")

        action = spec.get("action")
        if not action:
            errors.append("DeepLogPolicy spec missing required field 'action'.")
        elif action not in VALID_ACTIONS:
            errors.append(f"Invalid action '{action}'. Valid actions: {sorted(VALID_ACTIONS)}")

        threshold = spec.get("anomalyThreshold")
        if threshold is not None:
            if not isinstance(threshold, (int, float)) or threshold < 0.0 or threshold > 1.0:
                errors.append(f"anomalyThreshold must be between 0.0 and 1.0, got: {threshold}")

    def _validate_daemonset(self, doc: Dict[str, Any], errors: List[str], warnings: List[str]) -> None:
        spec = doc.get("spec", {})
        template = spec.get("template", {}) if isinstance(spec, dict) else {}
        pod_spec = template.get("spec", {}) if isinstance(template, dict) else {}

        # Check eBPF specific requirements
        name = doc.get("metadata", {}).get("name", "")
        if "ebpf" in name.lower():
            if not pod_spec.get("hostPID"):
                errors.append("eBPF DaemonSet must have 'spec.template.spec.hostPID: true'.")

            containers = pod_spec.get("containers", [])
            if not isinstance(containers, list) or len(containers) == 0:
                errors.append("DaemonSet must define at least one container.")
            else:
                probe_container = containers[0]
                sec_ctx = probe_container.get("securityContext", {})
                if not sec_ctx.get("privileged"):
                    errors.append("eBPF DaemonSet container must have 'securityContext.privileged: true'.")

                # Check for /sys/fs/bpf volume mount
                v_mounts = probe_container.get("volumeMounts", [])
                has_bpf = any(
                    isinstance(m, dict) and m.get("mountPath") == "/sys/fs/bpf"
                    for m in v_mounts
                )
                if not has_bpf:
                    warnings.append("eBPF DaemonSet container does not mount '/sys/fs/bpf'.")

    def _validate_deployment(self, doc: Dict[str, Any], errors: List[str], warnings: List[str]) -> None:
        spec = doc.get("spec", {})
        if not isinstance(spec, dict):
            errors.append("Deployment 'spec' must be a dictionary.")
            return
        if not spec.get("selector"):
            errors.append("Deployment spec missing 'selector'.")
        if not spec.get("template"):
            errors.append("Deployment spec missing 'template'.")

    def validate_file(self, file_path: Union[str, Path]) -> List[ManifestValidationResult]:
        """Validate all documents inside a YAML file."""
        docs = self.load_manifest(file_path)
        return [self.validate_manifest(d) for d in docs]

    def validate_all(self, directory: Optional[Union[str, Path]] = None) -> Dict[str, List[ManifestValidationResult]]:
        """Validate all YAML manifests in the specified directory or deploy/k8s/."""
        scan_dir = Path(directory) if directory else (self.root / "deploy" / "k8s")
        if not scan_dir.is_absolute():
            scan_dir = self.root / scan_dir

        results: Dict[str, List[ManifestValidationResult]] = {}
        for yml_path in scan_dir.rglob("*.yaml"):
            # Skip Helm templates with raw Go expressions if they can't be parsed directly
            if "templates" in yml_path.parts:
                continue
            rel = str(yml_path.relative_to(self.root))
            results[rel] = self.validate_file(yml_path)
        return results

    # -----------------------------------------------------------------------
    # Programmatic Generation
    # -----------------------------------------------------------------------

    def generate_deeplog_cluster_cr(
        self,
        name: str,
        namespace: str = "default",
        replicas: int = 3,
        image: str = "deeplog/engine:latest",
        model_path: str = "/models/deeplog_model_full.pt",
        ebpf_enabled: bool = True,
        bpf_mount_path: str = "/sys/fs/bpf",
        ring_buffer_size_mb: int = 64,
        sampling_rate: float = 1.0,
        resources: Optional[Dict[str, Any]] = None,
        federation_enabled: bool = False,
        mesh_port: int = 9443,
        safety_filter_enabled: bool = True,
        max_oscillation_frequency: float = 0.5,
        labels: Optional[Dict[str, str]] = None,
        annotations: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Programmatically generate a DeepLogCluster Custom Resource instance."""
        cluster_labels = {
            "app.kubernetes.io/name": "deeplog",
            "app.kubernetes.io/component": "cluster",
            "app.kubernetes.io/managed-by": "deeplog-k8s-manager",
        }
        if labels:
            cluster_labels.update(labels)

        default_resources = {
            "requests": {"cpu": "500m", "memory": "1Gi"},
            "limits": {"cpu": "2000m", "memory": "4Gi"},
        }

        cr = {
            "apiVersion": FULL_API_VERSION,
            "kind": CLUSTER_KIND,
            "metadata": {
                "name": name,
                "namespace": namespace,
                "labels": cluster_labels,
            },
            "spec": {
                "replicas": replicas,
                "image": image,
                "imagePullPolicy": "IfNotPresent",
                "modelPath": model_path,
                "ebpfProbe": {
                    "enabled": ebpf_enabled,
                    "bpfMountPath": bpf_mount_path,
                    "ringBufferSizeMb": ring_buffer_size_mb,
                    "samplingRate": sampling_rate,
                    "dropNormalEvents": True,
                },
                "resources": resources or default_resources,
                "federation": {
                    "enabled": federation_enabled,
                    "meshPort": mesh_port,
                    "privacyBudgetEpsilon": 1.0,
                    "peers": [],
                },
                "safetyFilter": {
                    "enabled": safety_filter_enabled,
                    "maxOscillationFrequency": max_oscillation_frequency,
                    "barrierDecayGamma": 0.1,
                },
            },
            "status": {
                "phase": "Pending",
                "readyReplicas": 0,
                "totalReplicas": replicas,
                "ebpfNodesActive": 0,
                "anomaliesDetected": 0,
                "conditions": [
                    {
                        "type": "Initialized",
                        "status": "True",
                        "lastTransitionTime": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        "reason": "ClusterCreated",
                        "message": f"DeepLogCluster '{name}' defined with {replicas} desired replicas.",
                    }
                ],
            },
        }
        if annotations:
            cr["metadata"]["annotations"] = annotations
        return cr

    def generate_deeplog_policy_cr(
        self,
        name: str,
        namespace: str = "default",
        detection_mode: str = "ensemble",
        action: str = "Alert",
        anomaly_threshold: float = 0.85,
        consecutive_violations: int = 3,
        target_namespaces: Optional[List[str]] = None,
        target_labels: Optional[Dict[str, str]] = None,
        rules: Optional[List[Dict[str, Any]]] = None,
        alert_channels: Optional[List[Dict[str, Any]]] = None,
        labels: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Programmatically generate a DeepLogPolicy Custom Resource instance."""
        if detection_mode not in VALID_DETECTION_MODES:
            raise ValueError(f"Invalid detectionMode '{detection_mode}'. Expected one of {VALID_DETECTION_MODES}")
        if action not in VALID_ACTIONS:
            raise ValueError(f"Invalid action '{action}'. Expected one of {VALID_ACTIONS}")

        policy_labels = {
            "app.kubernetes.io/name": "deeplog",
            "app.kubernetes.io/component": "policy",
        }
        if labels:
            policy_labels.update(labels)

        target_workloads: Dict[str, Any] = {}
        if target_namespaces:
            target_workloads["namespaces"] = target_namespaces
        if target_labels:
            target_workloads["matchLabels"] = target_labels

        cr = {
            "apiVersion": FULL_API_VERSION,
            "kind": POLICY_KIND,
            "metadata": {
                "name": name,
                "namespace": namespace,
                "labels": policy_labels,
            },
            "spec": {
                "detectionMode": detection_mode,
                "action": action,
                "anomalyThreshold": float(anomaly_threshold),
                "consecutiveViolations": int(consecutive_violations),
                "targetWorkloads": target_workloads,
                "rules": rules or [
                    {
                        "name": "HighAnomalyScore",
                        "condition": f"score > {anomaly_threshold}",
                        "severity": "High",
                    }
                ],
                "alertChannels": alert_channels or [
                    {
                        "type": "webhook",
                        "endpoint": "http://alert-manager.monitoring.svc:9093",
                    }
                ],
            },
            "status": {
                "active": True,
                "matchedPods": 0,
                "violations": 0,
                "lastTriggered": "",
            },
        }
        return cr

    def generate_ebpf_daemonset(
        self,
        name: str = "deeplog-ebpf-probe",
        namespace: str = "deeplog-system",
        image: str = "deeplog/ebpf-probe:latest",
        bpf_mount_path: str = "/sys/fs/bpf",
        ring_buffer_size_mb: int = 64,
        service_account_name: str = "deeplog-ebpf-node",
    ) -> Dict[str, Any]:
        """Programmatically generate a production-ready eBPF probe DaemonSet manifest."""
        return {
            "apiVersion": "apps/v1",
            "kind": "DaemonSet",
            "metadata": {
                "name": name,
                "namespace": namespace,
                "labels": {
                    "app.kubernetes.io/name": name,
                    "app.kubernetes.io/component": "ebpf-probe",
                },
            },
            "spec": {
                "selector": {
                    "matchLabels": {
                        "app.kubernetes.io/name": name,
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app.kubernetes.io/name": name,
                            "app.kubernetes.io/component": "ebpf-probe",
                        }
                    },
                    "spec": {
                        "hostPID": True,
                        "hostNetwork": True,
                        "dnsPolicy": "ClusterFirstWithHostNet",
                        "serviceAccountName": service_account_name,
                        "tolerations": [
                            {"operator": "Exists", "effect": "NoSchedule"},
                            {"operator": "Exists", "effect": "NoExecute"},
                        ],
                        "containers": [
                            {
                                "name": "ebpf-probe",
                                "image": image,
                                "imagePullPolicy": "IfNotPresent",
                                "securityContext": {
                                    "privileged": True,
                                    "capabilities": {
                                        "add": [
                                            "SYS_ADMIN",
                                            "CAP_BPF",
                                            "CAP_PERFMON",
                                            "CAP_NET_ADMIN",
                                            "SYS_RESOURCE",
                                        ]
                                    },
                                },
                                "env": [
                                    {
                                        "name": "NODE_NAME",
                                        "valueFrom": {"fieldRef": {"fieldPath": "spec.nodeName"}},
                                    },
                                    {"name": "BPF_FS_PATH", "value": bpf_mount_path},
                                    {"name": "EBPF_RINGBUF_SIZE_MB", "value": str(ring_buffer_size_mb)},
                                ],
                                "resources": {
                                    "requests": {"cpu": "200m", "memory": "256Mi"},
                                    "limits": {"cpu": "1000m", "memory": "1Gi"},
                                },
                                "volumeMounts": [
                                    {
                                        "name": "bpf-fs",
                                        "mountPath": bpf_mount_path,
                                        "mountPropagation": "Bidirectional",
                                    },
                                    {
                                        "name": "debugfs",
                                        "mountPath": "/sys/kernel/debug",
                                        "mountPropagation": "HostToContainer",
                                    },
                                ],
                            }
                        ],
                        "volumes": [
                            {
                                "name": "bpf-fs",
                                "hostPath": {"path": bpf_mount_path, "type": "DirectoryOrCreate"},
                            },
                            {
                                "name": "debugfs",
                                "hostPath": {"path": "/sys/kernel/debug", "type": "DirectoryOrCreate"},
                            },
                        ],
                    },
                },
            },
        }

    # -----------------------------------------------------------------------
    # Cluster Inspection & Status Reconciliation
    # -----------------------------------------------------------------------

    def inspect_cluster_status(self, cluster_cr: Dict[str, Any]) -> ClusterStatusInspection:
        """Inspect and compute health metrics for a DeepLogCluster resource."""
        spec = cluster_cr.get("spec", {})
        status = cluster_cr.get("status", {})

        desired = int(spec.get("replicas", 1))
        ready = int(status.get("readyReplicas", 0))
        ebpf_active = int(status.get("ebpfNodesActive", 0))
        anomalies = int(status.get("anomaliesDetected", 0))
        phase = status.get("phase", "Pending")

        health_ratio = (ready / desired) if desired > 0 else 0.0
        health_percentage = min(100.0, max(0.0, health_ratio * 100.0))

        conditions = status.get("conditions", [])
        cond_map = {c.get("type", "Unknown"): c.get("status", "False") for c in conditions if isinstance(c, dict)}

        is_healthy = (phase == "Running" and ready == desired)
        msg = f"Cluster in phase '{phase}' with {ready}/{desired} ready replicas."

        return ClusterStatusInspection(
            phase=phase,
            health_percentage=round(health_percentage, 2),
            ready_replicas=ready,
            desired_replicas=desired,
            ebpf_nodes_active=ebpf_active,
            anomalies_detected=anomalies,
            is_healthy=is_healthy,
            conditions_summary=cond_map,
            message=msg,
        )

    def reconcile_cluster_status(
        self,
        cluster_cr: Dict[str, Any],
        pod_statuses: List[Dict[str, Any]],
        ebpf_daemonset_status: Optional[Dict[str, Any]] = None,
        anomalies_increment: int = 0,
    ) -> Dict[str, Any]:
        """Reconcile and advance the operational status of a DeepLogCluster.

        Args:
            cluster_cr: DeepLogCluster manifest dictionary.
            pod_statuses: List of pod status dictionaries (each having phase, ready).
            ebpf_daemonset_status: Optional DaemonSet status dict (e.g. numberReady).
            anomalies_increment: Number of newly detected anomalies to add.

        Returns:
            Updated DeepLogCluster manifest dictionary with reconciled status.
        """
        updated = copy.deepcopy(cluster_cr)
        spec = updated.setdefault("spec", {})
        status = updated.setdefault("status", {})

        desired = int(spec.get("replicas", 1))

        # Count ready pods
        ready_count = 0
        running_count = 0
        for pod in pod_statuses:
            pod_phase = pod.get("phase", "")
            is_ready = pod.get("ready", False)
            if pod_phase == "Running":
                running_count += 1
            if is_ready:
                ready_count += 1

        status["readyReplicas"] = ready_count
        status["totalReplicas"] = len(pod_statuses)

        # eBPF active nodes
        if ebpf_daemonset_status:
            status["ebpfNodesActive"] = int(ebpf_daemonset_status.get("numberReady", 0))

        # Update anomalies count
        current_anomalies = int(status.get("anomaliesDetected", 0))
        status["anomaliesDetected"] = current_anomalies + anomalies_increment

        # Phase determination
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        conditions = status.setdefault("conditions", [])

        if ready_count == desired and desired > 0:
            status["phase"] = "Running"
            conditions.append({
                "type": "Ready",
                "status": "True",
                "lastTransitionTime": now_iso,
                "reason": "AllReplicasReady",
                "message": f"All {desired} DeepLog inference pods are healthy.",
            })
        elif ready_count > 0:
            status["phase"] = "Degraded"
            conditions.append({
                "type": "Ready",
                "status": "False",
                "lastTransitionTime": now_iso,
                "reason": "PartiallyReady",
                "message": f"Cluster degraded: {ready_count}/{desired} pods ready.",
            })
        elif running_count > 0:
            status["phase"] = "Initializing"
            conditions.append({
                "type": "Ready",
                "status": "False",
                "lastTransitionTime": now_iso,
                "reason": "ContainersStarting",
                "message": "Inference pods starting.",
            })
        else:
            status["phase"] = "Pending"
            conditions.append({
                "type": "Ready",
                "status": "False",
                "lastTransitionTime": now_iso,
                "reason": "AwaitingPods",
                "message": "No pods running.",
            })

        return updated

    def evaluate_policy(
        self,
        policy_cr: Dict[str, Any],
        anomaly_score: float,
        consecutive_violations: int = 1,
    ) -> PolicyEvaluationResult:
        """Evaluate incoming anomaly scores against a DeepLogPolicy Custom Resource."""
        spec = policy_cr.get("spec", {})
        threshold = float(spec.get("anomalyThreshold", 0.85))
        required_consec = int(spec.get("consecutiveViolations", 1))
        action = spec.get("action", "Alert")

        triggered = (anomaly_score >= threshold) and (consecutive_violations >= required_consec)
        effective_action = action if triggered else "None"

        if triggered:
            msg = (
                f"Policy violation: anomaly_score {anomaly_score:.3f} >= threshold {threshold:.3f} "
                f"for {consecutive_violations}/{required_consec} consecutive events. Action: {action}."
            )
        else:
            msg = (
                f"Policy compliant: anomaly_score {anomaly_score:.3f} (threshold {threshold:.3f}), "
                f"consecutive {consecutive_violations}/{required_consec}."
            )

        return PolicyEvaluationResult(
            triggered=triggered,
            action=effective_action,
            anomaly_score=anomaly_score,
            threshold=threshold,
            consecutive_violations=consecutive_violations,
            required_consecutive=required_consec,
            message=msg,
        )
