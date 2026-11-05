"""Unit tests for Kubernetes Deployment Manifests, CRD, Helm Charts, and K8sManager.

Validates:
1. Production CRD manifest (deeplog_crd.yaml) OpenAPI v3 schema for DeepLogCluster & DeepLogPolicy.
2. Production eBPF probe DaemonSet (deeplog_ebpf_daemonset.yaml) securityContext, hostPID, and BPF mounts.
3. Helm chart structure, Chart.yaml metadata, values.yaml configuration, and templates.
4. K8sManager parsing, dumping, and round-trip fidelity.
5. K8sManager manifest validation (schema checking, error and warning detection).
6. K8sManager programmatic generation of CRD instances and DaemonSet manifests.
7. K8sManager cluster status inspection, health metrics, and autonomous reconciliation.
8. K8sManager policy threshold evaluation and autonomous mitigation trigger logic.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pytest

from deeplog import (
    ClusterStatusInspection,
    K8sManager,
    ManifestValidationResult,
    PolicyEvaluationResult,
)


@pytest.fixture
def k8s_mgr() -> K8sManager:
    """Fixture returning K8sManager pointing to the workspace root."""
    return K8sManager()


class TestCRDManifest:
    """Validation tests for deploy/k8s/crd/deeplog_crd.yaml."""

    def test_crd_file_exists(self, k8s_mgr: K8sManager):
        """CRD file exists in expected directory deploy/k8s/crd/."""
        assert k8s_mgr.crd_path.exists(), f"CRD file missing at {k8s_mgr.crd_path}"
        assert k8s_mgr.crd_path.is_file()

    def test_crd_multi_document_loading(self, k8s_mgr: K8sManager):
        """CRD file contains two valid YAML documents for DeepLogCluster and DeepLogPolicy."""
        docs = k8s_mgr.load_manifest(k8s_mgr.crd_path)
        assert len(docs) == 2, f"Expected 2 CRD documents, got {len(docs)}"

        kinds = [d.get("kind") for d in docs]
        assert kinds == ["CustomResourceDefinition", "CustomResourceDefinition"]

        names = [d.get("metadata", {}).get("name") for d in docs]
        assert "deeplogclusters.deeplog.ai" in names
        assert "deeplogpolicies.deeplog.ai" in names

    def test_deeplog_cluster_crd_schema(self, k8s_mgr: K8sManager):
        """Validate DeepLogCluster CRD structural schema and configurations."""
        docs = k8s_mgr.load_manifest(k8s_mgr.crd_path)
        cluster_crd = next(d for d in docs if d["metadata"]["name"] == "deeplogclusters.deeplog.ai")

        assert cluster_crd["apiVersion"] == "apiextensions.k8s.io/v1"
        spec = cluster_crd["spec"]
        assert spec["group"] == "deeplog.ai"
        assert spec["scope"] == "Namespaced"
        assert spec["names"]["kind"] == "DeepLogCluster"
        assert spec["names"]["plural"] == "deeplogclusters"
        assert "dlc" in spec["names"]["shortNames"]

        # Versions check
        versions = spec["versions"]
        assert len(versions) >= 1
        v1 = versions[0]
        assert v1["name"] == "v1alpha1"
        assert v1["storage"] is True
        assert v1["served"] is True

        # Subresources
        assert "status" in v1["subresources"]
        assert "scale" in v1["subresources"]
        assert v1["subresources"]["scale"]["specReplicasPath"] == ".spec.replicas"

        # OpenAPI v3 properties
        schema_props = v1["schema"]["openAPIV3Schema"]["properties"]["spec"]["properties"]
        assert "replicas" in schema_props
        assert "modelPath" in schema_props
        assert "ebpfProbe" in schema_props
        assert "resources" in schema_props
        assert "federation" in schema_props
        assert "safetyFilter" in schema_props

        # Printer columns
        col_names = [col["name"] for col in v1["additionalPrinterColumns"]]
        assert "Status" in col_names
        assert "Desired" in col_names
        assert "Ready" in col_names
        assert "Anomalies" in col_names

    def test_deeplog_policy_crd_schema(self, k8s_mgr: K8sManager):
        """Validate DeepLogPolicy CRD structural schema."""
        docs = k8s_mgr.load_manifest(k8s_mgr.crd_path)
        policy_crd = next(d for d in docs if d["metadata"]["name"] == "deeplogpolicies.deeplog.ai")

        assert policy_crd["apiVersion"] == "apiextensions.k8s.io/v1"
        spec = policy_crd["spec"]
        assert spec["group"] == "deeplog.ai"
        assert spec["scope"] == "Namespaced"
        assert spec["names"]["kind"] == "DeepLogPolicy"
        assert "dlp" in spec["names"]["shortNames"]

        v1 = spec["versions"][0]
        policy_props = v1["schema"]["openAPIV3Schema"]["properties"]["spec"]["properties"]
        assert "detectionMode" in policy_props
        assert "action" in policy_props
        assert "anomalyThreshold" in policy_props
        assert "rules" in policy_props


class TestEBPFDaemonSetManifest:
    """Validation tests for deploy/k8s/daemonset/deeplog_ebpf_daemonset.yaml."""

    def test_daemonset_file_exists(self, k8s_mgr: K8sManager):
        """DaemonSet manifest exists at deploy/k8s/daemonset/."""
        assert k8s_mgr.daemonset_path.exists(), f"Missing {k8s_mgr.daemonset_path}"
        assert k8s_mgr.daemonset_path.is_file()

    def test_daemonset_documents_presence(self, k8s_mgr: K8sManager):
        """DaemonSet manifest defines namespace, serviceaccount, rbac, and daemonset."""
        docs = k8s_mgr.load_manifest(k8s_mgr.daemonset_path)
        kinds = [d.get("kind") for d in docs]

        assert "Namespace" in kinds
        assert "ServiceAccount" in kinds
        assert "ClusterRole" in kinds
        assert "ClusterRoleBinding" in kinds
        assert "DaemonSet" in kinds

    def test_ebpf_daemonset_security_and_mounts(self, k8s_mgr: K8sManager):
        """Validate hostPID, privileged securityContext, and BPF mounts."""
        docs = k8s_mgr.load_manifest(k8s_mgr.daemonset_path)
        ds = next(d for d in docs if d.get("kind") == "DaemonSet")

        assert ds["metadata"]["name"] == "deeplog-ebpf-probe"
        pod_spec = ds["spec"]["template"]["spec"]

        # Critical eBPF requirements
        assert pod_spec["hostPID"] is True, "eBPF probe must have hostPID=True to trace host processes."
        assert pod_spec["hostNetwork"] is True

        # Container specs
        containers = pod_spec["containers"]
        assert len(containers) >= 1
        c = containers[0]
        assert c["name"] == "ebpf-probe"

        # Security context
        sec_ctx = c["securityContext"]
        assert sec_ctx["privileged"] is True, "eBPF probe container must be privileged."
        caps = sec_ctx.get("capabilities", {}).get("add", [])
        assert "CAP_BPF" in caps or "SYS_ADMIN" in caps

        # Volume mounts
        mounts = {m["mountPath"]: m for m in c["volumeMounts"]}
        assert "/sys/fs/bpf" in mounts, "Missing /sys/fs/bpf mount for eBPF maps."
        assert "/sys/kernel/debug" in mounts, "Missing /sys/kernel/debug mount for kprobes/tracepoints."
        assert mounts["/sys/fs/bpf"].get("mountPropagation") == "Bidirectional"

        # HostPath volumes
        vols = {v["name"]: v for v in pod_spec["volumes"]}
        assert "bpf-fs" in vols
        assert vols["bpf-fs"]["hostPath"]["path"] == "/sys/fs/bpf"

        # Tolerations for all nodes including control-plane
        tolerations = pod_spec.get("tolerations", [])
        assert len(tolerations) >= 2
        effects = [t.get("effect") for t in tolerations]
        assert "NoSchedule" in effects
        assert "NoExecute" in effects


class TestHelmChart:
    """Validation tests for deploy/k8s/helm/ chart structure and templates."""

    def test_helm_chart_metadata(self, k8s_mgr: K8sManager):
        """Chart.yaml has valid Helm v2 specification."""
        chart_file = k8s_mgr.helm_dir / "Chart.yaml"
        assert chart_file.exists()

        docs = k8s_mgr.load_manifest(chart_file)
        chart = docs[0]
        assert chart["apiVersion"] == "v2"
        assert chart["name"] == "deeplog"
        assert chart["type"] == "application"
        assert chart["version"] == "1.0.0"
        assert "ebpf" in chart.get("keywords", [])

    def test_helm_values_structure(self, k8s_mgr: K8sManager):
        """values.yaml contains required configuration sections."""
        values_file = k8s_mgr.helm_dir / "values.yaml"
        assert values_file.exists()

        docs = k8s_mgr.load_manifest(values_file)
        values = docs[0]

        assert "controller" in values
        assert values["controller"]["replicaCount"] == 3
        assert "autoscaling" in values["controller"]

        assert "ebpfProbe" in values
        assert values["ebpfProbe"]["enabled"] is True
        assert values["ebpfProbe"]["hostPID"] is True
        assert values["ebpfProbe"]["bpfMountPath"] == "/sys/fs/bpf"
        assert values["ebpfProbe"]["securityContext"]["privileged"] is True

        assert "service" in values
        assert "rbac" in values
        assert "serviceAccount" in values
        assert "federation" in values
        assert "safetyFilter" in values

    def test_helm_template_files_present(self, k8s_mgr: K8sManager):
        """All expected Helm deployment templates exist."""
        templates_dir = k8s_mgr.helm_dir / "templates"
        expected_templates = [
            "_helpers.tpl",
            "serviceaccount.yaml",
            "rbac.yaml",
            "configmap.yaml",
            "daemonset-ebpf.yaml",
            "deployment-controller.yaml",
            "service.yaml",
            "hpa.yaml",
            "deeplog-cluster.yaml",
        ]
        for tpl in expected_templates:
            tpl_path = templates_dir / tpl
            assert tpl_path.exists(), f"Helm template missing: {tpl}"
            content = tpl_path.read_text(encoding="utf-8")
            assert len(content) > 50, f"Template {tpl} appears empty"


class TestK8sManagerCore:
    """Tests for K8sManager core parsing, serialization, and round-tripping."""

    def test_parse_and_dump_roundtrip(self, k8s_mgr: K8sManager):
        """Validate parse_yaml and dump_yaml round-trip preservation."""
        sample_dict = {
            "apiVersion": "deeplog.ai/v1alpha1",
            "kind": "DeepLogCluster",
            "metadata": {
                "name": "production-cluster",
                "namespace": "deeplog-system",
                "labels": {"env": "prod", "tier": "critical"},
            },
            "spec": {
                "replicas": 5,
                "modelPath": "/opt/models/deeplog.pt",
                "ebpfProbe": {
                    "enabled": True,
                    "ringBufferSizeMb": 128,
                    "samplingRate": 0.5,
                },
                "tags": ["alpha", "beta", "gamma"],
            },
        }

        yaml_str = k8s_mgr.dump_yaml(sample_dict)
        assert "apiVersion: deeplog.ai/v1alpha1" in yaml_str
        assert "replicas: 5" in yaml_str

        parsed = k8s_mgr.parse_yaml(yaml_str)
        assert len(parsed) == 1
        res = parsed[0]
        assert res["apiVersion"] == sample_dict["apiVersion"]
        assert res["kind"] == sample_dict["kind"]
        assert res["metadata"]["name"] == "production-cluster"
        assert res["spec"]["replicas"] == 5
        assert res["spec"]["ebpfProbe"]["enabled"] is True
        assert res["spec"]["ebpfProbe"]["samplingRate"] == 0.5
        assert res["spec"]["tags"] == ["alpha", "beta", "gamma"]

    def test_multi_document_export_and_load(self, k8s_mgr: K8sManager):
        """Exporting a multi-document list writes documents separated by '---'."""
        doc1 = {"apiVersion": "v1", "kind": "ConfigMap", "metadata": {"name": "cfg-1"}}
        doc2 = {"apiVersion": "v1", "kind": "ConfigMap", "metadata": {"name": "cfg-2"}}

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "multi.yaml"
            k8s_mgr.export_manifest([doc1, doc2], out_file)
            assert out_file.exists()

            loaded = k8s_mgr.load_manifest(out_file)
            assert len(loaded) == 2
            assert loaded[0]["metadata"]["name"] == "cfg-1"
            assert loaded[1]["metadata"]["name"] == "cfg-2"


class TestK8sManagerValidation:
    """Tests for K8sManager manifest validation rules."""

    def test_validate_crd_file(self, k8s_mgr: K8sManager):
        """Validating the production CRD file returns valid status with no errors."""
        results = k8s_mgr.validate_file(k8s_mgr.crd_path)
        assert len(results) == 2
        for r in results:
            assert r.is_valid is True, f"CRD validation failed: {r.errors}"
            assert len(r.errors) == 0

    def test_validate_daemonset_file(self, k8s_mgr: K8sManager):
        """Validating the production DaemonSet file returns valid status."""
        results = k8s_mgr.validate_file(k8s_mgr.daemonset_path)
        assert len(results) >= 4
        for r in results:
            assert r.is_valid is True, f"DaemonSet manifest validation failed: {r.errors}"

    def test_validation_catches_missing_fields(self, k8s_mgr: K8sManager):
        """Manifest missing apiVersion or metadata.name is marked invalid."""
        invalid_manifest = {
            "kind": "DeepLogCluster",
            "metadata": {},
        }
        res = k8s_mgr.validate_manifest(invalid_manifest)
        assert res.is_valid is False
        assert any("apiVersion" in err for err in res.errors)
        assert any("metadata.name" in err for err in res.errors)

    def test_validation_catches_invalid_cluster_replicas(self, k8s_mgr: K8sManager):
        """Cluster with 0 or negative replicas fails validation."""
        cr = k8s_mgr.generate_deeplog_cluster_cr("test-cluster", replicas=0)
        res = k8s_mgr.validate_manifest(cr)
        assert res.is_valid is False
        assert any("replicas" in err for err in res.errors)

    def test_validation_catches_invalid_policy_mode_or_action(self, k8s_mgr: K8sManager):
        """Policy with invalid mode or action fails validation."""
        bad_policy = {
            "apiVersion": "deeplog.ai/v1alpha1",
            "kind": "DeepLogPolicy",
            "metadata": {"name": "bad-policy"},
            "spec": {
                "detectionMode": "invalid-ai-magic",
                "action": "DoNothingCrazy",
                "anomalyThreshold": 1.5,
            },
        }
        res = k8s_mgr.validate_manifest(bad_policy)
        assert res.is_valid is False
        assert any("detectionMode" in err for err in res.errors)
        assert any("action" in err for err in res.errors)
        assert any("anomalyThreshold" in err for err in res.errors)

    def test_validation_catches_unprivileged_ebpf_daemonset(self, k8s_mgr: K8sManager):
        """eBPF DaemonSet without hostPID or privileged securityContext fails validation."""
        bad_ds = {
            "apiVersion": "apps/v1",
            "kind": "DaemonSet",
            "metadata": {"name": "deeplog-ebpf-bad"},
            "spec": {
                "template": {
                    "spec": {
                        "hostPID": False,
                        "containers": [
                            {
                                "name": "probe",
                                "securityContext": {"privileged": False},
                            }
                        ],
                    }
                }
            },
        }
        res = k8s_mgr.validate_manifest(bad_ds)
        assert res.is_valid is False
        assert any("hostPID" in err for err in res.errors)
        assert any("privileged" in err for err in res.errors)


class TestK8sManagerProgrammaticGeneration:
    """Tests for programmatic CRD and DaemonSet instance generation."""

    def test_generate_deeplog_cluster_cr(self, k8s_mgr: K8sManager):
        """Generate DeepLogCluster CR and verify spec completeness."""
        cr = k8s_mgr.generate_deeplog_cluster_cr(
            name="dlc-staging",
            namespace="staging",
            replicas=4,
            image="deeplog/engine:v1.2.0",
            model_path="/mnt/models/extreme.pt",
            ebpf_enabled=True,
            bpf_mount_path="/sys/fs/bpf",
            ring_buffer_size_mb=128,
            sampling_rate=0.75,
            federation_enabled=True,
            safety_filter_enabled=True,
        )

        assert cr["apiVersion"] == "deeplog.ai/v1alpha1"
        assert cr["kind"] == "DeepLogCluster"
        assert cr["metadata"]["name"] == "dlc-staging"
        assert cr["metadata"]["namespace"] == "staging"
        assert cr["spec"]["replicas"] == 4
        assert cr["spec"]["image"] == "deeplog/engine:v1.2.0"
        assert cr["spec"]["modelPath"] == "/mnt/models/extreme.pt"
        assert cr["spec"]["ebpfProbe"]["ringBufferSizeMb"] == 128
        assert cr["spec"]["ebpfProbe"]["samplingRate"] == 0.75
        assert cr["spec"]["federation"]["enabled"] is True
        assert cr["spec"]["safetyFilter"]["enabled"] is True

        # Validation must pass
        val_res = k8s_mgr.validate_manifest(cr)
        assert val_res.is_valid is True

    def test_generate_deeplog_policy_cr(self, k8s_mgr: K8sManager):
        """Generate DeepLogPolicy CR and verify compliance."""
        pol = k8s_mgr.generate_deeplog_policy_cr(
            name="security-strict-policy",
            namespace="production",
            detection_mode="native-ebpf",
            action="Quarantine",
            anomaly_threshold=0.92,
            consecutive_violations=5,
            target_namespaces=["payments", "auth"],
            target_labels={"criticality": "tier-0"},
        )

        assert pol["apiVersion"] == "deeplog.ai/v1alpha1"
        assert pol["kind"] == "DeepLogPolicy"
        assert pol["spec"]["detectionMode"] == "native-ebpf"
        assert pol["spec"]["action"] == "Quarantine"
        assert pol["spec"]["anomalyThreshold"] == 0.92
        assert pol["spec"]["consecutiveViolations"] == 5
        assert pol["spec"]["targetWorkloads"]["namespaces"] == ["payments", "auth"]
        assert pol["spec"]["targetWorkloads"]["matchLabels"]["criticality"] == "tier-0"

        val_res = k8s_mgr.validate_manifest(pol)
        assert val_res.is_valid is True

    def test_generate_deeplog_policy_invalid_arguments_raise(self, k8s_mgr: K8sManager):
        """Generating policy with unsupported detectionMode or action raises ValueError."""
        with pytest.raises(ValueError, match="Invalid detectionMode"):
            k8s_mgr.generate_deeplog_policy_cr("bad", detection_mode="magic_mode")

        with pytest.raises(ValueError, match="Invalid action"):
            k8s_mgr.generate_deeplog_policy_cr("bad", action="Explode")

    def test_generate_ebpf_daemonset(self, k8s_mgr: K8sManager):
        """Generate eBPF DaemonSet programmatically and verify specification."""
        ds = k8s_mgr.generate_ebpf_daemonset(
            name="deeplog-ebpf-custom",
            namespace="monitoring",
            image="deeplog/ebpf-probe:v2",
            bpf_mount_path="/sys/fs/bpf",
            ring_buffer_size_mb=32,
        )

        assert ds["apiVersion"] == "apps/v1"
        assert ds["kind"] == "DaemonSet"
        assert ds["metadata"]["name"] == "deeplog-ebpf-custom"
        assert ds["spec"]["template"]["spec"]["hostPID"] is True
        assert ds["spec"]["template"]["spec"]["containers"][0]["securityContext"]["privileged"] is True

        val_res = k8s_mgr.validate_manifest(ds)
        assert val_res.is_valid is True


class TestK8sManagerStatusAndReconciliation:
    """Tests for cluster status inspection, pod reconciliation, and policy evaluation."""

    def test_inspect_cluster_status_running_and_healthy(self, k8s_mgr: K8sManager):
        """Inspect cluster with all replicas ready evaluates as healthy."""
        cr = k8s_mgr.generate_deeplog_cluster_cr("prod-cluster", replicas=3)
        cr["status"]["phase"] = "Running"
        cr["status"]["readyReplicas"] = 3
        cr["status"]["totalReplicas"] = 3
        cr["status"]["ebpfNodesActive"] = 5
        cr["status"]["anomaliesDetected"] = 12

        status_info = k8s_mgr.inspect_cluster_status(cr)
        assert isinstance(status_info, ClusterStatusInspection)
        assert status_info.phase == "Running"
        assert status_info.health_percentage == 100.0
        assert status_info.ready_replicas == 3
        assert status_info.desired_replicas == 3
        assert status_info.ebpf_nodes_active == 5
        assert status_info.anomalies_detected == 12
        assert status_info.is_healthy is True

    def test_inspect_cluster_status_degraded(self, k8s_mgr: K8sManager):
        """Inspect cluster with partial ready replicas evaluates as degraded."""
        cr = k8s_mgr.generate_deeplog_cluster_cr("prod-cluster", replicas=4)
        cr["status"]["phase"] = "Degraded"
        cr["status"]["readyReplicas"] = 2

        status_info = k8s_mgr.inspect_cluster_status(cr)
        assert status_info.phase == "Degraded"
        assert status_info.health_percentage == 50.0
        assert status_info.is_healthy is False

    def test_reconcile_cluster_status_transitions_to_running(self, k8s_mgr: K8sManager):
        """Reconciliation with all pods ready transitions phase from Pending to Running."""
        cr = k8s_mgr.generate_deeplog_cluster_cr("dlc-test", replicas=3)
        assert cr["status"]["phase"] == "Pending"

        pod_statuses = [
            {"name": "dlc-test-0", "phase": "Running", "ready": True},
            {"name": "dlc-test-1", "phase": "Running", "ready": True},
            {"name": "dlc-test-2", "phase": "Running", "ready": True},
        ]
        daemonset_status = {"numberReady": 4}

        reconciled = k8s_mgr.reconcile_cluster_status(
            cr,
            pod_statuses,
            ebpf_daemonset_status=daemonset_status,
            anomalies_increment=3,
        )

        assert reconciled["status"]["phase"] == "Running"
        assert reconciled["status"]["readyReplicas"] == 3
        assert reconciled["status"]["totalReplicas"] == 3
        assert reconciled["status"]["ebpfNodesActive"] == 4
        assert reconciled["status"]["anomaliesDetected"] == 3
        assert any(c["reason"] == "AllReplicasReady" for c in reconciled["status"]["conditions"])

    def test_reconcile_cluster_status_partial_degraded(self, k8s_mgr: K8sManager):
        """Reconciliation with only 1/3 pods ready sets phase to Degraded."""
        cr = k8s_mgr.generate_deeplog_cluster_cr("dlc-test", replicas=3)
        pod_statuses = [
            {"name": "dlc-test-0", "phase": "Running", "ready": True},
            {"name": "dlc-test-1", "phase": "Pending", "ready": False},
        ]

        reconciled = k8s_mgr.reconcile_cluster_status(cr, pod_statuses)
        assert reconciled["status"]["phase"] == "Degraded"
        assert reconciled["status"]["readyReplicas"] == 1
        assert any(c["reason"] == "PartiallyReady" for c in reconciled["status"]["conditions"])

    def test_evaluate_policy_triggers_action(self, k8s_mgr: K8sManager):
        """Policy evaluation triggers action when score and consecutive violations exceed threshold."""
        pol = k8s_mgr.generate_deeplog_policy_cr(
            name="alert-policy",
            action="Remediate",
            anomaly_threshold=0.80,
            consecutive_violations=3,
        )

        # Case 1: Below score threshold
        res1 = k8s_mgr.evaluate_policy(pol, anomaly_score=0.75, consecutive_violations=5)
        assert res1.triggered is False
        assert res1.action == "None"

        # Case 2: Above score, but insufficient consecutive events
        res2 = k8s_mgr.evaluate_policy(pol, anomaly_score=0.95, consecutive_violations=2)
        assert res2.triggered is False
        assert res2.action == "None"

        # Case 3: Above score and sufficient consecutive events -> Triggered!
        res3 = k8s_mgr.evaluate_policy(pol, anomaly_score=0.95, consecutive_violations=3)
        assert res3.triggered is True
        assert res3.action == "Remediate"
        assert "Remediate" in res3.message
