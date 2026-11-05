"""Unit tests validating Dockerfile, docker-compose.yml, and LaTeX Research Paper.

Validates:
1. Dockerfile:
   - Multi-stage build (builder, runtime)
   - Python 3.11/3.12-slim base image
   - Security-hardened non-root user (deeplog:deeplog, UID 10001)
   - Dependency pre-warming and layer caching
   - System package cleanup (rm -rf /var/lib/apt/lists/*)
   - Streamlit / API port exposure (8501, 8000)
   - Healthcheck directive against Streamlit health endpoint
   - Non-root entrypoint and execution parameters
2. docker-compose.yml:
   - Syntactically valid YAML structure
   - All 4 required services: deeplog-cockpit, kafka, clickhouse, prometheus
   - Port mappings (8501, 8000, 9092, 8123, 9000, 9090)
   - Healthchecks for all container services
   - Dependency ordering and health conditions
   - Custom bridge network bridging (deeplog-net)
   - Named volumes for data persistence
   - Prometheus scrape configuration
3. paper/deeplog_infinity_usenix.tex:
   - Publication-grade USENIX/ACM structure and styling
   - All required sections (Abstract, Introduction, Threat Model, System Architecture,
     Mathematical Foundations, Empirical Results, Discussion, Related Work, Conclusion)
   - Explicit coverage of all 7 Architectural Pillars
   - Mathematical Foundations:
     * QP Control Barrier Function (CBF) safety proof and forward invariance
     * Discrete Lyapunov stability contraction proof
     * HDC 10,000-bit hypervector Hamming distance Chernoff bounds
     * Open Jackson Queue Network equilibrium and Little's Law conservation
   - Empirical benchmark metrics:
     * SOTA F1-Score: 72.23%
     * Evaluation corpus: 570,204 blocks (11,077,032 transitions)
     * HDC Latency: 0.293 µs (0.29 µs)
     * Blast radius reduction: 95.5%
     * False positive alert reduction: 81.2%
   - LaTeX structural balance and syntax integrity
4. paper/references.bib:
   - Seminal citations: du2017deeplog, meng2019loganomaly, zhang2019logrobust,
     he2017drain, ames2019control, kanerva2009hyperdimensional, hafner2020dreamer
   - 100% citation resolution parity between .tex and .bib (zero broken citations)
   - Mandatory BibTeX fields (title, author, year)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Set

import pytest

from deeplog.k8s_manager import K8sManager


@pytest.fixture
def workspace_root() -> Path:
    """Fixture pointing to workspace root directory."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def k8s_mgr() -> K8sManager:
    """Fixture returning K8sManager for YAML parsing."""
    return K8sManager()


# ==============================================================================
# 1. Dockerfile Tests
# ==============================================================================

class TestDockerfile:
    """Validation test suite for Dockerfile."""

    def test_dockerfile_exists_and_substantial(self, workspace_root: Path):
        """Dockerfile exists at project root and is non-empty."""
        dockerfile = workspace_root / "Dockerfile"
        assert dockerfile.exists(), f"Dockerfile missing at {dockerfile}"
        assert dockerfile.is_file()
        content = dockerfile.read_text(encoding="utf-8")
        assert len(content) > 500, "Dockerfile content is suspiciously short"

    def test_dockerfile_multistage_syntax(self, workspace_root: Path):
        """Dockerfile implements multi-stage build pattern."""
        content = (workspace_root / "Dockerfile").read_text(encoding="utf-8")
        from_matches = re.findall(r"^FROM\s+(\S+)\s+AS\s+(\S+)", content, flags=re.MULTILINE | re.IGNORECASE)
        assert len(from_matches) >= 2, f"Expected at least 2 stages, found: {from_matches}"
        stages = [m[1].lower() for m in from_matches]
        assert "builder" in stages, f"Missing 'builder' stage in {stages}"
        assert any(s in ("runtime", "runner", "production") for s in stages), f"Missing runtime stage in {stages}"

    def test_dockerfile_base_image_python_slim(self, workspace_root: Path):
        """Dockerfile utilizes Python 3.11 or 3.12 slim base image."""
        content = (workspace_root / "Dockerfile").read_text(encoding="utf-8")
        base_images = re.findall(r"^FROM\s+(\S+)", content, flags=re.MULTILINE | re.IGNORECASE)
        for base in base_images:
            clean_base = base.lower()
            assert "python:3.11-slim" in clean_base or "python:3.12-slim" in clean_base, (
                f"Base image {base} is not Python 3.11-slim or 3.12-slim"
            )

    def test_dockerfile_security_nonroot_user(self, workspace_root: Path):
        """Dockerfile creates and runs as non-privileged non-root user (UID 10001)."""
        content = (workspace_root / "Dockerfile").read_text(encoding="utf-8")
        # Check non-root user creation
        assert re.search(r"useradd.*(?:-u\s+10001|deeplog)", content, re.IGNORECASE), "Non-root user creation missing"
        assert re.search(r"groupadd.*(?:-g\s+10001|deeplog)", content, re.IGNORECASE), "Non-root group creation missing"
        # Check USER switch instruction
        user_directives = re.findall(r"^USER\s+(\S+)", content, flags=re.MULTILINE)
        assert len(user_directives) >= 1, "No USER directive found in Dockerfile"
        final_user = user_directives[-1].strip()
        assert final_user in ("deeplog", "10001"), f"Final user must be non-root, got '{final_user}'"

    def test_dockerfile_dependency_caching_and_prewarming(self, workspace_root: Path):
        """Dockerfile caches dependency installation before copying application code."""
        content = (workspace_root / "Dockerfile").read_text(encoding="utf-8")
        # Ensure requirements.txt is copied and installed prior to general COPY
        req_copy_idx = content.find("COPY requirements.txt")
        pip_install_idx = content.find("pip install")
        app_copy_idx = content.find("COPY --chown=deeplog:deeplog app.py")
        if app_copy_idx == -1:
            app_copy_idx = content.find("COPY . /app")

        assert req_copy_idx != -1, "requirements.txt COPY directive missing"
        assert pip_install_idx != -1, "pip install directive missing"
        assert req_copy_idx < pip_install_idx, "requirements.txt must be copied before pip install"
        if app_copy_idx != -1:
            assert pip_install_idx < app_copy_idx, "pip install dependencies must occur before copying app source"

    def test_dockerfile_clean_apt_cache(self, workspace_root: Path):
        """Dockerfile purges apt cache to minimize image size and attack surface."""
        content = (workspace_root / "Dockerfile").read_text(encoding="utf-8")
        assert "rm -rf /var/lib/apt/lists/*" in content, "Missing apt cache purge command"

    def test_dockerfile_healthcheck_directive(self, workspace_root: Path):
        """Dockerfile declares robust container HEALTHCHECK."""
        content = (workspace_root / "Dockerfile").read_text(encoding="utf-8")
        assert "HEALTHCHECK" in content, "HEALTHCHECK instruction missing from Dockerfile"
        assert "--interval=" in content, "HEALTHCHECK missing --interval flag"
        assert "--timeout=" in content, "HEALTHCHECK missing --timeout flag"
        assert "--retries=" in content, "HEALTHCHECK missing --retries flag"
        assert "curl" in content or "wget" in content or "health" in content, "HEALTHCHECK test command missing"

    def test_dockerfile_ports_and_env(self, workspace_root: Path):
        """Dockerfile exposes proper ports (8501, 8000) and sets Python env flags."""
        content = (workspace_root / "Dockerfile").read_text(encoding="utf-8")
        assert "EXPOSE" in content
        assert "8501" in content
        assert "8000" in content
        assert "PYTHONUNBUFFERED=1" in content
        assert "PYTHONDONTWRITEBYTECODE=1" in content

    def test_dockerfile_entrypoint_and_cmd(self, workspace_root: Path):
        """Dockerfile defines ENTRYPOINT or CMD launching Streamlit cockpit."""
        content = (workspace_root / "Dockerfile").read_text(encoding="utf-8")
        assert "ENTRYPOINT" in content or "CMD" in content
        assert "streamlit" in content
        assert "app.py" in content


# ==============================================================================
# 2. docker-compose.yml Tests
# ==============================================================================

class TestDockerCompose:
    """Validation test suite for docker-compose.yml."""

    def test_compose_file_exists(self, workspace_root: Path):
        """docker-compose.yml exists at project root."""
        compose_file = workspace_root / "docker-compose.yml"
        assert compose_file.exists(), f"docker-compose.yml missing at {compose_file}"
        assert compose_file.is_file()

    def test_compose_valid_yaml_structure(self, workspace_root: Path, k8s_mgr: K8sManager):
        """docker-compose.yml is valid YAML and contains services and networks."""
        compose_file = workspace_root / "docker-compose.yml"
        docs = k8s_mgr.load_manifest(compose_file)
        assert len(docs) >= 1, "No YAML documents parsed from docker-compose.yml"
        compose_dict = docs[0]
        assert "services" in compose_dict, "Missing 'services' key in docker-compose.yml"
        assert "networks" in compose_dict, "Missing 'networks' key in docker-compose.yml"
        assert "volumes" in compose_dict, "Missing 'volumes' key in docker-compose.yml"

    def test_compose_all_four_services_defined(self, workspace_root: Path, k8s_mgr: K8sManager):
        """All 4 required services are defined: deeplog-cockpit, kafka, clickhouse, prometheus."""
        docs = k8s_mgr.load_manifest(workspace_root / "docker-compose.yml")
        services = docs[0]["services"]
        required_services = {"deeplog-cockpit", "kafka", "clickhouse", "prometheus"}
        assert required_services.issubset(set(services.keys())), (
            f"Missing required services: {required_services - set(services.keys())}"
        )

    def test_compose_deeplog_cockpit_service(self, workspace_root: Path, k8s_mgr: K8sManager):
        """deeplog-cockpit service is configured correctly."""
        docs = k8s_mgr.load_manifest(workspace_root / "docker-compose.yml")
        cockpit = docs[0]["services"]["deeplog-cockpit"]

        # Port checks (8501 and 8000)
        ports = cockpit.get("ports", [])
        port_strs = [str(p) for p in ports]
        assert any("8501" in p for p in port_strs), f"Port 8501 not found in {ports}"
        assert any("8000" in p for p in port_strs), f"Port 8000 not found in {ports}"

        # Dependency checks
        depends_on = cockpit.get("depends_on", {})
        assert "kafka" in depends_on, "deeplog-cockpit must depend on kafka"
        assert "clickhouse" in depends_on, "deeplog-cockpit must depend on clickhouse"
        assert "prometheus" in depends_on, "deeplog-cockpit must depend on prometheus"

        # Healthcheck check
        assert "healthcheck" in cockpit, "deeplog-cockpit missing healthcheck"
        test_cmd = str(cockpit["healthcheck"].get("test", ""))
        assert "8501" in test_cmd or "health" in test_cmd

        # Networks check
        assert "deeplog-net" in cockpit.get("networks", [])

    def test_compose_kafka_service(self, workspace_root: Path, k8s_mgr: K8sManager):
        """kafka broker service is configured correctly."""
        docs = k8s_mgr.load_manifest(workspace_root / "docker-compose.yml")
        kafka = docs[0]["services"]["kafka"]

        ports = [str(p) for p in kafka.get("ports", [])]
        assert any("9092" in p for p in ports), f"Port 9092 missing in {ports}"
        assert "healthcheck" in kafka, "kafka service missing healthcheck"
        assert "deeplog-net" in kafka.get("networks", [])
        assert any("kafka_data" in str(v) for v in kafka.get("volumes", []))

    def test_compose_clickhouse_service(self, workspace_root: Path, k8s_mgr: K8sManager):
        """clickhouse analytics sink service is configured correctly."""
        docs = k8s_mgr.load_manifest(workspace_root / "docker-compose.yml")
        clickhouse = docs[0]["services"]["clickhouse"]

        ports = [str(p) for p in clickhouse.get("ports", [])]
        assert any("8123" in p for p in ports), f"Port 8123 missing in {ports}"
        assert "healthcheck" in clickhouse, "clickhouse service missing healthcheck"
        assert "deeplog-net" in clickhouse.get("networks", [])
        assert any("clickhouse_data" in str(v) for v in clickhouse.get("volumes", []))

    def test_compose_prometheus_service(self, workspace_root: Path, k8s_mgr: K8sManager):
        """prometheus metrics exporter service is configured correctly."""
        docs = k8s_mgr.load_manifest(workspace_root / "docker-compose.yml")
        prom = docs[0]["services"]["prometheus"]

        ports = [str(p) for p in prom.get("ports", [])]
        assert any("9090" in p for p in ports), f"Port 9090 missing in {ports}"
        assert "healthcheck" in prom, "prometheus service missing healthcheck"
        assert "deeplog-net" in prom.get("networks", [])
        assert any("prometheus_data" in str(v) for v in prom.get("volumes", []))

    def test_compose_network_and_volumes_declared(self, workspace_root: Path, k8s_mgr: K8sManager):
        """Top-level bridge network and persistent volumes are declared."""
        docs = k8s_mgr.load_manifest(workspace_root / "docker-compose.yml")
        compose_dict = docs[0]

        networks = compose_dict.get("networks", {})
        assert "deeplog-net" in networks
        net_cfg = networks["deeplog-net"]
        if isinstance(net_cfg, dict):
            assert net_cfg.get("driver") in (None, "bridge")

        volumes = compose_dict.get("volumes", {})
        required_vols = {"deeplog_data", "kafka_data", "clickhouse_data", "prometheus_data"}
        assert required_vols.issubset(set(volumes.keys())), f"Missing volumes: {required_vols - set(volumes.keys())}"

    def test_prometheus_configuration_file(self, workspace_root: Path, k8s_mgr: K8sManager):
        """Prometheus config deploy/prometheus/prometheus.yml exists and targets cockpit."""
        prom_cfg = workspace_root / "deploy" / "prometheus" / "prometheus.yml"
        assert prom_cfg.exists(), f"Prometheus config missing at {prom_cfg}"
        docs = k8s_mgr.load_manifest(prom_cfg)
        assert len(docs) >= 1
        cfg = docs[0]
        assert "scrape_configs" in cfg
        job_names = [j.get("job_name") for j in cfg["scrape_configs"]]
        assert "deeplog-cockpit" in job_names


# ==============================================================================
# 3. Academic Paper (.tex) Tests
# ==============================================================================

class TestAcademicPaperTex:
    """Validation test suite for paper/deeplog_infinity_usenix.tex."""

    @pytest.fixture
    def tex_content(self, workspace_root: Path) -> str:
        tex_path = workspace_root / "paper" / "deeplog_infinity_usenix.tex"
        assert tex_path.exists(), f"LaTeX file missing at {tex_path}"
        assert tex_path.is_file()
        return tex_path.read_text(encoding="utf-8")

    def test_tex_file_substantial_length(self, tex_content: str):
        """LaTeX document has substantial publication-grade length (>20,000 characters)."""
        assert len(tex_content) > 20000, f"Paper length ({len(tex_content)}) is too brief"

    def test_tex_usenix_formatting_conventions(self, tex_content: str):
        """LaTeX paper follows USENIX / ACM CCS formatting conventions."""
        assert "\\documentclass" in tex_content
        assert "letterpaper" in tex_content
        assert "twocolumn" in tex_content
        assert "amsmath" in tex_content
        assert "amsthm" in tex_content
        assert "\\begin{document}" in tex_content
        assert "\\end{document}" in tex_content

    def test_tex_required_academic_sections(self, tex_content: str):
        """All standard academic conference research sections are present."""
        assert "\\begin{abstract}" in tex_content and "\\end{abstract}" in tex_content
        assert re.search(r"\\section\{Introduction\}", tex_content)
        assert re.search(r"\\section\{Background and Threat Model\}|\\section\{Threat Model", tex_content)
        assert re.search(r"\\section\{System Architecture\}", tex_content)
        assert re.search(r"\\section\{Mathematical Foundations\}", tex_content)
        assert re.search(r"\\section\{Empirical Benchmark Results\}", tex_content)
        assert re.search(r"\\section\{Discussion\}", tex_content)
        assert re.search(r"\\section\{Related Work\}", tex_content)
        assert re.search(r"\\section\{Conclusion\}", tex_content)

    def test_tex_all_seven_architectural_pillars_covered(self, tex_content: str):
        """Paper explicitly details all 7 Architectural Pillars."""
        assert "Pillar I:" in tex_content and ("eBPF" in tex_content or "HDC" in tex_content)
        assert "Pillar II:" in tex_content and ("OTel" in tex_content or "MELT" in tex_content)
        assert "Pillar III:" in tex_content and ("Physics" in tex_content or "PISNO" in tex_content)
        assert "Pillar IV:" in tex_content and ("AST" in tex_content or "Neuro-Symbolic" in tex_content)
        assert "Pillar V:" in tex_content and ("World Model" in tex_content or "RSSM" in tex_content)
        assert "Pillar VI:" in tex_content and ("Safety" in tex_content or "CBF" in tex_content)
        assert "Pillar VII:" in tex_content and ("Threat Mesh" in tex_content or "Zero-Knowledge" in tex_content)

    def test_tex_mathematical_foundations_qp_cbf_proof(self, tex_content: str):
        """Paper contains Quadratic Programming Control Barrier Function (QP-CBF) safety proof."""
        # Theorem and proof environments
        assert "Control Barrier Function" in tex_content
        assert "Forward Invariance" in tex_content
        assert "L_f h" in tex_content or "L_g h" in tex_content
        assert "\\alpha(h(" in tex_content or "\\alpha" in tex_content
        assert "\\begin{proof}" in tex_content
        assert "Nagumo" in tex_content or "forward invariant" in tex_content

    def test_tex_mathematical_foundations_lyapunov_stability_proof(self, tex_content: str):
        """Paper contains Discrete Lyapunov stability contraction proof."""
        assert "Lyapunov" in tex_content
        assert "\\Delta V(" in tex_content or "V(\\mathbf{e}_{t+1})" in tex_content
        assert "\\kappa" in tex_content
        assert "contraction" in tex_content or "exponentially stable" in tex_content

    def test_tex_mathematical_foundations_hdc_bounds(self, tex_content: str):
        """Paper contains HDC 10,000-bit Hamming distance Chernoff collision bound."""
        assert "10,000" in tex_content or "10{,}000" in tex_content
        assert "Hamming distance" in tex_content
        assert "Chernoff" in tex_content or "Hoeffding" in tex_content
        assert "e^{-50}" in tex_content or "10^{-22}" in tex_content

    def test_tex_mathematical_foundations_jackson_littles_law(self, tex_content: str):
        """Paper contains Open Jackson Network and Little's Law conservation theorem."""
        assert "Jackson" in tex_content
        assert "Little's Law" in tex_content or "Little's formula" in tex_content
        assert "L_i = \\lambda_i W_i" in tex_content or "L = \\lambda W" in tex_content
        assert "(\\mathbf{I} - \\mathbf{P}^T)^{-1}" in tex_content or "I - P^T" in tex_content

    def test_tex_empirical_benchmark_metrics(self, tex_content: str):
        """Paper reports exact empirical benchmark results and hardware latencies."""
        # SOTA F1-Score of 72.23%
        assert "72.23" in tex_content
        # Complete 570k evaluation corpus
        assert "570,204" in tex_content or "570k" in tex_content
        # Sub-microsecond latency 0.29 / 0.293 µs
        assert "0.293" in tex_content or "0.29" in tex_content
        # Chaos Gym blast radius reduction 95.5%
        assert "95.5" in tex_content
        # False alarm reduction 81.2%
        assert "81.2" in tex_content

    def test_tex_structural_environments_balanced(self, tex_content: str):
        """Paper has balanced LaTeX begin/end block environments."""
        env_names = [
            "document",
            "abstract",
            "theorem",
            "lemma",
            "definition",
            "proof",
            "equation",
            "align",
            "table",
            "table*",
            "figure",
            "tabular",
            "itemize",
            "enumerate",
        ]
        for env in env_names:
            begins = len(re.findall(rf"\\begin\{{{re.escape(env)}\}}", tex_content))
            ends = len(re.findall(rf"\\end\{{{re.escape(env)}\}}", tex_content))
            assert begins == ends, f"Mismatched environment '{env}': {begins} begins vs {ends} ends"


# ==============================================================================
# 4. BibTeX Bibliography (.bib) Tests
# ==============================================================================

class TestBibTeXReferences:
    """Validation test suite for paper/references.bib."""

    @pytest.fixture
    def bib_content(self, workspace_root: Path) -> str:
        bib_path = workspace_root / "paper" / "references.bib"
        assert bib_path.exists(), f"BibTeX file missing at {bib_path}"
        assert bib_path.is_file()
        return bib_path.read_text(encoding="utf-8")

    def test_bib_seminal_citations_present(self, bib_content: str):
        """Seminal foundational citations are all defined in BibTeX."""
        seminal_keys = [
            "du2017deeplog",
            "meng2019loganomaly",
            "zhang2019logrobust",
            "he2017drain",
            "ames2019control",
            "kanerva2009hyperdimensional",
            "hafner2020dreamer",
        ]
        for key in seminal_keys:
            pattern = rf"@\w+\s*\{{\s*{re.escape(key)}\s*,"
            assert re.search(pattern, bib_content, re.IGNORECASE), f"Missing seminal citation: '{key}'"

    def test_bib_entries_field_completeness(self, bib_content: str):
        """Every BibTeX entry contains title, author, and year."""
        entry_blocks = re.findall(r"@\w+\s*\{([^,]+),([^@]+)\}", bib_content)
        assert len(entry_blocks) >= 10, f"Expected at least 10 BibTeX entries, got {len(entry_blocks)}"
        for key, body in entry_blocks:
            k = key.strip()
            assert "author" in body.lower(), f"BibTeX entry '{k}' missing 'author'"
            assert "title" in body.lower(), f"BibTeX entry '{k}' missing 'title'"
            assert "year" in body.lower(), f"BibTeX entry '{k}' missing 'year'"

    def test_all_paper_citations_resolve_in_bib(self, workspace_root: Path, bib_content: str):
        """Every citation \\cite{...} in the .tex paper resolves to an entry in .bib."""
        tex_content = (workspace_root / "paper" / "deeplog_infinity_usenix.tex").read_text(encoding="utf-8")

        # Extract all citation keys from \cite{key1, key2}
        cite_matches = re.findall(r"\\cite\{([^}]+)\}", tex_content)
        paper_keys: Set[str] = set()
        for match in cite_matches:
            for k in match.split(","):
                cleaned = k.strip()
                if cleaned:
                    paper_keys.add(cleaned)

        # Extract all defined BibTeX keys
        bib_keys: Set[str] = set(re.findall(r"@\w+\s*\{\s*([^,]+)\s*,", bib_content))

        assert len(paper_keys) >= 10, f"Extracted too few citation keys from paper: {len(paper_keys)}"
        unresolved = paper_keys - bib_keys
        assert not unresolved, f"Unresolved citations found in paper: {unresolved}"
