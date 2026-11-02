"""
Unit Tests for DeepLog-Omni Pillar III: Code-Grounded Neuro-Symbolic AST Duality
===============================================================================
Validates:
1. ASTShadowGraph:
   - Java lexical/AST parsing (classes, inner classes, methods, constructors, parameters, line bounds, calls, logs).
   - Python native AST parsing (classes, functions, async functions, calls, exceptions).
   - Directed call-graph construction, caller/callee indexing, and multi-hop causal call-stack resolution.
   - Comment/string stripping preserving line indexing.
2. LogToCodeMapper:
   - Bidirectional provenance mapping for all 29 canonical HDFS templates (E1..E29).
   - Forward lookup (Event ID -> CodeLocation).
   - Reverse lookup (File/method/line -> CodeLocations).
   - Raw log text matching to exact code locations.
   - Dynamic custom mapping registration.
   - Anomalous event sequence to call-hierarchy resolution.
3. GitLineageTracer:
   - Blame resolution for canonical HDFS faulting locations.
   - Live git blame execution on workspace repository files.
   - Commit history retrieval and formatted markdown summaries.
4. AutonomousPatchSynthesizer:
   - Multi-pattern root cause inference (Timeout, Resource Leak, Deadlock, Mirror Failover, State NPE).
   - Unified Git diff patch synthesis (.patch) matching Unified Diff specification.
   - In-memory dry-run patch application verification.
   - Automated unit test recommendations and runnable JUnit mock test code generation.
   - Production-grade Pull Request markdown document compilation.
"""

import os
import tempfile
import pytest
import networkx as nx

from deeplog.neuro_symbolic import (
    ASTShadowGraph,
    LogToCodeMapper,
    GitLineageTracer,
    AutonomousPatchSynthesizer,
    CodeLocation,
    CodeSymbol,
    GitBlameInfo,
    PatchProposal,
    CallStackResolution,
    CANONICAL_HDFS_CODE_MAPPINGS,
    CANONICAL_HDFS_SOURCES,
)


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def sample_java_code() -> str:
    """Canonical Java snippet with outer class, inner class, calls, and logs."""
    return """package org.apache.hadoop.hdfs.server.datanode;

import java.io.IOException;
import java.net.Socket;

public class SamplePipeline {
    private Socket socket;

    public SamplePipeline(Socket socket) {
        this.socket = socket;
    }

    public void startStream() throws IOException {
        LOG.info("Starting stream to client");
        processPacket();
    }

    private void processPacket() throws IOException {
        try {
            readPayload();
        } catch (IOException ioe) {
            LOG.warn("Failed reading payload: " + ioe);
            throw ioe;
        }
    }

    private void readPayload() throws IOException {
        LOG.info("Reading payload data bytes");
    }

    public static class WorkerThread extends Thread {
        public void run() {
            LOG.info("WorkerThread execution loop");
        }
    }
}
"""


@pytest.fixture
def sample_python_code() -> str:
    """Canonical Python snippet with class, functions, calls, and logging."""
    return '''"""Sample streaming pipeline module."""
import logging

logger = logging.getLogger(__name__)

class TelemetryCollector:
    """Collects and aggregates distributed streaming logs."""

    def __init__(self, endpoint: str):
        self.endpoint = endpoint

    def poll_events(self, timeout: float = 5.0):
        logger.info(f"Polling events from {self.endpoint}")
        try:
            records = self._fetch_raw()
            return self._parse(records)
        except TimeoutError as exc:
            logger.error(f"Timeout connecting to {self.endpoint}: {exc}")
            raise

    def _fetch_raw(self):
        return ["log1", "log2"]

    def _parse(self, records):
        return len(records)
'''


@pytest.fixture
def hdfs_shadow_graph() -> ASTShadowGraph:
    """Pre-populated shadow graph with canonical HDFS source files."""
    return LogToCodeMapper.get_default_hdfs_shadow_graph()


@pytest.fixture
def default_mapper(hdfs_shadow_graph: ASTShadowGraph) -> LogToCodeMapper:
    """LogToCodeMapper initialized with canonical HDFS shadow graph."""
    return LogToCodeMapper(hdfs_shadow_graph)


@pytest.fixture
def patch_synthesizer(hdfs_shadow_graph: ASTShadowGraph, default_mapper: LogToCodeMapper) -> AutonomousPatchSynthesizer:
    """Patch synthesizer with default mapper and shadow graph."""
    return AutonomousPatchSynthesizer(shadow_graph=hdfs_shadow_graph, mapper=default_mapper)


# ==============================================================================
# 1. ASTShadowGraph Tests
# ==============================================================================


class TestASTShadowGraph:
    """Tests for multi-language AST parsing, line resolution, and call graphs."""

    def test_java_parsing_classes_and_methods(self, sample_java_code: str):
        graph = ASTShadowGraph()
        symbols = graph.parse_java(sample_java_code, file_path="SamplePipeline.java")

        # Must find outer class and inner class
        class_names = [s.name for s in symbols if s.symbol_type == "class"]
        assert "SamplePipeline" in class_names
        assert "WorkerThread" in class_names

        # Must find constructor and methods
        method_names = [s.name for s in symbols if s.symbol_type in ("method", "constructor")]
        assert "SamplePipeline" in method_names  # constructor
        assert "startStream" in method_names
        assert "processPacket" in method_names
        assert "readPayload" in method_names
        assert "run" in method_names

        # Verify method properties
        process_sym = next(s for s in symbols if s.name == "processPacket")
        assert process_sym.symbol_type == "method"
        assert process_sym.class_name == "SamplePipeline"
        assert "readPayload" in process_sym.calls
        assert "IOException" in process_sym.exceptions_handled or "IOException" in process_sym.exceptions_thrown
        assert len(process_sym.log_statements) > 0
        assert process_sym.line_start < process_sym.line_end

    def test_python_parsing_classes_and_functions(self, sample_python_code: str):
        graph = ASTShadowGraph()
        symbols = graph.parse_python(sample_python_code, file_path="pipeline.py")

        class_symbols = [s for s in symbols if s.symbol_type == "class"]
        assert len(class_symbols) == 1
        assert class_symbols[0].name == "TelemetryCollector"

        func_names = [s.name for s in symbols if s.symbol_type in ("method", "function")]
        assert "poll_events" in func_names
        assert "_fetch_raw" in func_names
        assert "_parse" in func_names

        poll_sym = next(s for s in symbols if s.name == "poll_events")
        assert "TimeoutError" in poll_sym.exceptions_handled
        assert "_fetch_raw" in poll_sym.calls
        assert "_parse" in poll_sym.calls
        assert poll_sym.line_start < poll_sym.line_end

    def test_strip_comments_and_strings_line_integrity(self, sample_java_code: str):
        stripped = ASTShadowGraph._strip_comments_and_strings(sample_java_code)
        orig_lines = sample_java_code.splitlines()
        stripped_lines = stripped.splitlines()
        assert len(orig_lines) == len(stripped_lines), "Line count must be strictly preserved"

    def test_call_graph_construction_and_queries(self, sample_java_code: str):
        graph = ASTShadowGraph()
        graph.add_file("SamplePipeline.java", content=sample_java_code, language="java")

        # startStream calls processPacket, processPacket calls readPayload
        callers_of_read = graph.get_callers("readPayload")
        caller_names = [s.name for s in callers_of_read]
        assert "processPacket" in caller_names

        callees_of_start = graph.get_callees("startStream")
        callee_names = [s.name for s in callees_of_start]
        assert "processPacket" in callee_names

    def test_causal_call_stack_path_resolution(self, sample_java_code: str):
        graph = ASTShadowGraph()
        graph.add_file("SamplePipeline.java", content=sample_java_code, language="java")

        # Multi-hop path: startStream -> processPacket -> readPayload
        path = graph.get_causal_call_stack("startStream", "readPayload")
        assert len(path) == 3
        assert any("startStream" in p for p in path)
        assert any("processPacket" in p for p in path)
        assert any("readPayload" in p for p in path)

    def test_find_symbol_at_line(self, sample_java_code: str):
        graph = ASTShadowGraph()
        graph.add_file("SamplePipeline.java", content=sample_java_code, language="java")

        # Find symbol for startStream (approx line 14-16)
        sym = graph.find_symbol_at_line("SamplePipeline.java", 15)
        assert sym is not None
        assert sym.name == "startStream"

    def test_canonical_hdfs_shadow_graph_completeness(self, hdfs_shadow_graph: ASTShadowGraph):
        # Must index DataXceiver, BlockReceiver, FSNamesystem, etc.
        assert "DataXceiver" in hdfs_shadow_graph.class_symbols
        assert "BlockReceiver" in hdfs_shadow_graph.class_symbols
        assert "FSNamesystem" in hdfs_shadow_graph.class_symbols

        write_block_syms = hdfs_shadow_graph.find_methods_by_name("writeBlock")
        assert len(write_block_syms) >= 1
        wb = write_block_syms[0]
        assert wb.class_name == "DataXceiver"
        assert "receiveBlock" in wb.calls or len(wb.calls) > 0

    def test_add_source_tree_indexing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = os.path.join(tmpdir, "module1.py")
            with open(file1, "w") as f:
                f.write("def foo():\n    return 42\n")

            file2 = os.path.join(tmpdir, "Service.java")
            with open(file2, "w") as f:
                f.write("public class Service {\n    public void serve() {}\n}\n")

            graph = ASTShadowGraph()
            count = graph.add_source_tree(tmpdir)
            assert count == 2
            assert graph.find_methods_by_name("foo") != []
            assert graph.find_methods_by_name("serve") != []


# ==============================================================================
# 2. LogToCodeMapper Tests
# ==============================================================================


class TestLogToCodeMapper:
    """Tests for bidirectional log-to-code mapping and call stack resolution."""

    def test_all_29_hdfs_mappings_exist_and_valid(self, default_mapper: LogToCodeMapper):
        mappings = default_mapper.get_canonical_hdfs_mappings()
        assert len(mappings) == 29

        for i in range(1, 30):
            event_id = f"E{i}"
            assert event_id in mappings
            loc = mappings[event_id]
            assert isinstance(loc, CodeLocation)
            assert loc.event_id == event_id
            assert loc.file_path.endswith(".java")
            assert loc.line_number > 0
            assert loc.class_name != ""
            assert loc.method_name != ""
            assert loc.log_template != ""

    def test_forward_mapping_normalization(self, default_mapper: LogToCodeMapper):
        loc_str = default_mapper.map_event_to_code("E7")
        loc_int = default_mapper.map_event_to_code(7)
        loc_raw = default_mapper.map_event_to_code("7")

        assert loc_str.method_name == "writeBlock"
        assert loc_str.class_name == "DataXceiver"
        assert loc_str.line_number == 315
        assert loc_int == loc_str
        assert loc_raw == loc_str
        assert loc_str.is_fault_indicator is True

    def test_reverse_mapping_by_file_and_line(self, default_mapper: LogToCodeMapper):
        # E7 is DataXceiver.java at line 315
        results = default_mapper.map_code_to_events("DataXceiver.java", line_number=315)
        assert len(results) >= 1
        assert any(r.event_id == "E7" for r in results)

    def test_reverse_mapping_by_method(self, default_mapper: LogToCodeMapper):
        # DataXceiver#writeBlock hosts E5, E6, E7
        results = default_mapper.map_code_to_events("DataXceiver.java", method_name="writeBlock")
        event_ids = {r.event_id for r in results}
        assert "E5" in event_ids
        assert "E6" in event_ids
        assert "E7" in event_ids

    def test_raw_log_matching(self, default_mapper: LogToCodeMapper):
        raw_log = "081109 203518 143 INFO dfs.DataNode$DataXceiver: Receiving block blk_-1608999832283419080 src: /10.250.19.102:54106 dest: /10.250.19.102:50010"
        loc = default_mapper.map_log_message_to_code(raw_log)
        assert loc is not None
        assert loc.event_id == "E5"
        assert loc.method_name == "writeBlock"

    def test_custom_mapping_registration(self, default_mapper: LogToCodeMapper):
        custom_loc = CodeLocation(
            event_id="CUSTOM_99",
            file_path="src/gateway/Router.py",
            class_name="Router",
            method_name="forward_request",
            line_number=88,
            log_template="Route request [*] failed with timeout",
            log_level="ERROR",
            is_fault_indicator=True,
        )
        default_mapper.register_mapping("CUSTOM_99", custom_loc)

        retrieved = default_mapper.map_event_to_code("CUSTOM_99")
        assert retrieved == custom_loc
        assert retrieved.location_str == "Router.py:forward_request:88"

    def test_map_sequence_to_call_stack(self, default_mapper: LogToCodeMapper):
        # Trace: E5 (Receiving block) -> E6 (Received block) -> E7 (writeBlock exception)
        trace = ["E5", "E6", "E7"]
        resolution = default_mapper.map_sequence_to_call_stack(trace)

        assert isinstance(resolution, CallStackResolution)
        assert len(resolution.call_hierarchy) == 3
        assert resolution.root_cause_location is not None
        assert resolution.root_cause_location.event_id == "E7"
        assert 2 in resolution.anomalous_indices
        assert "DataXceiver#writeBlock" in resolution.causal_chain
        assert "Faulting Origin" in resolution.diagnostic_summary


# ==============================================================================
# 3. GitLineageTracer Tests
# ==============================================================================


class TestGitLineageTracer:
    """Tests for commit history, blame resolution, and summary formatting."""

    def test_canonical_hdfs_blame_resolution(self):
        tracer = GitLineageTracer()
        blame = tracer.trace_blame("org/apache/hadoop/hdfs/server/datanode/DataXceiver.java", 315)

        assert isinstance(blame, GitBlameInfo)
        assert blame.author_name == "Dhruba Borthakur"
        assert "dhruba@apache.org" in blame.author_email
        assert blame.pr_reference == "HDFS-364"
        assert "socket timeouts" in blame.commit_message.lower()

    def test_live_workspace_git_blame(self):
        # Run blame on existing repository file
        tracer = GitLineageTracer()
        blame = tracer.trace_blame("src/deeplog/deeplog.py", 2)

        assert isinstance(blame, GitBlameInfo)
        assert blame.commit_hash != ""
        assert blame.author_name != ""

    def test_commit_history_retrieval(self):
        tracer = GitLineageTracer()
        history = tracer.trace_history("src/deeplog/deeplog.py", max_commits=3)

        assert len(history) >= 1
        assert "hash" in history[0]
        assert "author" in history[0]

    def test_format_blame_summary(self):
        tracer = GitLineageTracer()
        blame = GitBlameInfo(
            commit_hash="a4e18f2d658c97354b032d8471b058fe8b109e2a",
            author_name="Dhruba Borthakur",
            author_email="dhruba@apache.org",
            commit_date="2008-08-14 18:22:15 -0700",
            commit_message="HDFS-364: Handle socket timeouts gracefully",
            file_path="DataXceiver.java",
            line_number=315,
            pr_reference="HDFS-364",
        )
        summary = tracer.format_blame_summary(blame)
        assert "Dhruba Borthakur" in summary
        assert "HDFS-364" in summary
        assert "a4e18f2d65" in summary


# ==============================================================================
# 4. AutonomousPatchSynthesizer Tests
# ==============================================================================


class TestAutonomousPatchSynthesizer:
    """Tests for root cause inference, unified diff generation, and dry-run application."""

    def test_infer_root_cause_categories(self, patch_synthesizer: AutonomousPatchSynthesizer):
        # Timeout
        cat1, _ = patch_synthesizer.infer_root_cause(["E5", "E29"])
        assert cat1 == "TIMEOUT_EXHAUSTION"

        # Deadlock / Interruption
        cat2, _ = patch_synthesizer.infer_root_cause(["E5", "E8"])
        assert cat2 == "CONCURRENCY_DEADLOCK"

        # Mirror Failover
        cat3, _ = patch_synthesizer.infer_root_cause(["E5", "E12"])
        assert cat3 == "IO_MIRROR_FAILOVER"

        # State NPE
        cat4, _ = patch_synthesizer.infer_root_cause(["E20"])
        assert cat4 == "STATE_VALIDATION_NPE"

        # Resource leak
        cat5, _ = patch_synthesizer.infer_root_cause(["E5", "E6", "E7"])
        assert cat5 == "RESOURCE_LEAK_SOCKET"

    def test_synthesize_patch_timeout_guard(self, patch_synthesizer: AutonomousPatchSynthesizer):
        # Trace with writeBlock temporal socket timeout
        proposal = patch_synthesizer.synthesize_patch(
            event_sequence=["E5", "E6", "E7"],
            temporal_anomaly=True,
            incident_id="INC-20261102-TIMEOUT",
        )

        assert isinstance(proposal, PatchProposal)
        assert proposal.category == "TIMEOUT_EXHAUSTION"
        assert "DataXceiver" in proposal.target_class
        assert proposal.verify_patch_syntax() is True

        # Check unified diff contents
        assert "--- a/" in proposal.patch_diff
        assert "+++ b/" in proposal.patch_diff
        assert "+ " in proposal.patch_diff or "+" in proposal.patch_diff
        assert "s.setSoTimeout" in proposal.patch_diff

        # Check dry-run verification
        assert proposal.dry_run_verified is True

    def test_synthesize_patch_resource_leak(self, patch_synthesizer: AutonomousPatchSynthesizer):
        # Trace with E7 exception
        proposal = patch_synthesizer.synthesize_patch(
            event_sequence=["E5", "E6", "E7"],
            incident_id="INC-20261102-LEAK",
        )

        assert proposal.category == "RESOURCE_LEAK_SOCKET"
        assert proposal.verify_patch_syntax() is True
        assert "s.close()" in proposal.patch_diff
        assert proposal.dry_run_verified is True

    def test_synthesize_patch_deadlock_guard(self, patch_synthesizer: AutonomousPatchSynthesizer):
        # Trace with E8 packet responder interrupted
        source = CANONICAL_HDFS_SOURCES["org/apache/hadoop/hdfs/server/datanode/BlockReceiver.java"]
        proposal = patch_synthesizer.synthesize_patch(
            event_sequence=["E5", "E8"],
            source_code=source,
            file_path="org/apache/hadoop/hdfs/server/datanode/BlockReceiver.java",
            incident_id="INC-20261102-DEADLOCK",
        )

        assert proposal.category == "CONCURRENCY_DEADLOCK"
        assert proposal.verify_patch_syntax() is True
        assert "isInterrupted()" in proposal.patch_diff
        assert proposal.dry_run_verified is True

    def test_unit_test_code_generation(self, patch_synthesizer: AutonomousPatchSynthesizer):
        loc = CodeLocation(
            event_id="E7",
            file_path="DataXceiver.java",
            class_name="DataXceiver",
            method_name="writeBlock",
            line_number=315,
            log_template="writeBlock exception",
        )
        test_code, desc = patch_synthesizer.generate_unit_test(loc, "TIMEOUT_EXHAUSTION")

        assert "public class TestDataXceiverAutopatch" in test_code
        assert "testWriteBlockTimeoutExhaustion" in test_code
        assert "@Test" in test_code
        assert "mock(Socket.class)" in test_code

    def test_pr_proposal_markdown_generation(self, patch_synthesizer: AutonomousPatchSynthesizer):
        proposal = patch_synthesizer.synthesize_patch(
            event_sequence=["E5", "E6", "E7"],
            incident_id="INC-PR-001",
        )

        pr = proposal.pr_description
        assert "# [DeepLog-Omni Autopatch]" in pr
        assert "INC-PR-001" in pr
        assert "Git Blame Lineage" in pr
        assert "Proposed Git Diff Patch" in pr
        assert "Automated Unit Test Plan" in pr
        assert "Risk & Impact Assessment" in pr

    def test_patch_file_save_and_verify(self, patch_synthesizer: AutonomousPatchSynthesizer):
        proposal = patch_synthesizer.synthesize_patch(
            event_sequence=["E5", "E6", "E7"],
            incident_id="INC-DISK-001",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            patch_file = os.path.join(tmpdir, "fix.patch")
            pr_file = os.path.join(tmpdir, "proposal.md")

            proposal.save_patch(patch_file)
            proposal.save_pr(pr_file)

            assert os.path.exists(patch_file)
            assert os.path.exists(pr_file)

            with open(patch_file, "r") as f:
                content = f.read()
            assert content == proposal.patch_diff

    def test_in_memory_unified_diff_application_exactness(self):
        orig_source = (
            "public void serve() {\n"
            "    LOG.info(\"Serving\");\n"
            "    doWork();\n"
            "}\n"
        )
        patched_target = (
            "public void serve() {\n"
            "    LOG.info(\"Serving\");\n"
            "    try {\n"
            "        doWork();\n"
            "    } catch (Exception e) {\n"
            "        LOG.error(\"Error\", e);\n"
            "    }\n"
            "}\n"
        )

        import difflib
        diff = "".join(
            difflib.unified_diff(
                orig_source.splitlines(keepends=True),
                patched_target.splitlines(keepends=True),
                fromfile="a/Server.java",
                tofile="b/Server.java",
            )
        )

        applied = AutonomousPatchSynthesizer.apply_unified_diff(orig_source, diff)
        assert applied == patched_target
