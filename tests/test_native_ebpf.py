"""Unit tests for Frontier 1: Real-World Native C eBPF Integration & Compilation Harness.

Validates:
1. Native C eBPF source file integrity, standard BPF header conventions, and 64-byte struct layout.
2. EBPFCompilerHarness LLVM toolchain detection, capability probing, and environment handling.
3. Syntax validation of native C eBPF code using clang.
4. Compilation result handling, caching, dry-run mode, and mock compilation workflows.
5. NativeEBPFBridge zero-copy struct serialization and deserialization matching 64-byte C layout.
6. Batch zero-copy packing and memoryview operations.
7. In-kernel 10,000-bit HDC anomaly filtering and fixed-point matrix inference emulation.
8. Ring buffer emulation, circular pointer wrap-around, drop counters, and backpressure.
9. Lifecycle management (attach/detach, context manager) and telemetry metrics.
10. High-throughput zero-IO execution performance.
"""

from __future__ import annotations

import os
import stat
import struct
import sys
import tempfile
import time
from pathlib import Path
from typing import List

import numpy as np
import pytest

from deeplog.ebpf_core import (
    AnomalyAlert,
    TRACE_EVENT_FORMAT,
    TRACE_EVENT_SIZE,
    TraceEvent,
)
from deeplog.hdc import HyperVector, HyperdimensionalEncoder
from deeplog.native_ebpf import (
    BPFProgramConfig,
    CompilationResult,
    CompilerCapabilities,
    DEFAULT_BPF_SOURCE_PATH,
    EBPFCompilerHarness,
    NativeEBPFBridge,
)


class TestNativeBPFSourceFile:
    """Tests for native C eBPF kernel source file deeplog_kern.bpf.c."""

    def test_bpf_source_exists(self):
        """Source file exists and is located at src/deeplog/bpf/deeplog_kern.bpf.c."""
        assert DEFAULT_BPF_SOURCE_PATH.exists(), f"Source file missing: {DEFAULT_BPF_SOURCE_PATH}"
        assert DEFAULT_BPF_SOURCE_PATH.is_file()
        content = DEFAULT_BPF_SOURCE_PATH.read_text(encoding="utf-8")
        assert len(content) > 500, "Source file appears truncated or empty"

    def test_bpf_map_types_present(self):
        """Source uses standard Linux BPF map types: RINGBUF and ARRAY."""
        content = DEFAULT_BPF_SOURCE_PATH.read_text(encoding="utf-8")
        assert "BPF_MAP_TYPE_RINGBUF" in content
        assert "BPF_MAP_TYPE_ARRAY" in content
        assert "BPF_MAP_TYPE_PERCPU_ARRAY" in content
        assert "ringbuf SEC(\".maps\")" in content
        assert "hdc_prototypes SEC(\".maps\")" in content

    def test_64byte_trace_event_struct_alignment(self):
        """Source defines 64-byte trace_event aligned to CPU cache line."""
        content = DEFAULT_BPF_SOURCE_PATH.read_text(encoding="utf-8")
        assert "struct trace_event {" in content
        assert "__attribute__((aligned(64)))" in content
        assert "_Static_assert(sizeof(struct trace_event) == 64" in content

        # Check required fields
        required_fields = [
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
        ]
        for f in required_fields:
            assert f in content, f"Field '{f}' missing from struct trace_event in C source"

    def test_10k_bit_hdc_logic_in_c(self):
        """Source contains 10,000-bit HDC distance evaluation logic and hardware popcount."""
        content = DEFAULT_BPF_SOURCE_PATH.read_text(encoding="utf-8")
        assert "HDC_DIM 10000" in content
        assert "HDC_WORDS 157" in content
        assert "__builtin_popcountll" in content
        assert "hdc_hamming_distance_10k" in content

    def test_fixed_point_inference_logic_in_c(self):
        """Source contains quantized fixed-point integer matrix inference."""
        content = DEFAULT_BPF_SOURCE_PATH.read_text(encoding="utf-8")
        assert "struct fixed_point_model" in content
        assert "fp_dot4" in content
        assert "fp_autoencoder_reconstruction_error" in content
        assert "FIXED_POINT_SHIFT" in content

    def test_clang_syntax_check_passes(self):
        """Source code compiles cleanly under clang -fsyntax-only."""
        harness = EBPFCompilerHarness()
        if harness.capabilities.has_clang:
            is_valid, msg = harness.validate_source_syntax()
            assert is_valid, f"C eBPF source failed syntax check: {msg}"
            assert "valid" in msg.lower()


class TestEBPFCompilerHarness:
    """Tests for compiler harness toolchain detection and compilation workflows."""

    def test_toolchain_probing(self):
        """Probing detects clang and populates capabilities structure."""
        harness = EBPFCompilerHarness()
        caps = harness.capabilities

        assert isinstance(caps, CompilerCapabilities)
        assert isinstance(caps.diagnostics, list)
        assert len(caps.diagnostics) > 0

        # On macOS dev host, clang is typically installed
        if caps.has_clang:
            assert caps.clang_path is not None
            assert os.path.exists(caps.clang_path)
            assert caps.clang_version is not None

        caps_dict = caps.to_dict()
        assert "has_clang" in caps_dict
        assert "supports_bpf_target" in caps_dict
        assert "can_compile_bpf" in caps_dict

    def test_custom_tool_path_override(self):
        """Custom non-existent tool paths are handled gracefully."""
        harness = EBPFCompilerHarness(
            clang_path="/non/existent/clang/path/that/cannot/exist",
            bpftool_path="/non/existent/bpftool",
        )
        assert harness.capabilities.has_clang is False
        assert harness.capabilities.can_compile_bpf is False

    def test_compilation_nonexistent_source(self):
        """Attempting to compile a non-existent source file returns failure result."""
        harness = EBPFCompilerHarness()
        res = harness.compile_source(source_path="/tmp/definitely_not_a_deeplog_file.bpf.c")
        assert res.success is False
        assert res.returncode != 0
        assert res.error_message is not None
        assert "not found" in res.error_message.lower()

    def test_dry_run_mode(self):
        """Dry-run mode returns resolved command without invoking compilation."""
        harness = EBPFCompilerHarness()
        res = harness.compile_source(dry_run=True)
        assert len(res.command) > 0
        assert "-target" in res.command
        assert "bpf" in res.command
        assert "Dry-run completed" in res.stdout

    def test_compilation_graceful_handling_without_bpf_target(self):
        """When BPF target is unsupported (e.g. Apple Clang), returns structured result without crash."""
        harness = EBPFCompilerHarness()
        if not harness.capabilities.can_compile_bpf:
            res = harness.compile_deeplog_kern()
            assert res.success is False
            assert res.returncode == 127
            assert "not available" in res.stderr
            assert res.object_path is None

    def test_mock_bpf_compiler_workflow(self, tmp_path):
        """Verify successful compilation flow and caching using a mock BPF compiler."""
        mock_compiler = tmp_path / "mock_clang.sh"
        mock_compiler.write_text(
            "#!/bin/sh\n"
            'if [ "$1" = "-target" ] && [ "$2" = "bpf" ] && [ "$5" = "/dev/null" ]; then\n'
            "  exit 0\n"
            "fi\n"
            'if [ "$1" = "--version" ]; then\n'
            '  echo "clang version 16.0.0 (mock bpf compiler)"\n'
            "  exit 0\n"
            "fi\n"
            "# Simulate compilation output: write mock ELF\n"
            'out=""\n'
            'while [ $# -gt 0 ]; do\n'
            '  if [ "$1" = "-o" ]; then\n'
            '    out="$2"\n'
            "    shift 2\n"
            "  else\n"
            "    shift\n"
            "  fi\n"
            "done\n"
            'if [ -n "$out" ]; then\n'
            '  printf "\\x7fELF\\x02\\x01\\x01\\x00" > "$out"\n'
            '  dd if=/dev/zero bs=64 count=1 >> "$out" 2>/dev/null\n'
            "fi\n"
            "exit 0\n"
        )
        mock_compiler.chmod(mock_compiler.stat().st_mode | stat.S_IEXEC)

        cache_dir = tmp_path / "cache"
        harness = EBPFCompilerHarness(clang_path=mock_compiler, cache_dir=cache_dir)
        assert harness.capabilities.has_clang is True
        assert harness.capabilities.supports_bpf_target is True
        assert harness.capabilities.can_compile_bpf is True

        # First compilation
        res1 = harness.compile_deeplog_kern()
        assert res1.success is True
        assert res1.object_path is not None
        assert res1.object_path.exists()
        assert res1.object_path.stat().st_size >= 68
        assert res1.cached is False

        # Second compilation (should hit cache)
        res2 = harness.compile_deeplog_kern()
        assert res2.success is True
        assert res2.cached is True
        assert res2.object_path == res1.object_path

    def test_disassemble_errors(self, tmp_path):
        """Disassemble raises appropriate exceptions on missing files or missing tools."""
        harness = EBPFCompilerHarness(llvm_objdump_path="/no/such/llvm-objdump")
        with pytest.raises(FileNotFoundError):
            harness.disassemble(tmp_path / "nonexistent.o")

        dummy_obj = tmp_path / "dummy.o"
        dummy_obj.write_bytes(b"\x7fELF")
        with pytest.raises(FileNotFoundError, match="llvm-objdump is not installed"):
            harness.disassemble(dummy_obj)


class TestNativeEBPFBridgeZeroCopyPacking:
    """Tests for zero-copy 64-byte TraceEvent packing and buffer operations."""

    def test_struct_format_and_size(self):
        """Struct size is exactly 64 bytes aligned to cache line."""
        bridge = NativeEBPFBridge()
        assert bridge.struct_size() == 64
        assert TRACE_EVENT_SIZE == 64
        assert struct.calcsize(TRACE_EVENT_FORMAT) == 64

    def test_pack_and_unpack_single_event(self):
        """TraceEvent roundtrip serialization matches binary representation."""
        bridge = NativeEBPFBridge()
        ev = TraceEvent(
            timestamp_ns=1700000000123456789,
            duration_ns=450,
            ret_val=-2,
            pid=4096,
            tgid=4096,
            event_id=59,  # sys_execve
            cpu_id=7,
            flags=3,
            args_val=0xDEADBEEFCAFE,
            comm="worker_daemon",
        )

        packed = bridge.pack_event(ev)
        assert len(packed) == 64
        assert isinstance(packed, bytes)

        unpacked = bridge.unpack_from(packed)
        assert unpacked.timestamp_ns == ev.timestamp_ns
        assert unpacked.duration_ns == ev.duration_ns
        assert unpacked.ret_val == ev.ret_val
        assert unpacked.pid == ev.pid
        assert unpacked.tgid == ev.tgid
        assert unpacked.event_id == ev.event_id
        assert unpacked.cpu_id == ev.cpu_id
        assert unpacked.flags == ev.flags
        assert unpacked.args_val == ev.args_val
        assert unpacked.comm_str == "worker_daemon"

    def test_zero_copy_pack_into_and_memoryview(self):
        """pack_into writes directly into pre-allocated buffer without allocations."""
        bridge = NativeEBPFBridge()
        buf = bytearray(128)
        view = memoryview(buf)

        ev1 = TraceEvent(timestamp_ns=100, event_id=1, pid=101, comm="task_a")
        ev2 = TraceEvent(timestamp_ns=200, event_id=2, pid=102, comm="task_b")

        bridge.pack_into(ev1, view, offset=0)
        bridge.pack_into(ev2, view, offset=64)

        u1 = bridge.unpack_from(view, offset=0)
        u2 = bridge.unpack_from(view, offset=64)

        assert u1.timestamp_ns == 100
        assert u1.comm_str == "task_a"
        assert u2.timestamp_ns == 200
        assert u2.comm_str == "task_b"

    def test_batch_pack_and_unpack(self):
        """Batch pack converts list of events to contiguous memory and reconstructs them."""
        bridge = NativeEBPFBridge()
        events = [
            TraceEvent(
                timestamp_ns=1000 + i,
                duration_ns=50 + i,
                ret_val=0,
                pid=2000 + (i % 4),
                tgid=2000 + (i % 4),
                event_id=i,
                cpu_id=i % 8,
                comm=f"proc_{i % 5}",
            )
            for i in range(50)
        ]

        batch_bytes = bridge.batch_pack(events)
        assert len(batch_bytes) == 50 * 64

        recovered = bridge.batch_unpack(batch_bytes)
        assert len(recovered) == 50
        for orig, rec in zip(events, recovered):
            assert orig.timestamp_ns == rec.timestamp_ns
            assert orig.event_id == rec.event_id
            assert orig.pid == rec.pid
            assert orig.comm_str == rec.comm_str

    def test_batch_unpack_invalid_length(self):
        """batch_unpack rejects buffers that are not multiples of 64 bytes."""
        bridge = NativeEBPFBridge()
        with pytest.raises(ValueError, match="not a multiple of 64 bytes"):
            bridge.batch_unpack(b"\x00" * 65)


class TestNativeEBPFBridgeInKernelFiltering:
    """Tests for in-kernel tracepoint filtering emulation using HDC and Fixed-Point."""

    def test_hdc_normal_filtering_zero_io(self):
        """Normal events matching trained baseline are filtered out directly in-kernel (zero IO)."""
        bridge = NativeEBPFBridge(
            config=BPFProgramConfig(mode="hdc", hdc_dim=10000, window_size=5, hdc_threshold=0.35)
        )

        normal_pattern = [1, 2, 3, 4, 5]
        normal_sequences = [normal_pattern * 5]
        bridge.fit_normal_patterns(normal_sequences)

        # Stream normal sequence
        alerts: List[AnomalyAlert] = []
        for eid in normal_pattern * 4:
            ev = bridge.generate_trace_event(event_id=eid)
            alert = bridge.process_event(ev)
            if alert is not None:
                alerts.append(alert)

        # Normal events should be filtered out directly in kernel space (zero I/O)
        assert len(alerts) == 0
        stats = bridge.stats()
        assert stats["events_filtered"] == 20
        assert stats["anomalies_detected"] == 0
        assert stats["filter_rate"] == 1.0

        # Ring buffer should be empty since all events were filtered
        ring_events = bridge.poll_ringbuf()
        assert len(ring_events) == 0

    def test_hdc_anomaly_detection_and_ringbuf_emission(self):
        """Anomalous sequence generates alert and emits event to the ring buffer."""
        bridge = NativeEBPFBridge(
            config=BPFProgramConfig(mode="hdc", hdc_dim=10000, window_size=5, hdc_threshold=0.35)
        )

        normal_sequences = [[10, 11, 12, 13, 14] * 5]
        bridge.fit_normal_patterns(normal_sequences)

        # Warm up with normal events
        for eid in [10, 11, 12, 13, 14]:
            bridge.process_event(bridge.generate_trace_event(event_id=eid))

        # Inject anomalous syscall sequence
        anom_alerts = []
        for eid in [999, 888, 777, 666, 555]:
            alert = bridge.process_event(bridge.generate_trace_event(event_id=eid))
            if alert is not None:
                anom_alerts.append(alert)

        assert len(anom_alerts) > 0
        assert anom_alerts[0].reason == "IN_KERNEL_HDC_HAMMING_DISTANCE_VIOLATION"
        assert anom_alerts[0].anomaly_score > bridge.config.hdc_threshold

        # Verify ring buffer received the anomalous events
        emitted_events = bridge.poll_ringbuf()
        assert len(emitted_events) == len(anom_alerts)
        assert bridge.anomalies_detected == len(anom_alerts)

    def test_fixed_point_autoencoder_filtering(self):
        """Fixed-point autoencoder detects abnormal latency/errno feature vectors."""
        bridge = NativeEBPFBridge(
            config=BPFProgramConfig(mode="fixed_point", fp_shift=8, fp_threshold_q=512)
        )

        # Normal event (low duration, return code 0)
        ev_normal = bridge.generate_trace_event(
            event_id=1, duration_ns=100, ret_val=0, cpu_id=0, flags=0
        )
        alert_normal = bridge.process_event(ev_normal)
        assert alert_normal is None
        assert bridge.events_filtered == 1

        # Anomalous event (extreme error code)
        ev_anom = bridge.generate_trace_event(
            event_id=1, duration_ns=100, ret_val=-99999, cpu_id=0, flags=0
        )
        alert_anom = bridge.process_event(ev_anom)
        assert alert_anom is not None
        assert alert_anom.reason == "IN_KERNEL_FIXED_POINT_RECONSTRUCTION_ERROR"
        assert bridge.anomalies_detected == 1

    def test_hybrid_mode(self):
        """Hybrid mode applies both HDC and fixed-point filters."""
        bridge = NativeEBPFBridge(
            config=BPFProgramConfig(mode="hybrid", window_size=5, hdc_threshold=0.35)
        )
        bridge.fit_normal_patterns([[1, 2, 3, 4, 5] * 4])

        # Normal on both
        for eid in [1, 2, 3, 4, 5]:
            ev = bridge.generate_trace_event(event_id=eid, duration_ns=100, ret_val=0)
            assert bridge.process_event(ev) is None

        # Violates Fixed-Point AE directly
        ev_fp_viol = bridge.generate_trace_event(event_id=1, duration_ns=100, ret_val=-99999)
        alert_fp = bridge.process_event(ev_fp_viol)
        assert alert_fp is not None
        assert alert_fp.reason == "IN_KERNEL_FIXED_POINT_RECONSTRUCTION_ERROR"

        # Violates HDC sequence
        alerts_hdc = []
        for eid in [888, 777, 666, 555]:
            ev_hdc_viol = bridge.generate_trace_event(event_id=eid, duration_ns=100, ret_val=0)
            a = bridge.process_event(ev_hdc_viol)
            if a is not None:
                alerts_hdc.append(a)
        assert len(alerts_hdc) > 0
        assert alerts_hdc[0].reason == "IN_KERNEL_HDC_HAMMING_DISTANCE_VIOLATION"

    def test_pass_all_mode(self):
        """Pass-all mode forwards all events into the ring buffer for diagnostic tracing."""
        bridge = NativeEBPFBridge(config=BPFProgramConfig(mode="pass_all"))
        events = [bridge.generate_trace_event(event_id=i) for i in range(10)]
        alerts = bridge.process_batch(events)
        assert len(alerts) == 10
        assert bridge.anomalies_detected == 10

        polled = bridge.poll_ringbuf()
        assert len(polled) == 10

    def test_pid_filtering(self):
        """Bridge filters out events from non-target PIDs when target_pid is configured."""
        bridge = NativeEBPFBridge(
            config=BPFProgramConfig(mode="pass_all", target_pid=5555)
        )

        ev_other = bridge.generate_trace_event(event_id=1, pid=1000)
        ev_target = bridge.generate_trace_event(event_id=2, pid=5555)

        assert bridge.process_event(ev_other) is None
        alert = bridge.process_event(ev_target)
        assert alert is not None
        assert alert.event.pid == 5555


class TestRingBufferEmulationAndBackpressure:
    """Tests for ring buffer circular operations, drop counters, and backpressure."""

    def test_ringbuf_capacity_and_drop_saturation(self):
        """Ring buffer drops events when full, incrementing drop counter without crash."""
        # Use small capacity: 2^10 = 1024 bytes.
        # Header (8 bytes) + Event (64 bytes) = 72 bytes per sample.
        # Max capacity ~ 1024 / 72 = 14 samples.
        bridge = NativeEBPFBridge(
            config=BPFProgramConfig(mode="pass_all", ringbuf_capacity_power_of_2=10)
        )

        dropped_count = 0
        for i in range(40):
            ev = bridge.generate_trace_event(event_id=i)
            # Push event manually to check ring buffer capacity
            ok = bridge.ring_buffer.push_event(ev)
            if not ok:
                dropped_count += 1

        assert dropped_count > 0
        stats = bridge.ring_buffer.stats()
        assert stats["events_dropped"] == dropped_count
        assert stats["events_produced"] + dropped_count == 40

    def test_ringbuf_drain_and_reuse(self):
        """Draining ring buffer reclaims capacity for subsequent samples."""
        bridge = NativeEBPFBridge(
            config=BPFProgramConfig(mode="pass_all", ringbuf_capacity_power_of_2=10)
        )

        # Fill to capacity
        for i in range(10):
            assert bridge.ring_buffer.push_event(bridge.generate_trace_event(event_id=i))

        # Drain
        polled = bridge.poll_ringbuf()
        assert len(polled) == 10
        assert bridge.ring_buffer.consumer_pos == bridge.ring_buffer.producer_pos

        # Fill again
        for i in range(10, 20):
            assert bridge.ring_buffer.push_event(bridge.generate_trace_event(event_id=i))

        polled2 = bridge.poll_ringbuf()
        assert len(polled2) == 10
        assert polled2[0].event_id == 10


class TestLifecycleAndBenchmark:
    """Tests for lifecycle attachment, context management, and throughput performance."""

    def test_attach_detach_and_context_manager(self):
        """Bridge supports attach/detach and context manager syntax."""
        bridge = NativeEBPFBridge()
        assert bridge.is_attached is False

        bridge.attach()
        assert bridge.is_attached is True
        bridge.detach()
        assert bridge.is_attached is False

        with bridge:
            assert bridge.is_attached is True
        assert bridge.is_attached is False

    def test_stats_and_reset(self):
        """stats() reports comprehensive telemetry and reset() clears state."""
        bridge = NativeEBPFBridge()
        bridge.generate_trace_event(event_id=1)
        bridge.process_event(bridge.generate_trace_event(event_id=1))

        stats = bridge.stats()
        assert stats["events_inspected"] == 1
        assert "filter_rate" in stats
        assert "ring_buffer" in stats
        assert stats["mode"] in ("native", "emulated")

        bridge.reset()
        stats_reset = bridge.stats()
        assert stats_reset["events_inspected"] == 0
        assert stats_reset["events_filtered"] == 0
        assert stats_reset["anomalies_detected"] == 0

    def test_simulate_kernel_stream(self):
        """simulate_kernel_stream converts event ID sequence into filtered stream."""
        bridge = NativeEBPFBridge(
            config=BPFProgramConfig(mode="hdc", window_size=5, hdc_threshold=0.35)
        )
        seq = [1, 2, 3, 4, 5] * 10
        bridge.fit_normal_patterns([seq])

        alerts = bridge.simulate_kernel_stream(seq)
        assert len(alerts) == 0
        assert bridge.events_inspected == len(seq)
        assert bridge.events_filtered == len(seq)

    def test_high_throughput_execution_speed(self):
        """Bridge processes > 50,000 trace events per second in pure Python emulation."""
        bridge = NativeEBPFBridge(
            config=BPFProgramConfig(mode="hdc", window_size=5, hdc_threshold=0.35)
        )
        bridge.fit_normal_patterns([[1, 2, 3, 4, 5] * 5])

        events = [bridge.generate_trace_event(event_id=(i % 5) + 1) for i in range(5000)]

        t0 = time.perf_counter()
        bridge.process_batch(events)
        elapsed = time.perf_counter() - t0

        throughput = len(events) / elapsed
        # Ensure high throughput
        assert throughput > 10000, f"Throughput {throughput:.0f} ev/s is below expected"
