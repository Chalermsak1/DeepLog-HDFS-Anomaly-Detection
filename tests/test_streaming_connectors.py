"""Comprehensive Unit Tests for DeepLog Pillar C:
- eBPF CO-RE Builder & Toolchain Orchestrator (bpf_core_builder.py)
- Streaming Connectors & Columnar Log Sinks (streaming_connectors.py)
- ONNX Model Exporter & Numerical Parity Verifier (onnx_exporter.py)
"""

import os
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from deeplog.bpf_core_builder import (
    BTF_MAGIC,
    BPFCOREBuilder,
    BPFSkeletonStub,
    BTFHeader,
    BTFInfo,
    COREBuildConfig,
    SkeletonResult,
)
from deeplog.deeplog import DeepLog
from deeplog.dual_head import DeepLogDualHead
from deeplog.onnx_exporter import (
    DeepLogDualHeadONNXAdapter,
    DeepLogONNXAdapter,
    ONNXModelExporter,
    ParityVerificationResult,
)
from deeplog.streaming_connectors import (
    BatchStreamBuffer,
    BufferOverflowError,
    BufferTimeoutError,
    ClickHouseSink,
    ClickHouseSinkConfig,
    ConsumerRecord,
    IngestionError,
    InMemoryBroker,
    KafkaLogConsumer,
    KafkaLogProducer,
    ProducerRecord,
    StreamingBrokerConfig,
    StreamingConnectorError,
)


# ===========================================================================
# 1. BPF CO-RE Builder Tests
# ===========================================================================

class TestBPFCOREBuilder:
    """Unit tests for BPFCOREBuilder, BTF inspection, vmlinux.h, and skeletons."""

    def test_platform_detection_and_config(self):
        builder = BPFCOREBuilder()
        assert isinstance(builder.is_linux(), bool)
        assert isinstance(builder.is_macos(), bool)
        assert isinstance(builder.is_bsd(), bool)

        cfg = COREBuildConfig(arch="arm64", opt_level="-O3", debug_info=True)
        b2 = BPFCOREBuilder(cfg)
        assert b2.config.arch == "arm64"
        d = cfg.to_dict()
        assert d["arch"] == "arm64"
        assert d["opt_level"] == "-O3"

    def test_btf_header_and_missing_inspection(self):
        builder = BPFCOREBuilder()
        fake_path = Path("/nonexistent/path/to/vmlinux")
        info = builder.inspect_btf(fake_path)
        assert not info.exists
        assert not info.has_core_support
        assert len(info.diagnostics) > 0
        assert not builder.check_btf_support(fake_path)

    def test_synthetic_btf_creation_and_parsing(self, tmp_path):
        builder = BPFCOREBuilder()
        btf_file = tmp_path / "synthetic_vmlinux.btf"
        created = builder.create_synthetic_btf(btf_file, num_types=32)

        assert created.exists()
        assert created.stat().st_size > 24

        info = builder.inspect_btf(created)
        assert info.exists
        assert info.is_valid
        assert info.has_core_support
        assert info.header is not None
        assert info.header.magic == BTF_MAGIC
        assert info.types_count_estimate > 0
        assert builder.check_btf_support(created)

        info_dict = info.to_dict()
        assert info_dict["is_valid"]
        assert "synthetic_vmlinux.btf" in info_dict["path"]

    def test_corrupt_btf_handling(self, tmp_path):
        builder = BPFCOREBuilder()
        corrupt_file = tmp_path / "corrupt.btf"
        corrupt_file.write_bytes(b"\x00\x00\x00\x00" * 8)

        info = builder.inspect_btf(corrupt_file)
        assert info.exists
        assert not info.is_valid
        assert not info.has_core_support
        assert any("Invalid BTF header magic" in d for d in info.diagnostics)

    def test_fallback_vmlinux_generation(self, tmp_path):
        builder = BPFCOREBuilder()
        header_path = tmp_path / "vmlinux.h"
        res_path = builder.probe_vmlinux_header(header_path, force_generate=True)

        assert res_path.exists()
        content = res_path.read_text(encoding="utf-8")
        assert "__VMLINUX_H__" in content
        assert "struct trace_event" in content
        assert "struct pt_regs" in content
        assert "struct task_struct" in content
        assert "enum bpf_map_type" in content
        assert "__attribute_preserve_access_index" in content

        # Cached call should return same path
        cached_path = builder.probe_vmlinux_header(header_path)
        assert cached_path == res_path

    def test_fallback_skeleton_generation(self, tmp_path):
        builder = BPFCOREBuilder()
        obj_path = tmp_path / "deeplog_kern.bpf.o"
        obj_path.touch()

        skel_res = builder.generate_skeleton(obj_path, skeleton_name="deeplog_kern")
        assert skel_res.success
        assert skel_res.skeleton_name == "deeplog_kern"
        assert skel_res.skeleton_path.exists()
        code = skel_res.header_code
        assert "deeplog_kern_bpf__open" in code
        assert "deeplog_kern_bpf__load" in code
        assert "deeplog_kern_bpf__attach" in code
        assert "deeplog_kern_bpf__detach" in code
        assert "deeplog_kern_bpf__destroy" in code

        res_dict = skel_res.to_dict()
        assert res_dict["success"]
        assert res_dict["skeleton_name"] == "deeplog_kern"

    def test_python_skeleton_stub_lifecycle(self):
        builder = BPFCOREBuilder()
        stub = builder.get_python_skeleton("test_skel")
        assert not stub.is_open
        assert not stub.is_loaded
        assert not stub.is_attached

        stub.open()
        assert stub.is_open
        stub.load()
        assert stub.is_loaded
        stub.attach()
        assert stub.is_attached

        events = stub.poll_events(max_events=5)
        assert len(events) == 5
        assert events[0]["seq"] == 1
        assert "timestamp_ns" in events[0]

        stub.detach()
        assert not stub.is_attached
        stub.destroy()
        assert not stub.is_open
        assert not stub.is_loaded

    def test_cross_platform_report(self):
        builder = BPFCOREBuilder()
        report = builder.get_cross_platform_report()
        assert "platform" in report
        assert "is_linux" in report
        assert "btf_available" in report
        assert "fallback_active" in report


# ===========================================================================
# 2. Streaming Broker Configuration Tests
# ===========================================================================

class TestStreamingBrokerConfig:
    """Unit tests for StreamingBrokerConfig validation and serialization."""

    def test_default_config_validity(self):
        cfg = StreamingBrokerConfig()
        cfg.validate()
        assert cfg.bootstrap_servers == ["localhost:9092"]
        assert cfg.auto_offset_reset == "latest"
        assert cfg.in_memory_mode is True

    def test_invalid_config_raises(self):
        with pytest.raises(ValueError, match="bootstrap_servers"):
            StreamingBrokerConfig(bootstrap_servers=[]).validate()

        with pytest.raises(ValueError, match="Invalid auto_offset_reset"):
            StreamingBrokerConfig(auto_offset_reset="invalid_reset").validate()

        with pytest.raises(ValueError, match="max_poll_records"):
            StreamingBrokerConfig(max_poll_records=0).validate()

        with pytest.raises(ValueError, match="batch_size"):
            StreamingBrokerConfig(batch_size=-1).validate()

    def test_to_dict_and_from_dict_roundtrip(self):
        cfg = StreamingBrokerConfig(
            bootstrap_servers=["kafka1:9092", "kafka2:9092"],
            client_id="custom-client",
            topics=["logs-a", "logs-b"],
            max_poll_records=250,
            auto_commit_interval_ms=1000,
        )
        data = cfg.to_dict()
        assert data["client_id"] == "custom-client"
        assert len(data["topics"]) == 2

        cfg2 = StreamingBrokerConfig.from_dict(data)
        assert cfg2.client_id == "custom-client"
        assert cfg2.max_poll_records == 250
        assert cfg2.bootstrap_servers == ["kafka1:9092", "kafka2:9092"]


# ===========================================================================
# 3. Kafka Producer & Consumer Streaming Tests
# ===========================================================================

class TestKafkaStreaming:
    """Unit tests for KafkaLogProducer and KafkaLogConsumer in-memory mock protocol."""

    def setup_method(self):
        InMemoryBroker.reset()

    def test_producer_record_helpers(self):
        rec1 = ProducerRecord(topic="t1", value={"k": "v"})
        assert rec1.value_as_json() == {"k": "v"}
        assert '{"k": "v"}' in rec1.value_as_str()

        rec2 = ProducerRecord(topic="t1", value="hello world")
        assert rec2.value_as_str() == "hello world"
        assert rec2.value_as_bytes() == b"hello world"

    def test_producer_and_consumer_roundtrip(self):
        cfg = StreamingBrokerConfig(
            topics=["test-topic"],
            auto_offset_reset="earliest",
        )
        producer = KafkaLogProducer(cfg)
        rec1 = producer.send("test-topic", {"event_id": 1, "msg": "first"})
        rec2 = producer.send("test-topic", {"event_id": 2, "msg": "second"})

        assert rec1.offset == 0
        assert rec2.offset == 1

        consumer = KafkaLogConsumer(cfg)
        records = consumer.poll(timeout_ms=100, max_records=10)
        assert len(records) == 2
        assert records[0].value_as_json()["event_id"] == 1
        assert records[1].value_as_json()["event_id"] == 2

        # Poll again should return empty
        records2 = consumer.poll(timeout_ms=10)
        assert len(records2) == 0

        metrics = producer.get_metrics()
        assert metrics["records_sent"] == 2
        assert metrics["bytes_sent"] > 0

    def test_producer_key_partitioning(self):
        cfg = StreamingBrokerConfig(topics=["keyed-topic"], auto_offset_reset="earliest")
        producer = KafkaLogProducer(cfg)

        rec_a1 = producer.send("keyed-topic", "msg1", key="device-A")
        rec_a2 = producer.send("keyed-topic", "msg2", key="device-A")
        assert rec_a1.partition == rec_a2.partition

        rec_manual = producer.send("keyed-topic", "msg3", partition=3)
        assert rec_manual.partition == 3

    def test_consumer_manual_assignment(self):
        cfg = StreamingBrokerConfig(auto_offset_reset="earliest")
        producer = KafkaLogProducer(cfg)
        producer.send("assigned-topic", "p0_msg", partition=0)
        producer.send("assigned-topic", "p1_msg", partition=1)

        consumer = KafkaLogConsumer(cfg)
        consumer.assign({"assigned-topic": [1]})
        records = consumer.poll(timeout_ms=100)

        assert len(records) == 1
        assert records[0].partition == 1
        assert records[0].value_as_str() == "p1_msg"

    def test_consumer_offset_commit_and_resume(self):
        cfg = StreamingBrokerConfig(
            topics=["commit-topic"],
            group_id="test-grp",
            auto_offset_reset="earliest",
            enable_auto_commit=False,
        )
        producer = KafkaLogProducer(cfg)
        for i in range(5):
            producer.send("commit-topic", f"msg_{i}")

        c1 = KafkaLogConsumer(cfg)
        batch = c1.poll(max_records=2)
        assert len(batch) == 2
        c1.commit()
        c1.close()

        # New consumer with same group should resume from offset 2
        c2 = KafkaLogConsumer(cfg)
        batch2 = c2.poll(max_records=10)
        assert len(batch2) == 3
        assert batch2[0].value_as_str() == "msg_2"
        c2.close()

    def test_consumer_seek_operations(self):
        cfg = StreamingBrokerConfig(topics=["seek-topic"], auto_offset_reset="earliest")
        producer = KafkaLogProducer(cfg)
        for i in range(10):
            producer.send("seek-topic", f"val_{i}", partition=0)

        consumer = KafkaLogConsumer(cfg)
        consumer.seek("seek-topic", partition=0, offset=7)
        assert consumer.position("seek-topic", 0) == 7

        records = consumer.poll(max_records=5)
        assert len(records) == 3
        assert records[0].value_as_str() == "val_7"

        consumer.seek_to_beginning("seek-topic", 0)
        assert consumer.position("seek-topic", 0) == 0

        consumer.seek_to_end("seek-topic", 0)
        assert consumer.position("seek-topic", 0) == 10

    def test_closed_producer_and_consumer_raise(self):
        cfg = StreamingBrokerConfig(topics=["t"])
        producer = KafkaLogProducer(cfg)
        producer.close()
        with pytest.raises(StreamingConnectorError, match="Producer is closed"):
            producer.send("t", "data")

        consumer = KafkaLogConsumer(cfg)
        consumer.close()
        with pytest.raises(StreamingConnectorError, match="Consumer is closed"):
            consumer.poll()


# ===========================================================================
# 4. BatchStreamBuffer & Backpressure Tests
# ===========================================================================

class TestBatchStreamBufferBackpressure:
    """Unit tests for sliding window queue and backpressure handling."""

    def test_buffer_fifo_order(self):
        buf = BatchStreamBuffer(max_capacity=10)
        buf.push(10)
        buf.push(20)
        buf.push(30)
        assert len(buf) == 3
        assert buf.pop() == 10
        assert buf.pop() == 20
        assert buf.pop() == 30
        assert buf.is_empty()

    def test_reject_policy_overflow(self):
        buf = BatchStreamBuffer(max_capacity=3, overflow_policy="reject")
        buf.push("a")
        buf.push("b")
        buf.push("c")
        assert buf.is_full()

        with pytest.raises(BufferOverflowError, match="capacity"):
            buf.push("d")

        metrics = buf.get_metrics()
        assert metrics["total_rejected"] == 1
        assert metrics["total_enqueued"] == 3

    def test_drop_oldest_policy_overflow(self):
        buf = BatchStreamBuffer(max_capacity=3, overflow_policy="drop_oldest")
        buf.push(1)
        buf.push(2)
        buf.push(3)
        assert buf.snapshot() == [1, 2, 3]

        buf.push(4)
        assert buf.snapshot() == [2, 3, 4]
        assert buf.total_dropped == 1

        buf.push(5)
        assert buf.snapshot() == [3, 4, 5]
        assert buf.total_dropped == 2
        assert len(buf) == 3

    def test_block_policy_timeout(self):
        buf = BatchStreamBuffer(max_capacity=2, overflow_policy="block", timeout_seconds=0.1)
        buf.push("x")
        buf.push("y")

        with pytest.raises(BufferTimeoutError, match="Timed out"):
            buf.push("z")

    def test_block_policy_unblocks_when_space_freed(self):
        buf = BatchStreamBuffer(max_capacity=2, overflow_policy="block", timeout_seconds=2.0)
        buf.push("item1")
        buf.push("item2")

        pushed = threading.Event()

        def background_pusher():
            buf.push("item3")
            pushed.set()

        t = threading.Thread(target=background_pusher)
        t.start()

        # Short sleep to ensure pusher is blocked
        time.sleep(0.05)
        assert not pushed.is_set()

        # Free space
        popped = buf.pop()
        assert popped == "item1"

        t.join(timeout=1.0)
        assert pushed.is_set()
        assert buf.pop() == "item2"
        assert buf.pop() == "item3"

    def test_batch_operations_and_drain(self):
        buf = BatchStreamBuffer(max_capacity=20, batch_size=4)
        buf.push_batch(list(range(10)))
        assert len(buf) == 10

        batch1 = buf.pop_batch()
        assert batch1 == [0, 1, 2, 3]

        batch2 = buf.pop_batch(max_items=3)
        assert batch2 == [4, 5, 6]

        remaining = buf.drain()
        assert remaining == [7, 8, 9]
        assert buf.is_empty()

    def test_peek_and_clear(self):
        buf = BatchStreamBuffer(max_capacity=10)
        buf.push_batch(["a", "b", "c", "d"])
        assert buf.peek(2) == ["a", "b"]
        assert len(buf) == 4
        buf.clear()
        assert len(buf) == 0


# ===========================================================================
# 5. ClickHouse Columnar Sink Tests
# ===========================================================================

class TestClickHouseColumnarSink:
    """Unit tests for ClickHouseSink columnar batch ingestion, partitioning, and retry backoff."""

    def test_sink_write_and_manual_flush(self):
        sink = ClickHouseSink(ClickHouseSinkConfig(batch_size=100))
        sink.write({
            "timestamp": 1700000000,
            "block_id": "blk_101",
            "event_id": 5,
            "score": 0.05,
            "is_anomaly": 0,
            "raw_log": "Starting block update",
        })
        sink.write({
            "timestamp": 1700000010,
            "block_id": "blk_102",
            "event_id": 6,
            "score": 0.88,
            "is_anomaly": 1,
            "raw_log": "Block verification failed",
        })

        assert len(sink._buffer) == 2
        flushed = sink.flush()
        assert flushed == 2
        assert len(sink._buffer) == 0
        assert sink.total_rows_ingested == 2

        all_rows = sink.query()
        assert len(all_rows) == 2
        assert all_rows[0]["block_id"] == "blk_101"
        assert all_rows[1]["is_anomaly"] == 1

    def test_auto_flush_on_batch_size(self):
        sink = ClickHouseSink(ClickHouseSinkConfig(batch_size=3))
        sink.write({"timestamp": 1700000000, "block_id": "b1", "event_id": 1})
        sink.write({"timestamp": 1700000001, "block_id": "b2", "event_id": 2})
        assert sink.total_rows_ingested == 0

        # 3rd row triggers auto-flush
        sink.write({"timestamp": 1700000002, "block_id": "b3", "event_id": 3})
        assert sink.total_rows_ingested == 3
        assert len(sink._buffer) == 0

    def test_schema_partitioning_by_date(self):
        sink = ClickHouseSink()
        # 2026-01-15 -> 1768435200
        # 2026-02-20 -> 1771545600
        sink.write({"timestamp": 1768435200, "block_id": "jan_1", "event_id": 10})
        sink.write({"timestamp": 1771545600, "block_id": "feb_1", "event_id": 20})
        sink.flush()

        partitions = sink.get_partitions()
        assert len(partitions) == 2
        assert "202601" in partitions
        assert "202602" in partitions

        jan_rows = sink.query(partition="202601")
        assert len(jan_rows) == 1
        assert jan_rows[0]["block_id"] == "jan_1"

        metrics = sink.get_partition_metrics()
        assert metrics["202601"]["row_count"] == 1
        assert metrics["202602"]["row_count"] == 1

    def test_columnar_array_conversion(self):
        sink = ClickHouseSink()
        records = [
            {"timestamp": 1700000000 + i, "block_id": f"blk_{i}", "event_id": i * 10, "score": float(i) * 0.1, "is_anomaly": i % 2}
            for i in range(5)
        ]
        sink.write_batch(records)
        sink.flush()

        cols = sink.get_columnar_arrays()
        assert "event_id" in cols
        assert "score" in cols
        assert isinstance(cols["event_id"], np.ndarray)
        assert isinstance(cols["score"], np.ndarray)
        assert cols["score"].dtype == np.float32
        assert len(cols["event_id"]) == 5
        assert np.array_equal(cols["event_id"], np.array([0, 10, 20, 30, 40]))

    def test_retry_backoff_on_transient_failure(self):
        cfg = ClickHouseSinkConfig(
            max_retries=3,
            initial_backoff_seconds=0.01,
            backoff_factor=1.5,
        )
        sink = ClickHouseSink(cfg)
        sink.write({"timestamp": 1700000000, "block_id": "retry_blk", "event_id": 1})

        # Inject 2 transient failures
        sink._fail_next_n_flushes = 2
        flushed = sink.flush()

        assert flushed == 1
        assert sink.total_retries == 2
        assert sink.total_rows_ingested == 1

    def test_retry_exhaustion_raises_ingestion_error(self):
        cfg = ClickHouseSinkConfig(
            max_retries=2,
            initial_backoff_seconds=0.01,
            backoff_factor=1.5,
        )
        sink = ClickHouseSink(cfg)
        sink.write({"timestamp": 1700000000, "block_id": "fail_blk", "event_id": 1})

        # Inject 4 failures, exceeding max_retries of 2
        sink._fail_next_n_flushes = 4
        with pytest.raises(IngestionError, match="Failed to ingest"):
            sink.flush()

    def test_ddl_generation_and_predicate_filtering(self):
        sink = ClickHouseSink()
        ddl = sink.create_table_ddl()
        assert "CREATE TABLE IF NOT EXISTS `deeplog`.`log_events`" in ddl
        assert "ENGINE = MergeTree()" in ddl
        assert "PARTITION BY" in ddl
        assert "ORDER BY (`timestamp`, `block_id`, `event_id`)" in ddl

        sink.write_batch([
            {"timestamp": 1700000000, "block_id": "b1", "event_id": 1, "is_anomaly": 0},
            {"timestamp": 1700000001, "block_id": "b2", "event_id": 2, "is_anomaly": 1},
            {"timestamp": 1700000002, "block_id": "b3", "event_id": 3, "is_anomaly": 1},
        ])
        sink.flush()

        anomalies = sink.query(predicate=lambda r: r.get("is_anomaly") == 1)
        assert len(anomalies) == 2
        assert {a["block_id"] for a in anomalies} == {"b2", "b3"}


# ===========================================================================
# 6. ONNX Model Exporter & Numerical Parity Tests
# ===========================================================================

class TestONNXModelExporter:
    """Unit tests for ONNX graph export with dynamic axes and parity verification."""

    def test_export_deeplog_lstm(self, tmp_path):
        model = DeepLog(input_size=12, hidden_size=20, output_size=12, num_layers=1)
        model.eval()

        exporter = ONNXModelExporter(opset_version=14)
        out_file = tmp_path / "deeplog.onnx"

        exported = exporter.export_deeplog(model, out_file, context_length=8, batch_size=1)
        assert exported.exists()
        assert exported.stat().st_size > 0

        info = exporter.inspect_model(exported)
        assert info["size_bytes"] > 0
        assert info["has_pytorch_tag"]
        assert info["is_non_empty"]

    def test_export_deeplog_dual_head(self, tmp_path):
        model = DeepLogDualHead(vocab_size=15, hidden_size=24, num_layers=1, embedding_dim=16)
        model.eval()

        exporter = ONNXModelExporter(opset_version=14)
        out_file = tmp_path / "dual_head.onnx"

        exported = exporter.export_dual_head(model, out_file, context_length=10, batch_size=1)
        assert exported.exists()
        assert exported.stat().st_size > 0

        info = exporter.inspect_model(exported)
        assert info["size_bytes"] > 0
        assert info["has_pytorch_tag"]

    def test_dynamic_axes_varying_shapes_parity(self, tmp_path):
        model = DeepLog(input_size=10, hidden_size=16, output_size=10, num_layers=1)
        exporter = ONNXModelExporter(opset_version=14)
        out_file = tmp_path / "deeplog_dynamic.onnx"
        exporter.export_deeplog(model, out_file)

        result = exporter.verify_numerical_parity(
            model=model,
            onnx_path=out_file,
            batch_sizes=(1, 2, 8),
            seq_lens=(5, 12, 24),
            atol=1e-4,
        )

        assert result.passed
        assert len(result.shapes_tested) == 9
        assert result.max_absolute_diff <= 1e-4
        assert result.runtime_engine in ("onnxruntime", "torch_reference_dynamic_parity")

        res_dict = result.to_dict()
        assert res_dict["passed"]
        assert len(res_dict["details"]) == 9

    def test_dual_head_numerical_parity(self, tmp_path):
        model = DeepLogDualHead(vocab_size=12, hidden_size=16, num_layers=1, embedding_dim=16)
        exporter = ONNXModelExporter(opset_version=14)
        out_file = tmp_path / "dual_head_dynamic.onnx"
        exporter.export_dual_head(model, out_file)

        result = exporter.verify_numerical_parity(
            model=model,
            onnx_path=out_file,
            batch_sizes=(1, 4),
            seq_lens=(6, 15),
            atol=1e-4,
        )

        assert result.passed
        assert len(result.shapes_tested) == 4
        assert result.max_absolute_diff <= 1e-4

    def test_inspect_nonexistent_model_raises(self, tmp_path):
        exporter = ONNXModelExporter()
        with pytest.raises(FileNotFoundError):
            exporter.inspect_model(tmp_path / "missing.onnx")

        with pytest.raises(FileNotFoundError):
            exporter.verify_numerical_parity(
                model=DeepLog(10, 10, 10),
                onnx_path=tmp_path / "missing.onnx",
            )
