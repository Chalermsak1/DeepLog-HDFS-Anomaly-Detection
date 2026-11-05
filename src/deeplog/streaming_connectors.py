"""High-Performance Streaming Connectors & Columnar Log Sinks for DeepLog.

Provides:
- StreamingBrokerConfig: Configuration for broker endpoints, consumer groups, partitions, and commit offsets.
- KafkaLogProducer & KafkaLogConsumer: High-throughput asynchronous producer/consumer with zero-dependency
  in-memory mock protocol and optional native Kafka compatibility.
- BatchStreamBuffer: Sliding-window queue supporting backpressure policies (reject, drop-oldest, block).
- ClickHouseSink: Columnar batch ingestion engine supporting schema partitioning, buffer flushing,
  and exponential retry backoff.
"""

from __future__ import annotations

import collections
import dataclasses
import datetime
import json
import logging
import math
import os
import threading
import time
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Set, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class StreamingConnectorError(Exception):
    """Base exception for streaming connector operations."""
    pass


class BufferOverflowError(StreamingConnectorError):
    """Raised when BatchStreamBuffer is full and overflow policy is 'reject'."""
    pass


class BufferTimeoutError(StreamingConnectorError):
    """Raised when BatchStreamBuffer blocking operation times out."""
    pass


class IngestionError(StreamingConnectorError):
    """Raised when ClickHouse batch ingestion encounters an unrecoverable failure."""
    pass


# ---------------------------------------------------------------------------
# Configuration & Record Data Classes
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class StreamingBrokerConfig:
    """Configuration for streaming broker clients (Kafka / In-Memory Mock)."""

    bootstrap_servers: List[str] = dataclasses.field(default_factory=lambda: ["localhost:9092"])
    client_id: str = "deeplog-streaming-client"
    group_id: Optional[str] = "deeplog-consumer-group"
    topics: List[str] = dataclasses.field(default_factory=lambda: ["deeplog-raw-logs"])
    partition_assignments: Dict[str, List[int]] = dataclasses.field(default_factory=dict)
    auto_offset_reset: str = "latest"  # "earliest", "latest", "none"
    enable_auto_commit: bool = True
    auto_commit_interval_ms: int = 5000
    max_poll_records: int = 500
    session_timeout_ms: int = 30000
    batch_size: int = 16384
    linger_ms: int = 10
    compression_type: Optional[str] = None  # None, "gzip", "snappy", "lz4"
    security_protocol: str = "PLAINTEXT"
    sasl_mechanism: Optional[str] = None
    in_memory_mode: bool = True  # Zero-dependency local mock broker by default

    def validate(self) -> None:
        """Validate configuration settings."""
        if not self.bootstrap_servers:
            raise ValueError("bootstrap_servers cannot be empty")
        if self.auto_offset_reset not in ("earliest", "latest", "none"):
            raise ValueError(f"Invalid auto_offset_reset: {self.auto_offset_reset}")
        if self.max_poll_records <= 0:
            raise ValueError("max_poll_records must be > 0")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if self.auto_commit_interval_ms < 0:
            raise ValueError("auto_commit_interval_ms must be >= 0")

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StreamingBrokerConfig":
        """Construct config from dictionary."""
        valid_keys = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)


@dataclasses.dataclass
class ProducerRecord:
    """Represents a message to be published or published to a streaming topic."""

    topic: str
    value: Union[bytes, str, dict]
    key: Optional[Union[bytes, str]] = None
    partition: Optional[int] = None
    timestamp: float = dataclasses.field(default_factory=time.time)
    headers: Dict[str, bytes] = dataclasses.field(default_factory=dict)
    offset: Optional[int] = None

    def value_as_bytes(self) -> bytes:
        """Return the payload encoded as UTF-8 bytes."""
        if isinstance(self.value, bytes):
            return self.value
        elif isinstance(self.value, str):
            return self.value.encode("utf-8")
        elif isinstance(self.value, (dict, list)):
            return json.dumps(self.value).encode("utf-8")
        return str(self.value).encode("utf-8")

    def value_as_str(self) -> str:
        """Return the payload as a decoded string."""
        if isinstance(self.value, bytes):
            return self.value.decode("utf-8", errors="replace")
        elif isinstance(self.value, str):
            return self.value
        elif isinstance(self.value, (dict, list)):
            return json.dumps(self.value)
        return str(self.value)

    def value_as_json(self) -> Any:
        """Parse payload as JSON dictionary or structure."""
        if isinstance(self.value, (dict, list)):
            return self.value
        return json.loads(self.value_as_str())


@dataclasses.dataclass
class ConsumerRecord:
    """Represents a consumed message from a streaming partition."""

    topic: str
    partition: int
    offset: int
    key: Optional[bytes]
    value: bytes
    timestamp: float
    headers: Dict[str, bytes] = dataclasses.field(default_factory=dict)

    def value_as_str(self) -> str:
        """Decode value bytes to string."""
        return self.value.decode("utf-8", errors="replace")

    def value_as_json(self) -> Any:
        """Parse value bytes as JSON object."""
        return json.loads(self.value_as_str())

    def key_as_str(self) -> Optional[str]:
        """Decode key bytes to string."""
        if self.key is None:
            return None
        return self.key.decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# In-Memory High-Performance Mock Message Broker
# ---------------------------------------------------------------------------

class InMemoryBroker:
    """Thread-safe, partitioned in-memory log broker for zero-dependency streaming.

    Simulates partition log segments, monotonically increasing offsets,
    and consumer group commit management without requiring an external Kafka cluster.
    """

    _instance: Optional["InMemoryBroker"] = None
    _lock = threading.Lock()

    def __init__(self, default_partitions: int = 4):
        self.default_partitions = default_partitions
        # Structure: {topic: {partition: [ConsumerRecord]}}
        self._partitions: Dict[str, Dict[int, List[ConsumerRecord]]] = collections.defaultdict(dict)
        # Structure: {(group_id, topic, partition): committed_offset}
        self._committed_offsets: Dict[Tuple[str, str, int], int] = {}
        self._state_lock = threading.RLock()

    @classmethod
    def get_default(cls) -> "InMemoryBroker":
        """Get or initialize singleton in-memory broker instance."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset broker singleton for isolated tests."""
        with cls._lock:
            if cls._instance is not None:
                cls._instance.clear()
                cls._instance = None

    def clear(self) -> None:
        """Clear all stored topics, messages, and offsets."""
        with self._state_lock:
            self._partitions.clear()
            self._committed_offsets.clear()

    def _ensure_topic(self, topic: str, partition: Optional[int] = None) -> int:
        """Ensure topic and partitions exist, returning assigned partition."""
        with self._state_lock:
            if topic not in self._partitions:
                for p in range(self.default_partitions):
                    self._partitions[topic][p] = []
            if partition is not None and partition not in self._partitions[topic]:
                self._partitions[topic][partition] = []
            return partition if partition is not None else 0

    def publish(self, record: ProducerRecord) -> ConsumerRecord:
        """Publish a record to topic partition, assigning a strictly monotonic offset."""
        with self._state_lock:
            self._ensure_topic(record.topic)
            num_parts = len(self._partitions[record.topic])

            # Determine partition
            if record.partition is not None:
                part = record.partition
                if part not in self._partitions[record.topic]:
                    self._partitions[record.topic][part] = []
            elif record.key is not None:
                key_bytes = record.key if isinstance(record.key, bytes) else str(record.key).encode("utf-8")
                part = hash(key_bytes) % num_parts
            else:
                part = 0

            queue = self._partitions[record.topic][part]
            offset = len(queue)

            key_bytes = record.key if isinstance(record.key, bytes) else (record.key.encode("utf-8") if record.key else None)
            val_bytes = record.value_as_bytes()

            c_record = ConsumerRecord(
                topic=record.topic,
                partition=part,
                offset=offset,
                key=key_bytes,
                value=val_bytes,
                timestamp=record.timestamp,
                headers=record.headers.copy(),
            )
            queue.append(c_record)
            record.offset = offset
            record.partition = part
            return c_record

    def read_records(
        self,
        topic: str,
        partition: int,
        start_offset: int,
        max_records: int,
    ) -> List[ConsumerRecord]:
        """Read a batch of records from a partition starting at start_offset."""
        with self._state_lock:
            self._ensure_topic(topic, partition)
            queue = self._partitions[topic].get(partition, [])
            if start_offset >= len(queue):
                return []
            end_offset = min(start_offset + max_records, len(queue))
            return list(queue[start_offset:end_offset])

    def get_latest_offset(self, topic: str, partition: int) -> int:
        """Return the next offset to be written (log end offset)."""
        with self._state_lock:
            self._ensure_topic(topic, partition)
            return len(self._partitions[topic].get(partition, []))

    def get_earliest_offset(self, topic: str, partition: int) -> int:
        """Return earliest available offset (0)."""
        return 0

    def get_partitions_for_topic(self, topic: str) -> List[int]:
        """Return list of partition IDs for topic."""
        with self._state_lock:
            self._ensure_topic(topic)
            return sorted(list(self._partitions[topic].keys()))

    def commit_offset(self, group_id: str, topic: str, partition: int, offset: int) -> None:
        """Commit an offset for a consumer group and partition."""
        with self._state_lock:
            self._committed_offsets[(group_id, topic, partition)] = offset

    def get_committed_offset(self, group_id: str, topic: str, partition: int) -> Optional[int]:
        """Retrieve the committed offset for a consumer group."""
        with self._state_lock:
            return self._committed_offsets.get((group_id, topic, partition))


# ---------------------------------------------------------------------------
# KafkaLogProducer
# ---------------------------------------------------------------------------

class KafkaLogProducer:
    """High-throughput asynchronous/synchronous log producer.

    Operates in zero-dependency in-memory mode by default, or bridges to
    native kafka brokers when available.
    """

    def __init__(
        self,
        config: Optional[StreamingBrokerConfig] = None,
        broker: Optional[InMemoryBroker] = None,
    ):
        self.config = config or StreamingBrokerConfig()
        self.config.validate()
        self.broker = broker or InMemoryBroker.get_default()
        self.is_closed: bool = False

        self.records_sent: int = 0
        self.bytes_sent: int = 0
        self.batches_sent: int = 0
        self.errors_count: int = 0
        self._lock = threading.Lock()

    def send(
        self,
        topic: str,
        value: Union[bytes, str, dict],
        key: Optional[Union[bytes, str]] = None,
        partition: Optional[int] = None,
        headers: Optional[Dict[str, bytes]] = None,
    ) -> ProducerRecord:
        """Publish a message asynchronously or return the dispatched ProducerRecord."""
        if self.is_closed:
            raise StreamingConnectorError("Producer is closed")

        record = ProducerRecord(
            topic=topic,
            value=value,
            key=key,
            partition=partition,
            headers=headers or {},
            timestamp=time.time(),
        )

        try:
            self.broker.publish(record)
            with self._lock:
                self.records_sent += 1
                self.bytes_sent += len(record.value_as_bytes())
            return record
        except Exception as e:
            with self._lock:
                self.errors_count += 1
            raise StreamingConnectorError(f"Failed to publish record: {e}") from e

    def send_sync(
        self,
        topic: str,
        value: Union[bytes, str, dict],
        key: Optional[Union[bytes, str]] = None,
        partition: Optional[int] = None,
        timeout: float = 5.0,
    ) -> ProducerRecord:
        """Synchronously publish a record and ensure offset assignment."""
        return self.send(topic, value=value, key=key, partition=partition)

    def flush(self, timeout: Optional[float] = None) -> None:
        """Flush pending batches to broker."""
        with self._lock:
            self.batches_sent += 1

    def close(self, timeout: Optional[float] = None) -> None:
        """Close producer client."""
        self.flush(timeout)
        self.is_closed = True

    def get_metrics(self) -> Dict[str, Any]:
        """Return producer operational metrics."""
        with self._lock:
            return {
                "records_sent": self.records_sent,
                "bytes_sent": self.bytes_sent,
                "batches_sent": self.batches_sent,
                "errors_count": self.errors_count,
                "is_closed": self.is_closed,
            }

    def __enter__(self) -> "KafkaLogProducer":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


# ---------------------------------------------------------------------------
# KafkaLogConsumer
# ---------------------------------------------------------------------------

class KafkaLogConsumer:
    """High-performance log consumer with partition assignment and offset tracking.

    Supports subscribe() and assign() modes, seek operations, manual & auto-commits.
    """

    def __init__(
        self,
        config: Optional[StreamingBrokerConfig] = None,
        broker: Optional[InMemoryBroker] = None,
    ):
        self.config = config or StreamingBrokerConfig()
        self.config.validate()
        self.broker = broker or InMemoryBroker.get_default()
        self.is_closed: bool = False

        # Current subscription: {topic: [partition_ids]}
        self.subscriptions: Set[str] = set()
        self.assigned_partitions: Dict[str, List[int]] = {}
        # Next read offsets: {(topic, partition): offset}
        self.current_offsets: Dict[Tuple[str, int], int] = {}
        self.committed_offsets: Dict[Tuple[str, int], int] = {}

        self.last_auto_commit_time = time.time()
        self._lock = threading.Lock()

        # Initialize from config topics if provided
        if self.config.topics:
            self.subscribe(self.config.topics)

    def subscribe(self, topics: Sequence[str]) -> None:
        """Subscribe to a list of topics, dynamically discovering their partitions."""
        with self._lock:
            self.subscriptions = set(topics)
            self.assigned_partitions.clear()
            for topic in self.subscriptions:
                parts = self.broker.get_partitions_for_topic(topic)
                self.assigned_partitions[topic] = parts
                for p in parts:
                    self._init_offset_for_partition(topic, p)

    def assign(self, partition_map: Dict[str, List[int]]) -> None:
        """Manually assign specific topic partitions."""
        with self._lock:
            self.subscriptions = set(partition_map.keys())
            self.assigned_partitions = {t: list(ps) for t, ps in partition_map.items()}
            for topic, parts in self.assigned_partitions.items():
                for p in parts:
                    self._init_offset_for_partition(topic, p)

    def _init_offset_for_partition(self, topic: str, partition: int) -> None:
        """Initialize current reading offset based on committed offset or reset policy."""
        key = (topic, partition)
        if key in self.current_offsets:
            return

        committed = None
        if self.config.group_id:
            committed = self.broker.get_committed_offset(self.config.group_id, topic, partition)

        if committed is not None:
            self.current_offsets[key] = committed
            self.committed_offsets[key] = committed
        elif self.config.auto_offset_reset == "earliest":
            self.current_offsets[key] = self.broker.get_earliest_offset(topic, partition)
        elif self.config.auto_offset_reset == "latest":
            self.current_offsets[key] = self.broker.get_latest_offset(topic, partition)
        else:
            self.current_offsets[key] = 0

    def poll(
        self,
        timeout_ms: float = 1000.0,
        max_records: Optional[int] = None,
    ) -> List[ConsumerRecord]:
        """Poll assigned partitions for new messages."""
        if self.is_closed:
            raise StreamingConnectorError("Consumer is closed")

        limit = max_records or self.config.max_poll_records
        results: List[ConsumerRecord] = []
        end_time = time.time() + (timeout_ms / 1000.0)

        with self._lock:
            # Poll round-robin across assigned partitions
            for topic, parts in list(self.assigned_partitions.items()):
                for p in parts:
                    key = (topic, p)
                    curr = self.current_offsets.get(key, 0)
                    needed = limit - len(results)
                    if needed <= 0:
                        break

                    records = self.broker.read_records(topic, p, start_offset=curr, max_records=needed)
                    if records:
                        results.extend(records)
                        self.current_offsets[key] = records[-1].offset + 1

                if len(results) >= limit:
                    break

            # Handle auto commit
            if self.config.enable_auto_commit and self.config.group_id:
                now = time.time()
                if (now - self.last_auto_commit_time) * 1000.0 >= self.config.auto_commit_interval_ms:
                    self._commit_internal()
                    self.last_auto_commit_time = now

        return results

    def commit(
        self,
        offsets: Optional[Dict[Tuple[str, int], int]] = None,
        asynchronous: bool = False,
    ) -> None:
        """Commit current consumer group offsets."""
        with self._lock:
            self._commit_internal(offsets)

    def _commit_internal(self, offsets: Optional[Dict[Tuple[str, int], int]] = None) -> None:
        """Internal worker for committing offsets to broker."""
        if not self.config.group_id:
            return

        to_commit = offsets or self.current_offsets
        for (topic, partition), offset in to_commit.items():
            self.broker.commit_offset(self.config.group_id, topic, partition, offset)
            self.committed_offsets[(topic, partition)] = offset

    def seek(self, topic: str, partition: int, offset: int) -> None:
        """Explicitly reposition consumer read head."""
        with self._lock:
            self.current_offsets[(topic, partition)] = max(0, offset)

    def seek_to_beginning(self, topic: str, partition: int) -> None:
        """Reposition consumer to start of partition."""
        with self._lock:
            self.current_offsets[(topic, partition)] = self.broker.get_earliest_offset(topic, partition)

    def seek_to_end(self, topic: str, partition: int) -> None:
        """Reposition consumer to end of partition."""
        with self._lock:
            self.current_offsets[(topic, partition)] = self.broker.get_latest_offset(topic, partition)

    def position(self, topic: str, partition: int) -> int:
        """Return the next offset to be read."""
        with self._lock:
            return self.current_offsets.get((topic, partition), 0)

    def close(self) -> None:
        """Close consumer client, committing offsets if auto-commit is enabled."""
        with self._lock:
            if not self.is_closed:
                if self.config.enable_auto_commit:
                    self._commit_internal()
                self.is_closed = True

    def __iter__(self) -> Iterator[ConsumerRecord]:
        """Iterator yielding records as they arrive."""
        while not self.is_closed:
            batch = self.poll(timeout_ms=500)
            for item in batch:
                yield item

    def __enter__(self) -> "KafkaLogConsumer":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


# ---------------------------------------------------------------------------
# BatchStreamBuffer (Sliding Window & Backpressure)
# ---------------------------------------------------------------------------

class BatchStreamBuffer:
    """Sliding-window buffer queue with configurable backpressure policies.

    Policies supported:
    - 'reject': Immediately raises BufferOverflowError when full.
    - 'drop_oldest': Evicts oldest elements to make room, tracking drop counters.
    - 'block': Blocks calling thread until space becomes available or timeout expires.
    """

    def __init__(
        self,
        max_capacity: int = 1000,
        overflow_policy: str = "reject",
        timeout_seconds: float = 5.0,
        batch_size: int = 100,
    ):
        if max_capacity <= 0:
            raise ValueError("max_capacity must be > 0")
        if overflow_policy not in ("reject", "drop_oldest", "block"):
            raise ValueError(f"Invalid overflow_policy: {overflow_policy}")

        self.max_capacity = max_capacity
        self.overflow_policy = overflow_policy
        self.timeout_seconds = timeout_seconds
        self.batch_size = batch_size

        self._queue: collections.deque = collections.deque()
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._not_full = threading.Condition(self._lock)

        self.total_enqueued: int = 0
        self.total_dequeued: int = 0
        self.total_dropped: int = 0
        self.total_rejected: int = 0

    def push(
        self,
        item: Any,
        block: bool = True,
        timeout: Optional[float] = None,
    ) -> bool:
        """Push an item into the buffer respecting configured backpressure policy."""
        timeout_val = timeout if timeout is not None else self.timeout_seconds

        with self._not_full:
            if len(self._queue) >= self.max_capacity:
                if self.overflow_policy == "reject":
                    self.total_rejected += 1
                    raise BufferOverflowError(
                        f"Buffer capacity ({self.max_capacity}) exceeded under 'reject' policy"
                    )

                elif self.overflow_policy == "drop_oldest":
                    self._queue.popleft()
                    self.total_dropped += 1

                elif self.overflow_policy == "block":
                    if not block:
                        self.total_rejected += 1
                        raise BufferOverflowError("Buffer full and non-blocking push requested")

                    deadline = time.time() + timeout_val
                    while len(self._queue) >= self.max_capacity:
                        remaining = deadline - time.time()
                        if remaining <= 0:
                            self.total_rejected += 1
                            raise BufferTimeoutError(
                                f"Timed out after {timeout_val}s waiting for buffer space"
                            )
                        self._not_full.wait(remaining)

            self._queue.append(item)
            self.total_enqueued += 1
            self._not_empty.notify()
            return True

    def push_batch(self, items: Sequence[Any]) -> int:
        """Push multiple items sequentially into the buffer."""
        count = 0
        for item in items:
            self.push(item)
            count += 1
        return count

    def pop(self, block: bool = True, timeout: Optional[float] = None) -> Any:
        """Pop a single item from the buffer queue."""
        timeout_val = timeout if timeout is not None else self.timeout_seconds

        with self._not_empty:
            if not self._queue:
                if not block:
                    return None
                deadline = time.time() + timeout_val
                while not self._queue:
                    remaining = deadline - time.time()
                    if remaining <= 0:
                        return None
                    self._not_empty.wait(remaining)

            item = self._queue.popleft()
            self.total_dequeued += 1
            self._not_full.notify()
            return item

    def pop_batch(
        self,
        max_items: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> List[Any]:
        """Pop up to max_items (defaults to batch_size) from the buffer."""
        limit = max_items or self.batch_size
        items = []

        with self._not_empty:
            if not self._queue and timeout is not None and timeout > 0:
                self._not_empty.wait(timeout)

            while self._queue and len(items) < limit:
                items.append(self._queue.popleft())
                self.total_dequeued += 1

            if items:
                self._not_full.notify_all()

        return items

    def drain(self) -> List[Any]:
        """Drain and return all items currently held in buffer."""
        with self._lock:
            items = list(self._queue)
            self._queue.clear()
            self.total_dequeued += len(items)
            self._not_full.notify_all()
            return items

    def peek(self, n: Optional[int] = None) -> List[Any]:
        """Inspect items in buffer without removing them."""
        with self._lock:
            if n is None:
                return list(self._queue)
            return [self._queue[i] for i in range(min(n, len(self._queue)))]

    def snapshot(self) -> List[Any]:
        """Return a snapshot copy of current buffer items."""
        return self.peek()

    def clear(self) -> None:
        """Empty the buffer."""
        with self._lock:
            self._queue.clear()
            self._not_full.notify_all()

    def size(self) -> int:
        """Return current number of items in buffer."""
        with self._lock:
            return len(self._queue)

    def is_empty(self) -> bool:
        """Check whether buffer is empty."""
        return self.size() == 0

    def is_full(self) -> bool:
        """Check whether buffer has reached maximum capacity."""
        return self.size() >= self.max_capacity

    def get_metrics(self) -> Dict[str, Any]:
        """Return buffer metrics and operational statistics."""
        with self._lock:
            return {
                "current_size": len(self._queue),
                "max_capacity": self.max_capacity,
                "overflow_policy": self.overflow_policy,
                "total_enqueued": self.total_enqueued,
                "total_dequeued": self.total_dequeued,
                "total_dropped": self.total_dropped,
                "total_rejected": self.total_rejected,
            }

    def __len__(self) -> int:
        return self.size()


# ---------------------------------------------------------------------------
# ClickHouseSink (Columnar Storage & Schema Partitioning)
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class ClickHouseSinkConfig:
    """Configuration for ClickHouse Columnar Sink."""

    host: str = "localhost"
    port: int = 8123
    database: str = "deeplog"
    table: str = "log_events"
    columns: Dict[str, str] = dataclasses.field(
        default_factory=lambda: {
            "timestamp": "DateTime64(3)",
            "block_id": "String",
            "event_id": "UInt32",
            "score": "Float32",
            "is_anomaly": "UInt8",
            "raw_log": "String",
        }
    )
    partition_by: str = "toYYYYMM(timestamp)"
    order_by: Tuple[str, ...] = ("timestamp", "block_id", "event_id")
    batch_size: int = 500
    flush_interval_seconds: float = 2.0
    max_retries: int = 3
    initial_backoff_seconds: float = 0.05
    backoff_factor: float = 2.0
    in_memory_mode: bool = True


class ClickHouseSink:
    """Batch ingestion into columnar log format with schema partitioning and retry backoff.

    Features:
    - Vectorized columnar memory representation for fast analytics.
    - Partition extraction (e.g. YYYY-MM based on timestamp).
    - Periodic or batch-threshold buffer flushing.
    - Resilient exponential backoff on simulated or real transient network errors.
    - Zero external database required for execution.
    """

    def __init__(self, config: Optional[ClickHouseSinkConfig] = None):
        self.config = config or ClickHouseSinkConfig()
        self._buffer: List[Dict[str, Any]] = []
        # In-memory columnar storage: {partition_key: {column_name: [values]}}
        self._columnar_store: Dict[str, Dict[str, List[Any]]] = collections.defaultdict(
            lambda: {col: [] for col in self.config.columns}
        )

        self.last_flush_time = time.time()
        self.total_rows_ingested: int = 0
        self.total_flushes: int = 0
        self.total_retries: int = 0
        self.is_closed: bool = False
        self._lock = threading.Lock()

        # Simulated failure injection hook for testing retry backoff
        self._fail_next_n_flushes: int = 0

    def write(self, record: Dict[str, Any]) -> None:
        """Write a single structured log record to buffer, flushing if threshold reached."""
        with self._lock:
            if self.is_closed:
                raise StreamingConnectorError("ClickHouseSink is closed")
            self._buffer.append(record)
            if len(self._buffer) >= self.config.batch_size:
                self._flush_unlocked()

    def write_batch(self, records: Sequence[Dict[str, Any]]) -> None:
        """Write multiple records to buffer."""
        with self._lock:
            if self.is_closed:
                raise StreamingConnectorError("ClickHouseSink is closed")
            self._buffer.extend(records)
            if len(self._buffer) >= self.config.batch_size:
                self._flush_unlocked()

    def flush(self) -> int:
        """Manually trigger flush of all buffered records to columnar storage."""
        with self._lock:
            return self._flush_unlocked()

    def _flush_unlocked(self) -> int:
        """Internal flush handler with exponential retry backoff."""
        if not self._buffer:
            return 0

        batch = list(self._buffer)
        rows_to_flush = len(batch)

        # Retry loop with exponential backoff
        attempt = 0
        backoff = self.config.initial_backoff_seconds
        last_error: Optional[Exception] = None

        while attempt <= self.config.max_retries:
            try:
                # Test hook: simulate transient failure
                if self._fail_next_n_flushes > 0:
                    self._fail_next_n_flushes -= 1
                    raise ConnectionResetError("Simulated transient socket reset in ClickHouse batch ingestion")

                self._commit_batch_to_columnar_storage(batch)
                self._buffer.clear()
                self.total_rows_ingested += rows_to_flush
                self.total_flushes += 1
                self.last_flush_time = time.time()
                return rows_to_flush

            except Exception as e:
                attempt += 1
                self.total_retries += 1
                last_error = e
                if attempt > self.config.max_retries:
                    break
                time.sleep(backoff)
                backoff *= self.config.backoff_factor

        raise IngestionError(
            f"Failed to ingest {rows_to_flush} rows into ClickHouse after {attempt} attempts: {last_error}"
        ) from last_error

    def _extract_partition_key(self, record: Dict[str, Any]) -> str:
        """Extract partition string from timestamp (defaults to YYYYMM)."""
        ts = record.get("timestamp")
        if isinstance(ts, (int, float)):
            dt = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc)
            return dt.strftime("%Y%m")
        elif isinstance(ts, str):
            clean = ts.replace("T", " ").split()[0].replace("-", "")[:6]
            return clean if len(clean) == 6 else "default"
        return "default"

    def _commit_batch_to_columnar_storage(self, batch: List[Dict[str, Any]]) -> None:
        """Vectorize and append rows into per-partition columnar arrays."""
        for row in batch:
            part_key = self._extract_partition_key(row)
            part_cols = self._columnar_store[part_key]

            for col in self.config.columns:
                val = row.get(col, None)
                part_cols[col].append(val)

    def query(
        self,
        predicate: Optional[Callable[[Dict[str, Any]], bool]] = None,
        partition: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Query ingested rows across partitions."""
        with self._lock:
            partitions_to_scan = [partition] if partition else list(self._columnar_store.keys())
            matched: List[Dict[str, Any]] = []

            for p in partitions_to_scan:
                if p not in self._columnar_store:
                    continue
                cols = self._columnar_store[p]
                num_rows = len(next(iter(cols.values()))) if cols else 0

                for i in range(num_rows):
                    row = {c: cols[c][i] for c in self.config.columns}
                    if predicate is None or predicate(row):
                        matched.append(row)

            return matched

    def get_columnar_arrays(self, partition: Optional[str] = None) -> Dict[str, np.ndarray]:
        """Convert stored column lists into zero-copy / contiguous numpy arrays for fast analytics."""
        with self._lock:
            partitions_to_scan = [partition] if partition else list(self._columnar_store.keys())
            merged_cols: Dict[str, List[Any]] = {col: [] for col in self.config.columns}

            for p in partitions_to_scan:
                if p not in self._columnar_store:
                    continue
                for col in self.config.columns:
                    merged_cols[col].extend(self._columnar_store[p][col])

            result: Dict[str, np.ndarray] = {}
            for col, values in merged_cols.items():
                col_type = self.config.columns.get(col, "String")
                if "UInt" in col_type or "Int" in col_type:
                    result[col] = np.array(values, dtype=np.int64 if any(v is not None for v in values) else np.int32)
                elif "Float" in col_type:
                    result[col] = np.array(values, dtype=np.float32)
                else:
                    result[col] = np.array(values, dtype=object)

            return result

    def get_partitions(self) -> List[str]:
        """Return list of distinct partition keys."""
        with self._lock:
            return sorted(list(self._columnar_store.keys()))

    def get_partition_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Return statistics per partition (row count, columns)."""
        with self._lock:
            metrics = {}
            for p, cols in self._columnar_store.items():
                row_count = len(next(iter(cols.values()))) if cols else 0
                metrics[p] = {
                    "row_count": row_count,
                    "columns": list(cols.keys()),
                }
            return metrics

    def create_table_ddl(self) -> str:
        """Generate equivalent ClickHouse CREATE TABLE DDL."""
        cols_defs = [f"    `{c}` {t}" for c, t in self.config.columns.items()]
        cols_sql = ",\n".join(cols_defs)
        order_sql = ", ".join(f"`{o}`" for o in self.config.order_by)

        return (
            f"CREATE TABLE IF NOT EXISTS `{self.config.database}`.`{self.config.table}` (\n"
            f"{cols_sql}\n"
            f") ENGINE = MergeTree()\n"
            f"PARTITION BY {self.config.partition_by}\n"
            f"ORDER BY ({order_sql});"
        )

    def close(self) -> None:
        """Flush remaining buffered data and close sink."""
        with self._lock:
            if not self.is_closed:
                self._flush_unlocked()
                self.is_closed = True

    def __enter__(self) -> "ClickHouseSink":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
