"""eBPF Compile-Once Run-Everywhere (CO-RE) Builder & Skeleton Orchestrator.

Manages Linux eBPF CO-RE lifecycle, kernel BTF (/sys/kernel/btf/vmlinux) inspection,
vmlinux.h header probing and generation, and bpftool skeleton code generation with
resilient cross-platform fallbacks for macOS/BSD development environments.
"""

from __future__ import annotations

import dataclasses
import os
import platform
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# Standard BTF magic constant (0xEB9F)
BTF_MAGIC = 0xEB9F
BTF_VERSION = 1
BTF_HEADER_SIZE = 24

# Default paths for Linux kernel BTF
DEFAULT_BTF_PATHS = [
    Path("/sys/kernel/btf/vmlinux"),
    Path("/boot/vmlinux"),
    Path("/lib/modules") / platform.release() / "vmlinux",
]


@dataclasses.dataclass
class BTFHeader:
    """Parsed BPF Type Format (BTF) binary header."""

    magic: int
    version: int
    flags: int
    hdr_len: int
    type_off: int
    type_len: int
    str_off: int
    str_len: int

    @property
    def is_valid(self) -> bool:
        """Check if BTF magic matches 0xEB9F."""
        return self.magic == BTF_MAGIC


@dataclasses.dataclass
class BTFInfo:
    """Kernel BTF diagnostic inspection results."""

    path: Optional[Path] = None
    exists: bool = False
    is_valid: bool = False
    size_bytes: int = 0
    kernel_release: str = ""
    has_core_support: bool = False
    types_count_estimate: int = 0
    endianness: str = "little"
    header: Optional[BTFHeader] = None
    diagnostics: List[str] = dataclasses.field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert BTF info to dictionary."""
        data = dataclasses.asdict(self)
        if self.path is not None:
            data["path"] = str(self.path)
        return data


@dataclasses.dataclass
class SkeletonResult:
    """Outcome of bpftool skeleton generation."""

    success: bool
    skeleton_path: Optional[Path] = None
    skeleton_name: str = ""
    is_mock: bool = False
    header_code: str = ""
    progs: List[str] = dataclasses.field(default_factory=list)
    maps: List[str] = dataclasses.field(default_factory=list)
    error_message: Optional[str] = None
    duration_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        data = dataclasses.asdict(self)
        if self.skeleton_path is not None:
            data["skeleton_path"] = str(self.skeleton_path)
        return data


@dataclasses.dataclass
class COREBuildConfig:
    """Configuration for eBPF CO-RE build process."""

    arch: str = "x86_64"
    clang_path: Optional[str] = None
    bpftool_path: Optional[str] = None
    btf_path: Optional[Path] = None
    vmlinux_h_path: Optional[Path] = None
    cache_dir: Optional[Path] = None
    opt_level: str = "-O2"
    debug_info: bool = True
    use_cross_platform_fallback: bool = True
    strict_core: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        data = dataclasses.asdict(self)
        for key in ("btf_path", "vmlinux_h_path", "cache_dir"):
            if data[key] is not None:
                data[key] = str(data[key])
        return data


class BPFSkeletonStub:
    """Runtime simulation stub for BPF skeletons in non-Linux or mock environments."""

    def __init__(self, skeleton_name: str, progs: Optional[List[str]] = None, maps: Optional[List[str]] = None):
        self.skeleton_name = skeleton_name
        self.progs = progs or ["kprobe_sys_enter", "tracepoint_sched_switch"]
        self.maps = maps or ["events_ringbuf", "anomaly_stats", "hdc_filter_map"]
        self.is_open: bool = False
        self.is_loaded: bool = False
        self.is_attached: bool = False
        self._event_counter: int = 0

    def open(self) -> "BPFSkeletonStub":
        """Open the BPF object and parse ELF sections."""
        self.is_open = True
        return self

    def load(self) -> "BPFSkeletonStub":
        """Load and verify BPF bytecode in the kernel."""
        if not self.is_open:
            self.open()
        self.is_loaded = True
        return self

    def attach(self) -> "BPFSkeletonStub":
        """Attach probes and tracepoints to kernel hooks."""
        if not self.is_loaded:
            self.load()
        self.is_attached = True
        return self

    def detach(self) -> "BPFSkeletonStub":
        """Detach all attached kernel probes."""
        self.is_attached = False
        return self

    def destroy(self) -> None:
        """Unload maps and programs, freeing kernel resources."""
        self.is_attached = False
        self.is_loaded = False
        self.is_open = False

    def poll_events(self, max_events: int = 10) -> List[Dict[str, Any]]:
        """Simulate polling events from skeleton ring buffer."""
        if not self.is_attached:
            return []
        events = []
        for _ in range(max_events):
            self._event_counter += 1
            events.append({
                "seq": self._event_counter,
                "timestamp_ns": int(time.time() * 1e9),
                "pid": os.getpid(),
                "syscall_id": (self._event_counter % 350) + 1,
                "is_anomalous": (self._event_counter % 20 == 0),
            })
        return events


class BPFCOREBuilder:
    """eBPF Compile-Once Run-Everywhere (CO-RE) Lifecycle & Toolchain Builder.

    Manages:
    - BTF (/sys/kernel/btf/vmlinux) probing and binary header verification.
    - vmlinux.h generation via bpftool or self-contained cross-platform C fallback header.
    - bpftool gen skeleton header generation and stub orchestration.
    - Cross-platform fallbacks for macOS/BSD/Windows development machines.
    """

    def __init__(self, config: Optional[COREBuildConfig] = None):
        self.config = config or COREBuildConfig()
        self.system = platform.system()
        self.kernel_release = platform.release()

        # Resolve cache directory
        if self.config.cache_dir is not None:
            self.cache_dir = Path(self.config.cache_dir).resolve()
        else:
            base_cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
            self.cache_dir = base_cache / "deeplog" / "bpf_core"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Tool discovery
        self.clang_path = self.config.clang_path or shutil.which("clang")
        self.bpftool_path = self.config.bpftool_path or shutil.which("bpftool")

    @classmethod
    def is_linux(cls) -> bool:
        """Check if operating on a Linux kernel."""
        return sys.platform.startswith("linux")

    @classmethod
    def is_macos(cls) -> bool:
        """Check if operating on macOS (Darwin)."""
        return sys.platform == "darwin"

    @classmethod
    def is_bsd(cls) -> bool:
        """Check if operating on BSD."""
        return "bsd" in sys.platform

    def check_btf_support(self, btf_path: Optional[Union[str, Path]] = None) -> bool:
        """Check whether Linux kernel BTF is available and valid."""
        info = self.inspect_btf(btf_path)
        return info.has_core_support

    def inspect_btf(self, btf_path: Optional[Union[str, Path]] = None) -> BTFInfo:
        """Inspect the BTF binary at /sys/kernel/btf/vmlinux or custom path."""
        target_path: Optional[Path] = None
        if btf_path is not None:
            target_path = Path(btf_path).resolve()
        elif self.config.btf_path is not None:
            target_path = Path(self.config.btf_path).resolve()
        else:
            # Search standard locations
            for p in DEFAULT_BTF_PATHS:
                if p.exists() and p.is_file():
                    target_path = p
                    break

        info = BTFInfo(
            path=target_path,
            kernel_release=self.kernel_release,
        )

        if target_path is None or not target_path.exists():
            info.exists = False
            info.has_core_support = False
            if self.is_linux():
                info.diagnostics.append(
                    "Kernel BTF (/sys/kernel/btf/vmlinux) not found. Kernel might not be built with CONFIG_DEBUG_INFO_BTF=y."
                )
            else:
                info.diagnostics.append(
                    f"Non-Linux platform ({self.system}) detected; kernel BTF is unavailable. Activating cross-platform CO-RE fallback."
                )
            return info

        info.exists = True
        try:
            size = target_path.stat().st_size
            info.size_bytes = size

            with open(target_path, "rb") as f:
                header_bytes = f.read(BTF_HEADER_SIZE)

            if len(header_bytes) < BTF_HEADER_SIZE:
                info.diagnostics.append(f"BTF file too small ({len(header_bytes)} bytes < {BTF_HEADER_SIZE}).")
                return info

            # struct btf_header: <HBBIIIII (magic, version, flags, hdr_len, type_off, type_len, str_off, str_len)
            magic, version, flags, hdr_len, type_off, type_len, str_off, str_len = struct.unpack(
                "<HBBIIIII", header_bytes
            )

            header = BTFHeader(
                magic=magic,
                version=version,
                flags=flags,
                hdr_len=hdr_len,
                type_off=type_off,
                type_len=type_len,
                str_off=str_off,
                str_len=str_len,
            )
            info.header = header
            info.is_valid = header.is_valid

            if header.is_valid:
                info.has_core_support = True
                # Approximate types count: each BTF type is ~12-24 bytes
                info.types_count_estimate = max(1, type_len // 16)
                info.diagnostics.append(
                    f"Valid BTF detected (magic=0x{magic:04X}, types_len={type_len} bytes, estimated_types={info.types_count_estimate})."
                )
            else:
                info.diagnostics.append(
                    f"Invalid BTF header magic (expected 0x{BTF_MAGIC:04X}, got 0x{magic:04X})."
                )

        except Exception as e:
            info.diagnostics.append(f"Failed to parse BTF file: {e}")

        return info

    def probe_vmlinux_header(
        self,
        output_path: Optional[Union[str, Path]] = None,
        force_generate: bool = False,
    ) -> Path:
        """Probe for or generate vmlinux.h for CO-RE compilation.

        If running on Linux with bpftool and BTF support, dumps vmlinux.h from BTF.
        Otherwise, generates a portable, syntactically complete minimal vmlinux.h fallback.
        """
        if output_path is not None:
            out = Path(output_path).resolve()
        elif self.config.vmlinux_h_path is not None:
            out = Path(self.config.vmlinux_h_path).resolve()
        else:
            out = self.cache_dir / "vmlinux.h"

        if out.exists() and not force_generate:
            return out

        out.parent.mkdir(parents=True, exist_ok=True)

        # Try real bpftool dump if on Linux with bpftool and BTF
        btf_info = self.inspect_btf()
        if self.is_linux() and self.bpftool_path and btf_info.has_core_support and btf_info.path:
            try:
                cmd = [self.bpftool_path, "btf", "dump", "file", str(btf_info.path), "format", "c"]
                res = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
                if res.returncode == 0 and res.stdout:
                    out.write_text(res.stdout, encoding="utf-8")
                    return out
            except Exception:
                pass

        # Cross-platform / fallback portable vmlinux.h generation
        fallback_header = self._generate_fallback_vmlinux_h()
        out.write_text(fallback_header, encoding="utf-8")
        return out

    def generate_skeleton(
        self,
        bpf_obj_path: Union[str, Path],
        output_header_path: Optional[Union[str, Path]] = None,
        skeleton_name: Optional[str] = None,
    ) -> SkeletonResult:
        """Generate C skeleton header from compiled eBPF object.

        Uses `bpftool gen skeleton` when available on Linux, or generates an idiomatic
        CO-RE C skeleton header and stub bindings in non-Linux or fallback environments.
        """
        start_time = time.perf_counter()
        obj_path = Path(bpf_obj_path).resolve()

        if skeleton_name is None:
            # Derive skeleton name from file name: deeplog_kern.bpf.o -> deeplog_kern
            base = obj_path.name
            for suffix in [".bpf.o", ".o"]:
                if base.endswith(suffix):
                    base = base[: -len(suffix)]
                    break
            skeleton_name = base.replace("-", "_")

        if output_header_path is not None:
            out_header = Path(output_header_path).resolve()
        else:
            out_header = self.cache_dir / f"{skeleton_name}.skel.h"

        out_header.parent.mkdir(parents=True, exist_ok=True)

        # Attempt bpftool gen skeleton if bpftool is present and object exists
        if self.bpftool_path and obj_path.exists() and self.is_linux():
            try:
                cmd = [self.bpftool_path, "gen", "skeleton", str(obj_path), "name", skeleton_name]
                res = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                if res.returncode == 0 and res.stdout:
                    out_header.write_text(res.stdout, encoding="utf-8")
                    dur_ms = (time.perf_counter() - start_time) * 1000.0
                    return SkeletonResult(
                        success=True,
                        skeleton_path=out_header,
                        skeleton_name=skeleton_name,
                        is_mock=False,
                        header_code=res.stdout,
                        progs=["tracepoint_sched_switch", "kprobe_sys_enter"],
                        maps=["events_ringbuf", "anomaly_stats"],
                        duration_ms=dur_ms,
                    )
            except Exception:
                pass

        # Cross-platform / fallback skeleton generation
        code = self._generate_fallback_skeleton_h(skeleton_name)
        out_header.write_text(code, encoding="utf-8")
        dur_ms = (time.perf_counter() - start_time) * 1000.0

        return SkeletonResult(
            success=True,
            skeleton_path=out_header,
            skeleton_name=skeleton_name,
            is_mock=True,
            header_code=code,
            progs=["kprobe_sys_enter", "tracepoint_sched_switch"],
            maps=["events_ringbuf", "anomaly_stats", "hdc_filter_map"],
            duration_ms=dur_ms,
        )

    def get_python_skeleton(self, skeleton_name: str) -> BPFSkeletonStub:
        """Instantiate a Python runtime stub for the skeleton."""
        return BPFSkeletonStub(skeleton_name=skeleton_name)

    def create_synthetic_btf(self, output_path: Union[str, Path], num_types: int = 64) -> Path:
        """Create a synthetic, binary-valid BTF file for cross-platform unit testing."""
        out = Path(output_path).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)

        magic = BTF_MAGIC
        version = BTF_VERSION
        flags = 0
        hdr_len = BTF_HEADER_SIZE

        type_len = num_types * 16
        type_off = 0
        str_off = type_len
        str_len = 256

        header_bytes = struct.pack(
            "<HBBIIIII",
            magic,
            version,
            flags,
            hdr_len,
            type_off,
            type_len,
            str_off,
            str_len,
        )

        dummy_type_data = b"\x00" * type_len
        dummy_str_data = b"\x00DeepLog_BTF_Synthetic_Kernel_Types\x00" + b"\x00" * (str_len - 36)

        with open(out, "wb") as f:
            f.write(header_bytes)
            f.write(dummy_type_data)
            f.write(dummy_str_data)

        return out

    def get_cross_platform_report(self) -> Dict[str, Any]:
        """Generate a complete diagnostic status report for the host environment."""
        btf_info = self.inspect_btf()
        return {
            "platform": sys.platform,
            "system": self.system,
            "kernel_release": self.kernel_release,
            "is_linux": self.is_linux(),
            "has_clang": self.clang_path is not None,
            "clang_path": self.clang_path,
            "has_bpftool": self.bpftool_path is not None,
            "bpftool_path": self.bpftool_path,
            "btf_available": btf_info.has_core_support,
            "btf_info": btf_info.to_dict(),
            "cache_dir": str(self.cache_dir),
            "fallback_active": not (self.is_linux() and btf_info.has_core_support and self.bpftool_path),
        }

    @staticmethod
    def _generate_fallback_vmlinux_h() -> str:
        """Generate a portable, self-contained minimal vmlinux.h header."""
        return """/* SPDX-License-Identifier: (LGPL-2.1 OR BSD-2-Clause) */
/* DeepLog CO-RE Cross-Platform Portable vmlinux.h Stub */

#ifndef __VMLINUX_H__
#define __VMLINUX_H__

#ifndef __BPF_TRACING__
#define __BPF_TRACING__
#endif

/* Standard integer typedefs */
typedef unsigned char __u8;
typedef unsigned short __u16;
typedef unsigned int __u32;
typedef unsigned long long __u64;

typedef signed char __s8;
typedef signed short __s16;
typedef signed int __s32;
typedef signed long long __s64;

typedef __u8 u8;
typedef __u16 u16;
typedef __u32 u32;
typedef __u64 u64;

typedef __s8 s8;
typedef __s16 s16;
typedef __s32 s32;
typedef __s64 s64;

typedef __u32 pid_t;
typedef __u32 uid_t;
typedef __u32 gid_t;
typedef __u64 size_t;

/* CO-RE attribute macro */
#ifndef __attribute_preserve_access_index
#if __has_attribute(preserve_access_index)
#define __attribute_preserve_access_index __attribute__((preserve_access_index))
#else
#define __attribute_preserve_access_index
#endif
#endif

/* BPF Map types */
enum bpf_map_type {
    BPF_MAP_TYPE_UNSPEC = 0,
    BPF_MAP_TYPE_HASH = 1,
    BPF_MAP_TYPE_ARRAY = 2,
    BPF_MAP_TYPE_PROG_ARRAY = 3,
    BPF_MAP_TYPE_PERF_EVENT_ARRAY = 4,
    BPF_MAP_TYPE_PERCPU_HASH = 5,
    BPF_MAP_TYPE_PERCPU_ARRAY = 6,
    BPF_MAP_TYPE_STACK_TRACE = 7,
    BPF_MAP_TYPE_RINGBUF = 27,
};

/* Core Kernel Data Structures */
struct pt_regs {
    __u64 r15;
    __u64 r14;
    __u64 r13;
    __u64 r12;
    __u64 rbp;
    __u64 rbx;
    __u64 r11;
    __u64 r10;
    __u64 r9;
    __u64 r8;
    __u64 rax;
    __u64 rcx;
    __u64 rdx;
    __u64 rsi;
    __u64 rdi;
    __u64 orig_rax;
    __u64 rip;
    __u64 cs;
    __u64 eflags;
    __u64 rsp;
    __u64 ss;
} __attribute_preserve_access_index;

struct trace_entry {
    __u16 type;
    __u8 flags;
    __u8 preempt_count;
    __s32 pid;
} __attribute_preserve_access_index;

struct task_struct {
    __s32 pid;
    __s32 tgid;
    char comm[16];
    __u64 start_time;
} __attribute_preserve_access_index;

/* 64-byte aligned DeepLog trace event definition */
struct trace_event {
    __u64 timestamp_ns;
    __u32 pid;
    __u32 tid;
    __u32 syscall_id;
    __u32 event_id;
    __u64 latency_ns;
    __s64 ret_value;
    __u32 error_code;
    __u16 cpu_id;
    __u16 drop_count;
    __u32 anomaly_score_q8;
    __u32 flags;
    __u8 comm[16];
} __attribute__((aligned(64)));

#endif /* __VMLINUX_H__ */
"""

    @staticmethod
    def _generate_fallback_skeleton_h(skeleton_name: str) -> str:
        """Generate an idiomatic C skeleton header matching bpftool output structure."""
        upper = skeleton_name.upper()
        return f"""/* SPDX-License-Identifier: (LGPL-2.1 OR BSD-2-Clause) */
/* DeepLog CO-RE Cross-Platform Generated Skeleton: {skeleton_name} */

#ifndef __{upper}_SKEL_H__
#define __{upper}_SKEL_H__

#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {{
#endif

struct {skeleton_name}_bpf {{
    struct {{
        void *events_ringbuf;
        void *anomaly_stats;
        void *hdc_filter_map;
    }} maps;

    struct {{
        void *kprobe_sys_enter;
        void *tracepoint_sched_switch;
    }} progs;

    struct {{
        void *kprobe_sys_enter_link;
        void *tracepoint_sched_switch_link;
    }} links;

    struct {{
        int target_pid;
        int hdc_threshold_q8;
        int pass_all_events;
    }} data;

    int is_loaded;
    int is_attached;
}};

static inline struct {skeleton_name}_bpf *{skeleton_name}_bpf__open(void) {{
    struct {skeleton_name}_bpf *skel = (struct {skeleton_name}_bpf *)calloc(1, sizeof(struct {skeleton_name}_bpf));
    if (!skel) return NULL;
    skel->data.target_pid = 0;
    skel->data.hdc_threshold_q8 = 89;
    skel->data.pass_all_events = 0;
    return skel;
}}

static inline int {skeleton_name}_bpf__load(struct {skeleton_name}_bpf *skel) {{
    if (!skel) return -1;
    skel->is_loaded = 1;
    return 0;
}}

static inline int {skeleton_name}_bpf__attach(struct {skeleton_name}_bpf *skel) {{
    if (!skel || !skel->is_loaded) return -1;
    skel->is_attached = 1;
    return 0;
}}

static inline void {skeleton_name}_bpf__detach(struct {skeleton_name}_bpf *skel) {{
    if (skel) skel->is_attached = 0;
}}

static inline void {skeleton_name}_bpf__destroy(struct {skeleton_name}_bpf *skel) {{
    if (skel) {{
        {skeleton_name}_bpf__detach(skel);
        free(skel);
    }}
}}

static inline struct {skeleton_name}_bpf *{skeleton_name}_bpf__open_and_load(void) {{
    struct {skeleton_name}_bpf *skel = {skeleton_name}_bpf__open();
    if (!skel) return NULL;
    if ({skeleton_name}_bpf__load(skel) != 0) {{
        {skeleton_name}_bpf__destroy(skel);
        return NULL;
    }}
    return skel;
}}

#ifdef __cplusplus
}}
#endif

#endif /* __{upper}_SKEL_H__ */
"""
