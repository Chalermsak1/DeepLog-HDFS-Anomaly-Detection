"""
DeepLog GitOps & SLM Dataset Curation Module
=============================================
Provides production-grade GitOps lifecycle management, unified patch application,
pull request synthesis with CBF safety certificates & Mermaid sequence diagrams,
GitHub webhook emulation, ArgoCD deployment synchronization with automated rollbacks,
and instruction-tuning SLM dataset curation (Alpaca, ShareGPT, HuggingFace formats).

Components:
-----------
1. `GitOpsManager`:
   Pure-Python git operations manager: branch creation, unified diff patch application,
   syntax/dry-run validation harness, status tracking, staging, commit history, and rollback.
2. `GitPRSpec`:
   Rich Pull Request descriptor with markdown rendering, AST blame provenance,
   CBF formal safety certificates, Mermaid sequence diagrams, and suggested reviewers.
3. `GitHubWebhookEmulator`:
   Webhook dispatcher with HMAC-SHA256 signing and mock GitHub/GitLab API client
   for PR management, review comments, labels, status checks, and merge events.
4. `ArgoCDSyncEngine`:
   GitOps continuous deployment simulator tracking sync states (Synced/OutOfSync),
   health states (Healthy/Degraded), automated synchronization, and safety-driven rollbacks.
5. `SLMDatasetCurator`:
   Curator and extractor for training Small Language Models (SLMs) on self-healing incidents,
   exporting instruction datasets containing (system_prompt, incident_telemetry,
   ast_cause, formal_safety_proof, git_patch).
"""

from __future__ import annotations

import ast
import copy
from dataclasses import asdict, dataclass, field
import datetime
import difflib
import enum
import hashlib
import hmac
import json
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import tempfile
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union
import uuid


def _utc_now_iso() -> str:
    """Return timezone-aware ISO 8601 UTC timestamp string."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


# ==============================================================================
# Enums and Core Value Objects
# ==============================================================================


class PRStatus(str, enum.Enum):
    """Lifecycle status of a Pull Request."""
    OPEN = "open"
    CLOSED = "closed"
    MERGED = "merged"
    DRAFT = "draft"


class SyncStatus(str, enum.Enum):
    """ArgoCD synchronization status."""
    SYNCED = "Synced"
    OUT_OF_SYNC = "OutOfSync"
    UNKNOWN = "Unknown"


class HealthStatus(str, enum.Enum):
    """ArgoCD application health status."""
    HEALTHY = "Healthy"
    PROGRESSING = "Progressing"
    DEGRADED = "Degraded"
    SUSPENDED = "Suspended"
    MISSING = "Missing"
    UNKNOWN = "Unknown"


class SLMDatasetFormat(str, enum.Enum):
    """Supported dataset formats for SLM fine-tuning."""
    ALPACA = "alpaca"
    SHAREGPT = "sharegpt"
    HUGGINGFACE = "huggingface"
    RAW = "raw"


class GitFileStatus(str, enum.Enum):
    """Status of an individual file in GitOpsManager."""
    UNMODIFIED = "unmodified"
    MODIFIED = "modified"
    ADDED = "added"
    DELETED = "deleted"
    UNTRACKED = "untracked"


@dataclass
class GitBranch:
    """Represents a Git branch reference."""
    name: str
    commit_sha: Optional[str] = None
    created_at: str = field(default_factory=_utc_now_iso)
    protected: bool = False


@dataclass
class GitCommit:
    """Represents a Git commit."""
    commit_sha: str
    parent_sha: Optional[str]
    message: str
    author_name: str
    author_email: str
    timestamp: str
    tree_snapshot: Dict[str, str] = field(default_factory=dict)
    files_changed: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert commit to dictionary."""
        return {
            "commit_sha": self.commit_sha,
            "parent_sha": self.parent_sha,
            "message": self.message,
            "author_name": self.author_name,
            "author_email": self.author_email,
            "timestamp": self.timestamp,
            "files_changed": list(self.files_changed),
        }


@dataclass
class GitStatus:
    """Git working tree and staging area status."""
    branch: str
    staged: List[str] = field(default_factory=list)
    modified: List[str] = field(default_factory=list)
    untracked: List[str] = field(default_factory=list)
    deleted: List[str] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        """Returns True if there are no staged, modified, untracked, or deleted files."""
        return not (self.staged or self.modified or self.untracked or self.deleted)

    def to_dict(self) -> Dict[str, Any]:
        """Convert status to dictionary representation."""
        return {
            "branch": self.branch,
            "is_clean": self.is_clean,
            "staged": list(self.staged),
            "modified": list(self.modified),
            "untracked": list(self.untracked),
            "deleted": list(self.deleted),
        }


@dataclass
class PatchResult:
    """Result of a unified diff patch application."""
    success: bool
    target_file: str
    applied_hunks: int = 0
    total_hunks: int = 0
    modified_lines_count: int = 0
    error: Optional[str] = None
    dry_run: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert patch result to dictionary."""
        return asdict(self)


@dataclass
class ValidationResult:
    """Result of executing the validation test harness."""
    passed: bool
    exit_code: int = 0
    stdout: str = ""
    stderr: str = ""
    test_count: int = 0
    failed_count: int = 0
    duration_seconds: float = 0.0
    issues: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert validation result to dictionary."""
        return asdict(self)


# ==============================================================================
# Unified Diff Parser & Pure-Python Patch Engine
# ==============================================================================


@dataclass
class DiffHunk:
    """Parsed single hunk from a unified diff."""
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: List[str]


@dataclass
class ParsedFilePatch:
    """Parsed per-file diff section."""
    old_file: str
    new_file: str
    hunks: List[DiffHunk]


def parse_unified_diff(diff_text: str) -> List[ParsedFilePatch]:
    """
    Parse a unified diff string into structured file patches and hunks.

    Parameters
    ----------
    diff_text : str
        Standard unified diff format text.

    Returns
    -------
    List[ParsedFilePatch]
        List of parsed file patches with line bounds and hunk contents.
    """
    patches: List[ParsedFilePatch] = []
    lines = diff_text.splitlines()
    i = 0

    while i < len(lines):
        line = lines[i]
        if line.startswith("--- "):
            old_file_line = line
            i += 1
            if i < len(lines) and lines[i].startswith("+++ "):
                new_file_line = lines[i]
                i += 1
                old_file = old_file_line[4:].strip().split("\t")[0]
                new_file = new_file_line[4:].strip().split("\t")[0]
                hunks: List[DiffHunk] = []

                while i < len(lines):
                    if lines[i].startswith("--- "):
                        break
                    if lines[i].startswith("@@ "):
                        hunk_header = lines[i]
                        i += 1
                        m = re.match(
                            r"^@@\s+-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@",
                            hunk_header,
                        )
                        if not m:
                            raise ValueError(f"Malformed unified diff hunk header: {hunk_header}")
                        old_start = int(m.group(1))
                        old_count = int(m.group(2)) if m.group(2) is not None else 1
                        new_start = int(m.group(3))
                        new_count = int(m.group(4)) if m.group(4) is not None else 1

                        hunk_lines: List[str] = []
                        while i < len(lines):
                            cur = lines[i]
                            if cur.startswith("@@ ") or cur.startswith("--- "):
                                break
                            if cur.startswith((" ", "+", "-", "\\")):
                                hunk_lines.append(cur)
                                i += 1
                            elif cur == "":
                                hunk_lines.append(" ")
                                i += 1
                            else:
                                break
                        hunks.append(DiffHunk(old_start, old_count, new_start, new_count, hunk_lines))
                    else:
                        i += 1
                patches.append(ParsedFilePatch(old_file, new_file, hunks))
            else:
                i += 1
        else:
            i += 1

    return patches


def _strip_file_path(path: str, strip_count: int = 1) -> str:
    """Strip leading directory components (like a/ or b/)."""
    if path == "/dev/null":
        return path
    parts = path.replace("\\", "/").split("/")
    if len(parts) > strip_count:
        return "/".join(parts[strip_count:])
    return path


def apply_hunks_to_content(
    original_content: str,
    hunks: List[DiffHunk],
    reverse: bool = False,
) -> Tuple[str, int]:
    """
    Apply a list of diff hunks to original string content.

    Returns
    -------
    Tuple[str, int]
        Patched content string and count of applied hunks.
    """
    if not hunks:
        return original_content, 0

    lines = original_content.splitlines(keepends=True)
    if not lines and original_content == "":
        lines = []

    # If reversing, invert hunks: swap + and -, swap line coordinates
    active_hunks = hunks
    if reverse:
        inverted_hunks: List[DiffHunk] = []
        for h in hunks:
            inv_lines: List[str] = []
            for l in h.lines:
                if l.startswith("+"):
                    inv_lines.append("-" + l[1:])
                elif l.startswith("-"):
                    inv_lines.append("+" + l[1:])
                else:
                    inv_lines.append(l)
            inverted_hunks.append(
                DiffHunk(
                    old_start=h.new_start,
                    old_count=h.new_count,
                    new_start=h.old_start,
                    new_count=h.old_count,
                    lines=inv_lines,
                )
            )
        active_hunks = inverted_hunks

    applied = 0
    # Apply hunks in order. Track line offset between original and modified text.
    offset = 0

    for hunk in active_hunks:
        old_expected_lines: List[str] = []
        new_replacement_lines: List[str] = []

        for hl in hunk.lines:
            if hl.startswith("\\"):
                continue
            prefix = hl[0]
            val = hl[1:]
            if not val.endswith("\n"):
                val = val + "\n"

            if prefix == " ":
                old_expected_lines.append(val)
                new_replacement_lines.append(val)
            elif prefix == "-":
                old_expected_lines.append(val)
            elif prefix == "+":
                new_replacement_lines.append(val)

        # Determine target insertion/replacement index
        start_idx = max(0, hunk.old_start - 1 + offset)

        # Context matching with tolerance for trailing whitespace
        match_found = False
        actual_start = start_idx

        # Try exact start_idx first, then search window +/- 5 lines
        search_range = [start_idx] + [
            start_idx + delta
            for delta in (1, -1, 2, -2, 3, -3, 4, -4, 5, -5)
            if 0 <= start_idx + delta <= len(lines)
        ]

        if not old_expected_lines:
            # Pure addition (e.g. creating new file or empty file)
            match_found = True
            actual_start = min(start_idx, len(lines))
        else:
            for cand in search_range:
                cand_slice = lines[cand : cand + len(old_expected_lines)]
                if len(cand_slice) == len(old_expected_lines):
                    if all(
                        c.rstrip("\r\n") == e.rstrip("\r\n")
                        for c, e in zip(cand_slice, old_expected_lines)
                    ):
                        match_found = True
                        actual_start = cand
                        break

        if not match_found and old_expected_lines:
            raise ValueError(
                f"Patch context mismatch at line {hunk.old_start}: expected "
                f"{len(old_expected_lines)} lines matching hunk context."
            )

        # Splice lines
        end_idx = actual_start + len(old_expected_lines)
        lines[actual_start:end_idx] = new_replacement_lines
        offset += len(new_replacement_lines) - len(old_expected_lines)
        applied += 1

    result = "".join(lines)
    return result, applied


# ==============================================================================
# GitOpsManager
# ==============================================================================


class GitOpsManager:
    """
    Pure-Python GitOps workspace manager.

    Capabilities:
    - Branch creation, switching, listing, deletion, and protection.
    - Unified diff patch application with dry-run verification and rollback.
    - Validation test harness executing AST syntax checks and custom test suites.
    - Complete Git state tracking (staged, modified, untracked, clean, commits).
    - Conflict detection between concurrent branch modifications.
    """

    def __init__(
        self,
        repo_dir: Optional[Union[str, Path]] = None,
        initial_branch: str = "main",
        author_name: str = "DeepLog Autopilot",
        author_email: str = "deeplog@autonomous.sre",
    ):
        self._temp_dir: Optional[tempfile.TemporaryDirectory] = None
        if repo_dir is None:
            self._temp_dir = tempfile.TemporaryDirectory(prefix="deeplog_gitops_")
            self.repo_dir = Path(self._temp_dir.name)
        else:
            self.repo_dir = Path(repo_dir)

        self.repo_dir.mkdir(parents=True, exist_ok=True)
        self.default_author_name = author_name
        self.default_author_email = author_email

        self.current_branch = initial_branch
        self.branches: Dict[str, GitBranch] = {
            initial_branch: GitBranch(name=initial_branch, commit_sha=None)
        }
        self.commits: Dict[str, GitCommit] = {}
        self.branch_heads: Dict[str, Optional[str]] = {initial_branch: None}
        self.staged_files: Set[str] = set()

        # In-memory snapshot of files per commit
        self._last_committed_snapshot: Dict[str, str] = {}

    def cleanup(self) -> None:
        """Clean up temporary directory if created."""
        if self._temp_dir is not None:
            try:
                self._temp_dir.cleanup()
            except Exception:
                pass
            self._temp_dir = None

    # --------------------------------------------------------------------------
    # Branch Management
    # --------------------------------------------------------------------------

    def create_branch(
        self,
        branch_name: str,
        from_branch: Optional[str] = None,
        checkout: bool = True,
    ) -> GitBranch:
        """
        Create a new Git branch.

        Parameters
        ----------
        branch_name : str
            Target branch name (e.g. 'deeplog/remediation-E14').
        from_branch : Optional[str]
            Source branch to fork from (default: current_branch).
        checkout : bool
            Whether to switch HEAD to the new branch immediately.

        Returns
        -------
        GitBranch
            Created branch descriptor.
        """
        if branch_name in self.branches:
            raise ValueError(f"Branch '{branch_name}' already exists.")

        source = from_branch or self.current_branch
        if source not in self.branches:
            raise ValueError(f"Source branch '{source}' does not exist.")

        head_sha = self.branch_heads.get(source)
        branch = GitBranch(name=branch_name, commit_sha=head_sha)
        self.branches[branch_name] = branch
        self.branch_heads[branch_name] = head_sha

        if checkout:
            self.current_branch = branch_name

        return branch

    def checkout(self, branch_name: str, create: bool = False) -> None:
        """
        Switch current working branch.

        Parameters
        ----------
        branch_name : str
            Name of branch to checkout.
        create : bool
            If True, creates branch if missing.
        """
        if branch_name not in self.branches:
            if create:
                self.create_branch(branch_name, checkout=True)
                return
            raise ValueError(f"Branch '{branch_name}' does not exist.")

        self.current_branch = branch_name

    def list_branches(self) -> List[str]:
        """List all branch names."""
        return sorted(list(self.branches.keys()))

    def get_current_branch(self) -> str:
        """Get the active branch name."""
        return self.current_branch

    def delete_branch(self, branch_name: str, force: bool = False) -> bool:
        """
        Delete a branch.

        Cannot delete current branch or protected branch without force.
        """
        if branch_name == self.current_branch:
            raise ValueError(f"Cannot delete active branch '{branch_name}'. Checkout another branch first.")
        if branch_name not in self.branches:
            return False
        if self.branches[branch_name].protected and not force:
            raise PermissionError(f"Branch '{branch_name}' is protected.")

        del self.branches[branch_name]
        self.branch_heads.pop(branch_name, None)
        return True

    # --------------------------------------------------------------------------
    # Working Directory Inspection & Git Status
    # --------------------------------------------------------------------------

    def _read_disk_files(self) -> Dict[str, str]:
        """Read all tracked and untracked files relative to repo root."""
        disk_files: Dict[str, str] = {}
        for root, dirs, files in os.walk(self.repo_dir):
            # Ignore hidden files, venv, pycache
            dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__" and d != "venv"]
            for f in files:
                if f.startswith("."):
                    continue
                full_path = Path(root) / f
                rel_path = str(full_path.relative_to(self.repo_dir)).replace("\\", "/")
                try:
                    disk_files[rel_path] = full_path.read_text(encoding="utf-8")
                except Exception:
                    # Fallback for binary / read errors
                    pass
        return disk_files

    def status(self) -> GitStatus:
        """
        Inspect working tree and staging area status.

        Returns
        -------
        GitStatus
            Staged, modified, untracked, deleted file status.
        """
        current_disk = self._read_disk_files()
        head_sha = self.branch_heads.get(self.current_branch)
        head_commit = self.commits.get(head_sha) if head_sha else None
        committed_tree = head_commit.tree_snapshot if head_commit else self._last_committed_snapshot

        staged: List[str] = sorted(list(self.staged_files))
        modified: List[str] = []
        untracked: List[str] = []
        deleted: List[str] = []

        # Check disk files against committed tree
        for rel_path, content in current_disk.items():
            if rel_path in committed_tree:
                if content != committed_tree[rel_path] and rel_path not in staged:
                    modified.append(rel_path)
            else:
                if rel_path not in staged:
                    untracked.append(rel_path)

        # Check for deleted files
        for rel_path in committed_tree:
            if rel_path not in current_disk and rel_path not in staged:
                deleted.append(rel_path)

        return GitStatus(
            branch=self.current_branch,
            staged=sorted(staged),
            modified=sorted(modified),
            untracked=sorted(untracked),
            deleted=sorted(deleted),
        )

    def stage(self, file_paths: Union[str, Sequence[str]]) -> None:
        """Add file path(s) to staging area."""
        if isinstance(file_paths, str):
            file_paths = [file_paths]
        for p in file_paths:
            normalized = p.replace("\\", "/").lstrip("/")
            self.staged_files.add(normalized)

    def stage_all(self) -> None:
        """Stage all modified, untracked, and deleted files."""
        current_disk = self._read_disk_files()
        head_sha = self.branch_heads.get(self.current_branch)
        head_commit = self.commits.get(head_sha) if head_sha else None
        committed_tree = head_commit.tree_snapshot if head_commit else self._last_committed_snapshot

        for p in current_disk:
            self.staged_files.add(p)
        for p in committed_tree:
            if p not in current_disk:
                self.staged_files.add(p)

    def unstage(self, file_path: str) -> None:
        """Remove file from staging area."""
        normalized = file_path.replace("\\", "/").lstrip("/")
        self.staged_files.discard(normalized)

    # --------------------------------------------------------------------------
    # Commits, History, and Rollbacks
    # --------------------------------------------------------------------------

    def commit(
        self,
        message: str,
        author_name: Optional[str] = None,
        author_email: Optional[str] = None,
    ) -> GitCommit:
        """
        Record staged changes into a new GitCommit on current branch.

        Parameters
        ----------
        message : str
            Commit message summary.
        author_name : Optional[str]
            Author committer name.
        author_email : Optional[str]
            Author committer email.

        Returns
        -------
        GitCommit
            Newly minted commit record.
        """
        disk_files = self._read_disk_files()
        parent_sha = self.branch_heads.get(self.current_branch)
        parent_commit = self.commits.get(parent_sha) if parent_sha else None
        old_tree = parent_commit.tree_snapshot if parent_commit else dict(self._last_committed_snapshot)

        # New tree: copy old tree, update with staged disk files, remove deleted files
        new_tree = dict(old_tree)
        files_changed: List[str] = []

        if not self.staged_files:
            # Auto-stage everything if staging is empty
            self.stage_all()

        for staged_p in self.staged_files:
            if staged_p in disk_files:
                if old_tree.get(staged_p) != disk_files[staged_p]:
                    new_tree[staged_p] = disk_files[staged_p]
                    files_changed.append(staged_p)
            else:
                if staged_p in new_tree:
                    del new_tree[staged_p]
                    files_changed.append(staged_p)

        timestamp = _utc_now_iso()
        sha_raw = f"{parent_sha or 'root'}:{message}:{timestamp}:{sorted(list(new_tree.keys()))}"
        commit_sha = hashlib.sha1(sha_raw.encode("utf-8")).hexdigest()

        commit = GitCommit(
            commit_sha=commit_sha,
            parent_sha=parent_sha,
            message=message,
            author_name=author_name or self.default_author_name,
            author_email=author_email or self.default_author_email,
            timestamp=timestamp,
            tree_snapshot=new_tree,
            files_changed=sorted(files_changed),
        )

        self.commits[commit_sha] = commit
        self.branch_heads[self.current_branch] = commit_sha
        self.branches[self.current_branch].commit_sha = commit_sha
        self._last_committed_snapshot = new_tree
        self.staged_files.clear()
        return commit

    def get_log(self, max_count: int = 20, branch: Optional[str] = None) -> List[GitCommit]:
        """Traverse commit history backwards from branch HEAD."""
        target_branch = branch or self.current_branch
        head_sha = self.branch_heads.get(target_branch)
        result: List[GitCommit] = []
        cur_sha = head_sha

        while cur_sha and len(result) < max_count:
            commit = self.commits.get(cur_sha)
            if not commit:
                break
            result.append(commit)
            cur_sha = commit.parent_sha

        return result

    def revert_last_commit(self) -> Optional[GitCommit]:
        """
        Revert working directory to parent of current commit (pure rollback).

        Returns
        -------
        Optional[GitCommit]
            Parent commit restored, or None if at root.
        """
        head_sha = self.branch_heads.get(self.current_branch)
        if not head_sha:
            return None
        current_commit = self.commits.get(head_sha)
        if not current_commit or not current_commit.parent_sha:
            return None

        parent_commit = self.commits.get(current_commit.parent_sha)
        if not parent_commit:
            return None

        # Restore working tree on disk to parent tree
        current_disk = self._read_disk_files()
        for p in current_disk:
            if p not in parent_commit.tree_snapshot:
                full_p = self.repo_dir / p
                if full_p.exists():
                    full_p.unlink()

        for p, content in parent_commit.tree_snapshot.items():
            full_p = self.repo_dir / p
            full_p.parent.mkdir(parents=True, exist_ok=True)
            full_p.write_text(content, encoding="utf-8")

        self.branch_heads[self.current_branch] = parent_commit.commit_sha
        self.branches[self.current_branch].commit_sha = parent_commit.commit_sha
        self._last_committed_snapshot = parent_commit.tree_snapshot
        self.staged_files.clear()
        return parent_commit

    # --------------------------------------------------------------------------
    # Patch Application & Dry Run Harness
    # --------------------------------------------------------------------------

    def apply_patch(
        self,
        patch_diff: str,
        check_only: bool = False,
        reverse: bool = False,
        strip: int = 1,
    ) -> PatchResult:
        """
        Apply a unified diff patch to files in the repository.

        Parameters
        ----------
        patch_diff : str
            Unified diff string.
        check_only : bool
            If True, verifies that the patch applies cleanly without writing changes.
        reverse : bool
            If True, reverses the patch (swaps additions and deletions).
        strip : int
            Number of leading directory components to strip (default: 1 for a/ and b/).

        Returns
        -------
        PatchResult
            Status indicating success, applied hunks, target file, and any error message.
        """
        if not patch_diff or not patch_diff.strip():
            return PatchResult(
                success=False,
                target_file="",
                error="Empty patch diff provided.",
                dry_run=check_only,
            )

        try:
            parsed_patches = parse_unified_diff(patch_diff)
        except Exception as e:
            return PatchResult(
                success=False,
                target_file="",
                error=f"Diff parsing error: {e}",
                dry_run=check_only,
            )

        if not parsed_patches:
            return PatchResult(
                success=False,
                target_file="",
                error="No valid file patches found in unified diff.",
                dry_run=check_only,
            )

        total_hunks = sum(len(p.hunks) for p in parsed_patches)
        applied_hunks = 0
        modified_lines_count = 0
        primary_target_file = ""

        # First pass: check and compute patched contents for all files
        planned_writes: Dict[Path, str] = {}

        for fp in parsed_patches:
            target_rel = _strip_file_path(fp.new_file if not reverse else fp.old_file, strip)
            if target_rel == "/dev/null":
                target_rel = _strip_file_path(fp.old_file if not reverse else fp.new_file, strip)

            if not primary_target_file:
                primary_target_file = target_rel

            target_path = self.repo_dir / target_rel

            # Read existing content
            if target_path.exists():
                orig_content = target_path.read_text(encoding="utf-8")
            else:
                orig_content = ""

            try:
                patched_content, hunks_count = apply_hunks_to_content(
                    orig_content,
                    fp.hunks,
                    reverse=reverse,
                )
            except Exception as e:
                return PatchResult(
                    success=False,
                    target_file=target_rel,
                    applied_hunks=applied_hunks,
                    total_hunks=total_hunks,
                    error=f"Hunk application failed for '{target_rel}': {e}",
                    dry_run=check_only,
                )

            applied_hunks += hunks_count
            planned_writes[target_path] = patched_content

            # Count modified lines
            diff_lines = difflib.unified_diff(
                orig_content.splitlines(),
                patched_content.splitlines(),
            )
            modified_lines_count += sum(
                1 for l in diff_lines if l.startswith(("+", "-")) and not l.startswith(("+++", "---"))
            )

        # Second pass: commit to disk if not check_only
        if not check_only:
            for path, new_content in planned_writes.items():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(new_content, encoding="utf-8")

        return PatchResult(
            success=True,
            target_file=primary_target_file,
            applied_hunks=applied_hunks,
            total_hunks=total_hunks,
            modified_lines_count=modified_lines_count,
            dry_run=check_only,
        )

    def dry_run_patch(self, patch_diff: str, reverse: bool = False) -> PatchResult:
        """Verify patch application without mutating repository files."""
        return self.apply_patch(patch_diff, check_only=True, reverse=reverse)

    # --------------------------------------------------------------------------
    # Validation Test Harness
    # --------------------------------------------------------------------------

    def run_validation_suite(
        self,
        test_commands: Optional[List[str]] = None,
        custom_validator: Optional[Callable[[], bool]] = None,
        python_syntax_check: bool = True,
        timeout_seconds: float = 10.0,
    ) -> ValidationResult:
        """
        Execute pre-merge validation tests and static syntax inspections.

        Parameters
        ----------
        test_commands : Optional[List[str]]
            Shell commands to execute in repo root.
        custom_validator : Optional[Callable[[], bool]]
            Custom Python test hook returning True on pass.
        python_syntax_check : bool
            Whether to inspect all .py files for valid AST compilation.
        timeout_seconds : float
            Maximum execution time for shell commands.

        Returns
        -------
        ValidationResult
            Validation suite execution summary.
        """
        start_time = time.time()
        issues: List[str] = []
        test_count = 0
        failed_count = 0
        stdout_buf: List[str] = []
        stderr_buf: List[str] = []

        # 1. AST syntax inspection on touched Python files
        if python_syntax_check:
            for root, _, files in os.walk(self.repo_dir):
                for f in files:
                    if f.endswith(".py"):
                        test_count += 1
                        py_path = Path(root) / f
                        try:
                            source = py_path.read_text(encoding="utf-8")
                            ast.parse(source, filename=str(py_path))
                        except SyntaxError as e:
                            failed_count += 1
                            issues.append(f"SyntaxError in {f}:{e.lineno}: {e.msg}")
                        except Exception as e:
                            failed_count += 1
                            issues.append(f"Parsing error in {f}: {e}")

        # 2. Custom callable validation
        if custom_validator is not None:
            test_count += 1
            try:
                passed = custom_validator()
                if not passed:
                    failed_count += 1
                    issues.append("Custom validation callable returned False.")
            except Exception as e:
                failed_count += 1
                issues.append(f"Custom validation exception: {e}")

        # 3. External test commands
        if test_commands:
            for cmd in test_commands:
                test_count += 1
                try:
                    res = subprocess.run(
                        cmd,
                        shell=True,
                        cwd=str(self.repo_dir),
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        timeout=timeout_seconds,
                    )
                    stdout_buf.append(res.stdout)
                    stderr_buf.append(res.stderr)
                    if res.returncode != 0:
                        failed_count += 1
                        issues.append(f"Command '{cmd}' failed with exit code {res.returncode}")
                except subprocess.TimeoutExpired:
                    failed_count += 1
                    issues.append(f"Command '{cmd}' timed out after {timeout_seconds}s")
                except Exception as e:
                    failed_count += 1
                    issues.append(f"Command '{cmd}' failed with error: {e}")

        duration = time.time() - start_time
        all_passed = (failed_count == 0)

        return ValidationResult(
            passed=all_passed,
            exit_code=0 if all_passed else 1,
            stdout="\n".join(stdout_buf),
            stderr="\n".join(stderr_buf),
            test_count=test_count,
            failed_count=failed_count,
            duration_seconds=duration,
            issues=issues,
        )

    # --------------------------------------------------------------------------
    # Diff Generation & Branch Conflict Detection
    # --------------------------------------------------------------------------

    def create_diff(self, file_path: Optional[str] = None) -> str:
        """
        Generate unified diff between working directory and HEAD commit.

        Parameters
        ----------
        file_path : Optional[str]
            Optional path to restrict diff to a single file.
        """
        head_sha = self.branch_heads.get(self.current_branch)
        head_commit = self.commits.get(head_sha) if head_sha else None
        committed_tree = head_commit.tree_snapshot if head_commit else self._last_committed_snapshot
        disk_files = self._read_disk_files()

        targets = [file_path] if file_path else sorted(list(set(disk_files.keys()) | set(committed_tree.keys())))
        diff_chunks: List[str] = []

        for p in targets:
            if not p:
                continue
            old_lines = committed_tree.get(p, "").splitlines(keepends=True)
            new_lines = disk_files.get(p, "").splitlines(keepends=True)
            if old_lines != new_lines:
                chunk = difflib.unified_diff(
                    old_lines,
                    new_lines,
                    fromfile=f"a/{p}",
                    tofile=f"b/{p}",
                )
                diff_chunks.extend(chunk)

        return "".join(diff_chunks)

    def detect_branch_conflict(self, branch_a: str, branch_b: str) -> List[str]:
        """
        Detect file-level and line-level merge conflicts between two branches.

        Returns
        -------
        List[str]
            List of conflicting file paths.
        """
        sha_a = self.branch_heads.get(branch_a)
        sha_b = self.branch_heads.get(branch_b)
        if not sha_a or not sha_b:
            return []

        tree_a = self.commits[sha_a].tree_snapshot
        tree_b = self.commits[sha_b].tree_snapshot

        conflicts: List[str] = []
        common_files = set(tree_a.keys()) & set(tree_b.keys())

        for p in common_files:
            if tree_a[p] != tree_b[p]:
                # Both branches modified the same file; check if changes conflict
                lines_a = tree_a[p].splitlines()
                lines_b = tree_b[p].splitlines()
                diff_ratio = difflib.SequenceMatcher(None, lines_a, lines_b).quick_ratio()
                if diff_ratio < 0.99:
                    conflicts.append(p)

        return sorted(conflicts)


# ==============================================================================
# GitPRSpec & Provenance Descriptors
# ==============================================================================


@dataclass
class ASTBlameProvenance:
    """AST root cause lineage metadata."""
    commit_sha: str
    author: str
    commit_date: str
    commit_message: str
    file_path: str
    line_number: int
    method_name: str = ""
    class_name: str = ""
    exception_type: str = ""
    code_line: str = ""
    pr_reference: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert provenance to dictionary."""
        return asdict(self)

    @classmethod
    def from_blame_info(cls, blame_obj: Any) -> ASTBlameProvenance:
        """Instantiate from GitBlameInfo or dictionary."""
        if hasattr(blame_obj, "to_dict"):
            d = blame_obj.to_dict()
        elif isinstance(blame_obj, dict):
            d = blame_obj
        else:
            d = getattr(blame_obj, "__dict__", {})

        return cls(
            commit_sha=d.get("commit_sha") or d.get("commit_hash", "HEAD~1"),
            author=d.get("author") or d.get("author_name", "Distributed Systems Engineer"),
            commit_date=d.get("commit_date", _utc_now_iso()),
            commit_message=d.get("commit_message", "Refactor connection handler"),
            file_path=d.get("file_path", "src/core/handler.py"),
            line_number=int(d.get("line_number", 42)),
            method_name=d.get("method_name", "handle_request"),
            class_name=d.get("class_name", "RequestHandler"),
            exception_type=d.get("exception_type", "SocketTimeoutException"),
            code_line=d.get("code_line", ""),
            pr_reference=d.get("pr_reference"),
        )


@dataclass
class CBFSafetyCertificate:
    """Formal mathematical Control Barrier Function (CBF) safety certificate."""
    is_safe: bool
    margin: float
    invariants_evaluated: List[Dict[str, Any]] = field(default_factory=list)
    violated_invariants: List[str] = field(default_factory=list)
    active_invariants: List[str] = field(default_factory=list)
    lie_derivatives: Dict[str, float] = field(default_factory=dict)
    lyapunov_drift: Optional[float] = None
    lyapunov_stable: Optional[bool] = None
    proof_summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert certificate to dictionary."""
        return asdict(self)

    @classmethod
    def from_safety_report(cls, report_obj: Any) -> CBFSafetyCertificate:
        """Instantiate from SafetyProofReport, dictionary, or duck-typed object."""
        if hasattr(report_obj, "to_dict"):
            d = report_obj.to_dict()
        elif isinstance(report_obj, dict):
            d = report_obj
        else:
            d = getattr(report_obj, "__dict__", {})

        return cls(
            is_safe=bool(d.get("is_safe", True)),
            margin=float(d.get("margin", 0.35)),
            invariants_evaluated=list(d.get("invariants_evaluated", [])),
            violated_invariants=list(d.get("violated_invariants", [])),
            active_invariants=list(d.get("active_invariants", [])),
            lie_derivatives=dict(d.get("lie_derivatives", {})),
            lyapunov_drift=d.get("lyapunov_drift"),
            lyapunov_stable=d.get("lyapunov_stable"),
            proof_summary=str(d.get("mathematical_proof") or d.get("proof_summary", "Formal CBF verified: safe")),
        )


@dataclass
class GitPRSpec:
    """
    Rich Pull Request specification for autonomous GitOps remediation.

    Contains Markdown summary, AST blame provenance, CBF formal safety certificate,
    Mermaid sequence diagram, unified git diff, and suggested reviewers.
    """

    pr_id: str
    title: str
    branch_name: str
    target_file: str
    summary: str
    patch_diff: str
    base_branch: str = "main"
    ast_blame: Optional[ASTBlameProvenance] = None
    safety_certificate: Optional[CBFSafetyCertificate] = None
    suggested_reviewers: List[str] = field(default_factory=list)
    labels: List[str] = field(default_factory=lambda: ["deeplog-autofix", "gitops", "cbf-verified"])
    status: PRStatus = PRStatus.OPEN
    created_at: str = field(default_factory=_utc_now_iso)
    mermaid_diagram: str = ""
    validation_result: Optional[ValidationResult] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.suggested_reviewers:
            self.suggested_reviewers = self._derive_reviewers()
        if not self.mermaid_diagram:
            self.mermaid_diagram = self.generate_mermaid_sequence_diagram()

    def _derive_reviewers(self) -> List[str]:
        """Derive recommended reviewers from AST blame and target file path."""
        reviewers: Set[str] = set()
        if self.ast_blame and self.ast_blame.author:
            clean_author = re.sub(r"[^a-zA-Z0-9_-]", "", self.ast_blame.author.lower().replace(" ", "-"))
            reviewers.add(f"@{clean_author}")

        target_lower = self.target_file.lower()
        if "datanode" in target_lower or "hdfs" in target_lower:
            reviewers.add("@hdfs-core-team")
        elif "k8s" in target_lower or "deploy" in target_lower or "yaml" in target_lower:
            reviewers.add("@kubernetes-platform-ops")
        elif "network" in target_lower or "socket" in target_lower:
            reviewers.add("@networking-lead")
        else:
            reviewers.add("@sre-oncall-lead")

        reviewers.add("@deeplog-ai-guard")
        return sorted(list(reviewers))

    def generate_mermaid_sequence_diagram(self) -> str:
        """
        Generate a strictly valid Mermaid sequence diagram depicting self-healing workflow.

        Returns
        -------
        str
            Mermaid sequenceDiagram definition.
        """
        safe_str = "PASSED" if (not self.safety_certificate or self.safety_certificate.is_safe) else "BLOCKED"
        target_name = Path(self.target_file).name or "source_code"

        diagram = (
            "sequenceDiagram\n"
            "    autonumber\n"
            "    actor SRE as SRE Operator\n"
            "    participant D as DeepLog Detector\n"
            "    participant AST as AST Blame Tracer\n"
            "    participant CBF as CBF Safety Verifier\n"
            "    participant GitOps as GitOps Manager\n"
            "    participant GH as GitHub Webhook\n"
            "    participant Argo as ArgoCD Sync Engine\n\n"
            "    D->>AST: Log Anomaly Detected\n"
            f"    AST->>CBF: Trace Root Cause ({target_name})\n"
            f"    CBF->>GitOps: Issue Safety Proof [{safe_str}]\n"
            f"    GitOps->>GH: Open Remediation PR #{self.pr_id}\n"
            "    GH->>Argo: Webhook Dispatch (pull_request.merged)\n"
            "    Argo->>Argo: Sync Manifests & Health Check\n"
            "    alt Invariant Violation / Degraded\n"
            "        Argo-->>GitOps: Automated Rollback Triggered\n"
            "    else Invariants Satisfied\n"
            "        Argo-->>SRE: Self-Healing Confirmed Healthy\n"
            "    end"
        )
        return diagram

    def to_markdown(self) -> str:
        """
        Render full GitHub-Flavored Markdown PR document.

        Returns
        -------
        str
            Formatted Markdown text ready for GitHub/GitLab PR body.
        """
        lines = [
            f"# {self.title}",
            "",
            f"**PR Identifier:** `{self.pr_id}` | **Branch:** `{self.branch_name}` &rarr; `{self.base_branch}` | **Status:** `{self.status.value.upper()}`",
            "",
            "## 1. Executive Summary",
            "",
            self.summary.strip(),
            "",
            "## 2. Operational Sequence Workflow",
            "",
            "```mermaid",
            self.mermaid_diagram.strip(),
            "```",
            "",
            "## 3. AST Blame & Provenance",
            "",
        ]

        if self.ast_blame:
            lines.extend([
                "| Attribute | Value |",
                "| :--- | :--- |",
                f"| **Target File** | `{self.ast_blame.file_path}:{self.ast_blame.line_number}` |",
                f"| **Method / Class** | `{self.ast_blame.class_name}#{self.ast_blame.method_name}` |",
                f"| **Faulting Commit** | `{self.ast_blame.commit_sha}` |",
                f"| **Original Author** | {self.ast_blame.author} |",
                f"| **Commit Message** | {self.ast_blame.commit_message} |",
                f"| **Exception Type** | `{self.ast_blame.exception_type}` |",
                "",
            ])
        else:
            lines.extend([f"Target file: `{self.target_file}`", ""])

        lines.extend([
            "## 4. Formal CBF Safety Certificate",
            "",
        ])

        if self.safety_certificate:
            cert = self.safety_certificate
            status_emoji = "PASSED" if cert.is_safe else "VIOLATION"
            lines.extend([
                f"- **Overall Safety Invariant:** `{status_emoji}`",
                f"- **Minimum Barrier Margin $h(x)$:** `{cert.margin:+.4f}`",
                f"- **Discrete Lyapunov Contraction:** `{'Stable' if cert.lyapunov_stable else 'Unstable/N/A'}` "
                f"(Drift $\\Delta V$: `{cert.lyapunov_drift if cert.lyapunov_drift is not None else 'N/A'}`)",
                f"- **Proof Statement:** {cert.proof_summary}",
                "",
            ])
            if cert.invariants_evaluated:
                lines.extend([
                    "| Invariant Name | Threshold | Actual Value | Margin | Satisfied |",
                    "| :--- | :--- | :--- | :--- | :--- |",
                ])
                for inv in cert.invariants_evaluated:
                    lines.append(
                        f"| {inv.get('name', 'Invariant')} | {inv.get('threshold', 0.0)} | "
                        f"{inv.get('actual', 0.0)} | {inv.get('margin', 0.0):+.4f} | "
                        f"{'YES' if inv.get('satisfied', True) else 'NO'} |"
                    )
                lines.append("")
        else:
            lines.extend(["*No formal CBF certificate attached.*", ""])

        lines.extend([
            "## 5. Unified Git Diff Patch",
            "",
            "```diff",
            self.patch_diff.strip(),
            "```",
            "",
            "## 6. Suggested Reviewers & Metadata",
            "",
            f"- **Suggested Reviewers:** {', '.join(self.suggested_reviewers)}",
            f"- **Labels:** {', '.join(f'`{lbl}`' for lbl in self.labels)}",
            f"- **Timestamp:** `{self.created_at}`",
        ])

        return "\n".join(lines)

    def save_markdown(self, output_path: Union[str, Path]) -> Path:
        """Write PR markdown document to file."""
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_markdown(), encoding="utf-8")
        return p.resolve()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize PR spec to dictionary."""
        return {
            "pr_id": self.pr_id,
            "title": self.title,
            "branch_name": self.branch_name,
            "base_branch": self.base_branch,
            "target_file": self.target_file,
            "summary": self.summary,
            "patch_diff": self.patch_diff,
            "ast_blame": self.ast_blame.to_dict() if self.ast_blame else None,
            "safety_certificate": self.safety_certificate.to_dict() if self.safety_certificate else None,
            "suggested_reviewers": list(self.suggested_reviewers),
            "labels": list(self.labels),
            "status": self.status.value,
            "created_at": self.created_at,
            "mermaid_diagram": self.mermaid_diagram,
            "validation_result": self.validation_result.to_dict() if self.validation_result else None,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> GitPRSpec:
        """Deserialize PR spec from dictionary."""
        blame = ASTBlameProvenance(**d["ast_blame"]) if d.get("ast_blame") else None
        cert = CBFSafetyCertificate(**d["safety_certificate"]) if d.get("safety_certificate") else None
        val_res = ValidationResult(**d["validation_result"]) if d.get("validation_result") else None

        return cls(
            pr_id=d["pr_id"],
            title=d["title"],
            branch_name=d["branch_name"],
            base_branch=d.get("base_branch", "main"),
            target_file=d["target_file"],
            summary=d["summary"],
            patch_diff=d["patch_diff"],
            ast_blame=blame,
            safety_certificate=cert,
            suggested_reviewers=d.get("suggested_reviewers", []),
            labels=d.get("labels", ["deeplog-autofix"]),
            status=PRStatus(d.get("status", "open")),
            created_at=d.get("created_at", _utc_now_iso()),
            mermaid_diagram=d.get("mermaid_diagram", ""),
            validation_result=val_res,
            metadata=d.get("metadata", {}),
        )


# ==============================================================================
# GitHub Webhook Emulator & Mock Git API Client
# ==============================================================================


@dataclass
class WebhookEvent:
    """Delivered webhook event payload."""
    event_id: str
    event_type: str
    action: Optional[str]
    payload: Dict[str, Any]
    signature: str
    timestamp: str


@dataclass
class WebhookDelivery:
    """Record of a webhook dispatch attempt."""
    delivery_id: str
    subscriber_id: str
    event: WebhookEvent
    status_code: int = 200
    success: bool = True
    error: Optional[str] = None


class GitHubWebhookEmulator:
    """
    Mock GitHub/GitLab API client and Webhook Dispatcher.

    Capabilities:
    - Webhook subscription with HMAC-SHA256 signature verification.
    - Full PR lifecycle: creation, inline/general comments, labeling, merging.
    - Commit status checks (pending, success, failure).
    - Event dispatching triggering subscriber callbacks and logging delivery audit history.
    """

    def __init__(self, default_secret: str = "deeplog-gitops-secret"):
        self.default_secret = default_secret
        self._subscribers: Dict[str, Dict[str, Any]] = {}
        self.delivery_history: List[WebhookDelivery] = []
        self._prs: Dict[int, Dict[str, Any]] = {}
        self._comments: Dict[int, List[Dict[str, Any]]] = {}
        self._commit_statuses: Dict[str, List[Dict[str, Any]]] = {}
        self._next_pr_number = 1

    # --------------------------------------------------------------------------
    # Webhook Subscription & Dispatch
    # --------------------------------------------------------------------------

    def register_webhook(
        self,
        events: Sequence[str],
        callback: Callable[[WebhookEvent], Any],
        secret: Optional[str] = None,
    ) -> str:
        """
        Register a webhook subscriber callback.

        Parameters
        ----------
        events : Sequence[str]
            List of subscribed event names (e.g. ['pull_request', 'push', '*']).
        callback : Callable[[WebhookEvent], Any]
            Callback invoked on matching event dispatch.
        secret : Optional[str]
            Secret key used for HMAC-SHA256 payload signing.

        Returns
        -------
        str
            Unique subscriber registration ID.
        """
        sub_id = str(uuid.uuid4())
        self._subscribers[sub_id] = {
            "events": set(events),
            "callback": callback,
            "secret": secret or self.default_secret,
        }
        return sub_id

    def unregister_webhook(self, subscriber_id: str) -> bool:
        """Unregister a webhook subscriber by ID."""
        return self._subscribers.pop(subscriber_id, None) is not None

    def compute_signature(self, payload_bytes: bytes, secret: str) -> str:
        """Compute GitHub-compliant HMAC-SHA256 signature (sha256=...)."""
        mac = hmac.new(secret.encode("utf-8"), payload_bytes, hashlib.sha256)
        return f"sha256={mac.hexdigest()}"

    def verify_signature(self, payload_bytes: bytes, signature: str, secret: str) -> bool:
        """Verify HMAC-SHA256 signature."""
        expected = self.compute_signature(payload_bytes, secret)
        return hmac.compare_digest(expected, signature)

    def dispatch_event(
        self,
        event_type: str,
        payload: Dict[str, Any],
        action: Optional[str] = None,
    ) -> List[WebhookDelivery]:
        """
        Dispatch a webhook event to all matching subscribers.

        Parameters
        ----------
        event_type : str
            Event name (e.g. 'pull_request', 'push', 'status').
        payload : Dict[str, Any]
            Event body payload.
        action : Optional[str]
            Sub-action (e.g. 'opened', 'closed', 'synchronize').

        Returns
        -------
        List[WebhookDelivery]
            Deliveries recorded.
        """
        deliveries: List[WebhookDelivery] = []
        payload_copy = copy.deepcopy(payload)
        if action and "action" not in payload_copy:
            payload_copy["action"] = action

        payload_bytes = json.dumps(payload_copy, sort_keys=True).encode("utf-8")
        timestamp = _utc_now_iso()
        event_id = str(uuid.uuid4())

        matching_subs = [
            (sub_id, sub_info)
            for sub_id, sub_info in self._subscribers.items()
            if "*" in sub_info["events"] or event_type in sub_info["events"]
        ]

        if not matching_subs:
            sig = self.compute_signature(payload_bytes, self.default_secret)
            event = WebhookEvent(
                event_id=event_id,
                event_type=event_type,
                action=action,
                payload=payload_copy,
                signature=sig,
                timestamp=timestamp,
            )
            delivery = WebhookDelivery(
                delivery_id=str(uuid.uuid4()),
                subscriber_id="audit-log",
                event=event,
                success=True,
                status_code=200,
            )
            deliveries.append(delivery)
            self.delivery_history.append(delivery)
        else:
            for sub_id, sub_info in matching_subs:
                sig = self.compute_signature(payload_bytes, sub_info["secret"])
                event = WebhookEvent(
                    event_id=event_id,
                    event_type=event_type,
                    action=action,
                    payload=payload_copy,
                    signature=sig,
                    timestamp=timestamp,
                )

                delivery = WebhookDelivery(
                    delivery_id=str(uuid.uuid4()),
                    subscriber_id=sub_id,
                    event=event,
                )

                try:
                    sub_info["callback"](event)
                    delivery.success = True
                    delivery.status_code = 200
                except Exception as e:
                    delivery.success = False
                    delivery.status_code = 500
                    delivery.error = str(e)

                deliveries.append(delivery)
                self.delivery_history.append(delivery)

        return deliveries

    # --------------------------------------------------------------------------
    # Mock GitHub API Client Methods
    # --------------------------------------------------------------------------

    def create_pull_request(self, spec: GitPRSpec) -> Dict[str, Any]:
        """
        Create a pull request in the mock GitHub repository.

        Dispatches 'pull_request' webhook with action 'opened'.
        """
        pr_number = self._next_pr_number
        self._next_pr_number += 1

        pr_data: Dict[str, Any] = {
            "number": pr_number,
            "id": spec.pr_id,
            "title": spec.title,
            "state": "open",
            "body": spec.to_markdown(),
            "head": {"ref": spec.branch_name},
            "base": {"ref": spec.base_branch},
            "labels": [{"name": lbl} for lbl in spec.labels],
            "requested_reviewers": [{"login": r.lstrip("@")} for r in spec.suggested_reviewers],
            "merged": False,
            "mergeable": True,
            "created_at": spec.created_at,
            "patch_diff": spec.patch_diff,
            "spec": spec.to_dict(),
        }

        self._prs[pr_number] = pr_data
        self._comments[pr_number] = []

        # Dispatch webhook
        self.dispatch_event(
            event_type="pull_request",
            action="opened",
            payload={"pull_request": pr_data, "number": pr_number},
        )

        return pr_data

    def get_pull_request(self, pr_number: int) -> Optional[Dict[str, Any]]:
        """Get PR by number."""
        return self._prs.get(pr_number)

    def list_pull_requests(self, state: str = "open") -> List[Dict[str, Any]]:
        """List PRs by state ('open', 'closed', 'all')."""
        if state == "all":
            return list(self._prs.values())
        return [pr for pr in self._prs.values() if pr.get("state") == state]

    def add_review_comment(
        self,
        pr_number: int,
        comment: str,
        file_path: Optional[str] = None,
        line: Optional[int] = None,
        author: str = "deeplog-bot",
    ) -> Dict[str, Any]:
        """Add a comment or inline code review comment to a PR."""
        if pr_number not in self._prs:
            raise KeyError(f"PR #{pr_number} not found.")

        comment_record: Dict[str, Any] = {
            "id": str(uuid.uuid4()),
            "pr_number": pr_number,
            "author": author,
            "body": comment,
            "path": file_path,
            "line": line,
            "created_at": _utc_now_iso(),
        }
        self._comments[pr_number].append(comment_record)

        self.dispatch_event(
            event_type="pull_request_review_comment" if file_path else "issue_comment",
            action="created",
            payload={"comment": comment_record, "pull_request": self._prs[pr_number]},
        )
        return comment_record

    def get_review_comments(self, pr_number: int) -> List[Dict[str, Any]]:
        """Get all review comments for a PR."""
        return list(self._comments.get(pr_number, []))

    def add_labels(self, pr_number: int, labels: Sequence[str]) -> List[str]:
        """Add labels to a PR and dispatch 'labeled' event."""
        if pr_number not in self._prs:
            raise KeyError(f"PR #{pr_number} not found.")

        existing = {lbl["name"] for lbl in self._prs[pr_number].get("labels", [])}
        for lbl in labels:
            if lbl not in existing:
                self._prs[pr_number]["labels"].append({"name": lbl})

        updated = [l["name"] for l in self._prs[pr_number]["labels"]]
        self.dispatch_event(
            event_type="pull_request",
            action="labeled",
            payload={"pull_request": self._prs[pr_number], "added_labels": list(labels)},
        )
        return updated

    def set_commit_status(
        self,
        commit_sha: str,
        state: str,
        context: str = "deeplog/safety-verification",
        description: str = "",
        target_url: str = "",
    ) -> Dict[str, Any]:
        """
        Record a commit status check (pending, success, failure, error).
        """
        valid_states = {"pending", "success", "failure", "error"}
        if state not in valid_states:
            raise ValueError(f"Invalid status state '{state}'. Expected one of {valid_states}")

        status_record: Dict[str, Any] = {
            "sha": commit_sha,
            "state": state,
            "context": context,
            "description": description,
            "target_url": target_url,
            "created_at": _utc_now_iso(),
        }

        if commit_sha not in self._commit_statuses:
            self._commit_statuses[commit_sha] = []
        self._commit_statuses[commit_sha].append(status_record)

        self.dispatch_event(
            event_type="status",
            payload={"status": status_record, "sha": commit_sha},
        )
        return status_record

    def get_commit_statuses(self, commit_sha: str) -> List[Dict[str, Any]]:
        """Get all commit status checks for a given SHA."""
        return list(self._commit_statuses.get(commit_sha, []))

    def merge_pull_request(
        self,
        pr_number: int,
        commit_title: Optional[str] = None,
        merge_method: str = "squash",
        merged_by: str = "gitops-bot",
    ) -> Dict[str, Any]:
        """
        Merge an open pull request.

        Dispatches 'pull_request' (closed, merged=True) and 'push' events.
        """
        if pr_number not in self._prs:
            raise KeyError(f"PR #{pr_number} not found.")

        pr = self._prs[pr_number]
        if pr["state"] != "open":
            raise ValueError(f"Cannot merge PR #{pr_number} in state '{pr['state']}'.")

        pr["state"] = "closed"
        pr["merged"] = True
        pr["merged_at"] = _utc_now_iso()
        pr["merged_by"] = {"login": merged_by}
        pr["merge_method"] = merge_method

        merge_sha = hashlib.sha1(f"merge:{pr_number}:{pr['merged_at']}".encode("utf-8")).hexdigest()
        pr["merge_commit_sha"] = merge_sha

        # 1. Dispatch PR closed/merged event
        self.dispatch_event(
            event_type="pull_request",
            action="closed",
            payload={"pull_request": pr, "merged": True, "number": pr_number},
        )

        # 2. Dispatch push event onto base branch
        self.dispatch_event(
            event_type="push",
            payload={
                "ref": f"refs/heads/{pr['base']['ref']}",
                "after": merge_sha,
                "head_commit": {
                    "id": merge_sha,
                    "message": commit_title or f"Merge PR #{pr_number}: {pr['title']}",
                    "author": {"name": merged_by},
                },
                "repository": {"name": "deeplog-gitops"},
            },
        )

        return pr

    def close_pull_request(self, pr_number: int, closed_by: str = "gitops-bot") -> Dict[str, Any]:
        """Close PR without merging."""
        if pr_number not in self._prs:
            raise KeyError(f"PR #{pr_number} not found.")

        pr = self._prs[pr_number]
        pr["state"] = "closed"
        pr["merged"] = False
        pr["closed_at"] = _utc_now_iso()

        self.dispatch_event(
            event_type="pull_request",
            action="closed",
            payload={"pull_request": pr, "merged": False, "number": pr_number},
        )
        return pr

    def clear_history(self) -> None:
        """Clear delivery audit history."""
        self.delivery_history.clear()


# ==============================================================================
# ArgoCD Continuous Deployment Sync Engine & Rollback Triggers
# ==============================================================================


@dataclass
class ArgoRevisionRecord:
    """Historical record of an applied GitOps revision."""
    revision_id: str
    commit_sha: str
    synced_at: str
    sync_status: SyncStatus
    health_status: HealthStatus
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SyncOperationResult:
    """Outcome of an ArgoCD sync operation."""
    success: bool
    app_name: str
    revision: str
    sync_status: SyncStatus
    health_status: HealthStatus
    message: str
    duration_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert sync result to dictionary."""
        return {
            "success": self.success,
            "app_name": self.app_name,
            "revision": self.revision,
            "sync_status": self.sync_status.value,
            "health_status": self.health_status.value,
            "message": self.message,
            "duration_seconds": self.duration_seconds,
        }


@dataclass
class HealthEvaluationResult:
    """ArgoCD application health assessment."""
    app_name: str
    health_status: HealthStatus
    is_healthy: bool
    degraded_reasons: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    evaluated_at: str = field(default_factory=_utc_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        """Convert health evaluation to dictionary."""
        return {
            "app_name": self.app_name,
            "health_status": self.health_status.value,
            "is_healthy": self.is_healthy,
            "degraded_reasons": list(self.degraded_reasons),
            "metrics": dict(self.metrics),
            "evaluated_at": self.evaluated_at,
        }


@dataclass
class RollbackResult:
    """Outcome of an automated rollback trigger."""
    success: bool
    app_name: str
    from_revision: str
    to_revision: str
    reason: str
    new_health_status: HealthStatus
    timestamp: str = field(default_factory=_utc_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        """Convert rollback result to dictionary."""
        return {
            "success": self.success,
            "app_name": self.app_name,
            "from_revision": self.from_revision,
            "to_revision": self.to_revision,
            "reason": self.reason,
            "new_health_status": self.new_health_status.value,
            "timestamp": self.timestamp,
        }


@dataclass
class ArgoApplication:
    """ArgoCD Application Custom Resource representation."""
    app_name: str
    repo_url: str
    target_revision: str = "main"
    path: str = "deploy/"
    auto_sync: bool = True
    auto_rollback: bool = True
    prune: bool = False
    active_revision: Optional[str] = None
    sync_status: SyncStatus = SyncStatus.OUT_OF_SYNC
    health_status: HealthStatus = HealthStatus.HEALTHY
    history: List[ArgoRevisionRecord] = field(default_factory=list)


class ArgoCDSyncEngine:
    """
    Simulates GitOps deployment synchronization, health assessment, and safety rollbacks.

    Capabilities:
    - Tracks application sync and health states.
    - Automated sync triggered by GitHub push/merge webhooks.
    - Continuous health check evaluation against error rates, pod statuses, and CBF margins.
    - Automatic rollback to previous known-healthy revision when health degrades or CBF margin is violated.
    """

    def __init__(self, initial_revision: str = "v1.0.0-baseline"):
        self.initial_revision = initial_revision
        self.applications: Dict[str, ArgoApplication] = {}
        self.rollback_history: List[RollbackResult] = []

    def register_application(
        self,
        app_name: str,
        repo_url: str = "https://github.com/deeplog/production-deployments.git",
        target_revision: str = "main",
        path: str = "k8s/overlays/production",
        auto_sync: bool = True,
        auto_rollback: bool = True,
        prune: bool = False,
    ) -> ArgoApplication:
        """Register a new ArgoCD application for synchronization."""
        app = ArgoApplication(
            app_name=app_name,
            repo_url=repo_url,
            target_revision=target_revision,
            path=path,
            auto_sync=auto_sync,
            auto_rollback=auto_rollback,
            prune=prune,
            active_revision=self.initial_revision,
            sync_status=SyncStatus.OUT_OF_SYNC,
            health_status=HealthStatus.HEALTHY,
        )
        self.applications[app_name] = app
        return app

    def get_application(self, app_name: str) -> Optional[ArgoApplication]:
        """Get application by name."""
        return self.applications.get(app_name)

    def list_applications(self) -> List[ArgoApplication]:
        """List all registered applications."""
        return list(self.applications.values())

    # --------------------------------------------------------------------------
    # Deployment Sync
    # --------------------------------------------------------------------------

    def sync(
        self,
        app_name: str,
        revision: Optional[str] = None,
        prune: bool = False,
        dry_run: bool = False,
    ) -> SyncOperationResult:
        """
        Synchronize application state to target revision.

        Parameters
        ----------
        app_name : str
            Name of application.
        revision : Optional[str]
            Target commit SHA or tag. If omitted, uses app.target_revision.
        prune : bool
            Whether to prune orphaned Kubernetes resources.
        dry_run : bool
            Preview sync without mutating active deployment.

        Returns
        -------
        SyncOperationResult
            Outcome of sync operation.
        """
        if app_name not in self.applications:
            raise KeyError(f"ArgoCD Application '{app_name}' not registered.")

        app = self.applications[app_name]
        target_rev = revision or app.target_revision

        if dry_run:
            return SyncOperationResult(
                success=True,
                app_name=app_name,
                revision=target_rev,
                sync_status=app.sync_status,
                health_status=app.health_status,
                message=f"Dry run sync preview for revision '{target_rev}' succeeded.",
                duration_seconds=0.01,
            )

        start_time = time.time()
        prev_rev = app.active_revision or "none"
        app.active_revision = target_rev
        app.sync_status = SyncStatus.SYNCED
        app.health_status = HealthStatus.HEALTHY

        record = ArgoRevisionRecord(
            revision_id=str(uuid.uuid4())[:8],
            commit_sha=target_rev,
            synced_at=_utc_now_iso(),
            sync_status=app.sync_status,
            health_status=app.health_status,
            details={"previous_revision": prev_rev, "pruned": prune},
        )
        app.history.append(record)

        duration = time.time() - start_time
        return SyncOperationResult(
            success=True,
            app_name=app_name,
            revision=target_rev,
            sync_status=app.sync_status,
            health_status=app.health_status,
            message=f"Successfully synchronized application '{app_name}' to revision '{target_rev}'.",
            duration_seconds=duration,
        )

    # --------------------------------------------------------------------------
    # Health Assessment & Automated Rollback
    # --------------------------------------------------------------------------

    def evaluate_health(
        self,
        app_name: str,
        error_rate: float = 0.0,
        pod_health: Optional[Dict[str, str]] = None,
        cbf_margin: Optional[float] = None,
        cluster_state: Optional[Any] = None,
    ) -> HealthEvaluationResult:
        """
        Evaluate application health metrics and trigger automated rollback if degraded.

        Degraded criteria:
        - HTTP 5xx error rate > 0.05 (5%)
        - CBF barrier margin < 0.0 (formal safety invariant violation!)
        - Pods in CrashLoopBackOff, Error, or OOMKilled
        - Cluster CPU / Memory headroom < 0.05
        """
        if app_name not in self.applications:
            raise KeyError(f"ArgoCD Application '{app_name}' not registered.")

        app = self.applications[app_name]
        degraded_reasons: List[str] = []
        metrics: Dict[str, float] = {"error_rate": error_rate}

        if error_rate > 0.05:
            degraded_reasons.append(f"High error rate detected: {error_rate * 100:.1f}% exceeds 5.0% threshold")

        if cbf_margin is not None:
            metrics["cbf_margin"] = cbf_margin
            if cbf_margin < 0.0:
                degraded_reasons.append(f"Formal CBF safety violation: margin h(x) = {cbf_margin:.4f} < 0.0")

        if pod_health:
            for pod, st in pod_health.items():
                if st in {"CrashLoopBackOff", "Error", "Failed", "OOMKilled"}:
                    degraded_reasons.append(f"Pod '{pod}' failed with status '{st}'")

        if cluster_state is not None:
            cpu_h = getattr(cluster_state, "cpu_headroom", 0.5)
            mem_h = getattr(cluster_state, "memory_headroom", 0.5)
            metrics["cpu_headroom"] = cpu_h
            metrics["memory_headroom"] = mem_h
            if cpu_h < 0.05:
                degraded_reasons.append(f"Critical CPU exhaustion: headroom {cpu_h * 100:.1f}% < 5.0%")
            if mem_h < 0.05:
                degraded_reasons.append(f"Critical memory exhaustion: headroom {mem_h * 100:.1f}% < 5.0%")

        is_degraded = len(degraded_reasons) > 0
        assessed_status = HealthStatus.DEGRADED if is_degraded else HealthStatus.HEALTHY
        if is_degraded:
            app.health_status = HealthStatus.DEGRADED
            # Automated rollback trigger if enabled
            if app.auto_rollback:
                self.trigger_rollback(
                    app_name=app_name,
                    reason=f"Automated rollback triggered: {'; '.join(degraded_reasons)}",
                )
        else:
            app.health_status = HealthStatus.HEALTHY

        return HealthEvaluationResult(
            app_name=app_name,
            health_status=assessed_status,
            is_healthy=not is_degraded,
            degraded_reasons=degraded_reasons,
            metrics=metrics,
        )

    def trigger_rollback(
        self,
        app_name: str,
        target_revision: Optional[str] = None,
        reason: str = "Manual or automated rollback initiated",
    ) -> RollbackResult:
        """
        Revert application deployment to a prior healthy revision.

        Parameters
        ----------
        app_name : str
            Application name.
        target_revision : Optional[str]
            Explicit revision SHA to revert to. If None, looks up the most recent
            healthy revision in history before current active revision.

        Returns
        -------
        RollbackResult
            Rollback execution result.
        """
        if app_name not in self.applications:
            raise KeyError(f"ArgoCD Application '{app_name}' not registered.")

        app = self.applications[app_name]
        from_rev = app.active_revision or "unknown"
        to_rev = target_revision

        if to_rev is None:
            # Find last healthy revision in history
            for record in reversed(app.history):
                if record.commit_sha != from_rev and record.health_status == HealthStatus.HEALTHY:
                    to_rev = record.commit_sha
                    break

        if to_rev is None:
            # Fall back to initial revision baseline
            to_rev = self.initial_revision

        app.active_revision = to_rev
        app.sync_status = SyncStatus.SYNCED
        app.health_status = HealthStatus.HEALTHY

        # Record rollback in application revision history
        record = ArgoRevisionRecord(
            revision_id=str(uuid.uuid4())[:8],
            commit_sha=to_rev,
            synced_at=_utc_now_iso(),
            sync_status=app.sync_status,
            health_status=app.health_status,
            details={"rollback_from": from_rev, "reason": reason},
        )
        app.history.append(record)

        result = RollbackResult(
            success=True,
            app_name=app_name,
            from_revision=from_rev,
            to_revision=to_rev,
            reason=reason,
            new_health_status=app.health_status,
        )
        self.rollback_history.append(result)
        return result

    # --------------------------------------------------------------------------
    # Integration with Webhook Emulator
    # --------------------------------------------------------------------------

    def connect_webhook_emulator(
        self,
        emulator: GitHubWebhookEmulator,
        app_name: str,
    ) -> str:
        """
        Register a webhook subscriber on GitHubWebhookEmulator to auto-sync on push/merge.

        Returns
        -------
        str
            Subscriber ID.
        """
        def _on_event(event: WebhookEvent) -> None:
            if event.event_type == "push":
                after_sha = event.payload.get("after")
                if after_sha and app_name in self.applications:
                    app = self.applications[app_name]
                    if app.auto_sync:
                        self.sync(app_name=app_name, revision=after_sha)
            elif event.event_type == "pull_request" and event.action == "closed":
                pr = event.payload.get("pull_request", {})
                if pr.get("merged"):
                    merge_sha = pr.get("merge_commit_sha")
                    if merge_sha and app_name in self.applications:
                        app = self.applications[app_name]
                        if app.auto_sync:
                            self.sync(app_name=app_name, revision=merge_sha)

        return emulator.register_webhook(
            events=["push", "pull_request"],
            callback=_on_event,
        )


# ==============================================================================
# SLM Dataset Curator (Alpaca / ShareGPT / HuggingFace Formats)
# ==============================================================================


@dataclass
class SLMSample:
    """
    Structured self-healing incident training sample for Small Language Models.

    Tuple components:
    - system_prompt
    - incident_telemetry
    - ast_cause
    - formal_safety_proof
    - git_patch
    """

    system_prompt: str
    incident_telemetry: Dict[str, Any]
    ast_cause: Dict[str, Any]
    formal_safety_proof: Dict[str, Any]
    git_patch: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_raw(self) -> Dict[str, Any]:
        """Convert sample to raw DeepLog tuple format."""
        return {
            "system_prompt": self.system_prompt,
            "incident_telemetry": self.incident_telemetry,
            "ast_cause": self.ast_cause,
            "formal_safety_proof": self.formal_safety_proof,
            "git_patch": self.git_patch,
            "metadata": self.metadata,
        }

    def to_alpaca(self) -> Dict[str, str]:
        """
        Convert to Stanford Alpaca instruction-tuning schema:
        {"instruction": ..., "input": ..., "output": ...}
        """
        instruction = (
            "Analyze the distributed system incident telemetry, pinpoint the source AST root cause, "
            "verify formal Control Barrier Function (CBF) safety invariants, and synthesize a production "
            "unified git diff patch."
        )

        input_payload = {
            "incident_telemetry": self.incident_telemetry,
            "context": self.metadata.get("category", "DISTRIBUTED_SYSTEM_ANOMALY"),
        }
        input_str = json.dumps(input_payload, indent=2)

        output_parts = [
            "### AST Root Cause Diagnosis",
            f"- File: `{self.ast_cause.get('file_path', 'unknown')}` Line `{self.ast_cause.get('line_number', 0)}`",
            f"- Class/Method: `{self.ast_cause.get('class_name', '')}#{self.ast_cause.get('method_name', '')}`",
            f"- Blamed Commit: `{self.ast_cause.get('commit_sha', 'HEAD~1')}` by {self.ast_cause.get('author', 'Engineer')}",
            f"- Exception: `{self.ast_cause.get('exception_type', 'SystemException')}`",
            "",
            "### Formal CBF Safety Verification",
            f"- Safe: `{'YES' if self.formal_safety_proof.get('is_safe', True) else 'NO'}`",
            f"- Barrier Margin: `{self.formal_safety_proof.get('margin', 0.0):+.4f}`",
            f"- Lyapunov Stability: `{'STABLE' if self.formal_safety_proof.get('lyapunov_stable', True) else 'UNSTABLE'}`",
            f"- Proof: {self.formal_safety_proof.get('proof_summary', 'Invariants satisfied.')}",
            "",
            "### Unified Git Diff Patch",
            "```diff",
            self.git_patch.strip(),
            "```",
        ]
        output_str = "\n".join(output_parts)

        return {
            "instruction": instruction,
            "input": input_str,
            "output": output_str,
        }

    def to_sharegpt(self) -> Dict[str, Any]:
        """
        Convert to ShareGPT multi-turn conversation format:
        {"conversations": [{"from": "system", "value": ...}, {"from": "human", "value": ...}, {"from": "gpt", "value": ...}]}
        """
        alp = self.to_alpaca()
        return {
            "conversations": [
                {"from": "system", "value": self.system_prompt},
                {
                    "from": "human",
                    "value": f"{alp['instruction']}\n\nIncident Telemetry:\n{alp['input']}",
                },
                {"from": "gpt", "value": alp["output"]},
            ]
        }

    def to_huggingface(self) -> Dict[str, Any]:
        """
        Convert to HuggingFace ChatML / messages schema:
        {"messages": [{"role": "system", "content": ...}, {"role": "user", "content": ...}, {"role": "assistant", "content": ...}]}
        """
        alp = self.to_alpaca()
        return {
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": f"{alp['instruction']}\n\nIncident Telemetry:\n{alp['input']}",
                },
                {"role": "assistant", "content": alp["output"]},
            ]
        }


DEFAULT_SLM_SYSTEM_PROMPT = (
    "You are DeepLog-Omni Autopilot, an expert Autonomous Site Reliability Engineer (SRE). "
    "Your objective is to ingest high-frequency telemetry anomalies, map them through neuro-symbolic "
    "AST code duality, verify provable safety invariants using Control Barrier Functions (CBFs), and "
    "synthesize production-ready unified git diff patches."
)


class SLMDatasetCurator:
    """
    Curator for Small Language Model (SLM) instruction-tuning datasets.

    Extracts incident histories and formats them into Alpaca, ShareGPT,
    and HuggingFace JSONL fine-tuning corpora.
    """

    def __init__(self, system_prompt: str = DEFAULT_SLM_SYSTEM_PROMPT):
        self.system_prompt = system_prompt
        self.samples: List[SLMSample] = []

    def add_sample(self, sample: SLMSample) -> None:
        """Add a curated sample to the collection."""
        self.samples.append(sample)

    def create_sample(
        self,
        incident_telemetry: Dict[str, Any],
        ast_cause: Dict[str, Any],
        formal_safety_proof: Dict[str, Any],
        git_patch: str,
        system_prompt: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SLMSample:
        """Construct and register an SLMSample."""
        sample = SLMSample(
            system_prompt=system_prompt or self.system_prompt,
            incident_telemetry=incident_telemetry,
            ast_cause=ast_cause,
            formal_safety_proof=formal_safety_proof,
            git_patch=git_patch,
            metadata=metadata or {},
        )
        self.add_sample(sample)
        return sample

    def from_incident_run(
        self,
        incident_report: Any,
        patch_proposal: Any,
        blame_info: Optional[Any] = None,
        safety_proof: Optional[Any] = None,
        system_prompt: Optional[str] = None,
    ) -> SLMSample:
        """
        Extract an SLMSample from DeepLog runtime incident run objects.
        """
        # 1. Telemetry
        if hasattr(incident_report, "to_dict"):
            telemetry = incident_report.to_dict()
        elif isinstance(incident_report, dict):
            telemetry = incident_report
        else:
            telemetry = {
                "summary": str(getattr(incident_report, "summary", "Log anomaly detected")),
                "anomaly_score": float(getattr(incident_report, "anomaly_score", 0.95)),
                "templates": getattr(incident_report, "event_templates", ["E14"]),
            }

        # 2. AST Cause
        if blame_info:
            blame_dict = ASTBlameProvenance.from_blame_info(blame_info).to_dict()
        elif hasattr(patch_proposal, "blame_info") and patch_proposal.blame_info:
            blame_dict = ASTBlameProvenance.from_blame_info(patch_proposal.blame_info).to_dict()
        else:
            blame_dict = {
                "file_path": getattr(patch_proposal, "target_file", "org/apache/hadoop/hdfs/DataXceiver.java"),
                "class_name": getattr(patch_proposal, "target_class", "DataXceiver"),
                "method_name": getattr(patch_proposal, "target_method", "writeBlock"),
                "line_number": int(getattr(patch_proposal, "faulting_line", 248)),
                "commit_sha": "a7b3c9d1",
                "author": "Distributed Systems Engineer",
                "exception_type": "SocketTimeoutException",
            }

        # 3. Safety Proof
        if safety_proof:
            safety_dict = CBFSafetyCertificate.from_safety_report(safety_proof).to_dict()
        else:
            safety_dict = {
                "is_safe": True,
                "margin": 0.32,
                "lyapunov_stable": True,
                "proof_summary": "All CBF invariants satisfied with positive margin.",
            }

        # 4. Patch
        patch_str = getattr(patch_proposal, "patch_diff", str(patch_proposal))

        return self.create_sample(
            incident_telemetry=telemetry,
            ast_cause=blame_dict,
            formal_safety_proof=safety_dict,
            git_patch=patch_str,
            system_prompt=system_prompt,
            metadata={"category": getattr(patch_proposal, "category", "INCIDENT_REMEDIATION")},
        )

    # --------------------------------------------------------------------------
    # Synthetic Incidents Generator
    # --------------------------------------------------------------------------

    def generate_synthetic_incidents(
        self,
        count: int = 25,
        seed: int = 42,
    ) -> List[SLMSample]:
        """
        Generate varied, deterministic synthetic incident samples across canonical failure modes.

        Failure archetypes:
        1. TIMEOUT_EXHAUSTION: Socket timeout on datanode block transfer (E14)
        2. RESOURCE_LEAK: Unclosed socket / FD leak in BlockReceiver (E7)
        3. DEADLOCK_CONCURRENCY: Lock inversion in NameNode FSNamesystem (E29)
        4. MIRROR_FAILOVER: Pipeline mirror disconnection in DataStreamer (E18)
        5. BUFFER_OVERFLOW: Heap exhaustion in packet streaming buffer (E26)
        """
        rng = random.Random(seed)
        archetypes = [
            {
                "category": "TIMEOUT_EXHAUSTION",
                "event_id": "E14",
                "file": "org/apache/hadoop/hdfs/server/datanode/DataXceiver.java",
                "cls": "DataXceiver",
                "method": "writeBlock",
                "line": 248,
                "author": "Alice Chen <alice@apache.org>",
                "exception": "java.net.SocketTimeoutException",
                "invariant": "cluster_cpu_headroom >= 0.15",
                "diff": (
                    "--- a/org/apache/hadoop/hdfs/server/datanode/DataXceiver.java\n"
                    "+++ b/org/apache/hadoop/hdfs/server/datanode/DataXceiver.java\n"
                    "@@ -248,2 +248,4 @@\n"
                    "-    socket.setSoTimeout(1000);\n"
                    "+    // DeepLog Autopilot: Exponential backoff socket timeout\n"
                    "+    int timeout = Math.min(60000, 1000 * (1 << retryCount));\n"
                    "+    socket.setSoTimeout(timeout);\n"
                ),
            },
            {
                "category": "RESOURCE_LEAK",
                "event_id": "E7",
                "file": "org/apache/hadoop/hdfs/server/datanode/BlockReceiver.java",
                "cls": "BlockReceiver",
                "method": "receiveBlock",
                "line": 182,
                "author": "Bob Miller <bmiller@apache.org>",
                "exception": "java.io.IOException: Too many open files",
                "invariant": "file_descriptor_usage <= 0.80",
                "diff": (
                    "--- a/org/apache/hadoop/hdfs/server/datanode/BlockReceiver.java\n"
                    "+++ b/org/apache/hadoop/hdfs/server/datanode/BlockReceiver.java\n"
                    "@@ -182,2 +182,6 @@\n"
                    "-    FileOutputStream out = new FileOutputStream(blockFile);\n"
                    "-    out.write(buffer);\n"
                    "+    // DeepLog Autopilot: Deterministic try-with-resources\n"
                    "+    try (FileOutputStream out = new FileOutputStream(blockFile)) {\n"
                    "+        out.write(buffer);\n"
                    "+    }\n"
                ),
            },
            {
                "category": "DEADLOCK_CONCURRENCY",
                "event_id": "E29",
                "file": "org/apache/hadoop/hdfs/server/namenode/FSNamesystem.java",
                "cls": "FSNamesystem",
                "method": "writeLockInterruptibly",
                "line": 412,
                "author": "Charlie Zhang <czhang@apache.org>",
                "exception": "java.lang.IllegalMonitorStateException",
                "invariant": "lock_wait_seconds <= 2.0",
                "diff": (
                    "--- a/org/apache/hadoop/hdfs/server/namenode/FSNamesystem.java\n"
                    "+++ b/org/apache/hadoop/hdfs/server/namenode/FSNamesystem.java\n"
                    "@@ -412,2 +412,5 @@\n"
                    "-    namesystemLock.writeLock().lock();\n"
                    "+    // DeepLog Autopilot: Timed lock acquisition to prevent deadlocks\n"
                    "+    if (!namesystemLock.writeLock().tryLock(5, TimeUnit.SECONDS)) {\n"
                    "+        throw new LockAcquisitionTimeoutException(\"Write lock acquisition timed out\");\n"
                    "+    }\n"
                ),
            },
            {
                "category": "MIRROR_FAILOVER",
                "event_id": "E18",
                "file": "org/apache/hadoop/hdfs/server/datanode/DataStreamer.java",
                "cls": "DataStreamer",
                "method": "setupPipelineForBlockRecovery",
                "line": 310,
                "author": "Dave Vance <dvance@apache.org>",
                "exception": "org.apache.hadoop.hdfs.protocol.PipelineAckException",
                "invariant": "traffic_drop <= 0.05",
                "diff": (
                    "--- a/org/apache/hadoop/hdfs/server/datanode/DataStreamer.java\n"
                    "+++ b/org/apache/hadoop/hdfs/server/datanode/DataStreamer.java\n"
                    "@@ -310,2 +310,5 @@\n"
                    "-    connectToMirror(nodes[0]);\n"
                    "+    // DeepLog Autopilot: Circuit-breaker mirror failover with fallback node\n"
                    "+    DatanodeInfo targetNode = selectHealthyMirror(nodes);\n"
                    "+    connectToMirror(targetNode);\n"
                ),
            },
            {
                "category": "BUFFER_OVERFLOW",
                "event_id": "E26",
                "file": "org/apache/hadoop/hdfs/server/datanode/DataXceiverServer.java",
                "cls": "DataXceiverServer",
                "method": "allocateBufferPool",
                "line": 95,
                "author": "Elena Rossi <erossi@apache.org>",
                "exception": "java.lang.OutOfMemoryError: Java heap space",
                "invariant": "memory_headroom >= 0.20",
                "diff": (
                    "--- a/org/apache/hadoop/hdfs/server/datanode/DataXceiverServer.java\n"
                    "+++ b/org/apache/hadoop/hdfs/server/datanode/DataXceiverServer.java\n"
                    "@@ -95,2 +95,5 @@\n"
                    "-    byte[] packetBuffer = new byte[unboundedLength];\n"
                    "+    // DeepLog Autopilot: Bounded buffer pool with backpressure check\n"
                    "+    int boundedSize = Math.min(unboundedLength, MAX_PACKET_BUFFER_SIZE);\n"
                    "+    byte[] packetBuffer = bufferPool.borrow(boundedSize);\n"
                ),
            },
        ]

        generated: List[SLMSample] = []
        for i in range(count):
            arch = archetypes[i % len(archetypes)]
            cycle = i // len(archetypes)
            line_offset = cycle * 10
            line_num = arch["line"] + line_offset
            jitter = rng.uniform(0.01, 0.15)
            margin = round(0.20 + rng.uniform(0.05, 0.25), 4)
            anomaly_score = round(0.85 + rng.uniform(0.05, 0.14), 4)

            # In diff, adjust line numbers to match line_num if cycle > 0
            diff_text = arch["diff"]
            if cycle > 0:
                diff_text = diff_text.replace(f"@@ -{arch['line']},", f"@@ -{line_num},")

            telemetry = {
                "incident_id": f"INC-2026-{1000 + i}",
                "event_template": arch["event_id"],
                "anomaly_score": anomaly_score,
                "log_sequence": [arch["event_id"] for _ in range(rng.randint(3, 8))],
                "time_delta_delay_sec": round(1.5 + rng.uniform(0.5, 4.0), 3),
                "queue_depth": rng.randint(25, 120),
                "cpu_utilization": round(0.70 + jitter, 3),
            }

            ast_cause = {
                "file_path": arch["file"],
                "class_name": arch["cls"],
                "method_name": arch["method"],
                "line_number": line_num,
                "commit_sha": hashlib.sha1(f"commit:{i}:{arch['file']}".encode("utf-8")).hexdigest()[:8],
                "author": arch["author"],
                "exception_type": arch["exception"],
            }

            safety_proof = {
                "is_safe": True,
                "margin": margin,
                "invariants_evaluated": [
                    {
                        "name": arch["invariant"],
                        "threshold": 0.15,
                        "actual": round(0.15 + margin, 4),
                        "margin": margin,
                        "satisfied": True,
                    }
                ],
                "lyapunov_drift": -0.12,
                "lyapunov_stable": True,
                "proof_summary": f"CBF condition dot{{h}}(x) + alpha(h(x)) >= 0 certified with margin {margin}.",
            }

            sample = self.create_sample(
                incident_telemetry=telemetry,
                ast_cause=ast_cause,
                formal_safety_proof=safety_proof,
                git_patch=diff_text,
                metadata={
                    "category": arch["category"],
                    "incident_index": i,
                    "generated_at": _utc_now_iso(),
                },
            )
            generated.append(sample)

        return generated

    # --------------------------------------------------------------------------
    # Dataset Filtering, Deduplication & Splitting
    # --------------------------------------------------------------------------

    def filter_samples(
        self,
        category: Optional[str] = None,
        min_margin: Optional[float] = None,
    ) -> List[SLMSample]:
        """Filter samples by root cause category or minimum safety margin."""
        res: List[SLMSample] = []
        for s in self.samples:
            if category and s.metadata.get("category") != category:
                continue
            if min_margin is not None:
                m = s.formal_safety_proof.get("margin", 0.0)
                if m < min_margin:
                    continue
            res.append(s)
        return res

    def deduplicate(self) -> int:
        """Remove duplicate samples based on hash of patch and AST cause."""
        seen: Set[str] = set()
        deduped: List[SLMSample] = []
        for s in self.samples:
            key = f"{s.ast_cause.get('file_path')}:{s.ast_cause.get('line_number')}:{s.git_patch.strip()}"
            if key not in seen:
                seen.add(key)
                deduped.append(s)

        removed = len(self.samples) - len(deduped)
        self.samples = deduped
        return removed

    def split(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
    ) -> Tuple[List[SLMSample], List[SLMSample], List[SLMSample]]:
        """Split dataset into train, validation, and test subsets."""
        rng = random.Random(seed)
        shuffled = list(self.samples)
        rng.shuffle(shuffled)

        n = len(shuffled)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train = shuffled[:n_train]
        val = shuffled[n_train : n_train + n_val]
        test = shuffled[n_train + n_val :]
        return train, val, test

    # --------------------------------------------------------------------------
    # Schema Validation & Export
    # --------------------------------------------------------------------------

    @staticmethod
    def validate_schema(data: Dict[str, Any], format_type: Union[SLMDatasetFormat, str]) -> bool:
        """
        Validate whether a dictionary satisfies the target SLM dataset format schema.

        Parameters
        ----------
        data : Dict[str, Any]
            Sample dictionary.
        format_type : Union[SLMDatasetFormat, str]
            Format name ('alpaca', 'sharegpt', 'huggingface', 'raw').

        Returns
        -------
        bool
            True if valid, False otherwise.
        """
        fmt = SLMDatasetFormat(format_type) if isinstance(format_type, str) else format_type

        if fmt == SLMDatasetFormat.ALPACA:
            return (
                isinstance(data.get("instruction"), str)
                and len(data["instruction"].strip()) > 0
                and "input" in data
                and isinstance(data.get("output"), str)
                and len(data["output"].strip()) > 0
            )
        elif fmt == SLMDatasetFormat.SHAREGPT:
            convs = data.get("conversations")
            if not isinstance(convs, list) or len(convs) < 2:
                return False
            valid_roles = {"system", "human", "gpt"}
            return all(
                isinstance(c, dict)
                and c.get("from") in valid_roles
                and isinstance(c.get("value"), str)
                for c in convs
            )
        elif fmt == SLMDatasetFormat.HUGGINGFACE:
            msgs = data.get("messages")
            if not isinstance(msgs, list) or len(msgs) < 2:
                return False
            valid_roles = {"system", "user", "assistant"}
            return all(
                isinstance(m, dict)
                and m.get("role") in valid_roles
                and isinstance(m.get("content"), str)
                for m in msgs
            )
        elif fmt == SLMDatasetFormat.RAW:
            required = {
                "system_prompt",
                "incident_telemetry",
                "ast_cause",
                "formal_safety_proof",
                "git_patch",
            }
            return required.issubset(data.keys())

        return False

    def export_dataset(
        self,
        output_path: Union[str, Path],
        format_type: Union[SLMDatasetFormat, str] = SLMDatasetFormat.ALPACA,
        split_ratios: Optional[Tuple[float, float, float]] = None,
    ) -> Dict[str, int]:
        """
        Export dataset to JSONL files.

        Parameters
        ----------
        output_path : Union[str, Path]
            Base file path (e.g. 'data/slm_finetuning.jsonl').
        format_type : Union[SLMDatasetFormat, str]
            Format ('alpaca', 'sharegpt', 'huggingface', 'raw').
        split_ratios : Optional[Tuple[float, float, float]]
            Optional (train, val, test) ratio tuple. If provided, writes
            <base>_train.jsonl, <base>_val.jsonl, <base>_test.jsonl.

        Returns
        -------
        Dict[str, int]
            Map of file paths to written sample counts.
        """
        fmt = SLMDatasetFormat(format_type) if isinstance(format_type, str) else format_type
        base_p = Path(output_path)
        base_p.parent.mkdir(parents=True, exist_ok=True)

        def _format_sample(s: SLMSample) -> Dict[str, Any]:
            if fmt == SLMDatasetFormat.ALPACA:
                return s.to_alpaca()
            elif fmt == SLMDatasetFormat.SHAREGPT:
                return s.to_sharegpt()
            elif fmt == SLMDatasetFormat.HUGGINGFACE:
                return s.to_huggingface()
            else:
                return s.to_raw()

        results: Dict[str, int] = {}

        if split_ratios:
            train, val, test = self.split(split_ratios[0], split_ratios[1], split_ratios[2])
            stem = base_p.stem
            suffix = base_p.suffix or ".jsonl"

            splits = [("train", train), ("val", val), ("test", test)]
            for name, subset in splits:
                split_p = base_p.parent / f"{stem}_{name}{suffix}"
                with open(split_p, "w", encoding="utf-8") as f:
                    for s in subset:
                        formatted = _format_sample(s)
                        f.write(json.dumps(formatted) + "\n")
                results[str(split_p)] = len(subset)
        else:
            with open(base_p, "w", encoding="utf-8") as f:
                for s in self.samples:
                    formatted = _format_sample(s)
                    f.write(json.dumps(formatted) + "\n")
            results[str(base_p)] = len(self.samples)

        return results

    def get_statistics(self) -> Dict[str, Any]:
        """Calculate token estimations, category counts, and dataset metrics."""
        category_counts: Dict[str, int] = {}
        total_tokens_est = 0

        for s in self.samples:
            cat = s.metadata.get("category", "UNKNOWN")
            category_counts[cat] = category_counts.get(cat, 0) + 1

            alp = s.to_alpaca()
            text_block = f"{alp['instruction']} {alp['input']} {alp['output']}"
            # Approximate 1 token ~ 4 characters
            total_tokens_est += max(1, len(text_block) // 4)

        return {
            "total_samples": len(self.samples),
            "estimated_tokens": total_tokens_est,
            "average_tokens_per_sample": round(total_tokens_est / max(1, len(self.samples)), 1),
            "categories": category_counts,
        }
