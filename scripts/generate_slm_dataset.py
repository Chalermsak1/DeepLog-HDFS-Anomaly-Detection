#!/usr/bin/env python3
"""
CLI Tool: Generate and Export SLM Instruction-Tuning Datasets for DeepLog
========================================================================
Extracts and synthesizes rich incident histories containing:
- Expert SRE system prompt
- Incident telemetry (log sequences, delay times, queue states)
- AST root cause diagnosis and commit blame lineage
- Formal Control Barrier Function (CBF) provable safety certificate
- Production-grade unified git diff patch

Formats supported:
- Alpaca (instruction / input / output)
- ShareGPT (multi-turn conversations with system, human, gpt)
- HuggingFace (messages format for ChatML / instruction tuning)
- Raw (DeepLog 5-tuple JSONL)

Usage examples:
---------------
# 1. Generate 25 Alpaca format samples
python scripts/generate_slm_dataset.py --output data/slm_alpaca.jsonl --format alpaca --num-samples 25 --stats

# 2. Generate ShareGPT format dataset with 80/10/10 train/val/test split
python scripts/generate_slm_dataset.py --output data/slm_sharegpt.jsonl --format sharegpt --num-samples 50 --split 0.8,0.1,0.1

# 3. Generate HuggingFace messages dataset with schema validation
python scripts/generate_slm_dataset.py --output data/slm_hf.jsonl --format huggingface --num-samples 30 --validate
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

try:
    from deeplog.gitops import (
        SLMDatasetCurator,
        SLMDatasetFormat,
    )
except ImportError:
    # If running before package install
    sys.path.insert(0, str(REPO_ROOT))
    from src.deeplog.gitops import (
        SLMDatasetCurator,
        SLMDatasetFormat,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and export fine-tuning JSONL datasets for Small Language Models (SLMs).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="data/slm_dataset.jsonl",
        help="Target output file path for the JSONL dataset.",
    )
    parser.add_argument(
        "-f", "--format",
        type=str,
        default="alpaca",
        choices=["alpaca", "sharegpt", "huggingface", "raw"],
        help="Target SLM training schema format.",
    )
    parser.add_argument(
        "-n", "--num-samples",
        type=int,
        default=25,
        help="Number of synthetic self-healing incident samples to synthesize.",
    )
    parser.add_argument(
        "-s", "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic incident generation.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Optional comma-separated train,val,test ratio string (e.g. '0.8,0.1,0.1').",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print dataset token estimations, category breakdowns, and summary statistics.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate all exported JSONL records against target format schema.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print("=" * 72)
    print(" DeepLog SLM Fine-Tuning Dataset Generator")
    print("=" * 72)
    print(f" Target Format : {args.format.upper()}")
    print(f" Sample Count  : {args.num_samples}")
    print(f" Random Seed   : {args.seed}")
    print(f" Output Target : {args.output}")

    curator = SLMDatasetCurator()
    print("\n[*] Synthesizing incidents across failure archetypes...")
    samples = curator.generate_synthetic_incidents(count=args.num_samples, seed=args.seed)
    print(f"[+] Successfully synthesized {len(samples)} incident samples.")

    # Parse split ratios if provided
    split_tuple = None
    if args.split:
        try:
            parts = [float(x.strip()) for x in args.split.split(",")]
            if len(parts) != 3 or abs(sum(parts) - 1.0) > 1e-4:
                raise ValueError("Split must be 3 floats summing to 1.0 (e.g. '0.8,0.1,0.1').")
            split_tuple = (parts[0], parts[1], parts[2])
            print(f"[*] Splitting dataset: Train={parts[0]:.1%}, Val={parts[1]:.1%}, Test={parts[2]:.1%}")
        except Exception as e:
            print(f"[!] Error parsing --split argument: {e}", file=sys.stderr)
            return 1

    format_enum = SLMDatasetFormat(args.format)
    print(f"[*] Exporting dataset to {args.output}...")
    written_files = curator.export_dataset(
        output_path=args.output,
        format_type=format_enum,
        split_ratios=split_tuple,
    )

    for path_str, count in written_files.items():
        print(f"  -> Wrote {count:4d} records to `{path_str}`")

    # Validation
    if args.validate:
        print("\n[*] Validating exported JSONL schemas...")
        validation_failed = False
        total_validated = 0
        for path_str in written_files:
            with open(path_str, "r", encoding="utf-8") as f:
                for line_idx, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as e:
                        print(f"[!] JSON parse error in {path_str}:{line_idx}: {e}", file=sys.stderr)
                        validation_failed = True
                        break

                    if not curator.validate_schema(record, format_enum):
                        print(f"[!] Schema validation failed in {path_str}:{line_idx}", file=sys.stderr)
                        validation_failed = True
                        break
                    total_validated += 1

        if validation_failed:
            print("[!] Validation FAILED.", file=sys.stderr)
            return 2
        print(f"[+] Schema validation PASSED ({total_validated} records verified).")

    # Statistics
    if args.stats or True:
        stats = curator.get_statistics()
        print("\n" + "-" * 72)
        print(" Dataset Summary Statistics")
        print("-" * 72)
        print(f" Total Samples         : {stats['total_samples']}")
        print(f" Estimated Token Count : {stats['estimated_tokens']:,}")
        print(f" Mean Tokens / Sample  : {stats['average_tokens_per_sample']}")
        print(" Failure Mode Breakdown:")
        for cat, cnt in sorted(stats["categories"].items()):
            pct = (cnt / max(1, stats["total_samples"])) * 100
            print(f"   - {cat:<24} : {cnt:3d} ({pct:5.1f}%)")
        print("-" * 72)

    print("\n[✓] SLM dataset generation completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
