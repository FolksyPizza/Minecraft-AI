#!/usr/bin/env python3
"""Download auxiliary public datasets and export train splits to JSONL."""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="LoRA/raw")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    kotlin_ds = load_dataset("JetBrains/Kotlin_HumanEval")
    java_ds = load_dataset("Nan-Do/instructional_code-search-net-java")

    kotlin_out = out_dir / "kotlin_dataset.jsonl"
    java_out = out_dir / "java_dataset.jsonl"

    kotlin_ds["train"].to_json(str(kotlin_out), orient="records", lines=True)
    java_ds["train"].to_json(str(java_out), orient="records", lines=True)

    print(f"Wrote {kotlin_out}")
    print(f"Wrote {java_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
