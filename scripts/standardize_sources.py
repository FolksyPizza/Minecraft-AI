#!/usr/bin/env python3
"""Standardize heterogeneous JSONL sources into prompt/completion JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def standardize_file(file_path: Path, prompt_col: str, completion_col: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            prompt = data.get(prompt_col)
            completion = data.get(completion_col)
            if isinstance(prompt, str) and prompt.strip() and isinstance(completion, str) and completion.strip():
                rows.append({"prompt": prompt.strip(), "completion": completion.strip()})
    return rows


def write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out_f:
        for row in rows:
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--java-file", default="LoRA/raw/java_dataset.jsonl")
    ap.add_argument("--kotlin-file", default="LoRA/raw/kotlin_dataset.jsonl")
    ap.add_argument("--out", default="LoRA/labeled/sources/java_kotlin_standardized.jsonl")
    args = ap.parse_args()

    mapping = [
        {"file": Path(args.java_file).resolve(), "prompt_col": "INSTRUCTION", "completion_col": "RESPONSE"},
        {"file": Path(args.kotlin_file).resolve(), "prompt_col": "prompt", "completion_col": "canonical_solution"},
    ]

    all_rows: list[dict[str, str]] = []
    for m in mapping:
        if m["file"].exists():
            all_rows.extend(standardize_file(m["file"], m["prompt_col"], m["completion_col"]))

    write_jsonl(Path(args.out).resolve(), all_rows)
    print(f"Wrote {len(all_rows)} rows -> {Path(args.out).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
