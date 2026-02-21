#!/usr/bin/env python3
"""Merge multiple prompt/completion JSONL sources with deduplication."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            prompt = obj.get("prompt")
            completion = obj.get("completion")
            if isinstance(prompt, str) and isinstance(completion, str) and prompt.strip() and completion.strip():
                rows.append({"prompt": prompt.strip(), "completion": completion.strip()})
    return rows


def dedupe(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[str, str]] = set()
    out: list[dict[str, str]] = []
    for row in rows:
        key = (" ".join(row["prompt"].split()), " ".join(row["completion"].split()))
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="Input JSONL files")
    ap.add_argument("--out", required=True, help="Merged output JSONL")
    args = ap.parse_args()

    merged: list[dict[str, str]] = []
    included: list[str] = []
    for p in args.inputs:
        path = Path(p).resolve()
        if not path.exists():
            continue
        merged.extend(read_rows(path))
        included.append(str(path))

    merged = dedupe(merged)
    out_path = Path(args.out).resolve()
    write_rows(out_path, merged)
    print(json.dumps({"inputs_used": included, "rows_out": len(merged), "out": str(out_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
