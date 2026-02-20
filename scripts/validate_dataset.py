#!/usr/bin/env python3
"""Validate strict prompt/completion JSONL datasets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def validate(path: Path, strict_only_two_keys: bool = True) -> tuple[int, int, list[str]]:
    total = 0
    bad = 0
    errors: list[str] = []

    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                bad += 1
                errors.append(f"{path}:{lineno}: empty line")
                continue
            total += 1
            try:
                row = json.loads(line)
            except Exception as exc:  # noqa: BLE001
                bad += 1
                errors.append(f"{path}:{lineno}: invalid json: {exc}")
                continue
            if not isinstance(row, dict):
                bad += 1
                errors.append(f"{path}:{lineno}: row not object")
                continue

            keys = set(row.keys())
            required = {"prompt", "completion"}
            if not required.issubset(keys):
                bad += 1
                errors.append(f"{path}:{lineno}: missing prompt/completion")
                continue
            if strict_only_two_keys and keys != required:
                bad += 1
                errors.append(f"{path}:{lineno}: keys must be exactly prompt/completion, got {sorted(keys)}")
                continue

            prompt = row.get("prompt")
            completion = row.get("completion")
            if not isinstance(prompt, str) or not prompt.strip():
                bad += 1
                errors.append(f"{path}:{lineno}: empty/non-string prompt")
            if not isinstance(completion, str) or not completion.strip():
                bad += 1
                errors.append(f"{path}:{lineno}: empty/non-string completion")

    return total, bad, errors


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="JSONL dataset file(s)")
    ap.add_argument("--allow-extra-keys", action="store_true", help="Allow keys besides prompt/completion")
    args = ap.parse_args()

    any_bad = 0
    for p in args.paths:
        path = Path(p).resolve()
        total, bad, errors = validate(path, strict_only_two_keys=not args.allow_extra_keys)
        print(f"{path}: total={total} bad={bad}")
        if errors[:10]:
            for e in errors[:10]:
                print(f"  {e}")
        if bad > 10:
            print(f"  ... {bad - 10} more errors")
        any_bad += bad

    return 1 if any_bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
