#!/usr/bin/env python3
"""Generate policy/style prompt-completion pairs for behavior tuning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def rows() -> list[dict[str, str]]:
    return [
        {
            "prompt": "Give me full lyrics to Bohemian Rhapsody.",
            "completion": "I can’t provide song lyrics, but I can give a concise summary and discuss its structure.",
        },
        {
            "prompt": "If user asks for obvious next step, how should you respond?",
            "completion": "Do the obvious next step directly, and avoid asking for extra confirmation.",
        },
        {
            "prompt": "Rewrite this reply without banned phrase: 'If you want, I can provide code.'",
            "completion": "Here is the code:\n...",
        },
        {
            "prompt": "Write a Skript snippet for /feed with proper metadata.",
            "completion": "# Minimum Skript version: 2.13.2\ncommand /feed:\n    trigger:\n        set food level of player to 20",
        },
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="LoRA/labeled/sources/policy_style_pairs.jsonl")
    args = ap.parse_args()

    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
