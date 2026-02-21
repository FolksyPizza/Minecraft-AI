#!/usr/bin/env python3
"""Validate that required addon names are sufficiently represented in source rows."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ADDON_RE = re.compile(r"Addon:\s*([^\n\r]+?)\s+(?:Syntax|Description:|$)", re.IGNORECASE)


def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def matches_alias(name: str, aliases: list[str]) -> bool:
    n = normalize(name)
    return any(a in n for a in aliases)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--min-per-addon", type=int, default=20)
    args = ap.parse_args()

    required = {
        "skbee": ["skbee"],
        "skript-reflect": ["skript-reflect", "skript reflect", "skriptmirror"],
        "skript-gui": ["skript-gui", "skript gui"],
        "skript-yaml": ["skript-yaml", "skript yaml"],
        "poask": ["poask", "poa sk"],
        "hippo": ["hippo"],
    }

    counts = {k: 0 for k in required}
    total = 0

    with Path(args.in_path).resolve().open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            total += 1
            obj = json.loads(line)
            prompt = obj.get("prompt", "")
            if not isinstance(prompt, str):
                continue
            m = ADDON_RE.search(prompt)
            if not m:
                continue
            addon = m.group(1)
            for key, aliases in required.items():
                if matches_alias(addon, aliases):
                    counts[key] += 1

    missing = [k for k, v in counts.items() if v < args.min_per_addon]
    report = {
        "rows_scanned": total,
        "min_per_addon": args.min_per_addon,
        "counts": counts,
        "missing_or_low": missing,
    }
    print(json.dumps(report, indent=2))

    if missing:
        raise SystemExit(
            "[error] required addon coverage below threshold for: " + ", ".join(missing)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
