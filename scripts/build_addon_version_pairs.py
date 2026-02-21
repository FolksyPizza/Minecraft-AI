#!/usr/bin/env python3
"""Build explicit syntax->addon/version supervision pairs from SkriptHub-derived rows."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


TITLE_RE = re.compile(r"Title:\s*([^\n\r]+?)\s+(?:Addon:|Syntax Pattern:|Description:|$)", re.IGNORECASE)
ADDON_RE = re.compile(r"Addon:\s*([^\n\r]+?)\s+(?:Syntax|Description:|$)", re.IGNORECASE)
SYNTAX_RE = re.compile(r"(?:Syntax Pattern:|Syntax:)\s*([^\n\r]+?)(?:\s+Description:|\s+Return only|\s+Format exactly|$)", re.IGNORECASE)
COMPAT_RE = re.compile(r"compatible_addon_version\s*=\s*([^\n\r]*)", re.IGNORECASE)


def read_rows(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            p = obj.get("prompt")
            c = obj.get("completion")
            if isinstance(p, str) and isinstance(c, str) and p.strip() and c.strip():
                rows.append({"prompt": p.strip(), "completion": c.strip()})
    return rows


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def extract(prompt: str) -> tuple[str, str, str]:
    title = ""
    addon = ""
    syntax = ""

    m = TITLE_RE.search(prompt)
    if m:
        title = norm(m.group(1))
    m = ADDON_RE.search(prompt)
    if m:
        addon = norm(m.group(1))
    m = SYNTAX_RE.search(prompt)
    if m:
        syntax = norm(m.group(1))

    return title, addon, syntax


def dedupe(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[str, str]] = set()
    out: list[dict[str, str]] = []
    for row in rows:
        key = (norm(row["prompt"]), norm(row["completion"]))
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_path", required=True)
    args = ap.parse_args()

    rows = read_rows(Path(args.in_path).resolve())

    # Gather best-known addon/version per (title, syntax).
    facts: dict[tuple[str, str], dict[str, str]] = {}

    for row in rows:
        prompt = row["prompt"]
        completion = row["completion"]
        title, addon, syntax = extract(prompt)

        if not title and not syntax:
            continue

        key = (title, syntax)
        info = facts.setdefault(key, {"title": title, "syntax": syntax, "addon": "", "version": ""})
        if addon and not info["addon"]:
            info["addon"] = addon

        m = COMPAT_RE.search(completion)
        if m:
            v = norm(m.group(1))
            if v:
                info["version"] = v

    out_rows: list[dict[str, str]] = []
    for info in facts.values():
        title = info["title"]
        syntax = info["syntax"]
        addon = info["addon"] or "unknown"
        version = info["version"] or "unknown"

        context = f"Title: {title}\\nSyntax: {syntax}" if syntax else f"Title: {title}"

        out_rows.append(
            {
                "prompt": (
                    "Given this Skript syntax entry, return only the addon name that provides it.\\n"
                    f"{context}"
                ),
                "completion": addon,
            }
        )

        out_rows.append(
            {
                "prompt": (
                    "Given this Skript syntax entry, return only the addon version where this syntax is"
                    " documented/introduced. If unknown, return 'unknown'.\\n"
                    f"{context}\\nAddon: {addon}"
                ),
                "completion": version,
            }
        )

        out_rows.append(
            {
                "prompt": (
                    "Return compact JSON with keys addon and introduced_addon_version for this Skript syntax.\\n"
                    f"{context}"
                ),
                "completion": json.dumps(
                    {"addon": addon, "introduced_addon_version": version}, ensure_ascii=False, separators=(",", ":")
                ),
            }
        )

    out_rows = dedupe(out_rows)
    write_rows(Path(args.out_path).resolve(), out_rows)

    known_version = sum(1 for r in out_rows if r["completion"] not in {"unknown", '{"addon":"unknown","introduced_addon_version":"unknown"}'})
    print(
        json.dumps(
            {
                "source_rows": len(rows),
                "facts": len(facts),
                "output_rows": len(out_rows),
                "rows_with_known_version_or_addon": known_version,
                "out": str(Path(args.out_path).resolve()),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
