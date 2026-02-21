#!/usr/bin/env python3
"""Augment Skript syntax-style rows with concrete one-line command examples."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


PLACEHOLDER_DEFAULTS = [
    (re.compile(r"%player%"), "player"),
    (re.compile(r"%offlineplayer%"), "player"),
    (re.compile(r"%number%"), "1"),
    (re.compile(r"%integer%"), "1"),
    (re.compile(r"%string%"), '"example"'),
    (re.compile(r"%text%"), '"example"'),
    (re.compile(r"%boolean%"), "true"),
    (re.compile(r"%world%"), "world"),
    (re.compile(r"%item(stack)?%"), "stone"),
    (re.compile(r"%location%"), "player's location"),
    (re.compile(r"%entity%"), "player"),
    (re.compile(r"%block%"), "stone"),
]


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


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def looks_like_syntax_pattern(text: str) -> bool:
    return any(marker in text for marker in ("%", "[", "]", "(", ")", "|"))


def choose_first_nonempty_line(text: str) -> str:
    for line in re.split(r"[\r\n]+", text):
        line = line.strip()
        if line:
            return line
    return text.strip()


def remove_optional_brackets(text: str) -> str:
    out = text
    while True:
        nxt = re.sub(r"\[[^\[\]]*\]", "", out)
        if nxt == out:
            break
        out = nxt
    return out


def choose_first_alternative(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        inner = match.group(1)
        return inner.split("|", 1)[0].strip()

    out = text
    while True:
        nxt = re.sub(r"\(([^()]*)\)", repl, out)
        if nxt == out:
            break
        out = nxt
    return out


def substitute_placeholders(text: str) -> str:
    out = text
    for pat, replacement in PLACEHOLDER_DEFAULTS:
        out = pat.sub(replacement, out)
    out = re.sub(r"%[^%]+%", "value", out)
    return out


def concretize_pattern(pattern: str) -> str:
    out = choose_first_nonempty_line(pattern)
    out = remove_optional_brackets(out)
    out = choose_first_alternative(out)
    out = substitute_placeholders(out)
    out = re.sub(r"\s+", " ", out).strip()
    return out


def build_augmented_rows(source_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    extra: list[dict[str, str]] = []
    for row in source_rows:
        comp = row["completion"]
        if not looks_like_syntax_pattern(comp):
            continue
        concrete = concretize_pattern(comp)
        if not concrete or concrete == comp:
            continue
        extra.append(
            {
                "prompt": (
                    "Convert this Skript syntax pattern into one concrete executable command "
                    f"with no optional segments or alternatives:\n{comp}"
                ),
                "completion": concrete,
            }
        )
    return extra


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="Input JSONL with prompt/completion rows")
    ap.add_argument("--out", dest="out_path", required=True, help="Output enriched JSONL")
    args = ap.parse_args()

    source_rows = read_rows(Path(args.in_path).resolve())
    augmented = build_augmented_rows(source_rows)
    merged = dedupe(source_rows + augmented)
    write_rows(Path(args.out_path).resolve(), merged)

    print(
        json.dumps(
            {
                "source_rows": len(source_rows),
                "augmented_rows": len(augmented),
                "total_rows": len(merged),
                "out_path": str(Path(args.out_path).resolve()),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
