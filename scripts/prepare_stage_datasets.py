#!/usr/bin/env python3
"""Compose stage datasets for a normal-coder-first two-stage curriculum."""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path


META_PROMPT_PATTERNS = [
    r"classify",
    r"identify the addon",
    r"infer the documentation title",
    r"syntax type",
    r"return only the exact syntax pattern",
    r"documentation snippet",
    r"title:",
    r"addon:",
    r"syntax pattern:",
    r"given this skript syntax entry",
    r"return compact json with keys addon",
    r"return only the addon name",
    r"return only the addon version",
    r"compatible_addon_version",
]


def read_pairs(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            prompt = obj.get("prompt")
            completion = obj.get("completion")
            if isinstance(prompt, str) and prompt.strip() and isinstance(completion, str) and completion.strip():
                rows.append({"prompt": prompt.strip(), "completion": completion.strip()})
    return rows


def dedupe(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[str, str]] = set()
    out: list[dict[str, str]] = []
    for r in rows:
        key = (" ".join(r["prompt"].split()), " ".join(r["completion"].split()))
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def is_meta_task(prompt: str, completion: str) -> bool:
    p = prompt.lower()
    c = completion.strip().lower()
    if any(re.search(pat, p) for pat in META_PROMPT_PATTERNS):
        return True
    if c.startswith("compatible_addon_version="):
        return True
    if len(c.split()) <= 2 and c in {"effect", "condition", "expression", "event", "section", "function"}:
        return True
    if p.startswith("identify") or p.startswith("classify"):
        return True
    return False


def is_template_like(text: str) -> bool:
    t = text.strip()
    return (
        ("%" in t)
        or ("[" in t and "]" in t)
        or ("(" in t and ")" in t and "|" in t)
    )


def sample_rows(rnd: random.Random, rows: list[dict[str, str]], k: int) -> list[dict[str, str]]:
    if k <= 0:
        return []
    if len(rows) <= k:
        return rows[:]
    return rnd.sample(rows, k)


def fill_to_target(
    rnd: random.Random,
    base: list[dict[str, str]],
    fallback: list[dict[str, str]],
    target: int,
) -> list[dict[str, str]]:
    out = base[:]
    if len(out) >= target:
        return out[:target]

    seen = {(" ".join(r["prompt"].split()), " ".join(r["completion"].split())) for r in out}
    for row in fallback:
        if len(out) >= target:
            break
        key = (" ".join(row["prompt"].split()), " ".join(row["completion"].split()))
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out[:target]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minecraft-source", required=True, help="Minecraft-focused labeled JSONL")
    ap.add_argument(
        "--minecraft-fact-sources",
        nargs="*",
        default=[],
        help="Optional fact-style Minecraft JSONL sources (addon/version attribution)",
    )
    ap.add_argument("--general-sources", nargs="*", default=[], help="General coding JSONL sources")
    ap.add_argument("--stage1-out", required=True)
    ap.add_argument("--stage2-out", required=True)
    ap.add_argument("--report-out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stage1-general-share", type=float, default=0.85)
    ap.add_argument("--stage2-general-share", type=float, default=0.55)
    ap.add_argument("--stage2-template-share-cap", type=float, default=0.08)
    ap.add_argument("--stage2-fact-share-cap", type=float, default=0.05)
    ap.add_argument("--max-stage1-rows", type=int, default=120000)
    ap.add_argument("--max-stage2-rows", type=int, default=140000)
    ap.add_argument("--min-general-rows", type=int, default=5000)
    args = ap.parse_args()

    rnd = random.Random(args.seed)

    mc_raw = dedupe(read_pairs(Path(args.minecraft_source).resolve()))
    mc_meta = [r for r in mc_raw if is_meta_task(r["prompt"], r["completion"])]
    mc_rows = [r for r in mc_raw if not is_meta_task(r["prompt"], r["completion"])]

    mc_fact_rows: list[dict[str, str]] = []
    for src in args.minecraft_fact_sources:
        p = Path(src).resolve()
        if p.exists():
            mc_fact_rows.extend(read_pairs(p))
    mc_fact_rows = dedupe(mc_fact_rows)

    general_rows: list[dict[str, str]] = []
    for src in args.general_sources:
        p = Path(src).resolve()
        if p.exists():
            general_rows.extend(read_pairs(p))
    general_rows = dedupe(general_rows)
    if len(general_rows) < args.min_general_rows:
        raise SystemExit(
            f"[error] insufficient general coding rows ({len(general_rows)}). "
            f"Need at least {args.min_general_rows}. "
            "Enable FETCH_GENERAL_SOURCES=1 or add more general datasets."
        )

    mc_template = [r for r in mc_rows if is_template_like(r["completion"])]
    mc_concrete = [r for r in mc_rows if not is_template_like(r["completion"])]
    mc_fact_pool = dedupe(mc_meta + mc_fact_rows)

    rnd.shuffle(general_rows)
    rnd.shuffle(mc_template)
    rnd.shuffle(mc_concrete)
    rnd.shuffle(mc_fact_pool)

    # Stage 1: keep model broadly capable.
    stage1_target = min(args.max_stage1_rows, max(len(general_rows), 40000))
    stage1_general_target = int(stage1_target * args.stage1_general_share)
    stage1_mc_target = stage1_target - stage1_general_target

    stage1_general = sample_rows(rnd, general_rows, stage1_general_target)
    stage1_mc = sample_rows(rnd, mc_concrete, stage1_mc_target)
    stage1 = dedupe(stage1_general + stage1_mc)
    stage1 = fill_to_target(rnd, stage1, general_rows + mc_concrete, min(stage1_target, len(stage1) + 20000))
    rnd.shuffle(stage1)

    # Stage 2: domain adaptation without losing general coding behavior.
    stage2_target = min(args.max_stage2_rows, max(len(mc_rows), 60000))
    stage2_general_target = int(stage2_target * args.stage2_general_share)
    stage2_mc_target = stage2_target - stage2_general_target

    stage2_template_cap = int(stage2_target * args.stage2_template_share_cap)
    stage2_fact_cap = int(stage2_target * args.stage2_fact_share_cap)
    stage2_template_target = min(stage2_template_cap, max(0, stage2_mc_target // 4), len(mc_template))
    stage2_fact_target = min(stage2_fact_cap, max(0, stage2_mc_target // 5), len(mc_fact_pool))
    stage2_concrete_target = max(0, stage2_mc_target - stage2_template_target - stage2_fact_target)

    stage2_general = sample_rows(rnd, general_rows, stage2_general_target)
    stage2_concrete = sample_rows(rnd, mc_concrete, stage2_concrete_target)
    stage2_template_rows = sample_rows(rnd, mc_template, stage2_template_target)
    stage2_fact_rows = sample_rows(rnd, mc_fact_pool, stage2_fact_target)

    stage2 = dedupe(stage2_general + stage2_concrete + stage2_template_rows + stage2_fact_rows)
    stage2 = fill_to_target(rnd, stage2, general_rows + mc_concrete, stage2_target)
    rnd.shuffle(stage2)

    write_jsonl(Path(args.stage1_out).resolve(), stage1)
    write_jsonl(Path(args.stage2_out).resolve(), stage2)

    report = {
        "minecraft_rows_raw": len(mc_raw),
        "minecraft_rows_meta_filtered": len(mc_meta),
        "minecraft_rows_after_meta_filter": len(mc_rows),
        "minecraft_concrete_rows": len(mc_concrete),
        "minecraft_template_rows": len(mc_template),
        "minecraft_fact_pool_rows": len(mc_fact_pool),
        "general_rows": len(general_rows),
        "stage1_rows": len(stage1),
        "stage2_rows": len(stage2),
        "stage1_general_share_target": args.stage1_general_share,
        "stage2_general_share_target": args.stage2_general_share,
        "stage2_template_share_cap": args.stage2_template_share_cap,
        "stage2_fact_share_cap": args.stage2_fact_share_cap,
        "stage2_template_rows_selected": len(stage2_template_rows),
        "stage2_fact_rows_selected": len(stage2_fact_rows),
        "stage2_general_rows_selected": len(stage2_general),
        "stage2_concrete_rows_selected": len(stage2_concrete),
    }
    Path(args.report_out).resolve().write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
