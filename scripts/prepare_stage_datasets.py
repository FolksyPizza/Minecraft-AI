#!/usr/bin/env python3
"""Compose stage datasets for two-stage LoRA curriculum."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


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


def dynamic_non_mc_ratio(m_count: int) -> float:
    if m_count >= 50000:
        return 0.15
    if m_count >= 30000:
        return 0.20
    if m_count >= 15000:
        return 0.25
    return 0.30


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minecraft-source", required=True, help="Minecraft-focused labeled JSONL")
    ap.add_argument("--general-sources", nargs="*", default=[], help="General coding JSONL sources")
    ap.add_argument("--stage1-out", required=True)
    ap.add_argument("--stage2-out", required=True)
    ap.add_argument("--report-out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rnd = random.Random(args.seed)

    mc_rows = dedupe(read_pairs(Path(args.minecraft_source).resolve()))

    general_rows: list[dict[str, str]] = []
    for src in args.general_sources:
        p = Path(src).resolve()
        if p.exists():
            general_rows.extend(read_pairs(p))
    general_rows = dedupe(general_rows)

    rnd.shuffle(mc_rows)
    rnd.shuffle(general_rows)

    # Stage 1: general-first with small Minecraft anchors.
    anchor = min(max(500, len(mc_rows) // 20), max(1, len(mc_rows))) if mc_rows else 0
    stage1 = dedupe(general_rows + mc_rows[:anchor])
    rnd.shuffle(stage1)

    # Stage 2: Minecraft-primary with dynamic non-Minecraft cap.
    ratio = dynamic_non_mc_ratio(len(mc_rows))
    max_general_for_stage2 = int(len(mc_rows) * ratio / (1.0 - ratio)) if mc_rows else 0
    selected_general = general_rows[:max_general_for_stage2]
    stage2 = dedupe(mc_rows + selected_general)
    rnd.shuffle(stage2)

    write_jsonl(Path(args.stage1_out).resolve(), stage1)
    write_jsonl(Path(args.stage2_out).resolve(), stage2)

    report = {
        "minecraft_rows": len(mc_rows),
        "general_rows": len(general_rows),
        "stage1_rows": len(stage1),
        "stage2_rows": len(stage2),
        "stage1_minecraft_anchor_rows": anchor,
        "stage2_non_minecraft_ratio_cap": ratio,
        "stage2_selected_general_rows": len(selected_general),
        "stage2_actual_non_minecraft_share": (len(selected_general) / len(stage2)) if stage2 else 0.0,
    }
    Path(args.report_out).resolve().write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
