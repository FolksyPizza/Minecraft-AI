#!/usr/bin/env python3
"""Download auxiliary datasets and mine concrete Skript lines from curated GitHub repos."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

import yaml


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None)


def _normalize_line(line: str) -> str:
    out = line.strip().replace("\t", " ")
    out = re.sub(r"\s+", " ", out)
    return out


def _looks_concrete_skript(line: str) -> bool:
    if not line or line.startswith("#"):
        return False
    if line.endswith(":"):
        return False
    if line.startswith(("command ", "on ", "every ", "function ", "options:")):
        return False
    if "{" in line or "}" in line:
        return False
    if line.count("%") >= 2:
        return False
    if len(line.split()) < 2:
        return False
    return any(ch.isalpha() for ch in line)


def _line_to_pair(repo: str, rel_path: str, line: str) -> dict[str, str]:
    topic = rel_path.split("/")[-1].replace(".sk", "")
    prompt = (
        "Write one concrete, executable Skript line for Minecraft server automation. "
        f"Context: {repo} / {topic}."
    )
    return {"prompt": prompt, "completion": line}


def _dedupe(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[str, str]] = set()
    out: list[dict[str, str]] = []
    for row in rows:
        key = (" ".join(row["prompt"].split()), " ".join(row["completion"].split()))
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _download_hf(raw_dir: Path) -> dict[str, str]:
    from datasets import load_dataset

    kotlin_ds = load_dataset("JetBrains/Kotlin_HumanEval")
    java_ds = load_dataset("Nan-Do/instructional_code-search-net-java")

    kotlin_out = raw_dir / "kotlin_dataset.jsonl"
    java_out = raw_dir / "java_dataset.jsonl"

    kotlin_ds["train"].to_json(str(kotlin_out), orient="records", lines=True)
    java_ds["train"].to_json(str(java_out), orient="records", lines=True)
    return {"kotlin": str(kotlin_out), "java": str(java_out)}


def _load_github_sources(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    sources = data.get("sources", [])
    if not isinstance(sources, list):
        raise ValueError(f"Invalid github sources config: {path}")
    return [s for s in sources if isinstance(s, dict) and s.get("enabled", True)]


def _clone_or_update(repo: str, ref: str | None, dest: Path) -> None:
    url = f"https://github.com/{repo}.git"
    if not dest.exists():
        if ref:
            _run(["git", "clone", "--depth", "1", "--branch", ref, url, str(dest)])
        else:
            _run(["git", "clone", "--depth", "1", url, str(dest)])
        return
    if ref:
        _run(["git", "fetch", "origin", ref, "--depth", "1"], cwd=dest)
        _run(["git", "checkout", "-q", "FETCH_HEAD"], cwd=dest)
    else:
        _run(["git", "fetch", "--depth", "1", "origin"], cwd=dest)
        _run(["git", "checkout", "-q", "origin/HEAD"], cwd=dest)


def _extract_skript_pairs(repo_cfg: dict, repo_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    include_globs = repo_cfg.get("include_globs") or ["**/*.sk"]
    max_lines = int(repo_cfg.get("max_lines", 10000))
    repo_name = str(repo_cfg["repo"])

    files: list[Path] = []
    for pattern in include_globs:
        files.extend(repo_dir.glob(pattern))

    seen_files = set()
    for path in sorted(files):
        if path.is_dir():
            continue
        if path in seen_files:
            continue
        seen_files.add(path)

        rel = path.relative_to(repo_dir).as_posix()
        try:
            with path.open("r", encoding="utf-8", errors="ignore") as f:
                if path.suffix.lower() == ".sk":
                    for raw in f:
                        line = _normalize_line(raw)
                        if not _looks_concrete_skript(line):
                            continue
                        rows.append(_line_to_pair(repo_name, rel_path=rel, line=line))
                        if len(rows) >= max_lines:
                            return rows
                elif path.suffix.lower() in {".md", ".markdown"}:
                    in_code = False
                    for raw in f:
                        line = raw.rstrip("\n")
                        stripped = line.strip()
                        if stripped.startswith("```"):
                            fence = stripped.strip("`").strip().lower()
                            if not in_code and fence in {"", "skript", "vb", "text"}:
                                in_code = True
                            else:
                                in_code = False
                            continue
                        if not in_code:
                            continue
                        candidate = _normalize_line(line)
                        if not _looks_concrete_skript(candidate):
                            continue
                        rows.append(_line_to_pair(repo_name, rel_path=rel, line=candidate))
                        if len(rows) >= max_lines:
                            return rows
        except OSError:
            continue
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="raw")
    ap.add_argument("--github-cache-dir", default="raw/github")
    ap.add_argument("--github-config", default="configs/github_sources.yaml")
    ap.add_argument("--github-out", default="labeled/sources/github_skript_concrete.jsonl")
    ap.add_argument("--manifest-out", default="raw/source_manifests/github_scan_manifest.json")
    ap.add_argument("--skip-hf", action="store_true", default=False)
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir).resolve()
    raw_dir.mkdir(parents=True, exist_ok=True)

    github_cache = Path(args.github_cache_dir).resolve()
    github_cache.mkdir(parents=True, exist_ok=True)

    hf_paths = {} if args.skip_hf else _download_hf(raw_dir)

    cfg_path = Path(args.github_config).resolve()
    sources = _load_github_sources(cfg_path)

    all_pairs: list[dict[str, str]] = []
    manifest = {
        "config": str(cfg_path),
        "sources": [],
        "hf_downloads": hf_paths,
    }

    for src in sources:
        repo = str(src["repo"])
        ref = src.get("ref")
        ref = str(ref).strip() if ref is not None else ""
        repo_dir = github_cache / repo.replace("/", "__")

        status = {"repo": repo, "ref": ref, "license": src.get("license"), "pairs": 0, "error": None}
        try:
            _clone_or_update(repo=repo, ref=(ref or None), dest=repo_dir)
            pairs = _extract_skript_pairs(src, repo_dir)
            all_pairs.extend(pairs)
            status["pairs"] = len(pairs)
            status["local_path"] = str(repo_dir)
            status["url"] = f"https://github.com/{repo}"
        except Exception as exc:  # noqa: BLE001
            status["error"] = str(exc)
        manifest["sources"].append(status)

    all_pairs = _dedupe(all_pairs)
    github_out = Path(args.github_out).resolve()
    _write_jsonl(github_out, all_pairs)

    manifest["total_pairs"] = len(all_pairs)
    manifest["github_out"] = str(github_out)
    manifest_out = Path(args.manifest_out).resolve()
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
