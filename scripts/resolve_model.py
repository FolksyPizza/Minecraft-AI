#!/usr/bin/env python3
"""Resolve an anonymously-downloadable base model from a public candidate list."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml
from huggingface_hub import snapshot_download
from transformers import AutoConfig


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "public_models.yaml"


def _load_candidates(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    candidates = data.get("models", [])
    if not isinstance(candidates, list) or not candidates:
        raise ValueError(f"No models configured in {path}")
    return candidates


def _try_hf(repo_id: str, revision: str | None = None) -> tuple[bool, str]:
    try:
        local_dir = snapshot_download(
            repo_id=repo_id,
            revision=revision,
            token=None,
            local_files_only=False,
            resume_download=True,
        )
        AutoConfig.from_pretrained(local_dir, trust_remote_code=True, token=None)
        return True, local_dir
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def resolve_model(config_path: Path) -> dict:
    attempts: list[dict] = []
    for candidate in _load_candidates(config_path):
        repo_id = candidate.get("repo_id")
        if not repo_id:
            continue
        ok, info = _try_hf(repo_id=repo_id, revision=candidate.get("revision"))
        attempt = {"repo_id": repo_id, "ok": ok, "info": info}
        attempts.append(attempt)
        if ok:
            return {
                "resolved": True,
                "repo_id": repo_id,
                "local_path": info,
                "trust_remote_code": bool(candidate.get("trust_remote_code", True)),
                "attempts": attempts,
            }

    return {"resolved": False, "attempts": attempts}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to public_models.yaml")
    ap.add_argument("--json-out", default="", help="Optional output JSON path")
    args = ap.parse_args()

    result = resolve_model(Path(args.config).resolve())
    if args.json_out:
        out_path = Path(args.json_out).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(result, indent=2))
    if result.get("resolved"):
        return 0

    print("No anonymously downloadable model could be resolved.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
