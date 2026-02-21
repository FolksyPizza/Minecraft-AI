#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> int:
    ap = argparse.ArgumentParser(description="Merge LoRA adapter into base model and save full HF model")
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-Coder-7B")
    ap.add_argument("--adapter", default="/Users/william/Desktop/sk/LoRA/output/stage2_minecraft_adapter")
    ap.add_argument("--out", default="/Users/william/Desktop/sk/LoRA/inference/merged_qwen2.5_coder_7b_minecraft")
    args = ap.parse_args()

    adapter_path = Path(args.adapter).resolve()
    out_path = Path(args.out).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else torch.float16

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, str(adapter_path))
    merged = model.merge_and_unload()

    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    merged.save_pretrained(str(out_path), safe_serialization=True)
    tok.save_pretrained(str(out_path))

    print(f"Merged model written to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
