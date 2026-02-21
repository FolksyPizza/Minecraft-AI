#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def build_prompt(user_prompt: str) -> str:
    # Match the exact training text format used in train_lora_cuda.py.
    return f"<|user|>\n{user_prompt}\n<|assistant|>\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Run local inference with base model + LoRA adapter")
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-Coder-7B")
    ap.add_argument("--adapter", default="/Users/william/Desktop/sk/LoRA/output/stage2_minecraft_adapter")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--repetition-penalty", type=float, default=1.15)
    ap.add_argument("--no-repeat-ngram-size", type=int, default=4)
    ap.add_argument("--load-in-8bit", action="store_true", default=False)
    ap.add_argument("--load-in-4bit", action="store_true", default=False)
    args = ap.parse_args()

    if args.load_in_8bit and args.load_in_4bit:
        raise SystemExit("Only one of --load-in-8bit/--load-in-4bit can be set")
    if (args.load_in_8bit or args.load_in_4bit) and not torch.cuda.is_available():
        raise SystemExit("8-bit/4-bit loading is CUDA-only in this script. On Apple Silicon, run without quant flags.")

    adapter_path = Path(args.adapter).resolve()
    if not adapter_path.exists():
        raise SystemExit(f"Adapter path not found: {adapter_path}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    has_cuda = torch.cuda.is_available()
    has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    bf16 = has_cuda and torch.cuda.is_bf16_supported()
    device = "cuda" if has_cuda else ("mps" if has_mps else "cpu")
    model_kwargs = {
        "trust_remote_code": True,
    }
    if has_cuda:
        model_kwargs["device_map"] = "auto"

    if args.load_in_8bit or args.load_in_4bit:
        compute_dtype = torch.bfloat16 if bf16 else torch.float16
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=args.load_in_8bit,
            load_in_4bit=args.load_in_4bit,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16 if bf16 else (torch.float16 if has_mps else torch.float32)

    base = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)
    model = PeftModel.from_pretrained(base, str(adapter_path))
    if not has_cuda:
        model.to(device)
    model.eval()

    text = build_prompt(args.prompt)
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.temperature > 0,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            pad_token_id=tokenizer.eos_token_id,
        )

    gen = tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    print(gen.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
