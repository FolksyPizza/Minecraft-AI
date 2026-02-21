#!/usr/bin/env python3
"""Two-stage LoRA training on CUDA with DeepSpeed and anonymous model download."""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from huggingface_hub import snapshot_download
from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)


DEFAULT_PUBLIC_MODELS = Path(__file__).resolve().parent / "configs" / "public_models.yaml"


def _load_candidates(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    models = data.get("models", [])
    if not isinstance(models, list) or not models:
        raise ValueError(f"No models in {path}")
    return models


def resolve_anonymous_model(preferred_model: str | None, public_models_yaml: Path) -> tuple[str, bool]:
    candidates = _load_candidates(public_models_yaml)
    ordered: list[dict] = []

    if preferred_model:
        ordered.append({"repo_id": preferred_model, "trust_remote_code": True})
    ordered.extend(candidates)

    seen = set()
    final = []
    for c in ordered:
        repo = c.get("repo_id")
        if not repo or repo in seen:
            continue
        seen.add(repo)
        final.append(c)

    errors = []
    for c in final:
        repo = c["repo_id"]
        trust = bool(c.get("trust_remote_code", True))
        try:
            local = snapshot_download(repo_id=repo, token=None, resume_download=True)
            AutoTokenizer.from_pretrained(local, trust_remote_code=trust, token=None)
            print(f"[model] resolved anonymously: {repo}")
            return repo, trust
        except Exception as exc:  # noqa: BLE001
            errors.append({"repo_id": repo, "error": str(exc)})
            print(f"[model] skip {repo}: {exc}")

    raise RuntimeError(f"No anonymous model could be resolved. Attempts: {json.dumps(errors, indent=2)[:4000]}")


def load_pairs_dataset(path: Path, tokenizer, max_seq_len: int, seed: int):
    ds = load_dataset("json", data_files=str(path), split="train")

    def to_text(ex):
        prompt = ex["prompt"].strip()
        completion = ex["completion"].strip()
        return {"text": f"<|user|>\n{prompt}\n<|assistant|>\n{completion}"}

    ds = ds.map(to_text, remove_columns=ds.column_names)

    def tokenize(batch):
        tokens = tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_seq_len,
            padding=False,
        )
        tokens["labels"] = [ids.copy() for ids in tokens["input_ids"]]
        return tokens

    ds = ds.map(tokenize, batched=True, remove_columns=["text"])
    ds = ds.shuffle(seed=seed)

    # create eval split if large enough
    if len(ds) >= 1000:
        split = ds.train_test_split(test_size=0.02, seed=seed)
        return split["train"], split["test"]
    return ds, None


def choose_precision() -> tuple[bool, bool]:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return True, False
    return False, True


def build_training_args(
    output_dir: Path,
    deepspeed_config: Path | None,
    lr: float,
    epochs: float,
    batch_size: int,
    grad_accum: int,
    save_steps: int,
    eval_steps: int,
    logging_steps: int,
    bf16: bool,
    fp16: bool,
) -> TrainingArguments:
    kwargs = dict(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=max(1, batch_size),
        gradient_accumulation_steps=grad_accum,
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=3,
        bf16=bf16,
        fp16=fp16,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        report_to=[],
        dataloader_num_workers=2,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
    )
    if deepspeed_config is not None:
        kwargs["deepspeed"] = str(deepspeed_config)
    return TrainingArguments(**kwargs)


def train_stage(
    stage_name: str,
    model_name: str,
    trust_remote_code: bool,
    dataset_path: Path,
    output_dir: Path,
    deepspeed_config: Path | None,
    max_seq_len: int,
    batch_size: int,
    grad_accum: int,
    epochs: float,
    lr: float,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: list[str],
    save_steps: int,
    eval_steps: int,
    logging_steps: int,
    seed: int,
    packing: bool,
    previous_adapter: Path | None,
    load_in_8bit: bool,
    load_in_4bit: bool,
) -> Path:
    if load_in_8bit and load_in_4bit:
        raise ValueError("Only one of load_in_8bit/load_in_4bit can be enabled.")

    bf16, fp16 = choose_precision()
    print(f"[{stage_name}] precision bf16={bf16} fp16={fp16}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code, token=None)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model_kwargs = {
        "trust_remote_code": trust_remote_code,
        "token": None,
    }
    if load_in_8bit or load_in_4bit:
        compute_dtype = torch.bfloat16 if bf16 else torch.float16
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        model_kwargs["device_map"] = {"": local_rank}
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16 if bf16 else torch.float16

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.config.use_cache = False

    if load_in_8bit or load_in_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    if previous_adapter is None:
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
    else:
        model = PeftModel.from_pretrained(model, str(previous_adapter), is_trainable=True)

    train_ds, eval_ds = load_pairs_dataset(dataset_path, tokenizer=tokenizer, max_seq_len=max_seq_len, seed=seed)

    args = build_training_args(
        output_dir=output_dir,
        deepspeed_config=deepspeed_config,
        lr=lr,
        epochs=epochs,
        batch_size=batch_size,
        grad_accum=grad_accum,
        save_steps=save_steps,
        eval_steps=eval_steps,
        logging_steps=logging_steps,
        bf16=bf16,
        fp16=fp16,
    )

    trainer = Trainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        args=args,
    )

    trainer.train()
    metrics = trainer.evaluate() if eval_ds is not None else {}

    adapter_out = output_dir / f"{stage_name}_adapter"
    adapter_out.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(adapter_out))
    tokenizer.save_pretrained(str(adapter_out))

    metrics_path = output_dir / f"{stage_name}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[{stage_name}] adapter saved: {adapter_out}")
    print(f"[{stage_name}] metrics: {metrics_path}")

    return adapter_out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="deepseek-ai/DeepSeek-Coder-V2-Lite-Base")
    ap.add_argument("--public_models_yaml", default=str(DEFAULT_PUBLIC_MODELS))
    ap.add_argument("--stage1_dataset", required=True)
    ap.add_argument("--stage2_dataset", required=True)
    ap.add_argument("--output_dir", default="LoRA/output")
    ap.add_argument("--deepspeed_config", default="LoRA/deepspeed/zero3.json")
    ap.add_argument("--disable_deepspeed", action="store_true", default=False)
    ap.add_argument("--load_in_8bit", action="store_true", default=False)
    ap.add_argument("--load_in_4bit", action="store_true", default=False)
    ap.add_argument("--max_seq_len", type=int, default=2048)
    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=16)
    ap.add_argument("--num_train_epochs_stage1", type=float, default=1.0)
    ap.add_argument("--num_train_epochs_stage2", type=float, default=1.0)
    ap.add_argument("--learning_rate_stage1", type=float, default=2e-4)
    ap.add_argument("--learning_rate_stage2", type=float, default=1.5e-4)
    ap.add_argument("--lora_r", type=int, default=64)
    ap.add_argument("--lora_alpha", type=int, default=128)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument(
        "--target_modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated target modules",
    )
    ap.add_argument("--save_steps", type=int, default=500)
    ap.add_argument("--eval_steps", type=int, default=500)
    ap.add_argument("--logging_steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--packing", action="store_true", default=True)
    ap.add_argument("--minecraft_only", action="store_true", default=False)
    args = ap.parse_args()

    set_seed(args.seed)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name, trust_remote_code = resolve_anonymous_model(
        preferred_model=args.model_name,
        public_models_yaml=Path(args.public_models_yaml).resolve(),
    )

    ds_config = None if args.disable_deepspeed else Path(args.deepspeed_config).resolve()
    if ds_config is None:
        print("[train] deepspeed disabled; using torch DDP")
    else:
        print(f"[train] deepspeed config: {ds_config}")
    print(f"[train] quantization load_in_8bit={args.load_in_8bit} load_in_4bit={args.load_in_4bit}")

    run_info = {
        "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "resolved_model": model_name,
        "trust_remote_code": trust_remote_code,
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
        "deepspeed_config": str(ds_config) if ds_config is not None else None,
        "minecraft_only": args.minecraft_only,
    }
    (output_dir / "run_info.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")

    target_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()]

    stage1_adapter: Path | None = None
    if args.minecraft_only:
        print("[train] minecraft_only=1 -> skipping stage1_general")
    else:
        stage1_adapter = train_stage(
            stage_name="stage1_general",
            model_name=model_name,
            trust_remote_code=trust_remote_code,
            dataset_path=Path(args.stage1_dataset).resolve(),
            output_dir=output_dir,
            deepspeed_config=ds_config,
            max_seq_len=args.max_seq_len,
            batch_size=args.per_device_train_batch_size,
            grad_accum=args.gradient_accumulation_steps,
            epochs=args.num_train_epochs_stage1,
            lr=args.learning_rate_stage1,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            save_steps=args.save_steps,
            eval_steps=args.eval_steps,
            logging_steps=args.logging_steps,
            seed=args.seed,
            packing=args.packing,
            previous_adapter=None,
            load_in_8bit=args.load_in_8bit,
            load_in_4bit=args.load_in_4bit,
        )

    stage2_adapter = train_stage(
        stage_name="stage2_minecraft",
        model_name=model_name,
        trust_remote_code=trust_remote_code,
        dataset_path=Path(args.stage2_dataset).resolve(),
        output_dir=output_dir,
        deepspeed_config=ds_config,
        max_seq_len=args.max_seq_len,
        batch_size=args.per_device_train_batch_size,
        grad_accum=args.gradient_accumulation_steps,
        epochs=args.num_train_epochs_stage2,
        lr=args.learning_rate_stage2,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        seed=args.seed,
        packing=args.packing,
        previous_adapter=stage1_adapter,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
    )

    summary = {
        "stage1_adapter": str(stage1_adapter) if stage1_adapter is not None else None,
        "stage2_adapter_final": str(stage2_adapter),
        "resolved_model": model_name,
        "minecraft_only": args.minecraft_only,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
