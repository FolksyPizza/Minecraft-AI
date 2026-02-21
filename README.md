# LoRA Adapter for DeepSeek/Qwen

This folder is portable. Copy `LoRA/` to your CUDA machine, then run one command:

```bash
bash scripts/run_train.sh
```

The script will:
1. Create `.venv`
2. Install `requirements.txt`
3. Optionally scan curated GitHub repos for concrete `.sk` command lines
4. Enrich Minecraft syntax rows with concrete one-line command variants
5. Validate/prepare stage datasets
6. Resolve a **public model anonymously** (no HF login/token)
7. Launch distributed two-stage LoRA training across all visible GPUs with DeepSpeed

## Requirements

- Linux CUDA machine (At least 1 GPU, and 64GB of RAM recommended)
- Python 3.10+
- NVIDIA driver + CUDA runtime working
- Internet access to fetch public models/datasets

Default list:
- `deepseek-ai/DeepSeek-Coder-V2-Lite-Base`
- `Qwen/Qwen2.5-Coder-7B`
- `codellama/CodeLlama-7b-hf`

## Main Files

- `train_lora_cuda.py`: two-stage trainer (general -> minecraft)
- `scripts/run_train.sh`: one-command setup + launch
- `scripts/prepare_stage_datasets.py`: dynamic 15-30% non-Minecraft cap composition
- `scripts/validate_dataset.py`: strict dataset validation
- `scripts/download_sources.py`: Hugging Face + curated GitHub source ingestion
- `scripts/merge_pair_sources.py`: merges/dedupes multiple prompt/completion sources
- `scripts/enrich_minecraft_concrete.py`: converts syntax-style rows into concrete command completions
- `scripts/build_addon_version_pairs.py`: builds explicit syntax -> addon/version supervision rows
- `deepspeed/zero3.json`: default ZeRO-3 sharding
- `deepspeed/zero2.json`: fallback config

## Dataset Inputs

Primary Minecraft source included:
- `labeled/sources/skripthub_pairs.jsonl`

Optional general sources (if present, auto-used):
- `labeled/sources/java_kotlin_standardized.jsonl`
- `labeled/sources/python_pytorch_pairs.jsonl`
- `labeled/sources/cpp_c_cuda_pairs.jsonl`
- `labeled/sources/sql_r_pairs.jsonl`

Generated training files:
- `labeled/final_general_stage1.jsonl`
- `labeled/final_minecraft_primary.jsonl`

## Running

```bash
bash scripts/run_train.sh
```

Optional env overrides:

```bash
MODEL_NAME=deepseek-ai/DeepSeek-Coder-V2-Lite-Base \
MAX_SEQ_LEN=2048 \
PER_DEVICE_TRAIN_BATCH_SIZE=1 \
GRAD_ACCUM_STEPS=16 \
EPOCHS_STAGE1=1.0 \
EPOCHS_STAGE2=1.0 \
bash scripts/run_train.sh
```

Force stage dataset regeneration (after source updates):

```bash
REBUILD_STAGE_DATA=1 bash scripts/run_train.sh
```

Fetch/update curated GitHub sources first (internet required):

```bash
FETCH_GITHUB_SOURCES=1 REBUILD_STAGE_DATA=1 bash scripts/run_train.sh
```

Build a more \"normal coder\" dataset mix (general-first):

```bash
FETCH_GENERAL_SOURCES=1 FETCH_GITHUB_SOURCES=1 REBUILD_STAGE_DATA=1 \
STAGE1_GENERAL_SHARE=0.85 STAGE2_GENERAL_SHARE=0.55 STAGE2_TEMPLATE_SHARE_CAP=0.08 \
bash scripts/run_train.sh
```

Recommended anti-overfit mix for this repo:

```bash
FETCH_GENERAL_SOURCES=1 FETCH_GITHUB_SOURCES=1 REBUILD_STAGE_DATA=1 \
STAGE1_GENERAL_SHARE=0.88 STAGE2_GENERAL_SHARE=0.60 \
STAGE2_TEMPLATE_SHARE_CAP=0.05 STAGE2_FACT_SHARE_CAP=0.05 \
MAX_SEQ_LEN=1024 GRAD_ACCUM_STEPS=16 \
LOAD_IN_8BIT=1 LOAD_IN_4BIT=0 \
bash scripts/run_train.sh
```

Curated addon sources include:
- `ShaneBeee/SkBee`
- `SkriptLang/skript-reflect`
- `APickledWalrus/skript-gui`
- `Sashie/skript-yaml`
- `Pesekjak/Hippo`
- `Ekpoa/PoaSkRewritev2`

Defaults in `scripts/run_train.sh` now prefer diversity:
- `FETCH_GENERAL_SOURCES=1`
- `FETCH_GITHUB_SOURCES=1`
- `MIN_GENERAL_ROWS=5000` (hard floor to avoid over-specialized adapters)
- `STAGE2_FACT_SHARE_CAP=0.05` (keeps addon/version grounding, but prevents fact-task takeover)

## Outputs

- `output/stage1_general_adapter/`
- `output/stage2_minecraft_adapter/`
- `output/stage1_general_metrics.json`
- `output/stage2_minecraft_metrics.json`
- `output/summary.json`
- `reports/mix_policy_report.json`

## Notes

- Uses BF16 if available, else FP16.
- Uses all GPUs detected by `torch.cuda.device_count()`.
- Keep dataset rows strictly as `{"prompt":"...", "completion":"..."}`.
