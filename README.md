# LoRA Training (CUDA, Multi-GPU, Anonymous Model Pull)

This folder is portable. Copy `LoRA/` to your CUDA machine, then run one command:

```bash
bash scripts/run_train.sh
```

The script will:
1. Create `.venv`
2. Install `requirements.txt`
3. Validate/prepare stage datasets
4. Resolve a **public model anonymously** (no HF login/token)
5. Launch distributed two-stage LoRA training across all visible GPUs with DeepSpeed

## Requirements

- Linux CUDA machine (dual RTX 3090 supported)
- Python 3.10+
- NVIDIA driver + CUDA runtime working
- Internet access to fetch public models/datasets

## Anonymous Model Policy

No account sign-in is required. The resolver tries `configs/public_models.yaml` in order and skips any gated/private model.

Default list:
- `deepseek-ai/DeepSeek-Coder-V2-Lite-Base`
- `Qwen/Qwen2.5-Coder-7B`
- `codellama/CodeLlama-7b-hf`

## Main Files

- `train_lora_cuda.py`: two-stage trainer (general -> minecraft)
- `scripts/run_train.sh`: one-command setup + launch
- `scripts/prepare_stage_datasets.py`: dynamic 15-30% non-Minecraft cap composition
- `scripts/validate_dataset.py`: strict dataset validation
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
