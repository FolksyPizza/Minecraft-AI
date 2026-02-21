#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "[setup] creating venv at ${VENV_DIR}"
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"
export SETUPTOOLS_USE_DISTUTILS=local
python -m pip install --upgrade pip setuptools==69.5.1 wheel
python -m pip install -r "${ROOT_DIR}/requirements.txt"

python - <<'PY'
import sys
if sys.version_info < (3, 10) or sys.version_info >= (3, 13):
    raise SystemExit(
        f"[error] unsupported Python {sys.version.split()[0]}; use Python 3.10, 3.11, or 3.12"
    )
print(f"[env] python={sys.version.split()[0]}")
PY

python - <<'PY'
import importlib
pkgs = ["torch", "transformers", "peft", "accelerate", "datasets", "rich", "bitsandbytes"]
parts = []
for name in pkgs:
    try:
        mod = importlib.import_module(name)
        parts.append(f"{name}={getattr(mod, '__version__', 'unknown')}")
    except Exception as exc:  # noqa: BLE001
        parts.append(f"{name}=import-error:{exc.__class__.__name__}")
print("[env] " + ", ".join(parts))
PY

# Compose stage datasets if missing.
STAGE1="${ROOT_DIR}/labeled/final_general_stage1.jsonl"
STAGE2="${ROOT_DIR}/labeled/final_minecraft_primary.jsonl"
MC_SOURCE_RAW="${ROOT_DIR}/labeled/sources/skripthub_pairs.jsonl"
MC_SOURCE_GITHUB="${ROOT_DIR}/labeled/sources/github_skript_concrete.jsonl"
MC_SOURCE_ADDON_VERSION="${ROOT_DIR}/labeled/sources/skript_addon_version_pairs.jsonl"
MC_SOURCE_COMBINED="${ROOT_DIR}/labeled/sources/skripthub_plus_github_pairs.jsonl"
MC_SOURCE_ENRICHED="${ROOT_DIR}/labeled/sources/skripthub_pairs_enriched.jsonl"
MC_SOURCE="${MC_SOURCE_ENRICHED}"

if [[ ! -f "${MC_SOURCE_RAW}" ]]; then
  echo "[error] missing Minecraft source dataset: ${MC_SOURCE_RAW}" >&2
  exit 1
fi

if [[ "${FETCH_GITHUB_SOURCES:-0}" == "1" || "${FETCH_GENERAL_SOURCES:-0}" == "1" ]]; then
  DL_ARGS=(
    --raw-dir "${ROOT_DIR}/raw"
    --github-cache-dir "${ROOT_DIR}/raw/github"
    --github-config "${ROOT_DIR}/configs/github_sources.yaml"
    --github-out "${MC_SOURCE_GITHUB}"
    --manifest-out "${ROOT_DIR}/raw/source_manifests/github_scan_manifest.json"
  )
  if [[ "${FETCH_GENERAL_SOURCES:-0}" != "1" ]]; then
    DL_ARGS+=(--skip-hf)
  fi
  python "${ROOT_DIR}/scripts/download_sources.py" \
    "${DL_ARGS[@]}"
fi

if [[ "${FETCH_GENERAL_SOURCES:-0}" == "1" ]]; then
  python "${ROOT_DIR}/scripts/standardize_sources.py" \
    --java-file "${ROOT_DIR}/raw/java_dataset.jsonl" \
    --kotlin-file "${ROOT_DIR}/raw/kotlin_dataset.jsonl" \
    --out "${ROOT_DIR}/labeled/sources/java_kotlin_standardized.jsonl"
fi

python "${ROOT_DIR}/scripts/build_addon_version_pairs.py" \
  --in "${MC_SOURCE_RAW}" \
  --out "${MC_SOURCE_ADDON_VERSION}"

python "${ROOT_DIR}/scripts/merge_pair_sources.py" \
  --inputs "${MC_SOURCE_RAW}" "${MC_SOURCE_GITHUB}" "${MC_SOURCE_ADDON_VERSION}" \
  --out "${MC_SOURCE_COMBINED}"

python "${ROOT_DIR}/scripts/enrich_minecraft_concrete.py" \
  --in "${MC_SOURCE_COMBINED}" \
  --out "${MC_SOURCE_ENRICHED}"

if [[ "${REBUILD_STAGE_DATA:-0}" == "1" || ! -f "${STAGE1}" || ! -f "${STAGE2}" ]]; then
  echo "[data] building stage datasets"

  GENERAL_SOURCES=()
  for p in \
    "${ROOT_DIR}/labeled/sources/java_kotlin_standardized.jsonl" \
    "${ROOT_DIR}/labeled/sources/python_pytorch_pairs.jsonl" \
    "${ROOT_DIR}/labeled/sources/cpp_c_cuda_pairs.jsonl" \
    "${ROOT_DIR}/labeled/sources/sql_r_pairs.jsonl"; do
    [[ -f "$p" ]] && GENERAL_SOURCES+=("$p")
  done

  python "${ROOT_DIR}/scripts/prepare_stage_datasets.py" \
    --minecraft-source "${MC_SOURCE}" \
    --general-sources "${GENERAL_SOURCES[@]}" \
    --stage1-out "${STAGE1}" \
    --stage2-out "${STAGE2}" \
    --report-out "${ROOT_DIR}/reports/mix_policy_report.json" \
    --stage1-general-share "${STAGE1_GENERAL_SHARE:-0.85}" \
    --stage2-general-share "${STAGE2_GENERAL_SHARE:-0.55}" \
    --stage2-template-share-cap "${STAGE2_TEMPLATE_SHARE_CAP:-0.08}" \
    --max-stage1-rows "${MAX_STAGE1_ROWS:-120000}" \
    --max-stage2-rows "${MAX_STAGE2_ROWS:-140000}"
fi

python "${ROOT_DIR}/scripts/validate_dataset.py" "${STAGE1}" "${STAGE2}"

GPU_COUNT="$(python - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)"

if [[ "${GPU_COUNT}" -lt 1 ]]; then
  echo "[error] no CUDA GPUs detected" >&2
  exit 1
fi

echo "[train] using ${GPU_COUNT} GPU(s)"

MASTER_PORT="${MASTER_PORT:-29500}"
TRAIN_ARGS=()
export ACCELERATE_USE_DEEPSPEED=false

if [[ "${USE_DEEPSPEED:-0}" == "1" ]]; then
  if [[ -z "${CUDA_HOME:-}" ]]; then
    if command -v nvcc >/dev/null 2>&1; then
      CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
      export CUDA_HOME
    fi
  fi

  if [[ -z "${CUDA_HOME:-}" || ! -d "${CUDA_HOME}" ]]; then
    echo "[warn] CUDA_HOME not set/found; disabling DeepSpeed and using torch DDP"
    TRAIN_ARGS+=(--disable_deepspeed)
  else
    echo "[train] CUDA_HOME=${CUDA_HOME}"
    export ACCELERATE_USE_DEEPSPEED=true
    python -m pip install deepspeed==0.15.4
    TRAIN_ARGS+=(--deepspeed_config "${ROOT_DIR}/deepspeed/zero3.json")
  fi
else
  echo "[train] USE_DEEPSPEED=0 (default) -> using torch DDP"
  TRAIN_ARGS+=(--disable_deepspeed)
fi

if [[ -z "${LOAD_IN_8BIT:-}" && -z "${LOAD_IN_4BIT:-}" ]]; then
  LOAD_IN_8BIT=1
  echo "[train] quantization defaulting to LOAD_IN_8BIT=1"
fi

if [[ "${LOAD_IN_8BIT:-0}" == "1" ]]; then
  TRAIN_ARGS+=(--load_in_8bit)
fi
if [[ "${LOAD_IN_4BIT:-0}" == "1" ]]; then
  TRAIN_ARGS+=(--load_in_4bit)
fi

torchrun \
  --nproc_per_node="${GPU_COUNT}" \
  --master_port="${MASTER_PORT}" \
  "${ROOT_DIR}/train_lora_cuda.py" \
  --model_name "${MODEL_NAME:-deepseek-ai/DeepSeek-Coder-V2-Lite-Base}" \
  --public_models_yaml "${ROOT_DIR}/configs/public_models.yaml" \
  --stage1_dataset "${STAGE1}" \
  --stage2_dataset "${STAGE2}" \
  --output_dir "${ROOT_DIR}/output" \
  --max_seq_len "${MAX_SEQ_LEN:-2048}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE:-1}" \
  --gradient_accumulation_steps "${GRAD_ACCUM_STEPS:-16}" \
  --num_train_epochs_stage1 "${EPOCHS_STAGE1:-1.0}" \
  --num_train_epochs_stage2 "${EPOCHS_STAGE2:-1.0}" \
  --learning_rate_stage1 "${LR_STAGE1:-2e-4}" \
  --learning_rate_stage2 "${LR_STAGE2:-1.5e-4}" \
  --lora_r "${LORA_R:-64}" \
  --lora_alpha "${LORA_ALPHA:-128}" \
  --lora_dropout "${LORA_DROPOUT:-0.05}" \
  --save_steps "${SAVE_STEPS:-500}" \
  --eval_steps "${EVAL_STEPS:-500}" \
  --logging_steps "${LOGGING_STEPS:-20}" \
  --seed "${SEED:-42}" \
  "${TRAIN_ARGS[@]}" \
  --packing
