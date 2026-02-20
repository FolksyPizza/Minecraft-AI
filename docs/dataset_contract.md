# Dataset Contract

## Labeled LoRA JSONL

Each line must be valid JSON with exactly:

```json
{"prompt": "...", "completion": "..."}
```

Rules:
- both values must be non-empty strings
- no extra keys in final stage datasets
- UTF-8 encoding
- one JSON object per line

## Unlabeled JSONL

Used for optional pretraining corpora. Not used directly by `train_lora_cuda.py`.

## Validation

Run:

```bash
python scripts/validate_dataset.py labeled/final_general_stage1.jsonl labeled/final_minecraft_primary.jsonl
```
