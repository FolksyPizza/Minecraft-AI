# Config Reference

## Training Script (`train_lora_cuda.py`)

Key arguments:
- `--model_name`: preferred base model (anonymous resolution + fallback still applies)
- `--public_models_yaml`: candidate list for no-auth model pull
- `--stage1_dataset`: general coding stage JSONL
- `--stage2_dataset`: minecraft-primary stage JSONL
- `--deepspeed_config`: ZeRO config JSON path
- `--max_seq_len`: context length (default 2048)
- `--per_device_train_batch_size`
- `--gradient_accumulation_steps`
- `--num_train_epochs_stage1`
- `--num_train_epochs_stage2`
- `--learning_rate_stage1`
- `--learning_rate_stage2`
- `--lora_r`, `--lora_alpha`, `--lora_dropout`
- `--target_modules`: comma-separated module names

## DeepSpeed

- `deepspeed/zero3.json`: default for memory efficiency
- `deepspeed/zero2.json`: fallback option

## Stage Composer (`scripts/prepare_stage_datasets.py`)

- Minecraft data stays primary in stage2.
- Non-Minecraft share is dynamically capped to 15-30%.
- Stage1 uses general-first data with a small Minecraft anchor.
