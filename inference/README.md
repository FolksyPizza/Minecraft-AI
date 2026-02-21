# Inference Setup

Use Python 3.10-3.12. Python 3.13+ often fails due missing wheels (`tokenizers`/`safetensors`).

Install inference dependencies in your active virtualenv:

```bash
cd /Users/william/Desktop/sk/LoRA
python3 -m pip install -r inference/requirements.txt
```

Run the adapter:

```bash
python3 inference/run_adapter.py \
  --base-model Qwen/Qwen2.5-Coder-7B \
  --adapter /Users/william/Desktop/sk/LoRA/output/stage2_minecraft_adapter \
  --prompt "Write one final Skript heal command line for a target player and amount." \
  --temperature 0.6 \
  --top-p 0.9 \
  --max-new-tokens 96
```
