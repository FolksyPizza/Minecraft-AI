# Inference Setup

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
  --prompt "Write a Skript command to heal a player."
```
