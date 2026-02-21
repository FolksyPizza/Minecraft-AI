# Ollama Conversion (Qwen2.5-Coder-7B + LoRA Adapter)

## 1) Merge adapter into full HF model

```bash
cd /Users/william/Desktop/sk/LoRA
python3 inference/merge_adapter.py \
  --base-model Qwen/Qwen2.5-Coder-7B \
  --adapter /Users/william/Desktop/sk/LoRA/output/stage2_minecraft_adapter \
  --out /Users/william/Desktop/sk/LoRA/inference/merged_qwen2.5_coder_7b_minecraft
```

## 2) Convert merged HF model -> GGUF (using llama.cpp)

```bash
cd /Users/william/Desktop
[ -d llama.cpp ] || git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
python3 -m pip install -r requirements.txt

python3 convert_hf_to_gguf.py \
  /Users/william/Desktop/sk/LoRA/inference/merged_qwen2.5_coder_7b_minecraft \
  --outfile /Users/william/Desktop/sk/LoRA/inference/qwen2.5-coder-7b-minecraft-f16.gguf \
  --outtype f16
```

## 3) Quantize GGUF for Ollama

```bash
cd /Users/william/Desktop/llama.cpp
cmake -B build
cmake --build build -j

./build/bin/llama-quantize \
  /Users/william/Desktop/sk/LoRA/inference/qwen2.5-coder-7b-minecraft-f16.gguf \
  /Users/william/Desktop/sk/LoRA/inference/qwen2.5-coder-7b-minecraft-q4_k_m.gguf \
  q4_k_m
```

## 4) Create Ollama model

Create `/Users/william/Desktop/sk/LoRA/inference/Modelfile`:

```text
FROM /Users/william/Desktop/sk/LoRA/inference/qwen2.5-coder-7b-minecraft-q4_k_m.gguf
PARAMETER temperature 0.2
```

Then:

```bash
cd /Users/william/Desktop/sk/LoRA/inference
ollama create minecraft-coder -f Modelfile
ollama run minecraft-coder
```

## Notes

- Keep `output/stage2_minecraft_adapter` as your training artifact of record.
- GGUF is a deployment format; re-export after future retrains.
