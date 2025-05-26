# 🌙 Dreamloop

A minimal Transformer-based language model written in C++. Loads a pre-trained model and tokenizer, then generates text one token at a time — building stories from your imagination.

---

### ✨ Features

- Pure C++ implementation — no deep learning frameworks
- Efficient memory handling (no repeated allocations)
- Runs a small GPT-style model from binary weights
- Generates coherent sequences based on a transformer architecture

---

### 🔧 Requirements

- C++17 or later
- Windows (uses `windows.h` and UTF-8 output)
- Pretrained model and tokenizer:  ### from https://github.com/karpathy/llama2.c
  - `model.bin` ### OG model, rename to model.bin
  - `tokenizer.bin` ### tokenizer.bin

---
