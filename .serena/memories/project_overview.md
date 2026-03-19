# text-to-lora-demo: Project Overview

## Purpose
Proof-of-concept applying Text-to-LoRA (T2L, Sakana AI, ICML 2025) to consumer interview analysis.

**Current experiment (Test 2):** Binary pain point presence detection from consumer utterances.
Compares T2L zero-shot adapter vs QLoRA fine-tuning at various data sizes (0, 50, 100, 200, ~220 examples).

**Archived experiment (Test 1):** 8-class JTBD node classification (55.8% acc, macro F1 0.32).

## Tech Stack
- Python 3.10+, managed via `uv`
- HuggingFace ecosystem: `transformers`, `peft`, `datasets`, `accelerate`, `bitsandbytes`
- Base model: Mistral-7B-Instruct-v0.2 with QLoRA (4-bit, r=8, alpha=16, q_proj+v_proj)
- Training: Google Colab (T4/A100) or Hugging Face Jobs (a10g-large)
- Experiment tracking: Trackio
- Notebooks: Jupyter

## Codebase Structure
```
src/data_utils.py          # shared label schema, prompt formatting, loaders
scripts/test1/             # archived 8-class scripts
scripts/test2/             # active binary detection scripts
  prepare_dataset_v2.py    # generates data/processed/ splits
  push_dataset_to_hub.py   # pushes to HF Hub
  train_qlora_hf_jobs.py   # HF Jobs training script
  evaluate_qlora_hf_jobs.py
notebooks/
  00_data_prep.ipynb       # EDA, CPU
  01_baselines.ipynb       # Stage 0 (zero-shot) + Stage 1 (T2L), A100
  02_lora_finetune.ipynb   # Stage 2 QLoRA, A100
data/processed/            # committed JSONL splits (train/val/test)
```

## Data
- Raw interviews from `interview-system-v2/synthetic_interviews` (not committed)
- Only JTBD methodology interviews included
- ~313 utterances, binary pain_point labels
