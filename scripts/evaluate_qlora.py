#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch",
#   "transformers>=4.40.0",
#   "peft>=0.7.0",
#   "datasets>=2.14.0",
#   "scikit-learn>=1.0.0",
#   "matplotlib",
#   "seaborn",
#   "tqdm",
# ]
# ///
"""
Evaluate Stage 2 QLoRA model on the test set.

Usage:
    uv run scripts/evaluate_qlora.py

Output:
    - Accuracy and macro F1 score
    - Classification report
    - Confusion matrix (saved as .png)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from datasets import load_dataset
from peft import PeftModel
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm

# Add src to path for imports
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))

from data_utils import NODE_TYPES, format_prompt, extract_label_from_response


# ── Configuration ─────────────────────────────────────────────────────────────

HF_TOKEN = os.environ.get("HF_TOKEN", None)
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
QLORA_ADAPTER = "michaelarutyunov/jtbd-qlora-mistral7b"
DATASET_REPO = "michaelarutyunov/jtbd-classification"
MAX_NEW_TOKENS = 12


# ── Load dataset ───────────────────────────────────────────────────────────────

# Try Hub first, fall back to local data
try:
    print(f"Loading dataset from Hub: {DATASET_REPO}")
    raw = load_dataset(DATASET_REPO, token=HF_TOKEN)
    test_dataset = raw["test"]
except Exception as e:
    print(f"Hub load failed: {e}")
    print("Falling back to local data...")
    from data_utils import load_splits, to_hf_dataset
    splits = load_splits(str(repo_root / "data" / "processed"))
    from datasets import Dataset
    test_dataset = Dataset.from_list(splits["test"])

print(f"Test set: {len(test_dataset)} examples")


# ── Load model + adapter ─────────────────────────────────────────────────────────

print(f"\nLoading tokenizer: {BASE_MODEL}")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=HF_TOKEN)
tokenizer.pad_token = tokenizer.eos_token

print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    token=HF_TOKEN,
)
model.eval()

print(f"Loading QLoRA adapter: {QLORA_ADAPTER}")
model = PeftModel.from_pretrained(model, QLORA_ADAPTER)
model.eval()
print("Model + adapter loaded.")


# ── Inference ───────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_label(prompt: str) -> str:
    """Greedy decode; extract first valid label from generation."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return extract_label_from_response(raw)


print("\nRunning inference on test set...")
preds, labels = [], []
for ex in tqdm(test_dataset, desc="Stage 2 (QLoRA)"):
    prompt = format_prompt(ex, include_answer=False)
    pred = predict_label(prompt)
    preds.append(pred)
    labels.append(ex["node_type"])


# ── Metrics ─────────────────────────────────────────────────────────────────────

# Filter out "unknown" predictions for fair comparison
valid_mask = [p != "unknown" for p in preds]
valid_preds = [p for p, m in zip(preds, valid_mask) if m]
valid_labels = [l for l, m in zip(labels, valid_mask) if m]

acc = sum(p == l for p, l in zip(valid_preds, valid_labels)) / len(valid_labels)
macro_f1 = f1_score(valid_labels, valid_preds, labels=NODE_TYPES, average="macro", zero_division=0)
unknown_pct = sum(p == "unknown" for p in preds) / len(preds) * 100

print(f"\n{'='*60}")
print(f"Stage 2 (QLoRA) Results:")
print(f"{'='*60}")
print(f"Accuracy:    {acc:.3f}")
print(f"Macro F1:    {macro_f1:.3f}")
print(f"Unknown:     {sum(p == 'unknown' for p in preds)} / {len(preds)} ({unknown_pct:.1f}%)")
print(f"Valid:       {len(valid_preds)} / {len(preds)}")
print(f"{'='*60}")

print("\nClassification report:")
print(classification_report(valid_labels, valid_preds, labels=NODE_TYPES, zero_division=0))


# ── Confusion matrix ───────────────────────────────────────────────────────────

cm = confusion_matrix(valid_labels, valid_preds, labels=NODE_TYPES)
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=NODE_TYPES,
    yticklabels=NODE_TYPES,
    ax=ax,
)
ax.set_title("Stage 2: QLoRA Fine-tuned — Confusion Matrix")
ax.set_ylabel("True")
ax.set_xlabel("Predicted")
plt.tight_layout()

output_path = repo_root / "results" / "stage2_confusion_matrix.png"
output_path.parent.mkdir(exist_ok=True)
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"\nConfusion matrix saved to: {output_path}")
plt.close()


# ── Print comparison with baselines (if available) ─────────────────────────────

print("\n" + "="*60)
print("Comparison with baselines (from notebook 01):")
print("="*60)
print(f"{'Stage':<20} {'Accuracy':>10} {'Macro F1':>10} {'Unknown %':>10}")
print("-"*60)
print(f"{'Stage 0 (zero-shot)':<20} {0.426:>10.3f} {'N/A':>10} {23/197*100:>10.1f}")
print(f"{'Stage 1 (T2L)':<20} {'N/A':>10} {'N/A':>10} {'N/A':>10}")
print(f"{'Stage 2 (QLoRA)':<20} {acc:>10.3f} {macro_f1:>10.3f} {unknown_pct:>10.1f}")
print("="*60)
