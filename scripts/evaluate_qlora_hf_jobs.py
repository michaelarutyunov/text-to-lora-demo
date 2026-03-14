#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "transformers>=4.40.0",
#   "peft>=0.7.0",
#   "datasets>=2.14.0",
#   "scikit-learn>=1.0.0",
#   "torch",
#   "matplotlib",
#   "seaborn",
#   "tqdm",
# ]
# ///
"""
Evaluate Stage 2 QLoRA model on test set via Hugging Face Jobs.

Submitted via Hugging Face Jobs:
    hf jobs uv run \\
        --flavor a10g-small \\
        --timeout 30m \\
        scripts/evaluate_qlora_hf_jobs.py

Environment variables (set via --env or HF Jobs secrets):
    HF_TOKEN        Hugging Face token (for gated model access)
"""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from datasets import load_dataset
from peft import PeftModel
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm

# ── Configuration ─────────────────────────────────────────────────────────────

HF_TOKEN = os.environ.get("HF_TOKEN", None)
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
QLORA_ADAPTER = "michaelarutyunov/jtbd-qlora-mistral7b"
DATASET_REPO = "michaelarutyunov/jtbd-classification"
MAX_NEW_TOKENS = 12

# ── Inlined label schema and prompt format ─────────────────────────────────────

NODE_TYPES: list[str] = [
    "pain_point",
    "gain_point",
    "emotional_job",
    "solution_approach",
    "job_statement",
    "job_context",
    "job_trigger",
    "social_job",
]

PROMPT_TEMPLATE = (
    "[CONTEXT]: {context}\n"
    "[QUOTE]: {quote}\n"
    "Classify the quote into one of: {labels}\n"
    "Answer:"
)

LABELS_STR = " | ".join(NODE_TYPES)


def format_prompt(example: dict, include_answer: bool = False) -> str:
    context = (example.get("context") or "").strip() or "[no context available]"
    prompt = PROMPT_TEMPLATE.format(
        context=context,
        quote=example["quote"].strip(),
        labels=LABELS_STR,
    )
    if include_answer:
        prompt += f" {example['node_type']}"
    return prompt


def extract_label_from_response(response: str) -> str:
    """Extract the first valid node type from the model response."""
    response_clean = response.strip().lower().replace("-", "_")
    for label in NODE_TYPES:
        if label in response_clean:
            return label
    return "unknown"


# ── Load dataset ───────────────────────────────────────────────────────────────

print(f"Loading dataset: {DATASET_REPO}")
raw = load_dataset(DATASET_REPO, token=HF_TOKEN)
test_dataset = raw["test"]
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

output_path = "stage2_confusion_matrix.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"\nConfusion matrix saved to: {output_path}")
plt.close()


# ── Print comparison with baselines ───────────────────────────────────────────

print("\n" + "="*60)
print("Comparison with baselines:")
print("="*60)
print(f"{'Stage':<25} {'Accuracy':>10} {'Macro F1':>10} {'Unknown %':>10}")
print("-"*60)
print(f"{'Stage 0 (zero-shot)':<25} {0.426:>10.3f} {'N/A':>10} {23/197*100:>10.1f}")
print(f"{'Stage 2 (QLoRA)':<25} {acc:>10.3f} {macro_f1:>10.3f} {unknown_pct:>10.1f}")
print("="*60)
