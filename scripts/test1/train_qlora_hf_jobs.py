#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "trl>=0.12.0",
#     "peft>=0.7.0",
#     "transformers>=4.40.0",
#     "accelerate>=0.24.0",
#     "bitsandbytes>=0.43.0",
#     "datasets>=2.14.0",
#     "trackio",
# ]
# ///
"""
QLoRA fine-tuning of Mistral-7B-Instruct on the JTBD classification dataset.

Submitted via Hugging Face Jobs:
    hf jobs uv run \\
        --flavor a10g-large \\
        --timeout 2h \\
        --secrets HF_TOKEN \\
        scripts/train_qlora_hf_jobs.py

Environment variables (set via --env or HF Jobs secrets):
    HF_TOKEN        Hugging Face token (write access required)
    DATASET_REPO    Hub dataset repo, e.g. yourname/jtbd-classification
    OUTPUT_REPO     Hub model repo for the trained adapter

LoRA config (r=8, alpha=16, q_proj+v_proj) matches T2L's adapter_config.json
for a fair Stage 2 vs Stage 1 comparison.
"""

from __future__ import annotations

import os

import torch
import trackio
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

# ── Configuration ─────────────────────────────────────────────────────────────

HF_TOKEN = os.environ["HF_TOKEN"]
DATASET_REPO = os.environ.get("DATASET_REPO", "michaelarutyunov/jtbd-classification")
OUTPUT_REPO = os.environ.get("OUTPUT_REPO", "michaelarutyunov/jtbd-qlora-mistral7b")

BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

# LoRA — matches T2L adapter_config.json (r=8, alpha=16, q_proj+v_proj)
# Keeping these identical ensures Stage 2 and Stage 1 comparisons are fair.
LORA_R = 8
LORA_ALPHA = 16
LORA_TARGET_MODULES = ["q_proj", "v_proj"]

# Training
NUM_EPOCHS = 3
PER_DEVICE_BATCH = 2
GRAD_ACCUM = 8          # effective batch = 16
LR = 2e-4
MAX_SEQ_LEN = 512
WARMUP_RATIO = 0.05
WARMUP_STEPS = None     # Computed from dataset size
WEIGHT_DECAY = 0.01
SEED = 42

# ── Inlined label schema (mirrors src/data_utils.py) ─────────────────────────
# data_utils.py is not available in the ephemeral HF Jobs container; the
# relevant constants and prompt logic are copied here verbatim.

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


# ── Load dataset ──────────────────────────────────────────────────────────────

print(f"Loading dataset: {DATASET_REPO}")
raw = load_dataset(DATASET_REPO, token=HF_TOKEN)
print(f"Splits: { {k: len(v) for k, v in raw.items()} }")

# Add the formatted prompt+answer column that SFTTrainer will consume
def add_text(batch: dict) -> dict:
    batch["text"] = [
        format_prompt(
            {"quote": q, "context": c, "node_type": nt},
            include_answer=True,
        )
        for q, c, nt in zip(batch["quote"], batch["context"], batch["node_type"])
    ]
    return batch

dataset = raw.map(add_text, batched=True, remove_columns=raw["train"].column_names)
train_dataset = dataset["train"]
eval_dataset = dataset["val"]

print(f"Train: {len(train_dataset)} | Eval: {len(eval_dataset)}")
print("Sample:", train_dataset[0]["text"][:200])

# Compute warmup_steps from dataset size
num_training_samples = len(train_dataset)
effective_batch_size = PER_DEVICE_BATCH * GRAD_ACCUM
steps_per_epoch = num_training_samples // effective_batch_size
total_steps = steps_per_epoch * NUM_EPOCHS
warmup_steps = int(total_steps * WARMUP_RATIO)
print(f"Steps: {total_steps} total, {warmup_steps} warmup ({WARMUP_RATIO*100:.1f}%)")

# ── LoRA config ───────────────────────────────────────────────────────────────

peft_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=LORA_TARGET_MODULES,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# ── Training config ───────────────────────────────────────────────────────────

training_args = SFTConfig(
    # Hub
    output_dir="jtbd-qlora-mistral7b",
    push_to_hub=True,
    hub_model_id=OUTPUT_REPO,
    hub_strategy="every_save",

    # Training
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=PER_DEVICE_BATCH,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    warmup_steps=warmup_steps,  # Fixed: use warmup_steps instead of warmup_ratio
    weight_decay=WEIGHT_DECAY,
    max_length=MAX_SEQ_LEN,
    seed=SEED,
    bf16=torch.cuda.is_bf16_supported(),
    gradient_checkpointing=True,

    # Evaluation & checkpointing
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",

    # Logging
    logging_steps=10,
    report_to="trackio",
    project="jtbd-classification",
    run_name="qlora-mistral7b-stage2",
)

# ── Model loading (4-bit quantization) ───────────────────────────────────────────

# Load the base model with 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

print(f"\nLoading model with 4-bit quantization: {BASE_MODEL}")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    token=HF_TOKEN,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

# ── Trainer ───────────────────────────────────────────────────────────────────

print(f"\nBase model: {BASE_MODEL}")
print(f"Output:     {OUTPUT_REPO}")
print(f"LoRA:       r={LORA_R}, alpha={LORA_ALPHA}, targets={LORA_TARGET_MODULES}")

trainer = SFTTrainer(
    model=model,  # Pass the loaded model directly
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    args=training_args,
)

print("\nStarting training...")
trainer.train()

print("Pushing adapter to Hub...")
trainer.push_to_hub()
trackio.finish()

print(f"\nDone! Adapter at: https://huggingface.co/{OUTPUT_REPO}")
print(f"Metrics at:      https://huggingface.co/spaces/YOUR_HF_USERNAME/trackio")
