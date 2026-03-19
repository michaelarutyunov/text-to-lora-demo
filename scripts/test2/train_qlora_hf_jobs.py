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
#     "huggingface-hub>=0.22.0",
#     "tqdm>=4.66.0",
# ]
# ///
"""
QLoRA training for Test 2 binary JTBD detectors.

Trains 12 QLoRA adapters: 3 detectors × 4 dataset sizes (50, 100, 200, full).
Each adapter is a binary yes/no classifier for a specific JTBD node type.

Submitted via Hugging Face Jobs:
    hf jobs submit \\
        --flavor a100-large \\
        --timeout 8h \\
        --secrets HF_TOKEN \\
        uv run scripts/test2/train_qlora_hf_jobs.py

Environment variables:
    HF_TOKEN        Hugging Face token (write access required)

LoRA config (r=8, alpha=16, q_proj+v_proj) matches T2L adapters
for a fair Stage 2 vs Stage 1 comparison.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

# ── Configuration ─────────────────────────────────────────────────────────────

HF_TOKEN = os.environ["HF_TOKEN"]
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

# HuggingFace dataset repository IDs (one per detector)
DATASET_REPOS = {
    "job_trigger": "michaelarutyunov/jtbd-binary-job-trigger",
    "solution_approach": "michaelarutyunov/jtbd-binary-solution-approach",
    "pain_point": "michaelarutyunov/jtbd-binary-pain-point",
}

# Detectors and dataset sizes
DETECTORS = ["job_trigger", "solution_approach", "pain_point"]
SIZES = ["50", "100", "200", "full"]

# Hub repo pattern: michaelarutyunov/jtbd-qlora-{detector}-{size}
HUB_REPO_PATTERN = "michaelarutyunov/jtbd-qlora-{detector}-{size}"

# LoRA — matches T2L adapter_config.json (r=8, alpha=16, q_proj+v_proj)
LORA_R = 8
LORA_ALPHA = 16
LORA_TARGET_MODULES = ["q_proj", "v_proj"]
LORA_DROPOUT = 0.05

# Common training params
LR = 2e-4
MAX_SEQ_LEN = 512
WARMUP_RATIO = 0.05
WEIGHT_DECAY = 0.01
SEED = 42

# Data paths (relative to project root)
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "test2"
EVAL_SCRIPT = Path(__file__).resolve().parent / "evaluate_conditions.py"

# ── Adaptive training config ─────────────────────────────────────────────────

TRAINING_CONFIGS = {
    "50": {"epochs": 10, "batch_size": 2, "grad_accum": 4},
    "100": {"epochs": 6, "batch_size": 2, "grad_accum": 4},
    "200": {"epochs": 4, "batch_size": 2, "grad_accum": 8},
    "full": {"epochs": 3, "batch_size": 2, "grad_accum": 8},
}


# ── Data loading ──────────────────────────────────────────────────────────────


def load_jsonl(path: Path) -> list[dict]:
    """Load JSONL file into list of dicts."""
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _load_split_from_hub(detector: str, split: str) -> list[dict]:
    """Download one split from HF Hub and convert int labels to strings."""
    from datasets import load_dataset as hf_load_dataset

    repo_id = DATASET_REPOS[detector]
    print(f"  Downloading {split} split from Hub: {repo_id}")
    ds = hf_load_dataset(repo_id, split=split, token=HF_TOKEN)
    return [
        {
            "utterance": r["utterance"],
            "label": "yes" if r["label"] == 1 else "no",
            "source_file": r["source_file"],
            "turn_number": r["turn_number"],
        }
        for r in ds
    ]


def load_and_subsample(detector: str, size: str) -> tuple[list[dict], list[dict]]:
    """Load training data and subsample to requested size.

    Returns (train_examples, val_examples).
    Tries local JSONL first (dev runs), falls back to HF Hub (HF Jobs).
    Subsampling is deterministic (sorted by source_file + turn_number, then sliced).
    """
    train_file = DATA_DIR / f"{detector}_train.jsonl"
    val_file = DATA_DIR / f"{detector}_val.jsonl"

    if train_file.exists() and val_file.exists():
        train_data = load_jsonl(train_file)
        val_data = load_jsonl(val_file)
    else:
        # HF Jobs: local data/test2/ not available — download from Hub
        train_data = _load_split_from_hub(detector, "train")
        val_data = _load_split_from_hub(detector, "val")

    if size != "full":
        n = int(size)
        # Deterministic subsample: sort then take first n
        train_data.sort(key=lambda x: (x["source_file"], x["turn_number"]))
        train_data = train_data[:n]

    return train_data, val_data


# ── Prompt formatting ─────────────────────────────────────────────────────────


def format_prompt(utterance: str, detector_name: str) -> str:
    """Format utterance into classification prompt (matches evaluate_conditions.py)."""
    label = detector_name.replace("_", " ")
    return (
        f"<s>[INST] [UTTERANCE]: {utterance}\n"
        f"Does this contain a {label}? Answer yes or no.\n"
        f"Answer: [/INST]"
    )


def format_training_example(example: dict, detector: str) -> str:
    """Format a single training example with prompt + response."""
    prompt = format_prompt(example["utterance"], detector)
    return f"{prompt}{example['label']}</s>"


def prepare_dataset(examples: list[dict], detector: str) -> Dataset:
    """Convert raw examples to HF Dataset with formatted text."""
    texts = [format_training_example(ex, detector) for ex in examples]
    return Dataset.from_dict({"text": texts})


# ── Model loading ─────────────────────────────────────────────────────────────


def load_base_model():
    """Load 4-bit quantized base model."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        token=HF_TOKEN,
    )
    model.config.use_cache = False
    return model


def load_tokenizer():
    """Load tokenizer with padding configured."""
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


# ── Training ──────────────────────────────────────────────────────────────────


def train_one(detector: str, size: str) -> str:
    """Train a single QLoRA adapter and push to Hub.

    Returns the Hub repo ID.
    """
    print(f"\n{'=' * 60}")
    print(f"Training: {detector} / {size}")
    print(f"{'=' * 60}")

    # Load data
    train_examples, val_examples = load_and_subsample(detector, size)
    print(f"  Train: {len(train_examples)} examples, Val: {len(val_examples)} examples")

    train_ds = prepare_dataset(train_examples, detector)
    val_ds = prepare_dataset(val_examples, detector)

    # Load model and tokenizer (fresh each run to avoid adapter contamination)
    model = load_base_model()
    tokenizer = load_tokenizer()

    # LoRA config
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Training config (adaptive by size)
    cfg = TRAINING_CONFIGS[size]
    hub_repo = HUB_REPO_PATTERN.format(detector=detector, size=size)
    output_dir = f"/tmp/qlora-{detector}-{size}"

    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["grad_accum"],
        learning_rate=LR,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        max_length=MAX_SEQ_LEN,
        seed=SEED,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        hub_model_id=hub_repo,
        hub_token=HF_TOKEN,
        push_to_hub=True,
        hub_strategy="end",
        report_to="none",
        dataset_text_field="text",
    )

    # Train
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    trainer.train()

    # Push to Hub
    trainer.push_to_hub(commit_message=f"QLoRA {detector} trained on {size} examples")
    print(f"  Pushed to: {hub_repo}")

    # Cleanup
    del trainer, model
    torch.cuda.empty_cache()

    return hub_repo


# ── Evaluate script updater ──────────────────────────────────────────────────


def update_evaluate_script(results: dict[str, dict[str, str]]) -> None:
    """Update QLORA_ADAPTERS in evaluate_conditions.py with trained Hub IDs.

    Args:
        results: {detector: {size: hub_repo_id}}
    """
    content = EVAL_SCRIPT.read_text()

    for detector in DETECTORS:
        for size in SIZES:
            hub_id = results.get(detector, {}).get(size)
            if hub_id is None:
                continue
            content = _replace_in_detector_block(content, detector, size, hub_id)

    EVAL_SCRIPT.write_text(content)
    print(f"\nUpdated {EVAL_SCRIPT} with trained adapter IDs")


def _replace_in_detector_block(
    content: str, detector: str, size: str, hub_id: str
) -> str:
    """Replace None with hub_id for a specific detector/size in QLORA_ADAPTERS."""
    # Find the detector's dict block within QLORA_ADAPTERS
    # Pattern: "detector": {\n ... "size": None ... }
    detector_pattern = rf'("{detector}":\s*\{{[^}}]*?)("{size}":\s*)None([^}}]*?\}})'
    match = re.search(detector_pattern, content, re.DOTALL)
    if match:
        replacement = f'{match.group(1)}{match.group(2)}"{hub_id}"{match.group(3)}'
        content = content[: match.start()] + replacement + content[match.end() :]
    return content


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    print("QLoRA Training for Test 2 Binary Detectors")
    print(f"Base model: {BASE_MODEL}")
    print(f"Data dir: {DATA_DIR}")
    print(f"Detectors: {DETECTORS}")
    print(f"Sizes: {SIZES}")

    results: dict[str, dict[str, str]] = {}

    for detector in DETECTORS:
        results[detector] = {}
        for size in SIZES:
            hub_id = train_one(detector, size)
            results[detector][size] = hub_id

    # Update evaluate_conditions.py (only works on local runs where the file exists)
    try:
        update_evaluate_script(results)
    except FileNotFoundError:
        print(f"\nSkipping evaluate_conditions.py update (file not found at {EVAL_SCRIPT})")

    # Summary
    print("\n" + "=" * 60)
    print("Training complete! All adapters:")
    print("=" * 60)
    for detector in DETECTORS:
        for size in SIZES:
            print(f"  {detector}/{size}: {results[detector][size]}")


if __name__ == "__main__":
    main()
