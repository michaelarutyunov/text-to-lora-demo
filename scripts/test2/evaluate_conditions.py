#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch>=2.0.0",
#     "transformers>=4.40.0",
#     "peft>=0.10.0",
#     "datasets>=2.18.0",
#     "numpy>=1.24.0",
#     "scikit-learn>=1.3.0",
#     "matplotlib>=3.7.0",
#     "seaborn>=0.12.0",
#     "pandas>=2.0.0",
# ]
# ///

"""
Test 2 Phase 5: Full Evaluation Pipeline

Evaluates all 24 experimental conditions:
- Stages 0, 1a, 1b, 1c, 2a, 2b, 2c, 2d
- Across 3 detectors: job_trigger, solution_approach, pain_point

Outputs:
- Metrics JSON with macro F1, per-class precision/recall/F1
- Confusion matrices as PNG files
- Three plots: learning curve, D2L knowledge transfer, confusion matrices
"""

import json
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from datasets import Dataset
from peft import PeftModel
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import AutoModelForCausalLM, AutoTokenizer


# =============================================================================
# Configuration
# =============================================================================

BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

# HuggingFace repository IDs (to be created during Phase 2-4)
D2L_ADAPTER_ID = "michaelarutyunov/jtbd-d2l-mistral7b"  # TODO: Update after Phase 2
T2L_ADAPTERS = {
    "job_trigger": "michaelarutyunov/jtbd-t2l-job-trigger-mistral7b",  # TODO: Update after Phase 3
    "solution_approach": "michaelarutyunov/jtbd-t2l-solution-approach-mistral7b",  # TODO: Update after Phase 3
    "pain_point": "michaelarutyunov/jtbd-t2l-pain-point-mistral7b",  # TODO: Update after Phase 3
}

QLORA_ADAPTERS = {
    # QLoRA adapters trained on 50, 100, 200, full examples
    # TODO: Update after QLoRA training
    "job_trigger": {
        "50": None,
        "100": None,
        "200": None,
        "full": None,
    },
    "solution_approach": {
        "50": None,
        "100": None,
        "200": None,
        "full": None,
    },
    "pain_point": {
        "50": None,
        "100": None,
        "200": None,
        "full": None,
    },
}

# Local dataset paths
DATA_DIR = Path("/home/mikhailarutyunov/projects/text-to-lora-demo/data/test2")
OUTPUT_DIR = Path(
    "/home/mikhailarutyunov/projects/text-to-lora-demo/outputs/test2/evaluation"
)

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# Data Loading
# =============================================================================


def load_test_data(detector_name: str) -> Dataset:
    """Load test set for a specific detector."""
    test_file = DATA_DIR / f"{detector_name}_test.jsonl"

    examples = []
    with open(test_file) as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))

    return Dataset.from_list(examples)


# =============================================================================
# Prompt Formatting
# =============================================================================


def format_prompt(utterance: str, detector_name: str) -> str:
    """Format utterance into classification prompt."""
    label = detector_name.replace("_", " ")
    prompt = f"<s>[INST] [UTTERANCE]: {utterance}\nDoes this contain a {label}? Answer yes or no.\nAnswer: [/INST]"
    return prompt


def extract_answer(output_text: str) -> str:
    """Extract yes/no answer from model output."""
    output = output_text.strip().lower()
    if "yes" in output and "no" not in output:
        return "yes"
    elif "no" in output:
        return "no"
    else:
        return "unknown"


# =============================================================================
# Model Loading
# =============================================================================


def load_base_model():
    """Load base Mistral-7B model and tokenizer."""
    print(f"Loading base model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def load_adapter(model, adapter_id: str):
    """Load a single adapter onto the model."""
    if adapter_id is None:
        return model
    print(f"Loading adapter: {adapter_id}")
    return PeftModel.from_pretrained(model, adapter_id, is_trainable=False)


def load_stacked_adapters(base_model, d2l_id: str, t2l_id: str):
    """Load D2L + T2L adapters stacked."""
    print(f"Loading stacked adapters: D2L={d2l_id}, T2L={t2l_id}")

    # Load T2L adapter first
    model = load_adapter(base_model, t2l_id)

    # Add D2L adapter
    model.load_adapter(d2l_id, adapter_name="d2l")

    # Enable both adapters
    model.set_adapter(["default", "d2l"])
    model.eval()

    return model


# =============================================================================
# Inference
# =============================================================================


def run_inference(
    model, tokenizer, dataset: Dataset, detector_name: str, batch_size: int = 4
):
    """Run inference on dataset and return predictions."""
    predictions = []

    for i in range(0, len(dataset), batch_size):
        batch = dataset[i : i + batch_size]
        utterances = batch["utterance"]

        prompts = [format_prompt(u, detector_name) for u in utterances]
        inputs = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode only the generated portion
        input_length = inputs["input_ids"].shape[1]
        generated_texts = tokenizer.batch_decode(
            outputs[:, input_length:], skip_special_tokens=True
        )

        for text in generated_texts:
            predictions.append(extract_answer(text))

    return predictions


# =============================================================================
# Metrics Computation
# =============================================================================


def compute_metrics(y_true, y_pred) -> dict[str, Any]:
    """Compute classification metrics."""
    # Filter out unknown predictions
    valid_mask = [p != "unknown" for p in y_pred]
    y_true_valid = [y_true[i] for i, v in enumerate(valid_mask) if v]
    y_pred_valid = [y_pred[i] for i, v in enumerate(valid_mask) if v]

    # Convert to binary labels
    label_map = {"yes": 1, "no": 0}
    y_true_binary = [label_map.get(lbl, 0) for lbl in y_true_valid]
    y_pred_binary = [label_map.get(lbl, 0) for lbl in y_pred_valid]

    # Compute metrics
    accuracy = (
        sum(a == b for a, b in zip(y_true_binary, y_pred_binary)) / len(y_true_binary)
        if y_true_binary
        else 0
    )

    precision = precision_score(
        y_true_binary, y_pred_binary, pos_label=1, zero_division=0
    )
    recall = recall_score(y_true_binary, y_pred_binary, pos_label=1, zero_division=0)
    f1 = f1_score(y_true_binary, y_pred_binary, pos_label=1, zero_division=0)

    # No (negative) class metrics
    precision_neg = precision_score(
        y_true_binary, y_pred_binary, pos_label=0, zero_division=0
    )
    recall_neg = recall_score(
        y_true_binary, y_pred_binary, pos_label=0, zero_division=0
    )
    f1_neg = f1_score(y_true_binary, y_pred_binary, pos_label=0, zero_division=0)

    macro_f1 = (f1 + f1_neg) / 2

    return {
        "accuracy": accuracy,
        "precision_yes": float(precision),
        "recall_yes": float(recall),
        "f1_yes": float(f1),
        "precision_no": float(precision_neg),
        "recall_no": float(recall_neg),
        "f1_no": float(f1_neg),
        "macro_f1": float(macro_f1),
        "n_valid": len(y_true_valid),
        "n_unknown": len(y_pred) - len(y_true_valid),
    }


def compute_confusion_matrix(y_true, y_pred) -> np.ndarray:
    """Compute confusion matrix."""
    label_map = {"yes": 1, "no": 0}
    y_true_binary = [label_map.get(lbl, 0) for lbl in y_true]
    y_pred_binary = [label_map.get(lbl, 0) for lbl in y_pred]

    return confusion_matrix(
        y_true_binary, y_pred_binary, labels=[1, 0]
    )  # yes, no order


# =============================================================================
# Evaluation Stages
# =============================================================================


def evaluate_stage_0(model, tokenizer, dataset: Dataset, detector_name: str) -> dict:
    """Stage 0: Zero-shot (no adapters)."""
    print(f"\n=== Stage 0: Zero-shot ({detector_name}) ===")
    predictions = run_inference(model, tokenizer, dataset, detector_name)
    y_true = dataset["label"]

    metrics = compute_metrics(y_true, predictions)
    cm = compute_confusion_matrix(y_true, predictions)

    return {"predictions": predictions, "metrics": metrics, "confusion_matrix": cm}


def evaluate_stage_1a(
    base_model, tokenizer, dataset: Dataset, detector_name: str
) -> Optional[dict]:
    """Stage 1a: T2L only (task adapter)."""
    print(f"\n=== Stage 1a: T2L only ({detector_name}) ===")
    adapter_id = T2L_ADAPTERS[detector_name]
    if adapter_id is None:
        print(f"Warning: T2L adapter not available for {detector_name}")
        return None

    model = load_adapter(base_model, adapter_id)
    predictions = run_inference(model, tokenizer, dataset, detector_name)
    y_true = dataset["label"]

    metrics = compute_metrics(y_true, predictions)
    cm = compute_confusion_matrix(y_true, predictions)

    del model
    torch.cuda.empty_cache()

    return {"predictions": predictions, "metrics": metrics, "confusion_matrix": cm}


def evaluate_stage_1b(
    base_model, tokenizer, dataset: Dataset, detector_name: str
) -> Optional[dict]:
    """Stage 1b: D2L only (methodology adapter)."""
    print(f"\n=== Stage 1b: D2L only ({detector_name}) ===")
    if D2L_ADAPTER_ID is None:
        print("Warning: D2L adapter not available")
        return None

    model = load_adapter(base_model, D2L_ADAPTER_ID)
    predictions = run_inference(model, tokenizer, dataset, detector_name)
    y_true = dataset["label"]

    metrics = compute_metrics(y_true, predictions)
    cm = compute_confusion_matrix(y_true, predictions)

    del model
    torch.cuda.empty_cache()

    return {"predictions": predictions, "metrics": metrics, "confusion_matrix": cm}


def evaluate_stage_1c(
    base_model, tokenizer, dataset: Dataset, detector_name: str
) -> Optional[dict]:
    """Stage 1c: D2L + T2L stacked."""
    print(f"\n=== Stage 1c: D2L + T2L ({detector_name}) ===")
    t2l_id = T2L_ADAPTERS[detector_name]
    if D2L_ADAPTER_ID is None or t2l_id is None:
        print(f"Warning: Stacked adapters not available for {detector_name}")
        return None

    model = load_stacked_adapters(base_model, D2L_ADAPTER_ID, t2l_id)
    predictions = run_inference(model, tokenizer, dataset, detector_name)
    y_true = dataset["label"]

    metrics = compute_metrics(y_true, predictions)
    cm = compute_confusion_matrix(y_true, predictions)

    del model
    torch.cuda.empty_cache()

    return {"predictions": predictions, "metrics": metrics, "confusion_matrix": cm}


def evaluate_stage_2(
    base_model, tokenizer, dataset: Dataset, detector_name: str, n_examples: str
) -> Optional[dict]:
    """Stage 2: QLoRA with n examples (50, 100, 200, full)."""
    print(f"\n=== Stage 2: QLoRA-{n_examples} ({detector_name}) ===")
    adapter_id = QLORA_ADAPTERS[detector_name][n_examples]
    if adapter_id is None:
        print(f"Warning: QLoRA-{n_examples} adapter not available for {detector_name}")
        return None

    model = load_adapter(base_model, adapter_id)
    predictions = run_inference(model, tokenizer, dataset, detector_name)
    y_true = dataset["label"]

    metrics = compute_metrics(y_true, predictions)
    cm = compute_confusion_matrix(y_true, predictions)

    del model
    torch.cuda.empty_cache()

    return {"predictions": predictions, "metrics": metrics, "confusion_matrix": cm}


# =============================================================================
# Plotting
# =============================================================================


def plot_learning_curve(results: dict, output_path: Path):
    """Plot 1: Learning curve per detector."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    detectors = ["job_trigger", "solution_approach", "pain_point"]
    stages_x = ["0", "1a", "1b", "1c", "2a", "2b", "2c", "2d"]
    stage_labels = ["Zero", "T2L", "D2L", "D2L+T2L", "50", "100", "200", "Full"]

    for idx, detector in enumerate(detectors):
        ax = axes[idx]

        # Collect macro F1 values
        f1_values = []
        for stage in [
            "stage_0",
            "stage_1a",
            "stage_1b",
            "stage_1c",
            "stage_2_50",
            "stage_2_100",
            "stage_2_200",
            "stage_2_full",
        ]:
            result = results.get(detector, {}).get(stage)
            if result and result.get("metrics"):
                f1_values.append(result["metrics"]["macro_f1"])
            else:
                f1_values.append(np.nan)

        # Plot
        ax.plot(range(len(stages_x)), f1_values, marker="o", linewidth=2)
        ax.set_xticks(range(len(stages_x)))
        ax.set_xticklabels(stage_labels, rotation=45, ha="right")
        ax.set_ylabel("Macro F1")
        ax.set_title(f"{detector.replace('_', ' ').title()}")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path / "learning_curve.png", dpi=150)
    plt.close()
    print(f"Saved: {output_path / 'learning_curve.png'}")


def plot_d2l_knowledge_transfer(results: dict, output_path: Path):
    """Plot 2: D2L knowledge transfer bar chart."""
    fig, ax = plt.subplots(figsize=(10, 6))

    detectors = ["job_trigger", "solution_approach", "pain_point"]
    detector_labels = ["Job Trigger", "Solution Approach", "Pain Point"]

    zero_shot_f1 = []
    d2l_only_f1 = []

    for detector in detectors:
        zero = (
            results.get(detector, {})
            .get("stage_0", {})
            .get("metrics", {})
            .get("macro_f1", 0)
        )
        d2l = (
            results.get(detector, {})
            .get("stage_1b", {})
            .get("metrics", {})
            .get("macro_f1", 0)
        )
        zero_shot_f1.append(zero)
        d2l_only_f1.append(d2l)

    x = np.arange(len(detectors))
    width = 0.35

    ax.bar(x - width / 2, zero_shot_f1, width, label="Zero-shot", alpha=0.8)
    ax.bar(x + width / 2, d2l_only_f1, width, label="D2L only", alpha=0.8)

    ax.set_ylabel("Macro F1")
    ax.set_title("D2L Knowledge Transfer: Zero-shot vs D2L-only")
    ax.set_xticks(x)
    ax.set_xticklabels(detector_labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 1)

    # Add value labels on bars
    for i, (z, d) in enumerate(zip(zero_shot_f1, d2l_only_f1)):
        ax.text(
            i - width / 2, z + 0.02, f"{z:.2f}", ha="center", va="bottom", fontsize=9
        )
        ax.text(
            i + width / 2, d + 0.02, f"{d:.2f}", ha="center", va="bottom", fontsize=9
        )

    plt.tight_layout()
    plt.savefig(output_path / "d2l_knowledge_transfer.png", dpi=150)
    plt.close()
    print(f"Saved: {output_path / 'd2l_knowledge_transfer.png'}")


def plot_confusion_matrices(results: dict, output_path: Path):
    """Plot 3: Confusion matrices comparison."""
    detectors = ["job_trigger", "solution_approach", "pain_point"]
    stages = ["stage_0", "stage_1c", "stage_2_full"]
    stage_labels = ["Zero-shot", "D2L+T2L", "QLoRA-Full"]

    fig, axes = plt.subplots(len(stages), len(detectors), figsize=(12, 12))

    for stage_idx, (stage, stage_label) in enumerate(zip(stages, stage_labels)):
        for detector_idx, detector in enumerate(detectors):
            ax = axes[stage_idx, detector_idx]

            result = results.get(detector, {}).get(stage)
            if result and result.get("confusion_matrix") is not None:
                cm = result["confusion_matrix"]
            else:
                cm = np.array([[0, 0], [0, 0]])

            # Plot heatmap
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                ax=ax,
                xticklabels=["Yes", "No"],
                yticklabels=["Yes", "No"],
            )

            if stage_idx == 0:
                ax.set_title(f"{detector.replace('_', ' ').title()}", fontweight="bold")
            if detector_idx == 0:
                ax.set_ylabel(stage_label, fontweight="bold")

    plt.suptitle("Confusion Matrices Comparison", fontsize=14, y=1.00)
    plt.tight_layout()
    plt.savefig(output_path / "confusion_matrices.png", dpi=150)
    plt.close()
    print(f"Saved: {output_path / 'confusion_matrices.png'}")


def save_confusion_matrix_individual(
    cm: np.ndarray, stage: str, detector: str, output_path: Path
):
    """Save individual confusion matrix as PNG."""
    fig, ax = plt.subplots(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        xticklabels=["Yes", "No"],
        yticklabels=["Yes", "No"],
    )

    ax.set_title(f"{detector.replace('_', ' ').title()} - {stage}")
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")

    plt.tight_layout()
    filename = f"{detector}_{stage}_confusion_matrix.png"
    plt.savefig(output_path / filename, dpi=150, bbox_inches="tight")
    plt.close()


# =============================================================================
# Main
# =============================================================================


def main():
    """Run full evaluation pipeline."""
    print("=" * 60)
    print("Test 2 Phase 5: Full Evaluation Pipeline")
    print("=" * 60)

    # Setup output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load base model once
    base_model, tokenizer = load_base_model()

    # Results storage
    results = {}

    # Evaluate each detector
    for detector in ["job_trigger", "solution_approach", "pain_point"]:
        print(f"\n{'=' * 60}")
        print(f"Detector: {detector}")
        print(f"{'=' * 60}")

        # Load test data
        dataset = load_test_data(detector)
        print(f"Test set size: {len(dataset)} utterances")

        results[detector] = {}

        # Stage 0: Zero-shot
        result = evaluate_stage_0(base_model, tokenizer, dataset, detector)
        results[detector]["stage_0"] = result
        save_confusion_matrix_individual(
            result["confusion_matrix"], "zero_shot", detector, OUTPUT_DIR
        )

        # Stage 1a: T2L only
        result = evaluate_stage_1a(base_model, tokenizer, dataset, detector)
        if result:
            results[detector]["stage_1a"] = result
            save_confusion_matrix_individual(
                result["confusion_matrix"], "t2l_only", detector, OUTPUT_DIR
            )

        # Stage 1b: D2L only
        result = evaluate_stage_1b(base_model, tokenizer, dataset, detector)
        if result:
            results[detector]["stage_1b"] = result
            save_confusion_matrix_individual(
                result["confusion_matrix"], "d2l_only", detector, OUTPUT_DIR
            )

        # Stage 1c: D2L + T2L
        result = evaluate_stage_1c(base_model, tokenizer, dataset, detector)
        if result:
            results[detector]["stage_1c"] = result
            save_confusion_matrix_individual(
                result["confusion_matrix"], "d2l_t2l", detector, OUTPUT_DIR
            )

        # Stage 2: QLoRA variants
        for n_examples in ["50", "100", "200", "full"]:
            result = evaluate_stage_2(
                base_model, tokenizer, dataset, detector, n_examples
            )
            if result:
                results[detector][f"stage_2_{n_examples}"] = result
                save_confusion_matrix_individual(
                    result["confusion_matrix"],
                    f"qlora_{n_examples}",
                    detector,
                    OUTPUT_DIR,
                )

    # Save metrics as JSON
    metrics_path = OUTPUT_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        # Convert numpy types for JSON serialization
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            raise TypeError

        json.dump(results, f, indent=2, default=convert)
    print(f"\nSaved metrics to: {metrics_path}")

    # Generate plots
    print("\nGenerating plots...")
    plot_learning_curve(results, OUTPUT_DIR)
    plot_d2l_knowledge_transfer(results, OUTPUT_DIR)
    plot_confusion_matrices(results, OUTPUT_DIR)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    summary_data = []
    for detector in ["job_trigger", "solution_approach", "pain_point"]:
        print(f"\n{detector.replace('_', ' ').title()}:")
        print(f"{'Stage':<15} {'Macro F1':<10} {'Accuracy':<10}")
        print("-" * 35)

        for stage_key, stage_label in [
            ("stage_0", "Zero-shot"),
            ("stage_1a", "T2L only"),
            ("stage_1b", "D2L only"),
            ("stage_1c", "D2L+T2L"),
            ("stage_2_50", "QLoRA-50"),
            ("stage_2_100", "QLoRA-100"),
            ("stage_2_200", "QLoRA-200"),
            ("stage_2_full", "QLoRA-Full"),
        ]:
            result = results.get(detector, {}).get(stage_key)
            if result and result.get("metrics"):
                metrics = result["metrics"]
                f1 = metrics["macro_f1"]
                acc = metrics["accuracy"]
                print(f"{stage_label:<15} {f1:<10.3f} {acc:<10.3f}")

                summary_data.append(
                    {
                        "detector": detector,
                        "stage": stage_label,
                        "macro_f1": f1,
                        "accuracy": acc,
                        "precision_yes": metrics["precision_yes"],
                        "recall_yes": metrics["recall_yes"],
                        "precision_no": metrics["precision_no"],
                        "recall_no": metrics["recall_no"],
                    }
                )

    # Save summary as CSV
    summary_df = pd.DataFrame(summary_data)
    summary_path = OUTPUT_DIR / "summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved summary to: {summary_path}")

    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
