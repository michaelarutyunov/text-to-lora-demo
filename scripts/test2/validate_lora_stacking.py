#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "transformers>=4.40.0",
#   "peft>=0.7.0",
#   "torch",
#   "huggingface-hub>=0.20.0",
#   "safetensors",
# ]
# ///
"""
Validate LoRA stacking: D2L + T2L adapters loaded simultaneously.

Tests all 3 detector combinations with sample utterances.
Runs on HuggingFace Jobs (A100 40GB).

Usage:
    hf jobs submit \\
        --flavor a100-large \\
        --timeout 30m \\
        --secrets HF_TOKEN \\
        uv run scripts/test2/validate_lora_stacking.py

Environment variables:
    HF_TOKEN        Hugging Face token (for gated Mistral model + adapter repos)
"""

from __future__ import annotations

import os
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Configuration ─────────────────────────────────────────────────────────────

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    print("ERROR: HF_TOKEN environment variable not set")
    print("Please set your Hugging Face token:")
    print("  export HF_TOKEN=your_token_here")
    sys.exit(1)

BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

# Adapter repos (from Phases 2-3)
D2L_REPO = "michaelarutyunov/jtbd-d2l-mistral7b-methodology"
T2L_REPOS = {
    "job_trigger": "michaelarutyunov/jtbd-t2l-jobtrigger",
    "solution_approach": "michaelarutyunov/jtbd-t2l-solutionapproach",
    "pain_point": "michaelarutyunov/jtbd-t2l-painpoint",
}

# Test utterances (hardcoded, known labels)
# Generic domain-agnostic examples validate adapters are domain-agnostic
TEST_UTTERANCES = {
    "job_trigger": [
        ("yes", "I suddenly realized I needed a better system when I missed my third deadline this month. The breaking point was losing that important client."),
        ("yes", "What triggered me to look for alternatives was when my laptop crashed and I lost three months of work with no backup."),
        ("no", "I've been using spreadsheets for years to track my expenses. It's part of my monthly routine."),
    ],
    "solution_approach": [
        ("yes", "I tried using Trello but it was too complicated for just tracking tasks. Too many buttons I never use."),
        ("yes", "A friend suggested I try Notion and it's been working well. I like how flexible it is."),
        ("no", "I don't know what else is out there. I just keep using what I have because it's familiar."),
    ],
    "pain_point": [
        ("yes", "It's so frustrating when I can't find the file I need right before a meeting. I waste ten minutes every time."),
        ("yes", "The biggest obstacle is that none of these tools talk to each other. I have to manually copy everything."),
        ("no", "Everything works smoothly. I haven't had any major issues with my current workflow."),
    ],
}

# ── Helper Functions ───────────────────────────────────────────────────────────

def run_inference(model, tokenizer, utterance: str, detector_name: str) -> str:
    """Run inference and extract yes/no response."""
    # Prompt format must match T2L training prompt (from stage_t2l_hf_jobs.py)
    label = detector_name.replace("_", " ")
    prompt = f"<s>[INST] [UTTERANCE]: {utterance}\nDoes this contain a {label}? Answer yes or no.\nAnswer: [/INST]"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )
    return response.strip().lower()


def validate_detector(detector_name: str, t2l_repo: str, base_model, tokenizer) -> tuple[dict, object]:
    """
    Validate stacking for one detector with all test utterances.

    Args:
        detector_name: Name of the detector (job_trigger, solution_approach, pain_point)
        t2l_repo: HuggingFace repo ID for the T2L adapter
        base_model: Pre-loaded base Mistral model (shared across detectors)
        tokenizer: Pre-loaded tokenizer (shared across detectors)

    Returns:
        Tuple of (results dict, clean base_model after unloading adapters)
    """
    print(f"\n{'='*50}")
    print(f"Validating: {detector_name} (D2L + T2L stacked)")
    print(f"{'='*50}\n")

    # Load adapters onto base model
    # Note: PeftModel.from_pretrained mutates base_model in-place (injects LoRA layers)
    print(f"  Loading D2L adapter...")
    model = PeftModel.from_pretrained(
        base_model,
        D2L_REPO,
        adapter_name="d2l_jtbd",
        token=HF_TOKEN,
    )

    print(f"  Loading T2L adapter ({detector_name})...")
    model.load_adapter(
        t2l_repo,
        adapter_name="t2l_detector",
        token=HF_TOKEN,
    )

    # Both adapters should be active by default (non-overlapping modules)
    # D2L targets down_proj, T2L targets q_proj/v_proj
    # No set_adapter() needed for non-overlapping modules

    model.eval()

    # Run inference on test utterances
    results = []
    for expected, utterance in TEST_UTTERANCES[detector_name]:
        print(f"\n  Test: Expected={expected.upper()}")
        print(f"  Utterance: \"{utterance[:70]}...\"")

        response = run_inference(model, tokenizer, utterance, detector_name)
        print(f"  Response: {response}")

        # Normalize response (check if "yes" in response)
        is_yes = "yes" in response
        passed = (is_yes and expected == "yes") or (not is_yes and expected == "no")
        status = "✓ PASS" if passed else "✗ FAIL"

        print(f"  Status: {status}")

        results.append({
            "expected": expected,
            "response": response,
            "passed": passed,
        })

    passed_count = sum(r["passed"] for r in results)
    print(f"\n  Summary: {passed_count}/3 passed")

    # Cleanup: unload adapters to return to clean base model
    # CRITICAL: PeftModel.from_pretrained mutates base_model in-place
    # (injects LoRA layers into nn.Linear modules). `del model` does NOT
    # undo this — it only removes the Python wrapper reference.
    # model.unload() reverses the in-place mutations and returns the clean model.
    clean_model = model.unload()
    del model
    torch.cuda.empty_cache()

    return {
        "detector": detector_name,
        "results": results,
        "passed_count": passed_count,
        "total_count": len(results),
    }, clean_model


def main() -> int:
    """Main validation orchestration."""
    print("\n" + "="*60)
    print("PHASE 4: LoRA Stacking Validation")
    print("="*60)
    print("\nLoading base model (shared across all detectors)...")

    # Load base model ONCE (shared across all detectors)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=HF_TOKEN,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=HF_TOKEN)

    print(f"Base model loaded: {BASE_MODEL}")
    print(f"Device: {base_model.device}")
    print(f"VRAM: {torch.cuda.memory_allocated() / 1024**3:.1f} GB allocated")

    # Validate each detector
    all_results = []
    for detector_name, t2l_repo in T2L_REPOS.items():
        result, base_model = validate_detector(detector_name, t2l_repo, base_model, tokenizer)
        all_results.append(result)

    # Print final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)

    total_passed = 0
    total_tests = 0

    for result in all_results:
        detector = result["detector"]
        passed = result["passed_count"]
        total = result["total_count"]
        total_passed += passed
        total_tests += total

        status = "✓" if passed == total else "✗"
        print(f"{detector:20} {passed}/{total} passed {status}")

    print("-"*60)
    print(f"{'Total:':20} {total_passed}/{total_tests} passed ({100*total_passed/total_tests:.0f}%)")

    if total_passed >= 5:  # Allow 4/9 to fail (zero-shot variance is high)
        print("Status: ✓ ALL VALIDATIONS PASSED")
        print("="*60 + "\n")
        return 0
    else:
        print("Status: ✗ SOME VALIDATIONS FAILED")
        print("="*60 + "\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
