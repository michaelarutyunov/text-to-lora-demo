#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch==2.6.0",
#     "transformers==4.51.3",
#     "peft>=0.14.0",
#     "accelerate==1.6.0",
#     "safetensors",
#     "huggingface-hub>=0.20.0",
#     "datasets>=2.14.0",
#     "numpy>=1.24.0",
# ]
#
# [tool.uv]
# torch-backend = "cu124"
# ///

"""
D2L Investigation: Why does the Doc-to-LoRA adapter have zero effect on
classification predictions?

Tests three hypotheses:

H1 — Near-zero weights: the hypernetwork generated a no-op adapter.
     Compares L2 / Frobenius norms of D2L vs T2L adapter tensors.

H2 — Architectural irrelevance: down_proj modifications don't propagate
     to the final "yes"/"no" token logit for short generation tasks.
     Compares logits and per-layer hidden-state norms with / without D2L.

H3 — Stacking silences T2L: loading D2L first with adapter_name="default"
     means the second adapter ("t2l_detector") is not active during forward.
     Checks model.active_adapters, tries set_adapter([both]), reverse order.

Results are uploaded to michaelarutyunov/jtbd-test2-d2l-investigation.
"""

import json
import os
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import HfApi, hf_hub_download
from peft import PeftModel
from safetensors.torch import load_file as st_load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

HF_TOKEN = os.environ.get("HF_TOKEN")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
D2L_ADAPTER_ID = "michaelarutyunov/jtbd-d2l-mistral7b-methodology"
T2L_ADAPTERS = {
    "job_trigger": "michaelarutyunov/jtbd-t2l-jobtrigger",
    "solution_approach": "michaelarutyunov/jtbd-t2l-solutionapproach",
    "pain_point": "michaelarutyunov/jtbd-t2l-painpoint",
}
RESULTS_REPO = "michaelarutyunov/jtbd-test2-d2l-investigation"
OUTPUT_DIR = Path("/tmp/d2l_investigation")
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================================
# Utilities
# ============================================================================


def format_prompt(utterance: str, detector_name: str) -> str:
    label = detector_name.replace("_", " ")
    return (
        f"<s>[INST] [UTTERANCE]: {utterance}\n"
        f"Does this contain a {label}? Answer yes or no.\nAnswer: [/INST]"
    )


def cleanup(model):
    """Unload PEFT adapter and return clean base model."""
    if isinstance(model, PeftModel):
        clean = model.unload()
        del model
        torch.cuda.empty_cache()
        return clean
    return model


# ============================================================================
# H1: Weight Magnitude Analysis
# ============================================================================


def _load_adapter_weights(adapter_id: str) -> dict:
    """Download adapter safetensors and return as CPU float32 tensors."""
    try:
        path = hf_hub_download(
            repo_id=adapter_id,
            filename="adapter_model.safetensors",
            token=HF_TOKEN,
        )
        return {k: v.float() for k, v in st_load_file(path).items()}
    except Exception:
        path = hf_hub_download(
            repo_id=adapter_id, filename="adapter_model.bin", token=HF_TOKEN
        )
        return {
            k: v.float() for k, v in torch.load(path, map_location="cpu").items()
        }


def _load_adapter_config(adapter_id: str) -> dict:
    path = hf_hub_download(
        repo_id=adapter_id, filename="adapter_config.json", token=HF_TOKEN
    )
    with open(path) as f:
        return json.load(f)


def _norm_stats(weights: dict, substring: str) -> dict:
    matching = {k: v for k, v in weights.items() if substring in k}
    if not matching:
        return {"n_tensors": 0}
    l2s = [v.norm(2).item() for v in matching.values()]
    frobs = [v.norm("fro").item() for v in matching.values()]
    maxabs = [v.abs().max().item() for v in matching.values()]
    return {
        "n_tensors": len(matching),
        "mean_l2_norm": float(np.mean(l2s)),
        "max_l2_norm": float(np.max(l2s)),
        "mean_frobenius_norm": float(np.mean(frobs)),
        "max_frobenius_norm": float(np.max(frobs)),
        "mean_max_abs": float(np.mean(maxabs)),
        "max_max_abs": float(np.max(maxabs)),
        # Include per-tensor view for the first 5 tensors
        "sample_tensors": {
            k: {
                "shape": list(v.shape),
                "l2_norm": float(v.norm(2).item()),
                "max_abs": float(v.abs().max().item()),
            }
            for k, v in list(matching.items())[:5]
        },
    }


def run_h1_analysis() -> dict:
    print("\n" + "=" * 60)
    print("H1: WEIGHT MAGNITUDE ANALYSIS")
    print("=" * 60)

    results = {}

    # --- D2L adapter ---
    print(f"\nD2L adapter: {D2L_ADAPTER_ID}")
    d2l_config = _load_adapter_config(D2L_ADAPTER_ID)
    d2l_weights = _load_adapter_weights(D2L_ADAPTER_ID)
    d2l_all_keys = list(d2l_weights.keys())[:10]
    print(f"  Config: target_modules={d2l_config.get('target_modules')}, "
          f"r={d2l_config.get('r')}, alpha={d2l_config.get('lora_alpha')}")
    print(f"  Sample keys: {d2l_all_keys[:3]}")

    d2l_down = _norm_stats(d2l_weights, "down_proj")
    d2l_lora_A = _norm_stats(d2l_weights, "lora_A")
    d2l_lora_B = _norm_stats(d2l_weights, "lora_B")

    print(f"  down_proj tensors: {d2l_down.get('n_tensors', 0)}")
    print(f"  lora_A mean_L2={d2l_lora_A.get('mean_l2_norm', 'N/A'):.6f}  "
          f"max_abs={d2l_lora_A.get('mean_max_abs', 'N/A'):.6f}")
    print(f"  lora_B mean_L2={d2l_lora_B.get('mean_l2_norm', 'N/A'):.6f}  "
          f"max_abs={d2l_lora_B.get('mean_max_abs', 'N/A'):.6f}")

    results["d2l"] = {
        "adapter_id": D2L_ADAPTER_ID,
        "config": d2l_config,
        "all_key_sample": d2l_all_keys,
        "down_proj_stats": d2l_down,
        "lora_A_stats": d2l_lora_A,
        "lora_B_stats": d2l_lora_B,
    }

    # --- T2L adapters ---
    for det_name, adapter_id in T2L_ADAPTERS.items():
        print(f"\nT2L adapter ({det_name}): {adapter_id}")
        t2l_config = _load_adapter_config(adapter_id)
        t2l_weights = _load_adapter_weights(adapter_id)
        print(f"  Config: target_modules={t2l_config.get('target_modules')}, "
              f"r={t2l_config.get('r')}, alpha={t2l_config.get('lora_alpha')}")

        t2l_q = _norm_stats(t2l_weights, "q_proj")
        t2l_v = _norm_stats(t2l_weights, "v_proj")
        t2l_A = _norm_stats(t2l_weights, "lora_A")
        t2l_B = _norm_stats(t2l_weights, "lora_B")

        print(f"  q_proj lora_A mean_L2={t2l_q.get('mean_l2_norm', 'N/A'):.6f}  "
              f"lora_B mean_L2={t2l_v.get('mean_l2_norm', 'N/A'):.6f}")
        print(f"  lora_A mean_L2={t2l_A.get('mean_l2_norm', 'N/A'):.6f}  "
              f"max_abs={t2l_A.get('mean_max_abs', 'N/A'):.6f}")
        print(f"  lora_B mean_L2={t2l_B.get('mean_l2_norm', 'N/A'):.6f}  "
              f"max_abs={t2l_B.get('mean_max_abs', 'N/A'):.6f}")

        results[f"t2l_{det_name}"] = {
            "adapter_id": adapter_id,
            "config": t2l_config,
            "q_proj_stats": t2l_q,
            "v_proj_stats": t2l_v,
            "lora_A_stats": t2l_A,
            "lora_B_stats": t2l_B,
        }

    return results


# ============================================================================
# H2: Logit & Hidden-State Analysis
# ============================================================================


def _get_logits_and_hidden_states(model, tokenizer, prompt: str) -> dict:
    """Forward pass returning logits and per-layer hidden-state norms."""
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)

    last_logits = outputs.logits[0, -1, :].float().cpu()

    # Resolve "yes" / "no" token IDs (try with and without leading space)
    def first_id(surface: str) -> int:
        ids = tokenizer.encode(surface, add_special_tokens=False)
        return ids[0]

    yes_id = first_id(" yes") if tokenizer.encode(" yes", add_special_tokens=False) else first_id("yes")
    no_id = first_id(" no") if tokenizer.encode(" no", add_special_tokens=False) else first_id("no")
    Yes_id = first_id("Yes")
    No_id = first_id("No")

    top5_indices = torch.topk(last_logits, 5).indices.tolist()
    top5_tokens = [
        (tokenizer.decode([idx]), float(last_logits[idx].item()))
        for idx in top5_indices
    ]

    # Hidden state L2 norms at last token position, per layer
    hs_norms = [
        float(layer_hs[0, -1, :].float().norm(2).item())
        for layer_hs in outputs.hidden_states
    ]

    return {
        "yes_id": yes_id,
        "no_id": no_id,
        "yes_logit": float(last_logits[yes_id].item()),
        "no_logit": float(last_logits[no_id].item()),
        "Yes_logit": float(last_logits[Yes_id].item()),
        "No_logit": float(last_logits[No_id].item()),
        "logit_diff_yes_minus_no": float(
            last_logits[yes_id].item() - last_logits[no_id].item()
        ),
        "top5_tokens": top5_tokens,
        "hidden_state_norms": hs_norms,
    }


def run_h2_analysis(base_model, tokenizer) -> tuple[dict, object]:
    """
    Returns (results_dict, clean_base_model).

    For each test example:
    1. Get logits from base model (no adapter).
    2. Load D2L adapter, get logits again.
    3. Check whether logits are numerically identical.
    4. Check whether any hidden-state norms changed.
    """
    print("\n" + "=" * 60)
    print("H2: LOGIT & HIDDEN-STATE ANALYSIS")
    print("=" * 60)

    test_examples = [
        (
            "I started looking for a new job when my commute became unbearable.",
            "job_trigger",
        ),
        (
            "The checkout process is really frustrating and takes too long.",
            "pain_point",
        ),
        (
            "I use a spreadsheet to track all my expenses manually.",
            "solution_approach",
        ),
    ]

    results = {}

    for utterance, detector_name in test_examples:
        prompt = format_prompt(utterance, detector_name)
        key = f"{detector_name}"

        print(f"\n[{detector_name}] {utterance[:60]}...")

        # Base (no adapter)
        base_stats = _get_logits_and_hidden_states(base_model, tokenizer, prompt)
        print(f"  Base: yes={base_stats['yes_logit']:.5f}  no={base_stats['no_logit']:.5f}  "
              f"diff={base_stats['logit_diff_yes_minus_no']:.5f}")
        print(f"  Base top3: {base_stats['top5_tokens'][:3]}")

        # With D2L adapter
        model_d2l = PeftModel.from_pretrained(
            base_model, D2L_ADAPTER_ID, is_trainable=False, token=HF_TOKEN
        )
        model_d2l.eval()

        # Inspect active adapters
        try:
            active = model_d2l.active_adapters
        except Exception as e:
            active = f"error: {e}"
        print(f"  D2L active_adapters: {active}")

        d2l_stats = _get_logits_and_hidden_states(model_d2l, tokenizer, prompt)
        print(f"  D2L:  yes={d2l_stats['yes_logit']:.5f}  no={d2l_stats['no_logit']:.5f}  "
              f"diff={d2l_stats['logit_diff_yes_minus_no']:.5f}")
        print(f"  D2L top3: {d2l_stats['top5_tokens'][:3]}")

        yes_delta = abs(base_stats["yes_logit"] - d2l_stats["yes_logit"])
        no_delta = abs(base_stats["no_logit"] - d2l_stats["no_logit"])
        hs_deltas = [
            abs(a - b)
            for a, b in zip(
                base_stats["hidden_state_norms"], d2l_stats["hidden_state_norms"]
            )
        ]
        max_hs_delta = max(hs_deltas) if hs_deltas else 0.0
        print(f"  |Δyes_logit|={yes_delta:.2e}  |Δno_logit|={no_delta:.2e}  "
              f"max|Δhs_norm|={max_hs_delta:.2e}")

        # Clean up
        base_model = cleanup(model_d2l)

        results[key] = {
            "utterance": utterance,
            "detector": detector_name,
            "base": base_stats,
            "d2l": d2l_stats,
            "d2l_active_adapters": str(active),
            "abs_yes_logit_delta": yes_delta,
            "abs_no_logit_delta": no_delta,
            "max_hidden_state_norm_delta": max_hs_delta,
            "logits_numerically_identical": yes_delta < 1e-6 and no_delta < 1e-6,
            "hidden_states_numerically_identical": max_hs_delta < 1e-6,
        }

    return results, base_model


# ============================================================================
# H3: Adapter Stacking Analysis
# ============================================================================


def run_h3_analysis(base_model, tokenizer) -> tuple[dict, object]:
    """
    Returns (results_dict, clean_base_model).

    Loads adapters in multiple configurations for job_trigger and checks:
    - Which adapters are reported as active
    - Whether logits match base / T2L-only / zero-shot
    - What happens with set_adapter([both]) vs individual names
    - Whether load order (D2L first vs T2L first) matters
    """
    print("\n" + "=" * 60)
    print("H3: ADAPTER STACKING ANALYSIS")
    print("=" * 60)

    utterance = "I started looking for a new job when my commute became unbearable."
    detector_name = "job_trigger"
    t2l_id = T2L_ADAPTERS[detector_name]
    prompt = format_prompt(utterance, detector_name)

    results = {}

    # --- Baseline: base model ---
    base_stats = _get_logits_and_hidden_states(base_model, tokenizer, prompt)
    results["base"] = {
        "yes_logit": base_stats["yes_logit"],
        "no_logit": base_stats["no_logit"],
        "top5": base_stats["top5_tokens"],
    }
    print(f"\nBase:    yes={base_stats['yes_logit']:.5f}  no={base_stats['no_logit']:.5f}")

    # --- T2L only ---
    model_t2l = PeftModel.from_pretrained(
        base_model, t2l_id, is_trainable=False, token=HF_TOKEN
    )
    model_t2l.eval()
    t2l_stats = _get_logits_and_hidden_states(model_t2l, tokenizer, prompt)
    t2l_active = str(getattr(model_t2l, "active_adapters", "unavailable"))
    results["t2l_only"] = {
        "yes_logit": t2l_stats["yes_logit"],
        "no_logit": t2l_stats["no_logit"],
        "top5": t2l_stats["top5_tokens"],
        "active_adapters": t2l_active,
    }
    print(f"T2L:     yes={t2l_stats['yes_logit']:.5f}  no={t2l_stats['no_logit']:.5f}  "
          f"active={t2l_active}")
    base_model = cleanup(model_t2l)

    # --- D2L only ---
    model_d2l = PeftModel.from_pretrained(
        base_model, D2L_ADAPTER_ID, is_trainable=False, token=HF_TOKEN
    )
    model_d2l.eval()
    d2l_stats = _get_logits_and_hidden_states(model_d2l, tokenizer, prompt)
    d2l_active = str(getattr(model_d2l, "active_adapters", "unavailable"))
    results["d2l_only"] = {
        "yes_logit": d2l_stats["yes_logit"],
        "no_logit": d2l_stats["no_logit"],
        "top5": d2l_stats["top5_tokens"],
        "active_adapters": d2l_active,
    }
    print(f"D2L:     yes={d2l_stats['yes_logit']:.5f}  no={d2l_stats['no_logit']:.5f}  "
          f"active={d2l_active}")
    base_model = cleanup(model_d2l)

    # --- Stacked: D2L first (original load order) ---
    print("\nLoading stacked (D2L='default' first, then T2L='t2l_detector')...")
    model_stk = PeftModel.from_pretrained(
        base_model, D2L_ADAPTER_ID, adapter_name="default", is_trainable=False, token=HF_TOKEN
    )
    model_stk.load_adapter(t2l_id, adapter_name="t2l_detector", token=HF_TOKEN)
    model_stk.eval()

    active_after_load = str(getattr(model_stk, "active_adapters", "unavailable"))
    print(f"  active_adapters immediately after load: {active_after_load}")

    stk_default = _get_logits_and_hidden_states(model_stk, tokenizer, prompt)
    results["stacked_d2l_first_default_active"] = {
        "yes_logit": stk_default["yes_logit"],
        "no_logit": stk_default["no_logit"],
        "top5": stk_default["top5_tokens"],
        "active_adapters": active_after_load,
        "note": "forward pass with whatever adapter(s) are active after loading both",
    }
    print(f"  default active: yes={stk_default['yes_logit']:.5f}  no={stk_default['no_logit']:.5f}")

    # Try set_adapter("t2l_detector") — only the T2L adapter
    try:
        model_stk.set_adapter("t2l_detector")
        active_t2l_only = str(getattr(model_stk, "active_adapters", "unavailable"))
        stk_t2l = _get_logits_and_hidden_states(model_stk, tokenizer, prompt)
        results["stacked_set_t2l_only"] = {
            "yes_logit": stk_t2l["yes_logit"],
            "no_logit": stk_t2l["no_logit"],
            "top5": stk_t2l["top5_tokens"],
            "active_adapters": active_t2l_only,
            "matches_t2l_standalone": abs(stk_t2l["yes_logit"] - t2l_stats["yes_logit"]) < 1e-4,
        }
        print(f"  set_adapter('t2l_detector'): yes={stk_t2l['yes_logit']:.5f}  no={stk_t2l['no_logit']:.5f}  "
              f"matches_t2l_standalone={results['stacked_set_t2l_only']['matches_t2l_standalone']}")
    except Exception as e:
        results["stacked_set_t2l_only"] = {"error": str(e)}
        print(f"  set_adapter('t2l_detector') failed: {e}")

    # Try set_adapter(["default", "t2l_detector"]) — both simultaneously
    try:
        model_stk.set_adapter(["default", "t2l_detector"])
        active_both = str(getattr(model_stk, "active_adapters", "unavailable"))
        stk_both = _get_logits_and_hidden_states(model_stk, tokenizer, prompt)
        results["stacked_set_both"] = {
            "yes_logit": stk_both["yes_logit"],
            "no_logit": stk_both["no_logit"],
            "top5": stk_both["top5_tokens"],
            "active_adapters": active_both,
        }
        print(f"  set_adapter([both]): yes={stk_both['yes_logit']:.5f}  no={stk_both['no_logit']:.5f}")
    except Exception as e:
        results["stacked_set_both"] = {"error": str(e)}
        print(f"  set_adapter([both]) failed: {e}")

    base_model = cleanup(model_stk)

    # --- Stacked: T2L first (reverse order) ---
    print("\nLoading stacked (T2L='default' first, then D2L='d2l')...")
    model_stk_rev = PeftModel.from_pretrained(
        base_model, t2l_id, adapter_name="default", is_trainable=False, token=HF_TOKEN
    )
    model_stk_rev.load_adapter(D2L_ADAPTER_ID, adapter_name="d2l", token=HF_TOKEN)
    model_stk_rev.eval()

    active_rev = str(getattr(model_stk_rev, "active_adapters", "unavailable"))
    stk_rev = _get_logits_and_hidden_states(model_stk_rev, tokenizer, prompt)
    results["stacked_t2l_first_default_active"] = {
        "yes_logit": stk_rev["yes_logit"],
        "no_logit": stk_rev["no_logit"],
        "top5": stk_rev["top5_tokens"],
        "active_adapters": active_rev,
        "matches_t2l_standalone": abs(stk_rev["yes_logit"] - t2l_stats["yes_logit"]) < 1e-4,
    }
    print(f"  T2L-first default active: yes={stk_rev['yes_logit']:.5f}  no={stk_rev['no_logit']:.5f}  "
          f"matches_t2l_standalone={results['stacked_t2l_first_default_active']['matches_t2l_standalone']}")

    base_model = cleanup(model_stk_rev)

    return results, base_model


# ============================================================================
# Summary & Upload
# ============================================================================


def print_summary(results: dict) -> None:
    print("\n" + "=" * 60)
    print("INVESTIGATION SUMMARY")
    print("=" * 60)

    # H1
    h1 = results.get("h1_weights", {})
    d2l_A = h1.get("d2l", {}).get("lora_A_stats", {})
    d2l_B = h1.get("d2l", {}).get("lora_B_stats", {})
    t2l = h1.get("t2l_job_trigger", {})
    t2l_A = t2l.get("lora_A_stats", {})
    t2l_B = t2l.get("lora_B_stats", {})

    print("\nH1 — Weight magnitudes:")
    print(f"  D2L lora_A: mean_L2={d2l_A.get('mean_l2_norm', 0):.6f}  "
          f"max_abs={d2l_A.get('max_max_abs', 0):.6f}")
    print(f"  D2L lora_B: mean_L2={d2l_B.get('mean_l2_norm', 0):.6f}  "
          f"max_abs={d2l_B.get('max_max_abs', 0):.6f}")
    print(f"  T2L lora_A: mean_L2={t2l_A.get('mean_l2_norm', 0):.6f}  "
          f"max_abs={t2l_A.get('max_max_abs', 0):.6f}")
    print(f"  T2L lora_B: mean_L2={t2l_B.get('mean_l2_norm', 0):.6f}  "
          f"max_abs={t2l_B.get('max_max_abs', 0):.6f}")
    ratio_A = (
        d2l_A.get("mean_l2_norm", 0) / t2l_A.get("mean_l2_norm", 1)
        if t2l_A.get("mean_l2_norm", 0) > 0
        else float("nan")
    )
    print(f"  D2L/T2L lora_A ratio: {ratio_A:.4f}x")

    # H2
    h2 = results.get("h2_logits", {})
    print("\nH2 — Logit changes with D2L applied:")
    for det, info in h2.items():
        identical = info.get("logits_numerically_identical", "?")
        hs_identical = info.get("hidden_states_numerically_identical", "?")
        print(f"  {det}: logits_identical={identical}  hidden_states_identical={hs_identical}  "
              f"|Δyes|={info.get('abs_yes_logit_delta', 0):.2e}")

    # H3
    h3 = results.get("h3_stacking", {})
    print("\nH3 — Stacking conditions:")
    for cond in [
        "base",
        "t2l_only",
        "d2l_only",
        "stacked_d2l_first_default_active",
        "stacked_set_t2l_only",
        "stacked_set_both",
        "stacked_t2l_first_default_active",
    ]:
        info = h3.get(cond, {})
        if "error" in info:
            print(f"  {cond}: ERROR — {info['error']}")
        elif "yes_logit" in info:
            extra = ""
            if "matches_t2l_standalone" in info:
                extra = f"  matches_t2l={info['matches_t2l_standalone']}"
            print(f"  {cond}: yes={info['yes_logit']:.5f}  no={info['no_logit']:.5f}"
                  f"  active={info.get('active_adapters', '?')}{extra}")


def upload_results(results: dict) -> None:
    api = HfApi(token=HF_TOKEN)
    api.create_repo(RESULTS_REPO, repo_type="dataset", exist_ok=True, private=True)

    report_path = OUTPUT_DIR / "investigation_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)

    api.upload_file(
        path_or_fileobj=str(report_path),
        path_in_repo="investigation_report.json",
        repo_id=RESULTS_REPO,
        repo_type="dataset",
    )
    print(f"\nResults uploaded to: https://huggingface.co/datasets/{RESULTS_REPO}")


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    print("=" * 60)
    print("D2L Investigation Script")
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print("=" * 60)

    results = {}

    # H1: Weight inspection — no base model needed
    results["h1_weights"] = run_h1_analysis()

    # Load base model once for H2 + H3
    print(f"\nLoading base model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=HF_TOKEN,
    )
    base_model.eval()

    # H2: Logit analysis
    h2_results, base_model = run_h2_analysis(base_model, tokenizer)
    results["h2_logits"] = h2_results

    # H3: Stacking analysis
    h3_results, base_model = run_h3_analysis(base_model, tokenizer)
    results["h3_stacking"] = h3_results

    # Print summary and upload
    print_summary(results)
    upload_results(results)


if __name__ == "__main__":
    main()
