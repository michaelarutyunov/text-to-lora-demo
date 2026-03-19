#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "transformers>=4.40.0",
#   "peft>=0.14.0",
#   "torch==2.6.0",
#   "sentence-transformers",
#   "huggingface-hub>=0.20.0",
#   "safetensors",
#   "inflect",
#   "pyyaml",
#   "accelerate",
#   "datasets",
#   "einops",
# ]
#
# [tool.uv]
# torch-backend = "cu124"
# ///
"""
Generate T2L adapters for binary JTBD node type detection.

Runs on HuggingFace Jobs (A100 40GB). Generates 3 LoRA adapters:
  - job_trigger: Detect trigger events that prompted solution-seeking
  - solution_approach: Detect current or potential solution methods
  - pain_point: Detect frustrations, obstacles, or difficulties

Each adapter is generated from a task description alone (zero-shot).

Submitted via Hugging Face Jobs:
    hf jobs submit \\
        --flavor a100-large \\
        --timeout 1h \\
        --secrets HF_TOKEN \\
        uv run scripts/test2/stage_t2l_hf_jobs.py

Environment variables:
    HF_TOKEN        Hugging Face token (for gated Mistral model)
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timezone

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Configuration ─────────────────────────────────────────────────────────────

HF_TOKEN = os.environ.get("HF_TOKEN")
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

# T2L hypernetwork location
T2L_REPO_URL = "https://github.com/SakanaAI/text-to-lora.git"
T2L_REPO_DIR = "/tmp/text-to-lora"

# Task descriptions for each detector
DETECTORS = {
    "job_trigger": {
        "task_description": (
            "Binary classification: given a consumer utterance from a JTBD interview, "
            "determine whether it describes a trigger event — something that initiated "
            "or prompted the person to seek a solution. Output 'yes' or 'no'."
        ),
        "output_repo": "michaelarutyunov/jtbd-t2l-jobtrigger",
        "output_dir": "/tmp/t2l_job_trigger",
    },
    "solution_approach": {
        "task_description": (
            "Binary classification: given a consumer utterance from a JTBD interview, "
            "determine whether it describes a solution approach — a current or potential "
            "method the person has tried or considered. Output 'yes' or 'no'."
        ),
        "output_repo": "michaelarutyunov/jtbd-t2l-solutionapproach",
        "output_dir": "/tmp/t2l_solution_approach",
    },
    "pain_point": {
        "task_description": (
            "Binary classification: given a consumer utterance from a JTBD interview, "
            "determine whether it contains a pain point — a frustration, obstacle, or "
            "difficulty in getting a job done. Output 'yes' or 'no'."
        ),
        "output_repo": "michaelarutyunov/jtbd-t2l-painpoint",
        "output_dir": "/tmp/t2l_pain_point",
    },
}

# Trust remote code for embedding models (avoids interactive prompt on HF Jobs)
os.environ["HF_HUB_TRUST_REMOTE_CODE"] = "1"

# ── Helper Functions ────────────────────────────────────────────────────────────

def get_git_commit(repo_dir: str) -> str:
    """Get the git commit hash of a repository."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return "unknown"


def clone_t2l_repo() -> str:
    """Clone T2L repository and return the directory path."""
    print(f"Cloning T2L repo from {T2L_REPO_URL}...")
    
    if os.path.exists(T2L_REPO_DIR):
        print(f"  T2L repo already exists at {T2L_REPO_DIR}")
        return T2L_REPO_DIR
    
    subprocess.run(
        ["git", "clone", "--depth", "1", T2L_REPO_URL, T2L_REPO_DIR],
        check=True,
    )
    
    # Install T2L's transitive deps (but NOT T2L itself — avoid transformers version conflict)
    print(f"  Installing T2L transitive dependencies...")
    subprocess.run(
        ["uv", "pip", "install", "--python", sys.executable,
         "torchmetrics", "inflect", "rouge-score", "wandb"],
        check=True,
    )

    # Patch model_loading.py to default use_flash_attn=False
    # T2L doesn't need flash-attn for its forward pass, only the loader defaults to it
    model_loading_path = os.path.join(T2L_REPO_DIR, "src/hyper_llm_modulator/utils/model_loading.py")
    with open(model_loading_path) as f:
        src = f.read()
    src = src.replace("use_flash_attn=True", "use_flash_attn=False")
    # Fix chat template path to be absolute (relative fails when CWD != repo root)
    src = src.replace(
        'template_path = f"chat_templates/',
        f'template_path = f"{T2L_REPO_DIR}/chat_templates/',  # T2L_REPO_DIR is resolved at script load time
    )
    with open(model_loading_path, "w") as f:
        f.write(src)
    print(f"  Patched model_loading.py: use_flash_attn=False, absolute template path")

    commit_hash = get_git_commit(T2L_REPO_DIR)
    print(f"  T2L repo cloned: {T2L_REPO_DIR}")
    print(f"  Commit: {commit_hash}")

    return T2L_REPO_DIR


def download_t2l_checkpoint() -> str:
    """Download T2L hypernetwork checkpoint from HuggingFace Hub."""
    print(f"\nDownloading T2L checkpoint from SakanaAI/text-to-lora...")
    
    ckpt_dir = snapshot_download(
        repo_id="SakanaAI/text-to-lora",
        allow_patterns="trained_t2l/mistral_7b_t2l/*",
        token=HF_TOKEN,
    )
    
    checkpoint_path = f"{ckpt_dir}/trained_t2l/mistral_7b_t2l/hypermod.pt"
    print(f"  Checkpoint downloaded: {checkpoint_path}")
    
    return checkpoint_path


def generate_t2l_adapter(
    checkpoint_path: str,
    task_description: str,
    detector_name: str,
    output_dir: str,
):
    """
    Generate a single T2L adapter from a task description.
    
    This follows the Sakana AI T2L method:
    1. Load hypernetwork checkpoint
    2. Embed task description
    3. Generate LoRA weights via hypernetwork forward pass
    4. Save as PEFT adapter
    """
    print(f"\n{'='*60}")
    print(f"Generating T2L adapter: {detector_name}")
    print(f"{'='*60}")
    
    # Add T2L repo to path
    sys.path.insert(0, os.path.join(T2L_REPO_DIR, "src"))
    
    from hyper_llm_modulator.hyper_modulator import load_hypermod_checkpoint, save_lora
    from hyper_llm_modulator.utils import embed_texts
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load hypernetwork checkpoint
    print(f"  Loading hypernetwork checkpoint...")
    (args, hypermod, _base_model, _base_tok,
     emb_model, emb_tok, task_desc_fmt, pooling_fn) = load_hypermod_checkpoint(
        checkpoint_path, device
    )
    
    # Free the duplicate base model loaded by hypernetwork to recover VRAM
    del _base_model, _base_tok
    torch.cuda.empty_cache()
    print(f"  Checkpoint loaded; duplicate base model released")
    
    # Embed task description and generate LoRA weights
    print(f"  Generating LoRA weights from task description...")
    with torch.no_grad():
        task_emb = embed_texts(
            [task_description], 
            emb_model, 
            emb_tok, 
            task_desc_fmt, 
            pooling_fn, 
            device
        )
        encoder_out = hypermod.task_encoder(task_emb)
        encoded = encoder_out["encoded_task_emb"].detach()
        
        n_layers = 32  # Mistral-7B has 32 transformer layers
        layer_indices = torch.tensor(
            range(n_layers),
            dtype=torch.long,
            device=device,
        )
        
        # Generate LoRA weights for all layers
        lora_sd = hypermod.gen_lora(layer_indices, encoded)
    
    # Use the PEFT config from the hypernetwork (already loaded from checkpoint)
    peft_cfg = hypermod.peft_config
    
    # Save adapter
    print(f"  Saving adapter to {output_dir}...")
    save_lora(lora_sd, peft_cfg, output_dir)
    print(f"  Adapter saved successfully")
    
    # Extract metadata
    metadata = {
        "detector_type": detector_name,
        "task_description": task_description,
        "base_model": BASE_MODEL,
        "lora_r": peft_cfg.r,
        "lora_alpha": peft_cfg.lora_alpha,
        "target_modules": list(peft_cfg.target_modules),
        "generation_timestamp": datetime.now(timezone.utc).isoformat(),
        "t2l_checkpoint": checkpoint_path,
        "t2l_repo_commit": get_git_commit(T2L_REPO_DIR),
    }
    
    return metadata


def save_metadata(output_dir: str, metadata: dict) -> None:
    """Save metadata as JSON."""
    import json
    
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"  Saved metadata.json")


def create_readme(output_dir: str, detector_name: str, metadata: dict) -> None:
    """Create README.md for the adapter."""
    readme = f"""# T2L Adapter: {detector_name}

Text-to-LoRA (T2L) adapter for JTBD node type detection.

## Detector Type

{detector_name}

## Task Description

```
{metadata['task_description']}
```

## Base Model

{BASE_MODEL}

## Generation Method

Generated using Sakana AI's Text-to-LoRA hypernetwork:
- Paper: [ICML 2025](https://openreview.net/forum?id=xxxxx)
- Code: https://github.com/SakanaAI/text-to-lora

This adapter was created from a task description alone (zero-shot, no labeled examples).

## Configuration

- LoRA rank (r): {metadata['lora_r']}
- LoRA alpha: {metadata['lora_alpha']}
- Target modules: {metadata['target_modules']}

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    "{BASE_MODEL}",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# Load T2L adapter
model = PeftModel.from_pretrained(
    model,
    "{metadata.get('output_repo', 'path/to/adapter')}",
)

# Use for inference
prompt = "<s>[INST] [UTTERANCE]: Your utterance here\\nDoes this contain a {detector_name}? Answer yes or no.\\nAnswer: [/INST]"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=5)
response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(response)
```

## Integration with D2L

This adapter targets `q_proj` and `v_proj` (attention layers).
Can be stacked with a D2L adapter (targets `down_proj`) for combined knowledge:

```python
model.load_adapter("path/to/d2l_adapter", adapter_name="d2l")
model.load_adapter("path/to/t2l_{detector_name}", adapter_name="t2l")
model.set_adapter(["d2l", "t2l_{detector_name}"])
```

## Metadata

- Generated: {metadata['generation_timestamp']}
- T2L checkpoint: {metadata['t2l_checkpoint']}
- T2L repo commit: {metadata['t2l_repo_commit']}

---

**Generated as part of Test 2: JTBD Node Type Detection with D2L + T2L**
"""

    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write(readme)
    
    print(f"  Created README.md")


def push_to_huggingface(output_dir: str, repo_id: str) -> None:
    """Push adapter to HuggingFace Hub."""
    print(f"\n  Pushing to HuggingFace Hub: {repo_id}")
    
    from huggingface_hub import HfApi
    
    api = HfApi(token=HF_TOKEN)
    
    # Create repo if it doesn't exist
    try:
        api.create_repo(
            repo_id=repo_id,
            exist_ok=True,
            repo_type="model",
        )
        print(f"  Repository ensured: {repo_id}")
    except Exception as e:
        print(f"  Note: {e}")
    
    # Upload files
    files_to_upload = [
        "adapter_model.safetensors",
        "adapter_config.json", 
        "metadata.json",
        "README.md",
    ]
    
    for filename in files_to_upload:
        file_path = os.path.join(output_dir, filename)
        if os.path.exists(file_path):
            api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=filename,
                repo_id=repo_id,
                repo_type="model",
            )
            print(f"  Uploaded: {filename}")
    
    print(f"  Adapter pushed to: https://huggingface.co/{repo_id}")


def validate_adapter(adapter_dir: str, detector_name: str, base_model, tokenizer) -> None:
    """
    Basic validation: load adapter onto pre-loaded base model and run a test.
    """
    print(f"\n  Validating adapter: {detector_name}")

    from peft import PeftModel

    # Load adapter onto base model
    print(f"    Loading adapter...")
    model_with_adapter = PeftModel.from_pretrained(base_model, adapter_dir)
    model_with_adapter.eval()

    # Test inference
    test_utterance = "I've been struggling to find a good way to track my expenses."
    test_prompt = f"<s>[INST] [UTTERANCE]: {test_utterance}\nDoes this contain a {detector_name.replace('_', ' ')}? Answer yes or no.\nAnswer: [/INST]"

    inputs = tokenizer(test_prompt, return_tensors="pt").to(base_model.device)

    with torch.no_grad():
        outputs = model_with_adapter.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"    Test utterance: {test_utterance}")
    print(f"    Response: {response}")
    print(f"    Validation complete")

    # Unload adapter to free VRAM for next detector
    del model_with_adapter
    torch.cuda.empty_cache()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    """Main entry point."""
    print("="*60)
    print("Phase 3: T2L Adapter Generation")
    print("="*60)
    print(f"Base model: {BASE_MODEL}")
    print(f"Detectors: {len(DETECTORS)}")
    
    # Clone T2L repo
    clone_t2l_repo()
    
    # Download checkpoint
    checkpoint_path = download_t2l_checkpoint()
    
    # Load base model ONCE for validation (reused across all detectors)
    print(f"\nLoading base model for validation...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=HF_TOKEN,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Base model loaded")

    # Generate adapters for each detector
    results = {}

    for detector_name, config in DETECTORS.items():
        try:
            # Generate adapter
            metadata = generate_t2l_adapter(
                checkpoint_path=checkpoint_path,
                task_description=config["task_description"],
                detector_name=detector_name,
                output_dir=config["output_dir"],
            )

            # Add output repo to metadata
            metadata["output_repo"] = config["output_repo"]

            # Save docs (save_lora already creates adapter_config.json and weights)
            save_metadata(config["output_dir"], metadata)
            create_readme(config["output_dir"], detector_name, metadata)

            # Validate
            validate_adapter(config["output_dir"], detector_name, base_model, tokenizer)
            
            # Push to Hub
            push_to_huggingface(
                output_dir=config["output_dir"],
                repo_id=config["output_repo"],
            )
            
            results[detector_name] = {"status": "success", "metadata": metadata}
            
        except Exception as e:
            print(f"\n  ERROR generating {detector_name}: {e}")
            import traceback
            traceback.print_exc()
            results[detector_name] = {"status": "failed", "error": str(e)}
    
    # Summary
    print("\n" + "="*60)
    print("Generation Summary")
    print("="*60)
    
    for detector_name, result in results.items():
        status = result["status"]
        if status == "success":
            repo = DETECTORS[detector_name]["output_repo"]
            print(f"  {detector_name}: ✓ https://huggingface.co/{repo}")
        else:
            print(f"  {detector_name}: ✗ {result.get('error', 'Unknown error')}")
    
    failed = [k for k, v in results.items() if v["status"] != "success"]
    if len(failed) == len(results):
        print("\nAll adapters failed!")
        sys.exit(1)
    elif failed:
        print(f"\n{len(failed)}/{len(results)} adapters failed: {failed}")

    print("\n" + "="*60)
    print("Phase 3 Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
