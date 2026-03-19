#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "transformers==4.51.3",
#   "peft>=0.14.0",
#   "torch==2.6.0",
#   "sentence-transformers",
#   "huggingface-hub>=0.20.0",
#   "safetensors",
#   "einops",
#   "jaxtyping",
#   "bitsandbytes>=0.43.0",
#   "accelerate==1.6.0",
#   "datasets>=2.14.0",
# ]
#
# [tool.uv]
# torch-backend = "cu124"
# ///
"""
Generate D2L adapter by internalizing JTBD methodology via Sakana AI's
Doc-to-LoRA hypernetwork. Runs on HuggingFace Jobs (A100 40GB).

This is the ACTUAL D2L method — a single hypernetwork forward pass that
converts a document into LoRA weights. No training loop.

Submitted via Hugging Face Jobs:
    hf jobs submit \\
        --flavor a100-large \\
        --timeout 1h \\
        --secrets HF_TOKEN \\
        uv run scripts/test2/stage_d2l_hf_jobs.py

Environment variables:
    HF_TOKEN        Hugging Face token (for gated Mistral model)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

import torch
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer


# ── Configuration ─────────────────────────────────────────────────────────────

HF_TOKEN = os.environ.get("HF_TOKEN")
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
DATASET_REPO = "michaelarutyunov/test2-binary-detectors"
METHODOLOGY_FILENAME = "jtbd_methodology_v2_prose.md"
OUTPUT_REPO = "michaelarutyunov/jtbd-d2l-mistral7b-methodology"
OUTPUT_DIR = "/tmp/d2l_jtbd_methodology"

D2L_REPO_URL = "https://github.com/SakanaAI/doc-to-lora.git"
D2L_REPO_DIR = "/tmp/doc-to-lora"

# ── Validation Questions ───────────────────────────────────────────────────────

VALIDATION_QUESTIONS = [
    "What is Jobs-to-be-Done (JTBD) methodology?",
    "What's the difference between a job trigger and a job context?",
    "How do you distinguish pain_point from gain_point in JTBD interviews?",
    "What are the 8 node types defined in the JTBD methodology?",
]

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


def download_methodology_document() -> str:
    """Download JTBD methodology document from dataset repo."""
    print(f"Downloading methodology document from {DATASET_REPO}...")

    methodology_path = hf_hub_download(
        repo_id=DATASET_REPO,
        repo_type="dataset",
        filename=METHODOLOGY_FILENAME,
        token=HF_TOKEN,
    )

    print(f"✓ Methodology document downloaded: {methodology_path}")

    with open(methodology_path, "r", encoding="utf-8") as f:
        methodology_text = f.read()

    return methodology_text


def load_d2l_hypernetwork(checkpoint_path: str, device: torch.device):
    """Load D2L hypernetwork from checkpoint."""
    print(f"\nLoading D2L hypernetwork from {checkpoint_path}...")

    # Add D2L repo to path
    sys.path.insert(0, D2L_REPO_DIR + "/src")

    from ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel

    # Load checkpoint
    state_dict = torch.load(checkpoint_path, weights_only=False, map_location=device)

    # Validate checkpoint format
    required_keys = ["hypernet_config", "ctx_encoder_args", "base_model_name_or_path"]
    missing_keys = [k for k in required_keys if k not in state_dict]
    if missing_keys:
        raise RuntimeError(
            f"Invalid D2L checkpoint format. Missing keys: {missing_keys}\n"
            f"Got keys: {list(state_dict.keys())}"
        )

    print(f"✓ Checkpoint loaded successfully")
    print(f"  - Base model: {state_dict['base_model_name_or_path']}")
    print(f"  - Target modules: {state_dict['hypernet_config'].lora_config.target_modules}")
    print(f"  - LoRA rank: {state_dict['hypernet_config'].lora_config.r}")

    # Load model
    model = ModulatedPretrainedModel.from_state_dict(
        state_dict,
        train=False,  # inference mode
        use_sequence_packing=False,
        base_model_kwargs={"output_hidden_states": True},  # REQUIRED for context encoder
        use_flash_attn=False,  # True requires flash-attn wheel matching CUDA version; False is safer on HF Jobs
    )

    model.eval()
    print(f"✓ D2L hypernetwork loaded")

    return model, state_dict


def extract_and_save_lora_weights(model, methodology_text: str, output_dir: str, state_dict: dict):
    """
    Extract generated LoRA weights from model.generated_loras
    and save in PEFT format.
    """
    print(f"\nExtracting LoRA weights...")

    # Internalize methodology
    print("  - Internalizing methodology document...")
    model.internalize(methodology_text)

    # Extract weights
    loras = model.generated_loras

    if loras is None:
        raise RuntimeError(
            "No generated LoRA weights found. "
            "Did internalize() complete successfully?"
        )

    print(f"  - Generated LoRA keys: {list(loras.keys())}")

    # Get layer count
    target_module = list(loras.keys())[0]  # Should be "down_proj"
    n_layers = loras[target_module]["A"].shape[1]  # [batch, n_layers, r, d_in]
    print(f"  - Number of layers: {n_layers}")

    # Build PEFT state dict
    peft_state_dict = {}

    for layer_idx in range(n_layers):
        # Extract down_proj weights for this layer (batch index 0)
        lora_A = loras[target_module]["A"][0, layer_idx]  # [r, d_in]
        lora_B = loras[target_module]["B"][0, layer_idx]  # [r, d_out]

        # PEFT naming convention for Mistral-7B
        # Format: base_model.model.model.layers.{i}.mlp.{module}.lora_{A|B}.weight
        # NOTE: Do NOT embed adapter name (e.g. ".default.") in key paths —
        # PEFT >=0.14 strips adapter names during loading, causing silent key mismatch
        # NOTE: D2L outputs lora_B as [r, d_out] but PEFT expects [d_out, r] — transpose
        peft_state_dict[
            f"base_model.model.model.layers.{layer_idx}.mlp.down_proj.lora_A.weight"
        ] = lora_A.cpu()
        peft_state_dict[
            f"base_model.model.model.layers.{layer_idx}.mlp.down_proj.lora_B.weight"
        ] = lora_B.t().contiguous().cpu()

    print(f"  - Extracted {len(peft_state_dict)} weight tensors")

    # Read adapter config — try model attribute first, fall back to state_dict
    ckpt_lora_config = None
    # Try model's hypernet_config (most reliable after from_state_dict)
    if hasattr(model, 'hypernet_config') and hasattr(model.hypernet_config, 'lora_config'):
        ckpt_lora_config = model.hypernet_config.lora_config
    # Try state_dict (may be consumed by from_state_dict)
    elif "hypernet_config" in state_dict and hasattr(state_dict["hypernet_config"], 'lora_config'):
        ckpt_lora_config = state_dict["hypernet_config"].lora_config

    if ckpt_lora_config is not None:
        lora_config = {
            "r": ckpt_lora_config.r,
            "lora_alpha": ckpt_lora_config.lora_alpha,
            "target_modules": list(ckpt_lora_config.target_modules),
            "lora_dropout": getattr(ckpt_lora_config, "lora_dropout", 0.0),
            "bias": "none",
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "base_model_name_or_path": BASE_MODEL,
        }
    else:
        # Fallback: use known values from the checkpoint we already printed
        print("  - WARNING: Could not read lora_config from model/state_dict, using defaults")
        lora_config = {
            "r": 8,
            "lora_alpha": 45.254833995939045,
            "target_modules": ["down_proj"],
            "lora_dropout": 0.0,
            "bias": "none",
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "base_model_name_or_path": BASE_MODEL,
        }

    # Add metadata for reproducibility
    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "d2l_checkpoint": "SakanaAI/doc-to-lora (mistral_7b_d2l/checkpoint-20000)",
        "d2l_repo_commit": get_git_commit(D2L_REPO_DIR),
        "methodology_doc_hash": hashlib.sha256(methodology_text.encode()).hexdigest()[:16],
        "methodology_doc_length": len(methodology_text),
        "base_model": BASE_MODEL,
        "target_modules": lora_config["target_modules"],
        "lora_r": lora_config["r"],
        "lora_alpha": lora_config["lora_alpha"],
        "n_layers": n_layers,
    }

    # Save adapter
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nSaving adapter to {output_dir}...")

    # Save weights
    save_file(peft_state_dict, f"{output_dir}/adapter_model.safetensors")
    print(f"  ✓ Saved: adapter_model.safetensors")

    # Save config
    with open(f"{output_dir}/adapter_config.json", "w") as f:
        json.dump(lora_config, f, indent=2)
    print(f"  ✓ Saved: adapter_config.json")

    # Save metadata
    with open(f"{output_dir}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✓ Saved: metadata.json")

    # Save README
    readme_content = f"""# D2L Adapter: JTBD Methodology for Mistral-7B-Instruct-v0.2

This adapter was generated using Sakana AI's Doc-to-LoRA (D2L) hypernetwork
by internalizing the JTBD (Jobs-to-be-Done) methodology document.

## Generation Details

- **Method**: D2L hypernetwork forward pass (no training loop)
- **Document**: JTBD methodology v2 prose ({metadata['methodology_doc_length']} chars)
- **Base Model**: {BASE_MODEL}
- **Target Modules**: {metadata['target_modules']}
- **LoRA Rank**: {metadata['lora_r']}
- **LoRA Alpha**: {metadata['lora_alpha']}

## Metadata

```json
{json.dumps(metadata, indent=2)}
```

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    "{BASE_MODEL}",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# Load D2L adapter
model = PeftModel.from_pretrained(
    model,
    "{OUTPUT_REPO}",
    adapter_name="d2l_jtbd",
)

# Generate with internalized JTBD knowledge
# ...
```

## Generated

- {metadata['generated_at']}
- D2L checkpoint: {metadata['d2l_checkpoint']}
- D2L repo commit: {metadata['d2l_repo_commit']}
"""

    with open(f"{output_dir}/README.md", "w") as f:
        f.write(readme_content)
    print(f"  ✓ Saved: README.md")

    return output_dir


def validate_adapter(tokenizer, model_with_d2l, device):
    """Run validation probe to test methodology internalization."""
    print(f"\n" + "="*60)
    print("Validation Probe: Testing JTBD Methodology Internalization")
    print("="*60)

    for i, question in enumerate(VALIDATION_QUESTIONS, 1):
        print(f"\nQuestion {i}: {question}")
        print("-" * 60)

        # Format prompt
        prompt = f"<s>[INST] {question} [/INST]"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # Generate response
        with torch.no_grad():
            outputs = model_with_d2l.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode response
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        print(f"Response: {response[:200]}")
        if len(response) > 200:
            print(f"... (truncated, total {len(response)} chars)")

    print(f"\n" + "="*60)
    print("Validation probe complete. Please review responses above.")
    print("="*60)


def push_to_huggingface(output_dir: str):
    """Push generated adapter to HuggingFace Hub."""
    print(f"\nPushing adapter to {OUTPUT_REPO}...")

    # Create a new repo or update existing
    from huggingface_hub import HfApi

    api = HfApi(token=HF_TOKEN)

    # Create repo if it doesn't exist
    api.create_repo(repo_id=OUTPUT_REPO, exist_ok=True)

    # Upload each file
    files_to_upload = [
        "adapter_model.safetensors",
        "adapter_config.json",
        "README.md",
        "metadata.json",
    ]

    for filename in files_to_upload:
        file_path = os.path.join(output_dir, filename)
        print(f"  Uploading {filename}...")
        api.upload_file(
            repo_id=OUTPUT_REPO,
            path_in_repo=filename,
            path_or_fileobj=file_path,
        )
        print(f"    ✓ Uploaded: {filename}")

    print(f"\n✓ Adapter pushed to: https://huggingface.co/{OUTPUT_REPO}")


# ── Flash Attention Setup ─────────────────────────────────────────────────────


def _ensure_flash_attn():
    """
    Ensure flash-attn is available for D2L's Idefics2Perceiver.

    The D2L repo's custom idefics2.py ONLY supports flash_attention_2 —
    there is no eager/sdpa fallback. Strategy:
      1. Try importing flash_attn (already installed → done)
      2. Try pip install flash-attn (multiple approaches)
      3. If all fail, patch D2L source to add eager attention support
    """
    try:
        import flash_attn  # noqa: F401
        print("✓ flash-attn already available")
        return
    except ImportError:
        pass

    # Detect CUDA and PyTorch versions for pre-built wheel selection
    cuda_version = ""
    torch_version = ""
    python_version = ""
    try:
        result = subprocess.run(["python3", "-c",
            "import torch; print(torch.version.cuda); print(torch.__version__); "
            "import sys; print(f'{sys.version_info.major}{sys.version_info.minor}')"],
            capture_output=True, text=True)
        lines = result.stdout.strip().split("\n")
        if len(lines) >= 3:
            cuda_version = lines[0]  # e.g. "12.4"
            torch_version = lines[1]  # e.g. "2.6.0"
            python_version = lines[2]  # e.g. "312"
        print(f"  CUDA: {cuda_version}, PyTorch: {torch_version}, Python: {python_version}")
    except Exception as e:
        print(f"  Could not detect versions: {e}")

    # uv environments don't include pip. Use `uv pip` to install into the active env,
    # falling back to system pip if uv isn't available.
    import shutil
    uv_path = shutil.which("uv")
    if uv_path:
        pip_cmd = ["uv", "pip", "install", "--python", sys.executable]
    else:
        pip_cmd = [sys.executable, "-m", "pip", "install"]

    # Ensure build tools are available
    subprocess.run(pip_cmd + ["setuptools", "wheel", "packaging", "ninja"],
                   capture_output=True, text=True)

    # Try multiple install strategies
    install_cmds = [
        # Strategy 1: with no-build-isolation (if source build needed)
        pip_cmd + ["flash-attn", "--no-build-isolation"],
        # Strategy 2: standard install
        pip_cmd + ["flash-attn"],
    ]

    # Strategy 0: try pre-built wheel from flash-attn GitHub releases
    # Naming convention: flash_attn-{ver}+cu12torch{major.minor}cxx11abi{TRUE|FALSE}-cp{py}-cp{py}-linux_x86_64.whl
    if torch_version and python_version:
        torch_short = ".".join(torch_version.split(".")[:2])  # "2.6"
        print(f"  Looking for pre-built wheel: cu12, torch{torch_short}, cp{python_version}")
        # v2.7.4.post1 is the exact version from D2L's install.sh
        # cxx11abiFALSE first — matches torch installed with --torch-backend=cu124
        for fa_ver in ["2.7.4.post1", "2.8.3", "2.7.3"]:
            for abi in ["FALSE", "TRUE"]:
                whl_name = f"flash_attn-{fa_ver}+cu12torch{torch_short}cxx11abi{abi}-cp{python_version}-cp{python_version}-linux_x86_64.whl"
                whl_url = f"https://github.com/Dao-AILab/flash-attention/releases/download/v{fa_ver}/{whl_name}"
                install_cmds.insert(0, pip_cmd + [whl_url])

    for cmd in install_cmds:
        cmd_str = " ".join(cmd)
        # Truncate long URLs for display
        display = cmd_str if len(cmd_str) < 120 else cmd_str[:60] + "..." + cmd_str[-50:]
        print(f"  Trying: {display}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        if result.returncode == 0:
            try:
                import importlib
                # Force re-import in case a previous attempt left a broken module
                if "flash_attn" in sys.modules:
                    del sys.modules["flash_attn"]
                import flash_attn  # noqa: F401, F811
                print(f"✓ flash-attn installed successfully (version {flash_attn.__version__})")
                return
            except Exception as e:
                print(f"  Install succeeded but import fails: {type(e).__name__}: {e}")
                continue
        # Show last part of error
        err = (result.stderr or result.stdout)[-200:]
        print(f"  Failed: {err}")

    print("  All flash-attn install strategies failed.")
    print("  Falling back to patching D2L source for eager attention...")
    _patch_d2l_eager_attention()
    _monkey_patch_transformers_flash_check()


def _patch_d2l_eager_attention():
    """
    Patch the D2L repo's custom idefics2.py to support eager attention.

    Changes:
    1. Uncomment "eager": Idefics2PerceiverAttention in IDEFICS2_PERCEIVER_ATTENTION_CLASSES
    2. Remove the assert _use_flash_attention_2 in Idefics2PerceiverResampler
    3. Force config._attn_implementation = "eager" in ModulatedPretrainedModel
    """
    import re

    # ── Patch 1: idefics2.py ──
    idefics2_path = os.path.join(D2L_REPO_DIR, "src/ctx_to_lora/modeling/idefics2.py")
    with open(idefics2_path, "r") as f:
        content = f.read()

    original = content

    # Inject pure-Python shims for flash_attn utilities used in forward pass
    # unpad_input/pad_input are tensor ops, not CUDA kernels
    shim_code = '''
# ── flash_attn shims (patched: flash_attn not installed) ──
try:
    from flash_attn.bert_padding import unpad_input, pad_input
except ImportError:
    import torch as _torch

    def unpad_input(hidden_states, attention_mask):
        seqlens_in_batch = attention_mask.sum(dim=-1, dtype=_torch.int32)
        indices = _torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
        max_seqlen_in_batch = seqlens_in_batch.max().item()
        cu_seqlens = _torch.nn.functional.pad(
            _torch.cumsum(seqlens_in_batch, dim=0, dtype=_torch.int32), (1, 0)
        )
        return (
            hidden_states.reshape(-1, hidden_states.shape[-1])[indices],
            indices,
            cu_seqlens,
            max_seqlen_in_batch,
            seqlens_in_batch,
        )

    def pad_input(hidden_states, indices, batch, seqlen):
        output = _torch.zeros(
            batch * seqlen, *hidden_states.shape[1:],
            device=hidden_states.device, dtype=hidden_states.dtype,
        )
        output[indices] = hidden_states
        return output.view(batch, seqlen, *hidden_states.shape[1:])
# ── end shims ──
'''

    # Insert shims after the imports section (before any class definitions)
    # Find the first class definition and insert before it
    class_match = re.search(r'^class ', content, re.MULTILINE)
    if class_match:
        content = content[:class_match.start()] + shim_code + "\n" + content[class_match.start():]

    # Append monkey-patch at end of file: make eager attention accept extra kwargs
    # The flash variant's forward() accepts is_cross_attn etc. but eager doesn't
    content += '''
# ── patched: make eager attention forward accept extra kwargs ──
import inspect as _inspect
_orig_eager_forward = Idefics2PerceiverAttention.forward
_eager_params = set(_inspect.signature(_orig_eager_forward).parameters.keys()) - {"self"}
def _patched_eager_forward(self, *args, **kwargs):
    filtered = {k: v for k, v in kwargs.items() if k in _eager_params}
    return _orig_eager_forward(self, *args, **filtered)
Idefics2PerceiverAttention.forward = _patched_eager_forward
'''

    # Uncomment the eager attention class — handle varying comment styles
    content = re.sub(
        r'#\s*"eager"\s*:\s*Idefics2PerceiverAttention\s*,?',
        '"eager": Idefics2PerceiverAttention,',
        content,
    )

    # If eager isn't in the dict at all (not even commented), inject it
    if '"eager": Idefics2PerceiverAttention' not in content:
        content = re.sub(
            r'(IDEFICS2_PERCEIVER_ATTENTION_CLASSES\s*=\s*\{)',
            r'\1\n    "eager": Idefics2PerceiverAttention,',
            content,
        )

    # Remove/comment out the flash attention assert
    content = re.sub(
        r'(\s+)assert.*_use_flash_attention_2.*\n',
        r'\1# assert removed: eager attention fallback enabled\n',
        content,
    )

    # Force eager on ALL super().__init__() calls that pass a config
    # Multiple classes (Idefics2Perceiver, Idefics2PerceiverResampler) call
    # super().__init__(some_config) which triggers transformers flash_attn check
    for config_arg in ["encoder_config", "decoder_config", "config"]:
        content = re.sub(
            rf'(\s+)(super\(\)\.__init__\({config_arg}\))',
            rf'\1{config_arg}._attn_implementation = "eager"  # patched: no flash_attn\n\1\2',
            content,
        )

    patched_idefics = content != original
    with open(idefics2_path, "w") as f:
        f.write(content)
    print(f"  {'✓' if patched_idefics else '⚠'} Patched {idefics2_path}: "
          f"{'enabled eager attention' if patched_idefics else 'no changes needed or pattern not found'}")

    # ── Patch 2: hypernet.py — force eager attn_implementation ──
    hypernet_path = os.path.join(D2L_REPO_DIR, "src/ctx_to_lora/modeling/hypernet.py")
    with open(hypernet_path, "r") as f:
        content = f.read()

    original = content

    # More flexible pattern: match "model = cls(base_model, hypernet_config, ...)"
    content = re.sub(
        r'(\s+)(model\s*=\s*cls\(base_model,\s*hypernet_config,\s*ctx_encoder_args.*?\))',
        r"""\1# Force eager attention (flash_attn not available)
\1if hasattr(hypernet_config, 'agg_config') and hasattr(hypernet_config.agg_config, 'config'):
\1    if hasattr(hypernet_config.agg_config.config, '_attn_implementation'):
\1        hypernet_config.agg_config.config._attn_implementation = 'eager'
\1\2""",
        content,
        count=1,
    )

    patched_hypernet = content != original
    with open(hypernet_path, "w") as f:
        f.write(content)
    print(f"  {'✓' if patched_hypernet else '⚠'} Patched {hypernet_path}: "
          f"{'forced eager attn_implementation' if patched_hypernet else 'no changes needed or pattern not found'}")

    if not patched_idefics and not patched_hypernet:
        print("  ⚠ WARNING: No patches applied! D2L source structure may have changed.")
        print("  Dumping relevant sections for debugging:")
        _dump_d2l_attention_sections()


def _monkey_patch_transformers_flash_check():
    """
    Global monkey-patch: intercept transformers' flash_attention_2 validation.

    When flash_attn isn't installed, transformers raises ImportError in
    _flash_attn_2_can_dispatch(). We patch this to silently redirect
    any flash_attention_2 config to eager instead.
    """
    import transformers.modeling_utils as tmu

    original_check = tmu.PreTrainedModel._check_and_adjust_attn_implementation

    def _patched_check(self_or_cls, config, *args, **kwargs):
        # Redirect flash_attention_2 → eager before the original check runs
        if hasattr(config, '_attn_implementation') and config._attn_implementation == 'flash_attention_2':
            config._attn_implementation = 'eager'
        return original_check(self_or_cls, config, *args, **kwargs)

    tmu.PreTrainedModel._check_and_adjust_attn_implementation = _patched_check
    print("  ✓ Monkey-patched transformers: flash_attention_2 → eager redirect")


def _dump_d2l_attention_sections():
    """Print relevant code sections to help debug patching failures."""
    for fname in ["idefics2.py", "hypernet.py"]:
        fpath = os.path.join(D2L_REPO_DIR, f"src/ctx_to_lora/modeling/{fname}")
        if not os.path.exists(fpath):
            print(f"    {fname}: FILE NOT FOUND")
            continue
        with open(fpath) as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            if any(kw in line.lower() for kw in ["attention_classes", "flash_attention", "_attn_implementation", "model = cls("]):
                start = max(0, i - 2)
                end = min(len(lines), i + 3)
                print(f"    {fname}:{i+1}: {''.join(lines[start:end])}")


# ── Main Pipeline ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate D2L adapter from JTBD methodology")
    parser.add_argument("--dry-run", action="store_true",
                       help="Validate setup without running HF Jobs")
    args = parser.parse_args()

    if args.dry_run:
        print("Dry-run mode: validating setup...")
        print("  ✓ HF_TOKEN: " + ("set" if HF_TOKEN else "NOT SET"))
        print("  ✓ Methodology file: " + ("exists" if os.path.exists("data/test2/jtbd_methodology_v2_prose.md") else "NOT FOUND"))
        print("\nTo run on HF Jobs:")
        print("  hf jobs submit --flavor a100-large --timeout 1h --secrets HF_TOKEN \\")
        print("      uv run scripts/test2/stage_d2l_hf_jobs.py")
        return

    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        # Step 1: Clone D2L repo
        if not os.path.exists(D2L_REPO_DIR):
            print(f"\nCloning D2L repository...")
            subprocess.run(
                ["git", "clone", "--depth", "1", D2L_REPO_URL, D2L_REPO_DIR],
                check=True,
            )
            print(f"✓ D2L repo cloned to {D2L_REPO_DIR}")
        else:
            print(f"✓ D2L repo already exists at {D2L_REPO_DIR}")

        # Step 1b: Ensure flash-attn is available (required by D2L's perceiver)
        # The D2L repo's custom Idefics2Perceiver ONLY supports flash_attention_2
        # (no eager/sdpa fallback). Try pip install; if it fails, patch the source.
        _ensure_flash_attn()

        # Step 2: Download D2L checkpoint
        print(f"\nDownloading D2L checkpoint...")
        ckpt_dir = snapshot_download(
            repo_id="SakanaAI/doc-to-lora",
            allow_patterns="mistral_7b_d2l/checkpoint-20000/*",
            token=HF_TOKEN,
        )
        checkpoint_path = f"{ckpt_dir}/mistral_7b_d2l/checkpoint-20000/pytorch_model.bin"
        print(f"✓ Checkpoint downloaded: {checkpoint_path}")

        # Step 3: Download methodology document
        methodology_text = download_methodology_document()
        print(f"  Methodology document length: {len(methodology_text)} characters")

        # Step 4: Load base model
        print(f"\nLoading base model: {BASE_MODEL}")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=HF_TOKEN)
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.bfloat16,
            device_map=device,
            token=HF_TOKEN,
        )
        base_model.eval()
        print(f"✓ Base model loaded")

        # Step 5: Load D2L hypernetwork
        d2l_model, state_dict = load_d2l_hypernetwork(checkpoint_path, device)

        # Step 6: Generate and save adapter
        extract_and_save_lora_weights(
            d2l_model, methodology_text, OUTPUT_DIR, state_dict
        )

        # Step 7: Validate adapter loading
        print(f"\nValidating adapter can be loaded...")
        # Debug: verify config file contents
        config_path = f"{OUTPUT_DIR}/adapter_config.json"
        with open(config_path) as f:
            saved_config = json.load(f)
        print(f"  adapter_config.json contents: {json.dumps(saved_config, indent=2)}")
        assert "peft_type" in saved_config, f"peft_type missing from config! Keys: {list(saved_config.keys())}"

        from peft import PeftModel, LoraConfig as PeftLoraConfig
        import peft
        print(f"  PEFT version: {peft.__version__}")

        # Use LoraConfig directly instead of relying on from_pretrained auto-detection
        peft_config = PeftLoraConfig(
            r=saved_config["r"],
            lora_alpha=saved_config["lora_alpha"],
            target_modules=saved_config["target_modules"],
            lora_dropout=saved_config.get("lora_dropout", 0.0),
            bias=saved_config.get("bias", "none"),
            task_type=saved_config.get("task_type", "CAUSAL_LM"),
        )

        sys.stdout.flush()
        try:
            model_with_adapter = PeftModel.from_pretrained(
                base_model,
                OUTPUT_DIR,
            )
        except KeyError:
            print("  from_pretrained auto-detect failed, loading with explicit config...")
            model_with_adapter = PeftModel(base_model, peft_config)
            # Load weights manually
            from safetensors.torch import load_file
            adapter_weights = load_file(f"{OUTPUT_DIR}/adapter_model.safetensors")
            model_with_adapter.load_state_dict(adapter_weights, strict=False)
        print(f"✓ Adapter successfully loaded back into model")

        # Step 8: Run validation probe
        validate_adapter(tokenizer, model_with_adapter, device)

        # Step 9: Push to HuggingFace
        push_to_huggingface(OUTPUT_DIR)

        print(f"\n{'='*60}")
        print("D2L Adapter Generation Complete!")
        print(f"{'='*60}")
        print(f"\nAdapter saved to: {OUTPUT_DIR}")
        print(f"Pushed to: https://huggingface.co/{OUTPUT_REPO}")
        print(f"\nTo use this adapter:")
        print(f"  from peft import PeftModel")
        print(f"  model = PeftModel.from_pretrained(base_model, '{OUTPUT_REPO}')")

    except Exception as e:
        print(f"\n❌ Error during D2L adapter generation:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
