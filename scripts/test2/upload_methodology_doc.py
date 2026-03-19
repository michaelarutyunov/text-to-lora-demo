#!/usr/bin/env python3
"""
Upload JTBD methodology document to HuggingFace dataset repository.

This script uploads the methodology prose document to the dataset repo,
making it accessible to the D2L generation script via hf_hub_download().

Usage:
    export HF_TOKEN=your_token
    python scripts/test2/upload_methodology_doc.py
"""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import HfApi

# ── Configuration ─────────────────────────────────────────────────────────────

HF_TOKEN = os.environ.get("HF_TOKEN")
DATASET_REPO = "michaelarutyunov/test2-binary-detectors"
METHODOLOGY_FILE = Path("data/test2/jtbd_methodology_v2_prose.md")
METHODOLOGY_REPO_PATH = "jtbd_methodology_v2_prose.md"

# ── Validation ─────────────────────────────────────────────────────────────────

if not METHODOLOGY_FILE.exists():
    raise FileNotFoundError(
        f"Methodology file not found: {METHODOLOGY_FILE}\n"
        "Please ensure the prose methodology document exists before uploading."
    )

if not HF_TOKEN:
    raise ValueError(
        "HF_TOKEN environment variable not set.\n"
        "Please set it with: export HF_TOKEN=your_token"
    )

# ── Upload Methodology Document ───────────────────────────────────────────────

api = HfApi(token=HF_TOKEN)

# Create repo if it doesn't exist
api.create_repo(repo_id=DATASET_REPO, repo_type="dataset", exist_ok=True)

print(f"Uploading methodology document to {DATASET_REPO}...")
print(f"  Local:  {METHODOLOGY_FILE}")
print(f"  Remote: {METHODOLOGY_REPO_PATH}")

api.upload_file(
    repo_id=DATASET_REPO,
    repo_type="dataset",
    path_in_repo=METHODOLOGY_REPO_PATH,
    path_or_fileobj=str(METHODOLOGY_FILE),
)

print(f"✓ Methodology document uploaded successfully!")
print(f"\nTo download in other scripts:")
print(f"  from huggingface_hub import hf_hub_download")
print(f"  path = hf_hub_download(")
print(f"      repo_id='{DATASET_REPO}',")
print(f"      filename='{METHODOLOGY_REPO_PATH}'")
print(f"  )")
