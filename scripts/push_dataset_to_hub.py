#!/usr/bin/env python3
"""
Push the local JTBD classification dataset to Hugging Face Hub.

Usage:
    uv run scripts/push_dataset_to_hub.py --repo YOUR_HF_USERNAME/jtbd-classification

The script loads data/processed/{train,val,test}.jsonl and pushes them as a
DatasetDict with train/val/test splits. The repo is created as private by default.

Prerequisites:
    - Run scripts/prepare_dataset.py first to generate the JSONL files
    - Set HF_TOKEN in .env or environment (token needs write permissions)
"""

# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "datasets>=2.14.0",
#     "huggingface-hub>=0.20.0",
#     "python-dotenv",
# ]
# ///

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def main() -> None:
    parser = argparse.ArgumentParser(description="Push JTBD dataset to Hugging Face Hub")
    parser.add_argument(
        "--repo",
        required=True,
        help="Hub repo ID, e.g. yourname/jtbd-classification",
    )
    parser.add_argument(
        "--data-dir",
        default="data/processed",
        help="Directory containing train.jsonl, val.jsonl, test.jsonl (default: data/processed)",
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Make the dataset repo public (default: private)",
    )
    args = parser.parse_args()

    # Resolve paths relative to repo root (script may be called from anywhere)
    repo_root = Path(__file__).parent.parent
    data_dir = repo_root / args.data_dir

    for split in ("train", "val", "test"):
        p = data_dir / f"{split}.jsonl"
        if not p.exists():
            sys.exit(
                f"Missing {p}\n"
                "Run scripts/prepare_dataset.py first to generate the JSONL files."
            )

    sys.path.insert(0, str(repo_root / "src"))
    from data_utils import load_splits, to_hf_dataset  # noqa: E402

    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN not set. Add it to .env or export it as an environment variable.")

    print(f"Loading splits from {data_dir} ...")
    splits = load_splits(data_dir)
    for name, examples in splits.items():
        print(f"  {name}: {len(examples)} examples")

    from datasets import DatasetDict

    dataset_dict = DatasetDict(
        {name: to_hf_dataset(examples) for name, examples in splits.items()}
    )
    print(f"\nDatasetDict:\n{dataset_dict}")

    private = not args.public
    print(f"\nPushing to Hub: {args.repo}  (private={private}) ...")
    dataset_dict.push_to_hub(
        args.repo,
        token=token,
        private=private,
    )
    print(f"\nDone! Dataset available at: https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
