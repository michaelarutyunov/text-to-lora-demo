#!/usr/bin/env python3
"""
Push Test 2 parallel binary detector datasets to Hugging Face Hub.

Creates three separate dataset repos, one for each binary detector:
- job_trigger
- solution_approach
- pain_point

Usage:
    uv run scripts/test2/push_datasets_to_hub.py --repo YOUR_HF_USERNAME/jtbd-binary-{detector}
    # Or push all three at once:
    for det in job_trigger solution_approach pain_point; do
        uv run scripts/test2/push_datasets_to_hub.py --repo YOUR_USERNAME/jtbd-binary-$det --detector $det
    done

The script loads data/test2/{detector}_{train,val,test}.jsonl and pushes them as
a DatasetDict with train/val/test splits. The repo is created as private by default.

Prerequisites:
    - Run scripts/test2/prepare_dataset_v3.py first to generate the JSONL files
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
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Valid detector names
DETECTORS = [
    "job_trigger",
    "solution_approach",
    "pain_point",
]

# Binary label mappings
LABEL2ID = {"no": 0, "yes": 1}
ID2LABEL = {0: "no", 1: "yes"}


def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file into a list of dicts."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def load_binary_detector_data(
    data_dir: Path,
    detector: str,
) -> dict[str, list[dict]]:
    """
    Load train/val/test splits for one binary detector.

    Returns:
        Dict mapping split name → list of example dicts.
    """
    splits = {}
    for split in ("train", "val", "test"):
        path = data_dir / f"{detector}_{split}.jsonl"
        if not path.exists():
            sys.exit(
                f"Missing {path}\n"
                f"Run scripts/test2/prepare_dataset_v3.py first to generate datasets."
            )
        splits[split] = load_jsonl(path)
    return splits


def to_hf_dataset(examples: list[dict]):
    """
    Convert a list of binary classification examples to a HuggingFace Dataset.

    Adds 'label' (int) column where no=0, yes=1.
    """
    from datasets import Dataset

    records = []
    for ex in examples:
        label_int = LABEL2ID[ex["label"]]
        records.append(
            {
                "utterance": ex["utterance"],
                "label": label_int,
                "source_file": ex.get("source_file", ""),
                "turn_number": ex.get("turn_number", -1),
            }
        )
    return Dataset.from_list(records)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Push Test 2 binary detector dataset to Hugging Face Hub"
    )
    parser.add_argument(
        "--repo",
        required=True,
        help="Hub repo ID, e.g. yourname/jtbd-binary-job-trigger",
    )
    parser.add_argument(
        "--detector",
        required=True,
        choices=DETECTORS,
        help="Which detector to push (job_trigger, solution_approach, or pain_point)",
    )
    parser.add_argument(
        "--data-dir",
        default="data/test2",
        help="Directory containing detector JSONL files (default: data/test2)",
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Make the dataset repo public (default: private)",
    )
    args = parser.parse_args()

    # Resolve paths relative to repo root
    repo_root = Path(__file__).parent.parent.parent
    data_dir = repo_root / args.data_dir

    # Load the detector's data
    print(f"Loading {args.detector} data from {data_dir} ...")
    splits = load_binary_detector_data(data_dir, args.detector)
    for name, examples in splits.items():
        yes_count = sum(1 for e in examples if e["label"] == "yes")
        no_count = len(examples) - yes_count
        print(f"  {name}: {len(examples)} examples (yes={yes_count}, no={no_count})")

    # Convert to HuggingFace DatasetDict
    from datasets import DatasetDict

    dataset_dict = DatasetDict(
        {name: to_hf_dataset(examples) for name, examples in splits.items()}
    )
    print(f"\nDatasetDict:\n{dataset_dict}")

    # Add dataset card metadata
    detector_name = args.detector.replace("_", " ").title()
    card = f"""# {detector_name} Binary Detection Dataset

Binary classification dataset for detecting JTBD {args.detector} nodes in consumer interview utterances.

## Dataset Description

This dataset contains utterances from Jobs-to-Be-Done (JTBD) interviews, labeled for whether they contain a **{args.detector}** node.

- **Task**: Binary classification (yes/no)
- **Classes**:
  - `no` (0): Utterance does not contain a {args.detector}
  - `yes` (1): Utterance contains a {args.detector}
- **Data Source**: 27 v2 JTBD interviews from interview-system-v2
- **Split**: Interview-level stratified split (19 train / 3 val / 5 test interviews)

## Labels

Labels are derived from graph node `source_quotes`: a turn is labeled "yes" if any {args.detector} node has a source_quote that appears in that turn's response text.

## Splits

- **train**: {len(splits['train'])} utterances from 19 interviews
- **val**: {len(splits['val'])} utterances from 3 interviews
- **test**: {len(splits['test'])} utterances from 5 interviews

## Data Fields

- `utterance` (string): Full interviewee response text
- `label` (int): Binary label (0=no, 1=yes)
- `source_file` (string): Original interview filename
- `turn_number` (int): Turn index within interview

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("{args.repo}")
train = dataset["train"]
```

## License

Please cite the associated research paper if using this dataset.
"""
    # Save README.md to include as dataset card
    repo_id = args.repo
    card_path = repo_root / "data" / "test2" / f"{args.detector}_README.md"
    card_path.parent.mkdir(parents=True, exist_ok=True)
    card_path.write_text(card, encoding="utf-8")

    dataset_dict.push_to_hub(
        repo_id,
        token=os.environ.get("HF_TOKEN"),
        private=not args.public,
    )

    # Push the README separately
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.upload_file(
        path_or_fileobj=str(card_path),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
    )

    private = not args.public
    print(f"\n✅ Done! Dataset available at: https://huggingface.co/datasets/{args.repo}")
    print(f"   (private={private})")


if __name__ == "__main__":
    main()
