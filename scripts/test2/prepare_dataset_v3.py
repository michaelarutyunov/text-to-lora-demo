"""
prepare_dataset_v3.py
──────────────────────
Reads JTBD v2 interview JSON files from interview-system-v2, extracts
utterance-level binary labels for multiple node types, and writes
interview-level stratified train/val/test splits.

Phase 1 of Test 2: Parallel binary detectors
Creates 3 independent binary classification datasets, one for each detector:
- job_trigger (34% yes, 66% no)
- solution_approach (64% yes, 36% no)
- pain_point (87% yes, 13% no)

Labels are derived from source_quotes substring matching: a turn is labelled
"yes" if any node of the target type has a source_quote that appears in that
turn's response text.

All three datasets use identical interview-level splits for fair comparison.

Usage (from repo root in WSL):
    python scripts/test2/prepare_dataset_v3.py

Output files (JSONL, one example per line):
    data/test2/job_trigger_train.jsonl
    data/test2/job_trigger_val.jsonl
    data/test2/job_trigger_test.jsonl
    data/test2/solution_approach_train.jsonl
    data/test2/solution_approach_val.jsonl
    data/test2/solution_approach_test.jsonl
    data/test2/pain_point_train.jsonl
    data/test2/pain_point_val.jsonl
    data/test2/pain_point_test.jsonl

Each record:
    {
        "utterance":    str,   # full interviewee response
        "label":        str,   # "yes" or "no"
        "source_file":  str,   # interview filename (no path)
        "turn_number":  int    # turn index within interview
    }
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from sklearn.model_selection import GroupShuffleSplit

# ── Constants ─────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATA_DIR = REPO_ROOT.parent / "interview-system-v2" / "synthetic_interviews"
OUTPUT_DIR = REPO_ROOT / "data" / "test2"

# Three detectors to create binary datasets for
DETECTORS = [
    "job_trigger",
    "solution_approach",
    "pain_point",
]

# Target split ratios for interviews (not utterances)
# 22 train / 4 val / 5 test interviews
TRAIN_RATIO = 0.71
VAL_RATIO = 0.13
TEST_RATIO = 0.16
RANDOM_SEED = 42


# ── Helpers ───────────────────────────────────────────────────────────────────


def is_v2_jtbd(metadata: dict) -> bool:
    """Check if interview uses jobs_to_be_done_v2 methodology."""
    methodology = metadata.get("methodology", "")
    return methodology.lower() == "jobs_to_be_done_v2"


def find_v2_interview_files(data_dir: Path) -> list[Path]:
    """Collect all v2 JTBD interview JSON files."""
    all_files = sorted(data_dir.glob("*.json"))
    v2_files = []
    for f in all_files:
        try:
            with open(f, encoding="utf-8") as fp:
                data = json.load(fp)
                if is_v2_jtbd(data.get("metadata", {})):
                    v2_files.append(f)
        except Exception:
            continue
    return v2_files


def extract_node_quotes_by_type(nodes: list[dict]) -> dict[str, list[str]]:
    """
    Collect all source_quote strings grouped by node_type.

    Returns:
        {node_type: [quote1, quote2, ...]}
    """
    quotes_by_type = defaultdict(list)
    for node in nodes:
        node_type = node.get("node_type", "").lower()
        if node_type not in DETECTORS:
            continue
        for q in node.get("source_quotes") or []:
            if q and q.strip():
                quotes_by_type[node_type].append(q.strip())
    return dict(quotes_by_type)


def turn_has_node_type(response: str, quotes: list[str]) -> bool:
    """
    Check if any quote for this node type appears in the response.

    source_quotes are verbatim spans extracted from interview responses, so
    substring matching is reliable. We match on the first 40 chars to be
    robust against minor trailing whitespace differences.
    """
    if not response:
        return False
    response_lower = response.lower()
    for q in quotes:
        if q[:40].lower() in response_lower:
            return True
    return False


def extract_examples(interview_path: Path) -> list[dict]:
    """
    Extract utterance-level examples from one v2 interview JSON.

    Returns a list of dicts with keys: utterance, source_file, turn_number,
    plus one binary label per detector (has_<node_type>).
    """
    with open(interview_path, encoding="utf-8") as f:
        data = json.load(f)

    if not is_v2_jtbd(data.get("metadata", {})):
        return []

    nodes = (data.get("graph") or {}).get("nodes", [])
    quotes_by_type = extract_node_quotes_by_type(nodes)

    examples = []
    for turn in data.get("turns", []):
        response = (turn.get("response") or "").strip()
        if not response:
            continue

        example = {
            "utterance": response,
            "source_file": interview_path.name,
            "turn_number": turn.get("turn_number", -1),
        }

        # Add binary labels for each detector
        for detector in DETECTORS:
            quotes = quotes_by_type.get(detector, [])
            example[f"has_{detector}"] = turn_has_node_type(response, quotes)

        examples.append(example)

    return examples


def interview_level_split(
    examples: list[dict],
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    seed: int = RANDOM_SEED,
) -> dict[str, list[dict]]:
    """
    Split examples by interview (GroupShuffleSplit on source_file).

    This ensures utterances from the same interview don't leak across splits.
    All three detectors will use the same interview splits for fair comparison.
    """
    # Group examples by interview
    interviews: dict[str, list[dict]] = defaultdict(list)
    for ex in examples:
        interviews[ex["source_file"]].append(ex)

    interview_files = list(interviews.keys())

    # First split: train vs (val + test)
    gss1 = GroupShuffleSplit(
        n_splits=1, test_size=(val_ratio + test_ratio), random_state=seed
    )
    train_idx, val_test_idx = next(gss1.split(interview_files, groups=interview_files))

    # Second split: val vs test
    val_test_files = [interview_files[i] for i in val_test_idx]
    test_size = test_ratio / (val_ratio + test_ratio)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    val_idx, test_idx = next(gss2.split(val_test_files, groups=val_test_files))

    # Collect examples by split
    train_files = [interview_files[i] for i in train_idx]
    val_files = [val_test_files[i] for i in val_idx]
    test_files = [val_test_files[i] for i in test_idx]

    splits = {}
    for split_name, files in [
        ("train", train_files),
        ("val", val_files),
        ("test", test_files),
    ]:
        split_examples = []
        for f in files:
            split_examples.extend(interviews[f])
        splits[split_name] = split_examples

    return splits


def write_jsonl(examples: list[dict], path: Path) -> None:
    """Write examples to JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def print_report(
    all_examples: list[dict],
    splits: dict[str, list[dict]],
) -> None:
    """Print summary statistics for all detectors."""
    total = len(all_examples)
    n_interviews = len(set(e["source_file"] for e in all_examples))

    print(f"\n{'='*70}")
    print(f"DATASET SUMMARY  ({total} utterances from {n_interviews} interviews)")
    print(f"{'='*70}")

    # Interview split
    train_interviews = set(e["source_file"] for e in splits["train"])
    val_interviews = set(e["source_file"] for e in splits["val"])
    test_interviews = set(e["source_file"] for e in splits["test"])

    print(f"\nInterview split:")
    print(f"  Train:  {len(train_interviews):2d} interviews")
    print(f"  Val:    {len(val_interviews):2d} interviews")
    print(f"  Test:   {len(test_interviews):2d} interviews")

    # Utterance split
    print(f"\nUtterance split:")
    for split_name, split_ex in splits.items():
        print(f"  {split_name:<8} {len(split_ex):4d} utterances")

    # Per-detector statistics
    for detector in DETECTORS:
        label_key = f"has_{detector}"
        yes_count = sum(1 for e in all_examples if e[label_key])
        no_count = total - yes_count
        yes_pct = yes_count / total * 100

        print(f"\n{detector.replace('_', ' ').title()}:")
        print(f"  Yes:    {yes_count:4d}  ({yes_pct:5.1f}%)")
        print(f"  No:     {no_count:4d}  ({100-yes_pct:5.1f}%)")

        # Per-split breakdown
        print(f"  Split breakdown:")
        for split_name, split_ex in splits.items():
            split_yes = sum(1 for e in split_ex if e[label_key])
            split_no = len(split_ex) - split_yes
            print(f"    {split_name:<8} {split_yes:3d} yes / {split_no:3d} no")

    print()


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Prepare Test 2 parallel binary detector datasets (v2 interviews only)"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Directory containing interview JSON files (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Output directory for JSONL splits (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed for splits (default: 42)",
    )
    args = parser.parse_args()

    if not args.data_dir.exists():
        print(f"ERROR: data-dir not found: {args.data_dir}", file=sys.stderr)
        sys.exit(1)

    # Find v2 interviews
    v2_files = find_v2_interview_files(args.data_dir)
    print(f"Found {len(v2_files)} v2 JTBD interviews in {args.data_dir}")

    if not v2_files:
        print("ERROR: no v2 JTBD interviews found.", file=sys.stderr)
        sys.exit(1)

    # Extract all examples (with labels for all detectors)
    all_examples = []
    for v2_file in v2_files:
        try:
            exs = extract_examples(v2_file)
            all_examples.extend(exs)
        except Exception as e:
            print(f"  WARNING: failed to parse {v2_file.name}: {e}", file=sys.stderr)

    print(f"Extracted {len(all_examples)} utterances from v2 interviews")

    if not all_examples:
        print("ERROR: no examples extracted.", file=sys.stderr)
        sys.exit(1)

    # Interview-level split (same for all detectors)
    splits = interview_level_split(all_examples, seed=args.seed)

    # Write separate dataset for each detector
    for detector in DETECTORS:
        label_key = f"has_{detector}"
        for split_name, split_examples in splits.items():
            # Convert to simple binary label format
            binary_examples = []
            for ex in split_examples:
                binary_examples.append(
                    {
                        "utterance": ex["utterance"],
                        "label": "yes" if ex[label_key] else "no",
                        "source_file": ex["source_file"],
                        "turn_number": ex["turn_number"],
                    }
                )

            out_path = args.output_dir / f"{detector}_{split_name}.jsonl"
            write_jsonl(binary_examples, out_path)
            print(f"Wrote {len(binary_examples):4d} examples → {out_path}")

    print_report(all_examples, splits)
    print("Done. Commit data/test2/ and push to HuggingFace Hub.")


if __name__ == "__main__":
    main()
