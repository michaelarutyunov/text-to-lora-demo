"""
prepare_dataset.py
──────────────────
Reads JTBD interview JSON files from interview-system-v2, extracts
quote-level classification examples, and writes stratified train/val/test
splits to data/processed/.

Usage (from repo root in WSL):
    python scripts/prepare_dataset.py
    python scripts/prepare_dataset.py --data-dir /path/to/synthetic_interviews
    python scripts/prepare_dataset.py --data-dir /path/to/dir --recurse

Output files (JSONL, one example per line):
    data/processed/train.jsonl
    data/processed/val.jsonl
    data/processed/test.jsonl

Each record:
    {
        "quote":       str,   # verbatim span from utterance
        "context":     str,   # full response utterance the quote comes from
        "node_type":   str,   # classification label
        "source_file": str,   # interview filename (no path)
        "node_id":     str    # node id for traceability
    }
"""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

from sklearn.model_selection import train_test_split

# ── Constants ─────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = REPO_ROOT.parent / "interview-system-v2" / "synthetic_interviews"
OUTPUT_DIR = REPO_ROOT / "data" / "processed"

# Canonical label set — social_job exists in some files as an 8th type.
# We include it but flag it as low-resource in the report.
NODE_TYPES = [
    "pain_point",
    "gain_point",
    "emotional_job",
    "solution_approach",
    "job_statement",
    "job_context",
    "job_trigger",
    "social_job",
]

SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
RANDOM_SEED = 42


# ── Helpers ───────────────────────────────────────────────────────────────────


def is_jtbd(metadata: dict) -> bool:
    """Return True if the interview uses a JTBD methodology."""
    methodology = metadata.get("methodology", "")
    return methodology.lower().startswith("jobs_to_be_done")


def find_interview_files(data_dir: Path, recurse: bool) -> list[Path]:
    """Collect all .json files from data_dir, optionally recursing into subdirs."""
    pattern = "**/*.json" if recurse else "*.json"
    return sorted(data_dir.glob(pattern))


def extract_examples(interview_path: Path) -> list[dict]:
    """
    Extract quote-level classification examples from one interview JSON.

    Handles two layouts:
      - graph.nodes[].source_quotes   (list of verbatim spans)
      - turns[].nodes_added[]         (cross-reference for utterance context)

    Returns a list of dicts with keys: quote, context, node_type,
    source_file, node_id.
    """
    with open(interview_path, encoding="utf-8") as f:
        data = json.load(f)

    if not is_jtbd(data.get("metadata", {})):
        return []

    # Build utterance_id → response text lookup from turns
    utterance_map: dict[str, str] = {}
    for turn in data.get("turns", []):
        for uid in turn.get("utterance_ids", []):
            utterance_map[uid] = turn.get("response", "")
        # Also index by turn index if utterance_ids absent
        turn_idx = str(turn.get("turn_index", ""))
        if turn_idx and turn_idx not in utterance_map:
            utterance_map[turn_idx] = turn.get("response", "")

    examples = []
    nodes = data.get("graph", {}).get("nodes", [])

    for node in nodes:
        node_type = node.get("node_type", "").lower()
        if node_type not in NODE_TYPES:
            continue  # skip unknown types (shouldn't happen after enum above)

        node_id = node.get("id", "")
        source_quotes = node.get("source_quotes", [])
        source_utt_ids = node.get("source_utterance_ids", [])

        # Resolve context: try utterance_ids first, fall back to empty string
        context = ""
        for uid in source_utt_ids:
            ctx = utterance_map.get(str(uid), "")
            if ctx:
                context = ctx
                break

        # One example per source quote
        for quote in source_quotes:
            if not quote.strip():
                continue
            examples.append(
                {
                    "quote": quote.strip(),
                    "context": context.strip(),
                    "node_type": node_type,
                    "source_file": interview_path.name,
                    "node_id": node_id,
                }
            )

        # If no source_quotes but node has a label, use the label as a fallback
        # (preserves examples for sparse classes even without verbatim spans)
        if not source_quotes:
            label_text = node.get("label", "").strip()
            if label_text:
                examples.append(
                    {
                        "quote": label_text,
                        "context": context.strip(),
                        "node_type": node_type,
                        "source_file": interview_path.name,
                        "node_id": node_id,
                    }
                )

    return examples


def stratified_split(
    examples: list[dict],
    ratios: dict = SPLIT_RATIOS,
    seed: int = RANDOM_SEED,
) -> dict[str, list[dict]]:
    """
    Stratified train/val/test split by node_type.

    Classes with fewer than 3 examples are lumped into train only
    to avoid empty strata in small splits.
    """
    labels = [e["node_type"] for e in examples]
    label_counts = Counter(labels)

    # Separate low-resource examples (can't stratify with < 3 per class)
    min_for_split = 3
    low_res = [e for e in examples if label_counts[e["node_type"]] < min_for_split]
    splittable = [e for e in examples if label_counts[e["node_type"]] >= min_for_split]

    if not splittable:
        return {"train": examples, "val": [], "test": []}

    splittable_labels = [e["node_type"] for e in splittable]
    val_test_size = ratios["val"] + ratios["test"]
    test_fraction_of_remainder = ratios["test"] / val_test_size

    train_ex, val_test_ex = train_test_split(
        splittable,
        test_size=val_test_size,
        stratify=splittable_labels,
        random_state=seed,
    )
    val_test_labels = [e["node_type"] for e in val_test_ex]
    val_ex, test_ex = train_test_split(
        val_test_ex,
        test_size=test_fraction_of_remainder,
        stratify=val_test_labels,
        random_state=seed,
    )

    # Low-resource examples go to train only
    train_ex = train_ex + low_res

    return {"train": train_ex, "val": val_ex, "test": test_ex}


def write_jsonl(examples: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def print_report(examples: list[dict], splits: dict[str, list[dict]]) -> None:
    total = len(examples)
    print(f"\n{'='*60}")
    print(f"DATASET SUMMARY  ({total} total examples)")
    print(f"{'='*60}")

    type_counts = Counter(e["node_type"] for e in examples)
    print("\nNode type distribution:")
    for nt in NODE_TYPES:
        c = type_counts.get(nt, 0)
        bar = "█" * (c // 5)
        flag = " ⚠ low-resource" if c < 20 else ""
        print(f"  {nt:<22} {c:4d}  {c/total*100:5.1f}%  {bar}{flag}")

    print(f"\nSplit sizes:")
    for split_name, split_ex in splits.items():
        sc = Counter(e["node_type"] for e in split_ex)
        print(f"  {split_name:<8} {len(split_ex):4d} examples")
        for nt in NODE_TYPES:
            if sc.get(nt, 0) > 0:
                print(f"           {nt:<22} {sc[nt]}")

    file_counts = Counter(e["source_file"] for e in examples)
    print(f"\nSource files ({len(file_counts)}):")
    for fname, c in sorted(file_counts.items()):
        print(f"  {fname}  ({c} examples)")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Prepare JTBD node classification dataset")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Directory containing interview JSON files (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--recurse",
        action="store_true",
        default=True,
        help="Also scan subdirectories (default: True)",
    )
    parser.add_argument(
        "--no-recurse",
        dest="recurse",
        action="store_false",
        help="Only scan top-level directory",
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
        print("Hint: pass --data-dir /path/to/synthetic_interviews", file=sys.stderr)
        sys.exit(1)

    # Collect and parse
    json_files = find_interview_files(args.data_dir, args.recurse)
    print(f"Found {len(json_files)} JSON files in {args.data_dir}")

    all_examples: list[dict] = []
    skipped = 0
    for jf in json_files:
        try:
            exs = extract_examples(jf)
            all_examples.extend(exs)
            if not exs:
                skipped += 1
        except Exception as e:
            print(f"  WARNING: failed to parse {jf.name}: {e}", file=sys.stderr)
            skipped += 1

    print(f"Skipped {skipped} non-JTBD or unparseable files")
    print(f"Extracted {len(all_examples)} examples from JTBD interviews")

    if not all_examples:
        print("ERROR: no examples extracted. Check --data-dir.", file=sys.stderr)
        sys.exit(1)

    # Split and write
    splits = stratified_split(all_examples, seed=args.seed)
    for split_name, split_examples in splits.items():
        out_path = args.output_dir / f"{split_name}.jsonl"
        write_jsonl(split_examples, out_path)
        print(f"Wrote {len(split_examples):4d} examples → {out_path}")

    print_report(all_examples, splits)
    print("Done. Commit data/processed/ and push to share with Colab.")


if __name__ == "__main__":
    main()
