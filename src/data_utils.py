"""
data_utils.py
─────────────
Shared utilities for loading and formatting the JTBD classification dataset.
Imported by all three notebooks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

# ── Label schema ──────────────────────────────────────────────────────────────

NODE_TYPES: list[str] = [
    "pain_point",
    "gain_point",
    "emotional_job",
    "solution_approach",
    "job_statement",
    "job_context",
    "job_trigger",
    "social_job",
]

LABEL2ID: dict[str, int] = {nt: i for i, nt in enumerate(NODE_TYPES)}
ID2LABEL: dict[int, str] = {i: nt for nt, i in LABEL2ID.items()}

# Human-readable descriptions for T2L task description prompt
NODE_TYPE_DESCRIPTIONS: dict[str, str] = {
    "pain_point":        "A frustration, obstacle, or negative experience the person encounters",
    "gain_point":        "A benefit, positive outcome, or value the person wants to achieve",
    "emotional_job":     "An emotional state the person wants to feel or avoid",
    "solution_approach": "A current workaround, tool, or strategy the person uses",
    "job_statement":     "A high-level functional goal expressed as a job-to-be-done",
    "job_context":       "Situational context or circumstances that frame when/where a job arises",
    "job_trigger":       "An event or condition that prompts the person to seek a solution",
    "social_job":        "A social impression or identity goal the person wants to achieve",
}

# ── T2L task description ──────────────────────────────────────────────────────

T2L_TASK_DESCRIPTION = (
    "Classify a quote from a consumer interview into one of eight JTBD node types. "
    "The quote is a verbatim span from the interviewee's response. "
    "The full utterance providing context is also given. "
    "Node types: "
    + "; ".join(f"{nt} ({NODE_TYPE_DESCRIPTIONS[nt]})" for nt in NODE_TYPES)
    + ". Output only the node type label, nothing else."
)


# ── Prompt formatting ─────────────────────────────────────────────────────────

PROMPT_TEMPLATE = (
    "[CONTEXT]: {context}\n"
    "[QUOTE]: {quote}\n"
    "Classify the quote into one of: {labels}\n"
    "Answer:"
)


def format_prompt(example: dict, include_answer: bool = False) -> str:
    """
    Format a single example as a classification prompt.

    Args:
        example:        dict with keys 'quote', 'context', 'node_type'
        include_answer: if True, append the label (for SFT training)

    Returns:
        Formatted prompt string.
    """
    labels_str = " | ".join(NODE_TYPES)
    context = example.get("context", "").strip() or "[no context available]"
    prompt = PROMPT_TEMPLATE.format(
        context=context,
        quote=example["quote"].strip(),
        labels=labels_str,
    )
    if include_answer:
        prompt += f" {example['node_type']}"
    return prompt


# ── Dataset loading ───────────────────────────────────────────────────────────

def load_jsonl(path: str | Path) -> list[dict]:
    """Load a JSONL file into a list of dicts."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {path}\n"
            "Run scripts/prepare_dataset.py first (locally) and push to GitHub."
        )
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def load_splits(
    data_dir: str | Path,
    splits: Optional[list[str]] = None,
) -> dict[str, list[dict]]:
    """
    Load train/val/test splits from a directory of JSONL files.

    Args:
        data_dir: path to directory containing train.jsonl, val.jsonl, test.jsonl
        splits:   which splits to load (default: all three)

    Returns:
        Dict mapping split name → list of example dicts.
    """
    if splits is None:
        splits = ["train", "val", "test"]
    data_dir = Path(data_dir)
    return {split: load_jsonl(data_dir / f"{split}.jsonl") for split in splits}


def to_hf_dataset(examples: list[dict]):
    """
    Convert a list of example dicts to a HuggingFace Dataset.
    Requires `datasets` to be installed.

    Adds 'label' (int) and 'prompt' (formatted string) columns.
    """
    from datasets import Dataset  # lazy import — not needed for all uses

    records = []
    for ex in examples:
        records.append(
            {
                "quote": ex["quote"],
                "context": ex.get("context", ""),
                "node_type": ex["node_type"],
                "label": LABEL2ID[ex["node_type"]],
                "prompt": format_prompt(ex, include_answer=False),
                "prompt_with_answer": format_prompt(ex, include_answer=True),
                "source_file": ex.get("source_file", ""),
                "node_id": ex.get("node_id", ""),
            }
        )
    return Dataset.from_list(records)


# ── Evaluation helpers ────────────────────────────────────────────────────────

def extract_label_from_response(response: str) -> str:
    """
    Parse the model's text output to extract the predicted node type label.
    Handles common formatting variations (casing, trailing punctuation, etc.).

    Returns the matched NODE_TYPES string, or 'unknown' if no match found.
    """
    response = response.strip().lower()
    # Strip common artifacts
    for ch in [".", ",", ":", ";", "'", '"', "`"]:
        response = response.replace(ch, "")
    response = response.strip()

    # Direct match
    if response in NODE_TYPES:
        return response

    # Partial / substring match (handles "The answer is pain_point" etc.)
    for nt in NODE_TYPES:
        if nt in response:
            return nt

    return "unknown"
