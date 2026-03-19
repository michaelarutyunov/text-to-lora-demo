#!/usr/bin/env python3
"""
Test 2 experiment orchestrator.

Submits all HF Jobs in the correct order, polling for completion before
advancing. Phases 2 (D2L) and 3 (T2L) are submitted in parallel.

Usage:
    python scripts/test2/submit_experiment.py
    python scripts/test2/submit_experiment.py --flavor l40s
    python scripts/test2/submit_experiment.py --flavor a100-large --skip-to qlora
    python scripts/test2/submit_experiment.py --dry-run

Phases:
    d2l       Phase 2: Generate D2L adapter (~30 min)
    t2l       Phase 3: Generate T2L adapters x3 (~35 min)
    validate  Phase 4: LoRA stacking validation (~20 min)
    qlora     Phase 5a: QLoRA training x12 (~3-3.5 hrs)
    evaluate  Phase 5b: Full evaluation pipeline (~1.5-2 hrs)

Dependency graph:
    d2l ──┐
          ├──▶ validate ──▶ qlora ──▶ evaluate
    t2l ──┘

Hardware flavors and prices (per hour):
    t4-small       $0.40   16GB VRAM  (too small for Mistral-7B)
    a10g-small     $1.00   24GB VRAM  (marginal for bfloat16)
    l40s           $1.80   48GB VRAM  (recommended — good value)
    a100-large     $2.50   80GB VRAM  (default in phase scripts)

Environment variables:
    HF_TOKEN       Hugging Face token (required)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

PHASE_ORDER = ["d2l", "t2l", "validate", "qlora", "evaluate"]

# Phases that can run in parallel (submitted simultaneously)
PARALLEL_PHASES = {"d2l", "t2l"}

PHASES: dict[str, dict] = {
    "d2l": {
        "label": "Phase 2: D2L adapter",
        "script": "scripts/test2/stage_d2l_hf_jobs.py",
        "timeout": "1h",
        "est_minutes": 30,
    },
    "t2l": {
        "label": "Phase 3: T2L adapters",
        "script": "scripts/test2/stage_t2l_hf_jobs.py",
        "timeout": "1h",
        "est_minutes": 35,
    },
    "validate": {
        "label": "Phase 4: LoRA stacking validation",
        "script": "scripts/test2/validate_lora_stacking.py",
        "timeout": "30m",
        "est_minutes": 20,
        "depends_on": ["d2l", "t2l"],
    },
    "qlora": {
        "label": "Phase 5a: QLoRA training",
        "script": "scripts/test2/train_qlora_hf_jobs.py",
        "timeout": "8h",
        "est_minutes": 210,
        "depends_on": ["validate"],
    },
    "evaluate": {
        "label": "Phase 5b: Evaluation pipeline",
        "script": "scripts/test2/evaluate_conditions.py",
        "timeout": "4h",
        "est_minutes": 105,
        "depends_on": ["qlora"],
    },
}

POLL_INTERVAL_SECONDS = 30
TERMINAL_STAGES = {"COMPLETED", "ERROR", "CANCELED"}


# ── Job tracking ───────────────────────────────────────────────────────────────


@dataclass
class JobRun:
    phase: str
    job_id: str
    submitted_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    status: str = "PENDING"
    error: str | None = None


# ── HF CLI helpers ─────────────────────────────────────────────────────────────


def submit_job(phase: str, flavor: str, dry_run: bool) -> JobRun | None:
    """Submit one phase as a HF Job and return a JobRun."""
    cfg = PHASES[phase]
    script_path = REPO_ROOT / cfg["script"]

    cmd = [
        "hf", "jobs", "uv", "run",
        "--flavor", flavor,
        "--timeout", cfg["timeout"],
        "--secrets", "HF_TOKEN",
        str(script_path),
    ]

    print(f"\n[{_now()}] Submitting: {cfg['label']}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"  Estimated runtime: ~{cfg['est_minutes']} min")

    if dry_run:
        print("  [DRY RUN] Skipping actual submission.")
        return JobRun(phase=phase, job_id="dry-run-id", status="COMPLETED")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )

    if result.returncode != 0:
        print(f"  ERROR: submission failed\n{result.stderr}", file=sys.stderr)
        return None

    output = result.stdout + result.stderr
    job_id = _parse_job_id(output)

    if not job_id:
        print(f"  ERROR: could not parse job ID from output:\n{output}", file=sys.stderr)
        return None

    print(f"  Submitted: job_id={job_id}")
    print(f"  URL: https://huggingface.co/jobs/michaelarutyunov/{job_id}")
    return JobRun(phase=phase, job_id=job_id)


def poll_job(run: JobRun) -> str:
    """Query job status via `hf jobs inspect`. Returns stage string."""
    result = subprocess.run(
        ["hf", "jobs", "inspect", run.job_id],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return "UNKNOWN"

    try:
        data = json.loads(result.stdout)
        # inspect returns a list with one item
        if isinstance(data, list):
            data = data[0]
        return data.get("status", {}).get("stage", "UNKNOWN")
    except (json.JSONDecodeError, KeyError, IndexError):
        return "UNKNOWN"


def wait_for_jobs(runs: list[JobRun], dry_run: bool) -> bool:
    """
    Poll all jobs until they reach a terminal state.

    Returns True if all completed successfully, False if any failed.
    """
    if dry_run:
        return True

    pending = {r.job_id: r for r in runs}
    last_print: dict[str, str] = {}

    while pending:
        time.sleep(POLL_INTERVAL_SECONDS)

        for job_id, run in list(pending.items()):
            stage = poll_job(run)
            run.status = stage

            if stage != last_print.get(job_id):
                print(f"  [{_now()}] {PHASES[run.phase]['label']}: {stage}")
                last_print[job_id] = stage

            if stage in TERMINAL_STAGES:
                del pending[job_id]

    # Check for failures
    failed = [r for r in runs if r.status != "COMPLETED"]
    if failed:
        for r in failed:
            print(f"\n  FAILED: {PHASES[r.phase]['label']} (status={r.status})")
        return False
    return True


def _parse_job_id(output: str) -> str | None:
    """Extract job ID (24-char hex) from hf jobs output."""
    import re
    # Job IDs are 24-character hex strings
    matches = re.findall(r"\b([0-9a-f]{24})\b", output)
    return matches[0] if matches else None


def _now() -> str:
    return datetime.now().strftime("%H:%M:%S")


# ── Cost estimation ────────────────────────────────────────────────────────────

# Hourly prices from HF Jobs pricing page
FLAVOR_PRICES: dict[str, float] = {
    "t4-small": 0.40,
    "t4-medium": 0.60,
    "l4x1": 0.80,
    "l4x4": 3.80,
    "a10g-small": 1.00,
    "a10g-large": 1.50,
    "a10g-largex2": 3.00,
    "a10g-largex4": 5.00,
    "l40sx1": 1.80,
    "l40sx4": 8.30,
    "l40sx8": 23.50,
    "a100-large": 2.50,
    "a100x4": 10.00,
    "a100x8": 20.00,
}


def print_cost_estimate(flavor: str, phases_to_run: list[str]) -> None:
    price = FLAVOR_PRICES.get(flavor)
    if price is None:
        print(f"  (unknown flavor '{flavor}' — cannot estimate cost)")
        return

    total_minutes = sum(PHASES[p]["est_minutes"] for p in phases_to_run)
    # d2l and t2l overlap
    if "d2l" in phases_to_run and "t2l" in phases_to_run:
        overlap = min(PHASES["d2l"]["est_minutes"], PHASES["t2l"]["est_minutes"])
        total_minutes -= overlap

    total_hours = total_minutes / 60
    cost = total_hours * price

    print(f"\nCost estimate ({flavor} @ ${price:.2f}/hr):")
    for p in phases_to_run:
        mins = PHASES[p]["est_minutes"]
        phase_cost = (mins / 60) * price
        label = PHASES[p]["label"]
        print(f"  {label:<40} ~{mins:3d} min  ~${phase_cost:.2f}")
    if "d2l" in phases_to_run and "t2l" in phases_to_run:
        print(f"  (d2l + t2l run in parallel — overlap deducted)")
    print(f"  {'Total':<40} ~{total_minutes:3d} min  ~${cost:.2f}")


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Test 2 experiment orchestrator — submits HF Jobs in order",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Hardware flavors (GPU VRAM / price per hour):
  l40s         48GB / $1.80   recommended — good value for Mistral-7B
  a100-large   80GB / $2.50   default in phase scripts, most headroom

Examples:
  python scripts/test2/submit_experiment.py
  python scripts/test2/submit_experiment.py --flavor l40s
  python scripts/test2/submit_experiment.py --skip-to qlora
  python scripts/test2/submit_experiment.py --only evaluate
  python scripts/test2/submit_experiment.py --dry-run
        """,
    )
    parser.add_argument(
        "--flavor",
        default="a100-large",
        choices=list(FLAVOR_PRICES.keys()),
        help="HF Jobs hardware flavor (default: a100-large)",
    )
    parser.add_argument(
        "--skip-to",
        choices=PHASE_ORDER,
        metavar="PHASE",
        help="Skip earlier phases and start from this phase",
    )
    parser.add_argument(
        "--only",
        choices=PHASE_ORDER,
        metavar="PHASE",
        help="Run only this single phase",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print submission commands without executing",
    )
    args = parser.parse_args()

    # Determine which phases to run
    if args.only:
        phases_to_run = [args.only]
    elif args.skip_to:
        start_idx = PHASE_ORDER.index(args.skip_to)
        phases_to_run = PHASE_ORDER[start_idx:]
    else:
        phases_to_run = list(PHASE_ORDER)

    print("=" * 60)
    print("Test 2 Experiment Orchestrator")
    print("=" * 60)
    print(f"Flavor:  {args.flavor}")
    print(f"Phases:  {', '.join(phases_to_run)}")
    print(f"Dry run: {args.dry_run}")
    print_cost_estimate(args.flavor, phases_to_run)
    print()

    if not args.dry_run and not os.environ.get("HF_TOKEN"):
        print("ERROR: HF_TOKEN environment variable not set", file=sys.stderr)
        return 1

    completed_phases: set[str] = set()
    job_runs: dict[str, JobRun] = {}

    # Process phases respecting parallel groups and dependencies
    i = 0
    while i < len(phases_to_run):
        phase = phases_to_run[i]

        # Check if this phase and the next are both in PARALLEL_PHASES
        # If so, submit them together and wait for both
        if (
            phase in PARALLEL_PHASES
            and i + 1 < len(phases_to_run)
            and phases_to_run[i + 1] in PARALLEL_PHASES
        ):
            parallel_batch = [phases_to_run[i], phases_to_run[i + 1]]
            print(f"\n--- Submitting in parallel: {', '.join(parallel_batch)} ---")

            runs = []
            for p in parallel_batch:
                run = submit_job(p, args.flavor, args.dry_run)
                if run is None:
                    print(f"Aborting: failed to submit {p}", file=sys.stderr)
                    return 1
                job_runs[p] = run
                runs.append(run)

            print(f"\nWaiting for parallel phases to complete...")
            success = wait_for_jobs(runs, args.dry_run)
            if not success:
                print("Aborting: one or more parallel phases failed.", file=sys.stderr)
                return 1

            completed_phases.update(parallel_batch)
            i += 2

        else:
            # Check dependencies
            deps = PHASES[phase].get("depends_on", [])
            missing_deps = [d for d in deps if d not in completed_phases and d in PHASE_ORDER]
            if missing_deps:
                print(
                    f"WARNING: {phase} depends on {missing_deps} which were not run in this session. "
                    f"Assuming they completed in a prior run.",
                )

            print(f"\n--- {PHASES[phase]['label']} ---")
            run = submit_job(phase, args.flavor, args.dry_run)
            if run is None:
                print(f"Aborting: failed to submit {phase}", file=sys.stderr)
                return 1

            job_runs[phase] = run

            print(f"\nWaiting for completion (polling every {POLL_INTERVAL_SECONDS}s)...")
            success = wait_for_jobs([run], args.dry_run)
            if not success:
                print(f"Aborting: {phase} failed.", file=sys.stderr)
                return 1

            completed_phases.add(phase)
            i += 1

    # Summary
    print("\n" + "=" * 60)
    print("All phases completed successfully!")
    print("=" * 60)
    for phase, run in job_runs.items():
        label = PHASES[phase]["label"]
        print(f"  {label:<40} {run.job_id}  [{run.status}]")

    return 0


if __name__ == "__main__":
    sys.exit(main())
