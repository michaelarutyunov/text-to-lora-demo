# Text-to-LoRA: JTBD Node Detection Demo

Proof-of-concept applying [Text-to-LoRA (T2L)](https://github.com/SakanaAI/text-to-lora)
and [Doc-to-LoRA (D2L)](https://github.com/SakanaAI/doc-to-lora) (Sakana AI, ICML 2025)
to consumer interview analysis using Jobs-to-be-Done (JTBD) methodology.

## Experiments

### Test 1 — 8-class JTBD Node Classification (archived)

Quote-level classification into 8 JTBD node types from pre-extracted spans.
Results: 55.8% accuracy, macro F1 0.32. See `test_1_eval.md` for full results.

### Test 2 — Binary JTBD Node Detection with D2L + T2L (current)

**Task**: Independent binary detectors for 3 JTBD node types (job_trigger, solution_approach, pain_point). Each answers: "Does this utterance contain a [node_type]? yes/no."

**Key innovation**: D2L internalizes JTBD methodology knowledge into LoRA weights, T2L generates task-specific adapters — both stacked on the same base model with non-overlapping target modules.

**Conditions**: Zero-shot, T2L-only, D2L-only, D2L+T2L stacked, QLoRA at 50/100/200/full examples.

Full experiment design, how-to-run guide, and troubleshooting: [`scripts/test2/TEST_2.md`](scripts/test2/TEST_2.md)

## Repository Structure

```
text-to-lora-demo/
├── data/
│   ├── processed/               # Test 1 data (archived)
│   └── test2/                   # Test 2 per-detector binary datasets
├── scripts/
│   ├── test1/                   # Test 1 scripts (archived)
│   └── test2/                   # Test 2 scripts
│       ├── TEST_2.md            # Experiment design & how-to-run
│       ├── test_2_debug_notes.md # Debug history across all phases
│       ├── prepare_dataset_v3.py
│       ├── push_datasets_to_hub.py
│       ├── upload_methodology_doc.py
│       ├── stage_d2l_hf_jobs.py  # D2L adapter generation (HF Jobs)
│       ├── stage_t2l_hf_jobs.py  # T2L adapter generation (HF Jobs)
│       ├── validate_lora_stacking.py
│       ├── train_qlora_hf_jobs.py # QLoRA training (HF Jobs)
│       ├── evaluate_conditions.py # Full evaluation pipeline
│       └── submit_experiment.py   # Orchestrator for phases 2-5
├── notebooks/                   # Test 1 notebooks (archived)
├── EVAL.md                      # Results and analysis
├── requirements.txt
└── README.md
```

## Quick Start (Test 2)

### Prerequisites
- Python 3.10+, [`uv`](https://github.com/astral-sh/uv), [`hf` CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli)
- `export HF_TOKEN=your_token_here` (write access required)

### 1. Prepare data (local)
```bash
uv run scripts/test2/prepare_dataset_v3.py
uv run scripts/test2/upload_methodology_doc.py
for det in job_trigger solution_approach pain_point; do
    uv run scripts/test2/push_datasets_to_hub.py \
        --repo michaelarutyunov/jtbd-binary-${det//_/-} --detector $det
done
```

### 2. Run experiment (HF Jobs)
```bash
# Submits all phases (D2L, T2L, validate, QLoRA, evaluate) in order
python3 scripts/test2/submit_experiment.py --flavor a100-large

# Monitor
hf jobs ps
hf jobs logs <job-id>
```

See [`scripts/test2/TEST_2.md`](scripts/test2/TEST_2.md) for hardware options, resume commands, and troubleshooting.

### 3. Download results (local)
```bash
# Evaluation outputs are uploaded to Hub automatically
uv run python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('michaelarutyunov/jtbd-test2-evaluation',
                  repo_type='dataset',
                  local_dir='outputs/test2/evaluation')
"
```

Results analysis: [`scripts/test2/test_2_eval.md`](scripts/test2/test_2_eval.md)

## Technical Details

- **Base model**: Mistral-7B-Instruct-v0.2
- **D2L**: Targets `down_proj`, requires flash-attn + A100
- **T2L**: Targets `q_proj`/`v_proj`, no flash-attn needed
- **LoRA stacking**: Non-overlapping modules allow simultaneous loading
- **QLoRA**: r=8, alpha=16, q_proj+v_proj — matches T2L for fair comparison

## Hub Artifacts

| Artifact | Hub ID |
|----------|--------|
| D2L adapter | `michaelarutyunov/jtbd-d2l-mistral7b-methodology` |
| T2L adapters | `michaelarutyunov/jtbd-t2l-{jobtrigger,solutionapproach,painpoint}` |
| QLoRA adapters | `michaelarutyunov/jtbd-qlora-{detector}-{size}` |
| Datasets | `michaelarutyunov/jtbd-binary-{job-trigger,solution-approach,pain-point}` |
| Evaluation outputs | `michaelarutyunov/jtbd-test2-evaluation` |

## References

- Sakana AI Text-to-LoRA: ICML 2025 (Charakorn et al.), [github](https://github.com/SakanaAI/text-to-lora)
- Sakana AI Doc-to-LoRA: arXiv:2602.15902, [github](https://github.com/SakanaAI/doc-to-lora)
- PEFT multi-adapter: [docs](https://huggingface.co/docs/peft)
