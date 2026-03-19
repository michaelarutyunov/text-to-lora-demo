# Test 2: JTBD Node Type Detection — Binary Detectors with D2L + T2L

## Background

Test 1 demonstrated Text-to-LoRA (T2L) on an 8-class JTBD node classification task.
Results: 55.8% accuracy, macro F1 0.32. Key issues: class imbalance, niche domain, narrow scope.

The original Test 2 plan (binary pain point presence detection) was abandoned after analysis revealed:

1. **Confounded labels**: v1 interviews (7) had broken graph extraction → all labeled "no" regardless of content. Dropped entirely.
2. **Severe imbalance in v2**: pain_point presence is 87/13 yes/no across 31 v2 interviews (362 utterances). JTBD interviews are designed to surface pain points, making binary presence detection near-trivial.
3. **LLM-generated data problem**: Since labels come from LLM extraction, any classification task derived from them is trivially solvable by another LLM — making T2L's value invisible over zero-shot.
4. **Multi-label reality**: Utterances typically contain multiple node types simultaneously. Multi-class single-label classification forces an artificial choice.

## Revised Approach: Parallel Binary Detectors

### Core idea

Instead of one multi-class classifier, build **independent binary detectors** for each JTBD node type. Each detector answers: "Does this utterance contain a [node_type]? yes/no."

This mirrors a practical production architecture:

```
Utterance
    ├── pain_point detector ──→ yes/no
    ├── gain_point detector ──→ yes/no
    ├── job_trigger detector ──→ yes/no
    ├── solution_approach detector ──→ yes/no
    ├── ...
    │
    ▼
Detected types: [pain_point, solution_approach]
    │
    ▼
Fast LLM call (targeted): "Extract the specific pain_point
and solution_approach labels from this utterance"
    │
    ▼
Structured output for knowledge graph
```

**What this replaces**: One expensive long LLM call per utterance that does detection + extraction + graph construction with the full JTBD methodology in context (~2000+ tokens of system prompt).

### Why this design

1. **Handles multi-label naturally** — each detector is independent
2. **Binary classification is T2L's sweet spot** — short task description, clear yes/no output
3. **D2L adds genuine value** — the model needs JTBD methodology knowledge to distinguish pain_point from gain_point, job_trigger from job_context
4. **Different node types have different balances** — provides varied experimental conditions
5. **Production-relevant** — demonstrates a real cost-reduction architecture, not a toy task
6. **Novel research angle** — testing whether D2L's factual internalization transfers to domain-specific classification (untested in literature)

### Selected detectors (3 of 7)

Selected to cover a range of class balances and difficulty levels:

| Detector | % yes | % no | Why selected |
|----------|-------|------|-------------|
| `job_trigger` | 34% | 66% | Best balance. Hardest to detect without JTBD knowledge — "trigger" is a domain-specific concept. |
| `solution_approach` | 64% | 36% | Good balance. Intuitive concept. Tests whether D2L helps distinguish from pain_point/gain_point. |
| `pain_point` | 87% | 13% | Most relatable for audiences. Imbalanced but evaluated with macro F1 + minority-class metrics. |

### D2L methodology document

The JTBD methodology is already codified in `interview-system-v2/config/methodologies/jobs_to_be_done_v2.yaml` (~160 lines). This contains:

- Node type definitions with descriptions and examples
- Extraction guidelines (linguistic markers for each type)
- Relationship examples showing how types interact in real utterances
- Extractability criteria (what IS and ISN'T a valid extraction)
- Concept naming conventions

This YAML (converted to prose if needed) is the D2L input document. It's ~4 pages — within D2L's validated range.

## Data

### Source
- **31 v2 interviews** from interview-system-v2 (v1 interviews dropped — unreliable labels)
- **362 utterances** total
- Labels derived from graph node `source_utterance_ids` linkage (existing pipeline)

### Splitting strategy
- **Interview-level splitting** (GroupShuffleSplit on `source_file`) — prevents data leakage from same-speaker utterances appearing in train and test
- **Target split**: 22 train / 4 val / 5 test (~250 / ~50 / ~60 utterances)
- Each binary detector uses the same splits but different labels

### Label generation
For each of the 3 selected node types, each utterance gets:
- `yes` if any graph node of that type has a `source_utterance_id` matching the utterance
- `no` otherwise

## Experimental Design

### Conditions

For each of the 3 binary detectors:

| Stage | Method | JTBD knowledge | Task instruction | Training data | What it tests |
|-------|--------|----------------|-----------------|---------------|--------------|
| 0 | Zero-shot | None (prompt only) | None (prompt only) | 0 | Baseline — Mistral-7B's naive understanding |
| 1a | T2L only | None | LoRA from task description | 0 | T2L's value without domain knowledge |
| 1b | D2L only | LoRA from methodology doc | None (prompt only) | 0 | D2L's knowledge transfer to classification |
| 1c | D2L + T2L | Both LoRA adapters stacked | Both | 0 | Combined effect — **the key hypothesis** |
| 2a | QLoRA | None | None | 50 | Supervised with limited data |
| 2b | QLoRA | None | None | 100 | Approaching crossover |
| 2c | QLoRA | None | None | 200 | Should surpass zero-shot methods |
| 2d | QLoRA | None | None | full (~250) | Best supervised performance |

### T2L task descriptions (per detector)

```python
# pain_point
"Binary classification: given a consumer utterance from a JTBD interview, "
"determine whether it contains a pain point — a frustration, obstacle, or "
"difficulty in getting a job done. Output 'yes' or 'no'."

# job_trigger
"Binary classification: given a consumer utterance from a JTBD interview, "
"determine whether it describes a trigger event — something that initiated "
"or prompted the person to seek a solution. Output 'yes' or 'no'."

# solution_approach
"Binary classification: given a consumer utterance from a JTBD interview, "
"determine whether it describes a solution approach — a current or potential "
"method the person has tried or considered. Output 'yes' or 'no'."
```

### D2L document

The `jobs_to_be_done_v2.yaml` methodology file, converted to readable prose. Contains node definitions, extraction guidelines, relationship examples, and extractability criteria. ~160 lines / ~4 pages.

### LoRA stacking (D2L + T2L)

T2L targets `q_proj, v_proj`. D2L targets `down_proj`. Non-overlapping modules — can coexist without interference. Implementation:
1. Run each hypernetwork once to generate LoRA weight matrices
2. Save both as standard PEFT adapters
3. Load both into PeftModel simultaneously (~50 lines of glue code)

### Evaluation metrics

- **Primary**: Macro F1 (treats both classes equally — critical for imbalanced pain_point detector)
- **Secondary**: Per-class precision, recall, F1
- **Diagnostic**: Confusion matrix per condition per detector

### Visualization

**Plot 1 — Learning curve per detector**: X-axis = training examples (0, 50, 100, 200, full), Y-axis = macro F1. Horizontal lines for zero-shot, T2L, D2L, D2L+T2L. QLoRA as ascending curve. Crossover point highlighted.

**Plot 2 — D2L knowledge transfer**: Bar chart comparing zero-shot vs D2L-only across all 3 detectors. Shows which node types benefit most from internalized methodology knowledge.

**Plot 3 — Confusion matrices**: Side-by-side for zero-shot / D2L+T2L / QLoRA-full, showing how domain knowledge resolves classification errors.

## Technical Constraints

### Model
- **Base model**: Mistral-7B-Instruct-v0.2 (required — T2L hypernetwork is architecture-specific)
- **Chat template**: `[INST]...[/INST]` format consistent across all stages

### Compute
- T2L + D2L adapter generation: A100 40GB (hypernetwork forward pass)
- QLoRA training: A100 40GB via HF Jobs
- Inference/evaluation: A100 40GB

### Adaptive batch size
For QLoRA at 50 examples: reduce effective batch size to ensure sufficient training steps (minimum ~50 steps). At 50 examples with batch_size=4 and gradient_accumulation=2, that's ~6 steps/epoch × 3 epochs = ~18 steps — increase epochs to 8-10.

## Risk Register

| Risk | Impact | Mitigation |
|------|--------|-----------|
| D2L doesn't transfer to classification | Central hypothesis fails | Report as negative result — still novel. Fall back to T2L-only narrative. |
| LoRA stacking causes interference | D2L+T2L condition unusable | Report D2L and T2L as independent conditions. Non-overlapping modules reduce this risk. |
| D2L can't process YAML format | Can't generate adapter | Convert methodology to prose/markdown. |
| 362 utterances too few for learning curve | QLoRA stages overlap in confidence intervals | Report with error bars. 5-fold cross-validation at interview level if needed. |

## Implementation Phases

### Phase 1: Data preparation ✓
1. Write `prepare_dataset_v3.py` — interview-level splitting, per-node-type binary labels, drop v1
2. Generate 3 JSONL datasets (one per detector) with same splits
3. Push to HuggingFace Hub (`michaelarutyunov/jtbd-binary-{job-trigger,solution-approach,pain-point}`)

### Phase 2: D2L adapter ✓
1. Convert JTBD methodology YAML to prose document → uploaded to Hub
2. Run D2L hypernetwork on Mistral-7B with methodology document (HF Jobs, A100)
3. Saved as standard PEFT adapter → `michaelarutyunov/jtbd-d2l-mistral7b-methodology`

### Phase 3: T2L adapters (×3) ✓
1. Run T2L hypernetwork with each task description (HF Jobs, A100)
2. Saved 3 PEFT adapters → `michaelarutyunov/jtbd-t2l-{jobtrigger,solutionapproach,painpoint}`

### Phase 4: LoRA stacking ✓
1. Validated D2L + T2L adapters load simultaneously (non-overlapping modules)
2. Inference works correctly with stacked adapters (6/9 test cases passed, threshold 5/9)

### Phase 5a: QLoRA training (in progress)
1. Training 12 adapters: 3 detectors × 4 sizes (50, 100, 200, full)
2. Push to Hub as `michaelarutyunov/jtbd-qlora-{detector}-{size}`

### Phase 5b: Evaluation (pending)
1. Run all conditions (0, 1a, 1b, 1c, 2a-2d) × 3 detectors = 24 evaluation runs
2. Collect metrics, generate plots
3. Write up results

### Phase 6: Writeup
1. Update EVAL.md with results and analysis
2. Update README with Test 2 narrative
3. Draft LinkedIn post narrative

## Success Criteria

1. D2L+T2L outperforms zero-shot on at least 2 of 3 detectors (macro F1)
2. D2L-only shows measurable improvement over zero-shot on job_trigger (the most JTBD-specific concept)
3. QLoRA crossover point is visible on learning curve (D2L+T2L competitive with QLoRA-50)
4. Confusion matrices show qualitatively different error patterns between conditions
5. Commercially compelling cost-reduction narrative with concrete token savings estimate

## How to Run Test 2

### Prerequisites

1. **HuggingFace token** with write access: `export HF_TOKEN=your_token_here`
2. **Interview-system-v2 access**: Data pipeline requires `interview-system-v2/synthetic_interviews/`
3. **Python 3.10+** and [`uv`](https://github.com/astral-sh/uv) installed

---

### Phase 1: Data Preparation (local, ~5 min)

```bash
# Generate JSONL datasets for all 3 detectors
uv run scripts/test2/prepare_dataset_v3.py

# Upload methodology doc to HF Hub (creates repo if needed)
uv run scripts/test2/upload_methodology_doc.py

# Push the 3 binary detector datasets to HF Hub
for det in job_trigger solution_approach pain_point; do
    uv run scripts/test2/push_datasets_to_hub.py \
        --repo michaelarutyunov/jtbd-binary-${det//_/-} \
        --detector $det
done
```

---

### Phases 2–5: Run on HF Jobs (automated)

The orchestrator submits all HF Jobs in the correct order, runs phases 2 and 3 in parallel, and polls for completion before advancing.

```bash
# Recommended: L40S (48GB VRAM, ~$11 total)
python3 scripts/test2/submit_experiment.py --flavor l40sx1

# Alternative: A100 (80GB VRAM, ~$15 total)
python3 scripts/test2/submit_experiment.py --flavor a100-large

# Preview commands and cost without submitting
python3 scripts/test2/submit_experiment.py --dry-run --flavor l40sx1
```

**Available flavors** (single-GPU options suitable for Mistral-7B):

| Flavor | VRAM | Price/hr |
|--------|------|----------|
| `l40sx1` | 48GB | $1.80 |
| `a100-large` | 80GB | $2.50 |

**Resume from a specific phase** (if a job fails and needs resubmitting):
```bash
python3 scripts/test2/submit_experiment.py --flavor l40sx1 --skip-to qlora
python3 scripts/test2/submit_experiment.py --flavor l40sx1 --only evaluate
```

**Total runtime**: ~4–5 hours wall-clock (phases 2+3 run in parallel)

---

### Phase 6: Writeup (local, after experiment completes)

1. Update `EVAL.md` with results from `outputs/test2/evaluation/summary.csv`
2. Update `README.md` with Test 2 narrative
3. Draft LinkedIn post

---

## Troubleshooting

### "command not found: python"
Use `python3` instead of `python` on this machine.

### HF Jobs flavor name not recognised
Run `hf jobs hardware` to see the current list of valid flavor names.

### "CUDA out of memory" during evaluation
Reduce batch size by editing `evaluate_conditions.py`:
```python
# In run_inference(), change default:
def run_inference(..., batch_size: int = 2):  # was 4
```

### Confusion matrix has all zeros
Model is predicting only one class. Check `metrics.json`:
```bash
cat outputs/test2/evaluation/metrics.json | jq '.job_trigger.stage_0.predictions[:10]'
```
If all "unknown": prompt format mismatch — verify `format_prompt()` is consistent across all scripts.

### D2L: `KeyError: 'peft_type'` or "Found missing adapter keys"
- Pin `peft>=0.14.0` (older versions can't auto-detect adapter format).
- PEFT key format must be `base_model.model.model.layers.{i}.mlp.down_proj.lora_{A|B}.default.weight` — note the double `.model` prefix and `.default` adapter name.

### T2L: `ModuleNotFoundError` for hyper_llm_modulator or its transitive deps
The T2L repo uses eager top-level imports that pull in `torchmetrics`, `inflect`, `rouge-score`, `wandb` even for generation-only paths. These are installed at runtime in `stage_t2l_hf_jobs.py`. If a new import surfaces, add it to the `uv pip install` call in `clone_t2l_repo()`.

### T2L: `FlashAttention2 has been toggled on`
T2L does **not** require flash-attn (unlike D2L). The script patches `use_flash_attn=True` → `False` in the cloned repo's `model_loading.py`. If this fails, the patch target may have changed upstream.

### QLoRA: TRL API errors (`max_seq_length`, `tokenizer`)
TRL ≥0.12 renamed parameters: `max_seq_length` → `max_length` in `SFTConfig`, `tokenizer` → `processing_class` in `SFTTrainer`. These are already fixed in the current scripts.

### Detailed debug log
See `scripts/test2/test_2_debug_notes.md` for the full submission-by-submission debug history across all phases.

---

## References

- Sakana AI Text-to-LoRA: ICML 2025 (Charakorn et al.), github.com/SakanaAI/text-to-lora
- Sakana AI Doc-to-LoRA: arXiv:2602.15902, github.com/SakanaAI/doc-to-lora
- PEFT multi-adapter: huggingface.co/docs/peft (add_weighted_adapter, multi-adapter loading)
- Persona drift (background): COLM 2024, "Measuring and Controlling Persona Drift in Language Model Dialogs"
