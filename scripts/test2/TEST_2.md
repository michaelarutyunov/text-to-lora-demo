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

### Phase 1: Data preparation
1. Write `prepare_dataset_v2.py` — interview-level splitting, per-node-type binary labels, drop v1
2. Generate 3 JSONL datasets (one per detector) with same splits
3. Push to HuggingFace Hub

### Phase 2: D2L adapter
1. Convert JTBD methodology YAML to prose document
2. Run D2L hypernetwork on Mistral-7B with methodology document
3. Save as standard PEFT adapter
4. Validate: probe model with JTBD knowledge questions to confirm internalization

### Phase 3: T2L adapters (×3)
1. Run T2L hypernetwork with each task description
2. Save 3 PEFT adapters (one per detector)

### Phase 4: LoRA stacking
1. Write adapter extraction + combination script
2. Test loading D2L + T2L adapters simultaneously
3. Validate inference works correctly with stacked adapters

### Phase 5: Evaluation
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

This section provides step-by-step instructions to execute all phases of Test 2.

### Prerequisites

1. **HuggingFace token** with write access
   ```bash
   export HF_TOKEN=your_token_here
   # Or login once:
   huggingface-cli login
   ```

2. **Required Python packages** (installed via `uv` automatically from PEP 723 headers):
   - T2L/D2L: `torch`, `transformers`, `peft`, `text-to-lora`
   - Data prep: `datasets`, `pandas`
   - Evaluation: `scikit-learn`, `matplotlib`, `seaborn`

3. **Compute requirements**:
   - **Local**: CPU for data prep, A100 40GB for adapter generation/training
   - **HuggingFace Jobs**: A100 40GB (`a100-large` flavor)

4. **Interview-system-v2 access**: The data pipeline requires access to `interview-system-v2/synthetic_interviews`

---

### Phase 1: Data Preparation (Local, ~5 min)

Generate binary classification datasets for all 3 detectors.

```bash
cd /home/mikhailarutyunov/projects/text-to-lora-demo
uv run scripts/test2/prepare_dataset_v3.py
```

**Expected output**:
```
Loaded 31 interviews
Total utterances: 362
Interview-level split: 22 train / 4 val / 5 test

Generated datasets:
- data/test2/job_trigger_train.jsonl (80 utterances)
- data/test2/job_trigger_val.jsonl (15 utterances)
- data/test2/job_trigger_test.jsonl (60 utterances)
- data/test2/solution_approach_{train,val,test}.jsonl
- data/test2/pain_point_{train,val,test}.jsonl
```

**Push to Hub** (makes datasets accessible for HF Jobs):
```bash
uv run scripts/test2/push_datasets_to_hub.py
```

---

### Phase 2: D2L Adapter Generation (A100 40GB, ~30 min)

Generate a domain knowledge adapter from the JTBD methodology document.

#### Step 2a: Upload methodology document to HuggingFace

```bash
uv run scripts/test2/upload_methodology_doc.py
```

This uploads `interview-system-v2/config/methodologies/jobs_to_be_done_v2.yaml` as a readable prose document to HF Hub.

**Expected output**: Hub repository URL (e.g., `https://huggingface.co/datasets/michaelarutyunov/jtbd-methodology-v2`)

#### Step 2b: Run D2L hypernetwork on HF Jobs

```bash
hf jobs uv run \
    --flavor a100-large \
    --timeout 1h \
    --secrets HF_TOKEN \
    scripts/test2/stage_d2l_hf_jobs.py
```

**What this does**:
1. Downloads the methodology document from Hub
2. Runs Doc-to-LoRA hypernetwork on Mistral-7B
3. Saves adapter as PEFT format to Hub: `michaelarutyunov/jtbd-d2l-mistral7b`

**Expected output**:
```
Doc-to-LoRA adapter saved to: michaelarutyunov/jtbd-d2l-mistral7b
```

---

### Phase 3: T2L Adapter Generation (A100 40GB, ~30 min)

Generate task-specific adapters from short text descriptions (one per detector).

```bash
hf jobs uv run \
    --flavor a100-large \
    --timeout 1h \
    --secrets HF_TOKEN \
    scripts/test2/stage_t2l_hf_jobs.py
```

**What this does**:
1. For each detector (job_trigger, solution_approach, pain_point):
   - Runs Text-to-LoRA hypernetwork on Mistral-7B
   - Uses the task description as the conditioning prompt
   - Saves adapter as PEFT format to Hub

**Expected output**:
```
T2L adapters saved:
- michaelarutyunov/jtbd-t2l-job-trigger-mistral7b
- michaelarutyunov/jtbd-t2l-solution-approach-mistral7b
- michaelarutyunov/jtbd-t2l-pain-point-mistral7b
```

---

### Phase 4: LoRA Stacking Validation (A100 40GB, ~15 min)

Verify that D2L and T2L adapters can be loaded simultaneously without interference.

```bash
hf jobs uv run \
    --flavor a10g-small \
    --timeout 30m \
    --secrets HF_TOKEN \
    scripts/test2/validate_lora_stacking.py
```

**What this does**:
1. Loads base Mistral-7B model
2. For each detector, tests loading D2L + T2L stacked adapters
3. Runs inference on 9 generic test utterances
4. Validates that adapters produce expected outputs

**Expected output**:
```
=== Testing LoRA Stacking ===
job_trigger detector:
  Zero-shot: 2/9 correct
  D2L + T2L stacked: 7/9 correct ✓

solution_approach detector:
  Zero-shot: 3/9 correct
  D2L + T2L stacked: 8/9 correct ✓

pain_point detector:
  Zero-shot: 6/9 correct
  D2L + T2L stacked: 8/9 correct ✓

All detectors passed stacking validation!
```

**If this fails**: Check PEFT version compatibility and adapter target modules. D2L should target `down_proj`, T2L should target `q_proj, v_proj` (non-overlapping).

---

### Phase 5: Full Evaluation (A100 40GB, ~2-3 hours)

Run all 24 experimental conditions and generate metrics/plots.

```bash
hf jobs uv run \
    --flavor a100-large \
    --timeout 4h \
    --secrets HF_TOKEN \
    scripts/test2/evaluate_conditions.py
```

**What this does**:
1. Evaluates all 24 conditions (0, 1a, 1b, 1c, 2a-2d × 3 detectors)
2. Computes macro F1, precision, recall for each condition
3. Generates 3 visualization plots
4. Saves metrics as JSON and summary CSV

**Expected outputs** (saved to `outputs/test2/evaluation/`):
```
metrics.json                    # All raw metrics and predictions
summary.csv                     # Human-readable summary table
learning_curve.png              # Plot 1: Learning curves per detector
d2l_knowledge_transfer.png      # Plot 2: D2L vs zero-shot comparison
confusion_matrices.png          # Plot 3: Confusion matrix grid
{detector}_{stage}_confusion_matrix.png  # Individual confusion matrices
```

**Sample terminal output**:
```
============================================================
Test 2 Phase 5: Full Evaluation Pipeline
============================================================
Loading base model: mistralai/Mistral-7B-Instruct-v0.2

============================================================
Detector: job_trigger
============================================================
Test set size: 60 utterances

=== Stage 0: Zero-shot (job_trigger) ===
=== Stage 1a: T2L only (job_trigger) ===
...
=== Stage 2: QLoRA-full (job_trigger) ===

============================================================
Detector: solution_approach
============================================================
...

============================================================
SUMMARY
============================================================

Job_Trigger:
Stage           Macro F1   Accuracy
-----------------------------------
Zero            0.450      0.550
T2L             0.520      0.620
D2L             0.580      0.640
D2L+T2L         0.650      0.720
QLoRA-50        0.680      0.740
QLoRA-100       0.710      0.780
QLoRA-200       0.740      0.810
QLoRA-Full      0.770      0.850
...
```

---

### Phase 6: Writeup (Local, ~1-2 hours)

Document results and create shareable content.

#### 6a. Update EVAL.md

Add results section to `/home/mikhailarutyunov/projects/text-to-lora-demo/EVAL.md`:

```markdown
---
## Test 2: Binary Detectors with D2L + T2L

**Date**: March 2026
**Approach**: Parallel binary detectors with Doc-to-LoRA + Text-to-LoRA

### Executive Summary

| Detector | Best Method | Macro F1 | Zero-shot F1 | Improvement |
|----------|-------------|----------|--------------|-------------|
| job_trigger | D2L+T2L | 0.65 | 0.45 | +44% |
| solution_approach | QLoRA-100 | 0.72 | 0.50 | +44% |
| pain_point | D2L+T2L | 0.78 | 0.65 | +20% |

**Key findings**:
- D2L knowledge transfer improves JTBD-specific concepts (job_trigger)
- T2L effectively captures task structure
- D2L+T2L stacking matches QLoRA-50 performance without training data
...
```

#### 6b. Update README

Add Test 2 section to project README:

```markdown
### Test 2: Binary Detectors (2026)

Explores parallel binary detection architecture with domain knowledge internalization via Doc-to-LoRA.

**Key results**: D2L+T2L stacked adapters achieve 44% improvement over zero-shot on JTBD-specific concepts without task-specific training data.

**Links**: [Full methodology](scripts/test2/TEST_2.md) | [Implementation log](scripts/test2/implementation_log.md)
```

#### 6c. Draft LinkedIn post

Create narrative highlighting:
1. **Problem**: Expensive LLM calls for JTBD extraction
2. **Solution**: Parallel binary detectors with LoRA adapters
3. **Innovation**: Domain knowledge internalization via Doc-to-LoRA
4. **Results**: 44% improvement without training data, 10x cost reduction

---

## Troubleshooting

### Issue: "Module not found" errors

**Cause**: PEP 723 dependencies not installed

**Fix**: Ensure you're using `uv run` (not `python`):
```bash
uv run scripts/test2/prepare_dataset_v3.py  # ✓
python scripts/test2/prepare_dataset_v3.py   # ✗
```

### Issue: HF Jobs fails with "quota exceeded"

**Cause**: No available A100 GPUs in your HF account

**Fix**:
1. Check your quota: https://huggingface.co/settings/billing
2. Try smaller flavor: `--flavor a10g-small` (may OOM with Mistral-7B)
3. Run locally if you have A100/A10G: Remove `hf jobs uv run` prefix

### Issue: "CUDA out of memory" during evaluation

**Cause**: Batch size too large for GPU

**Fix**: Reduce batch size in `evaluate_conditions.py`:
```python
# In main():
result = evaluate_stage_0(base_model, tokenizer, dataset, detector_name)
# Change to:
result = evaluate_stage_0(base_model, tokenizer, dataset, detector_name, batch_size=2)
```

Or add this near the top of the file:
```python
# Change default batch size
DEFAULT_BATCH_SIZE = 2  # Instead of 4
```

### Issue: "Adapter not found" during evaluation

**Cause**: Adapter IDs in `evaluate_conditions.py` are placeholders

**Fix**: Update the adapter ID constants after Phases 2-3 complete:
```python
D2L_ADAPTER_ID = "michaelarutyunov/jtbd-d2l-mistral7b"  # Update this
T2L_ADAPTERS = {
    "job_trigger": "michaelarutyunov/jtbd-t2l-job-trigger-mistral7b",  # Update these
    ...
}
```

### Issue: LoRA stacking produces wrong predictions

**Cause**: Adapters targeting overlapping modules causing interference

**Fix**: Verify target modules are non-overlapping:
```python
# T2L should target: q_proj, v_proj
# D2L should target: down_proj

# Check in adapter config:
# cat ~/.cache/huggingface/hub/.../adapter_config.json
# "target_modules": ["q_proj", "v_proj"]  # T2L
# "target_modules": ["down_proj"]            # D2L
```

If overlap exists, re-train one adapter with correct targets.

### Issue: Confusion matrix has all zeros

**Cause**: Model predictions are all "unknown" or all one class

**Diagnosis**: Check predictions in `metrics.json`:
```bash
cat outputs/test2/evaluation/metrics.json | jq '.job_trigger.stage_0.predictions' | head -20
```

**Fix**:
- If all "unknown": Prompt format mismatch, check `format_prompt()` matches training
- If all "yes" or all "no": Class imbalance, use balanced accuracy or weighted loss
- If random predictions: Adapter not loaded correctly, verify `model.eval()` was called

---

## Quick Reference

**Run all phases sequentially**:
```bash
# Phase 1: Data (local)
uv run scripts/test2/prepare_dataset_v3.py
uv run scripts/test2/push_datasets_to_hub.py

# Phase 2: D2L (HF Jobs)
uv run scripts/test2/upload_methodology_doc.py
hf jobs uv run --flavor a100-large --timeout 1h --secrets HF_TOKEN scripts/test2/stage_d2l_hf_jobs.py

# Phase 3: T2L (HF Jobs)
hf jobs uv run --flavor a100-large --timeout 1h --secrets HF_TOKEN scripts/test2/stage_t2l_hf_jobs.py

# Phase 4: Validate stacking (HF Jobs)
hf jobs uv run --flavor a10g-small --timeout 30m --secrets HF_TOKEN scripts/test2/validate_lora_stacking.py

# Phase 5: Evaluate (HF Jobs)
hf jobs uv run --flavor a100-large --timeout 4h --secrets HF_TOKEN scripts/test2/evaluate_conditions.py

# Phase 6: Writeup (local)
# Manually update EVAL.md, README.md, draft LinkedIn post
```

**Total compute time**: ~4-5 hours on A100 40GB
**Total cost**: ~$20-30 on HuggingFace Jobs (as of March 2026)

---

## References

- Sakana AI Text-to-LoRA: ICML 2025 (Charakorn et al.), github.com/SakanaAI/text-to-lora
- Sakana AI Doc-to-LoRA: arXiv:2602.15902, github.com/SakanaAI/doc-to-lora
- PEFT multi-adapter: huggingface.co/docs/peft (add_weighted_adapter, multi-adapter loading)
- Persona drift (background): COLM 2024, "Measuring and Controlling Persona Drift in Language Model Dialogs"
