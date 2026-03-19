# Test 2 Implementation Summary

## Phase 1: Data Preparation — `prepare_dataset_v3.py` + `push_datasets_to_hub.py`

- v2-only filtering via `is_v2_jtbd()` (v1 interviews dropped — unreliable labels)
- Interview-level splitting with `GroupShuffleSplit` (prevents data leakage from same-speaker utterances)
- 3 detectors: `job_trigger`, `solution_approach`, `pain_point`
- Binary labels via `source_quotes` substring matching (first 40 chars)
- Same splits across all detectors for fair comparison
- Output format: `{utterance, label, source_file, turn_number}`
- Data split: 238 train / 27 val / 48 test = 313 utterances per detector

## Phase 2: D2L Adapter — `stage_d2l_hf_jobs.py`

- Uses actual D2L hypernetwork (single forward pass, not SFT)
- Downloads JTBD methodology prose document from dataset repo
- Loads `SakanaAI/doc-to-lora` checkpoint (`mistral_7b_d2l/checkpoint-20000`)
- Extracts LoRA weights from `model.generated_loras` and saves in PEFT format
- Reads `r`, `alpha`, `target_modules` from checkpoint (not hardcoded)
- D2L targets `down_proj` (MLP layers)
- Validates adapter loads back + runs probe questions for methodology internalization
- Pushes to `michaelarutyunov/jtbd-d2l-mistral7b-methodology`

## Phase 3: T2L Adapters — `stage_t2l_hf_jobs.py`

- Uses actual T2L hypernetwork (single forward pass per detector, not SFT)
- Task descriptions match TEST_2.md spec exactly (one per detector)
- Downloads `SakanaAI/text-to-lora` checkpoint (`trained_t2l/mistral_7b_t2l/hypermod.pt`)
- T2L targets `q_proj, v_proj` (attention layers — non-overlapping with D2L's `down_proj`)
- Generates 3 adapters, validates each, pushes to Hub
- Frees duplicate base model loaded by hypernetwork to manage VRAM

## Phase 4: LoRA Stacking Validation — `validate_lora_stacking.py`

- Loads D2L first (`adapter_name="d2l_jtbd"`), then T2L (`adapter_name="t2l_detector"`)
- No `set_adapter()` call needed — modules are non-overlapping
- Uses `model.unload()` to properly clean base model between detectors
- Test utterances with known expected labels for sanity checking
- Allows 2/9 failures for model variance

## Phase 5: Evaluation — `evaluate_conditions.py`

- All 8 stages: 0 (zero-shot), 1a (T2L only), 1b (D2L only), 1c (D2L+T2L), 2a-2d (QLoRA 50/100/200/full)
- 3 detectors × 8 stages = 24 experimental conditions
- Consistent prompt format across all scripts: `<s>[INST] [UTTERANCE]: {utterance}\nDoes this contain a {label}? Answer yes or no.\nAnswer: [/INST]`
- Primary metric: macro F1 (treats both classes equally — critical for imbalanced detectors)
- Secondary metrics: per-class precision/recall/F1, confusion matrices
- 3 plots: learning curve, D2L knowledge transfer bar chart, confusion matrices comparison
- Proper adapter load/unload cycle to maintain clean base model state

## Phase 5b: QLoRA Training — `train_qlora_hf_jobs.py`

- 3 detectors × 4 sizes (50, 100, 200, full) = 12 adapters
- LoRA config matches T2L (`r=8, alpha=16, q_proj+v_proj`) for fair comparison
- Adaptive training config: more epochs for smaller datasets (10 epochs at 50 examples, 3 at full)
- Deterministic subsampling (sorted by source_file + turn_number, then sliced)
- 4-bit quantization via `BitsAndBytesConfig` (proper QLoRA)
- Auto-updates `evaluate_conditions.py` QLORA_ADAPTERS dict after training

## Cross-Script Consistency

| Aspect | d2l | t2l | validate | evaluate | train_qlora |
|--------|:---:|:---:|:---:|:---:|:---:|
| Base model: Mistral-7B-Instruct-v0.2 | ✓ | ✓ | ✓ | ✓ | ✓ |
| Prompt format consistent | — | ✓ | ✓ | ✓ | ✓ |
| D2L targets `down_proj` | ✓ | — | ✓ | ✓ | — |
| T2L targets `q_proj,v_proj` | — | ✓ | ✓ | ✓ | ✓ |

## Notes

- QLoRA targets the same modules as T2L (`q_proj, v_proj`) — intentional for fair parameter-budget comparison, but means QLoRA adapters cannot be stacked with T2L (not required by experimental design)
- `evaluate_conditions.py` reads from local JSONL (string labels "yes"/"no"), while Hub datasets use int labels (0/1) — different code paths, both correct
- Methodology document (`jtbd_methodology_v2_prose.md`) is hosted on dataset repo and downloaded at D2L generation time
