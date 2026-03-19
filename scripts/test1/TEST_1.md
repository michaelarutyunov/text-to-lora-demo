# Test 1: 8-Class JTBD Node Classification

## Background

Initial proof-of-concept applying [Text-to-LoRA (T2L)](https://github.com/SakanaAI/text-to-lora) (Charakorn et al., ICML 2025, Sakana AI) to consumer interview analysis using the Jobs-to-Be-Done (JTBD) framework.

The core research question: **Can T2L provide zero-shot performance competitive with supervised fine-tuning for niche domain classification tasks?**

### Why Text-to-LoRA?

T2L's value proposition is **cold-start learning** — generating task-specific LoRA adapters from natural language task descriptions alone, without any training data. This is powerful for:
- Rapid prototyping in niche domains
- Low-resource scenarios where labeling is expensive
- Exploratory analysis before committing to data collection

### The JTBD Classification Task

JTBD interviews produce a knowledge graph where each node represents a meaningful concept (pain point, gain point, job statement, etc.) extracted from consumer responses. The task: **classify verbatim quote spans into one of 8 JTBD node types.**

This is a realistic, challenging NLP task:
- **Domain-specific**: Requires understanding JTBD taxonomy
- **Subtle distinctions**: e.g., `job_context` vs `job_statement` vs `job_trigger`
- **Real-world data**: Derived from actual consumer interviews
- **Limited labeled data**: 977 examples across 8 classes (sparse for some types)

## Options Considered

### Option A: Utterance-level binary classification ✅
Classify full consumer responses as "contains pain point" vs "does not."
**Selected for Test 2** — simpler, more relatable, better T2L fit.

### Option B: Pain point extraction (NER)
Extract pain point spans from raw utterances.
**Rejected** — Structured extraction, not classification. T2L strength is task description → adapter mapping.

### Option C: Sentence-level multi-label
Split utterances into sentences, predict all applicable node types.
**Rejected** — Requires fuzzy quote-to-sentence alignment; LLM-extracted quotes don't map cleanly to sentence boundaries.

### Option D: Quote-level multi-class classification ✅
Classify pre-extracted quote spans into 8 JTBD node types.
**Selected for Test 1** — demonstrates T2L on complex, nuanced classification.

## Chosen Approach: Quote-Level 8-Class Classification

**Task**: Given a verbatim quote span extracted from a consumer interview response, classify it into one of 8 JTBD node types.

### Why this approach

1. **Directly derived from data** — Quotes are pre-extracted by the interview system, no additional processing needed
2. **Showcases T2L's capabilities** — Multi-class, nuanced distinctions, domain-specific
3. **Realistic complexity** — Mirrors actual JTBD analysis workflow
4. **Clear evaluation metrics** — Standard classification accuracy and macro F1

### The 8 JTBD Node Types

| Class | Description | Example |
|-------|-------------|---------|
| `pain_point` | Consumer frustration or problem | *"It takes too long to find what I need"* |
| `gain_point` | Desired outcome or benefit | *"I want to feel confident in my choice"* |
| `emotional_job` | Emotional motivation | *"I feel anxious about making the wrong decision"* |
| `solution_approach` | How consumer solves the problem | *"I usually ask friends for recommendations"* |
| `job_statement` | Core job the consumer is trying to do | *"I need to compare products before buying"* |
| `job_context` | Circumstantial context | *"I'm shopping on my phone during lunch break"* |
| `job_trigger` | Event that initiates the job | *"My old laptop finally died"* |
| `social_job` | Social dimension of the job | *"I want to look knowledgeable in front of my team"* |

### Data characteristics

From 8 JTBD interviews, extracted 977 quote-level examples:

| Class | Count | Percentage |
|-------|-------|-----------|
| pain_point | 291 | 29.8% |
| gain_point | 156 | 16.0% |
| emotional_job | 94 | 9.6% |
| solution_approach | 94 | 9.6% |
| job_statement | 55 | 5.6% |
| job_context | 55 | 5.6% |
| job_trigger | 31 | 3.2% |
| social_job | 8 | 0.8% |

**Class imbalance**: `pain_point` dominates (37.3% of training data), `social_job` is sparse (1.0%). This imbalance significantly impacts macro F1.

Dataset splits (stratified by node type):
- **Train**: 780 examples (79.8%)
- **Test**: 197 examples (20.2%)
- **Val**: Not used in evaluation (held out for hyperparameter tuning)

## Experiment Design: Two-Stage Comparison

### Stage 0: Zero-Shot Prompting Baseline

**Purpose**: Establish what Mistral-7B-Instruct-v0.2 can do out-of-the-box with only prompt engineering.

**Method**: Provide the model with a detailed description of the 8 JTBD node types and ask it to classify each quote.

**Hypothesis**: Zero-shot performance will be modest due to domain specificity and nuanced class distinctions.

### Stage 2: Supervised QLoRA Fine-tuning

**Purpose**: Measure the upper bound of task-specific fine-tuning with 780 labeled examples.

**Method**: Train a QLoRA adapter (r=8, alpha=16, 4-bit quantization) on the training set for 3 epochs.

**Hypothesis**: Fine-tuning will significantly outperform zero-shot, but rare classes will suffer due to imbalance.

### Missing Stage: T2L Zero-Shot Adapter

**Critical gap**: Stage 1 (T2L) was not completed due to A100 GPU availability. T2L's `load_hypermod_checkpoint` internally loads a second copy of Mistral-7B, exceeding T4 GPU memory (16 GB).

**Impact**: Cannot assess T2L's zero-shot value proposition — the key contribution of the paper.

## Results Summary

| Stage | Method | Accuracy | Macro F1 | Unknown % |
|-------|--------|----------|----------|-----------|
| 0 | Zero-shot prompting | 42.6% | N/A | 11.7% |
| 2 | QLoRA fine-tuned | **55.8%** | **0.320** | **0.0%** |

**Key Finding**: Fine-tuning on 780 task-specific examples yields a **31% relative improvement** in accuracy over zero-shot baseline, with complete elimination of uncertain predictions.

### Per-Class Performance (Stage 2)

| Class | Precision | Recall | F1 | Support | Rank |
|-------|-----------|--------|-----|---------|------|
| pain_point | **0.63** | **0.91** | **0.75** | 79 | 1 |
| gain_point | 0.49 | **0.79** | **0.60** | 24 | 2 |
| solution_approach | **0.67** | 0.28 | 0.39 | 29 | 3 |
| emotional_job | 0.35 | 0.27 | 0.31 | 22 | 4 |
| job_trigger | 0.40 | 0.22 | 0.29 | 9 | 5 |
| job_statement | 0.33 | 0.18 | 0.23 | 17 | 6 |
| job_context | 0.00 | 0.00 | 0.00 | 16 | 7 |
| social_job | 0.00 | 0.00 | 0.00 | 1 | 8 |

**Critical insight**: Model learned majority classes well (`pain_point`, `gain_point`) but failed completely on rare classes (`job_context`, `social_job`).

## Analysis

### Strengths

1. **Strong majority class performance**
   - `pain_point`: 91% recall with 75% F1
   - `gain_point`: 79% recall with 60% F1
   - These 2 classes comprise ~57% of data, driving overall accuracy

2. **Zero uncertainty**
   - Stage 0 produced 23/197 unknown predictions (11.7%)
   - Stage 2 produced 0/197 unknown predictions
   - Model learned to always make a confident prediction

3. **High precision on some classes**
   - `solution_approach`: 67% precision
   - `pain_point`: 63% precision
   - Model learned distinctive features

### Weaknesses

1. **Rare class failure**
   - `job_context`: 0/16 correct (0%)
   - `social_job`: 0/1 correct (0%)
   - Severe underrepresentation in training data (<8% combined)

2. **Class imbalance effects**
   - Macro F1 (0.32) << Accuracy (0.56)
   - Model optimizes for majority classes
   - 3 classes get zero recall despite having training examples

3. **Domain complexity**
   - JTBD taxonomy requires explanation
   - Nuanced distinctions (`job_context` vs `job_statement` vs `job_trigger`)
   - Not instantly relatable to broader audiences

### Why This Led to Test 2

Three realizations from Test 1 results:

1. **Task mismatch**: T2L shines at zero-shot cold-start, but Test 1 focused on supervised fine-tuning comparison without running Stage 1 (T2L) itself.

2. **Audience accessibility**: 8-class JTBD node classification is niche — requires explaining the taxonomy. Hard to communicate value to non-JTBD audiences.

3. **Class imbalance**: Severe imbalance (`social_job` at 0.8%) makes macro F1 uninterpretable. Need simpler, balanced task.

4. **Portfolio impact**: Binary pain point detection is instantly relatable — *"detects customer pain points in interview transcripts"* vs *"classifies quotes into 8 JTBD node types."*

## Key Considerations

### T2L is Model-Specific

The T2L hypernetwork generates LoRA adapters parameterized for Mistral-7B's architecture. Adapters cannot be transferred to other model families without retraining the hypernetwork. This constrains both T2L and QLoRA stages to `Mistral-7B-Instruct-v0.2`.

### Hardware Requirements

- **T2L Stage 1**: Requires A100 (40 GB) — `load_hypermod_checkpoint` loads a second copy of Mistral-7B; combined VRAM exceeds T4 (16 GB)
- **QLoRA Stage 2**: Requires A100 (24 GB) or T4 (16 GB) with 4-bit quantization

### Missing Evaluation Pieces

- **Stage 1 (T2L)**: Not run due to GPU constraints
- **Learning curve**: Only two data points (0 and 780 examples). Need intermediate points (50, 100, 200) to characterize where supervised fine-tuning overtakes T2L.

## Artifacts

### Trained Model
- **Hub**: `michaelarutyunov/jtbd-qlora-mistral7b`
- **Adapter size**: ~27 MB
- **Files**: `adapter_config.json`, `adapter_model.safetensors`

### Training Configuration
```yaml
Base Model: mistralai/Mistral-7B-Instruct-v0.2
Method: QLoRA (4-bit quantization)

LoRA Config:
  rank: 8
  lora_alpha: 16
  target_modules: [q_proj, v_proj]
  lora_dropout: 0.05

Training:
  epochs: 3
  batch_size: 2 (per device)
  gradient_accumulation: 8 (effective batch: 16)
  learning_rate: 2e-4
  warmup_steps: 22 (~5% of total steps)
  max_seq_length: 512
  weight_decay: 0.01

Hardware: Hugging Face Jobs (a10g-large)
Training time: ~2 hours
```

### Outputs
- **Confusion matrix**: https://huggingface.co/michaelarutyunov/jtbd-qlora-mistral7b/blob/main/stage2_confusion_matrix.png
- **Classification report**: See EVAL.md
- **Comparison table**: See EVAL.md

## Conclusion

**Test 1 successfully demonstrates the value of task-specific training data for niche domain classification.**

- **31% relative accuracy improvement** over zero-shot baseline
- **Complete elimination** of uncertain predictions
- **Strong performance** on majority classes (pain_point, gain_point)

**Primary limitations**:
- Class imbalance causes poor performance on rare classes
- 8-class JTBD taxonomy is not instantly relatable
- Missing T2L Stage 1 due to GPU constraints

**Decision**: Pivot to Test 2 — binary pain point presence detection on full utterances. Simpler task, better balance, more relatable narrative, and clearer path to T2L evaluation.

---

## Next Steps (Transition to Test 2)

1. **Generate more interviews** — run interview-system-v2 overnight to reach ~400 utterances (~280 train / 60 val / 60 test)
2. **Write `prepare_dataset_v2.py`** — utterance-level, binary labels, derived from graph nodes
3. **Run Stage 1 (T2L zero-shot)** — critical missing piece from Test 1; requires A100 GPU
4. **Run Stage 2 learning curve** — QLoRA at 50, 100, 200, full training sizes on HF Jobs
5. **Plot crossover** — accuracy vs training examples, T2L as horizontal baseline
6. **Update README and EVAL.md** — document Test 1 and Test 2 results, narrative

See `scripts/test2/TEST_2.md` for detailed Test 2 design.
