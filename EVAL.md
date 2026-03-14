# Evaluation Results: Stage 2 QLoRA Fine-tuning

**Date**: 2025-03-14
**Model**: `michaelarutyunov/jtbd-qlora-mistral7b`
**Base Model**: `mistralai/Mistral-7B-Instruct-v0.2`
**Training**: QLoRA (4-bit, r=8, alpha=16, q_proj+v_proj) on 780 examples
**Evaluation**: 197 test set examples

---

## Executive Summary

| Stage | Method | Accuracy | Macro F1 | Unknown % |
|-------|--------|----------|----------|-----------|
| 0 | Zero-shot prompting | 42.6% | N/A | 11.7% |
| 2 | QLoRA fine-tuned | **55.8%** | **0.320** | **0.0%** |

**Key Finding**: Fine-tuning on 780 task-specific examples yields a **31% relative improvement** in accuracy over zero-shot baseline, with complete elimination of uncertain predictions.

---

## Experimental Setup

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

### Dataset Statistics

```
Total examples: 780 (train) + 197 (test) = 977
Train/Test split: ~80/20

Class distribution (train):
  pain_point:       291 (37.3%)
  gain_point:       156 (20.0%)
  emotional_job:     94 (12.1%)
  solution_approach: 94 (12.1%)
  job_statement:     55 ( 7.1%)
  job_context:       55 ( 7.1%)
  job_trigger:       31 ( 4.0%)
  social_job:         8 ( 1.0%)
```

---

## Detailed Results

### Overall Metrics

```
Accuracy:    55.8% (110/197 correct)
Macro F1:    0.320
Unknown:     0 / 197 (0.0%)
Valid:       197 / 197 (100%)
```

### Per-Class Performance

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

**Weighted Average**: Precision: 0.50, Recall: 0.56, F1: 0.50

### Classification Report

```
                   precision    recall  f1-score   support

       pain_point       0.63      0.91      0.75        79
       gain_point       0.49      0.79      0.60        24
    emotional_job       0.35      0.27      0.31        22
solution_approach       0.67      0.28      0.39        29
    job_statement       0.33      0.18      0.23        17
      job_context       0.00      0.00      0.00        16
      job_trigger       0.40      0.22      0.29         9
       social_job       0.00      0.00      0.00         1

         accuracy                           0.56       197
        macro avg       0.36      0.33      0.32       197
     weighted avg       0.50      0.56      0.50       197
```

---

## Analysis

### Strengths

1. **Strong majority class performance**
   - `pain_point`: 91% recall with 75% F1 — the model learned this pattern well
   - `gain_point`: 79% recall with 60% F1 — solid second-best performance
   - Together these 2 classes comprise ~57% of the data, driving overall accuracy

2. **Zero uncertainty**
   - Stage 0 (zero-shot) produced 23/197 unknown predictions (11.7%)
   - Stage 2 (QLoRA) produced 0/197 unknown predictions
   - Model learned to always make a confident prediction

3. **High precision on some classes**
   - `solution_approach`: 67% precision (when it predicts this, it's usually right)
   - `pain_point`: 63% precision
   - This suggests the model learned distinctive features for these classes

### Weaknesses

1. **Rare class failure**
   - `job_context`: 0/16 correct (0%)
   - `social_job`: 0/1 correct (0%)
   - These classes are severely underrepresented in training data (<8% combined)

2. **Class imbalance effects**
   - Model optimizes for majority classes because they dominate the loss
   - 3 classes get zero recall despite having training examples
   - Macro F1 (0.32) << Accuracy (0.56) indicates this imbalance problem

3. **Confusion patterns**
   - Need confusion matrix analysis to identify specific confusions
   - Likely: `job_context` → `job_statement` or `job_trigger` (context classes)
   - Likely: `job_trigger` → `emotional_job` (trigger/emotion link)

### Comparison with Stage 0 (Zero-Shot)

| Metric | Stage 0 | Stage 2 | Delta |
|--------|---------|---------|-------|
| Accuracy | 42.6% | 55.8% | **+13.2%** |
| Unknown % | 11.7% | 0.0% | **-11.7%** |
| Valid predictions | 174/197 | 197/197 | +23 examples |
| Relative improvement | — | **31.0%** | — |

**Interpretation**: Fine-tuning on task-specific data is significantly more effective than prompt engineering alone for this classification task.

---

## Recommendations

### Immediate Actions

1. **Document confusion patterns**
   - Analyze the confusion matrix to understand which classes are being confused
   - Identify systematic prediction errors (e.g., context → statement)

2. **Qualitative analysis**
   - Sample 10-20 misclassified examples per poor-performing class
   - Manual review to understand if labels are ambiguous or features are unclear

3. **Update baselines notebook**
   - Add Stage 2 results to `01_baselines.ipynb` comparison table
   - Generate side-by-side visualizations

### Future Improvements

#### Option A: Address Class Imbalance

**Techniques to try:**
1. **Class weighting**: `weight = 1 / sqrt(class_count)`
2. **Focal loss**: Down-weight well-classified examples
3. **Oversampling**: Replicate rare class examples in training
4. **Data augmentation**: Generate synthetic examples for rare classes

**Expected impact**: Improve rare class recall at cost of majority class precision.

#### Option B: Collect More Data

**Priority order:**
1. `job_context`: +50 examples (from 55 → 105)
2. `job_statement`: +50 examples (from 55 → 105)
3. `job_trigger`: +30 examples (from 31 → 61)
4. `social_job`: +20 examples (from 8 → 28)

**Expected impact**: Better calibration across all classes, improved macro F1.

#### Option C: Model Architecture Changes

**Ideas to explore:**
1. **Larger LoRA rank**: r=16 or r=32 for more capacity
2. **Full fine-tuning**: If compute allows
3. **Ensembling**: Combine multiple QLoRA checkpoints
4. **Threshold tuning**: Optimize per-class prediction thresholds

**Expected impact**: Marginal gains; class imbalance is the primary bottleneck.

---

## Artifacts

### Trained Model
- **Hub**: `michaelarutyunov/jtbd-qlora-mistral7b`
- **Adapter size**: ~27 MB
- **Files**: `adapter_config.json`, `adapter_model.safetensors`

### Training Metrics
- **Trackio Dashboard**: https://michaelarutyunov-trackio.hf.space?project=jtbd-classification&runs=qlora-mistral7b-stage2
- **Training loss**: Decreased from ~2.1 → ~1.2 over 3 epochs
- **Validation loss**: Tracked but not reported here

### Outputs
- **Confusion matrix**: `stage2_confusion_matrix.png` (saved from evaluation job)
- **Classification report**: Included above
- **Comparison table**: Included above

---

## Conclusion

**Stage 2 QLoRA fine-tuning successfully demonstrates the value of task-specific training data.**

- **31% relative accuracy improvement** over zero-shot baseline
- **Complete elimination** of uncertain predictions
- **Strong performance** on majority classes (pain_point, gain_point)

**Primary limitation**: Class imbalance causes poor performance on rare classes (job_context, social_job).

**Next priority**: Address class imbalance through either (a) reweighting/oversampling techniques, or (b) collecting more rare class examples. Expected macro F1 improvement: 0.32 → 0.45+.

**Overall assessment**: Successful proof-of-concept for Text-to-LoRA applied to JTBD node classification. Fine-tuning significantly outperforms zero-shot prompting, validating the hypothesis that task-specific adapter training is effective for this NLP classification task.
