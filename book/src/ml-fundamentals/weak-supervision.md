# Weak Supervision Theory

Weak supervision uses noisy, limited, or imprecise labels to train models when perfect labels are unavailable or expensive.

## The Labeling Bottleneck

| Data Type | Scale | Label Cost |
|-----------|-------|------------|
| Web text | Billions | $0 (unlabeled) |
| Reviews with stars | Millions | Free (noisy) |
| Expert annotations | Thousands | $50-500/sample |

Weak supervision bridges the gap between unlabeled and perfectly labeled data.

## Types of Weak Supervision

### 1. Incomplete Supervision

Only some samples are labeled:

```
Dataset: [x₁, x₂, x₃, x₄, x₅, ...]
Labels:  [y₁,  ?,  ?, y₄,  ?, ...]
```

**Approaches:** Semi-supervised learning, self-training

### 2. Inexact Supervision

Labels at coarser granularity:

```
Document: "The movie was great but too long"
Document label: Positive (but sentence 2 is negative)
```

**Approaches:** Multiple instance learning, attention

### 3. Inaccurate Supervision

Labels contain errors:

```
True label: Positive
Noisy label: Negative (human error)
```

**Approaches:** Noise modeling, co-teaching

## Labeling Functions

Programmatic rules that generate noisy labels:

```python
# Labeling function for sentiment
def lf_positive_words(text):
    if any(word in text for word in ["great", "amazing", "excellent"]):
        return POSITIVE
    return ABSTAIN

def lf_negative_words(text):
    if any(word in text for word in ["terrible", "awful", "bad"]):
        return NEGATIVE
    return ABSTAIN
```

### Properties

| Property | Description |
|----------|-------------|
| Coverage | Fraction of samples labeled |
| Accuracy | Correctness when not abstaining |
| Overlap | Agreement between LFs |
| Conflict | Disagreement between LFs |

## Label Model

Aggregate multiple labeling functions:

```
       LF₁    LF₂    LF₃    LF₄
         \    /  \  /    /
          ▼  ▼    ▼▼    ▼
         Probabilistic Label
               │
               ▼
          True Label (latent)
```

**Data Programming (Snorkel):**
- Model LF accuracies and correlations
- Infer probabilistic labels
- Train end model on soft labels

## Noise-Aware Training

### Forward Correction

Model the noise transition:

```
P(ỹ|x) = Σᵧ P(ỹ|y) · P(y|x)
           │
      Noise matrix T
```

### Backward Correction

Weight loss by estimated noise:

```
L = Σᵢ wᵢ · loss(fθ(xᵢ), ỹᵢ)
```

Where wᵢ reflects label confidence.

### Co-Teaching

Two networks teach each other:

```
Network A → Select small-loss samples → Train Network B
Network B → Select small-loss samples → Train Network A
```

Exploits memorization difference for clean vs noisy samples.

## Semi-Supervised Learning

Use unlabeled data with few labels:

### Self-Training

```
1. Train on labeled data
2. Predict on unlabeled data
3. Add confident predictions to training set
4. Repeat
```

### Consistency Regularization

```
L = L_supervised + λ · ||f(x) - f(aug(x))||²
```

Predictions should be consistent under augmentation.

### MixMatch / FixMatch

Combine:
- Pseudo-labeling
- Consistency regularization
- Data augmentation

## Crowdsourcing

Aggregate labels from multiple annotators:

### Majority Vote

```
ŷ = mode(y₁, y₂, ..., yₙ)
```

Simple but ignores annotator quality.

### Dawid-Skene Model

Model annotator reliability:

```
P(yⱼ|y*) = confusion matrix for annotator j
```

EM algorithm estimates true labels and annotator accuracies.

## Quality Estimation

### Label Quality Score

```
Score(x, ỹ) = P(y* = ỹ | x, model)
```

Low scores indicate potential label errors.

### Confident Learning

1. Estimate joint P(y*, ỹ)
2. Identify samples where y* ≠ ỹ
3. Prune, re-weight, or correct

## References

- Ratner, A., et al. (2017). "Snorkel: Rapid Training Data Creation with Weak Supervision." VLDB.
- Han, B., et al. (2018). "Co-teaching: Robust Training of Deep Neural Networks with Extremely Noisy Labels." NeurIPS.
- Northcutt, C., et al. (2021). "Confident Learning: Estimating Uncertainty in Dataset Labels." JAIR.
