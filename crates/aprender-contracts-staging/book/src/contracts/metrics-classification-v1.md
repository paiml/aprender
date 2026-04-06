# metrics-classification-v1

**Version:** 1.0.0

Classification metrics — evaluation measures for discrete predictions

## References

- Manning, Raghavan & Schütze (2008) Introduction to Information Retrieval
- Sokolova & Lapalme (2009) A Systematic Analysis of Performance Measures

## Equations

### accuracy

$$
accuracy = |{i : ŷᵢ = yᵢ}| / n
$$

**Domain:** $y, ŷ \in {0, ..., C-1}ⁿ, n \geq 1$

**Codomain:** $accuracy \in [0, 1]$

**Invariants:**

- $accuracy \in [0, 1] (bounded)$
- $accuracy = 1.0 iff ŷᵢ = yᵢ for all i$
- $accuracy = 0.0 iff ŷᵢ \neq yᵢ for all i$

### confusion_matrix

$$
CM[i,j] = |{k : yₖ = i ∧ ŷₖ = j}|
$$

**Domain:** $y, ŷ \in {0, ..., C-1}ⁿ$

**Codomain:** $CM \in ℕ^{C×C}$

**Invariants:**

- $\sumᵢⱼ CM[i,j] = n (all samples accounted for)$
- $CM[i,j] \geq 0 (non-negative counts)$
- $\sumⱼ CM[i,j] = support(class i)$

### f1_score

$$
F1 = 2 · precision · recall / (precision + recall)
$$

**Domain:** $precision, recall \in [0, 1], precision + recall > 0$

**Codomain:** $F1 \in [0, 1]$

**Invariants:**

- $F1 \in [0, 1]$
- $F1 \leq max(precision, recall) (harmonic \leq arithmetic mean)$
- $F1 = precision = recall when precision = recall$
- $F1 = 0 when precision = 0 or recall = 0$

### precision

$$
precision_c = TP_c / (TP_c + FP_c)
$$

**Domain:** $y, ŷ \in {0, ..., C-1}ⁿ, TP_c + FP_c > 0$

**Codomain:** $precision \in [0, 1]$

**Invariants:**

- $precision \in [0, 1]$
- $precision = 1.0 when FP = 0 and TP > 0$
- $micro_precision = accuracy (for multi-class single-label)$

### recall

$$
recall_c = TP_c / (TP_c + FN_c)
$$

**Domain:** $y, ŷ \in {0, ..., C-1}ⁿ, TP_c + FN_c > 0$

**Codomain:** $recall \in [0, 1]$

**Invariants:**

- $recall \in [0, 1]$
- $recall = 1.0 when FN = 0 and TP > 0$
- $micro_recall = accuracy (for multi-class single-label)$

## Proof Obligations

| # | Type | Property | Formal |
|---|------|----------|--------|
| 1 | bound | Accuracy bounded | $accuracy \in [0, 1] for all y, ŷ$ |
| 2 | bound | Precision bounded | $precision \in [0, 1] for all y, ŷ$ |
| 3 | bound | Recall bounded | $recall \in [0, 1] for all y, ŷ$ |
| 4 | bound | F1 bounded | $F1 \in [0, 1]$ |
| 5 | invariant | F1 harmonic mean property | $F1 \leq max(precision, recall)$ |
| 6 | invariant | Confusion matrix row sums | $\sumᵢⱼ CM[i,j] = n$ |
| 7 | equivalence | Perfect classification identity | $accuracy = 1, precision = 1, recall = 1, F1 = 1 when ŷ = y$ |
| 8 | invariant | Micro-average identity | $micro_precision = micro_recall = accuracy$ |

## Falsification Tests

| ID | Rule | Prediction | If Fails |
|----|------|------------|----------|
| FALSIFY-CM-001 | Accuracy bounded | accuracy ∈ [0, 1] for random predictions | Division by zero or count error |
| FALSIFY-CM-002 | Precision bounded | precision ∈ [0, 1] for all averaging modes | TP/FP counting error or division error |
| FALSIFY-CM-003 | F1 harmonic mean | F1 ≤ max(precision, recall) for all inputs | Harmonic mean formula incorrect |
| FALSIFY-CM-004 | Confusion matrix conservation | sum(CM) = n for all inputs | Sample missed or double-counted |
| FALSIFY-CM-005 | Perfect classification | accuracy = precision = recall = F1 = 1 when ŷ = y | Edge case in perfect prediction handling |
| FALSIFY-CM-006 | Micro-average identity | micro_precision = micro_recall = accuracy | Micro-averaging aggregation error |
| FALSIFY-CM-007 | Recall bounded | recall ∈ [0, 1] for all averaging modes | TP/FN counting error or division error |
| FALSIFY-CM-008 | F1 bounded | F1 ∈ [0, 1] for arbitrary precision and recall inputs | Harmonic mean overflow or division by zero in F1 computation |

## Kani Harnesses

| ID | Obligation | Bound | Strategy |
|----|------------|-------|----------|
| KANI-CM-001 | 1 | 4 | exhaustive |
| KANI-CM-002 | 2 | 4 | exhaustive |
| KANI-METRIC-003 | Accuracy bounded | 8 | stub_float |
| KANI-METRIC-004 | Precision bounded | 8 | stub_float |
| KANI-METRIC-005 | Recall bounded | 8 | stub_float |
| KANI-METRIC-006 | F1 bounded | 8 | stub_float |
| KANI-METRIC-007 | F1 harmonic mean property | 8 | exhaustive |
| KANI-METRIC-008 | Confusion matrix row sums | 8 | stub_float |
| KANI-METRIC-009 | Perfect classification identity | 8 | exhaustive |
| KANI-METRIC-010 | Micro-average identity | 8 | exhaustive |

## QA Gate

**Classification Metrics Contract** (F-CM-001)

Classification metric correctness quality gate

**Checks:** accuracy_bounded, precision_bounded, f1_harmonic_mean, confusion_matrix_conservation, perfect_classification, micro_average_identity

**Pass criteria:** All 8 falsification tests pass

