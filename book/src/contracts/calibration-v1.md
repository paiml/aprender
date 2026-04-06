# calibration-v1

**Version:** 1.0.0

Calibration metrics — evaluation and correction of probabilistic predictions

## References

- Naeini, Cooper & Hauskrecht (2015) Obtaining Well Calibrated Probabilities Using Bayesian Binning into Quantiles
- Guo et al. (2017) On Calibration of Modern Neural Networks
- Platt (1999) Probabilistic Outputs for Support Vector Machines

## Equations

### expected_calibration_error

$$
ECE = \sum_b (|B_b|/n) |acc(B_b) - conf(B_b)|
$$

**Domain:** $probabilities \in [0,1]ⁿ, labels \in {0,1}ⁿ, n_bins \geq 1$

**Codomain:** $ECE \in [0, 1]$

**Invariants:**

- $ECE \in [0, 1] (weighted average of absolute differences, each in [0,1])$
- $ECE = 0 for perfectly calibrated predictions$
- $ECE monotone in calibration deviation$

### isotonic_regression

$$
ĝ = argmin_{g monotone} \sum(g(f_i) - y_i)²
$$

**Domain:** $f \in [0,1]ⁿ (probabilities), labels \in {0,1}ⁿ$

**Codomain:** $calibrated_probs \in [0, 1]ⁿ$

**Invariants:**

- $Output is monotone non-decreasing$
- $Output \in [0, 1]$
- $Isotonic fit minimizes sum of squared residuals among monotone functions$

### maximum_calibration_error

$$
MCE = max_b |acc(B_b) - conf(B_b)|
$$

**Domain:** $probabilities \in [0,1]ⁿ, labels \in {0,1}ⁿ, n_bins \geq 1$

**Codomain:** $MCE \in [0, 1]$

**Invariants:**

- $MCE \in [0, 1] (absolute difference of values in [0,1])$
- $MCE \geq ECE (max \geq weighted average)$
- $MCE = 0 for perfectly calibrated predictions$

### platt_scaling

```
σ(Af + B) where A,B = argmin -Σ[t_i log(σ(Af_i+B)) + (1-t_i)log(1-σ(Af_i+B))]
```

**Domain:** $f \in \mathbb{R}ⁿ (logits), labels \in {0,1}ⁿ$

**Codomain:** $calibrated_probs \in (0, 1)ⁿ$

**Invariants:**

- $Output probabilities \in (0, 1)$
- $Monotone: f_i > f_j and A > 0 ⟹ \sigma(Af_i+B) > \sigma(Af_j+B)$

### reliability_diagram

```
For each bin b: (mean_confidence(B_b), mean_accuracy(B_b))
```

**Domain:** $probabilities \in [0,1]ⁿ, labels \in {0,1}ⁿ, n_bins \geq 1$

**Codomain:** `bins: Vec<(confidence ∈ [0,1], accuracy ∈ [0,1])>`

**Invariants:**

- $Bin confidence \in [0, 1]$
- $Bin accuracy \in [0, 1]$
- $Perfect calibration: all bins lie on the diagonal (confidence \approx accuracy)$

## Proof Obligations

| # | Type | Property | Formal |
|---|------|----------|--------|
| 1 | bound | ECE bounded | $ECE \in [0, 1] for all valid probability-label pairs$ |
| 2 | bound | MCE bounded | $MCE \in [0, 1] for all valid probability-label pairs$ |
| 3 | invariant | MCE dominates ECE | $MCE \geq ECE for any binning$ |
| 4 | invariant | Perfect calibration zero error | $perfectly calibrated ⟹ ECE = 0 ∧ MCE = 0$ |
| 5 | bound | Platt output bounded | $\sigma(Af+B) \in (0, 1) for all f \in \mathbb{R}$ |
| 6 | invariant | Isotonic monotonicity | $f_i \leq f_j ⟹ ĝ(f_i) \leq ĝ(f_j)$ |
| 7 | bound | Reliability bin bounds | $confidence, accuracy \in [0, 1] for all bins$ |

## Falsification Tests

| ID | Rule | Prediction | If Fails |
|----|------|------------|----------|
| FALSIFY-CAL-001 | ECE bounded | ECE ∈ [0, 1] for random probabilities and labels | Bin weight normalization error |
| FALSIFY-CAL-002 | MCE bounded | MCE ∈ [0, 1] for random probabilities and labels | Max over empty bins or unnormalized |
| FALSIFY-CAL-003 | MCE dominates ECE | MCE ≥ ECE for same inputs | Different binning used for ECE vs MCE |
| FALSIFY-CAL-004 | Perfect calibration zero | ECE ≈ 0 and MCE ≈ 0 when predictions match empirical frequencies | Off-by-one in bin boundary or floating point accumulation |
| FALSIFY-CAL-005 | Platt output bounded | All Platt-scaled outputs in (0, 1) | Sigmoid overflow/underflow |
| FALSIFY-CAL-006 | Isotonic monotonicity | Isotonic regression output is monotone non-decreasing | Isotonic regression solver violated monotonicity constraint |
| FALSIFY-CAL-007 | Reliability bin bounds | All bin (confidence, accuracy) pairs in [0,1]² | Division by zero in empty bins |

## Kani Harnesses

| ID | Obligation | Bound | Strategy |
|----|------------|-------|----------|
| KANI-CAL-001 | CAL-BND-001 | 8 | stub_float |
| KANI-CAL-002 | CAL-BND-002 | 8 | stub_float |
| KANI-CALIBR-003 | ECE bounded | 8 | stub_float |
| KANI-CALIBR-004 | MCE bounded | 8 | stub_float |
| KANI-CALIBR-005 | MCE dominates ECE | 8 | exhaustive |
| KANI-CALIBR-006 | Perfect calibration zero error | 8 | exhaustive |
| KANI-CALIBR-007 | Platt output bounded | 8 | stub_float |
| KANI-CALIBR-008 | Isotonic monotonicity | 8 | exhaustive |
| KANI-CALIBR-009 | Reliability bin bounds | 8 | stub_float |

## QA Gate

**Calibration Metrics Contract** (F-CAL-001)

Calibration metric correctness quality gate

**Checks:** ece_bounded, mce_bounded, mce_dominates_ece, perfect_calibration_zero, platt_bounded, isotonic_monotonicity, reliability_bounded

**Pass criteria:** All 7 falsification tests pass + Kani harnesses verify

