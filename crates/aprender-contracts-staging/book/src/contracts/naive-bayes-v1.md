# naive-bayes-v1

**Version:** 1.0.0

Gaussian Naive Bayes — probabilistic classifier assuming feature independence

## References

- Murphy (2012) Machine Learning: A Probabilistic Perspective, §3.5
- Bishop (2006) Pattern Recognition and Machine Learning, §4.2.2

## Equations

### class_prior

$$
P(C_k) = |{i : y_i = k}| / n
$$

**Domain:** $labels \in {0..K-1}ⁿ, n \geq 1, K \geq 2$

**Codomain:** $priors \in (0, 1)^K with \sum P(C_k) = 1$

**Invariants:**

- $P(C_k) \in (0, 1) for each class k present in training$
- $\sum_k P(C_k) = 1 (valid probability distribution)$
- $P(C_k) > 0 for all observed classes$

### gaussian_likelihood

```
P(x_j | C_k) = (1/√(2πσ²_jk)) exp(-(x_j - μ_jk)²/(2σ²_jk))
```

**Domain:** $x_j \in \mathbb{R}, μ_jk \in \mathbb{R}, \sigma²_jk > 0$

**Codomain:** $P(x_j | C_k) > 0$

**Invariants:**

- $Likelihood > 0 (Gaussian PDF is strictly positive)$
- $Log-likelihood is finite for finite inputs and \sigma > 0$

### log_posterior

$$
log P(C_k | x) ∝ log P(C_k) + \sum_j log P(x_j | C_k)
$$

**Domain:** $x \in \mathbb{R}^d, fitted model (priors, means, variances)$

**Codomain:** $unnormalized log-posteriors \in \mathbb{R}^K$

**Invariants:**

- $Predicted class = argmax_k log P(C_k | x)$
- $Posterior probabilities sum to 1 after normalization$
- $Prediction is deterministic for same input$

## Proof Obligations

| # | Type | Property | Formal |
|---|------|----------|--------|
| 1 | invariant | Prior sums to 1 | $\sum_k P(C_k) = 1 after fit$ |
| 2 | bound | Prior bounded | $P(C_k) \in (0, 1) for all observed classes$ |
| 3 | invariant | Posterior probability valid | $Normalized posteriors sum to 1 and each \in [0, 1]$ |
| 4 | invariant | Prediction deterministic | $predict(x) = predict(x) for all x$ |
| 5 | invariant | Fit-predict class range | $predict(x) \in training_classes for all x$ |

## Falsification Tests

| ID | Rule | Prediction | If Fails |
|----|------|------------|----------|
| FALSIFY-NB-001 | Prior sums to 1 | sum of class priors = 1.0 after fit | Normalization not applied or integer truncation |
| FALSIFY-NB-002 | Prior bounded | each prior ∈ (0, 1) | Empty class handling incorrect |
| FALSIFY-NB-003 | Prediction in training classes | all predictions ∈ {classes seen during fit} | Argmax over log-posteriors returns invalid index |
| FALSIFY-NB-004 | Prediction deterministic | predict(x) = predict(x) for same input | Non-deterministic floating point or random state leakage |
| FALSIFY-NB-005 | Separable data high accuracy | accuracy > 0.9 on well-separated Gaussian clusters | Likelihood computation error or variance estimation bug |

## Kani Harnesses

| ID | Obligation | Bound | Strategy |
|----|------------|-------|----------|
| KANI-NB-001 | NB-INV-001 | 8 | stub_float |
| KANI-NB-002 | NB-BND-001 | 8 | stub_float |
| KANI-NAIVE_-003 | Prior sums to 1 | 8 | stub_float |
| KANI-NAIVE_-004 | Prior bounded | 8 | stub_float |
| KANI-NAIVE_-005 | Posterior probability valid | 8 | exhaustive |
| KANI-NAIVE_-006 | Prediction deterministic | 8 | exhaustive |
| KANI-NAIVE_-007 | Fit-predict class range | 8 | exhaustive |

## QA Gate

**Naive Bayes Contract** (F-NB-001)

Gaussian Naive Bayes correctness quality gate

**Checks:** prior_sum_to_one, prior_bounded, prediction_in_classes, prediction_deterministic, separable_high_accuracy

**Pass criteria:** All 5 falsification tests pass + Kani harnesses verify

