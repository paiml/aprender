# active-learning-v1

**Version:** 1.0.0

Active learning query strategies for label-efficient training

## References

- Settles (2012) Active Learning, Synthesis Lectures on AI and ML
- Lewis & Gale (1994) A Sequential Algorithm for Training Text Classifiers

## Equations

### entropy_score

```
H(p) = -sum_i(p_i * ln(p_i))
```

**Domain:** $p in R^k, p_i in [0, 1], sum(p_i) = 1, 0*ln(0) defined as 0$

**Codomain:** $H >= 0$

**Invariants:**

- $Entropy is 0 for degenerate distributions (single class has probability 1)$
- $Entropy is maximized at ln(k) for uniform distribution$
- $Entropy is always non-negative$

### margin_score

$$
m(p) = 1 - (p_(1) - p_(2))
$$

**Domain:** $p sorted descending, p_i in [0, 1], sum(p_i) = 1, |p| >= 2$

**Codomain:** $m in [0, 1]$

**Invariants:**

- $Score is 0 when top class has probability 1 (maximum margin)$
- $Score is 1 when top two classes have equal probability (zero margin)$
- $Score is always in [0, 1] for valid probability vectors$

### qbc_score

$$
H_vote(x) = -sum_c(V(c)/C * ln(V(c)/C))
$$

**Domain:** $V(c) = vote count for class c, C = committee size, sum(V(c)) = C$

**Codomain:** $H_vote >= 0$

**Invariants:**

- $Vote entropy is 0 when all committee members agree$
- $Vote entropy is maximized when votes are uniformly split$
- $Vote entropy is always non-negative$

### uncertainty_score

```
u(p) = 1 - max_i(p_i)
```

**Domain:** $p in R^k, p_i in [0, 1], sum(p_i) = 1$

**Codomain:** $u in [0, 1]$

**Invariants:**

- $Score is 0 when model is perfectly confident (one class has probability 1)$
- $Score is 1 - 1/k when uniform distribution over k classes$
- $Score is always in [0, 1] for valid probability vectors$

## Proof Obligations

| # | Type | Property | Formal |
|---|------|----------|--------|
| 1 | bound | Uncertainty score in [0, 1] | $forall p valid prob vec: 0 <= u(p) <= 1$ |
| 2 | bound | Margin score in [0, 1] | $forall p valid prob vec with \|p\| >= 2: 0 <= m(p) <= 1$ |
| 3 | bound | Entropy is non-negative | $forall p valid prob vec: H(p) >= 0$ |
| 4 | bound | Vote entropy is non-negative | $forall committee predictions: H_vote >= 0$ |
| 5 | invariant | Higher uncertainty selects more ambiguous samples | $u(uniform(k)) >= u(one_hot(k)) for all k >= 2$ |

## Falsification Tests

| ID | Rule | Prediction | If Fails |
|----|------|------------|----------|
| FALSIFY-AL-001 | Uncertainty score bounds | 0 <= u(p) <= 1 for any valid probability vector | Uncertainty score formula violates [0, 1] range |
| FALSIFY-AL-002 | Margin score bounds | 0 <= m(p) <= 1 for any valid probability vector with >= 2 classes | Margin score formula violates [0, 1] range |
| FALSIFY-AL-003 | Entropy non-negativity | H(p) >= 0 for any valid probability vector | Entropy computation produces negative values (numerical error) |
| FALSIFY-AL-004 | Vote entropy non-negativity | H_vote >= 0 for any committee vote distribution | Vote entropy computation produces negative values |
| FALSIFY-AL-005 | Uncertainty monotonicity | Uniform distribution has higher uncertainty than one-hot | Uncertainty does not rank ambiguous samples higher |
| FALSIFY-AL-006 | Entropy finiteness | H(p) is finite for any valid probability vector | 0*ln(0) not handled correctly, produces NaN or Inf |

## Kani Harnesses

| ID | Obligation | Bound | Strategy |
|----|------------|-------|----------|
| KANI-AL-001 | AL-BND-001 | 8 | stub_float |
| KANI-AL-002 | AL-BND-002 | 8 | stub_float |
| KANI-AL-003 | AL-BND-003 | 8 | stub_float |
| KANI-ACTIVE-004 | Uncertainty score in [0, 1] | 8 | exhaustive |
| KANI-ACTIVE-005 | Margin score in [0, 1] | 8 | exhaustive |
| KANI-ACTIVE-006 | Entropy is non-negative | 8 | exhaustive |
| KANI-ACTIVE-007 | Vote entropy is non-negative | 8 | exhaustive |
| KANI-ACTIVE-008 | Higher uncertainty selects more ambiguous samples | 8 | exhaustive |

## QA Gate

**Active Learning Contract** (F-AL-001)

Active learning query strategy correctness quality gate

**Checks:** uncertainty_bounds, margin_bounds, entropy_non_negativity, vote_entropy_non_negativity, uncertainty_monotonicity, entropy_finiteness

**Pass criteria:** All 6 falsification tests pass + Kani harnesses verify

