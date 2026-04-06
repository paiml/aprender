# decision-tree-v1

**Version:** 1.0.0

Decision tree — CART algorithm with Gini impurity and MSE splitting

## References

- Breiman, Friedman, Olshen, Stone (1984) Classification and Regression Trees
- Hastie, Tibshirani, Friedman (2009) Elements of Statistical Learning, §9.2

## Equations

### gini_impurity

$$
G(S) = 1 - \sum_k p_k² where p_k = |S_k|/|S|
$$

**Domain:** $S = multiset of class labels, |S| \geq 1$

**Codomain:** $G \in [0, 1 - 1/K] ⊂ [0, 1)$

**Invariants:**

- $G \in [0, 1) (bounded by construction)$
- $G = 0 iff all elements have the same class (pure node)$
- $G is maximal when all classes equally represented: G = 1 - 1/K$

### gini_split

```
G_split = (|S_L|/|S|)G(S_L) + (|S_R|/|S|)G(S_R)
```

**Domain:** $S partitioned into S_L, S_R with |S_L|, |S_R| \geq 1$

**Codomain:** $G_split \in [0, 1)$

**Invariants:**

- $G_split \leq G(S) (splitting never increases impurity)$
- $G_split \in [0, 1)$
- $G_split = 0 iff both children are pure$

### mse_split

$$
MSE(S) = (1/|S|) \sum(y_i - ȳ)² where ȳ = mean(S)
$$

**Domain:** $S = set of real-valued targets, |S| \geq 1$

**Codomain:** $MSE \in [0, ∞)$

**Invariants:**

- $MSE \geq 0 (sum of squares)$
- $MSE = 0 iff all targets identical$
- $MSE = Var(S) (variance of the target set)$

### prediction

$$
Classifier: majority_class(leaf), Regressor: mean(leaf_targets)
$$

**Domain:** $x \in \mathbb{R}^d (feature vector), fitted tree$

**Codomain:** $ŷ \in classes (classifier) or ŷ \in \mathbb{R} (regressor)$

**Invariants:**

- $Prediction is deterministic for same input$
- $Prediction depends only on features used in splits along root-to-leaf path$

## Proof Obligations

| # | Type | Property | Formal |
|---|------|----------|--------|
| 1 | bound | Gini bounded | $G(S) \in [0, 1) for all non-empty S$ |
| 2 | invariant | Gini pure node | $G(S) = 0 iff \|unique(S)\| = 1$ |
| 3 | invariant | Gini split reduction | `G_split(S_L, S_R) ≤ G(S) for any partition` |
| 4 | bound | MSE non-negative | $MSE(S) \geq 0 for all S$ |
| 5 | invariant | MSE zero for constant | $all targets identical ⟹ MSE = 0$ |
| 6 | invariant | Prediction deterministic | $predict(x, tree) = predict(x, tree) for all x$ |
| 7 | invariant | Fit-predict consistency | $Trained classifier predicts only observed classes$ |

## Falsification Tests

| ID | Rule | Prediction | If Fails |
|----|------|------------|----------|
| FALSIFY-DT-001 | Gini bounded | gini_impurity ∈ [0, 1) for random label sets | Division by zero for empty set or normalization error |
| FALSIFY-DT-002 | Gini pure node | gini_impurity = 0 when all labels identical | Off-by-one in class counting |
| FALSIFY-DT-003 | Gini split reduction | weighted child Gini ≤ parent Gini | Weight calculation error or incorrect split |
| FALSIFY-DT-004 | MSE non-negative | MSE ≥ 0 for random target sets | Negative variance from catastrophic cancellation |
| FALSIFY-DT-005 | MSE zero for constant | MSE ≈ 0 when all targets identical | Float drift in mean computation |
| FALSIFY-DT-006 | Prediction deterministic | predict(x, tree) = predict(x, tree) for same input | Non-deterministic state in tree traversal |
| FALSIFY-DT-007 | Fit-predict class range | Classifier predictions ⊆ training classes | Leaf prediction uses incorrect class label |

## Kani Harnesses

| ID | Obligation | Bound | Strategy |
|----|------------|-------|----------|
| KANI-DT-001 | DT-BND-001 | 8 | stub_float |
| KANI-DT-002 | DT-BND-002 | 8 | stub_float |
| KANI-DECISI-003 | Gini bounded | 8 | stub_float |
| KANI-DECISI-004 | Gini pure node | 8 | exhaustive |
| KANI-DECISI-005 | Gini split reduction | 8 | exhaustive |
| KANI-DECISI-006 | MSE non-negative | 8 | exhaustive |
| KANI-DECISI-007 | MSE zero for constant | 8 | exhaustive |
| KANI-DECISI-008 | Prediction deterministic | 8 | exhaustive |
| KANI-DECISI-009 | Fit-predict consistency | 8 | exhaustive |

## QA Gate

**Decision Tree Contract** (F-DT-001)

Decision tree correctness quality gate

**Checks:** gini_bounded, gini_pure_zero, gini_split_reduction, mse_non_negative, mse_constant_zero, prediction_deterministic, fit_predict_class_range

**Pass criteria:** All 7 falsification tests pass + Kani harnesses verify

