# loss-functions-v1

**Version:** 1.0.0

Loss functions — differentiable objective functions for neural network training

## References

- Bishop (2006) Pattern Recognition and Machine Learning
- Goodfellow, Bengio & Courville (2016) Deep Learning

## Equations

### bce

$$
BCE = -(1/n) \sum[yᵢ·\log(ŷᵢ) + (1-yᵢ)·\log(1-ŷᵢ)]
$$

**Domain:** $y \in {0,1}ⁿ, ŷ \in (0,1)ⁿ, n \geq 1$

**Codomain:** $BCE \in [0, ∞)$

**Invariants:**

- $BCE \geq 0 (non-negativity from -log on (0,1))$
- $BCE = 0 iff ŷᵢ = yᵢ for all i (perfect prediction)$
- $BCE \to ∞ as ŷ \to 0 for y=1, or ŷ \to 1 for y=0$

### huber

$$
L_\delta(a) = ½a² if |a| \leq \delta, else \delta(|a| - ½\delta)
$$

**Domain:** $a = y - ŷ \in \mathbb{R}, \delta > 0$

**Codomain:** $L_\delta \in [0, ∞)$

**Invariants:**

- $L_\delta \geq 0 (non-negativity)$
- $L_\delta = 0 iff a = 0$
- $L_\delta is differentiable everywhere (C¹ smooth)$
- $L_\delta \to ½a² as \delta \to ∞ (approaches MSE)$
- $L_\delta \to \delta|a| as \delta \to 0 (approaches MAE)$

### l1_loss

$$
L1 = (1/n) \sum|yᵢ - ŷᵢ|
$$

**Domain:** $y, ŷ \in \mathbb{R}ⁿ, n \geq 1$

**Codomain:** $L1 \in [0, ∞)$

**Invariants:**

- $L1 \geq 0$
- $L1 = 0 iff ŷ = y$
- $L1(y, ŷ) = L1(ŷ, y) (symmetry)$
- $L1 = MAE (identical function)$

### mse_loss

$$
MSE = (1/n) \sum(yᵢ - ŷᵢ)²
$$

**Domain:** $y, ŷ \in \mathbb{R}ⁿ, n \geq 1$

**Codomain:** $MSE \in [0, ∞)$

**Invariants:**

- $MSE \geq 0$
- $MSE = 0 iff ŷ = y$
- $gradient: ∂MSE/∂ŷ = 2(ŷ-y)/n$

### nll

$$
NLL = -(1/n) \sum \log(p_{yᵢ}) where p = softmax(logits)
$$

**Domain:** $logits \in \mathbb{R}^{n×C}, y \in {0..C-1}ⁿ$

**Codomain:** $NLL \in [0, ∞)$

**Invariants:**

- $NLL \geq 0 (non-negativity from -log of probability)$
- $NLL = 0 iff predicted probability of true class = 1$
- $NLL \geq -\log(1/C) for uniform predictions$

### smooth_l1

$$
SL1(a) = ½a²/\beta if |a| < \beta, else |a| - ½\beta
$$

**Domain:** $a = y - ŷ \in \mathbb{R}, \beta > 0$

**Codomain:** $SL1 \in [0, ∞)$

**Invariants:**

- $SL1 \geq 0$
- $SL1 = 0 iff a = 0$
- $SL1 is C¹ smooth$

## Proof Obligations

| # | Type | Property | Formal |
|---|------|----------|--------|
| 1 | bound | All losses non-negative | $L(y, ŷ) \geq 0 for all loss functions$ |
| 2 | equivalence | Zero loss at perfect prediction | $L(y, y) = 0 for all y$ |
| 3 | invariant | BCE monotonicity | $BCE increases as predictions diverge from targets$ |
| 4 | invariant | Huber smoothness | $Huber loss is C¹ at transition point \|a\| = \delta$ |
| 5 | invariant | L1 symmetry | $L1(y, ŷ) = L1(ŷ, y)$ |
| 6 | bound | NLL lower bound | $NLL \geq 0$ |

## Falsification Tests

| ID | Rule | Prediction | If Fails |
|----|------|------------|----------|
| FALSIFY-LF-001 | Non-negativity | All loss values ≥ 0 for random inputs | Sign error in loss formula |
| FALSIFY-LF-002 | Zero at perfect prediction | L(y, y) = 0 for all loss functions | Floating-point accumulation error |
| FALSIFY-LF-003 | BCE non-negativity | BCE ≥ 0 for y ∈ {0,1}, ŷ ∈ (0.001, 0.999) | Log of value outside (0,1) |
| FALSIFY-LF-004 | Huber transition continuity | \|L_δ(δ+ε) - L_δ(δ-ε)\| < 2ε for small ε | Discontinuity at Huber transition point |
| FALSIFY-LF-005 | L1 symmetry | L1(y, ŷ) = L1(ŷ, y) for random inputs | Argument-order dependence |
| FALSIFY-LF-006 | NLL lower bound | NLL ≥ 0 for random logits and valid class indices | Softmax normalization error or log of non-positive value |

## Kani Harnesses

| ID | Obligation | Bound | Strategy |
|----|------------|-------|----------|
| KANI-LF-001 | 1 | 4 | stub_float |
| KANI-LF-002 | 2 | 4 | stub_float |
| KANI-LOSS_F-003 | All losses non-negative | 8 | exhaustive |
| KANI-LOSS_F-004 | Zero loss at perfect prediction | 8 | exhaustive |
| KANI-LOSS_F-005 | BCE monotonicity | 8 | exhaustive |
| KANI-LOSS_F-006 | Huber smoothness | 8 | exhaustive |
| KANI-LOSS_F-007 | L1 symmetry | 8 | exhaustive |
| KANI-LOSS_F-008 | NLL lower bound | 8 | stub_float |

## QA Gate

**Loss Functions Contract** (F-LF-001)

Loss function correctness quality gate

**Checks:** non_negativity, zero_at_perfect, bce_non_negativity, huber_continuity, l1_symmetry, nll_lower_bound

**Pass criteria:** All 6 falsification tests pass

