# preprocessing-normalization-v1

**Version:** 1.0.0

Preprocessing normalization — data scaling and standardization transforms

## References

- Scikit-learn: Preprocessing data (StandardScaler, MinMaxScaler)
- Bishop (2006) Pattern Recognition and Machine Learning, §1.1

## Equations

### minmax_scaler

$$
x_scaled = (x - x_min) / (x_max - x_min) * (max - min) + min
$$

**Domain:** $X \in \mathbb{R}^{n×d}, target range [min, max]$

**Codomain:** $X_scaled \in [min, max]^{n×d} (for training data)$

**Invariants:**

- $X_scaled \in [min, max] for training data (exact bounds)$
- $x_min maps to min, x_max maps to max$
- $Inverse transform recovers original$
- $Monotone: x_i \leq x_j ⟹ scaled(x_i) \leq scaled(x_j)$

### robust_scaler

$$
z = (x - median) / IQR where IQR = Q3 - Q1
$$

**Domain:** $X \in \mathbb{R}^{n×d}, n \geq 4$

**Codomain:** $Z \in \mathbb{R}^{n×d}$

**Invariants:**

- $median(Z_j) \approx 0 for each feature j$
- $IQR(Z_j) \approx 1 for each feature j (when IQR > 0)$
- $Robust to outliers (only uses quartiles)$

### standard_scaler

$$
z = (x - μ) / \sigma where μ = mean(X), \sigma = std(X)
$$

**Domain:** $X \in \mathbb{R}^{n×d}, n \geq 2$

**Codomain:** $Z \in \mathbb{R}^{n×d}$

**Invariants:**

- $mean(Z_j) \approx 0 for each feature j (within float tolerance)$
- $std(Z_j) \approx 1 for each feature j (when \sigma_j > 0)$
- $Inverse transform recovers original: x = z * \sigma + μ$

## Proof Obligations

| # | Type | Property | Formal |
|---|------|----------|--------|
| 1 | invariant | StandardScaler zero mean | $\|mean(Z_j)\| < \varepsilon for each feature j$ |
| 2 | invariant | StandardScaler unit variance | $\|std(Z_j) - 1\| < \varepsilon for each feature j where \sigma_j > \varepsilon$ |
| 3 | bound | MinMaxScaler bounded | $X_scaled \in [min, max] for training data$ |
| 4 | invariant | MinMaxScaler extremes | $scaled(x_min) = min, scaled(x_max) = max$ |
| 5 | invariant | StandardScaler inverse | $inverse_transform(transform(X)) \approx X$ |

## Falsification Tests

| ID | Rule | Prediction | If Fails |
|----|------|------------|----------|
| FALSIFY-PP-001 | StandardScaler zero mean | mean of each standardized feature ≈ 0 | Mean subtraction not applied per-feature |
| FALSIFY-PP-002 | StandardScaler unit variance | std of each standardized feature ≈ 1 | Division by std not applied or Bessel correction wrong |
| FALSIFY-PP-003 | MinMaxScaler bounded | all transformed values in [min, max] | Range computation error or off-by-one |
| FALSIFY-PP-004 | MinMaxScaler extremes | min(X) maps to min, max(X) maps to max | Boundary condition not handled |
| FALSIFY-PP-005 | StandardScaler inverse roundtrip | inverse_transform(transform(X)) ≈ X | Inverse formula incorrect |
| FALSIFY-PP-006 | MinMaxScaler inverse roundtrip | inverse_transform(transform(X)) ≈ X | Inverse formula incorrect |

## Kani Harnesses

| ID | Obligation | Bound | Strategy |
|----|------------|-------|----------|
| KANI-PP-001 | PP-INV-001 | 8 | stub_float |
| KANI-PP-002 | PP-BND-001 | 8 | stub_float |
| KANI-PREPRO-003 | StandardScaler zero mean | 8 | exhaustive |
| KANI-PREPRO-004 | StandardScaler unit variance | 8 | exhaustive |
| KANI-PREPRO-005 | MinMaxScaler bounded | 8 | stub_float |
| KANI-PREPRO-006 | MinMaxScaler extremes | 8 | exhaustive |
| KANI-PREPRO-007 | StandardScaler inverse | 8 | exhaustive |
| KANI-PREPRO-008 | MinMaxScaler inverse | 8 | exhaustive |

## QA Gate

**Preprocessing Normalization Contract** (F-PP-001)

Preprocessing normalization correctness quality gate

**Checks:** standard_zero_mean, standard_unit_variance, minmax_bounded, minmax_extremes, standard_inverse, minmax_inverse

**Pass criteria:** All 6 falsification tests pass + Kani harnesses verify

