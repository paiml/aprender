# drift-detection-v1

**Version:** 1.0.0

Data drift detection -- univariate and performance drift with threshold-based classification

## References

- Gama et al. (2004) Learning with Drift Detection, SBIA
- Webb et al. (2016) Characterizing Concept Drift, DMKD

## Equations

### classify_drift

$$
status = NoDrift if score < warn_threshold, Warning if score < drift_threshold, Drift otherwise
$$

**Domain:** $score >= 0, 0 < warn_threshold < drift_threshold$

**Codomain:** $status in {NoDrift, Warning, Drift}$

**Invariants:**

- $NoDrift < Warning < Drift (ordered severity)$
- $Thresholds partition [0, infinity) into exactly 3 regions$
- $score = 0 always yields NoDrift$

### min_samples_guard

$$
detect(data) = NoDrift if |data| < min_samples
$$

**Domain:** $data in R^n, min_samples >= 1$

**Codomain:** $DriftStatus$

**Invariants:**

- $Insufficient data never triggers drift alarm$
- $min_samples is a strict lower bound$

### performance_drift

$$
perf_drift = |metric_ref - metric_cur| / metric_ref
$$

**Domain:** $metric_ref > 0, metric_cur >= 0$

**Codomain:** $perf_drift in [0, infinity)$

**Invariants:**

- $perf_drift >= 0$
- $perf_drift = 0 when metric_ref = metric_cur$

### univariate_drift

$$
drift_score = |mu_ref - mu_cur| / sigma_ref
$$

**Domain:** $mu_ref, mu_cur in R, sigma_ref > 0$

**Codomain:** $drift_score in [0, infinity)$

**Invariants:**

- $drift_score >= 0 (absolute value divided by positive sigma)$
- $drift_score = 0 when mu_ref = mu_cur (no drift)$
- $Larger shift produces larger score$

## Proof Obligations

| # | Type | Property | Formal |
|---|------|----------|--------|
| 1 | bound | Drift score non-negative | $drift_score >= 0 for all inputs$ |
| 2 | invariant | DriftStatus transitions correct | $NoDrift if score < warn, Warning if warn <= score < drift, Drift if score >= drift$ |
| 3 | invariant | min_samples respected | $\|data\| < min_samples implies status = NoDrift$ |
| 4 | invariant | Identical distributions yield NoDrift | $mu_ref = mu_cur implies drift_score = 0 implies NoDrift$ |

## Falsification Tests

| ID | Rule | Prediction | If Fails |
|----|------|------------|----------|
| FALSIFY-DRIFT-001 | Drift score non-negative | drift_score >= 0 for all inputs | Sign error in drift score computation |
| FALSIFY-DRIFT-002 | DriftStatus ordering | Higher scores produce equal or higher severity status | Threshold comparison logic inverted or off-by-one |
| FALSIFY-DRIFT-003 | min_samples guard | Data below min_samples always returns NoDrift | min_samples guard bypassed or not checked |
| FALSIFY-DRIFT-004 | Identical distribution no drift | Same distribution as reference yields NoDrift | Floating-point rounding in mean/std computation |

## Kani Harnesses

| ID | Obligation | Bound | Strategy |
|----|------------|-------|----------|
| KANI-DRIFT-001 | DRIFT-BND-001 | 8 | stub_float |
| KANI-DRIFT-002 | DRIFT-INV-001 | 8 | stub_float |
| KANI-DRIFT_-003 | Drift score non-negative | 8 | exhaustive |
| KANI-DRIFT_-004 | DriftStatus transitions correct | 8 | exhaustive |
| KANI-DRIFT_-005 | min_samples respected | 8 | exhaustive |
| KANI-DRIFT_-006 | Identical distributions yield NoDrift | 8 | exhaustive |

## QA Gate

**Drift Detection Contract** (F-DRIFT-001)

Data drift detection correctness quality gate

**Checks:** drift_score_nonneg, status_ordering, min_samples_guard, identical_no_drift

**Pass criteria:** All 4 falsification tests pass + Kani harnesses verify

