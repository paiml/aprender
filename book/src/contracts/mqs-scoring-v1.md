# mqs-scoring-v1

**Version:** 1.0.0

Model Quality Score (MQS) — composite quality metric for ML model certification (QUAL+PERF+STAB+COMP+EDGE+REGR)

## References

- apr-model-qa-playbook — production model quality assurance pipeline
- Breck et al. (2017) ML Test Score: A Rubric for ML Production Readiness
- Sculley et al. (2015) Hidden Technical Debt in Machine Learning Systems

## Equations

### mqs_composite

```
mqs: ModelEvidence -> MqsResult
  MqsResult {
    raw: f64,        -- sum of dimension scores (0-1050)
    normalized: f64, -- raw / 10.5 mapped to [0, 100]
    grade: Grade,    -- A+/A/A-/B+/.../F
    dimensions: DimensionBreakdown,
  }
  raw = QUAL + PERF + STAB + COMP + EDGE + REGR
  Where each dimension in [0, 175]:
    QUAL = quality_checks_passed / quality_checks_total * 175
    PERF = performance_within_budget ? latency_ratio * 175 : 0
    STAB = stability_variance < threshold ? (1 - variance/threshold) * 175 : 0
    COMP = compatibility_checks_passed / compatibility_checks_total * 175
    EDGE = edge_cases_passed / edge_cases_total * 175
    REGR = regression_tests_passed / regression_tests_total * 175

```

**Domain:** $ModelEvidence with non-negative check counts, dimension totals > 0$

**Codomain:** $MqsResult with raw in [0, 1050], normalized in [0, 100]$

**Invariants:**

- $0 <= raw <= 1050 (6 dimensions * 175 max each)$
- $0 <= normalized <= 100$
- $raw = sum of all 6 dimension scores$
- $Each dimension score in [0, 175]$

### mqs_deterministic

```
deterministic: ModelEvidence -> bool
  For all e: mqs(e).raw == mqs(e).raw (same evidence, same score)

```

**Domain:** $Any valid ModelEvidence$

**Codomain:** $bool = true$

**Invariants:**

- $No randomness in scoring pipeline$
- $Floating point operations are deterministic (same platform)$

### mqs_grade

$$
grade: normalized -> Grade
  A+ if normalized >= 97
  A  if normalized >= 93
  A- if normalized >= 90
  B+ if normalized >= 85
  B  if normalized >= 80
  C  if normalized >= 70
  D  if normalized >= 60
  F  otherwise

$$

**Domain:** $normalized in [0, 100]$

**Codomain:** $Grade in {A+, A, A-, B+, B, C, D, F}$

**Invariants:**

- $Grade monotonically non-decreasing with normalized score$

### mqs_pass_rate

```
mqs_pass_rate: Vec<MqsResult> -> f64
  pass_rate = models_passing_all_gates / total_models_evaluated
  Where passing = normalized >= pass_threshold (default: 70.0)

```

**Domain:** `Vec<MqsResult> with len >= 1`

**Codomain:** $pass_rate in [0.0, 1.0]$

**Invariants:**

- $pass_rate = 1.0 iff all models pass$
- $pass_rate = 0.0 iff no models pass$

## Proof Obligations

| # | Type | Property | Formal |
|---|------|----------|--------|
| 1 | bound | MQS raw bounded | $0 <= mqs(e).raw <= 1050 for all valid ModelEvidence e$ |
| 2 | bound | MQS normalized bounded | $0 <= mqs(e).normalized <= 100 for all valid ModelEvidence e$ |
| 3 | invariant | Deterministic scoring | `mqs(e1) == mqs(e2) when e1 == e2` |
| 4 | invariant | Dimension sum | $raw = QUAL + PERF + STAB + COMP + EDGE + REGR$ |
| 5 | monotonicity | Grade monotonic | $normalized(a) > normalized(b) => grade(a) >= grade(b)$ |
| 6 | postcondition | Pass rate bounded | `0.0 <= mqs_pass_rate(results) <= 1.0` |

## Falsification Tests

| ID | Rule | Prediction | If Fails |
|----|------|------------|----------|
| FALSIFY-MQS-001 | Raw score bounded | MQS raw score always in [0, 1050] | Dimension weights or normalization incorrect |
| FALSIFY-MQS-002 | Normalized bounded | MQS normalized always in [0, 100] | Normalization divisor incorrect |
| FALSIFY-MQS-003 | Deterministic | Same ModelEvidence always produces same MqsResult | Non-deterministic state or randomness in scorer |
| FALSIFY-MQS-004 | All-pass yields max raw | Perfect scores in all 6 dimensions yields raw=1050 | Dimension weight calculation error |
| FALSIFY-MQS-005 | Zero checks yields zero | Zero passed checks yields raw=0 | Division by zero or default score not zero |
| FALSIFY-MQS-006 | Pass rate bounded | mqs_pass_rate always in [0.0, 1.0] | Pass rate calculation denominator error |

## Kani Harnesses

| ID | Obligation | Bound | Strategy |
|----|------------|-------|----------|
| KANI-MQS-001 | MQS raw bounded | 16 | exhaustive |
| KANI-MQS-002 | MQS normalized bounded | 16 | exhaustive |
| KANI-MQS-003 | Dimension sum | 8 | exhaustive |
| KANI-MQS_SC-004 | Deterministic scoring | 8 | exhaustive |
| KANI-MQS_SC-005 | Grade monotonic | 8 | exhaustive |
| KANI-MQS_SC-006 | Pass rate bounded | 8 | stub_float |

## QA Gate

**mqs-scoring-v1 Contract** (F-MSV-001)

Quality gate for Model Quality Score (MQS) — composite quality metric for ML 

**Checks:** validation, falsification

