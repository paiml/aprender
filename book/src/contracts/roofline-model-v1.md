# roofline-model-v1

**Version:** 1.0.0

Roofline model — performance bound analysis for LLM inference

## References

- Williams et al. (2009) Roofline: An Insightful Visual Performance Model
- Qwen3 Performance Parity Spec — throughput analysis
- Ivanov et al. (2021) Data Movement Is All You Need

## Equations

### bandwidth_ceiling

```
bw_ceiling = effective_bandwidth_GB_s / (model_bytes / 1e9)
```

**Domain:** `effective_bandwidth_GB_s > 0, model_bytes > 0`

**Codomain:** $bw_ceiling > 0 (tokens/second)$

**Invariants:**

- $Higher bandwidth \to higher ceiling$
- $Larger model \to lower ceiling$

### compute_ceiling

```
compute_ceiling = effective_GFLOPS / ops_per_token
```

**Domain:** `effective_GFLOPS > 0, ops_per_token > 0`

**Codomain:** $compute_ceiling > 0 (tokens/second)$

**Invariants:**

- $Higher GFLOPS \to higher ceiling$

### model_bytes

```
model_bytes = total_params × bits_per_weight / 8
```

**Domain:** `total_params ∈ ℤ⁺, bits_per_weight ∈ {2, 4, 8, 16, 32}`

**Invariants:**

- $model_bytes > 0 for valid model$
- $model_bytes monotonically increases with total_params$

### throughput_bound

$$
throughput <= min(bw_ceiling, compute_ceiling)
$$

**Domain:** $bw_ceiling > 0, compute_ceiling > 0$

**Invariants:**

- $Throughput cannot exceed either ceiling$
- $Memory-bound iff bw_ceiling < compute_ceiling$

## Proof Obligations

| # | Type | Property | Formal |
|---|------|----------|--------|
| 1 | bound | Ceilings positive | $bw_ceiling > 0 ∧ compute_ceiling > 0 for valid inputs$ |
| 2 | invariant | Memory-bound classification | $bw_ceiling < compute_ceiling ⟹ system is memory-bound$ |
| 3 | bound | Throughput bounded | $throughput <= min(bw_ceiling, compute_ceiling)$ |
| 4 | monotonicity | Model bytes monotonic | $more params (same quant) \to more bytes \to lower bw ceiling$ |
| 5 | equivalence | SIMD roofline equivalence |  |

## Falsification Tests

| ID | Rule | Prediction | If Fails |
|----|------|------------|----------|
| FALSIFY-RM-001 | Positive ceilings | bw and compute ceilings > 0 for all valid hardware/model specs | Division by zero in ceiling calculation |
| FALSIFY-RM-002 | Memory-bound classification | Classification matches ceiling comparison | Incorrect bottleneck identification |
| FALSIFY-RM-003 | Throughput bounded | No throughput exceeds both ceilings | Bound violation in roofline model |
| FALSIFY-RM-004 | Monotonicity | Doubling params halves bandwidth ceiling | Non-monotonic model size behavior |
| FALSIFY-RM-005 | SIMD roofline equivalence | SIMD roofline matches scalar |  compare scalar vs SIMD roofline calc:SIMD roofline bounds differ |

## Kani Harnesses

| ID | Obligation | Bound | Strategy |
|----|------------|-------|----------|
| KANI-RM-001 | RM-BND-001 | 4 | bounded_int |
| KANI-ROOFLI-002 | Ceilings positive | 8 | exhaustive |
| KANI-ROOFLI-003 | Memory-bound classification | 8 | stub_float |
| KANI-ROOFLI-004 | Throughput bounded | 8 | stub_float |
| KANI-ROOFLI-005 | Model bytes monotonic | 8 | exhaustive |
| KANI-ROOFLI-006 | SIMD roofline equivalence | 8 | exhaustive |

## QA Gate

**Roofline Model Contract** (F-RM-001)

Performance bound analysis quality gate

**Checks:** positive_ceilings, memory_bound_classification, throughput_bounded, monotonicity

**Pass criteria:** All 5 falsification tests pass

