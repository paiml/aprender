# batchnorm-kernel-v1

**Version:** 1.0.0

BatchNorm kernel — batch normalization with running statistics

## References

- Ioffe & Szegedy (2015) Batch Normalization: Accelerating Deep Network Training

## Equations

### batchnorm_eval

```
BN_eval(x)_i = gamma_i * (x_i - mu_run) / sqrt(sigma_run^2 + eps) + beta_i
```

**Domain:** $x in R^C, mu_run in R^C, sigma_run in R_>=0^C$

**Codomain:** $BN_eval(x) in R^C$

**Invariants:**

- $Uses running stats, not batch stats$
- $Deterministic (same output for same input)$

### batchnorm_train

$$
BN(x)_i = gamma_i * (x_i - mu_B) / \sqrt{sigma_B^2 + eps} + beta_i
$$

**Domain:** $x in R^{N x C}, gamma in R^C, beta in R^C, eps > 0$

**Codomain:** $BN(x) in R^{N x C}$

**Invariants:**

- $mu_B = (1/N) * sum_n x_{n,c} per channel c (batch mean)$
- $sigma_B^2 = (1/N) * sum_n (x_{n,c} - mu_B)^2 per channel c$
- $Output has zero mean and unit variance per channel (before affine)$

### running_stats

$$
mu_run = (1-m)*mu_run + m*mu_B, sigma_run = (1-m)*sigma_run + m*sigma_B
$$

**Domain:** $momentum m in (0, 1)$

**Codomain:** $mu_run in R^C, sigma_run in R_>=0^C$

**Invariants:**

- $Running stats are exponential moving averages$
- $sigma_run >= 0 (non-negative variance)$

## Proof Obligations

| # | Type | Property | Formal |
|---|------|----------|--------|
| 1 | invariant | Training output standardized | $\|mean(BN(x)[:, c]) - beta_c\| < eps per channel c when gamma=1$ |
| 2 | bound | Denominator strictly positive | $\sqrt{sigma_B^2 + eps} > 0 when eps > 0$ |
| 3 | invariant | Running variance non-negative | $sigma_run >= 0 after any number of updates$ |
| 4 | equivalence | Eval mode uses running stats | `BN_eval(x) uses mu_run/sigma_run, not batch statistics` |
| 5 | equivalence | SIMD matches scalar within ULP |  |

## Kernel Phases

1. **compute_batch_stats**: Compute per-channel mean and variance across batch — *sigma^2 >= 0*
2. **normalize**: Subtract mean, divide by sqrt(var + eps) — *denominator > 0*
3. **affine_transform**: Apply gamma * normalized + beta — *Output dimension preserved*
4. **update_running_stats**: EMA update of running mean and variance (training only) — *Running variance stays non-negative*

## SIMD Dispatch

| Kernel | ISA | Target |
|--------|-----|--------|
| batchnorm | avx2 | `batchnorm_avx2` |
| batchnorm | ptx | `batchnorm_ptx` |
| batchnorm | scalar | `batchnorm_scalar` |

## Falsification Tests

| ID | Rule | Prediction | If Fails |
|----|------|------------|----------|
| FALSIFY-BN-001 | Training standardization | \|mean(BN(x)[:,c])\| < 1e-5 per channel when gamma=1, beta=0 | Batch mean not correctly subtracted |
| FALSIFY-BN-002 | Denominator safety | No NaN/Inf when all inputs are equal (zero variance) | Epsilon not added before sqrt |
| FALSIFY-BN-003 | Running stats non-negativity | sigma_run >= 0 after 1000 random updates | Floating-point error in EMA accumulation |
| FALSIFY-BN-004 | Eval uses running stats | BN_eval(x) != BN_train(x) when running stats differ from batch | Mode flag ignored, always using batch stats |
| FALSIFY-BN-005 | SIMD equivalence | \|batchnorm_avx2(x) - batchnorm_scalar(x)\| < 8 ULP | SIMD reduction for mean/variance differs |
| FALSIFY-BN-006 | Boundary - batch size 1 | BN with batch_size=1 has zero variance, output = beta when gamma=1 | Edge case in variance computation for N=1 |

## Kani Harnesses

| ID | Obligation | Bound | Strategy |
|----|------------|-------|----------|
| KANI-BN-001 | BN-BND-001 | 8 | stub_float |
| KANI-BN-002 | BN-INV-003 | 4 | stub_float |
| KANI-BATCHN-003 | Training output standardized | 8 | exhaustive |
| KANI-BATCHN-004 | Denominator strictly positive | 8 | exhaustive |
| KANI-BATCHN-005 | Running variance non-negative | 8 | exhaustive |
| KANI-BATCHN-006 | Eval mode uses running stats | 8 | exhaustive |
| KANI-BATCHN-007 | SIMD matches scalar within ULP | 8 | exhaustive |

## QA Gate

**BatchNorm Contract** (F-BN-001)

Batch normalization with running statistics quality gate

**Checks:** standardization, denominator_safety, running_stats_validity, eval_determinism

**Pass criteria:** All 6 falsification tests pass + Kani harnesses verify

