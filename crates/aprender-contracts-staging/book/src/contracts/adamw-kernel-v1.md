# adamw-kernel-v1

**Version:** 1.0.0

AdamW kernel — Adam optimizer with decoupled weight decay

## References

- Loshchilov & Hutter (2017) Decoupled Weight Decay Regularization
- Kingma & Ba (2014) Adam: A Method for Stochastic Optimization

## Dependency Graph

```mermaid
graph LR
    classification_finetune_v1["classification-finetune-v1"] --> adamw_kernel_v1["adamw-kernel-v1"]
```

## Equations

### adam_moments

$$
m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
$$

**Domain:** $g_t in R^d, m_0 = 0, beta1 in (0, 1)$

**Codomain:** $m_t in R^d$

**Invariants:**

- $m_t is exponential moving average of gradients$
- $|m_t| bounded by max(|g_1|, ..., |g_t|) when beta1 < 1$

### adam_variance

$$
v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
$$

**Domain:** $g_t in R^d, v_0 = 0, beta2 in (0, 1)$

**Codomain:** $v_t in R_>=0^d$

**Invariants:**

- $v_t >= 0 (non-negative second moment)$
- $v_t is exponential moving average of squared gradients$

### bias_correction

```
m_hat_t = m_t / (1 - beta1^t), v_hat_t = v_t / (1 - beta2^t)
```

**Domain:** $t >= 1, beta1 in (0,1), beta2 in (0,1)$

**Codomain:** `m_hat_t in R^d, v_hat_t in R_>=0^d`

**Invariants:**

- $Correction factor > 1 for all t >= 1$
- $Correction approaches 1 as t -> inf$

### weight_update

```
theta_t = theta_{t-1} - lr * (m_hat_t / (sqrt(v_hat_t) + eps) + lambda * theta_{t-1})
```

**Domain:** $theta in R^d, lr > 0, lambda >= 0, eps > 0$

**Codomain:** $theta_t in R^d$

**Invariants:**

- $Weight decay applied AFTER Adam update (decoupled)$
- $Update finite when inputs finite and eps > 0$

## Proof Obligations

| # | Type | Property | Formal |
|---|------|----------|--------|
| 1 | precondition | Hyperparameters valid, inputs finite | $lr > 0 ∧ \beta1 \in (0,1) ∧ \beta2 \in (0,1) ∧ \varepsilon > 0 ∧ \lambda \geq 0 ∧ t \geq 1 ∧ \forall i: isFinite(g_i)$ |
| 2 | postcondition | Updated weights finite, moments non-negative | `∀i: isFinite(θ_i) ∧ v_t_i ≥ 0` |
| 3 | frame | Only theta, m, v are modified; gradients and hyperparams unchanged | $modifies(\theta, m, v) ∧ preserves(g, lr, \beta1, \beta2, \varepsilon, \lambda)$ |
| 4 | loop_invariant | Second moment remains non-negative across all training steps | `∀ step t, ∀i: v_t_i ≥ 0` |
| 5 | loop_variant | Training step counter advances | $V = max_steps - t, V \geq 0, V strictly decreasing$ |
| 6 | old_state | Moments are exponential moving averages of old values | $m_t = \beta1 · old(m_{t-1}) + (1-\beta1) · g_t$ |
| 7 | invariant | Decoupled weight decay | $Weight decay term is lambda * theta, not lambda * theta in gradient$ |
| 8 | bound | Second moment non-negative | $v_t >= 0 for all t and all dimensions$ |
| 9 | bound | Bias-corrected moments finite | `m_hat_t and v_hat_t are finite when g_t is finite` |
| 10 | invariant | Bias correction factor | $1 / (1 - beta^t) > 1 for t >= 1 and beta in (0, 1)$ |
| 11 | equivalence | SIMD matches scalar within ULP |  |

## Kernel Phases

1. **update_first_moment**: m_t = beta1 * m_{t-1} + (1 - beta1) * g_t — *m_t is linear combination of m_{t-1} and g_t*
2. **update_second_moment**: v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2 — *v_t >= 0*
3. **bias_correct**: Compute bias-corrected m_hat and v_hat — *Correction factor > 1*
4. **adam_step**: Compute lr * m_hat / (sqrt(v_hat) + eps) — *Step is finite when eps > 0*
5. **weight_decay**: Subtract lr * lambda * theta (decoupled) — *Decay applied to theta, not gradient*

## SIMD Dispatch

| Kernel | ISA | Target |
|--------|-----|--------|
| adamw | avx2 | `adamw_step_avx2` |
| adamw | ptx | `adamw_ptx` |
| adamw | scalar | `adamw_step_scalar` |

## Falsification Tests

| ID | Rule | Prediction | If Fails |
|----|------|------------|----------|
| FALSIFY-AW-001 | Decoupled weight decay | AdamW(g, lambda) != Adam(g + lambda*theta) for lambda > 0 | Weight decay is coupled (L2 reg instead of decoupled) |
| FALSIFY-AW-002 | Second moment non-negativity | v_t >= 0 for all t after random gradient updates | Floating-point underflow in EMA update |
| FALSIFY-AW-003 | Bias correction | 1/(1-beta^t) > 1 for t in [1, 10000] and beta in (0, 1) | Integer overflow in power computation or division by zero |
| FALSIFY-AW-004 | Update finiteness | theta_t is finite when g_t is finite and eps > 0 | Division by near-zero denominator when eps too small |
| FALSIFY-AW-005 | SIMD equivalence | \|adamw_avx2(args) - adamw_scalar(args)\| < 8 ULP | SIMD sqrt or reciprocal approximation differs |
| FALSIFY-AW-006 | Boundary - zero gradient | With g=0, only weight decay modifies theta | Bias correction or moment update incorrect at zero |
| FALSIFY-AW-007 | Precondition - invalid hyperparameters | adamw_step with β1=0 or β2=1 or ε=0 returns Err or panics | Missing hyperparameter validation |
| FALSIFY-AW-008 | Frame condition | Gradient vector unchanged after adamw_step | Optimizer modifies gradient buffer |
| FALSIFY-AW-009 | Loop invariant - moment non-negativity across steps | v_t ≥ 0 after 1000 consecutive random gradient steps | EMA accumulation produces negative second moment |
| FALSIFY-AW-010 | Old state - moment EMA | m_t = β1·old_m + (1-β1)·g_t verified at each step | First moment update formula incorrect |
| FALSIFY-AW-011 | Loop variant - step counter | Training loop terminates when step reaches max_steps | Loop counter not advancing or off-by-one |

## Kani Harnesses

| ID | Obligation | Bound | Strategy |
|----|------------|-------|----------|
| KANI-AW-001 | AW-INV-001 | 4 | stub_float |
| KANI-AW-002 | AW-BND-001 | 8 | stub_float |
| KANI-AW-003 | AW-BND-002 | 4 | stub_float |
| KANI-ADAMW_-004 | Hyperparameters valid, inputs finite | 8 | exhaustive |
| KANI-ADAMW_-005 | Updated weights finite, moments non-negative | 8 | exhaustive |
| KANI-ADAMW_-006 | Only theta, m, v are modified; gradients and hyperparams unchanged | 8 | exhaustive |
| KANI-ADAMW_-007 | Second moment remains non-negative across all training steps | 8 | exhaustive |
| KANI-ADAMW_-008 | Training step counter advances | 8 | exhaustive |
| KANI-ADAMW_-009 | Moments are exponential moving averages of old values | 8 | exhaustive |
| KANI-ADAMW_-010 | Decoupled weight decay | 8 | exhaustive |
| KANI-ADAMW_-011 | Second moment non-negative | 8 | exhaustive |
| KANI-ADAMW_-012 | Bias-corrected moments finite | 8 | exhaustive |
| KANI-ADAMW_-013 | Bias correction factor | 8 | exhaustive |
| KANI-ADAMW_-014 | SIMD matches scalar within ULP | 8 | exhaustive |

## QA Gate

**AdamW Contract** (F-AW-001)

Decoupled weight decay optimizer quality gate

**Checks:** decoupled_decay, moment_positivity, update_finiteness, simd_equivalence

**Pass criteria:** All 11 falsification tests pass + Kani harnesses verify

