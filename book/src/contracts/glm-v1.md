# glm-v1

**Version:** 1.0.0

Generalized Linear Models -- Poisson, Gamma, and Binomial regression with canonical link functions

## References

- Nelder & Wedderburn (1972) Generalized Linear Models, JRSS
- McCullagh & Nelder (1989) Generalized Linear Models, 2nd ed.

## Equations

### binomial_link

$$
g(p) = ln(p/(1-p)), g^{-1}(eta) = 1/(1+\exp(-eta))
$$

**Domain:** $p in (0, 1), eta in R$

**Codomain:** $g(p) in R, g^{-1}(eta) in (0, 1)$

**Invariants:**

- $Logit maps (0,1) to R bijectively$
- $Predicted probability always in (0, 1)$
- $g(g^{-1}(eta)) = eta (inverse round-trip)$

### gamma_link

$$
g(mu) = 1/mu, g^{-1}(eta) = 1/eta
$$

**Domain:** $mu > 0 (Gamma mean), eta > 0 (linear predictor restricted)$

**Codomain:** $g(mu) > 0, g^{-1}(eta) > 0$

**Invariants:**

- $Link function is strictly monotone on (0, inf)$
- $Predicted mean always positive$
- $g(g^{-1}(eta)) = eta for eta > 0$

### irls_fit

$$
beta^{(k+1)} = (X^T W^{(k)} X)^{-1} X^T W^{(k)} z^{(k)}
$$

**Domain:** $X in R^{n x p}, W diagonal weights, z working response$

**Codomain:** $beta in R^p$

**Invariants:**

- $Deviance decreases monotonically: D(beta^{(k+1)}) <= D(beta^{(k)})$
- $Converges when ||beta^{(k+1)} - beta^{(k)}|| < tol$

### poisson_link

$$
g(mu) = ln(mu), g^{-1}(eta) = \exp(eta)
$$

**Domain:** $mu > 0 (Poisson mean), eta in R (linear predictor)$

**Codomain:** $g(mu) in R, g^{-1}(eta) > 0$

**Invariants:**

- $Link function is strictly monotone (bijective)$
- $\exp(eta) > 0 for all eta (mean always positive)$
- $g(g^{-1}(eta)) = eta (inverse round-trip)$

## Proof Obligations

| # | Type | Property | Formal |
|---|------|----------|--------|
| 1 | invariant | Link function invertible | $g(g^{-1}(eta)) = eta for all eta in domain$ |
| 2 | bound | Predicted mean in valid range | $Poisson: mu > 0, Gamma: mu > 0, Binomial: 0 < p < 1$ |
| 3 | invariant | IRLS convergence | $D^{(k+1)} <= D^{(k)} (deviance non-increasing)$ |
| 4 | invariant | Predictions finite | `forall i: \|y_hat_i\| < infinity for bounded input` |

## Falsification Tests

| ID | Rule | Prediction | If Fails |
|----|------|------------|----------|
| FALSIFY-GLM-001 | Link function round-trip | g(g^{-1}(eta)) = eta within floating-point tolerance | Link or inverse link implementation error |
| FALSIFY-GLM-002 | Predicted mean in valid range | Poisson mu > 0, Gamma mu > 0, Binomial p in (0,1) | Inverse link not constraining output range |
| FALSIFY-GLM-003 | IRLS convergence | Deviance decreases or stays same across IRLS iterations | IRLS weight or working response computation error |
| FALSIFY-GLM-004 | Prediction finiteness | All predictions are finite for bounded input data | Numerical overflow in exp() or division by zero in reciprocal link |

## Kani Harnesses

| ID | Obligation | Bound | Strategy |
|----|------------|-------|----------|
| KANI-GLM-001 | GLM-INV-001 | 8 | stub_float |
| KANI-GLM-002 | GLM-BND-001 | 8 | stub_float |
| KANI-GLM_V1-003 | Link function invertible | 8 | exhaustive |
| KANI-GLM_V1-004 | Predicted mean in valid range | 8 | exhaustive |
| KANI-GLM_V1-005 | IRLS convergence | 8 | exhaustive |
| KANI-GLM_V1-006 | Predictions finite | 8 | exhaustive |

## QA Gate

**GLM Contract** (F-GLM-001)

Generalized Linear Models correctness quality gate

**Checks:** link_roundtrip, mean_valid_range, irls_convergence, prediction_finiteness

**Pass criteria:** All 4 falsification tests pass + Kani harnesses verify

