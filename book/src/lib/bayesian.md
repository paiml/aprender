<!-- PCU: lib-bayesian | contract: contracts/apr-page-lib-bayesian-v1.yaml -->

# Module: `aprender::bayesian`

Public module of the `aprender-core` crate.

## Source

[`crates/aprender-core/src/bayesian/mod.rs`](https://github.com/paiml/aprender/blob/main/crates/aprender-core/src/bayesian/mod.rs) or directory.

## Example

```rust
use aprender::bayesian::{BetaBinomial, NormalInverseGamma, BayesianLinearRegression};
// See `cargo doc -p aprender-core --open` for full API reference.
```

## Module summary

`aprender::bayesian` exposes the closed-form conjugate-prior distributions
that make Bayesian updating tractable (Beta-Binomial, Dirichlet-Multinomial,
Gamma-Poisson, Normal-Inverse-Gamma) plus full Bayesian linear and logistic
regression. Conjugate priors give you posterior mean / mode / variance in
constant time after each observation — ideal for online experiment
analysis (A/B tests), bandits, and small-sample inference where you need
uncertainty estimates not just point predictions.

## Key types

| Type | Description |
|------|-------------|
| `BetaBinomial` | Beta prior over a binomial proportion. `uniform()`, `jeffreys()`, `new(alpha, beta)` constructors; `update(successes, trials)` for sequential inference. |
| `DirichletMultinomial` | Dirichlet prior over multinomial probabilities. |
| `GammaPoisson` | Gamma prior over a Poisson rate. |
| `NormalInverseGamma` | NIG prior over unknown mean + variance. |
| `BayesianLinearRegression` | Linear regression with full posterior over coefficients + noise. Exposes `predict`, `log_likelihood`, `aic`, `bic`. |
| `BayesianLogisticRegression` | Laplace-approximated Bayesian logistic regression. |

## Usage patterns

### Pattern 1: Sequential A/B-test analysis with Beta-Binomial

```rust
use aprender::bayesian::BetaBinomial;

// Start from a Jeffreys prior, then ingest 7 successes out of 10 trials.
let mut posterior = BetaBinomial::jeffreys();
posterior.update(7, 10);

let mean = posterior.posterior_mean();
let var = posterior.posterior_variance();
println!("posterior mean = {:.3}  variance = {:.4}", mean, var);
assert!(mean > 0.5);
```

### Pattern 2: Bayesian linear regression with predictive variance

```rust
use aprender::bayesian::BayesianLinearRegression;
use aprender::primitives::{Matrix, Vector};

let x = Matrix::from_vec(4, 1, vec![1.0, 2.0, 3.0, 4.0]).expect("4x1");
let y = Vector::from_slice(&[3.0, 5.0, 7.0, 9.0]);

let mut blr = BayesianLinearRegression::new(1);
blr.fit(&x, &y).expect("blr fit");

let coefs = blr.posterior_mean().expect("fitted");
let noise = blr.noise_variance().expect("fitted");
println!("posterior_mean = {:?}  noise_var = {:?}", coefs, noise);

let x_test = Matrix::from_vec(1, 1, vec![5.0]).expect("1x1");
let preds = blr.predict(&x_test).expect("predict");
println!("y(5) ≈ {:.2}", preds[0]);
```

## See also

- [`linear_model`](linear_model.md) — frequentist alternatives (OLS / Ridge / Lasso)
- [`glm`](glm.md) — generalized linear models for non-Gaussian responses
- [`naive_bayes` (in `classification`)](classification.md) — Bayes classifier variant
- [`monte_carlo`](monte_carlo.md) — when the posterior is not conjugate, fall through to MCMC

## Full API

Run `cargo doc -p aprender-core --open` for the rendered rustdoc, or browse
[docs.rs/aprender](https://docs.rs/aprender) for the published version.
