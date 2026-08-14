<!-- PCU: lib-glm | contract: contracts/apr-page-lib-glm-v1.yaml -->

# Module: `aprender::glm`

Public module of the `aprender-core` crate.

## Source

[`crates/aprender-core/src/glm/mod.rs`](https://github.com/paiml/aprender/blob/main/crates/aprender-core/src/glm/mod.rs) or directory.

## Example

```rust
use aprender::glm::{GLM, Family, Link};
// See `cargo doc -p aprender-core --open` for full API reference.
```

## Module summary

`aprender::glm` provides generalized linear models — the natural extension of
linear regression to non-Gaussian response distributions. A single `GLM`
struct is parameterised by a `Family` enum (Poisson, Negative Binomial,
Gamma, Binomial, Gaussian, …) and an optional `Link` function (Log, Inverse,
Logit, Identity). Fitting uses iteratively reweighted least squares (IRLS).
Reach for GLM when you have count data (Poisson / NegBinomial), positive
continuous outcomes (Gamma), proportions (Binomial), or any other exponential
family.

## Key types

| Type | Description |
|------|-------------|
| `GLM` | The generalized linear model. Builder: `with_link`, `with_max_iter`, `with_tolerance`, `with_dispersion`. |
| `Family` | Response distribution: `Poisson`, `NegativeBinomial`, `Gamma`, `Binomial`, `Gaussian`. |
| `Link` | Link function: `Log`, `Inverse`, `Logit`, `Identity`. Canonical link inferred per family if not set. |

## Usage patterns

### Pattern 1: Poisson regression for count data

```rust
use aprender::glm::{GLM, Family, Link};
use aprender::primitives::{Matrix, Vector};

let x = Matrix::from_vec(5, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0]).expect("5x1");
let y = Vector::from_slice(&[1.0, 3.0, 5.0, 8.0, 13.0]);

let mut glm = GLM::new(Family::Poisson)
    .with_link(Link::Log)
    .with_max_iter(100)
    .with_tolerance(1e-6);

glm.fit(&x, &y).expect("Poisson GLM fit");

let test = Matrix::from_vec(1, 1, vec![6.0]).expect("1x1");
let pred = glm.predict(&test).expect("predict");
println!("expected count at x=6 ≈ {:.2}", pred[0]);
```

### Pattern 2: Gamma regression for positive continuous data

```rust
use aprender::glm::{GLM, Family, Link};
use aprender::primitives::{Matrix, Vector};

let x = Matrix::from_vec(5, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0]).expect("5x1");
// strictly positive y (e.g. insurance claim amounts)
let y = Vector::from_slice(&[1.2, 2.5, 5.1, 9.8, 19.5]);

let mut glm = GLM::new(Family::Gamma)
    .with_link(Link::Log)
    .with_dispersion(0.1);

glm.fit(&x, &y).expect("Gamma GLM fit");

let coefs = glm.coefficients().expect("fitted");
println!("coefficients: {:?}", coefs);
```

## See also

- [`linear_model`](linear_model.md) — Gaussian-noise linear regression (a special case of GLM)
- [`bayesian`](bayesian.md) — Bayesian counterparts via conjugate priors
- [`classification`](classification.md) — logistic regression is a special Binomial-family GLM
- [`metrics`](metrics.md) — deviance and other GLM-specific scores

## Full API

Run `cargo doc -p aprender-core --open` for the rendered rustdoc, or browse
[docs.rs/aprender](https://docs.rs/aprender) for the published version.
