# aprender-contracts-macros

[![Crates.io](https://img.shields.io/crates/v/aprender-contracts-macros.svg)](https://crates.io/crates/aprender-contracts-macros)
[![docs.rs](https://docs.rs/aprender-contracts-macros/badge.svg)](https://docs.rs/aprender-contracts-macros)
[![CI](https://github.com/paiml/aprender/actions/workflows/ci.yml/badge.svg)](https://github.com/paiml/aprender/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Procedural macros for compile-time contract enforcement in the [Aprender](https://github.com/paiml/aprender) ML framework. Bind Rust functions to YAML contracts and express pre/post-conditions that are verified at compile time.

> Previously published as `provable-contracts-macros`.

## Install

```toml
[dependencies]
aprender-contracts-macros = "0.29"
```

Or with cargo-add:

```bash
cargo add aprender-contracts-macros
```

## Quick Start

```rust
use aprender_contracts_macros::{contract, ensures, invariant, requires};

// Bind to a YAML contract file and document the governing equation.
// The macro verifies the contract YAML exists at compile time.
#[contract("linear-regression-v1", equation = "y_hat = X * w + b")]
pub fn predict(weights: &[f64], bias: f64, x: &[f64]) -> Vec<f64> {
    x.iter().map(|xi| xi * weights[0] + bias).collect()
}

// Enforce preconditions — checked in debug builds, elided in release.
#[requires(xs.len() == ys.len(), "input slices must have equal length")]
#[requires(!xs.is_empty(), "cannot fit on empty data")]
pub fn fit(xs: &[f64], ys: &[f64]) -> (f64, f64) {
    // ...
}

// Enforce postconditions on the return value.
#[ensures(ret >= 0.0, "MSE is always non-negative")]
pub fn mean_squared_error(predictions: &[f64], targets: &[f64]) -> f64 {
    // ...
}

// Enforce struct invariants checked at entry and exit of every method.
#[invariant(self.n_clusters > 0)]
impl KMeans {
    pub fn fit(&mut self, data: &[Vec<f64>]) { /* ... */ }
    pub fn predict(&self, x: &[Vec<f64>]) -> Vec<usize> { /* ... */ }
}
```

## Macros

| Macro | Description |
|---|---|
| `#[contract("name")]` | Bind a function to a named YAML contract; fails at compile time if the contract file is missing |
| `#[contract("name", equation = "...")]` | Same, plus embed the governing mathematical equation in generated docs |
| `#[requires(expr, "msg")]` | Assert a precondition before the function body executes |
| `#[ensures(ret > 0.0, "msg")]` | Assert a postcondition on the return value (bound as `ret`) |
| `#[invariant(self.valid())]` | Assert a struct invariant on entry and exit of every annotated `impl` block method |

## Documentation

- [API docs (docs.rs)](https://docs.rs/aprender-contracts-macros)
- [Full monorepo](https://github.com/paiml/aprender)
- [Contract library crate](https://crates.io/crates/aprender-contracts)

## License

MIT. See [LICENSE](../../LICENSE).
