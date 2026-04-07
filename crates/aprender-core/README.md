# aprender-core

Core ML library for the [Aprender](https://github.com/paiml/aprender) framework.

This crate provides the ML library (`use aprender::*`). Install the CLI via `cargo install aprender`.

## Usage

```toml
[dependencies]
aprender-core = "0.29"
```

```rust
use aprender::linear_regression::LinearRegression;
use aprender::traits::Estimator;
```

Part of the [Aprender monorepo](https://github.com/paiml/aprender) — 70 workspace crates.
