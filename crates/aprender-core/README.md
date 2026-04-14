# aprender-core

[![Crates.io](https://img.shields.io/crates/v/aprender-core.svg)](https://crates.io/crates/aprender-core)
[![docs.rs](https://docs.rs/aprender-core/badge.svg)](https://docs.rs/aprender-core)
[![CI](https://github.com/paiml/aprender/actions/workflows/ci.yml/badge.svg)](https://github.com/paiml/aprender/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

The core ML library for the [Aprender](https://github.com/paiml/aprender) framework — pure Rust, zero unsafe, 13,026 tests.

Published as `aprender-core` on crates.io; import as `use aprender::*`. Part of a 70-crate monorepo that ships the `apr` CLI (`cargo install aprender`).

## Install

```toml
[dependencies]
aprender-core = "0.29"

# Optional features
aprender-core = { version = "0.29", features = ["format-compression", "format-quantize"] }
```

Or with cargo-add:

```bash
cargo add aprender-core
```

## Quick Start

```rust
use aprender::linear_regression::LinearRegression;
use aprender::traits::Estimator;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Training data: y = 2x + 1
    let x = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0]];
    let y = vec![3.0, 5.0, 7.0, 9.0];

    let mut model = LinearRegression::new();
    model.fit(&x, &y)?;

    let predictions = model.predict(&vec![vec![5.0], vec![6.0]])?;
    println!("{predictions:?}"); // [11.0, 13.0]
    Ok(())
}
```

```rust
use aprender::kmeans::KMeans;
use aprender::traits::UnsupervisedEstimator;

let data = vec![
    vec![1.0, 2.0], vec![1.5, 1.8], vec![5.0, 8.0], vec![8.0, 8.0],
];
let mut model = KMeans::new(2, 100);
model.fit(&data)?;
let labels = model.predict(&data)?;
```

## Features

**Supervised Learning**
- Linear Regression, Logistic Regression
- Decision Trees (CART), Random Forest, Gradient Boosting (GBM)
- K-Nearest Neighbors (KNN), Support Vector Machines (SVM)

**Unsupervised Learning**
- K-Means clustering, PCA decomposition, ICA

**Time Series**
- ARIMA (AutoRegressive Integrated Moving Average)

**Bayesian Inference**
- Conjugate priors, Bayesian Linear Regression
- Generalized Linear Models (Poisson, Gamma, Binomial)

**Graph Neural Networks**
- Dijkstra, A*, PageRank, community detection

**Text Processing**
- BPE and SentencePiece tokenizers, stop words, stemming
- Chat template engine (minijinja), chat markup language

**APR Model Format**
- Read/write the native `.apr` binary format
- `format-compression` feature: LZ4 and Zstd tensor compression
- `format-quantize` feature: Q4K and Q6K quantization

## Optional Feature Flags

| Feature | Description |
|---|---|
| `format-compression` | LZ4 and Zstd compressed tensor storage |
| `format-quantize` | Q4K / Q6K quantization for model export |

## Documentation

- [API docs (docs.rs)](https://docs.rs/aprender-core)
- [Full monorepo](https://github.com/paiml/aprender)
- [APR format specification](https://github.com/paiml/aprender/blob/main/contracts/tensor-layout-v1.yaml)
- [EXTREME TDD book](https://github.com/paiml/aprender/tree/main/docs/book)

## License

MIT. See [LICENSE](../../LICENSE).
