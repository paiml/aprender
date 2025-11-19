# aprender

Next Generation Machine Learning, Statistics and Deep Learning in PURE Rust

[![CI](https://github.com/paiml/aprender/actions/workflows/ci.yml/badge.svg)](https://github.com/paiml/aprender/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/paiml/aprender/branch/main/graph/badge.svg)](https://codecov.io/gh/paiml/aprender)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![TDG Score](https://img.shields.io/badge/TDG-93.3%2F100-brightgreen)](https://github.com/noahgift/pmat)
[![Crates.io](https://img.shields.io/crates/v/aprender.svg)](https://crates.io/crates/aprender)
[![Docs.rs](https://docs.rs/aprender/badge.svg)](https://docs.rs/aprender)

## Overview

Aprender is a lightweight, pure Rust machine learning library designed for efficiency and ease of use. Built with EXTREME TDD methodology, it provides reliable implementations of core ML algorithms with comprehensive test coverage.

## Features

### Core Primitives
- **Vector** - 1D numerical array with statistical operations (mean, sum, dot, norm, variance)
  - Powered by [trueno](https://github.com/paiml/trueno) v0.4.0 for SIMD acceleration
- **Matrix** - 2D numerical array with linear algebra (matmul, transpose, Cholesky decomposition)
  - SIMD-optimized operations via trueno backend
- **DataFrame** - Named column container for ML data preparation workflows

### Machine Learning Models
- **LinearRegression** - Ordinary Least Squares via normal equations
- **KMeans** - K-means++ initialization with Lloyd's algorithm
- **DecisionTreeClassifier** - GINI-based decision tree with configurable max depth
- **RandomForestClassifier** - Bootstrap aggregating ensemble with majority voting

### Model Selection & Evaluation
- **train_test_split** - Random train/test splitting with reproducible seeds
- **KFold** - K-fold cross-validator with optional shuffling
- **cross_validate** - Automated cross-validation with statistics (mean, std, min, max)

### Model Persistence
- **Serialization** - Save/load models to disk (serde + bincode)
- Works with all models: LinearRegression, KMeans, DecisionTree, RandomForest

### Metrics
- Regression: `r_squared`, `mse`, `rmse`, `mae`
- Clustering: `silhouette_score`, `inertia`

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
aprender = "0.4.0"
```

## Quick Start

### Linear Regression

```rust
use aprender::prelude::*;

fn main() {
    // Features: [sqft, bedrooms]
    let x = Matrix::from_vec(5, 2, vec![
        1500.0, 3.0,
        2000.0, 4.0,
        1200.0, 2.0,
        1800.0, 3.0,
        2500.0, 5.0,
    ]).unwrap();

    // Target: price
    let y = Vector::from_slice(&[250.0, 350.0, 180.0, 280.0, 450.0]);

    let mut model = LinearRegression::new();
    model.fit(&x, &y).expect("Failed to fit");

    let predictions = model.predict(&x);
    let r2 = model.score(&x, &y);

    println!("R² Score: {:.4}", r2);
}
```

### K-Means Clustering

```rust
use aprender::prelude::*;

fn main() {
    let x = Matrix::from_vec(6, 2, vec![
        1.0, 2.0,
        1.5, 1.8,
        5.0, 8.0,
        6.0, 8.0,
        1.0, 0.6,
        9.0, 11.0,
    ]).unwrap();

    let mut kmeans = KMeans::new(2)
        .with_max_iter(100)
        .with_random_state(42);

    kmeans.fit(&x).expect("Failed to fit");

    let labels = kmeans.predict(&x);
    let score = silhouette_score(&x, &labels);

    println!("Silhouette Score: {:.4}", score);
}
```

### Random Forest Classification

```rust
use aprender::prelude::*;
use aprender::tree::RandomForestClassifier;

fn main() {
    // Iris dataset (simplified)
    let x = Matrix::from_vec(12, 2, vec![
        1.4, 0.2, 1.3, 0.2, 1.5, 0.2, 1.7, 0.4,  // Setosa
        4.7, 1.4, 4.5, 1.5, 4.9, 1.5, 4.6, 1.3,  // Versicolor
        6.0, 2.5, 5.9, 2.1, 6.1, 2.3, 5.8, 2.2,  // Virginica
    ]).unwrap();
    let y = vec![0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2];

    let mut rf = RandomForestClassifier::new(20)
        .with_max_depth(5)
        .with_random_state(42);

    rf.fit(&x, &y).expect("Failed to fit");

    let predictions = rf.predict(&x);
    let accuracy = rf.score(&x, &y);

    println!("Accuracy: {:.1}%", accuracy * 100.0);
}
```

### Cross-Validation

```rust
use aprender::prelude::*;
use aprender::model_selection::{cross_validate, KFold};

fn main() {
    let x = Matrix::from_vec(100, 1, (0..100).map(|i| i as f32).collect()).unwrap();
    let y = Vector::from_vec((0..100).map(|i| 2.0 * i as f32 + 1.0).collect());

    let model = LinearRegression::new();
    let kfold = KFold::new(10).with_random_state(42);

    let results = cross_validate(&model, &x, &y, &kfold).unwrap();

    println!("Mean R²: {:.4} ± {:.4}", results.mean(), results.std());
    println!("Min/Max: {:.4} / {:.4}", results.min(), results.max());
}
```

## Examples

Run the included examples:

```bash
cargo run --example boston_housing       # Linear regression demo
cargo run --example iris_clustering      # K-Means clustering demo
cargo run --example dataframe_basics     # DataFrame operations demo
cargo run --example decision_tree_iris   # Decision Tree classifier demo
cargo run --example random_forest_iris   # Random Forest ensemble demo
cargo run --example cross_validation     # Cross-validation demo
cargo run --example model_persistence    # Model save/load demo
```

## Quality Metrics

- **TDG Score**: 93.3/100 (A grade)
- **Total Tests**: 184 passing
- **Property Tests**: 22 (proptest)
- **Doc Tests**: 16
- **Coverage**: ~97%
- **Max Cyclomatic Complexity**: ≤10
- **Clippy Warnings**: 0

## Documentation

- **EXTREME TDD Book**: https://paiml.github.io/aprender/
- **API Reference**: Run `cargo doc --open` or visit [docs.rs/aprender](https://docs.rs/aprender)

## Roadmap

See [ROADMAP.md](ROADMAP.md) for planned features and version roadmap.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome! Please ensure:
- All tests pass: `cargo test --all`
- No clippy warnings: `cargo clippy --all-targets`
- Code is formatted: `cargo fmt`
