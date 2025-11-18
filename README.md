# aprender

Next Generation Machine Learning, Statistics and Deep Learning in PURE Rust

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![TDG Score](https://img.shields.io/badge/TDG-94.1%2F100-brightgreen)](https://github.com/noahgift/pmat)

## Overview

Aprender is a lightweight, pure Rust machine learning library designed for efficiency and ease of use. Built with EXTREME TDD methodology, it provides reliable implementations of core ML algorithms with comprehensive test coverage.

## Features (v0.1.0)

### Core Primitives
- **Vector** - 1D numerical array with statistical operations (mean, sum, dot, norm, variance)
- **Matrix** - 2D numerical array with linear algebra (matmul, transpose, Cholesky decomposition)
- **DataFrame** - Named column container for ML data preparation workflows

### Machine Learning Models
- **LinearRegression** - Ordinary Least Squares via normal equations
- **KMeans** - K-means++ initialization with Lloyd's algorithm

### Metrics
- Regression: `r_squared`, `mse`, `rmse`, `mae`
- Clustering: `silhouette_score`, `inertia`

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
aprender = "0.1.0"
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

### DataFrame Operations

```rust
use aprender::prelude::*;

fn main() {
    let columns = vec![
        ("age".to_string(), Vector::from_slice(&[25.0, 30.0, 35.0])),
        ("income".to_string(), Vector::from_slice(&[50000.0, 60000.0, 75000.0])),
    ];

    let df = DataFrame::new(columns).expect("Failed to create DataFrame");

    println!("Shape: {:?}", df.shape());
    println!("Mean age: {:.1}", df.column("age").unwrap().mean());

    // Convert to Matrix for ML
    let matrix = df.to_matrix();
}
```

## Examples

Run the included examples:

```bash
cargo run --example boston_housing    # Linear regression demo
cargo run --example iris_clustering   # K-Means clustering demo
cargo run --example dataframe_basics  # DataFrame operations demo
```

## Quality Metrics

- **TDG Score**: 94.1/100 (A grade)
- **Unit Tests**: 120
- **Property Tests**: 19 (proptest)
- **Doc Tests**: 13
- **Max Cyclomatic Complexity**: 5

## API Reference

Generate documentation:

```bash
cargo doc --open
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome! Please ensure:
- All tests pass: `cargo test --all`
- No clippy warnings: `cargo clippy --all-targets`
- Code is formatted: `cargo fmt`
