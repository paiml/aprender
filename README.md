<p align="center">
  <img src=".github/apr-format-hero.svg" alt="aprender" width="800">
</p>

<h1 align="center">aprender</h1>

<p align="center">
  <b>A production-ready machine learning library written in pure Rust.</b>
</p>

<p align="center">
  <a href="https://crates.io/crates/aprender"><img src="https://img.shields.io/crates/v/aprender.svg" alt="Crates.io"></a>
  <a href="https://docs.rs/aprender"><img src="https://docs.rs/aprender/badge.svg" alt="Documentation"></a>
  <a href="https://github.com/paiml/aprender/actions/workflows/ci.yml"><img src="https://github.com/paiml/aprender/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License"></a>
</p>

---

Aprender provides implementations of classical machine learning algorithms optimized for performance and safety. The library requires no external dependencies beyond the Rust standard library and offers seamless compilation to WebAssembly.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Algorithms](#algorithms)
- [Model Persistence](#model-persistence)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Pure Rust** — Zero C/C++ dependencies, memory-safe, thread-safe by default
- **SIMD Acceleration** — Vectorized operations via [trueno](https://github.com/paiml/trueno) backend
- **WebAssembly Ready** — Compile to WASM for browser and edge deployment
- **Native Model Format** — `.apr` format with encryption, signatures, and zero-copy loading
- **Interoperability** — Export to SafeTensors and GGUF formats

## Installation

Add aprender to your `Cargo.toml`:

```toml
[dependencies]
aprender = "0.11"
```

### Optional Features

```toml
[dependencies]
aprender = { version = "0.11", features = ["format-encryption", "hf-hub-integration"] }
```

| Feature | Description |
|---------|-------------|
| `format-encryption` | AES-256-GCM encryption for model files |
| `format-signing` | Ed25519 digital signatures |
| `format-compression` | Zstd compression |
| `hf-hub-integration` | Hugging Face Hub push/pull support |
| `gpu` | GPU acceleration via wgpu |

## Quick Start

```rust
use aprender::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Training data
    let x = Matrix::from_vec(4, 2, vec![
        1.0, 2.0,
        2.0, 3.0,
        3.0, 4.0,
        4.0, 5.0,
    ])?;
    let y = Vector::from_slice(&[3.0, 5.0, 7.0, 9.0]);

    // Train model
    let mut model = LinearRegression::new();
    model.fit(&x, &y)?;

    // Evaluate
    println!("R² = {:.4}", model.score(&x, &y));

    Ok(())
}
```

## Algorithms

### Supervised Learning

| Algorithm | Description |
|-----------|-------------|
| `LinearRegression` | Ordinary least squares regression |
| `LogisticRegression` | Binary and multiclass classification |
| `DecisionTreeClassifier` | GINI-based decision trees |
| `RandomForestClassifier` | Bootstrap aggregating ensemble |
| `GradientBoostingClassifier` | Adaptive boosting with residual learning |
| `NaiveBayes` | Gaussian naive Bayes classifier |
| `KNeighborsClassifier` | k-nearest neighbors |
| `LinearSVM` | Support vector machine with hinge loss |

### Unsupervised Learning

| Algorithm | Description |
|-----------|-------------|
| `KMeans` | k-means++ initialization with Lloyd's algorithm |
| `DBSCAN` | Density-based spatial clustering |
| `PCA` | Principal component analysis |
| `IsolationForest` | Anomaly detection |

### Additional Modules

- **Graph Analysis** — PageRank, betweenness centrality, community detection
- **Time Series** — ARIMA forecasting
- **Text Processing** — Tokenization, TF-IDF, stemming
- **Neural Networks** — Sequential models, transformers, mixture of experts

## Model Persistence

The `.apr` format provides secure, efficient model serialization:

```rust
use aprender::format::{save, load, ModelType, SaveOptions};

// Save with encryption
save(&model, ModelType::LinearRegression, "model.apr",
    SaveOptions::default()
        .with_encryption("password")
        .with_compression(true))?;

// Load
let model: LinearRegression = load("model.apr", ModelType::LinearRegression)?;
```

### Format Capabilities

- **Security** — AES-256-GCM encryption, Ed25519 signatures, X25519 key exchange
- **Performance** — Memory-mapped loading, 600x faster than standard deserialization
- **Integrity** — CRC32 checksums with automatic corruption detection
- **Commercial** — License blocks, watermarking, buyer-specific encryption

## Documentation

| Resource | Link |
|----------|------|
| API Reference | [docs.rs/aprender](https://docs.rs/aprender) |
| User Guide | [paiml.github.io/aprender](https://paiml.github.io/aprender/) |
| Examples | [`examples/`](examples/) |
| Model Format Spec | [`docs/specifications/model-format-spec.md`](docs/specifications/model-format-spec.md) |

## Contributing

We welcome contributions. Please ensure your changes pass the test suite:

```bash
cargo test --all-features
cargo clippy --all-targets -- -D warnings
cargo fmt --check
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

Aprender is distributed under the MIT License. See [LICENSE](LICENSE) for details.

---

<p align="center">
  <sub>Built by <a href="https://paiml.com">Paiml</a></sub>
</p>
