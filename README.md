<div align="center">

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

</div>

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
aprender = "0.13"
```

### Optional Features

```toml
[dependencies]
aprender = { version = "0.13", features = ["format-encryption", "hf-hub-integration"] }
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
- **Metaheuristics** — ACO, Tabu Search, DE, PSO, GA, CMA-ES

### Related Crates

| Crate | Description |
|-------|-------------|
| [`aprender-tsp`](https://crates.io/crates/aprender-tsp) | TSP solver with CLI and `.apr` model persistence |
| [`aprender-shell`](https://crates.io/crates/aprender-shell) | AI-powered shell completion trained on your history |

### Resources

| Resource | Description |
|----------|-------------|
| [apr-cookbook](https://github.com/paiml/apr-cookbook) | 50+ idiomatic Rust examples for `.apr` format, WASM deployment, and SIMD acceleration |

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

## APR CLI Tool

The `apr` CLI provides comprehensive model operations for the `.apr` format.

### Installation

```bash
cargo install apr-cli
```

### Commands

| Command | Description |
|---------|-------------|
| `apr run` | Run model directly (auto-download, cache, execute) |
| `apr compile` | Build standalone executable with embedded model |
| `apr inspect` | Inspect model metadata, vocab, and structure |
| `apr debug` | Simple debugging output ("drama" mode available) |
| `apr validate` | Validate model integrity and quality |
| `apr diff` | Compare two models |
| `apr tensors` | List tensor names, shapes, and statistics |
| `apr trace` | Layer-by-layer trace analysis |
| `apr lint` | Check for best practices and conventions |
| `apr explain` | Explain errors, architecture, and tensors |
| `apr canary` | Regression testing via tensor statistics |
| `apr export` | Export to SafeTensors, GGUF formats |
| `apr import` | Import from HuggingFace, SafeTensors |
| `apr convert` | Quantization (int8, int4, fp16) and optimization |
| `apr merge` | Merge models (average, weighted strategies) |
| `apr tui` | Interactive terminal UI |
| `apr probar` | Export for visual testing |

### Quick Examples

```bash
# Run model directly (auto-downloads if needed)
apr run hf://openai/whisper-tiny --input audio.wav

# Build standalone executable with embedded model
apr compile whisper.apr --quantize int8 -o whisper-cli

# Validate model integrity
apr validate model.apr --quality

# Convert with quantization
apr convert model.safetensors --quantize int8 -o model-int8.apr

# Lint for best practices
apr lint model.apr

# Export to GGUF (llama.cpp compatible)
apr export model.apr --format gguf -o model.gguf

# Merge models (ensemble)
apr merge model1.apr model2.apr --strategy average -o ensemble.apr

# Create regression test
apr canary create model.apr --input ref.wav --output canary.json

# Check model against canary
apr canary check optimized.apr --canary canary.json
```

## Documentation

| Resource | Link |
|----------|------|
| API Reference | [docs.rs/aprender](https://docs.rs/aprender) |
| User Guide | [paiml.github.io/aprender](https://paiml.github.io/aprender/) |
| Examples | [`examples/`](examples/) |
| APR Format Spec | [`docs/specifications/APR-SPEC.md`](docs/specifications/APR-SPEC.md) |

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
