<h1 align="center">aprender</h1>

<p align="center">
  <strong>Production ML Library in Pure Rust</strong>
</p>

<p align="center">
  <a href="https://crates.io/crates/aprender">
    <img src="https://img.shields.io/crates/v/aprender.svg" alt="crates.io">
  </a>
  <a href="https://docs.rs/aprender">
    <img src="https://docs.rs/aprender/badge.svg" alt="docs.rs">
  </a>
  <a href="https://github.com/paiml/aprender/actions/workflows/ci.yml">
    <img src="https://github.com/paiml/aprender/actions/workflows/ci.yml/badge.svg" alt="CI">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT License">
  </a>
  <a href="#">
    <img src="https://img.shields.io/badge/MSRV-1.89-blue.svg" alt="MSRV: 1.89">
  </a>
</p>

A pure Rust machine learning library with SIMD acceleration, GPU
inference, and the APR v2 model format. No Python dependencies, no C
bindings -- memory-safe, thread-safe, and WebAssembly-ready.

---

[Features](#features) | [Installation](#installation) | [Quick Start](#quick-start) | [Algorithms](#algorithms) | [APR Format](#apr-format) | [Architecture](#architecture) | [Quality](#quality) | [Sovereign Stack](#sovereign-ai-stack) | [Documentation](#documentation) | [License](#license)

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Algorithms](#algorithms)
- [APR Format](#apr-format)
- [Architecture](#architecture)
- [Quality](#quality)
- [Sovereign AI Stack](#sovereign-ai-stack)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)

## What is aprender?

Aprender is a production-ready machine learning library written entirely
in Rust. It provides classical and modern ML algorithms -- from linear
regression and k-means to neural networks and transformers -- with SIMD
acceleration via [trueno](https://github.com/paiml/trueno) and GPU
inference via [realizar](https://github.com/paiml/realizar). Models
serialize to the APR v2 format with LZ4/ZSTD compression, zero-copy
loading, and optional AES-256-GCM encryption.

Aprender is the ML foundation of the PAIML Sovereign AI Stack.

## Features

- **Pure Rust** -- Zero C/C++ dependencies, memory-safe and thread-safe
  by default.
- **SIMD Acceleration** -- Vectorized operations via
  [trueno](https://github.com/paiml/trueno) 0.16 (AVX2/AVX-512/NEON).
- **GPU Inference** -- CUDA-accelerated inference via
  [realizar](https://github.com/paiml/realizar) (67.8 tok/s 7B, 851
  tok/s 1.5B batched on RTX 4090).
- **APR v2 Format** -- Native model serialization with LZ4/ZSTD
  compression, zero-copy loading, and Int4/Int8 quantization.
- **Multi-Format** -- Native `.apr`, SafeTensors (single + sharded), and
  GGUF support.
- **WebAssembly Ready** -- Compile to WASM for browser and edge
  deployment.
- **12,974+ Tests** -- 96.35% coverage, 73 provable contracts.

## Installation

Add aprender to your `Cargo.toml`:

```toml
[dependencies]
aprender = "0.27"
```

### Optional Features

```toml
[dependencies]
aprender = { version = "0.27", features = ["format-encryption", "hf-hub-integration"] }
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
    println!("R-squared = {:.4}", model.score(&x, &y));

    Ok(())
}
```

## Algorithms

### Linear Models

| Algorithm | Module |
|-----------|--------|
| `LinearRegression` | `linear_model` |
| `LogisticRegression` | `linear_model` |
| `LinearSVM` | `classification` |

### Clustering

| Algorithm | Module |
|-----------|--------|
| `KMeans` | `cluster` |
| `DBSCAN` | `cluster` |

### Classification

| Algorithm | Module |
|-----------|--------|
| `NaiveBayes` | `classification` |
| `KNeighborsClassifier` | `classification` |
| `RandomForestClassifier` | `ensemble` |
| `GradientBoostingClassifier` | `ensemble` |

### Decision Trees

| Algorithm | Module |
|-----------|--------|
| `DecisionTreeClassifier` | `tree` |

### NLP and Text

| Capability | Module |
|------------|--------|
| Tokenization (BPE, WordPiece) | `text` |
| TF-IDF vectorization | `text` |
| Stemming | `text` |
| Chat templates (ChatML, Llama2, Mistral, Phi) | `text` |

### Time Series

| Capability | Module |
|------------|--------|
| ARIMA forecasting | `time_series` |

### Graph Analysis

| Capability | Module |
|------------|--------|
| PageRank | `graph` |
| Betweenness centrality | `graph` |
| Community detection | `graph` |

### Bayesian Inference

| Capability | Module |
|------------|--------|
| Gaussian Naive Bayes | `bayesian` |

### Neural Networks

| Capability | Module |
|------------|--------|
| Sequential models | `nn` |
| Transformer layers | `nn` |
| Mixture of Experts | `nn` |

### Decomposition

| Algorithm | Module |
|-----------|--------|
| PCA | `decomposition` |

### Model Serialization (APR v2)

| Capability | Module |
|------------|--------|
| Save/load with encryption | `format` |
| LZ4/ZSTD compression | `format` |
| Zero-copy memory-mapped loading | `format` |
| Ed25519 signatures | `format` |

### Online Learning

| Capability | Module |
|------------|--------|
| Incremental model updates | `online` |

### Recommendation Systems

| Capability | Module |
|------------|--------|
| Collaborative filtering | `recommend` |

## APR Format

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

| Feature | APR v1 | APR v2 |
|---------|--------|--------|
| Tensor Compression | None | LZ4/ZSTD |
| Index Format | JSON | Binary |
| Zero-Copy Loading | Partial | Full |
| Quantization | Int8 | Int4/Int8 |
| Streaming | No | Yes |

## Architecture

```
aprender
  primitives/       Core tensor types (Matrix, Vector)
  linear_model/     Linear and logistic regression
  cluster/          KMeans, DBSCAN
  classification/   SVM, Naive Bayes, KNN
  tree/             Decision trees
  ensemble/         Random forest, gradient boosting
  text/             Tokenization, TF-IDF, chat templates
  time_series/      ARIMA
  graph/            PageRank, centrality, community detection
  bayesian/         Bayesian inference
  glm/              Generalized linear models
  decomposition/    PCA, dimensionality reduction
  nn/               Neural networks, transformers, MoE
  online/           Online / incremental learning
  recommend/        Recommendation systems
  synthetic/        Synthetic data generation
  format/           APR v2 serialization (encryption, compression)
  serialization/    Legacy serialization
  prelude/          Convenient re-exports
```

Key dependency: [trueno](https://crates.io/crates/trueno) provides the
SIMD-accelerated compute primitives (AVX2/AVX-512/NEON).

## Quality

- **12,974+ tests**, 96.35% line coverage
- Zero clippy warnings (`-D warnings`)
- 73 provable contracts via
  [provable-contracts](https://github.com/paiml/provable-contracts)
- TDG Score: A+ (95.2/100)
- Mutation testing target: >80% mutation score
- Zero SATD (self-admitted technical debt)

## Sovereign AI Stack

Aprender is the ML layer of the PAIML Sovereign AI Stack -- a pure Rust
ecosystem for privacy-preserving ML infrastructure.

| Layer | Crate | Purpose |
|-------|-------|---------|
| Compute | [trueno](https://crates.io/crates/trueno) | SIMD/GPU primitives (AVX2/AVX-512/NEON, wgpu) |
| ML | **aprender** | ML algorithms, APR v2 format |
| Training | [entrenar](https://crates.io/crates/entrenar) | Autograd, LoRA/QLoRA, quantization |
| Inference | [realizar](https://crates.io/crates/realizar) | APR/GGUF/SafeTensors inference, GPU kernels |
| Speech | [whisper-apr](https://crates.io/crates/whisper-apr) | Pure Rust Whisper ASR |
| Distribution | [repartir](https://crates.io/crates/repartir) | Distributed compute (CPU/GPU/Remote) |
| Simulation | [simular](https://crates.io/crates/simular) | Monte Carlo, physics, optimization |
| Registry | [pacha](https://crates.io/crates/pacha) | Model registry with Ed25519 signatures |
| Orchestration | [batuta](https://crates.io/crates/batuta) | Stack coordination and CLI |

### Related Crates

| Crate | Description |
|-------|-------------|
| [aprender-tsp](https://crates.io/crates/aprender-tsp) | TSP solver with CLI and `.apr` model persistence |
| [aprender-shell](https://crates.io/crates/aprender-shell) | AI-powered shell completion trained on your history |
| [apr-cookbook](https://github.com/paiml/apr-cookbook) | 50+ idiomatic Rust examples for `.apr` format and SIMD acceleration |

## Documentation

| Resource | Link |
|----------|------|
| API Reference | [docs.rs/aprender](https://docs.rs/aprender) |
| User Guide | [paiml.github.io/aprender](https://paiml.github.io/aprender/) |
| Examples | [`examples/`](examples/) |
| APR Format Spec | [`docs/specifications/archive/APR-SPEC.md`](docs/specifications/archive/APR-SPEC.md) |

## Contributing

1. Fork the repository
2. Make your changes on the `master` branch
3. Run quality gates: `cargo test --all-features && cargo clippy --all-targets -- -D warnings && cargo fmt --check`
4. Submit a pull request

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT

---

<p align="center">
  <sub>Built by <a href="https://paiml.com">PAIML</a></sub>
</p>
