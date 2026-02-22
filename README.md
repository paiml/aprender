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
- [Usage](#usage)
- [Algorithms](#algorithms)
- [Model Persistence](#model-persistence)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Pure Rust** — Zero C/C++ dependencies, memory-safe, thread-safe by default
- **SIMD Acceleration** — Vectorized operations via [trueno](https://github.com/paiml/trueno) backend
- **GPU Inference** — CUDA-accelerated inference via [realizar](https://github.com/paiml/realizar) (67.8 tok/s 7B, 851 tok/s 1.5B)
- **Multi-Format** — Native `.apr`, SafeTensors (single + sharded), and GGUF support
- **WebAssembly Ready** — Compile to WASM for browser and edge deployment
- **11,251 Tests** — 96.35% coverage, zero SATD, TDG 96.9/100 A+

## Installation

Add aprender to your `Cargo.toml`:

```toml
[dependencies]
aprender = "0.25"
```

### Optional Features

```toml
[dependencies]
aprender = { version = "0.25", features = ["format-encryption", "hf-hub-integration"] }
```

| Feature | Description |
|---------|-------------|
| `format-encryption` | AES-256-GCM encryption for model files |
| `format-signing` | Ed25519 digital signatures |
| `format-compression` | Zstd compression |
| `hf-hub-integration` | Hugging Face Hub push/pull support |
| `gpu` | GPU acceleration via wgpu |

## Usage

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
- **Text Processing** — Tokenization, TF-IDF, stemming, chat templates
- **Neural Networks** — Sequential models, transformers, mixture of experts
- **Metaheuristics** — ACO, Tabu Search, DE, PSO, GA, CMA-ES

### Chat Templates

Format LLM conversations for different model families with automatic template detection:

```rust
use aprender::text::chat_template::{
    auto_detect_template, ChatMessage, ChatTemplateEngine
};

// Auto-detect template from model name
let template = auto_detect_template("Qwen2-0.5B-Instruct");

let messages = vec![
    ChatMessage::system("You are a helpful assistant."),
    ChatMessage::user("Hello!"),
];

let formatted = template.format_conversation(&messages)?;
```

**Supported Formats:**

| Format | Models | System Prompt |
|--------|--------|---------------|
| ChatML | Qwen2, Yi, OpenHermes | Yes |
| Llama2 | TinyLlama, Vicuna, LLaMA 2 | Yes |
| Mistral | Mistral-7B, Mixtral | No |
| Phi | Phi-2, Phi-3 | Yes |
| Alpaca | Alpaca, Guanaco | Yes |
| Raw | Fallback | Passthrough |
| Custom | Any (Jinja2) | Configurable |

See [`examples/chat_template.rs`](examples/chat_template.rs) for complete usage.

**Verification:** All templates are 100% tested via bashrs probar playbooks. See [`docs/model-verification-checklist.md`](docs/model-verification-checklist.md) for coverage status.

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
| `apr serve` | Start inference server (REST API, streaming, metrics) |
| `apr chat` | Interactive chat with language models |
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
| `apr pull` | Download and cache model from HuggingFace (Ollama-style UX) |
| `apr list` | List cached models |
| `apr rm` | Remove model from cache |
| `apr convert` | Quantization (int8, int4, fp16) and optimization |
| `apr merge` | Merge models (average, weighted strategies) |
| `apr tui` | Interactive terminal UI |
| `apr probar` | Export for visual testing |
| `apr tree` | Model architecture tree view |
| `apr hex` | Hex dump tensor data |
| `apr flow` | Data flow visualization |
| `apr bench` | Benchmark throughput (spec H12: >= 10 tok/s) |
| `apr eval` | Evaluate model perplexity (spec H13: PPL <= 20) |
| `apr profile` | Deep profiling with Roofline analysis |
| `apr qa` | Falsifiable QA checklist for model releases |
| `apr qualify` | Cross-subcommand smoke test (does every tool handle this model?) |
| `apr showcase` | Qwen2.5-Coder showcase demo |
| `apr check` | Model self-test: 10-stage pipeline integrity check |
| `apr publish` | Publish model to HuggingFace Hub |
| `apr cbtop` | ComputeBrick pipeline monitor |
| `apr compare-hf` | Compare APR model against HuggingFace source |

### Quick Examples

```bash
# Run model directly (auto-downloads if needed)
apr run hf://openai/whisper-tiny --input audio.wav

# Download and cache models (Ollama-style UX)
apr pull hf://Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF -o ./models/
apr list  # List cached models

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

# Publish to HuggingFace Hub
apr publish ./model-dir/ org/model-name --license mit
```

## Showcase: Qwen2.5-Coder Inference

Multi-model inference across Qwen2.5-Coder 0.5B, 1.5B, 3B, 7B, and 14B — all formats (SafeTensors, GGUF, APR), both CPU and GPU:

```bash
# Run any model (auto-downloads if needed)
apr run hf://Qwen/Qwen2.5-Coder-7B-Instruct --prompt "Write hello world in Rust"

# Sharded SafeTensors supported (3B+)
apr serve /path/to/model.safetensors.index.json --port 8080

# Interactive chat
apr chat qwen2.5-coder-1.5b-q4_k_m.gguf

# Production server (OpenAI-compatible API)
apr serve qwen2.5-coder-7b-q4_k_m.gguf --port 8080 --gpu
```

### Benchmark Results (2026-02-11)

**7B Q4_K_M on RTX 4090:**

| Mode | Throughput | vs Ollama | Status |
|------|------------|-----------|--------|
| GPU Decode | **67.8 tok/s** | **0.6x** (Grade D) | Pass |
| CPU (GGUF) | 8 tok/s | — | Pass |

**1.5B Q4_K_M on RTX 4090:**

| Mode | Throughput | vs Ollama | Status |
|------|------------|-----------|--------|
| GPU Batched (M=16) | **851.8 tok/s** | **2.93x** | Pass |
| GPU Single | 120.1 tok/s | 1.0x | Pass |
| CPU | 25.3 tok/s | 1.69x | Pass |

**Supported model sizes:** 0.5B, 1.5B, 3B, 7B, 14B (SafeTensors sharded, GGUF Q4_K, APR native).

See [`docs/specifications/qwen2.5-coder-showcase-demo.md`](docs/specifications/qwen2.5-coder-showcase-demo.md) for full benchmark methodology and the 43-round Popperian falsification protocol (206 bugs found and fixed).

## QA & Testing

The project includes comprehensive QA infrastructure for model validation:

```bash
# Run 7-gate QA suite on any model
apr qa model.gguf

# QA with throughput assertions
apr qa model.gguf --assert-tps 100 --json

# MVP playbook testing (18-cell matrix: 3 formats × 2 backends × 3 modalities)
cd apr-model-qa-playbook
apr-qa run playbooks/models/qwen2.5-coder-7b-mvp.playbook.yaml \
  --model-path /path/to/model.safetensors.index.json
```

**QA Gates (7 falsifiable gates):**
1. Tensor contract validation
2. Golden output verification
3. Throughput measurement
4. Ollama parity comparison
5. GPU speedup verification
6. Format parity (SafeTensors vs GGUF vs APR)
7. PTX parity (GPU kernel correctness)

**QA Matrix Coverage:**
- **Modalities**: `run`, `chat`, `serve`
- **Formats**: GGUF, SafeTensors (including sharded), APR
- **Backends**: CPU, GPU
- **Models tested**: 0.5B, 1.5B, 3B, 7B, 14B
- **Falsification**: 43 rounds, 206 bugs found, 155/163 gates passing (95.1%)

## Documentation

| Resource | Link |
|----------|------|
| API Reference | [docs.rs/aprender](https://docs.rs/aprender) |
| User Guide | [paiml.github.io/aprender](https://paiml.github.io/aprender/) |
| Examples | [`examples/`](examples/) |
| APR Format Spec | [`docs/specifications/APR-SPEC.md`](docs/specifications/APR-SPEC.md) |
| QA Protocol | [`docs/specifications/qa-showcase-methodology.md`](docs/specifications/qa-showcase-methodology.md) |
| Qualify Matrix | [`docs/qualify-matrix.md`](docs/qualify-matrix.md) |
🤖 [Coursera Hugging Face AI Development Specialization](https://www.coursera.org/specializations/hugging-face-ai-development) - Build Production AI systems with Hugging Face in Pure Rust

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
