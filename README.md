<p align="center">
  <img src="docs/hero.svg" alt="aprender — Next-generation ML framework in pure Rust" width="100%">
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
  <a href="https://crates.io/crates/aprender">
    <img src="https://img.shields.io/crates/d/aprender.svg" alt="Downloads">
  </a>
  <a href="https://blog.rust-lang.org/2025/03/10/Rust-1.86.0.html">
    <img src="https://img.shields.io/badge/MSRV-1.86-blue.svg" alt="MSRV 1.86">
  </a>
</p>

## Table of Contents

- [Quick Start](#quick-start)
- [What is Aprender?](#what-is-aprender)
- [Supported Architectures](#supported-architectures)
- [Install](#install)
- [CLI Examples](#cli-examples)
- [Library Usage](#library-usage)
- [Architecture](#architecture)
- [Performance](#performance)
- [Framework Comparison](#framework-comparison)
- [Provable Contracts](#provable-contracts)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)

## Quick Start

```bash
cargo install aprender
apr pull qwen2.5-coder-1.5b
apr run qwen2.5-coder-1.5b "What is 2+2?"
# => 2 + 2 = 4.
```

## What is Aprender?

Aprender is a complete ML framework built from scratch in Rust. One `cargo install`,
one `apr` binary, 57 commands covering the full ML lifecycle.

> **Last verified**: 2026-04-09 against apr 0.29.3 on Rust 1.93. All examples below are CI-tested.

| Stage | Commands | What it does |
|-------|----------|-------------|
| **Inference** | `apr run`, `apr chat`, `apr serve` | Run models locally (GGUF, SafeTensors, APR) |
| **Training** | `apr finetune`, `apr train`, `apr distill` | LoRA/QLoRA fine-tuning, knowledge distillation |
| **Model Ops** | `apr convert`, `apr quantize`, `apr merge`, `apr export` | Format conversion, quantization, model merging |
| **Inspection** | `apr inspect`, `apr validate`, `apr tensors`, `apr diff` | Model debugging, validation, comparison |
| **Profiling** | `apr profile`, `apr bench`, `apr qa` | Roofline analysis, benchmarks, QA gates |
| **Registry** | `apr pull`, `apr list`, `apr rm`, `apr publish` | HuggingFace Hub integration, model cache |
| **GPU** | `apr gpu`, `apr parity`, `apr ptx` | GPU status, CPU/GPU parity checks, PTX analysis |
| **Monitoring** | `apr tui`, `apr monitor`, `apr cbtop` | Terminal UI, training monitor, ComputeBrick pipeline |

### Numbers

- **70** workspace crates (was 20 separate repos)
- **25,806** tests, all passing
- **547** provable contracts (equation-based verification)
- **57** CLI commands with contract coverage
- **0** `[patch.crates-io]` — clean workspace deps

## Supported Architectures

| Architecture | Models | Inference | Training | Formats |
|-------------|--------|-----------|----------|---------|
| Qwen2 | Qwen2.5-Coder 0.5B-32B | Yes | LoRA | GGUF, SafeTensors, APR |
| Qwen3 | Qwen3-0.6B-235B | Yes | LoRA | GGUF, SafeTensors, APR |
| LLaMA | LLaMA-2 7B-70B, LLaMA-3 8B-70B | Yes | LoRA | GGUF, SafeTensors, APR |
| Mistral | Mistral 7B, Mixtral 8x7B | Yes | LoRA | GGUF, SafeTensors, APR |
| Phi | Phi-2, Phi-3 Mini/Small/Medium | Yes | LoRA | GGUF, SafeTensors, APR |
| Gemma | Gemma 2B-27B | Yes | LoRA | GGUF, SafeTensors, APR |
| GPT-NeoX | Pythia, StableLM | Yes | - | GGUF, SafeTensors |
| DeepSeek | DeepSeek-V2 | Yes | - | GGUF, SafeTensors |
| Falcon | Falcon 7B-180B | Yes | - | GGUF, SafeTensors |
| StarCoder | StarCoder, StarCoder2 | Yes | - | GGUF, SafeTensors |

**Not yet supported**: Qwen3.5 (SSM/Gated Delta Net layers — [GH-704](https://github.com/paiml/aprender/issues/704)),
Mamba, RWKV, and other pure-SSM architectures.

## Install

```bash
# Install the `apr` binary
cargo install aprender

# Verify
apr --version
```

## CLI Examples

```bash
# Run inference
apr run hf://Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF "Explain quicksort"
apr chat hf://meta-llama/Llama-3-8B-Instruct-GGUF

# Serve model as API
apr serve model.gguf --port 8080

# Inspect model
apr inspect model.gguf
apr validate model.apr --quality --strict
apr tensors model.gguf | head -20

# Fine-tune with LoRA
apr finetune model.gguf --adapter lora --rank 64 --data train.jsonl

# Convert formats
apr convert model.safetensors --quantize q4_k -o model.gguf
apr export model.apr --format gguf -o model.gguf

# Profile
apr profile model.gguf --roofline
apr bench model.gguf --assert-tps 100
```

## Library Usage

The ML library is available as `aprender` on crates.io:

```toml
[dependencies]
aprender-core = "0.29"
```

```rust
use aprender::linear_regression::LinearRegression;
use aprender::traits::Estimator;

let model = LinearRegression::new();
model.fit(&x_train, &y_train)?;
let predictions = model.predict(&x_test)?;
```

Algorithms: Linear/Logistic Regression, Decision Trees, Random Forest, GBM,
Naive Bayes, KNN, SVM, K-Means, PCA, ARIMA, ICA, GLMs, Graph algorithms,
Bayesian inference, text processing, audio processing.

## Architecture

Monorepo with 70 crates in flat `crates/aprender-*` layout
(same pattern as [Polars](https://github.com/pola-rs/polars),
[Burn](https://github.com/tracel-ai/burn),
[Nushell](https://github.com/nushell/nushell)):

```text
paiml/aprender/
├── Cargo.toml                    # Workspace root + `cargo install aprender`
├── crates/
│   ├── aprender-core/            # ML library (use aprender::*)
│   ├── apr-cli/                  # CLI logic (57 commands)
│   ├── aprender-compute/         # SIMD/GPU compute (was: trueno)
│   ├── aprender-gpu/             # CUDA PTX kernels (was: trueno-gpu)
│   ├── aprender-serve/           # Inference server (was: realizar)
│   ├── aprender-train/           # Training loops (was: entrenar)
│   ├── aprender-orchestrate/     # Agents, RAG (was: batuta)
│   ├── aprender-contracts/       # Provable contracts (was: provable-contracts)
│   ├── aprender-profile/         # Profiling (was: renacer)
│   ├── aprender-present-*/       # TUI framework (was: presentar)
│   ├── aprender-db/              # Embedded analytics DB
│   ├── aprender-graph/           # Graph database
│   ├── aprender-rag/             # RAG pipeline
│   └── ... (70 crates total)
├── contracts/                    # 547 provable YAML contracts
└── book/                         # mdBook documentation
```

## Performance

| Model | Format | Speed | Hardware |
|-------|--------|-------|----------|
| Qwen2.5-Coder 1.5B | Q4_K | 40+ tok/s | CPU (AVX2) |
| Qwen2.5-Coder 7B | Q4_K | 225+ tok/s | RTX 4090 |
| TinyLlama 1.1B | Q4_0 | 17 tok/s | CPU (APR format) |

## Framework Comparison

Benchmarked against real inference engines on Qwen2.5-Coder 7B Q4_K (RTX 4090).
Data from [candle-vs-apr](https://github.com/paiml/candle-vs-apr) proof-of-concept:

### Inference Speed (single request, decode tok/s)

| Engine | tok/s | vs Candle | Architecture |
|--------|-------|-----------|--------------|
| llama.cpp b7746 | **443.6** | 1.95x | C++, Flash Attention |
| **aprender (realizr)** | **369.9** | **1.63x** | Rust, CUDA graph + Flash Decoding |
| Candle | 227.4 | 1.00x | Rust, per-op dispatch |

### Batched Throughput (aprender only — Candle has no server)

| Concurrency | Agg tok/s | Scaling | Method |
|-------------|-----------|---------|--------|
| 1 | 367 | 1.0x | Single request |
| 8 | 954 | 2.6x | Continuous batching |
| 32 | **3,220** | **8.8x** | Orca-style iteration scheduling |

### Why Aprender Beats Candle

| aprender advantage | Candle limitation | Impact |
|---|---|---|
| CUDA graph (647 kernels, 1 launch) | Per-op dispatch (~640 launches) | **+26%** |
| Flash Decoding (chunked KV) | Standard SDPA | **+15%** long ctx |
| Fused DP4A GEMV (4-bit native) | Separate dequant + matmul | **~10%** |
| Continuous batching server | CLI only, no server | **8.8x** at c=32 |

### ML Training: apr vs Ludwig

From [ground-truth-apr-ludwig](https://github.com/paiml/ground-truth-apr-ludwig)
(21 recipes, Popperian falsification methodology):

| Capability | aprender | Ludwig | Notes |
|-----------|----------|--------|-------|
| Classification (Iris, Wine) | `apr finetune` | `ludwig train` | Both achieve >95% accuracy |
| LoRA fine-tuning | `apr finetune --lora` | Not native | apr: rank-64 in minutes |
| Quantization (INT8/INT4) | `apr quantize` | Not supported | apr-native capability |
| Model merging | `apr merge --strategy ties` | Not supported | TIES/DARE/SLERP |
| Provable contracts | 547 YAML contracts | None | Equation-based verification |
| Single binary | `cargo install aprender` | `pip install ludwig` | Rust vs Python |

*All benchmarks reproducible from linked repos with `cargo test`.*

## Provable Contracts

Every CLI command and kernel has a provable contract (`contracts/*.yaml`)
with equations, preconditions, postconditions, and falsification tests:

```yaml
equations:
  validate_exit_code:
    formula: exit_code = if score < 50 then 5 else 0
    invariants:
    - score < 50 implies exit_code != 0
falsification_tests:
- id: FALSIFY-CLI-001
  prediction: apr validate bad-model.apr exits non-zero
```

547 contracts across inference, training, quantization, attention, FFN,
tokenization, model formats, and CLI safety.

## Documentation

- [API Reference (docs.rs)](https://docs.rs/aprender) — Full Rust API documentation
- [Book (mdBook)](book/) — Tutorials, guides, and deep dives
- [Specifications](docs/specifications/) — 394 spec files across 22 subsystems
- [Contracts](contracts/) — 547 provable YAML contracts with equations and falsification tests
- [CLI Reference](https://docs.rs/aprender/latest/aprender/) — 57 commands documented

## Migration from Old Crates

All old crate names still work via backward-compatible shim crates:

| Old | New | Status |
|-----|-----|--------|
| `trueno = "0.18"` | `aprender-compute = "0.29"` | Shim available |
| `entrenar = "0.7"` | `aprender-train = "0.29"` | Shim available |
| `realizar = "0.8"` | `aprender-serve = "0.29"` | Shim available |
| `batuta = "0.7"` | `aprender-orchestrate = "0.29"` | Shim available |

Old repositories are archived and read-only. All development happens here.

## Contributing

```bash
git clone https://github.com/paiml/aprender
cd aprender
cargo test --workspace --lib    # 25,806 tests
cargo check --workspace         # 70 crates
apr --help                      # 57 commands
```

## License

MIT
