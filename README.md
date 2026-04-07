<h1 align="center">aprender</h1>

<p align="center">
  <strong>Next-generation ML framework in pure Rust. One install, one binary, full stack.</strong>
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
</p>

## Quick Start

```bash
cargo install aprender
apr pull qwen2.5-coder-1.5b
apr run qwen2.5-coder-1.5b "What is 2+2?"
```

## What is Aprender?

Aprender is a complete ML framework built from scratch in Rust. One `cargo install`,
one `apr` binary, 57 commands covering the full ML lifecycle:

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
- **25,391** tests, all passing
- **405** provable contracts (equation-based verification)
- **57** CLI commands with contract coverage
- **0** `[patch.crates-io]` — clean workspace deps

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

```
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
├── contracts/                    # 405 provable YAML contracts
└── book/                         # mdBook documentation
```

## Performance

| Model | Format | Speed | Hardware |
|-------|--------|-------|----------|
| Qwen2.5-Coder 1.5B | Q4_K | 40+ tok/s | CPU (AVX2) |
| Qwen2.5-Coder 7B | Q4_K | 225+ tok/s | RTX 4090 |
| TinyLlama 1.1B | Q4_0 | 17 tok/s | CPU (APR format) |

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

405 contracts across inference, training, quantization, attention, FFN,
tokenization, model formats, and CLI safety.

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
cargo test --workspace --lib    # 25,391 tests
cargo check --workspace         # 70 crates
apr --help                      # 57 commands
```

## License

MIT
