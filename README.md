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
</p>

## Quick Start

```bash
cargo install aprender
apr pull qwen2.5-coder-1.5b
apr run qwen2.5-coder-1.5b "What is 2+2?"
```

## What is Aprender?

A complete ML framework in pure Rust. One `cargo install`, one `apr` binary,
the full model lifecycle — inference, training, quantization, profiling,
publishing — all backed by YAML provable contracts that fail CI on drift.

### At HEAD

| Metric | Count | Source of truth |
|-------:|------:|---|
| Workspace crates | **80** workspace crates | `ls crates/` |
| Provable contracts | **1105** provable contracts | `find contracts/ -name '*.yaml'` |
| CLI commands | **80** CLI commands | `apr --help` |

These numbers are enforced by [`contracts/readme-claims-v1.yaml`](contracts/readme-claims-v1.yaml).
Drift between this table and live repo state fails `bash scripts/check_readme_claims.sh`
→ see [FALSIFY-README-001..004](contracts/readme-claims-v1.yaml).

### Command surface

| Stage | Commands |
|---|---|
| **Inference** | `apr run`, `apr chat`, `apr serve` |
| **Training** | `apr finetune`, `apr train`, `apr pretrain`, `apr distill` |
| **Model ops** | `apr convert`, `apr quantize`, `apr merge`, `apr export`, `apr compile` |
| **Inspection** | `apr inspect`, `apr validate`, `apr tensors`, `apr diff`, `apr trace`, `apr lint` |
| **Profiling** | `apr profile`, `apr bench`, `apr qa` |
| **Registry** | `apr pull`, `apr list`, `apr rm`, `apr publish`, `apr registry` |
| **GPU** | `apr gpu`, `apr parity`, `apr ptx` |
| **Observability** | `apr tui`, `apr monitor`, `apr cbtop` |

## Cookbook

End-to-end recipes (data prep → train → quantize → publish → serve) live in
[`paiml/apr-cookbook`](https://github.com/paiml/apr-cookbook) — 341 worked
examples with local `book/src/` walkthroughs.

```bash
git clone https://github.com/paiml/apr-cookbook  # ../apr-cookbook
cd ../apr-cookbook
apr cookbook list                              # 341 recipes
apr cookbook run train-tiny-from-scratch       # runs end-to-end
```

## Install

```bash
cargo install aprender    # installs the `apr` binary
apr --version
```

## CLI examples

```bash
# Run inference (local or HF)
apr run hf://Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF "Explain quicksort"
apr chat hf://meta-llama/Llama-3-8B-Instruct-GGUF

# Serve
apr serve model.gguf --port 8080

# Inspect
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

## Library usage

```toml
[dependencies]
aprender = "0.31"
```

```rust
use aprender::linear_regression::LinearRegression;
use aprender::traits::Estimator;

let model = LinearRegression::new();
model.fit(&x_train, &y_train)?;
let predictions = model.predict(&x_test)?;
```

Algorithms: Linear/Logistic Regression, Decision Trees, Random Forest, GBM,
Naive Bayes, KNN, SVM, K-Means, PCA, ARIMA, ICA, GLMs, graph algorithms,
Bayesian inference, text + audio processing.

## Architecture

Monorepo, flat `crates/aprender-*` layout (same pattern as
[Polars](https://github.com/pola-rs/polars),
[Burn](https://github.com/tracel-ai/burn),
[Nushell](https://github.com/nushell/nushell)):

```
paiml/aprender/
├── Cargo.toml                      # Workspace root + `cargo install aprender`
├── crates/
│   ├── aprender-core/              # ML library (use aprender::*)
│   ├── apr-cli/                    # CLI logic (80 subcommands)
│   ├── aprender-compute/           # SIMD/GPU compute kernels
│   ├── aprender-gpu/               # CUDA PTX
│   ├── aprender-serve/             # Inference server
│   ├── aprender-train/             # Training loops
│   ├── aprender-orchestrate/       # Agents + RAG
│   ├── aprender-contracts/         # Provable contracts engine
│   ├── aprender-profile/           # Profiling
│   ├── aprender-db/ aprender-graph/ aprender-rag/
│   └── ... (80 crates total)
├── contracts/                      # 1105 provable YAML contracts
└── book/                           # mdBook documentation
```

## Performance

| Model | Format | Speed | Hardware |
|-------|--------|-------|----------|
| Qwen2.5-Coder 1.5B | Q4_K | 40+ tok/s | CPU (AVX2) |
| Qwen2.5-Coder 7B | Q4_K | 225+ tok/s | RTX 4090 |
| TinyLlama 1.1B | Q4_0 | 17 tok/s | CPU (APR format) |

Reproduced from [candle-vs-apr](https://github.com/paiml/candle-vs-apr) and
[ground-truth-apr-ludwig](https://github.com/paiml/ground-truth-apr-ludwig).

## Provable contracts

Every CLI command and kernel is bound to a YAML contract with equations,
preconditions, postconditions, and falsification tests:

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

1105 contracts across inference, training, quantization, attention, FFN,
tokenization, model formats, CLI safety — and this README itself.

## Migration from old crates

| Old | New | Status |
|-----|-----|--------|
| `trueno = "0.18"` | `aprender-compute = "0.31"` | Shim available |
| `entrenar = "0.7"` | `aprender-train = "0.31"` | Shim available |
| `realizar = "0.8"` | `aprender-serve = "0.31"` | Shim available |
| `batuta = "0.7"` | `aprender-orchestrate = "0.31"` | Shim available |

Old repositories are archived. All development happens here.

## Contributing

```bash
git clone https://github.com/paiml/aprender
cd aprender
cargo test --workspace --lib
cargo check --workspace
apr --help
bash scripts/check_readme_claims.sh    # README contract gate
```

## License

MIT
