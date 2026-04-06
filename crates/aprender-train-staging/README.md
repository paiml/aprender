<h1 align="center">entrenar</h1>

<p align="center">
  <strong>Training Framework for the Sovereign AI Stack</strong>
</p>

<p align="center">
  <a href="https://crates.io/crates/entrenar">
    <img src="https://img.shields.io/crates/v/entrenar.svg" alt="crates.io">
  </a>
  <a href="https://docs.rs/entrenar">
    <img src="https://docs.rs/entrenar/badge.svg" alt="docs.rs">
  </a>
  <a href="https://github.com/paiml/entrenar/actions">
    <img src="https://github.com/paiml/entrenar/actions/workflows/ci.yml/badge.svg"
         alt="CI">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT License">
  </a>
  <a href="https://blog.rust-lang.org/2025/05/15/Rust-1.87.0.html">
    <img src="https://img.shields.io/badge/rust-1.87%2B-orange.svg" alt="Rust 1.87+">
  </a>
</p>

A pure Rust training framework providing autograd, LoRA/QLoRA
fine-tuning, quantization (Int4/Int8), model merging, knowledge
distillation, and Compiler-in-the-Loop (CITL) training. Built on
[trueno](https://crates.io/crates/trueno) for SIMD-accelerated compute
and [aprender](https://crates.io/crates/aprender) for ML algorithms.

---

[Features](#features) | [Installation](#installation) | [Usage](#usage) | [Architecture](#architecture) | [Quality](#quality) | [Sovereign Stack](#sovereign-ai-stack) | [Documentation](#documentation) | [License](#license)

---

## Table of Contents

- [What is entrenar?](#what-is-entrenar)
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Architecture](#architecture)
- [Quality](#quality)
- [Sovereign AI Stack](#sovereign-ai-stack)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)

## What is entrenar?

**Entrenar** (Spanish: "to train") is a production-grade neural network
training library in pure Rust. It provides everything needed to train,
fine-tune, quantize, merge, and distill models -- with no Python
dependency.

Core capabilities:

- **Autograd Engine** -- Tape-based reverse-mode automatic differentiation
- **Optimizers** -- SGD, Adam, AdamW with cosine scheduling and gradient clipping
- **LoRA / QLoRA** -- Parameter-efficient fine-tuning with 4-bit quantized base weights
- **Quantization** -- QAT, PTQ, GGUF-compatible Q4_0/Q8_0, NF4 training
- **Model Merging** -- TIES, DARE, SLERP algorithms
- **Knowledge Distillation** -- Multi-teacher, progressive layer-wise
- **CITL** -- Compiler-in-the-Loop training for transpiler optimization
- **GPU Training** -- WGPU backend (AMD/Intel/cross-platform), CUDA/cuBLAS (NVIDIA)
- **Monitoring** -- Real-time metrics, drift detection, Andon alerts

Part of the [PAIML Sovereign AI Stack](https://github.com/paiml).

## Installation

### Library

Add to your `Cargo.toml`:

```toml
[dependencies]
entrenar = "0.7"
```

### CLI

```bash
cargo install entrenar
```

### From source

```bash
git clone https://github.com/paiml/entrenar
cd entrenar
cargo install --path .
```

## Usage

### Basic Training

```rust
use entrenar::train::{Trainer, TrainConfig, MSELoss, EarlyStopping};
use entrenar::optim::Adam;
use entrenar::Tensor;

let params = vec![Tensor::zeros(784 * 128, true)];
let optimizer = Adam::new(0.001, 0.9, 0.999, 1e-8);

let mut trainer = Trainer::new(params, Box::new(optimizer), TrainConfig::default());
trainer.set_loss(Box::new(MSELoss));
trainer.add_callback(EarlyStopping::new(5, 0.001));

let result = trainer.train(100, || batches.clone(), |x| model.forward(x));
println!("Final loss: {:.4}", result.final_loss);
```

### Autograd

```rust
use entrenar::autograd::{matmul, softmax, layer_norm, attention};

let y = matmul(&x, &w);
let s = softmax(&logits);
let n = layer_norm(&x, &gamma, &beta);
let a = attention(&q, &k, &v);
```

### LoRA / QLoRA Fine-Tuning

```rust
use entrenar::lora::{LoRALayer, QLoRALayer};

// Standard LoRA
let lora = LoRALayer::new(4096, 4096, 16, 32.0);

// QLoRA: 4-bit base + FP16 adapters (7B model: 28GB -> 3.5GB)
let qlora = QLoRALayer::new(base_weights, 16, 32.0);
```

### Quantization

```rust
use entrenar::quant::{FakeQuantize, PTQCalibrator, GGUFQuantizer};

let fq = FakeQuantize::new(8, true);            // QAT with STE
let calibrator = PTQCalibrator::percentile(0.999); // Post-training
let quantizer = GGUFQuantizer::q4_0();           // GGUF export
```

### Model Merging

```rust
use entrenar::merge::{TiesMerge, DareMerge, SlerpMerge};

let merged = TiesMerge::new(0.2).merge(&models, &weights);
let merged = DareMerge::new(0.9).merge(&base, &finetuned);
let merged = SlerpMerge::new().merge(&a, &b, 0.5);
```

### Declarative Configuration

```yaml
# train.yaml
model:
  path: base-model.gguf
data:
  train: train.parquet
  batch_size: 8
optimizer:
  name: adamw
  lr: 0.0001
lora:
  rank: 64
  alpha: 16
training:
  epochs: 10
  grad_clip: 1.0
```

```bash
entrenar train train.yaml
```

### CLI Commands

```bash
entrenar train config.yaml --epochs 10
entrenar quantize model.safetensors --bits 4 --output model_q4.json
entrenar merge model1.safetensors model2.safetensors --method ties
entrenar bench config.yaml --warmup 5 --iterations 100
entrenar inspect model.safetensors -v
entrenar audit predictions.parquet --type bias --threshold 0.8
entrenar monitor data.parquet --threshold 0.2
```

## Features

### Autograd Engine

Tape-based reverse-mode automatic differentiation with verified
gradients. Supports matmul, softmax, layer normalization, and scaled
dot-product attention. All gradients validated against finite-difference
reference implementations.

### LoRA / QLoRA Fine-Tuning

Parameter-efficient fine-tuning with up to 99.75% parameter reduction.
QLoRA combines 4-bit NF4 quantized base weights with FP16 low-rank
adapters, reducing 7B model memory from 28GB to 3.5GB. PEFT-compatible
adapter export for interoperability with HuggingFace tooling.

### Quantization

Three quantization strategies: Quantization-Aware Training (QAT) with
straight-through estimator, Post-Training Quantization (PTQ) with
percentile calibration, and GGUF-compatible Q4_0/Q8_0 export for
llama.cpp interoperability. NF4 training with cuBLAS backward pass
support.

### Model Merging

Three model merging algorithms for combining fine-tuned checkpoints:
TIES (Trim, Elect Sign, Merge) for multi-model consolidation, DARE
(Dropout And Rescale) for parameter-efficient merging, and SLERP
(Spherical Linear Interpolation) for smooth two-model blending.

### Knowledge Distillation

Temperature-scaled KD loss with configurable alpha weighting between
hard and soft targets. Multi-teacher ensemble distillation with
weighted aggregation. Progressive layer-wise distillation for
large-to-small model transfer.

### CITL (Compiler-in-the-Loop)

Training loop that incorporates compiler feedback for transpiler
optimization. Uses RAG-based fix suggestions via trueno-rag to
guide training toward compilable outputs. Designed for the
depyler/bashrs/decy transpilation stack.

### GPU Training

WGPU backend for cross-platform GPU training (AMD, Intel, Apple
Silicon). NVIDIA CUDA/cuBLAS backend for dedicated GPU acceleration.
NVML integration for real-time GPU monitoring. VRAM ledger with
file-based locking for multi-process coordination.

### Monitoring

Toyota Way-inspired quality monitoring with real-time metrics
collection, drift detection (z-score based), and Andon alert system
for automatic anomaly notification. NaN/Inf detection, gradient
explosion guards, and loss divergence tracking.

### Feature Flags

| Flag | Purpose |
|------|---------|
| `gpu` | GPU-accelerated training via wgpu |
| `cuda` | NVIDIA CUDA/cuBLAS training |
| `citl` | Compiler-in-the-Loop with trueno-rag |
| `monitor` | Training monitoring with trueno-db persistence |
| `server` | REST/HTTP API server via axum |
| `parquet` | Parquet batch loading via alimentar |
| `hub` | HuggingFace Hub model fetching |
| `wasm` | Browser-compatible WASM build |
| `tracing` | Renacer distributed tracing integration |
| `nvml` | Real GPU monitoring via NVIDIA NVML |

## Architecture

```
entrenar/
  autograd/     Tape-based automatic differentiation
  optim/        SGD, Adam, AdamW, schedulers
  lora/         LoRA, QLoRA fine-tuning
  quant/        QAT, PTQ, GGUF quantization
  merge/        TIES, DARE, SLERP merging
  distill/      Knowledge distillation
  finetune/     ClassifyPipeline, ClassifyTrainer, evaluation
  eval/         Classification metrics, drift detection, Andon
  train/        Trainer, callbacks, metrics, WGPU transformer trainer
  monitor/      Real-time monitoring, Andon alerts
  config/       Declarative YAML configuration
  io/           Model persistence (SafeTensors, APR)
```

## Quality

| Metric | Value |
|--------|-------|
| Tests | 7,527+ passing |
| Coverage | 96% |
| TDG Score | A+ (96.8/100) |
| Critical Defects | 0 |
| Property Tests | 200K+ iterations |
| Gradient Checking | Finite-difference validated |
| Mutation Testing | >80% kill rate |
| MSRV | 1.87 |

## Sovereign AI Stack

| Crate | Purpose | Version |
|-------|---------|---------|
| [trueno](https://crates.io/crates/trueno) | SIMD/GPU compute primitives | 0.16.x |
| [aprender](https://crates.io/crates/aprender) | ML algorithms, APR v2 format | 0.27.x |
| **entrenar** | **Training and optimization** | **0.7.x** |
| [realizar](https://crates.io/crates/realizar) | Inference engine (APR/GGUF/SafeTensors) | 0.8.x |
| [repartir](https://crates.io/crates/repartir) | Distributed compute (CPU/GPU/Remote) | 2.0.x |
| [whisper-apr](https://crates.io/crates/whisper-apr) | Pure Rust Whisper ASR | 0.2.x |
| [simular](https://crates.io/crates/simular) | Simulation engine | 0.3.x |
| [batuta](https://crates.io/crates/batuta) | Stack orchestration | 0.7.x |

## Documentation

- [API Reference](https://docs.rs/entrenar) -- Generated from source
- [Book](book/) -- Comprehensive guide with examples
- [Examples](examples/) -- Runnable training, merging, and monitoring examples

## Contributing

1. Fork the repository
2. Create your changes on the `master` branch
3. Run quality gates: `make lint && make test`
4. Run coverage: `make coverage`
5. Submit a pull request

## Cookbook

See [entrenar-cookbook](https://github.com/paiml/entrenar-cookbook) for
examples and recipes.

## License

MIT
