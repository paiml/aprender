<p align="center">
  <img src="assets/hero.svg" width="800" alt="realizar - Production LLM Inference Engine">
</p>

<h1 align="center">realizar</h1>

<p align="center">
  <strong>Production LLM Inference Engine for the Sovereign AI Stack</strong>
</p>

<p align="center">
  <a href="https://crates.io/crates/realizar">
    <img src="https://img.shields.io/crates/v/realizar.svg" alt="crates.io">
  </a>
  <a href="https://docs.rs/realizar">
    <img src="https://docs.rs/realizar/badge.svg" alt="docs.rs">
  </a>
  <a href="https://github.com/paiml/realizar/actions">
    <img src="https://github.com/paiml/realizar/actions/workflows/ci.yml/badge.svg" alt="CI">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT License">
  </a>
  <a href="https://www.rust-lang.org">
    <img src="https://img.shields.io/badge/rust-1.89%2B-orange.svg" alt="Rust 1.89+">
  </a>
</p>

<p align="center">
  <a href="#what-is-realizar">What is realizar?</a> |
  <a href="#installation">Installation</a> |
  <a href="#usage">Usage</a> |
  <a href="#features">Features</a> |
  <a href="#quality">Quality</a> |
  <a href="#sovereign-ai-stack">Stack</a> |
  <a href="#documentation">Docs</a>
</p>

---

## Table of Contents

- [What is realizar?](#what-is-realizar)
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Benchmarks](#benchmarks)
- [Quality](#quality)
- [Sovereign AI Stack](#sovereign-ai-stack)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)

## What is realizar?

realizar is a pure Rust LLM inference engine. It loads models in APR v2,
GGUF, and SafeTensors formats, runs transformer inference with quantized
kernels (Q4\_K through Q8\_0), and serves predictions over an
OpenAI-compatible REST API.

Key design decisions:

- **Row-major mandate** -- All tensors are row-major internally. GGUF
  column-major data is transposed at import by aprender. This matches
  PyTorch/SafeTensors layout and simplifies kernel implementations.
- **Pure Rust CUDA** -- GPU acceleration via trueno-gpu generates PTX
  directly from Rust. No nvcc, no LLVM, no C++ dependencies.
- **Cost-based dispatch** -- Backend selection (GPU/SIMD/scalar) uses a
  5x PCIe cost model to avoid GPU overhead on small workloads.

## Installation

### CLI

```bash
cargo install realizar
```

### Library

Add to your `Cargo.toml`:

```toml
[dependencies]
realizar = "0.8"
```

## Usage

### Serving

```bash
# Start demo server
realizar serve --demo --port 8080

# Health check
curl http://localhost:8080/health

# Prometheus metrics
curl http://localhost:8080/metrics
```

### OpenAI-Compatible API

```bash
# Chat completions
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50
  }'

# Streaming
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

### Library API

```rust
use realizar::chat_template::{auto_detect_template, ChatMessage};

let template = auto_detect_template("Qwen2-0.5B-Instruct");
let messages = vec![
    ChatMessage::system("You are a helpful assistant."),
    ChatMessage::user("Hello!"),
];
let formatted = template.format_conversation(&messages)?;
```

### Tracing

Use the `X-Trace-Level` header for inference debugging:

```bash
# Brick-level: token-by-token timing
curl -H "X-Trace-Level: brick" -X POST http://localhost:8080/v1/chat/completions ...

# Layer-level: per-layer timing breakdown
curl -H "X-Trace-Level: layer" -X POST http://localhost:8080/v1/chat/completions ...
```

## Features

### Model Formats

| Format | Description |
|--------|-------------|
| APR v2 | Native format with LZ4/ZSTD compression, zero-copy loading, Int4/Int8 quantization |
| GGUF | llama.cpp-compatible quantized models |
| SafeTensors | HuggingFace full-precision format |

### GPU Kernels

| Kernel | Purpose |
|--------|---------|
| `GemmKernel` | Matrix multiplication (naive, tiled, tensor core) |
| `AttentionKernel` | FlashAttention-style tiled attention |
| `SoftmaxKernel` | Numerically stable with warp shuffle |
| `LayerNormKernel` | Fused layer normalization |
| `QuantizeKernel` | Q4\_K dequantization fused with matmul |
| `Q5KKernel` | Q5\_K dequantization |
| `Q6KKernel` | Q6\_K dequantization |

### Quantization

Q4\_0, Q8\_0, Q4\_K, Q5\_K, Q6\_K -- SIMD-accelerated on AVX2, AVX-512,
and NEON. GPU dequantization fused with matrix operations to avoid
memory round-trips.

### KV Cache

Autoregressive decoding with persistent key-value cache. Supports
grouped-query attention (GQA) for models like Qwen2.5 and Llama 3.

### Chat Templates

Automatic template detection from model metadata:

| Format | Models | System Prompt |
|--------|--------|---------------|
| ChatML | Qwen2, Yi, OpenHermes | Yes |
| Llama2 | TinyLlama, Vicuna, LLaMA 2 | Yes |
| Mistral | Mistral-7B, Mixtral | No |
| Phi | Phi-2, Phi-3 | Yes |
| Alpaca | Alpaca, Guanaco | Yes |
| Raw | Fallback | Passthrough |
| Custom | Any (Jinja2) | Configurable |

### Feature Flags

| Flag | Description |
|------|-------------|
| `default` | server + cli + gpu |
| `cuda` | NVIDIA CUDA support (pure Rust PTX, no nvcc) |
| `minimal` | Core inference only |
| `bench-http` | External server benchmarking |

## Benchmarks

### LLM Inference (GPU)

| Model | Size | Format | Backend | Throughput |
|-------|------|--------|---------|------------|
| Qwen2.5-Coder Q4\_K\_M | 1.5B | APR | RTX 4090 (CUDA) | 240 tok/s |
| Phi-2 Q4\_K\_M | 2.7B | GGUF | RTX 4090 (CUDA) | 276 tok/s |
| Phi-2 Q4\_K\_M | 2.7B | GGUF | llama.cpp CUDA | 256 tok/s |
| Phi-2 Q4\_K\_M | 2.7B | GGUF | Ollama CUDA | 228 tok/s |

realizar achieves 8--21% faster inference than llama.cpp/Ollama via pure
Rust CUDA PTX generation.

### Classical ML (APR Format)

| Model | Parameters | Latency | Throughput |
|-------|------------|---------|------------|
| Iris | 131 | 103ns | 9.6M inferences/sec |
| MNIST | 103K | 73us | 13.6K inferences/sec |
| Large NN | 1M | 410us | 2.4K inferences/sec |

Methodology follows Hoefler & Belli SC'15 (CV-based stopping, warmup
iterations discarded).

## Quality

- **15,000+ tests** across unit, integration, and property-based suites
- **95%+ line coverage** via cargo-llvm-cov
- **Zero clippy warnings** with `-D warnings`
- **Mutation testing** via cargo-mutants
- **Provable contracts** -- 1,725 bindings with AllImplemented policy

## Sovereign AI Stack

realizar is the inference layer of the PAIML Sovereign AI Stack:

| Layer | Crate | Purpose |
|-------|-------|---------|
| Compute | [trueno](https://crates.io/crates/trueno) | SIMD/GPU primitives (AVX2/AVX-512/NEON, wgpu) |
| ML | [aprender](https://crates.io/crates/aprender) | ML algorithms, APR v2 format |
| Training | [entrenar](https://crates.io/crates/entrenar) | Autograd, LoRA/QLoRA, quantization |
| Inference | **realizar** | LLM inference, GPU kernels, model serving |
| Speech | [whisper-apr](https://crates.io/crates/whisper-apr) | Pure Rust Whisper ASR |
| Distribution | [repartir](https://crates.io/crates/repartir) | Distributed compute (CPU/GPU/Remote) |
| Registry | [pacha](https://crates.io/crates/pacha) | Model registry with Ed25519 signatures |
| Orchestration | [batuta](https://crates.io/crates/batuta) | Stack coordination and CLI |

## Documentation

- **API docs**: [docs.rs/realizar](https://docs.rs/realizar)
- **Repository**: [github.com/paiml/realizar](https://github.com/paiml/realizar)
- **Cookbook**: [github.com/paiml/apr-cookbook](https://github.com/paiml/apr-cookbook)

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines, or open an issue to discuss your idea first.

## License

MIT -- [Pragmatic AI Labs](https://paiml.com)
