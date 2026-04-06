<h1 align="center">batuta</h1>

<p align="center">
  <strong>Sovereign AI Orchestration -- agents, inference, analysis, and transpilation in pure Rust</strong>
</p>

<p align="center">
  <a href="https://crates.io/crates/batuta">
    <img src="https://img.shields.io/crates/v/batuta.svg" alt="crates.io">
  </a>
  <a href="https://docs.rs/batuta">
    <img src="https://docs.rs/batuta/badge.svg" alt="docs.rs">
  </a>
  <a href="https://github.com/paiml/batuta/actions">
    <img src="https://github.com/paiml/batuta/actions/workflows/ci.yml/badge.svg" alt="CI">
  </a>
  <a href="https://paiml.github.io/batuta/">
    <img src="https://img.shields.io/badge/book-online-brightgreen" alt="Book">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT License">
  </a>
</p>

<div align="center">

[Installation](#installation) | [Quick Start](#quick-start) | [Features](#features) | [Stack Components](#stack-components) | [Documentation](#documentation)

</div>

---

## Table of Contents

- [What is Batuta?](#what-is-batuta)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Features](#features)
- [Agent Runtime](#agent-runtime)
- [Stack Components](#stack-components)
- [CLI Reference](#cli-reference)
- [Privacy Tiers](#privacy-tiers)
- [Quality](#quality)
- [Development](#development)
- [Documentation](#documentation)
- [License](#license)

## What is Batuta?

Batuta is the orchestration CLI for the **Sovereign AI Stack** -- a
pure-Rust ecosystem for privacy-preserving ML infrastructure. It
coordinates 15+ crates spanning compute, training, inference, and
serving -- with zero Python dependencies.

```bash
# Analyze any codebase
batuta analyze --tdg .

# Query the stack oracle
batuta oracle "How do I serve a Llama model locally?"

# Serve models (OpenAI-compatible API)
batuta serve ./model.gguf --port 8080

# Autonomous coding agent
batuta code --prompt "Summarize this codebase"
```

## Installation

```bash
# From crates.io
cargo install batuta

# With autonomous agents
cargo install batuta --features agents
```

## Quick Start

### Analyze a project

```bash
batuta analyze --tdg .
```

```
Analysis Results
  Files: 440 total, 98,000 lines
  Languages: Rust (95%), TOML (3%), Markdown (2%)
  TDG Score: 98.4 (Grade: A+)
```

### Hunt for bugs

```bash
batuta bug-hunter analyze .
```

Finds unwraps, panics, unsafe blocks, error swallowing, and 20+ fault
patterns across your codebase.

### Query the stack oracle

```bash
batuta oracle "How do I train a random forest?"
```

Returns component recommendations with working code examples and TDD
test companions.

### Serve a model

```bash
batuta serve ./model.gguf --port 8080
```

Starts an OpenAI-compatible server at
`http://localhost:8080/v1/chat/completions`.

## Features

- **Code Analysis** -- TDG scoring, bug hunting, Popperian
  falsification testing
- **Oracle Queries** -- Natural language queries with RAG-based
  documentation search
- **Model Serving** -- OpenAI-compatible endpoints with privacy tiers
  (Sovereign/Private/Standard)
- **Autonomous Agents** -- Perceive-reason-act loop with 9 tools and
  formal contract invariants
- **Stack Orchestration** -- Version drift detection, publish-status,
  release pipelines for 15+ crates
- **Transpilation** -- Python/Shell/C to Rust conversion via
  depyler/bashrs/decy
- **Playbooks** -- Deterministic YAML pipelines with BLAKE3
  content-addressed caching

## Agent Runtime

The agent runtime provides an autonomous coding assistant with local
LLM inference:

```bash
# Single-prompt mode (GPU-accelerated)
batuta code --prompt "Explain the error handling in this project"

# Interactive chat
batuta code

# With explicit model
batuta code --model ./Qwen3-8B-Q4_K_M.gguf --prompt "Add unit tests"
```

Agents are configured via TOML manifests with capability-gated tools
(shell, filesystem, network, RAG, MCP), privacy enforcement, and
circuit-breaker guards.

See the [Agent Runtime Book
Chapter](https://paiml.github.io/batuta/part3/agent-runtime.html) for
details.

## Stack Components

```
+-------------------------------------------------------------+
|                    batuta (Orchestration)                    |
+-------------------------------------------------------------+
|  whisper-apr (ASR)  |  realizar (Inference)  | pacha (Reg)  |
+---------------------+------------------------+--------------+
|   aprender (ML)   |  entrenar (Training)  | jugar (Games)  |
+-------------------+-----------------------+----------------+
|   simular (Sim)   |   profesor (Edu)      |                |
+-------------------+-----------------------+----------------+
|                 repartir (Distributed Compute)              |
+-------------------------------------------------------------+
|  trueno-zram (Compression)  |  trueno-ublk (Block Device)  |
+-----------------------------+------------------------------+
|               trueno (SIMD/GPU Compute Primitives)          |
+-------------------------------------------------------------+
```

| Component | Version | Description |
|-----------|---------|-------------|
| [trueno](https://crates.io/crates/trueno) | 0.16 | SIMD/GPU compute (AVX2/AVX-512/NEON, wgpu, LZ4) |
| [aprender](https://crates.io/crates/aprender) | 0.27 | ML algorithms: regression, trees, clustering, NLP |
| [entrenar](https://crates.io/crates/entrenar) | 0.7 | Training: autograd, LoRA/QLoRA, quantization |
| [realizar](https://crates.io/crates/realizar) | 0.8 | LLM inference for GGUF/SafeTensors/APR models |
| [repartir](https://crates.io/crates/repartir) | 2.0 | Distributed compute (CPU/GPU/Remote executors) |
| [whisper-apr](https://crates.io/crates/whisper-apr) | 0.2 | Pure Rust Whisper ASR (WASM-first) |
| [ttop](https://crates.io/crates/ttop) | 2.0 | Sovereign system monitor (14 panels, GPU support) |
| [presentar-terminal](https://crates.io/crates/presentar-terminal) | 0.3 | Zero-alloc TUI rendering |
| [pacha](https://crates.io/crates/pacha) | 0.2 | Model registry with Ed25519 signatures |
| [renacer](https://crates.io/crates/renacer) | 0.10 | Syscall tracing with semantic validation |
| [pmat](https://crates.io/crates/pmat) | 3.x | Code quality analysis and TDG scoring |

## CLI Reference

```
batuta analyze        Analyze project structure, languages, TDG score
batuta bug-hunter     Proactive bug hunting (fault patterns, mutation targets)
batuta falsify        Popperian falsification checklist
batuta oracle         Natural language queries about the Sovereign AI Stack
batuta serve          ML model serving (OpenAI-compatible API)
batuta code           Autonomous coding agent (local LLM inference)
batuta stack          Stack version management, drift detection
batuta playbook       Deterministic YAML pipeline runner
batuta transpile      Code transpilation (Python/Shell/C -> Rust)
batuta hf             HuggingFace Hub integration
```

## Privacy Tiers

| Tier | Behavior | Use Case |
|------|----------|----------|
| **Sovereign** | Blocks ALL external API calls | Healthcare, Government |
| **Private** | VPC/dedicated endpoints only | Financial services |
| **Standard** | Public APIs allowed | General deployment |

## Quality

| Metric | Value |
|--------|-------|
| Tests | 6,258 passing |
| Coverage | 95%+ line coverage |
| TDG Score | 94.7 (A) |
| Clippy | Zero warnings |
| Contracts | 13 provable contracts, 129 FALSIFY tests |

### Falsifiable Quality Commitments

| Commitment | Threshold | Verification |
|------------|-----------|--------------|
| Test coverage | >= 95% line coverage | `cargo llvm-cov` (CI enforced) |
| Clippy clean | Zero warnings | `cargo clippy -- -D warnings` |
| Contract enforcement | 13 contracts, 129 tests | `pv lint` + FALSIFY suite |
| TDG grade | A or above | `pmat analyze tdg` |
| Build time | < 2 minutes incremental | `cargo build --timings` |

## Development

```bash
git clone https://github.com/paiml/batuta.git
cd batuta

cargo build --release          # Build
cargo test --lib               # Unit tests
cargo clippy -- -D warnings    # Lint
make book                      # Build documentation
```

## Documentation

- [The Batuta Book](https://paiml.github.io/batuta/) -- Comprehensive
  guide
- [API Documentation](https://docs.rs/batuta) -- Rust API reference
- [Sovereign AI Stack
  Book](https://paiml.github.io/sovereign-ai-stack-book/) -- Full
  stack tutorial
- [Batuta Cookbook](https://github.com/paiml/batuta-cookbook) -- Runnable
  recipes for orchestration, transpilation, and EXTREME TDD

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License -- see [LICENSE](LICENSE) for details.

---

<div align="center">
  <sub>Built with Extreme TDD | Part of the <a href="https://github.com/paiml">PAIML Sovereign AI Stack</a></sub>
</div>
