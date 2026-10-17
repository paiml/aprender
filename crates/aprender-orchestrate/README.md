<h1 align="center">aprender-orchestrate</h1>

<p align="center">
  <strong>Sovereign AI Orchestration -- agents, inference, analysis, and transpilation in pure Rust</strong>
</p>

<p align="center">
  <a href="https://crates.io/crates/aprender-orchestrate">
    <img src="https://img.shields.io/crates/v/aprender-orchestrate.svg" alt="crates.io">
  </a>
  <a href="https://docs.rs/aprender-orchestrate">
    <img src="https://docs.rs/aprender-orchestrate/badge.svg" alt="docs.rs">
  </a>
  <a href="https://github.com/paiml/aprender/actions/workflows/ci.yml">
    <img src="https://github.com/paiml/aprender/actions/workflows/ci.yml/badge.svg" alt="CI">
  </a>
  <a href="https://github.com/paiml/aprender/tree/main/crates/aprender-orchestrate/book">
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
aprender-orchestrate analyze --tdg .

# Query the stack oracle
aprender-orchestrate oracle "How do I serve a Llama model locally?"

# Serve models (OpenAI-compatible API)
aprender-orchestrate serve ./model.gguf --port 8080

# Autonomous coding agent
aprender-orchestrate code --prompt "Summarize this codebase"
```

## Installation

```bash
# From crates.io
cargo install aprender-orchestrate

# With autonomous agents
cargo install aprender-orchestrate --features agents
```

## Quick Start

### Analyze a project

```bash
aprender-orchestrate analyze --tdg .
```

```
Analysis Results
  Files: 440 total, 98,000 lines
  Languages: Rust (95%), TOML (3%), Markdown (2%)
  TDG Score: 98.4 (Grade: A+)
```

### Hunt for bugs

```bash
aprender-orchestrate bug-hunter analyze .
```

Finds unwraps, panics, unsafe blocks, error swallowing, and 20+ fault
patterns across your codebase.

### Query the stack oracle

```bash
aprender-orchestrate oracle "How do I train a random forest?"
```

Returns component recommendations with working code examples and TDD
test companions.

### Serve a model

```bash
aprender-orchestrate serve ./model.gguf --port 8080
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
aprender-orchestrate code --prompt "Explain the error handling in this project"

# Interactive chat
aprender-orchestrate code

# With explicit model
aprender-orchestrate code --model ./Qwen3-8B-Q4_K_M.gguf --prompt "Add unit tests"
```

Agents are configured via TOML manifests with capability-gated tools
(shell, filesystem, network, RAG, MCP), privacy enforcement, and
circuit-breaker guards.

See the [Agent Runtime Book
Chapter](https://github.com/paiml/aprender/tree/main/crates/aprender-orchestrate/book) for
details.

## Stack Components

```
+-------------------------------------------------------------+
|                    aprender-orchestrate                     |
+-------------------------------------------------------------+
| whisper-apr (ASR) |   aprender-serve   | aprender-registry  |
+-------------------+--------------------+--------------------+
|   aprender (ML)   |   aprender-train   |   jugar (Games)    |
+-------------------+--------------------+--------------------+
| aprender-simulate |   profesor (Edu)   |                    |
+-------------------+--------------------+--------------------+
|          aprender-distribute (Distributed Compute)          |
+-------------------------------------------------------------+
| aprender-zram (Compression)  |  trueno-ublk (Block Device)  |
+------------------------------+------------------------------+
|        aprender-compute (SIMD/GPU Compute Primitives)       |
+-------------------------------------------------------------+
```

| Component | Version | Description |
|-----------|---------|-------------|
| [aprender-compute](https://crates.io/crates/aprender-compute) | 0.71 | SIMD/GPU compute (AVX2/AVX-512/NEON, wgpu, LZ4) |
| [aprender](https://crates.io/crates/aprender) | 0.71 | ML algorithms: regression, trees, clustering, NLP |
| [aprender-train](https://crates.io/crates/aprender-train) | 0.71 | Training: autograd, LoRA/QLoRA, quantization |
| [aprender-serve](https://crates.io/crates/aprender-serve) | 0.71 | LLM inference for GGUF/SafeTensors/APR models |
| [aprender-distribute](https://crates.io/crates/aprender-distribute) | 0.71 | Distributed compute (CPU/GPU/Remote executors) |
| [whisper-apr](https://crates.io/crates/whisper-apr) | 0.2 | Pure Rust Whisper ASR (WASM-first) |
| aprender-viz-ttop (not published) | 0.71 | Sovereign system monitor (14 panels, GPU support) |
| [aprender-present-terminal](https://crates.io/crates/aprender-present-terminal) | 0.71 | Zero-alloc TUI rendering |
| [aprender-registry](https://crates.io/crates/aprender-registry) | 0.71 | Model registry with Ed25519 signatures |
| [aprender-profile](https://crates.io/crates/aprender-profile) | 0.71 | Syscall tracing with semantic validation |
| [pmat](https://crates.io/crates/pmat) | 3.x | Code quality analysis and TDG scoring |

## CLI Reference

```
aprender-orchestrate analyze        Analyze project structure, languages, TDG score
aprender-orchestrate bug-hunter     Proactive bug hunting (fault patterns, mutation targets)
aprender-orchestrate falsify        Popperian falsification checklist
aprender-orchestrate oracle         Natural language queries about the Sovereign AI Stack
aprender-orchestrate serve          ML model serving (OpenAI-compatible API)
aprender-orchestrate code           Autonomous coding agent (local LLM inference)
aprender-orchestrate stack          Stack version management, drift detection
aprender-orchestrate playbook       Deterministic YAML pipeline runner
aprender-orchestrate transpile      Code transpilation (Python/Shell/C -> Rust)
aprender-orchestrate hf             HuggingFace Hub integration
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
git clone https://github.com/paiml/aprender.git
cd aprender/crates/aprender-orchestrate

cargo build --release          # Build
cargo test --lib               # Unit tests
cargo clippy -- -D warnings    # Lint
make book                      # Build documentation
```

## Documentation

- [The Batuta Book](https://github.com/paiml/aprender/tree/main/crates/aprender-orchestrate/book) -- Comprehensive
  guide
- [API Documentation](https://docs.rs/aprender-orchestrate) -- Rust API reference
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

---

Part of the [Aprender monorepo](https://github.com/paiml/aprender) — 70 workspace crates.
