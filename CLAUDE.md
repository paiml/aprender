# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Batuta is the orchestration framework for the **Sovereign AI Stack** — a pure-Rust ecosystem for privacy-preserving ML infrastructure. It coordinates stack components (trueno, aprender, pacha, realizar) and provides transpilation pipelines for converting Python/C/Shell to Rust.

### Stack Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      batuta (Orchestration)                 │
├─────────────────────────────────────────────────────────────┤
│  whisper.apr (ASR)  │  realizar (Inference)  │ pacha (Reg)  │
├─────────────────────┴────────────────────────┴──────────────┤
│   aprender (ML)   │  entrenar (Training)  │ jugar (Games)   │
├───────────────────┴───────────────────────┴─────────────────┤
│   simular (Simulation)   │   profesor (Education)           │
├──────────────────────────┴──────────────────────────────────┤
│                 repartir (Distributed Compute)              │
│           CPU (Rayon) │ GPU (wgpu) │ Remote (TCP/TLS)       │
├─────────────────────────────────────────────────────────────┤
│  trueno-zram (Compression)  │  trueno-ublk (Block Device)   │
├─────────────────────────────┴───────────────────────────────┤
│               trueno (SIMD/GPU Compute Primitives)          │
│         AVX2/AVX-512/NEON │ wgpu │ LZ4/ZSTD compression     │
└─────────────────────────────────────────────────────────────┘
```

## Build and Development Commands

```bash
# Build
cargo build                    # Debug build
cargo build --release --locked # Release build

# Testing (uses nextest for parallelism)
make test-fast                 # Fast unit tests (<30s target)
make test                      # Standard tests (<2min target)
make test-full                 # All features enabled
cargo test --lib               # Unit tests only
cargo test --test '*'          # Integration tests only

# Single test
cargo test test_name           # Run specific test
cargo nextest run test_name    # With nextest

# Linting and formatting
make lint                      # Clippy with -D warnings
make fmt                       # Format code
make fmt-check                 # Check formatting

# Coverage (two-phase pattern, temporarily disables mold linker)
make coverage                  # HTML + LCOV reports in target/coverage/

# Quality tiers (Certeza Methodology)
make tier1                     # On-save (<1s): fmt, clippy, check
make tier2                     # Pre-commit (<5s): test --lib, clippy
make tier3                     # Pre-push (1-5min): full tests
make tier4                     # CI/CD: release tests + pmat analysis

# Mutation testing
make mutants-fast              # Quick sample (~5 min)
make mutants                   # Full suite (~30-60 min)
make mutants-file FILE=src/backend.rs  # Specific file

# WASM build
make wasm                      # Debug WASM
make wasm-release              # Optimized WASM

## Code Search (pmat query)

Use `pmat query` instead of grep for code discovery. Returns quality-annotated, ranked results.

**NEVER use grep or rg for code discovery. ALWAYS use pmat query.**

```bash
# Find functions by intent
pmat query "pipeline transpilation" --limit 10

# Find high-quality code
pmat query "oracle recommendation" --min-grade A --exclude-tests

# Find with fault annotations (unwrap, panic, unsafe)
pmat query "backend dispatch" --faults

# Filter by complexity
pmat query "stack dependency" --max-complexity 15

# Cross-project search
pmat query "simd kernel" --include-project ../trueno
pmat query "training loop" --include-project ../entrenar

# Include source code in results
pmat query "bug hunter" --include-source --limit 5

# Git history search (find code by commit intent via RRF fusion)
pmat query "fix clippy warnings" -G
pmat query "pipeline refactor" --git-history

# Enrichment flags (combine freely)
pmat query "oracle dispatch" --churn              # git volatility (commit count, churn score)
pmat query "stack dependency" --duplicates         # code clone detection (MinHash+LSH)
pmat query "recipe handler" --entropy              # pattern diversity (repetitive vs unique)
pmat query "bug hunter" --churn --duplicates --entropy --faults -G  # full audit
```

# Documentation
make book                      # Build mdBook
make book-serve                # Serve at localhost:3000
cargo doc --no-deps --open     # API docs
```

## Architecture

### Core Modules

- **`src/pipeline.rs`**: 5-phase transpilation pipeline (Analysis → Transpilation → Optimization → Validation → Build) with Jidoka stop-on-error validation
- **`src/backend.rs`**: Cost-based GPU/SIMD/Scalar selection using 5× PCIe rule (Gregg & Hazelwood, 2011)
- **`src/oracle/`**: Knowledge graph for stack component recommendations with natural language queries. All code examples (34 cookbook recipes + 5 recommender snippets) include TDD test companions (`#[cfg(test)]` modules). Use `--format code` to get code + test companions.
- **`src/serve/`**: Model serving with failover, circuit breakers, privacy tiers (Sovereign/Private/Standard)
- **`src/agent/`**: Autonomous coding assistant (`batuta code` / `apr code`). 9 tools, multi-turn, session persistence, model discovery (APR-preferred over GGUF). Entry: `agent/code.rs`. Wired into apr-cli via PMAT-182.
- **`src/stack/`**: Dependency graph management, release orchestration, quality gates across stack components

### ML Converters

- **`src/numpy_converter.rs`**: NumPy → Trueno operation mapping
- **`src/sklearn_converter.rs`**: scikit-learn → Aprender algorithm mapping
- **`src/pytorch_converter.rs`**: PyTorch → Realizar operation mapping (inference-only)

### Feature Flags

- `native` (default): Full CLI, filesystem, tracing, TUI dashboard
- `rag` (default): SQLite+FTS5 RAG oracle
- `agents` (default): Autonomous agent runtime (`batuta code`, perceive-reason-act loop)
- `inference` (default): RealizarDriver for local LLM inference (GGUF/APR)
- `agents-inference`: Agent with local inference via RealizarDriver
- `agents-rag`: Agent with trueno-rag document retrieval
- `wasm`: Browser-compatible build (no filesystem, in-memory analysis)
- `trueno-integration`: SIMD/GPU tensor operations
- `oracle-mode`: Knowledge graph with trueno-graph and trueno-db

### External Tool Integration

Batuta orchestrates external transpilers detected via PATH:
- **Depyler**: Python → Rust
- **Bashrs**: Shell → Rust
- **Decy**: C/C++ → Rust
- **PMAT**: Quality analysis and TDG scoring

## Design Principles

Toyota Production System principles applied:
- **Jidoka**: Stop-on-error in pipelines, automatic failover
- **Poka-Yoke**: Privacy tiers prevent data leakage
- **Heijunka**: Load leveling via spillover routing
- **Muda**: Cost circuit breakers prevent waste
- **Kaizen**: Continuous optimization via MoE backend selection

## LAYOUT-002: Row-Major Mandate (Stack-Wide Policy)

**The entire Sovereign AI Stack uses ROW-MAJOR tensor layout. GGUF column-major data is transposed at import.**

This is a critical architectural decision that affects aprender, realizar, and all model conversion pipelines.

### Layout Architecture

```
External Formats                    Stack Internal (Row-Major)
────────────────                    ──────────────────────────
SafeTensors (row-major) ──────────► APR v2 ──► realizar ──► output
                         (native)       ↑
GGUF (column-major) ─────────────────┘
                    (transposed by aprender)
```

### Why Row-Major?

1. **PyTorch/SafeTensors compatibility** - Native HuggingFace format is row-major
2. **Cache efficiency** - Row-major matches C memory layout (contiguous rows)
3. **Kernel simplicity** - realizar's fused Q4K/Q6K kernels expect row-major

### Implementation

| Component | Responsibility |
|-----------|----------------|
| **aprender** | Transposes GGUF→row-major during `apr import` |
| **realizar** | Assumes row-major, uses fused_q4k_parallel_matvec |
| **trueno** | Provides both colmajor/rowmajor kernels (use row-major for APR) |

### Garbage Output = Layout Bug

If you see output like `"olumbia+lsi nunca/localENTS"` instead of coherent text:
- **Root cause**: Column-major data fed to row-major kernel
- **Fix**: Ensure GGUF was converted through aprender's converter
- **Documentation**: See `aprender/CLAUDE.md` LAYOUT-002 section

## Quality Standards

- 95% minimum test coverage (90% enforced, 95% preferred)
- Zero clippy warnings (with `-D warnings`)
- Mutation testing target: >80% mutation score
- TDG Score: maintain A grade (≥85)
- Pre-commit checks must complete in <30s

## Release Policy (HARD GATE — ZERO EXCEPTIONS)

**No `cargo publish` for ANY crate in the Sovereign AI Stack unless ALL of the following pass:**

1. **Clean-room build** — `make clean-room-p1` (or `clean-room-all`) in `../infra/machines/clean-room/` must exit 0. This builds in a Docker container with NO sibling repo mounts — only crates.io deps resolve.
2. **GitHub CI** — ALL GitHub Actions workflows must be green for the target repo AND all downstream dependents.
3. **provable-contracts version pin** — ALL `provable-contracts` and `provable-contracts-macros` deps MUST include `version = "0.2"` alongside any `path = "..."`. Path-only deps break clean-room.

```bash
# CORRECT: version + path (uses path locally, crates.io in clean-room)
provable-contracts-macros = { version = "0.2", path = "../provable-contracts/crates/provable-contracts-macros" }

# WRONG: path-only (breaks clean-room container)
provable-contracts-macros = { path = "../provable-contracts/crates/provable-contracts-macros" }
```

**Release workflow:**
```bash
# 1. Ensure all deps have version pins
grep -rn 'provable-contracts.*path' Cargo.toml crates/*/Cargo.toml | grep -v 'version'
# If any match: ADD version = "0.2" alongside the path

# 2. Run clean-room
cd ../infra/machines/clean-room && make clean-room-p1

# 3. Verify CI green on GitHub
gh run list --repo paiml/<crate> --limit 1

# 4. Only then publish
cargo publish
```

**If clean-room fails:** Apply Five Whys root cause analysis. Fix the root cause. NEVER bypass with `--no-verify` or manual workarounds.

## Sovereign AI Stack Ecosystem

### Checking for Updates

```bash
# Check latest versions of all PAIML stack crates
make stack-versions              # or: batuta stack versions

# JSON output for tooling
make stack-versions-json         # or: batuta stack versions --format json

# Check local vs crates.io
make stack-outdated

# Update dependencies
cargo update trueno aprender realizar pacha renacer
```

### Publish Status (O(1) Cached)

```bash
# Check which crates need publishing - O(1) with cache
make stack-publish-status        # or: batuta stack publish-status

# Force refresh (cold cache)
make stack-publish-status-refresh

# Performance:
# - Cold cache: ~7s (parallel crates.io fetches)
# - Warm cache: <100ms (hash-based invalidation)
```

Cache invalidation triggers:
- Cargo.toml content changed
- Git HEAD moved (new commit)
- crates.io TTL expired (15 min)

### Stack Components (crates.io)

| Layer | Crate | Version | Purpose |
|-------|-------|---------|---------|
| Compute | `trueno` | **0.16.x** | SIMD/GPU primitives (AVX2/AVX-512/NEON, wgpu, LZ4) |
| Compute | `trueno-db` | 0.3.x | GPU-first analytics database, SQL interface |
| Compute | `trueno-graph` | 0.1.x | Graph database for code analysis |
| Compute | `trueno-rag` | 0.2.x | RAG pipeline (chunking, BM25+vector, RRF) |
| Compute | `trueno-viz` | 0.2.x | Terminal/PNG visualization |
| UI | `presentar-terminal` | 0.3.x | Zero-alloc TUI backend (CellBuffer, DiffRenderer) |
| Monitor | `ttop` | **2.0.x** | Sovereign system monitor (14 panels, NVIDIA+AMD GPU) |
| Compression | `trueno-zram-core` | 0.3.x | SIMD compression (LZ4/ZSTD, AVX2/AVX-512/NEON, CUDA) |
| Block Device | `trueno-ublk` | 0.3.x | GPU-accelerated ZRAM replacement via ublk |
| Distribution | `repartir` | 2.0.x | Distributed compute (CPU/GPU/Remote, work-stealing) |
| ML | `aprender` | **0.27.x** | ML algorithms, APR v2 format (LZ4/ZSTD compression) |
| Training | `entrenar` | 0.7.x | Autograd, LoRA/QLoRA, quantization, model merge, CITL |
| Inference | `realizar` | **0.8.x** | APR v2/GGUF/SafeTensors inference, GPU kernels |
| Speech | `whisper-apr` | 0.2.x | Pure Rust Whisper ASR (WASM-first, Int4/Int8 quant) |
| Simulation | `simular` | 0.3.x | Unified simulation (Monte Carlo, physics, optimization) |
| Games | `jugar` | 0.1.x | Game engine (ECS, physics, AI, render, audio, WASM) |
| Education | `profesor` | 0.1.x* | Educational platform (courses, quizzes, labs) |
| Data | `alimentar` | 0.2.x | Zero-copy Parquet/Arrow data loading |
| Registry | `pacha` | 0.2.x | Model registry with Ed25519 signatures |
| Tracing | `renacer` | 0.10.x | Syscall tracer with source correlation |
| Quality | `apr-qa` | 0.1.x | APR model QA playbook (test gen, runner, reports) |
| Quality | `provable-contracts` | 0.2.x | YAML contract → Kani verification for ML kernels |
| Quality | `tiny-model-ground-truth` | 0.1.x | Popperian falsification for model conversion parity |
| Transpilers | `depyler`, `bashrs`, `decy` | - | Python/Shell/C → Rust |
| Orchestration | `batuta` | 0.7.x | Stack coordination and CLI |

*Not yet published to crates.io

### APR v2 Model Format

The `.apr` format is the stack's native model serialization:

| Feature | APR v1 | APR v2 |
|---------|--------|--------|
| Tensor Compression | None | LZ4/ZSTD |
| Index Format | JSON | Binary |
| Zero-Copy Loading | Partial | Full |
| Quantization | Int8 | Int4/Int8 |
| Streaming | No | Yes |

```rust
// APR v2 with compression
use aprender::apr::{AprModel, Compression};

let model = AprModel::load_compressed("model.apr", Compression::Lz4)?;
```

### repartir Feature Flags

| Feature | Purpose |
|---------|---------|
| `cpu` (default) | Local multi-core execution with work-stealing |
| `gpu` | wgpu GPU compute (Vulkan/Metal/DX12/WebGPU) |
| `remote` | TCP-based distributed execution across machines |
| `remote-tls` | TLS-secured remote execution |
| `tensor` | trueno SIMD tensor integration |
| `checkpoint` | trueno-db + Parquet state persistence |
| `tui` | Job flow TUI visualization |
| `full` | All features enabled |

### Stack Quality Metrics (PMAT)

Updated 2026-04-05. Status reflects current clippy/test health across the stack.

| Crate | Version | Tests | Clippy | Status |
|-------|---------|-------|--------|--------|
| `batuta` | 0.7.3 | 6258+ | clean | stable (GPU inference, 13 contracts, 129 FALSIFY) |
| `trueno` | 0.16.3 | - | clean | stable |
| `aprender` | 0.27.5 | - | clean | stable |
| `realizar` | 0.8.3 | 15039+ | clean | stable |
| `ttop` | **2.0.0** | 78 | clean | **stable (sovereign, no ratatui)** |
| `presentar-terminal` | 0.3.5 | 4652+ | clean | stable |
| `entrenar` | 0.7.x | - | - | being fixed |
| `simular` | 0.3.1 | - | clean | fixed (was failing) |
| `jugar` | 0.1.x | 1525 | clean | stable |
| `alimentar` | 0.2.x | 1820 | clean | stable |
| `repartir` | 2.0.x | 84 | clean | stable |
| `pacha` | 0.2.x | - | - | being fixed |
| `renacer` | 0.10.x | - | - | being fixed |
| `profesor` | 0.1.x | - | clean | stable |
| `provable-contracts` | 0.2.x | - | clean | stable |
| `tiny-model-ground-truth` | 0.1.x | - | clean | stable |

### Staying Current

My knowledge has a cutoff date. To get latest stack features:

```bash
# Fetch latest from crates.io (cached 15 min)
batuta stack versions

# Check docs.rs for API changes
# https://docs.rs/trueno, https://docs.rs/aprender, etc.

# RSS feeds for releases
# https://crates.io/api/v1/crates/{crate}/versions.rss
```

## Key Dependencies

- **trueno**: SIMD/GPU compute with LZ4 compression (0.16.x)
- **repartir**: Distributed compute with CPU/GPU/Remote executors (2.0.x)
- **aprender**: ML algorithms with APR v2 format, LZ4/ZSTD compression (0.27.x)
- **realizar**: Inference engine with APR v2, GPU kernels (0.8.x)
- **whisper-apr**: Pure Rust Whisper ASR, WASM-first (0.2.x)
- **trueno-zram-core**: SIMD/GPU memory compression (0.3.x)
- **trueno-ublk**: GPU-accelerated block device via ublk (0.3.x)
- **entrenar**: Training with autograd, LoRA/QLoRA, CITL (0.7.x)
- **simular**: Simulation engine with Jidoka guards, Heijunka scheduling (0.3.x)
- **jugar**: Game engine with ECS, physics, AI, WASM support (0.1.x)
- **profesor**: Educational platform with quizzes, labs (0.1.x, not on crates.io)
- **renacer**: Syscall tracing for semantic validation (0.10.x)
- **pacha**: Model registry integration (0.2.x)
- **alimentar**: Data loading with Parquet/Arrow (0.2.x)
- **provable-contracts**: YAML contract verification for ML kernels (0.2.x)

### Stack Inter-dependencies

```
whisper-apr ► trueno (0.16), aprender (0.27), realizar (0.8)
realizar ───► trueno (0.16), aprender (0.27), alimentar (0.2), pacha (0.2)
aprender ───► trueno (0.16), alimentar (0.2), entrenar (0.7)
entrenar ───► trueno (0.16), aprender (0.27), trueno-db, trueno-rag
trueno-zram-core ► trueno (0.16), CUDA optional
trueno-ublk ► trueno-zram-core, trueno-zram-adaptive, libublk
repartir ───► trueno (0.16), trueno-db (checkpoint), wgpu (gpu)
jugar ──────► trueno (0.16), aprender (0.27)
simular ────► jugar-probar (testing)
profesor ───► (no_std, minimal deps for WASM)
```

### GPU Kernel Capabilities (realizar)

| Kernel | Purpose |
|--------|---------|
| `GemmKernel` | Matrix multiplication (naive, tiled, tensor core) |
| `AttentionKernel` | FlashAttention-style tiled attention |
| `SoftmaxKernel` | Numerically stable with warp shuffle |
| `LayerNormKernel` | Fused layer normalization |
| `QuantizeKernel` | Q4_K dequantization fused with matmul |
| `Q5KKernel` | Q5_K dequantization |
| `Q6KKernel` | Q6_K dequantization |

### trueno-zram Compression

| Algorithm | Throughput | Use Case |
|-----------|------------|----------|
| LZ4 | 3+ GB/s | High-speed, general purpose |
| ZSTD | 13 GB/s (AVX-512) | Better ratio, compressible data |
| Same-Fill | 2048:1 | Zero/repeated pages |

```rust
use trueno_zram_core::{CompressorBuilder, Algorithm};

let compressor = CompressorBuilder::new()
    .algorithm(Algorithm::Lz4)
    .build()?;

let compressed = compressor.compress(&page)?;
```

### Distributed Computing with repartir

```bash
# Run distributed computing example
cargo run --example repartir_distributed --features distributed

# Start remote worker (on each node)
cargo run --bin repartir-worker --features remote -- --bind 0.0.0.0:9000

# TUI job flow monitor
cargo run --bin job-flow --features tui,remote
```

Multi-machine GPU/SIMD pattern:
```rust
use repartir::{Pool, task::{Task, Backend}};
use repartir::executor::remote::RemoteExecutor;

// Connect to GPU workers across machines
let executor = RemoteExecutor::builder()
    .add_worker("node1:9000")  // GPU node 1
    .add_worker("node2:9000")  // GPU node 2
    .build().await?;

let task = Task::builder()
    .binary("./gpu-workload")
    .backend(Backend::Gpu)
    .build()?;

let result = executor.execute(task).await?;
```

## Project-Specific Commands

```bash
# Stack orchestration
batuta stack check              # Dependency health
batuta stack status             # TUI dashboard
batuta stack versions           # Check crates.io versions
batuta stack quality            # Quality matrix
batuta stack gate               # CI quality gate

# Oracle mode (natural language queries)
batuta oracle "How do I train a model?"
batuta oracle --list            # List all components
batuta oracle --recipe ml-random-forest --format code  # Code + TDD test companion
batuta oracle --cookbook --format code   # All recipes with test companions

# Oracle RAG mode (indexed documentation search)
batuta oracle --rag-index       # Index stack docs + ground truth corpora
batuta oracle --rag "tokenization"  # Search indexed docs

# Oracle PMAT query (function-level quality-annotated search)
batuta oracle --pmat-query "error handling"                    # Search functions
batuta oracle --pmat-query "serialize" --pmat-min-grade A      # Grade filter
batuta oracle --pmat-query "cache" --pmat-max-complexity 10    # Complexity filter
batuta oracle --pmat-query "error" --rag                       # Combined function + doc search
batuta oracle --pmat-query "alloc" --pmat-include-source       # Include source code

# Analysis
batuta analyze --languages --tdg .
```

## Ground Truth Corpora

The Oracle RAG mode indexes external ground truth corpora for cross-language knowledge:

### HuggingFace Ground Truth Corpus

Location: `../hf-ground-truth-corpus`

A curated collection of production-ready Python recipes for HuggingFace ML workflows:
- **95%+ test coverage** with property-based testing (Hypothesis)
- **Module structure**: `hf_gtc.hub`, `hf_gtc.inference`, `hf_gtc.preprocessing`, `hf_gtc.training`
- **Cross-references**: Maps Python patterns to Rust equivalents (candle/trueno)

Oracle query examples:
```bash
batuta oracle --rag "How do I tokenize text for BERT?"
# Returns: hf_gtc/preprocessing/tokenization.py + candle equivalent

batuta oracle --rag "sentiment analysis pipeline"
# Returns: hf_gtc/inference/pipelines.py patterns
```

### TGI Ground Truth Corpus

Location: `../tgi-ground-truth-corpus`

Production-ready Rust patterns for LLM inference serving, adapted from HuggingFace TGI:
- **Inference patterns**: continuous batching, KV cache, speculative decoding
- **Quantization**: Q4/Q5/Q6 kernels, calibration strategies
- **Serving**: router, scheduler, streaming SSE, request validation
- **Integration**: Maps TGI patterns to Sovereign AI Stack (realizar)

Oracle query examples:
```bash
batuta oracle --rag "continuous batching implementation"
# Returns: tgi-ground-truth-corpus/src/batching.rs + book/patterns/batching.md

batuta oracle --rag "KV cache optimization"
# Returns: tgi-ground-truth-corpus/src/kv_cache.rs patterns
```

### Databricks Ground Truth Corpus

Location: `../databricks-ground-truth-corpus`

Popperian falsification corpus for Databricks open-source projects:
- **Methodology**: Attempt to break, not verify (129/322 tests passing)
- **Domains**: SDK parity, MegaBlocks MoE, Lilac data quality, Spark extensions, Benchmarks
- **Test signals**: PII detection, dedup, language ID, text statistics, pandas API parity
- **Integration**: Validates Databricks ecosystem patterns

Oracle query examples:
```bash
batuta oracle --rag "PII detection patterns"
# Returns: lilac/scripts/test_pii_detection.py patterns

batuta oracle --rag "MinHash near-duplicate detection"
# Returns: lilac/scripts/test_dedup_detection.py implementation
```

### Ludwig Ground Truth Corpus

Location: `../ludwig-ground-truth-corpus`

Popperian falsification corpus for Ludwig declarative deep learning framework:
- **Methodology**: Attempt to break, not verify (~280 pre-registered tests)
- **Domains**: Config validation, feature encoding, preprocessing, ECD architecture,
  training determinism, serving fidelity, LLM fine-tuning
- **Test signals**: Shape invariants, normalization properties, LoRA math, loss monotonicity
- **Integration**: Validates declarative ML patterns (encoder-combiner-decoder)

Oracle query examples:
```bash
batuta oracle --rag "Ludwig config validation"
# Returns: config-validation/scripts/test_config_validation.py patterns

batuta oracle --rag "LoRA adapter properties"
# Returns: llm-fine-tuning/scripts/test_lora_properties.py implementation
```

### Tiny Model Ground Truth Corpus

Location: `../tiny-model-ground-truth`

Popperian falsification test suite for model conversion parity:
- **Methodology**: Generate HuggingFace oracle outputs, validate against realizar inference
- **Domains**: GGUF/SafeTensors/APR format conversions, quantization drift
- **Test signals**: Output parity, KL divergence, roundtrip fidelity
- **Integration**: Validates realizar and aprender conversion pipelines

Oracle query examples:
```bash
batuta oracle --rag "model conversion parity"
# Returns: tiny-model-ground-truth oracle generation patterns

batuta oracle --rag "quantization drift measurement"
# Returns: tiny-model-ground-truth drift validation tests
```

### Extending Ground Truth

To add new ground truth corpora:
1. **Rust corpora**: Add to `rust_corpus_dirs` in `src/cli/oracle/rag_index.rs:IndexConfig::new()`
2. **Python corpora**: Add to `python_corpus_dirs` in the same function
3. Ensure corpus has CLAUDE.md and README.md for P0/P1 indexing
4. Source in `src/**/*.rs` or `src/**/*.py` is indexed as P2
5. mdBook docs in `book/src/**/*.md` are indexed as P1
6. Run `batuta oracle --rag-index` to rebuild index

### Private Repos (`.batuta-private.toml`)

For private intellectual property that should be discoverable via RAG but never committed to GitHub, create a `.batuta-private.toml` file at the project root (git-ignored):

```toml
[private]
rust_stack_dirs = [
    "../rmedia",
    "../infra",
]

rust_corpus_dirs = [
    "../internal-cookbook",
]

python_corpus_dirs = []
```

Private directories are merged into the standard index at runtime. The CLI shows a confirmation:

```
Private: 2 private directories merged from .batuta-private.toml
```

- Missing file: silently ignored (no warning)
- Malformed TOML: warning printed, indexing continues without private dirs
- Empty `[private]` section: no-op
- Nonexistent directories: handled gracefully at scan time

## Claude Code Integration

This project includes `.claude/commands/` for quick access to common tasks:

- `/stack-versions` - Check latest PAIML crate versions from crates.io
- `/stack-check` - Run dependency health check
- `/quality` - Run full quality gate (fmt, clippy, test, coverage)
- `/update-deps` - Check and apply stack dependency updates

These commands provide pre-configured workflows for maintaining the Sovereign AI Stack.
