# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Contract-First Design

This project follows contract-first development with provable-contracts.
Contracts live in `../provable-contracts/contracts/alimentar/`.
Run `pmat comply check` to validate contract compliance.

## Project Overview

alimentar ("to feed" in Spanish) is a pure Rust data loading, transformation, and distribution library for the paiml sovereign AI stack. It provides HuggingFace-compatible functionality with sovereignty-first design (local storage default, no mandatory cloud dependency).

## Design Principles

1. **Sovereign-first** - Local storage default, no mandatory cloud dependency
2. **Pure Rust** - No Python, no FFI (WASM-compatible)
3. **Zero-copy** - Arrow RecordBatch throughout
4. **Ecosystem aligned** - Arrow 53, Parquet 53 (matches trueno-db, trueno-graph)

## Code Search (pmat query)

**NEVER use grep or rg for code discovery.** Use `pmat query` instead -- it returns quality-annotated, ranked results with TDG scores and fault annotations.

```bash
# Find functions by intent
pmat query "parquet reader" --limit 10

# Find high-quality code
pmat query "arrow conversion" --min-grade A --exclude-tests

# Find with fault annotations (unwrap, panic, unsafe, etc.)
pmat query "data loading" --faults

# Filter by complexity
pmat query "schema validation" --max-complexity 10

# Cross-project search
pmat query "tensor conversion" --include-project ../trueno

# Git history search (find code by commit intent via RRF fusion)
pmat query "fix column reader" -G
pmat query "deserialize" --git-history

# Enrichment flags (combine freely)
pmat query "deserialize" --churn              # git volatility (commit count, churn score)
pmat query "schema" --duplicates           # code clone detection (MinHash+LSH)
pmat query "data pipeline" --entropy           # pattern diversity (repetitive vs unique)
pmat query "parquet loading" --churn --duplicates --entropy --faults -G  # full audit
```

## Build Commands

```bash
# Build
cargo build
cargo build --release

# Test
cargo test
cargo test --all-features

# Lint
cargo fmt --check
cargo clippy -- -D warnings

# Quality gates (when Makefile exists)
make check          # lint + test
make quality-gate   # lint + test + coverage (blocks if <90%)
make mutants        # mutation testing
make coverage       # coverage report
```

## Quality Standards (EXTREME TDD)

| Metric | Target |
|--------|--------|
| Test coverage | ≥85% (HTTP/HF Hub/S3 require network, not testable without mocking) |
| Mutation score | ≥85% |
| Cyclomatic complexity | ≤15 |
| SATD comments | 0 |
| unwrap() calls | 0 (use clippy disallowed-methods) |
| TDG grade | ≥B+ |
| WASM binary | <500KB |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        alimentar                            │
├─────────────────────────────────────────────────────────────┤
│  Importers          │  Core            │  Exporters         │
│  ─────────          │  ────            │  ─────────         │
│  • HuggingFace Hub  │  • Dataset       │  • Local FS        │
│  • Local files      │  • DataLoader    │  • S3-compatible   │
│  • S3-compatible    │  • Transforms    │  • Registry API    │
│  • HTTP/HTTPS       │  • Streaming     │                    │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
   trueno-db             aprender              trueno-viz
   (storage)             (ML/DL)               (WASM/browser)
```

## Core Types

- **Dataset trait** - `len()`, `get()`, `schema()`, `iter()` returning Arrow RecordBatches
- **ArrowDataset** - In-memory or memory-mapped dataset backed by Arrow
- **StreamingDataset** - Lazy/streaming dataset with prefetch
- **DataLoader** - Batching iterator with shuffle, drop_last, num_workers (0 for WASM)
- **Transform trait** - `apply(batch) -> Result<RecordBatch>` for data transformations
- **StorageBackend trait** - Async trait for list/get/put/delete/exists operations

## Feature Flags

```toml
default = ["local", "tokio-runtime"]
local = []                           # Local filesystem
s3 = ["aws-sdk-s3"]                  # S3-compatible backends
http = ["reqwest"]                   # HTTP sources
hf-hub = ["http"]                    # HuggingFace Hub import
tokio-runtime = ["tokio"]            # Async runtime (non-WASM)
wasm = ["wasm-bindgen", "js-sys"]    # Browser/WASM target
```

## WASM Constraints

When targeting WASM:
- No filesystem access → use `MemoryBackend` or `HttpBackend`
- No multi-threading → `num_workers = 0`
- No tokio → use `wasm-bindgen-futures`
- Use `#[cfg(target_arch = "wasm32")]` for WASM-specific code

## Search Ownership

- **alimentar owns**: Registry metadata search (text/tag matching on index)
- **trueno-db owns**: SQL/filter queries, vector/semantic search (delegate to it)

## Configuration Files

| File | Purpose |
|------|---------|
| `.pmat-gates.toml` | Quality gate thresholds |
| `.cargo-mutants.toml` | Mutation testing config |
| `deny.toml` | Dependency policy |
| `renacer.toml` | Deep inspection config |

## CLI Commands (when implemented)

```bash
alimentar import hf squad --output ./data/squad
alimentar convert data.csv data.parquet
alimentar registry list|push|pull
alimentar info|head|schema ./data/train.parquet
```


## Stack Documentation Search

Query this component's documentation and the entire Sovereign AI Stack using batuta's RAG Oracle:

```bash
# Index all stack documentation (run once, persists to ~/.cache/batuta/rag/)
batuta oracle --rag-index

# Search across the entire stack
batuta oracle --rag "your question here"

# Examples
batuta oracle --rag "SIMD matrix multiplication"
batuta oracle --rag "how to train a model"
batuta oracle --rag "tokenization for BERT"

# Check index status
batuta oracle --rag-stats
```

The RAG index includes CLAUDE.md, README.md, and source files from all stack components plus Python ground truth corpora for cross-language pattern matching.

Index auto-updates via post-commit hooks and `ora-fresh` on shell login.
To manually check freshness: `ora-fresh`
To force full reindex: `batuta oracle --rag-index --force`
