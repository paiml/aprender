# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## CRITICAL: Contract-First Design

**NEVER write code before writing a provable contract.**

All code changes MUST have a corresponding contract (YAML in ../provable-contracts/contracts/<project>/ or .pmat-work/<TICKET>/contract.json) BEFORE implementation. This is enforced by `pmat comply` CB-1400.

- Use `pmat comply check` to verify contract coverage
- Minimum verification level: L1 (recommended L3+)
- See docs/agent-instructions/provable-contract-first-agents.md for the full workflow

## Project Overview

**renacer** is a pure Rust system call tracer with source-aware correlation for Rust binaries. It provides `strace`-like functionality with enhanced debugging capabilities including source code mapping and semantic analysis.

## Code Search (pmat query)

**NEVER use grep or rg for code discovery.** Use `pmat query` instead -- it returns quality-annotated, ranked results with TDG scores and fault annotations.

```bash
# Find functions by intent
pmat query "syscall tracing" --limit 10

# Find high-quality code
pmat query "source correlation" --min-grade A --exclude-tests

# Find with fault annotations (unwrap, panic, unsafe, etc.)
pmat query "trace collection" --faults

# Filter by complexity
pmat query "syscall handler" --max-complexity 10

# Cross-project search
pmat query "binary analysis" --include-project ../trueno

# Git history search (find code by commit intent via RRF fusion)
pmat query "fix ptrace attach" -G
pmat query "trace parser" --git-history

# Enrichment flags (combine freely)
pmat query "trace parser" --churn              # git volatility (commit count, churn score)
pmat query "syscall table" --duplicates           # code clone detection (MinHash+LSH)
pmat query "event handler" --entropy           # pattern diversity (repetitive vs unique)
pmat query "syscall tracing" --churn --duplicates --entropy --faults -G  # full audit
```

## Build and Test Commands

```bash
# Build
cargo build                    # Debug build
cargo build --release          # Release build

# Test
cargo test                     # Run all tests
cargo test --lib               # Unit tests only

# Run
cargo run -- <binary>          # Trace a binary
cargo run -- --help            # Show options

# Lint
cargo clippy -- -D warnings    # Clippy with strict warnings
cargo fmt --check              # Check formatting
```

## Architecture

### Core Components

- **Tracer**: ptrace-based syscall interception
- **Decoder**: System call argument parsing and formatting
- **Correlator**: Source code location mapping via DWARF debug info
- **Reporter**: Output formatting (text, JSON, flamegraph)

### Key Features

- Pure Rust implementation (no C dependencies for core tracing)
- Source-aware correlation for Rust binaries
- JSON output for tooling integration
- Semantic validation of transpiled code

## Integration with Sovereign AI Stack

renacer is used by batuta for semantic validation during transpilation:

```rust
use renacer::{Tracer, TracerConfig};

// Trace original and transpiled binaries
let original_trace = Tracer::new(config).trace("./original")?;
let transpiled_trace = Tracer::new(config).trace("./transpiled")?;

// Compare syscall sequences for semantic equivalence
assert!(original_trace.semantically_equivalent(&transpiled_trace));
```

## Dependencies

- `nix`: Linux ptrace interface
- `libc`: System call numbers and structures
- `clap`: CLI argument parsing
- `serde`/`serde_json`: Trace serialization

## Stack Documentation Search

**IMPORTANT: Proactively use the batuta RAG oracle when:**
- Looking up patterns from other stack components (trueno SIMD, aprender ML, realizar inference)
- Finding how transpilers use renacer for semantic validation (depyler, bashrs, decy)
- Understanding ptrace/syscall patterns across the stack
- Researching anomaly detection and ML-based analysis approaches

```bash
# Index all stack documentation (run once, persists to ~/.cache/batuta/rag/)
batuta oracle --rag-index

# Search across the entire stack
batuta oracle --rag "your question here"

# Renacer-specific examples
batuta oracle --rag "syscall tracing semantic validation"
batuta oracle --rag "DWARF debug info source correlation"
batuta oracle --rag "ptrace process tracing patterns"
batuta oracle --rag "anomaly detection KMeans clustering"
batuta oracle --rag "OpenTelemetry OTLP export integration"
```

The RAG index (341+ docs) includes CLAUDE.md, README.md, and source files from all stack components plus Python ground truth corpora for cross-language pattern matching.

Index auto-updates via post-commit hooks and `ora-fresh` on shell login.
To manually check freshness: `ora-fresh`
To force full reindex: `batuta oracle --rag-index --force`
