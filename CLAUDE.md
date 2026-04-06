# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**simular** is a unified simulation engine for the Sovereign AI Stack. It provides falsifiable, reproducible simulations across multiple domains: physics, Monte Carlo, machine learning, and optimization. The project is currently in RFC/specification phase with implementation pending.

## Three Foundational Methodologies

All code must adhere to these methodologies:

### 1. Toyota Production System (TPS)
- **Jidoka**: Stop-on-error - halt simulation on NaN, Inf, energy drift, or constraint violations
- **Poka-Yoke**: Mistake-proofing via type-safe units (`uom` crate), compile-time validation
- **Zero SATD**: No TODO, FIXME, HACK comments - create GitHub issues instead

### 2. JPL Power of 10 Rules
- Max nesting depth: 4
- Max function length: 60 lines
- Max cyclomatic complexity: 15
- Min 2 assertions per function
- All loops must have fixed bounds
- No heap allocation after initialization
- `Result<T, E>` everywhere - no `.unwrap()` in production code
- Warnings as errors: `-D warnings`

### 3. Popperian Falsification
- Every simulation hypothesis must be falsifiable
- Implement `FalsifiableHypothesis` trait for testable predictions
- Property-based testing with `proptest` for falsification
- Reference validation against known-good data (JPL ephemeris)

## Quality Gates

PMAT enforces these thresholds via pre-commit hooks:

| Metric | Threshold |
|--------|-----------|
| Min quality grade | B+ |
| Test coverage | 95% |
| Max complexity | 15 |
| Max nesting | 4 |
| Max function lines | 60 |

## Code Search (pmat query)

**NEVER use grep or rg for code discovery.** Use `pmat query` instead -- it returns quality-annotated, ranked results with TDG scores and fault annotations.

```bash
# Find functions by intent
pmat query "monte carlo simulation" --limit 10

# Find high-quality code
pmat query "physics solver" --min-grade A --exclude-tests

# Find with fault annotations (unwrap, panic, unsafe, etc.)
pmat query "random sampling" --faults

# Filter by complexity
pmat query "optimization step" --max-complexity 10

# Cross-project search
pmat query "simulation engine" --include-project ../trueno

# Git history search (find code by commit intent via RRF fusion)
pmat query "fix particle collision" -G
pmat query "physics step" --git-history

# Enrichment flags (combine freely)
pmat query "physics step" --churn              # git volatility (commit count, churn score)
pmat query "solver" --duplicates           # code clone detection (MinHash+LSH)
pmat query "integration" --entropy           # pattern diversity (repetitive vs unique)
pmat query "monte carlo" --churn --duplicates --entropy --faults -G  # full audit
```

## Commands

```bash
# PMAT compliance check
pmat comply check

# Quality analysis
pmat tdg .
pmat analyze .

# Quality gate (CI/CD)
pmat quality-gate --strict

# Build (once Cargo.toml exists)
cargo build
cargo test
cargo clippy -- -D warnings
cargo fmt --check
```

## Architecture

### Sovereign AI Stack Integration

simular integrates with sibling crates:
- **trueno**: SIMD/GPU compute backend for Monte Carlo and physics
- **aprender**: ML algorithms (Gaussian Process surrogates)
- **entrenar**: Training (autograd, gradients for optimization)
- **realizar**: Fast inference
- **alimentar**: Parquet/Arrow data I/O
- **pacha**: Model/checkpoint registry
- **renacer**: Syscall tracing for validation

### Planned Module Structure

```
simular/
├── src/
│   ├── config/      # YAML schema, Serde validation, Poka-Yoke
│   ├── engine/      # Core loop, scheduler, deterministic RNG, Jidoka guards
│   ├── domains/     # Physics, Monte Carlo, optimization, ML engines
│   ├── replay/      # Checkpointing, event journal, time-travel scrubber
│   ├── viz/         # TUI (ratatui), WebGL (wgpu), export (Parquet, MP4)
│   └── falsification/  # Hypothesis testing, oracles, sensitivity analysis
```

### Key Design Patterns

1. **Deterministic RNG**: Partitioned PCG seeds for reproducible parallel execution
2. **Event Scheduler**: Priority heap with sequence numbers for deterministic ordering
3. **Incremental Checkpointing**: Copy-on-write with zstd compression
4. **Symplectic Integrators**: Störmer-Verlet for energy-preserving physics

## Configuration

Simulations are configured via YAML with schema validation. Key sections:
- `reproducibility`: Seed, IEEE strict mode
- `domains`: Physics engine, Monte Carlo, optimization, ML
- `replay`: Checkpoint intervals, journal persistence
- `falsification`: Null hypothesis, criteria, statistical tests

## Current Status

- Specification complete: `docs/specifications/unified-simulation-engine-spec.md`
- PMAT compliance: Configured and enforced
- Implementation: Pending (no `Cargo.toml` yet)

## Roadmap Reference

See `roadmap.yaml` for milestones:
1. **M1 Foundation**: Cargo workspace, CI/CD, reproducibility subsystem
2. **M2 Domain Engines**: Physics, Monte Carlo, optimization, ML
3. **M3 Replay System**: Time-travel debugging
4. **M4 Visualization**: TUI, WebGL, export
5. **M5 Integration**: Full Sovereign AI Stack integration


## Stack Documentation Search

**IMPORTANT: Proactively use the batuta RAG oracle when:**
- Looking up patterns from other stack components (trueno SIMD, aprender ML, realizar inference)
- Finding simulation and physics patterns from ground truth corpora
- Understanding Jidoka guards and Heijunka scheduling across the stack
- Researching Monte Carlo, optimization, and orbit mechanics approaches

```bash
# Index all stack documentation (run once, persists to ~/.cache/batuta/rag/)
batuta oracle --rag-index

# Search across the entire stack
batuta oracle --rag "your question here"

# Simular-specific examples
batuta oracle --rag "Monte Carlo simulation convergence"
batuta oracle --rag "Jidoka guards stop-on-error patterns"
batuta oracle --rag "Heijunka load leveling scheduling"
batuta oracle --rag "orbit mechanics Keplerian propagation"
batuta oracle --rag "TSP optimization GRASP metaheuristic"
```

The RAG index (341+ docs) includes CLAUDE.md, README.md, and source files from all stack components plus Python ground truth corpora for cross-language pattern matching.

Index auto-updates via post-commit hooks and `ora-fresh` on shell login.
To manually check freshness: `ora-fresh`
To force full reindex: `batuta oracle --rag-index --force`
