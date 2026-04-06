# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Verificar is a unified combinatorial test generation and synthetic data factory for PAIML transpiler projects (depyler, bashrs, ruchy, decy). It generates verified `(source, target, correctness)` tuples at scale, creating training data for domain-specific code intelligence models.

## Build Commands

```bash
# Build
cargo build --release

# Run tests
cargo test
cargo test --all-features

# Single test
cargo test test_name

# Coverage (95% minimum required)
cargo llvm-cov --fail-under 95

# Linting
cargo fmt --check
cargo clippy -- -D warnings

# Mutation testing (85% minimum)
cargo mutants --min-score 85
```

## Quality Standards

- **95% minimum test coverage** (enforced via pre-commit)
- **85% minimum mutation score**
- **A- minimum TDG grade** (Technical Debt Grade via pmat)
- **Zero SATD comments** (no TODO/FIXME accumulation)
- All public APIs documented
- Run `certeza` before commits: `cd ../certeza && cargo run -- check ../verificar`

## Architecture

### Core Components

```
src/
├── grammar/      # Language grammar definitions (tree-sitter, pest PEGs)
├── generator/    # Combinatorial program generation engine
├── mutator/      # AST mutation operators (AOR, ROR, LOR, BSR, etc.)
├── oracle/       # Verification oracle (sandbox execution, I/O diffing)
├── ml/           # ML model training (bug prediction, embeddings)
└── data/         # Data pipeline (Parquet output)
```

### Key Traits

```rust
/// Implemented by each transpiler (depyler, bashrs, ruchy, decy)
pub trait Transpiler {
    fn source_language(&self) -> Language;
    fn target_language(&self) -> Language;
    fn transpile(&self, source: &str) -> Result<String, TranspileError>;
    fn grammar(&self) -> &Grammar;
}

/// Standardized oracle interface
trait VerificationOracle {
    fn execute_source(&self, code: &str, input: &str) -> ExecutionResult;
    fn execute_target(&self, code: &str, input: &str) -> ExecutionResult;
    fn compare(&self, source: &ExecutionResult, target: &ExecutionResult) -> Verdict;
}
```

### Sampling Strategies

- **Exhaustive**: Enumerate all programs up to depth N
- **CoverageGuided**: Prioritize unexplored AST paths (NAUTILUS-style)
- **Swarm**: Random feature subsets per batch
- **Boundary**: Edge values emphasized (0, -1, MAX_INT, empty collections)

### PAIML Ecosystem Dependencies

- **trueno**: SIMD-accelerated tensor operations (always use latest crates.io version)
- **aprender**: Classical ML (RandomForest, GradientBoosting for bug prediction)
- **entrenar**: LLM fine-tuning with LoRA
- **certeza**: Quality gate enforcement
- **renacer**: Runtime tracing with source map correlation

## Generation Priority

Based on organizational intelligence analysis of 1,296 defect-fix commits:

| Priority | Category | Allocation | Rationale |
|----------|----------|------------|-----------|
| P0 | ASTTransform | 50% | Universal dominant defect (40-62%) |
| P1 | OwnershipBorrow | 20% | Rust-specific (15-20%) |
| P2 | StdlibMapping | 15% | API translation errors |
| P3 | Language-specific | 15% | bashrs security, decy memory, etc. |

## Git Workflow

Work directly on master branch only. Never create feature branches.


## Stack Documentation Search (RAG Oracle)

**IMPORTANT: Proactively use the batuta RAG oracle when:**
- Looking up patterns from other stack components
- Finding cross-language equivalents (Python HuggingFace → Rust)
- Understanding how similar problems are solved elsewhere in the stack

```bash
# Search across the entire Sovereign AI Stack
batuta oracle --rag "your question here"

# Reindex after changes (auto-runs via post-commit hook + ora-fresh)
batuta oracle --rag-index

# Check index freshness (runs automatically on shell login)
ora-fresh
```

The RAG index covers 5000+ documents across the Sovereign AI Stack.
Index auto-updates via post-commit hooks and `ora-fresh` on shell login.
To manually check freshness: `ora-fresh`
To force full reindex: `batuta oracle --rag-index --force`

## Contract-First Development

All new features and bug fixes must follow provable-contract-first methodology:
1. Write or update the contract YAML in `../provable-contracts/contracts/verificar/`
2. Run `pmat comply check` to validate compliance
3. Implement the code to satisfy the contract
4. Run `pmat comply check` again to confirm
