# verificar

[![Crates.io](https://img.shields.io/crates/v/verificar.svg)](https://crates.io/crates/verificar)
[![Documentation](https://docs.rs/verificar/badge.svg)](https://docs.rs/verificar)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Synthetic Data Factory for Domain-Specific Code Intelligence**

Verificar is a unified combinatorial test generation and synthetic data factory for PAIML transpiler projects (depyler, bashrs, ruchy, decy). It generates verified `(source, target, correctness)` tuples at scale, creating training data for domain-specific code intelligence models.

## Features

- **Multi-Language Support**: Generate test programs in Python, Bash, C, TypeScript, and Ruchy
- **Combinatorial Generation**: Exhaustive enumeration of valid programs up to configurable depth
- **Mutation Testing**: AST-level mutation operators (AOR, ROR, LOR, BSR, etc.)
- **Verification Oracle**: Sandboxed execution with I/O diffing for correctness verification
- **ML Pipeline**: Bug prediction models and embeddings for code intelligence
- **Parquet Output**: Efficient columnar storage for large-scale data processing

## Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                       VERIFICAR CORE                        │
├─────────────────────────────────────────────────────────────┤
│  Grammar    →   Generator   →   Mutator   →   Oracle       │
│  Definitions    Engine         Engine         Verification  │
└─────────────────────────────────────────────────────────────┘
```

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
verificar = "0.3"
```

Or with optional features:

```toml
[dependencies]
verificar = { version = "0.3", features = ["parquet", "ml"] }
```

## Quick Start

### Library Usage

```rust
use verificar::generator::{Generator, SamplingStrategy};
use verificar::Language;

// Create a generator for Python
let generator = Generator::new(Language::Python);

// Generate test cases using coverage-guided sampling
let strategy = SamplingStrategy::CoverageGuided {
    coverage_map: None,
    max_depth: 3,
    seed: 42,
};
let test_cases = generator.generate(strategy, 100);
```

### CLI Usage

```bash
# Generate Python test programs
verificar generate --language python --count 1000 --output corpus.json

# Generate with specific sampling strategy
verificar generate --language bash --strategy swarm --count 500

# Generate depyler-specific patterns
verificar depyler --category file_io --count 100 --output depyler_tests/
```

## Supported Languages

| Language | Grammar | Description |
|----------|---------|-------------|
| Python | `PythonGrammar` | Functions, control flow, type hints (depyler source) |
| Bash | `BashGrammar` | Variables, pipes, conditionals (bashrs source) |
| C | `CGrammar` | Functions, pointers, memory operations (decy source) |
| TypeScript | `TypeScriptGrammar` | Interfaces, generics, type annotations (decy target) |
| Ruchy | `RuchyGrammar` | Custom DSL programs |
| Rust | - | Common target language |

## Sampling Strategies

- **Exhaustive**: Enumerate all programs up to depth N
- **CoverageGuided**: Prioritize unexplored AST paths (NAUTILUS-style)
- **Swarm**: Random feature subsets per batch
- **Boundary**: Edge values emphasized (0, -1, MAX_INT, empty collections)

## Generation Priority

Based on organizational intelligence analysis of 1,296 defect-fix commits:

| Priority | Category | Allocation | Rationale |
|----------|----------|------------|-----------|
| P0 | ASTTransform | 50% | Universal dominant defect (40-62%) |
| P1 | OwnershipBorrow | 20% | Rust-specific (15-20%) |
| P2 | StdlibMapping | 15% | API translation errors |
| P3 | Language-specific | 15% | bashrs security, decy memory, etc. |

## Features

| Feature | Description |
|---------|-------------|
| `parquet` | Enable Parquet data output |
| `ml` | Enable ML pipeline (aprender integration) |
| `tree-sitter` | Use tree-sitter for grammar parsing |
| `pest` | Use pest for PEG grammars |
| `full` | Enable all features |

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome! Please read the [CLAUDE.md](CLAUDE.md) for development guidelines.
