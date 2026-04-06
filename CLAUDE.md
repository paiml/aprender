# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Contract-First Design

This project follows contract-first development with provable-contracts.
Contracts live in `../provable-contracts/contracts/pacha/`.
Run `pmat comply check` to validate contract compliance.

## Project Overview

Pacha is a unified registry for machine learning artifacts—models, datasets, and training recipes—with full lineage tracking, semantic versioning, and cryptographic integrity. It is part of the Pragmatic AI Labs Sovereign AI Stack.

## Code Search (pmat query)

**NEVER use grep or rg for code discovery.** Use `pmat query` instead -- it returns quality-annotated, ranked results with TDG scores and fault annotations.

```bash
# Find functions by intent
pmat query "model registry" --limit 10

# Find high-quality code
pmat query "signature verification" --min-grade A --exclude-tests

# Find with fault annotations (unwrap, panic, unsafe, etc.)
pmat query "artifact storage" --faults

# Filter by complexity
pmat query "registry lookup" --max-complexity 10

# Cross-project search
pmat query "model format" --include-project ../aprender

# Git history search (find code by commit intent via RRF fusion)
pmat query "fix signature validation" -G
pmat query "registry" --git-history

# Enrichment flags (combine freely)
pmat query "registry" --churn              # git volatility (commit count, churn score)
pmat query "artifact" --duplicates           # code clone detection (MinHash+LSH)
pmat query "verification" --entropy           # pattern diversity (repetitive vs unique)
pmat query "model signing" --churn --duplicates --entropy --faults -G  # full audit
```

## Build Commands

```bash
cargo build                    # Build library and CLI
cargo test                     # Run all tests (131 tests)
cargo clippy                   # Run linter
cargo fmt                      # Format code
cargo run -- --help            # Show CLI help
cargo run -- stats             # Show registry statistics
```

## Architecture

### Module Structure

```
src/
├── lib.rs              # Library root with prelude
├── main.rs             # CLI binary
├── error.rs            # Error types (PachaError, Result)
├── storage/            # Content-addressed storage
│   ├── content_address.rs  # BLAKE3 hashing
│   └── object_store.rs     # File-based object storage
├── model/              # Model registry types
│   ├── version.rs      # Semantic versioning
│   ├── stage.rs        # Lifecycle stages
│   └── card.rs         # Model cards
├── data/               # Dataset registry types
│   ├── version.rs      # Dataset versioning
│   └── datasheet.rs    # Dataset documentation
├── recipe/             # Training recipe types
│   ├── version.rs      # Recipe versioning
│   └── hyperparams.rs  # Hyperparameter types
├── experiment/         # Experiment tracking
├── lineage/            # Model lineage graph
└── registry/           # Main registry implementation
    ├── mod.rs          # Registry API
    └── database.rs     # SQLite storage
```

### Key Types

- `ContentAddress` - BLAKE3-based content addressing for deduplication
- `ModelVersion`, `DatasetVersion`, `RecipeVersion` - Semantic versioning
- `ModelCard` - Model documentation following Mitchell et al. (2019)
- `Datasheet` - Dataset documentation following Gebru et al. (2021)
- `TrainingRecipe` - Complete training specification for reproducibility
- `ExperimentRun` - Training execution tracking with metrics

### Storage

- **Objects**: Content-addressed files in `~/.pacha/objects/` with BLAKE3 hash prefix sharding
- **Metadata**: SQLite database at `~/.pacha/registry.db`

## CLI Interface

```bash
# Initialize registry
pacha init

# Model operations
pacha model register <name> <artifact> -v <version> [-d <description>]
pacha model list [<name>]
pacha model get <name> -v <version>
pacha model download <name> -v <version> -o <output>
pacha model stage <name> -v <version> -t <development|staging|production|archived>

# Dataset operations
pacha data register <name> <data> -v <version> [-p <purpose>]
pacha data list [<name>]
pacha data get <name> -v <version>
pacha data download <name> -v <version> -o <output>

# Recipe operations
pacha recipe register <recipe.toml>
pacha recipe list [<name>]
pacha recipe get <name> -v <version>
pacha recipe validate <name> -v <version>

# Experiment tracking
pacha run list <recipe> -v <version>
pacha run get <id>
pacha run compare <id1> <id2> ...
pacha run best <recipe> -v <version> -m <metric> [--minimize]

# Statistics
pacha stats
```

## Integration Points

Pacha integrates with the Sovereign AI Stack:
- **alimentar** - Data loading with `.ald` encrypted format
- **aprender** - Model training with `.apr` encrypted format
- **entrenar** - Training pipelines and hyperparameter optimization
- **realizar** - Model serving and inference

## Testing

The project uses EXTREME TDD with:
- Unit tests for all modules
- Property-based testing with `proptest`
- Integration tests for registry operations

Run specific tests:
```bash
cargo test storage::          # Storage module tests
cargo test model::            # Model module tests
cargo test registry::         # Registry tests
cargo test --test <name>      # Specific test file
```

## Design Principles

Following Toyota Way methodology:
- **Muda (Waste Elimination)** - Content-addressed deduplication
- **Jidoka (Built-in Quality)** - Cryptographic integrity verification
- **Kaizen (Continuous Improvement)** - Incremental lineage tracking


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
