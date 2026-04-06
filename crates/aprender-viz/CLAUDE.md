# CLAUDE.md

## Contract-First Design

This project follows contract-first development with provable-contracts.
Contracts live in `../provable-contracts/contracts/trueno-viz/`.
Run `pmat comply check` to validate contract compliance.

## Project Overview

trueno-viz — Terminal/PNG visualization for the Sovereign AI Stack

## Code Search (pmat query)

**NEVER use grep or rg for code discovery.** Use `pmat query` instead -- it returns quality-annotated, ranked results with TDG scores and fault annotations.

```bash
# Find functions by intent
pmat query "chart rendering" --limit 10

# Find high-quality code
pmat query "terminal output" --min-grade A --exclude-tests

# Find with fault annotations (unwrap, panic, unsafe, etc.)
pmat query "plot data" --faults

# Filter by complexity
pmat query "color mapping" --max-complexity 10

# Cross-project search
pmat query "tensor visualization" --include-project ../trueno

# Git history search (find code by commit intent via RRF fusion)
pmat query "fix axis labels" -G
pmat query "fix axis labels" --git-history

# Enrichment flags (combine freely)
pmat query "renderer" --churn              # git volatility (commit count, churn score)
pmat query "chart builder" --duplicates          # code clone detection (MinHash+LSH)
pmat query "layout" --entropy             # pattern diversity (repetitive vs unique)
pmat query "visualization" --churn --duplicates --entropy --faults -G  # full audit
```

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
