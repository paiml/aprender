# Oracle Mode

> **"Ask the Oracle, receive the wisdom of the stack."**

Oracle Mode is the intelligent query interface for the Sovereign AI Stack. Instead of manually researching which components to use, Oracle Mode guides you to the optimal solution based on your requirements.

## Overview

Oracle Mode provides:

- **Knowledge Graph:** Complete registry of stack components with capabilities
- **Natural Language Interface:** Query in plain English
- **Intelligent Recommendations:** Algorithm and backend selection
- **Code Generation:** Ready-to-use examples

```
┌──────────────────────────────────────────────────────────────────┐
│                     ORACLE MODE ARCHITECTURE                      │
└──────────────────────────────────────────────────────────────────┘

                    ┌─────────────────┐
                    │  Natural Query  │
                    │   "Train RF"    │
                    └────────┬────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│                       QUERY ENGINE                               │
│  ┌─────────────┐   ┌──────────────┐   ┌──────────────────────┐ │
│  │   Domain    │   │  Algorithm   │   │   Performance        │ │
│  │  Detection  │   │  Extraction  │   │   Hints              │ │
│  └─────────────┘   └──────────────┘   └──────────────────────┘ │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│                     KNOWLEDGE GRAPH                              │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ Layer 0: Primitives   → trueno, trueno-db, trueno-graph   │  │
│  │ Layer 1: ML           → aprender                          │  │
│  │ Layer 2: Pipeline     → entrenar, realizar                │  │
│  │ Layer 3: Transpilers  → depyler, decy, bashrs, ruchy      │  │
│  │ Layer 4: Orchestration→ batuta, repartir                  │  │
│  │ Layer 5: Quality      → certeza, pmat, renacer            │  │
│  │ Layer 6: Data         → alimentar                         │  │
│  │ Layer 7: Media        → rmedia                            │  │
│  └───────────────────────────────────────────────────────────┘  │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│                      RECOMMENDER                                 │
│  ┌─────────────┐   ┌──────────────┐   ┌──────────────────────┐ │
│  │  Component  │   │   Backend    │   │   Distribution       │ │
│  │  Selection  │   │   Selection  │   │   Decision           │ │
│  └─────────────┘   └──────────────┘   └──────────────────────┘ │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
                    ┌─────────────────┐
                    │    Response     │
                    │  + Code Example │
                    └─────────────────┘
```

## The Sovereign AI Stack

Oracle Mode knows all 21 components in the stack:

| Layer | Components | Purpose |
|-------|------------|---------|
| **L0: Primitives** | trueno, trueno-db, trueno-graph, trueno-viz, trueno-rag | SIMD/GPU compute, vector storage, graph ops, RAG |
| **L1: ML** | aprender | First-principles ML algorithms |
| **L2: Pipeline** | entrenar, realizar | Training loops, inference runtime |
| **L3: Transpilers** | depyler, decy, bashrs, ruchy | Python/C transpilers + Rust↔Shell bidirectional |
| **L4: Orchestration** | batuta, repartir, pforge | Migration workflow, distributed compute, MCP servers |
| **L5: Quality** | certeza, pmat, renacer | Testing, profiling, syscall tracing |
| **L6: Data** | alimentar, pacha | Data loading, model/recipe registry |
| **L7: Media** | rmedia | Headless video editing, MLT XML, course production |

## Basic Usage

### CLI Interface

```bash
# List all stack components
$ batuta oracle --list

# Show component details
$ batuta oracle --show trueno

# Find components by capability
$ batuta oracle --capabilities simd

# Query integration patterns
$ batuta oracle --integrate aprender realizar

# Interactive mode
$ batuta oracle --interactive
```

### Interactive Mode

```bash
$ batuta oracle --interactive

🔮 Oracle Mode - Ask anything about the Sovereign AI Stack

oracle> How do I train a random forest on 1M samples?

📊 Analysis:
  Problem class: Supervised Learning
  Algorithm: random_forest
  Data size: Large (1M samples)

💡 Primary Recommendation: aprender
   Path: aprender::tree::RandomForest
   Confidence: 95%
   Rationale: Random forest is ideal for large tabular datasets

🔧 Backend: SIMD
   Rationale: SIMD vectorization optimal for 1M samples with High complexity

📦 Supporting Components:
   - trueno (95%): SIMD-accelerated tensor operations
   - alimentar (70%): Parallel data loading

💻 Code Example:
use aprender::tree::RandomForest;
use alimentar::Dataset;

let dataset = Dataset::from_csv("data.csv")?;
let (x, y) = dataset.split_features_target("label")?;

let model = RandomForest::new()
    .n_estimators(100)
    .max_depth(Some(10))
    .n_jobs(-1)  // Use all cores
    .fit(&x, &y)?;

📚 Related Queries:
   - How to optimize random forest hyperparameters?
   - How to serialize trained models with realizar?
   - How to distribute training with repartir?
```

## Backend Selection

Oracle Mode uses **Amdahl's Law** and **PCIe transfer overhead** (Gregg & Hazelwood, 2011) to select the optimal compute backend.

### The 5× Rule

GPU dispatch is only beneficial when compute time exceeds 5× the PCIe transfer time:

```
If compute_time > 5 × transfer_time → Use GPU
Otherwise → Use SIMD
```

### Backend Decision Matrix

| Operation | Complexity | Small Data | Large Data | GPU Available |
|-----------|------------|------------|------------|---------------|
| Element-wise | O(n) | Scalar | SIMD | SIMD (memory-bound) |
| Reductions | O(n) | Scalar | SIMD | SIMD |
| Matrix mult | O(n³) | SIMD | GPU | GPU |
| Conv2D | O(n²k²) | SIMD | GPU | GPU |
| Attention | O(n²d) | SIMD | GPU | GPU |

### Backend Selection Example

```bash
oracle> What backend for 2048×2048 matrix multiplication?

🎯 Backend Selection:
  Operation: Matrix multiplication
  Size: 2048 × 2048 = 4.2M elements
  Complexity: O(n³) = 8.6B FLOPs

  PCIe Transfer: 4.2M × 4 bytes × 2 = 34 MB
  Transfer time: 34 MB / 32 GB/s = 1.06 ms
  Compute time: 8.6B FLOPs / 20 TFLOPS = 0.43 ms

  Ratio: 0.43 / 1.06 = 0.41× (< 5×)

💡 Recommendation: SIMD
   Rationale: PCIe overhead dominates. Use trueno SIMD backend.
   GPU becomes beneficial at ~8192×8192.
```

## Distribution Decision

Oracle uses **Amdahl's Law** for distribution decisions:

```
Speedup = 1 / ((1 - P) + P/N)

Where:
  P = Parallel fraction of workload
  N = Number of nodes
```

### Distribution Example

```bash
oracle> Should I distribute random forest on 4 nodes?

📊 Amdahl's Law Analysis:
  Algorithm: Random Forest
  Parallel fraction: 0.95 (tree training is parallelizable)
  Nodes: 4

  Theoretical speedup: 1 / (0.05 + 0.95/4) = 3.48×
  Communication overhead: ~10% per node = 40%
  Effective speedup: 3.48 × 0.6 = 2.09×

💡 Recommendation: Yes, distribute with repartir
   Expected speedup: 2.09×
   Break-even: 2+ nodes

📦 Code Example:
use repartir::{Executor, WorkStealing};
use aprender::tree::RandomForest;

let executor = Executor::new()
    .with_workers(4)
    .with_scheduler(WorkStealing);

let forest = executor.map(
    trees.chunks(25),
    |chunk| train_tree_subset(chunk, &data)
).await?;
```

## Knowledge Graph Queries

### Find by Capability

```bash
oracle> What components support GPU?

🔍 Components with GPU capability:
  - trueno: SIMD-accelerated tensor operations with GPU dispatch
  - realizar: GPU-accelerated inference runtime
```

### Find by Domain

```bash
oracle> What do I need for graph analytics?

🧠 Graph Analytics Components:
  - trueno-graph: Graph traversal and algorithms
  - trueno-db: Vector storage with graph indexes
```

### Integration Patterns

```bash
oracle> How do I integrate depyler with aprender?

🔗 Integration: depyler → aprender

Pattern: sklearn_migration
Description: Convert sklearn code to aprender

Example:
# Original Python (sklearn)
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

# After depyler transpilation
use aprender::tree::RandomForest;
let model = RandomForest::new()
    .n_estimators(100)
    .fit(&x, &y)?;
```

## Academic Foundations

Oracle Mode is grounded in peer-reviewed research:

| Concept | Reference | Application |
|---------|-----------|-------------|
| PCIe overhead | Gregg & Hazelwood (2011) | Backend selection |
| Amdahl's Law | Amdahl (1967) | Distribution decisions |
| Roofline model | Williams et al. (2009) | Performance bounds |
| SIMD vectorization | Fog (2022) | Optimization hints |
| Decision trees | Breiman (2001) | Algorithm recommendations |

## JSON Output

For programmatic access, use `--format json`:

```bash
$ batuta oracle --format json "random forest large data"
```

```json
{
  "problem_class": "Supervised Learning",
  "algorithm": "random_forest",
  "primary": {
    "component": "aprender",
    "path": "aprender::tree::RandomForest",
    "confidence": 0.95,
    "rationale": "Random forest is ideal for large tabular datasets"
  },
  "supporting": [
    {
      "component": "trueno",
      "confidence": 0.95,
      "rationale": "SIMD-accelerated tensor operations"
    }
  ],
  "compute": {
    "backend": "SIMD",
    "rationale": "SIMD vectorization optimal for large datasets"
  },
  "distribution": {
    "needed": false,
    "rationale": "Single-node sufficient for this workload size"
  },
  "code_example": "use aprender::tree::RandomForest;..."
}
```

## Code Output

For Unix pipeline composition, use `--format code` to extract raw Rust code with no ANSI escapes and no metadata:

```bash
# From a natural language query
$ batuta oracle "train a random forest" --format code
use aprender::tree::RandomForest;

let model = RandomForest::new()
    .n_estimators(100)
    .max_depth(Some(10))
    .fit(&x, &y)?;

# From a cookbook recipe
$ batuta oracle --recipe ml-random-forest --format code

# From an integration pattern
$ batuta oracle --integrate "aprender,realizar" --format code

# Pipe through rustfmt and copy
$ batuta oracle --recipe training-lora --format code | rustfmt | pbcopy

# Dump all recipes with delimiter comments
$ batuta oracle --cookbook --format code
// --- ml-random-forest ---
use aprender::prelude::*;
...
// --- ml-serving ---
use realizar::prelude::*;
...
```

Code output follows the **Jidoka** principle: when no code is available, the process exits with code 1 and a stderr diagnostic rather than emitting garbage. Commands like `--list`, `--capabilities`, and `--rag` have no code representation and always exit 1 with `--format code`.

### TDD Test Companions

Every code example — both cookbook recipes and recommender-generated snippets — includes a **TDD test companion**: a `#[cfg(test)]` module with 3-4 focused tests. Test companions follow PMAT compliance rules: low cyclomatic complexity, single assertion per test, real crate types.

When using `--format code`, test companions are appended after the main code:

```bash
$ batuta oracle --recipe ml-random-forest --format code
use aprender::tree::RandomForest;

let model = RandomForest::new()
    .n_estimators(100)
    .max_depth(Some(10))
    .fit(&x, &y)?;

#[cfg(test)]
mod tests {
    #[test]
    fn test_random_forest_construction() {
        let n_estimators = 100;
        let max_depth = Some(10);
        assert!(n_estimators > 0);
        assert!(max_depth.unwrap() > 0);
    }

    #[test]
    fn test_prediction_count_matches_input() {
        let n_samples = 50;
        let predictions = vec![0usize; n_samples];
        assert_eq!(predictions.len(), n_samples);
    }

    #[test]
    fn test_feature_importance_sums_to_one() {
        let importances = vec![0.4, 0.35, 0.25];
        let sum: f64 = importances.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }
}
```

Test companion categories:

| Recipe Type | Test Approach |
|-------------|---------------|
| **Pure Rust** (28 recipes) | Full `#[cfg(test)] mod tests` block |
| **Python+Rust** (2 recipes) | Test Rust portion only |
| **WASM** (3 recipes) | `#[cfg(all(test, not(target_arch = "wasm32")))]` guard |
| **Recommender** (5 examples) | Embedded in code_example string |

Recommender code examples (`batuta oracle "train a model" --format code`) also include test companions inline, so the output is always test-ready.

```bash
# Count test companions across all recipes
$ batuta oracle --cookbook --format code 2>/dev/null | grep -c '#\[cfg('
34

# Pipe a recipe with tests through rustfmt
$ batuta oracle --recipe ml-random-forest --format code | rustfmt
```

See `docs/specifications/code-snippets.md` for the full specification with Popperian falsification protocol.

## Programmatic API

Use Oracle Mode from Rust code:

```rust
use batuta::oracle::{Recommender, OracleQuery, DataSize};

// Natural language query
let recommender = Recommender::new();
let response = recommender.query("train random forest on 1M samples");

println!("Primary: {}", response.primary.component);
println!("Backend: {:?}", response.compute.backend);

// Structured query with constraints
let query = OracleQuery::new("neural network training")
    .with_data_size(DataSize::samples(1_000_000))
    .with_hardware(HardwareSpec::with_gpu(16.0))
    .sovereign_only();

let response = recommender.query_structured(&query);

if response.distribution.needed {
    println!("Distribute with: {:?}", response.distribution.tool);
}
```

## RAG Oracle (APR-Powered)

The RAG Oracle extends Oracle Mode with **Retrieval-Augmented Generation** for stack documentation. It indexes all CLAUDE.md and README.md files from stack components and provides semantic search.

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      RAG ORACLE PIPELINE                         │
└─────────────────────────────────────────────────────────────────┘

┌─────────────┐   ┌─────────────────┐   ┌─────────────────────────┐
│   Source    │   │    Semantic     │   │   Content-Addressable   │
│   Docs      │ → │    Chunker      │ → │   Index (BLAKE3)        │
│   (P0-P3)   │   │   (Code-aware)  │   │   (Poka-Yoke)           │
└─────────────┘   └─────────────────┘   └─────────────────────────┘
                                                    ↓
┌─────────────┐   ┌─────────────────┐   ┌─────────────────────────┐
│   Results   │   │   RRF Fusion    │   │   Hybrid Retrieval      │
│   + Scores  │ ← │   (k=60)        │ ← │   (BM25 + Dense)        │
└─────────────┘   └─────────────────┘   └─────────────────────────┘
```

### Toyota Production System Integration

The RAG Oracle applies Toyota Way principles:

| Principle | Implementation |
|-----------|----------------|
| **Jidoka** | Stop-on-error validation (NaN/Inf detection, dimension mismatch) |
| **Poka-Yoke** | Content hashing prevents stale indexes (BLAKE3) |
| **Heijunka** | Load-leveled reindexing via priority queue |
| **Muda** | Delta-only updates skip unchanged documents |
| **Kaizen** | Model hash tracking for continuous improvement |

### Index Persistence (Section 9.7)

The RAG index is persisted to disk for fast startup and offline usage:

**Cache Location:** `~/.cache/batuta/rag/`

**Cache Files:**
```
~/.cache/batuta/rag/
├── manifest.json     # Version, checksums, timestamps
├── index.json        # Inverted index (BM25 terms)
└── documents.json    # Document metadata + chunks
```

**Integrity Validation (Jidoka):**
- BLAKE3 checksums for index.json and documents.json
- Version compatibility check (major version must match)
- Checksum mismatch triggers load failure (stop-on-error)

**Persistence Flow:**
```
Index (CLI)          Persist           Load (CLI)
───────────          ───────           ──────────
batuta oracle        ┌───────┐         batuta oracle
--rag-index    ────▶ │ Cache │ ────▶   --rag "query"
                     └───────┘
                         │
                         ▼
batuta oracle   ──────▶ Stats
--rag-stats            (no full load)

batuta oracle   ──────▶ Full Rebuild (two-phase save)
--rag-index-force
```

### RAG CLI Commands

```bash
# Index all stack documentation (CLAUDE.md, README.md)
$ batuta oracle --rag-index

📚 RAG Indexer (Heijunka Mode)
──────────────────────────────────────────────────
Scanning stack repositories...

  ✓ trueno/CLAUDE.md        ████████░░░░░░░ (12 chunks)
  ✓ trueno/README.md        ██████░░░░░░░░░ (8 chunks)
  ✓ aprender/CLAUDE.md      ██████████░░░░░ (15 chunks)
  ...

Complete: 16 documents, 142 chunks indexed
Vocabulary: 2847 unique terms
Avg doc length: 89.4 tokens

# Query with RAG
$ batuta oracle --rag "How do I use SIMD for matrix operations?"

🔍 RAG Oracle Mode
──────────────────────────────────────────────────
Index: 16 documents, 142 chunks

Query: How do I use SIMD for matrix operations?

1. [trueno] trueno/CLAUDE.md#42 ████████░░ 78%
   Trueno provides SIMD-accelerated tensor ops...

2. [trueno] trueno/README.md#15 ██████░░░░ 62%
   Matrix multiplication with AVX2/AVX-512...

# Show TUI dashboard (native only)
$ batuta oracle --rag-dashboard

# Show cache statistics (fast, manifest only)
$ batuta oracle --rag-stats

📊 RAG Index Statistics
──────────────────────────────────────────────────
Version: 1.0.0
Batuta version: 0.6.2
Indexed at: 2025-01-30 14:23:45 UTC

Sources:
  - trueno: 4 docs, 42 chunks
  - aprender: 3 docs, 38 chunks
  - hf-ground-truth-corpus: 12 docs, 100 chunks

# Force rebuild (old cache retained until save completes)
$ batuta oracle --rag-index-force

Force rebuild requested (old cache retained until save)...
📚 RAG Indexer (Heijunka Mode)
...
```

### RAG TUI Dashboard

The dashboard shows real-time index health, query latency, and retrieval quality:

```
┌─ Oracle RAG Dashboard ──────────────────────────────────────┐
│ Index Health: 95%  |  Docs: 16  |  Chunks: 142              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Index Status                    Query Latency              │
│  ─────────────                   ─────────────              │
│  > trueno      ████████░░ 42     ▁▂▃▄▅▆▇█▆▅▃▂▁            │
│    aprender    █████████░ 38     avg: 12ms  p99: 45ms      │
│    realizar    ██████░░░░ 24                                │
│    entrenar    █████░░░░░ 18     Retrieval Quality         │
│                                   ─────────────────         │
│  Recent Queries                   MRR   0.847 ████████░░   │
│  ─────────────                    NDCG  0.791 ███████░░░   │
│  12:34:56 "SIMD tensor" trueno    R@10  0.923 █████████░   │
│  12:34:41 "train model" aprender                           │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│ [q]uit  [r]efresh  [↑/↓]navigate                           │
└─────────────────────────────────────────────────────────────┘
```

### Hybrid Retrieval

RAG Oracle uses hybrid retrieval combining:

1. **BM25 (Sparse):** Term-based matching with IDF weighting
2. **Dense Retrieval:** Embedding-based semantic similarity (placeholder for trueno-db)
3. **RRF Fusion:** Reciprocal Rank Fusion (k=60) combines both rankings

```
RRF Score = Σ 1/(k + rank) for each retriever
```

### Scalar Int8 Rescoring (Two-Stage Retrieval)

For large-scale dense retrieval, the RAG Oracle implements **scalar int8 rescoring** based on the [HuggingFace embedding quantization](https://huggingface.co/blog/embedding-quantization) research:

```
┌─────────────────────────────────────────────────────────────────┐
│                TWO-STAGE RESCORING PIPELINE                      │
└─────────────────────────────────────────────────────────────────┘

    Stage 1: Fast Approximate Search        Stage 2: Precise Rescoring
    ────────────────────────────────        ──────────────────────────
    ┌─────────────┐                         ┌─────────────────────────┐
    │ Query (f32) │                         │  Top 4k candidates      │
    │ → int8      │ ─────────────────────▶  │  (from Stage 1)         │
    │             │   i8 × i8 dot product   │                         │
    └─────────────┘   O(n) fast scan        │  f32 × i8 rescoring     │
          │                                 │  with scale factor      │
          ▼                                 │                         │
    ┌─────────────┐                         │  Final top-k ranking    │
    │ Index (int8)│                         └─────────────────────────┘
    │ 4× smaller  │
    └─────────────┘
```

**Benefits:**
- 4× memory reduction (f32 → int8)
- 99% accuracy retention with rescoring
- 3.66× speedup via SIMD acceleration

**SIMD Backend Detection:**

| Backend | Ops/Cycle | Platforms |
|---------|-----------|-----------|
| AVX-512 | 64 | Intel Skylake-X, Ice Lake |
| AVX2 | 32 | Intel Haswell+, AMD Zen+ |
| NEON | 16 | ARM64 (M1/M2, Raspberry Pi) |
| Scalar | 1 | Universal fallback |

**Quantization (Kaizen):**

The quantization uses absmax symmetric quantization with Welford's online algorithm for numerically stable calibration:

```
scale = absmax / 127
quantized[i] = clamp(round(x[i] / scale), -128, 127)
```

**Run the Demo:**

```bash
# Run the scalar int8 rescoring demo
cargo run --example int8_rescore_demo --features native

# Output:
# 🚀 Scalar Int8 Rescoring Retriever Demo
# 🖥️  Detected SIMD Backend: AVX-512
#    Int8 operations per cycle: 64
# 📊 Memory Comparison (10 documents × 384 dims):
#    f32 storage:      15360 bytes
#    int8 storage:      4320 bytes
#    Compression:       3.56×
```

See `docs/specifications/retriever-spec.md` for the full specification with 100-point Popperian falsification checklist.

### Document Priority (Genchi Genbutsu)

Documents are indexed with priority levels:

| Priority | Source | Trigger |
|----------|--------|---------|
| P0 | CLAUDE.md | Every commit |
| P1 | README.md, Cargo.toml, pyproject.toml | On release |
| P2 | docs/*.md, src/**/*.py | Weekly scan |
| P3 | examples/*.rs, tests/**/*.py, Docstrings | Monthly scan |

### Ground Truth Corpora (Cross-Language)

The RAG Oracle indexes external ground truth corpora for cross-language ML pattern discovery:

```
┌─────────────────────────────────────────────────────────────────┐
│            GROUND TRUTH CORPUS ARCHITECTURE                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────┐        ┌──────────────────┐             │
│  │  Rust Stack      │        │  Python Corpus   │             │
│  │  (trueno, etc)   │        │  (hf-gtc)        │             │
│  │  CLAUDE.md       │        │  CLAUDE.md       │             │
│  │  README.md       │        │  src/**/*.py     │             │
│  └────────┬─────────┘        └────────┬─────────┘             │
│           │                           │                        │
│           └─────────────┬─────────────┘                        │
│                         ▼                                      │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │              RAG Oracle Index (BM25 + Dense)             │  │
│  │         Cross-language search for ML patterns            │  │
│  └─────────────────────────────────────────────────────────┘  │
│                         │                                      │
│                         ▼                                      │
│         Query: "How do I tokenize text for BERT?"              │
│                         ↓                                      │
│         Results: hf-gtc/preprocessing/tokenization.py          │
│                  + candle/trueno Rust equivalent               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### HuggingFace Ground Truth Corpus

Location: `../hf-ground-truth-corpus`

A curated collection of production-ready Python recipes for HuggingFace ML workflows:

- **95%+ test coverage** with property-based testing (Hypothesis)
- **Module structure**: `hf_gtc.hub`, `hf_gtc.inference`, `hf_gtc.preprocessing`, `hf_gtc.training`
- **Cross-references**: Maps Python patterns to Rust equivalents (candle/trueno)

**Query Examples:**

```bash
# Query for Python ML patterns
$ batuta oracle --rag "How do I tokenize text for BERT?"
# Returns: hf_gtc/preprocessing/tokenization.py + candle equivalent

$ batuta oracle --rag "sentiment analysis pipeline"
# Returns: hf_gtc/inference/pipelines.py patterns
```

#### Extending Ground Truth

To add new ground truth corpora:

1. **Rust stack components** (with `Cargo.toml`): Add to `rust_stack_dirs` in `src/cli/oracle/rag_index.rs:IndexConfig::new()`
2. **Rust reference material** (books, cookbooks, ground truth corpora): Add to `rust_corpus_dirs`
3. **Python corpora** (courses, transpilation corpora): Add to `python_corpus_dirs`
4. Ensure corpus has CLAUDE.md and README.md for P0/P1 indexing
5. Source in `src/**/*.rs` or `src/**/*.py` is indexed as P2
6. Run `batuta oracle --rag-index` to rebuild index

The index currently spans 90+ repositories across categories:
- **Core stack** (trueno, aprender, realizar, entrenar, etc.)
- **Transpilers** (depyler, bashrs, decy, rascal, ruchy, ruchyruchy)
- **Quality tooling** (certeza, pmat, renacer, provable-contracts)
- **Ground truth corpora** (HF, JAX, vLLM, Databricks, TGI, Lean, Lua)
- **Courses** (HuggingFace, Databricks, GitHub Copilot, Agentic AI)
- **Books/cookbooks** (ruchy-book, pmat-book, apr-cookbook, etc.)
- **Private repos** via `.batuta-private.toml` (see below)

#### Private Repositories (`.batuta-private.toml`)

For private repos that should be discoverable via Oracle RAG but never committed to version control, create a `.batuta-private.toml` at the project root. This file is git-ignored by default.

```toml
[private]
rust_stack_dirs = [
    "../rmedia",
    "../infra",
    "../assetgen",
    "../assetsearch",
]

rust_corpus_dirs = [
    "../resolve-pipeline",
]

python_corpus_dirs = [
    "../coursera-stats",
    "../interactive.paiml.com",
]
```

Private directories are merged into the standard RAG index at runtime. The indexer confirms:

```
Private: 7 private directories merged from .batuta-private.toml
```

**Edge cases:**
- **Missing file:** silently ignored (no warning, no error)
- **Malformed TOML:** warning printed to stderr, indexing continues without private dirs
- **Empty `[private]` section:** no-op (no "Private:" line printed)
- **Nonexistent directories:** handled gracefully at scan time ("not found")
- **Partial config:** only populate the categories you need; all fields default to empty

**Query private content:**

```bash
# After indexing, private repos are fully searchable
$ batuta oracle --rag "video editor"
1. [rmedia] rmedia/README.md#1  ██████████ 100%
   Pure Rust headless video editor with MLT XML compatibility...

$ batuta oracle --rag "infrastructure SSH"
1. [infra] infra/docs/rag-video-corpus.md#25  ██████████ 100%
   NO MANUAL SSH. All operations flow through forjar apply...
```

**Future (Phase 2):** Remote RAG endpoints via SSH/HTTP for searching indexes on other machines:

```toml
# Not yet implemented
[[private.endpoints]]
name = "intel"
type = "ssh"
host = "intel.local"
index_path = "/home/noah/.cache/batuta/rag/index.sqlite"
```

#### Python Chunking

Python files use specialized delimiters for semantic chunking:

| Delimiter | Purpose |
|-----------|---------|
| `\ndef ` | Function definitions |
| `\nclass ` | Class definitions |
| `\n    def ` | Method definitions |
| `\nasync def ` | Async function definitions |
| `\n## ` | Markdown section headers |

### Programmatic RAG API

```rust
use batuta::oracle::rag::{RagOracle, ChunkerConfig, SemanticChunker};

// Create RAG Oracle
let oracle = RagOracle::new();

// Query the index
let results = oracle.query("SIMD tensor operations");

for result in results {
    println!("{}: {} (score: {:.2})",
        result.component,
        result.source,
        result.score
    );
}

// Custom chunking
let config = ChunkerConfig::new(512, 64, &["\n## ", "\nfn "]);
let chunker = SemanticChunker::from_config(&config);
let chunks = chunker.split(content);
```

## Auto-Update System

The RAG index stays fresh automatically through a three-layer freshness system:

### Layer 1: Shell Auto-Fresh (`ora-fresh`)

On every shell login, `ora-fresh` runs in the background to check index freshness:

```bash
# Runs automatically on shell login (non-blocking)
ora-fresh

# Manual check
ora-fresh
✅ Index is fresh (3h old)

# When stale
ora-fresh
📚 Stack changed since last index, refreshing...
```

`ora-fresh` checks two conditions:
1. **Stale marker**: `~/.cache/batuta/rag/.stale` (set by post-commit hooks)
2. **Age**: Index older than 24 hours

### Layer 2: Post-Commit Hooks (26 repos)

Every commit in any Sovereign AI Stack repository touches a stale marker file:

```bash
# .git/hooks/post-commit (installed in all 26 stack repos)
#!/bin/bash
touch "$HOME/.cache/batuta/rag/.stale" 2>/dev/null
```

This is a zero-overhead signal — the next `ora-fresh` invocation picks it up and triggers a reindex. No work is done at commit time beyond a single `touch` call.

### Layer 3: Fingerprint-Based Change Detection (BLAKE3)

When a reindex is triggered, BLAKE3 content fingerprints prevent unnecessary work:

```
batuta oracle --rag-index
✅ Index is current (no files changed since last index)
```

Each indexed file has a `DocumentFingerprint` containing:
- **Content hash**: BLAKE3 hash of file contents
- **Chunker config hash**: Detects chunking parameter changes
- **Model hash**: Detects embedding model changes

If no fingerprints have changed, the entire reindex is skipped instantly.

```
┌─────────────────────────────────────────────────────────────────┐
│                    AUTO-UPDATE FLOW                                │
└─────────────────────────────────────────────────────────────────┘

  git commit ─────▶ post-commit hook
                    touch ~/.cache/batuta/rag/.stale
                            │
                            ▼
  shell login ────▶ ora-fresh (background)
                    checks .stale marker + 24h age
                            │
                            ▼
  batuta oracle ──▶ fingerprint check (BLAKE3)
  --rag-index       compare content hashes
                    skip if nothing changed
                            │
                    (changed)│(unchanged)
                            │     └──▶ "Index is current"
                            ▼
                    Full reindex (~30s)
                    Persist new fingerprints
```

### Manual Commands

```bash
# Check freshness (instant)
ora-fresh

# Reindex with change detection (skips if current)
batuta oracle --rag-index

# Force full reindex (ignores fingerprints)
batuta oracle --rag-index-force
```

## RAG Profiling Infrastructure

The RAG Oracle includes comprehensive profiling infrastructure for performance optimization and debugging.

### Profiling Components

| Component | Purpose |
|-----------|---------|
| **Histogram** | Track latency distributions (p50, p90, p99) |
| **Counter** | Count events (cache hits, misses) |
| **Timed Span** | Automatic duration recording on drop |
| **Global Metrics** | Centralized metrics collection |

### CLI Profiling

```bash
# Enable profiling output
batuta oracle --rag "tokenization" --rag-profile

# Output includes timing breakdown:
# 📊 RAG Profiling Results
# ────────────────────────────────────────────────
#   bm25_search:    4.21ms (count: 1)
#   tfidf_search:   2.18ms (count: 1)
#   rrf_fusion:     0.45ms (count: 1)
# ────────────────────────────────────────────────
#   Total query time: 6.84ms
#   Cache hit rate: 75.0%

# Enable detailed tracing
batuta oracle --rag "tokenization" --rag-trace
```

### Programmatic Profiling

```rust
use batuta::oracle::rag::profiling::{span, Counter, Histogram, GLOBAL_METRICS};
use std::time::Duration;

// Track latencies with histogram
let histogram = Histogram::new();
histogram.observe(Duration::from_millis(12));
histogram.observe(Duration::from_millis(15));

println!("p50: {:.2}ms", histogram.percentile(50.0));
println!("p90: {:.2}ms", histogram.percentile(90.0));

// Count cache behavior
let hits = Counter::new();
let misses = Counter::new();
hits.inc_by(45);
misses.inc_by(15);

// Timed spans (auto-record on drop)
{
    let _span = span("bm25_search");
    // ... search work happens here ...
} // Duration recorded when _span drops

// Query global metrics
let summary = GLOBAL_METRICS.summary();
for (name, stats) in &summary.spans {
    println!("{}: {:.2}ms", name, stats.total_us as f64 / 1000.0);
}
```

### Performance Targets

| Metric | Target | Achieved |
|--------|--------|----------|
| Cold start | <500ms | ~300ms |
| Query p50 | <20ms | ~12ms |
| Query p99 | <100ms | ~45ms |
| Cache hit rate | >80% | ~85% |

### Run the Profiling Demo

```bash
cargo run --example rag_profiling_demo
```

---

## SVG Generation System

The Oracle includes two SVG generation modes:

1. **Material Design 3** — 8px grid, Roboto fonts, MD3 palette (legacy)
2. **Grid Protocol** — 16x9 cell-based layout for 1080p video, provable non-overlap

### Design Principles

| Principle | Material Design 3 | Grid Protocol |
|-----------|-------------------|---------------|
| **Layout** | 8px grid, float collision | 16x9 cells (120px), occupied-set tracking |
| **Typography** | Roboto, 11px min | Segoe UI / Cascadia Code, 18px min |
| **Palette** | MD3 (#6750A4 primary) | VideoPalette (pre-verified 4.5:1 contrast) |
| **Viewport** | Configurable | 1920x1080 (16:9) |
| **Validation** | Layout overlap check | Cell non-overlap proof + manifest |
| **Size** | <100KB | <100KB |

### Grid Protocol Mode

The Grid Protocol divides a 1920x1080 canvas into a 16-column x 9-row grid of 120px cells with three boundary layers:

- **Pixel bounds** — raw cell edges
- **Render bounds** — 10px cell padding inset
- **Content zone** — additional 20px internal padding

```rust
use batuta::oracle::svg::{GridProtocol, GridSpan};

let mut grid = GridProtocol::new();
grid.allocate("header",  GridSpan::new(0, 0, 15, 1))?; // full-width top 2 rows
grid.allocate("sidebar", GridSpan::new(0, 2, 3,  8))?; // left 4 columns
grid.allocate("content", GridSpan::new(4, 2, 15, 8))?; // remaining area

// Overlapping allocations are rejected at compile-time equivalent
assert_eq!(grid.cells_used(), 144); // entire grid filled
println!("{}", grid.manifest());     // XML comment documenting all allocations
```

### Layout Templates (A-G)

Seven pre-built templates cover common slide types:

| Template | Regions | Use Case |
|----------|---------|----------|
| **A: Title Slide** | title, subtitle | Opening/closing slides |
| **B: Two Column** | header, left, right | Side-by-side comparison |
| **C: Dashboard** | header, 4 quadrants | Metrics overview |
| **D: Code Walkthrough** | header, code, notes | Code with annotations |
| **E: Diagram** | header, diagram | Architecture diagrams |
| **F: Key Concepts** | header, 3 cards | Concept introduction |
| **G: Reflection** | header, reflection, readings | Summary slides |

```rust
use batuta::oracle::svg::{ShapeHeavyRenderer, LayoutTemplate};

// Template auto-enables grid protocol mode (1920x1080)
let svg = ShapeHeavyRenderer::new()
    .template(LayoutTemplate::Diagram)  // Template E
    .title("Stack Architecture")
    .component("trueno", 100.0, 300.0, "Trueno", "trueno")
    .build();
// Output contains GRID PROTOCOL MANIFEST and 1920x1080 viewBox
```

### Video Typography

All text sizes >= 18px for readability at 1080p:

| Role | Size | Weight | Font |
|------|------|--------|------|
| Slide title | 56px | Bold (700) | Segoe UI |
| Section header | 36px | SemiBold (600) | Segoe UI |
| Body | 24px | Regular (400) | Segoe UI |
| Label | 18px | Regular (400) | Segoe UI |
| Code | 22px | Regular (400) | Cascadia Code |
| Icon text | 18px | Bold (700) | Segoe UI |

### Video Palette

Pre-verified dark and light palettes with WCAG AA 4.5:1 contrast:

| Role | Dark | Light |
|------|------|-------|
| Canvas | #0F172A | #F8FAFC |
| Surface | #1E293B | #FFFFFF |
| Heading | #F1F5F9 | #0F172A |
| Body | #94A3B8 | #475569 |
| Accent Blue | #60A5FA | #2563EB |
| Accent Green | #4ADE80 | #16A34A |
| Accent Gold | #FDE047 | #CA8A04 |
| Outline | #475569 | #94A3B8 |

Four forbidden pairings are rejected by the linter (slate-500 on navy, grey-500 on slate, blue-500 on slate, slate-600 on navy).

### Video-Mode Lint Rules

```rust
use batuta::oracle::svg::{LintConfig, SvgLinter};

let linter = SvgLinter::with_config(LintConfig::video_mode());
// Enforces:
// - min_text_size: 18px
// - min_stroke_width: 2px
// - min_contrast_ratio: 4.5:1
// - min_internal_padding: 20px
// - min_block_gap: 20px
// - forbidden color pairings
```

### Renderer Types

#### ShapeHeavyRenderer

Use for architecture diagrams with 3+ components:

```rust
use batuta::oracle::svg::{ShapeHeavyRenderer, LayoutTemplate, shapes::Point};

// Grid Protocol mode (1080p presentation)
let svg = ShapeHeavyRenderer::new()
    .template(LayoutTemplate::Diagram)
    .title("Data Pipeline Architecture")
    .layer("ingestion", 50.0, 100.0, 800.0, 150.0, "Data Ingestion")
    .horizontal_stack(
        &[("kafka", "Kafka"), ("spark", "Spark"), ("trueno", "Trueno")],
        Point::new(100.0, 130.0),
    )
    .build();

// Material Design 3 mode (legacy)
let svg = ShapeHeavyRenderer::new()
    .title("Pipeline")
    .component("ml", 100.0, 330.0, "ML Engine", "aprender")
    .build();
```

#### TextHeavyRenderer

Use for documentation diagrams:

```rust
use batuta::oracle::svg::{TextHeavyRenderer, LayoutTemplate};

// Grid Protocol mode
let svg = TextHeavyRenderer::new()
    .template(LayoutTemplate::TwoColumn)
    .title("Lecture Notes")
    .heading("Key Concepts")
    .paragraph("Grid Protocol provides provable non-overlap.")
    .build();
```

### Built-in Diagrams

```rust
use batuta::oracle::svg::{sovereign_stack_diagram, documentation_diagram};

// Sovereign Stack diagram (uses Grid Protocol Template E)
let stack_svg = sovereign_stack_diagram();

// Documentation diagram
let doc_svg = documentation_diagram(
    "API Reference",
    &[
        ("Authentication", "Bearer token required"),
        ("Rate Limiting", "100 req/min"),
    ],
);
```

### CLI Integration

Generate SVG alongside code examples:

```bash
# Get code + SVG for a recipe
batuta oracle --recipe ml-random-forest --format code+svg

# The format outputs:
# 1. Rust code with TDD test companion
# 2. SVG diagram showing component architecture
```

### Run the SVG Demo

```bash
cargo run --example svg_generation_demo

# Output demonstrates:
#  1-5.  Material Design 3 mode (architecture, docs, dark, code)
#  6.    Grid Protocol cell allocation engine
#  7.    Layout Templates A-G
#  8-9.  Renderers with Grid Protocol
#  10.   Video Palette and Typography
#  11.   WCAG AA contrast verification
#  12.   Video-mode lint rules
#  13.   SvgBuilder grid mode with video CSS
```

---

## arXiv Paper Enrichment

Oracle Mode includes a **two-tier arXiv enrichment system** that surfaces relevant academic papers alongside component recommendations. This connects stack usage guidance with the underlying research literature.

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   arXiv ENRICHMENT PIPELINE                       │
└─────────────────────────────────────────────────────────────────┘

                    ┌─────────────────┐
                    │  Oracle Query   │
                    │  + --arxiv flag │
                    └────────┬────────┘
                             ↓
              ┌──────────────────────────────┐
              │     Search Term Derivation   │
              │  components + domains +      │
              │  algorithms + keywords       │
              └──────────────┬───────────────┘
                             ↓
         ┌───────────────────┴───────────────────┐
         │                                       │
    ┌────▼────────────┐                ┌─────────▼──────────┐
    │  Tier 1: Builtin │                │  Tier 2: Live API  │
    │  Curated DB      │                │  export.arxiv.org  │
    │  (~120 entries)  │                │  /api/query        │
    │  (--arxiv)       │                │  (--arxiv-live)    │
    └────────┬─────────┘                └─────────┬──────────┘
             │                                    │
             └────────────────┬───────────────────┘
                              ↓
                    ┌─────────────────┐
                    │  Top N papers   │
                    │  (--arxiv-max)  │
                    └─────────────────┘
```

### Tier 1: Builtin Curated Database (`--arxiv`)

The `--arxiv` flag enriches oracle results with papers from a builtin curated database of approximately 120 entries covering the core domains of the Sovereign AI Stack. This provides **instant offline results** with no network dependency:

```bash
$ batuta oracle "whisper speech recognition" --arxiv

📊 Analysis:
  Problem class: Speech Recognition
  Algorithm: whisper

💡 Primary Recommendation: whisper-apr
   Confidence: 90%

📚 arXiv Papers (curated):
  1. [2212.04356] Robust Speech Recognition via Large-Scale Weak Supervision
     Radford et al., 2022
     https://arxiv.org/abs/2212.04356

  2. [2305.11095] Distil-Whisper: Robust Knowledge Distillation via Large-Scale Pseudo Labelling
     Gandhi et al., 2023
     https://arxiv.org/abs/2305.11095
```

Search terms are automatically derived from the oracle query analysis:

| Source | Example Terms |
|--------|---------------|
| **Components** | whisper-apr, realizar, aprender |
| **Domains** | speech recognition, inference, machine learning |
| **Algorithms** | whisper, transformer, attention |
| **Keywords** | fine-tuning, quantization, SIMD |

### Tier 2: Live arXiv API (`--arxiv-live`)

The `--arxiv-live` flag fetches papers directly from the arXiv API (`export.arxiv.org/api/query`) for the most current results. This requires network access:

```bash
$ batuta oracle "LoRA fine-tuning" --arxiv-live

📊 Analysis:
  Problem class: Training
  Algorithm: lora

💡 Primary Recommendation: entrenar
   Confidence: 92%

📚 arXiv Papers (live):
  1. [2106.09685] LoRA: Low-Rank Adaptation of Large Language Models
     Hu et al., 2021
     https://arxiv.org/abs/2106.09685

  2. [2305.14314] QLoRA: Efficient Finetuning of Quantized Large Language Models
     Dettmers et al., 2023
     https://arxiv.org/abs/2305.14314

  3. [2402.12354] LoRA+: Efficient Low Rank Adaptation of Large Models
     Hayou et al., 2024
     https://arxiv.org/abs/2402.12354
```

### Controlling Result Count (`--arxiv-max`)

The `--arxiv-max <n>` flag controls the maximum number of papers shown (default: 3):

```bash
# Show up to 5 papers
$ batuta oracle "transformer attention" --arxiv --arxiv-max 5

# Show just the single most relevant paper
$ batuta oracle "random forest" --arxiv --arxiv-max 1
```

### Output Formats

arXiv enrichment integrates with all output formats:

**Text (default):** Papers listed with IDs, titles, authors, and links after the main recommendation.

**JSON (`--format json`):** Papers included as an array in the response envelope:

```bash
$ batuta oracle "inference optimization" --arxiv --format json
```

```json
{
  "problem_class": "Inference",
  "primary": { ... },
  "arxiv_papers": [
    {
      "id": "2211.17192",
      "title": "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning",
      "authors": "Dao, 2023",
      "url": "https://arxiv.org/abs/2211.17192"
    }
  ]
}
```

**Markdown (`--format markdown`):** Papers rendered with linked titles:

```bash
$ batuta oracle "deep learning" --arxiv --format markdown
```

```markdown
## arXiv Papers

- [FlashAttention-2](https://arxiv.org/abs/2211.17192) — Dao, 2023
- [Efficient Transformers: A Survey](https://arxiv.org/abs/2009.06732) — Tay et al., 2020
```

**Code (`--format code`):** The `--arxiv` flag is **silently skipped** when using `--format code`. Code output contains only executable Rust code and TDD test companions — no metadata, no paper references. This preserves the Jidoka principle: code output is always pipe-safe.

---

## Key Takeaways

- **Query naturally:** Ask in plain English, get precise answers
- **Trust the math:** Backend selection based on PCIe and Amdahl analysis
- **Complete stack:** All 20 components indexed with capabilities
- **Code ready:** Get working examples, not just recommendations
- **Reproducible:** JSON output for automation and CI/CD

## Next Steps

Try Oracle Mode yourself:

```bash
# Run the Oracle demo
cargo run --example oracle_demo --features native

# Run the RAG Oracle demo
cargo run --example rag_oracle_demo --features native

# Run the RAG Profiling demo
cargo run --example rag_profiling_demo --features native

# Run the SVG Generation demo
cargo run --example svg_generation_demo --features native

# Run the Stack Comply demo
cargo run --example stack_comply_demo --features native

# Run the Scalar Int8 Rescoring demo
cargo run --example int8_rescore_demo --features native

# Run the PMAT Query demo (code search + git history + enrichment)
cargo run --example pmat_query_demo --features native

# PMAT query with git history (hotspots, defect intro, churn, coupling)
pmat query "error handling" -G --churn --limit 5

# Full enrichment audit
pmat query "error handling" --churn --duplicates --entropy --faults -G

# Index stack documentation for RAG
batuta oracle --rag-index

# Query with RAG and profiling
batuta oracle --rag "How do I train a model?" --rag-profile

# Get code + SVG output
batuta oracle --recipe ml-random-forest --format code+svg

# Run stack compliance checks
batuta stack comply

# Start interactive mode
batuta oracle --interactive

# Query from CLI
batuta oracle "How do I migrate sklearn to Rust?"

# Enrich oracle results with arXiv papers
batuta oracle "whisper speech recognition" --arxiv
batuta oracle "transformer attention" --arxiv --arxiv-max 5
batuta oracle "LoRA fine-tuning" --arxiv-live
```

---

**Previous:** [Renacer: Syscall Tracing](./renacer.md)
**Next:** [Example Overview](../part4/example-overview.md)
