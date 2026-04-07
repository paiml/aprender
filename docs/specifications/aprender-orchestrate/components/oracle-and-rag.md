# Oracle and RAG Specification

> Parent: [batuta-spec.md](../batuta-spec.md)
> Sources: oracle-mode-spec, apr-powered-rag-oracle, sqlite-rag-integration, improve-oracle, pmat-query-batuta-oracle-integration, code-snippets

---

## 1. Oracle Mode Overview

Oracle Mode provides an intelligent query interface for the Sovereign AI Stack, enabling developers to query capabilities, receive component recommendations, understand integration patterns, and navigate the toolchain.

### Component Taxonomy (Layers)

| Layer | Components | Capabilities |
|-------|-----------|--------------|
| **Compute** | trueno, trueno-db, trueno-graph | SIMD/GPU ops, vector store, graph analytics |
| **ML** | aprender, entrenar | Supervised/unsupervised ML, training (LoRA/QLoRA) |
| **Inference** | realizar | Model serving (.apr/.gguf/.safetensors) |
| **Transpilation** | depyler, decy, bashrs | Python/C/Shell to Rust |
| **Distribution** | repartir | Work-stealing, CPU/GPU/remote executors |
| **Quality** | pmat, certeza, renacer | Static analysis, coverage, syscall tracing |

### Capability Matrix

| Problem Domain | Primary Tool | Backend Selection |
|----------------|-------------|-------------------|
| Linear Algebra | trueno | SIMD (<100K elements), GPU (>100K) |
| ML Training | aprender + entrenar | CPU (small data), GPU (large) |
| ML Inference | realizar | Lambda (serverless), Edge (embedded) |
| Python Migration | depyler | N/A |
| Vector Search | trueno-db | HNSW index |
| Distributed Training | repartir + entrenar | Multi-node |

### Query Examples

| Query | Recommended Stack |
|-------|-------------------|
| "Convert sklearn pipeline to Rust" | depyler -> aprender (sklearn_converter) |
| "Serve model with <10ms latency" | realizar (Lambda) + aprender (.apr format) |
| "Train LLM on 8 GPUs" | repartir (gpu_executor) + entrenar (LoRA) |
| "Detect anomalies in time series" | aprender (IsolationForest) + trueno-db |
| "Ensure GDPR compliance" | batuta (sovereign mode) + local execution |

---

## 2. Recommendation Engine

### Backend Selection Algorithm

Based on operation complexity and the 5x PCIe rule (Gregg & Hazelwood, 2011):

| Op Complexity | GPU Threshold | SIMD Threshold |
|---------------|--------------|----------------|
| Low (element-wise) | >1M elements | >1K elements |
| Medium (reductions) | >100K elements | >100 elements |
| High (matrix ops) | >10K elements | >10 elements |

### Distribution Decision

Uses Amdahl's Law: distribute only when `speedup > 1.5` and `communication_overhead < 20%` of single-node time.

### Integration Pipelines

```
ML Training:  alimentar -> aprender -> entrenar -> realizar -> serve
Python Migr:  source -> batuta analyze -> depyler -> batuta validate
Distributed:  data shards -> repartir -> entrenar -> repartir (aggregate)
```

---

## 3. RAG System (SQLite+FTS5)

### Problem

Oracle RAG index exceeded 540 MB as JSON (445 MB inverted index + 96 MB documents). Every query deserialized the full corpus into memory.

### Solution

SQLite+FTS5 as default storage backend in trueno-rag, replacing JSON. Benefits:
- **10-50 ms** median query latency (vs 3-5s cold load)
- **~30 MB RSS** during query (vs ~540 MB)
- Incremental document updates (vs full rewrite)
- Concurrent access via WAL mode

### Performance Requirements

| Metric | Target | Hard Ceiling |
|--------|--------|-------------|
| Median query latency (p50) | 10-50 ms | 50 ms |
| p99 query latency | < 200 ms | 200 ms |
| Cold open + first query | < 500 ms | 1 s |
| Index update (single doc) | < 100 ms | 200 ms |
| Full reindex (5000 docs) | < 60 s | 120 s |
| Incremental reindex (0 changes) | < 1 s | 5 s |
| Disk footprint | < 60 MB | 100 MB |
| RSS during query | < 30 MB | 50 MB |

### Architecture

```
trueno-rag/src/
  +-- index.rs         # SparseIndex trait, BM25Index (in-memory)
  +-- sqlite/
  |   +-- mod.rs       # SqliteIndex, SqliteStore
  |   +-- schema.rs    # DDL, migrations
  |   +-- fts.rs       # FTS5 search, BM25 ranking
  +-- lib.rs           # Feature-gate sqlite module
```

### Schema (Key Tables)

- **documents** -- id, title, source, content, metadata (JSON), chunk_count
- **chunks** -- id, doc_id (FK), content, position, metadata
- **chunks_fts** -- FTS5 virtual table (`tokenize='porter unicode61'`), external content from chunks
- **fingerprints** -- doc_path, blake3_hash, chunk_count (incremental reindexing)
- **metadata** -- key/value store (replaces manifest.json)

### Connection Pragmas

```sql
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA cache_size = -65536;    -- 64 MB page cache
PRAGMA mmap_size = 268435456;  -- 256 MB mmap I/O
PRAGMA foreign_keys = ON;
PRAGMA busy_timeout = 5000;
```

### Falsification Criteria

| ID | Claim | Falsified If |
|----|-------|-------------|
| F-QUERY | 50 ms median | p50 > 50 ms on 5000-doc corpus, warm cache |
| F-STORAGE | 10-18x reduction | SQLite file > 100 MB (< 5.4x reduction) |
| F-RANKING | FTS5 equivalent to hand-rolled BM25 | NDCG@10 drops > 5% on 50-query test set |
| F-MEMORY | Low RSS | VmRSS > 50 MB during query |
| F-CONCURRENCY | WAL concurrent access | Reader blocks > 100 ms during writer txn |
| F-INCREMENTAL | Fast no-op reindex | > 5 s with zero document changes |

---

## 4. PMAT Query Integration

### Problem

Two disjoint search modalities -- document search (oracle --rag, chunk-level) and code search (pmat query, function-level) -- required separate invocations and mental correlation.

### Solution: Hybrid Retrieval via `--pmat-query`

```bash
# Function-level code search with quality signals
batuta oracle --pmat-query "error handling"

# Combined code + document search
batuta oracle --pmat-query "serialize" --rag

# Cross-project search
batuta oracle --pmat-query "simd kernel" --pmat-all-local

# Quality-filtered results
batuta oracle --pmat-query "cache" --pmat-min-grade A --pmat-max-complexity 10
```

### Data Flow

```
User Query
  |
  +--[--pmat-query]--> pmat query (JSON) --> PmatQueryResult[]
  |                                           (file, function, TDG, complexity)
  |                                           |
  |                                     Quality Summary (grade distribution, mean cx)
  |
  +--[--rag]---------> SQLite FTS5 --> RetrievalResult[]
  |                                    + RAG backlinks for pmat files
  |
  +--[RRF Fusion]---> Merged ranked results
```

### v2.0 Enhancements

1. **RRF-fused ranking** -- Reciprocal Rank Fusion merges code + document results
2. **Cross-project search** -- local workspace discovery via `--pmat-all-local`
3. **Result caching** -- hash-based invalidation
4. **Quality summaries** -- grade distribution, mean complexity
5. **Documentation backlinks** -- automatic links from code results to related docs

---

## 5. Code Snippet Generation (`--format code`)

### Purpose

Emit raw, pipeable Rust code without ANSI decoration:

```bash
batuta oracle "train a model" --format code | rustfmt | pbcopy
batuta oracle --recipe training-lora --format code > example.rs
```

### Code Sources

| Source | Trigger | Origin |
|--------|---------|--------|
| Natural language query | `batuta oracle "..."` | `Recommender::generate_code_example` |
| Named recipe | `--recipe <name>` | `Recipe.code` field in knowledge graph |
| Integration pattern | `oracle integrate A B` | `Recommender::integration_code` |

### Design Rules

- No ANSI escapes in output (prevents pipe corruption)
- Exit code 1 when no code available (Jidoka: never emit garbage)
- All 34 cookbook recipes include TDD test companions (`#[cfg(test)]` modules)
- Normalize all 3 code sources to identical format (Heijunka)

---

## 6. Oracle Improvements (Roadmap)

### Stack Comply

Cross-project consistency enforcement:
- Makefile target consistency across 15+ crates
- Cargo.toml dependency/metadata compliance
- CI workflow parity
- Code duplication detection (MinHash + LSH)

```bash
batuta stack comply
batuta stack comply --rule makefile-targets --fix --dry-run
```

### Falsify

Popperian falsification QA generator -- automatically generate falsifiable test criteria for specification claims.

### RAG Optimization

Indexing performance improvements, delta-only updates, external content FTS5 (eliminates shadow table duplication).

### SVG Generation

Visual architecture diagrams alongside code snippets for Oracle responses.

---

## 7. CLI Reference

```bash
# Natural language query
batuta oracle "How do I train a model?"

# Component listing
batuta oracle --list

# Recipe with code + test companion
batuta oracle --recipe ml-random-forest --format code

# All cookbook recipes
batuta oracle --cookbook --format code

# RAG search (indexed docs)
batuta oracle --rag-index          # Build/update index
batuta oracle --rag "tokenization" # Search

# PMAT function search
batuta oracle --pmat-query "error handling"
batuta oracle --pmat-query "serialize" --pmat-min-grade A
batuta oracle --pmat-query "cache" --pmat-max-complexity 10
batuta oracle --pmat-query "error" --rag  # Combined search
batuta oracle --pmat-query "alloc" --pmat-include-source
```
