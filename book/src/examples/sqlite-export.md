# SQLite+FTS5 Export

Demonstrates building a SQLite index with BM25 full-text search. The `sqlite` feature is enabled by default.

```bash
cargo run --example sqlite_export
```

## What It Shows

1. Creating an in-memory `SqliteIndex`
2. Inserting documents with multiple chunks
3. BM25-ranked full-text search via FTS5
4. The persistence workflow for disk-based indices

## Key Concepts

**FTS5 implicit AND**: Multi-word queries require all terms to appear in the same chunk. Use focused keyword queries for best results.

**Porter stemming**: The FTS5 tokenizer uses Porter stemming, so "learning" matches "learn" and "deploying" matches "deployment".

**BM25 scoring**: Higher scores indicate more relevant results. The score accounts for term frequency and inverse document frequency.

## Code Walkthrough

### Creating and Populating an Index

```rust
use trueno_rag::sqlite::SqliteIndex;

// In-memory (for testing) or file-based (for production)
let index = SqliteIndex::open_in_memory()?;
// let index = SqliteIndex::open("index.sqlite")?;

// Insert a document with its chunks
index.insert_document(
    "doc-id",
    Some("Document Title"),
    Some("source/path.md"),
    "Full document content...",
    &[
        ("doc-id#chunk-0".into(), "First chunk text...".into()),
        ("doc-id#chunk-1".into(), "Second chunk text...".into()),
    ],
    None, // Optional fingerprint for incremental reindexing
)?;

// Optimize after batch inserts
index.optimize()?;
```

### Searching

```rust
let results = index.search_fts("neural networks", 5)?;
for result in &results {
    println!("[BM25: {:.3}] {} -> {}", result.score, result.doc_id, result.content);
}
```

## CLI Equivalent

The `--sqlite` flag on `trueno-rag index` creates a SQLite+FTS5 index alongside the standard JSON index:

```bash
trueno-rag index --path /data/corpus --output /data/index --recursive --dedup --sqlite
```

This produces both `index.json` (for hybrid retrieval) and `index.sqlite` (for BM25 search via batuta oracle).
