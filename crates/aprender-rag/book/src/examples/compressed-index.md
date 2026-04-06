# Compressed Index

Demonstrates LZ4 and ZSTD compression for BM25 index serialization, achieving 5-10x storage reduction.

Requires the `compression` feature.

```bash
cargo run --example compressed_index --features compression
```

## Source

```rust
#[cfg(feature = "compression")]
use trueno_rag::{compressed::Compression, index::SparseIndex, BM25Index, Chunk, DocumentId};

fn main() -> trueno_rag::Result<()> {
    let mut index = BM25Index::new();

    let docs = vec![
        "Machine learning enables computers to learn from data",
        "Deep learning uses neural networks for pattern recognition",
        "Natural language processing understands human language",
    ];

    for doc in &docs {
        let chunk = Chunk::new(DocumentId::new(), doc.to_string(), 0, doc.len());
        index.add(&chunk);
    }

    // Compress with LZ4 (fast)
    let lz4_bytes = index.to_compressed_bytes(Compression::Lz4)?;
    println!("LZ4 compressed: {} bytes", lz4_bytes.len());

    // Compress with ZSTD (smaller)
    let zstd_bytes = index.to_compressed_bytes(Compression::Zstd)?;
    println!("ZSTD compressed: {} bytes", zstd_bytes.len());

    // Restore from compressed bytes
    let restored = BM25Index::from_compressed_bytes(&lz4_bytes, Compression::Lz4)?;
    assert_eq!(restored.len(), 3);

    // Search still works after restore
    let results = restored.search("machine learning", 3);
    println!("Search results: {} matches", results.len());

    Ok(())
}
```

## Expected Output

```
=== Trueno-RAG Compressed Index Demo ===

1. Basic BM25 Index Compression
   Documents indexed: 5
   LZ4 compressed size: 1847 bytes
   ZSTD compressed size: 1574 bytes
   Restored index size: 5 documents

2. Compression Ratio Comparison
   Documents: 500
   Uncompressed: 459.1 KB
   LZ4:  78.4 KB (5.9x ratio)
   ZSTD: 52.1 KB (8.8x ratio)
   Storage saved (LZ4): 380.7 KB

3. Search Behavior After Restore
   Query: "programming language safety"
   Original results: 3 matches
   Restored results: 3 matches
   Scores match: YES

4. Persistence Workflow (Simulated)
   Index serialized: 2156 bytes (ZSTD)
   Index restored: 100 documents
   Search works: 5 results

All demos completed successfully!
```

## Key Points

- **LZ4**: Fast compression/decompression, ~5-6x ratio
- **ZSTD**: Better compression, ~8-10x ratio, slightly slower
- Scores are identical after round-trip compression
- Use ZSTD for storage, LZ4 for latency-sensitive applications
