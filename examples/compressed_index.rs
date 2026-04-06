//! Compressed BM25 Index Demo (GH-2)
//!
//! Run with: `cargo run --example compressed_index --features compression`
//!
//! This example demonstrates LZ4/ZSTD compression for BM25 index serialization,
//! reducing storage footprint by 5-10x for typical RAG indices.

#[cfg(feature = "compression")]
use trueno_rag::{compressed::Compression, index::SparseIndex, BM25Index, Chunk, DocumentId};

#[cfg(feature = "compression")]
fn main() -> trueno_rag::Result<()> {
    println!("=== Trueno-RAG Compressed Index Demo ===\n");

    // Demo basic compression
    demo_basic_compression()?;

    // Demo compression ratio comparison
    demo_compression_comparison()?;

    // Demo search after restore
    demo_search_after_restore()?;

    // Demo file persistence (simulated)
    demo_persistence_workflow()?;

    println!("All demos completed successfully!");
    Ok(())
}

#[cfg(feature = "compression")]
fn demo_basic_compression() -> trueno_rag::Result<()> {
    println!("1. Basic BM25 Index Compression");
    println!("   -----------------------------");

    let mut index = BM25Index::new();

    // Add some documents
    let docs = vec![
        "Machine learning enables computers to learn from data",
        "Deep learning uses neural networks for pattern recognition",
        "Natural language processing understands human language",
        "Computer vision analyzes and interprets images",
        "Reinforcement learning trains agents through rewards",
    ];

    for doc in &docs {
        let chunk = create_chunk(doc);
        index.add(&chunk);
    }

    println!("   Documents indexed: {}", index.len());

    // Compress with LZ4
    let lz4_bytes = index.to_compressed_bytes(Compression::Lz4)?;
    println!("   LZ4 compressed size: {} bytes", lz4_bytes.len());

    // Compress with ZSTD
    let zstd_bytes = index.to_compressed_bytes(Compression::Zstd)?;
    println!("   ZSTD compressed size: {} bytes", zstd_bytes.len());

    // Restore from compressed bytes
    let restored = BM25Index::from_compressed_bytes(&lz4_bytes, Compression::Lz4)?;
    println!("   Restored index size: {} documents", restored.len());

    println!();
    Ok(())
}

#[cfg(feature = "compression")]
fn demo_compression_comparison() -> trueno_rag::Result<()> {
    println!("2. Compression Ratio Comparison");
    println!("   -----------------------------");

    let mut index = BM25Index::new();

    // Build a larger index
    for i in 0..500 {
        let content = format!(
            "Document {} about machine learning artificial intelligence deep neural networks \
             natural language processing computer vision reinforcement learning {}",
            i,
            "data science ".repeat(10)
        );
        index.add(&create_chunk(&content));
    }

    // Get uncompressed size (bincode)
    let uncompressed = bincode::serialize(&index).expect("serialize");
    let lz4_bytes = index.to_compressed_bytes(Compression::Lz4)?;
    let zstd_bytes = index.to_compressed_bytes(Compression::Zstd)?;

    println!("   Documents: {}", index.len());
    println!("   Uncompressed: {:.1} KB", uncompressed.len() as f64 / 1024.0);
    println!(
        "   LZ4:  {:.1} KB ({:.1}x ratio)",
        lz4_bytes.len() as f64 / 1024.0,
        uncompressed.len() as f64 / lz4_bytes.len() as f64
    );
    println!(
        "   ZSTD: {:.1} KB ({:.1}x ratio)",
        zstd_bytes.len() as f64 / 1024.0,
        uncompressed.len() as f64 / zstd_bytes.len() as f64
    );
    println!(
        "   Storage saved (LZ4): {:.1} KB",
        (uncompressed.len() - lz4_bytes.len()) as f64 / 1024.0
    );

    println!();
    Ok(())
}

#[cfg(feature = "compression")]
fn demo_search_after_restore() -> trueno_rag::Result<()> {
    println!("3. Search Behavior After Restore");
    println!("   ------------------------------");

    let mut index = BM25Index::new();

    // Index documents about programming languages
    let docs = vec![
        ("Rust is a systems programming language focused on safety and performance", "rust"),
        ("Python is popular for data science and machine learning applications", "python"),
        ("JavaScript powers interactive web applications in browsers", "javascript"),
        ("Go provides simple concurrency with goroutines and channels", "go"),
        ("TypeScript adds static typing to JavaScript for better tooling", "typescript"),
    ];

    for (content, _lang) in &docs {
        index.add(&create_chunk(content));
    }

    // Search before compression
    let query = "programming language safety";
    let original_results = index.search(query, 3);
    println!("   Query: \"{}\"", query);
    println!("   Original results: {} matches", original_results.len());

    // Compress and restore
    let compressed = index.to_compressed_bytes(Compression::Lz4)?;
    let restored = BM25Index::from_compressed_bytes(&compressed, Compression::Lz4)?;

    // Search after restore
    let restored_results = restored.search(query, 3);
    println!("   Restored results: {} matches", restored_results.len());

    // Verify scores match
    let scores_match = original_results
        .iter()
        .zip(restored_results.iter())
        .all(|((_, s1), (_, s2))| (s1 - s2).abs() < 1e-5);

    println!("   Scores match: {}", if scores_match { "YES" } else { "NO" });

    println!();
    Ok(())
}

#[cfg(feature = "compression")]
fn demo_persistence_workflow() -> trueno_rag::Result<()> {
    println!("4. Persistence Workflow (Simulated)");
    println!("   ---------------------------------");

    let mut index = BM25Index::new();

    // Build index
    for i in 0..100 {
        let content = format!("Document {} with searchable content about AI and ML", i);
        index.add(&create_chunk(&content));
    }

    // Serialize to bytes (would write to file in real usage)
    let compressed = index.to_compressed_bytes(Compression::Zstd)?;
    println!("   Index serialized: {} bytes (ZSTD)", compressed.len());

    // Simulate file write/read
    // std::fs::write("index.rag.zst", &compressed)?;
    // let loaded = std::fs::read("index.rag.zst")?;

    // Deserialize
    let restored = BM25Index::from_compressed_bytes(&compressed, Compression::Zstd)?;
    println!("   Index restored: {} documents", restored.len());

    // Verify functionality
    let results = restored.search("AI ML content", 5);
    println!("   Search works: {} results", results.len());

    println!();
    println!("   Typical workflow:");
    println!("   1. Build index from documents");
    println!("   2. Serialize: index.to_compressed_bytes(Compression::Zstd)?");
    println!("   3. Save to file: std::fs::write(\"index.rag.zst\", &bytes)?");
    println!("   4. Load from file: let bytes = std::fs::read(\"index.rag.zst\")?");
    println!("   5. Deserialize: BM25Index::from_compressed_bytes(&bytes, Compression::Zstd)?");

    println!();
    Ok(())
}

#[cfg(feature = "compression")]
fn create_chunk(content: &str) -> Chunk {
    Chunk::new(DocumentId::new(), content.to_string(), 0, content.len())
}

#[cfg(not(feature = "compression"))]
fn main() {
    eprintln!("This example requires the 'compression' feature.");
    eprintln!("Run with: cargo run --example compressed_index --features compression");
    std::process::exit(1);
}
