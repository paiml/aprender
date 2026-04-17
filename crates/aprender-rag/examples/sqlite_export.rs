//! SQLite+FTS5 Export Example (PMAT-017)
//!
//! Demonstrates building a SQLite index with BM25 full-text search,
//! suitable for use with batuta oracle's multi-index search.
//!
//! Run with: cargo run --example sqlite_export
//!
//! (The `sqlite` feature is enabled by default.)

use trueno_rag::sqlite::SqliteIndex;

fn main() -> trueno_rag::Result<()> {
    println!("=== SQLite+FTS5 Export Example ===\n");

    // 1. Create an in-memory SQLite index
    let index = SqliteIndex::open_in_memory()?;

    // 2. Insert documents with chunks
    let documents = vec![
        (
            "rust-ownership",
            "Rust Ownership Model",
            vec![
                "Rust uses an ownership system to manage memory. Each value has exactly one owner.",
                "When the owner goes out of scope, the value is dropped. This prevents memory leaks.",
                "Borrowing allows references to values without taking ownership, enforced at compile time.",
            ],
        ),
        (
            "ml-basics",
            "Machine Learning Basics",
            vec![
                "Machine learning enables computers to learn from data without being explicitly programmed.",
                "Supervised learning uses labeled training data to learn a mapping from inputs to outputs.",
                "Neural networks are composed of layers that progressively extract higher-level features.",
            ],
        ),
        (
            "devops-cicd",
            "CI/CD Pipeline Design",
            vec![
                "Continuous integration merges code changes frequently, running automated tests on each merge.",
                "Continuous deployment automatically releases validated changes to production environments.",
                "The PDCA cycle (Plan-Do-Check-Act) drives iterative improvement in DevOps processes.",
            ],
        ),
    ];

    for (doc_id, title, chunk_texts) in &documents {
        let full_content = chunk_texts.join("\n\n");
        let chunks: Vec<(String, String)> = chunk_texts
            .iter()
            .enumerate()
            .map(|(i, text)| (format!("{doc_id}#chunk-{i}"), (*text).to_string()))
            .collect();

        index.insert_document(doc_id, Some(title), Some(doc_id), &full_content, &chunks, None)?;
    }

    // 3. Optimize FTS5 segments for search performance
    index.optimize()?;

    println!("Indexed {} documents, {} chunks\n", index.document_count()?, index.chunk_count()?);

    // 4. BM25 full-text search
    // Note: FTS5 uses implicit AND — all terms must appear in a single chunk.
    // Use focused keyword queries for best results (like real RAG retrieval).
    let queries = vec![
        ("ownership borrowing", "Matches chunk with both terms"),
        ("neural networks", "Matches neural networks chunk"),
        ("PDCA cycle", "Matches DevOps PDCA chunk"),
        ("continuous deployment", "Matches CI/CD deployment chunk"),
        ("memory leaks", "Matches ownership memory management chunk"),
        ("supervised learning", "Matches ML training chunk"),
    ];

    for (query, description) in queries {
        println!("Query: \"{}\"  ({})\n", query, description);

        let results = index.search_fts(query, 3)?;

        if results.is_empty() {
            println!("  No results found\n");
        } else {
            for (i, result) in results.iter().enumerate() {
                let preview = if result.content.len() > 70 {
                    format!("{}...", &result.content[..70])
                } else {
                    result.content.clone()
                };
                println!(
                    "  {}. [BM25: {:.3}] doc={}, chunk={}",
                    i + 1,
                    result.score,
                    result.doc_id,
                    result.chunk_id
                );
                println!("     {}\n", preview);
            }
        }
        println!("{}\n", "-".repeat(60));
    }

    // 5. File-based persistence workflow
    println!("=== Persistence Workflow ===\n");
    println!("  # Create index on disk:");
    println!("  let index = SqliteIndex::open(\"index.sqlite\")?;");
    println!("  index.insert_document(...)?;");
    println!("  index.optimize()?;");
    println!();
    println!("  # CLI equivalent:");
    println!("  trueno-rag index --path /data/corpus --output /data/index --sqlite");
    println!();
    println!("  # Use with batuta oracle:");
    println!("  scp intel:/data/index/index.sqlite ~/.cache/batuta/rag/video-corpus.sqlite");
    println!("  batuta oracle --rag \"your query\"");

    Ok(())
}
