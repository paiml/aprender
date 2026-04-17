//! Chunking Strategies Example
//!
//! Run with: cargo run --example chunking_strategies

use trueno_rag::{
    chunk::{
        Chunker, FixedSizeChunker, ParagraphChunker, RecursiveChunker, SentenceChunker,
        StructuralChunker,
    },
    Document,
};

fn main() -> trueno_rag::Result<()> {
    println!("=== Chunking Strategies Comparison ===\n");

    // Sample document
    let content = r"# Introduction to Machine Learning

Machine learning is transforming industries. It enables computers to learn from data.

## Supervised Learning

In supervised learning, models learn from labeled examples. The algorithm learns to map inputs to outputs.

## Unsupervised Learning

Unsupervised learning finds patterns in unlabeled data. Clustering and dimensionality reduction are common techniques.

## Deep Learning

Deep learning uses neural networks with many layers. It excels at complex pattern recognition tasks.";

    let doc = Document::new(content).with_title("ML Guide");

    // Test different chunkers
    let chunkers: Vec<(&str, Box<dyn Chunker>)> = vec![
        ("RecursiveChunker(100, 20)", Box::new(RecursiveChunker::new(100, 20))),
        ("FixedSizeChunker(80, 10)", Box::new(FixedSizeChunker::new(80, 10))),
        ("SentenceChunker(2, 0)", Box::new(SentenceChunker::new(2, 0))),
        ("ParagraphChunker(1)", Box::new(ParagraphChunker::new(1))),
        ("StructuralChunker(true, 500)", Box::new(StructuralChunker::new(true, 500))),
    ];

    for (name, chunker) in chunkers {
        println!("--- {} ---", name);

        let chunks = chunker.chunk(&doc)?;
        println!("Chunks created: {}\n", chunks.len());

        for (i, chunk) in chunks.iter().enumerate() {
            let preview = if chunk.content.len() > 60 {
                format!("{}...", &chunk.content[..60])
            } else {
                chunk.content.clone()
            };
            let preview = preview.replace('\n', "\\n");
            println!("  {}: [{}..{}] {}", i + 1, chunk.start_offset, chunk.end_offset, preview);

            if !chunk.metadata.headers.is_empty() {
                println!("     Headers: {:?}", chunk.metadata.headers);
            }
        }
        println!();
    }

    Ok(())
}
