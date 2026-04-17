//! Basic RAG Pipeline Example
//!
//! Run with: cargo run --example basic_rag

use trueno_rag::{
    chunk::RecursiveChunker, embed::MockEmbedder, fusion::FusionStrategy,
    pipeline::RagPipelineBuilder, rerank::LexicalReranker, Document,
};

fn main() -> trueno_rag::Result<()> {
    println!("=== Basic RAG Pipeline Example ===\n");

    // 1. Build the pipeline
    let mut pipeline = RagPipelineBuilder::new()
        .chunker(RecursiveChunker::new(512, 50))
        .embedder(MockEmbedder::new(384))
        .reranker(LexicalReranker::new())
        .fusion(FusionStrategy::RRF { k: 60.0 })
        .max_context_tokens(4096)
        .build()?;

    // 2. Prepare documents
    let documents = vec![
        Document::new(
            "Machine learning is a subset of artificial intelligence \
             that enables computers to learn from data without being \
             explicitly programmed. It uses algorithms to identify \
             patterns in data and make predictions.",
        )
        .with_title("Machine Learning Basics"),
        Document::new(
            "Deep learning is a type of machine learning that uses \
             neural networks with many layers. It has achieved \
             breakthrough results in image recognition, natural \
             language processing, and game playing.",
        )
        .with_title("Deep Learning Overview"),
        Document::new(
            "Natural language processing (NLP) is a field of AI \
             that focuses on the interaction between computers and \
             human language. It enables machines to read, understand, \
             and generate human language.",
        )
        .with_title("NLP Introduction"),
        Document::new(
            "Rust is a systems programming language focused on safety, \
             speed, and concurrency. It achieves memory safety without \
             garbage collection through its ownership system.",
        )
        .with_title("Rust Programming"),
    ];

    // 3. Index documents
    let chunk_count = pipeline.index_documents(&documents)?;
    println!("Indexed {} documents with {} chunks\n", pipeline.document_count(), chunk_count);

    // 4. Query the pipeline
    let queries = vec![
        "What is machine learning and how does it relate to AI?",
        "Tell me about neural networks",
        "What programming language is memory safe?",
    ];

    for query in queries {
        println!("Query: {}\n", query);

        let (results, context) = pipeline.query_with_context(query, 3)?;

        println!("Top Results:");
        for (i, result) in results.iter().enumerate() {
            let preview = &result.chunk.content[..60.min(result.chunk.content.len())];
            println!("  {}. [Score: {:.3}] {}...", i + 1, result.best_score(), preview);
        }

        println!("\nAssembled Context:");
        println!("{}", context.format_with_citations());

        println!("\nCitations:");
        println!("{}", context.citation_list());
        println!("\n{}\n", "=".repeat(60));
    }

    Ok(())
}
