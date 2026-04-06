# Basic RAG Pipeline

Complete example of a basic RAG pipeline.

```rust
use trueno_rag::{
    pipeline::RagPipelineBuilder,
    chunk::RecursiveChunker,
    embed::MockEmbedder,
    rerank::LexicalReranker,
    fusion::FusionStrategy,
    Document,
};

fn main() -> trueno_rag::Result<()> {
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
             patterns in data."
        ).with_title("Machine Learning Basics"),

        Document::new(
            "Deep learning is a type of machine learning that uses \
             neural networks with many layers. It has achieved \
             breakthrough results in image recognition, natural \
             language processing, and game playing."
        ).with_title("Deep Learning Overview"),

        Document::new(
            "Natural language processing (NLP) is a field of AI \
             that focuses on the interaction between computers and \
             human language. It enables machines to read, understand, \
             and generate human language."
        ).with_title("NLP Introduction"),
    ];

    // 3. Index documents
    let chunk_count = pipeline.index_documents(&documents)?;
    println!("Indexed {} documents with {} chunks",
             pipeline.document_count(),
             chunk_count);

    // 4. Query the pipeline
    let query = "What is machine learning and how does it relate to AI?";
    let (results, context) = pipeline.query_with_context(query, 5)?;

    // 5. Display results
    println!("\n=== Query: {} ===\n", query);

    println!("Top Results:");
    for (i, result) in results.iter().enumerate() {
        println!("{}. [Score: {:.3}] {}...",
                 i + 1,
                 result.best_score(),
                 &result.chunk.content[..50.min(result.chunk.content.len())]);
    }

    println!("\n=== Assembled Context ===\n");
    println!("{}", context.format_with_citations());

    println!("\n=== Citations ===\n");
    println!("{}", context.citation_list());

    Ok(())
}
```

## Output

```
Indexed 3 documents with 3 chunks

=== Query: What is machine learning and how does it relate to AI? ===

Top Results:
1. [Score: 0.823] Machine learning is a subset of artificial intel...
2. [Score: 0.654] Deep learning is a type of machine learning that...
3. [Score: 0.412] Natural language processing (NLP) is a field of ...

=== Assembled Context ===

Machine learning is a subset of artificial intelligence that enables
computers to learn from data without being explicitly programmed.
It uses algorithms to identify patterns in data. [1]

Deep learning is a type of machine learning that uses neural networks
with many layers. [2]

=== Citations ===

[1] Machine Learning Basics
[2] Deep Learning Overview
```
