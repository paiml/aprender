# Quick Start

Get up and running with Trueno-RAG in minutes.

## Add to Cargo.toml

```toml
[dependencies]
trueno-rag = "0.1"
```

## Basic Usage

```rust
use trueno_rag::{
    pipeline::RagPipelineBuilder,
    embed::MockEmbedder,
    rerank::NoOpReranker,
    Document,
};

fn main() -> trueno_rag::Result<()> {
    // Create a pipeline with defaults
    let mut pipeline = RagPipelineBuilder::new()
        .embedder(MockEmbedder::new(384))
        .reranker(NoOpReranker::new())
        .build()?;

    // Index some documents
    let docs = vec![
        Document::new("Machine learning enables computers to learn from data."),
        Document::new("Deep learning uses neural networks with many layers."),
        Document::new("Natural language processing handles text and speech."),
    ];

    pipeline.index_documents(&docs)?;

    // Query the pipeline
    let results = pipeline.query("What is machine learning?", 3)?;

    for result in results {
        println!("Score: {:.3} - {}", result.best_score(), result.chunk.content);
    }

    Ok(())
}
```

## With Context Assembly

```rust
let (results, context) = pipeline.query_with_context("machine learning", 5)?;

// Get formatted context with citations
println!("Context:\n{}", context.format_with_citations());

// Get citation list
println!("\nSources:\n{}", context.citation_list());
```

## Customizing the Pipeline

```rust
use trueno_rag::{
    chunk::RecursiveChunker,
    fusion::FusionStrategy,
    rerank::LexicalReranker,
};

let pipeline = RagPipelineBuilder::new()
    .chunker(RecursiveChunker::new(512, 50))  // 512 chars, 50 overlap
    .embedder(MockEmbedder::new(384))
    .reranker(LexicalReranker::new())
    .fusion(FusionStrategy::RRF { k: 60.0 })
    .max_context_tokens(4096)
    .build()?;
```

## Running Examples

Try the included examples to see Trueno-RAG in action:

```bash
# Run all examples
make examples

# Or run individually
cargo run --example basic_rag
cargo run --example chunking_strategies
cargo run --example hybrid_search
cargo run --example metrics_evaluation
```

## Next Steps

- Learn about [Document Chunking](./concepts/chunking.md)
- Explore [Fusion Strategies](./concepts/fusion.md)
- See [Advanced Examples](./examples/basic.md)
