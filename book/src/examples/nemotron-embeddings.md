# Nemotron Embeddings

NVIDIA Embed Nemotron 8B embeddings via GGUF format with asymmetric retrieval (different prefixes for queries vs documents).

Requires the `nemotron` feature and a downloaded GGUF model.

```bash
export NEMOTRON_MODEL_PATH=/path/to/NV-Embed-v2-Q4_K.gguf
cargo run --example nemotron_embeddings --features nemotron
```

## Source

```rust
#[cfg(feature = "nemotron")]
use trueno_rag::embed::{cosine_similarity, Embedder, NemotronConfig, NemotronEmbedder};

fn main() -> trueno_rag::Result<()> {
    let model_path = std::env::var("NEMOTRON_MODEL_PATH")
        .unwrap_or_else(|_| "models/NV-Embed-v2-Q4_K.gguf".to_string());

    let config = NemotronConfig::new(&model_path)
        .with_gpu(true)
        .with_batch_size(8)
        .with_normalize(true);

    let embedder = NemotronEmbedder::new(config)?;
    println!("Loaded: {} dimensions", embedder.dimension());

    let documents = vec![
        "Machine learning enables systems to learn from data.",
        "Neural networks are inspired by biological neural networks.",
        "The stock market saw significant gains today.",
    ];

    // Embed documents
    let doc_embeddings: Vec<_> = documents.iter()
        .map(|doc| embedder.embed_document(doc).unwrap())
        .collect();

    // Embed query (asymmetric — uses instruction prefix)
    let query = "What is machine learning?";
    let query_embedding = embedder.embed_query(query)?;

    // Rank by similarity
    let mut scored: Vec<_> = documents.iter()
        .zip(doc_embeddings.iter())
        .map(|(doc, emb)| (cosine_similarity(&query_embedding, emb), doc))
        .collect();
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    for (i, (score, doc)) in scored.iter().enumerate() {
        println!("{}. [{:.4}] {}", i + 1, score, doc);
    }

    Ok(())
}
```

## Key Concepts

- **Asymmetric retrieval**: Queries and documents use different embedding prefixes, so `embed_query()` and `embed_document()` produce different vectors for the same text
- **GGUF format**: Quantized model files (Q4_K, Q5_K, Q8_0) for efficient CPU/GPU inference
- **4096 dimensions**: Nemotron produces larger embeddings than MiniLM (384) for higher quality

## Expected Output

```
Loading Nemotron model from: models/NV-Embed-v2-Q4_K.gguf
Loaded Nemotron embedder: 4096 dimensions

=== Embedding Documents ===

Document: Machine learning enables systems to learn from data...
  Embedding dim: 4096
Document: Neural networks are inspired by biological neural ne...
  Embedding dim: 4096
Document: The stock market saw significant gains today...
  Embedding dim: 4096

=== Query: What is machine learning? ===

Query embedding dim: 4096

=== Similarities ===

1. [0.8234] Machine learning enables systems to learn from data.
2. [0.7156] Neural networks are inspired by biological neural networks.
3. [0.2341] The stock market saw significant gains today.
```
