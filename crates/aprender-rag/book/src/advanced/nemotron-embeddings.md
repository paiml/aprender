# Nemotron Embeddings (GH-3)

NVIDIA Embed Nemotron 8B provides high-quality 4096-dimensional
embeddings via GGUF model inference. Built on Llama 3.1 8B
architecture with specialized training for retrieval tasks.

## Quick Start

```rust
use trueno_rag::embed::{NemotronEmbedder, NemotronConfig, Embedder, cosine_similarity};

// Configure the embedder
let config = NemotronConfig::new("models/NV-Embed-v2-Q4_K.gguf")
    .with_gpu(true)         // Use GPU if available
    .with_normalize(true);  // L2 normalize embeddings

let embedder = NemotronEmbedder::new(config)?;

// Embed query with asymmetric prefix
let query_emb = embedder.embed_query("What is machine learning?")?;

// Embed document without prefix
let doc_emb = embedder.embed_document("Machine learning is a branch of AI...")?;

// Compute similarity
let similarity = cosine_similarity(&query_emb, &doc_emb);
```

## Model Download

Nemotron requires a GGUF model file. Download from Hugging Face:

```bash
# Example: Download quantized model (~4GB)
wget https://huggingface.co/nvidia/NV-Embed-v2-GGUF/resolve/main/NV-Embed-v2-Q4_K.gguf

# Set environment variable for examples
export NEMOTRON_MODEL_PATH=/path/to/NV-Embed-v2-Q4_K.gguf
```

## Asymmetric Retrieval

Nemotron uses different prefixes for queries vs documents:

```rust
use trueno_rag::embed::{NemotronEmbedder, NemotronConfig, Embedder};

let config = NemotronConfig::new("model.gguf");
let embedder = NemotronEmbedder::new(config)?;

// Query gets instruction prefix: "Instruct: Given a query, retrieve relevant documents\nQuery: "
let query_emb = embedder.embed_query("What is RAG?")?;

// Document gets no prefix (empty string)
let doc_emb = embedder.embed_document("RAG combines retrieval with generation...")?;

// This asymmetry improves retrieval quality
```

## Configuration Options

```rust
use trueno_rag::embed::NemotronConfig;

let config = NemotronConfig::new("model.gguf")
    .with_gpu(true)                    // GPU acceleration
    .with_batch_size(8)                // Batch processing
    .with_max_length(8192)             // Max sequence length
    .with_normalize(true)              // L2 normalization
    .with_query_prefix("Query: ")      // Custom query prefix
    .with_passage_prefix("");          // Custom passage prefix
```

## Integration with Pipeline

```rust
use trueno_rag::{
    pipeline::RagPipelineBuilder,
    chunk::RecursiveChunker,
    embed::{NemotronEmbedder, NemotronConfig},
    rerank::LexicalReranker,
    fusion::FusionStrategy,
    Document,
};

let config = NemotronConfig::new("models/NV-Embed-v2-Q4_K.gguf")
    .with_gpu(true);
let embedder = NemotronEmbedder::new(config)?;

let mut pipeline = RagPipelineBuilder::new()
    .chunker(RecursiveChunker::new(512, 50))
    .embedder(embedder)
    .reranker(LexicalReranker::new())
    .fusion(FusionStrategy::RRF { k: 60.0 })
    .build()?;

// Index documents with Nemotron embeddings (4096 dims)
let doc = Document::new("Your content here...").with_title("Title");
pipeline.index_document(&doc)?;
```

## Comparison with FastEmbed

| Feature | FastEmbed | Nemotron |
|---------|-----------|----------|
| Dimensions | 384-768 | 4096 |
| Model Size | 90-550MB | 4-32GB |
| Speed | Fast (ONNX) | Slower (GGUF) |
| Quality | Good | Excellent |
| GPU Support | ONNX Runtime | Via realizar |
| Asymmetric | No | Yes |
| Context Length | 512-8192 | 8192 |

## API Reference

### NemotronConfig

```rust
pub struct NemotronConfig {
    pub model_path: PathBuf,      // Path to GGUF model
    pub use_gpu: bool,            // GPU acceleration
    pub batch_size: usize,        // Batch processing size
    pub query_prefix: String,     // Query instruction prefix
    pub passage_prefix: String,   // Document prefix
    pub max_length: usize,        // Max sequence length
    pub normalize: bool,          // L2 normalization
}

impl NemotronConfig {
    pub fn new(model_path: impl AsRef<Path>) -> Self;
    pub fn with_gpu(self, use_gpu: bool) -> Self;
    pub fn with_batch_size(self, batch_size: usize) -> Self;
    pub fn with_query_prefix(self, prefix: impl Into<String>) -> Self;
    pub fn with_passage_prefix(self, prefix: impl Into<String>) -> Self;
    pub fn with_max_length(self, max_length: usize) -> Self;
    pub fn with_normalize(self, normalize: bool) -> Self;
}
```

### NemotronEmbedder

```rust
impl NemotronEmbedder {
    pub fn new(config: NemotronConfig) -> Result<Self>;
    pub fn config(&self) -> &NemotronConfig;
}

impl Embedder for NemotronEmbedder {
    fn embed(&self, text: &str) -> Result<Vec<f32>>;
    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;
    fn dimension(&self) -> usize;  // 4096 for Nemotron 8B
    fn model_id(&self) -> &str;    // "nvidia/NV-Embed-v2"
    fn embed_query(&self, query: &str) -> Result<Vec<f32>>;
    fn embed_document(&self, document: &str) -> Result<Vec<f32>>;
}
```

## Feature Flag

Nemotron embeddings require the `nemotron` feature:

```toml
[dependencies]
trueno-rag = { version = "0.1.8", features = ["nemotron"] }
```

This adds the realizar dependency for GGUF model inference.

## Running the Example

```bash
# Set model path
export NEMOTRON_MODEL_PATH=/path/to/model.gguf

# Run example
cargo run --example nemotron_embeddings --features nemotron
```

## Best Practices

1. **GPU Acceleration**: Enable GPU for 10-50x speedup on large batches
2. **Batch Processing**: Process multiple documents together for efficiency
3. **Model Quantization**: Use Q4_K or Q5_K quantization for memory efficiency
4. **Caching**: Store embeddings with your index to avoid recomputation
5. **Hybrid Retrieval**: Combine with BM25 for robust results

```rust
// Production pattern
let config = NemotronConfig::new("models/NV-Embed-v2-Q4_K.gguf")
    .with_gpu(true)
    .with_batch_size(16)
    .with_normalize(true);

let embedder = NemotronEmbedder::new(config)?;

let mut pipeline = RagPipelineBuilder::new()
    .chunker(RecursiveChunker::new(512, 50))
    .embedder(embedder)
    .fusion(FusionStrategy::RRF { k: 60.0 })
    .build()?;
```

## Troubleshooting

### Model Not Found

```
Error: Model file not found: /path/to/model.gguf
```

Ensure the model path is correct and the file exists.

### GGUF Parse Error

```
Error: Failed to parse GGUF model
```

Verify the GGUF file is valid and not corrupted. Re-download if necessary.

### Out of Memory

For large models, use quantized versions (Q4_K, Q5_K) or disable GPU:

```rust
let config = NemotronConfig::new("model.gguf")
    .with_gpu(false);  // CPU inference
```
