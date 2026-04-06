# Installation

## Requirements

- Rust 1.70+ (stable)
- Cargo

## From crates.io

```bash
cargo add trueno-rag
```

## From Source

```bash
git clone https://github.com/noahgift/trueno-rag
cd trueno-rag
cargo build --release
```

## Running Tests

```bash
# All tests
cargo test

# Fast tests (release mode)
make test-fast

# With coverage
make coverage
```

## Running Examples

Trueno-RAG includes several examples demonstrating key features:

```bash
# Run all examples
make examples

# Run individual examples (default features)
cargo run --example basic_rag
cargo run --example chunking_strategies
cargo run --example hybrid_search
cargo run --example metrics_evaluation
cargo run --example eval_hybrid

# Run examples with optional features
cargo run --example compressed_index --features compression
cargo run --example semantic_embeddings --features embeddings
cargo run --example nemotron_embeddings --features nemotron
```

### Available Examples

| Example | Feature | Description |
|---------|---------|-------------|
| `basic_rag` | default | Complete RAG pipeline with indexing and querying |
| `chunking_strategies` | default | Comparison of different chunking approaches |
| `hybrid_search` | default | Dense + sparse hybrid retrieval with fusion strategies |
| `metrics_evaluation` | default | Retrieval quality metrics (Recall, MRR, NDCG, MAP) |
| `eval_hybrid` | default | Hybrid vs dense vs sparse retrieval comparison with metrics |
| `compressed_index` | `compression` | LZ4/ZSTD index compression (5-10x ratios) |
| `semantic_embeddings` | `embeddings` | Production semantic search with FastEmbed ONNX |
| `nemotron_embeddings` | `nemotron` | NVIDIA Embed Nemotron 8B via GGUF |

## Feature Flags

### Library (`trueno-rag`)

| Feature | Description |
|---------|-------------|
| `sqlite` (default) | SQLite+FTS5 persistent BM25 index |
| `compression` | LZ4/ZSTD index compression |
| `embeddings` | Semantic embeddings via ONNX FastEmbed |
| `nemotron` | NVIDIA Embed Nemotron 8B via GGUF inference |
| `multivector` | ColBERT-style multi-vector with WARP algorithm |
| `transcription` | Speech-to-text via whisper-apr |
| `eval` | Evaluation pipeline (sample, retrieve, metrics, compare, gate) |

### CLI (`trueno-rag-cli`)

The CLI has its own feature flags that forward to the library:

| Feature | Description |
|---------|-------------|
| `sqlite` | Enable `--sqlite` flag on `index` subcommand (exports `index.sqlite`) |
| `embeddings` | Enable semantic embedder for `index` and `query` |
| `nemotron` | Enable Nemotron embedder |
| `transcription` | Enable `transcribe` subcommand |
| `eval` | Enable `eval` subcommands |

Build with multiple features:

```bash
cargo build --release -p trueno-rag-cli --features eval,embeddings,sqlite
```
