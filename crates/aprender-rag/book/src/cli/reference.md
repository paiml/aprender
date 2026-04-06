# CLI Reference

The `trueno-rag` CLI provides indexing, querying, transcription, and evaluation capabilities.

## Installation

```bash
cargo install trueno-rag-cli

# Or build from source
cargo build --release -p trueno-rag-cli
```

## Subcommands

### `demo`

Run a demo RAG query with built-in sample documents.

```bash
trueno-rag demo --query "What is machine learning?" --top-k 3
```

| Flag | Default | Description |
|------|---------|-------------|
| `--query` | "What is machine learning?" | Query string |
| `--top-k` | 3 | Number of results |

### `index`

Index documents from a file or directory into a persistent index.

```bash
# Basic indexing
trueno-rag index --path docs/ --output index/

# Recursive with timestamp chunking (for transcripts)
trueno-rag index --path /data/courses --output /data/index \
  --chunk-strategy timestamp --recursive --dimension 4096 --jobs 16

# Semantic embeddings
trueno-rag index --path docs/ --output index/ \
  --embedder semantic --model mini-lm

# Exclude directories
trueno-rag index --path /data --output index/ \
  --recursive --exclude "*/RAW" --exclude "*/RAW/*"

# Export SQLite+FTS5 index alongside JSON (for BM25 search via batuta oracle)
trueno-rag index --path /data/courses --output /data/index \
  --recursive --dedup --sqlite
```

| Flag | Default | Description |
|------|---------|-------------|
| `--path` | (required) | Path to document(s) or directory |
| `--output` | (required) | Output directory for index |
| `--chunk-size` | 512 | Chunk size in characters (recursive chunker) |
| `--chunk-overlap` | 64 | Chunk overlap in characters |
| `--dimension` | 256 | Embedding dimension (TF-IDF only) |
| `--embedder` | tfidf | Embedder type: `tfidf` or `semantic` |
| `--model` | mini-lm | Semantic model: `mini-lm`, `bge-small`, `bge-base` |
| `--recursive` | false | Recursively scan subdirectories |
| `--chunk-strategy` | auto | Chunking: `auto`, `recursive`, `timestamp` |
| `--jobs` | 1 | Parallel loading jobs |
| `--manifest` | false | Write a JSON manifest of indexed files |
| `--exclude` | (none) | Glob patterns to exclude (repeatable) |
| `--dedup` | false | Deduplicate chunks with identical content (keeps first occurrence) |
| `--sqlite` | false | Also export a SQLite+FTS5 index (`index.sqlite`) for BM25 search |

### `query`

Query an existing index. Default mode is **hybrid** (BM25 + dense with RRF fusion), which combines keyword matching with semantic understanding.

Dense and hybrid modes **auto-detect** the index's embedder type: if the index was built with `--embedder semantic`, queries use the same semantic model (BGE-small, BGE-base, or MiniLM via ONNX). Otherwise, TF-IDF is used. No extra flags needed.

```bash
# Hybrid retrieval (default — BM25 + semantic RRF fusion)
trueno-rag query "how does Kubernetes handle pod scheduling" --index index/

# BM25-only (keyword-rich queries)
trueno-rag query "AWS Lambda function" --index index/ --mode sparse

# Dense cosine similarity (auto-detects TF-IDF or semantic)
trueno-rag query "machine learning" --index index/ --mode dense

# Hybrid retrieval (BM25 + dense with RRF fusion)
trueno-rag query "how does Kubernetes handle pod scheduling" \
  --index index/ --mode hybrid

# JSON output with custom fusion
trueno-rag query "AWS Lambda" --index index/ --format json \
  --mode hybrid --fusion rrf --fusion-k 30 --candidates 100

# Hybrid + lexical reranking (fetches 3x candidates, re-orders by term coverage)
trueno-rag query "how does Kubernetes handle pod scheduling" \
  --index index/ --mode hybrid --rerank lexical
```

| Flag | Default | Description |
|------|---------|-------------|
| (positional) | (required) | Query string |
| `--index` | (required) | Path to index directory |
| `--top-k` | 5 | Number of results |
| `--format` | text | Output format: `text` or `json` |
| `--mode` | hybrid | Retrieval mode: `dense`, `sparse`, `hybrid` |
| `--fusion` | rrf | Fusion strategy (hybrid only): `rrf`, `linear`, `dbsf` |
| `--fusion-k` | (varies) | Fusion parameter: RRF k value or Linear dense_weight |
| `--candidates` | 50 | Candidates per source (hybrid only) |
| `--rerank` | none | Reranking strategy: `none`, `lexical` |

### `transcribe`

Batch transcribe media files to SRT sidecars using whisper-apr.

Requires the `transcription` feature: `cargo build --features transcription`

```bash
# Basic transcription
trueno-rag transcribe --path /data/courses --recursive --model base.apr

# With hotword biasing and parallel jobs
trueno-rag transcribe --path /data/courses \
  --recursive --skip-existing --jobs 16 \
  --model /data/models/base.apr \
  --hotwords hotwords.txt \
  --exclude "*/RAW" --exclude "*/RAW/*"

# Dry run (list files only)
trueno-rag transcribe --path /data/courses --recursive --dry-run
```

| Flag | Default | Description |
|------|---------|-------------|
| `--path` | (required) | Directory containing media files |
| `--recursive` | false | Scan subdirectories |
| `--skip-existing` | true | Skip files with existing .srt/.vtt sidecars |
| `--jobs` | 1 | Parallel transcription jobs |
| `--model` | (none) | Path to Whisper .apr model file |
| `--backend` | cpu | Compute backend: `cpu`, `gpu`, `cuda` |
| `--dry-run` | false | List files without transcribing |
| `--hotwords` | (none) | Path to hotwords file (one per line) |
| `--exclude` | (none) | Glob patterns to exclude (repeatable) |

### `info`

Show pipeline component information.

```bash
trueno-rag info
```

---

## Eval Subcommands

The evaluation pipeline requires the `eval` feature:

```bash
cargo build --release -p trueno-rag-cli --features eval
```

### `eval sample`

Sample chunks from an index for ground truth generation. No API needed.

```bash
trueno-rag eval sample \
  --index /path/to/index \
  --output sampled-chunks.jsonl \
  --sample-size 250 --seed 42
```

| Flag | Default | Description |
|------|---------|-------------|
| `--index` | (required) | Path to index directory (containing index.json) |
| `--output` | (required) | Output path for sampled chunks JSONL |
| `--sample-size` | 250 | Number of chunks to sample |
| `--seed` | 42 | Random seed for reproducibility |

Sampling is stratified by course directory. Filters: min 50 words, min 15 unique words, skips navigational boilerplate.

### `eval generate`

Generate synthetic ground truth questions via the Claude API.

Requires `ANTHROPIC_API_KEY` environment variable.

```bash
# Full generation
trueno-rag eval generate \
  --index /path/to/index \
  --output ground-truth.jsonl \
  --sample-size 250 --seed 42

# Dry run (sample only, no API calls)
trueno-rag eval generate \
  --index /path/to/index --output /dev/null --dry-run
```

| Flag | Default | Description |
|------|---------|-------------|
| `--index` | (required) | Path to index directory |
| `--output` | (required) | Output path for ground truth JSONL |
| `--sample-size` | 250 | Number of query-chunk pairs |
| `--seed` | 42 | Random seed |
| `--model` | claude-sonnet-4-20250514 | Claude model for generation |
| `--dry-run` | false | Sample only, no API calls |

### `eval retrieve`

Run retrieval queries from ground truth against an index.

Dense and hybrid modes auto-detect the index's embedder type (semantic or TF-IDF).

```bash
# Dense retrieval (auto-detects TF-IDF or semantic embeddings)
trueno-rag eval retrieve \
  --index /path/to/index \
  --ground-truth ground-truth.jsonl \
  --output retrieval-results.jsonl \
  --top-k 10 --mode dense

# Sparse retrieval (BM25 only)
trueno-rag eval retrieve \
  --index /path/to/index \
  --ground-truth ground-truth.jsonl \
  --output retrieval-results-sparse.jsonl \
  --mode sparse

# Hybrid retrieval (BM25 + dense with RRF fusion)
trueno-rag eval retrieve \
  --index /path/to/index \
  --ground-truth ground-truth.jsonl \
  --output retrieval-results-hybrid.jsonl \
  --mode hybrid --fusion rrf --candidates 50

# Hybrid + lexical reranking (fetches 3x candidates, re-orders by term coverage)
trueno-rag eval retrieve \
  --index /path/to/index \
  --ground-truth ground-truth.jsonl \
  --output retrieval-results-reranked.jsonl \
  --mode hybrid --fusion rrf --rerank lexical
```

| Flag | Default | Description |
|------|---------|-------------|
| `--index` | (required) | Path to index directory |
| `--ground-truth` | (required) | Path to ground truth JSONL |
| `--output` | (required) | Output path for retrieval results JSONL |
| `--top-k` | 10 | Results per query |
| `--mode` | dense | Retrieval mode: `dense`, `sparse`, `hybrid` |
| `--fusion` | rrf | Fusion strategy (hybrid only): `rrf`, `linear`, `dbsf` |
| `--fusion-k` | (varies) | Fusion parameter: RRF k value or Linear dense_weight |
| `--candidates` | 50 | Candidates per source (hybrid only) |
| `--rerank` | none | Reranking strategy: `none`, `lexical` |

### `eval judge`

Judge retrieval results for relevance via the Claude API.

Requires `ANTHROPIC_API_KEY` environment variable.

```bash
trueno-rag eval judge \
  --retrieval-results retrieval-results.jsonl \
  --ground-truth ground-truth.jsonl \
  --output results.json \
  --cache judge-cache.json
```

| Flag | Default | Description |
|------|---------|-------------|
| `--retrieval-results` | (required) | Retrieval results JSONL |
| `--ground-truth` | (required) | Ground truth JSONL (for metadata) |
| `--output` | (required) | Output path for eval results JSON |
| `--cache` | judge-cache.json | SHA256-keyed judge cache path |
| `--top-k` | 10 | Results to judge per query |
| `--model` | claude-sonnet-4-20250514 | Claude model for judging |

### `eval metrics`

Compute IR metrics from pre-judged results. No API needed.

```bash
trueno-rag eval metrics \
  --retrieval-results retrieval-results.jsonl \
  --judgments judgments.jsonl \
  --output results.json
```

| Flag | Default | Description |
|------|---------|-------------|
| `--retrieval-results` | (required) | Retrieval results JSONL |
| `--judgments` | (required) | Judgments JSONL |
| `--output` | (required) | Output path for eval results JSON |

### `eval compare`

Compare two evaluation result files, printing metric deltas.

```bash
trueno-rag eval compare \
  --baseline results-baseline.json \
  --candidate results-hybrid.json
```

### `eval gate`

Regression gate. Exits with non-zero status if metrics fall below thresholds.

```bash
trueno-rag eval gate \
  --results results.json \
  --min-mrr 0.50 --min-hit5 0.70
```

| Flag | Default | Description |
|------|---------|-------------|
| `--results` | (required) | Eval results JSON |
| `--min-mrr` | 0.50 | Minimum MRR threshold |
| `--min-hit5` | 0.70 | Minimum Hit@5 threshold |
