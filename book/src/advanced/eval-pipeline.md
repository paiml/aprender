# Evaluation Pipeline

Trueno-RAG includes a built-in evaluation framework for measuring retrieval quality using synthetic ground truth and LLM-as-judge scoring.

## Overview

The evaluation pipeline has 7 steps:

```
sample → generate → retrieve → judge → metrics → compare → gate
```

| Step | Command | API Required |
|------|---------|-------------|
| 1. Sample chunks | `eval sample` | No |
| 2. Generate questions | `eval generate` | Yes (Claude API) |
| 3. Run retrieval | `eval retrieve` | No |
| 4. Judge relevance | `eval judge` | Yes (Claude API) |
| 5. Compute metrics | `eval metrics` | No |
| 6. Compare results | `eval compare` | No |
| 7. Regression gate | `eval gate` | No |

Steps 2 and 4 require an `ANTHROPIC_API_KEY`. The remaining steps run entirely offline.

## Step 1: Sample Chunks

Stratified sampling from an index. Groups chunks by source directory and samples 2-3 from each, filtering out navigational boilerplate (min 50 words, min 15 unique words).

```bash
trueno-rag eval sample \
  --index /path/to/index \
  --output sampled-chunks.jsonl \
  --sample-size 250 --seed 42
```

Output format (`sampled-chunks.jsonl`):

```json
{"content": "When a CloudFormation stack update fails...", "source": "/data/courses/52-weeks-aws/build/22.0.srt", "start_secs": 145.0, "end_secs": 175.0, "domain": "aws", "course": "52-weeks-aws"}
```

## Step 2: Generate Questions

Generate one evaluation question per sampled chunk. Questions are designed to be answerable only from the chunk text.

```bash
trueno-rag eval generate \
  --index /path/to/index \
  --output ground-truth.jsonl \
  --sample-size 250 --seed 42
```

Output format (`ground-truth.jsonl`):

```json
{"query": "How does CloudFormation handle rollback on stack update failure?", "chunk_content": "When a CloudFormation stack update fails...", "chunk_source": "/data/courses/52-weeks-aws/build/22.0.srt", "domain": "aws", "course": "52-weeks-aws"}
```

## Step 3: Run Retrieval

Run each ground-truth query against the index. Supports three retrieval modes:

```bash
# Dense: TF-IDF cosine similarity
trueno-rag eval retrieve \
  --index /path/to/index \
  --ground-truth ground-truth.jsonl \
  --output retrieval-dense.jsonl \
  --mode dense --top-k 10

# Sparse: BM25 term matching
trueno-rag eval retrieve \
  --index /path/to/index \
  --ground-truth ground-truth.jsonl \
  --output retrieval-sparse.jsonl \
  --mode sparse --top-k 10

# Hybrid: BM25 + TF-IDF with fusion
trueno-rag eval retrieve \
  --index /path/to/index \
  --ground-truth ground-truth.jsonl \
  --output retrieval-hybrid.jsonl \
  --mode hybrid --fusion rrf --candidates 50 --top-k 10
```

### Retrieval Modes

Dense and hybrid modes **auto-detect** the index's embedder type. If the index was built with `--embedder semantic`, queries use the same model (BGE-small, BGE-base, or MiniLM via ONNX). Otherwise, TF-IDF is used.

| Mode | Method | Best For |
|------|--------|----------|
| `dense` | Cosine similarity (auto-detects TF-IDF or semantic) | Semantic matching |
| `sparse` | BM25 term frequency | Exact keyword matching |
| `hybrid` | BM25 + dense with fusion (recommended) | Combines keyword + semantic (best MRR) |

### Reranking

Optional post-retrieval reranking fetches 3x candidates and re-orders by the reranker's scoring:

```bash
trueno-rag eval retrieve \
  --index /path/to/index \
  --ground-truth ground-truth.jsonl \
  --output retrieval-reranked.jsonl \
  --mode hybrid --fusion rrf --rerank lexical
```

| Reranker | Description |
|----------|-------------|
| `none` | No reranking (default) |
| `lexical` | Term coverage + exact match reranking |

### Fusion Strategies

| Strategy | Flag | Description |
|----------|------|-------------|
| RRF | `--fusion rrf` | Reciprocal Rank Fusion (k=60 default) |
| Linear | `--fusion linear` | Weighted linear combination |
| DBSF | `--fusion dbsf` | Distribution-Based Score Fusion |

## Step 4: Judge Relevance

Judge each (query, retrieved chunk) pair for content relevance using an LLM.

```bash
trueno-rag eval judge \
  --retrieval-results retrieval-hybrid.jsonl \
  --ground-truth ground-truth.jsonl \
  --output results.json \
  --cache judge-cache.json
```

Judgments are cached by SHA256 hash of (query, chunk content), so re-runs are free.

## Step 5: Compute Metrics

Compute IR metrics from retrieval results and judgments.

```bash
trueno-rag eval metrics \
  --retrieval-results retrieval-hybrid.jsonl \
  --judgments judgments.jsonl \
  --output results.json
```

### Metrics

| Metric | Description |
|--------|-------------|
| MRR | Mean Reciprocal Rank — 1/rank of first relevant result |
| NDCG@k | Normalized Discounted Cumulative Gain — position-weighted ranking quality |
| Recall@k | Fraction of relevant content found in top-k |
| Precision@k | Fraction of top-k results that are relevant |
| Hit Rate@k | Whether any relevant result appears in top-k |
| MAP | Mean Average Precision |

Metrics are computed at k=5 and k=10, reported both in aggregate and by domain.

## Step 6: Compare Results

Compare two evaluation runs to measure improvement.

```bash
trueno-rag eval compare \
  --baseline results-dense.json \
  --candidate results-hybrid.json
```

Prints metric deltas with directional arrows showing improvement or regression.

## Step 7: Regression Gate

CI-friendly gate that exits non-zero if metrics fall below thresholds.

```bash
trueno-rag eval gate \
  --results results.json \
  --min-mrr 0.50 --min-hit5 0.70
```

## Domain Classification

Chunks are automatically classified into domains based on their source directory path:

| Domain | Example Patterns |
|--------|-----------------|
| `aws` | `52-weeks-aws`, `aws-cloud-practitioner`, `sagemaker` |
| `ml` | `machine-learning`, `pytorch`, `deep-learning` |
| `k8s` | `kubernetes`, `docker`, `container` |
| `lang` | `rust`, `python`, `golang` |
| `devops` | `devops`, `github-actions`, `terraform` |
| `data` | `data-engineering`, `databricks`, `spark` |
| `cloud` | `going-pro-cloud-computing`, `duke-cloud` |
| `other` | Everything else |

## Feature Flags

All eval subcommands require the `eval` feature. For semantic embedding support, also enable `embeddings`:

```bash
# Eval only (TF-IDF + BM25)
cargo build --release -p trueno-rag-cli --features eval

# Eval + semantic embeddings (BGE-small, BGE-base, MiniLM via ONNX)
cargo build --release -p trueno-rag-cli --features eval,embeddings
```
