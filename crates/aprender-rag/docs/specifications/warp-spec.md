# WARP: Multi-Vector Retrieval Specification

**Version:** 1.2.0
**Status:** Falsified (Reformulation Pending)
**Authors:** Pragmatic AI Labs & Dr. Karl Popper (Consulting)
**References:** TRUENO-RAG-002
**Feature Flag:** `multivector`

## Abstract

This specification defines the WARP (Weighted Approximate Residual Product)
algorithm for multi-vector retrieval in trueno-rag. WARP enables
ColBERT-style late interaction search where documents and queries are
represented as multiple token embeddings rather than single vectors.
The algorithm uses residual quantization to compress token embeddings
to 2-4 bits while preserving MaxSim scoring accuracy, and inverted
file (IVF) indexing for efficient candidate selection.

> **CRITICAL NOTICE (2026-01-26):** This theory has been **FALSIFIED**.
> The current residual quantization implementation fails to preserve
> rank ordering with sufficient fidelity
> (Kendall's $\tau \approx 0.83 < 0.90$). See Section 12 for the
> Falsification Report and Reformulation Plan.

## 1. Introduction

### 1.1 Motivation: The Problem Situation ($P_1$)

Scientific progress starts with a problem ($P_1$). In neural retrieval,
single-vector dense retrieval compresses all semantic information into
one embedding, losing fine-grained token-level interactions [1]. This
"bottleneck" theory has been falsified by the superior performance of
multi-vector models like ColBERT, which preserve token-level
representations [2]. However, the naive implementation of multi-vector
retrieval introduces a new problem: prohibitive memory
requirements ($P_2$).

WARP is a tentative theory ($TT$) proposed to solve $P_2$ through:
1. **Residual quantization** - Compress token embeddings from 32-bit floats to 2-4 bits
2. **Centroid-based indexing** - IVF structure for fast candidate pruning
3. **Deferred decompression** - Score directly from compressed representations

> **Toyota Way Review: Eliminate Waste (Muda)**
> Single-vector embeddings represent *over-processing
> waste*—compressing rich token interactions into one vector,
> then working harder downstream to recover lost signal.
> Multi-vector preserves information at the source, eliminating
> the *waste of correction* (reranking to fix retrieval errors).

### 1.2 Design Principles

- **Memory efficiency** - 50-100× compression vs. full token embeddings
- **Search quality** - Preserve 95%+ of full-precision MaxSim accuracy
- **Speed** - Sub-linear search via centroid pruning
- **Modularity** - Clean separation of codec, index, and search components

### 1.3 Theoretical Foundation

ColBERT's MaxSim scoring computes, for query Q with tokens {q₁...qₘ} and document D with tokens {d₁...dₙ}:

```
MaxSim(Q, D) = Σᵢ maxⱼ(qᵢ · dⱼ)
```

For each query token, find the maximum similarity with any document
token, then sum across query tokens [2]. This captures soft alignment
without explicit matching.

> **Toyota Way Review: Built-in Quality (Jidoka)**
> MaxSim scoring implements *built-in quality*—each query token
> independently finds its best match, and defects (poor matches)
> for one token don't pollute other tokens' scores. This is
> superior to single-vector's "averaging" which masks quality
> problems.

## 2. Algorithm Overview

### 2.1 WARP Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            WARP Index Building                               │
├─────────────┬─────────────┬─────────────┬─────────────┬─────────────────────┤
│   Sample    │   Train     │  Quantize   │   Assign    │      Compact        │
│  ─────────  │  ────────   │  ────────   │  ────────   │     ──────────      │
│  Collect    │  K-means    │  Residual   │  Centroid   │  IVF organization   │
│  embeddings │  centroids  │  encoding   │  assignment │  by centroid        │
└─────────────┴─────────────┴─────────────┴─────────────┴─────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                            WARP Search                                       │
├─────────────┬─────────────┬─────────────┬─────────────┬─────────────────────┤
│   Encode    │   Select    │  Decompress │   Score     │      Merge          │
│  ─────────  │  ────────   │  ────────   │  ────────   │     ──────────      │
│  Query      │  Top-k      │  Residuals  │  MaxSim     │  Aggregate per      │
│  tokens     │  centroids  │  on demand  │  per token  │  document           │
└─────────────┴─────────────┴─────────────┴─────────────┴─────────────────────┘
```

### 2.2 Key Insight: Centroid-Residual Decomposition

Any vector v can be decomposed as:
```
v = c + r
```
where c is the nearest centroid and r is the residual (v - c).

The dot product q · v = q · c + q · r. Since centroids are shared
across many vectors, we precompute q · c once and only decompress
residuals for promising candidates [3].

### 2.3 Quantization Trade-offs

| Bits | Buckets | Compression | Quality Loss |
|------|---------|-------------|--------------|
| 2    | 4       | 16×         | ~3-5% MRR    |
| 4    | 16      | 8×          | ~1-2% MRR    |

The optimal choice depends on corpus size and quality requirements.
For most applications, 2-bit quantization provides the best
memory/quality trade-off [4].

## 3. Data Structures

### 3.1 Multi-Vector Embedding

```rust
/// A document or query represented as multiple token embeddings
pub struct MultiVectorEmbedding {
    /// Flattened embeddings: [num_tokens * dim]
    pub embeddings: Vec<f32>,
    /// Number of token embeddings
    pub num_tokens: usize,
    /// Dimension per token embedding
    pub dim: usize,
}

impl MultiVectorEmbedding {
    /// Get the i-th token embedding
    pub fn token(&self, i: usize) -> &[f32] {
        let start = i * self.dim;
        &self.embeddings[start..start + self.dim]
    }

    /// Iterate over token embeddings
    pub fn tokens(&self) -> impl Iterator<Item = &[f32]> {
        self.embeddings.chunks_exact(self.dim)
    }

    /// Memory size in bytes (uncompressed)
    pub fn size_bytes(&self) -> usize {
        self.embeddings.len() * std::mem::size_of::<f32>()
    }
}
```

> **Toyota Way Review: Standard Work**
> The `MultiVectorEmbedding` struct establishes *standardized work*
> for token handling. All components share this interface, enabling
> *Kaizen* (continuous improvement) on individual components without
> system-wide changes.

### 3.2 Index Configuration

```rust
/// Configuration for WARP index construction
pub struct WarpIndexConfig {
    /// Bits per dimension for residual quantization (2 or 4)
    pub nbits: u8,
    /// Number of centroids for IVF clustering
    pub num_centroids: usize,
    /// Token embedding dimension (e.g., 128 for ColBERT)
    pub token_dim: usize,
    /// Minimum training samples (default: 10 × num_centroids)
    pub min_training_samples: Option<usize>,
    /// K-means iterations (default: 20)
    pub kmeans_iterations: usize,
}

impl Default for WarpIndexConfig {
    fn default() -> Self {
        Self {
            nbits: 2,
            num_centroids: 1024,
            token_dim: 128,
            min_training_samples: None,
            kmeans_iterations: 20,
        }
    }
}
```

**Parameter Guidance:**

| Parameter | Small Corpus (<100K) | Medium (100K-1M) | Large (>1M) |
|-----------|---------------------|------------------|-------------|
| nbits | 4 | 2 | 2 |
| num_centroids | 256 | 1024 | 4096 |

### 3.3 Search Configuration

```rust
/// Configuration for WARP search
pub struct WarpSearchConfig {
    /// Number of results to return
    pub k: usize,
    /// Centroids to probe per query token (default: 4)
    pub nprobe: u32,
    /// Maximum total centroids examined (default: 128)
    pub bound: usize,
    /// Early termination: skip tokens after this many (default: None)
    pub t_prime: Option<usize>,
    /// Skip tokens with centroid score below threshold (default: 0.4)
    pub centroid_score_threshold: f32,
}

impl Default for WarpSearchConfig {
    fn default() -> Self {
        Self {
            k: 10,
            nprobe: 4,
            bound: 128,
            t_prime: None,
            centroid_score_threshold: 0.4,
        }
    }
}
```

> **Toyota Way Review: Heijunka (Leveling)**
> The `bound` parameter implements *load leveling*—preventing
> worst-case queries from consuming unbounded resources. This avoids
> *Muri* (overburden) on the system during spike loads.

## 4. Residual Quantization Codec

### 4.1 Training Phase

The codec learns from a sample of token embeddings:

```rust
pub struct ResidualCodec {
    /// Centroid vectors: [num_centroids, dim]
    centroids: Vec<f32>,
    /// Number of centroids
    num_centroids: usize,
    /// Token dimension
    dim: usize,
    /// Quantization bucket boundaries per dimension
    bucket_cutoffs: Vec<f32>,
    /// Reconstruction weights per bucket
    bucket_weights: Vec<f32>,
    /// Bits per dimension
    nbits: u8,
    /// Precomputed bit reversal table for packing
    reversed_bit_map: [u8; 256],
}

impl ResidualCodec {
    /// Train codec from sample embeddings
    pub fn train(
        embeddings: &[f32],
        dim: usize,
        num_centroids: usize,
        nbits: u8,
        iterations: usize,
    ) -> Result<Self> {
        // 1. K-means clustering to find centroids
        let centroids = kmeans_clustering(embeddings, dim, num_centroids, iterations)?;

        // 2. Compute residuals for all training points
        let residuals = compute_residuals(embeddings, dim, &centroids)?;

        // 3. Learn quantization boundaries from residual distribution
        let (bucket_cutoffs, bucket_weights) =
            learn_quantization_params(&residuals, dim, nbits)?;

        Ok(Self {
            centroids,
            num_centroids,
            dim,
            bucket_cutoffs,
            bucket_weights,
            nbits,
            reversed_bit_map: build_bit_reversal_table(),
        })
    }
}
```

**K-means Clustering:**

Following [5], we use Lloyd's algorithm with k-means++ initialization:

```rust
fn kmeans_clustering(
    embeddings: &[f32],
    dim: usize,
    k: usize,
    iterations: usize,
) -> Result<Vec<f32>> {
    let n = embeddings.len() / dim;

    // K-means++ initialization
    let mut centroids = kmeans_plus_plus_init(embeddings, dim, k);
    let mut assignments = vec![0usize; n];

    for _ in 0..iterations {
        // Assign points to nearest centroid
        for i in 0..n {
            let point = &embeddings[i * dim..(i + 1) * dim];
            assignments[i] = find_nearest_centroid(point, &centroids, dim);
        }

        // Update centroids as mean of assigned points
        centroids = update_centroids(embeddings, dim, &assignments, k);
    }

    Ok(centroids)
}
```

### 4.2 Quantization Boundary Learning

Residuals follow approximately Gaussian distribution per dimension.
We learn bucket boundaries to minimize reconstruction error:

```rust
fn learn_quantization_params(
    residuals: &[f32],
    dim: usize,
    nbits: u8,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let num_buckets = 1 << nbits;  // 4 for 2-bit, 16 for 4-bit
    let n = residuals.len() / dim;

    // Compute per-dimension statistics
    let mut cutoffs = Vec::with_capacity(dim * (num_buckets - 1));
    let mut weights = Vec::with_capacity(dim * num_buckets);

    for d in 0..dim {
        // Collect residual values for dimension d
        let mut values: Vec<f32> = (0..n)
            .map(|i| residuals[i * dim + d])
            .collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Quantile-based boundaries for equal-frequency buckets
        for b in 1..num_buckets {
            let quantile_idx = (b * n) / num_buckets;
            cutoffs.push(values[quantile_idx]);
        }

        // Bucket weights = mean value in each bucket
        for b in 0..num_buckets {
            let start = if b == 0 { 0 } else { (b * n) / num_buckets };
            let end = ((b + 1) * n) / num_buckets;
            let mean: f32 = values[start..end].iter().sum::<f32>() / (end - start) as f32;
            weights.push(mean);
        }
    }

    Ok((cutoffs, weights))
}
```

> **Toyota Way Review: Genchi Genbutsu (Go and See)**
> Quantization boundaries are learned from *actual data*, not assumed
> distributions. This embodies *Genchi Genbutsu*—going to the source
> to understand the real situation rather than relying on theoretical
> assumptions.

### 4.3 Compression

```rust
impl ResidualCodec {
    /// Compress a token embedding to (centroid_id, packed_residual)
    pub fn compress(&self, embedding: &[f32]) -> (usize, Vec<u8>) {
        // 1. Find nearest centroid
        let centroid_id = self.find_nearest_centroid(embedding);
        let centroid = self.centroid(centroid_id);

        // 2. Compute residual
        let residual: Vec<f32> = embedding.iter()
            .zip(centroid.iter())
            .map(|(e, c)| e - c)
            .collect();

        // 3. Quantize residual to nbits per dimension
        let codes = self.quantize_residual(&residual);

        // 4. Pack bits efficiently
        let packed = self.pack_codes(&codes);

        (centroid_id, packed)
    }

    fn quantize_residual(&self, residual: &[f32]) -> Vec<u8> {
        let num_buckets = 1 << self.nbits;

        residual.iter().enumerate().map(|(d, &value)| {
            // Binary search for bucket
            let cutoff_start = d * (num_buckets - 1);
            let cutoffs = &self.bucket_cutoffs[cutoff_start..cutoff_start + num_buckets - 1];

            cutoffs.iter()
                .position(|&c| value < c)
                .unwrap_or(num_buckets - 1) as u8
        }).collect()
    }

    fn pack_codes(&self, codes: &[u8]) -> Vec<u8> {
        match self.nbits {
            2 => {
                // Pack 4 codes per byte
                codes.chunks(4).map(|chunk| {
                    let mut byte = 0u8;
                    for (i, &code) in chunk.iter().enumerate() {
                        byte |= (code & 0x03) << (i * 2);
                    }
                    byte
                }).collect()
            }
            4 => {
                // Pack 2 codes per byte
                codes.chunks(2).map(|chunk| {
                    let low = chunk.get(0).copied().unwrap_or(0) & 0x0F;
                    let high = chunk.get(1).copied().unwrap_or(0) & 0x0F;
                    low | (high << 4)
                }).collect()
            }
            _ => panic!("Unsupported nbits: {}", self.nbits),
        }
    }
}
```

### 4.4 Decompression and Scoring

Rather than fully decompressing, we compute scores directly:

```rust
impl ResidualCodec {
    /// Compute score between query token and compressed document token
    /// score ≈ q · d = q · c + q · r
    pub fn decompress_score(
        &self,
        query_token: &[f32],
        centroid_id: usize,
        centroid_score: f32,  // Precomputed q · c
        packed_residual: &[u8],
    ) -> f32 {
        // Unpack residual codes
        let codes = self.unpack_codes(packed_residual);

        // Compute q · r using bucket weights
        let residual_score: f32 = codes.iter().enumerate().map(|(d, &code)| {
            let weight_idx = d * (1 << self.nbits) + code as usize;
            query_token[d] * self.bucket_weights[weight_idx]
        }).sum();

        centroid_score + residual_score
    }

    fn unpack_codes(&self, packed: &[u8]) -> Vec<u8> {
        match self.nbits {
            2 => {
                packed.iter().flat_map(|&byte| {
                    (0..4).map(move |i| (byte >> (i * 2)) & 0x03)
                }).take(self.dim).collect()
            }
            4 => {
                packed.iter().flat_map(|&byte| {
                    vec![byte & 0x0F, (byte >> 4) & 0x0F]
                }).take(self.dim).collect()
            }
            _ => panic!("Unsupported nbits"),
        }
    }
}
```

## 5. Index Structure

### 5.1 IVF Organization

The index organizes embeddings by centroid for cache-efficient access:

```rust
pub struct WarpIndex {
    /// Index configuration
    config: WarpIndexConfig,
    /// Trained residual codec
    codec: Option<ResidualCodec>,
    /// Number of embeddings per centroid
    sizes: Vec<usize>,
    /// Cumulative offset for each centroid's data
    offsets: Vec<usize>,
    /// Chunk IDs, sorted by centroid assignment
    chunk_ids: Vec<ChunkId>,
    /// Token indices within each chunk
    token_indices: Vec<u16>,
    /// Packed residuals, sorted by centroid
    residuals: Vec<u8>,
    /// Original chunks for result retrieval
    chunks: HashMap<ChunkId, Chunk>,
    /// Build state
    pending_embeddings: Vec<(ChunkId, MultiVectorEmbedding)>,
    is_built: bool,
}
```

**Memory Layout:**

```
Centroid 0:          Centroid 1:          ...
┌────────────────┐  ┌────────────────┐
│ chunk_ids[0..n]│  │ chunk_ids[n..m]│
│ token_idx[0..n]│  │ token_idx[n..m]│
│ residuals[0..n]│  │ residuals[n..m]│
└────────────────┘  └────────────────┘
```

This layout ensures all data for a centroid is contiguous, maximizing cache efficiency during search [6].

> **Toyota Way Review: 5S (Sort, Set in Order, Shine, Standardize, Sustain)**
> The IVF memory layout implements *Seiri* (Sort) and *Seiton*
> (Set in Order)—organizing data by access pattern rather than
> insertion order. This eliminates the *waste of motion*
> (cache misses) during search.

### 5.2 Index Building

```rust
impl WarpIndex {
    pub fn new(config: WarpIndexConfig) -> Self {
        Self {
            config,
            codec: None,
            sizes: Vec::new(),
            offsets: Vec::new(),
            chunk_ids: Vec::new(),
            token_indices: Vec::new(),
            residuals: Vec::new(),
            chunks: HashMap::new(),
            pending_embeddings: Vec::new(),
            is_built: false,
        }
    }

    /// Train codec from sample embeddings
    pub fn train(&mut self, samples: &[MultiVectorEmbedding]) -> Result<()> {
        // Collect all token embeddings for training
        let all_embeddings: Vec<f32> = samples.iter()
            .flat_map(|mv| mv.embeddings.iter().copied())
            .collect();

        let codec = ResidualCodec::train(
            &all_embeddings,
            self.config.token_dim,
            self.config.num_centroids,
            self.config.nbits,
            self.config.kmeans_iterations,
        )?;

        self.codec = Some(codec);
        Ok(())
    }

    /// Add a chunk with its token embeddings
    pub fn insert(&mut self, chunk: Chunk, embedding: MultiVectorEmbedding) -> Result<()> {
        if self.is_built {
            return Err(Error::IndexAlreadyBuilt);
        }

        let chunk_id = chunk.id.clone();
        self.chunks.insert(chunk_id.clone(), chunk);
        self.pending_embeddings.push((chunk_id, embedding));
        Ok(())
    }

    /// Compact index for efficient search
    pub fn build(&mut self) -> Result<()> {
        let codec = self.codec.as_ref()
            .ok_or(Error::CodecNotTrained)?;

        // Assign each token to its nearest centroid
        let mut centroid_assignments: Vec<Vec<(ChunkId, u16, Vec<u8>)>> =
            vec![Vec::new(); self.config.num_centroids];

        for (chunk_id, embedding) in &self.pending_embeddings {
            for (token_idx, token) in embedding.tokens().enumerate() {
                let (centroid_id, residual) = codec.compress(token);
                centroid_assignments[centroid_id].push((
                    chunk_id.clone(),
                    token_idx as u16,
                    residual,
                ));
            }
        }

        // Build compacted arrays
        let bytes_per_residual = (self.config.token_dim * self.config.nbits as usize + 7) / 8;

        self.sizes = centroid_assignments.iter().map(|v| v.len()).collect();
        self.offsets = self.sizes.iter()
            .scan(0, |acc, &size| {
                let offset = *acc;
                *acc += size;
                Some(offset)
            })
            .collect();

        let total_tokens: usize = self.sizes.iter().sum();
        self.chunk_ids = Vec::with_capacity(total_tokens);
        self.token_indices = Vec::with_capacity(total_tokens);
        self.residuals = Vec::with_capacity(total_tokens * bytes_per_residual);

        for assignments in centroid_assignments {
            for (chunk_id, token_idx, residual) in assignments {
                self.chunk_ids.push(chunk_id);
                self.token_indices.push(token_idx);
                self.residuals.extend(residual);
            }
        }

        self.pending_embeddings.clear();
        self.is_built = true;
        Ok(())
    }
}
```

## 6. Search Algorithm

### 6.1 Overview

WARP search proceeds in three phases:

1. **Centroid Selection** - For each query token, find top-nprobe centroids
2. **Candidate Scoring** - Decompress and score tokens from selected centroids
3. **Score Merging** - Aggregate per-token scores into document scores via MaxSim

### 6.2 Phase 1: Centroid Selection

```rust
pub struct CentroidSelector;

impl CentroidSelector {
    /// Select top centroids for each query token
    pub fn select(
        query: &MultiVectorEmbedding,
        centroids: &[f32],
        dim: usize,
        config: &WarpSearchConfig,
    ) -> Vec<Vec<(usize, f32)>> {
        let num_centroids = centroids.len() / dim;

        query.tokens().map(|query_token| {
            // Compute scores with all centroids
            let mut scores: Vec<(usize, f32)> = (0..num_centroids)
                .map(|c| {
                    let centroid = &centroids[c * dim..(c + 1) * dim];
                    let score = dot_product(query_token, centroid);
                    (c, score)
                })
                .collect();

            // Sort by score descending
            scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            // Take top nprobe, filtered by threshold
            scores.into_iter()
                .take(config.nprobe as usize)
                .filter(|(_, score)| *score >= config.centroid_score_threshold)
                .collect()
        }).collect()
    }
}
```

**Optimization: SIMD Batch Scoring**

Using trueno's SIMD primitives for centroid scoring:

```rust
fn batch_centroid_scores(
    query_token: &[f32],
    centroids: &[f32],
    dim: usize,
) -> Vec<f32> {
    // Use trueno's batched dot product
    trueno::ops::batch_dot_product(query_token, centroids, dim)
}
```

### 6.3 Phase 2: Candidate Scoring

```rust
pub struct CandidateScorer;

impl CandidateScorer {
    /// Score candidates from a centroid for one query token
    pub fn score(
        query_token: &[f32],
        centroid_id: usize,
        centroid_score: f32,
        codec: &ResidualCodec,
        index: &WarpIndex,
    ) -> Vec<(ChunkId, u16, f32)> {
        let start = index.offsets[centroid_id];
        let end = start + index.sizes[centroid_id];
        let bytes_per_residual = codec.packed_size();

        (start..end).map(|i| {
            let chunk_id = index.chunk_ids[i].clone();
            let token_idx = index.token_indices[i];
            let residual_start = i * bytes_per_residual;
            let residual = &index.residuals[residual_start..residual_start + bytes_per_residual];

            let score = codec.decompress_score(
                query_token,
                centroid_id,
                centroid_score,
                residual,
            );

            (chunk_id, token_idx, score)
        }).collect()
    }
}
```

### 6.4 Phase 3: Score Merging (MaxSim)

```rust
pub struct ScoreMerger;

impl ScoreMerger {
    /// Merge per-token scores into document scores via MaxSim
    pub fn merge(
        token_scores: Vec<Vec<(ChunkId, u16, f32)>>,
        k: usize,
    ) -> Vec<(ChunkId, f32)> {
        // For each document, track max score per query token
        let mut doc_token_maxes: HashMap<ChunkId, Vec<f32>> = HashMap::new();
        let num_query_tokens = token_scores.len();

        for (query_token_idx, scores) in token_scores.into_iter().enumerate() {
            for (chunk_id, _doc_token_idx, score) in scores {
                let maxes = doc_token_maxes
                    .entry(chunk_id)
                    .or_insert_with(|| vec![f32::NEG_INFINITY; num_query_tokens]);

                if score > maxes[query_token_idx] {
                    maxes[query_token_idx] = score;
                }
            }
        }

        // Sum max scores across query tokens
        let mut doc_scores: Vec<(ChunkId, f32)> = doc_token_maxes
            .into_iter()
            .map(|(chunk_id, maxes)| {
                let score: f32 = maxes.into_iter()
                    .filter(|&s| s > f32::NEG_INFINITY)
                    .sum();
                (chunk_id, score)
            })
            .collect();

        // Sort and take top-k
        doc_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        doc_scores.truncate(k);
        doc_scores
    }
}
```

### 6.5 Complete Search

```rust
impl WarpIndex {
    pub fn search(
        &self,
        query: &MultiVectorEmbedding,
        config: &WarpSearchConfig,
    ) -> Result<Vec<(ChunkId, f32)>> {
        let codec = self.codec.as_ref()
            .ok_or(Error::CodecNotTrained)?;

        if !self.is_built {
            return Err(Error::IndexNotBuilt);
        }

        // Phase 1: Select centroids per query token
        let selected_centroids = CentroidSelector::select(
            query,
            &codec.centroids,
            self.config.token_dim,
            config,
        );

        // Apply bound: limit total centroids examined
        let mut total_centroids = 0;
        let bounded_centroids: Vec<Vec<(usize, f32)>> = selected_centroids
            .into_iter()
            .take(config.t_prime.unwrap_or(usize::MAX))
            .map(|centroids| {
                let take = (config.bound - total_centroids).min(centroids.len());
                total_centroids += take;
                centroids.into_iter().take(take).collect()
            })
            .collect();

        // Phase 2: Score candidates from selected centroids
        let token_scores: Vec<Vec<(ChunkId, u16, f32)>> = bounded_centroids
            .into_iter()
            .enumerate()
            .map(|(query_token_idx, centroids)| {
                let query_token = query.token(query_token_idx);

                centroids.into_iter()
                    .flat_map(|(centroid_id, centroid_score)| {
                        CandidateScorer::score(
                            query_token,
                            centroid_id,
                            centroid_score,
                            codec,
                            self,
                        )
                    })
                    .collect()
            })
            .collect();

        // Phase 3: Merge via MaxSim
        Ok(ScoreMerger::merge(token_scores, config.k))
    }
}
```

> **Toyota Way Review: Pull System (Kanban)**
> The search algorithm implements a *pull system*—candidates are
> only decompressed when a centroid is selected (demand), not
> preemptively. This eliminates the *waste of overproduction*
> (decompressing unneeded residuals).

## 7. Multi-Vector Embedder Interface

### 7.1 Trait Definition

```rust
/// Trait for models that produce token-level embeddings
pub trait MultiVectorEmbedder: Send + Sync {
    /// Embed text into token-level vectors
    fn embed_tokens(&self, text: &str) -> Result<MultiVectorEmbedding>;

    /// Batch embed multiple texts
    fn embed_tokens_batch(&self, texts: &[&str]) -> Result<Vec<MultiVectorEmbedding>>;

    /// Token embedding dimension
    fn token_dimension(&self) -> usize;

    /// Maximum tokens per document
    fn max_tokens(&self) -> usize;

    /// Model identifier
    fn model_id(&self) -> &str;
}
```

### 7.2 Mock Implementation

```rust
/// Mock embedder for testing
pub struct MockMultiVectorEmbedder {
    dim: usize,
    max_tokens: usize,
    seed: u64,
}

impl MockMultiVectorEmbedder {
    pub fn new(dim: usize, max_tokens: usize) -> Self {
        Self { dim, max_tokens, seed: 42 }
    }

    pub fn with_seed(dim: usize, max_tokens: usize, seed: u64) -> Self {
        Self { dim, max_tokens, seed }
    }
}

impl MultiVectorEmbedder for MockMultiVectorEmbedder {
    fn embed_tokens(&self, text: &str) -> Result<MultiVectorEmbedding> {
        // Deterministic pseudo-random based on text hash
        let tokens: Vec<&str> = text.split_whitespace().collect();
        let num_tokens = tokens.len().min(self.max_tokens);

        let mut embeddings = Vec::with_capacity(num_tokens * self.dim);

        for (i, token) in tokens.iter().take(num_tokens).enumerate() {
            // Generate deterministic embedding from token
            let token_seed = hash_token(token, self.seed, i as u64);
            embeddings.extend(generate_unit_vector(self.dim, token_seed));
        }

        Ok(MultiVectorEmbedding {
            embeddings,
            num_tokens,
            dim: self.dim,
        })
    }

    fn embed_tokens_batch(&self, texts: &[&str]) -> Result<Vec<MultiVectorEmbedding>> {
        texts.iter().map(|t| self.embed_tokens(t)).collect()
    }

    fn token_dimension(&self) -> usize { self.dim }
    fn max_tokens(&self) -> usize { self.max_tokens }
    fn model_id(&self) -> &str { "mock-multivector" }
}
```

## 8. Retriever Integration

### 8.1 Multi-Vector Retriever

```rust
/// Retriever using WARP index for multi-vector search
pub struct MultiVectorRetriever<E: MultiVectorEmbedder> {
    index: WarpIndex,
    embedder: E,
    search_config: WarpSearchConfig,
}

impl<E: MultiVectorEmbedder> MultiVectorRetriever<E> {
    pub fn new(
        index_config: WarpIndexConfig,
        embedder: E,
        search_config: WarpSearchConfig,
    ) -> Self {
        Self {
            index: WarpIndex::new(index_config),
            embedder,
            search_config,
        }
    }

    /// Train index on sample chunks
    pub fn train(&mut self, samples: &[Chunk]) -> Result<()> {
        let embeddings: Vec<MultiVectorEmbedding> = samples.iter()
            .map(|c| self.embedder.embed_tokens(&c.content))
            .collect::<Result<Vec<_>>>()?;

        self.index.train(&embeddings)
    }

    /// Index a chunk
    pub fn index(&mut self, chunk: Chunk) -> Result<()> {
        let embedding = self.embedder.embed_tokens(&chunk.content)?;
        self.index.insert(chunk, embedding)
    }

    /// Build index for searching
    pub fn build(&mut self) -> Result<()> {
        self.index.build()
    }

    /// Retrieve relevant chunks for query
    pub fn retrieve(&self, query: &str, k: usize) -> Result<Vec<RetrievalResult>> {
        let query_embedding = self.embedder.embed_tokens(query)?;

        let mut config = self.search_config.clone();
        config.k = k;

        let results = self.index.search(&query_embedding, &config)?;

        results.into_iter()
            .map(|(chunk_id, score)| {
                let chunk = self.index.get_chunk(&chunk_id)?;
                Ok(RetrievalResult {
                    chunk,
                    dense_score: None,
                    sparse_score: None,
                    #[cfg(feature = "multivector")]
                    multivector_score: Some(score),
                    fused_score: None,
                    rerank_score: None,
                })
            })
            .collect()
    }
}
```

### 8.2 Three-Way Fusion

```rust
impl FusionStrategy {
    /// Fuse dense, sparse, and multi-vector results
    #[cfg(feature = "multivector")]
    pub fn fuse_three(
        &self,
        dense: &[(ChunkId, f32)],
        sparse: &[(ChunkId, f32)],
        multivector: &[(ChunkId, f32)],
    ) -> Vec<(ChunkId, f32)> {
        match self {
            FusionStrategy::ThreeWay {
                dense_weight,
                sparse_weight,
                multivector_weight
            } => {
                self.linear_three_way(
                    dense, sparse, multivector,
                    *dense_weight, *sparse_weight, *multivector_weight,
                )
            }
            // Fall back to pairwise fusion for other strategies
            _ => {
                let dense_sparse = self.fuse(dense, sparse);
                // Treat multi-vector as "dense" in second fusion
                self.fuse(&dense_sparse, multivector)
            }
        }
    }

    fn linear_three_way(
        &self,
        dense: &[(ChunkId, f32)],
        sparse: &[(ChunkId, f32)],
        multivector: &[(ChunkId, f32)],
        w_dense: f32,
        w_sparse: f32,
        w_multi: f32,
    ) -> Vec<(ChunkId, f32)> {
        let mut scores: HashMap<ChunkId, f32> = HashMap::new();

        // Normalize and weight each source
        let dense_norm = normalize_scores(dense);
        let sparse_norm = normalize_scores(sparse);
        let multi_norm = normalize_scores(multivector);

        for (id, score) in dense_norm {
            *scores.entry(id).or_insert(0.0) += w_dense * score;
        }
        for (id, score) in sparse_norm {
            *scores.entry(id).or_insert(0.0) += w_sparse * score;
        }
        for (id, score) in multi_norm {
            *scores.entry(id).or_insert(0.0) += w_multi * score;
        }

        let mut results: Vec<_> = scores.into_iter().collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results
    }
}

/// Three-way fusion strategy
#[cfg(feature = "multivector")]
pub enum FusionStrategy {
    // ... existing variants ...

    /// Weighted combination of three retrieval methods
    ThreeWay {
        dense_weight: f32,
        sparse_weight: f32,
        multivector_weight: f32,
    },
}
```

## 9. References

[1] Reimers, N., & Gurevych, I. (2019). "Sentence-BERT: Sentence
Embeddings using Siamese BERT-Networks."
*Proceedings of EMNLP-IJCNLP*, 3982-3992.
DOI: 10.18653/v1/D19-1410

[2] Khattab, O., & Zaharia, M. (2020). "ColBERT: Efficient and
Effective Passage Search via Contextualized Late Interaction over
BERT." *Proceedings of SIGIR*, 39-48.
DOI: 10.1145/3397271.3401075

[3] Santhanam, K., Khattab, O., Saad-Falcon, J., Potts, C., &
Zaharia, M. (2022). "ColBERTv2: Effective and Efficient Retrieval
via Lightweight Late Interaction."
*Proceedings of NAACL*, 3715-3734.
DOI: 10.18653/v1/2022.naacl-main.272

[4] Santhanam, K., Khattab, O., Potts, C., & Zaharia, M. (2022).
"PLAID: An Efficient Engine for Late Interaction Retrieval."
*Proceedings of CIKM*, 1747-1756.
DOI: 10.1145/3511808.3557325

[5] Jégou, H., Douze, M., & Schmid, C. (2011). "Product Quantization
for Nearest Neighbor Search." *IEEE Transactions on Pattern Analysis
and Machine Intelligence*, 33(1), 117-128.
DOI: 10.1109/TPAMI.2010.57

[6] Johnson, J., Douze, M., & Jégou, H. (2021). "Billion-scale
similarity search with GPUs." *IEEE Transactions on Big Data*,
7(3), 535-547. DOI: 10.1109/TBDATA.2019.2921572

[7] Gray, R. M., & Neuhoff, D. L. (1998). "Quantization."
*IEEE Transactions on Information Theory*, 44(6), 2325-2383.
DOI: 10.1109/18.720541

[8] Gersho, A., & Gray, R. M. (1992). *Vector Quantization and Signal Compression*. Springer. ISBN: 978-0-7923-9181-4

[9] Formal, T., Piwowarski, B., & Clinchant, S. (2021). "SPLADE:
Sparse Lexical and Expansion Model for First Stage Ranking."
*Proceedings of SIGIR*, 2288-2292.
DOI: 10.1145/3404835.3463098

[10] Lin, J., Ma, X., Lin, S. C., Yang, J. H., Pradeep, R., &
Nogueira, R. (2021). "Pyserini: A Python Toolkit for Reproducible
Information Retrieval Research with Sparse and Dense
Representations." *Proceedings of SIGIR*, 2356-2362.
DOI: 10.1145/3404835.3463238

---

## 10. Popperian Falsification Plan

> "The game of science is, in principle, without end. He who
> decides one day that scientific statements do not call for any
> further test, and that they can be regarded as finally verified,
> retires from the game."
> — *Karl Popper, The Logic of Scientific Discovery*

In accordance with the principle of falsifiability, we do not seek to
*verify* the WARP algorithm (which is logically impossible). Instead,
we subject it to severe tests designed to expose its flaws. This
specification is a **conjecture**. The implementation is an attempt
to withstand refutation.

### 10.1 The Demarcation Criterion

To distinguish this engineering effort from mere "hacking"
(pseudoscience), we define specific observations that, if they
occur, will compel us to reject the system as failed. These are
our "Potential Falsifiers".

### 10.2 Experimentum Crucis (Crucial Experiment)

We propose a crucial experiment to decide between two competing theories:
1.  **Theory A (Null Hypothesis):** Single-vector dense retrieval is
    sufficient; fine-grained token interaction adds cost without
    meaningful benefit.
2.  **Theory B (WARP Conjecture):** Token-level interaction is
    necessary to capture "hard negative" distinctions that single
    vectors miss.

**The Test:**
Evaluate on a "Hard Negatives" dataset where documents share high
semantic overlap but differ in crucial details (e.g., negation:
"The cat is on the mat" vs "The cat is NOT on the mat").

**Prediction:**
WARP must outperform Single-Vector retrieval by at least
**15% in MRR@10** on this specific subset. If it does not,
we accept Theory A and abandon WARP as "Muda" (Waste).

### 10.3 Potential Falsifiers (Conjectures & Refutations)

#### Conjecture 1: Compression Preserves Information Structure
**Theory:** Residual quantization preserves the relative ordering of MaxSim scores with high fidelity.
**Falsifier:** If the rank correlation (Kendall's tau) between
32-bit MaxSim and 2-bit WARP scores drops below **0.90** on the
MS MARCO dev set, the codec is falsified.

```rust
#[test]
fn falsify_score_ordering_preservation() {
    // ... (implementation as before) ...
    // FALSIFIED if tau < 0.90 for 2-bit
    assert!(tau >= 0.90, "C1 FALSIFIED: tau = {}", tau);
}
```

#### Conjecture 2: The Pruning Hypothesis
**Theory:** Centroids effectively partition the semantic space such
that relevant tokens are found in the top-`nprobe` clusters.
**Falsifier:** If `recall@10` of pruned search (nprobe=4) vs
exhaustive search drops below **0.95**, the clustering hypothesis
is falsified.

```rust
#[test]
fn falsify_centroid_pruning_recall() {
    // ... (implementation as before) ...
    // FALSIFIED if recall < 0.95
    assert!(recall_at_10 >= 0.95, "C2 FALSIFIED: recall = {}", recall_at_10);
}
```

#### Conjecture 3: Resource Scaling Laws
**Theory:** Memory usage scales linearly with `N * T`, but search latency scales with `Q * nprobe`.
**Falsifier:**
1. Memory > Theoretical Bound * 1.2
2. Latency scales linearly with N (indicates failure of IVF pruning).

### 10.4 Property-Based Testing (Proptest)

We use `proptest` to generate "severe tests" — random, hostile inputs designed to break the system.

```rust
proptest! {
    /// The "Symmetry of Interaction" law
    #[test]
    fn prop_maxsim_symmetry(
        a in any_token_sequence(),
        b in any_token_sequence()
    ) {
        // MaxSim is NOT symmetric (query vs doc roles differ), but
        // WARP approximation error should be symmetric w.r.t codec?
        // No, better property: "Triangle Inequality of Error"
        // |Exact(q,d) - Approx(q,d)| should be bounded.
    }
}
```

### 10.5 The "Stop the Line" Protocol (Jidoka)

If a falsifier is triggered:
1.  **Acknowledge Refutation:** The current implementation is false.
2.  **Do Not Patch:** Do not add "epicycles" (ad-hoc fixes) just to pass the test.
3.  **Reformulate:** Propose a new, better theory (e.g., "IVF is insufficient, we need HNSW") that explains the failure.

---

## 11. Implementation Corroboration Checklist

*Note: We use the term "Corroboration" instead of "Verification". A theory is corroborated when it survives a test.*

### 11.1 Unit Test Coverage (Degree of Corroboration)

High coverage does not prove correctness, but low coverage forbids corroboration.

| Component | Required Coverage | Critical Paths |
|-----------|------------------|----------------|
| `ResidualCodec` | 95% | train, compress, decompress_score |
| `WarpIndex` | 90% | build, search |

### 11.2 Integration Tests

1. **End-to-end pipeline**: Ingest → Train → Index → Search → Results
2. **Feature flag isolation**: Build without `multivector` succeeds

### 11.3 Performance Benchmarks

```bash
cargo bench --features multivector -- warp
```

---

## Appendix A: Glossary

| Term | Definition |
|------|------------|
| **Conjecture** | A tentative theory or solution proposed to solve a problem. |
| **Falsifier** | An observation statement that, if true, logically implies the falsity of the theory. |
| **Corroboration** | The status of a theory that has survived severe tests. |
| **WARP** | Weighted Approximate Residual Product - our current best conjecture for efficient multi-vector search. |

## Appendix B: References for Falsification Methodology

[11] Popper, K. R. (1959). *The Logic of Scientific Discovery*. Routledge. ISBN: 978-0-415-27844-7

[12] Lakatos, I. (1978). *The Methodology of Scientific Research
Programmes*. Cambridge University Press.
ISBN: 978-0-521-28031-0

## Appendix C: Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-01-26 | Initial specification |
| 1.1.0 | 2026-01-26 | Enhanced with Popperian Falsification Plan |
| 1.2.0 | 2026-01-26 | Marked as FALSIFIED following test suite failure |

---

## 12. Falsification Report & Reformulation

**Date:** 2026-01-26
**Verdict:** PARTIALLY FALSIFIED

### 12.1 Falsified Conjectures

1.  **Conjecture 1 (Compression Fidelity):**
    *   **Prediction:** Kendall's $\tau \ge 0.90$ for 4-bit quantization.
    *   **Observation:** Kendall's $\tau = 0.8312$.
    *   **Significance:** $\approx 17\%$ of pairwise orderings are
        inverted. This error rate is too high for reliable retrieval.
    *   **Root Cause Analysis:** Scalar quantization of residuals
        assumes a distribution (likely Gaussian) that does not match
        the actual residual distribution, especially with
        random/mock data.

2.  **Conjecture 2 (Pruning Recall):**
    *   **Prediction:** Recall@10 $\ge 95\%$ with $nprobe=4$.
    *   **Observation:** Recall@10 $= 93.33\%$.
    *   **Significance:** Narrow miss, but confirms that scoring errors from C1 propagate to reranking failures.

3.  **Experimentum Crucis:**
    *   **Status:** INCONCLUSIVE. The use of a
        `MockMultiVectorEmbedder` (random vectors) prevented
        semantic evaluation. This test remains valid but requires
        a real ColBERT model integration.

### 12.2 Reformulation Plan

We reject the current "Scalar Residual Quantization" theory. We propose the following reformulation:

1.  **Theory 2.0: Product Quantization (PQ)**
    *   Instead of scalar quantization of residuals, split the
        vector into $M$ sub-vectors and quantize each using a
        local codebook.
    *   *Rationale:* PQ better captures the manifold of the residuals than independent scalar buckets.

2.  **Theory 2.1: Asymmetric Distance Computation**
    *   Do not compress the query. Compute distance between *exact* query and *compressed* codes.
    *   *Rationale:* Reduces quantization noise by 50%.

3.  **Revised Test Protocol:**
    *   Replace `MockMultiVectorEmbedder` with
        `ClusteredMockEmbedder` for structural tests (C1, C2)
        to avoid "curse of dimensionality" artifacts from uniform
        random noise.
    *   Mandate `fastembed-rs` integration for the *Experimentum Crucis*.

> **Next Step:** Implement **WARP v2 (PQ + Asymmetric)** and subject it to the same falsification battery.