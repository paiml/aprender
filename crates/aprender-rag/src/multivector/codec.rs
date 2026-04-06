//! Residual quantization codec for WARP
//!
//! This module implements the residual quantization scheme used to compress
//! token embeddings in the WARP algorithm. Each vector is decomposed into:
//! - A centroid (learned via k-means)
//! - A residual (difference from centroid), quantized to 2-4 bits per dimension
//!
//! The codec enables efficient scoring without full decompression by using
//! precomputed centroid scores and bucket weights.

use crate::multivector::types::WarpIndexConfig;
use crate::Result;
use serde::{Deserialize, Serialize};

/// Residual quantization codec for compressing token embeddings.
///
/// The codec learns centroids via k-means clustering, then quantizes the
/// residuals (v - centroid) to a small number of bits per dimension.
///
/// # Compression Process
///
/// 1. Find nearest centroid for input vector
/// 2. Compute residual = vector - centroid
/// 3. Quantize each dimension to `nbits` using learned bucket boundaries
/// 4. Pack quantized values into bytes
///
/// # Scoring
///
/// Score computation avoids full decompression:
/// ```text
/// q · v ≈ q · c + Σ_d q[d] × bucket_weight[d, code[d]]
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResidualCodec {
    /// Centroid vectors: [num_centroids × dim], flattened
    centroids: Vec<f32>,
    /// Number of centroids
    num_centroids: usize,
    /// Token dimension
    dim: usize,
    /// Quantization bucket boundaries per dimension: [dim × (num_buckets - 1)]
    bucket_cutoffs: Vec<f32>,
    /// Reconstruction weights per bucket: [dim × num_buckets]
    bucket_weights: Vec<f32>,
    /// Bits per dimension (2 or 4)
    nbits: u8,
}

impl ResidualCodec {
    /// Train a codec from sample embeddings.
    ///
    /// # Arguments
    ///
    /// * `embeddings` - Flattened sample embeddings [n × dim]
    /// * `dim` - Embedding dimension
    /// * `num_centroids` - Number of k-means centroids
    /// * `nbits` - Bits per dimension (2 or 4)
    /// * `iterations` - K-means iterations
    ///
    /// # Errors
    ///
    /// Returns an error if training data is insufficient or parameters invalid.
    pub fn train(
        embeddings: &[f32],
        dim: usize,
        num_centroids: usize,
        nbits: u8,
        iterations: usize,
    ) -> Result<Self> {
        if nbits != 2 && nbits != 4 {
            return Err(crate::Error::InvalidInput("nbits must be 2 or 4".to_string()));
        }

        if dim == 0 {
            return Err(crate::Error::InvalidInput("dim must be > 0".to_string()));
        }
        let n = embeddings.len() / dim;
        if n < num_centroids {
            return Err(crate::Error::InvalidInput(format!(
                "Insufficient training data: {n} samples for {num_centroids} centroids"
            )));
        }

        // Contract: embedding-algebra-v1.yaml precondition (pv codegen)
        contract_pre_embedding_lookup!(embeddings);

        // Step 1: K-means clustering to find centroids
        let centroids = Self::kmeans_clustering(embeddings, dim, num_centroids, iterations);

        // Step 2: Compute residuals for all training points
        let residuals = Self::compute_all_residuals(embeddings, dim, &centroids, num_centroids);

        // Step 3: Learn quantization boundaries from residual distribution
        let (bucket_cutoffs, bucket_weights) =
            Self::learn_quantization_params(&residuals, dim, nbits);

        Ok(Self { centroids, num_centroids, dim, bucket_cutoffs, bucket_weights, nbits })
    }

    /// Create a codec with pre-trained parameters.
    ///
    /// # Panics
    ///
    /// Panics if `dim == 0` (poka-yoke: division-by-zero guard).
    #[must_use]
    pub fn with_params(
        centroids: Vec<f32>,
        num_centroids: usize,
        dim: usize,
        bucket_cutoffs: Vec<f32>,
        bucket_weights: Vec<f32>,
        nbits: u8,
    ) -> Self {
        assert!(dim > 0, "dim must be > 0: division by zero in centroid/residual arithmetic");
        Self { centroids, num_centroids, dim, bucket_cutoffs, bucket_weights, nbits }
    }

    /// Get the number of centroids.
    #[must_use]
    pub fn num_centroids(&self) -> usize {
        self.num_centroids
    }

    /// Get the embedding dimension.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Get bits per dimension.
    #[must_use]
    pub fn nbits(&self) -> u8 {
        self.nbits
    }

    /// Get the packed residual size in bytes.
    #[must_use]
    pub fn packed_size(&self) -> usize {
        (self.dim * self.nbits as usize + 7) / 8
    }

    /// Get centroid slice by ID.
    #[must_use]
    pub fn centroid(&self, id: usize) -> &[f32] {
        let start = id * self.dim;
        &self.centroids[start..start + self.dim]
    }

    /// Get all centroids as a flat slice.
    #[must_use]
    pub fn centroids(&self) -> &[f32] {
        &self.centroids
    }

    /// Find the nearest centroid for a vector.
    #[must_use]
    pub fn find_nearest_centroid(&self, embedding: &[f32]) -> usize {
        // Contract: configuration-v1.yaml precondition (pv codegen)
        contract_pre_configuration!(embedding);
        let mut best_id = 0;
        let mut best_dist = f32::MAX;

        for c in 0..self.num_centroids {
            let centroid = self.centroid(c);
            let dist = Self::squared_distance(embedding, centroid);
            if dist < best_dist {
                best_dist = dist;
                best_id = c;
            }
        }

        best_id
    }

    /// Compress an embedding to (centroid_id, packed_residual).
    #[must_use]
    pub fn compress(&self, embedding: &[f32]) -> (usize, Vec<u8>) {
        // Contract: embedding-algebra-v1.yaml precondition (pv codegen)
        contract_pre_embedding_lookup!(embedding);
        // Find nearest centroid
        let centroid_id = self.find_nearest_centroid(embedding);
        let centroid = self.centroid(centroid_id);

        // Compute residual
        let residual: Vec<f32> =
            embedding.iter().zip(centroid.iter()).map(|(e, c)| e - c).collect();

        // Quantize residual
        let codes = self.quantize_residual(&residual);

        // Pack codes into bytes
        let packed = self.pack_codes(&codes);

        (centroid_id, packed)
    }

    /// Compute score between query token and compressed document token.
    ///
    /// score ≈ q · d = q · c + q · r
    ///
    /// # Arguments
    ///
    /// * `query_token` - Query embedding
    /// * `centroid_id` - Assigned centroid
    /// * `centroid_score` - Precomputed q · c
    /// * `packed_residual` - Packed quantized residual
    #[must_use]
    pub fn decompress_score(
        &self,
        query_token: &[f32],
        centroid_id: usize,
        centroid_score: f32,
        packed_residual: &[u8],
    ) -> f32 {
        let _ = centroid_id; // Centroid info already in centroid_score

        // Unpack residual codes
        let codes = self.unpack_codes(packed_residual);

        // Compute q · r using bucket weights
        let num_buckets = 1usize << self.nbits;
        let residual_score: f32 = codes
            .iter()
            .enumerate()
            .map(|(d, &code)| {
                let weight_idx = d * num_buckets + code as usize;
                query_token[d] * self.bucket_weights[weight_idx]
            })
            .sum();

        centroid_score + residual_score
    }

    /// Compute dot product between query and centroid.
    #[must_use]
    pub fn centroid_score(&self, query_token: &[f32], centroid_id: usize) -> f32 {
        let centroid = self.centroid(centroid_id);
        Self::dot_product(query_token, centroid)
    }

    /// Quantize a residual vector to codes.
    fn quantize_residual(&self, residual: &[f32]) -> Vec<u8> {
        let num_buckets = 1usize << self.nbits;

        residual
            .iter()
            .enumerate()
            .map(|(d, &value)| {
                // Binary search for bucket
                let cutoff_start = d * (num_buckets - 1);
                let cutoffs = &self.bucket_cutoffs[cutoff_start..cutoff_start + num_buckets - 1];

                // Find first cutoff >= value
                cutoffs.iter().position(|&c| value < c).unwrap_or(num_buckets - 1) as u8
            })
            .collect()
    }

    /// Pack quantization codes into bytes.
    fn pack_codes(&self, codes: &[u8]) -> Vec<u8> {
        match self.nbits {
            2 => {
                // Pack 4 codes per byte
                codes
                    .chunks(4)
                    .map(|chunk| {
                        let mut byte = 0u8;
                        for (i, &code) in chunk.iter().enumerate() {
                            byte |= (code & 0x03) << (i * 2);
                        }
                        byte
                    })
                    .collect()
            }
            4 => {
                // Pack 2 codes per byte
                codes
                    .chunks(2)
                    .map(|chunk| {
                        let low = chunk.first().copied().unwrap_or(0) & 0x0F;
                        let high = chunk.get(1).copied().unwrap_or(0) & 0x0F;
                        low | (high << 4)
                    })
                    .collect()
            }
            _ => panic!("Unsupported nbits: {}", self.nbits),
        }
    }

    /// Unpack codes from packed bytes.
    fn unpack_codes(&self, packed: &[u8]) -> Vec<u8> {
        match self.nbits {
            2 => packed
                .iter()
                .flat_map(|&byte| (0..4).map(move |i| (byte >> (i * 2)) & 0x03))
                .take(self.dim)
                .collect(),
            4 => packed
                .iter()
                .flat_map(|&byte| vec![byte & 0x0F, (byte >> 4) & 0x0F])
                .take(self.dim)
                .collect(),
            _ => panic!("Unsupported nbits: {}", self.nbits),
        }
    }

    // ============ K-means Implementation ============

    /// K-means clustering with k-means++ initialization.
    fn kmeans_clustering(embeddings: &[f32], dim: usize, k: usize, iterations: usize) -> Vec<f32> {
        let n = embeddings.len() / dim;

        // K-means++ initialization
        let mut centroids = Self::kmeans_plus_plus_init(embeddings, dim, k);
        let mut assignments = vec![0usize; n];

        for _ in 0..iterations {
            // Assign points to nearest centroid
            for i in 0..n {
                let point = &embeddings[i * dim..(i + 1) * dim];
                let mut best_dist = f32::MAX;
                let mut best_c = 0;

                for c in 0..k {
                    let centroid = &centroids[c * dim..(c + 1) * dim];
                    let dist = Self::squared_distance(point, centroid);
                    if dist < best_dist {
                        best_dist = dist;
                        best_c = c;
                    }
                }
                assignments[i] = best_c;
            }

            // Update centroids as mean of assigned points
            let mut new_centroids = vec![0.0f32; k * dim];
            let mut counts = vec![0usize; k];

            for i in 0..n {
                let c = assignments[i];
                counts[c] += 1;
                let point = &embeddings[i * dim..(i + 1) * dim];
                for d in 0..dim {
                    new_centroids[c * dim + d] += point[d];
                }
            }

            for c in 0..k {
                if counts[c] > 0 {
                    for d in 0..dim {
                        new_centroids[c * dim + d] /= counts[c] as f32;
                    }
                } else {
                    // Keep old centroid if no points assigned
                    for d in 0..dim {
                        new_centroids[c * dim + d] = centroids[c * dim + d];
                    }
                }
            }

            centroids = new_centroids;
        }

        centroids
    }

    /// K-means++ initialization.
    fn kmeans_plus_plus_init(embeddings: &[f32], dim: usize, k: usize) -> Vec<f32> {
        let n = embeddings.len() / dim;
        let mut centroids = Vec::with_capacity(k * dim);
        let mut rng_state = 42u64; // Simple deterministic RNG

        // Choose first centroid uniformly at random
        let first_idx = Self::simple_random(&mut rng_state, n);
        centroids.extend_from_slice(&embeddings[first_idx * dim..(first_idx + 1) * dim]);

        let mut distances = vec![f32::MAX; n];

        for _ in 1..k {
            let num_centroids = centroids.len() / dim;

            // Update distances to nearest centroid
            for i in 0..n {
                let point = &embeddings[i * dim..(i + 1) * dim];
                let centroid = &centroids[(num_centroids - 1) * dim..num_centroids * dim];
                let dist = Self::squared_distance(point, centroid);
                distances[i] = distances[i].min(dist);
            }

            // Choose next centroid with probability proportional to D²
            let total: f32 = distances.iter().sum();
            if total <= 0.0 {
                // All points are centroids already, pick random
                let idx = Self::simple_random(&mut rng_state, n);
                centroids.extend_from_slice(&embeddings[idx * dim..(idx + 1) * dim]);
                continue;
            }

            let threshold = Self::simple_random_f32(&mut rng_state) * total;
            let mut cumsum = 0.0f32;
            let mut chosen = 0;

            for (i, &d) in distances.iter().enumerate() {
                cumsum += d;
                if cumsum >= threshold {
                    chosen = i;
                    break;
                }
            }

            centroids.extend_from_slice(&embeddings[chosen * dim..(chosen + 1) * dim]);
        }

        centroids
    }

    /// Compute residuals for all embeddings.
    fn compute_all_residuals(
        embeddings: &[f32],
        dim: usize,
        centroids: &[f32],
        num_centroids: usize,
    ) -> Vec<f32> {
        let n = embeddings.len() / dim;
        let mut residuals = Vec::with_capacity(n * dim);

        for i in 0..n {
            let point = &embeddings[i * dim..(i + 1) * dim];

            // Find nearest centroid
            let mut best_c = 0;
            let mut best_dist = f32::MAX;
            for c in 0..num_centroids {
                let centroid = &centroids[c * dim..(c + 1) * dim];
                let dist = Self::squared_distance(point, centroid);
                if dist < best_dist {
                    best_dist = dist;
                    best_c = c;
                }
            }

            // Compute residual
            let centroid = &centroids[best_c * dim..(best_c + 1) * dim];
            for d in 0..dim {
                residuals.push(point[d] - centroid[d]);
            }
        }

        residuals
    }

    /// Learn quantization bucket boundaries and weights from residuals.
    fn learn_quantization_params(residuals: &[f32], dim: usize, nbits: u8) -> (Vec<f32>, Vec<f32>) {
        let num_buckets = 1usize << nbits;
        let n = residuals.len() / dim;

        let mut cutoffs = Vec::with_capacity(dim * (num_buckets - 1));
        let mut weights = Vec::with_capacity(dim * num_buckets);

        for d in 0..dim {
            // Collect residual values for dimension d
            let mut values: Vec<f32> = (0..n).map(|i| residuals[i * dim + d]).collect();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            // Quantile-based boundaries for equal-frequency buckets
            for b in 1..num_buckets {
                let quantile_idx = (b * n) / num_buckets;
                cutoffs.push(values[quantile_idx.min(n - 1)]);
            }

            // Bucket weights = mean value in each bucket
            for b in 0..num_buckets {
                let start = (b * n) / num_buckets;
                let end = ((b + 1) * n) / num_buckets;
                let end = end.max(start + 1).min(n);

                let sum: f32 = values[start..end].iter().sum();
                let mean = sum / (end - start) as f32;
                weights.push(mean);
            }
        }

        (cutoffs, weights)
    }

    // ============ Math Utilities ============

    fn squared_distance(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
    }

    fn dot_product(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    fn simple_random(state: &mut u64, max: usize) -> usize {
        *state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        ((*state >> 33) as usize) % max
    }

    fn simple_random_f32(state: &mut u64) -> f32 {
        *state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        ((*state >> 33) as f32) / (u32::MAX as f32)
    }
}

/// Builder for creating a `ResidualCodec` from a `WarpIndexConfig`.
pub struct ResidualCodecBuilder {
    config: WarpIndexConfig,
}

impl ResidualCodecBuilder {
    /// Create a new builder from config.
    #[must_use]
    pub fn new(config: WarpIndexConfig) -> Self {
        Self { config }
    }

    /// Train the codec from sample embeddings.
    pub fn train(&self, embeddings: &[f32]) -> Result<ResidualCodec> {
        // Contract: embedding-algebra-v1.yaml precondition (pv codegen)
        contract_pre_embedding_lookup!(embeddings);
        ResidualCodec::train(
            embeddings,
            self.config.token_dim,
            self.config.num_centroids,
            self.config.nbits,
            self.config.kmeans_iterations,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_test_embeddings(n: usize, dim: usize) -> Vec<f32> {
        let mut embeddings = Vec::with_capacity(n * dim);
        let mut rng_state = 12345u64;

        for _ in 0..(n * dim) {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let val = ((rng_state >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0;
            embeddings.push(val);
        }

        embeddings
    }

    // ============ Basic Codec Tests ============

    #[test]
    fn test_codec_train_2bit() {
        let embeddings = generate_test_embeddings(1000, 32);
        let codec = ResidualCodec::train(&embeddings, 32, 16, 2, 5).unwrap();

        assert_eq!(codec.num_centroids(), 16);
        assert_eq!(codec.dim(), 32);
        assert_eq!(codec.nbits(), 2);
    }

    #[test]
    fn test_codec_train_4bit() {
        let embeddings = generate_test_embeddings(1000, 32);
        let codec = ResidualCodec::train(&embeddings, 32, 16, 4, 5).unwrap();

        assert_eq!(codec.nbits(), 4);
    }

    #[test]
    fn test_codec_train_insufficient_data() {
        let embeddings = generate_test_embeddings(5, 32);
        let result = ResidualCodec::train(&embeddings, 32, 16, 2, 5);

        assert!(result.is_err());
    }

    #[test]
    fn test_codec_train_invalid_nbits() {
        let embeddings = generate_test_embeddings(100, 32);
        let result = ResidualCodec::train(&embeddings, 32, 16, 3, 5);

        assert!(result.is_err());
    }

    /// Regression test for paiml/trueno-rag#15: train() rejects dim=0.
    #[test]
    fn test_codec_train_dim_zero() {
        let result = ResidualCodec::train(&[], 0, 4, 2, 3);
        assert!(result.is_err());
    }

    /// Regression test for paiml/trueno-rag#15: with_params() rejects dim=0.
    #[test]
    #[should_panic(expected = "dim must be > 0")]
    fn test_codec_with_params_dim_zero() {
        let _ = ResidualCodec::with_params(vec![], 0, 0, vec![], vec![], 2);
    }

    // ============ Compression Tests ============

    #[test]
    fn test_codec_compress() {
        let embeddings = generate_test_embeddings(500, 32);
        let codec = ResidualCodec::train(&embeddings, 32, 16, 2, 5).unwrap();

        let test_vec = &embeddings[0..32];
        let (centroid_id, packed) = codec.compress(test_vec);

        assert!(centroid_id < 16);
        assert_eq!(packed.len(), codec.packed_size());
    }

    #[test]
    fn test_codec_packed_size_2bit() {
        let embeddings = generate_test_embeddings(500, 128);
        let codec = ResidualCodec::train(&embeddings, 128, 16, 2, 5).unwrap();

        // 128 dims × 2 bits = 256 bits = 32 bytes
        assert_eq!(codec.packed_size(), 32);
    }

    #[test]
    fn test_codec_packed_size_4bit() {
        let embeddings = generate_test_embeddings(500, 128);
        let codec = ResidualCodec::train(&embeddings, 128, 16, 4, 5).unwrap();

        // 128 dims × 4 bits = 512 bits = 64 bytes
        assert_eq!(codec.packed_size(), 64);
    }

    // ============ Pack/Unpack Tests ============

    #[test]
    fn test_pack_unpack_2bit() {
        let embeddings = generate_test_embeddings(500, 8);
        let codec = ResidualCodec::train(&embeddings, 8, 16, 2, 5).unwrap();

        let codes: Vec<u8> = vec![0, 1, 2, 3, 0, 1, 2, 3];
        let packed = codec.pack_codes(&codes);
        let unpacked = codec.unpack_codes(&packed);

        assert_eq!(codes, unpacked);
    }

    #[test]
    fn test_pack_unpack_4bit() {
        let embeddings = generate_test_embeddings(500, 8);
        let codec = ResidualCodec::train(&embeddings, 8, 16, 4, 5).unwrap();

        let codes: Vec<u8> = vec![0, 5, 10, 15, 1, 6, 11, 14];
        let packed = codec.pack_codes(&codes);
        let unpacked = codec.unpack_codes(&packed);

        assert_eq!(codes, unpacked);
    }

    // ============ Scoring Tests ============

    #[test]
    fn test_decompress_score() {
        let embeddings = generate_test_embeddings(500, 32);
        let codec = ResidualCodec::train(&embeddings, 32, 16, 2, 5).unwrap();

        let query = &embeddings[0..32];
        let doc = &embeddings[32..64];

        // Compress document
        let (centroid_id, packed) = codec.compress(doc);

        // Compute centroid score
        let centroid_score = codec.centroid_score(query, centroid_id);

        // Compute approximate score
        let approx_score = codec.decompress_score(query, centroid_id, centroid_score, &packed);

        // Compute exact score
        let exact_score: f32 = query.iter().zip(doc.iter()).map(|(q, d)| q * d).sum();

        // Approximate score should be close to exact (within reasonable tolerance)
        let error = (approx_score - exact_score).abs();
        assert!(
            error < exact_score.abs() * 0.5 + 1.0,
            "Error too large: approx={}, exact={}, error={}",
            approx_score,
            exact_score,
            error
        );
    }

    #[test]
    fn test_centroid_score() {
        let embeddings = generate_test_embeddings(500, 32);
        let codec = ResidualCodec::train(&embeddings, 32, 16, 2, 5).unwrap();

        let query = &embeddings[0..32];
        let centroid = codec.centroid(0);

        let expected: f32 = query.iter().zip(centroid.iter()).map(|(q, c)| q * c).sum();
        let actual = codec.centroid_score(query, 0);

        assert!((expected - actual).abs() < 1e-6);
    }

    // ============ K-means Tests ============

    #[test]
    fn test_find_nearest_centroid() {
        let embeddings = generate_test_embeddings(500, 32);
        let codec = ResidualCodec::train(&embeddings, 32, 16, 2, 5).unwrap();

        // A centroid should be nearest to itself
        let centroid_0 = codec.centroid(0).to_vec();
        let nearest = codec.find_nearest_centroid(&centroid_0);
        assert_eq!(nearest, 0);
    }

    // ============ Builder Tests ============

    #[test]
    fn test_codec_builder() {
        let config = WarpIndexConfig::new(2, 16, 32).with_kmeans_iterations(5);
        let builder = ResidualCodecBuilder::new(config);

        let embeddings = generate_test_embeddings(500, 32);
        let codec = builder.train(&embeddings).unwrap();

        assert_eq!(codec.num_centroids(), 16);
        assert_eq!(codec.dim(), 32);
    }

    // ============ Serialization Tests ============

    #[test]
    fn test_codec_serialization() {
        let embeddings = generate_test_embeddings(500, 16);
        let codec = ResidualCodec::train(&embeddings, 16, 8, 2, 5).unwrap();

        let json = serde_json::to_string(&codec).unwrap();
        let deserialized: ResidualCodec = serde_json::from_str(&json).unwrap();

        assert_eq!(codec.num_centroids(), deserialized.num_centroids());
        assert_eq!(codec.dim(), deserialized.dim());
        assert_eq!(codec.nbits(), deserialized.nbits());
    }

    // ============ Property-Based Tests ============

    use proptest::prelude::*;

    proptest! {
        #[test]
        fn prop_compress_produces_valid_centroid_id(
            seed in 0u64..1000
        ) {
            let mut embeddings = Vec::with_capacity(200 * 16);
            let mut rng_state = seed;
            for _ in 0..(200 * 16) {
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                embeddings.push(((rng_state >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0);
            }

            let codec = ResidualCodec::train(&embeddings, 16, 8, 2, 3).unwrap();
            let test_vec = &embeddings[0..16];
            let (centroid_id, _) = codec.compress(test_vec);

            prop_assert!(centroid_id < 8);
        }

        #[test]
        fn prop_packed_size_matches_config(
            nbits in prop::sample::select(vec![2u8, 4]),
            dim in 8usize..64
        ) {
            let num_samples = 100 * dim;
            let embeddings = generate_test_embeddings(num_samples / dim, dim);

            if let Ok(codec) = ResidualCodec::train(&embeddings, dim, 8, nbits, 3) {
                let expected_size = (dim * nbits as usize + 7) / 8;
                prop_assert_eq!(codec.packed_size(), expected_size);
            }
        }
    }
}
