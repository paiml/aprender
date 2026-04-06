//! Active Learning with Thompson Sampling
//!
//! Implements active learning for dynamic sampling strategy adjustment
//! based on oracle feedback using Thompson Sampling on code clusters.
//!
//! # Architecture
//!
//! ```text
//! Code → Embedding → Clustering → Thompson Sampling → Sample Selection
//!                                        ↑
//!                                  Oracle Feedback
//! ```
//!
//! # Reference
//! - VER-052: Active Learning - Thompson Sampling exploration
//! - Spieker et al. (2017): "Reinforcement Learning for Automatic Test Case Prioritization"

use rand::Rng;
use rand_distr::{Beta, Distribution};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Code embedding for clustering
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CodeEmbedding {
    /// Feature vector (n-gram counts, normalized)
    pub features: Vec<f32>,
    /// Dimensionality
    pub dim: usize,
}

impl CodeEmbedding {
    /// Create empty embedding
    #[must_use]
    pub fn new(dim: usize) -> Self {
        Self {
            features: vec![0.0; dim],
            dim,
        }
    }

    /// Create from feature vector
    #[must_use]
    pub fn from_vec(features: Vec<f32>) -> Self {
        let dim = features.len();
        Self { features, dim }
    }

    /// L2 norm of embedding
    #[must_use]
    pub fn norm(&self) -> f32 {
        self.features.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    /// Normalize to unit vector
    pub fn normalize(&mut self) {
        let norm = self.norm();
        if norm > 0.0 {
            for x in &mut self.features {
                *x /= norm;
            }
        }
    }

    /// Cosine similarity with another embedding
    #[must_use]
    pub fn cosine_similarity(&self, other: &Self) -> f32 {
        if self.dim != other.dim {
            return 0.0;
        }

        let dot: f32 = self
            .features
            .iter()
            .zip(&other.features)
            .map(|(a, b)| a * b)
            .sum();

        let norm_a = self.norm();
        let norm_b = other.norm();

        if norm_a > 0.0 && norm_b > 0.0 {
            dot / (norm_a * norm_b)
        } else {
            0.0
        }
    }

    /// Euclidean distance to another embedding
    #[must_use]
    pub fn euclidean_distance(&self, other: &Self) -> f32 {
        if self.dim != other.dim {
            return f32::MAX;
        }

        self.features
            .iter()
            .zip(&other.features)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt()
    }
}

/// Simple code embedder using n-gram features
#[derive(Debug, Clone)]
pub struct CodeEmbedder {
    /// N-gram size
    n: usize,
    /// Vocabulary size (hash buckets)
    vocab_size: usize,
}

impl Default for CodeEmbedder {
    fn default() -> Self {
        Self::new(3, 128)
    }
}

impl CodeEmbedder {
    /// Create embedder with n-gram size and vocabulary size
    #[must_use]
    pub fn new(n: usize, vocab_size: usize) -> Self {
        Self { n, vocab_size }
    }

    /// Embed code string to vector
    #[must_use]
    pub fn embed(&self, code: &str) -> CodeEmbedding {
        let mut features = vec![0.0f32; self.vocab_size];

        // Extract character n-grams
        let chars: Vec<char> = code.chars().collect();
        if chars.len() >= self.n {
            for window in chars.windows(self.n) {
                let hash = self.hash_ngram(window);
                features[hash] += 1.0;
            }
        }

        // Also add word unigrams
        for word in code.split_whitespace() {
            let hash = self.hash_word(word);
            features[hash] += 1.0;
        }

        let mut embedding = CodeEmbedding::from_vec(features);
        embedding.normalize();
        embedding
    }

    fn hash_ngram(&self, chars: &[char]) -> usize {
        let mut hash = 0usize;
        for (i, &c) in chars.iter().enumerate() {
            hash = hash.wrapping_add((c as usize).wrapping_mul(31_usize.wrapping_pow(i as u32)));
        }
        hash % self.vocab_size
    }

    fn hash_word(&self, word: &str) -> usize {
        let mut hash = 0usize;
        for (i, c) in word.chars().enumerate() {
            hash = hash.wrapping_add((c as usize).wrapping_mul(37_usize.wrapping_pow(i as u32)));
        }
        hash % self.vocab_size
    }
}

/// K-means cluster
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Cluster {
    /// Cluster ID
    pub id: usize,
    /// Centroid
    pub centroid: CodeEmbedding,
    /// Number of samples in cluster
    pub size: usize,
    /// Sum of distances to centroid (for silhouette calculation)
    pub intra_distance: f32,
}

impl Cluster {
    /// Create new cluster
    #[must_use]
    pub fn new(id: usize, centroid: CodeEmbedding) -> Self {
        Self {
            id,
            centroid,
            size: 0,
            intra_distance: 0.0,
        }
    }

    /// Average intra-cluster distance
    #[must_use]
    pub fn avg_intra_distance(&self) -> f32 {
        if self.size > 0 {
            self.intra_distance / self.size as f32
        } else {
            0.0
        }
    }
}

/// K-means clustering result
#[derive(Debug, Clone)]
pub struct ClusteringResult {
    /// Clusters
    pub clusters: Vec<Cluster>,
    /// Assignment of each sample to cluster
    pub assignments: Vec<usize>,
    /// Silhouette score (-1 to 1, higher = better)
    pub silhouette_score: f32,
    /// Number of iterations
    pub iterations: usize,
}

/// Simple K-means clustering
#[derive(Debug, Clone)]
pub struct KMeansClustering {
    /// Number of clusters
    k: usize,
    /// Max iterations
    max_iter: usize,
    /// Random seed
    seed: u64,
}

impl Default for KMeansClustering {
    fn default() -> Self {
        Self::new(5)
    }
}

impl KMeansClustering {
    /// Create with k clusters
    #[must_use]
    pub fn new(k: usize) -> Self {
        Self {
            k,
            max_iter: 100,
            seed: 42,
        }
    }

    /// Set max iterations
    #[must_use]
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set random seed
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Fit clusters to embeddings
    pub fn fit(&self, embeddings: &[CodeEmbedding]) -> ClusteringResult {
        if embeddings.is_empty() {
            return ClusteringResult {
                clusters: vec![],
                assignments: vec![],
                silhouette_score: 0.0,
                iterations: 0,
            };
        }

        let dim = embeddings[0].dim;
        let actual_k = self.k.min(embeddings.len());

        // Initialize centroids (k-means++ style)
        let mut rng = rand::rng();
        let mut centroids = self.init_centroids(embeddings, actual_k, &mut rng);

        let mut assignments = vec![0usize; embeddings.len()];
        let mut iterations = 0;

        for iter in 0..self.max_iter {
            iterations = iter + 1;

            // Assign samples to nearest centroid
            let mut changed = false;
            for (i, emb) in embeddings.iter().enumerate() {
                let nearest = self.find_nearest_centroid(emb, &centroids);
                if assignments[i] != nearest {
                    assignments[i] = nearest;
                    changed = true;
                }
            }

            // Update centroids
            centroids = self.update_centroids(embeddings, &assignments, actual_k, dim);

            if !changed {
                break;
            }
        }

        // Build cluster objects
        let mut clusters: Vec<Cluster> = centroids
            .into_iter()
            .enumerate()
            .map(|(id, centroid)| Cluster::new(id, centroid))
            .collect();

        // Calculate cluster sizes and intra-distances
        for (i, &cluster_id) in assignments.iter().enumerate() {
            if cluster_id < clusters.len() {
                clusters[cluster_id].size += 1;
                clusters[cluster_id].intra_distance +=
                    embeddings[i].euclidean_distance(&clusters[cluster_id].centroid);
            }
        }

        // Calculate silhouette score
        let silhouette_score = self.calculate_silhouette(embeddings, &assignments, &clusters);

        ClusteringResult {
            clusters,
            assignments,
            silhouette_score,
            iterations,
        }
    }

    fn init_centroids<R: Rng>(
        &self,
        embeddings: &[CodeEmbedding],
        k: usize,
        rng: &mut R,
    ) -> Vec<CodeEmbedding> {
        if embeddings.is_empty() || k == 0 {
            return vec![];
        }

        let mut centroids = Vec::with_capacity(k);

        // First centroid: random
        let first_idx = rng.random_range(0..embeddings.len());
        centroids.push(embeddings[first_idx].clone());

        // K-means++: choose remaining centroids proportional to squared distance
        for _ in 1..k {
            let distances: Vec<f32> = embeddings
                .iter()
                .map(|emb| {
                    centroids
                        .iter()
                        .map(|c| emb.euclidean_distance(c))
                        .fold(f32::MAX, f32::min)
                        .powi(2)
                })
                .collect();

            let total: f32 = distances.iter().sum();
            if total <= 0.0 {
                break;
            }

            let threshold = rng.random::<f32>() * total;
            let mut cumsum = 0.0;
            for (i, &d) in distances.iter().enumerate() {
                cumsum += d;
                if cumsum >= threshold {
                    centroids.push(embeddings[i].clone());
                    break;
                }
            }
        }

        centroids
    }

    fn find_nearest_centroid(&self, emb: &CodeEmbedding, centroids: &[CodeEmbedding]) -> usize {
        centroids
            .iter()
            .enumerate()
            .map(|(i, c)| (i, emb.euclidean_distance(c)))
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(0, |(i, _)| i)
    }

    fn update_centroids(
        &self,
        embeddings: &[CodeEmbedding],
        assignments: &[usize],
        k: usize,
        dim: usize,
    ) -> Vec<CodeEmbedding> {
        let mut sums: Vec<Vec<f32>> = vec![vec![0.0; dim]; k];
        let mut counts = vec![0usize; k];

        for (i, &cluster_id) in assignments.iter().enumerate() {
            if cluster_id < k {
                counts[cluster_id] += 1;
                for (j, &val) in embeddings[i].features.iter().enumerate() {
                    if j < dim {
                        sums[cluster_id][j] += val;
                    }
                }
            }
        }

        sums.into_iter()
            .zip(counts)
            .map(|(sum, count)| {
                if count > 0 {
                    let features: Vec<f32> = sum.into_iter().map(|s| s / count as f32).collect();
                    CodeEmbedding::from_vec(features)
                } else {
                    CodeEmbedding::new(dim)
                }
            })
            .collect()
    }

    fn calculate_silhouette(
        &self,
        embeddings: &[CodeEmbedding],
        assignments: &[usize],
        clusters: &[Cluster],
    ) -> f32 {
        if embeddings.len() <= 1 || clusters.len() <= 1 {
            return 0.0;
        }

        let mut total_score = 0.0;
        let mut count = 0;

        for (i, emb) in embeddings.iter().enumerate() {
            let cluster_id = assignments[i];
            if cluster_id >= clusters.len() {
                continue;
            }

            // a(i): average distance to points in same cluster
            let a = clusters[cluster_id].avg_intra_distance();

            // b(i): minimum average distance to points in other clusters
            let b = clusters
                .iter()
                .filter(|c| c.id != cluster_id)
                .map(|c| emb.euclidean_distance(&c.centroid))
                .fold(f32::MAX, f32::min);

            if b < f32::MAX {
                let max_ab = a.max(b);
                if max_ab > 0.0 {
                    total_score += (b - a) / max_ab;
                    count += 1;
                }
            }
        }

        if count > 0 {
            total_score / count as f32
        } else {
            0.0
        }
    }
}

/// Active learner using Thompson Sampling on clusters
#[derive(Debug)]
pub struct ActiveLearner {
    /// Code embedder
    embedder: CodeEmbedder,
    /// Clustering algorithm
    clustering: KMeansClustering,
    /// Current clustering result
    cluster_result: Option<ClusteringResult>,
    /// Success counts per cluster (alpha for Beta dist)
    success_counts: HashMap<usize, f64>,
    /// Failure counts per cluster (beta for Beta dist)
    failure_counts: HashMap<usize, f64>,
    /// Total samples
    total_samples: usize,
    /// Exploration rate
    exploration_rate: f64,
}

impl Default for ActiveLearner {
    fn default() -> Self {
        Self::new(5)
    }
}

impl ActiveLearner {
    /// Create active learner with k clusters
    #[must_use]
    pub fn new(k: usize) -> Self {
        Self {
            embedder: CodeEmbedder::default(),
            clustering: KMeansClustering::new(k),
            cluster_result: None,
            success_counts: HashMap::new(),
            failure_counts: HashMap::new(),
            total_samples: 0,
            exploration_rate: 0.1,
        }
    }

    /// Create with custom embedder
    #[must_use]
    pub fn with_embedder(mut self, embedder: CodeEmbedder) -> Self {
        self.embedder = embedder;
        self
    }

    /// Set exploration rate
    #[must_use]
    pub fn with_exploration_rate(mut self, rate: f64) -> Self {
        self.exploration_rate = rate.clamp(0.0, 1.0);
        self
    }

    /// Fit clusters on code samples
    pub fn fit(&mut self, codes: &[&str]) {
        let embeddings: Vec<CodeEmbedding> = codes.iter().map(|c| self.embedder.embed(c)).collect();

        self.cluster_result = Some(self.clustering.fit(&embeddings));
    }

    /// Get cluster for a code sample
    #[must_use]
    pub fn get_cluster(&self, code: &str) -> Option<usize> {
        let embedding = self.embedder.embed(code);
        self.cluster_result.as_ref().map(|result| {
            result
                .clusters
                .iter()
                .enumerate()
                .map(|(i, c)| (i, embedding.euclidean_distance(&c.centroid)))
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map_or(0, |(i, _)| i)
        })
    }

    /// Sample cluster using Thompson Sampling
    ///
    /// Returns cluster ID with high expected value (exploration vs exploitation)
    ///
    /// # Panics
    ///
    /// Panics if Beta distribution creation fails with default parameters (1.0, 1.0),
    /// which should never happen for valid alpha/beta values.
    pub fn sample_cluster(&self) -> Option<usize> {
        let result = self.cluster_result.as_ref()?;
        if result.clusters.is_empty() {
            return None;
        }

        let mut rng = rand::rng();

        // Sample from Beta distribution for each cluster
        let scores: Vec<(usize, f64)> = result
            .clusters
            .iter()
            .map(|c| {
                // Get counts with prior (Beta(1,1) = uniform)
                let alpha = self.failure_counts.get(&c.id).copied().unwrap_or(0.0) + 1.0;
                let beta = self.success_counts.get(&c.id).copied().unwrap_or(0.0) + 1.0;

                // Sample from Beta distribution
                #[allow(clippy::unwrap_used)]
                let beta_dist =
                    Beta::new(alpha, beta).unwrap_or_else(|_| Beta::new(1.0, 1.0).unwrap());
                let score = beta_dist.sample(&mut rng);

                (c.id, score)
            })
            .collect();

        // Return cluster with highest sampled score
        scores
            .into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(id, _)| id)
    }

    /// Select samples for next batch using Thompson Sampling
    ///
    /// Returns indices of codes to sample next
    pub fn select_batch(&self, codes: &[&str], batch_size: usize) -> Vec<usize> {
        if codes.is_empty() || batch_size == 0 {
            return vec![];
        }

        let Some(_result) = &self.cluster_result else {
            return (0..batch_size.min(codes.len())).collect();
        };

        let mut rng = rand::rng();
        let mut selected = Vec::with_capacity(batch_size);
        let mut remaining: Vec<usize> = (0..codes.len()).collect();

        while selected.len() < batch_size && !remaining.is_empty() {
            // Sample cluster using Thompson Sampling
            let target_cluster = self.sample_cluster().unwrap_or(0);

            // Find samples in target cluster
            let in_cluster: Vec<usize> = remaining
                .iter()
                .filter(|&&i| {
                    self.get_cluster(codes[i])
                        .is_some_and(|c| c == target_cluster)
                })
                .copied()
                .collect();

            if in_cluster.is_empty() {
                // Fallback: random selection
                let idx = rng.random_range(0..remaining.len());
                let sample_idx = remaining.remove(idx);
                selected.push(sample_idx);
            } else {
                // Select from target cluster
                let idx = rng.random_range(0..in_cluster.len());
                let sample_idx = in_cluster[idx];
                remaining.retain(|&x| x != sample_idx);
                selected.push(sample_idx);
            }
        }

        selected
    }

    /// Update with oracle feedback
    ///
    /// # Arguments
    /// * `code` - The code that was verified
    /// * `revealed_bug` - True if verification revealed a bug
    pub fn update_feedback(&mut self, code: &str, revealed_bug: bool) {
        if let Some(cluster_id) = self.get_cluster(code) {
            if revealed_bug {
                *self.failure_counts.entry(cluster_id).or_insert(0.0) += 1.0;
            } else {
                *self.success_counts.entry(cluster_id).or_insert(0.0) += 1.0;
            }
        }
        self.total_samples += 1;
    }

    /// Get current silhouette score
    #[must_use]
    pub fn silhouette_score(&self) -> f32 {
        self.cluster_result
            .as_ref()
            .map_or(0.0, |r| r.silhouette_score)
    }

    /// Get cluster statistics
    #[must_use]
    pub fn cluster_stats(&self) -> Vec<ClusterStats> {
        self.cluster_result
            .as_ref()
            .map(|r| {
                r.clusters
                    .iter()
                    .map(|c| {
                        let successes = self.success_counts.get(&c.id).copied().unwrap_or(0.0);
                        let failures = self.failure_counts.get(&c.id).copied().unwrap_or(0.0);
                        let total = successes + failures;

                        ClusterStats {
                            cluster_id: c.id,
                            size: c.size,
                            bug_rate: if total > 0.0 { failures / total } else { 0.5 },
                            #[allow(clippy::cast_sign_loss)]
                            samples_tried: total.max(0.0) as usize,
                        }
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Check if exploration should be prioritized
    #[must_use]
    pub fn should_explore(&self) -> bool {
        let mut rng = rand::rng();
        rng.random::<f64>() < self.exploration_rate
    }

    /// Get total samples processed
    #[must_use]
    pub fn total_samples(&self) -> usize {
        self.total_samples
    }
}

/// Statistics for a cluster
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterStats {
    /// Cluster ID
    pub cluster_id: usize,
    /// Number of samples in cluster
    pub size: usize,
    /// Bug revelation rate (0-1)
    pub bug_rate: f64,
    /// Number of samples tried from this cluster
    pub samples_tried: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_codes() -> Vec<&'static str> {
        vec![
            "def add(a, b):\n    return a + b",
            "def sub(a, b):\n    return a - b",
            "for i in range(10):\n    print(i)",
            "while True:\n    break",
            "if x > 0:\n    return x\nelse:\n    return -x",
            "class Foo:\n    def __init__(self):\n        pass",
            "x = [1, 2, 3]\ny = sum(x)",
            "import os\npath = os.getcwd()",
        ]
    }

    // ========== CodeEmbedding Tests ==========

    #[test]
    fn test_code_embedding_new() {
        let emb = CodeEmbedding::new(64);
        assert_eq!(emb.dim, 64);
        assert_eq!(emb.features.len(), 64);
    }

    #[test]
    fn test_code_embedding_from_vec() {
        let features = vec![1.0, 2.0, 3.0];
        let emb = CodeEmbedding::from_vec(features.clone());
        assert_eq!(emb.features, features);
    }

    #[test]
    fn test_code_embedding_norm() {
        let emb = CodeEmbedding::from_vec(vec![3.0, 4.0]);
        assert!((emb.norm() - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_code_embedding_normalize() {
        let mut emb = CodeEmbedding::from_vec(vec![3.0, 4.0]);
        emb.normalize();
        assert!((emb.norm() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_code_embedding_cosine_similarity_same() {
        let emb = CodeEmbedding::from_vec(vec![1.0, 2.0, 3.0]);
        assert!((emb.cosine_similarity(&emb) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_code_embedding_cosine_similarity_orthogonal() {
        let emb1 = CodeEmbedding::from_vec(vec![1.0, 0.0]);
        let emb2 = CodeEmbedding::from_vec(vec![0.0, 1.0]);
        assert!(emb1.cosine_similarity(&emb2).abs() < 0.001);
    }

    #[test]
    fn test_code_embedding_euclidean_distance() {
        let emb1 = CodeEmbedding::from_vec(vec![0.0, 0.0]);
        let emb2 = CodeEmbedding::from_vec(vec![3.0, 4.0]);
        assert!((emb1.euclidean_distance(&emb2) - 5.0).abs() < 0.001);
    }

    // ========== CodeEmbedder Tests ==========

    #[test]
    fn test_code_embedder_default() {
        let embedder = CodeEmbedder::default();
        assert_eq!(embedder.n, 3);
        assert_eq!(embedder.vocab_size, 128);
    }

    #[test]
    fn test_code_embedder_embed() {
        let embedder = CodeEmbedder::default();
        let emb = embedder.embed("def foo(): return 1");
        assert_eq!(emb.dim, 128);
        assert!(emb.norm() > 0.0);
    }

    #[test]
    fn test_code_embedder_similar_code() {
        let embedder = CodeEmbedder::default();
        let emb1 = embedder.embed("def add(a, b): return a + b");
        let emb2 = embedder.embed("def add(x, y): return x + y");
        let emb3 = embedder.embed("class Foo: pass");

        // Similar functions should be more similar than different constructs
        let sim_12 = emb1.cosine_similarity(&emb2);
        let sim_13 = emb1.cosine_similarity(&emb3);
        assert!(sim_12 > sim_13);
    }

    #[test]
    fn test_code_embedder_empty() {
        let embedder = CodeEmbedder::default();
        let emb = embedder.embed("");
        assert_eq!(emb.dim, 128);
    }

    // ========== KMeansClustering Tests ==========

    #[test]
    fn test_kmeans_default() {
        let kmeans = KMeansClustering::default();
        assert_eq!(kmeans.k, 5);
    }

    #[test]
    fn test_kmeans_fit_empty() {
        let kmeans = KMeansClustering::new(3);
        let result = kmeans.fit(&[]);
        assert!(result.clusters.is_empty());
        assert!(result.assignments.is_empty());
    }

    #[test]
    fn test_kmeans_fit() {
        let embedder = CodeEmbedder::default();
        let codes = sample_codes();
        let embeddings: Vec<CodeEmbedding> = codes.iter().map(|c| embedder.embed(c)).collect();

        let kmeans = KMeansClustering::new(3).with_seed(42);
        let result = kmeans.fit(&embeddings);

        assert_eq!(result.clusters.len(), 3);
        assert_eq!(result.assignments.len(), codes.len());
    }

    #[test]
    fn test_kmeans_silhouette_bounded() {
        let embedder = CodeEmbedder::default();
        let codes = sample_codes();
        let embeddings: Vec<CodeEmbedding> = codes.iter().map(|c| embedder.embed(c)).collect();

        let kmeans = KMeansClustering::new(3);
        let result = kmeans.fit(&embeddings);

        // Silhouette should be in [-1, 1]
        assert!(result.silhouette_score >= -1.0);
        assert!(result.silhouette_score <= 1.0);
    }

    // ========== ActiveLearner Tests ==========

    #[test]
    fn test_active_learner_new() {
        let learner = ActiveLearner::new(5);
        assert_eq!(learner.total_samples(), 0);
    }

    #[test]
    fn test_active_learner_fit() {
        let mut learner = ActiveLearner::new(3);
        let codes = sample_codes();

        learner.fit(&codes);

        assert!(learner.silhouette_score() >= -1.0);
    }

    #[test]
    fn test_active_learner_get_cluster() {
        let mut learner = ActiveLearner::new(3);
        let codes = sample_codes();

        learner.fit(&codes);

        let cluster = learner.get_cluster(codes[0]);
        assert!(cluster.is_some());
    }

    #[test]
    fn test_active_learner_sample_cluster() {
        let mut learner = ActiveLearner::new(3);
        let codes = sample_codes();

        learner.fit(&codes);

        let cluster = learner.sample_cluster();
        assert!(cluster.is_some());
    }

    #[test]
    fn test_active_learner_select_batch() {
        let mut learner = ActiveLearner::new(3);
        let codes = sample_codes();

        learner.fit(&codes);

        let batch = learner.select_batch(&codes, 3);
        assert_eq!(batch.len(), 3);
        // All indices should be unique
        let mut sorted = batch.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), batch.len());
    }

    #[test]
    fn test_active_learner_update_feedback() {
        let mut learner = ActiveLearner::new(3);
        let codes = sample_codes();

        learner.fit(&codes);

        learner.update_feedback(codes[0], true);
        learner.update_feedback(codes[1], false);

        assert_eq!(learner.total_samples(), 2);
    }

    #[test]
    fn test_active_learner_cluster_stats() {
        let mut learner = ActiveLearner::new(3);
        let codes = sample_codes();

        learner.fit(&codes);

        // Add some feedback
        for (i, &code) in codes.iter().enumerate() {
            learner.update_feedback(code, i % 2 == 0);
        }

        let stats = learner.cluster_stats();
        assert!(!stats.is_empty());
    }

    #[test]
    fn test_active_learner_exploration_rate() {
        let learner = ActiveLearner::new(3).with_exploration_rate(1.0);

        // With rate=1.0, should always explore
        let mut explored = 0;
        for _ in 0..100 {
            if learner.should_explore() {
                explored += 1;
            }
        }
        assert_eq!(explored, 100);
    }

    // ========== Debug Tests ==========

    #[test]
    fn test_code_embedding_debug() {
        let emb = CodeEmbedding::new(4);
        let debug = format!("{emb:?}");
        assert!(debug.contains("CodeEmbedding"));
    }

    #[test]
    fn test_code_embedder_debug() {
        let embedder = CodeEmbedder::default();
        let debug = format!("{embedder:?}");
        assert!(debug.contains("CodeEmbedder"));
    }

    #[test]
    fn test_cluster_debug() {
        let cluster = Cluster::new(0, CodeEmbedding::new(4));
        let debug = format!("{cluster:?}");
        assert!(debug.contains("Cluster"));
    }

    #[test]
    fn test_active_learner_debug() {
        let learner = ActiveLearner::new(3);
        let debug = format!("{learner:?}");
        assert!(debug.contains("ActiveLearner"));
    }
}

/// Property-based tests
#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// Embedding norm is non-negative
        #[test]
        fn prop_embedding_norm_nonnegative(features in proptest::collection::vec(-100.0f32..100.0, 1..50)) {
            let emb = CodeEmbedding::from_vec(features);
            prop_assert!(emb.norm() >= 0.0);
        }

        /// Cosine similarity is bounded [-1, 1]
        #[test]
        fn prop_cosine_bounded(
            f1 in proptest::collection::vec(-10.0f32..10.0, 1..20),
            f2 in proptest::collection::vec(-10.0f32..10.0, 1..20),
        ) {
            let dim = f1.len().min(f2.len());
            let emb1 = CodeEmbedding::from_vec(f1[..dim].to_vec());
            let emb2 = CodeEmbedding::from_vec(f2[..dim].to_vec());

            let sim = emb1.cosine_similarity(&emb2);
            prop_assert!(sim >= -1.0 - 0.001);
            prop_assert!(sim <= 1.0 + 0.001);
        }

        /// Euclidean distance is non-negative
        #[test]
        fn prop_euclidean_nonnegative(
            f1 in proptest::collection::vec(-100.0f32..100.0, 1..20),
            f2 in proptest::collection::vec(-100.0f32..100.0, 1..20),
        ) {
            let dim = f1.len().min(f2.len());
            let emb1 = CodeEmbedding::from_vec(f1[..dim].to_vec());
            let emb2 = CodeEmbedding::from_vec(f2[..dim].to_vec());

            prop_assert!(emb1.euclidean_distance(&emb2) >= 0.0);
        }

        /// Normalized vectors have unit norm
        #[test]
        fn prop_normalized_unit_norm(features in proptest::collection::vec(0.1f32..10.0, 1..20)) {
            let mut emb = CodeEmbedding::from_vec(features);
            emb.normalize();

            // Should be close to 1.0 (allow small floating point error)
            prop_assert!((emb.norm() - 1.0).abs() < 0.01);
        }

        /// Batch selection returns valid indices
        #[test]
        fn prop_batch_indices_valid(batch_size in 1usize..10) {
            let mut learner = ActiveLearner::new(3);
            let codes: Vec<&str> = vec![
                "x = 1",
                "y = 2",
                "z = 3",
                "def f(): pass",
                "class C: pass",
            ];

            learner.fit(&codes);

            let batch = learner.select_batch(&codes, batch_size);

            for &idx in &batch {
                prop_assert!(idx < codes.len());
            }
        }
    }
}
