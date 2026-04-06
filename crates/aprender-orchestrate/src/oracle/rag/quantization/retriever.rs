//! Two-stage rescoring retriever
//!
//! Stage 1: Fast approximate retrieval with int8 embeddings
//! Stage 2: Precise rescoring with f32 query x i8 docs
//!
//! Achieves 99% accuracy retention with 3.66x speedup.

// Library code - usage from examples and integration tests
use super::calibration::CalibrationStats;
use super::embedding::QuantizedEmbedding;
use super::error::{validate_embedding, QuantizationError};
use super::simd::SimdBackend;

/// Two-stage rescoring retriever configuration
///
/// Following the scalar int8 rescoring specification.
#[derive(Debug, Clone)]
pub struct RescoreRetrieverConfig {
    /// Number of candidates to retrieve in stage 1 (multiplier x top_k)
    /// Optimal: 4-5x for 99% accuracy retention
    pub rescore_multiplier: usize,
    /// Final number of results to return
    pub top_k: usize,
    /// Minimum calibration samples required
    pub min_calibration_samples: usize,
    /// SIMD backend (auto-detected if None)
    pub simd_backend: Option<SimdBackend>,
}

impl Default for RescoreRetrieverConfig {
    fn default() -> Self {
        Self {
            rescore_multiplier: 4, // Optimal per specification
            top_k: 10,
            min_calibration_samples: 1000,
            simd_backend: None, // Auto-detect
        }
    }
}

/// Two-stage rescoring retriever
///
/// Stage 1: Fast approximate retrieval with int8 embeddings
/// Stage 2: Precise rescoring with f32 query x i8 docs
///
/// Achieves 99% accuracy retention with 3.66x speedup.
#[derive(Debug)]
pub struct RescoreRetriever {
    /// Int8 quantized document embeddings
    embeddings: Vec<QuantizedEmbedding>,
    /// Document IDs corresponding to embeddings
    doc_ids: Vec<String>,
    /// Calibration statistics
    calibration: CalibrationStats,
    /// Configuration
    config: RescoreRetrieverConfig,
    /// SIMD backend
    backend: SimdBackend,
}

impl RescoreRetriever {
    /// Create new rescoring retriever
    pub fn new(dims: usize, config: RescoreRetrieverConfig) -> Self {
        let backend = config.simd_backend.unwrap_or_else(SimdBackend::detect);
        Self {
            embeddings: Vec::new(),
            doc_ids: Vec::new(),
            calibration: CalibrationStats::new(dims),
            config,
            backend,
        }
    }

    /// Add embedding to calibration set (Kaizen)
    pub fn add_calibration_sample(&mut self, embedding: &[f32]) -> Result<(), QuantizationError> {
        self.calibration.update(embedding)
    }

    /// Index a document with its embedding
    pub fn index_document(
        &mut self,
        doc_id: &str,
        embedding: &[f32],
    ) -> Result<(), QuantizationError> {
        // Update calibration
        self.calibration.update(embedding)?;

        // Quantize embedding
        let quantized = QuantizedEmbedding::from_f32(embedding, &self.calibration)?;

        self.embeddings.push(quantized);
        self.doc_ids.push(doc_id.to_string());

        Ok(())
    }

    /// Stage 1: Fast int8 retrieval
    ///
    /// Returns (doc_index, approximate_score) pairs
    fn stage1_retrieve(&self, query_i8: &[i8]) -> Vec<(usize, i32)> {
        let num_candidates = self.config.top_k * self.config.rescore_multiplier;

        let mut scores: Vec<(usize, i32)> = self
            .embeddings
            .iter()
            .enumerate()
            .map(|(i, emb)| {
                let score = self.backend.dot_i8(query_i8, &emb.values);
                (i, score)
            })
            .collect();

        // Sort descending by score
        scores.sort_by(|a, b| b.1.cmp(&a.1));
        scores.truncate(num_candidates);

        scores
    }

    /// Stage 2: Precise rescoring with f32 query
    fn stage2_rescore(&self, query: &[f32], candidates: Vec<(usize, i32)>) -> Vec<RescoreResult> {
        let mut results: Vec<RescoreResult> = candidates
            .into_iter()
            .map(|(doc_idx, approx_score)| {
                let emb = &self.embeddings[doc_idx];
                let precise_score = self.backend.dot_f32_i8(query, &emb.values, emb.params.scale);

                RescoreResult {
                    doc_id: self.doc_ids[doc_idx].clone(),
                    score: precise_score,
                    approx_score,
                }
            })
            .collect();

        // Sort by precise score descending
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(self.config.top_k);

        results
    }

    /// Full two-stage retrieval
    ///
    /// # Arguments
    /// * `query` - f32 query embedding
    ///
    /// # Returns
    /// Top-k results with precise scores
    pub fn retrieve(&self, query: &[f32]) -> Result<Vec<RescoreResult>, QuantizationError> {
        // Validate query embedding
        validate_embedding(query, self.calibration.dims)?;

        // Stage 1: Quantize query and retrieve candidates
        let query_quantized = QuantizedEmbedding::from_f32(query, &self.calibration)?;
        let candidates = self.stage1_retrieve(&query_quantized.values);

        // Stage 2: Rescore with f32 precision
        Ok(self.stage2_rescore(query, candidates))
    }

    /// Get number of indexed documents
    pub fn len(&self) -> usize {
        self.embeddings.len()
    }

    /// Check if index is empty
    pub fn is_empty(&self) -> bool {
        self.embeddings.is_empty()
    }

    /// Get calibration statistics
    pub fn calibration(&self) -> &CalibrationStats {
        &self.calibration
    }

    /// Get total memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        self.embeddings.iter().map(|e| e.memory_size()).sum()
    }
}

/// Result from rescoring retrieval
#[derive(Debug, Clone)]
pub struct RescoreResult {
    /// Document ID
    pub doc_id: String,
    /// Precise score from stage 2 (f32 x i8)
    pub score: f32,
    /// Approximate score from stage 1 (i8 x i8)
    pub approx_score: i32,
}
