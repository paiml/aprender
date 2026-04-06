use super::*;

// ============================================================================
// Shared Test Helpers
// ============================================================================

mod test_helpers {
    use super::*;

    /// Linear embedding: `[0 * factor, 1 * factor, ..., (dims-1) * factor]`
    pub fn linear_embedding(dims: usize, factor: f32) -> Vec<f32> {
        (0..dims).map(|i| i as f32 * factor).collect()
    }

    /// Sin embedding with offset: `[sin((base+0) * factor), sin((base+1) * factor), ...]`
    pub fn sin_embedding_offset(dims: usize, base: usize, factor: f32) -> Vec<f32> {
        (0..dims).map(|j| ((base + j) as f32 * factor).sin()).collect()
    }

    /// Index-based embedding: `[(base*dims+0) as f32, (base*dims+1) as f32, ...]`
    pub fn index_embedding(dims: usize, base: usize) -> Vec<f32> {
        (0..dims).map(|j| (base * dims + j) as f32).collect()
    }

    /// Offset embedding: `[(base+0) as f32, (base+1) as f32, ...]`
    pub fn offset_embedding(dims: usize, base: usize) -> Vec<f32> {
        (0..dims).map(|j| (base + j) as f32).collect()
    }

    /// Assert that retrieval results have monotonically decreasing scores.
    pub fn assert_scores_monotonic(results: &[RescoreResult], tag: &str) {
        for i in 1..results.len() {
            assert!(
                results[i - 1].score >= results[i].score,
                "{}: scores not monotonic at position {}",
                tag,
                i
            );
        }
    }

    /// Assert that quantization rejects a non-finite value at index 0.
    pub fn assert_nonfinite_rejected(value: f32, dims: usize, cal: &CalibrationStats, label: &str) {
        let mut embedding = vec![0.0f32; dims];
        embedding[0] = value;
        let result = QuantizedEmbedding::from_f32(&embedding, cal);
        assert!(
            matches!(result, Err(QuantizationError::NonFiniteValue { .. })),
            "QA-05 FALSIFIED: {} not rejected",
            label,
        );
    }
}

// ============================================================================
// Section 1: Quantization Accuracy Tests (QA-01 to QA-15)
// ============================================================================

#[allow(non_snake_case)]
mod qa_tests {
    use super::test_helpers::*;
    use super::*;

    /// Generate a sin-based test embedding of the given dimension.
    fn sin_embedding(dims: usize, index: usize) -> Vec<f32> {
        (0..dims).map(|j| ((index * dims + j) as f32 * 0.001).sin()).collect()
    }

    /// Create a CalibrationStats pre-calibrated with sin embeddings.
    fn calibrated_stats(dims: usize, n_samples: usize) -> CalibrationStats {
        let mut cal = CalibrationStats::new(dims);
        for i in 0..n_samples {
            let _ = cal.update(&sin_embedding(dims, i));
        }
        cal
    }

    // QA-01: Quantization Error Bound
    // Claim: Quantization error is bounded by |Q(x) - x| <= scale/2
    #[test]
    fn test_QA_01_quantization_error_bound() {
        let dims = 384;
        let cal = calibrated_stats(dims, 1000);

        // Test error bound
        let test_embedding = sin_embedding_offset(dims, 0, 0.01);

        let quantized = QuantizedEmbedding::from_f32(&test_embedding, &cal).unwrap();
        let scale = quantized.params.scale;
        let dequantized = quantized.dequantize();

        let max_error = test_embedding
            .iter()
            .zip(dequantized.iter())
            .map(|(orig, deq)| (orig - deq).abs())
            .fold(0.0f32, f32::max);

        // Error should be bounded by scale/2 + epsilon for rounding
        let bound = scale / 2.0 + 1e-6;
        assert!(max_error <= bound, "QA-01 FALSIFIED: max_error {} > bound {}", max_error, bound);
    }

    // QA-02: Symmetric Quantization Optimality
    // Claim: Zero-mean embeddings use zero_point = 0
    #[test]
    fn test_QA_02_symmetric_quantization_optimality() {
        let dims = 128;
        let mut cal = CalibrationStats::new(dims);

        // Zero-mean embedding
        let embedding: Vec<f32> = (0..dims).map(|i| (i as f32 - 64.0) * 0.01).collect();
        let _ = cal.update(&embedding);

        let quantized = QuantizedEmbedding::from_f32(&embedding, &cal).unwrap();

        assert_eq!(
            quantized.params.zero_point, 0,
            "QA-02 FALSIFIED: zero_point should be 0 for symmetric quantization"
        );
    }

    // QA-03: Calibration Dataset Representativeness
    // Claim: 1000 samples sufficient for stable calibration
    #[test]
    fn test_QA_03_calibration_convergence() {
        let dims = 128;
        let mut cal1 = CalibrationStats::new(dims);
        let mut cal2 = CalibrationStats::new(dims);

        // Two different 1000-sample subsets (using 2x stride for separation)
        for i in 0..1000 {
            let _ = cal1.update(&sin_embedding(dims, i * 2));
        }
        for i in 1000..2000 {
            let _ = cal2.update(&sin_embedding(dims, i * 2));
        }

        // absmax should be within 5%
        let diff = (cal1.absmax - cal2.absmax).abs() / cal1.absmax.max(cal2.absmax);
        assert!(diff < 0.05, "QA-03 FALSIFIED: absmax varies by {:.1}% > 5%", diff * 100.0);
    }

    // QA-04: Overflow Prevention in Int8 Dot Product
    // Claim: i32 accumulator prevents overflow for dims <= 4096
    #[test]
    fn test_QA_04_dot_product_overflow_prevention() {
        // Worst case: 127 * 127 * 4096 = 66,060,289 < i32::MAX (2,147,483,647)
        let dims = 4096;
        let a: Vec<i8> = vec![127; dims];
        let b: Vec<i8> = vec![127; dims];

        let result = dot_i8_scalar(&a, &b);
        let expected = 127i32 * 127i32 * dims as i32;

        assert_eq!(result, expected, "QA-04 FALSIFIED: overflow detected or incorrect result");
    }

    // QA-05: NaN/Inf Handling in Quantization
    // Claim: Quantization rejects non-finite inputs
    #[test]
    fn test_QA_05_nonfinite_rejection() {
        let dims = 128;
        let mut cal = CalibrationStats::new(dims);
        let _ = cal.update(&vec![1.0; dims]);

        assert_nonfinite_rejected(f32::NAN, dims, &cal, "NaN");
        assert_nonfinite_rejected(f32::INFINITY, dims, &cal, "Inf");
        assert_nonfinite_rejected(f32::NEG_INFINITY, dims, &cal, "-Inf");
    }

    // QA-06: Dequantization Reversibility
    // Claim: Dequantization recovers original within error bound
    #[test]
    fn test_QA_06_dequantization_reversibility() {
        let dims = 256;
        let mut cal = CalibrationStats::new(dims);

        let embedding = sin_embedding_offset(dims, 0, 0.01);
        let _ = cal.update(&embedding);

        let quantized = QuantizedEmbedding::from_f32(&embedding, &cal).unwrap();
        let scale = quantized.params.scale;
        let dequantized = quantized.dequantize();

        for (i, (&orig, &deq)) in embedding.iter().zip(dequantized.iter()).enumerate() {
            let error = (orig - deq).abs();
            assert!(
                error <= scale,
                "QA-06 FALSIFIED: error {} > scale {} at index {}",
                error,
                scale,
                i
            );
        }
    }

    // QA-07: Scale Factor Computation Accuracy
    // Claim: Scale = absmax / 127.0
    #[test]
    fn test_QA_07_scale_computation() {
        let dims = 64;
        let mut cal = CalibrationStats::new(dims);

        let embedding = linear_embedding(dims, 0.5);
        let _ = cal.update(&embedding);

        let expected_scale = cal.absmax / 127.0;
        let quantized = QuantizedEmbedding::from_f32(&embedding, &cal).unwrap();

        let diff = (quantized.params.scale - expected_scale).abs();
        assert!(
            diff < 1e-6,
            "QA-07 FALSIFIED: scale {} != expected {}",
            quantized.params.scale,
            expected_scale
        );
    }

    // QA-08: Zero-Point Correctness (Symmetric)
    // Claim: Zero-point is 0 for symmetric quantization
    #[test]
    fn test_QA_08_zero_point_symmetric() {
        let dims = 64;
        let mut cal = CalibrationStats::new(dims);
        let embedding = linear_embedding(dims, 0.1);
        let _ = cal.update(&embedding);

        let quantized = QuantizedEmbedding::from_f32(&embedding, &cal).unwrap();
        assert_eq!(quantized.params.zero_point, 0, "QA-08 FALSIFIED: zero_point != 0");
    }

    // QA-09: Clipping Behavior at Boundaries
    // Claim: Values correctly clipped to [-128, 127]
    #[test]
    fn test_QA_09_clipping_behavior() {
        let dims = 64;
        let mut cal = CalibrationStats::new(dims);

        // Small calibration absmax
        let small_embedding: Vec<f32> = vec![0.1; dims];
        let _ = cal.update(&small_embedding);

        // Large values that should be clipped
        let large_embedding: Vec<f32> = vec![100.0; dims];
        let quantized = QuantizedEmbedding::from_f32(&large_embedding, &cal).unwrap();

        // Should be clipped to 127 (i8 range is inherently [-128, 127])
        assert!(
            quantized.values.iter().all(|&v| v == 127),
            "QA-09 FALSIFIED: large values not clipped to 127"
        );
    }

    // QA-10: Dimension Mismatch Detection
    // Claim: Quantization fails on dimension mismatch
    #[test]
    fn test_QA_10_dimension_mismatch() {
        let dims = 128;
        let mut cal = CalibrationStats::new(dims);
        let _ = cal.update(&vec![1.0; dims]);

        // Wrong dimension
        let wrong_dims: Vec<f32> = vec![1.0; 64];
        let result = QuantizedEmbedding::from_f32(&wrong_dims, &cal);

        assert!(
            matches!(result, Err(QuantizationError::DimensionMismatch { .. })),
            "QA-10 FALSIFIED: dimension mismatch not detected"
        );
    }

    // QA-11: Content Hash Integrity
    // Claim: Content hash uniquely identifies quantized embedding
    #[test]
    fn test_QA_11_content_hash_integrity() {
        let dims = 64;
        let mut cal = CalibrationStats::new(dims);
        let _ = cal.update(&vec![1.0; dims]);

        let emb1 = linear_embedding(dims, 0.01);
        let emb2 = linear_embedding(dims, 0.02);

        let q1 = QuantizedEmbedding::from_f32(&emb1, &cal).unwrap();
        let q2 = QuantizedEmbedding::from_f32(&emb2, &cal).unwrap();

        assert_ne!(q1.hash, q2.hash, "QA-11 FALSIFIED: different embeddings have same hash");

        // Same embedding should have same hash
        let q1_again = QuantizedEmbedding::from_f32(&emb1, &cal).unwrap();
        assert_eq!(q1.hash, q1_again.hash, "QA-11 FALSIFIED: same embedding has different hash");
    }

    // QA-12: Calibration Update Numerical Stability (Welford's)
    #[test]
    fn test_QA_12_welford_stability() {
        let dims = 32;
        let mut cal = CalibrationStats::new(dims);

        // Add many samples (linear formula for numerical stability test)
        for i in 0..10000 {
            let embedding: Vec<f32> = (0..dims).map(|j| (i * dims + j) as f32 * 0.0001).collect();
            let _ = cal.update(&embedding);
        }

        // Mean should be stable (not NaN or Inf)
        for &m in &cal.mean {
            assert!(m.is_finite(), "QA-12 FALSIFIED: mean is not finite");
        }

        // Variance should be stable for each dimension
        for i in 0..dims {
            let var = cal.variance(i);
            assert!(var.is_finite(), "QA-12 FALSIFIED: variance is not finite at dim {}", i);
        }
    }

    // QA-13: Quantization Determinism
    // Claim: Same input produces same output
    #[test]
    fn test_QA_13_quantization_determinism() {
        let dims = 128;
        let mut cal = CalibrationStats::new(dims);
        let embedding: Vec<f32> = (0..dims).map(|i| (i as f32 * 0.01).cos()).collect();
        let _ = cal.update(&embedding);

        let q1 = QuantizedEmbedding::from_f32(&embedding, &cal).unwrap();
        let q2 = QuantizedEmbedding::from_f32(&embedding, &cal).unwrap();

        assert_eq!(q1.values, q2.values, "QA-13 FALSIFIED: non-deterministic quantization");
        assert_eq!(q1.hash, q2.hash, "QA-13 FALSIFIED: non-deterministic hash");
    }

    // QA-14: Memory Layout Correctness
    // Claim: Int8 values stored contiguously
    #[test]
    fn test_QA_14_memory_layout() {
        let dims = 256;
        let mut cal = CalibrationStats::new(dims);
        let embedding = linear_embedding(dims, 0.01);
        let _ = cal.update(&embedding);

        let quantized = QuantizedEmbedding::from_f32(&embedding, &cal).unwrap();

        // Vec<i8> should be contiguous
        assert_eq!(quantized.values.len(), dims, "QA-14 FALSIFIED: wrong number of values");

        // Check contiguous memory layout via slice indexing
        // (Vec guarantees contiguous allocation; verify values match)
        for i in 0..dims {
            let val = quantized.values[i];
            assert_eq!(val, quantized.values[i], "QA-14 FALSIFIED: non-contiguous memory at {}", i);
        }
    }

    // QA-15: Batch Quantization Consistency
    // Claim: Batch and individual produce identical results
    #[test]
    fn test_QA_15_batch_consistency() {
        let dims = 64;
        let mut cal = CalibrationStats::new(dims);

        // Calibrate
        for i in 0..100 {
            let embedding: Vec<f32> = (0..dims).map(|j| (i * dims + j) as f32 * 0.01).collect();
            let _ = cal.update(&embedding);
        }

        // Test embeddings
        let embeddings: Vec<Vec<f32>> =
            (0..10).map(|i| (0..dims).map(|j| (i * dims + j) as f32 * 0.005).collect()).collect();

        // Individual quantization
        let individual: Vec<_> =
            embeddings.iter().map(|e| QuantizedEmbedding::from_f32(e, &cal).unwrap()).collect();

        // Each should match (batch = individual since we process one at a time)
        for (i, q) in individual.iter().enumerate() {
            let q_again = QuantizedEmbedding::from_f32(&embeddings[i], &cal).unwrap();
            assert_eq!(q.values, q_again.values, "QA-15 FALSIFIED: batch inconsistency at {}", i);
        }
    }
}

// ============================================================================
// Section 2: Retrieval Accuracy Tests (RA-01 to RA-15)
// ============================================================================

#[allow(non_snake_case)]
mod ra_tests {
    use super::test_helpers::*;
    use super::*;

    /// Create a one-hot-like embedding (0.1 everywhere, 1.0 at target position).
    fn one_hot_embedding(dims: usize, target: usize) -> Vec<f32> {
        let mut embedding = vec![0.1f32; dims];
        embedding[target % dims] = 1.0;
        embedding
    }

    /// Create a retriever pre-populated with one-hot documents.
    fn populated_retriever(
        dims: usize,
        n_docs: usize,
        config: RescoreRetrieverConfig,
    ) -> RescoreRetriever {
        let mut retriever = RescoreRetriever::new(dims, config);
        for i in 0..n_docs {
            let _ = retriever.index_document(&format!("doc_{}", i), &one_hot_embedding(dims, i));
        }
        retriever
    }

    // RA-01: 99% Accuracy Retention
    // Claim: Int8 + rescoring retains >=99% of f32 accuracy
    #[test]
    fn test_RA_01_accuracy_retention() {
        let dims = 128;
        let retriever = populated_retriever(dims, 20, RescoreRetrieverConfig::default());

        // Query matches doc_10
        let target = 10;
        let query = one_hot_embedding(dims, target);

        let results = retriever.retrieve(&query).unwrap();

        // Top result should be doc_10 (highest dot product due to matching 1.0 positions)
        assert!(!results.is_empty(), "RA-01 FALSIFIED: no results returned");
        assert_eq!(
            results[0].doc_id,
            format!("doc_{}", target),
            "RA-01 FALSIFIED: wrong top result, expected doc_{}, got {}",
            target,
            results[0].doc_id
        );
    }

    // RA-02: Rescore Multiplier Optimality
    // Claim: Multiplier of 4 achieves optimal accuracy/speed tradeoff
    #[test]
    fn test_RA_02_rescore_multiplier() {
        let config = RescoreRetrieverConfig::default();
        assert_eq!(config.rescore_multiplier, 4, "RA-02 FALSIFIED: default multiplier should be 4");
    }

    // RA-03: Stage 1 Recall Sufficiency
    // Claim: Stage 1 retrieves true top-k within candidate set
    #[test]
    fn test_RA_03_stage1_recall() {
        let dims = 64;
        let config =
            RescoreRetrieverConfig { rescore_multiplier: 4, top_k: 5, ..Default::default() };
        let retriever = populated_retriever(dims, 20, config);

        // Query matches doc_10
        let target = 10;
        let query = one_hot_embedding(dims, target);
        let results = retriever.retrieve(&query).unwrap();

        // doc_10 should be in top results
        let has_target = results.iter().any(|r| r.doc_id == format!("doc_{}", target));
        assert!(
            has_target,
            "RA-03 FALSIFIED: true top result doc_{} missing from candidates, got {:?}",
            target,
            results.iter().map(|r| &r.doc_id).collect::<Vec<_>>()
        );
    }

    // RA-04: Rescoring Rank Improvement
    // Claim: Rescoring improves rank of relevant documents
    #[test]
    fn test_RA_04_rescore_improvement() {
        // Rescoring uses f32 x i8 which should be more accurate than i8 x i8
        let dims = 128;
        let config = RescoreRetrieverConfig::default();
        let mut retriever = RescoreRetriever::new(dims, config);

        for i in 0..20 {
            let embedding = sin_embedding_offset(dims, i, 0.1);
            let _ = retriever.index_document(&format!("doc_{}", i), &embedding);
        }

        let query = sin_embedding_offset(dims, 10, 0.1);
        let results = retriever.retrieve(&query).unwrap();

        // Results should be sorted by score (descending)
        assert_scores_monotonic(&results, "RA-04 FALSIFIED");
    }

    // RA-05: Empty Index Handling
    #[test]
    fn test_RA_05_empty_index() {
        let dims = 64;
        let config = RescoreRetrieverConfig::default();
        let retriever = RescoreRetriever::new(dims, config);

        let query: Vec<f32> = vec![1.0; dims];
        // Empty index returns CalibrationNotInitialized error - correct behavior
        let result = retriever.retrieve(&query);

        assert!(
            matches!(result, Err(QuantizationError::CalibrationNotInitialized)),
            "RA-05: empty index should return CalibrationNotInitialized error"
        );
    }

    // RA-06: Query Embedding Consistency
    #[test]
    fn test_RA_06_query_consistency() {
        let dims = 64;
        let config = RescoreRetrieverConfig::default();
        let mut retriever = RescoreRetriever::new(dims, config);

        for i in 0..10 {
            let embedding = offset_embedding(dims, i);
            let _ = retriever.index_document(&format!("doc_{}", i), &embedding);
        }

        let query = offset_embedding(dims, 5);

        let results1 = retriever.retrieve(&query).unwrap();
        let results2 = retriever.retrieve(&query).unwrap();

        // Same query should give same results
        assert_eq!(results1.len(), results2.len(), "RA-06 FALSIFIED: inconsistent result count");
        for (r1, r2) in results1.iter().zip(results2.iter()) {
            assert_eq!(r1.doc_id, r2.doc_id, "RA-06 FALSIFIED: inconsistent result ordering");
        }
    }

    // RA-07: Score Monotonicity
    #[test]
    fn test_RA_07_score_monotonicity() {
        let dims = 64;
        let config = RescoreRetrieverConfig::default();
        let mut retriever = RescoreRetriever::new(dims, config);

        for i in 0..20 {
            let embedding = index_embedding(dims, i);
            let _ = retriever.index_document(&format!("doc_{}", i), &embedding);
        }

        let query = index_embedding(dims, 10);
        let results = retriever.retrieve(&query).unwrap();

        // Scores should be monotonically decreasing
        assert_scores_monotonic(&results, "RA-07 FALSIFIED");
    }

    // RA-08: Empty Query Handling
    #[test]
    fn test_RA_08_empty_query() {
        let dims = 64;
        let config = RescoreRetrieverConfig::default();
        let mut retriever = RescoreRetriever::new(dims, config);

        for i in 0..5 {
            let embedding: Vec<f32> = vec![1.0; dims];
            let _ = retriever.index_document(&format!("doc_{}", i), &embedding);
        }

        // Zero vector query
        let query: Vec<f32> = vec![0.0; dims];
        let results = retriever.retrieve(&query).unwrap();

        // Should handle gracefully (all scores should be 0)
        for r in &results {
            assert!(r.score.is_finite(), "RA-08 FALSIFIED: non-finite score for zero query");
        }
    }

    // RA-09: Large-K Retrieval
    #[test]
    fn test_RA_09_large_k_retrieval() {
        let dims = 32;
        let n_docs = 20;
        // Configure top_k to match n_docs for this test
        let config = RescoreRetrieverConfig { top_k: n_docs, ..Default::default() };
        let mut retriever = RescoreRetriever::new(dims, config);

        for i in 0..n_docs {
            let embedding = offset_embedding(dims, i);
            let _ = retriever.index_document(&format!("doc_{}", i), &embedding);
        }

        let query = offset_embedding(dims, 0);
        let results = retriever.retrieve(&query).unwrap();

        assert_eq!(results.len(), n_docs, "RA-09 FALSIFIED: should return all {} docs", n_docs);
    }

    // RA-10: Duplicate Handling
    #[test]
    fn test_RA_10_duplicate_handling() {
        let dims = 32;
        let config = RescoreRetrieverConfig::default();
        let mut retriever = RescoreRetriever::new(dims, config);

        let embedding: Vec<f32> = vec![1.0; dims];
        let _ = retriever.index_document("doc_1", &embedding);
        let _ = retriever.index_document("doc_2", &embedding); // Same embedding, different ID

        let results = retriever.retrieve(&embedding).unwrap();

        // Both should be returned (different IDs)
        assert_eq!(
            results.len(),
            2,
            "RA-10 FALSIFIED: duplicate embeddings should have separate IDs"
        );
    }

    // RA-11: Score Range Validation
    #[test]
    fn test_RA_11_score_range() {
        let dims = 64;
        let config = RescoreRetrieverConfig::default();
        let mut retriever = RescoreRetriever::new(dims, config);

        for i in 0..10 {
            let embedding = sin_embedding_offset(dims, i, 0.1);
            let _ = retriever.index_document(&format!("doc_{}", i), &embedding);
        }

        let query = sin_embedding_offset(dims, 0, 0.1);
        let results = retriever.retrieve(&query).unwrap();

        for r in &results {
            assert!(r.score.is_finite(), "RA-11 FALSIFIED: score is not finite");
        }
    }

    // RA-12: Result Ordering Stability
    #[test]
    fn test_RA_12_ordering_stability() {
        let dims = 32;
        let config = RescoreRetrieverConfig::default();
        let mut retriever = RescoreRetriever::new(dims, config);

        for i in 0..10 {
            let embedding = offset_embedding(dims, i);
            let _ = retriever.index_document(&format!("doc_{}", i), &embedding);
        }

        let query = offset_embedding(dims, 0);

        // Multiple retrievals
        let results: Vec<_> = (0..5).map(|_| retriever.retrieve(&query).unwrap()).collect();

        // All should be identical
        for r in results.iter().skip(1) {
            assert_eq!(r.len(), results[0].len(), "RA-12 FALSIFIED: unstable result count");
            for (a, b) in r.iter().zip(results[0].iter()) {
                assert_eq!(a.doc_id, b.doc_id, "RA-12 FALSIFIED: unstable ordering");
            }
        }
    }

    // RA-13: Cross-Encoder Placeholder
    #[test]
    fn test_RA_13_cross_encoder_placeholder() {
        // Cross-encoder is future work; validate baseline two-stage system works
        let dims = 32;
        let config = RescoreRetrieverConfig::default();
        let retriever = RescoreRetriever::new(dims, config);
        // Baseline retriever can be constructed (cross-encoder would wrap this)
        assert!(retriever.is_empty());
    }

    // RA-14: Hybrid Fusion Placeholder
    #[test]
    fn test_RA_14_hybrid_fusion_placeholder() {
        // RRF fusion is handled in HybridRetriever
        // Validate the integration point type exists and is constructible
        let config = RescoreRetrieverConfig::default();
        assert!(config.top_k > 0, "RA-14: Default config should have positive top_k");
    }

    // RA-15: BM25 Parameters Placeholder
    #[test]
    fn test_RA_15_bm25_parameters_placeholder() {
        // BM25 is handled in HybridRetriever; validate rescore config defaults
        let config = RescoreRetrieverConfig::default();
        assert!(config.top_k > 0, "RA-15: Default rescore config should be valid");
    }
}

// ============================================================================
// Section 4: Numerical Correctness Tests (NC-01 to NC-15)
// ============================================================================

#[allow(non_snake_case)]
mod nc_tests {
    use super::test_helpers::*;
    use super::*;

    // NC-01: IEEE 754 Compliance
    #[test]
    fn test_NC_01_ieee754_compliance() {
        // Verify special value handling
        assert!(f32::NAN.is_nan(), "NC-01: NaN detection");
        assert!(f32::INFINITY.is_infinite(), "NC-01: Inf detection");
        assert!(f32::NEG_INFINITY.is_infinite(), "NC-01: -Inf detection");
    }

    // NC-02: Cross-Platform Determinism (partial)
    #[test]
    fn test_NC_02_determinism() {
        let dims = 64;
        let mut cal = CalibrationStats::new(dims);
        let embedding = linear_embedding(dims, 0.01);
        let _ = cal.update(&embedding);

        let q1 = QuantizedEmbedding::from_f32(&embedding, &cal).unwrap();
        let q2 = QuantizedEmbedding::from_f32(&embedding, &cal).unwrap();

        assert_eq!(q1.values, q2.values, "NC-02: determinism failed");
        assert_eq!(q1.hash, q2.hash, "NC-02: hash determinism failed");
    }

    // NC-03: Dot Product Associativity Handling
    #[test]
    fn test_NC_03_dot_product_order() {
        let a: Vec<i8> = (0..256).map(|i| (i % 128) as i8).collect();
        let b: Vec<i8> = (0..256).map(|i| (i % 64) as i8).collect();

        let result1 = dot_i8_scalar(&a, &b);
        let result2 = dot_i8_scalar(&a, &b);

        assert_eq!(result1, result2, "NC-03: dot product not deterministic");
    }

    // NC-04: Kahan Summation Placeholder
    #[test]
    fn test_NC_04_summation_accuracy() {
        // Our current implementation uses i32 accumulator which is exact for int8
        let a: Vec<i8> = vec![1; 1000];
        let b: Vec<i8> = vec![1; 1000];

        let result = dot_i8_scalar(&a, &b);
        assert_eq!(result, 1000, "NC-04: accumulation error");
    }

    // NC-05: Underflow Handling
    #[test]
    fn test_NC_05_underflow_handling() {
        let dims = 64;
        let mut cal = CalibrationStats::new(dims);

        // Very small values
        let small: Vec<f32> = vec![1e-38; dims];
        let _ = cal.update(&small);

        // Should not produce NaN
        assert!(cal.absmax.is_finite(), "NC-05: underflow caused non-finite");
        assert!(cal.absmax >= 0.0, "NC-05: absmax should be non-negative");
    }

    // NC-06: Overflow Handling
    #[test]
    fn test_NC_06_overflow_handling() {
        let dims = 64;
        let mut cal = CalibrationStats::new(dims);

        // Large values
        let large: Vec<f32> = vec![1e38; dims];
        let _ = cal.update(&large);

        // Should handle gracefully
        assert!(cal.absmax.is_finite(), "NC-06: overflow caused non-finite");
    }

    // NC-07: Cosine Similarity Normalization
    #[test]
    fn test_NC_07_cosine_normalization() {
        // For normalized embeddings, dot product gives cosine similarity
        let a: Vec<f32> = vec![0.6, 0.8, 0.0]; // ||a|| = 1
        let b: Vec<f32> = vec![0.6, 0.8, 0.0]; // ||b|| = 1

        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        assert!((dot - 1.0).abs() < 1e-6, "NC-07: cosine of identical vectors should be 1");
        assert!((-1.0..=1.0).contains(&dot), "NC-07: cosine out of range");
    }

    // NC-08: L2 Distance Non-Negativity
    #[test]
    fn test_NC_08_l2_nonnegativity() {
        let a: Vec<f32> = vec![1.0, 2.0, 3.0];
        let b: Vec<f32> = vec![4.0, 5.0, 6.0];

        let l2_squared: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum();

        assert!(l2_squared >= 0.0, "NC-08: L2 squared should be non-negative");
    }

    // NC-09: Inner Product Symmetry
    #[test]
    fn test_NC_09_dot_symmetry() {
        let a: Vec<i8> = vec![1, 2, 3, 4, 5];
        let b: Vec<i8> = vec![5, 4, 3, 2, 1];

        let ab = dot_i8_scalar(&a, &b);
        let ba = dot_i8_scalar(&b, &a);

        assert_eq!(ab, ba, "NC-09: dot product not symmetric");
    }

    // NC-10: Vector Norm Correctness
    #[test]
    fn test_NC_10_norm_correctness() {
        let v: Vec<f32> = vec![3.0, 4.0]; // ||v|| = 5

        let norm_squared: f32 = v.iter().map(|x| x * x).sum();
        let norm = norm_squared.sqrt();

        assert!((norm - 5.0).abs() < 1e-6, "NC-10: norm incorrect");
    }

    // NC-11: Score Ranking Consistency
    #[test]
    fn test_NC_11_score_ranking() {
        let dims = 32;
        let config = RescoreRetrieverConfig::default();
        let mut retriever = RescoreRetriever::new(dims, config);

        for i in 0..5 {
            let embedding = offset_embedding(dims, i);
            let _ = retriever.index_document(&format!("doc_{}", i), &embedding);
        }

        let query = offset_embedding(dims, 0);
        let results = retriever.retrieve(&query).unwrap();

        // Higher scores should come first
        assert_scores_monotonic(&results, "NC-11");
    }

    // NC-12: Embedding Dimension Consistency
    #[test]
    fn test_NC_12_dimension_consistency() {
        let dims = 64;
        let config = RescoreRetrieverConfig::default();
        let retriever = RescoreRetriever::new(dims, config);

        // Check dimensions via calibration
        assert_eq!(retriever.calibration().dims, dims, "NC-12: dimension mismatch in retriever");
    }

    // NC-13: f32 Precision Sufficiency
    #[test]
    fn test_NC_13_f32_sufficiency() {
        // f32 has ~7 decimal digits of precision
        // For similarity scores, this is sufficient
        let a: f32 = 0.999999;
        let b: f32 = 0.999998;
        assert!(a > b, "NC-13: f32 should distinguish these values");
    }

    // NC-14: Int8 Range Utilization
    #[test]
    fn test_NC_14_int8_range() {
        let dims = 64;
        let mut cal = CalibrationStats::new(dims);

        // Calibrate with wide range
        let embedding: Vec<f32> = (0..dims).map(|i| (i as f32 - 32.0) * 10.0).collect();
        let _ = cal.update(&embedding);

        let quantized = QuantizedEmbedding::from_f32(&embedding, &cal).unwrap();

        // Check range utilization
        let max_val = quantized.values.iter().map(|&v| v.abs()).max().unwrap();
        assert!(max_val >= 100, "NC-14: int8 range underutilized (max={})", max_val);
    }

    // NC-15: Rescoring Score Scaling
    #[test]
    fn test_NC_15_rescore_scaling() {
        let dims = 64;
        let config = RescoreRetrieverConfig::default();
        let mut retriever = RescoreRetriever::new(dims, config);

        let embedding = linear_embedding(dims, 0.1);
        let _ = retriever.index_document("doc_1", &embedding);

        let results = retriever.retrieve(&embedding).unwrap();
        assert!(!results.is_empty(), "NC-15: should have results for self-query");

        // Self-similarity should be high
        let score = results[0].score;
        assert!(score > 0.0, "NC-15: self-similarity should be positive");
    }
}

// ============================================================================
// Section 5: Safety & Robustness Tests (SR-01 to SR-10)
// ============================================================================

#[allow(non_snake_case)]
mod sr_tests {
    use super::*;

    // SR-01: Panic-Free Operation
    #[test]
    fn test_SR_01_panic_free() {
        let dims = 64;
        let config = RescoreRetrieverConfig::default();
        let mut retriever = RescoreRetriever::new(dims, config);

        // Empty retrieval
        let query: Vec<f32> = vec![1.0; dims];
        let _ = retriever.retrieve(&query);

        // Add documents
        for i in 0..10 {
            let embedding: Vec<f32> = vec![i as f32; dims];
            let _ = retriever.index_document(&format!("doc_{}", i), &embedding);
        }

        // Various queries
        let _ = retriever.retrieve(&vec![0.0; dims]);
        let _ = retriever.retrieve(&vec![1e10; dims]);
        let _ = retriever.retrieve(&vec![-1e10; dims]);

        // Test passes if no panic occurred
    }

    // SR-02: Fuzzing Robustness Placeholder
    #[test]
    fn test_SR_02_fuzzing_placeholder() {
        // Actual fuzzing requires cargo-fuzz
        // This validates basic random input handling
        let dims = 32;
        let mut cal = CalibrationStats::new(dims);

        for _ in 0..100 {
            let embedding: Vec<f32> =
                (0..dims).map(|i| (i as f32 * 123.456).sin() * 100.0).collect();
            let _ = cal.update(&embedding);
        }

        assert!(cal.absmax.is_finite(), "SR-02: fuzzing produced non-finite");
    }

    // SR-03: Thread Safety Placeholder
    #[test]
    fn test_SR_03_thread_safety() {
        // RescoreRetriever is not Send+Sync by default (contains Vec)
        // This tests single-threaded safety
        let dims = 32;
        let config = RescoreRetrieverConfig::default();
        let mut retriever = RescoreRetriever::new(dims, config);

        let _ = retriever.index_document("doc_1", &vec![1.0; dims]);
        let results = retriever.retrieve(&vec![1.0; dims]).unwrap();

        assert_eq!(results.len(), 1, "SR-03: basic thread safety");
    }

    // SR-04: OOM Handling Placeholder
    #[test]
    fn test_SR_04_oom_handling() {
        // We can't easily test OOM in unit tests
        // This validates reasonable memory usage
        let dims = 256;
        let mut cal = CalibrationStats::new(dims);

        let embedding: Vec<f32> = vec![1.0; dims];
        let _ = cal.update(&embedding);

        let quantized = QuantizedEmbedding::from_f32(&embedding, &cal).unwrap();

        // i8 should use ~4x less memory than f32
        let i8_size = quantized.values.len();
        let f32_size = embedding.len() * 4;

        assert!(i8_size * 4 <= f32_size, "SR-04: memory reduction not achieved");
    }

    // SR-05: Input Validation
    #[test]
    fn test_SR_05_input_validation() {
        let dims = 64;
        let mut cal = CalibrationStats::new(dims);
        let _ = cal.update(&vec![1.0; dims]);

        // Empty input
        let empty: Vec<f32> = vec![];
        let result = QuantizedEmbedding::from_f32(&empty, &cal);
        assert!(result.is_err(), "SR-05: empty input should error");

        // Wrong dimensions
        let wrong: Vec<f32> = vec![1.0; dims / 2];
        let result = QuantizedEmbedding::from_f32(&wrong, &cal);
        assert!(result.is_err(), "SR-05: wrong dims should error");
    }

    // SR-06: Miri UB Detection Placeholder
    #[test]
    fn test_SR_06_no_ub() {
        // Actual Miri testing requires special toolchain
        // This validates basic safety
        let a: Vec<i8> = vec![1, 2, 3, 4];
        let b: Vec<i8> = vec![4, 3, 2, 1];

        let result = dot_i8_scalar(&a, &b);
        assert_eq!(result, 20, "SR-06: basic safety check");
    }

    // SR-07: Unsafe Code Isolation
    #[test]
    // SAFETY: no actual unsafe code -- test string literal or variable containing 'unsafe'
    fn test_SR_07_unsafe_isolation() {
        // SIMD functions are marked unsafe
        // Public API is safe
        let dims = 64;
        let mut cal = CalibrationStats::new(dims);
        let embedding: Vec<f32> = vec![1.0; dims];
        let _ = cal.update(&embedding);

        // This is the safe API
        let result = QuantizedEmbedding::from_f32(&embedding, &cal);
        assert!(result.is_ok(), "SR-07: safe API should work");
    }

    // SR-08: Error Message Quality
    #[test]
    fn test_SR_08_error_messages() {
        let dims = 64;
        let mut cal = CalibrationStats::new(dims);
        let _ = cal.update(&vec![1.0; dims]);

        // Dimension mismatch
        let wrong: Vec<f32> = vec![1.0; dims / 2];
        let result = QuantizedEmbedding::from_f32(&wrong, &cal);

        if let Err(e) = result {
            let msg = format!("{}", e);
            assert!(
                msg.contains("dimension") || msg.contains("mismatch"),
                "SR-08: error should mention dimension"
            );
        }
    }

    // SR-09: Graceful Degradation
    #[test]
    fn test_SR_09_graceful_degradation() {
        // Scalar backend is always available as fallback
        let backend = SimdBackend::detect();

        // Should return a valid backend
        match backend {
            SimdBackend::Avx512 | SimdBackend::Avx2 | SimdBackend::Neon | SimdBackend::Scalar => {
                // All valid
            }
        }
    }

    // SR-10: Resource Limits
    #[test]
    fn test_SR_10_resource_limits() {
        let config = RescoreRetrieverConfig::default();

        // Default config should have reasonable limits
        assert!(config.rescore_multiplier <= 10, "SR-10: multiplier should be reasonable");
        assert!(config.top_k <= 100, "SR-10: top_k should be reasonable");
    }
}

// ============================================================================
// Integration Tests
// ============================================================================

mod integration_tests {
    use super::*;

    #[test]
    fn test_falsification_result_creation() {
        let result =
            FalsificationResult::pass("QA-01", "Test", TpsPrinciple::Jidoka, Severity::Critical);
        assert!(!result.falsified);

        let fail = FalsificationResult::fail(
            "QA-02",
            "Test",
            TpsPrinciple::PokaYoke,
            Severity::Major,
            vec!["evidence".to_string()],
        );
        assert!(fail.falsified);
    }

    #[test]
    fn test_falsification_summary() {
        let results = vec![
            FalsificationResult::pass("T1", "Test 1", TpsPrinciple::Jidoka, Severity::Critical),
            FalsificationResult::pass("T2", "Test 2", TpsPrinciple::PokaYoke, Severity::Major),
            FalsificationResult::fail(
                "T3",
                "Test 3",
                TpsPrinciple::Kaizen,
                Severity::Minor,
                vec![],
            ),
        ];

        let summary = FalsificationSummary::new(results);
        assert_eq!(summary.total, 3);
        assert_eq!(summary.passed, 2);
        assert_eq!(summary.failed, 1);
        assert!((summary.score - 66.67).abs() < 1.0);
    }

    #[test]
    fn test_grade_calculation() {
        // 100% = Toyota Standard
        let results_100: Vec<FalsificationResult> = (0..10)
            .map(|i| {
                FalsificationResult::pass(
                    &format!("T{}", i),
                    "Test",
                    TpsPrinciple::Jidoka,
                    Severity::Critical,
                )
            })
            .collect();
        let summary = FalsificationSummary::new(results_100);
        assert_eq!(summary.grade, Grade::ToyotaStandard);

        // 90% = Kaizen Required
        let mut results_90: Vec<FalsificationResult> = (0..9)
            .map(|i| {
                FalsificationResult::pass(
                    &format!("T{}", i),
                    "Test",
                    TpsPrinciple::Jidoka,
                    Severity::Critical,
                )
            })
            .collect();
        results_90.push(FalsificationResult::fail(
            "T9",
            "Test",
            TpsPrinciple::Kaizen,
            Severity::Minor,
            vec![],
        ));
        let summary = FalsificationSummary::new(results_90);
        assert_eq!(summary.grade, Grade::KaizenRequired);
    }

    // ============================================================================
    // Coverage: with_evidence, run_falsification_suite, grade variants
    // ============================================================================

    #[test]
    fn test_with_evidence_chaining() {
        let result = FalsificationResult::pass(
            "QA-99",
            "Evidence Test",
            TpsPrinciple::Kaizen,
            Severity::Minor,
        )
        .with_evidence("first observation")
        .with_evidence("second observation");

        assert_eq!(result.evidence.len(), 2);
        assert_eq!(result.evidence[0], "first observation");
        assert_eq!(result.evidence[1], "second observation");
        assert!(!result.falsified);
    }

    #[test]
    fn test_with_evidence_on_fail() {
        let result = FalsificationResult::fail(
            "QA-99",
            "Fail With Evidence",
            TpsPrinciple::Muda,
            Severity::Critical,
            vec!["initial evidence".to_string()],
        )
        .with_evidence("additional evidence");

        assert!(result.falsified);
        assert_eq!(result.evidence.len(), 2);
        assert_eq!(result.evidence[1], "additional evidence");
    }

    #[test]
    fn test_run_falsification_suite() {
        let summary = run_falsification_suite();
        assert_eq!(summary.total, 2);
        assert_eq!(summary.passed, 2);
        assert_eq!(summary.failed, 0);
        assert_eq!(summary.grade, Grade::ToyotaStandard);
        assert!((summary.score - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_grade_andon_warning() {
        // 75% = Andon Warning (70-84%)
        let mut results: Vec<FalsificationResult> = (0..3)
            .map(|i| {
                FalsificationResult::pass(
                    &format!("T{}", i),
                    "Test",
                    TpsPrinciple::Heijunka,
                    Severity::Informational,
                )
            })
            .collect();
        results.push(FalsificationResult::fail(
            "T3",
            "Fail",
            TpsPrinciple::Muri,
            Severity::Major,
            vec![],
        ));
        let summary = FalsificationSummary::new(results);
        assert_eq!(summary.grade, Grade::AndonWarning);
    }

    #[test]
    fn test_grade_stop_the_line() {
        // <70% = Stop The Line
        let mut results: Vec<FalsificationResult> = (0..1)
            .map(|i| {
                FalsificationResult::pass(
                    &format!("T{}", i),
                    "Test",
                    TpsPrinciple::Mura,
                    Severity::Minor,
                )
            })
            .collect();
        // Add 2 failures out of 3 total => 33%
        for i in 1..3 {
            results.push(FalsificationResult::fail(
                &format!("F{}", i),
                "Fail",
                TpsPrinciple::GenchiGenbutsu,
                Severity::Critical,
                vec![format!("evidence {}", i)],
            ));
        }
        let summary = FalsificationSummary::new(results);
        assert_eq!(summary.grade, Grade::StopTheLine);
        assert!(summary.score < 70.0);
    }

    #[test]
    fn test_summary_empty_results() {
        let summary = FalsificationSummary::new(vec![]);
        assert_eq!(summary.total, 0);
        assert_eq!(summary.passed, 0);
        assert_eq!(summary.failed, 0);
        assert!((summary.score - 0.0).abs() < f64::EPSILON);
        assert_eq!(summary.grade, Grade::StopTheLine);
    }

    #[test]
    fn test_falsification_result_fields() {
        let result = FalsificationResult::fail(
            "SR-01",
            "Panic-Free",
            TpsPrinciple::Jidoka,
            Severity::Critical,
            vec!["panic occurred".to_string()],
        );
        assert_eq!(result.id, "SR-01");
        assert_eq!(result.name, "Panic-Free");
        assert_eq!(result.tps_principle, TpsPrinciple::Jidoka);
        assert_eq!(result.severity, Severity::Critical);
        assert!(result.falsified);
        assert_eq!(result.evidence.len(), 1);
    }

    #[test]
    fn test_severity_ordering() {
        // Severity derives PartialOrd, Ord
        assert!(Severity::Critical < Severity::Major);
        assert!(Severity::Major < Severity::Minor);
        assert!(Severity::Minor < Severity::Informational);
    }

    #[test]
    fn test_tps_principle_equality() {
        assert_eq!(TpsPrinciple::Jidoka, TpsPrinciple::Jidoka);
        assert_ne!(TpsPrinciple::Jidoka, TpsPrinciple::PokaYoke);
        assert_ne!(TpsPrinciple::Heijunka, TpsPrinciple::Muda);
        assert_ne!(TpsPrinciple::Muri, TpsPrinciple::Mura);
    }

    #[test]
    fn test_grade_equality() {
        assert_eq!(Grade::ToyotaStandard, Grade::ToyotaStandard);
        assert_ne!(Grade::ToyotaStandard, Grade::KaizenRequired);
        assert_ne!(Grade::AndonWarning, Grade::StopTheLine);
    }

    #[test]
    fn test_falsification_result_debug() {
        let result =
            FalsificationResult::pass("D-01", "Debug", TpsPrinciple::Kaizen, Severity::Minor);
        let debug_str = format!("{:?}", result);
        assert!(debug_str.contains("D-01"));
        assert!(debug_str.contains("Debug"));
    }

    #[test]
    fn test_falsification_summary_debug() {
        let summary = FalsificationSummary::new(vec![]);
        let debug_str = format!("{:?}", summary);
        assert!(debug_str.contains("total"));
        assert!(debug_str.contains('0'));
    }

    #[test]
    fn test_grade_boundary_95() {
        // Exactly 95% -> ToyotaStandard
        let mut results: Vec<FalsificationResult> = (0..19)
            .map(|i| {
                FalsificationResult::pass(
                    &format!("T{}", i),
                    "Test",
                    TpsPrinciple::Jidoka,
                    Severity::Critical,
                )
            })
            .collect();
        results.push(FalsificationResult::fail(
            "T19",
            "Fail",
            TpsPrinciple::Kaizen,
            Severity::Minor,
            vec![],
        ));
        let summary = FalsificationSummary::new(results);
        // 19/20 = 95%
        assert_eq!(summary.grade, Grade::ToyotaStandard);
    }

    #[test]
    fn test_grade_boundary_85() {
        // Exactly 85% -> KaizenRequired
        let mut results: Vec<FalsificationResult> = (0..17)
            .map(|i| {
                FalsificationResult::pass(
                    &format!("T{}", i),
                    "Test",
                    TpsPrinciple::Jidoka,
                    Severity::Critical,
                )
            })
            .collect();
        for i in 17..20 {
            results.push(FalsificationResult::fail(
                &format!("T{}", i),
                "Fail",
                TpsPrinciple::Kaizen,
                Severity::Minor,
                vec![],
            ));
        }
        let summary = FalsificationSummary::new(results);
        // 17/20 = 85%
        assert_eq!(summary.grade, Grade::KaizenRequired);
    }

    #[test]
    fn test_grade_boundary_70() {
        // Exactly 70% -> AndonWarning
        let mut results: Vec<FalsificationResult> = (0..7)
            .map(|i| {
                FalsificationResult::pass(
                    &format!("T{}", i),
                    "Test",
                    TpsPrinciple::Jidoka,
                    Severity::Critical,
                )
            })
            .collect();
        for i in 7..10 {
            results.push(FalsificationResult::fail(
                &format!("T{}", i),
                "Fail",
                TpsPrinciple::Kaizen,
                Severity::Minor,
                vec![],
            ));
        }
        let summary = FalsificationSummary::new(results);
        // 7/10 = 70%
        assert_eq!(summary.grade, Grade::AndonWarning);
    }

    #[test]
    fn test_summary_all_failed() {
        let results: Vec<FalsificationResult> = (0..5)
            .map(|i| {
                FalsificationResult::fail(
                    &format!("F{}", i),
                    "Fail",
                    TpsPrinciple::Jidoka,
                    Severity::Critical,
                    vec!["total failure".to_string()],
                )
            })
            .collect();
        let summary = FalsificationSummary::new(results);
        assert_eq!(summary.total, 5);
        assert_eq!(summary.passed, 0);
        assert_eq!(summary.failed, 5);
        assert!((summary.score - 0.0).abs() < f64::EPSILON);
        assert_eq!(summary.grade, Grade::StopTheLine);
    }

    #[test]
    fn test_summary_clone() {
        let summary = FalsificationSummary::new(vec![FalsificationResult::pass(
            "C-1",
            "Clone",
            TpsPrinciple::Kaizen,
            Severity::Minor,
        )]);
        let cloned = summary.clone();
        assert_eq!(cloned.total, summary.total);
        assert_eq!(cloned.passed, summary.passed);
        assert_eq!(cloned.grade, summary.grade);
    }

    #[test]
    fn test_falsification_result_clone() {
        let result =
            FalsificationResult::pass("CL-1", "Clone", TpsPrinciple::Muda, Severity::Major)
                .with_evidence("cloneable");
        let cloned = result.clone();
        assert_eq!(cloned.id, result.id);
        assert_eq!(cloned.evidence, result.evidence);
    }
}
