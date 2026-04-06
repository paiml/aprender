//! Tests for quantization module
//!
//! Following Popperian falsification checklist from specification

use crate::oracle::rag::quantization::{
    dot_i8_scalar, CalibrationStats, QuantizationError, QuantizationParams, QuantizedEmbedding,
    RescoreRetriever, RescoreRetrieverConfig, SimdBackend,
};

// ========================================================================
// SECTION 1: Quantization Accuracy Tests (QA-01 to QA-15)
// Following Popperian falsification checklist from specification
// ========================================================================

mod quantization_accuracy {
    use super::*;

    /// QA-01: Quantization Error Bound
    /// Claim: |Q(x) - x| <= scale/2
    #[test]
    fn qa_01_quantization_error_bound() {
        let params = QuantizationParams::new(0.1, 384).unwrap();
        let max_error = params.max_error_bound();

        // Test many values
        for i in -1000..=1000 {
            let x = i as f32 * 0.01;
            let q = params.quantize_value(x);
            let dq = params.dequantize_value(q);
            let error = (dq - x).abs();

            // Error should be within bound (with small epsilon for floating point)
            assert!(
                error <= max_error + 1e-6,
                "Error {} exceeds bound {} for x={}",
                error,
                max_error,
                x
            );
        }
    }

    /// QA-02: Symmetric Quantization (zero_point = 0)
    #[test]
    fn qa_02_symmetric_quantization() {
        let params = QuantizationParams::new(0.1, 384).unwrap();
        assert_eq!(params.zero_point, 0, "Zero point must be 0 for symmetric");
    }

    /// QA-03: Calibration Convergence
    #[test]
    fn qa_03_calibration_convergence() {
        let mut cal = CalibrationStats::new(384);

        // Add many samples
        for _ in 0..1000 {
            let embedding: Vec<f32> = (0..384).map(|i| (i as f32 * 0.01).sin()).collect();
            cal.update(&embedding).unwrap();
        }

        // Absmax should stabilize
        let absmax1 = cal.absmax;

        for _ in 0..100 {
            let embedding: Vec<f32> = (0..384).map(|i| (i as f32 * 0.01).sin()).collect();
            cal.update(&embedding).unwrap();
        }

        let absmax2 = cal.absmax;

        // Should not change significantly
        assert!(
            (absmax2 - absmax1).abs() < 0.01,
            "Calibration should converge: {} vs {}",
            absmax1,
            absmax2
        );
    }

    /// QA-04: Overflow Prevention in Dot Product
    #[test]
    fn qa_04_overflow_prevention() {
        let backend = SimdBackend::Scalar;

        // Maximum possible values: 127 x 127 x 4096 = 66,044,672
        // This fits in i32 (max 2,147,483,647)
        let a = vec![127i8; 4096];
        let b = vec![127i8; 4096];

        let result = backend.dot_i8(&a, &b);

        // Should not overflow
        assert_eq!(result, 127 * 127 * 4096);
    }

    /// QA-05: NaN/Inf Handling
    #[test]
    fn qa_05_nan_handling() {
        let cal = CalibrationStats::new(4);
        let embedding = vec![1.0, f32::NAN, 3.0, 4.0];

        let result = QuantizedEmbedding::from_f32(&embedding, &cal);
        assert!(matches!(result, Err(QuantizationError::NonFiniteValue { .. })));
    }

    #[test]
    fn qa_05_inf_handling() {
        let cal = CalibrationStats::new(4);
        let embedding = vec![1.0, f32::INFINITY, 3.0, 4.0];

        let result = QuantizedEmbedding::from_f32(&embedding, &cal);
        assert!(matches!(result, Err(QuantizationError::NonFiniteValue { .. })));
    }

    /// QA-06: Dequantization Reversibility
    #[test]
    fn qa_06_dequantization_reversibility() {
        let mut cal = CalibrationStats::new(4);
        let original = vec![0.5, -0.3, 0.8, -0.1];
        cal.update(&original).unwrap();

        let quantized = QuantizedEmbedding::from_f32(&original, &cal).unwrap();
        let dequantized = quantized.dequantize();

        let params = cal.to_quant_params().unwrap();
        let max_error = params.scale;

        for (i, (&orig, &deq)) in original.iter().zip(dequantized.iter()).enumerate() {
            let error = (orig - deq).abs();
            assert!(
                error <= max_error,
                "Error {} exceeds bound {} at index {}",
                error,
                max_error,
                i
            );
        }
    }

    /// QA-07: Scale Factor Computation
    #[test]
    fn qa_07_scale_computation() {
        let absmax = 12.7;
        let params = QuantizationParams::from_absmax(absmax, 384).unwrap();

        let expected_scale = absmax / 127.0;
        assert!(
            (params.scale - expected_scale).abs() < 1e-6,
            "Scale {} != expected {}",
            params.scale,
            expected_scale
        );
    }

    /// QA-08: Zero Point for Symmetric
    #[test]
    fn qa_08_zero_point_symmetric() {
        let params = QuantizationParams::new(0.1, 384).unwrap();
        assert_eq!(params.zero_point, 0);

        let params2 = QuantizationParams::from_absmax(12.7, 384).unwrap();
        assert_eq!(params2.zero_point, 0);
    }

    /// QA-09: Clipping at Boundaries
    #[test]
    fn qa_09_clipping_boundaries() {
        let params = QuantizationParams::new(0.01, 384).unwrap();

        // Large positive should clip to 127
        let large = 10000.0;
        assert_eq!(params.quantize_value(large), 127);

        // Large negative should clip to -128
        let small = -10000.0;
        assert_eq!(params.quantize_value(small), -128);
    }

    /// QA-10: Dimension Mismatch Detection
    #[test]
    fn qa_10_dimension_mismatch() {
        let cal = CalibrationStats::new(384);
        let wrong_dims = vec![1.0; 256];

        let result = QuantizedEmbedding::from_f32(&wrong_dims, &cal);
        assert!(matches!(result, Err(QuantizationError::DimensionMismatch { .. })));
    }

    /// QA-11: Content Hash Integrity
    #[test]
    fn qa_11_content_hash_integrity() {
        let mut cal = CalibrationStats::new(4);
        let embedding = vec![0.5, -0.3, 0.8, -0.1];
        cal.update(&embedding).unwrap();

        let quantized = QuantizedEmbedding::from_f32(&embedding, &cal).unwrap();
        assert!(quantized.verify_integrity());

        // Tamper with values
        let mut tampered = quantized.clone();
        tampered.values[0] = 99;
        assert!(!tampered.verify_integrity());
    }

    /// QA-12: Welford's Numerical Stability
    #[test]
    fn qa_12_welford_stability() {
        let mut cal = CalibrationStats::new(4);

        // Add many samples with known statistics
        for i in 0..10000 {
            let val = (i as f32 * 0.001).sin();
            let embedding = vec![val; 4];
            cal.update(&embedding).unwrap();
        }

        // Mean should be close to actual mean of sin values
        // Variance should be positive and finite
        for i in 0..4 {
            assert!(cal.mean[i].is_finite());
            assert!(cal.variance(i).is_finite());
            assert!(cal.variance(i) >= 0.0);
        }
    }

    /// QA-13: Quantization Determinism
    #[test]
    fn qa_13_quantization_determinism() {
        let mut cal = CalibrationStats::new(4);
        let embedding = vec![0.5, -0.3, 0.8, -0.1];
        cal.update(&embedding).unwrap();

        let q1 = QuantizedEmbedding::from_f32(&embedding, &cal).unwrap();
        let q2 = QuantizedEmbedding::from_f32(&embedding, &cal).unwrap();

        assert_eq!(q1.values, q2.values);
        assert_eq!(q1.hash, q2.hash);
    }

    /// QA-14: Memory Layout (contiguous)
    #[test]
    fn qa_14_memory_layout() {
        let mut cal = CalibrationStats::new(384);
        let embedding: Vec<f32> = (0..384).map(|i| i as f32 * 0.01).collect();
        cal.update(&embedding).unwrap();

        let quantized = QuantizedEmbedding::from_f32(&embedding, &cal).unwrap();

        // Values should be contiguous
        assert_eq!(quantized.values.len(), 384);
        assert_eq!(quantized.values.capacity(), 384);
    }

    /// QA-15: Batch vs Individual Consistency
    #[test]
    fn qa_15_batch_consistency() {
        let mut cal1 = CalibrationStats::new(4);
        let mut cal2 = CalibrationStats::new(4);

        let embeddings: Vec<Vec<f32>> =
            (0..100).map(|i| vec![(i as f32 * 0.01).sin(); 4]).collect();

        // Individual updates
        for emb in &embeddings {
            cal1.update(emb).unwrap();
        }

        // Batch update
        cal2.update_batch(&embeddings).unwrap();

        assert!((cal1.absmax - cal2.absmax).abs() < 1e-6);
        assert_eq!(cal1.n_samples, cal2.n_samples);
        for i in 0..4 {
            assert!((cal1.mean[i] - cal2.mean[i]).abs() < 1e-6);
        }
    }
}

// ========================================================================
// SECTION 2: Retrieval Accuracy Tests (RA-01 to RA-15)
// ========================================================================

mod retrieval_accuracy {
    use super::*;

    /// RA-01: Accuracy Retention
    #[test]
    fn ra_01_retrieval_returns_results() {
        let config = RescoreRetrieverConfig {
            rescore_multiplier: 4,
            top_k: 5,
            min_calibration_samples: 1,
            simd_backend: Some(SimdBackend::Scalar),
        };
        let mut retriever = RescoreRetriever::new(4, config);

        // Index some documents
        for i in 0..10 {
            let emb = vec![i as f32 * 0.1; 4];
            retriever.index_document(&format!("doc{}", i), &emb).unwrap();
        }

        let query = vec![0.5; 4];
        let results = retriever.retrieve(&query).unwrap();

        assert!(!results.is_empty());
        assert!(results.len() <= 5);
    }

    /// RA-02: Rescore Multiplier Effect
    #[test]
    fn ra_02_rescore_multiplier() {
        let config = RescoreRetrieverConfig {
            rescore_multiplier: 4,
            top_k: 2,
            min_calibration_samples: 1,
            simd_backend: Some(SimdBackend::Scalar),
        };
        let mut retriever = RescoreRetriever::new(4, config);

        for i in 0..10 {
            let emb = vec![i as f32 * 0.1; 4];
            retriever.index_document(&format!("doc{}", i), &emb).unwrap();
        }

        // Stage 1 should retrieve 4 x 2 = 8 candidates
        let query = vec![0.5; 4];
        let results = retriever.retrieve(&query).unwrap();

        // Should return exactly top_k
        assert_eq!(results.len(), 2);
    }

    /// RA-05: Empty Index Handling
    #[test]
    fn ra_05_empty_index() {
        let config = RescoreRetrieverConfig::default();
        let mut retriever = RescoreRetriever::new(4, config);

        // Add calibration sample so we can query
        retriever.add_calibration_sample(&[1.0, 2.0, 3.0, 4.0]).unwrap();

        let query = vec![0.5; 4];
        let results = retriever.retrieve(&query).unwrap();

        // No documents indexed, so results should be empty
        assert!(results.is_empty());
    }

    /// RA-08: Empty Query Handling
    #[test]
    fn ra_08_empty_query() {
        let config = RescoreRetrieverConfig::default();
        let mut retriever = RescoreRetriever::new(4, config);

        retriever.index_document("doc1", &[1.0, 2.0, 3.0, 4.0]).unwrap();

        // Empty query should fail with dimension mismatch
        let empty_query: Vec<f32> = vec![];
        let result = retriever.retrieve(&empty_query);
        assert!(matches!(result, Err(QuantizationError::DimensionMismatch { .. })));
    }

    /// RA-11: Score Ordering
    #[test]
    fn ra_11_score_ordering() {
        let config = RescoreRetrieverConfig {
            rescore_multiplier: 4,
            top_k: 5,
            min_calibration_samples: 1,
            simd_backend: Some(SimdBackend::Scalar),
        };
        let mut retriever = RescoreRetriever::new(4, config);

        for i in 0..10 {
            let emb = vec![i as f32 * 0.1; 4];
            retriever.index_document(&format!("doc{}", i), &emb).unwrap();
        }

        let query = vec![0.5; 4];
        let results = retriever.retrieve(&query).unwrap();

        // Results should be sorted descending by score
        for i in 1..results.len() {
            assert!(
                results[i - 1].score >= results[i].score,
                "Results not sorted: {} < {}",
                results[i - 1].score,
                results[i].score
            );
        }
    }
}

// ========================================================================
// SECTION 3: Performance Tests (PF-01 to PF-15)
// ========================================================================

mod performance {
    use super::*;

    /// PF-02: Memory Reduction (4x)
    #[test]
    fn pf_02_memory_reduction() {
        let f32_size = 384 * 4; // 384 dims x 4 bytes
        let mut cal = CalibrationStats::new(384);
        let embedding: Vec<f32> = (0..384).map(|i| i as f32 * 0.01).collect();
        cal.update(&embedding).unwrap();

        let quantized = QuantizedEmbedding::from_f32(&embedding, &cal).unwrap();
        let i8_size = quantized.memory_size();

        // Should be significantly less than f32 (at least 2x, ideally ~4x)
        assert!(
            i8_size < f32_size,
            "Int8 size {} should be less than f32 size {}",
            i8_size,
            f32_size
        );
    }

    /// PF-05: SIMD Backend Detection
    #[test]
    fn pf_05_simd_detection() {
        let backend = SimdBackend::detect();

        // Should detect some backend
        match backend {
            SimdBackend::Avx512 => println!("Detected AVX-512"),
            SimdBackend::Avx2 => println!("Detected AVX2"),
            SimdBackend::Neon => println!("Detected NEON"),
            SimdBackend::Scalar => println!("Using scalar fallback"),
        }

        // Backend should work
        let a = vec![1i8; 32];
        let b = vec![2i8; 32];
        let result = backend.dot_i8(&a, &b);
        assert_eq!(result, 64); // 1 x 2 x 32 = 64
    }

    /// PF-06: Scalar Dot Product Correctness
    #[test]
    fn pf_06_scalar_dot_product() {
        let a = vec![1i8, 2, 3, 4];
        let b = vec![5i8, 6, 7, 8];

        let result = dot_i8_scalar(&a, &b);
        assert_eq!(result, 5 + 2 * 6 + 3 * 7 + 4 * 8);
    }

    /// Test SIMD backends produce same results as scalar
    #[test]
    fn pf_simd_matches_scalar() {
        let backend = SimdBackend::detect();

        // Test various sizes
        for size in [16, 32, 64, 128, 384, 1024] {
            let a: Vec<i8> = (0..size).map(|i| (i % 127) as i8).collect();
            let b: Vec<i8> = (0..size).map(|i| ((i * 7) % 127) as i8).collect();

            let scalar_result = dot_i8_scalar(&a, &b);
            let simd_result = backend.dot_i8(&a, &b);

            assert_eq!(
                scalar_result, simd_result,
                "SIMD mismatch for size {}: {} vs {}",
                size, scalar_result, simd_result
            );
        }
    }
}

// ========================================================================
// SECTION 4: Numerical Correctness Tests (NC-01 to NC-15)
// ========================================================================

mod numerical_correctness {
    use super::*;

    /// NC-05: Zero Vector Handling
    #[test]
    fn nc_05_zero_vector_handling() {
        let zero_embedding = vec![0.0f32; 4];
        let result = QuantizedEmbedding::from_f32_uncalibrated(&zero_embedding);
        assert!(result.is_ok());

        let quantized = result.unwrap();
        assert!(quantized.values.iter().all(|&v| v == 0));
    }

    /// NC-09: Dot Product Symmetry
    #[test]
    fn nc_09_dot_product_symmetry() {
        let backend = SimdBackend::Scalar;
        let a = vec![1i8, 2, 3, 4, 5];
        let b = vec![5i8, 4, 3, 2, 1];

        let ab = backend.dot_i8(&a, &b);
        let ba = backend.dot_i8(&b, &a);

        assert_eq!(ab, ba);
    }

    /// NC-13: f32 x i8 Rescoring Correctness
    #[test]
    fn nc_13_f32_i8_dot_product() {
        let backend = SimdBackend::Scalar;
        let query = vec![1.0f32, 2.0, 3.0, 4.0];
        let doc = vec![1i8, 2, 3, 4];
        let scale = 0.1;

        let result = backend.dot_f32_i8(&query, &doc, scale);

        // Expected: (1.0 x 1 x 0.1) + (2.0 x 2 x 0.1) + (3.0 x 3 x 0.1) + (4.0 x 4 x 0.1)
        //         = 0.1 + 0.4 + 0.9 + 1.6 = 3.0
        assert!((result - 3.0).abs() < 1e-6);
    }
}

// ========================================================================
// SECTION 5: Safety & Robustness Tests (SR-01 to SR-10)
// ========================================================================

mod safety_robustness {
    use super::*;

    /// SR-01: No Panics on Valid Input
    #[test]
    fn sr_01_no_panic_valid_input() {
        let mut cal = CalibrationStats::new(4);
        let embeddings = vec![
            vec![0.0, 0.0, 0.0, 0.0],
            vec![1.0, 1.0, 1.0, 1.0],
            vec![-1.0, -1.0, -1.0, -1.0],
            vec![0.5, -0.5, 0.5, -0.5],
        ];

        for emb in &embeddings {
            cal.update(emb).unwrap();
            QuantizedEmbedding::from_f32(emb, &cal).unwrap();
        }
    }

    /// SR-05: Input Validation
    #[test]
    fn sr_05_input_validation() {
        let cal = CalibrationStats::new(4);

        // Empty should fail
        assert!(QuantizedEmbedding::from_f32(&[], &cal).is_err());

        // Wrong dimensions should fail
        assert!(QuantizedEmbedding::from_f32(&[1.0, 2.0], &cal).is_err());

        // NaN should fail
        assert!(QuantizedEmbedding::from_f32(&[1.0, f32::NAN, 3.0, 4.0], &cal).is_err());

        // Inf should fail
        assert!(QuantizedEmbedding::from_f32(&[1.0, f32::INFINITY, 3.0, 4.0], &cal).is_err());
    }

    /// SR-08: Error Message Quality
    #[test]
    fn sr_08_error_messages() {
        let error = QuantizationError::DimensionMismatch { expected: 384, actual: 256 };
        let msg = error.to_string();
        assert!(msg.contains("384"));
        assert!(msg.contains("256"));

        let error = QuantizationError::NonFiniteValue { index: 42, value: f32::NAN };
        let msg = error.to_string();
        assert!(msg.contains("42"));
    }
}

// ========================================================================
// Property-Based Tests
// ========================================================================

mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        /// Quantization error is always bounded
        #[test]
        fn prop_quantization_error_bounded(
            values in prop::collection::vec(-10.0f32..10.0, 1..100)
        ) {
            let quantized = QuantizedEmbedding::from_f32_uncalibrated(&values);
            if let Ok(q) = quantized {
                let dequantized = q.dequantize();
                let max_error = q.params.scale;

                for (i, (&orig, &deq)) in values.iter().zip(dequantized.iter()).enumerate() {
                    let error = (orig - deq).abs();
                    prop_assert!(
                        error <= max_error + 1e-5,
                        "Error {} > bound {} at index {}",
                        error, max_error, i
                    );
                }
            }
        }

        /// Dot product is symmetric
        #[test]
        fn prop_dot_symmetric(
            a in prop::collection::vec(-128i8..127, 1..100),
            b_seed in 0u64..1000
        ) {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};

            // Generate b with same length as a
            let mut hasher = DefaultHasher::new();
            b_seed.hash(&mut hasher);
            let seed = hasher.finish();

            let b: Vec<i8> = (0..a.len())
                .map(|i| ((seed.wrapping_add(i as u64) % 256) as i32 - 128) as i8)
                .collect();

            let backend = SimdBackend::Scalar;
            let ab = backend.dot_i8(&a, &b);
            let ba = backend.dot_i8(&b, &a);

            prop_assert_eq!(ab, ba);
        }

        /// Calibration n_samples increases monotonically
        #[test]
        fn prop_calibration_samples_monotonic(
            embeddings in prop::collection::vec(
                prop::collection::vec(-1.0f32..1.0, 4..5),
                1..50
            )
        ) {
            let mut cal = CalibrationStats::new(4);
            let mut prev_samples = 0;

            for emb in &embeddings {
                if emb.len() == cal.dims
                    && cal.update(emb).is_ok() {
                        prop_assert!(cal.n_samples > prev_samples);
                        prev_samples = cal.n_samples;
                    }
            }
        }

        /// Retriever returns at most top_k results
        #[test]
        fn prop_retriever_respects_top_k(
            num_docs in 1usize..50,
            top_k in 1usize..20
        ) {
            let config = RescoreRetrieverConfig {
                rescore_multiplier: 4,
                top_k,
                min_calibration_samples: 1,
                simd_backend: Some(SimdBackend::Scalar),
            };
            let mut retriever = RescoreRetriever::new(4, config);

            for i in 0..num_docs {
                let emb = vec![(i as f32 * 0.1).sin(); 4];
                let _ = retriever.index_document(&format!("doc{}", i), &emb);
            }

            let query = vec![0.5; 4];
            if let Ok(results) = retriever.retrieve(&query) {
                prop_assert!(results.len() <= top_k);
            }
        }

        /// Memory usage is less than f32 equivalent
        #[test]
        fn prop_memory_reduction(
            dims in 64usize..1024
        ) {
            let f32_size = dims * 4;

            let mut cal = CalibrationStats::new(dims);
            let embedding: Vec<f32> = (0..dims).map(|i| (i as f32 * 0.01).sin()).collect();
            let _ = cal.update(&embedding);

            if let Ok(quantized) = QuantizedEmbedding::from_f32(&embedding, &cal) {
                let _i8_size = quantized.memory_size();
                // Overhead for params and hash, but core data should be smaller
                prop_assert!(
                    quantized.values.len() < f32_size,
                    "Raw values {} should be less than f32 {}",
                    quantized.values.len(),
                    f32_size
                );
            }
        }
    }
}

// ========================================================================
// SECTION 6: Explicit SIMD Backend Coverage
// ========================================================================

mod simd_backend_coverage {
    use super::*;

    /// Cover AVX2 path by explicitly constructing SimdBackend::Avx2
    #[test]
    fn avx2_dot_i8_matches_scalar() {
        let backend = SimdBackend::Avx2;

        for size in [16, 32, 33, 64, 100, 128, 384, 1024] {
            let a: Vec<i8> = (0..size).map(|i| ((i * 3) % 127) as i8).collect();
            let b: Vec<i8> = (0..size).map(|i| ((i * 7 + 13) % 127) as i8).collect();

            let scalar_result = dot_i8_scalar(&a, &b);
            let avx2_result = backend.dot_i8(&a, &b);

            assert_eq!(
                scalar_result, avx2_result,
                "AVX2 mismatch for size {}: scalar={} avx2={}",
                size, scalar_result, avx2_result
            );
        }
    }

    /// Cover AVX2 with negative values
    #[test]
    fn avx2_dot_i8_negative_values() {
        let backend = SimdBackend::Avx2;

        let a: Vec<i8> = (0..64).map(|i| -((i % 128) as i8)).collect();
        let b: Vec<i8> = (0..64).map(|i| ((i * 5) % 127) as i8).collect();

        let scalar_result = dot_i8_scalar(&a, &b);
        let avx2_result = backend.dot_i8(&a, &b);
        assert_eq!(scalar_result, avx2_result);
    }

    /// Cover AVX2 with maximum magnitude values
    #[test]
    fn avx2_dot_i8_max_values() {
        let backend = SimdBackend::Avx2;

        let a = vec![127i8; 4096];
        let b = vec![127i8; 4096];

        let expected = 127 * 127 * 4096;
        let result = backend.dot_i8(&a, &b);
        assert_eq!(result, expected);
    }

    /// Cover AVX2 remainder path (non-multiple of 32)
    #[test]
    fn avx2_dot_i8_remainder() {
        let backend = SimdBackend::Avx2;

        // 35 elements: 32 SIMD + 3 remainder
        let a: Vec<i8> = (0..35).map(|i| (i + 1) as i8).collect();
        let b: Vec<i8> = (0..35).map(|i| (i + 1) as i8).collect();

        let scalar_result = dot_i8_scalar(&a, &b);
        let avx2_result = backend.dot_i8(&a, &b);
        assert_eq!(scalar_result, avx2_result);
    }

    /// Cover AVX2 with zero-length input
    #[test]
    fn avx2_dot_i8_empty() {
        let backend = SimdBackend::Avx2;
        let result = backend.dot_i8(&[], &[]);
        assert_eq!(result, 0);
    }

    /// Cover AVX2 with sub-SIMD-width input (< 32 elements)
    #[test]
    fn avx2_dot_i8_small() {
        let backend = SimdBackend::Avx2;
        let a = vec![10i8; 5];
        let b = vec![20i8; 5];

        let result = backend.dot_i8(&a, &b);
        assert_eq!(result, 10 * 20 * 5);
    }

    /// Cover AVX-512 path by explicitly constructing SimdBackend::Avx512
    #[test]
    fn avx512_dot_i8_matches_scalar() {
        let backend = SimdBackend::Avx512;

        for size in [32, 64, 65, 128, 384, 1024] {
            let a: Vec<i8> = (0..size).map(|i| ((i * 3) % 127) as i8).collect();
            let b: Vec<i8> = (0..size).map(|i| ((i * 7 + 13) % 127) as i8).collect();

            let scalar_result = dot_i8_scalar(&a, &b);
            let avx512_result = backend.dot_i8(&a, &b);

            assert_eq!(
                scalar_result, avx512_result,
                "AVX512 mismatch for size {}: scalar={} avx512={}",
                size, scalar_result, avx512_result
            );
        }
    }

    /// Cover AVX-512 remainder path (non-multiple of 64)
    #[test]
    fn avx512_dot_i8_remainder() {
        let backend = SimdBackend::Avx512;

        // 70 elements: 64 SIMD + 6 remainder
        let a: Vec<i8> = (0..70).map(|i| (i + 1) as i8).collect();
        let b: Vec<i8> = (0..70).map(|i| (i + 1) as i8).collect();

        let scalar_result = dot_i8_scalar(&a, &b);
        let avx512_result = backend.dot_i8(&a, &b);
        assert_eq!(scalar_result, avx512_result);
    }

    /// Cover f32_i8 dot product with detected backend
    #[test]
    fn f32_i8_dot_detected_backend() {
        let backend = SimdBackend::detect();
        let query = vec![1.0f32, 0.5, -1.0, 2.0];
        let doc = vec![10i8, 20, -30, 40];
        let scale = 0.1;

        let result = backend.dot_f32_i8(&query, &doc, scale);
        // (1.0*10*0.1) + (0.5*20*0.1) + (-1.0*-30*0.1) + (2.0*40*0.1)
        // = 1.0 + 1.0 + 3.0 + 8.0 = 13.0
        assert!((result - 13.0).abs() < 1e-5);
    }

    /// Cover all backend variants have consistent f32_i8 results
    #[test]
    fn f32_i8_all_backends_consistent() {
        let query: Vec<f32> = (0..384).map(|i| (i as f32 * 0.01).sin()).collect();
        let doc: Vec<i8> = (0..384).map(|i| ((i * 3) % 127) as i8).collect();
        let scale = 0.05;

        let scalar_result = SimdBackend::Scalar.dot_f32_i8(&query, &doc, scale);
        let avx2_result = SimdBackend::Avx2.dot_f32_i8(&query, &doc, scale);
        let avx512_result = SimdBackend::Avx512.dot_f32_i8(&query, &doc, scale);

        // f32_i8 uses the same scalar loop for all backends
        assert!((scalar_result - avx2_result).abs() < 1e-5);
        assert!((scalar_result - avx512_result).abs() < 1e-5);
    }
}

// ========================================================================
// Integration Tests
// ========================================================================

mod integration {
    use super::*;

    /// Full pipeline test: calibrate -> index -> retrieve
    #[test]
    fn full_pipeline() {
        let config = RescoreRetrieverConfig {
            rescore_multiplier: 4,
            top_k: 3,
            min_calibration_samples: 10,
            simd_backend: Some(SimdBackend::Scalar),
        };
        let mut retriever = RescoreRetriever::new(384, config);

        // Add calibration samples
        for i in 0..100 {
            let emb: Vec<f32> = (0..384).map(|j| ((i * j) as f32 * 0.001).sin()).collect();
            retriever.add_calibration_sample(&emb).unwrap();
        }

        // Index documents
        for i in 0..50 {
            let emb: Vec<f32> = (0..384).map(|j| ((i + j) as f32 * 0.01).cos()).collect();
            retriever.index_document(&format!("doc{}", i), &emb).unwrap();
        }

        // Query
        let query: Vec<f32> = (0..384).map(|i| (i as f32 * 0.01).sin()).collect();
        let results = retriever.retrieve(&query).unwrap();

        // Should return top_k results
        assert_eq!(results.len(), 3);

        // Results should be sorted
        for i in 1..results.len() {
            assert!(results[i - 1].score >= results[i].score);
        }

        // All results should have valid doc_ids
        for result in &results {
            assert!(result.doc_id.starts_with("doc"));
        }
    }

    /// Test calibration -> quantization -> retrieval consistency
    #[test]
    fn calibration_consistency() {
        let mut cal = CalibrationStats::new(4);

        // Build calibration from known data
        let samples = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
        ];

        for sample in &samples {
            cal.update(sample).unwrap();
        }

        assert_eq!(cal.n_samples, 4);
        assert!((cal.absmax - 1.0).abs() < 1e-6);

        // Quantize and verify
        for sample in &samples {
            let q = QuantizedEmbedding::from_f32(sample, &cal).unwrap();
            assert!(q.verify_integrity());
        }
    }
}

// ========================================================================
// SECTION 7: Error Module Coverage (error.rs)
// ========================================================================

mod error_coverage {
    use super::*;
    use crate::oracle::rag::quantization::validate_embedding;

    /// Cover Display for EmptyEmbedding
    #[test]
    fn display_empty_embedding() {
        let err = QuantizationError::EmptyEmbedding;
        let msg = err.to_string();
        assert_eq!(msg, "Empty embedding");
    }

    /// Cover Display for CalibrationNotInitialized
    #[test]
    fn display_calibration_not_initialized() {
        let err = QuantizationError::CalibrationNotInitialized;
        let msg = err.to_string();
        assert_eq!(msg, "Calibration not initialized");
    }

    /// Cover Display for InvalidScale
    #[test]
    fn display_invalid_scale() {
        let err = QuantizationError::InvalidScale { scale: -0.5 };
        let msg = err.to_string();
        assert!(msg.contains("-0.5"));
        assert!(msg.contains("Invalid scale"));
    }

    /// Cover Display for ComputationOverflow
    #[test]
    fn display_computation_overflow() {
        let err = QuantizationError::ComputationOverflow;
        let msg = err.to_string();
        assert_eq!(msg, "Computation overflow");
    }

    /// Cover std::error::Error impl
    #[test]
    fn error_trait_impl() {
        let err = QuantizationError::EmptyEmbedding;
        let _: &dyn std::error::Error = &err;
        // source() returns None by default
        assert!(std::error::Error::source(&err).is_none());
    }

    /// Cover validate_embedding with matching empty (0-dim expected, 0-dim actual)
    /// This tests the path where len == expected_dims but embedding.is_empty()
    #[test]
    fn validate_embedding_zero_dims_is_empty() {
        let result = validate_embedding(&[], 0);
        // len() == expected_dims (0 == 0), but then is_empty() returns EmptyEmbedding
        assert!(matches!(result, Err(QuantizationError::EmptyEmbedding)));
    }

    /// Cover validate_embedding success path
    #[test]
    fn validate_embedding_success() {
        let embedding = vec![1.0, 2.0, 3.0];
        let result = validate_embedding(&embedding, 3);
        assert!(result.is_ok());
    }

    /// Cover validate_embedding dimension mismatch (not empty)
    #[test]
    fn validate_embedding_dimension_mismatch() {
        let embedding = vec![1.0, 2.0];
        let result = validate_embedding(&embedding, 5);
        match result {
            Err(QuantizationError::DimensionMismatch { expected: 5, actual: 2 }) => {}
            other => panic!("Expected DimensionMismatch, got {:?}", other),
        }
    }

    /// Cover validate_embedding with NaN at various indices
    #[test]
    fn validate_embedding_nan_at_last_index() {
        let embedding = vec![1.0, 2.0, f32::NAN];
        let result = validate_embedding(&embedding, 3);
        match result {
            Err(QuantizationError::NonFiniteValue { index: 2, .. }) => {}
            other => panic!("Expected NonFiniteValue at index 2, got {:?}", other),
        }
    }

    /// Cover validate_embedding with negative infinity
    #[test]
    fn validate_embedding_neg_infinity() {
        let embedding = vec![f32::NEG_INFINITY, 2.0, 3.0];
        let result = validate_embedding(&embedding, 3);
        assert!(matches!(result, Err(QuantizationError::NonFiniteValue { index: 0, .. })));
    }

    /// Cover PartialEq derive for QuantizationError
    #[test]
    fn error_partial_eq() {
        let err1 = QuantizationError::EmptyEmbedding;
        let err2 = QuantizationError::EmptyEmbedding;
        let err3 = QuantizationError::ComputationOverflow;
        assert_eq!(err1, err2);
        assert_ne!(err1, err3);
    }

    /// Cover Clone derive for QuantizationError
    #[test]
    fn error_clone() {
        let err = QuantizationError::DimensionMismatch { expected: 10, actual: 5 };
        let cloned = err.clone();
        assert_eq!(err, cloned);
    }
}

// ========================================================================
// SECTION 8: Calibration Edge Cases (calibration.rs)
// ========================================================================

mod calibration_coverage {
    use super::*;

    /// Cover variance with n_samples=0 (returns 0.0)
    #[test]
    fn variance_zero_samples() {
        let cal = CalibrationStats::new(4);
        assert_eq!(cal.variance(0), 0.0);
        assert_eq!(cal.variance(3), 0.0);
    }

    /// Cover variance with n_samples=1 (returns 0.0)
    #[test]
    fn variance_one_sample() {
        let mut cal = CalibrationStats::new(4);
        cal.update(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        assert_eq!(cal.n_samples, 1);
        assert_eq!(cal.variance(0), 0.0);
        assert_eq!(cal.variance(3), 0.0);
    }

    /// Cover variance with index out of bounds (returns 0.0)
    #[test]
    fn variance_out_of_bounds() {
        let mut cal = CalibrationStats::new(4);
        cal.update(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        cal.update(&[2.0, 3.0, 4.0, 5.0]).unwrap();
        // Index 10 is out of bounds for dims=4
        assert_eq!(cal.variance(10), 0.0);
    }

    /// Cover std_dev delegates to variance
    #[test]
    fn std_dev_values() {
        let mut cal = CalibrationStats::new(2);
        cal.update(&[1.0, 2.0]).unwrap();
        cal.update(&[3.0, 4.0]).unwrap();
        let var = cal.variance(0);
        let sd = cal.std_dev(0);
        assert!((sd - var.sqrt()).abs() < 1e-6);
    }

    /// Cover std_dev with zero variance
    #[test]
    fn std_dev_zero_samples() {
        let cal = CalibrationStats::new(4);
        assert_eq!(cal.std_dev(0), 0.0);
    }

    /// Cover is_sufficient true and false paths
    #[test]
    fn is_sufficient_check() {
        let mut cal = CalibrationStats::new(4);
        assert!(!cal.is_sufficient(1));
        assert!(cal.is_sufficient(0));

        cal.update(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        assert!(cal.is_sufficient(1));
        assert!(!cal.is_sufficient(2));

        cal.update(&[2.0, 3.0, 4.0, 5.0]).unwrap();
        assert!(cal.is_sufficient(2));
        assert!(!cal.is_sufficient(3));
    }

    /// Cover to_quant_params with zero absmax (uses 1.0 fallback)
    #[test]
    fn to_quant_params_zero_absmax() {
        let mut cal = CalibrationStats::new(4);
        cal.update(&[0.0, 0.0, 0.0, 0.0]).unwrap();
        assert_eq!(cal.absmax, 0.0);

        let params = cal.to_quant_params().unwrap();
        // absmax was 0.0, so it becomes 1.0, scale = 1.0/127.0
        let expected_scale = 1.0 / 127.0;
        assert!((params.scale - expected_scale).abs() < 1e-6);
    }

    /// Cover to_quant_params with no samples (CalibrationNotInitialized)
    #[test]
    fn to_quant_params_no_samples() {
        let cal = CalibrationStats::new(4);
        let result = cal.to_quant_params();
        assert!(matches!(result, Err(QuantizationError::CalibrationNotInitialized)));
    }

    /// Cover update_batch with empty batch
    #[test]
    fn update_batch_empty() {
        let mut cal = CalibrationStats::new(4);
        let empty: Vec<Vec<f32>> = vec![];
        cal.update_batch(&empty).unwrap();
        assert_eq!(cal.n_samples, 0);
    }

    /// Cover calibration new creates correct dimensions
    #[test]
    fn calibration_new_dims() {
        let cal = CalibrationStats::new(128);
        assert_eq!(cal.dims, 128);
        assert_eq!(cal.mean.len(), 128);
        assert_eq!(cal.m2.len(), 128);
        assert_eq!(cal.n_samples, 0);
        assert_eq!(cal.absmax, 0.0);
    }

    /// Cover variance with 2 samples (denominator = n-1 = 1)
    #[test]
    fn variance_two_samples_welford() {
        let mut cal = CalibrationStats::new(1);
        cal.update(&[2.0]).unwrap();
        cal.update(&[4.0]).unwrap();
        // mean = 3.0, var = ((2-3)^2 + (4-3)^2) / (2-1) = 2.0
        let var = cal.variance(0);
        assert!((var - 2.0).abs() < 1e-5, "Expected variance ~2.0, got {}", var);
    }
}

// ========================================================================
// SECTION 9: Retriever Coverage (retriever.rs)
// ========================================================================

mod retriever_coverage {
    use super::*;

    /// Cover len() and is_empty() methods
    #[test]
    fn len_and_is_empty() {
        let config = RescoreRetrieverConfig {
            rescore_multiplier: 4,
            top_k: 5,
            min_calibration_samples: 1,
            simd_backend: Some(SimdBackend::Scalar),
        };
        let mut retriever = RescoreRetriever::new(4, config);

        assert!(retriever.is_empty());
        assert_eq!(retriever.len(), 0);

        retriever.index_document("doc0", &[1.0, 2.0, 3.0, 4.0]).unwrap();

        assert!(!retriever.is_empty());
        assert_eq!(retriever.len(), 1);
    }

    /// Cover calibration() accessor
    #[test]
    fn calibration_accessor() {
        let config = RescoreRetrieverConfig {
            rescore_multiplier: 4,
            top_k: 5,
            min_calibration_samples: 1,
            simd_backend: Some(SimdBackend::Scalar),
        };
        let mut retriever = RescoreRetriever::new(4, config);

        retriever.index_document("doc0", &[1.0, 2.0, 3.0, 4.0]).unwrap();

        let cal = retriever.calibration();
        assert_eq!(cal.dims, 4);
        assert_eq!(cal.n_samples, 1);
    }

    /// Cover memory_usage()
    #[test]
    fn memory_usage() {
        let config = RescoreRetrieverConfig {
            rescore_multiplier: 4,
            top_k: 5,
            min_calibration_samples: 1,
            simd_backend: Some(SimdBackend::Scalar),
        };
        let mut retriever = RescoreRetriever::new(4, config);

        assert_eq!(retriever.memory_usage(), 0);

        retriever.index_document("doc0", &[1.0, 2.0, 3.0, 4.0]).unwrap();

        assert!(retriever.memory_usage() > 0);
    }

    /// Cover stage1 top-k truncation when more docs than candidates
    #[test]
    fn stage1_truncation_with_many_docs() {
        let config = RescoreRetrieverConfig {
            rescore_multiplier: 2,
            top_k: 2,
            min_calibration_samples: 1,
            simd_backend: Some(SimdBackend::Scalar),
        };
        let mut retriever = RescoreRetriever::new(4, config);

        // Index 20 documents, more than rescore_multiplier * top_k = 4
        for i in 0..20 {
            let emb = vec![(i as f32) * 0.1, 0.0, 0.0, 0.0];
            retriever.index_document(&format!("doc{}", i), &emb).unwrap();
        }

        let query = vec![1.0, 0.0, 0.0, 0.0];
        let results = retriever.retrieve(&query).unwrap();

        // Should return exactly top_k = 2
        assert_eq!(results.len(), 2);

        // Results should be sorted descending
        assert!(results[0].score >= results[1].score);
    }

    /// Cover stage2 rescore precise sorting
    #[test]
    fn stage2_rescore_ordering() {
        let config = RescoreRetrieverConfig {
            rescore_multiplier: 10,
            top_k: 3,
            min_calibration_samples: 1,
            simd_backend: Some(SimdBackend::Scalar),
        };
        let mut retriever = RescoreRetriever::new(4, config);

        // Index docs with known scoring patterns
        retriever.index_document("low", &[0.1, 0.1, 0.1, 0.1]).unwrap();
        retriever.index_document("mid", &[0.5, 0.5, 0.5, 0.5]).unwrap();
        retriever.index_document("high", &[1.0, 1.0, 1.0, 1.0]).unwrap();

        let query = vec![1.0, 1.0, 1.0, 1.0];
        let results = retriever.retrieve(&query).unwrap();

        assert_eq!(results.len(), 3);
        // Highest similarity should be first
        assert_eq!(results[0].doc_id, "high");
    }

    /// Cover RescoreRetrieverConfig::default()
    #[test]
    fn default_config() {
        let config = RescoreRetrieverConfig::default();
        assert_eq!(config.rescore_multiplier, 4);
        assert_eq!(config.top_k, 10);
        assert_eq!(config.min_calibration_samples, 1000);
        assert!(config.simd_backend.is_none());
    }

    /// Cover retriever with auto-detected backend (None simd_backend)
    #[test]
    fn auto_detect_backend() {
        let config = RescoreRetrieverConfig {
            rescore_multiplier: 4,
            top_k: 5,
            min_calibration_samples: 1,
            simd_backend: None, // auto-detect
        };
        let mut retriever = RescoreRetriever::new(4, config);

        retriever.index_document("doc0", &[1.0, 2.0, 3.0, 4.0]).unwrap();

        let results = retriever.retrieve(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        assert_eq!(results.len(), 1);
    }

    /// Cover add_calibration_sample
    #[test]
    fn add_calibration_sample() {
        let config = RescoreRetrieverConfig {
            rescore_multiplier: 4,
            top_k: 5,
            min_calibration_samples: 1,
            simd_backend: Some(SimdBackend::Scalar),
        };
        let mut retriever = RescoreRetriever::new(4, config);

        retriever.add_calibration_sample(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        retriever.add_calibration_sample(&[5.0, 6.0, 7.0, 8.0]).unwrap();

        let cal = retriever.calibration();
        assert_eq!(cal.n_samples, 2);
    }

    /// Cover RescoreResult fields
    #[test]
    fn rescore_result_fields() {
        let config = RescoreRetrieverConfig {
            rescore_multiplier: 4,
            top_k: 5,
            min_calibration_samples: 1,
            simd_backend: Some(SimdBackend::Scalar),
        };
        let mut retriever = RescoreRetriever::new(4, config);

        retriever.index_document("test_doc", &[1.0, 2.0, 3.0, 4.0]).unwrap();

        let results = retriever.retrieve(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        assert_eq!(results.len(), 1);
        let r = &results[0];
        assert_eq!(r.doc_id, "test_doc");
        // approx_score should be an i32 from stage1
        let _approx: i32 = r.approx_score;
        // score should be f32 from stage2
        let _score: f32 = r.score;
    }
}

// ========================================================================
// SECTION 10: SIMD Tail Coverage (simd.rs)
// ========================================================================

mod simd_tail_coverage {
    use super::*;

    /// Cover scalar tail via SIMD backends with non-aligned vector lengths.
    /// AVX2 processes 32 elements at a time, so 33 elements triggers the tail.
    #[test]
    fn avx2_scalar_tail_single_element() {
        let backend = SimdBackend::Avx2;

        // 33 elements = 32 SIMD + 1 scalar tail
        let a: Vec<i8> = (0..33).map(|i| (i + 1) as i8).collect();
        let b: Vec<i8> = (0..33).map(|i| (i + 1) as i8).collect();

        let expected = dot_i8_scalar(&a, &b);
        let result = backend.dot_i8(&a, &b);
        assert_eq!(result, expected);
    }

    /// AVX-512 processes 64 elements, so 65 triggers the tail
    #[test]
    fn avx512_scalar_tail_single_element() {
        let backend = SimdBackend::Avx512;

        // 65 elements = 64 SIMD + 1 scalar tail
        let a: Vec<i8> = (0..65).map(|i| (i + 1) as i8).collect();
        let b: Vec<i8> = (0..65).map(|i| (i + 1) as i8).collect();

        let expected = dot_i8_scalar(&a, &b);
        let result = backend.dot_i8(&a, &b);
        assert_eq!(result, expected);
    }

    /// Cover with lengths that leave various tail sizes for AVX2
    #[test]
    fn avx2_various_tail_sizes() {
        let backend = SimdBackend::Avx2;

        // Test tail sizes 1..31
        for tail in 1..32 {
            let size = 32 + tail; // 32 SIMD + tail remainder
            let a: Vec<i8> = (0..size).map(|i| ((i * 3 + 7) % 127) as i8).collect();
            let b: Vec<i8> = (0..size).map(|i| ((i * 5 + 11) % 127) as i8).collect();

            let expected = dot_i8_scalar(&a, &b);
            let result = backend.dot_i8(&a, &b);
            assert_eq!(
                result, expected,
                "AVX2 tail mismatch for tail size {}: got {} expected {}",
                tail, result, expected
            );
        }
    }

    /// Cover with lengths that leave various tail sizes for AVX-512
    #[test]
    fn avx512_various_tail_sizes() {
        let backend = SimdBackend::Avx512;

        // Test tail sizes 1..63
        for tail in 1..64 {
            let size = 64 + tail; // 64 SIMD + tail remainder
            let a: Vec<i8> = (0..size).map(|i| ((i * 3 + 7) % 127) as i8).collect();
            let b: Vec<i8> = (0..size).map(|i| ((i * 5 + 11) % 127) as i8).collect();

            let expected = dot_i8_scalar(&a, &b);
            let result = backend.dot_i8(&a, &b);
            assert_eq!(
                result, expected,
                "AVX512 tail mismatch for tail size {}: got {} expected {}",
                tail, result, expected
            );
        }
    }

    /// Cover Scalar backend (the _ catch-all in dot_i8)
    #[test]
    fn scalar_backend_catchall() {
        let backend = SimdBackend::Scalar;
        let a = vec![1i8, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let b = vec![10i8, 9, 8, 7, 6, 5, 4, 3, 2, 1];

        let expected: i32 = a.iter().zip(b.iter()).map(|(&x, &y)| x as i32 * y as i32).sum();
        let result = backend.dot_i8(&a, &b);
        assert_eq!(result, expected);
    }

    /// Cover sub-SIMD-width input (less than one SIMD vector width)
    #[test]
    fn sub_width_inputs() {
        // Less than 32 (AVX2 width) and less than 64 (AVX-512 width)
        for &size in &[1, 2, 3, 5, 7, 15, 16, 31, 63] {
            let a: Vec<i8> = (0..size).map(|i| (i as i8) + 1).collect();
            let b: Vec<i8> = (0..size).map(|i| (i as i8) + 1).collect();

            let expected = dot_i8_scalar(&a, &b);

            let avx2_result = SimdBackend::Avx2.dot_i8(&a, &b);
            assert_eq!(avx2_result, expected, "AVX2 sub-width mismatch for size {}", size);

            let avx512_result = SimdBackend::Avx512.dot_i8(&a, &b);
            assert_eq!(avx512_result, expected, "AVX512 sub-width mismatch for size {}", size);

            let scalar_result = SimdBackend::Scalar.dot_i8(&a, &b);
            assert_eq!(scalar_result, expected, "Scalar mismatch for size {}", size);
        }
    }

    /// Cover f32_i8 dot product with various backends and empty vectors
    #[test]
    fn f32_i8_empty() {
        let backend = SimdBackend::Scalar;
        let result = backend.dot_f32_i8(&[], &[], 1.0);
        assert_eq!(result, 0.0);
    }

    /// Cover SimdBackend::detect() and verify it returns a valid backend
    #[test]
    fn detect_returns_valid_backend() {
        let backend = SimdBackend::detect();
        // On x86_64, should be Avx512 or Avx2 (never Neon)
        #[cfg(target_arch = "x86_64")]
        {
            assert!(
                backend == SimdBackend::Avx512
                    || backend == SimdBackend::Avx2
                    || backend == SimdBackend::Scalar,
                "x86_64 should detect Avx512, Avx2, or Scalar"
            );
        }
        // Verify the detected backend can compute dot products
        let a = vec![1i8; 16];
        let b = vec![2i8; 16];
        let result = backend.dot_i8(&a, &b);
        assert_eq!(result, 32);
    }

    /// Cover SimdBackend Debug and PartialEq derives
    #[test]
    fn simd_backend_debug_and_eq() {
        let scalar = SimdBackend::Scalar;
        let avx2 = SimdBackend::Avx2;
        let avx512 = SimdBackend::Avx512;

        assert_eq!(scalar, SimdBackend::Scalar);
        assert_ne!(scalar, avx2);
        assert_ne!(avx2, avx512);

        let debug = format!("{:?}", scalar);
        assert!(debug.contains("Scalar"));
        let debug = format!("{:?}", avx2);
        assert!(debug.contains("Avx2"));
        let debug = format!("{:?}", avx512);
        assert!(debug.contains("Avx512"));
    }

    /// Cover SimdBackend Clone and Copy
    #[test]
    fn simd_backend_clone_copy() {
        let original = SimdBackend::Avx2;
        let cloned = original; // Copy
        assert_eq!(original, cloned);

        let clone2 = original;
        assert_eq!(original, clone2);
    }

    /// Cover dot_i8_scalar with single element
    #[test]
    fn dot_i8_scalar_single_element() {
        let result = dot_i8_scalar(&[42], &[3]);
        assert_eq!(result, 126);
    }

    /// Cover dot_i8_scalar with empty input
    #[test]
    fn dot_i8_scalar_empty() {
        let result = dot_i8_scalar(&[], &[]);
        assert_eq!(result, 0);
    }

    /// Cover from_x86_features with AVX-512 available
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn from_x86_features_avx512() {
        let backend = SimdBackend::from_x86_features(true, true);
        assert_eq!(backend, SimdBackend::Avx512);
    }

    /// Cover from_x86_features with only AVX2 available
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn from_x86_features_avx2_only() {
        let backend = SimdBackend::from_x86_features(false, true);
        assert_eq!(backend, SimdBackend::Avx2);
    }

    /// Cover from_x86_features with no SIMD (scalar fallback)
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn from_x86_features_scalar() {
        let backend = SimdBackend::from_x86_features(false, false);
        assert_eq!(backend, SimdBackend::Scalar);
    }

    /// Cover from_x86_features with AVX-512 but no AVX2 (unusual but valid)
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn from_x86_features_avx512_no_avx2() {
        let backend = SimdBackend::from_x86_features(true, false);
        assert_eq!(backend, SimdBackend::Avx512);
    }
}
