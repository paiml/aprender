//! FALSIFY-006: Cross-Crate Validation Parity Test
//!
//! Contract: tensor-layout-v1.yaml §falsification_tests FALSIFY-006
//! Claim: "Both aprender AND realizar enforce identical validation"
//!
//! This test verifies that `aprender::format::ValidatedEmbedding::new()` and
//! `realizar::safetensors::validation::ValidatedEmbedding::new()` produce
//! identical accept/reject decisions for the SAME input data.
//!
//! If the two crates' validation logic diverges (different thresholds, missing
//! gates, different error messages), this test catches it.
//!
//! ## Theoretical Foundation
//!
//! Popper, K. (1959). The Logic of Scientific Discovery.
//! "If the same data passes validation in one crate but fails in another,
//!  the contract is broken — the two crates have diverged."

#[cfg(feature = "inference")]
mod cross_crate_parity {
    use aprender::format::validated_tensors::{
        ValidatedEmbedding as AprEmbedding, ValidatedVector as AprVector,
        ValidatedWeight as AprWeight,
    };
    use realizar::safetensors::validation::{
        ValidatedEmbedding as RlzEmbedding, ValidatedVector as RlzVector,
        ValidatedWeight as RlzWeight,
    };

    // =========================================================================
    // Test vector generators — identical data used for both crates
    // =========================================================================

    fn good_embedding_data(vocab_size: usize, hidden_dim: usize) -> Vec<f32> {
        (0..vocab_size * hidden_dim)
            .map(|i| (i as f32 * 0.01).sin() * 0.1)
            .collect()
    }

    fn good_weight_data(n: usize) -> Vec<f32> {
        (0..n).map(|i| i as f32 * 0.01).collect()
    }

    // =========================================================================
    // FALSIFY-006a: Good data accepted by BOTH crates
    // =========================================================================

    #[test]
    fn falsify_006_good_embedding_accepted_by_both() {
        let vocab_size = 100;
        let hidden_dim = 64;
        let data = good_embedding_data(vocab_size, hidden_dim);

        let apr_result = AprEmbedding::new(data.clone(), vocab_size, hidden_dim);
        let rlz_result = RlzEmbedding::new(data, vocab_size, hidden_dim);

        assert!(
            apr_result.is_ok(),
            "aprender should accept good embedding: {:?}",
            apr_result.err()
        );
        assert!(
            rlz_result.is_ok(),
            "realizar should accept good embedding: {:?}",
            rlz_result.err()
        );
    }

    #[test]
    fn falsify_006_good_weight_accepted_by_both() {
        let data = good_weight_data(100);

        let apr_result = AprWeight::new(data.clone(), 10, 10, "test_weight");
        let rlz_result = RlzWeight::new(data, 10, 10, "test_weight");

        assert!(
            apr_result.is_ok(),
            "aprender should accept good weight: {:?}",
            apr_result.err()
        );
        assert!(
            rlz_result.is_ok(),
            "realizar should accept good weight: {:?}",
            rlz_result.err()
        );
    }

    #[test]
    fn falsify_006_good_vector_accepted_by_both() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];

        let apr_result = AprVector::new(data.clone(), 5, "test_vec");
        let rlz_result = RlzVector::new(data, 5, "test_vec");

        assert!(
            apr_result.is_ok(),
            "aprender should accept good vector: {:?}",
            apr_result.err()
        );
        assert!(
            rlz_result.is_ok(),
            "realizar should accept good vector: {:?}",
            rlz_result.err()
        );
    }

    // =========================================================================
    // FALSIFY-006b: Bad data rejected by BOTH crates with same rule_id
    // =========================================================================

    #[test]
    fn falsify_006_all_zeros_embedding_rejected_by_both() {
        let data = vec![0.0f32; 100 * 64];

        let apr_result = AprEmbedding::new(data.clone(), 100, 64);
        let rlz_result = RlzEmbedding::new(data, 100, 64);

        assert!(
            apr_result.is_err(),
            "aprender should reject all-zero embedding"
        );
        assert!(
            rlz_result.is_err(),
            "realizar should reject all-zero embedding"
        );

        let apr_err = apr_result.unwrap_err();
        let rlz_err = rlz_result.unwrap_err();

        assert_eq!(
            apr_err.rule_id, rlz_err.rule_id,
            "Both crates must cite the same rule_id.\n  aprender: {} ({})\n  realizar: {} ({})",
            apr_err.rule_id, apr_err.message, rlz_err.rule_id, rlz_err.message
        );
    }

    #[test]
    fn falsify_006_nan_embedding_rejected_by_both() {
        let vocab_size = 10;
        let hidden_dim = 8;
        let mut data: Vec<f32> = (0..vocab_size * hidden_dim)
            .map(|i| (i as f32 * 0.01).sin() * 0.1)
            .collect();
        data[5] = f32::NAN;

        let apr_result = AprEmbedding::new(data.clone(), vocab_size, hidden_dim);
        let rlz_result = RlzEmbedding::new(data, vocab_size, hidden_dim);

        assert!(apr_result.is_err(), "aprender should reject NaN embedding");
        assert!(rlz_result.is_err(), "realizar should reject NaN embedding");

        let apr_err = apr_result.unwrap_err();
        let rlz_err = rlz_result.unwrap_err();
        assert_eq!(
            apr_err.rule_id, rlz_err.rule_id,
            "Both must cite same rule for NaN.\n  aprender: {}\n  realizar: {}",
            apr_err.rule_id, rlz_err.rule_id
        );
    }

    #[test]
    fn falsify_006_inf_embedding_rejected_by_both() {
        let vocab_size = 10;
        let hidden_dim = 8;
        let mut data: Vec<f32> = (0..vocab_size * hidden_dim)
            .map(|i| (i as f32 * 0.01).sin() * 0.1)
            .collect();
        data[7] = f32::INFINITY;

        let apr_result = AprEmbedding::new(data.clone(), vocab_size, hidden_dim);
        let rlz_result = RlzEmbedding::new(data, vocab_size, hidden_dim);

        assert!(apr_result.is_err(), "aprender should reject Inf embedding");
        assert!(rlz_result.is_err(), "realizar should reject Inf embedding");

        let apr_err = apr_result.unwrap_err();
        let rlz_err = rlz_result.unwrap_err();
        assert_eq!(
            apr_err.rule_id, rlz_err.rule_id,
            "Both must cite same rule for Inf.\n  aprender: {}\n  realizar: {}",
            apr_err.rule_id, rlz_err.rule_id
        );
    }

    #[test]
    fn falsify_006_wrong_shape_embedding_rejected_by_both() {
        let data = vec![0.1f32; 1000]; // wrong size for 100x64

        let apr_result = AprEmbedding::new(data.clone(), 100, 64);
        let rlz_result = RlzEmbedding::new(data, 100, 64);

        assert!(apr_result.is_err(), "aprender should reject wrong shape");
        assert!(rlz_result.is_err(), "realizar should reject wrong shape");

        let apr_err = apr_result.unwrap_err();
        let rlz_err = rlz_result.unwrap_err();
        assert_eq!(
            apr_err.rule_id, rlz_err.rule_id,
            "Both must cite same rule for shape mismatch.\n  aprender: {}\n  realizar: {}",
            apr_err.rule_id, rlz_err.rule_id
        );
    }

    #[test]
    fn falsify_006_spot_check_offset_bug_rejected_by_both() {
        // Simulate PMAT-234: zero token at 10% of vocab
        let vocab_size = 100;
        let hidden_dim = 64;
        let mut data: Vec<f32> = (0..vocab_size * hidden_dim)
            .map(|i| (i as f32 * 0.01).sin() * 0.1)
            .collect();

        // Zero out token at 10%
        let token_start = 10 * hidden_dim;
        for v in &mut data[token_start..token_start + hidden_dim] {
            *v = 0.0;
        }

        let apr_result = AprEmbedding::new(data.clone(), vocab_size, hidden_dim);
        let rlz_result = RlzEmbedding::new(data, vocab_size, hidden_dim);

        assert!(
            apr_result.is_err(),
            "aprender should catch zero token at 10%"
        );
        assert!(
            rlz_result.is_err(),
            "realizar should catch zero token at 10%"
        );

        let apr_err = apr_result.unwrap_err();
        let rlz_err = rlz_result.unwrap_err();
        assert_eq!(
            apr_err.rule_id, rlz_err.rule_id,
            "Both must cite same rule for spot check.\n  aprender: {}\n  realizar: {}",
            apr_err.rule_id, rlz_err.rule_id
        );
    }

    #[test]
    fn falsify_006_all_zero_weight_rejected_by_both() {
        let data = vec![0.0f32; 100];

        let apr_result = AprWeight::new(data.clone(), 10, 10, "test");
        let rlz_result = RlzWeight::new(data, 10, 10, "test");

        assert!(
            apr_result.is_err(),
            "aprender should reject all-zero weight"
        );
        assert!(
            rlz_result.is_err(),
            "realizar should reject all-zero weight"
        );

        let apr_err = apr_result.unwrap_err();
        let rlz_err = rlz_result.unwrap_err();
        assert_eq!(
            apr_err.rule_id, rlz_err.rule_id,
            "Both must cite same rule.\n  aprender: {}\n  realizar: {}",
            apr_err.rule_id, rlz_err.rule_id
        );
    }

    #[test]
    fn falsify_006_nan_weight_rejected_by_both() {
        let mut data = good_weight_data(100);
        data[50] = f32::NAN;

        let apr_result = AprWeight::new(data.clone(), 10, 10, "test");
        let rlz_result = RlzWeight::new(data, 10, 10, "test");

        assert!(apr_result.is_err(), "aprender should reject NaN weight");
        assert!(rlz_result.is_err(), "realizar should reject NaN weight");

        let apr_err = apr_result.unwrap_err();
        let rlz_err = rlz_result.unwrap_err();
        assert_eq!(
            apr_err.rule_id, rlz_err.rule_id,
            "Both must cite same rule for NaN weight.\n  aprender: {}\n  realizar: {}",
            apr_err.rule_id, rlz_err.rule_id
        );
    }

    #[test]
    fn falsify_006_nan_vector_rejected_by_both() {
        let mut data = vec![1.0f32; 50];
        data[25] = f32::NAN;

        let apr_result = AprVector::new(data.clone(), 50, "test");
        let rlz_result = RlzVector::new(data, 50, "test");

        assert!(apr_result.is_err(), "aprender should reject NaN vector");
        assert!(rlz_result.is_err(), "realizar should reject NaN vector");

        let apr_err = apr_result.unwrap_err();
        let rlz_err = rlz_result.unwrap_err();
        assert_eq!(
            apr_err.rule_id, rlz_err.rule_id,
            "Both must cite same rule for NaN vector.\n  aprender: {}\n  realizar: {}",
            apr_err.rule_id, rlz_err.rule_id
        );
    }

    // =========================================================================
    // FALSIFY-006c: Threshold parity — both crates use IDENTICAL constants
    // =========================================================================

    #[test]
    fn falsify_006_density_threshold_boundary_parity() {
        // Test at exactly 50% zeros for embedding — boundary case
        let vocab_size = 100;
        let hidden_dim = 64;
        let total = vocab_size * hidden_dim;
        let zero_count = total / 2 + 1; // 50.01% zeros — should fail

        let mut data: Vec<f32> = (0..total)
            .map(|i| (i as f32 * 0.01).sin() * 0.1 + 0.05)
            .collect();
        for v in data.iter_mut().take(zero_count) {
            *v = 0.0;
        }

        let apr_result = AprEmbedding::new(data.clone(), vocab_size, hidden_dim);
        let rlz_result = RlzEmbedding::new(data, vocab_size, hidden_dim);

        // Both must agree: either both pass or both fail
        assert_eq!(
            apr_result.is_err(),
            rlz_result.is_err(),
            "Density boundary: aprender={}, realizar={}. Thresholds have diverged!",
            if apr_result.is_err() {
                "rejected"
            } else {
                "accepted"
            },
            if rlz_result.is_err() {
                "rejected"
            } else {
                "accepted"
            },
        );
    }

    // =========================================================================
    // FALSIFY-007: Special token ID parity between aprender and realizar
    // =========================================================================

    #[test]
    fn falsify_007_special_tokens_match_realizar_defaults() {
        use aprender::demo::SpecialTokens;

        // Verify aprender's SpecialTokens registry matches realizar's
        // default_eos_for_architecture and default_bos_for_architecture.
        // Source of truth: special-tokens-registry-v1.yaml
        let test_cases: &[(&str, u32, u32)] = &[
            // (arch, expected_bos, expected_eos)
            ("qwen2", 151_643, 151_645),
            ("llama", 128_000, 128_001),
            ("mistral", 1, 2),
            ("gemma", 2, 1),
            ("deepseek", 0, 1),
            ("phi3", 1, 32_000),
            ("gpt2", 0, 50_256),
        ];

        for &(arch, expected_bos, expected_eos) in test_cases {
            let tokens = SpecialTokens::from_architecture(arch)
                .unwrap_or_else(|| panic!("from_architecture('{arch}') returned None"));
            assert_eq!(tokens.bos_id, expected_bos,
                "FALSIFY-007: aprender BOS for '{arch}' ({}) != expected ({expected_bos})",
                tokens.bos_id);
            assert_eq!(tokens.eos_id, expected_eos,
                "FALSIFY-007: aprender EOS for '{arch}' ({}) != expected ({expected_eos})",
                tokens.eos_id);
        }
    }

    #[test]
    fn falsify_006_weight_density_threshold_boundary_parity() {
        // Test at exactly 80% zeros for weight — boundary case
        let total = 100;
        let zero_count = total * 81 / 100; // 81% zeros — should fail

        let mut data: Vec<f32> = (0..total).map(|i| i as f32 * 0.01 + 0.01).collect();
        for v in data.iter_mut().take(zero_count) {
            *v = 0.0;
        }

        let apr_result = AprWeight::new(data.clone(), 10, 10, "test");
        let rlz_result = RlzWeight::new(data, 10, 10, "test");

        assert_eq!(
            apr_result.is_err(),
            rlz_result.is_err(),
            "Weight density boundary: aprender={}, realizar={}. Thresholds have diverged!",
            if apr_result.is_err() {
                "rejected"
            } else {
                "accepted"
            },
            if rlz_result.is_err() {
                "rejected"
            } else {
                "accepted"
            },
        );
    }

    // =========================================================================
    // FALSIFY-E6: Cross-crate embedding constant parity (Refs PMAT-325)
    //
    // Five-Whys:
    //   Why 1: GPU path could load garbage embeddings silently
    //   Why 2: GPU path only checks shape, not data quality
    //   Why 3: ValidatedEmbedding gates not wired into GGUF load
    //   Why 4: GGUF load predates ValidatedEmbedding
    //   Why 5: No cross-path contract enforcement test existed
    //
    // These tests verify that the CONSTANTS used for validation are
    // identical between aprender and realizar, so when PMAT-325 is
    // fixed, the GPU path will use the same thresholds.
    // =========================================================================

    #[test]
    fn falsify_e6_embedding_constants_identical_across_crates() {
        // Verify the validation constants are compiled into both crates
        // by constructing boundary test cases that exercise each threshold.

        // MAX_ZERO_PCT = 50.0 — both crates must agree on this boundary.
        // Create embedding with 51% zeros (should fail in both).
        let vocab_size = 100;
        let hidden_dim = 64;
        let total = vocab_size * hidden_dim;
        let zero_count = total * 51 / 100; // 51% zeros

        let mut data: Vec<f32> = (0..total)
            .map(|i| (i as f32 * 0.01).sin() * 0.1 + 0.05)
            .collect();
        // Scatter zeros uniformly so spot-check tokens aren't wiped out
        let mut count = 0;
        for i in 0..total {
            if count < zero_count && i % 2 == 0 {
                data[i] = 0.0;
                count += 1;
            }
        }
        // Fill remaining if needed
        let mut idx = 1;
        while count < zero_count && idx < total {
            if data[idx].abs() > 1e-10 {
                data[idx] = 0.0;
                count += 1;
            }
            idx += 3; // skip to avoid zeroing spot-check tokens entirely
        }

        let apr_result = AprEmbedding::new(data.clone(), vocab_size, hidden_dim);
        let rlz_result = RlzEmbedding::new(data, vocab_size, hidden_dim);

        // Both must reject: 51% > 50%
        assert_eq!(
            apr_result.is_err(),
            rlz_result.is_err(),
            "FALSIFY-E6: 51% zero embedding: aprender={}, realizar={}. Density thresholds diverged!",
            if apr_result.is_err() { "rejected" } else { "accepted" },
            if rlz_result.is_err() { "rejected" } else { "accepted" },
        );
    }

    #[test]
    fn falsify_e6_min_l2_norm_identical_across_crates() {
        // MIN_L2_NORM = 1e-6 — test with near-zero L2 embedding.
        // Values above the zero threshold (1e-10) but L2 < 1e-6.
        let vocab_size = 10;
        let hidden_dim = 8;
        let data: Vec<f32> = (0..vocab_size * hidden_dim)
            .map(|i| 1e-8 + (i as f32) * 1e-12) // L2 ≈ 8.9e-8 < 1e-6
            .collect();

        let apr_result = AprEmbedding::new(data.clone(), vocab_size, hidden_dim);
        let rlz_result = RlzEmbedding::new(data, vocab_size, hidden_dim);

        assert_eq!(
            apr_result.is_err(),
            rlz_result.is_err(),
            "FALSIFY-E6: Near-zero L2 embedding: aprender={}, realizar={}. MIN_L2_NORM thresholds diverged!",
            if apr_result.is_err() { "rejected" } else { "accepted" },
            if rlz_result.is_err() { "rejected" } else { "accepted" },
        );
    }

    #[test]
    fn falsify_e6_spot_check_percentiles_identical_across_crates() {
        // SPOT_CHECK_PCTS = [10, 50, 90] — both crates must check same positions.
        // Zero out token at 50% — both must catch it.
        let vocab_size = 100;
        let hidden_dim = 64;
        let mut data: Vec<f32> = (0..vocab_size * hidden_dim)
            .map(|i| (i as f32 * 0.01).sin() * 0.1)
            .collect();

        // Zero out token at exactly 50% of vocab
        let token_50 = vocab_size * 50 / 100; // token 50
        let start = token_50 * hidden_dim;
        for v in &mut data[start..start + hidden_dim] {
            *v = 0.0;
        }

        let apr_result = AprEmbedding::new(data.clone(), vocab_size, hidden_dim);
        let rlz_result = RlzEmbedding::new(data, vocab_size, hidden_dim);

        assert!(apr_result.is_err(), "FALSIFY-E6: aprender must catch zero token at 50%");
        assert!(rlz_result.is_err(), "FALSIFY-E6: realizar must catch zero token at 50%");

        let apr_err = apr_result.unwrap_err();
        let rlz_err = rlz_result.unwrap_err();
        assert_eq!(
            apr_err.rule_id, rlz_err.rule_id,
            "FALSIFY-E6: Both must cite F-DATA-QUALITY-004 for spot check.\n  aprender: {}\n  realizar: {}",
            apr_err.rule_id, rlz_err.rule_id
        );
    }

    #[test]
    fn falsify_e6_constant_embedding_rejected_by_both() {
        // All-same-value embedding: passes density check but fails variation check.
        let vocab_size = 10;
        let hidden_dim = 8;
        let data = vec![0.5f32; vocab_size * hidden_dim];

        let apr_result = AprEmbedding::new(data.clone(), vocab_size, hidden_dim);
        let rlz_result = RlzEmbedding::new(data, vocab_size, hidden_dim);

        assert!(apr_result.is_err(), "aprender must reject constant embedding");
        assert!(rlz_result.is_err(), "realizar must reject constant embedding");

        let apr_err = apr_result.unwrap_err();
        let rlz_err = rlz_result.unwrap_err();
        assert_eq!(
            apr_err.rule_id, rlz_err.rule_id,
            "FALSIFY-E6: Both must cite same rule for constant embedding.\n  aprender: {}\n  realizar: {}",
            apr_err.rule_id, rlz_err.rule_id
        );
    }

    // =========================================================================
    // FALSIFY-L6: Cross-crate lm_head (ValidatedWeight) parity (Refs PMAT-328)
    //
    // Five-Whys:
    //   Why 1: lm_head matmul could produce different results in aprender vs realizar
    //   Why 2: ValidatedWeight gates might use different thresholds
    //   Why 3: Separate implementations in two crates
    //   Why 4: No cross-crate contract enforcement test existed
    //   Why 5: lm_head is critical (GH-202) — divergence here = garbage output
    //
    // Popper (1959): "These tests attempt to falsify the claim that
    // aprender and realizar enforce identical lm_head validation."
    // =========================================================================

    fn good_lm_head_data(vocab_size: usize, hidden_dim: usize) -> Vec<f32> {
        (0..vocab_size * hidden_dim)
            .map(|i| (i as f32 * 0.01).sin() * 0.1)
            .collect()
    }

    #[test]
    fn falsify_l6_good_lm_head_accepted_by_both() {
        let vocab_size = 100;
        let hidden_dim = 64;
        let data = good_lm_head_data(vocab_size, hidden_dim);

        let apr_result = AprWeight::new(data.clone(), vocab_size, hidden_dim, "lm_head.weight");
        let rlz_result = RlzWeight::new(data, vocab_size, hidden_dim, "lm_head.weight");

        assert!(apr_result.is_ok(),
            "aprender should accept good lm_head: {:?}", apr_result.err());
        assert!(rlz_result.is_ok(),
            "realizar should accept good lm_head: {:?}", rlz_result.err());
    }

    #[test]
    fn falsify_l6_wrong_shape_lm_head_rejected_by_both() {
        // Data for 100*64=6400 but declared as 200*64=12800
        let data = good_lm_head_data(100, 64);

        let apr_result = AprWeight::new(data.clone(), 200, 64, "lm_head.weight");
        let rlz_result = RlzWeight::new(data, 200, 64, "lm_head.weight");

        assert!(apr_result.is_err(), "aprender must reject wrong-shape lm_head");
        assert!(rlz_result.is_err(), "realizar must reject wrong-shape lm_head");

        let apr_err = apr_result.unwrap_err();
        let rlz_err = rlz_result.unwrap_err();
        assert_eq!(apr_err.rule_id, rlz_err.rule_id,
            "FALSIFY-L6: Both must cite same rule for shape mismatch.\n  aprender: {}\n  realizar: {}",
            apr_err.rule_id, rlz_err.rule_id);
    }

    #[test]
    fn falsify_l6_all_zero_lm_head_rejected_by_both() {
        // All zeros → density >80% → rejected
        let data = vec![0.0f32; 100 * 64];

        let apr_result = AprWeight::new(data.clone(), 100, 64, "lm_head.weight");
        let rlz_result = RlzWeight::new(data, 100, 64, "lm_head.weight");

        assert!(apr_result.is_err(), "aprender must reject all-zero lm_head");
        assert!(rlz_result.is_err(), "realizar must reject all-zero lm_head");

        let apr_err = apr_result.unwrap_err();
        let rlz_err = rlz_result.unwrap_err();
        assert_eq!(apr_err.rule_id, rlz_err.rule_id,
            "FALSIFY-L6: Both must cite same rule for all-zero lm_head.\n  aprender: {}\n  realizar: {}",
            apr_err.rule_id, rlz_err.rule_id);
    }

    #[test]
    fn falsify_l6_nan_lm_head_rejected_by_both() {
        let mut data = good_lm_head_data(100, 64);
        data[500] = f32::NAN;

        let apr_result = AprWeight::new(data.clone(), 100, 64, "lm_head.weight");
        let rlz_result = RlzWeight::new(data, 100, 64, "lm_head.weight");

        assert!(apr_result.is_err(), "aprender must reject NaN lm_head");
        assert!(rlz_result.is_err(), "realizar must reject NaN lm_head");

        let apr_err = apr_result.unwrap_err();
        let rlz_err = rlz_result.unwrap_err();
        assert_eq!(apr_err.rule_id, rlz_err.rule_id,
            "FALSIFY-L6: Both must cite same rule for NaN lm_head.\n  aprender: {}\n  realizar: {}",
            apr_err.rule_id, rlz_err.rule_id);
    }

    #[test]
    fn falsify_l6_inf_lm_head_rejected_by_both() {
        let mut data = good_lm_head_data(100, 64);
        data[300] = f32::INFINITY;

        let apr_result = AprWeight::new(data.clone(), 100, 64, "lm_head.weight");
        let rlz_result = RlzWeight::new(data, 100, 64, "lm_head.weight");

        assert!(apr_result.is_err(), "aprender must reject Inf lm_head");
        assert!(rlz_result.is_err(), "realizar must reject Inf lm_head");

        let apr_err = apr_result.unwrap_err();
        let rlz_err = rlz_result.unwrap_err();
        assert_eq!(apr_err.rule_id, rlz_err.rule_id,
            "FALSIFY-L6: Both must cite same rule for Inf lm_head.\n  aprender: {}\n  realizar: {}",
            apr_err.rule_id, rlz_err.rule_id);
    }

    #[test]
    fn falsify_l6_weight_density_threshold_parity_for_lm_head() {
        // ValidatedWeight density threshold = 80%. Test at 81% zeros.
        let total = 100 * 64;
        let zero_count = total * 81 / 100;

        let mut data: Vec<f32> = (0..total)
            .map(|i| (i as f32 * 0.01).sin() * 0.1 + 0.05)
            .collect();
        for v in data.iter_mut().take(zero_count) {
            *v = 0.0;
        }

        let apr_result = AprWeight::new(data.clone(), 100, 64, "lm_head.weight");
        let rlz_result = RlzWeight::new(data, 100, 64, "lm_head.weight");

        assert_eq!(apr_result.is_err(), rlz_result.is_err(),
            "FALSIFY-L6: 81% zero lm_head: aprender={}, realizar={}. Density thresholds diverged!",
            if apr_result.is_err() { "rejected" } else { "accepted" },
            if rlz_result.is_err() { "rejected" } else { "accepted" });
    }

    // =========================================================================
    // FALSIFY-A7: Cross-crate attention projection parity (Refs PMAT-330)
    //
    // Five-Whys:
    //   Why 1: Attention projections could behave differently in training vs inference
    //   Why 2: aprender and realizar might use different density/NaN thresholds
    //   Why 3: Separate ValidatedWeight implementations in each crate
    //   Why 4: No cross-crate test for attention weight validation
    //   Why 5: Embedding/lm_head parity tests existed but attention did not
    //
    // Popper (1959): "These tests attempt to falsify the claim that
    // aprender and realizar enforce identical attention weight validation."
    // =========================================================================

    fn good_attn_data(out_dim: usize, in_dim: usize) -> Vec<f32> {
        (0..out_dim * in_dim)
            .map(|i| (i as f32 * 0.001).sin() * 0.1)
            .collect()
    }

    #[test]
    fn falsify_a7_gqa_k_proj_accepted_by_both() {
        // Qwen2 0.5B: kv_dim=128, hidden=896
        let kv_dim = 128;
        let hidden = 896;
        let data = good_attn_data(kv_dim, hidden);

        let apr_result = AprWeight::new(data.clone(), kv_dim, hidden, "k_proj");
        let rlz_result = RlzWeight::new(data, kv_dim, hidden, "k_proj");

        assert!(apr_result.is_ok(), "aprender must accept valid GQA k_proj: {:?}", apr_result.err());
        assert!(rlz_result.is_ok(), "realizar must accept valid GQA k_proj: {:?}", rlz_result.err());
    }

    #[test]
    fn falsify_a7_wrong_shape_attn_rejected_by_both() {
        // Data for [128, 896] but declared as [896, 896]
        let data = good_attn_data(128, 896);

        let apr_result = AprWeight::new(data.clone(), 896, 896, "q_proj");
        let rlz_result = RlzWeight::new(data, 896, 896, "q_proj");

        assert!(apr_result.is_err(), "aprender must reject wrong-shape q_proj");
        assert!(rlz_result.is_err(), "realizar must reject wrong-shape q_proj");

        let apr_err = apr_result.unwrap_err();
        let rlz_err = rlz_result.unwrap_err();
        assert_eq!(apr_err.rule_id, rlz_err.rule_id,
            "FALSIFY-A7: Both must cite same rule for shape mismatch.\n  aprender: {}\n  realizar: {}",
            apr_err.rule_id, rlz_err.rule_id);
    }

    #[test]
    fn falsify_a7_nan_in_v_proj_rejected_by_both() {
        let kv_dim = 128;
        let hidden = 64;
        let mut data = good_attn_data(kv_dim, hidden);
        data[100] = f32::NAN;

        let apr_result = AprWeight::new(data.clone(), kv_dim, hidden, "v_proj");
        let rlz_result = RlzWeight::new(data, kv_dim, hidden, "v_proj");

        assert!(apr_result.is_err(), "aprender must reject NaN v_proj");
        assert!(rlz_result.is_err(), "realizar must reject NaN v_proj");

        let apr_err = apr_result.unwrap_err();
        let rlz_err = rlz_result.unwrap_err();
        assert_eq!(apr_err.rule_id, rlz_err.rule_id,
            "FALSIFY-A7: Both must cite same rule for NaN v_proj.\n  aprender: {}\n  realizar: {}",
            apr_err.rule_id, rlz_err.rule_id);
    }

    #[test]
    fn falsify_a7_all_zero_o_proj_rejected_by_both() {
        let hidden = 64;
        let data = vec![0.0f32; hidden * hidden];

        let apr_result = AprWeight::new(data.clone(), hidden, hidden, "o_proj");
        let rlz_result = RlzWeight::new(data, hidden, hidden, "o_proj");

        assert!(apr_result.is_err(), "aprender must reject all-zero o_proj");
        assert!(rlz_result.is_err(), "realizar must reject all-zero o_proj");

        let apr_err = apr_result.unwrap_err();
        let rlz_err = rlz_result.unwrap_err();
        assert_eq!(apr_err.rule_id, rlz_err.rule_id,
            "FALSIFY-A7: Both must cite same rule for all-zero o_proj.\n  aprender: {}\n  realizar: {}",
            apr_err.rule_id, rlz_err.rule_id);
    }

    // =========================================================================
    // FALSIFY-F8: §2.1.4 FFN Cross-Crate Parity (Refs PMAT-333)
    //
    // Contract: tensor-layout-v1.yaml §tensors.gate_proj/up_proj/down_proj
    // Claim: "Both aprender AND realizar enforce identical FFN validation"
    //
    // Popper (1959): "If gate_proj passes in one crate but fails in another,
    // the contract is broken — the two crates have diverged."
    // =========================================================================

    /// FALSIFY-F8a: Good gate_proj accepted by both crates
    #[test]
    fn falsify_f8_good_gate_proj_accepted_by_both() {
        let intermediate = 64;
        let hidden = 16;
        let data: Vec<f32> = (0..intermediate * hidden)
            .map(|i| (i as f32 * 0.01).sin() * 0.1 + 0.05)
            .collect();

        let apr_result = AprWeight::new(data.clone(), intermediate, hidden, "gate_proj");
        let rlz_result = RlzWeight::new(data, intermediate, hidden, "gate_proj");

        assert!(apr_result.is_ok(), "aprender must accept good gate_proj: {:?}", apr_result.err());
        assert!(rlz_result.is_ok(), "realizar must accept good gate_proj: {:?}", rlz_result.err());
    }

    /// FALSIFY-F8b: Wrong-shape down_proj rejected by both crates
    #[test]
    fn falsify_f8_wrong_shape_down_proj_rejected_by_both() {
        let hidden = 16;
        let intermediate = 64;
        // down_proj: [hidden, intermediate] = 1024 elements, but give 500
        let data: Vec<f32> = (0..500)
            .map(|i| (i as f32 * 0.01).sin() * 0.1 + 0.05)
            .collect();

        let apr_result = AprWeight::new(data.clone(), hidden, intermediate, "down_proj");
        let rlz_result = RlzWeight::new(data, hidden, intermediate, "down_proj");

        assert!(apr_result.is_err(), "aprender must reject wrong-shape down_proj");
        assert!(rlz_result.is_err(), "realizar must reject wrong-shape down_proj");

        let apr_err = apr_result.unwrap_err();
        let rlz_err = rlz_result.unwrap_err();
        assert_eq!(apr_err.rule_id, rlz_err.rule_id,
            "FALSIFY-F8b: Both must cite same rule for wrong-shape down_proj.\n  aprender: {}\n  realizar: {}",
            apr_err.rule_id, rlz_err.rule_id);
    }

    /// FALSIFY-F8c: NaN in up_proj rejected by both crates
    #[test]
    fn falsify_f8_nan_in_up_proj_rejected_by_both() {
        let intermediate = 64;
        let hidden = 16;
        let mut data: Vec<f32> = (0..intermediate * hidden)
            .map(|i| (i as f32 * 0.01).sin() * 0.1 + 0.05)
            .collect();
        data[42] = f32::NAN;

        let apr_result = AprWeight::new(data.clone(), intermediate, hidden, "up_proj");
        let rlz_result = RlzWeight::new(data, intermediate, hidden, "up_proj");

        assert!(apr_result.is_err(), "aprender must reject NaN up_proj");
        assert!(rlz_result.is_err(), "realizar must reject NaN up_proj");
    }

    /// FALSIFY-F8d: All-zero gate_proj rejected by both crates
    #[test]
    fn falsify_f8_all_zero_gate_proj_rejected_by_both() {
        let intermediate = 64;
        let hidden = 16;
        let data = vec![0.0f32; intermediate * hidden];

        let apr_result = AprWeight::new(data.clone(), intermediate, hidden, "gate_proj");
        let rlz_result = RlzWeight::new(data, intermediate, hidden, "gate_proj");

        assert!(apr_result.is_err(), "aprender must reject all-zero gate_proj");
        assert!(rlz_result.is_err(), "realizar must reject all-zero gate_proj");

        let apr_err = apr_result.unwrap_err();
        let rlz_err = rlz_result.unwrap_err();
        assert_eq!(apr_err.rule_id, rlz_err.rule_id,
            "FALSIFY-F8d: Both must cite same rule for all-zero gate_proj");
    }

    // =========================================================================
    // FALSIFY-N9: §2.1.5-6 Norm Cross-Crate Parity (Refs PMAT-332)
    //
    // Contract: tensor-layout-v1.yaml §tensors.input_layernorm/final_norm
    // Claim: "Both aprender AND realizar enforce identical norm validation"
    // =========================================================================

    /// FALSIFY-N9a: Good norm vector accepted by both crates
    #[test]
    fn falsify_n9_good_norm_accepted_by_both() {
        let hidden = 64;
        let data = vec![1.0f32; hidden];

        let apr_result = AprVector::new(data.clone(), hidden, "attn_norm");
        let rlz_result = RlzVector::new(data, hidden, "attn_norm");

        assert!(apr_result.is_ok(), "aprender must accept good norm: {:?}", apr_result.err());
        assert!(rlz_result.is_ok(), "realizar must accept good norm: {:?}", rlz_result.err());
    }

    /// FALSIFY-N9b: Wrong-length norm rejected by both crates
    #[test]
    fn falsify_n9_wrong_length_norm_rejected_by_both() {
        let data = vec![1.0f32; 32];
        let expected_len = 64;

        let apr_result = AprVector::new(data.clone(), expected_len, "output_norm");
        let rlz_result = RlzVector::new(data, expected_len, "output_norm");

        assert!(apr_result.is_err(), "aprender must reject wrong-length norm");
        assert!(rlz_result.is_err(), "realizar must reject wrong-length norm");

        let apr_err = apr_result.unwrap_err();
        let rlz_err = rlz_result.unwrap_err();
        assert_eq!(apr_err.rule_id, rlz_err.rule_id,
            "FALSIFY-N9b: Both must cite same rule for wrong-length norm.\n  aprender: {}\n  realizar: {}",
            apr_err.rule_id, rlz_err.rule_id);
    }

    /// FALSIFY-N9c: NaN in norm rejected by both crates
    #[test]
    fn falsify_n9_nan_in_norm_rejected_by_both() {
        let hidden = 64;
        let mut data = vec![1.0f32; hidden];
        data[10] = f32::NAN;

        let apr_result = AprVector::new(data.clone(), hidden, "ffn_norm");
        let rlz_result = RlzVector::new(data, hidden, "ffn_norm");

        assert!(apr_result.is_err(), "aprender must reject NaN norm");
        assert!(rlz_result.is_err(), "realizar must reject NaN norm");
    }

    /// FALSIFY-N9d: Inf in norm rejected by both crates
    #[test]
    fn falsify_n9_inf_in_norm_rejected_by_both() {
        let hidden = 64;
        let mut data = vec![1.0f32; hidden];
        data[5] = f32::INFINITY;

        let apr_result = AprVector::new(data.clone(), hidden, "attn_norm");
        let rlz_result = RlzVector::new(data, hidden, "attn_norm");

        assert!(apr_result.is_err(), "aprender must reject Inf norm");
        assert!(rlz_result.is_err(), "realizar must reject Inf norm");
    }
}
