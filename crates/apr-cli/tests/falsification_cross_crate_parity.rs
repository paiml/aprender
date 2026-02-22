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

        assert!(apr_result.is_err(), "aprender should reject all-zero embedding");
        assert!(rlz_result.is_err(), "realizar should reject all-zero embedding");

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

        assert!(apr_result.is_err(), "aprender should catch zero token at 10%");
        assert!(rlz_result.is_err(), "realizar should catch zero token at 10%");

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

        assert!(apr_result.is_err(), "aprender should reject all-zero weight");
        assert!(rlz_result.is_err(), "realizar should reject all-zero weight");

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
            if apr_result.is_err() { "rejected" } else { "accepted" },
            if rlz_result.is_err() { "rejected" } else { "accepted" },
        );
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
            if apr_result.is_err() { "rejected" } else { "accepted" },
            if rlz_result.is_err() { "rejected" } else { "accepted" },
        );
    }
}
