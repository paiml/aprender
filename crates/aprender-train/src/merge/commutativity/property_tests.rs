//! Property-based tests for merge commutativity using proptest

use super::helpers::{make_model, make_multi_param_model, models_approx_equal};
use crate::merge::{dare_merge, slerp_merge, ties_merge, DareConfig, SlerpConfig, TiesConfig};
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    // ============================================================
    // SLERP Properties
    // ============================================================

    #[test]
    fn prop_slerp_commutativity(
        v1 in proptest::collection::vec(1.0f32..10.0, 3..8),
        v2 in proptest::collection::vec(1.0f32..10.0, 3..8),
        t in 0.01f32..0.99
    ) {
        let len = v1.len().min(v2.len());
        let v1: Vec<f32> = v1.into_iter().take(len).collect();
        let v2: Vec<f32> = v2.into_iter().take(len).collect();

        let m1 = make_model(v1);
        let m2 = make_model(v2);

        let c1 = SlerpConfig::new(t).expect("slerp config creation should succeed");
        let c2 = SlerpConfig::new(1.0 - t).expect("slerp config creation should succeed");

        let r1 = slerp_merge(&m1, &m2, &c1).expect("operation should succeed");
        let r2 = slerp_merge(&m2, &m1, &c2).expect("operation should succeed");

        prop_assert!(
            models_approx_equal(&r1, &r2, 1e-3),
            "slerp(A,B,t) should equal slerp(B,A,1-t)"
        );
    }

    #[test]
    fn prop_slerp_identity(
        values in proptest::collection::vec(-10.0f32..10.0, 3..8),
        t in 0.0f32..=1.0
    ) {
        let m = make_model(values);
        let config = SlerpConfig::new(t).expect("slerp config creation should succeed");

        let result = slerp_merge(&m, &m, &config).expect("config should be valid");

        prop_assert!(
            models_approx_equal(&result, &m, 1e-4),
            "slerp(A, A, t) should equal A"
        );
    }

    #[test]
    fn prop_slerp_boundary_t0(
        v1 in proptest::collection::vec(-10.0f32..10.0, 3..8),
        v2 in proptest::collection::vec(-10.0f32..10.0, 3..8)
    ) {
        let len = v1.len().min(v2.len());
        let v1: Vec<f32> = v1.into_iter().take(len).collect();
        let v2: Vec<f32> = v2.into_iter().take(len).collect();

        let m1 = make_model(v1);
        let m2 = make_model(v2);
        let config = SlerpConfig::new(0.0).expect("slerp config creation should succeed");

        let result = slerp_merge(&m1, &m2, &config).expect("config should be valid");

        prop_assert!(
            models_approx_equal(&result, &m1, 1e-5),
            "slerp(A, B, 0) should equal A"
        );
    }

    #[test]
    fn prop_slerp_boundary_t1(
        v1 in proptest::collection::vec(-10.0f32..10.0, 3..8),
        v2 in proptest::collection::vec(-10.0f32..10.0, 3..8)
    ) {
        let len = v1.len().min(v2.len());
        let v1: Vec<f32> = v1.into_iter().take(len).collect();
        let v2: Vec<f32> = v2.into_iter().take(len).collect();

        let m1 = make_model(v1);
        let m2 = make_model(v2);
        let config = SlerpConfig::new(1.0).expect("slerp config creation should succeed");

        let result = slerp_merge(&m1, &m2, &config).expect("config should be valid");

        prop_assert!(
            models_approx_equal(&result, &m2, 1e-5),
            "slerp(A, B, 1) should equal B"
        );
    }

    #[test]
    fn prop_slerp_midpoint_symmetric(
        v1 in proptest::collection::vec(1.0f32..10.0, 3..6),
        v2 in proptest::collection::vec(1.0f32..10.0, 3..6)
    ) {
        let len = v1.len().min(v2.len());
        let v1: Vec<f32> = v1.into_iter().take(len).collect();
        let v2: Vec<f32> = v2.into_iter().take(len).collect();

        let m1 = make_model(v1);
        let m2 = make_model(v2);
        let config = SlerpConfig::new(0.5).expect("slerp config creation should succeed");

        let r1 = slerp_merge(&m1, &m2, &config).expect("config should be valid");
        let r2 = slerp_merge(&m2, &m1, &config).expect("config should be valid");

        prop_assert!(
            models_approx_equal(&r1, &r2, 1e-4),
            "slerp at t=0.5 should be symmetric"
        );
    }

    // ============================================================
    // DARE Properties
    // ============================================================

    #[test]
    fn prop_dare_zero_drop_commutative(
        v1 in proptest::collection::vec(-10.0f32..10.0, 3..8),
        v2 in proptest::collection::vec(-10.0f32..10.0, 3..8)
    ) {
        let len = v1.len().min(v2.len());
        let v1: Vec<f32> = v1.into_iter().take(len).collect();
        let v2: Vec<f32> = v2.into_iter().take(len).collect();

        let base = make_model(vec![0.0; len]);
        let m1 = make_model(v1);
        let m2 = make_model(v2);

        let config = DareConfig::new(0.0).expect("config should be valid");

        let r1 = dare_merge(&[m1.clone(), m2.clone()], &base, &config).expect("config should be valid");
        let r2 = dare_merge(&[m2, m1], &base, &config).expect("config should be valid");

        prop_assert!(
            models_approx_equal(&r1, &r2, 1e-5),
            "DARE(drop=0) should be commutative"
        );
    }

    #[test]
    fn prop_dare_zero_drop_permutation_invariant(
        v1 in proptest::collection::vec(-5.0f32..5.0, 3..6),
        v2 in proptest::collection::vec(-5.0f32..5.0, 3..6),
        v3 in proptest::collection::vec(-5.0f32..5.0, 3..6)
    ) {
        let len = v1.len().min(v2.len()).min(v3.len());
        let v1: Vec<f32> = v1.into_iter().take(len).collect();
        let v2: Vec<f32> = v2.into_iter().take(len).collect();
        let v3: Vec<f32> = v3.into_iter().take(len).collect();

        let base = make_model(vec![0.0; len]);
        let m1 = make_model(v1);
        let m2 = make_model(v2);
        let m3 = make_model(v3);

        let config = DareConfig::new(0.0).expect("config should be valid");

        let r1 = dare_merge(&[m1.clone(), m2.clone(), m3.clone()], &base, &config).expect("config should be valid");
        let r2 = dare_merge(&[m3.clone(), m1.clone(), m2.clone()], &base, &config).expect("config should be valid");
        let r3 = dare_merge(&[m2, m3, m1], &base, &config).expect("config should be valid");

        prop_assert!(
            models_approx_equal(&r1, &r2, 1e-5) && models_approx_equal(&r2, &r3, 1e-5),
            "DARE(drop=0) should be permutation-invariant"
        );
    }

    #[test]
    fn prop_dare_identity_merge(
        values in proptest::collection::vec(-10.0f32..10.0, 3..8)
    ) {
        let base = make_model(vec![0.0; values.len()]);
        let m = make_model(values);

        let config = DareConfig::new(0.0).expect("config should be valid");
        let result = dare_merge(&[m.clone(), m.clone()], &base, &config).expect("config should be valid");

        prop_assert!(
            models_approx_equal(&result, &m, 1e-5),
            "DARE of identical models should preserve values"
        );
    }

    // ============================================================
    // TIES Properties
    // ============================================================

    #[test]
    fn prop_ties_permutation_invariant_2_models(
        v1 in proptest::collection::vec(-10.0f32..10.0, 4..8),
        v2 in proptest::collection::vec(-10.0f32..10.0, 4..8),
        density in 0.3f32..0.8
    ) {
        let len = v1.len().min(v2.len());
        let v1: Vec<f32> = v1.into_iter().take(len).collect();
        let v2: Vec<f32> = v2.into_iter().take(len).collect();

        let base = make_model(vec![0.0; len]);
        let m1 = make_model(v1);
        let m2 = make_model(v2);

        let config = TiesConfig::new(density).expect("config should be valid");

        let r1 = ties_merge(&[m1.clone(), m2.clone()], &base, &config).expect("config should be valid");
        let r2 = ties_merge(&[m2, m1], &base, &config).expect("config should be valid");

        prop_assert!(
            models_approx_equal(&r1, &r2, 1e-5),
            "TIES should be permutation-invariant for 2 models"
        );
    }

    #[test]
    fn prop_ties_permutation_invariant_3_models(
        v1 in proptest::collection::vec(-5.0f32..5.0, 4..6),
        v2 in proptest::collection::vec(-5.0f32..5.0, 4..6),
        v3 in proptest::collection::vec(-5.0f32..5.0, 4..6),
        density in 0.4f32..0.7
    ) {
        let len = v1.len().min(v2.len()).min(v3.len());
        let v1: Vec<f32> = v1.into_iter().take(len).collect();
        let v2: Vec<f32> = v2.into_iter().take(len).collect();
        let v3: Vec<f32> = v3.into_iter().take(len).collect();

        let base = make_model(vec![0.0; len]);
        let m1 = make_model(v1);
        let m2 = make_model(v2);
        let m3 = make_model(v3);

        let config = TiesConfig::new(density).expect("config should be valid");

        let r1 = ties_merge(&[m1.clone(), m2.clone(), m3.clone()], &base, &config).expect("config should be valid");
        let r2 = ties_merge(&[m2.clone(), m3.clone(), m1.clone()], &base, &config).expect("config should be valid");
        let r3 = ties_merge(&[m3, m1, m2], &base, &config).expect("config should be valid");

        prop_assert!(
            models_approx_equal(&r1, &r2, 1e-5) && models_approx_equal(&r2, &r3, 1e-5),
            "TIES should be permutation-invariant for 3 models"
        );
    }

    #[test]
    fn prop_ties_identity_preserves_sign(
        values in proptest::collection::vec(-10.0f32..10.0, 4..8)
            .prop_filter("non-zero values", |v| v.iter().all(|x| x.abs() > 0.1)),
        density in 0.7f32..0.95
    ) {
        let base = make_model(vec![0.0; values.len()]);
        let m = make_model(values.clone());

        let config = TiesConfig::new(density).expect("config should be valid");
        let result = ties_merge(&[m.clone(), m.clone()], &base, &config).expect("config should be valid");

        // Signs of non-zero results should match original
        let merged = result["w"].data();
        for (i, (orig, res)) in values.iter().zip(merged.iter()).enumerate() {
            if res.abs() > 1e-6 {
                prop_assert!(
                    orig.signum() == res.signum(),
                    "TIES identity should preserve sign at index {}: {} vs {}",
                    i, orig, res
                );
            }
        }
    }

    // ============================================================
    // Multi-parameter model tests
    // ============================================================

    #[test]
    fn prop_slerp_multi_param_commutativity(
        v1a in proptest::collection::vec(1.0f32..5.0, 3..5),
        v1b in proptest::collection::vec(1.0f32..5.0, 3..5),
        v2a in proptest::collection::vec(1.0f32..5.0, 3..5),
        v2b in proptest::collection::vec(1.0f32..5.0, 3..5),
        t in 0.1f32..0.9
    ) {
        let len_a = v1a.len().min(v2a.len());
        let len_b = v1b.len().min(v2b.len());

        let v1a: Vec<f32> = v1a.into_iter().take(len_a).collect();
        let v2a: Vec<f32> = v2a.into_iter().take(len_a).collect();
        let v1b: Vec<f32> = v1b.into_iter().take(len_b).collect();
        let v2b: Vec<f32> = v2b.into_iter().take(len_b).collect();

        let m1 = make_multi_param_model(vec![("a", v1a), ("b", v1b)]);
        let m2 = make_multi_param_model(vec![("a", v2a), ("b", v2b)]);

        let c1 = SlerpConfig::new(t).expect("slerp config creation should succeed");
        let c2 = SlerpConfig::new(1.0 - t).expect("slerp config creation should succeed");

        let r1 = slerp_merge(&m1, &m2, &c1).expect("operation should succeed");
        let r2 = slerp_merge(&m2, &m1, &c2).expect("operation should succeed");

        prop_assert!(
            models_approx_equal(&r1, &r2, 1e-3),
            "Multi-param SLERP should be commutative"
        );
    }
}
