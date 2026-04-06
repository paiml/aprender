//! Property-based tests for ensemble merging

use super::common::{make_model, models_approx_equal};
use crate::merge::ensemble::{ensemble_merge, EnsembleConfig, EnsembleStrategy, Model};
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    #[test]
    fn prop_uniform_average_is_mean(
        v1 in proptest::collection::vec(-10.0f32..10.0, 3..6),
        v2 in proptest::collection::vec(-10.0f32..10.0, 3..6),
        v3 in proptest::collection::vec(-10.0f32..10.0, 3..6)
    ) {
        let len = v1.len().min(v2.len()).min(v3.len());
        let v1: Vec<f32> = v1.into_iter().take(len).collect();
        let v2: Vec<f32> = v2.into_iter().take(len).collect();
        let v3: Vec<f32> = v3.into_iter().take(len).collect();

        let m1 = make_model(v1.clone());
        let m2 = make_model(v2.clone());
        let m3 = make_model(v3.clone());

        let config = EnsembleConfig::uniform_average();
        let result = ensemble_merge(&[m1, m2, m3], &config).expect("config should be valid");

        for i in 0..len {
            let expected = (v1[i] + v2[i] + v3[i]) / 3.0;
            prop_assert!(
                (result["w"].data()[i] - expected).abs() < 1e-4,
                "Uniform average mismatch at {}: {} vs {}",
                i, result["w"].data()[i], expected
            );
        }
    }

    #[test]
    fn prop_weighted_average_sums_correctly(
        v1 in proptest::collection::vec(-10.0f32..10.0, 3..6),
        v2 in proptest::collection::vec(-10.0f32..10.0, 3..6),
        w1 in 0.1f32..1.0,
        w2 in 0.1f32..1.0
    ) {
        let len = v1.len().min(v2.len());
        let v1: Vec<f32> = v1.into_iter().take(len).collect();
        let v2: Vec<f32> = v2.into_iter().take(len).collect();

        let m1 = make_model(v1.clone());
        let m2 = make_model(v2.clone());

        let config = EnsembleConfig::weighted_average(vec![w1, w2]);
        let result = ensemble_merge(&[m1, m2], &config).expect("config should be valid");

        let total = w1 + w2;
        for i in 0..len {
            let expected = (v1[i] * w1 + v2[i] * w2) / total;
            prop_assert!(
                (result["w"].data()[i] - expected).abs() < 1e-4,
                "Weighted average mismatch"
            );
        }
    }

    #[test]
    fn prop_uniform_average_permutation_invariant(
        v1 in proptest::collection::vec(-5.0f32..5.0, 3..5),
        v2 in proptest::collection::vec(-5.0f32..5.0, 3..5),
        v3 in proptest::collection::vec(-5.0f32..5.0, 3..5)
    ) {
        let len = v1.len().min(v2.len()).min(v3.len());
        let v1: Vec<f32> = v1.into_iter().take(len).collect();
        let v2: Vec<f32> = v2.into_iter().take(len).collect();
        let v3: Vec<f32> = v3.into_iter().take(len).collect();

        let m1 = make_model(v1);
        let m2 = make_model(v2);
        let m3 = make_model(v3);

        let config = EnsembleConfig::uniform_average();

        let r1 = ensemble_merge(&[m1.clone(), m2.clone(), m3.clone()], &config).expect("config should be valid");
        let r2 = ensemble_merge(&[m2.clone(), m3.clone(), m1.clone()], &config).expect("config should be valid");
        let r3 = ensemble_merge(&[m3, m1, m2], &config).expect("config should be valid");

        prop_assert!(models_approx_equal(&r1, &r2, 1e-5));
        prop_assert!(models_approx_equal(&r2, &r3, 1e-5));
    }

    #[test]
    fn prop_iterative_slerp_produces_finite_output(
        v1 in proptest::collection::vec(0.1f32..10.0, 3..6),
        v2 in proptest::collection::vec(0.1f32..10.0, 3..6),
        v3 in proptest::collection::vec(0.1f32..10.0, 3..6),
        t in 0.1f32..0.9
    ) {
        let len = v1.len().min(v2.len()).min(v3.len());
        let v1: Vec<f32> = v1.into_iter().take(len).collect();
        let v2: Vec<f32> = v2.into_iter().take(len).collect();
        let v3: Vec<f32> = v3.into_iter().take(len).collect();

        let m1 = make_model(v1);
        let m2 = make_model(v2);
        let m3 = make_model(v3);

        let config = EnsembleConfig::iterative_slerp(t);
        let result = ensemble_merge(&[m1, m2, m3], &config).expect("config should be valid");

        for val in result["w"].data() {
            prop_assert!(val.is_finite(), "SLERP produced non-finite value");
        }
    }

    #[test]
    fn prop_hierarchical_balanced_four_models(
        values in proptest::collection::vec(
            proptest::collection::vec(-5.0f32..5.0, 3..5),
            4..=4
        )
    ) {
        let len = values.iter().map(std::vec::Vec::len).min().unwrap_or(3);
        let models: Vec<Model> = values
            .into_iter()
            .map(|v| make_model(v.into_iter().take(len).collect()))
            .collect();

        let config = EnsembleConfig::hierarchical(
            EnsembleStrategy::WeightedAverage { weights: vec![0.5, 0.5] }
        );
        let result = ensemble_merge(&models, &config).expect("config should be valid");

        for val in result["w"].data() {
            prop_assert!(val.is_finite());
        }
    }

    #[test]
    fn prop_identity_single_weight(
        values in proptest::collection::vec(-10.0f32..10.0, 3..6)
    ) {
        let m1 = make_model(values.clone());
        let m2 = make_model(vec![0.0; values.len()]);

        // Weight 1.0 for m1, 0.0 for m2 should return m1
        let config = EnsembleConfig::weighted_average(vec![1.0, 0.0]);
        let result = ensemble_merge(&[m1.clone(), m2], &config).expect("config should be valid");

        prop_assert!(models_approx_equal(&result, &m1, 1e-5));
    }
}
