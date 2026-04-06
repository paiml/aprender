//! Counterfactual Property Tests

use crate::monitor::inference::Counterfactual;
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    #[test]
    fn prop_counterfactual_metrics(
        original in prop::collection::vec(-10.0f32..10.0, 2..20),
        deltas in prop::collection::vec(-5.0f32..5.0, 2..20),
    ) {
        let n = original.len().min(deltas.len());
        let original = original[..n].to_vec();
        let deltas = deltas[..n].to_vec();

        let counterfactual: Vec<f32> = original.iter().zip(deltas.iter())
            .map(|(o, d)| o + d)
            .collect();

        let cf = Counterfactual::new(
            original.clone(),
            0,
            0.9,
            counterfactual,
            1,
            0.85,
        );

        // L1 should be sum of absolute deltas for changed features
        let expected_l1: f32 = deltas.iter().map(|d| d.abs()).sum();
        prop_assert!((cf.sparsity - expected_l1).abs() < 1e-4, "L1 mismatch");

        // L2 should be sqrt of sum of squared deltas
        let expected_l2: f32 = deltas.iter().map(|d| d * d).sum::<f32>().sqrt();
        prop_assert!((cf.distance - expected_l2).abs() < 1e-4, "L2 mismatch");
    }

    #[test]
    fn prop_counterfactual_serialization_roundtrip(
        original in prop::collection::vec(-10.0f32..10.0, 2..10),
        deltas in prop::collection::vec(-5.0f32..5.0, 2..10),
    ) {
        let n = original.len().min(deltas.len());
        let original = original[..n].to_vec();
        let counterfactual: Vec<f32> = original.iter().zip(deltas.iter().take(n))
            .map(|(o, d)| o + d)
            .collect();

        let cf = Counterfactual::new(
            original,
            0,
            0.9,
            counterfactual,
            1,
            0.85,
        );

        let bytes = cf.to_bytes();
        let restored = Counterfactual::from_bytes(&bytes)
            .expect("Deserialization failed");

        prop_assert_eq!(cf.original_decision, restored.original_decision);
        prop_assert_eq!(cf.alternative_decision, restored.alternative_decision);
        prop_assert_eq!(cf.n_changes(), restored.n_changes());
    }
}
