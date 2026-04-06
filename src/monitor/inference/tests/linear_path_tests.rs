//! LinearPath Property Tests

use super::helpers::arb_linear_path;
use crate::monitor::inference::path::DecisionPath;
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    #[test]
    fn prop_linear_path_serialization_roundtrip(path in arb_linear_path()) {
        use crate::monitor::inference::LinearPath;

        let bytes = path.to_bytes();
        let restored = LinearPath::from_bytes(&bytes).expect("Deserialization failed");

        prop_assert_eq!(path.contributions.len(), restored.contributions.len());
        for (a, b) in path.contributions.iter().zip(restored.contributions.iter()) {
            prop_assert!((a - b).abs() < 1e-5, "Contribution mismatch: {} vs {}", a, b);
        }
        prop_assert!((path.intercept - restored.intercept).abs() < 1e-5);
        prop_assert!((path.logit - restored.logit).abs() < 1e-5);
        prop_assert!((path.prediction - restored.prediction).abs() < 1e-5);
        prop_assert_eq!(path.probability.is_some(), restored.probability.is_some());
    }

    #[test]
    fn prop_linear_path_confidence_bounds(path in arb_linear_path()) {
        let confidence = path.confidence();
        prop_assert!(confidence >= 0.0, "Confidence must be >= 0: {}", confidence);
        prop_assert!(confidence <= 1.0, "Confidence must be <= 1: {}", confidence);
    }

    #[test]
    fn prop_linear_path_top_features_sorted(path in arb_linear_path()) {
        let k = path.contributions.len().min(5);
        let top = path.top_features(k);

        // Check sorted by absolute value (descending)
        for i in 1..top.len() {
            prop_assert!(
                top[i-1].1.abs() >= top[i].1.abs(),
                "Not sorted: {} vs {}",
                top[i-1].1.abs(),
                top[i].1.abs()
            );
        }
    }

    #[test]
    fn prop_linear_path_feature_contributions_length(path in arb_linear_path()) {
        let contributions = path.feature_contributions();
        prop_assert_eq!(contributions.len(), path.contributions.len());
    }
}
