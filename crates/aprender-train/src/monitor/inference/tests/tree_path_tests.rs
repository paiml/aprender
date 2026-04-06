//! TreePath Property Tests

use super::helpers::arb_tree_path;
use crate::monitor::inference::path::DecisionPath;
use crate::monitor::inference::TreePath;
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    #[test]
    fn prop_tree_path_serialization_roundtrip(path in arb_tree_path()) {
        let bytes = path.to_bytes();
        let restored = TreePath::from_bytes(&bytes).expect("Deserialization failed");

        prop_assert_eq!(path.splits.len(), restored.splits.len());
        prop_assert_eq!(path.depth(), restored.depth());
        prop_assert!((path.leaf.prediction - restored.leaf.prediction).abs() < 1e-5);
        prop_assert_eq!(path.leaf.n_samples, restored.leaf.n_samples);
    }

    #[test]
    fn prop_tree_path_depth_equals_splits(path in arb_tree_path()) {
        prop_assert_eq!(path.depth(), path.splits.len());
    }

    #[test]
    fn prop_tree_path_confidence_bounds(path in arb_tree_path()) {
        let confidence = path.confidence();
        prop_assert!(confidence >= 0.0, "Confidence must be >= 0: {}", confidence);
        prop_assert!(confidence <= 1.0, "Confidence must be <= 1: {}", confidence);
    }
}
