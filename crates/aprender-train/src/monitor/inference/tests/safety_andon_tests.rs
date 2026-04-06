//! Safety Andon Property Tests

use crate::monitor::inference::SafetyIntegrityLevel;
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    #[test]
    fn prop_safety_andon_confidence_thresholds(
        sil in prop_oneof![
            Just(SafetyIntegrityLevel::QM),
            Just(SafetyIntegrityLevel::SIL1),
            Just(SafetyIntegrityLevel::SIL2),
            Just(SafetyIntegrityLevel::SIL3),
            Just(SafetyIntegrityLevel::SIL4),
        ]
    ) {
        let confidence = sil.min_confidence();
        prop_assert!(confidence >= 0.0);
        prop_assert!(confidence <= 1.0);
    }

    #[test]
    fn prop_safety_andon_latency_thresholds(
        sil in prop_oneof![
            Just(SafetyIntegrityLevel::QM),
            Just(SafetyIntegrityLevel::SIL1),
            Just(SafetyIntegrityLevel::SIL2),
            Just(SafetyIntegrityLevel::SIL3),
            Just(SafetyIntegrityLevel::SIL4),
        ]
    ) {
        let latency = sil.max_latency_ns();
        prop_assert!(latency > 0);
    }
}
