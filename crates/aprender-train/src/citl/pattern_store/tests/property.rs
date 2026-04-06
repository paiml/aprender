//! Property-based tests for pattern store.

use super::*;
use proptest::prelude::*;

proptest! {
    #[test]
    fn prop_fix_pattern_success_rate_bounded(
        successes in 0u32..100,
        failures in 0u32..100
    ) {
        let mut pattern = FixPattern::new("E0308", "diff");
        for _ in 0..successes {
            pattern.record_success();
        }
        for _ in 0..failures {
            pattern.record_failure();
        }

        let rate = pattern.success_rate();
        prop_assert!(rate >= 0.0);
        prop_assert!(rate <= 1.0);
    }

    #[test]
    fn prop_fix_pattern_searchable_contains_error_code(
        error_code in "[A-Z][0-9]{4}"
    ) {
        let pattern = FixPattern::new(&error_code, "diff");
        let text = pattern.to_searchable_text();
        prop_assert!(text.contains(&error_code));
    }

    #[test]
    fn prop_chunk_id_serialization_roundtrip(n in 0u128..1000) {
        let id = ChunkId(uuid::Uuid::from_u128(n));
        let json = serde_json::to_string(&id).expect("JSON serialization should succeed");
        let deserialized: ChunkId = serde_json::from_str(&json).expect("JSON deserialization should succeed");
        prop_assert_eq!(id, deserialized);
    }

    #[test]
    fn prop_suggestion_weighted_score_positive(
        score in 0.0f32..1.0,
        successes in 0u32..10,
        attempts in 1u32..20
    ) {
        let mut pattern = FixPattern::new("E0308", "diff");
        let actual_attempts = attempts.max(successes);
        for _ in 0..successes {
            pattern.record_success();
        }
        for _ in 0..(actual_attempts - successes) {
            pattern.record_failure();
        }

        let suggestion = FixSuggestion::new(pattern, score, 0);
        prop_assert!(suggestion.weighted_score() >= 0.0);
    }

    #[test]
    fn prop_apr_roundtrip_preserves_patterns(
        error_codes in proptest::collection::vec("[A-Z][0-9]{4}", 1..5),
        successes in proptest::collection::vec(0u32..10, 1..5)
    ) {
        let mut store = DecisionPatternStore::new().expect("operation should succeed");

        for (i, code) in error_codes.iter().enumerate() {
            let mut pattern = FixPattern::new(code, format!("fix{i}"));
            let success_count = successes.get(i).copied().unwrap_or(0);
            for _ in 0..success_count {
                pattern.record_success();
            }
            store.index_fix(pattern).expect("operation should succeed");
        }

        let temp_dir = tempfile::tempdir().expect("temp file creation should succeed");
        let path = temp_dir.path().join("patterns.apr");

        store.save_apr(&path).expect("save should succeed");
        let loaded = DecisionPatternStore::load_apr(&path).expect("load should succeed");

        prop_assert_eq!(store.len(), loaded.len());

        for code in &error_codes {
            let orig = store.patterns_for_error(code);
            let load = loaded.patterns_for_error(code);
            prop_assert_eq!(orig.len(), load.len());
        }
    }
}
