//! Property-based tests for monitor module

use crate::monitor::*;
use proptest::prelude::*;

proptest! {
    #[test]
    fn prop_mean_within_bounds(values in proptest::collection::vec(-1000.0f64..1000.0, 1..100)) {
        let mut collector = MetricsCollector::new();
        let min_val = values.iter().copied().fold(f64::INFINITY, f64::min);
        let max_val = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        for v in &values {
            collector.record(Metric::Loss, *v);
        }

        let summary = collector.summary();
        let stats = summary.get(&Metric::Loss).expect("key should exist");

        prop_assert!(stats.mean >= min_val);
        prop_assert!(stats.mean <= max_val);
    }

    #[test]
    fn prop_std_non_negative(values in proptest::collection::vec(-1000.0f64..1000.0, 2..100)) {
        let mut collector = MetricsCollector::new();
        for v in &values {
            collector.record(Metric::Loss, *v);
        }

        let summary = collector.summary();
        let stats = summary.get(&Metric::Loss).expect("key should exist");

        prop_assert!(stats.std >= 0.0);
    }

    #[test]
    fn prop_count_matches_insertions(values in proptest::collection::vec(-1000.0f64..1000.0, 1..100)) {
        let mut collector = MetricsCollector::new();
        for v in &values {
            collector.record(Metric::Loss, *v);
        }

        let summary = collector.summary();
        let stats = summary.get(&Metric::Loss).expect("key should exist");

        prop_assert_eq!(stats.count, values.len());
    }

    #[test]
    fn prop_min_max_correct(values in proptest::collection::vec(-1000.0f64..1000.0, 1..100)) {
        let mut collector = MetricsCollector::new();
        let expected_min = values.iter().copied().fold(f64::INFINITY, f64::min);
        let expected_max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        for v in &values {
            collector.record(Metric::Loss, *v);
        }

        let summary = collector.summary();
        let stats = summary.get(&Metric::Loss).expect("key should exist");

        prop_assert!((stats.min - expected_min).abs() < 1e-10);
        prop_assert!((stats.max - expected_max).abs() < 1e-10);
    }
}
