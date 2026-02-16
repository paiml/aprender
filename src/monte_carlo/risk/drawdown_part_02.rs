
#[cfg(test)]
mod tests {
    #[allow(clippy::wildcard_imports)]
    use super::super::*;
    use crate::monte_carlo::engine::PathMetadata;

    #[test]
    fn test_max_drawdown_basic() {
        let values = vec![100.0, 110.0, 90.0, 95.0, 85.0, 100.0];
        let max_dd = DrawdownAnalysis::max_drawdown(&values);

        // Peak 110, trough 85: (110-85)/110 = 0.2273
        assert!(
            (max_dd - 0.2273).abs() < 0.01,
            "Max drawdown = {max_dd}, expected ~0.227"
        );
    }

    #[test]
    fn test_max_drawdown_no_drawdown() {
        let values = vec![100.0, 101.0, 102.0, 103.0, 104.0];
        let max_dd = DrawdownAnalysis::max_drawdown(&values);

        assert!(max_dd.abs() < 1e-10, "No drawdown for monotonic increase");
    }

    #[test]
    fn test_max_drawdown_complete_loss() {
        let values = vec![100.0, 50.0, 25.0, 10.0, 0.0];
        let max_dd = DrawdownAnalysis::max_drawdown(&values);

        assert!(
            (max_dd - 1.0).abs() < 1e-10,
            "100% drawdown expected: {max_dd}"
        );
    }

    #[test]
    fn test_drawdown_series() {
        let values = vec![100.0, 110.0, 90.0, 100.0];
        let series = DrawdownAnalysis::drawdown_series(&values);

        assert_eq!(series.len(), 4);
        assert!(series[0].abs() < 1e-10); // No drawdown at start
        assert!(series[1].abs() < 1e-10); // New peak
        assert!((series[2] - (110.0 - 90.0) / 110.0).abs() < 1e-10); // ~0.182
        assert!((series[3] - (110.0 - 100.0) / 110.0).abs() < 1e-10); // ~0.091
    }

    #[test]
    fn test_max_drawdown_duration() {
        let values = vec![100.0, 110.0, 90.0, 95.0, 100.0, 110.0, 115.0];
        let duration = DrawdownAnalysis::max_drawdown_duration(&values);

        // In drawdown from index 2 to 5 (3 periods: 90->95->100->110 recovery)
        // Peak=110 at index 1, drawdown starts at index 2, recovers at index 5
        assert_eq!(duration, 3);
    }

    #[test]
    fn test_ulcer_index() {
        let values = vec![100.0, 110.0, 90.0, 100.0, 110.0];
        let ui = DrawdownAnalysis::ulcer_index(&values);

        assert!(ui >= 0.0);
        assert!(ui.is_finite());
    }

    #[test]
    fn test_pain_index() {
        let values = vec![100.0, 110.0, 90.0, 100.0, 110.0];
        let pi = DrawdownAnalysis::pain_index(&values);

        assert!(pi >= 0.0);
        assert!(pi.is_finite());
    }

    #[test]
    fn test_recovery_factor() {
        // 50% return with 20% max drawdown = 2.5 recovery factor
        let values = vec![100.0, 120.0, 100.0, 150.0];
        let rf = DrawdownAnalysis::recovery_factor(&values);

        // Return: (150-100)/100 = 0.5
        // Max DD: (120-100)/120 = 0.167
        // RF = 0.5 / 0.167 ≈ 3.0
        assert!(rf > 2.0 && rf < 4.0, "Recovery factor = {rf}");
    }

    #[test]
    fn test_drawdown_statistics() {
        let drawdowns = vec![0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.12, 0.08, 0.18, 0.22];
        let stats = DrawdownStatistics::from_drawdowns(&drawdowns);

        assert!(stats.mean > 0.0);
        assert!(stats.median > 0.0);
        assert!(stats.std >= 0.0);
        assert!(stats.worst >= stats.p99);
        assert!(stats.best <= stats.p5);
    }

    #[test]
    fn test_from_paths() {
        let paths: Vec<SimulationPath> = (0..100)
            .map(|i| {
                let values = vec![100.0, 105.0, 95.0, 100.0 + (i as f64 * 0.5)];
                SimulationPath::new(
                    vec![0.0, 0.25, 0.5, 1.0],
                    values,
                    PathMetadata {
                        path_id: i,
                        seed: 42,
                        is_antithetic: false,
                    },
                )
            })
            .collect();

        let stats = DrawdownAnalysis::from_paths(&paths);

        assert!(stats.mean > 0.0);
        assert!(stats.mean < 1.0);
    }

    #[test]
    fn test_exceeds_threshold() {
        let drawdowns = vec![0.05, 0.10, 0.15, 0.20, 0.25];
        let stats = DrawdownStatistics::from_drawdowns(&drawdowns);

        // At 95% confidence, check if drawdown exceeds 0.10
        // p95 should be around 0.24
        assert!(stats.exceeds_threshold(0.10, 0.95));
        assert!(!stats.exceeds_threshold(0.30, 0.95));
    }

    #[test]
    fn test_empty_inputs() {
        assert!(DrawdownAnalysis::max_drawdown(&[]).abs() < 1e-10);
        assert!(DrawdownAnalysis::drawdown_series(&[]).is_empty());
        assert_eq!(DrawdownAnalysis::max_drawdown_duration(&[]), 0);
        assert!(DrawdownAnalysis::ulcer_index(&[]).abs() < 1e-10);
    }

    #[test]
    fn test_single_value() {
        let values = vec![100.0];
        assert!(DrawdownAnalysis::max_drawdown(&values).abs() < 1e-10);
        assert_eq!(DrawdownAnalysis::drawdown_series(&values).len(), 1);
    }

    #[test]
    fn test_recovery_factor_initial_zero() {
        // first <= 0.0 should return 0.0
        let values = vec![0.0, 50.0, 100.0];
        let rf = DrawdownAnalysis::recovery_factor(&values);
        assert!(
            rf.abs() < 1e-10,
            "Recovery factor with zero initial should be 0: {rf}"
        );
    }

    #[test]
    fn test_recovery_factor_initial_negative() {
        let values = vec![-10.0, 50.0, 100.0];
        let rf = DrawdownAnalysis::recovery_factor(&values);
        assert!(
            rf.abs() < 1e-10,
            "Recovery factor with negative initial should be 0: {rf}"
        );
    }

    #[test]
    fn test_recovery_factor_no_drawdown_positive_return() {
        // Monotonically increasing: max_dd = 0, total_return > 0 => Infinity
        let values = vec![100.0, 110.0, 120.0, 130.0];
        let rf = DrawdownAnalysis::recovery_factor(&values);
        assert!(
            rf.is_infinite() && rf > 0.0,
            "Recovery factor with no drawdown and positive return should be Infinity: {rf}"
        );
    }

    #[test]
    fn test_recovery_factor_no_drawdown_no_return() {
        // Flat values: max_dd = 0, total_return = 0 => 0.0
        let values = vec![100.0, 100.0, 100.0];
        let rf = DrawdownAnalysis::recovery_factor(&values);
        assert!(
            rf.abs() < 1e-10,
            "Recovery factor with no drawdown and no return should be 0: {rf}"
        );
    }

    #[test]
    fn test_recovery_factor_single_value() {
        let values = vec![100.0];
        let rf = DrawdownAnalysis::recovery_factor(&values);
        assert!(
            rf.abs() < 1e-10,
            "Recovery factor with single value should be 0: {rf}"
        );
    }

    #[test]
    fn test_recovery_factor_empty() {
        let rf = DrawdownAnalysis::recovery_factor(&[]);
        assert!(
            rf.abs() < 1e-10,
            "Recovery factor with empty should be 0: {rf}"
        );
    }

    #[test]
    fn test_drawdown_series_zero_peak() {
        // peak <= 0 triggers the else branch returning 0.0 for drawdown
        let values = vec![0.0, 0.0, 0.0];
        let series = DrawdownAnalysis::drawdown_series(&values);
        assert_eq!(series.len(), 3);
        for dd in &series {
            assert!(
                dd.abs() < 1e-10,
                "Drawdown with zero peak should be 0: {dd}"
            );
        }
    }

    #[test]
    fn test_drawdown_series_negative_values() {
        // Negative starting values: peak stays at initial negative value
        let values = vec![-10.0, -5.0, -2.0];
        let series = DrawdownAnalysis::drawdown_series(&values);
        assert_eq!(series.len(), 3);
        // Peak starts at -10, then goes to -5 (new peak), then -2 (new peak)
        // All are new peaks, so all drawdowns should be 0
        for dd in &series {
            assert!(
                dd.abs() < 1e-10,
                "Drawdown with rising negative values: {dd}"
            );
        }
    }

    #[test]
    fn test_max_drawdown_zero_peak() {
        // Peak at 0: the else-if branch `peak > 0.0` is false, so drawdown not computed
        let values = vec![0.0, -5.0, -10.0];
        let dd = DrawdownAnalysis::max_drawdown(&values);
        // Peak=0 at start, values go negative but peak > 0.0 check fails
        assert!(dd.abs() < 1e-10, "Max drawdown with zero peak: {dd}");
    }

    #[test]
    fn test_from_paths_empty() {
        let stats = DrawdownAnalysis::from_paths(&[]);
        assert!(stats.mean.abs() < 1e-10);
        assert!(stats.worst.abs() < 1e-10 || stats.worst == f64::NEG_INFINITY);
    }

    #[test]
    fn test_drawdown_statistics_empty() {
        let stats = DrawdownStatistics::from_drawdowns(&[]);
        assert!(stats.mean.abs() < 1e-10);
        assert!(stats.median.abs() < 1e-10);
        assert!(stats.std.abs() < 1e-10);
    }

    #[test]
    fn test_exceeds_threshold_various_confidence_levels() {
        let drawdowns = vec![0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40];
        let stats = DrawdownStatistics::from_drawdowns(&drawdowns);

        // Test confidence >= 0.99 => uses p99
        assert!(stats.exceeds_threshold(0.01, 0.99));

        // Test confidence >= 0.95 but < 0.99 => uses p95
        assert!(stats.exceeds_threshold(0.01, 0.96));

        // Test confidence >= 0.75 but < 0.95 => uses p75
        assert!(stats.exceeds_threshold(0.01, 0.80));

        // Test confidence >= 0.50 but < 0.75 => uses median
        assert!(stats.exceeds_threshold(0.01, 0.60));

        // Test confidence < 0.50 => uses p25
        assert!(stats.exceeds_threshold(0.01, 0.30));

        // Test threshold too high for each level
        assert!(!stats.exceeds_threshold(0.99, 0.30));
    }

    #[test]
    fn test_max_drawdown_duration_no_drawdown() {
        // Monotonically increasing: never in drawdown
        let values = vec![100.0, 110.0, 120.0, 130.0];
        let duration = DrawdownAnalysis::max_drawdown_duration(&values);
        assert_eq!(duration, 0);
    }

    #[test]
    fn test_max_drawdown_duration_single() {
        let values = vec![100.0];
        let duration = DrawdownAnalysis::max_drawdown_duration(&values);
        assert_eq!(duration, 0);
    }

    #[test]
    fn test_pain_index_empty() {
        assert!(DrawdownAnalysis::pain_index(&[]).abs() < 1e-10);
    }

    // Property-based tests
    #[cfg(test)]
    mod proptests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn prop_max_drawdown_bounded(values in prop::collection::vec(1.0..1000.0f64, 10..100)) {
                let dd = DrawdownAnalysis::max_drawdown(&values);
                prop_assert!(dd >= 0.0 && dd <= 1.0, "Drawdown must be in [0, 1]: {dd}");
            }

            #[test]
            fn prop_drawdown_series_same_length(values in prop::collection::vec(1.0..1000.0f64, 1..100)) {
                let series = DrawdownAnalysis::drawdown_series(&values);
                prop_assert_eq!(series.len(), values.len());
            }

            #[test]
            fn prop_drawdown_series_non_negative(values in prop::collection::vec(1.0..1000.0f64, 1..100)) {
                let series = DrawdownAnalysis::drawdown_series(&values);
                for dd in series {
                    prop_assert!(dd >= 0.0, "Drawdown must be non-negative: {dd}");
                }
            }

            #[test]
            fn prop_ulcer_index_non_negative(values in prop::collection::vec(1.0..1000.0f64, 10..100)) {
                let ui = DrawdownAnalysis::ulcer_index(&values);
                prop_assert!(ui >= 0.0 && ui.is_finite());
            }

            #[test]
            fn prop_pain_leq_ulcer(values in prop::collection::vec(1.0..1000.0f64, 10..100)) {
                let pi = DrawdownAnalysis::pain_index(&values);
                let ui = DrawdownAnalysis::ulcer_index(&values);
                // Pain (linear mean) <= Ulcer (quadratic mean) by Jensen's inequality
                // With small tolerance for numerical errors
                prop_assert!(pi <= ui + 0.001, "Pain {pi} should be <= Ulcer {ui}");
            }
        }
    }
}
