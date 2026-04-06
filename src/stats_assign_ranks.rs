
#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // Additional Coverage Tests - Rank Assignment
    // ========================================================================

    #[test]
    fn test_assign_ranks_no_ties() {
        let sorted = vec![(1.0, 0), (2.0, 1), (3.0, 0), (4.0, 1)];
        let ranks = assign_ranks_with_ties(&sorted);

        assert_eq!(ranks.len(), 4);
        assert!((ranks[0].0 - 1.0).abs() < 1e-10);
        assert!((ranks[1].0 - 2.0).abs() < 1e-10);
        assert!((ranks[2].0 - 3.0).abs() < 1e-10);
        assert!((ranks[3].0 - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_assign_ranks_with_multiple_tie_groups() {
        // Two tie groups: (1.0, 1.0) and (3.0, 3.0)
        let sorted = vec![(1.0, 0), (1.0, 1), (3.0, 0), (3.0, 1)];
        let ranks = assign_ranks_with_ties(&sorted);

        // First two should have average rank 1.5
        assert!((ranks[0].0 - 1.5).abs() < 1e-10);
        assert!((ranks[1].0 - 1.5).abs() < 1e-10);
        // Last two should have average rank 3.5
        assert!((ranks[2].0 - 3.5).abs() < 1e-10);
        assert!((ranks[3].0 - 3.5).abs() < 1e-10);
    }

    #[test]
    fn test_assign_ranks_preserves_groups() {
        let sorted = vec![(1.0, 0), (2.0, 1), (3.0, 0)];
        let ranks = assign_ranks_with_ties(&sorted);

        assert_eq!(ranks[0].1, 0);
        assert_eq!(ranks[1].1, 1);
        assert_eq!(ranks[2].1, 0);
    }

    // ========================================================================
    // Additional Coverage Tests - AnalysisResult Fields
    // ========================================================================

    #[test]
    fn test_analysis_result_significant_flag() {
        // Test with alpha = 0.10 (more lenient)
        let control = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let treatment = vec![3.0, 4.0, 5.0, 6.0, 7.0];
        let result = analyze_t_test(&control, &treatment, 0.10);

        // With larger alpha, more likely to be significant
        // p_value < alpha should set significant = true
        assert_eq!(result.significant, result.p_value < 0.10);
    }

    // ========================================================================
    // Additional Coverage Tests - TestMethod Enum
    // ========================================================================

    #[test]
    fn test_test_method_equality() {
        assert_eq!(TestMethod::TTest, TestMethod::TTest);
        assert_eq!(TestMethod::LogTransformTTest, TestMethod::LogTransformTTest);
        assert_eq!(TestMethod::MannWhitneyU, TestMethod::MannWhitneyU);
        assert_ne!(TestMethod::TTest, TestMethod::MannWhitneyU);
    }

    #[test]
    fn test_test_method_clone() {
        let method = TestMethod::LogTransformTTest;
        let cloned = method.clone();
        assert_eq!(method, cloned);
    }

    // ========================================================================
    // Additional Coverage Tests - EffectSizeInterpretation Enum
    // ========================================================================

    #[test]
    fn test_effect_size_interpretation_equality() {
        assert_eq!(
            EffectSizeInterpretation::Negligible,
            EffectSizeInterpretation::Negligible
        );
        assert_ne!(
            EffectSizeInterpretation::Small,
            EffectSizeInterpretation::Large
        );
    }

    #[test]
    fn test_effect_size_interpretation_clone() {
        let interp = EffectSizeInterpretation::Medium;
        let cloned = interp;
        assert_eq!(interp, cloned);
    }

    // ========================================================================
    // Additional Coverage Tests - MannWhitneyResult
    // ========================================================================

    #[test]
    fn test_mann_whitney_result_clone() {
        let control = vec![1.0, 2.0, 3.0];
        let treatment = vec![4.0, 5.0, 6.0];
        let result = mann_whitney_u(&control, &treatment);
        let cloned = result.clone();

        assert_eq!(result.u_statistic, cloned.u_statistic);
        assert_eq!(result.p_value, cloned.p_value);
        assert_eq!(result.method, cloned.method);
    }

    // ========================================================================
    // Additional Coverage Tests - AnalysisResult Clone
    // ========================================================================

    #[test]
    fn test_analysis_result_clone() {
        let control = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let treatment = vec![2.0, 3.0, 4.0, 5.0, 6.0];
        let result = analyze_t_test(&control, &treatment, 0.05);
        let cloned = result.clone();

        assert_eq!(result.control_mean, cloned.control_mean);
        assert_eq!(result.treatment_mean, cloned.treatment_mean);
        assert_eq!(result.method, cloned.method);
    }

    // ========================================================================
    // Additional Coverage Tests - AnalysisConfig Clone
    // ========================================================================

    #[test]
    fn test_analysis_config_clone() {
        let config = AnalysisConfig {
            alpha: 0.01,
            auto_detect_skew: true,
        };
        let cloned = config.clone();

        assert_eq!(config.alpha, cloned.alpha);
        assert_eq!(config.auto_detect_skew, cloned.auto_detect_skew);
    }
include!("stats_mann_whitney.rs");
}
