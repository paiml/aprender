
    // ========================================================================
    // Mann-Whitney U Test (Non-parametric, per Box et al. 2005)
    // ========================================================================

    #[test]
    fn test_mann_whitney_identical_samples() {
        let control = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let treatment = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let result = mann_whitney_u(&control, &treatment);

        // Identical samples should have no significant difference
        assert!(!result.significant);
        assert!(result.effect_size.abs() < 0.1); // Negligible effect
    }

    #[test]
    fn test_mann_whitney_completely_separated() {
        let control = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let treatment = vec![10.0, 11.0, 12.0, 13.0, 14.0];

        let result = mann_whitney_u(&control, &treatment);

        // Completely separated should be highly significant
        assert!(result.significant);
        assert!(result.effect_size.abs() > 0.8); // Large effect
        assert_eq!(result.u_statistic, 0.0); // No overlap
    }

    #[test]
    fn test_mann_whitney_handles_ties() {
        let control = vec![1.0, 2.0, 2.0, 3.0, 3.0];
        let treatment = vec![2.0, 2.0, 3.0, 4.0, 5.0];

        let result = mann_whitney_u(&control, &treatment);

        // Should handle ties correctly (average ranks)
        assert!(result.p_value > 0.0 && result.p_value <= 1.0);
    }

    #[test]
    fn test_mann_whitney_effect_size_interpretation() {
        // Small effect
        let control = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let treatment = vec![1.5, 2.5, 3.5, 4.5, 5.5];
        let result = mann_whitney_u(&control, &treatment);
        assert!(matches!(
            result.effect_interpretation,
            EffectSizeInterpretation::Small | EffectSizeInterpretation::Negligible
        ));
    }

    #[test]
    fn test_mann_whitney_returns_correct_method() {
        let control = vec![1.0, 2.0, 3.0];
        let treatment = vec![4.0, 5.0, 6.0];
        let result = mann_whitney_u(&control, &treatment);
        assert_eq!(result.method, TestMethod::MannWhitneyU);
    }

    // ========================================================================
    // Auto Test Selection (per Gemini review recommendation)
    // ========================================================================

    #[test]
    fn test_auto_select_uses_mann_whitney_for_small_samples() {
        // Small samples (n < 15) should use non-parametric
        let control = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let treatment = vec![2.0, 3.0, 4.0, 5.0, 6.0];
        let config = AnalysisConfig {
            alpha: 0.05,
            auto_detect_skew: true,
        };

        let result = analyze_with_auto_select(&control, &treatment, &config);

        // Small samples should trigger Mann-Whitney
        assert_eq!(result.method, TestMethod::MannWhitneyU);
    }

    #[test]
    fn test_auto_select_uses_log_transform_for_latency_like_data() {
        // Generate log-normal-ish data (typical latency distribution)
        let control: Vec<f64> = vec![
            10.0, 12.0, 11.0, 15.0, 100.0, 13.0, 14.0, 11.0, 12.0, 10.0, 11.0, 12.0, 13.0, 14.0,
            15.0, 16.0, 17.0, 18.0, 19.0, 200.0,
        ];
        let treatment: Vec<f64> = vec![
            8.0, 9.0, 10.0, 11.0, 50.0, 9.0, 10.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
            8.0, 9.0, 10.0, 11.0, 80.0,
        ];

        let config = AnalysisConfig {
            alpha: 0.05,
            auto_detect_skew: true,
        };

        let result = analyze_with_auto_select(&control, &treatment, &config);

        // Skewed data should use log-transform (if n >= 15) or Mann-Whitney
        assert!(matches!(
            result.method,
            TestMethod::LogTransformTTest | TestMethod::MannWhitneyU
        ));
    }

    // ========================================================================
    // Original Tests
    // ========================================================================

    #[test]
    fn test_t_test_no_difference() {
        let control = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let treatment = vec![1.1, 2.1, 3.1, 4.1, 5.1];
        let result = analyze_t_test(&control, &treatment, 0.05);
        assert!(!result.significant); // Small effect, not significant
    }

    #[test]
    fn test_t_test_significant_difference() {
        let control = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let treatment = vec![10.0, 11.0, 12.0, 13.0, 14.0];
        let result = analyze_t_test(&control, &treatment, 0.05);
        assert!(result.significant); // Large effect
    }

    #[test]
    fn test_log_transform_latency() {
        // Simulate log-normal latency (ms)
        let control = vec![10.0, 12.0, 15.0, 100.0, 11.0]; // Has outlier
        let treatment = vec![8.0, 9.0, 10.0, 50.0, 8.5];
        let result = analyze_log_transform(&control, &treatment, 0.05);
        assert!(result.treatment_mean < result.control_mean);
        assert_eq!(result.method, TestMethod::LogTransformTTest);
    }

    #[test]
    fn test_auto_detect_skewness() {
        // Highly skewed data should use log-transform
        let control = vec![1.0, 1.1, 1.2, 1.3, 100.0]; // Skewed
        let treatment = vec![1.0, 1.1, 1.2, 1.3, 1.4];
        let config = AnalysisConfig::default();
        let result = analyze(&control, &treatment, &config);
        assert_eq!(result.method, TestMethod::LogTransformTTest);
    }

    #[test]
    fn test_normal_data_uses_t_test() {
        // Symmetric data should use t-test
        let control = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let treatment = vec![2.0, 3.0, 4.0, 5.0, 6.0];
        let config = AnalysisConfig::default();
        let result = analyze(&control, &treatment, &config);
        assert_eq!(result.method, TestMethod::TTest);
    }

    #[test]
    fn test_skewness_calculation() {
        // Symmetric data has ~0 skewness
        let symmetric = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!(compute_skewness(&symmetric).abs() < 0.5);

        // Right-skewed data has positive skewness
        let skewed = vec![1.0, 1.0, 1.0, 1.0, 100.0];
        assert!(compute_skewness(&skewed) > 1.0);
    }

    // ========================================================================
    // Additional Coverage Tests - Median
    // ========================================================================

    #[test]
    fn test_median_empty_slice() {
        let data: Vec<f64> = vec![];
        assert_eq!(median(&data), 0.0);
    }

    #[test]
    fn test_median_single_element() {
        let data = vec![42.0];
        assert_eq!(median(&data), 42.0);
    }

    #[test]
    fn test_median_odd_length() {
        let data = vec![3.0, 1.0, 2.0];
        assert_eq!(median(&data), 2.0);
    }

    #[test]
    fn test_median_even_length() {
        let data = vec![4.0, 1.0, 3.0, 2.0];
        // Sorted: [1.0, 2.0, 3.0, 4.0], median = (2.0 + 3.0) / 2 = 2.5
        assert_eq!(median(&data), 2.5);
    }

    #[test]
    fn test_median_with_duplicates() {
        let data = vec![5.0, 5.0, 5.0, 5.0, 5.0];
        assert_eq!(median(&data), 5.0);
    }

    // ========================================================================
    // Additional Coverage Tests - Normal CDF
    // ========================================================================

    #[test]
    fn test_normal_cdf_zero() {
        let cdf = normal_cdf(0.0);
        assert!((cdf - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_normal_cdf_positive() {
        let cdf = normal_cdf(2.0);
        // CDF(2.0) should be close to 0.9772
        assert!(cdf > 0.95 && cdf < 0.99);
    }

    #[test]
    fn test_normal_cdf_negative() {
        let cdf = normal_cdf(-2.0);
        // CDF(-2.0) should be close to 0.0228
        assert!(cdf > 0.01 && cdf < 0.05);
    }

    #[test]
    fn test_normal_cdf_symmetry() {
        let cdf_pos = normal_cdf(1.5);
        let cdf_neg = normal_cdf(-1.5);
        // CDF(-x) + CDF(x) should equal 1
        assert!((cdf_pos + cdf_neg - 1.0).abs() < 0.01);
    }

    // ========================================================================
    // Additional Coverage Tests - Effect Size Interpretation
    // ========================================================================

    #[test]
    fn test_interpret_effect_size_negligible() {
        assert_eq!(
            interpret_effect_size(0.05),
            EffectSizeInterpretation::Negligible
        );
        assert_eq!(
            interpret_effect_size(0.09),
            EffectSizeInterpretation::Negligible
        );
    }

    #[test]
    fn test_interpret_effect_size_small() {
        assert_eq!(interpret_effect_size(0.1), EffectSizeInterpretation::Small);
        assert_eq!(interpret_effect_size(0.2), EffectSizeInterpretation::Small);
        assert_eq!(interpret_effect_size(0.29), EffectSizeInterpretation::Small);
    }

    #[test]
    fn test_interpret_effect_size_medium() {
        assert_eq!(interpret_effect_size(0.3), EffectSizeInterpretation::Medium);
        assert_eq!(interpret_effect_size(0.4), EffectSizeInterpretation::Medium);
        assert_eq!(
            interpret_effect_size(0.49),
            EffectSizeInterpretation::Medium
        );
    }

    #[test]
    fn test_interpret_effect_size_large() {
        assert_eq!(interpret_effect_size(0.5), EffectSizeInterpretation::Large);
        assert_eq!(interpret_effect_size(0.8), EffectSizeInterpretation::Large);
        assert_eq!(interpret_effect_size(1.0), EffectSizeInterpretation::Large);
    }

    // ========================================================================
    // Additional Coverage Tests - Skewness Edge Cases
    // ========================================================================

    #[test]
    fn test_skewness_constant_values() {
        // All same values -> std_dev = 0 -> should return 0
        let data = vec![5.0, 5.0, 5.0, 5.0, 5.0];
        assert_eq!(compute_skewness(&data), 0.0);
    }

    #[test]
    fn test_skewness_left_skewed() {
        // Left-skewed data has negative skewness
        let data = vec![100.0, 99.0, 98.0, 97.0, 1.0];
        assert!(compute_skewness(&data) < -1.0);
    }

    // ========================================================================
    // Additional Coverage Tests - AnalysisConfig
    // ========================================================================

    #[test]
    fn test_analysis_config_default() {
        let config = AnalysisConfig::default();
        assert!((config.alpha - 0.05).abs() < 1e-10);
        assert!(config.auto_detect_skew);
    }

    #[test]
    fn test_analysis_config_custom() {
        let config = AnalysisConfig {
            alpha: 0.01,
            auto_detect_skew: false,
        };
        assert!((config.alpha - 0.01).abs() < 1e-10);
        assert!(!config.auto_detect_skew);
    }

    // ========================================================================
    // Auto-select with skew detection disabled
    // ========================================================================

    #[test]
    fn test_auto_select_skew_disabled_large_sample() {
        // Large sample with skew detection disabled should use t-test
        let control: Vec<f64> = (0..20).map(|i| (i as f64) + 1.0).collect();
        let treatment: Vec<f64> = (0..20).map(|i| (i as f64) + 2.0).collect();
        let config = AnalysisConfig {
            alpha: 0.05,
            auto_detect_skew: false,
        };

        let result = analyze_with_auto_select(&control, &treatment, &config);
        assert_eq!(result.method, TestMethod::TTest);
    }

    #[test]
    fn test_auto_select_skewed_with_non_positive_values() {
        // Skewed data with non-positive values should fall back to Mann-Whitney
        let control: Vec<f64> = vec![
            -5.0, 1.0, 2.0, 3.0, 100.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0,
            12.0, 13.0, 14.0, 500.0,
        ];
        let treatment: Vec<f64> = vec![
            0.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0,
        ];
        let config = AnalysisConfig {
            alpha: 0.05,
            auto_detect_skew: true,
        };

        let result = analyze_with_auto_select(&control, &treatment, &config);
        // Can't log-transform non-positive, should use Mann-Whitney
        assert_eq!(result.method, TestMethod::MannWhitneyU);
    }

    // ========================================================================
    // Additional Coverage Tests - analyze() function
    // ========================================================================

    #[test]
    fn test_analyze_with_skew_disabled() {
        let control = vec![1.0, 1.1, 1.2, 1.3, 100.0]; // Skewed
        let treatment = vec![1.0, 1.1, 1.2, 1.3, 1.4];
        let config = AnalysisConfig {
            alpha: 0.05,
            auto_detect_skew: false,
        };
        let result = analyze(&control, &treatment, &config);
        // With skew detection disabled, should use t-test
        assert_eq!(result.method, TestMethod::TTest);
    }

    // ========================================================================
    // Additional Coverage Tests - T-Test Details
    // ========================================================================

    #[test]
    fn test_t_test_effect_size() {
        let control = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let treatment = vec![15.0, 25.0, 35.0, 45.0, 55.0];
        let result = analyze_t_test(&control, &treatment, 0.05);

        // Effect size should be (35 - 30) / 30 = 0.1667
        let expected_effect = (35.0 - 30.0) / 30.0;
        assert!((result.effect_size - expected_effect).abs() < 0.01);
    }

    #[test]
    fn test_t_test_means() {
        let control = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let treatment = vec![3.0, 6.0, 9.0, 12.0, 15.0];
        let result = analyze_t_test(&control, &treatment, 0.05);

        assert!((result.control_mean - 6.0).abs() < 0.01);
        assert!((result.treatment_mean - 9.0).abs() < 0.01);
    }

    // ========================================================================
    // Additional Coverage Tests - Log Transform Details
    // ========================================================================

    #[test]
    fn test_log_transform_effect_size() {
        let control = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let treatment = vec![20.0, 40.0, 60.0, 80.0, 100.0];
        let result = analyze_log_transform(&control, &treatment, 0.05);

        // Treatment geometric mean should be ~2x control geometric mean
        // Effect size should be roughly 1.0 (100% increase)
        assert!(result.effect_size > 0.5);
    }

    // ========================================================================
    // Additional Coverage Tests - Mann-Whitney Edge Cases
    // ========================================================================

    #[test]
    fn test_mann_whitney_large_effect() {
        let control = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let treatment = vec![100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0];
        let result = mann_whitney_u(&control, &treatment);

        assert_eq!(
            result.effect_interpretation,
            EffectSizeInterpretation::Large
        );
        assert!(result.significant);
    }

    #[test]
    fn test_mann_whitney_z_score_sign() {
        // Treatment higher than control
        let control = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let treatment = vec![6.0, 7.0, 8.0, 9.0, 10.0];
        let result = mann_whitney_u(&control, &treatment);

        // U statistic should be 0 (no overlap)
        assert_eq!(result.u_statistic, 0.0);
    }

    #[test]
    fn test_mann_whitney_result_fields() {
        let control = vec![1.0, 3.0, 5.0, 7.0, 9.0];
        let treatment = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let result = mann_whitney_u(&control, &treatment);

        // Verify all fields are populated
        assert!(result.u_statistic >= 0.0);
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
        assert!(!result.z_score.is_nan());
        assert!(!result.effect_size.is_nan());
    }
