
#[cfg(test)]
mod tests {
    use super::*;

    /// Test: One-sample t-test
    #[test]
    fn test_ttest_1samp() {
        // Sample: [2.3, 2.5, 2.7, 2.9, 3.1]
        // Mean ≈ 2.7, testing against μ₀ = 2.5
        let sample = vec![2.3, 2.5, 2.7, 2.9, 3.1];
        let result = ttest_1samp(&sample, 2.5).expect("Valid t-test");

        // t = (2.7 - 2.5) / (s/√5)
        assert!(result.statistic > 0.0, "t-statistic should be positive");
        assert!(result.pvalue > 0.0 && result.pvalue < 1.0);
        assert_eq!(result.df, 4.0);
    }

    /// Test: Independent two-sample t-test (equal variances)
    #[test]
    fn test_ttest_ind_equal_var() {
        let group1 = vec![2.3, 2.5, 2.7, 2.9, 3.1];
        let group2 = vec![3.2, 3.4, 3.6, 3.8, 4.0];

        let result = ttest_ind(&group1, &group2, true).expect("Valid t-test");

        // Group2 mean > Group1 mean, so t should be negative
        assert!(result.statistic < 0.0, "t-statistic should be negative");
        assert!(result.pvalue > 0.0 && result.pvalue < 1.0);
        assert_eq!(result.df, 8.0); // n1 + n2 - 2 = 5 + 5 - 2 = 8
    }

    /// Test: Independent two-sample t-test (unequal variances - Welch's)
    #[test]
    fn test_ttest_ind_unequal_var() {
        let group1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let group2 = vec![10.0, 11.0, 12.0];

        let result = ttest_ind(&group1, &group2, false).expect("Valid Welch's t-test");

        assert!(result.statistic < 0.0); // group1 < group2
        assert!(result.pvalue > 0.0 && result.pvalue < 1.0);
        assert!(result.df > 0.0); // Welch-Satterthwaite df
    }

    /// Test: Paired t-test
    #[test]
    fn test_ttest_rel() {
        // Before-after measurements
        let before = vec![120.0, 122.0, 125.0, 128.0, 130.0];
        let after = vec![115.0, 118.0, 120.0, 123.0, 125.0];

        let result = ttest_rel(&before, &after).expect("Valid paired t-test");

        // After < Before, differences should be positive
        assert!(result.statistic > 0.0);
        assert!(result.pvalue > 0.0 && result.pvalue < 1.0);
        assert_eq!(result.df, 4.0); // n - 1
    }

    /// Test: Paired t-test with dimension mismatch
    #[test]
    fn test_ttest_rel_dimension_mismatch() {
        let sample1 = vec![1.0, 2.0, 3.0];
        let sample2 = vec![4.0, 5.0]; // Different size!

        let result = ttest_rel(&sample1, &sample2);
        assert!(result.is_err());
        let err = result.expect_err("Should be a dimension mismatch error");
        assert!(matches!(err, AprenderError::DimensionMismatch { .. }));
    }

    /// Test: t-test with insufficient data
    #[test]
    fn test_ttest_insufficient_data() {
        let sample = vec![1.0]; // Only 1 sample!
        let result = ttest_1samp(&sample, 0.0);
        assert!(result.is_err());
    }

    /// Test: Chi-square goodness-of-fit
    #[test]
    fn test_chisquare_goodness_of_fit() {
        // Testing if a die is fair
        // Observed: [8, 12, 10, 15, 9, 6] (60 rolls)
        // Expected: [10, 10, 10, 10, 10, 10] (uniform)
        let observed = vec![8.0, 12.0, 10.0, 15.0, 9.0, 6.0];
        let expected = vec![10.0, 10.0, 10.0, 10.0, 10.0, 10.0];

        let result = chisquare(&observed, &expected).expect("Valid chi-square test");

        // χ² = Σ (O-E)²/E
        assert!(result.statistic > 0.0);
        assert!(result.pvalue > 0.0 && result.pvalue < 1.0);
        assert_eq!(result.df, 5); // k - 1 = 6 - 1 = 5
    }

    /// Test: Chi-square with dimension mismatch
    #[test]
    fn test_chisquare_dimension_mismatch() {
        let observed = vec![10.0, 20.0, 30.0];
        let expected = vec![15.0, 25.0]; // Different size!

        let result = chisquare(&observed, &expected);
        assert!(result.is_err());
        let err = result.expect_err("Should be a dimension mismatch error");
        assert!(matches!(err, AprenderError::DimensionMismatch { .. }));
    }

    /// Test: Chi-square with invalid expected frequencies
    #[test]
    fn test_chisquare_invalid_expected() {
        let observed = vec![10.0, 20.0, 30.0];
        let expected = vec![15.0, 0.0, 25.0]; // Zero is invalid!

        let result = chisquare(&observed, &expected);
        assert!(result.is_err());
    }

    /// Test: One-way ANOVA with multiple groups
    #[test]
    fn test_f_oneway() {
        // Three groups with different means
        let group1 = vec![2.0, 2.5, 3.0, 2.8, 2.7];
        let group2 = vec![3.5, 3.8, 4.0, 3.7, 3.9];
        let group3 = vec![5.0, 5.2, 4.8, 5.1, 4.9];

        let result = f_oneway(&[group1, group2, group3]).expect("Valid ANOVA");

        // Groups have different means, F should be large
        assert!(result.statistic > 0.0);
        assert!(result.pvalue > 0.0 && result.pvalue < 1.0);
        assert_eq!(result.df_between, 2); // k - 1 = 3 - 1 = 2
        assert_eq!(result.df_within, 12); // n_total - k = 15 - 3 = 12
    }

    /// Test: ANOVA with identical groups (no difference)
    #[test]
    fn test_f_oneway_no_difference() {
        // Three groups with identical values
        let group1 = vec![3.0, 3.0, 3.0];
        let group2 = vec![3.0, 3.0, 3.0];
        let group3 = vec![3.0, 3.0, 3.0];

        let result = f_oneway(&[group1, group2, group3]).expect("Valid ANOVA");

        // No variance between groups, F should be ~0 (or NaN if MSwithin=0)
        assert!(result.statistic >= 0.0 || result.statistic.is_nan());
    }

    /// Test: ANOVA with insufficient groups
    #[test]
    fn test_f_oneway_insufficient_groups() {
        let group1 = vec![1.0, 2.0, 3.0];
        let result = f_oneway(&[group1]);
        assert!(result.is_err());
    }

    /// Test: ANOVA with empty group
    #[test]
    fn test_f_oneway_empty_group() {
        let group1 = vec![1.0, 2.0, 3.0];
        let group2 = vec![]; // Empty!
        let group3 = vec![4.0, 5.0, 6.0];

        let result = f_oneway(&[group1, group2, group3]);
        assert!(result.is_err());
    }

    /// Test: Normal CDF approximation
    #[test]
    fn test_normal_cdf() {
        // Standard normal values
        assert!((normal_cdf(0.0) - 0.5).abs() < 0.01);
        assert!(normal_cdf(1.96) > 0.97); // ~0.975
        assert!(normal_cdf(-1.96) < 0.03); // ~0.025
    }

    /// Test: Error function
    #[test]
    fn test_erf() {
        assert!((erf(0.0) - 0.0).abs() < 0.01);
        assert!(erf(1.0) > 0.8); // erf(1) ≈ 0.8427
        assert!(erf(-1.0) < -0.8); // erf(-1) ≈ -0.8427
    }

    /// Test: Gamma function
    #[test]
    fn test_gamma() {
        // Γ(1) = 1
        assert!((gamma(1.0) - 1.0).abs() < 0.1);
        // Γ(2) = 1! = 1
        assert!((gamma(2.0) - 1.0).abs() < 0.1);
        // Γ(3) = 2! = 2
        assert!((gamma(3.0) - 2.0).abs() < 0.2);
        // Γ(4) = 3! = 6
        assert!((gamma(4.0) - 6.0).abs() < 0.5);
    }

    /// Test: Real-world example - comparing two treatments
    #[test]
    fn test_real_world_treatment_comparison() {
        // Control group vs treatment group (blood pressure reduction)
        let control = vec![5.0, 7.0, 6.0, 8.0, 5.5, 6.5];
        let treatment = vec![12.0, 14.0, 13.0, 15.0, 11.0, 13.5];

        let result = ttest_ind(&control, &treatment, true).expect("Valid comparison");

        // Treatment should show significantly higher reduction
        assert!(result.statistic < 0.0); // control < treatment
                                         // With this difference, p-value should be small (< 0.05 typically)
        assert!(result.pvalue < 0.1, "Should show significant difference");
    }

    // =========================================================================
    // Additional coverage: Debug/Clone on result structs
    // =========================================================================

    #[test]
    fn test_ttest_result_debug_clone() {
        let result = ttest_1samp(&[2.0, 3.0, 4.0, 5.0, 6.0], 4.0).expect("Valid t-test");
        let debug_str = format!("{:?}", result);
        assert!(debug_str.contains("TTestResult"));
        assert!(debug_str.contains("statistic"));
        let cloned = result.clone();
        assert!((cloned.statistic - result.statistic).abs() < 1e-10);
        assert!((cloned.pvalue - result.pvalue).abs() < 1e-10);
        assert!((cloned.df - result.df).abs() < 1e-10);
    }

    #[test]
    fn test_chi_square_result_debug_clone() {
        let observed = vec![10.0, 20.0, 30.0];
        let expected = vec![20.0, 20.0, 20.0];
        let result = chisquare(&observed, &expected).expect("Valid chi-square");
        let debug_str = format!("{:?}", result);
        assert!(debug_str.contains("ChiSquareResult"));
        let cloned = result.clone();
        assert!((cloned.statistic - result.statistic).abs() < 1e-10);
        assert_eq!(cloned.df, result.df);
    }

    #[test]
    fn test_anova_result_debug_clone() {
        let g1 = vec![1.0, 2.0, 3.0];
        let g2 = vec![4.0, 5.0, 6.0];
        let result = f_oneway(&[g1, g2]).expect("Valid ANOVA");
        let debug_str = format!("{:?}", result);
        assert!(debug_str.contains("AnovaResult"));
        let cloned = result.clone();
        assert!((cloned.statistic - result.statistic).abs() < 1e-10);
        assert_eq!(cloned.df_between, result.df_between);
        assert_eq!(cloned.df_within, result.df_within);
    }

    // =========================================================================
    // Additional coverage: error return paths
    // =========================================================================

    #[test]
    fn test_ttest_ind_first_sample_too_small() {
        let s1 = vec![1.0]; // Only 1 element
        let s2 = vec![2.0, 3.0, 4.0];
        let result = ttest_ind(&s1, &s2, true);
        assert!(result.is_err());
    }

    #[test]
    fn test_ttest_ind_second_sample_too_small() {
        let s1 = vec![1.0, 2.0, 3.0];
        let s2 = vec![4.0]; // Only 1 element
        let result = ttest_ind(&s1, &s2, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_chisquare_too_few_categories() {
        let observed = vec![10.0];
        let expected = vec![10.0];
        let result = chisquare(&observed, &expected);
        assert!(result.is_err());
    }

    #[test]
    fn test_chisquare_negative_expected() {
        let observed = vec![10.0, 20.0, 30.0];
        let expected = vec![15.0, -5.0, 20.0]; // Negative expected
        let result = chisquare(&observed, &expected);
        assert!(result.is_err());
    }

    #[test]
    fn test_f_oneway_zero_groups() {
        let groups: &[Vec<f32>] = &[];
        let result = f_oneway(groups);
        assert!(result.is_err());
    }

    #[test]
    fn test_f_oneway_single_observation_per_group() {
        // Each group has 1 element => df_within = n_total - k = 2 - 2 = 0 => error
        let g1 = vec![1.0];
        let g2 = vec![5.0];
        let result = f_oneway(&[g1, g2]);
        assert!(result.is_err());
    }

    // =========================================================================
    // Additional coverage: distribution helper edge cases
    // =========================================================================

    #[test]
    fn test_t_distribution_large_df_uses_normal_approx() {
        // df > 30 triggers the normal approximation branch
        let large_sample: Vec<f32> = (0..50).map(|i| i as f32 * 0.1).collect();
        let result = ttest_1samp(&large_sample, 0.0).expect("Valid t-test with large n");
        // df = 49, which is > 30, so normal approximation path is taken
        assert!(result.df > 30.0);
        assert!(result.pvalue >= 0.0 && result.pvalue <= 1.0);
    }

    #[test]
    fn test_incomplete_beta_boundary_zero() {
        // x <= 0.0 returns 0.0
        let val = incomplete_beta(1.0, 1.0, 0.0);
        assert!((val - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_incomplete_beta_boundary_one() {
        // x >= 1.0 returns 1.0
        let val = incomplete_beta(1.0, 1.0, 1.0);
        assert!((val - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_incomplete_beta_else_branch() {
        // When x >= (a+1)/(a+b+2) the else branch is taken
        // For a=1, b=1: threshold = 2/4 = 0.5; x=0.8 triggers else branch
        let val = incomplete_beta(1.0, 1.0, 0.8);
        assert!(val > 0.0 && val <= 1.0);
    }

    #[test]
    fn test_incomplete_beta_if_branch() {
        // When x < (a+1)/(a+b+2) the if branch is taken
        // For a=1, b=1: threshold = 2/4 = 0.5; x=0.2 triggers if branch
        let val = incomplete_beta(1.0, 1.0, 0.2);
        assert!(val > 0.0 && val < 1.0);
    }

    #[test]
    fn test_incomplete_gamma_zero_x() {
        let val = incomplete_gamma(1.0, 0.0);
        assert!((val - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_incomplete_gamma_negative_x() {
        let val = incomplete_gamma(1.0, -1.0);
        assert!((val - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_incomplete_gamma_zero_a() {
        let val = incomplete_gamma(0.0, 1.0);
        assert!((val - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_incomplete_gamma_negative_a() {
        let val = incomplete_gamma(-1.0, 1.0);
        assert!((val - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_gamma_reflection_formula() {
        // z < 0.5 triggers the reflection formula branch
        let val = gamma(0.3);
        // Gamma(0.3) is approximately 2.9915...
        assert!(
            val > 2.5 && val < 3.5,
            "Gamma(0.3) should be ~2.99, got {val}"
        );
    }

    #[test]
    fn test_gamma_half() {
        // Gamma(0.5) = sqrt(pi) ~ 1.7724...
        let val = gamma(0.5);
        let sqrt_pi = std::f32::consts::PI.sqrt();
        assert!(
            (val - sqrt_pi).abs() < 0.2,
            "Gamma(0.5) should be ~sqrt(pi)={sqrt_pi}, got {val}"
        );
    }

    #[test]
    fn test_erf_large_positive() {
        // erf(3.0) should be very close to 1.0
        let val = erf(3.0);
        assert!(val > 0.99, "erf(3.0) should be ~1.0, got {val}");
    }

    #[test]
    fn test_erf_large_negative() {
        // erf(-3.0) should be very close to -1.0
        let val = erf(-3.0);
        assert!(val < -0.99, "erf(-3.0) should be ~-1.0, got {val}");
    }

    #[test]
    fn test_normal_cdf_extreme_positive() {
        let val = normal_cdf(5.0);
        assert!(val > 0.999, "Normal CDF at z=5 should be ~1.0");
    }

    #[test]
    fn test_normal_cdf_extreme_negative() {
        let val = normal_cdf(-5.0);
        assert!(val < 0.001, "Normal CDF at z=-5 should be ~0.0");
    }

    #[test]
    fn test_chi_square_pvalue_range() {
        // Ensure chi-square p-value is in [0, 1]
        let pval = chi_square_pvalue(5.0, 3);
        assert!(pval >= 0.0 && pval <= 1.0);
    }

    #[test]
    fn test_f_distribution_pvalue_range() {
        // Ensure f-distribution p-value is in [0, 1]
        let pval = f_distribution_pvalue(3.0, 2, 10);
        assert!(pval >= 0.0 && pval <= 1.0);
    }

    #[test]
    fn test_beta_function_basic() {
        // B(1,1) = Gamma(1)*Gamma(1)/Gamma(2) = 1*1/1 = 1
        let val = beta_function(1.0, 1.0);
        assert!((val - 1.0).abs() < 0.2, "B(1,1) should be ~1.0, got {val}");
    }

    #[test]
    fn test_ttest_1samp_mean_equals_population_mean() {
        // When the sample mean matches population mean, t-stat should be ~0
        let sample = vec![10.0, 10.0, 10.0, 10.0, 10.0];
        let result = ttest_1samp(&sample, 10.0).expect("Valid t-test");
        assert!(
            result.statistic.abs() < 1e-6 || result.statistic.is_nan(),
            "t-stat should be ~0 when means match"
        );
    }

    #[test]
    fn test_ttest_ind_equal_var_identical_samples() {
        let s = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = ttest_ind(&s, &s, true).expect("Valid t-test");
        assert!(
            result.statistic.abs() < 1e-6,
            "t-stat should be 0 for identical samples"
        );
    }

    #[test]
    fn test_ttest_ind_welch_identical_samples() {
        let s = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = ttest_ind(&s, &s, false).expect("Valid Welch's t-test");
        assert!(
            result.statistic.abs() < 1e-6,
            "t-stat should be 0 for identical samples"
        );
    }
}
