#[cfg(test)]
mod tests {
    #[allow(clippy::wildcard_imports)]
    use super::super::*;

    #[test]
    fn test_no_variance_reduction() {
        let mut rng = MonteCarloRng::new(42);
        let vr = VarianceReduction::None;
        let uniforms = vr.generate_uniforms(100, &mut rng);

        assert_eq!(uniforms.len(), 100);
        for &u in &uniforms {
            assert!(u >= 0.0 && u < 1.0);
        }
    }

    #[test]
    fn test_antithetic_uniforms() {
        let mut rng = MonteCarloRng::new(42);
        let vr = VarianceReduction::Antithetic;
        let uniforms = vr.generate_uniforms(100, &mut rng);

        assert_eq!(uniforms.len(), 100);

        // Check antithetic pairs
        for i in 0..50 {
            let u1 = uniforms[i * 2];
            let u2 = uniforms[i * 2 + 1];
            assert!(
                (u1 + u2 - 1.0).abs() < 1e-10,
                "Antithetic pair should sum to 1: {u1} + {u2}"
            );
        }
    }

    #[test]
    fn test_antithetic_normals() {
        let mut rng = MonteCarloRng::new(42);
        let vr = VarianceReduction::Antithetic;
        let normals = vr.generate_normals(100, &mut rng);

        assert_eq!(normals.len(), 100);

        // Check antithetic pairs sum to 0
        for i in 0..50 {
            let z1 = normals[i * 2];
            let z2 = normals[i * 2 + 1];
            assert!(
                (z1 + z2).abs() < 1e-10,
                "Antithetic normal pair should sum to 0: {z1} + {z2}"
            );
        }
    }

    #[test]
    fn test_stratified_coverage() {
        let mut rng = MonteCarloRng::new(42);
        let vr = VarianceReduction::Stratified { strata: 10 };
        let uniforms = vr.generate_uniforms(1000, &mut rng);

        assert_eq!(uniforms.len(), 1000);

        // Count samples in each stratum
        let mut counts = [0; 10];
        for &u in &uniforms {
            let stratum = (u * 10.0).floor() as usize;
            if stratum < 10 {
                counts[stratum] += 1;
            }
        }

        // Each stratum should have ~100 samples
        for (i, &count) in counts.iter().enumerate() {
            assert!(
                count >= 90 && count <= 110,
                "Stratum {i} has {count} samples, expected ~100"
            );
        }
    }

    #[test]
    fn test_latin_hypercube() {
        let mut rng = MonteCarloRng::new(42);
        let vr = VarianceReduction::LatinHypercube { samples: 100 };
        let uniforms = vr.generate_uniforms(100, &mut rng);

        assert_eq!(uniforms.len(), 100);

        // Check all values are in [0, 1)
        for &u in &uniforms {
            assert!(u >= 0.0 && u < 1.0);
        }
    }

    #[test]
    fn test_inverse_normal_cdf() {
        // Test known values
        assert!((inverse_normal_cdf(0.5) - 0.0).abs() < 1e-6);
        assert!((inverse_normal_cdf(0.84134) - 1.0).abs() < 1e-3);
        assert!((inverse_normal_cdf(0.15866) - (-1.0)).abs() < 1e-3);
        assert!((inverse_normal_cdf(0.97725) - 2.0).abs() < 1e-3);
    }

    #[test]
    fn test_antithetic_variance_reduction() {
        let n = 100_000;
        let mut rng = MonteCarloRng::new(42);

        // Estimate E[X^2] where X ~ U(0,1) using standard sampling
        let standard: Vec<f64> = (0..n)
            .map(|_| {
                let u = rng.uniform();
                u * u
            })
            .collect();

        // Same estimate using antithetic variates
        let mut rng = MonteCarloRng::new(42);
        let antithetic: Vec<f64> = (0..n / 2)
            .flat_map(|_| {
                let u = rng.uniform();
                let v = 1.0 - u;
                vec![u * u, v * v]
            })
            .collect();

        // Calculate variances
        let var_std = variance(&standard);
        let var_anti = variance(&antithetic);

        // Antithetic should have lower variance
        let reduction = var_anti / var_std;
        assert!(
            reduction < 1.0,
            "Antithetic should reduce variance: {reduction}"
        );
    }

    #[test]
    fn test_stratified_variance_reduction() {
        // Use different seeds to avoid correlation between samples
        let n = 50_000;
        let mut rng_std = MonteCarloRng::new(42);
        let mut rng_strat = MonteCarloRng::new(123);

        // Standard sampling
        let standard: Vec<f64> = (0..n).map(|_| rng_std.uniform().powi(2)).collect();

        // Stratified sampling with many strata
        let vr = VarianceReduction::Stratified { strata: 500 };
        let uniforms = vr.generate_uniforms(n, &mut rng_strat);
        let stratified: Vec<f64> = uniforms.iter().map(|&u| u.powi(2)).collect();

        let var_std = variance(&standard);
        let var_strat = variance(&stratified);

        // With proper stratification, variance should be similar or lower
        // The effect may be small for U² since it's a smooth function
        let reduction = var_strat / var_std;
        assert!(
            reduction < 1.1,
            "Stratified should not significantly increase variance: {reduction}"
        );
    }

    #[test]
    fn test_variance_ratio_estimates() {
        assert!((VarianceReduction::None.estimate_variance_ratio() - 1.0).abs() < 1e-10);
        assert!(VarianceReduction::Antithetic.estimate_variance_ratio() < 1.0);
        assert!(VarianceReduction::Stratified { strata: 10 }.estimate_variance_ratio() < 1.0);
    }

    #[test]
    fn test_odd_sample_sizes() {
        let mut rng = MonteCarloRng::new(42);

        // Odd number with antithetic
        let vr = VarianceReduction::Antithetic;
        let uniforms = vr.generate_uniforms(101, &mut rng);
        assert_eq!(uniforms.len(), 101);

        // Odd number with stratified
        let mut rng = MonteCarloRng::new(42);
        let vr = VarianceReduction::Stratified { strata: 10 };
        let uniforms = vr.generate_uniforms(101, &mut rng);
        assert_eq!(uniforms.len(), 101);
    }

    #[test]
    fn test_empirical_variance_reduction() {
        let standard = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let reduced = vec![2.8, 3.0, 3.2, 3.0, 3.0]; // Lower variance

        let ratio = empirical_variance_reduction(&standard, &reduced);
        assert!(ratio < 1.0);
    }

    // =========================================================================
    // Additional coverage tests
    // =========================================================================

    #[test]
    fn test_variance_reduction_default() {
        let vr = VarianceReduction::default();
        assert!(matches!(vr, VarianceReduction::Antithetic));
    }

    #[test]
    fn test_stratified_zero_strata() {
        let mut rng = MonteCarloRng::new(42);
        let vr = VarianceReduction::Stratified { strata: 0 };
        let uniforms = vr.generate_uniforms(100, &mut rng);
        assert!(uniforms.is_empty());
    }

    #[test]
    fn test_latin_hypercube_zero_samples() {
        let mut rng = MonteCarloRng::new(42);
        let vr = VarianceReduction::LatinHypercube { samples: 0 };
        let uniforms = vr.generate_uniforms(100, &mut rng);
        assert!(uniforms.is_empty());
    }

    #[test]
    fn test_latin_hypercube_more_requested_than_samples() {
        let mut rng = MonteCarloRng::new(42);
        let vr = VarianceReduction::LatinHypercube { samples: 50 };
        let uniforms = vr.generate_uniforms(100, &mut rng); // Request 100, LHC has 50
        assert_eq!(uniforms.len(), 100); // Should still return 100 (50 LHC + 50 random)
    }

    #[test]
    fn test_generate_normals_stratified() {
        let mut rng = MonteCarloRng::new(42);
        let vr = VarianceReduction::Stratified { strata: 10 };
        let normals = vr.generate_normals(100, &mut rng);
        assert_eq!(normals.len(), 100);
        for &z in &normals {
            assert!(z.is_finite());
        }
    }

    #[test]
    fn test_generate_normals_latin_hypercube() {
        let mut rng = MonteCarloRng::new(42);
        let vr = VarianceReduction::LatinHypercube { samples: 100 };
        let normals = vr.generate_normals(100, &mut rng);
        assert_eq!(normals.len(), 100);
        for &z in &normals {
            assert!(z.is_finite());
        }
    }

    #[test]
    fn test_generate_normals_none() {
        let mut rng = MonteCarloRng::new(42);
        let vr = VarianceReduction::None;
        let normals = vr.generate_normals(100, &mut rng);
        assert_eq!(normals.len(), 100);
        for &z in &normals {
            assert!(z.is_finite());
        }
    }

    #[test]
    fn test_antithetic_normals_odd_count() {
        let mut rng = MonteCarloRng::new(42);
        let vr = VarianceReduction::Antithetic;
        let normals = vr.generate_normals(101, &mut rng); // Odd count
        assert_eq!(normals.len(), 101);
    }

    #[test]
    fn test_antithetic_uniforms_odd_count() {
        let mut rng = MonteCarloRng::new(42);
        let vr = VarianceReduction::Antithetic;
        let uniforms = vr.generate_uniforms(101, &mut rng); // Odd count
        assert_eq!(uniforms.len(), 101);
    }

    #[test]
    fn test_empirical_variance_reduction_empty() {
        let empty: Vec<f64> = vec![];
        let ratio = empirical_variance_reduction(&empty, &[1.0, 2.0]);
        assert!((ratio - 1.0).abs() < 1e-10);

        let ratio = empirical_variance_reduction(&[1.0, 2.0], &empty);
        assert!((ratio - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_empirical_variance_reduction_zero_variance() {
        let constant = vec![5.0, 5.0, 5.0, 5.0];
        let varying = vec![1.0, 2.0, 3.0, 4.0];
        let ratio = empirical_variance_reduction(&constant, &varying);
        assert!((ratio - 1.0).abs() < 1e-10); // Zero variance in standard returns 1.0
    }

    #[test]
    fn test_empirical_variance_reduction_single_value() {
        let single = vec![5.0];
        let varying = vec![1.0, 2.0, 3.0];
        let ratio = empirical_variance_reduction(&single, &varying);
        // Single value has 0 variance
        assert!((ratio - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_variance_ratio_latin_hypercube() {
        let vr = VarianceReduction::LatinHypercube { samples: 100 };
        let ratio = vr.estimate_variance_ratio();
        assert!(ratio < 1.0);
        assert!(ratio > 0.0);
    }

    #[test]
    fn test_variance_reduction_debug() {
        let vr = VarianceReduction::Antithetic;
        let debug_str = format!("{:?}", vr);
        assert!(debug_str.contains("Antithetic"));

        let vr2 = VarianceReduction::Stratified { strata: 10 };
        let debug_str2 = format!("{:?}", vr2);
        assert!(debug_str2.contains("Stratified"));

        let vr3 = VarianceReduction::LatinHypercube { samples: 100 };
        let debug_str3 = format!("{:?}", vr3);
        assert!(debug_str3.contains("LatinHypercube"));
    }

    #[test]
    fn test_variance_reduction_clone() {
        let vr = VarianceReduction::Stratified { strata: 10 };
        let cloned = vr.clone();
        assert!(matches!(
            cloned,
            VarianceReduction::Stratified { strata: 10 }
        ));
    }

    #[test]
    fn test_inverse_normal_cdf_tails() {
        // Test lower tail (p < 0.02425)
        let z_low = inverse_normal_cdf(0.01);
        assert!(z_low < -2.0);

        // Test upper tail (p > 0.97575)
        let z_high = inverse_normal_cdf(0.99);
        assert!(z_high > 2.0);

        // Test extreme values - should clamp
        let z_extreme_low = inverse_normal_cdf(1e-20);
        assert!(z_extreme_low.is_finite());

        let z_extreme_high = inverse_normal_cdf(1.0 - 1e-20);
        assert!(z_extreme_high.is_finite());
    }

    // Property-based tests
    #[cfg(test)]
    mod proptests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn prop_uniforms_in_range(seed: u64, n in 10..500usize, strata in 2..20usize) {
                // Test all variance reduction methods
                for vr in [
                    VarianceReduction::None,
                    VarianceReduction::Antithetic,
                    VarianceReduction::Stratified { strata },
                    VarianceReduction::LatinHypercube { samples: n },
                ] {
                    let uniforms = vr.generate_uniforms(n, &mut MonteCarloRng::new(seed));
                    for &u in &uniforms {
                        prop_assert!(u >= 0.0 && u <= 1.0, "Uniform out of range: {u}");
                    }
                }
            }

            #[test]
            fn prop_normals_finite(seed: u64, n in 10..500usize) {
                for vr in [
                    VarianceReduction::None,
                    VarianceReduction::Antithetic,
                ] {
                    let normals = vr.generate_normals(n, &mut MonteCarloRng::new(seed));
                    for &z in &normals {
                        prop_assert!(z.is_finite(), "Normal not finite: {z}");
                    }
                }
            }
        }
    }
}
