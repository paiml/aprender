//! Variance reduction techniques for Monte Carlo simulations
//!
//! Implements antithetic variates, stratified sampling, and Latin hypercube
//! sampling to improve simulation efficiency.
//!
//! References:
//! - Hammersley & Handscomb (1964), "Monte Carlo Methods"
//! - Lavenberg & Welch (1981), "Control Variables"

use super::rng::MonteCarloRng;

/// Variance reduction technique configuration
#[derive(Debug, Clone, Default)]
pub enum VarianceReduction {
    /// No variance reduction (baseline)
    None,

    /// Antithetic variates for symmetric distributions
    ///
    /// For each uniform U, also use (1-U), creating negatively correlated pairs.
    /// Reduces variance when the function is monotonic.
    #[default]
    Antithetic,

    /// Stratified sampling with K strata
    ///
    /// Partitions `[0,1]` into K equal intervals and samples from each.
    /// Guarantees representation across the entire range.
    Stratified {
        /// Number of strata
        strata: usize,
    },

    /// Latin Hypercube Sampling
    ///
    /// Ensures each dimension is sampled uniformly across its range.
    /// Better coverage than simple random sampling.
    LatinHypercube {
        /// Number of samples
        samples: usize,
    },
}

impl VarianceReduction {
    /// Generate uniforms with variance reduction applied
    #[must_use]
    pub fn generate_uniforms(&self, n: usize, rng: &mut MonteCarloRng) -> Vec<f64> {
        match self {
            Self::None => (0..n).map(|_| rng.uniform()).collect(),

            Self::Antithetic => {
                let mut result = Vec::with_capacity(n);
                let pairs = n / 2;
                for _ in 0..pairs {
                    let u = rng.uniform();
                    result.push(u);
                    result.push(1.0 - u);
                }
                // Handle odd n
                if n % 2 == 1 {
                    result.push(rng.uniform());
                }
                result
            }

            Self::Stratified { strata } => {
                let k = *strata;
                if k == 0 {
                    return Vec::new();
                }

                let samples_per_stratum = n / k;
                let extra = n % k;
                let mut result = Vec::with_capacity(n);

                for i in 0..k {
                    let n_samples = samples_per_stratum + usize::from(i < extra);
                    let low = i as f64 / k as f64;
                    let high = (i + 1) as f64 / k as f64;

                    for _ in 0..n_samples {
                        let u = rng.uniform();
                        result.push(low + u * (high - low));
                    }
                }

                result
            }

            Self::LatinHypercube { samples } => {
                let n_samples = *samples.min(&n);
                if n_samples == 0 {
                    return Vec::new();
                }

                // Generate permutation
                let mut indices: Vec<usize> = (0..n_samples).collect();
                for i in (1..n_samples).rev() {
                    let j = (rng.uniform() * (i + 1) as f64) as usize;
                    indices.swap(i, j);
                }

                // Generate samples
                let mut result = Vec::with_capacity(n_samples);
                for (i, _) in indices.iter().enumerate().take(n_samples) {
                    let u = rng.uniform();
                    result.push((i as f64 + u) / n_samples as f64);
                }

                // If more samples needed, use standard sampling
                for _ in n_samples..n {
                    result.push(rng.uniform());
                }

                result
            }
        }
    }

    /// Generate standard normal samples with variance reduction
    #[must_use]
    pub fn generate_normals(&self, n: usize, rng: &mut MonteCarloRng) -> Vec<f64> {
        match self {
            Self::None => (0..n).map(|_| rng.standard_normal()).collect(),

            Self::Antithetic => {
                let mut result = Vec::with_capacity(n);
                let pairs = n / 2;
                for _ in 0..pairs {
                    let z = rng.standard_normal();
                    result.push(z);
                    result.push(-z);
                }
                // Handle odd n
                if n % 2 == 1 {
                    result.push(rng.standard_normal());
                }
                result
            }

            Self::Stratified { .. } | Self::LatinHypercube { .. } => {
                // Generate stratified uniforms and transform via inverse CDF
                let uniforms = self.generate_uniforms(n, rng);
                uniforms.iter().map(|&u| inverse_normal_cdf(u)).collect()
            }
        }
    }

    /// Estimate variance reduction factor compared to simple sampling
    ///
    /// Returns ratio of variance with technique vs without (< 1 means reduction)
    #[must_use]
    pub fn estimate_variance_ratio(&self) -> f64 {
        match self {
            Self::None => 1.0,
            Self::Antithetic => 0.5, // Theoretical best case for monotonic functions
            Self::Stratified { strata } => {
                // Roughly 1/K improvement for K strata
                1.0 / (*strata as f64).max(1.0)
            }
            Self::LatinHypercube { samples } => {
                // Similar to stratified
                1.0 / (*samples as f64).sqrt().max(1.0)
            }
        }
    }
}

/// Inverse normal CDF (quantile function) approximation
///
/// Uses the Acklam approximation which provides accuracy to ~1.15e-9
#[must_use]
#[allow(clippy::excessive_precision)]
pub fn inverse_normal_cdf(p: f64) -> f64 {
    // Clamp to avoid infinities
    let p = p.clamp(1e-15, 1.0 - 1e-15);

    // Acklam's approximation constants
    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_690e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239,
    ];

    const B: [f64; 5] = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];

    const C: [f64; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
        4.374_664_141_464_968,
        2.938_163_982_698_783,
    ];

    const D: [f64; 4] = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996,
        3.754_408_661_907_416,
    ];

    const P_LOW: f64 = 0.02425;
    const P_HIGH: f64 = 1.0 - P_LOW;

    if p < P_LOW {
        // Lower tail
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= P_HIGH {
        // Central region
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        // Upper tail
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}

/// Calculate empirical variance reduction from paired samples
#[must_use]
pub fn empirical_variance_reduction(standard_samples: &[f64], reduced_samples: &[f64]) -> f64 {
    if standard_samples.is_empty() || reduced_samples.is_empty() {
        return 1.0;
    }

    let var_standard = variance(standard_samples);
    let var_reduced = variance(reduced_samples);

    if var_standard > 0.0 {
        var_reduced / var_standard
    } else {
        1.0
    }
}

fn variance(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert!(matches!(cloned, VarianceReduction::Stratified { strata: 10 }));
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
