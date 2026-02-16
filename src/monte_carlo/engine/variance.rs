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

#[path = "variance_part_02.rs"]
mod variance_part_02;
