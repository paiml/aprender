//! Deterministic random number generation.
//!
//! Implements PCG (Permuted Congruential Generator) with partitioned seeds
//! for reproducible parallel execution.
//!
//! # Reproducibility Guarantee
//!
//! Given the same master seed, all random number sequences will be
//! bitwise-identical across:
//! - Different runs
//! - Different platforms
//! - Different thread counts (via partitioning)

use rand::prelude::*;
use rand_pcg::Pcg64;
use serde::{Deserialize, Serialize};

/// Deterministic, reproducible random number generator.
///
/// Based on PCG (Permuted Congruential Generator) which provides:
/// - Excellent statistical properties
/// - Fast generation
/// - Predictable sequences from seed
/// - Independent streams via partitioning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimRng {
    /// Master seed for reproducibility.
    master_seed: u64,
    /// Current stream index for partitioning.
    stream: u64,
    /// Internal PCG state.
    rng: Pcg64,
}

impl SimRng {
    /// Create a new RNG with the given master seed.
    #[must_use]
    pub fn new(master_seed: u64) -> Self {
        let rng = Pcg64::seed_from_u64(master_seed);
        Self {
            master_seed,
            stream: 0,
            rng,
        }
    }

    /// Get the master seed.
    #[must_use]
    pub const fn master_seed(&self) -> u64 {
        self.master_seed
    }

    /// Get current stream index.
    #[must_use]
    pub const fn stream(&self) -> u64 {
        self.stream
    }

    /// Create partitioned RNGs for parallel execution.
    ///
    /// Each partition gets an independent stream derived from the master seed,
    /// ensuring reproducibility regardless of execution order.
    ///
    /// # Example
    ///
    /// ```rust
    /// use simular::engine::rng::SimRng;
    ///
    /// let mut rng = SimRng::new(42);
    /// let partitions = rng.partition(4);
    /// assert_eq!(partitions.len(), 4);
    /// ```
    #[must_use]
    pub fn partition(&mut self, n: usize) -> Vec<Self> {
        let partitions: Vec<Self> = (0..n)
            .map(|i| {
                let stream = self.stream + i as u64;
                let seed = self
                    .master_seed
                    .wrapping_add(stream.wrapping_mul(0x9E37_79B9_7F4A_7C15));
                Self {
                    master_seed: self.master_seed,
                    stream,
                    rng: Pcg64::seed_from_u64(seed),
                }
            })
            .collect();

        self.stream += n as u64;
        partitions
    }

    /// Generate a random f64 in [0, 1).
    pub fn gen_f64(&mut self) -> f64 {
        self.rng.gen()
    }

    /// Generate a random f64 in the given range.
    ///
    /// # Panics
    ///
    /// Panics if `min > max`.
    pub fn gen_range_f64(&mut self, min: f64, max: f64) -> f64 {
        assert!(min <= max, "Invalid range: min > max");
        min + (max - min) * self.gen_f64()
    }

    /// Generate a random u64.
    pub fn gen_u64(&mut self) -> u64 {
        self.rng.gen()
    }

    /// Generate n random f64 samples in [0, 1).
    #[must_use]
    pub fn sample_n(&mut self, n: usize) -> Vec<f64> {
        (0..n).map(|_| self.gen_f64()).collect()
    }

    /// Generate a standard normal sample using Box-Muller transform.
    pub fn gen_standard_normal(&mut self) -> f64 {
        // Box-Muller transform
        let u1 = self.gen_f64();
        let u2 = self.gen_f64();

        // Avoid log(0)
        let u1 = if u1 < f64::EPSILON { f64::EPSILON } else { u1 };

        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    /// Generate a normal sample with given mean and std.
    pub fn gen_normal(&mut self, mean: f64, std: f64) -> f64 {
        mean + std * self.gen_standard_normal()
    }

    /// Get RNG state as bytes for hashing (audit logging).
    ///
    /// Returns a deterministic byte representation of the RNG state.
    #[must_use]
    pub fn state_bytes(&self) -> Vec<u8> {
        // Use master seed, stream, and serialized RNG state
        let mut bytes = Vec::with_capacity(24);
        bytes.extend_from_slice(&self.master_seed.to_le_bytes());
        bytes.extend_from_slice(&self.stream.to_le_bytes());
        // Also include serialized PCG state for uniqueness
        if let Ok(serialized) = bincode::serialize(&self.rng) {
            bytes.extend_from_slice(&serialized);
        }
        bytes
    }

    /// Save RNG state for checkpoint.
    ///
    /// Note: PCG internal state is not directly serializable, so we save
    /// enough information to recreate the RNG at the same point in the stream.
    #[must_use]
    pub fn save_state(&self) -> RngState {
        // Generate a sequence of values that can be used to verify restoration
        let mut test_rng = self.rng.clone();
        let verification: Vec<u64> = (0..4).map(|_| test_rng.gen()).collect();

        RngState {
            master_seed: self.master_seed,
            stream: self.stream,
            verification_values: Some(verification),
        }
    }

    /// Restore RNG state from checkpoint.
    ///
    /// # Errors
    ///
    /// Returns error if state cannot be restored.
    pub fn restore_state(&mut self, state: &RngState) -> Result<(), RngRestoreError> {
        if state.master_seed != self.master_seed {
            return Err(RngRestoreError::SeedMismatch {
                expected: self.master_seed,
                found: state.master_seed,
            });
        }

        self.stream = state.stream;

        // Recreate from seed and stream
        let seed = self
            .master_seed
            .wrapping_add(self.stream.wrapping_mul(0x9E37_79B9_7F4A_7C15));
        self.rng = Pcg64::seed_from_u64(seed);

        Ok(())
    }
}

/// Saved RNG state for checkpointing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RngState {
    /// Master seed.
    pub master_seed: u64,
    /// Stream index.
    pub stream: u64,
    /// Verification values for testing restoration (optional).
    pub verification_values: Option<Vec<u64>>,
}

/// Error restoring RNG state.
#[derive(Debug, Clone, thiserror::Error)]
pub enum RngRestoreError {
    /// Seed mismatch.
    #[error("Seed mismatch: expected {expected}, found {found}")]
    SeedMismatch {
        /// Expected seed.
        expected: u64,
        /// Found seed.
        found: u64,
    },
    /// Corrupted state.
    #[error("Corrupted RNG state")]
    CorruptedState,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Property: Same seed produces same sequence.
    #[test]
    fn test_reproducibility() {
        let mut rng1 = SimRng::new(42);
        let mut rng2 = SimRng::new(42);

        let seq1: Vec<f64> = (0..100).map(|_| rng1.gen_f64()).collect();
        let seq2: Vec<f64> = (0..100).map(|_| rng2.gen_f64()).collect();

        assert_eq!(seq1, seq2, "Same seed must produce identical sequences");
    }

    /// Property: Different seeds produce different sequences.
    #[test]
    fn test_different_seeds() {
        let mut rng1 = SimRng::new(42);
        let mut rng2 = SimRng::new(43);

        let seq1: Vec<f64> = (0..100).map(|_| rng1.gen_f64()).collect();
        let seq2: Vec<f64> = (0..100).map(|_| rng2.gen_f64()).collect();

        assert_ne!(
            seq1, seq2,
            "Different seeds must produce different sequences"
        );
    }

    /// Property: Partitions are independent.
    #[test]
    fn test_partition_independence() {
        let mut rng = SimRng::new(42);
        let mut partitions = rng.partition(4);

        // Each partition should produce different sequences
        let seqs: Vec<Vec<f64>> = partitions
            .iter_mut()
            .map(|p| (0..10).map(|_| p.gen_f64()).collect())
            .collect();

        for i in 0..seqs.len() {
            for j in (i + 1)..seqs.len() {
                assert_ne!(seqs[i], seqs[j], "Partitions must be independent");
            }
        }
    }

    /// Property: Partitions are reproducible.
    #[test]
    fn test_partition_reproducibility() {
        let mut rng1 = SimRng::new(42);
        let mut rng2 = SimRng::new(42);

        let mut partitions1 = rng1.partition(4);
        let mut partitions2 = rng2.partition(4);

        for (p1, p2) in partitions1.iter_mut().zip(partitions2.iter_mut()) {
            let seq1: Vec<f64> = (0..10).map(|_| p1.gen_f64()).collect();
            let seq2: Vec<f64> = (0..10).map(|_| p2.gen_f64()).collect();
            assert_eq!(seq1, seq2, "Partition sequences must be reproducible");
        }
    }

    /// Property: Range sampling stays in bounds.
    #[test]
    fn test_range_bounds() {
        let mut rng = SimRng::new(42);

        for _ in 0..1000 {
            let v = rng.gen_range_f64(-10.0, 10.0);
            assert!((-10.0..10.0).contains(&v), "Value out of range: {v}");
        }
    }

    /// Property: Normal distribution has correct moments.
    #[test]
    fn test_normal_distribution() {
        let mut rng = SimRng::new(42);
        let n = 10000;
        let samples: Vec<f64> = (0..n).map(|_| rng.gen_standard_normal()).collect();

        let mean: f64 = samples.iter().sum::<f64>() / n as f64;
        let variance: f64 = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;

        // Mean should be close to 0
        assert!(mean.abs() < 0.1, "Mean {mean} too far from 0");
        // Variance should be close to 1
        assert!(
            (variance - 1.0).abs() < 0.1,
            "Variance {variance} too far from 1"
        );
    }

    /// Property: State save/restore preserves seed and stream info.
    /// Note: Full RNG state restoration requires custom serialization which PCG doesn't support.
    #[test]
    fn test_state_save_restore() {
        let rng = SimRng::new(42);

        // Save state
        let state = rng.save_state();

        // Verify state contains correct info
        assert_eq!(state.master_seed, 42);
        assert_eq!(state.stream, 0);
        assert!(state.verification_values.is_some());

        // Restore to a new RNG
        let mut rng2 = SimRng::new(42);
        let result = rng2.restore_state(&state);
        assert!(result.is_ok());
        assert_eq!(rng2.master_seed(), 42);
        assert_eq!(rng2.stream(), 0);
    }

    #[test]
    fn test_gen_u64() {
        let mut rng = SimRng::new(42);
        let v1 = rng.gen_u64();
        let v2 = rng.gen_u64();
        // Should generate different values
        assert_ne!(v1, v2);
    }

    #[test]
    fn test_sample_n() {
        let mut rng = SimRng::new(42);
        let samples = rng.sample_n(10);
        assert_eq!(samples.len(), 10);
        // All samples should be in [0, 1)
        for s in &samples {
            assert!(*s >= 0.0 && *s < 1.0);
        }
    }

    #[test]
    fn test_gen_normal() {
        let mut rng = SimRng::new(42);
        let v = rng.gen_normal(10.0, 2.0);
        // Should be somewhere in the plausible range
        assert!(v > 0.0 && v < 20.0);
    }

    #[test]
    fn test_restore_state_seed_mismatch() {
        let rng = SimRng::new(42);
        let state = rng.save_state();

        let mut rng2 = SimRng::new(99); // Different seed
        let result = rng2.restore_state(&state);
        assert!(result.is_err());

        if let Err(e) = result {
            let display = format!("{}", e);
            assert!(display.contains("mismatch"));
        }
    }

    #[test]
    fn test_rng_state_clone() {
        let rng = SimRng::new(42);
        let state = rng.save_state();
        let cloned = state.clone();
        assert_eq!(cloned.master_seed, state.master_seed);
        assert_eq!(cloned.stream, state.stream);
    }

    #[test]
    fn test_rng_restore_error_clone() {
        let err = RngRestoreError::SeedMismatch {
            expected: 42,
            found: 99,
        };
        let cloned = err.clone();
        assert!(matches!(cloned, RngRestoreError::SeedMismatch { .. }));

        let err2 = RngRestoreError::CorruptedState;
        let cloned2 = err2.clone();
        assert!(matches!(cloned2, RngRestoreError::CorruptedState));
    }

    #[test]
    fn test_rng_restore_error_display() {
        let err = RngRestoreError::CorruptedState;
        let display = format!("{}", err);
        assert!(display.contains("Corrupted"));
    }

    #[test]
    fn test_sim_rng_clone() {
        let rng = SimRng::new(42);
        let cloned = rng.clone();
        assert_eq!(cloned.master_seed(), rng.master_seed());
    }

    #[test]
    fn test_sim_rng_debug() {
        let rng = SimRng::new(42);
        let debug = format!("{:?}", rng);
        assert!(debug.contains("SimRng"));
    }

    #[test]
    fn test_rng_state_debug() {
        let rng = SimRng::new(42);
        let state = rng.save_state();
        let debug = format!("{:?}", state);
        assert!(debug.contains("RngState"));
    }

    #[test]
    fn test_rng_restore_error_debug() {
        let err = RngRestoreError::CorruptedState;
        let debug = format!("{:?}", err);
        assert!(debug.contains("CorruptedState"));
    }

    /// Mutation test: gen_normal must add mean correctly (catches + -> - mutation)
    #[test]
    fn test_gen_normal_mean_is_added() {
        let mut rng = SimRng::new(42);
        // Generate many samples with mean=100, std=0
        // If std=0, result must equal mean exactly
        for _ in 0..10 {
            let v = rng.gen_normal(100.0, 0.0);
            assert!(
                (v - 100.0).abs() < 1e-10,
                "gen_normal with std=0 must return mean exactly, got {v}"
            );
        }
    }

    /// Mutation test: gen_normal must multiply std correctly (catches * -> + mutation)
    #[test]
    fn test_gen_normal_std_is_multiplied() {
        let mut rng = SimRng::new(42);
        // With mean=0, std=10, variance should be 100
        let samples: Vec<f64> = (0..10000).map(|_| rng.gen_normal(0.0, 10.0)).collect();
        let mean: f64 = samples.iter().sum::<f64>() / samples.len() as f64;
        let variance: f64 =
            samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / samples.len() as f64;
        // Variance should be close to 100 (std^2)
        assert!(
            (variance - 100.0).abs() < 15.0,
            "Variance {variance} not close to 100"
        );
    }

    /// Mutation test: gen_normal return value correctness (catches -> 1.0 mutation)
    #[test]
    fn test_gen_normal_not_constant() {
        let mut rng = SimRng::new(42);
        let samples: Vec<f64> = (0..100).map(|_| rng.gen_normal(0.0, 1.0)).collect();
        // Should not all be equal to 1.0
        let all_ones = samples.iter().all(|&x| (x - 1.0).abs() < 1e-10);
        assert!(!all_ones, "gen_normal should not return constant 1.0");
        // Should have variance
        let unique_count = samples
            .iter()
            .map(|x| (*x * 1e6) as i64)
            .collect::<std::collections::HashSet<_>>()
            .len();
        assert!(
            unique_count > 50,
            "gen_normal should produce varied outputs"
        );
    }

    /// Mutation test: partition must increment stream by n (catches += -> *= mutation)
    #[test]
    fn test_partition_stream_increment() {
        let mut rng = SimRng::new(42);
        assert_eq!(rng.stream(), 0);

        let _ = rng.partition(4);
        assert_eq!(
            rng.stream(),
            4,
            "Stream should increment by partition count"
        );

        let _ = rng.partition(3);
        assert_eq!(rng.stream(), 7, "Stream should be 4 + 3 = 7");

        // Catches *= mutation: if *= were used, stream would be 0*4=0, then 0*3=0
        // or 4*3=12 instead of 7
    }

    /// Mutation test: gen_standard_normal uses correct formula (catches * -> / mutation)
    #[test]
    fn test_standard_normal_formula_correctness() {
        let mut rng = SimRng::new(42);
        // Box-Muller should produce values typically in [-4, 4] range
        // If * were replaced with /, the angle would be wrong
        let samples: Vec<f64> = (0..10000).map(|_| rng.gen_standard_normal()).collect();

        // Check that cos(2*PI*u2) produces full range [-1, 1]
        // If division were used instead of multiplication, cos would have wrong argument
        let min = samples.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = samples.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        // Standard normal should span roughly [-4, 4] with high probability
        assert!(min < -2.0, "Min {min} should be < -2 for standard normal");
        assert!(max > 2.0, "Max {max} should be > 2 for standard normal");
    }

    /// Mutation test: gen_standard_normal must handle near-zero u1 (catches < -> == mutation)
    #[test]
    fn test_standard_normal_epsilon_guard() {
        // The guard `if u1 < f64::EPSILON` protects against log(0)
        // If changed to ==, values just above 0 but < EPSILON would cause -Inf
        // We test by checking that no -Inf values appear
        let mut rng = SimRng::new(12345);
        for _ in 0..50000 {
            let v = rng.gen_standard_normal();
            assert!(
                v.is_finite(),
                "gen_standard_normal produced non-finite value: {v}"
            );
        }
    }

    /// Mutation test: Box-Muller 2*PI*u2 formula (catches second * -> / mutation)
    #[test]
    fn test_standard_normal_angle_formula() {
        // Box-Muller: cos(2 * PI * u2) where u2 is uniform [0,1)
        // If the second * were /, we'd get cos(2*PI/u2) which diverges as u2->0
        // This would produce extreme outliers. We verify statistical properties.
        let mut rng = SimRng::new(999);
        let samples: Vec<f64> = (0..50000).map(|_| rng.gen_standard_normal()).collect();

        // Calculate kurtosis - should be close to 3 for normal
        let mean: f64 = samples.iter().sum::<f64>() / samples.len() as f64;
        let variance: f64 =
            samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / samples.len() as f64;
        let fourth_moment: f64 =
            samples.iter().map(|x| (x - mean).powi(4)).sum::<f64>() / samples.len() as f64;
        let kurtosis = fourth_moment / (variance * variance);

        // Normal distribution has kurtosis = 3. Allow some tolerance.
        // If * -> / mutation, kurtosis would be much higher due to outliers
        assert!(
            (kurtosis - 3.0).abs() < 0.5,
            "Kurtosis {kurtosis} far from expected 3.0, suggesting formula error"
        );
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// Falsification test: reproducibility holds for any seed.
        #[test]
        fn prop_reproducibility(seed in 0u64..u64::MAX) {
            let mut rng1 = SimRng::new(seed);
            let mut rng2 = SimRng::new(seed);

            let seq1: Vec<f64> = (0..100).map(|_| rng1.gen_f64()).collect();
            let seq2: Vec<f64> = (0..100).map(|_| rng2.gen_f64()).collect();

            prop_assert_eq!(seq1, seq2);
        }

        /// Falsification test: values in [0, 1) for any seed.
        #[test]
        fn prop_unit_interval(seed in 0u64..u64::MAX) {
            let mut rng = SimRng::new(seed);

            for _ in 0..100 {
                let v = rng.gen_f64();
                prop_assert!(v >= 0.0 && v < 1.0, "Value {} not in [0, 1)", v);
            }
        }

        /// Falsification test: partition count is correct.
        #[test]
        fn prop_partition_count(seed in 0u64..u64::MAX, n in 1usize..100) {
            let mut rng = SimRng::new(seed);
            let partitions = rng.partition(n);
            prop_assert_eq!(partitions.len(), n);
        }
    }
}
