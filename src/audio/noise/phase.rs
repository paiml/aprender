//! Phase generation for noise synthesis

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use std::f32::consts::PI;

/// Random phase generator for spectral synthesis
#[derive(Debug)]
pub struct PhaseGenerator {
    rng: SmallRng,
    coherence: f32,
    previous_phases: Vec<f32>,
}

impl PhaseGenerator {
    /// Create a new phase generator with the given seed
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self {
            rng: SmallRng::seed_from_u64(seed),
            coherence: 0.0,
            previous_phases: Vec::new(),
        }
    }

    /// Create with coherence (0.0 = fully random, 1.0 = fully correlated with previous)
    #[must_use]
    pub fn with_coherence(mut self, coherence: f32) -> Self {
        self.coherence = coherence.clamp(0.0, 1.0);
        self
    }

    /// Set coherence value
    pub fn set_coherence(&mut self, coherence: f32) {
        self.coherence = coherence.clamp(0.0, 1.0);
    }

    /// Generate random phases in the range [-PI, PI]
    pub fn generate(&mut self, n_freqs: usize) -> Vec<f32> {
        let mut phases = Vec::with_capacity(n_freqs);

        for i in 0..n_freqs {
            // Generate new random phase
            let random_phase = self.rng.gen_range(-PI..PI);

            let phase = if self.coherence > 0.0 && i < self.previous_phases.len() {
                // Blend with previous phase for coherence
                let prev = self.previous_phases[i];
                let blended = prev * self.coherence + random_phase * (1.0 - self.coherence);
                // Wrap to [-PI, PI]
                wrap_phase(blended)
            } else {
                random_phase
            };

            phases.push(phase);
        }

        // Store for next frame coherence
        self.previous_phases = phases.clone();

        phases
    }

    /// Reset the generator state (for deterministic testing)
    pub fn reset(&mut self, seed: u64) {
        self.rng = SmallRng::seed_from_u64(seed);
        self.previous_phases.clear();
    }
}

/// Wrap phase to [-PI, PI] range
#[inline]
fn wrap_phase(phase: f32) -> f32 {
    let mut p = phase;
    while p > PI {
        p -= 2.0 * PI;
    }
    while p < -PI {
        p += 2.0 * PI;
    }
    p
}

impl Default for PhaseGenerator {
    fn default() -> Self {
        Self::new(42)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========== NG8: Phase values bounded to [-PI, PI] ==========

    #[test]
    fn test_ng8_phase_bounds_basic() {
        let mut gen = PhaseGenerator::new(42);
        let phases = gen.generate(100);

        for &phase in &phases {
            assert!(
                phase >= -PI && phase <= PI,
                "Phase {} out of bounds [-PI, PI]",
                phase
            );
        }
    }

    #[test]
    fn test_ng8_phase_bounds_many_iterations() {
        let mut gen = PhaseGenerator::new(12345);

        for _ in 0..100 {
            let phases = gen.generate(512);
            for &phase in &phases {
                assert!(phase >= -PI && phase <= PI, "Phase {} out of bounds", phase);
            }
        }
    }

    #[test]
    fn test_ng8_phase_bounds_with_coherence() {
        let mut gen = PhaseGenerator::new(42).with_coherence(0.8);

        for _ in 0..50 {
            let phases = gen.generate(256);
            for &phase in &phases {
                assert!(
                    phase >= -PI && phase <= PI,
                    "Phase {} out of bounds with coherence",
                    phase
                );
            }
        }
    }

    #[test]
    fn test_ng8_phase_bounds_small_buffer() {
        let mut gen = PhaseGenerator::new(42);
        let phases = gen.generate(1);
        assert_eq!(phases.len(), 1);
        assert!(phases[0] >= -PI && phases[0] <= PI);
    }

    #[test]
    fn test_ng8_phase_bounds_large_buffer() {
        let mut gen = PhaseGenerator::new(42);
        let phases = gen.generate(4096);
        assert_eq!(phases.len(), 4096);
        for &phase in &phases {
            assert!(phase >= -PI && phase <= PI);
        }
    }

    // ========== NG9: Deterministic with fixed seed ==========

    #[test]
    fn test_ng9_deterministic_same_seed() {
        let mut gen1 = PhaseGenerator::new(42);
        let mut gen2 = PhaseGenerator::new(42);

        let phases1 = gen1.generate(100);
        let phases2 = gen2.generate(100);

        assert_eq!(phases1, phases2, "Same seed should produce same phases");
    }

    #[test]
    fn test_ng9_deterministic_after_reset() {
        let mut gen = PhaseGenerator::new(42);

        let phases1 = gen.generate(100);
        gen.reset(42);
        let phases2 = gen.generate(100);

        assert_eq!(phases1, phases2, "Reset should restore determinism");
    }

    #[test]
    fn test_ng9_deterministic_multiple_calls() {
        let mut gen1 = PhaseGenerator::new(123);
        let mut gen2 = PhaseGenerator::new(123);

        // Multiple calls should still match
        let p1a = gen1.generate(50);
        let p1b = gen1.generate(50);

        let p2a = gen2.generate(50);
        let p2b = gen2.generate(50);

        assert_eq!(p1a, p2a);
        assert_eq!(p1b, p2b);
    }

    #[test]
    fn test_ng9_different_seed_different_output() {
        let mut gen1 = PhaseGenerator::new(42);
        let mut gen2 = PhaseGenerator::new(43);

        let phases1 = gen1.generate(100);
        let phases2 = gen2.generate(100);

        assert_ne!(
            phases1, phases2,
            "Different seeds should produce different phases"
        );
    }

    // ========== NG10: Coherence=0 produces uniform distribution ==========

    #[test]
    fn test_ng10_zero_coherence_distribution() {
        let mut gen = PhaseGenerator::new(42).with_coherence(0.0);

        // Generate many samples
        let mut all_phases = Vec::new();
        for _ in 0..100 {
            all_phases.extend(gen.generate(100));
        }

        // Check distribution - should have reasonable spread across [-PI, PI]
        let n = all_phases.len() as f32;

        // Count phases in quadrants
        let q1 = all_phases
            .iter()
            .filter(|&&p| p >= 0.0 && p < PI / 2.0)
            .count();
        let q2 = all_phases
            .iter()
            .filter(|&&p| p >= PI / 2.0 && p <= PI)
            .count();
        let q3 = all_phases
            .iter()
            .filter(|&&p| p >= -PI && p < -PI / 2.0)
            .count();
        let q4 = all_phases
            .iter()
            .filter(|&&p| p >= -PI / 2.0 && p < 0.0)
            .count();

        // Each quadrant should have roughly 25% (with some tolerance)
        let expected = n / 4.0;
        let tolerance = expected * 0.3; // 30% tolerance

        assert!(
            (q1 as f32 - expected).abs() < tolerance,
            "Q1 distribution off: {} vs expected {}",
            q1,
            expected
        );
        assert!(
            (q2 as f32 - expected).abs() < tolerance,
            "Q2 distribution off: {} vs expected {}",
            q2,
            expected
        );
        assert!(
            (q3 as f32 - expected).abs() < tolerance,
            "Q3 distribution off: {} vs expected {}",
            q3,
            expected
        );
        assert!(
            (q4 as f32 - expected).abs() < tolerance,
            "Q4 distribution off: {} vs expected {}",
            q4,
            expected
        );
    }

    #[test]
    fn test_ng10_zero_coherence_mean_near_zero() {
        let mut gen = PhaseGenerator::new(42).with_coherence(0.0);

        let mut sum = 0.0;
        let mut count = 0;

        for _ in 0..100 {
            for phase in gen.generate(100) {
                sum += phase;
                count += 1;
            }
        }

        let mean = sum / count as f32;

        // Mean should be close to zero for uniform distribution over [-PI, PI]
        assert!(
            mean.abs() < 0.2,
            "Mean {} should be close to 0 for uniform distribution",
            mean
        );
    }

    #[test]
    fn test_ng10_full_coherence_stable() {
        let mut gen = PhaseGenerator::new(42).with_coherence(1.0);

        let phases1 = gen.generate(100);
        let phases2 = gen.generate(100);

        // With full coherence, phases should be very similar to previous
        let mut diff_sum = 0.0;
        for (a, b) in phases1.iter().zip(phases2.iter()) {
            diff_sum += (a - b).abs();
        }
        let avg_diff = diff_sum / 100.0;

        assert!(
            avg_diff < PI / 2.0,
            "Full coherence should have similar phases, avg_diff = {}",
            avg_diff
        );
    }

    #[test]
    fn test_ng10_partial_coherence_intermediate() {
        let mut gen = PhaseGenerator::new(42).with_coherence(0.5);

        let phases1 = gen.generate(100);
        let phases2 = gen.generate(100);

        // With partial coherence, phases should be somewhat correlated
        // but not identical (unless by chance)
        let mut diff_sum = 0.0;
        for (a, b) in phases1.iter().zip(phases2.iter()) {
            diff_sum += (a - b).abs();
        }

        // Just verify the phases are still bounded
        for &phase in &phases2 {
            assert!(phase >= -PI && phase <= PI);
        }

        // Some difference should exist
        assert!(diff_sum > 0.0);
    }

    // ========== Additional phase tests ==========

    #[test]
    fn test_wrap_phase_positive() {
        assert!((wrap_phase(PI + 1.0) - (1.0 - PI)).abs() < 0.001);
    }

    #[test]
    fn test_wrap_phase_negative() {
        assert!((wrap_phase(-PI - 1.0) - (PI - 1.0)).abs() < 0.001);
    }

    #[test]
    fn test_wrap_phase_in_range() {
        assert!((wrap_phase(0.5) - 0.5).abs() < f32::EPSILON);
        assert!((wrap_phase(-0.5) - (-0.5)).abs() < f32::EPSILON);
    }

    #[test]
    fn test_phase_generator_default() {
        let gen = PhaseGenerator::default();
        assert!((gen.coherence - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_set_coherence() {
        let mut gen = PhaseGenerator::new(42);
        gen.set_coherence(0.7);
        assert!((gen.coherence - 0.7).abs() < 0.001);
    }

    #[test]
    fn test_set_coherence_clamped() {
        let mut gen = PhaseGenerator::new(42);
        gen.set_coherence(2.0);
        assert!((gen.coherence - 1.0).abs() < f32::EPSILON);
        gen.set_coherence(-1.0);
        assert!((gen.coherence - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_generate_empty() {
        let mut gen = PhaseGenerator::new(42);
        let phases = gen.generate(0);
        assert!(phases.is_empty());
    }
}
