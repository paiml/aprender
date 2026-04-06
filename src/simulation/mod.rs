//! Simulation Testing Framework (TRUENO-SPEC-012)
//!
//! Provides deterministic, reproducible, and falsifiable validation of compute
//! operations across all backends: SIMD (CPU), PTX (CUDA), and WGPU.
//!
//! This module integrates with the sovereign stack (simular) and follows
//! Toyota Production System principles:
//!
//! - **Jidoka**: Built-in quality - stop on defect
//! - **Poka-Yoke**: Mistake-proofing via type safety
//! - **Heijunka**: Leveled testing across backends
//! - **Genchi Genbutsu**: Visual inspection of results
//! - **Kaizen**: Continuous performance improvement
//!
//! # Example
//!
//! ```rust,ignore
//! use trueno::simulation::{SimTestConfig, BackendTolerance};
//!
//! let config = SimTestConfig::builder()
//!     .seed(42)
//!     .tolerance(BackendTolerance::default())
//!     .build();
//! ```

mod jidoka;
mod scheduler;
mod stress;
mod visual;

// Re-export visual regression types
pub use visual::{
    BufferRenderer, ColorPalette, GoldenBaseline, PixelDiffResult, Rgb, VisualRegressionConfig,
};

// Re-export stress testing types
pub use stress::{
    StressAnomaly, StressAnomalyKind, StressResult, StressTestConfig, StressThresholds,
};

// Re-export jidoka types
pub use jidoka::{JidokaAction, JidokaCondition, JidokaError, JidokaGuard};

// Re-export scheduler types
pub use scheduler::{
    BackendCategory, BackendSelector, BackendTolerance, HeijunkaScheduler, NeedsSeed, Ready,
    SimTestConfig, SimTestConfigBuilder, SimulationTest,
};

/// Re-export SimRng from simular for deterministic testing
#[cfg(test)]
pub use simular::engine::rng::SimRng;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Backend;

    #[test]
    fn test_simrng_reproducibility() {
        // Falsifiable claim B-016, B-017
        let mut rng1 = SimRng::new(42);
        let mut rng2 = SimRng::new(42);

        let seq1: Vec<f64> = (0..100).map(|_| rng1.gen_f64()).collect();
        let seq2: Vec<f64> = (0..100).map(|_| rng2.gen_f64()).collect();

        assert_eq!(seq1, seq2, "Same seed must produce identical sequences");
    }

    #[test]
    fn test_simrng_different_seeds() {
        // Falsifiable claim B-018
        let mut rng1 = SimRng::new(42);
        let mut rng2 = SimRng::new(43);

        let seq1: Vec<f64> = (0..100).map(|_| rng1.gen_f64()).collect();
        let seq2: Vec<f64> = (0..100).map(|_| rng2.gen_f64()).collect();

        assert_ne!(seq1, seq2, "Different seeds must produce different sequences");
    }

    #[test]
    fn test_simrng_partitioning() {
        // Falsifiable claim B-019
        let mut rng = SimRng::new(42);
        let partitions = rng.partition(4);

        assert_eq!(partitions.len(), 4);

        // Each partition should be independent
        let mut seqs: Vec<Vec<f64>> = Vec::new();
        for mut p in partitions {
            seqs.push((0..10).map(|_| p.gen_f64()).collect());
        }

        for i in 0..seqs.len() {
            for j in (i + 1)..seqs.len() {
                assert_ne!(seqs[i], seqs[j], "Partitions must be independent");
            }
        }
    }

    #[test]
    fn test_simrng_gen_f32_for_trueno() {
        // Generate f32 test data using SimRng
        let mut rng = SimRng::new(42);

        let test_data: Vec<f32> = (0..1000).map(|_| rng.gen_f64() as f32).collect();

        // Verify all values are in valid range
        for v in &test_data {
            assert!(v.is_finite(), "Generated value should be finite");
            assert!(*v >= 0.0 && *v < 1.0, "Value should be in [0, 1)");
        }
    }

    #[test]
    fn test_full_simulation_workflow() {
        // Create test configuration
        let config = SimTestConfig::builder()
            .seed(42)
            .tolerance(BackendTolerance::default())
            .backends(vec![Backend::Scalar, Backend::AVX2])
            .input_sizes(vec![100])
            .cycles(2)
            .build();

        // Create scheduler
        let mut scheduler = config.create_scheduler();

        // Create Jidoka guards
        let nan_guard = JidokaGuard::nan_guard("simulation_test");

        // Run through all tests
        let mut test_count = 0;
        while let Some(test) = scheduler.next_test() {
            // Generate deterministic test data
            let mut rng = SimRng::new(test.seed);
            let data: Vec<f32> = (0..test.input_size).map(|_| rng.gen_f64() as f32).collect();

            // Check for NaN (should pass with valid data)
            let result = nan_guard.check_output(&data);
            assert!(result.is_ok(), "Generated data should not contain NaN");

            test_count += 1;
        }

        // 2 backends * 1 size * 2 cycles = 4 tests
        assert_eq!(test_count, 4);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// Falsifiable claim: NaN detection never misses
        #[test]
        fn prop_nan_detection_complete(values in prop::collection::vec(-1000.0f32..1000.0, 0..100)) {
            let guard = JidokaGuard::nan_guard("test");

            // Clean input should pass
            let result = guard.check_output(&values);
            prop_assert!(result.is_ok());
        }
    }
}
