//! Random Seed Management for Reproducible Testing
//!
//! This module provides deterministic random number generation for testing.
//! All random operations should use seeds from this module to ensure reproducibility.
//!
//! # Environment Variables
//!
//! - `PRESENTAR_TEST_SEED`: Main test seed (default: 42)
//! - `PROPTEST_SEED`: PropTest seed (default: 0xdeadbeef)
//! - `PRESENTAR_BENCH_SEED`: Benchmark seed (default: 12345)
//!
//! # Example
//!
//! ```rust
//! use presentar_terminal::seed::{get_test_seed, deterministic_rng};
//!
//! let seed = get_test_seed();
//! let mut rng = deterministic_rng();
//! ```

use std::sync::OnceLock;

/// Default test seed for reproducibility
pub const DEFAULT_TEST_SEED: u64 = 42;

/// Default PropTest seed
pub const DEFAULT_PROPTEST_SEED: u64 = 0xdeadbeef;

/// Default benchmark seed
pub const DEFAULT_BENCH_SEED: u64 = 12345;

static TEST_SEED: OnceLock<u64> = OnceLock::new();
static BENCH_SEED: OnceLock<u64> = OnceLock::new();

/// Get the test seed from environment or use default
pub fn get_test_seed() -> u64 {
    *TEST_SEED.get_or_init(|| {
        std::env::var("PRESENTAR_TEST_SEED")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(DEFAULT_TEST_SEED)
    })
}

/// Get the benchmark seed from environment or use default
pub fn get_bench_seed() -> u64 {
    *BENCH_SEED.get_or_init(|| {
        std::env::var("PRESENTAR_BENCH_SEED")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(DEFAULT_BENCH_SEED)
    })
}

/// Simple deterministic PRNG (xorshift64)
/// Used for test data generation without external dependencies
#[derive(Clone, Debug)]
pub struct DeterministicRng {
    state: u64,
}

impl DeterministicRng {
    /// Create new RNG with given seed
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Create RNG from test seed environment variable
    pub fn from_test_seed() -> Self {
        Self::new(get_test_seed())
    }

    /// Create RNG from benchmark seed environment variable
    pub fn from_bench_seed() -> Self {
        Self::new(get_bench_seed())
    }

    /// Generate next random u64
    pub fn next_u64(&mut self) -> u64 {
        // xorshift64 algorithm
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    /// Generate random f64 in [0, 1)
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() as f64) / (u64::MAX as f64)
    }

    /// Generate random f64 in [min, max)
    pub fn next_f64_range(&mut self, min: f64, max: f64) -> f64 {
        min + self.next_f64() * (max - min)
    }

    /// Reset RNG to original seed
    pub fn reset(&mut self, seed: u64) {
        self.state = seed;
    }
}

/// Create a deterministic RNG for testing
pub fn deterministic_rng() -> DeterministicRng {
    DeterministicRng::from_test_seed()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deterministic_rng_reproducible() {
        let mut rng1 = DeterministicRng::new(42);
        let mut rng2 = DeterministicRng::new(42);

        for _ in 0..100 {
            assert_eq!(rng1.next_u64(), rng2.next_u64());
        }
    }

    #[test]
    fn test_f64_range() {
        let mut rng = DeterministicRng::new(42);

        for _ in 0..100 {
            let v = rng.next_f64_range(0.0, 100.0);
            assert!((0.0..100.0).contains(&v));
        }
    }

    #[test]
    fn test_seed_from_env() {
        // Without env var, should use default
        let seed = get_test_seed();
        assert_eq!(seed, DEFAULT_TEST_SEED);
    }

    // Additional tests
    #[test]
    fn test_default_constants() {
        assert_eq!(DEFAULT_TEST_SEED, 42);
        assert_eq!(DEFAULT_PROPTEST_SEED, 0xdeadbeef);
        assert_eq!(DEFAULT_BENCH_SEED, 12345);
    }

    #[test]
    fn test_get_bench_seed() {
        let seed = get_bench_seed();
        assert_eq!(seed, DEFAULT_BENCH_SEED);
    }

    #[test]
    fn test_deterministic_rng_from_test_seed() {
        let mut rng = DeterministicRng::from_test_seed();
        let val1 = rng.next_u64();
        // Reset and check
        rng.reset(get_test_seed());
        let val2 = rng.next_u64();
        assert_eq!(val1, val2);
    }

    #[test]
    fn test_deterministic_rng_from_bench_seed() {
        let mut rng = DeterministicRng::from_bench_seed();
        let _val = rng.next_u64();
        // Just verify it doesn't panic
    }

    #[test]
    fn test_next_f64() {
        let mut rng = DeterministicRng::new(12345);
        for _ in 0..100 {
            let val = rng.next_f64();
            assert!((0.0..1.0).contains(&val));
        }
    }

    #[test]
    fn test_next_f64_range_negative() {
        let mut rng = DeterministicRng::new(42);
        for _ in 0..100 {
            let val = rng.next_f64_range(-50.0, 50.0);
            assert!((-50.0..50.0).contains(&val));
        }
    }

    #[test]
    fn test_reset() {
        let mut rng = DeterministicRng::new(42);
        let val1 = rng.next_u64();
        let val2 = rng.next_u64();

        rng.reset(42);
        let val3 = rng.next_u64();
        let val4 = rng.next_u64();

        assert_eq!(val1, val3);
        assert_eq!(val2, val4);
    }

    #[test]
    fn test_deterministic_rng_debug() {
        let rng = DeterministicRng::new(42);
        let debug = format!("{:?}", rng);
        assert!(debug.contains("DeterministicRng"));
    }

    #[test]
    fn test_deterministic_rng_clone() {
        let mut rng1 = DeterministicRng::new(42);
        let _ = rng1.next_u64();
        let mut rng2 = rng1.clone();

        // Both should produce same value now
        assert_eq!(rng1.next_u64(), rng2.next_u64());
    }

    #[test]
    fn test_deterministic_rng_different_seeds() {
        let mut rng1 = DeterministicRng::new(1);
        let mut rng2 = DeterministicRng::new(2);

        // Different seeds should produce different values
        assert_ne!(rng1.next_u64(), rng2.next_u64());
    }

    #[test]
    fn test_deterministic_rng_function() {
        let mut rng = deterministic_rng();
        let _val = rng.next_u64();
        // Verify it works
    }

    #[test]
    fn test_next_u64_not_zero() {
        let mut rng = DeterministicRng::new(42);
        // After a few iterations, shouldn't be stuck at 0
        for _ in 0..10 {
            rng.next_u64();
        }
        assert_ne!(rng.next_u64(), 0);
    }

    #[test]
    fn test_deterministic_rng_zero_seed() {
        // Edge case: zero seed
        let mut rng = DeterministicRng::new(0);
        // xorshift with 0 stays 0, but that's expected behavior
        let val = rng.next_u64();
        assert_eq!(val, 0);
    }

    #[test]
    fn test_deterministic_rng_max_seed() {
        // Edge case: max seed
        let mut rng = DeterministicRng::new(u64::MAX);
        let _val = rng.next_u64();
        // Should not panic
    }
}
