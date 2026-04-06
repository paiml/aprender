//! Thread configuration for inference operations
//!
//! Provides dynamic thread allocation following llama.cpp patterns:
//! - Prefill/batch: Use all cores for maximum throughput
//! - Decode: Use fewer cores to reduce cache thrashing
//!
//! ## Performance Notes
//!
//! Profiling shows optimal thread counts for memory-bound quantized matmuls:
//! - 48 threads: 11.9 tok/s (too much sync overhead)
//! - 24 threads: 18.7 tok/s
//! - 16 threads: 25.3 tok/s (optimal)
//! - 12 threads: 25.0 tok/s
//! - 8 threads:  21.9 tok/s

use crate::error::{RealizarError, Result};

/// Thread configuration for dynamic thread allocation
///
/// Per llama.cpp: batch processing uses more threads than single-token decode.
/// This reduces cache thrashing during decode phase.
///
/// # Example
///
/// ```
/// use realizar::inference::ThreadConfig;
///
/// let config = ThreadConfig::auto();
/// assert!(config.n_threads_batch >= 1);
/// assert!(config.n_threads_decode >= 1);
///
/// // Batch uses more threads than decode
/// assert!(config.n_threads_batch >= config.n_threads_decode);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ThreadConfig {
    /// Threads for batch/prefill operations (uses all cores)
    pub n_threads_batch: usize,
    /// Threads for single-token decode (uses fewer cores)
    pub n_threads_decode: usize,
}

impl ThreadConfig {
    /// Create optimal thread config based on available cores
    ///
    /// - Batch: Uses all available cores
    /// - Decode: Uses half cores (min 1) to reduce cache contention
    #[must_use]
    pub fn auto() -> Self {
        let num_cpus = rayon::current_num_threads();
        Self {
            n_threads_batch: num_cpus,
            n_threads_decode: (num_cpus / 2).max(1),
        }
    }

    /// Create with explicit thread counts
    ///
    /// Both values are clamped to minimum of 1.
    #[must_use]
    pub fn new(n_threads_batch: usize, n_threads_decode: usize) -> Self {
        Self {
            n_threads_batch: n_threads_batch.max(1),
            n_threads_decode: n_threads_decode.max(1),
        }
    }

    /// Get the number of threads for the current operation mode
    ///
    /// Returns `n_threads_batch` for prefill, `n_threads_decode` otherwise.
    #[must_use]
    pub fn threads_for(&self, is_prefill: bool) -> usize {
        if is_prefill {
            self.n_threads_batch
        } else {
            self.n_threads_decode
        }
    }
}

impl Default for ThreadConfig {
    fn default() -> Self {
        Self::auto()
    }
}

/// Execution mode for controlling parallelism
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InferenceMode {
    /// Prefill/prompt processing - use maximum threads
    Prefill,
    /// Single-token decode - use fewer threads to reduce cache thrashing
    Decode,
}

impl InferenceMode {
    /// Returns true if this is prefill mode
    #[must_use]
    pub fn is_prefill(self) -> bool {
        matches!(self, Self::Prefill)
    }

    /// Returns true if this is decode mode
    #[must_use]
    pub fn is_decode(self) -> bool {
        matches!(self, Self::Decode)
    }
}

/// Configure the global rayon thread pool with optimal thread count for inference
///
/// Uses ~16 threads by default for optimal memory bandwidth utilization on
/// modern multi-core CPUs. This can be overridden with `RAYON_NUM_THREADS` env var.
///
/// # Errors
///
/// Returns `InvalidConfiguration` if the thread pool has already been initialized.
///
/// # Example
///
/// ```ignore
/// // Call once at startup
/// realizar::inference::configure_optimal_thread_pool()?;
/// ```
pub fn configure_optimal_thread_pool() -> Result<()> {
    let optimal_threads = std::env::var("RAYON_NUM_THREADS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(16);

    configure_thread_pool(optimal_threads)
}

/// Configure the global rayon thread pool for inference
///
/// NOTE: This should be called once at startup. Rayon's global pool cannot
/// be resized dynamically.
///
/// # Errors
///
/// Returns `InvalidConfiguration` if the thread pool has already been initialized.
pub fn configure_thread_pool(num_threads: usize) -> Result<()> {
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build_global()
        .map_err(|e| {
            RealizarError::InvalidConfiguration(format!("Failed to configure thread pool: {e}"))
        })
}

// ============================================================================
// EXTREME TDD: Comprehensive Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------------
    // ThreadConfig Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_thread_config_auto_returns_valid_config() {
        let config = ThreadConfig::auto();
        assert!(config.n_threads_batch >= 1, "batch threads must be >= 1");
        assert!(config.n_threads_decode >= 1, "decode threads must be >= 1");
    }

    #[test]
    fn test_thread_config_auto_batch_gte_decode() {
        let config = ThreadConfig::auto();
        assert!(
            config.n_threads_batch >= config.n_threads_decode,
            "batch threads should be >= decode threads"
        );
    }

    #[test]
    fn test_thread_config_new_clamps_to_minimum() {
        let config = ThreadConfig::new(0, 0);
        assert_eq!(config.n_threads_batch, 1, "0 should be clamped to 1");
        assert_eq!(config.n_threads_decode, 1, "0 should be clamped to 1");
    }

    #[test]
    fn test_thread_config_new_preserves_valid_values() {
        let config = ThreadConfig::new(8, 4);
        assert_eq!(config.n_threads_batch, 8);
        assert_eq!(config.n_threads_decode, 4);
    }

    #[test]
    fn test_thread_config_threads_for_prefill() {
        let config = ThreadConfig::new(16, 8);
        assert_eq!(config.threads_for(true), 16);
    }

    #[test]
    fn test_thread_config_threads_for_decode() {
        let config = ThreadConfig::new(16, 8);
        assert_eq!(config.threads_for(false), 8);
    }

    #[test]
    fn test_thread_config_default_uses_auto() {
        let default_config = ThreadConfig::default();
        let auto_config = ThreadConfig::auto();
        assert_eq!(default_config, auto_config);
    }

    #[test]
    fn test_thread_config_clone() {
        let config = ThreadConfig::new(12, 6);
        let cloned = config;
        assert_eq!(config, cloned);
    }

    #[test]
    fn test_thread_config_debug() {
        let config = ThreadConfig::new(4, 2);
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("ThreadConfig"));
        assert!(debug_str.contains("4"));
        assert!(debug_str.contains("2"));
    }

    // ------------------------------------------------------------------------
    // InferenceMode Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_inference_mode_is_prefill() {
        assert!(InferenceMode::Prefill.is_prefill());
        assert!(!InferenceMode::Decode.is_prefill());
    }

    #[test]
    fn test_inference_mode_is_decode() {
        assert!(InferenceMode::Decode.is_decode());
        assert!(!InferenceMode::Prefill.is_decode());
    }

    #[test]
    fn test_inference_mode_equality() {
        assert_eq!(InferenceMode::Prefill, InferenceMode::Prefill);
        assert_eq!(InferenceMode::Decode, InferenceMode::Decode);
        assert_ne!(InferenceMode::Prefill, InferenceMode::Decode);
    }

    #[test]
    fn test_inference_mode_clone() {
        let mode = InferenceMode::Prefill;
        let cloned = mode;
        assert_eq!(mode, cloned);
    }

    #[test]
    fn test_inference_mode_debug() {
        assert_eq!(format!("{:?}", InferenceMode::Prefill), "Prefill");
        assert_eq!(format!("{:?}", InferenceMode::Decode), "Decode");
    }

    #[test]
    fn test_inference_mode_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(InferenceMode::Prefill);
        set.insert(InferenceMode::Decode);
        assert_eq!(set.len(), 2);
        assert!(set.contains(&InferenceMode::Prefill));
        assert!(set.contains(&InferenceMode::Decode));
    }

    // ------------------------------------------------------------------------
    // Integration Tests: ThreadConfig + InferenceMode
    // ------------------------------------------------------------------------

    #[test]
    fn test_config_with_mode() {
        let config = ThreadConfig::new(16, 4);

        let prefill_threads = config.threads_for(InferenceMode::Prefill.is_prefill());
        let decode_threads = config.threads_for(InferenceMode::Decode.is_prefill());

        assert_eq!(prefill_threads, 16);
        assert_eq!(decode_threads, 4);
    }

    // ------------------------------------------------------------------------
    // Edge Case Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_thread_config_with_one_thread() {
        let config = ThreadConfig::new(1, 1);
        assert_eq!(config.threads_for(true), 1);
        assert_eq!(config.threads_for(false), 1);
    }

    #[test]
    fn test_thread_config_large_values() {
        let config = ThreadConfig::new(1024, 512);
        assert_eq!(config.n_threads_batch, 1024);
        assert_eq!(config.n_threads_decode, 512);
    }

    #[test]
    fn test_thread_config_decode_larger_than_batch() {
        // This is allowed, though unusual
        let config = ThreadConfig::new(4, 8);
        assert_eq!(config.n_threads_batch, 4);
        assert_eq!(config.n_threads_decode, 8);
    }
}
