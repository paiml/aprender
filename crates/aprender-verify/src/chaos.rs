//! # Chaos Engineering Module
//!
//! Chaos engineering infrastructure adapted from **renacer v0.4.1**
//! (<https://github.com/paiml/renacer>).
//!
//! This module provides chaos engineering capabilities for testing system resilience
//! under adverse conditions. It integrates with the certeza testing framework to
//! validate behavior under resource constraints and failure scenarios.
//!
//! ## Chaos Testing Philosophy
//!
//! Chaos engineering validates that systems behave correctly under stress:
//! - Memory constraints (allocation pressure)
//! - CPU limits (throttling)
//! - Timeout conditions (deadlines)
//! - Signal injection (interrupts)
//!
//! ## Tier Integration
//!
//! - **Tier 2 (ON-COMMIT)**: Basic chaos tests with gentle constraints
//! - **Tier 3 (ON-MERGE/NIGHTLY)**: Aggressive chaos tests with extreme limits
//!
//! ## Example
//!
//! ```rust
//! use certeza::chaos::{ChaosConfig, ChaosResult};
//! use std::time::Duration;
//!
//! // Gentle chaos for development (renacer preset)
//! let config = ChaosConfig::gentle();
//!
//! // Aggressive chaos for CI (renacer preset)
//! let aggressive = ChaosConfig::aggressive();
//!
//! // Custom configuration (renacer builder pattern)
//! let custom = ChaosConfig::new()
//!     .with_memory_limit(128 * 1024 * 1024)  // 128MB
//!     .with_cpu_limit(0.5)                   // 50% CPU
//!     .with_timeout(Duration::from_secs(30))
//!     .with_signal_injection(true)
//!     .build();
//! ```
//!
//! ## Source
//!
//! This implementation is based on the renacer project's Sprint 29 chaos engineering
//! framework, adapted for certeza's tiered TDD-X approach.

use std::time::Duration;

/// Result type for chaos engineering operations.
///
/// Based on renacer's `ChaosResult<T>` pattern.
///
/// # Examples
///
/// ```rust
/// use certeza::chaos::{ChaosResult, ChaosError};
/// use std::time::Duration;
///
/// fn simulated_operation(limit: Duration) -> ChaosResult<String> {
///     let elapsed = Duration::from_secs(5);
///     if elapsed > limit {
///         Err(ChaosError::Timeout { elapsed, limit })
///     } else {
///         Ok("success".to_string())
///     }
/// }
/// ```
pub type ChaosResult<T> = Result<T, ChaosError>;

/// Error types for chaos engineering scenarios.
///
/// Based on renacer's chaos error taxonomy. These errors represent
/// intentional failure modes injected during chaos testing.
///
/// # Examples
///
/// ```rust
/// use certeza::chaos::ChaosError;
/// use std::time::Duration;
///
/// let error = ChaosError::MemoryLimitExceeded {
///     limit: 1024,
///     used: 2048,
/// };
///
/// assert_eq!(
///     format!("{}", error),
///     "Memory limit exceeded: 2048 > 1024 bytes"
/// );
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChaosError {
    /// Memory allocation exceeded configured limit.
    ///
    /// Represents out-of-memory conditions under resource constraints.
    MemoryLimitExceeded {
        /// Configured memory limit in bytes
        limit: usize,
        /// Actual memory usage in bytes
        used: usize,
    },

    /// Operation exceeded configured timeout.
    ///
    /// Represents deadline violations in time-constrained scenarios.
    Timeout {
        /// Actual elapsed time
        elapsed: Duration,
        /// Configured timeout limit
        limit: Duration,
    },

    /// Signal injection failed.
    ///
    /// Represents failures in interrupt simulation (SIGINT, SIGTERM, etc.).
    SignalInjectionFailed {
        /// Signal number (e.g., 2 for SIGINT, 15 for SIGTERM)
        signal: i32,
        /// Human-readable failure reason
        reason: String,
    },
}

impl std::fmt::Display for ChaosError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MemoryLimitExceeded { limit, used } => {
                write!(f, "Memory limit exceeded: {used} > {limit} bytes")
            }
            Self::Timeout { elapsed, limit } => {
                write!(f, "Timeout: {elapsed:?} > {limit:?}")
            }
            Self::SignalInjectionFailed { signal, reason } => {
                write!(f, "Signal injection failed ({signal}): {reason}")
            }
        }
    }
}

impl std::error::Error for ChaosError {}

/// Configuration for chaos engineering experiments.
///
/// Defines resource constraints and failure scenarios to inject during testing.
/// All constraints are optional - zero/default values mean no limit enforced.
///
/// # Examples
///
/// ```rust
/// use certeza::chaos::ChaosConfig;
/// use std::time::Duration;
///
/// // Default: no limits
/// let default = ChaosConfig::default();
/// assert_eq!(default.memory_limit, 0);
/// assert_eq!(default.cpu_limit, 0.0);
///
/// // Gentle constraints for local development
/// let gentle = ChaosConfig::gentle();
/// assert_eq!(gentle.memory_limit, 512 * 1024 * 1024);  // 512MB
///
/// // Aggressive constraints for CI/CD
/// let aggressive = ChaosConfig::aggressive();
/// assert_eq!(aggressive.memory_limit, 64 * 1024 * 1024);  // 64MB
/// assert!(aggressive.signal_injection);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct ChaosConfig {
    /// Maximum memory allocation in bytes (0 = unlimited).
    ///
    /// When non-zero, operations exceeding this limit should fail gracefully.
    pub memory_limit: usize,

    /// CPU utilization limit as fraction (0.0-1.0, 0.0 = unlimited).
    ///
    /// Represents the maximum CPU fraction available to the process.
    /// Values outside [0.0, 1.0] are automatically clamped.
    pub cpu_limit: f64,

    /// Operation timeout duration.
    ///
    /// Long-running operations should respect this deadline and fail
    /// gracefully if exceeded.
    pub timeout: Duration,

    /// Whether to inject simulated signal interrupts.
    ///
    /// When enabled, tests may simulate SIGINT, SIGTERM, or other signals
    /// to validate interrupt handling.
    pub signal_injection: bool,
}

impl Default for ChaosConfig {
    /// Creates a configuration with no limits (passthrough mode).
    ///
    /// Useful for baseline testing without chaos constraints.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use certeza::chaos::ChaosConfig;
    ///
    /// let config = ChaosConfig::default();
    /// assert_eq!(config.memory_limit, 0);
    /// assert_eq!(config.cpu_limit, 0.0);
    /// assert!(!config.signal_injection);
    /// ```
    fn default() -> Self {
        Self {
            memory_limit: 0,
            cpu_limit: 0.0,
            timeout: Duration::from_secs(60),
            signal_injection: false,
        }
    }
}

impl ChaosConfig {
    /// Creates a new chaos configuration with default (unlimited) settings.
    ///
    /// Use the builder pattern to customize constraints.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use certeza::chaos::ChaosConfig;
    /// use std::time::Duration;
    ///
    /// let config = ChaosConfig::new()
    ///     .with_memory_limit(256 * 1024 * 1024)
    ///     .with_timeout(Duration::from_secs(120))
    ///     .build();
    ///
    /// assert_eq!(config.memory_limit, 256 * 1024 * 1024);
    /// ```
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the memory limit in bytes.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use certeza::chaos::ChaosConfig;
    ///
    /// let config = ChaosConfig::new()
    ///     .with_memory_limit(128 * 1024 * 1024);  // 128MB
    ///
    /// assert_eq!(config.memory_limit, 128 * 1024 * 1024);
    /// ```
    #[must_use]
    pub const fn with_memory_limit(mut self, bytes: usize) -> Self {
        self.memory_limit = bytes;
        self
    }

    /// Sets the CPU utilization limit as a fraction (0.0-1.0).
    ///
    /// Values outside the valid range are automatically clamped.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use certeza::chaos::ChaosConfig;
    ///
    /// // Valid fraction
    /// let config = ChaosConfig::new().with_cpu_limit(0.5);
    /// assert_eq!(config.cpu_limit, 0.5);
    ///
    /// // Clamped to maximum
    /// let clamped = ChaosConfig::new().with_cpu_limit(2.0);
    /// assert_eq!(clamped.cpu_limit, 1.0);
    ///
    /// // Clamped to minimum
    /// let negative = ChaosConfig::new().with_cpu_limit(-0.5);
    /// assert_eq!(negative.cpu_limit, 0.0);
    /// ```
    #[must_use]
    pub const fn with_cpu_limit(mut self, fraction: f64) -> Self {
        self.cpu_limit = fraction.clamp(0.0, 1.0);
        self
    }

    /// Sets the operation timeout.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use certeza::chaos::ChaosConfig;
    /// use std::time::Duration;
    ///
    /// let config = ChaosConfig::new()
    ///     .with_timeout(Duration::from_secs(30));
    ///
    /// assert_eq!(config.timeout, Duration::from_secs(30));
    /// ```
    #[must_use]
    pub const fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Enables or disables signal injection.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use certeza::chaos::ChaosConfig;
    ///
    /// let config = ChaosConfig::new().with_signal_injection(true);
    /// assert!(config.signal_injection);
    /// ```
    #[must_use]
    pub const fn with_signal_injection(mut self, enabled: bool) -> Self {
        self.signal_injection = enabled;
        self
    }

    /// Finalizes the configuration (terminal builder method).
    ///
    /// This is a no-op that exists for ergonomic builder pattern completion.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use certeza::chaos::ChaosConfig;
    ///
    /// let config = ChaosConfig::new()
    ///     .with_memory_limit(256 * 1024 * 1024)
    ///     .build();
    /// ```
    #[must_use]
    pub const fn build(self) -> Self {
        self
    }

    /// Creates a gentle chaos configuration for local development.
    ///
    /// Applies moderate constraints that catch obvious issues without
    /// excessive developer friction.
    ///
    /// **Constraints:**
    /// - Memory: 512MB
    /// - CPU: 80% (0.8)
    /// - Timeout: 120 seconds
    /// - Signal injection: disabled
    ///
    /// # Examples
    ///
    /// ```rust
    /// use certeza::chaos::ChaosConfig;
    /// use std::time::Duration;
    ///
    /// let config = ChaosConfig::gentle();
    /// assert_eq!(config.memory_limit, 512 * 1024 * 1024);
    /// assert_eq!(config.cpu_limit, 0.8);
    /// assert_eq!(config.timeout, Duration::from_secs(120));
    /// assert!(!config.signal_injection);
    /// ```
    #[must_use]
    pub fn gentle() -> Self {
        Self::new()
            .with_memory_limit(512 * 1024 * 1024) // 512MB
            .with_cpu_limit(0.8) // 80% CPU
            .with_timeout(Duration::from_secs(120))
    }

    /// Creates an aggressive chaos configuration for CI/CD and stress testing.
    ///
    /// Applies severe constraints to expose edge cases and failure modes
    /// that might occur in production under load.
    ///
    /// **Constraints:**
    /// - Memory: 64MB
    /// - CPU: 25% (0.25)
    /// - Timeout: 10 seconds
    /// - Signal injection: enabled
    ///
    /// # Examples
    ///
    /// ```rust
    /// use certeza::chaos::ChaosConfig;
    /// use std::time::Duration;
    ///
    /// let config = ChaosConfig::aggressive();
    /// assert_eq!(config.memory_limit, 64 * 1024 * 1024);
    /// assert_eq!(config.cpu_limit, 0.25);
    /// assert_eq!(config.timeout, Duration::from_secs(10));
    /// assert!(config.signal_injection);
    /// ```
    #[must_use]
    pub fn aggressive() -> Self {
        Self::new()
            .with_memory_limit(64 * 1024 * 1024) // 64MB
            .with_cpu_limit(0.25) // 25% CPU
            .with_timeout(Duration::from_secs(10))
            .with_signal_injection(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ChaosConfig tests
    #[test]
    #[allow(clippy::float_cmp)]
    fn test_default_config() {
        let config = ChaosConfig::default();
        assert_eq!(config.memory_limit, 0);
        assert_eq!(config.cpu_limit, 0.0);
        assert_eq!(config.timeout, Duration::from_secs(60));
        assert!(!config.signal_injection);
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_builder_pattern() {
        let config = ChaosConfig::new()
            .with_memory_limit(1024)
            .with_cpu_limit(0.5)
            .with_timeout(Duration::from_secs(30))
            .with_signal_injection(true)
            .build();

        assert_eq!(config.memory_limit, 1024);
        assert_eq!(config.cpu_limit, 0.5);
        assert_eq!(config.timeout, Duration::from_secs(30));
        assert!(config.signal_injection);
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_cpu_limit_clamping() {
        let too_high = ChaosConfig::new().with_cpu_limit(2.0);
        assert_eq!(too_high.cpu_limit, 1.0);

        let too_low = ChaosConfig::new().with_cpu_limit(-1.0);
        assert_eq!(too_low.cpu_limit, 0.0);

        let valid = ChaosConfig::new().with_cpu_limit(0.75);
        assert_eq!(valid.cpu_limit, 0.75);
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_gentle_preset() {
        let config = ChaosConfig::gentle();
        assert_eq!(config.memory_limit, 512 * 1024 * 1024);
        assert_eq!(config.cpu_limit, 0.8);
        assert_eq!(config.timeout, Duration::from_secs(120));
        assert!(!config.signal_injection);
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_aggressive_preset() {
        let config = ChaosConfig::aggressive();
        assert_eq!(config.memory_limit, 64 * 1024 * 1024);
        assert_eq!(config.cpu_limit, 0.25);
        assert_eq!(config.timeout, Duration::from_secs(10));
        assert!(config.signal_injection);
    }

    #[test]
    #[allow(clippy::redundant_clone)]
    fn test_clone() {
        let original = ChaosConfig::gentle();
        let cloned = original.clone();
        assert_eq!(&original, &cloned);
    }

    #[test]
    fn test_debug_format() {
        let config = ChaosConfig::new();
        let debug_str = format!("{config:?}");
        assert!(debug_str.contains("ChaosConfig"));
        assert!(debug_str.contains("memory_limit"));
    }

    // ChaosError tests
    #[test]
    fn test_memory_limit_exceeded_display() {
        let error = ChaosError::MemoryLimitExceeded { limit: 1024, used: 2048 };
        assert_eq!(format!("{error}"), "Memory limit exceeded: 2048 > 1024 bytes");
    }

    #[test]
    fn test_timeout_display() {
        let error =
            ChaosError::Timeout { elapsed: Duration::from_secs(5), limit: Duration::from_secs(3) };
        let display = format!("{error}");
        assert!(display.contains("Timeout"));
        assert!(display.contains("5s"));
        assert!(display.contains("3s"));
    }

    #[test]
    fn test_signal_injection_failed_display() {
        let error = ChaosError::SignalInjectionFailed {
            signal: 2,
            reason: "Process not found".to_string(),
        };
        assert_eq!(format!("{error}"), "Signal injection failed (2): Process not found");
    }

    #[test]
    #[allow(clippy::redundant_clone)]
    fn test_chaos_error_clone() {
        let error = ChaosError::MemoryLimitExceeded { limit: 100, used: 200 };
        let cloned = error.clone();
        assert_eq!(&error, &cloned);
    }

    #[test]
    fn test_chaos_error_debug() {
        let error = ChaosError::Timeout {
            elapsed: Duration::from_secs(1),
            limit: Duration::from_millis(500),
        };
        let debug_str = format!("{error:?}");
        assert!(debug_str.contains("Timeout"));
    }

    // ChaosResult tests
    #[test]
    fn test_chaos_result_ok() {
        let result: ChaosResult<i32> = Ok(42);
        if let Ok(value) = result {
            assert_eq!(value, 42);
        } else {
            panic!("Expected Ok result");
        }
    }

    #[test]
    fn test_chaos_result_err() {
        let result: ChaosResult<i32> =
            Err(ChaosError::MemoryLimitExceeded { limit: 100, used: 200 });
        assert!(result.is_err());
    }
}
