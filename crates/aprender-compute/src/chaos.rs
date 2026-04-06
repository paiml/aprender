//! Chaos Engineering Configuration
//!
//! This module provides chaos engineering infrastructure for testing Trueno under
//! adverse conditions. Integrated from renacer v0.4.1 (https://github.com/paiml/renacer).
//!
//! # Examples
//!
//! ```
//! use trueno::chaos::ChaosConfig;
//! use std::time::Duration;
//!
//! // Use gentle preset for gradual stress testing
//! let gentle = ChaosConfig::gentle();
//!
//! // Use aggressive preset for extreme conditions
//! let aggressive = ChaosConfig::aggressive();
//!
//! // Custom configuration using builder pattern
//! let custom = ChaosConfig::new()
//!     .with_memory_limit(100 * 1024 * 1024)  // 100 MB
//!     .with_cpu_limit(0.5)                    // 50% CPU
//!     .with_timeout(Duration::from_secs(30))
//!     .with_signal_injection(true)
//!     .build();
//! ```

use std::time::Duration;

/// Chaos engineering configuration for stress testing
///
/// Provides configurable limits and constraints for chaos testing scenarios.
/// All limits are optional and disabled by default (value of 0 means no limit).
#[derive(Debug, Clone, PartialEq)]
pub struct ChaosConfig {
    /// Memory limit in bytes (0 = no limit)
    pub memory_limit: usize,
    /// CPU usage limit as fraction 0.0-1.0 (0.0 = no limit)
    pub cpu_limit: f64,
    /// Maximum execution time
    pub timeout: Duration,
    /// Whether to inject random signals during execution
    pub signal_injection: bool,
}

impl Default for ChaosConfig {
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
    /// Create a new chaos configuration with default values (no limits)
    pub fn new() -> Self {
        Self::default()
    }

    /// Set memory limit in bytes
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::chaos::ChaosConfig;
    ///
    /// let config = ChaosConfig::new()
    ///     .with_memory_limit(512 * 1024 * 1024); // 512 MB
    /// assert_eq!(config.memory_limit, 512 * 1024 * 1024);
    /// ```
    pub fn with_memory_limit(mut self, bytes: usize) -> Self {
        self.memory_limit = bytes;
        self
    }

    /// Set CPU usage limit as fraction (0.0-1.0)
    ///
    /// Values are automatically clamped to valid range.
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::chaos::ChaosConfig;
    ///
    /// let config = ChaosConfig::new().with_cpu_limit(0.75);
    /// assert_eq!(config.cpu_limit, 0.75);
    ///
    /// // Values outside range are clamped
    /// let clamped = ChaosConfig::new().with_cpu_limit(1.5);
    /// assert_eq!(clamped.cpu_limit, 1.0);
    /// ```
    pub fn with_cpu_limit(mut self, fraction: f64) -> Self {
        self.cpu_limit = fraction.clamp(0.0, 1.0);
        self
    }

    /// Set maximum execution timeout
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::chaos::ChaosConfig;
    /// use std::time::Duration;
    ///
    /// let config = ChaosConfig::new()
    ///     .with_timeout(Duration::from_secs(30));
    /// assert_eq!(config.timeout, Duration::from_secs(30));
    /// ```
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Enable/disable random signal injection during execution
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::chaos::ChaosConfig;
    ///
    /// let config = ChaosConfig::new().with_signal_injection(true);
    /// assert!(config.signal_injection);
    /// ```
    pub fn with_signal_injection(mut self, enabled: bool) -> Self {
        self.signal_injection = enabled;
        self
    }

    /// Finalize configuration (no-op, for builder pattern consistency)
    pub fn build(self) -> Self {
        self
    }

    /// Gentle chaos configuration preset
    ///
    /// - 512 MB memory limit
    /// - 80% CPU limit
    /// - 120 second timeout
    /// - No signal injection
    ///
    /// Suitable for gradual stress testing and CI environments.
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::chaos::ChaosConfig;
    /// use std::time::Duration;
    ///
    /// let config = ChaosConfig::gentle();
    /// assert_eq!(config.memory_limit, 512 * 1024 * 1024);
    /// assert_eq!(config.cpu_limit, 0.8);
    /// assert_eq!(config.timeout, Duration::from_secs(120));
    /// assert!(!config.signal_injection);
    /// ```
    pub fn gentle() -> Self {
        Self::new()
            .with_memory_limit(512 * 1024 * 1024)
            .with_cpu_limit(0.8)
            .with_timeout(Duration::from_secs(120))
    }

    /// Aggressive chaos configuration preset
    ///
    /// - 64 MB memory limit
    /// - 25% CPU limit
    /// - 10 second timeout
    /// - Signal injection enabled
    ///
    /// Suitable for extreme stress testing and finding edge cases.
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::chaos::ChaosConfig;
    /// use std::time::Duration;
    ///
    /// let config = ChaosConfig::aggressive();
    /// assert_eq!(config.memory_limit, 64 * 1024 * 1024);
    /// assert_eq!(config.cpu_limit, 0.25);
    /// assert_eq!(config.timeout, Duration::from_secs(10));
    /// assert!(config.signal_injection);
    /// ```
    pub fn aggressive() -> Self {
        Self::new()
            .with_memory_limit(64 * 1024 * 1024)
            .with_cpu_limit(0.25)
            .with_timeout(Duration::from_secs(10))
            .with_signal_injection(true)
    }
}

/// Result type for chaos operations
pub type ChaosResult<T> = Result<T, ChaosError>;

/// Errors that can occur during chaos testing
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChaosError {
    /// Memory limit was exceeded during execution
    MemoryLimitExceeded {
        /// Configured memory limit in bytes
        limit: usize,
        /// Actual memory used in bytes
        used: usize,
    },
    /// Execution exceeded timeout
    Timeout {
        /// Time elapsed before timeout
        elapsed: Duration,
        /// Configured timeout limit
        limit: Duration,
    },
    /// Signal injection failed
    SignalInjectionFailed {
        /// Signal number that failed to inject
        signal: i32,
        /// Reason for failure
        reason: String,
    },
}

impl std::fmt::Display for ChaosError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ChaosError::MemoryLimitExceeded { limit, used } => {
                write!(f, "Memory limit exceeded: {} > {} bytes", used, limit)
            }
            ChaosError::Timeout { elapsed, limit } => {
                write!(f, "Timeout: {:?} > {:?}", elapsed, limit)
            }
            ChaosError::SignalInjectionFailed { signal, reason } => {
                write!(f, "Signal injection failed ({}): {}", signal, reason)
            }
        }
    }
}

impl std::error::Error for ChaosError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = ChaosConfig::default();
        assert_eq!(config.memory_limit, 0);
        assert_eq!(config.cpu_limit, 0.0);
        assert_eq!(config.timeout, Duration::from_secs(60));
        assert!(!config.signal_injection);
    }

    #[test]
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
    fn test_cpu_limit_clamping() {
        let too_high = ChaosConfig::new().with_cpu_limit(1.5);
        assert_eq!(too_high.cpu_limit, 1.0);

        let too_low = ChaosConfig::new().with_cpu_limit(-0.5);
        assert_eq!(too_low.cpu_limit, 0.0);

        let valid = ChaosConfig::new().with_cpu_limit(0.75);
        assert_eq!(valid.cpu_limit, 0.75);
    }

    #[test]
    fn test_gentle_preset() {
        let gentle = ChaosConfig::gentle();
        assert_eq!(gentle.memory_limit, 512 * 1024 * 1024);
        assert_eq!(gentle.cpu_limit, 0.8);
        assert_eq!(gentle.timeout, Duration::from_secs(120));
        assert!(!gentle.signal_injection);
    }

    #[test]
    fn test_aggressive_preset() {
        let aggressive = ChaosConfig::aggressive();
        assert_eq!(aggressive.memory_limit, 64 * 1024 * 1024);
        assert_eq!(aggressive.cpu_limit, 0.25);
        assert_eq!(aggressive.timeout, Duration::from_secs(10));
        assert!(aggressive.signal_injection);
    }

    #[test]
    fn test_chaos_error_display() {
        let mem_err = ChaosError::MemoryLimitExceeded { limit: 1000, used: 2000 };
        assert_eq!(format!("{}", mem_err), "Memory limit exceeded: 2000 > 1000 bytes");

        let timeout_err =
            ChaosError::Timeout { elapsed: Duration::from_secs(5), limit: Duration::from_secs(3) };
        assert!(format!("{}", timeout_err).contains("Timeout"));

        let signal_err =
            ChaosError::SignalInjectionFailed { signal: 9, reason: "test failure".to_string() };
        assert_eq!(format!("{}", signal_err), "Signal injection failed (9): test failure");
    }
}
