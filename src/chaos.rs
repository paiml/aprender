//! Chaos Engineering Configuration
//!
//! Infrastructure for chaos engineering and fault injection testing,
//! ported from [renacer v0.4.1](https://github.com/paiml/renacer).
//!
//! # Examples
//!
//! ```
//! use aprender::chaos::ChaosConfig;
//! use std::time::Duration;
//!
//! // Use presets
//! let gentle = ChaosConfig::gentle();
//! let aggressive = ChaosConfig::aggressive();
//!
//! // Custom configuration
//! let custom = ChaosConfig::new()
//!     .with_memory_limit(100 * 1024 * 1024)
//!     .with_cpu_limit(0.5)
//!     .with_timeout(Duration::from_secs(30))
//!     .with_signal_injection(true)
//!     .build();
//!
//! assert_eq!(custom.memory_limit, 100 * 1024 * 1024);
//! assert_eq!(custom.cpu_limit, 0.5);
//! ```

use std::time::Duration;

/// Chaos engineering configuration for fault injection testing.
///
/// Provides builder pattern for configuring resource limits, timeouts,
/// and signal injection for testing system behavior under stress.
///
/// Based on renacer Sprint 29 chaos infrastructure.
#[derive(Debug, Clone, PartialEq)]
pub struct ChaosConfig {
    /// Memory limit in bytes (0 = unlimited)
    pub memory_limit: usize,
    /// CPU limit as fraction 0.0-1.0 (0.0 = unlimited)
    pub cpu_limit: f64,
    /// Maximum execution timeout
    pub timeout: Duration,
    /// Enable signal injection for fault testing
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
    /// Create a new chaos configuration with defaults.
    ///
    /// # Examples
    ///
    /// ```
    /// use aprender::chaos::ChaosConfig;
    /// use std::time::Duration;
    ///
    /// let config = ChaosConfig::new();
    /// assert_eq!(config.memory_limit, 0);
    /// assert_eq!(config.cpu_limit, 0.0);
    /// assert_eq!(config.timeout, Duration::from_secs(60));
    /// assert_eq!(config.signal_injection, false);
    /// ```
    pub fn new() -> Self {
        Self::default()
    }

    /// Set memory limit in bytes.
    ///
    /// # Examples
    ///
    /// ```
    /// use aprender::chaos::ChaosConfig;
    ///
    /// let config = ChaosConfig::new().with_memory_limit(512 * 1024 * 1024);
    /// assert_eq!(config.memory_limit, 512 * 1024 * 1024);
    /// ```
    pub fn with_memory_limit(mut self, bytes: usize) -> Self {
        self.memory_limit = bytes;
        self
    }

    /// Set CPU limit as fraction (0.0 = unlimited, 1.0 = full).
    ///
    /// Values are clamped to [0.0, 1.0] range.
    ///
    /// # Examples
    ///
    /// ```
    /// use aprender::chaos::ChaosConfig;
    ///
    /// let config = ChaosConfig::new().with_cpu_limit(0.75);
    /// assert_eq!(config.cpu_limit, 0.75);
    ///
    /// // Clamping to valid range
    /// let clamped = ChaosConfig::new().with_cpu_limit(1.5);
    /// assert_eq!(clamped.cpu_limit, 1.0);
    /// ```
    pub fn with_cpu_limit(mut self, fraction: f64) -> Self {
        self.cpu_limit = fraction.clamp(0.0, 1.0);
        self
    }

    /// Set execution timeout.
    ///
    /// # Examples
    ///
    /// ```
    /// use aprender::chaos::ChaosConfig;
    /// use std::time::Duration;
    ///
    /// let config = ChaosConfig::new().with_timeout(Duration::from_secs(120));
    /// assert_eq!(config.timeout, Duration::from_secs(120));
    /// ```
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Enable or disable signal injection.
    ///
    /// # Examples
    ///
    /// ```
    /// use aprender::chaos::ChaosConfig;
    ///
    /// let config = ChaosConfig::new().with_signal_injection(true);
    /// assert!(config.signal_injection);
    /// ```
    pub fn with_signal_injection(mut self, enabled: bool) -> Self {
        self.signal_injection = enabled;
        self
    }

    /// Finalize the configuration (no-op, for API consistency).
    ///
    /// # Examples
    ///
    /// ```
    /// use aprender::chaos::ChaosConfig;
    ///
    /// let config = ChaosConfig::new()
    ///     .with_memory_limit(100 * 1024 * 1024)
    ///     .build();
    /// assert_eq!(config.memory_limit, 100 * 1024 * 1024);
    /// ```
    pub fn build(self) -> Self {
        self
    }

    /// Gentle chaos preset: moderate limits, longer timeout.
    ///
    /// - Memory limit: 512 MB
    /// - CPU limit: 80%
    /// - Timeout: 120 seconds
    /// - Signal injection: disabled
    ///
    /// # Examples
    ///
    /// ```
    /// use aprender::chaos::ChaosConfig;
    /// use std::time::Duration;
    ///
    /// let gentle = ChaosConfig::gentle();
    /// assert_eq!(gentle.memory_limit, 512 * 1024 * 1024);
    /// assert_eq!(gentle.cpu_limit, 0.8);
    /// assert_eq!(gentle.timeout, Duration::from_secs(120));
    /// assert!(!gentle.signal_injection);
    /// ```
    pub fn gentle() -> Self {
        Self::new()
            .with_memory_limit(512 * 1024 * 1024)
            .with_cpu_limit(0.8)
            .with_timeout(Duration::from_secs(120))
    }

    /// Aggressive chaos preset: strict limits, short timeout, signal injection.
    ///
    /// - Memory limit: 64 MB
    /// - CPU limit: 25%
    /// - Timeout: 10 seconds
    /// - Signal injection: enabled
    ///
    /// # Examples
    ///
    /// ```
    /// use aprender::chaos::ChaosConfig;
    /// use std::time::Duration;
    ///
    /// let aggressive = ChaosConfig::aggressive();
    /// assert_eq!(aggressive.memory_limit, 64 * 1024 * 1024);
    /// assert_eq!(aggressive.cpu_limit, 0.25);
    /// assert_eq!(aggressive.timeout, Duration::from_secs(10));
    /// assert!(aggressive.signal_injection);
    /// ```
    pub fn aggressive() -> Self {
        Self::new()
            .with_memory_limit(64 * 1024 * 1024)
            .with_cpu_limit(0.25)
            .with_timeout(Duration::from_secs(10))
            .with_signal_injection(true)
    }
}

/// Result type for chaos operations.
pub type ChaosResult<T> = Result<T, ChaosError>;

/// Errors that can occur during chaos engineering tests.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChaosError {
    /// Memory limit exceeded
    MemoryLimitExceeded {
        /// Configured memory limit in bytes
        limit: usize,
        /// Actual memory used in bytes
        used: usize,
    },
    /// Execution timeout exceeded
    Timeout {
        /// Actual elapsed time
        elapsed: Duration,
        /// Configured timeout limit
        limit: Duration,
    },
    /// Signal injection failed
    SignalInjectionFailed {
        /// Signal number that failed
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
    fn test_chaos_config_new() {
        let config = ChaosConfig::new();
        assert_eq!(config.memory_limit, 0);
        assert_eq!(config.cpu_limit, 0.0);
        assert_eq!(config.timeout, Duration::from_secs(60));
        assert!(!config.signal_injection);
    }

    #[test]
    fn test_chaos_config_default() {
        let config = ChaosConfig::default();
        assert_eq!(config.memory_limit, 0);
        assert_eq!(config.cpu_limit, 0.0);
        assert_eq!(config.timeout, Duration::from_secs(60));
        assert!(!config.signal_injection);
    }

    #[test]
    fn test_with_memory_limit() {
        let config = ChaosConfig::new().with_memory_limit(100 * 1024 * 1024);
        assert_eq!(config.memory_limit, 100 * 1024 * 1024);
    }

    #[test]
    fn test_with_cpu_limit() {
        let config = ChaosConfig::new().with_cpu_limit(0.5);
        assert_eq!(config.cpu_limit, 0.5);
    }

    #[test]
    fn test_cpu_limit_clamping_high() {
        let config = ChaosConfig::new().with_cpu_limit(1.5);
        assert_eq!(config.cpu_limit, 1.0);
    }

    #[test]
    fn test_cpu_limit_clamping_low() {
        let config = ChaosConfig::new().with_cpu_limit(-0.5);
        assert_eq!(config.cpu_limit, 0.0);
    }

    #[test]
    fn test_with_timeout() {
        let config = ChaosConfig::new().with_timeout(Duration::from_secs(30));
        assert_eq!(config.timeout, Duration::from_secs(30));
    }

    #[test]
    fn test_with_signal_injection() {
        let config = ChaosConfig::new().with_signal_injection(true);
        assert!(config.signal_injection);
    }

    #[test]
    fn test_builder_pattern() {
        let config = ChaosConfig::new()
            .with_memory_limit(256 * 1024 * 1024)
            .with_cpu_limit(0.6)
            .with_timeout(Duration::from_secs(90))
            .with_signal_injection(true)
            .build();

        assert_eq!(config.memory_limit, 256 * 1024 * 1024);
        assert_eq!(config.cpu_limit, 0.6);
        assert_eq!(config.timeout, Duration::from_secs(90));
        assert!(config.signal_injection);
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
    fn test_chaos_error_memory_limit_display() {
        let err = ChaosError::MemoryLimitExceeded {
            limit: 1000,
            used: 2000,
        };
        assert_eq!(err.to_string(), "Memory limit exceeded: 2000 > 1000 bytes");
    }

    #[test]
    fn test_chaos_error_timeout_display() {
        let err = ChaosError::Timeout {
            elapsed: Duration::from_secs(15),
            limit: Duration::from_secs(10),
        };
        assert!(err.to_string().contains("Timeout"));
    }

    #[test]
    fn test_chaos_error_signal_injection_display() {
        let err = ChaosError::SignalInjectionFailed {
            signal: 9,
            reason: "Permission denied".to_string(),
        };
        assert_eq!(
            err.to_string(),
            "Signal injection failed (9): Permission denied"
        );
    }

    #[test]
    fn test_chaos_result_type_alias() {
        // Test that ChaosResult is properly aliased to Result<T, ChaosError>
        let _ok: ChaosResult<i32> = Ok(42);
        let _err: ChaosResult<i32> = Err(ChaosError::MemoryLimitExceeded {
            limit: 100,
            used: 200,
        });
        // Type alias compiles correctly
    }

    #[test]
    fn test_chaos_config_clone() {
        let config1 = ChaosConfig::gentle();
        let config2 = config1.clone();
        assert_eq!(config1, config2);
    }

    #[test]
    fn test_chaos_error_clone() {
        let err1 = ChaosError::Timeout {
            elapsed: Duration::from_secs(5),
            limit: Duration::from_secs(3),
        };
        let err2 = err1.clone();
        assert_eq!(err1, err2);
    }
}
