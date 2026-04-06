//! Property-based tests for chaos engineering module
//!
//! Based on renacer's chaos testing approach from Sprint 29.
//! Uses proptest to validate chaos configuration properties.

use certeza::chaos::{ChaosConfig, ChaosError, ChaosResult};
use proptest::prelude::*;
use std::time::Duration;

// ============================================================================
// Property-Based Tests for ChaosConfig
// ============================================================================

proptest! {
    /// Property: CPU limit is always clamped to [0.0, 1.0]
    #[test]
    #[allow(clippy::float_cmp)]
    fn test_cpu_limit_clamping(limit in any::<f64>()) {
        let config = ChaosConfig::new().with_cpu_limit(limit);
        prop_assert!(config.cpu_limit >= 0.0 && config.cpu_limit <= 1.0);
    }

    /// Property: Memory limit is always non-negative
    #[test]
    fn test_memory_limit_nonnegative(limit in any::<usize>()) {
        let config = ChaosConfig::new().with_memory_limit(limit);
        prop_assert_eq!(config.memory_limit, limit);
    }

    /// Property: Timeout is always set correctly
    #[test]
    fn test_timeout_preservation(secs in 0u64..10000) {
        let timeout = Duration::from_secs(secs);
        let config = ChaosConfig::new().with_timeout(timeout);
        prop_assert_eq!(config.timeout, timeout);
    }

    /// Property: Signal injection flag is preserved
    #[test]
    fn test_signal_injection_preservation(enabled in any::<bool>()) {
        let config = ChaosConfig::new().with_signal_injection(enabled);
        prop_assert_eq!(config.signal_injection, enabled);
    }

    /// Property: Builder pattern is order-independent (commutativity)
    #[test]
    fn test_builder_order_independence(
        mem in 0usize..1_000_000_000,
        cpu in 0.0f64..2.0,
        secs in 1u64..1000,
        signal in any::<bool>()
    ) {
        let timeout = Duration::from_secs(secs);

        let config1 = ChaosConfig::new()
            .with_memory_limit(mem)
            .with_cpu_limit(cpu)
            .with_timeout(timeout)
            .with_signal_injection(signal)
            .build();

        let config2 = ChaosConfig::new()
            .with_signal_injection(signal)
            .with_timeout(timeout)
            .with_cpu_limit(cpu)
            .with_memory_limit(mem)
            .build();

        prop_assert_eq!(config1, config2);
    }

    /// Property: Extreme CPU values are clamped correctly
    #[test]
    #[allow(clippy::float_cmp)]
    fn test_extreme_cpu_limits(limit in -1000.0f64..1000.0) {
        let config = ChaosConfig::new().with_cpu_limit(limit);

        if limit < 0.0 {
            prop_assert_eq!(config.cpu_limit, 0.0);
        } else if limit > 1.0 {
            prop_assert_eq!(config.cpu_limit, 1.0);
        } else {
            prop_assert_eq!(config.cpu_limit, limit);
        }
    }

    /// Property: Clone produces identical configuration
    #[test]
    fn test_clone_equivalence(
        mem in any::<usize>(),
        cpu in 0.0f64..1.0,
        secs in 1u64..1000
    ) {
        let original = ChaosConfig::new()
            .with_memory_limit(mem)
            .with_cpu_limit(cpu)
            .with_timeout(Duration::from_secs(secs));

        let cloned = original.clone();
        prop_assert_eq!(original, cloned);
    }
}

// ============================================================================
// Property-Based Tests for ChaosError
// ============================================================================

proptest! {
    /// Property: MemoryLimitExceeded display format is consistent
    #[test]
    fn test_memory_error_display(limit in 1usize..1_000_000, used in 1usize..1_000_000) {
        let error = ChaosError::MemoryLimitExceeded { limit, used };
        let display = format!("{error}");
        prop_assert!(display.contains(&limit.to_string()));
        prop_assert!(display.contains(&used.to_string()));
        prop_assert!(display.contains("Memory limit exceeded"));
    }

    /// Property: Timeout error display is consistent
    #[test]
    fn test_timeout_error_display(elapsed_secs in 1u64..1000, limit_secs in 1u64..1000) {
        let error = ChaosError::Timeout {
            elapsed: Duration::from_secs(elapsed_secs),
            limit: Duration::from_secs(limit_secs),
        };
        let display = format!("{error}");
        prop_assert!(display.contains("Timeout"));
    }

    /// Property: Signal injection error display is consistent
    #[test]
    fn test_signal_error_display(signal in 1i32..100, reason_len in 1usize..100) {
        let reason = "a".repeat(reason_len);
        let error = ChaosError::SignalInjectionFailed {
            signal,
            reason: reason.clone(),
        };
        let display = format!("{error}");
        prop_assert!(display.contains(&signal.to_string()));
        prop_assert!(display.contains(&reason));
        prop_assert!(display.contains("Signal injection failed"));
    }

    /// Property: Error cloning preserves equality
    #[test]
    fn test_error_clone_equality(limit in 1usize..1_000_000, used in 1usize..1_000_000) {
        let original = ChaosError::MemoryLimitExceeded { limit, used };
        let cloned = original.clone();
        prop_assert_eq!(original, cloned);
    }
}

// ============================================================================
// Property-Based Tests for ChaosResult
// ============================================================================

proptest! {
    /// Property: ChaosResult<T> preserves Ok values
    #[test]
    fn test_chaos_result_ok_preservation(value in any::<i32>()) {
        let result: ChaosResult<i32> = Ok(value);
        prop_assert!(result.is_ok());
        if let Ok(v) = result {
            prop_assert_eq!(v, value);
        }
    }

    /// Property: ChaosResult<T> preserves Err values
    #[test]
    fn test_chaos_result_err_preservation(limit in 1usize..1000, used in 1usize..1000) {
        let error = ChaosError::MemoryLimitExceeded { limit, used };
        let result: ChaosResult<i32> = Err(error.clone());
        prop_assert!(result.is_err());
        if let Err(e) = result {
            prop_assert_eq!(e, error);
        }
    }
}

// ============================================================================
// Unit Tests for Preset Configurations
// ============================================================================

#[cfg(test)]
mod unit_tests {
    use super::*;

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_gentle_preset_values() {
        let config = ChaosConfig::gentle();
        assert_eq!(config.memory_limit, 512 * 1024 * 1024);
        assert_eq!(config.cpu_limit, 0.8);
        assert_eq!(config.timeout, Duration::from_secs(120));
        assert!(!config.signal_injection);
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_aggressive_preset_values() {
        let config = ChaosConfig::aggressive();
        assert_eq!(config.memory_limit, 64 * 1024 * 1024);
        assert_eq!(config.cpu_limit, 0.25);
        assert_eq!(config.timeout, Duration::from_secs(10));
        assert!(config.signal_injection);
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_default_preset_values() {
        let config = ChaosConfig::default();
        assert_eq!(config.memory_limit, 0);
        assert_eq!(config.cpu_limit, 0.0);
        assert_eq!(config.timeout, Duration::from_secs(60));
        assert!(!config.signal_injection);
    }
}

// ============================================================================
// Integration Tests (Tier 2)
// ============================================================================

#[cfg(test)]
mod integration_tests {
    use super::*;

    /// Test that gentle preset is suitable for development
    #[test]
    #[allow(clippy::float_cmp)]
    fn test_gentle_allows_reasonable_operations() {
        let config = ChaosConfig::gentle();

        // Should allow reasonable memory allocation (< 512MB)
        assert!(config.memory_limit >= 100 * 1024 * 1024);

        // Should allow majority of CPU
        assert!(config.cpu_limit >= 0.5);

        // Should have reasonable timeout
        assert!(config.timeout >= Duration::from_secs(60));
    }

    /// Test that aggressive preset is strict for CI/CD
    #[test]
    #[allow(clippy::float_cmp)]
    fn test_aggressive_enforces_strict_limits() {
        let config = ChaosConfig::aggressive();

        // Should enforce tight memory limits
        assert!(config.memory_limit <= 128 * 1024 * 1024);

        // Should restrict CPU heavily
        assert!(config.cpu_limit <= 0.5);

        // Should have short timeout
        assert!(config.timeout <= Duration::from_secs(30));

        // Should inject signals
        assert!(config.signal_injection);
    }

    /// Test error conversion to Result
    #[test]
    fn test_chaos_result_error_propagation() {
        fn failing_operation() -> ChaosResult<()> {
            Err(ChaosError::Timeout {
                elapsed: Duration::from_secs(10),
                limit: Duration::from_secs(5),
            })
        }

        let result = failing_operation();
        assert!(result.is_err());

        match result {
            Err(ChaosError::Timeout { elapsed, limit }) => {
                assert_eq!(elapsed, Duration::from_secs(10));
                assert_eq!(limit, Duration::from_secs(5));
            }
            _ => panic!("Expected Timeout error"),
        }
    }
}
