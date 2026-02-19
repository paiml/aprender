#![allow(clippy::disallowed_methods)]
//! Property-based tests for chaos engineering infrastructure
//!
//! Tests chaos configuration using proptest to verify invariants
//! across random inputs. Based on renacer chaos testing patterns.

use aprender::chaos::{ChaosConfig, ChaosError, ChaosResult};
use proptest::prelude::*;
use std::time::Duration;

proptest! {
    /// CPU limit should always be clamped to [0.0, 1.0] range
    #[test]
    fn test_cpu_limit_clamping(limit in any::<f64>()) {
        let config = ChaosConfig::new().with_cpu_limit(limit);
        assert!(config.cpu_limit >= 0.0 && config.cpu_limit <= 1.0);
    }

    /// Memory limit should be accepted as-is (non-negative usize)
    #[test]
    fn test_memory_limit_nonnegative(limit in any::<usize>()) {
        let config = ChaosConfig::new().with_memory_limit(limit);
        assert_eq!(config.memory_limit, limit);
    }

    /// Signal injection flag should be set correctly
    #[test]
    fn test_signal_injection_flag(enabled in any::<bool>()) {
        let config = ChaosConfig::new().with_signal_injection(enabled);
        assert_eq!(config.signal_injection, enabled);
    }

    /// Timeout should be set correctly
    #[test]
    fn test_timeout_setting(secs in 1u64..10000u64) {
        let duration = Duration::from_secs(secs);
        let config = ChaosConfig::new().with_timeout(duration);
        assert_eq!(config.timeout, duration);
    }

    /// Builder pattern should preserve all settings
    #[test]
    fn test_builder_preserves_settings(
        mem in any::<usize>(),
        cpu in any::<f64>(),
        secs in 1u64..10000u64,
        sig in any::<bool>()
    ) {
        let duration = Duration::from_secs(secs);
        let config = ChaosConfig::new()
            .with_memory_limit(mem)
            .with_cpu_limit(cpu)
            .with_timeout(duration)
            .with_signal_injection(sig)
            .build();

        assert_eq!(config.memory_limit, mem);
        assert!(config.cpu_limit >= 0.0 && config.cpu_limit <= 1.0);
        assert_eq!(config.timeout, duration);
        assert_eq!(config.signal_injection, sig);
    }

    /// Clone should produce identical config
    #[test]
    fn test_config_clone_equality(
        mem in any::<usize>(),
        cpu in 0.0f64..1.0f64,
        secs in 1u64..1000u64
    ) {
        let config = ChaosConfig::new()
            .with_memory_limit(mem)
            .with_cpu_limit(cpu)
            .with_timeout(Duration::from_secs(secs))
            .build();

        let cloned = config.clone();
        assert_eq!(config, cloned);
    }
}

#[test]
fn test_chaos_error_memory_limit_exceeded() {
    let err = ChaosError::MemoryLimitExceeded {
        limit: 1000,
        used: 2000,
    };
    assert!(err.to_string().contains("2000"));
    assert!(err.to_string().contains("1000"));
}

#[test]
fn test_chaos_error_timeout() {
    let err = ChaosError::Timeout {
        elapsed: Duration::from_secs(15),
        limit: Duration::from_secs(10),
    };
    assert!(err.to_string().contains("Timeout"));
}

#[test]
fn test_chaos_error_signal_injection() {
    let err = ChaosError::SignalInjectionFailed {
        signal: 9,
        reason: "Permission denied".to_string(),
    };
    assert!(err.to_string().contains('9'));
    assert!(err.to_string().contains("Permission denied"));
}

#[test]
fn test_chaos_result_type() {
    // Test that ChaosResult is properly aliased to Result<T, ChaosError>
    let ok: ChaosResult<i32> = Ok(42);
    let err: ChaosResult<i32> = Err(ChaosError::MemoryLimitExceeded {
        limit: 100,
        used: 200,
    });
    // Type alias compiles correctly - verify they work
    assert!(ok.is_ok());
    assert!(err.is_err());
}

#[test]
fn test_gentle_preset_invariants() {
    let gentle = ChaosConfig::gentle();
    assert!(gentle.memory_limit > 0);
    assert!(gentle.cpu_limit > 0.0 && gentle.cpu_limit <= 1.0);
    assert!(gentle.timeout > Duration::from_secs(0));
    assert!(!gentle.signal_injection);
}

#[test]
fn test_aggressive_preset_invariants() {
    let aggressive = ChaosConfig::aggressive();
    assert!(aggressive.memory_limit > 0);
    assert!(aggressive.cpu_limit > 0.0 && aggressive.cpu_limit <= 1.0);
    assert!(aggressive.timeout > Duration::from_secs(0));
    assert!(aggressive.signal_injection);
}

#[test]
fn test_gentle_vs_aggressive() {
    let gentle = ChaosConfig::gentle();
    let aggressive = ChaosConfig::aggressive();

    // Gentle should have more generous limits
    assert!(gentle.memory_limit > aggressive.memory_limit);
    assert!(gentle.cpu_limit > aggressive.cpu_limit);
    assert!(gentle.timeout > aggressive.timeout);
}

#[test]
fn test_default_equals_new() {
    let default = ChaosConfig::default();
    let new = ChaosConfig::new();
    assert_eq!(default, new);
}

#[cfg(feature = "chaos-basic")]
#[test]
fn test_chaos_basic_feature_enabled() {
    // This test only runs when chaos-basic feature is enabled
    let config = ChaosConfig::new();
    assert!(config.cpu_limit >= 0.0);
}

#[cfg(feature = "chaos-full")]
#[test]
fn test_chaos_full_feature_enabled() {
    // This test only runs when chaos-full feature is enabled
    let config = ChaosConfig::aggressive();
    assert!(config.signal_injection);
}
