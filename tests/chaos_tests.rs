#![allow(clippy::disallowed_methods, clippy::float_cmp)]
//! Chaos Engineering Property-Based Tests
//!
//! These tests use proptest to verify chaos configuration behavior under
//! arbitrary inputs, following renacer v0.4.1 patterns.

use proptest::prelude::*;
use std::time::Duration;
use trueno::chaos::{ChaosConfig, ChaosError};

proptest! {
    /// CPU limit should always be clamped to [0.0, 1.0] range
    #[test]
    fn test_cpu_limit_clamping(limit in any::<f64>()) {
        let config = ChaosConfig::new().with_cpu_limit(limit);
        assert!(config.cpu_limit >= 0.0, "CPU limit {} should be >= 0.0", config.cpu_limit);
        assert!(config.cpu_limit <= 1.0, "CPU limit {} should be <= 1.0", config.cpu_limit);
    }

    /// Memory limit should always be non-negative and preserve input
    #[test]
    fn test_memory_limit_nonnegative(limit in any::<usize>()) {
        let config = ChaosConfig::new().with_memory_limit(limit);
        assert_eq!(config.memory_limit, limit, "Memory limit should preserve input value");
    }

    /// Timeout should preserve any duration value
    #[test]
    fn test_timeout_preservation(secs in 0u64..=3600) {
        let timeout = Duration::from_secs(secs);
        let config = ChaosConfig::new().with_timeout(timeout);
        assert_eq!(config.timeout, timeout, "Timeout should preserve input value");
    }

    /// Signal injection flag should preserve boolean value
    #[test]
    fn test_signal_injection_flag(enabled in any::<bool>()) {
        let config = ChaosConfig::new().with_signal_injection(enabled);
        assert_eq!(config.signal_injection, enabled, "Signal injection flag should preserve input");
    }

    /// Builder pattern should be chainable with arbitrary values
    #[test]
    fn test_builder_chaining(
        mem in any::<usize>(),
        cpu in any::<f64>(),
        secs in 0u64..=3600,
        sig in any::<bool>()
    ) {
        let config = ChaosConfig::new()
            .with_memory_limit(mem)
            .with_cpu_limit(cpu)
            .with_timeout(Duration::from_secs(secs))
            .with_signal_injection(sig)
            .build();

        assert_eq!(config.memory_limit, mem);
        assert!(config.cpu_limit >= 0.0 && config.cpu_limit <= 1.0);
        assert_eq!(config.timeout, Duration::from_secs(secs));
        assert_eq!(config.signal_injection, sig);
    }
}

#[test]
fn test_gentle_preset_properties() {
    let gentle = ChaosConfig::gentle();

    // Gentle should have reasonable limits
    assert!(gentle.memory_limit > 0, "Gentle preset should have memory limit");
    assert!(gentle.memory_limit >= 64 * 1024 * 1024, "Gentle should allow at least 64MB");
    assert!(gentle.cpu_limit > 0.0, "Gentle should have CPU limit");
    assert!(gentle.cpu_limit >= 0.5, "Gentle should allow >= 50% CPU");
    assert!(gentle.timeout >= Duration::from_secs(60), "Gentle should allow >= 60s timeout");
    assert!(!gentle.signal_injection, "Gentle should not inject signals by default");
}

#[test]
fn test_aggressive_preset_properties() {
    let aggressive = ChaosConfig::aggressive();

    // Aggressive should have strict limits
    assert!(aggressive.memory_limit > 0, "Aggressive preset should have memory limit");
    assert!(aggressive.memory_limit <= 128 * 1024 * 1024, "Aggressive should limit <= 128MB");
    assert!(aggressive.cpu_limit > 0.0, "Aggressive should have CPU limit");
    assert!(aggressive.cpu_limit <= 0.5, "Aggressive should limit <= 50% CPU");
    assert!(
        aggressive.timeout <= Duration::from_secs(30),
        "Aggressive should limit <= 30s timeout"
    );
}

#[test]
fn test_chaos_error_display_messages() {
    let mem_err = ChaosError::MemoryLimitExceeded { limit: 1000, used: 2000 };
    let msg = format!("{}", mem_err);
    assert!(msg.contains("Memory limit exceeded"));
    assert!(msg.contains("2000"));
    assert!(msg.contains("1000"));

    let timeout_err =
        ChaosError::Timeout { elapsed: Duration::from_secs(5), limit: Duration::from_secs(3) };
    let msg = format!("{}", timeout_err);
    assert!(msg.contains("Timeout"));

    let signal_err =
        ChaosError::SignalInjectionFailed { signal: 9, reason: "permission denied".to_string() };
    let msg = format!("{}", signal_err);
    assert!(msg.contains("Signal injection failed"));
    assert!(msg.contains('9'));
    assert!(msg.contains("permission denied"));
}

#[test]
fn test_default_is_permissive() {
    let default = ChaosConfig::default();

    // Default should impose no limits (permissive for testing)
    assert_eq!(default.memory_limit, 0, "Default should have no memory limit");
    assert_eq!(default.cpu_limit, 0.0, "Default should have no CPU limit");
    assert_eq!(default.timeout, Duration::from_secs(60), "Default timeout should be reasonable");
    assert!(!default.signal_injection, "Default should not inject signals");
}

#[test]
fn test_presets_are_distinct() {
    let gentle = ChaosConfig::gentle();
    let aggressive = ChaosConfig::aggressive();

    // Presets should have different characteristics
    assert!(gentle.memory_limit > aggressive.memory_limit, "Gentle should allow more memory");
    assert!(gentle.cpu_limit > aggressive.cpu_limit, "Gentle should allow more CPU");
    assert!(gentle.timeout > aggressive.timeout, "Gentle should allow longer timeout");
}
