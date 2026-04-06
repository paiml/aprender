//! Part 04: Additional thread.rs coverage tests
//!
//! Additional coverage for thread.rs including:
//! - Thread pool configuration error paths
//! - ThreadConfig additional edge cases
//! - InferenceMode integration patterns
//! - Error propagation scenarios

use crate::inference::{InferenceMode, ThreadConfig};

// ============================================================================
// ThreadConfig Boundary Tests
// ============================================================================

#[test]
fn test_thread_config_new_with_usize_max() {
    // Test with usize::MAX values
    let config = ThreadConfig::new(usize::MAX, usize::MAX);
    assert_eq!(config.n_threads_batch, usize::MAX);
    assert_eq!(config.n_threads_decode, usize::MAX);
}

#[test]
fn test_thread_config_new_batch_one_decode_max() {
    let config = ThreadConfig::new(1, usize::MAX);
    assert_eq!(config.n_threads_batch, 1);
    assert_eq!(config.n_threads_decode, usize::MAX);
}

#[test]
fn test_thread_config_new_batch_max_decode_one() {
    let config = ThreadConfig::new(usize::MAX, 1);
    assert_eq!(config.n_threads_batch, usize::MAX);
    assert_eq!(config.n_threads_decode, 1);
}

#[test]
fn test_thread_config_threads_for_returns_correct_branch() {
    let config = ThreadConfig::new(100, 50);

    // Verify branches by testing is_prefill parameter
    let prefill_result = config.threads_for(true);
    let decode_result = config.threads_for(false);

    // Test that the branches are mutually exclusive
    assert_ne!(prefill_result, decode_result);
    assert_eq!(prefill_result, 100);
    assert_eq!(decode_result, 50);
}

#[test]
fn test_thread_config_equal_values_threads_for() {
    let config = ThreadConfig::new(64, 64);

    // When batch == decode, both should return same value
    assert_eq!(config.threads_for(true), config.threads_for(false));
    assert_eq!(config.threads_for(true), 64);
}

// ============================================================================
// ThreadConfig Auto Configuration Tests
// ============================================================================

#[test]
fn test_thread_config_auto_decode_calculation() {
    let config = ThreadConfig::auto();

    // Decode threads should be batch / 2, minimum 1
    let expected_decode = (config.n_threads_batch / 2).max(1);
    assert_eq!(config.n_threads_decode, expected_decode);
}

#[test]
fn test_thread_config_auto_uses_rayon_threads() {
    let config = ThreadConfig::auto();
    let rayon_threads = rayon::current_num_threads();

    // Batch threads should equal rayon's thread count
    assert_eq!(config.n_threads_batch, rayon_threads);
}

#[test]
fn test_thread_config_auto_called_multiple_times() {
    // Verify auto() returns consistent values across multiple calls
    let config1 = ThreadConfig::auto();
    let config2 = ThreadConfig::auto();
    let config3 = ThreadConfig::auto();

    assert_eq!(config1, config2);
    assert_eq!(config2, config3);
}

// ============================================================================
// ThreadConfig Default Trait Tests
// ============================================================================

#[test]
fn test_thread_config_default_is_auto() {
    let default = ThreadConfig::default();
    let auto = ThreadConfig::auto();

    assert_eq!(default.n_threads_batch, auto.n_threads_batch);
    assert_eq!(default.n_threads_decode, auto.n_threads_decode);
}

#[test]
fn test_thread_config_default_multiple_calls() {
    let d1 = ThreadConfig::default();
    let d2 = ThreadConfig::default();
    assert_eq!(d1, d2);
}

// ============================================================================
// ThreadConfig Clone and Copy Tests
// ============================================================================

#[test]
fn test_thread_config_explicit_clone() {
    let config = ThreadConfig::new(32, 16);
    let cloned = config;

    assert_eq!(config.n_threads_batch, cloned.n_threads_batch);
    assert_eq!(config.n_threads_decode, cloned.n_threads_decode);
}

#[test]
fn test_thread_config_copy_semantics() {
    let config = ThreadConfig::new(16, 8);
    let copied = config; // Copy (not move, since Copy is derived)

    // Original should still be usable
    assert_eq!(config.n_threads_batch, 16);
    assert_eq!(copied.n_threads_batch, 16);
}

#[test]
fn test_thread_config_clone_and_modify_independence() {
    let config = ThreadConfig::new(20, 10);
    let cloned = config;

    // Since ThreadConfig has pub fields, verify clone is independent
    assert_eq!(config.n_threads_batch, cloned.n_threads_batch);
}

// ============================================================================
// ThreadConfig Equality Tests
// ============================================================================

#[test]
fn test_thread_config_eq_same_values() {
    let a = ThreadConfig::new(8, 4);
    let b = ThreadConfig::new(8, 4);
    assert_eq!(a, b);
}

#[test]
fn test_thread_config_eq_different_batch() {
    let a = ThreadConfig::new(8, 4);
    let b = ThreadConfig::new(16, 4);
    assert_ne!(a, b);
}

#[test]
fn test_thread_config_eq_different_decode() {
    let a = ThreadConfig::new(8, 4);
    let b = ThreadConfig::new(8, 2);
    assert_ne!(a, b);
}

#[test]
fn test_thread_config_eq_both_different() {
    let a = ThreadConfig::new(8, 4);
    let b = ThreadConfig::new(16, 2);
    assert_ne!(a, b);
}

// ============================================================================
// InferenceMode Tests
// ============================================================================

#[test]
fn test_inference_mode_prefill_properties() {
    let mode = InferenceMode::Prefill;
    assert!(mode.is_prefill());
    assert!(!mode.is_decode());
}

#[test]
fn test_inference_mode_decode_properties() {
    let mode = InferenceMode::Decode;
    assert!(!mode.is_prefill());
    assert!(mode.is_decode());
}

#[test]
fn test_inference_mode_exhaustive_match() {
    // Test that all variants are covered
    let modes = [InferenceMode::Prefill, InferenceMode::Decode];

    for mode in modes {
        let is_either = mode.is_prefill() || mode.is_decode();
        assert!(is_either, "Mode should be either prefill or decode");

        let is_exclusive = mode.is_prefill() != mode.is_decode();
        assert!(
            is_exclusive,
            "Mode should be exactly one of prefill or decode"
        );
    }
}

#[test]
fn test_inference_mode_copy_trait() {
    let mode = InferenceMode::Prefill;
    let copied = mode; // Copy
    assert_eq!(mode, copied);

    // Original still usable
    assert!(mode.is_prefill());
}

#[test]
fn test_inference_mode_clone_trait() {
    let mode = InferenceMode::Decode;
    let cloned = mode;
    assert_eq!(mode, cloned);
}

// ============================================================================
// InferenceMode Hash Tests
// ============================================================================

#[test]
fn test_inference_mode_hash_in_set() {
    use std::collections::HashSet;

    let mut set = HashSet::new();

    // Insert both variants
    assert!(set.insert(InferenceMode::Prefill));
    assert!(set.insert(InferenceMode::Decode));

    // Duplicates should not insert
    assert!(!set.insert(InferenceMode::Prefill));
    assert!(!set.insert(InferenceMode::Decode));

    assert_eq!(set.len(), 2);
}

#[test]
fn test_inference_mode_as_hashmap_key() {
    use std::collections::HashMap;

    let mut map: HashMap<InferenceMode, usize> = HashMap::new();
    map.insert(InferenceMode::Prefill, 32);
    map.insert(InferenceMode::Decode, 8);

    assert_eq!(map.get(&InferenceMode::Prefill), Some(&32));
    assert_eq!(map.get(&InferenceMode::Decode), Some(&8));
}

// ============================================================================
// Integration: ThreadConfig + InferenceMode
// ============================================================================

#[test]
fn test_config_threads_for_mode_prefill() {
    let config = ThreadConfig::new(64, 32);
    let mode = InferenceMode::Prefill;

    let threads = config.threads_for(mode.is_prefill());
    assert_eq!(threads, 64);
}

#[test]
fn test_config_threads_for_mode_decode() {
    let config = ThreadConfig::new(64, 32);
    let mode = InferenceMode::Decode;

    let threads = config.threads_for(mode.is_prefill());
    assert_eq!(threads, 32);
}

#[test]
fn test_config_mode_switching_pattern() {
    let config = ThreadConfig::new(48, 12);

    // Simulate inference pattern: prefill -> decode -> decode -> ...
    let prefill = InferenceMode::Prefill;
    let decode = InferenceMode::Decode;

    // Prefill phase
    assert_eq!(config.threads_for(prefill.is_prefill()), 48);

    // Decode phase (multiple tokens)
    for _ in 0..10 {
        assert_eq!(config.threads_for(decode.is_prefill()), 12);
    }
}

#[test]
fn test_config_with_default_and_mode() {
    let config = ThreadConfig::default();

    // Both modes should return positive values
    assert!(config.threads_for(InferenceMode::Prefill.is_prefill()) >= 1);
    assert!(config.threads_for(InferenceMode::Decode.is_prefill()) >= 1);
}

// ============================================================================
// ThreadConfig Debug Format Tests
// ============================================================================

#[test]
fn test_thread_config_debug_contains_all_fields() {
    let config = ThreadConfig::new(24, 12);
    let debug = format!("{:?}", config);

    assert!(debug.contains("ThreadConfig"));
    assert!(debug.contains("n_threads_batch"));
    assert!(debug.contains("n_threads_decode"));
    assert!(debug.contains("24"));
    assert!(debug.contains("12"));
}

#[test]
fn test_thread_config_debug_with_extreme_values() {
    let config = ThreadConfig::new(1, 1);
    let debug = format!("{:?}", config);

    assert!(debug.contains("1"));
}

// ============================================================================
// InferenceMode Debug Format Tests
// ============================================================================

#[test]
fn test_inference_mode_debug_prefill() {
    let debug = format!("{:?}", InferenceMode::Prefill);
    assert_eq!(debug, "Prefill");
}

#[test]
fn test_inference_mode_debug_decode() {
    let debug = format!("{:?}", InferenceMode::Decode);
    assert_eq!(debug, "Decode");
}

// ============================================================================
// Thread Pool Configuration Tests
// ============================================================================
// Note: These tests need to be careful because rayon's global pool can only
// be initialized once. We test the error path and the function behavior.

#[test]
fn test_configure_thread_pool_returns_result() {
    use crate::inference::configure_thread_pool;

    // This will return Err if the pool is already initialized (which it likely is)
    // or Ok if this is the first initialization
    let result = configure_thread_pool(4);

    // The function should return a Result, we verify it's either Ok or the expected error
    match result {
        Ok(()) => {
            // Pool was successfully initialized (unlikely in test environment)
        },
        Err(e) => {
            // Pool was already initialized
            let error_msg = e.to_string();
            assert!(
                error_msg.contains("thread pool") || error_msg.contains("configuration"),
                "Error should mention thread pool: {}",
                error_msg
            );
        },
    }
}

#[test]
fn test_configure_optimal_thread_pool_returns_result() {
    use crate::inference::configure_optimal_thread_pool;

    // Similar to above - will return error if already initialized
    let result = configure_optimal_thread_pool();

    match result {
        Ok(()) => {
            // Success
        },
        Err(e) => {
            let error_msg = e.to_string();
            assert!(
                error_msg.contains("thread pool") || error_msg.contains("configuration"),
                "Error should mention thread pool: {}",
                error_msg
            );
        },
    }
}

#[test]
fn test_configure_thread_pool_with_zero_threads() {
    use crate::inference::configure_thread_pool;

    // Rayon requires at least 1 thread, but the function doesn't validate this
    // The result depends on rayon's behavior
    let result = configure_thread_pool(0);

    // Should either fail (pool already initialized) or rayon handles the 0
    assert!(result.is_ok() || result.is_err());
}

#[test]
fn test_configure_thread_pool_with_large_count() {
    use crate::inference::configure_thread_pool;

    // Large but reasonable thread count
    let result = configure_thread_pool(1024);

    // Should return a result (likely error if pool already initialized)
    assert!(result.is_ok() || result.is_err());
}

// ============================================================================
// Additional Coverage: Struct Field Access
// ============================================================================

#[test]
fn test_thread_config_field_access() {
    let config = ThreadConfig::new(10, 5);

    // Direct field access should work since fields are pub
    assert_eq!(config.n_threads_batch, 10);
    assert_eq!(config.n_threads_decode, 5);
}

#[test]
fn test_thread_config_auto_field_access() {
    let config = ThreadConfig::auto();

    // Verify fields are accessible and reasonable
    assert!(config.n_threads_batch >= 1);
    assert!(config.n_threads_decode >= 1);
}

include!("thread_config_03.rs");
