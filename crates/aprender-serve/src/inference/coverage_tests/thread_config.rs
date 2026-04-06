//! Part 01: Thread configuration and inference mode tests
//!
//! Additional coverage for thread.rs including:
//! - Edge cases for ThreadConfig
//! - InferenceMode usage patterns
//! - Thread pool configuration error paths

use crate::inference::{InferenceMode, ThreadConfig};

// ============================================================================
// ThreadConfig Additional Coverage Tests
// ============================================================================

#[test]
fn test_thread_config_extremely_large_batch_threads() {
    // Test with very large thread count
    let config = ThreadConfig::new(usize::MAX / 2, 1);
    assert_eq!(config.n_threads_batch, usize::MAX / 2);
    assert_eq!(config.n_threads_decode, 1);
}

#[test]
fn test_thread_config_extremely_large_decode_threads() {
    // Test with very large decode thread count
    let config = ThreadConfig::new(1, usize::MAX / 2);
    assert_eq!(config.n_threads_batch, 1);
    assert_eq!(config.n_threads_decode, usize::MAX / 2);
}

#[test]
fn test_thread_config_new_with_equal_values() {
    let config = ThreadConfig::new(8, 8);
    assert_eq!(config.n_threads_batch, config.n_threads_decode);
    assert_eq!(config.threads_for(true), config.threads_for(false));
}

#[test]
fn test_thread_config_copy_trait() {
    let config = ThreadConfig::new(4, 2);
    let copied: ThreadConfig = config; // Copy
    assert_eq!(config, copied);
    // Verify original still usable (proving it was copied, not moved)
    assert_eq!(config.n_threads_batch, 4);
}

#[test]
fn test_thread_config_threads_for_all_modes() {
    let config = ThreadConfig::new(16, 4);

    // Test both modes in sequence
    let prefill = config.threads_for(true);
    let decode = config.threads_for(false);
    let prefill_again = config.threads_for(true);

    assert_eq!(prefill, 16);
    assert_eq!(decode, 4);
    assert_eq!(prefill_again, prefill); // Consistent results
}

#[test]
fn test_thread_config_partial_eq() {
    let config1 = ThreadConfig::new(8, 4);
    let config2 = ThreadConfig::new(8, 4);
    let config3 = ThreadConfig::new(8, 2);
    let config4 = ThreadConfig::new(4, 4);

    assert!(config1 == config2);
    assert!(config1 != config3);
    assert!(config1 != config4);
}

#[test]
fn test_thread_config_eq_trait() {
    let config1 = ThreadConfig::new(8, 4);
    let config2 = ThreadConfig::new(8, 4);

    // Test Eq reflexivity
    assert!(config1 == config1);
    // Test Eq symmetry
    assert!((config1 == config2) == (config2 == config1));
}

#[test]
fn test_thread_config_debug_format_details() {
    let config = ThreadConfig::new(12, 6);
    let debug_str = format!("{:?}", config);

    // Verify struct name and all fields present
    assert!(debug_str.contains("ThreadConfig"));
    assert!(debug_str.contains("n_threads_batch"));
    assert!(debug_str.contains("n_threads_decode"));
    assert!(debug_str.contains("12"));
    assert!(debug_str.contains("6"));
}

#[test]
fn test_thread_config_clamping_both_zero() {
    let config = ThreadConfig::new(0, 0);

    // Both should be clamped to 1
    assert_eq!(config.n_threads_batch, 1);
    assert_eq!(config.n_threads_decode, 1);
    assert!(config.n_threads_batch >= 1);
    assert!(config.n_threads_decode >= 1);
}

#[test]
fn test_thread_config_clamping_batch_only() {
    let config = ThreadConfig::new(0, 4);

    assert_eq!(config.n_threads_batch, 1);
    assert_eq!(config.n_threads_decode, 4);
}

#[test]
fn test_thread_config_clamping_decode_only() {
    let config = ThreadConfig::new(8, 0);

    assert_eq!(config.n_threads_batch, 8);
    assert_eq!(config.n_threads_decode, 1);
}

// ============================================================================
// InferenceMode Additional Coverage Tests
// ============================================================================

#[test]
fn test_inference_mode_copy_trait() {
    let mode = InferenceMode::Prefill;
    let copied: InferenceMode = mode; // Copy
    assert_eq!(mode, copied);
}

#[test]
fn test_inference_mode_all_variants_is_prefill() {
    assert!(InferenceMode::Prefill.is_prefill());
    assert!(!InferenceMode::Decode.is_prefill());
}

#[test]
fn test_inference_mode_all_variants_is_decode() {
    assert!(!InferenceMode::Prefill.is_decode());
    assert!(InferenceMode::Decode.is_decode());
}

#[test]
fn test_inference_mode_mutually_exclusive() {
    // Each mode should be exactly one of prefill or decode
    for mode in [InferenceMode::Prefill, InferenceMode::Decode] {
        let is_prefill = mode.is_prefill();
        let is_decode = mode.is_decode();

        // Exactly one should be true
        assert!(is_prefill ^ is_decode);
    }
}

#[test]
fn test_inference_mode_hash_different() {
    use std::collections::hash_map::DefaultHasher;

    use std::hash::{Hash, Hasher};

    fn calculate_hash<T: Hash>(t: &T) -> u64 {
        let mut s = DefaultHasher::new();
        t.hash(&mut s);
        s.finish()
    }

    let prefill_hash = calculate_hash(&InferenceMode::Prefill);
    let decode_hash = calculate_hash(&InferenceMode::Decode);

    // Different variants should have different hashes (with very high probability)
    assert_ne!(prefill_hash, decode_hash);
}

#[test]
fn test_inference_mode_hash_consistent() {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    fn calculate_hash<T: Hash>(t: &T) -> u64 {
        let mut s = DefaultHasher::new();
        t.hash(&mut s);
        s.finish()
    }

    // Same variant should have same hash
    let hash1 = calculate_hash(&InferenceMode::Prefill);
    let hash2 = calculate_hash(&InferenceMode::Prefill);
    assert_eq!(hash1, hash2);

    let hash3 = calculate_hash(&InferenceMode::Decode);
    let hash4 = calculate_hash(&InferenceMode::Decode);
    assert_eq!(hash3, hash4);
}

#[test]
fn test_inference_mode_in_hashmap() {
    use std::collections::HashMap;

    let mut map: HashMap<InferenceMode, &str> = HashMap::new();
    map.insert(InferenceMode::Prefill, "prefill value");
    map.insert(InferenceMode::Decode, "decode value");

    assert_eq!(map.get(&InferenceMode::Prefill), Some(&"prefill value"));
    assert_eq!(map.get(&InferenceMode::Decode), Some(&"decode value"));
}

#[test]
fn test_inference_mode_debug_format() {
    let prefill_debug = format!("{:?}", InferenceMode::Prefill);
    let decode_debug = format!("{:?}", InferenceMode::Decode);

    assert_eq!(prefill_debug, "Prefill");
    assert_eq!(decode_debug, "Decode");
}

#[test]
fn test_inference_mode_clone() {
    let original = InferenceMode::Decode;
    let cloned = original;
    assert_eq!(original, cloned);
}

// ============================================================================
// Integration Tests: ThreadConfig + InferenceMode
// ============================================================================

#[test]
fn test_thread_config_with_inference_mode_prefill() {
    let config = ThreadConfig::new(32, 8);
    let mode = InferenceMode::Prefill;

    let threads = config.threads_for(mode.is_prefill());
    assert_eq!(threads, 32);
}

#[test]
fn test_thread_config_with_inference_mode_decode() {
    let config = ThreadConfig::new(32, 8);
    let mode = InferenceMode::Decode;

    let threads = config.threads_for(mode.is_prefill());
    assert_eq!(threads, 8);
}

#[test]
fn test_thread_config_mode_switching() {
    let config = ThreadConfig::new(24, 12);

    // Simulate switching between modes during inference
    let mut mode = InferenceMode::Prefill;
    assert_eq!(config.threads_for(mode.is_prefill()), 24);

    mode = InferenceMode::Decode;
    assert_eq!(config.threads_for(mode.is_prefill()), 12);

    // Can switch back
    mode = InferenceMode::Prefill;
    assert_eq!(config.threads_for(mode.is_prefill()), 24);
}

#[test]
fn test_default_config_reasonable_values() {
    let config = ThreadConfig::default();

    // Auto config should give reasonable values
    assert!(config.n_threads_batch >= 1);
    assert!(config.n_threads_decode >= 1);

    // Batch threads should be >= decode threads (per llama.cpp pattern)
    assert!(config.n_threads_batch >= config.n_threads_decode);
}

#[test]
fn test_auto_config_at_least_half_for_decode() {
    let config = ThreadConfig::auto();

    // Decode should be at least 1
    assert!(config.n_threads_decode >= 1);

    // If batch is >= 2, decode should be at least half
    if config.n_threads_batch >= 2 {
        assert!(config.n_threads_decode * 2 >= config.n_threads_batch);
    }
}

// ============================================================================
// Stress Tests
// ============================================================================

#[test]
fn test_thread_config_many_instances() {
    // Create many instances to ensure no memory issues
    let configs: Vec<ThreadConfig> = (0..1000)
        .map(|i| ThreadConfig::new(i % 64 + 1, (i % 32) + 1))
        .collect();

    assert_eq!(configs.len(), 1000);

    for (i, config) in configs.iter().enumerate() {
        assert_eq!(config.n_threads_batch, (i % 64) + 1);
    }
}

#[test]
fn test_inference_mode_many_instances() {
    use std::collections::HashSet;

    // Create many mode instances
    let modes: Vec<InferenceMode> = (0..1000)
        .map(|i| {
            if i % 2 == 0 {
                InferenceMode::Prefill
            } else {
                InferenceMode::Decode
            }
        })
        .collect();

    let unique: HashSet<_> = modes.iter().collect();
    assert_eq!(unique.len(), 2); // Only 2 unique variants
}
