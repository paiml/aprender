//! FNV-1a Hash Property Tests and Unit Tests

use crate::monitor::inference::{fnv1a_hash, hash_features};
use proptest::prelude::*;

// =============================================================================
// FNV-1a Hash Property Tests
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    #[test]
    fn prop_fnv1a_deterministic(data in prop::collection::vec(any::<u8>(), 0..100)) {
        let hash1 = fnv1a_hash(&data);
        let hash2 = fnv1a_hash(&data);
        prop_assert_eq!(hash1, hash2, "Hash not deterministic");
    }

    #[test]
    fn prop_fnv1a_different_data(
        data1 in prop::collection::vec(any::<u8>(), 1..50),
        data2 in prop::collection::vec(any::<u8>(), 1..50),
    ) {
        if data1 != data2 {
            let hash1 = fnv1a_hash(&data1);
            let hash2 = fnv1a_hash(&data2);
            // Not guaranteed to be different (collisions exist) but very likely
            // This is a weak test - we just check it runs without panic
            let _ = (hash1, hash2);
        }
    }

    #[test]
    fn prop_hash_features_deterministic(features in prop::collection::vec(-100.0f32..100.0, 1..50)) {
        let hash1 = hash_features(&features);
        let hash2 = hash_features(&features);
        prop_assert_eq!(hash1, hash2, "Feature hash not deterministic");
    }
}

// =============================================================================
// Utility Function Unit Tests
// =============================================================================

#[test]
fn test_fnv1a_hash_empty() {
    let hash = fnv1a_hash(&[]);
    assert_eq!(hash, 0xcbf29ce484222325, "Empty input should return FNV offset basis");
}

#[test]
fn test_fnv1a_hash_deterministic() {
    let data = b"hello world";
    let hash1 = fnv1a_hash(data);
    let hash2 = fnv1a_hash(data);
    assert_eq!(hash1, hash2, "Same input should produce same hash");
}

#[test]
fn test_fnv1a_hash_different_inputs() {
    let hash1 = fnv1a_hash(b"hello");
    let hash2 = fnv1a_hash(b"world");
    assert_ne!(hash1, hash2, "Different inputs should produce different hashes");
}

#[test]
fn test_fnv1a_hash_single_byte() {
    let hash = fnv1a_hash(&[0x61]); // 'a'
    assert_ne!(hash, 0xcbf29ce484222325, "Single byte should change hash");
}

#[test]
fn test_hash_features_deterministic() {
    let features = [1.0f32, 2.0, 3.0, 4.0, 5.0];
    let hash1 = hash_features(&features);
    let hash2 = hash_features(&features);
    assert_eq!(hash1, hash2, "Same features should produce same hash");
}

#[test]
fn test_hash_features_different_inputs() {
    let features1 = [1.0f32, 2.0, 3.0];
    let features2 = [1.0f32, 2.0, 4.0];
    let hash1 = hash_features(&features1);
    let hash2 = hash_features(&features2);
    assert_ne!(hash1, hash2, "Different features should produce different hashes");
}

#[test]
fn test_hash_features_empty() {
    let features: [f32; 0] = [];
    let hash = hash_features(&features);
    assert_eq!(hash, 0xcbf29ce484222325, "Empty features should return FNV offset basis");
}
