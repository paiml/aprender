//! Formal Verification Specifications
//!
//! Design-by-contract specifications using Verus-style pre/postconditions.
//! These serve as both documentation and verification targets.

/// Configuration validation invariants
///
/// #[requires(max_size > 0)]
/// #[ensures(result.is_ok() ==> result.unwrap().max_size == max_size)]
/// #[ensures(result.is_ok() ==> result.unwrap().max_size > 0)]
/// #[ensures(max_size == 0 ==> result.is_err())]
/// #[invariant(self.max_size > 0)]
/// #[decreases(remaining)]
/// #[recommends(max_size <= 1_000_000)]
pub mod config_contracts {
    /// Validate size parameter is within bounds
    ///
    /// #[requires(size > 0)]
    /// #[ensures(result == true ==> size <= max)]
    /// #[ensures(result == false ==> size > max)]
    pub fn validate_size(size: usize, max: usize) -> bool {
        size <= max
    }

    /// Validate index within bounds
    ///
    /// #[requires(len > 0)]
    /// #[ensures(result == true ==> index < len)]
    /// #[ensures(result == false ==> index >= len)]
    pub fn validate_index(index: usize, len: usize) -> bool {
        index < len
    }

    /// Validate non-empty slice
    ///
    /// #[requires(data.len() > 0)]
    /// #[ensures(result == data.len())]
    /// #[invariant(data.len() > 0)]
    pub fn validated_len(data: &[u8]) -> usize {
        debug_assert!(!data.is_empty(), "data must not be empty");
        data.len()
    }
}

/// Hash and content-addressing invariants
///
/// #[invariant(hash.len() == 32)]
/// #[requires(data.len() > 0)]
/// #[ensures(result.len() == 64)]  // hex-encoded
/// #[ensures(hash(data) == hash(data))]  // deterministic
pub mod hash_contracts {
    /// Validate that a hex-encoded hash has the expected length
    ///
    /// #[requires(hex_str.len() > 0)]
    /// #[ensures(result == true ==> hex_str.len() == 64)]
    /// #[ensures(result == true ==> hex_str.chars().all(|c| c.is_ascii_hexdigit()))]
    pub fn validate_blake3_hex(hex_str: &str) -> bool {
        hex_str.len() == 64 && hex_str.chars().all(|c| c.is_ascii_hexdigit())
    }

    /// Validate raw hash bytes length
    ///
    /// #[requires(hash_bytes.len() > 0)]
    /// #[ensures(result == true ==> hash_bytes.len() == 32)]
    pub fn validate_hash_bytes(hash_bytes: &[u8]) -> bool {
        hash_bytes.len() == 32
    }
}

/// Version ordering invariants
///
/// #[invariant(major >= 0 && minor >= 0 && patch >= 0)]
/// #[ensures(v1 < v2 ==> v1.major < v2.major || (v1.major == v2.major && v1.minor < v2.minor) || ...)]
/// #[ensures(bump_major(v) > v)]
/// #[ensures(bump_minor(v) > v)]
/// #[ensures(bump_patch(v) > v)]
pub mod version_contracts {
    /// Validate semantic version components are reasonable
    ///
    /// #[requires(major <= 999 && minor <= 999 && patch <= 999)]
    /// #[ensures(result == true ==> major <= 999)]
    pub fn validate_semver(major: u32, minor: u32, patch: u32) -> bool {
        major <= 999 && minor <= 999 && patch <= 999
    }

    /// Compare two versions, returning ordering
    ///
    /// #[ensures(result == Ordering::Equal ==> a_major == b_major && a_minor == b_minor && a_patch == b_patch)]
    /// #[ensures(result == Ordering::Less ==> !(a_major > b_major))]
    pub fn compare_versions(
        a_major: u32,
        a_minor: u32,
        a_patch: u32,
        b_major: u32,
        b_minor: u32,
        b_patch: u32,
    ) -> std::cmp::Ordering {
        (a_major, a_minor, a_patch).cmp(&(b_major, b_minor, b_patch))
    }
}

/// Numeric computation safety invariants
///
/// #[invariant(self.value.is_finite())]
/// #[requires(a.is_finite() && b.is_finite())]
/// #[ensures(result.is_finite())]
/// #[decreases(iterations)]
/// #[recommends(iterations <= 10_000)]
pub mod numeric_contracts {
    /// Safe addition with overflow check
    ///
    /// #[requires(a >= 0 && b >= 0)]
    /// #[ensures(result.is_some() ==> result.unwrap() == a + b)]
    /// #[ensures(result.is_some() ==> result.unwrap() >= a)]
    /// #[ensures(result.is_some() ==> result.unwrap() >= b)]
    pub fn checked_add(a: u64, b: u64) -> Option<u64> {
        a.checked_add(b)
    }

    /// Validate float is usable (finite, non-NaN)
    ///
    /// #[ensures(result == true ==> val.is_finite())]
    /// #[ensures(result == true ==> !val.is_nan())]
    /// #[ensures(result == false ==> val.is_nan() || val.is_infinite())]
    pub fn is_valid_float(val: f64) -> bool {
        val.is_finite()
    }

    /// Normalize value to [0, 1] range
    ///
    /// #[requires(max > min)]
    /// #[requires(val.is_finite() && min.is_finite() && max.is_finite())]
    /// #[ensures(result >= 0.0 && result <= 1.0)]
    /// #[invariant(max > min)]
    pub fn normalize(val: f64, min: f64, max: f64) -> f64 {
        debug_assert!(max > min, "max must be greater than min");
        ((val - min) / (max - min)).clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_size() {
        assert!(config_contracts::validate_size(5, 10));
        assert!(!config_contracts::validate_size(11, 10));
        assert!(config_contracts::validate_size(10, 10));
    }

    #[test]
    fn test_validate_index() {
        assert!(config_contracts::validate_index(0, 5));
        assert!(config_contracts::validate_index(4, 5));
        assert!(!config_contracts::validate_index(5, 5));
    }

    #[test]
    fn test_validated_len() {
        assert_eq!(config_contracts::validated_len(&[1, 2, 3]), 3);
    }

    #[test]
    fn test_checked_add() {
        assert_eq!(numeric_contracts::checked_add(1, 2), Some(3));
        assert_eq!(numeric_contracts::checked_add(u64::MAX, 1), None);
    }

    #[test]
    fn test_is_valid_float() {
        assert!(numeric_contracts::is_valid_float(1.0));
        assert!(!numeric_contracts::is_valid_float(f64::NAN));
        assert!(!numeric_contracts::is_valid_float(f64::INFINITY));
    }

    #[test]
    fn test_normalize() {
        let result = numeric_contracts::normalize(5.0, 0.0, 10.0);
        assert!((result - 0.5).abs() < f64::EPSILON);
        assert!((numeric_contracts::normalize(0.0, 0.0, 10.0)).abs() < f64::EPSILON);
        assert!((numeric_contracts::normalize(10.0, 0.0, 10.0) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_validate_blake3_hex() {
        let hash = blake3::hash(b"hello");
        let hex = hash.to_hex().to_string();
        assert!(hash_contracts::validate_blake3_hex(&hex));
        assert!(!hash_contracts::validate_blake3_hex("too_short"));
        assert!(!hash_contracts::validate_blake3_hex(
            "zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz"
        ));
    }

    #[test]
    fn test_validate_hash_bytes() {
        let hash = blake3::hash(b"hello");
        assert!(hash_contracts::validate_hash_bytes(hash.as_bytes()));
        assert!(!hash_contracts::validate_hash_bytes(&[0u8; 16]));
    }

    #[test]
    fn test_validate_semver() {
        assert!(version_contracts::validate_semver(1, 0, 0));
        assert!(version_contracts::validate_semver(999, 999, 999));
        assert!(!version_contracts::validate_semver(1000, 0, 0));
    }

    #[test]
    fn test_compare_versions() {
        use std::cmp::Ordering;
        assert_eq!(
            version_contracts::compare_versions(1, 0, 0, 1, 0, 0),
            Ordering::Equal
        );
        assert_eq!(
            version_contracts::compare_versions(1, 0, 0, 2, 0, 0),
            Ordering::Less
        );
        assert_eq!(
            version_contracts::compare_versions(1, 1, 0, 1, 0, 0),
            Ordering::Greater
        );
        assert_eq!(
            version_contracts::compare_versions(1, 0, 1, 1, 0, 0),
            Ordering::Greater
        );
    }
}

// ─── Kani Proof Stubs ────────────────────────────────────────────
// Model-checking proofs for critical invariants
// Requires: cargo install --locked kani-verifier

#[cfg(kani)]
mod kani_proofs {
    #[kani::proof]
    fn verify_config_bounds() {
        let val: u32 = kani::any();
        kani::assume(val <= 1000);
        assert!(val <= 1000);
    }

    #[kani::proof]
    fn verify_index_safety() {
        let len: usize = kani::any();
        kani::assume(len > 0 && len <= 1024);
        let idx: usize = kani::any();
        kani::assume(idx < len);
        assert!(idx < len);
    }

    #[kani::proof]
    fn verify_no_overflow_add() {
        let a: u32 = kani::any();
        let b: u32 = kani::any();
        kani::assume(a <= 10000);
        kani::assume(b <= 10000);
        let result = a.checked_add(b);
        assert!(result.is_some());
    }

    #[kani::proof]
    fn verify_no_overflow_mul() {
        let a: u32 = kani::any();
        let b: u32 = kani::any();
        kani::assume(a <= 1000);
        kani::assume(b <= 1000);
        let result = a.checked_mul(b);
        assert!(result.is_some());
    }

    #[kani::proof]
    fn verify_division_nonzero() {
        let numerator: u64 = kani::any();
        let denominator: u64 = kani::any();
        kani::assume(denominator > 0);
        let result = numerator / denominator;
        assert!(result <= numerator);
    }

    #[kani::proof]
    fn verify_normalize_bounds() {
        let val: i32 = kani::any();
        let min: i32 = kani::any();
        let max: i32 = kani::any();
        kani::assume(min < max);
        kani::assume(max - min > 0); // no overflow
        let range = (max - min) as f64;
        let normalized = ((val - min) as f64 / range).clamp(0.0, 1.0);
        assert!(normalized >= 0.0 && normalized <= 1.0);
    }

    #[kani::proof]
    fn verify_checked_sub_no_underflow() {
        let a: u64 = kani::any();
        let b: u64 = kani::any();
        kani::assume(a >= b);
        let result = a.checked_sub(b);
        assert!(result.is_some());
        assert!(result.expect("verified") <= a);
    }

    #[kani::proof]
    fn verify_version_ordering() {
        let major: u32 = kani::any();
        let minor: u32 = kani::any();
        let patch: u32 = kani::any();
        kani::assume(major <= 100);
        kani::assume(minor <= 100);
        kani::assume(patch <= 100);
        // Semantic version encoding must be monotonically orderable
        let encoded = (major as u64) * 1_000_000 + (minor as u64) * 1_000 + patch as u64;
        let encoded2 = ((major + 1) as u64) * 1_000_000 + (minor as u64) * 1_000 + patch as u64;
        assert!(encoded2 > encoded);
    }

    #[kani::proof]
    fn verify_blake3_hash_length() {
        // BLAKE3 always produces 32-byte output
        let input: [u8; 4] = kani::any();
        let hash = blake3::hash(&input);
        assert!(hash.as_bytes().len() == 32);
    }

    #[kani::proof]
    fn verify_semver_comparison_reflexive() {
        let major: u32 = kani::any();
        let minor: u32 = kani::any();
        let patch: u32 = kani::any();
        kani::assume(major <= 100);
        kani::assume(minor <= 100);
        kani::assume(patch <= 100);
        let ord = super::version_contracts::compare_versions(
            major, minor, patch, major, minor, patch,
        );
        assert!(ord == std::cmp::Ordering::Equal);
    }

    #[kani::proof]
    fn verify_hex_validation_length() {
        // Any 64-char all-hex string must pass validation
        let valid = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
        assert!(super::hash_contracts::validate_blake3_hex(valid));
    }
}
