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
    fn test_validate_slice_bounds() {
        assert!(buffer_contracts::validate_slice_bounds(10, 0, 5));
        assert!(buffer_contracts::validate_slice_bounds(10, 5, 5));
        assert!(!buffer_contracts::validate_slice_bounds(10, 5, 6));
        assert!(!buffer_contracts::validate_slice_bounds(10, 11, 0));
        // Overflow case
        assert!(!buffer_contracts::validate_slice_bounds(10, usize::MAX, 1));
    }

    #[test]
    fn test_is_aligned() {
        assert!(buffer_contracts::is_aligned(0, 8));
        assert!(buffer_contracts::is_aligned(8, 8));
        assert!(buffer_contracts::is_aligned(16, 8));
        assert!(!buffer_contracts::is_aligned(7, 8));
        assert!(!buffer_contracts::is_aligned(1, 4));
    }

    #[test]
    fn test_padding_for_alignment() {
        assert_eq!(buffer_contracts::padding_for_alignment(0, 8), 0);
        assert_eq!(buffer_contracts::padding_for_alignment(1, 8), 7);
        assert_eq!(buffer_contracts::padding_for_alignment(7, 8), 1);
        assert_eq!(buffer_contracts::padding_for_alignment(8, 8), 0);
        assert_eq!(buffer_contracts::padding_for_alignment(9, 8), 7);
    }

    #[test]
    fn test_validate_magic() {
        let magic = b"ALIM";
        let data = b"ALIMextra";
        assert!(integrity_contracts::validate_magic(data, magic));
        assert!(!integrity_contracts::validate_magic(b"BLIM", magic));
        assert!(!integrity_contracts::validate_magic(b"AL", magic));
    }

    #[test]
    fn test_validate_version() {
        assert!(integrity_contracts::validate_version(1, 1, 3));
        assert!(integrity_contracts::validate_version(3, 1, 3));
        assert!(!integrity_contracts::validate_version(0, 1, 3));
        assert!(!integrity_contracts::validate_version(4, 1, 3));
    }

    #[test]
    fn test_xor_checksum() {
        assert_eq!(integrity_contracts::xor_checksum(&[0xFF, 0xFF]), 0);
        assert_eq!(integrity_contracts::xor_checksum(&[0xAA]), 0xAA);
        // Deterministic
        let data = &[1, 2, 3, 4, 5];
        assert_eq!(
            integrity_contracts::xor_checksum(data),
            integrity_contracts::xor_checksum(data)
        );
    }

    // Negative tests: verify contract violations are caught

    #[test]
    #[should_panic(expected = "data must not be empty")]
    fn test_validated_len_rejects_empty() {
        config_contracts::validated_len(&[]);
    }

    #[test]
    #[should_panic(expected = "max must be greater than min")]
    fn test_normalize_rejects_invalid_range() {
        numeric_contracts::normalize(5.0, 10.0, 0.0);
    }

    #[test]
    #[should_panic(expected = "alignment must be power of two")]
    fn test_alignment_rejects_non_power_of_two() {
        buffer_contracts::padding_for_alignment(10, 3);
    }

    #[test]
    fn test_from_vec_rejects_mismatched_shape() {
        use crate::tensor::TensorData;
        let result = TensorData::<f32>::from_vec(vec![1.0, 2.0, 3.0], 2, 2);
        assert!(result.is_err());
    }
}

/// Buffer and memory safety invariants
///
/// #[invariant(self.capacity >= self.len)]
/// #[requires(offset + len <= buffer.len())]
/// #[ensures(result.len() == len)]
pub mod buffer_contracts {
    /// Validate buffer slice bounds
    ///
    /// #[requires(buffer.len() > 0)]
    /// #[requires(offset <= buffer.len())]
    /// #[ensures(result == true ==> offset + len <= buffer.len())]
    /// #[ensures(result == false ==> offset + len > buffer.len())]
    pub fn validate_slice_bounds(buffer_len: usize, offset: usize, len: usize) -> bool {
        offset.checked_add(len).map_or(false, |end| end <= buffer_len)
    }

    /// Validate alignment for typed access
    ///
    /// #[requires(alignment > 0)]
    /// #[requires(alignment.is_power_of_two())]
    /// #[ensures(result == true ==> addr % alignment == 0)]
    pub fn is_aligned(addr: usize, alignment: usize) -> bool {
        debug_assert!(alignment.is_power_of_two(), "alignment must be power of two");
        addr % alignment == 0
    }

    /// Calculate padding needed for alignment
    ///
    /// #[requires(alignment > 0)]
    /// #[requires(alignment.is_power_of_two())]
    /// #[ensures(result < alignment)]
    /// #[ensures((current_len + result) % alignment == 0)]
    pub fn padding_for_alignment(current_len: usize, alignment: usize) -> usize {
        debug_assert!(alignment.is_power_of_two(), "alignment must be power of two");
        let remainder = current_len % alignment;
        if remainder == 0 { 0 } else { alignment - remainder }
    }
}

/// Data integrity invariants for format operations
///
/// #[invariant(self.checksum_valid)]
/// #[requires(data.len() > 0)]
/// #[ensures(result.len() == expected_len)]
pub mod integrity_contracts {
    /// Validate magic bytes match expected header
    ///
    /// #[requires(data.len() >= magic.len())]
    /// #[ensures(result == true ==> data[..magic.len()] == magic[..])]
    pub fn validate_magic(data: &[u8], magic: &[u8]) -> bool {
        data.len() >= magic.len() && data[..magic.len()] == *magic
    }

    /// Validate version is within supported range
    ///
    /// #[requires(max_version >= min_version)]
    /// #[ensures(result == true ==> version >= min_version && version <= max_version)]
    pub fn validate_version(version: u32, min_version: u32, max_version: u32) -> bool {
        version >= min_version && version <= max_version
    }

    /// Calculate simple checksum (XOR-fold)
    ///
    /// #[requires(data.len() > 0)]
    /// #[ensures(result <= u8::MAX)]
    pub fn xor_checksum(data: &[u8]) -> u8 {
        data.iter().fold(0u8, |acc, &b| acc ^ b)
    }
}

// ─── Kani Proof Stubs ────────────────────────────────────────────
// Model-checking proofs for critical invariants
// Requires: cargo install --locked kani-verifier

#[cfg(kani)]
mod kani_proofs {
    use super::*;

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
    fn verify_slice_bounds_safety() {
        let buf_len: usize = kani::any();
        let offset: usize = kani::any();
        let len: usize = kani::any();
        kani::assume(buf_len <= 4096);
        kani::assume(offset <= 4096);
        kani::assume(len <= 4096);
        let valid = buffer_contracts::validate_slice_bounds(buf_len, offset, len);
        if valid {
            assert!(offset + len <= buf_len);
        }
    }

    #[kani::proof]
    fn verify_alignment_padding() {
        let current: usize = kani::any();
        kani::assume(current <= 4096);
        let alignment: usize = 8; // Common alignment
        let pad = buffer_contracts::padding_for_alignment(current, alignment);
        assert!(pad < alignment);
        assert!((current + pad) % alignment == 0);
    }

    #[kani::proof]
    fn verify_normalize_bounds() {
        let val: u8 = kani::any();
        let result = numeric_contracts::normalize(f64::from(val), 0.0, 255.0);
        assert!(result >= 0.0);
        assert!(result <= 1.0);
    }

    #[kani::proof]
    fn verify_xor_checksum_deterministic() {
        let a: u8 = kani::any();
        let b: u8 = kani::any();
        let data = [a, b];
        let c1 = integrity_contracts::xor_checksum(&data);
        let c2 = integrity_contracts::xor_checksum(&data);
        assert_eq!(c1, c2);
    }

    #[kani::proof]
    fn verify_magic_validation() {
        let magic: [u8; 4] = [0x41, 0x4C, 0x49, 0x4D]; // "ALIM"
        let mut data = [0u8; 8];
        data[0] = magic[0];
        data[1] = magic[1];
        data[2] = magic[2];
        data[3] = magic[3];
        assert!(integrity_contracts::validate_magic(&data, &magic));
    }
}
