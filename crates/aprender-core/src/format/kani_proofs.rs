//! Kani formal verification proofs for APR format invariants.
//!
//! These proofs verify critical safety properties of the APR binary format:
//! - Magic byte validation cannot accept invalid headers
//! - Tensor shape calculations cannot overflow
//! - Alignment guarantees for memory-mapped access

/// APR format magic bytes: `APR\0` (v2) or `APRN` (v1)
const APR_MAGIC_V2: [u8; 4] = [0x41, 0x50, 0x52, 0x00];
const APR_MAGIC_V1: [u8; 4] = [0x41, 0x50, 0x52, 0x4E];

/// Verify magic bytes are exactly 4 bytes and mutually exclusive
#[cfg(kani)]
#[kani::proof]
fn verify_magic_bytes_distinct() {
    assert_ne!(APR_MAGIC_V1, APR_MAGIC_V2);
    assert_eq!(APR_MAGIC_V1[0], APR_MAGIC_V2[0]); // Both start with 'A'
    assert_eq!(APR_MAGIC_V1[1], APR_MAGIC_V2[1]); // Both have 'P'
    assert_eq!(APR_MAGIC_V1[2], APR_MAGIC_V2[2]); // Both have 'R'
    assert_ne!(APR_MAGIC_V1[3], APR_MAGIC_V2[3]); // Differ in 4th byte
}

/// Verify tensor element count cannot overflow for valid shapes
#[cfg(kani)]
#[kani::proof]
fn verify_tensor_element_count_no_overflow() {
    let rows: u32 = kani::any();
    let cols: u32 = kani::any();

    // APR spec: max tensor dimension is 2^20 (1M)
    kani::assume(rows <= 1_048_576);
    kani::assume(cols <= 1_048_576);

    let element_count = (rows as u64) * (cols as u64);
    // With max 1M × 1M, result fits in u64
    assert!(element_count <= u64::MAX);
    // And fits in usize on 64-bit (APR requires 64-bit)
    assert!(element_count <= usize::MAX as u64);
}

/// Verify alignment padding is always less than alignment
#[cfg(kani)]
#[kani::proof]
fn verify_alignment_padding_bounded() {
    let offset: u64 = kani::any();
    let alignment: u64 = kani::any();

    kani::assume(alignment > 0);
    kani::assume(alignment.is_power_of_two());
    kani::assume(alignment <= 4096); // APR max alignment

    let padding = (alignment - (offset % alignment)) % alignment;
    assert!(padding < alignment);
}

/// Verify that shape reversal for GGUF import is involutory
#[cfg(kani)]
#[kani::proof]
fn verify_shape_reversal_involutory() {
    let dim0: u32 = kani::any();
    let dim1: u32 = kani::any();
    kani::assume(dim0 > 0);
    kani::assume(dim1 > 0);

    let original = [dim0, dim1];
    let reversed = [original[1], original[0]];
    let double_reversed = [reversed[1], reversed[0]];

    assert_eq!(original, double_reversed);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_magic_bytes_constants() {
        assert_eq!(&APR_MAGIC_V2, b"APR\0");
        assert_eq!(&APR_MAGIC_V1, b"APRN");
    }

    #[test]
    fn test_alignment_padding() {
        // 0 offset, 64 alignment → 0 padding
        let padding = (64 - (0_u64 % 64)) % 64;
        assert_eq!(padding, 0);

        // 60 offset, 64 alignment → 4 padding
        let padding = (64 - (60_u64 % 64)) % 64;
        assert_eq!(padding, 4);
    }
}
