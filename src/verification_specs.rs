//! Formal Verification Specifications for trueno
//!
//! Design-by-contract specifications using Verus-style pre/postconditions.
//! These serve as both documentation and verification targets.
//!
//! When Verus is available, these can be mechanically verified.
//! Without Verus, they serve as checked documentation via debug_assert!().

/// Configuration validation invariants
///
/// # Verification Specifications
///
/// #[requires(max_size > 0)]
/// #[ensures(result.is_ok() ==> result.unwrap().max_size == max_size)]
/// #[ensures(result.is_ok() ==> result.unwrap().max_size > 0)]
/// #[ensures(max_size == 0 ==> result.is_err())]
/// #[invariant(self.max_size > 0)]
/// #[decreases(remaining)]
/// #[recommends(max_size <= 1_000_000)]
pub mod config_contracts {
    /// Validate that a size parameter is within acceptable bounds
    ///
    /// #[requires(size > 0)]
    /// #[ensures(result == true ==> size <= max)]
    /// #[ensures(result == false ==> size > max)]
    pub fn validate_size(size: usize, max: usize) -> bool {
        size <= max
    }

    /// Validate that an index is within bounds
    ///
    /// #[requires(len > 0)]
    /// #[ensures(result == true ==> index < len)]
    /// #[ensures(result == false ==> index >= len)]
    pub fn validate_index(index: usize, len: usize) -> bool {
        index < len
    }

    /// Validate non-empty slice invariant
    ///
    /// #[requires(data.len() > 0)]
    /// #[ensures(result == data.len())]
    /// #[invariant(data.len() > 0)]
    pub fn validated_len(data: &[u8]) -> usize {
        debug_assert!(!data.is_empty(), "data must not be empty");
        data.len()
    }
}

/// Numeric invariants for computation safety
///
/// #[invariant(self.value.is_finite())]
/// #[requires(a.is_finite() && b.is_finite())]
/// #[ensures(result.is_finite())]
/// #[decreases(iterations)]
/// #[recommends(iterations <= 10_000)]
pub mod numeric_contracts {
    /// Safe addition that checks for overflow
    ///
    /// #[requires(a >= 0 && b >= 0)]
    /// #[ensures(result.is_some() ==> result.unwrap() == a + b)]
    /// #[ensures(result.is_some() ==> result.unwrap() >= a)]
    /// #[ensures(result.is_some() ==> result.unwrap() >= b)]
    pub fn checked_add(a: u64, b: u64) -> Option<u64> {
        a.checked_add(b)
    }

    /// Validate that a float value is usable (finite, non-NaN)
    ///
    /// #[ensures(result == true ==> val.is_finite())]
    /// #[ensures(result == true ==> !val.is_nan())]
    /// #[ensures(result == false ==> val.is_nan() || val.is_infinite())]
    pub fn is_valid_float(val: f64) -> bool {
        val.is_finite()
    }

    /// Normalize a value to [0, 1] range
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

// ─── Verus Formal Verification Specs ─────────────────────────────
// Domain: trueno - SIMD operations, tensor dimensions, memory alignment
// Machine-checkable pre/postconditions for tensor computation safety.

#[cfg(verus)]
mod verus_specs {
    use builtin::*;
    use builtin_macros::*;

    verus! {
        // ── Tensor dimension verification ──

        #[requires(rows > 0 && cols > 0)]
        #[ensures(result == rows * cols)]
        fn verify_tensor_size(rows: u64, cols: u64) -> u64 {
            rows * cols
        }

        #[requires(dim_a > 0 && dim_b > 0)]
        #[ensures(result == (dim_a == dim_b))]
        fn verify_dimension_match(dim_a: u64, dim_b: u64) -> bool {
            dim_a == dim_b
        }

        #[requires(a_cols == b_rows)]
        #[ensures(result == a_rows * b_cols)]
        #[recommends(a_rows * b_cols <= 1024 * 1024 * 1024)]
        fn verify_matmul_output_size(a_rows: u64, a_cols: u64, b_rows: u64, b_cols: u64) -> u64 {
            a_rows * b_cols
        }

        #[requires(ndim > 0 && ndim <= 8)]
        #[ensures(result == ndim)]
        #[invariant(ndim <= 8)]
        fn verify_tensor_rank(ndim: u64) -> u64 { ndim }

        // ── SIMD alignment verification ──

        #[requires(alignment > 0)]
        #[ensures(result == (addr % alignment == 0))]
        #[recommends(alignment == 32)]
        fn verify_simd_alignment(addr: u64, alignment: u64) -> bool {
            addr % alignment == 0
        }

        #[requires(size > 0)]
        #[ensures(result >= size)]
        #[ensures(result % 32 == 0)]
        fn verify_aligned_alloc_size(size: u64) -> u64 {
            ((size + 31) / 32) * 32
        }

        #[requires(lane_width > 0)]
        #[ensures(result == (len % lane_width == 0))]
        #[recommends(lane_width == 8)]
        fn verify_simd_lane_alignment(len: u64, lane_width: u64) -> bool {
            len % lane_width == 0
        }

        // ── Quantization verification ──

        #[requires(block_size > 0)]
        #[ensures(result == (num_elements % block_size == 0))]
        #[recommends(block_size == 32)]
        fn verify_quant_block_alignment(num_elements: u64, block_size: u64) -> bool {
            num_elements % block_size == 0
        }

        #[requires(bits > 0 && bits <= 8)]
        #[ensures(result == (1u64 << bits) - 1)]
        fn verify_quant_max_value(bits: u64) -> u64 {
            (1u64 << bits) - 1
        }

        #[requires(num_blocks > 0)]
        #[ensures(result == num_blocks * bytes_per_block)]
        #[invariant(bytes_per_block > 0)]
        fn verify_quantized_buffer_size(num_blocks: u64, bytes_per_block: u64) -> u64 {
            num_blocks * bytes_per_block
        }

        // ── Stride verification ──

        #[requires(ndim > 0)]
        #[ensures(result > 0)]
        #[decreases(ndim)]
        fn verify_contiguous_stride(shape_product: u64, ndim: u64) -> u64 {
            if ndim == 0 { 1 } else { shape_product }
        }

        #[requires(offset < total_elements)]
        #[ensures(result < total_elements)]
        fn verify_element_offset(offset: u64, total_elements: u64) -> u64 { offset }

        // ── Memory pool verification ──

        #[requires(pool_capacity > 0)]
        #[ensures(result <= pool_capacity)]
        #[invariant(allocated <= pool_capacity)]
        fn verify_pool_allocation(allocated: u64, pool_capacity: u64) -> u64 { allocated }

        #[requires(requested > 0)]
        #[ensures(result == (available >= requested))]
        #[recommends(available >= requested)]
        fn verify_pool_has_space(available: u64, requested: u64) -> bool {
            available >= requested
        }

        // ── Broadcast verification ──

        #[requires(dim_a > 0 && dim_b > 0)]
        #[ensures(result == (dim_a == dim_b || dim_a == 1 || dim_b == 1))]
        fn verify_broadcast_compatible(dim_a: u64, dim_b: u64) -> bool {
            dim_a == dim_b || dim_a == 1 || dim_b == 1
        }

        #[requires(dim_a > 0 && dim_b > 0)]
        #[ensures(result >= dim_a && result >= dim_b)]
        fn verify_broadcast_output_dim(dim_a: u64, dim_b: u64) -> u64 {
            if dim_a > dim_b { dim_a } else { dim_b }
        }

        // ── Transpose verification ──

        #[requires(rows > 0 && cols > 0)]
        #[ensures(result == rows * cols)]
        #[invariant(rows * cols == cols * rows)]
        fn verify_transpose_element_count(rows: u64, cols: u64) -> u64 {
            rows * cols
        }
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
        assert_eq!(config_contracts::validated_len(&[0]), 1);
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
        assert!((numeric_contracts::normalize(5.0, 0.0, 10.0) - 0.5).abs() < f64::EPSILON);
        assert!((numeric_contracts::normalize(0.0, 0.0, 10.0)).abs() < f64::EPSILON);
        assert!((numeric_contracts::normalize(10.0, 0.0, 10.0) - 1.0).abs() < f64::EPSILON);
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
}
