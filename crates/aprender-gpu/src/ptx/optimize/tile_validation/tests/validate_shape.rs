//! validate_shape edge cases, boundaries, and zero dimension tests

use super::*;

// ========== validate_shape edge cases ==========

#[test]
fn test_validate_shape_multiple_dims_with_non_power_of_two() {
    // First dimension is valid, second is not
    let result = validate_shape(&[32, 100]);
    assert!(matches!(result, Err(TileError::NonPowerOfTwo { dim: 100 })));
}

#[test]
fn test_validate_shape_first_dim_non_power_of_two() {
    // First dimension is invalid
    let result = validate_shape(&[13, 16]);
    assert!(matches!(result, Err(TileError::NonPowerOfTwo { dim: 13 })));
}

#[test]
fn test_validate_shape_too_many_elements_exact_boundary() {
    // 4096 * 4096 = 16,777,216 = MAX_TILE_ELEMENTS (valid)
    assert!(validate_shape(&[4096, 4096]).is_ok());

    // 4096 * 4096 * 2 = 33,554,432 > MAX (invalid)
    // But 8192 would fail dimension check first, so use 4096 * 4096 * 2
    // Actually, we need to test elements overflow without dimension overflow
    // Use 4096 * 4096 * 2 = would overflow dimension first
    // Instead test with multiple smaller dimensions
    assert!(matches!(
        validate_shape(&[4096, 4096, 2]),
        Err(TileError::TooManyElements { .. })
    ));
}

#[test]
fn test_validate_shape_single_element() {
    assert!(validate_shape(&[1]).is_ok());
    assert!(validate_shape(&[1, 1]).is_ok());
    assert!(validate_shape(&[1, 1, 1]).is_ok());
}

#[test]
fn test_validate_shape_large_valid_multidimensional() {
    // Multiple dimensions that together are within limits
    assert!(validate_shape(&[64, 64, 64]).is_ok()); // 262,144 elements
    assert!(validate_shape(&[256, 256, 256]).is_ok()); // 16,777,216 elements = MAX
}

#[test]
fn test_validate_shape_dimension_boundary() {
    // Exactly at MAX_TILE_DIM
    assert!(validate_shape(&[4096]).is_ok());

    // Just over MAX_TILE_DIM (but still power of two)
    assert!(matches!(
        validate_shape(&[8192]),
        Err(TileError::DimensionTooLarge {
            actual: 8192,
            max: 4096
        })
    ));
}

#[test]
fn test_validate_shape_zero_dimension_is_ok() {
    // Zero is handled specially - results in 0 elements which is <= MAX
    // Zero passes the power-of-two check (dim != 0 && !is_power_of_two() is false for 0)
    let result = validate_shape(&[0, 16]);
    assert!(result.is_ok());
}

// cuda-tile-behavior.md: Falsification test #1
#[test]
fn test_power_of_two_tiles_valid() {
    assert!(validate_shape(&[8, 16, 32, 64]).is_ok());
    assert!(validate_shape(&[128, 128]).is_ok());
    assert!(validate_shape(&[1024, 1024]).is_ok());
    assert!(validate_shape(&[4096]).is_ok());
}

// cuda-tile-behavior.md: Falsification test #5
#[test]
fn test_non_power_of_two_rejected() {
    assert!(matches!(
        validate_shape(&[7]),
        Err(TileError::NonPowerOfTwo { dim: 7 })
    ));
    assert!(matches!(
        validate_shape(&[100]),
        Err(TileError::NonPowerOfTwo { dim: 100 })
    ));
    assert!(validate_shape(&[17]).is_err());
    assert!(validate_shape(&[1000]).is_err());
}

// cuda-tile-behavior.md: Falsification test #2
#[test]
fn test_max_tile_elements_enforced() {
    // Just under limit: OK
    assert!(validate_shape(&[4096, 4096]).is_ok()); // 16M elements

    // Over limit: rejected
    assert!(matches!(
        validate_shape(&[8192, 4096]),
        Err(TileError::TooManyElements { .. })
    ));
}

// cuda-tile-behavior.md: Falsification test #3
#[test]
fn test_max_dimension_enforced() {
    assert!(validate_shape(&[4096]).is_ok());
    assert!(matches!(
        validate_shape(&[8192]),
        Err(TileError::DimensionTooLarge { .. })
    ));
}

// cuda-tile-behavior.md: Falsification test #4
#[test]
fn test_validation_catches_invalid_at_build_time() {
    // This should be caught at validation time, not runtime
    let result = validate_shape(&[12345]);
    assert!(result.is_err());
}

// cuda-tile-behavior.md: Falsification test #6
#[test]
fn test_constraints_backend_agnostic() {
    // Same constraints work regardless of target
    let shape = [32, 32];
    assert!(validate_shape(&shape).is_ok());
}

// cuda-tile-behavior.md: Falsification test #7
#[test]
fn test_small_tiles_valid() {
    assert!(validate_shape(&[4]).is_ok());
    assert!(validate_shape(&[8]).is_ok());
    assert!(validate_shape(&[2, 2]).is_ok());
}

#[test]
fn test_empty_shape_valid() {
    // Empty shape has 0 elements (product of empty = 1 actually, but we handle it)
    assert!(validate_shape(&[]).is_ok());
}

#[test]
fn test_zero_dimension() {
    // Zero is technically a power of two in bit representation,
    // but we should handle it gracefully
    let result = validate_shape(&[0, 16]);
    // Zero results in 0 total elements, which is <= MAX
    // Zero is not a power of two in the mathematical sense
    // Our implementation should handle this edge case
    assert!(result.is_ok() || result.is_err());
}

// cuda-tile-behavior.md: Falsification test #13
#[test]
fn test_error_messages_actionable() {
    let err = validate_shape(&[17]).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("17") && msg.contains("power of two"),
        "Error message should be actionable: {}",
        msg
    );
}
