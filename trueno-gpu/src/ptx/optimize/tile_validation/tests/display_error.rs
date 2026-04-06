//! TileError Display, Error trait, and GpuError conversion tests

use super::*;

// ========== Display and Error trait tests ==========

#[test]
fn test_tile_error_display_too_many_elements() {
    let err = TileError::TooManyElements { actual: 20_000_000, max: 16_777_216 };
    let msg = err.to_string();
    assert!(msg.contains("20000000"), "Should contain actual count: {}", msg);
    assert!(msg.contains("16777216"), "Should contain max count: {}", msg);
    assert!(msg.contains("too many elements"), "Should describe the error: {}", msg);
}

#[test]
fn test_tile_error_display_dimension_too_large() {
    let err = TileError::DimensionTooLarge { actual: 8192, max: 4096 };
    let msg = err.to_string();
    assert!(msg.contains("8192"), "Should contain actual dimension: {}", msg);
    assert!(msg.contains("4096"), "Should contain max dimension: {}", msg);
    assert!(msg.contains("exceeds"), "Should describe the error: {}", msg);
}

#[test]
fn test_tile_error_display_invalid_wmma_shape() {
    let err = TileError::InvalidWmmaShape { shape: "m64n64k64".to_string() };
    let msg = err.to_string();
    assert!(msg.contains("m64n64k64"), "Should contain shape: {}", msg);
    assert!(msg.contains("Invalid WMMA"), "Should describe the error: {}", msg);
}

#[test]
fn test_tile_error_display_non_power_of_two() {
    let err = TileError::NonPowerOfTwo { dim: 123 };
    let msg = err.to_string();
    assert!(msg.contains("123"), "Should contain dimension: {}", msg);
    assert!(msg.contains("power of two"), "Should describe the error: {}", msg);
}

#[test]
fn test_tile_error_implements_std_error() {
    let err = TileError::NonPowerOfTwo { dim: 42 };
    // Verify Error trait is implemented (source() returns None by default)
    let std_err: &dyn std::error::Error = &err;
    assert!(std_err.source().is_none());
}

#[test]
fn test_tile_error_to_gpu_error_conversion() {
    let tile_err = TileError::TooManyElements { actual: 100, max: 50 };
    let gpu_err: GpuError = tile_err.clone().into();

    // Verify conversion produces InvalidParameter variant with the error message
    match gpu_err {
        GpuError::InvalidParameter(msg) => {
            assert!(msg.contains("100"), "Should contain actual: {}", msg);
            assert!(msg.contains("50"), "Should contain max: {}", msg);
        }
        _ => panic!("Expected InvalidParameter variant"),
    }
}

#[test]
fn test_tile_error_conversion_non_power_of_two() {
    let tile_err = TileError::NonPowerOfTwo { dim: 37 };
    let gpu_err: GpuError = tile_err.into();

    match gpu_err {
        GpuError::InvalidParameter(msg) => {
            assert!(msg.contains("37"));
            assert!(msg.contains("power of two"));
        }
        _ => panic!("Expected InvalidParameter"),
    }
}

#[test]
fn test_tile_error_conversion_dimension_too_large() {
    let tile_err = TileError::DimensionTooLarge { actual: 10000, max: 4096 };
    let gpu_err: GpuError = tile_err.into();

    match gpu_err {
        GpuError::InvalidParameter(msg) => {
            assert!(msg.contains("10000"));
            assert!(msg.contains("4096"));
        }
        _ => panic!("Expected InvalidParameter"),
    }
}

#[test]
fn test_tile_error_conversion_invalid_wmma() {
    let tile_err = TileError::InvalidWmmaShape { shape: "m99n99k99".to_string() };
    let gpu_err: GpuError = tile_err.into();

    match gpu_err {
        GpuError::InvalidParameter(msg) => {
            assert!(msg.contains("m99n99k99"));
        }
        _ => panic!("Expected InvalidParameter"),
    }
}
