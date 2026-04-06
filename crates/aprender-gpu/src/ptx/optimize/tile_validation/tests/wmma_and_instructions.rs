//! WMMA shape validation, instruction validation, TileError Clone/PartialEq/Debug, and constants

use super::*;

// ========== validate_wmma_shape tests ==========

#[test]
fn test_wmma_shape_validation_all_valid() {
    // Explicitly test all three valid shapes
    assert!(validate_wmma_shape(&WmmaShape::M16N16K16).is_ok());
    assert!(validate_wmma_shape(&WmmaShape::M8N32K16).is_ok());
    assert!(validate_wmma_shape(&WmmaShape::M32N8K16).is_ok());
}

#[test]
fn test_wmma_shape_validation_invalid_combinations() {
    // Various invalid combinations
    let cases = [
        WmmaShape { m: 8, n: 8, k: 8 },    // All 8s
        WmmaShape { m: 32, n: 32, k: 32 }, // All 32s
        WmmaShape { m: 16, n: 32, k: 16 }, // Wrong combination
        WmmaShape { m: 8, n: 16, k: 16 },  // Wrong combination
        WmmaShape { m: 1, n: 1, k: 1 },    // Minimal
        WmmaShape { m: 64, n: 64, k: 64 }, // Too large
    ];

    for shape in cases {
        assert!(
            validate_wmma_shape(&shape).is_err(),
            "Shape m{}n{}k{} should be invalid",
            shape.m,
            shape.n,
            shape.k
        );
    }
}

#[test]
fn test_wmma_invalid_error_message_format() {
    let shape = WmmaShape { m: 24, n: 24, k: 8 };
    let result = validate_wmma_shape(&shape);

    match result {
        Err(TileError::InvalidWmmaShape { shape: s }) => {
            assert_eq!(s, "m24n24k8");
        }
        _ => panic!("Expected InvalidWmmaShape error"),
    }
}

// ========== validate() instruction tests ==========

#[test]
fn test_validate_wmma_load_a() {
    let instructions = vec![PtxInstruction::new(PtxOp::WmmaLoadA, crate::ptx::types::PtxType::F16)];
    assert!(validate(&instructions).is_ok());
}

#[test]
fn test_validate_wmma_load_b() {
    let instructions = vec![PtxInstruction::new(PtxOp::WmmaLoadB, crate::ptx::types::PtxType::F16)];
    assert!(validate(&instructions).is_ok());
}

#[test]
fn test_validate_wmma_load_c() {
    let instructions = vec![PtxInstruction::new(PtxOp::WmmaLoadC, crate::ptx::types::PtxType::F32)];
    assert!(validate(&instructions).is_ok());
}

#[test]
fn test_validate_wmma_store_d() {
    let instructions =
        vec![PtxInstruction::new(PtxOp::WmmaStoreD, crate::ptx::types::PtxType::F32)];
    assert!(validate(&instructions).is_ok());
}

#[test]
fn test_validate_mixed_instructions_with_wmma() {
    let instructions = vec![
        PtxInstruction::new(PtxOp::Add, crate::ptx::types::PtxType::F32),
        PtxInstruction::new(PtxOp::WmmaLoadA, crate::ptx::types::PtxType::F16),
        PtxInstruction::new(PtxOp::Mul, crate::ptx::types::PtxType::F32),
        PtxInstruction::new(PtxOp::WmmaLoadB, crate::ptx::types::PtxType::F16),
        PtxInstruction::new(PtxOp::WmmaLoadC, crate::ptx::types::PtxType::F32),
        PtxInstruction::new(PtxOp::WmmaMma, crate::ptx::types::PtxType::F32),
        PtxInstruction::new(PtxOp::WmmaStoreD, crate::ptx::types::PtxType::F32),
        PtxInstruction::new(PtxOp::Sub, crate::ptx::types::PtxType::F32),
    ];
    assert!(validate(&instructions).is_ok());
}

#[test]
fn test_validate_all_wmma_ops_in_sequence() {
    let instructions = vec![
        PtxInstruction::new(PtxOp::WmmaLoadA, crate::ptx::types::PtxType::F16),
        PtxInstruction::new(PtxOp::WmmaLoadB, crate::ptx::types::PtxType::F16),
        PtxInstruction::new(PtxOp::WmmaLoadC, crate::ptx::types::PtxType::F32),
        PtxInstruction::new(PtxOp::WmmaMma, crate::ptx::types::PtxType::F32),
        PtxInstruction::new(PtxOp::WmmaStoreD, crate::ptx::types::PtxType::F32),
    ];
    assert!(validate(&instructions).is_ok());
}

// ========== TileError Clone and PartialEq tests ==========

#[test]
fn test_tile_error_clone() {
    let err1 = TileError::TooManyElements { actual: 1000, max: 500 };
    let err2 = err1.clone();
    assert_eq!(err1, err2);
}

#[test]
fn test_tile_error_partial_eq() {
    let err1 = TileError::NonPowerOfTwo { dim: 17 };
    let err2 = TileError::NonPowerOfTwo { dim: 17 };
    let err3 = TileError::NonPowerOfTwo { dim: 19 };

    assert_eq!(err1, err2);
    assert_ne!(err1, err3);
}

#[test]
fn test_tile_error_debug() {
    let err = TileError::DimensionTooLarge { actual: 5000, max: 4096 };
    let debug_str = format!("{:?}", err);
    assert!(debug_str.contains("DimensionTooLarge"));
    assert!(debug_str.contains("5000"));
    assert!(debug_str.contains("4096"));
}

// ========== Constants verification ==========

#[test]
fn test_constants_values() {
    assert_eq!(MAX_TILE_ELEMENTS, 16_777_216);
    assert_eq!(MAX_TILE_DIM, 4096);
}

// WMMA shape tests (additional)
#[test]
fn test_wmma_valid_shapes() {
    assert!(validate_wmma_shape(&WmmaShape::M16N16K16).is_ok());
    assert!(validate_wmma_shape(&WmmaShape::M8N32K16).is_ok());
    assert!(validate_wmma_shape(&WmmaShape::M32N8K16).is_ok());
}

#[test]
fn test_wmma_invalid_shapes() {
    let invalid = WmmaShape { m: 32, n: 32, k: 16 };
    assert!(validate_wmma_shape(&invalid).is_err());
}

// Integration test with instruction validation
#[test]
fn test_validate_instructions_empty() {
    assert!(validate(&[]).is_ok());
}

#[test]
fn test_validate_instructions_no_wmma() {
    let instructions = vec![
        PtxInstruction::new(PtxOp::Add, crate::ptx::types::PtxType::F32),
        PtxInstruction::new(PtxOp::Mul, crate::ptx::types::PtxType::F32),
    ];
    assert!(validate(&instructions).is_ok());
}

#[test]
fn test_validate_instructions_with_wmma() {
    let instructions = vec![PtxInstruction::new(PtxOp::WmmaMma, crate::ptx::types::PtxType::F32)];
    // Should validate the default WMMA shape
    assert!(validate(&instructions).is_ok());
}
