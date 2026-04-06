use super::super::super::super::*;

// ========================
// Coverage Tests (C001-C005): Assertions, BrickId, BrickCategory
// ========================

/// C001: ComputeAssertion::equiv creates equivalence with default tolerance
#[test]
fn test_c001_compute_assertion_equiv() {
    let assertion = ComputeAssertion::equiv(Backend::Scalar);
    if let ComputeAssertion::Equivalence { baseline, tolerance } = assertion {
        assert_eq!(baseline, Backend::Scalar);
        assert!((tolerance - 1e-5).abs() < 1e-10);
    } else {
        panic!("Expected Equivalence assertion");
    }
}

/// C002: assert_equiv builder method
#[test]
fn test_c002_compute_brick_assert_equiv() {
    let brick = ComputeBrick::new(AddOp::new(4)).assert_equiv(Backend::Scalar);
    // Verify assertion was added
    assert!(!brick.get_assertions().is_empty());
}

/// C003: BrickId Display trait
#[test]
fn test_c003_brick_id_display() {
    let id = BrickId::QkvProjection;
    let display = format!("{}", id);
    assert_eq!(display, "QkvProjection");

    let id2 = BrickId::RmsNorm;
    assert_eq!(format!("{}", id2), "RmsNorm");
}

/// C004: BrickCategory::name() all variants
#[test]
fn test_c004_brick_category_name() {
    assert_eq!(BrickCategory::Norm.name(), "Norm");
    assert_eq!(BrickCategory::Attention.name(), "Attention");
    assert_eq!(BrickCategory::Ffn.name(), "FFN");
    assert_eq!(BrickCategory::Other.name(), "Other");
}

/// C005: BrickCategory Display trait
#[test]
fn test_c005_brick_category_display() {
    assert_eq!(format!("{}", BrickCategory::Norm), "Norm");
    assert_eq!(format!("{}", BrickCategory::Attention), "Attention");
    assert_eq!(format!("{}", BrickCategory::Ffn), "FFN");
    assert_eq!(format!("{}", BrickCategory::Other), "Other");
}
