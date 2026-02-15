
// ==========================================================================
// FALSIFICATION: Column validation with valid uniform columns
// ==========================================================================
#[test]
fn test_column_validate_valid_uniform() {
    // 2x2 mask where each column is uniform
    let mask_data = Tensor::new(&[1.0, 0.0, 1.0, 0.0], &[2, 2]);
    let result = SparsityMask::new(mask_data, SparsityPattern::Column);
    assert!(result.is_ok(), "Uniform columns should be valid");
}
