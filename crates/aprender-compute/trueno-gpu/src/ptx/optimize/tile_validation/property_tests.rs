use super::*;
use proptest::prelude::*;

/// Generate power-of-two values
fn power_of_two() -> impl Strategy<Value = usize> {
    (0u32..12).prop_map(|exp| 1usize << exp) // 1, 2, 4, ..., 2048
}

/// Generate non-power-of-two values
fn non_power_of_two() -> impl Strategy<Value = usize> {
    (3usize..1000).prop_filter("not power of two", |&n| !n.is_power_of_two())
}

proptest! {
    /// All power-of-two single dimensions are valid (within limits)
    #[test]
    fn power_of_two_single_dim_valid(dim in power_of_two()) {
        if dim > 0 && dim <= MAX_TILE_DIM {
            prop_assert!(validate_shape(&[dim]).is_ok(),
                "Power of two {} should be valid", dim);
        }
    }

    /// All non-power-of-two dimensions are rejected
    #[test]
    fn non_power_of_two_rejected(dim in non_power_of_two()) {
        let result = validate_shape(&[dim]);
        prop_assert!(result.is_err(), "Non-power-of-two {} should be rejected", dim);
        if let Err(TileError::NonPowerOfTwo { dim: d }) = result {
            prop_assert_eq!(d, dim);
        }
    }

    /// Product of dimensions <= MAX_TILE_ELEMENTS is valid
    #[test]
    fn total_elements_within_limit(exp1 in 0u32..10, exp2 in 0u32..10) {
        let d1 = 1usize << exp1;
        let d2 = 1usize << exp2;
        let total = d1.saturating_mul(d2);

        if d1 <= MAX_TILE_DIM && d2 <= MAX_TILE_DIM && total <= MAX_TILE_ELEMENTS {
            prop_assert!(validate_shape(&[d1, d2]).is_ok(),
                "{}x{} = {} should be valid", d1, d2, total);
        }
    }

    /// TileError::Display is consistent
    #[test]
    fn tile_error_display_contains_values(dim in non_power_of_two()) {
        let err = TileError::NonPowerOfTwo { dim };
        let msg = err.to_string();
        prop_assert!(msg.contains(&dim.to_string()),
            "Error message should contain dimension: {}", msg);
    }

    /// validate always returns Ok or Err (never panics)
    #[test]
    fn validate_never_panics(dims in prop::collection::vec(0usize..10000, 0..5)) {
        // This just verifies no panic occurs
        let _ = validate_shape(&dims);
    }

    /// WMMA valid shapes pass validation
    #[test]
    fn wmma_valid_shapes_pass(_dummy in 0u8..3) {
        let shapes = [
            WmmaShape::M16N16K16,
            WmmaShape::M8N32K16,
            WmmaShape::M32N8K16,
        ];
        for shape in shapes {
            prop_assert!(validate_wmma_shape(&shape).is_ok(),
                "Valid WMMA shape {:?} should pass", shape);
        }
    }

    /// WMMA shapes with invalid dimensions fail
    #[test]
    fn wmma_invalid_shapes_fail(m in 1u32..100, n in 1u32..100, k in 1u32..100) {
        let shape = WmmaShape { m, n, k };

        // Only specific combinations are valid
        let is_valid = matches!(
            (m, n, k),
            (16, 16, 16) | (8, 32, 16) | (32, 8, 16)
        );

        let result = validate_wmma_shape(&shape);
        prop_assert_eq!(result.is_ok(), is_valid,
            "WMMA shape m{}n{}k{} validity mismatch", m, n, k);
    }
}
