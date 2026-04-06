use super::super::*;

// ========================================================================
// Phase 3: Packing Tests
// ========================================================================

#[test]
fn test_pack_a_layout() {
    // 4x3 matrix, pack first 4 rows
    let a = vec![
        1.0, 2.0, 3.0, // row 0
        4.0, 5.0, 6.0, // row 1
        7.0, 8.0, 9.0, // row 2
        10.0, 11.0, 12.0, // row 3
    ];

    let mut packed = vec![0.0; packed_a_size(4, 3)];
    pack_a(&a, 3, 4, 3, &mut packed);

    // Expected layout: column-major within MR-panels
    // For MR=8, we have one panel with 4 real rows + 4 zero padding
    // Col 0: [1, 4, 7, 10, 0, 0, 0, 0]
    // Col 1: [2, 5, 8, 11, 0, 0, 0, 0]
    // Col 2: [3, 6, 9, 12, 0, 0, 0, 0]
    assert_eq!(packed[0], 1.0); // (0,0)
    assert_eq!(packed[1], 4.0); // (1,0)
    assert_eq!(packed[2], 7.0); // (2,0)
    assert_eq!(packed[3], 10.0); // (3,0)
    assert_eq!(packed[4], 0.0); // padding
    assert_eq!(packed[MR], 2.0); // (0,1)
}

#[test]
fn test_pack_b_layout() {
    // 3x4 matrix
    let b = vec![
        1.0, 2.0, 3.0, 4.0, // row 0
        5.0, 6.0, 7.0, 8.0, // row 1
        9.0, 10.0, 11.0, 12.0, // row 2
    ];

    let mut packed = vec![0.0; packed_b_size(3, 4)];
    pack_b(&b, 4, 3, 4, &mut packed);

    // Expected: row-major within NR-panels
    // For NR=6, we have one panel with 4 real cols + 2 zero padding
    // Row 0: [1, 2, 3, 4, 0, 0]
    // Row 1: [5, 6, 7, 8, 0, 0]
    // Row 2: [9, 10, 11, 12, 0, 0]
    assert_eq!(packed[0], 1.0);
    assert_eq!(packed[1], 2.0);
    assert_eq!(packed[2], 3.0);
    assert_eq!(packed[3], 4.0);
    assert_eq!(packed[4], 0.0); // padding
    assert_eq!(packed[NR], 5.0); // row 1
}
