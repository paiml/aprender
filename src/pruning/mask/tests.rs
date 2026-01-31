use super::*;

// ==========================================================================
// FALSIFICATION: SparsityPattern validation
// ==========================================================================
#[test]
fn test_sparsity_pattern_nm_valid() {
    let pattern = SparsityPattern::NM { n: 2, m: 4 };
    assert!(
        pattern.is_valid(),
        "MSK-01 FALSIFIED: 2:4 pattern should be valid"
    );
}

#[test]
fn test_sparsity_pattern_nm_n_equals_m_valid() {
    // N=M is valid (no pruning)
    let pattern = SparsityPattern::NM { n: 4, m: 4 };
    assert!(
        pattern.is_valid(),
        "MSK-02 FALSIFIED: N=M pattern should be valid"
    );
}

#[test]
fn test_sparsity_pattern_nm_invalid_n_gt_m() {
    let pattern = SparsityPattern::NM { n: 5, m: 4 };
    assert!(
        !pattern.is_valid(),
        "MSK-03 FALSIFIED: N>M pattern should be invalid"
    );
}

#[test]
fn test_sparsity_pattern_nm_n_zero_valid() {
    // N=0 is valid (prune everything)
    let pattern = SparsityPattern::NM { n: 0, m: 4 };
    assert!(
        pattern.is_valid(),
        "MSK-04 FALSIFIED: N=0 pattern should be valid"
    );
}

#[test]
fn test_sparsity_pattern_block_valid() {
    let pattern = SparsityPattern::Block {
        height: 2,
        width: 2,
    };
    assert!(
        pattern.is_valid(),
        "MSK-05 FALSIFIED: block pattern should be valid"
    );
}

#[test]
fn test_sparsity_pattern_block_invalid_zero() {
    let pattern = SparsityPattern::Block {
        height: 0,
        width: 2,
    };
    assert!(
        !pattern.is_valid(),
        "MSK-06 FALSIFIED: zero height should be invalid"
    );
}

#[test]
fn test_sparsity_pattern_theoretical_sparsity() {
    let pattern = SparsityPattern::NM { n: 2, m: 4 };
    assert!(
        (pattern.theoretical_sparsity().unwrap() - 0.5).abs() < 1e-6,
        "MSK-07 FALSIFIED: 2:4 has 50% sparsity"
    );

    let pattern = SparsityPattern::NM { n: 1, m: 4 };
    assert!(
        (pattern.theoretical_sparsity().unwrap() - 0.75).abs() < 1e-6,
        "MSK-07 FALSIFIED: 1:4 has 75% sparsity"
    );
}

// ==========================================================================
// FALSIFICATION: SparsityMask creation validates binary values
// ==========================================================================
#[test]
fn test_sparsity_mask_accepts_binary() {
    let mask_data = Tensor::new(&[1.0, 0.0, 1.0, 0.0], &[4]);
    let mask = SparsityMask::new(mask_data, SparsityPattern::Unstructured);
    assert!(
        mask.is_ok(),
        "MSK-08 FALSIFIED: binary mask should be accepted"
    );
}

#[test]
fn test_sparsity_mask_rejects_non_binary() {
    let mask_data = Tensor::new(&[1.0, 0.5, 1.0, 0.0], &[4]);
    let mask = SparsityMask::new(mask_data, SparsityPattern::Unstructured);
    assert!(
        mask.is_err(),
        "MSK-09 FALSIFIED: non-binary mask should be rejected"
    );

    let err = mask.unwrap_err();
    match err {
        PruningError::InvalidMask { .. } => (),
        _ => panic!("MSK-09 FALSIFIED: Expected InvalidMask error"),
    }
}

#[test]
fn test_sparsity_mask_rejects_negative_values() {
    let mask_data = Tensor::new(&[1.0, -1.0, 1.0, 0.0], &[4]);
    let mask = SparsityMask::new(mask_data, SparsityPattern::Unstructured);
    assert!(
        mask.is_err(),
        "MSK-10 FALSIFIED: negative values should be rejected"
    );
}

// ==========================================================================
// FALSIFICATION: SparsityMask sparsity calculation
// ==========================================================================
#[test]
fn test_sparsity_mask_computes_sparsity_correctly() {
    // 2 zeros out of 4 = 50% sparsity
    let mask_data = Tensor::new(&[1.0, 0.0, 1.0, 0.0], &[4]);
    let mask = SparsityMask::new(mask_data, SparsityPattern::Unstructured).unwrap();
    assert!(
        (mask.sparsity() - 0.5).abs() < 1e-6,
        "MSK-11 FALSIFIED: sparsity should be 0.5"
    );
}

#[test]
fn test_sparsity_mask_all_ones_zero_sparsity() {
    let mask_data = Tensor::new(&[1.0, 1.0, 1.0, 1.0], &[4]);
    let mask = SparsityMask::new(mask_data, SparsityPattern::Unstructured).unwrap();
    assert!(
        (mask.sparsity() - 0.0).abs() < 1e-6,
        "MSK-12 FALSIFIED: all-ones mask has 0% sparsity"
    );
}

#[test]
fn test_sparsity_mask_all_zeros_full_sparsity() {
    let mask_data = Tensor::new(&[0.0, 0.0, 0.0, 0.0], &[4]);
    let mask = SparsityMask::new(mask_data, SparsityPattern::Unstructured).unwrap();
    assert!(
        (mask.sparsity() - 1.0).abs() < 1e-6,
        "MSK-13 FALSIFIED: all-zeros mask has 100% sparsity"
    );
}

// ==========================================================================
// FALSIFICATION: Mask application (Poka-Yoke shape validation)
// ==========================================================================
#[test]
fn test_mask_apply_correct_shape() {
    let mask_data = Tensor::new(&[1.0, 0.0, 1.0, 0.0], &[2, 2]);
    let mask = SparsityMask::new(mask_data, SparsityPattern::Unstructured).unwrap();

    let mut weights = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let result = mask.apply(&mut weights);

    assert!(
        result.is_ok(),
        "MSK-14 FALSIFIED: should apply successfully"
    );
    assert_eq!(
        weights.data(),
        &[1.0, 0.0, 3.0, 0.0],
        "MSK-14 FALSIFIED: weights should be masked"
    );
}

#[test]
fn test_mask_apply_wrong_shape_fails() {
    let mask_data = Tensor::new(&[1.0, 0.0, 1.0, 0.0], &[4]);
    let mask = SparsityMask::new(mask_data, SparsityPattern::Unstructured).unwrap();

    let mut weights = Tensor::new(&[1.0, 2.0, 3.0], &[3]); // Wrong shape!
    let result = mask.apply(&mut weights);

    assert!(result.is_err(), "MSK-15 FALSIFIED: wrong shape should fail");
    match result.unwrap_err() {
        PruningError::ShapeMismatch { expected, got } => {
            assert_eq!(expected, vec![4]);
            assert_eq!(got, vec![3]);
        }
        _ => panic!("MSK-15 FALSIFIED: Expected ShapeMismatch error"),
    }
}

// ==========================================================================
// FALSIFICATION: Mask application is idempotent (spec item 48)
// ==========================================================================
#[test]
fn test_mask_apply_idempotent() {
    let mask_data = Tensor::new(&[1.0, 0.0, 1.0, 0.0], &[4]);
    let mask = SparsityMask::new(mask_data, SparsityPattern::Unstructured).unwrap();

    let mut weights1 = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[4]);
    let mut weights2 = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[4]);

    mask.apply(&mut weights1).unwrap();
    mask.apply(&mut weights1).unwrap(); // Apply twice
    mask.apply(&mut weights2).unwrap(); // Apply once

    assert_eq!(
        weights1.data(),
        weights2.data(),
        "MSK-16 FALSIFIED: mask application should be idempotent"
    );
}

// ==========================================================================
// FALSIFICATION: N:M mask validates structure
// ==========================================================================
#[test]
fn test_nm_mask_validates_structure() {
    // Create a valid 2:4 mask
    let mask_data = Tensor::new(
        &[
            1.0, 1.0, 0.0, 0.0, // Group 1: 2 ones
            0.0, 1.0, 1.0, 0.0, // Group 2: 2 ones
        ],
        &[8],
    );

    let pattern = SparsityPattern::NM { n: 2, m: 4 };
    let mask = SparsityMask::new(mask_data, pattern);
    assert!(
        mask.is_ok(),
        "MSK-17 FALSIFIED: valid 2:4 mask should be accepted"
    );
}

#[test]
fn test_nm_mask_rejects_invalid_structure() {
    // Invalid: 3 ones in first group (should be 2)
    let mask_data = Tensor::new(
        &[
            1.0, 1.0, 1.0, 0.0, // Group 1: 3 ones (INVALID!)
            0.0, 1.0, 1.0, 0.0, // Group 2: 2 ones
        ],
        &[8],
    );

    let pattern = SparsityPattern::NM { n: 2, m: 4 };
    let mask = SparsityMask::new(mask_data, pattern);
    assert!(
        mask.is_err(),
        "MSK-18 FALSIFIED: invalid N:M structure should be rejected"
    );
}

// ==========================================================================
// FALSIFICATION: generate_unstructured_mask
// ==========================================================================
#[test]
fn test_generate_unstructured_mask_basic() {
    let scores = Tensor::new(&[0.1, 0.4, 0.2, 0.3], &[4]);
    let mask = generate_unstructured_mask(&scores, 0.5).unwrap();

    // 50% sparsity = prune 2 lowest scores (0.1, 0.2)
    // Keep: 0.4, 0.3 -> mask = [0, 1, 0, 1]
    assert!(
        (mask.sparsity() - 0.5).abs() < 1e-6,
        "MSK-19 FALSIFIED: should achieve ~50% sparsity"
    );
}

#[test]
fn test_generate_unstructured_mask_zero_sparsity() {
    let scores = Tensor::new(&[0.1, 0.2, 0.3, 0.4], &[4]);
    let mask = generate_unstructured_mask(&scores, 0.0).unwrap();

    assert!(
        (mask.sparsity() - 0.0).abs() < 1e-6,
        "MSK-20 FALSIFIED: 0% sparsity should keep all"
    );
}

#[test]
fn test_generate_unstructured_mask_full_sparsity() {
    let scores = Tensor::new(&[0.1, 0.2, 0.3, 0.4], &[4]);
    let mask = generate_unstructured_mask(&scores, 1.0).unwrap();

    assert!(
        (mask.sparsity() - 1.0).abs() < 1e-6,
        "MSK-21 FALSIFIED: 100% sparsity should prune all"
    );
}

#[test]
fn test_generate_unstructured_mask_invalid_sparsity() {
    let scores = Tensor::new(&[0.1, 0.2], &[2]);

    assert!(
        generate_unstructured_mask(&scores, -0.1).is_err(),
        "MSK-22 FALSIFIED: negative sparsity should error"
    );
    assert!(
        generate_unstructured_mask(&scores, 1.1).is_err(),
        "MSK-22 FALSIFIED: sparsity >1 should error"
    );
}

// ==========================================================================
// FALSIFICATION: generate_nm_mask
// ==========================================================================
#[test]
fn test_generate_nm_mask_2_4() {
    let scores = Tensor::new(&[0.1, 0.4, 0.2, 0.3, 0.5, 0.1, 0.3, 0.2], &[8]);
    let mask = generate_nm_mask(&scores, 2, 4).unwrap();

    // Each group of 4 should have exactly 2 non-zeros
    let data = mask.tensor().data();
    for chunk in data.chunks(4) {
        let ones = chunk.iter().filter(|&&v| v > 0.5).count();
        assert_eq!(
            ones, 2,
            "MSK-23 FALSIFIED: each group should have 2 non-zeros"
        );
    }
}

#[test]
fn test_generate_nm_mask_keeps_top_n() {
    // Group 1: [0.1, 0.4, 0.2, 0.3] -> keep 0.4, 0.3
    let scores = Tensor::new(&[0.1, 0.4, 0.2, 0.3], &[4]);
    let mask = generate_nm_mask(&scores, 2, 4).unwrap();

    let data = mask.tensor().data();
    assert_eq!(data[0], 0.0, "MSK-24 FALSIFIED: 0.1 should be pruned");
    assert_eq!(data[1], 1.0, "MSK-24 FALSIFIED: 0.4 should be kept");
    assert_eq!(data[2], 0.0, "MSK-24 FALSIFIED: 0.2 should be pruned");
    assert_eq!(data[3], 1.0, "MSK-24 FALSIFIED: 0.3 should be kept");
}

#[test]
fn test_generate_nm_mask_invalid_n_gt_m() {
    let scores = Tensor::new(&[0.1, 0.2, 0.3, 0.4], &[4]);
    let result = generate_nm_mask(&scores, 5, 4);
    assert!(result.is_err(), "MSK-25 FALSIFIED: N>M should error");
}

#[test]
fn test_generate_nm_mask_not_divisible() {
    let scores = Tensor::new(&[0.1, 0.2, 0.3], &[3]); // Not divisible by 4
    let result = generate_nm_mask(&scores, 2, 4);
    assert!(
        result.is_err(),
        "MSK-26 FALSIFIED: non-divisible length should error"
    );
}

// ==========================================================================
// FALSIFICATION: Helper methods
// ==========================================================================
#[test]
fn test_mask_nnz() {
    let mask = SparsityMask::new(
        Tensor::new(&[1.0, 0.0, 1.0, 0.0, 1.0], &[5]),
        SparsityPattern::Unstructured,
    )
    .unwrap();

    assert_eq!(mask.nnz(), 3, "MSK-27 FALSIFIED: nnz should be 3");
    assert_eq!(
        mask.num_zeros(),
        2,
        "MSK-27 FALSIFIED: num_zeros should be 2"
    );
}

#[test]
fn test_dense_mask() {
    let mask = SparsityMask::dense(&[3, 4]);
    assert_eq!(
        mask.shape(),
        &[3, 4],
        "MSK-28 FALSIFIED: shape should match"
    );
    assert_eq!(
        mask.sparsity(),
        0.0,
        "MSK-28 FALSIFIED: dense mask has 0 sparsity"
    );
    assert_eq!(mask.nnz(), 12, "MSK-28 FALSIFIED: all elements non-zero");
}

// ==========================================================================
// FALSIFICATION: generate_block_mask
// ==========================================================================
#[test]
fn test_generate_block_mask_basic() {
    // 4x4 tensor with 2x2 blocks = 4 blocks
    // Prune 50% = 2 blocks
    let scores = Tensor::new(
        &[
            1.0, 1.0, 2.0, 2.0, // Row 0
            1.0, 1.0, 2.0, 2.0, // Row 1
            3.0, 3.0, 4.0, 4.0, // Row 2
            3.0, 3.0, 4.0, 4.0, // Row 3
        ],
        &[4, 4],
    );

    let mask = generate_block_mask(&scores, 2, 2, 0.5).unwrap();

    // Should prune 2 lowest blocks (sums: 4, 8, 12, 16)
    // Block (0,0) sum=4, Block (0,1) sum=8 should be pruned
    assert!(
        (mask.sparsity() - 0.5).abs() < 1e-6,
        "MSK-29 FALSIFIED: should achieve 50% sparsity"
    );
}

#[test]
fn test_generate_block_mask_uniform_blocks() {
    // All blocks should be uniform (all 0s or all 1s)
    let scores = Tensor::new(&[1.0; 16], &[4, 4]);
    let mask = generate_block_mask(&scores, 2, 2, 0.5).unwrap();

    let data = mask.tensor().data();
    // Check each 2x2 block is uniform
    for br in 0..2 {
        for bc in 0..2 {
            let first = data[br * 2 * 4 + bc * 2];
            let uniform = (0..2).all(|r| {
                (0..2).all(|c| {
                    let idx = (br * 2 + r) * 4 + (bc * 2 + c);
                    (data[idx] - first).abs() < 1e-6
                })
            });
            assert!(
                uniform,
                "MSK-30 FALSIFIED: block ({}, {}) should be uniform",
                br, bc
            );
        }
    }
}

#[test]
fn test_generate_block_mask_not_divisible() {
    let scores = Tensor::new(&[1.0; 15], &[3, 5]);
    let result = generate_block_mask(&scores, 2, 2, 0.5);
    assert!(
        result.is_err(),
        "MSK-31 FALSIFIED: non-divisible shape should error"
    );
}

#[test]
fn test_generate_block_mask_1d_rejected() {
    let scores = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[4]);
    let result = generate_block_mask(&scores, 2, 2, 0.5);
    assert!(
        result.is_err(),
        "MSK-32 FALSIFIED: 1D tensor should be rejected"
    );
}

#[test]
fn test_generate_block_mask_invalid_sparsity() {
    let scores = Tensor::new(&[1.0; 16], &[4, 4]);
    assert!(
        generate_block_mask(&scores, 2, 2, -0.1).is_err(),
        "MSK-33 FALSIFIED: negative sparsity should error"
    );
    assert!(
        generate_block_mask(&scores, 2, 2, 1.1).is_err(),
        "MSK-33 FALSIFIED: sparsity >1 should error"
    );
}

// ==========================================================================
// FALSIFICATION: generate_row_mask
// ==========================================================================
#[test]
fn test_generate_row_mask_basic() {
    // 4 rows, prune 50% = 2 rows
    let scores = Tensor::new(
        &[
            1.0, 1.0, // Row 0: sum = 2 (lowest)
            2.0, 2.0, // Row 1: sum = 4
            3.0, 3.0, // Row 2: sum = 6
            4.0, 4.0, // Row 3: sum = 8 (highest)
        ],
        &[4, 2],
    );

    let mask = generate_row_mask(&scores, 0.5).unwrap();

    // Should prune rows 0 and 1 (lowest sums)
    assert!(
        (mask.sparsity() - 0.5).abs() < 1e-6,
        "MSK-34 FALSIFIED: should achieve 50% sparsity"
    );

    // Verify rows are uniform
    let data = mask.tensor().data();
    for r in 0..4 {
        let first = data[r * 2];
        assert!(
            (data[r * 2 + 1] - first).abs() < 1e-6,
            "MSK-34 FALSIFIED: row {} should be uniform",
            r
        );
    }
}

#[test]
fn test_generate_row_mask_keeps_highest() {
    let scores = Tensor::new(
        &[
            1.0, 1.0, // Row 0: sum = 2
            10.0, 10.0, // Row 1: sum = 20 (highest)
        ],
        &[2, 2],
    );

    let mask = generate_row_mask(&scores, 0.5).unwrap();
    let data = mask.tensor().data();

    // Row 0 should be pruned, row 1 kept
    assert_eq!(data[0], 0.0, "MSK-35 FALSIFIED: row 0 should be pruned");
    assert_eq!(data[1], 0.0, "MSK-35 FALSIFIED: row 0 should be pruned");
    assert_eq!(data[2], 1.0, "MSK-35 FALSIFIED: row 1 should be kept");
    assert_eq!(data[3], 1.0, "MSK-35 FALSIFIED: row 1 should be kept");
}

#[test]
fn test_generate_row_mask_1d_rejected() {
    let scores = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
    let result = generate_row_mask(&scores, 0.5);
    assert!(
        result.is_err(),
        "MSK-36 FALSIFIED: 1D tensor should be rejected"
    );
}

// ==========================================================================
// FALSIFICATION: generate_column_mask
// ==========================================================================
#[test]
fn test_generate_column_mask_basic() {
    // 2 rows, 4 columns, prune 50% = 2 columns
    let scores = Tensor::new(
        &[
            1.0, 2.0, 3.0, 4.0, // Row 0
            1.0, 2.0, 3.0, 4.0, // Row 1
        ],
        &[2, 4],
    );

    let mask = generate_column_mask(&scores, 0.5).unwrap();

    // Should prune columns 0 and 1 (lowest sums: 2, 4)
    assert!(
        (mask.sparsity() - 0.5).abs() < 1e-6,
        "MSK-37 FALSIFIED: should achieve 50% sparsity"
    );

    // Verify columns are uniform
    let data = mask.tensor().data();
    for c in 0..4 {
        let first = data[c];
        assert!(
            (data[4 + c] - first).abs() < 1e-6,
            "MSK-37 FALSIFIED: column {} should be uniform",
            c
        );
    }
}

#[test]
fn test_generate_column_mask_keeps_highest() {
    let scores = Tensor::new(
        &[
            1.0, 10.0, // Col 0: sum=2, Col 1: sum=20
            1.0, 10.0,
        ],
        &[2, 2],
    );

    let mask = generate_column_mask(&scores, 0.5).unwrap();
    let data = mask.tensor().data();

    // Column 0 should be pruned, column 1 kept
    assert_eq!(data[0], 0.0, "MSK-38 FALSIFIED: col 0 should be pruned");
    assert_eq!(data[1], 1.0, "MSK-38 FALSIFIED: col 1 should be kept");
    assert_eq!(data[2], 0.0, "MSK-38 FALSIFIED: col 0 should be pruned");
    assert_eq!(data[3], 1.0, "MSK-38 FALSIFIED: col 1 should be kept");
}

#[test]
fn test_generate_column_mask_1d_rejected() {
    let scores = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
    let result = generate_column_mask(&scores, 0.5);
    assert!(
        result.is_err(),
        "MSK-39 FALSIFIED: 1D tensor should be rejected"
    );
}

#[test]
fn test_generate_column_mask_invalid_sparsity() {
    let scores = Tensor::new(&[1.0; 4], &[2, 2]);
    assert!(
        generate_column_mask(&scores, -0.1).is_err(),
        "MSK-40 FALSIFIED: negative sparsity should error"
    );
}

// ==========================================================================
// FALSIFICATION: SparsityPattern theoretical_sparsity additional
// ==========================================================================
#[test]
fn test_sparsity_pattern_theoretical_sparsity_row_col() {
    let row = SparsityPattern::Row;
    assert!(
        row.theoretical_sparsity().is_none(),
        "MSK-41 FALSIFIED: Row should return None"
    );

    let col = SparsityPattern::Column;
    assert!(
        col.theoretical_sparsity().is_none(),
        "MSK-41 FALSIFIED: Column should return None"
    );
}

// ==========================================================================
// FALSIFICATION: SparsityMask tensor getter
// ==========================================================================
#[test]
fn test_sparsity_mask_tensor() {
    let mask_data = Tensor::new(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
    let mask = SparsityMask::new(mask_data, SparsityPattern::Unstructured).unwrap();

    assert_eq!(mask.tensor().shape(), &[2, 2]);
    assert_eq!(mask.tensor().data(), &[1.0, 0.0, 0.0, 1.0]);
}

// ==========================================================================
// FALSIFICATION: SparsityMask pattern getter
// ==========================================================================
#[test]
fn test_sparsity_mask_pattern() {
    let mask_data = Tensor::new(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
    let mask = SparsityMask::new(mask_data, SparsityPattern::Unstructured).unwrap();

    assert_eq!(mask.pattern(), SparsityPattern::Unstructured);
}

// ==========================================================================
// FALSIFICATION: generate_row_mask invalid sparsity positive
// ==========================================================================
#[test]
fn test_generate_row_mask_invalid_sparsity_positive() {
    let scores = Tensor::new(&[1.0; 4], &[2, 2]);
    assert!(
        generate_row_mask(&scores, 1.5).is_err(),
        "MSK-42 FALSIFIED: sparsity > 1.0 should error"
    );
}

// ==========================================================================
// FALSIFICATION: SparsityPattern Clone and Debug
// ==========================================================================
#[test]
fn test_sparsity_pattern_clone() {
    let nm = SparsityPattern::NM { n: 2, m: 4 };
    let cloned = nm.clone();
    assert_eq!(nm, cloned);
}

#[test]
fn test_sparsity_pattern_debug() {
    let block = SparsityPattern::Block {
        height: 2,
        width: 2,
    };
    let debug = format!("{:?}", block);
    assert!(debug.contains("Block"));
}

#[test]
fn test_sparsity_pattern_copy() {
    let unstructured = SparsityPattern::Unstructured;
    let copied = unstructured;
    assert_eq!(unstructured, copied);
}

// ==========================================================================
// FALSIFICATION: SparsityMask shape getter
// ==========================================================================
#[test]
fn test_sparsity_mask_shape() {
    let mask_data = Tensor::new(&[1.0; 6], &[2, 3]);
    let mask = SparsityMask::new(mask_data, SparsityPattern::Unstructured).unwrap();
    assert_eq!(mask.shape(), &[2, 3]);
}
