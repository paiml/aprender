use super::*;

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

// ==========================================================================
// FALSIFICATION: Block validation with <2D mask
// ==========================================================================
#[test]
fn test_block_validate_1d_mask() {
    let mask_data = Tensor::new(&[1.0, 0.0, 1.0, 0.0], &[4]);
    let result = SparsityMask::new(
        mask_data,
        SparsityPattern::Block {
            height: 2,
            width: 2,
        },
    );
    assert!(result.is_err(), "Block sparsity should require 2D mask");
}

// ==========================================================================
// FALSIFICATION: Block validation with non-divisible shape
// ==========================================================================
#[test]
fn test_block_validate_non_divisible_shape() {
    // 3x3 mask with 2x2 blocks - 3 is not divisible by 2
    let mask_data = Tensor::new(&[1.0; 9], &[3, 3]);
    let result = SparsityMask::new(
        mask_data,
        SparsityPattern::Block {
            height: 2,
            width: 2,
        },
    );
    assert!(result.is_err(), "Block should error on non-divisible shape");
}

// ==========================================================================
// FALSIFICATION: Block validation with non-uniform block
// ==========================================================================
#[test]
fn test_block_validate_non_uniform_block() {
    // 2x2 mask with 2x2 block, but values are not uniform within the block
    let mask_data = Tensor::new(&[1.0, 0.0, 1.0, 1.0], &[2, 2]);
    let result = SparsityMask::new(
        mask_data,
        SparsityPattern::Block {
            height: 2,
            width: 2,
        },
    );
    assert!(result.is_err(), "Block should error on non-uniform block");
}

// ==========================================================================
// FALSIFICATION: Row validation with <2D mask
// ==========================================================================
#[test]
fn test_row_validate_1d_mask() {
    let mask_data = Tensor::new(&[1.0, 0.0, 1.0, 0.0], &[4]);
    let result = SparsityMask::new(mask_data, SparsityPattern::Row);
    assert!(result.is_err(), "Row sparsity should require 2D mask");
}

// ==========================================================================
// FALSIFICATION: Row validation with non-uniform row
// ==========================================================================
#[test]
fn test_row_validate_non_uniform_row() {
    // 2x2 mask where first row is not uniform
    let mask_data = Tensor::new(&[1.0, 0.0, 1.0, 1.0], &[2, 2]);
    let result = SparsityMask::new(mask_data, SparsityPattern::Row);
    assert!(result.is_err(), "Row should error on non-uniform row");
}

// ==========================================================================
// FALSIFICATION: Column validation with <2D mask
// ==========================================================================
#[test]
fn test_column_validate_1d_mask() {
    let mask_data = Tensor::new(&[1.0, 0.0, 1.0, 0.0], &[4]);
    let result = SparsityMask::new(mask_data, SparsityPattern::Column);
    assert!(result.is_err(), "Column sparsity should require 2D mask");
}

// ==========================================================================
// FALSIFICATION: Column validation with non-uniform column
// ==========================================================================
#[test]
fn test_column_validate_non_uniform_column() {
    // 2x2 mask where first column is not uniform (top-left=1.0, bottom-left=0.0)
    let mask_data = Tensor::new(&[1.0, 1.0, 0.0, 1.0], &[2, 2]);
    let result = SparsityMask::new(mask_data, SparsityPattern::Column);
    assert!(result.is_err(), "Column should error on non-uniform column");
}

// ==========================================================================
// FALSIFICATION: SparsityMask with empty data
// ==========================================================================
#[test]
fn test_sparsity_mask_empty_data() {
    let mask_data = Tensor::new(&[], &[0]);
    let mask = SparsityMask::new(mask_data, SparsityPattern::Unstructured);
    assert!(mask.is_ok());
    let mask = mask.unwrap();
    assert!((mask.sparsity() - 0.0).abs() < 1e-6);
}

// ==========================================================================
// FALSIFICATION: SparsityPattern Default
// ==========================================================================
#[test]
fn test_sparsity_pattern_default() {
    let pattern = SparsityPattern::default();
    assert_eq!(pattern, SparsityPattern::Unstructured);
}

// ==========================================================================
// FALSIFICATION: generate_nm_mask with m=0
// ==========================================================================
#[test]
fn test_generate_nm_mask_m_zero() {
    let scores = Tensor::new(&[1.0; 4], &[4]);
    let result = generate_nm_mask(&scores, 1, 0);
    assert!(result.is_err(), "M=0 should error");
}

// ==========================================================================
// FALSIFICATION: generate_nm_mask with n > m
// ==========================================================================
#[test]
fn test_generate_nm_mask_n_greater_than_m() {
    let scores = Tensor::new(&[1.0; 4], &[4]);
    let result = generate_nm_mask(&scores, 3, 2);
    assert!(result.is_err(), "N > M should error");
}

// ==========================================================================
// FALSIFICATION: generate_nm_mask data not divisible by m (coverage boost)
// ==========================================================================
#[test]
fn test_generate_nm_mask_not_divisible_coverage() {
    let scores = Tensor::new(&[1.0; 5], &[5]); // 5 is not divisible by 4
    let result = generate_nm_mask(&scores, 2, 4);
    assert!(result.is_err(), "Length not divisible by M should error");
}

// ==========================================================================
// FALSIFICATION: generate_unstructured_mask with empty tensor
// ==========================================================================
#[test]
fn test_generate_unstructured_mask_empty() {
    let scores = Tensor::new(&[], &[0]);
    let mask = generate_unstructured_mask(&scores, 0.5);
    assert!(mask.is_ok());
}

// ==========================================================================
// FALSIFICATION: Block validation with valid uniform blocks
// ==========================================================================
#[test]
fn test_block_validate_valid_uniform() {
    // 4x4 mask with 2x2 blocks, all ones (uniform)
    let mask_data = Tensor::new(&[1.0; 16], &[4, 4]);
    let result = SparsityMask::new(
        mask_data,
        SparsityPattern::Block {
            height: 2,
            width: 2,
        },
    );
    assert!(result.is_ok(), "Uniform blocks should be valid");
}

// ==========================================================================
// FALSIFICATION: Row validation with valid uniform rows
// ==========================================================================
#[test]
fn test_row_validate_valid_uniform() {
    // 2x3 mask where each row is uniform
    let mask_data = Tensor::new(&[1.0, 1.0, 1.0, 0.0, 0.0, 0.0], &[2, 3]);
    let result = SparsityMask::new(mask_data, SparsityPattern::Row);
    assert!(result.is_ok(), "Uniform rows should be valid");
}
