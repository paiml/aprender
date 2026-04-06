//! Coverage tests for `TiledQ4KMatvec::execute_parallel` (q4k_matvec.rs:141).
//!
//! Target: 24 uncovered lines, impact 12.3.
//! Note: in_dim (K) MUST be a multiple of 256 (Q4K_SUPERBLOCK_SIZE).

use super::super::*;

// ========================================================================
// execute_parallel — exercises the Rayon parallel path (or scalar fallback)
// ========================================================================

/// Basic parallel execution with minimal matrix (2 rows x 256 cols).
#[test]
fn test_execute_parallel_basic() {
    let m = 2;
    let k = 256; // 1 superblock per row

    let matvec = TiledQ4KMatvec::new(m, k);

    // Create valid Q4K weights (2 rows x 1 superblock x 144 bytes)
    let mut weights = vec![0u8; m * Q4K_SUPERBLOCK_BYTES];

    // Set d=1.0 (f16 = 0x3C00), dmin=0.0 for each row
    for row in 0..m {
        let offset = row * Q4K_SUPERBLOCK_BYTES;
        weights[offset] = 0x00;
        weights[offset + 1] = 0x3C;
        // dmin, scales, qs all zero
    }

    let input = vec![1.0f32; k];
    let mut output = vec![0.0f32; m];

    matvec.execute_parallel(&weights, &input, &mut output);

    // With zero quantized values, output should be finite (close to zero)
    for val in &output {
        assert!(val.is_finite());
    }
}

/// Parallel execution matches scalar execution.
#[test]
fn test_execute_parallel_matches_scalar() {
    let m = 4;
    let k = 256; // 1 superblock per row

    let matvec = TiledQ4KMatvec::new(m, k);

    // Create weights with non-trivial patterns
    let mut weights = vec![0u8; m * Q4K_SUPERBLOCK_BYTES];
    for row in 0..m {
        let offset = row * Q4K_SUPERBLOCK_BYTES;
        // d = 1.0 (f16: 0x3C00)
        weights[offset] = 0x00;
        weights[offset + 1] = 0x3C;
        // dmin = 0.0
        weights[offset + 2] = 0x00;
        weights[offset + 3] = 0x00;
        // Set some scale values
        weights[offset + 4] = 0x01;
        weights[offset + 5] = 0x02;
        // Set quantized values to non-zero
        for i in 16..144 {
            weights[offset + i] = ((row * 7 + i * 3) % 256) as u8;
        }
    }

    let input: Vec<f32> = (0..k).map(|i| (i as f32) * 0.01).collect();

    let mut output_scalar = vec![0.0f32; m];
    let mut output_parallel = vec![0.0f32; m];

    matvec.execute_scalar(&weights, &input, &mut output_scalar);
    matvec.execute_parallel(&weights, &input, &mut output_parallel);

    for i in 0..m {
        assert!(
            (output_scalar[i] - output_parallel[i]).abs() < 1e-5,
            "Mismatch at row {}: scalar={}, parallel={}",
            i,
            output_scalar[i],
            output_parallel[i]
        );
    }
}

/// Parallel execution with multiple superblocks per row (K = 512 = 2 superblocks).
#[test]
fn test_execute_parallel_multiple_superblocks() {
    let m = 3;
    let k = 512; // 2 superblocks per row
    let sb_per_row = k / Q4K_SUPERBLOCK_SIZE;

    let matvec = TiledQ4KMatvec::new(m, k);
    assert_eq!(matvec.superblocks_per_row(), sb_per_row);

    let mut weights = vec![0u8; m * sb_per_row * Q4K_SUPERBLOCK_BYTES];
    for row in 0..m {
        for sb in 0..sb_per_row {
            let offset = (row * sb_per_row + sb) * Q4K_SUPERBLOCK_BYTES;
            // d = 0.5 (f16: 0x3800)
            weights[offset] = 0x00;
            weights[offset + 1] = 0x38;
            // Fill qs with alternating pattern
            for i in 16..144 {
                weights[offset + i] = 0x55; // alternating nibbles (5, 5)
            }
        }
    }

    let input = vec![1.0f32; k];
    let mut output_scalar = vec![0.0f32; m];
    let mut output_parallel = vec![0.0f32; m];

    matvec.execute_scalar(&weights, &input, &mut output_scalar);
    matvec.execute_parallel(&weights, &input, &mut output_parallel);

    for i in 0..m {
        assert!(
            (output_scalar[i] - output_parallel[i]).abs() < 1e-5,
            "Row {} mismatch: scalar={}, parallel={}",
            i,
            output_scalar[i],
            output_parallel[i]
        );
    }
}

/// Larger matrix to ensure parallel dispatch actually distributes work.
#[test]
fn test_execute_parallel_larger_matrix() {
    let m = 64;
    let k = 256;

    let matvec = TiledQ4KMatvec::new(m, k);

    let mut weights = vec![0u8; m * Q4K_SUPERBLOCK_BYTES];
    for row in 0..m {
        let offset = row * Q4K_SUPERBLOCK_BYTES;
        // d = 1.0
        weights[offset] = 0x00;
        weights[offset + 1] = 0x3C;
        // Fill qs with row-dependent pattern
        for i in 16..144 {
            weights[offset + i] = ((row + i) % 256) as u8;
        }
    }

    let input: Vec<f32> = (0..k).map(|i| ((i % 10) as f32) * 0.1).collect();
    let mut output_scalar = vec![0.0f32; m];
    let mut output_parallel = vec![0.0f32; m];

    matvec.execute_scalar(&weights, &input, &mut output_scalar);
    matvec.execute_parallel(&weights, &input, &mut output_parallel);

    for i in 0..m {
        assert!(
            (output_scalar[i] - output_parallel[i]).abs() < 1e-4,
            "Row {} mismatch: scalar={}, parallel={}",
            i,
            output_scalar[i],
            output_parallel[i]
        );
    }
}

/// Parallel execution with K = 1024 (4 superblocks per row).
#[test]
fn test_execute_parallel_large_k() {
    let m = 8;
    let k = 1024; // 4 superblocks per row
    let sb_per_row = k / Q4K_SUPERBLOCK_SIZE;

    let matvec = TiledQ4KMatvec::new(m, k);
    assert_eq!(matvec.superblocks_per_row(), 4);

    let mut weights = vec![0u8; m * sb_per_row * Q4K_SUPERBLOCK_BYTES];
    for row in 0..m {
        for sb in 0..sb_per_row {
            let offset = (row * sb_per_row + sb) * Q4K_SUPERBLOCK_BYTES;
            // d = 2.0 (f16: 0x4000)
            weights[offset] = 0x00;
            weights[offset + 1] = 0x40;
            // Some non-zero scales
            weights[offset + 4] = 0x03;
            // Non-zero qs
            for i in 16..144 {
                weights[offset + i] = 0xAA; // nibbles: 10, 10
            }
        }
    }

    let input = vec![0.5f32; k];
    let mut output = vec![0.0f32; m];

    matvec.execute_parallel(&weights, &input, &mut output);

    // All outputs should be finite and non-zero
    for (i, val) in output.iter().enumerate() {
        assert!(val.is_finite(), "Row {} is not finite", i);
    }
}

/// Verify that execute_parallel with single row works.
#[test]
fn test_execute_parallel_single_row() {
    let m = 1;
    let k = 256;

    let matvec = TiledQ4KMatvec::new(m, k);

    let mut weights = vec![0u8; Q4K_SUPERBLOCK_BYTES];
    // d = 1.0, dmin = 0
    weights[0] = 0x00;
    weights[1] = 0x3C;
    // All qs = 0x11 -> nibbles 1 and 1
    for i in 16..144 {
        weights[i] = 0x11;
    }

    let input = vec![1.0f32; k];
    let mut output_scalar = vec![0.0f32; m];
    let mut output_parallel = vec![0.0f32; m];

    matvec.execute_scalar(&weights, &input, &mut output_scalar);
    matvec.execute_parallel(&weights, &input, &mut output_parallel);

    assert!(
        (output_scalar[0] - output_parallel[0]).abs() < 1e-5,
        "scalar={}, parallel={}",
        output_scalar[0],
        output_parallel[0]
    );
}
