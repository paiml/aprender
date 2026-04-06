//! Parallel dispatch, compute_chunk_scalar, and multi-block tests.

use super::super::gemv::compute_chunk_scalar;
use super::super::*;

#[test]
fn test_parallel_dispatch_large_matrix() {
    // Test parallel path: total_work >= 8_000_000
    // Use 4096 x 2048 = 8_388_608 ops (triggers parallel)
    let out_dim = 4096;
    let in_dim = 2048; // Must be multiple of 256 (SUPER_BLOCK_SIZE)
    let total_work = out_dim * in_dim;
    assert!(total_work >= 8_000_000, "Test must trigger parallel path");

    let num_superblocks_per_row = (in_dim + SUPER_BLOCK_SIZE - 1) / SUPER_BLOCK_SIZE;
    let row_bytes = num_superblocks_per_row * SUPER_BLOCK_BYTES;
    let total_bytes = out_dim * row_bytes;

    // Create deterministic test data
    let mut q6k_data = vec![0u8; total_bytes];
    for row in 0..out_dim {
        for sb in 0..num_superblocks_per_row {
            let offset = row * row_bytes + sb * SUPER_BLOCK_BYTES;
            // d = 1.0 as f16
            q6k_data[offset] = 0x00;
            q6k_data[offset + 1] = 0x3C;
            // ql: 6-bit low parts
            for i in 0..128 {
                q6k_data[offset + 2 + i] = ((row + sb + i) % 64) as u8;
            }
            // qh: 2-bit high parts
            for i in 0..64 {
                q6k_data[offset + 130 + i] = ((row ^ sb ^ i) % 4) as u8;
            }
            // scales
            for i in 0..16 {
                q6k_data[offset + 194 + i] = 0x10;
            }
        }
    }

    let input: Vec<f32> = (0..in_dim).map(|i| (i % 10) as f32 * 0.1).collect();

    // Call dispatch - should use parallel path
    let result = matmul_q6k_f32_dispatch(&q6k_data, &input, out_dim, in_dim);

    // Verify dimensions and finiteness
    assert_eq!(result.len(), out_dim);
    for (i, &val) in result.iter().enumerate() {
        assert!(val.is_finite(), "Result[{}] is not finite: {}", i, val);
    }

    // Compare a few rows against scalar for consistency
    let scalar_result = matmul_q6k_f32_scalar(&q6k_data, &input, out_dim, in_dim);
    for i in (0..out_dim).step_by(512) {
        let diff = (result[i] - scalar_result[i]).abs();
        let tol = scalar_result[i].abs() * 0.01 + 1e-4;
        assert!(
            diff < tol,
            "Parallel vs scalar mismatch at row {}: parallel={}, scalar={}, diff={}",
            i,
            result[i],
            scalar_result[i],
            diff
        );
    }
}

#[test]
#[allow(deprecated)]
fn test_parallel_colmajor_large_matrix() {
    // Test colmajor path
    // ne0 = output dimension, ne1 = input dimension
    let ne0 = 2048; // output dimension, must be multiple of 256
    let ne1 = 4096; // input dimension

    let blocks_per_col = (ne0 + SUPER_BLOCK_SIZE - 1) / SUPER_BLOCK_SIZE;
    let col_bytes = blocks_per_col * SUPER_BLOCK_BYTES;
    let total_bytes = ne1 * col_bytes;

    let mut q6k_data = vec![0u8; total_bytes];
    for col in 0..ne1 {
        for sb in 0..blocks_per_col {
            let offset = col * col_bytes + sb * SUPER_BLOCK_BYTES;
            // d = 0.5 as f16
            q6k_data[offset] = 0x00;
            q6k_data[offset + 1] = 0x38;
            // ql
            for i in 0..128 {
                q6k_data[offset + 2 + i] = ((col ^ sb ^ i) % 64) as u8;
            }
            // qh
            for i in 0..64 {
                q6k_data[offset + 130 + i] = ((col + sb) % 4) as u8;
            }
            // scales
            for i in 0..16 {
                q6k_data[offset + 194 + i] = 0x20;
            }
        }
    }

    // Input must have length ne1
    let input: Vec<f32> = (0..ne1).map(|i| ((i % 7) as f32 - 3.0) * 0.1).collect();

    // Use colmajor dispatch
    let result = matmul_q6k_f32_colmajor_dispatch(&q6k_data, &input, ne0, ne1);

    // Output has ne0 elements
    assert_eq!(result.len(), ne0);
    for (i, &val) in result.iter().enumerate() {
        assert!(val.is_finite(), "Result[{}] is not finite: {}", i, val);
    }
}

#[test]
fn test_compute_chunk_scalar_small() {
    // Directly test compute_chunk_scalar
    let in_dim = 256;
    let out_dim = 4;
    let num_blocks_per_row = 1;
    let row_bytes = SUPER_BLOCK_BYTES;

    let mut q6k_data = vec![0u8; out_dim * row_bytes];
    for row in 0..out_dim {
        let offset = row * row_bytes;
        // d = 1.0 as f16
        q6k_data[offset] = 0x00;
        q6k_data[offset + 1] = 0x3C;
        // ql = all zeros
        for i in 0..128 {
            q6k_data[offset + 2 + i] = 0x00;
        }
        // qh = all zeros
        for i in 0..64 {
            q6k_data[offset + 130 + i] = 0x00;
        }
        // scales = 1
        for i in 0..16 {
            q6k_data[offset + 194 + i] = 0x01;
        }
    }

    let input = vec![1.0f32; in_dim];
    let mut chunk = vec![0.0f32; out_dim];

    compute_chunk_scalar(
        &q6k_data,
        &input,
        &mut chunk,
        0,
        out_dim,
        in_dim,
        num_blocks_per_row,
        row_bytes,
    );

    // Verify results are finite
    for (i, &val) in chunk.iter().enumerate() {
        assert!(val.is_finite(), "Chunk[{}] is not finite: {}", i, val);
    }
}

/// Test compute_chunk_scalar with start_row offset (exercises the
/// `out_idx = start_row + local_idx` path and skips first rows)
#[test]
fn test_compute_chunk_scalar_with_offset() {
    let in_dim = 256;
    let out_dim = 4;
    let num_blocks_per_row = 1;
    let row_bytes = SUPER_BLOCK_BYTES;

    // Build Q6K data: ql[128], qh[64], scales[16], d[2] = 210 bytes per row
    let mut q6k_data = Vec::new();
    for row in 0..out_dim {
        // ql: varying values
        for i in 0..128 {
            q6k_data.push(((row * 17 + i) % 256) as u8);
        }
        // qh: varying
        for i in 0..64 {
            q6k_data.push(((row * 7 + i) % 256) as u8);
        }
        // scales
        q6k_data.extend_from_slice(&[0x02u8; 16]);
        // d = 1.0 as f16
        q6k_data.extend_from_slice(&[0x00, 0x3C]);
    }

    let input: Vec<f32> = (0..in_dim).map(|i| (i as f32) * 0.01).collect();

    // Process only the last 2 rows (start_row=2)
    let mut chunk = vec![0.0f32; 2];
    compute_chunk_scalar(
        &q6k_data,
        &input,
        &mut chunk,
        2, // start_row offset
        out_dim,
        in_dim,
        num_blocks_per_row,
        row_bytes,
    );

    for &val in &chunk {
        assert!(val.is_finite());
    }
}

/// Test compute_chunk_scalar where out_idx exceeds out_dim (early break)
#[test]
fn test_compute_chunk_scalar_exceeds_outdim() {
    let in_dim = 256;
    let out_dim = 2;
    let num_blocks_per_row = 1;
    let row_bytes = SUPER_BLOCK_BYTES;

    let mut q6k_data = Vec::new();
    for _ in 0..out_dim {
        q6k_data.extend_from_slice(&[0x33u8; 128]); // ql
        q6k_data.extend_from_slice(&[0x11u8; 64]); // qh
        q6k_data.extend_from_slice(&[0x01u8; 16]); // scales
        q6k_data.extend_from_slice(&[0x00, 0x3C]); // d = 1.0
    }

    let input = vec![1.0f32; in_dim];

    // Chunk has 4 slots but out_dim=2, so only 2 should be written
    let mut chunk = vec![0.0f32; 4];
    compute_chunk_scalar(
        &q6k_data,
        &input,
        &mut chunk,
        0,
        out_dim,
        in_dim,
        num_blocks_per_row,
        row_bytes,
    );

    // First 2 should be populated, last 2 should remain zero
    for i in 0..2 {
        assert!(chunk[i].is_finite());
    }
    assert_eq!(chunk[2], 0.0, "Elements beyond out_dim should remain zero");
    assert_eq!(chunk[3], 0.0, "Elements beyond out_dim should remain zero");
}

/// Test scalar path with multiple super-blocks per row
#[test]
fn test_matmul_q6k_scalar_multiple_blocks() {
    let in_dim = 512; // 2 super-blocks
    let out_dim = 2;
    let num_blocks = 2;

    let mut q6k_data = Vec::new();
    for _ in 0..out_dim {
        for _ in 0..num_blocks {
            q6k_data.extend_from_slice(&[0x44u8; 128]); // ql
            q6k_data.extend_from_slice(&[0x00u8; 64]); // qh
            q6k_data.extend_from_slice(&[0x02u8; 16]); // scales = 2
            q6k_data.extend_from_slice(&[0x00, 0x3C]); // d = 1.0
        }
    }

    let input: Vec<f32> = (0..in_dim).map(|i| (i as f32) * 0.001).collect();
    let output_scalar = matmul_q6k_f32_scalar(&q6k_data, &input, out_dim, in_dim);
    let output_dispatch = matmul_q6k_f32_dispatch(&q6k_data, &input, out_dim, in_dim);

    assert_eq!(output_scalar.len(), out_dim);
    assert_eq!(output_dispatch.len(), out_dim);

    for (i, (s, d)) in output_scalar.iter().zip(output_dispatch.iter()).enumerate() {
        let diff = (s - d).abs();
        assert!(diff < 1e-3, "Row {}: scalar={} vs dispatch={}, diff={}", i, s, d, diff);
    }
}
