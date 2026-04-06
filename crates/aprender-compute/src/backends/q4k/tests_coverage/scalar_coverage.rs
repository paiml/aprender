use super::super::gemv::compute_chunk_q4k_scalar;
use super::super::*;

// =========================================================================
// Additional Q4K scalar coverage tests
// =========================================================================

/// Test compute_chunk_q4k_scalar with start_row offset
#[test]
fn test_compute_chunk_scalar_with_offset() {
    let in_dim = 256;
    let out_dim = 4;
    let num_blocks_per_row = 1;
    let row_bytes = SUPER_BLOCK_BYTES;

    let mut q4k_data = vec![0u8; out_dim * row_bytes];
    for row in 0..out_dim {
        let offset = row * row_bytes;
        q4k_data[offset] = 0x00;
        q4k_data[offset + 1] = 0x3C; // d = 1.0
        q4k_data[offset + 2] = 0x00;
        q4k_data[offset + 3] = 0x00; // dmin = 0.0
        for i in 0..12 {
            q4k_data[offset + 4 + i] = 0x01;
        }
        q4k_data[offset + 16..offset + 144].fill(0x55); // qs = 5|5
    }

    let input: Vec<f32> = (0..in_dim).map(|i| (i as f32) * 0.01).collect();

    // Process only the last 2 rows (start_row=2)
    let mut chunk = vec![0.0f32; 2];
    compute_chunk_q4k_scalar(
        &q4k_data,
        &input,
        &mut chunk,
        2, // start_row
        out_dim,
        in_dim,
        num_blocks_per_row,
        row_bytes,
    );

    for &val in &chunk {
        assert!(val.is_finite());
    }
}

/// Test compute_chunk_q4k_scalar where out_idx exceeds out_dim (early break)
#[test]
fn test_compute_chunk_scalar_exceeds_outdim() {
    let in_dim = 256;
    let out_dim = 2;
    let num_blocks_per_row = 1;
    let row_bytes = SUPER_BLOCK_BYTES;

    let mut q4k_data = vec![0u8; out_dim * row_bytes];
    for row in 0..out_dim {
        let offset = row * row_bytes;
        q4k_data[offset] = 0x00;
        q4k_data[offset + 1] = 0x3C;
        q4k_data[offset + 2] = 0x00;
        q4k_data[offset + 3] = 0x00;
        for i in 0..12 {
            q4k_data[offset + 4 + i] = 0x01;
        }
        q4k_data[offset + 16..offset + 144].fill(0x33);
    }

    let input = vec![1.0f32; in_dim];

    // Chunk has 4 slots but out_dim=2, so only 2 should be written
    let mut chunk = vec![0.0f32; 4];
    compute_chunk_q4k_scalar(
        &q4k_data,
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
fn test_matmul_q4k_scalar_multiple_blocks() {
    let in_dim = 512; // 2 super-blocks
    let out_dim = 2;
    let num_blocks = 2;

    let mut q4k_data = Vec::new();
    for _ in 0..out_dim {
        for _ in 0..num_blocks {
            q4k_data.extend_from_slice(&[0x00, 0x3C]); // d = 1.0
            q4k_data.extend_from_slice(&[0x00, 0x00]); // dmin = 0
            q4k_data.extend_from_slice(&[0x02u8; 12]); // scales = 2
            q4k_data.extend_from_slice(&[0x88u8; 128]); // qs = 8|8
        }
    }

    let input: Vec<f32> = (0..in_dim).map(|i| (i as f32) * 0.001).collect();
    let output_scalar = matmul_q4k_f32_scalar(&q4k_data, &input, out_dim, in_dim);
    let output_optimized = matmul_q4k_f32(&q4k_data, &input, out_dim, in_dim);

    assert_eq!(output_scalar.len(), out_dim);
    assert_eq!(output_optimized.len(), out_dim);

    for (i, (s, o)) in output_scalar.iter().zip(output_optimized.iter()).enumerate() {
        let diff = (s - o).abs();
        assert!(diff < 1e-3, "Row {}: scalar={} vs optimized={}, diff={}", i, s, o, diff);
    }
}
