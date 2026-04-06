use super::super::gemv::compute_chunk_q4k_scalar;
use super::super::*;

// =========================================================================
// BH-MUT boundary mutation tests (trueno #100)
// These tests detect off-by-one mutations in critical boundary constants.
// =========================================================================

/// BH-MUT-1: dequant data boundary -- sb_start + SUPER_BLOCK_BYTES > data.len()
///
/// Validates that dequantize_q4k_to_f32 handles exact-length data correctly
/// and stops at the boundary. A mutation from `>` to `>=` would incorrectly
/// skip the last valid superblock.
#[test]
fn test_bh_mut_dequant_data_length_boundary() {
    use super::super::dequant::dequantize_q4k_to_f32;

    // Build exactly 1 superblock of valid data (144 bytes)
    let mut data = vec![0u8; SUPER_BLOCK_BYTES];
    data[0] = 0x00;
    data[1] = 0x3C; // d = 1.0 in f16
    data[2] = 0x00;
    data[3] = 0x00; // dmin = 0.0
    for i in 4..16 {
        data[i] = 0x01; // scales
    }
    for i in 16..144 {
        data[i] = 0x11; // qs = 1|1 (both nibbles = 1)
    }

    // Exact boundary: data.len() == SUPER_BLOCK_BYTES
    let result = dequantize_q4k_to_f32(&data, 256);
    let non_zero = result.iter().filter(|&&v| v != 0.0).count();
    assert!(non_zero > 0, "Exact-boundary superblock must be processed");

    // One byte short: data.len() == SUPER_BLOCK_BYTES - 1
    let short_data = &data[..SUPER_BLOCK_BYTES - 1];
    let result_short = dequantize_q4k_to_f32(short_data, 256);
    let non_zero_short = result_short.iter().filter(|&&v| v != 0.0).count();
    assert_eq!(non_zero_short, 0, "Short data must not produce dequantized values");
}

/// BH-MUT-2: chunk loop bound -- for chunk in 0..4
///
/// Validates that exactly 4 chunks of 64 values (256 total) are processed
/// per superblock. A mutation from 4->3 would lose 64 values; 4->5 would
/// access out-of-bounds data.
#[test]
fn test_bh_mut_chunk_count_boundary() {
    let in_dim = 256;
    let out_dim = 1;
    let mut q4k_data = vec![0u8; SUPER_BLOCK_BYTES];
    q4k_data[0] = 0x00;
    q4k_data[1] = 0x3C; // d = 1.0
    q4k_data[2] = 0x00;
    q4k_data[3] = 0x00; // dmin = 0.0
    for i in 4..16 {
        q4k_data[i] = 0x01;
    }
    // Set distinct nibble patterns per chunk so each chunk contributes differently
    for chunk in 0..4u8 {
        let base = 16 + chunk as usize * 32;
        for i in 0..32 {
            q4k_data[base + i] = (chunk + 1) | ((chunk + 2) << 4);
        }
    }

    // Input: all 1.0 so dot product = sum of dequantized values
    let input = vec![1.0f32; in_dim];
    let result = matmul_q4k_f32_scalar(&q4k_data, &input, out_dim, in_dim);

    // Compute expected: sum over all 4 chunks
    let mut expected_sum = 0.0f32;
    for chunk in 0..4u8 {
        let d = 1.0f32;
        let d1 = d * 1.0; // scale = 0x01
        let d2 = d * 1.0;
        let low_nib = (chunk + 1) as f32;
        let high_nib = (chunk + 2) as f32;
        expected_sum += d1 * low_nib * 32.0;
        expected_sum += d2 * high_nib * 32.0;
    }

    // If chunk count mutated to 3, result would be ~75% of expected
    let ratio = result[0] / expected_sum;
    assert!(
        (ratio - 1.0).abs() < 0.01,
        "Expected ratio ~1.0, got {ratio:.4} (result={}, expected={expected_sum})",
        result[0]
    );
}

/// BH-MUT-3: out_idx >= out_dim early termination
///
/// Validates that compute_chunk_q4k_scalar stops exactly at out_dim.
/// A mutation from `>=` to `>` would write one element past the boundary.
#[test]
fn test_bh_mut_out_idx_boundary() {
    let in_dim = 256;
    let out_dim = 3;
    let num_blocks_per_row = 1;
    let row_bytes = SUPER_BLOCK_BYTES;

    // Build 4 rows of Q4K data (chunk has 4 slots but out_dim=3)
    let mut q4k_data = vec![0u8; 4 * row_bytes];
    for row in 0..4 {
        let offset = row * row_bytes;
        q4k_data[offset] = 0x00;
        q4k_data[offset + 1] = 0x3C;
        q4k_data[offset + 2] = 0x00;
        q4k_data[offset + 3] = 0x00;
        for i in 4..16 {
            q4k_data[offset + i] = 0x01;
        }
        q4k_data[offset + 16..offset + 144].fill(0x11);
    }

    let input = vec![1.0f32; in_dim];

    // Chunk has 4 slots but out_dim=3, so index 3 should NOT be written
    let mut chunk = vec![f32::NAN; 4];
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

    // First 3 elements should be computed (finite)
    for i in 0..3 {
        assert!(chunk[i].is_finite(), "chunk[{i}] should be computed, got {}", chunk[i]);
    }
    // Fourth element should still be NAN (out_idx=3 >= out_dim=3)
    assert!(chunk[3].is_nan(), "chunk[3] should be untouched (NAN), got {}", chunk[3]);
}
