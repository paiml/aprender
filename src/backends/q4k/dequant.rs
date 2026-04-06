//! Q4_K dequantization to F32.
//!
//! Provides full dequantization of Q4_K compressed data for golden test comparison.

use super::{parse_q4k_header, SUPER_BLOCK_BYTES, SUPER_BLOCK_SIZE};

/// Dequantize Q4_K data to F32 (for golden test comparison)
///
/// This function fully dequantizes Q4K data to F32, matching the
/// `dequantize_q4_k_to_f32` in aprender/src/format/converter.rs.
pub fn dequantize_q4k_to_f32(data: &[u8], num_elements: usize) -> Vec<f32> {
    contract_pre_dequant!();
    let num_blocks = (num_elements + SUPER_BLOCK_SIZE - 1) / SUPER_BLOCK_SIZE;
    let mut result = vec![0.0f32; num_blocks * SUPER_BLOCK_SIZE];

    for sb_idx in 0..num_blocks {
        let sb_start = sb_idx * SUPER_BLOCK_BYTES;
        let out_start = sb_idx * SUPER_BLOCK_SIZE;

        if sb_start + SUPER_BLOCK_BYTES > data.len() {
            break;
        }

        let sb_data = &data[sb_start..sb_start + SUPER_BLOCK_BYTES];
        let (d, dmin, scales, mins) = parse_q4k_header(sb_data);
        let qs = sb_data.get(16..144).expect("Q4_K: need ≥144 bytes for qs");

        let mut ys_index = out_start;

        for chunk in 0..4 {
            let q = &qs[chunk * 32..(chunk + 1) * 32];

            let scale_idx_low = chunk * 2;
            let scale_idx_high = chunk * 2 + 1;

            let d1 = d * f32::from(scales[scale_idx_low]);
            let dm1 = dmin * f32::from(mins[scale_idx_low]);
            let d2 = d * f32::from(scales[scale_idx_high]);
            let dm2 = dmin * f32::from(mins[scale_idx_high]);

            // First pass: 32 low nibbles
            for &byte in q {
                result[ys_index] = d1 * (byte & 0xF) as f32 - dm1;
                ys_index += 1;
            }

            // Second pass: 32 high nibbles
            for &byte in q {
                result[ys_index] = d2 * (byte >> 4) as f32 - dm2;
                ys_index += 1;
            }
        }
    }

    result.truncate(num_elements);
    contract_post_dequant!(result);
    result
}
