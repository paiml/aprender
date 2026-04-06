
/// Fused Q4_K with Q8 blocks dot product
///
/// Computes the dot product of Q4_K quantized weights with Q8_0 quantized activations.
pub fn fused_q4k_q8_dot(q4k_data: &[u8], q8_blocks: &[Q8_0Block]) -> Result<f32> {
    const SUPER_BLOCK_BYTES: usize = 144;

    // Validate Q4_K data length
    if !q4k_data.len().is_multiple_of(SUPER_BLOCK_BYTES) {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Q4_K data length {} is not a multiple of super-block size {}",
                q4k_data.len(),
                SUPER_BLOCK_BYTES
            ),
        });
    }

    let num_super_blocks = q4k_data.len() / SUPER_BLOCK_BYTES;
    let expected_values = num_super_blocks * QK_K; // 256 values per super-block
    let expected_q8_blocks = expected_values / 32;

    // Validate Q8 block count matches
    if q8_blocks.len() != expected_q8_blocks {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Q8_0 block count {} doesn't match expected {} (for {} Q4_K values)",
                q8_blocks.len(),
                expected_q8_blocks,
                expected_values
            ),
        });
    }

    // Accumulator for dot product result
    let mut acc = 0.0f32;
    let mut q8_block_idx = 0;

    for sb_idx in 0..num_super_blocks {
        let sb_start = sb_idx * SUPER_BLOCK_BYTES;

        // Read d (f16 -> f32) - super-block scale
        let d = read_f16(&q4k_data[sb_start..sb_start + 2]);

        // Read dmin (f16 -> f32) - super-block min
        let dmin = read_f16(&q4k_data[sb_start + 2..sb_start + 4]);

        // Read scales (12 bytes) - packed 6-bit scales for 8 blocks
        let mut scales = [0u8; 12];
        scales.copy_from_slice(&q4k_data[sb_start + 4..sb_start + 16]);

        // Read qs (128 bytes) - 256 4-bit quantized values
        let qs_start = sb_start + 16;
        let qs = &q4k_data[qs_start..qs_start + 128];

        // Process 8 blocks of 32 values each
        for block_idx in 0..8 {
            // Extract 6-bit scale and min for this block
            let (scale, min) = extract_scale_min(&scales, block_idx);

            // Get the Q8 block for this 32-value chunk
            let q8_block = &q8_blocks[q8_block_idx];
            let q8_scale = q8_block.scale;
            q8_block_idx += 1;

            // Process 32 values (16 bytes, 2 4-bit values per byte)
            let block_start = block_idx * 16;
            for byte_idx in 0..16 {
                let byte = qs[block_start + byte_idx];
                let q8_idx = byte_idx * 2;

                // Low 4 bits: fused dequant and accumulate
                #[allow(clippy::cast_possible_wrap)]
                let q4_low = (byte & 0x0F) as i8;
                let w_low = d * scale * f32::from(q4_low) - dmin * min;
                let a_low = q8_scale * f32::from(q8_block.quants[q8_idx]);
                acc += w_low * a_low;

                // High 4 bits: fused dequant and accumulate
                #[allow(clippy::cast_possible_wrap)]
                let q4_high = ((byte >> 4) & 0x0F) as i8;
                let w_high = d * scale * f32::from(q4_high) - dmin * min;
                let a_high = q8_scale * f32::from(q8_block.quants[q8_idx + 1]);
                acc += w_high * a_high;
            }
        }
    }

    Ok(acc)
}
