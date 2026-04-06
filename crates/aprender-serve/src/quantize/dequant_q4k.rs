
/// Dequantize `Q4_K` format weights
pub fn dequantize_q4_k(data: &[u8]) -> Result<Vec<f32>> {
    const SUPER_BLOCK_BYTES: usize = 144;

    if !data.len().is_multiple_of(SUPER_BLOCK_BYTES) {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Q4_K data length {} is not a multiple of super-block size {}",
                data.len(),
                SUPER_BLOCK_BYTES
            ),
        });
    }

    let num_super_blocks = data.len() / SUPER_BLOCK_BYTES;
    let mut result = vec![0.0f32; num_super_blocks * QK_K];

    for sb_idx in 0..num_super_blocks {
        let sb_start = sb_idx * SUPER_BLOCK_BYTES;
        let out_start = sb_idx * QK_K;

        let d = read_f16(&data[sb_start..sb_start + 2]);
        let dmin = read_f16(&data[sb_start + 2..sb_start + 4]);

        let mut scales = [0u8; 12];
        scales.copy_from_slice(&data[sb_start + 4..sb_start + 16]);

        let qs_start = sb_start + 16;
        let qs = &data[qs_start..qs_start + 128];

        let mut ys_index = out_start;

        for j in (0..QK_K).step_by(64) {
            let q = &qs[j / 2..j / 2 + 32];

            let is = j / 32;
            let (sc1, m1) = extract_scale_min(&scales, is);
            let d1 = d * sc1;
            let dm1 = dmin * m1;

            let (sc2, m2) = extract_scale_min(&scales, is + 1);
            let d2 = d * sc2;
            let dm2 = dmin * m2;

            for &byte in q {
                result[ys_index] = d1 * (byte & 0xF) as f32 - dm1;
                ys_index += 1;
            }

            for &byte in q {
                result[ys_index] = d2 * (byte >> 4) as f32 - dm2;
                ys_index += 1;
            }
        }
    }

    Ok(result)
}

/// Dequantize `Q5_K` format weights
pub fn dequantize_q5_k(data: &[u8]) -> Result<Vec<f32>> {
    const SUPER_BLOCK_BYTES: usize = 176;

    if !data.len().is_multiple_of(SUPER_BLOCK_BYTES) {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Q5_K data length {} is not a multiple of super-block size {}",
                data.len(),
                SUPER_BLOCK_BYTES
            ),
        });
    }

    let num_super_blocks = data.len() / SUPER_BLOCK_BYTES;
    let mut result = Vec::with_capacity(num_super_blocks * QK_K);

    for sb_idx in 0..num_super_blocks {
        let sb_start = sb_idx * SUPER_BLOCK_BYTES;

        let d = read_f16(&data[sb_start..sb_start + 2]);
        let dmin = read_f16(&data[sb_start + 2..sb_start + 4]);

        let mut scales = [0u8; 12];
        scales.copy_from_slice(&data[sb_start + 4..sb_start + 16]);

        let qh_start = sb_start + 16;
        let qh = &data[qh_start..qh_start + 32];

        let qs_low_start = sb_start + 48;
        let qs = &data[qs_low_start..qs_low_start + 128];

        for block_idx in 0..8 {
            let (scale, min) = extract_scale_min(&scales, block_idx);

            let block_start = block_idx * 16;
            let qh_block_start = block_idx * 4;

            for byte_idx in 0..16 {
                let qs_byte = qs[block_start + byte_idx];

                let high_bits_byte = qh[qh_block_start + byte_idx / 4];
                let bit_offset = (byte_idx % 4) * 2;

                let q_low_4bit = qs_byte & 0x0F;
                let q_low_high_bit = (high_bits_byte >> bit_offset) & 0x01;
                #[allow(clippy::cast_possible_wrap)]
                let q_low = ((q_low_high_bit << 4) | q_low_4bit) as i8;
                let value_low = d * scale * f32::from(q_low) - dmin * min;
                result.push(value_low);

                let q_high_4bit = (qs_byte >> 4) & 0x0F;
                let q_high_high_bit = (high_bits_byte >> (bit_offset + 1)) & 0x01;
                #[allow(clippy::cast_possible_wrap)]
                let q_high = ((q_high_high_bit << 4) | q_high_4bit) as i8;
                let value_high = d * scale * f32::from(q_high) - dmin * min;
                result.push(value_high);
            }
        }
    }

    Ok(result)
}

/// Dequantize `Q6_K` format weights
pub fn dequantize_q6_k(data: &[u8]) -> Result<Vec<f32>> {
    const SUPER_BLOCK_BYTES: usize = 210;

    if !data.len().is_multiple_of(SUPER_BLOCK_BYTES) {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Q6_K data length {} is not a multiple of super-block size {}",
                data.len(),
                SUPER_BLOCK_BYTES
            ),
        });
    }

    let num_super_blocks = data.len() / SUPER_BLOCK_BYTES;
    let mut result = vec![0.0f32; num_super_blocks * QK_K];

    for sb_idx in 0..num_super_blocks {
        let sb_start = sb_idx * SUPER_BLOCK_BYTES;
        let out_start = sb_idx * QK_K;

        let ql = &data[sb_start..sb_start + 128];
        let qh = &data[sb_start + 128..sb_start + 192];

        let mut scales = [0i8; 16];
        for (i, scale) in scales.iter_mut().enumerate() {
            #[allow(clippy::cast_possible_wrap)]
            {
                *scale = data[sb_start + 192 + i] as i8;
            }
        }

        let d = read_f16(&data[sb_start + 208..sb_start + 210]);

        for n in (0..QK_K).step_by(128) {
            let idx = n / 128;
            let sc = &scales[8 * idx..];
            let ql_slice = &ql[64 * idx..];
            let qh_slice = &qh[32 * idx..];

            for l in 0..32 {
                let is = l / 16;

                let q1 = ((ql_slice[l] & 0xF) | ((qh_slice[l] & 3) << 4)) as i32 - 32;
                let q2 = ((ql_slice[l + 32] & 0xF) | (((qh_slice[l] >> 2) & 3) << 4)) as i32 - 32;
                let q3 = ((ql_slice[l] >> 4) | (((qh_slice[l] >> 4) & 3) << 4)) as i32 - 32;
                let q4 = ((ql_slice[l + 32] >> 4) | (((qh_slice[l] >> 6) & 3) << 4)) as i32 - 32;

                result[out_start + n + l] = d * (sc[is] as f32) * (q1 as f32);
                result[out_start + n + l + 32] = d * (sc[is + 2] as f32) * (q2 as f32);
                result[out_start + n + l + 64] = d * (sc[is + 4] as f32) * (q3 as f32);
                result[out_start + n + l + 96] = d * (sc[is + 6] as f32) * (q4 as f32);
            }
        }
    }

    Ok(result)
}

/// Dequantize `Q2_K` format weights
pub fn dequantize_q2_k(data: &[u8]) -> Result<Vec<f32>> {
    const SUPER_BLOCK_BYTES: usize = 84;

    if !data.len().is_multiple_of(SUPER_BLOCK_BYTES) {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Q2_K data length {} is not a multiple of super-block size {}",
                data.len(),
                SUPER_BLOCK_BYTES
            ),
        });
    }

    let num_super_blocks = data.len() / SUPER_BLOCK_BYTES;
    let mut result = vec![0.0f32; num_super_blocks * QK_K];

    for sb_idx in 0..num_super_blocks {
        let sb_start = sb_idx * SUPER_BLOCK_BYTES;
        let out_start = sb_idx * QK_K;

        let scales_data = &data[sb_start..sb_start + 16];
        let qs = &data[sb_start + 16..sb_start + 80];

        let d = read_f16(&data[sb_start + 80..sb_start + 82]);
        let dmin = read_f16(&data[sb_start + 82..sb_start + 84]);

        for j in 0..16 {
            let sc = (scales_data[j] & 0x0F) as f32;
            let m = (scales_data[j] >> 4) as f32;

            let d_sc = d * sc;
            let dm = dmin * m;

            let qs_offset = j * 4;

            for k in 0..4 {
                let q_byte = qs[qs_offset + k];
                let q0 = (q_byte & 0x03) as f32;
                let q1 = ((q_byte >> 2) & 0x03) as f32;
                let q2 = ((q_byte >> 4) & 0x03) as f32;
                let q3 = ((q_byte >> 6) & 0x03) as f32;

                let base_idx = out_start + j * 16 + k * 4;
                result[base_idx] = d_sc * q0 - dm;
                result[base_idx + 1] = d_sc * q1 - dm;
                result[base_idx + 2] = d_sc * q2 - dm;
                result[base_idx + 3] = d_sc * q3 - dm;
            }
        }
    }

    Ok(result)
}

/// Helper: Read f16 from bytes and convert to f32
#[inline]
pub(crate) fn read_f16(bytes: &[u8]) -> f32 {
    let bits = u16::from_le_bytes([bytes[0], bytes[1]]);
    half::f16::from_bits(bits).to_f32()
}
