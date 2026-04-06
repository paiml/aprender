//! Dequantization helpers for APR quantized tensor formats (PMAT-COMPLY)
//!
//! Extracted from mod.rs for file health compliance.

/// Convert F16 (IEEE 754 half-precision) to F32
///
/// ONE PATH: Delegates to `trueno::f16_to_f32` (UCBD §4).
#[inline]
pub fn f16_to_f32(bits: u16) -> f32 {
    trueno::f16_to_f32(bits)
}

/// Dequantize F16 data to F32
pub fn dequantize_f16(bytes: &[u8], num_elements: usize) -> Vec<f32> {
    let mut result = Vec::with_capacity(num_elements);
    for chunk in bytes.chunks_exact(2) {
        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
        result.push(f16_to_f32(bits));
    }
    result.truncate(num_elements);
    result
}

/// Dequantize Q8_0 format (GGUF compatible)
/// Q8_0: blocks of 32 elements, each block has 2-byte f16 scale + 32 bytes of int8 quants
pub fn dequantize_q8_0(bytes: &[u8], num_elements: usize) -> Vec<f32> {
    const BLOCK_SIZE: usize = 32;
    const BLOCK_BYTES: usize = 2 + 32; // f16 scale + 32 int8 values

    let mut result = Vec::with_capacity(num_elements);
    let mut offset = 0;

    while result.len() < num_elements && offset + BLOCK_BYTES <= bytes.len() {
        // Read scale (f16)
        let scale_bits = u16::from_le_bytes([bytes[offset], bytes[offset + 1]]);
        let scale = f16_to_f32(scale_bits);
        offset += 2;

        // Read 32 int8 values
        for i in 0..BLOCK_SIZE {
            if result.len() >= num_elements {
                break;
            }
            let v = f32::from(bytes[offset + i] as i8);
            result.push(v * scale);
        }
        offset += 32;
    }

    result.truncate(num_elements);
    result
}

/// Extract and dequantize 32 nibbles from Q4_K bytes
#[inline]
fn push_q4k_nibbles(
    result: &mut Vec<f32>,
    num_elements: usize,
    bytes: &[u8],
    d_scale: f32,
    d_min: f32,
    shift: u8,
) {
    for &byte in bytes {
        if result.len() >= num_elements {
            break;
        }
        let q_val = ((byte >> shift) & 0x0F) as f32;
        result.push(d_scale * q_val - d_min);
    }
}

/// Dequantize Q4_K format (GGUF K-quants)
/// Q4_K: super blocks of 256 elements
/// Each super block: d (f16) + dmin (f16) + scales (12 bytes) + qs (128 bytes) = 144 bytes
///
/// LAYOUT-001 FIX: Element ordering must match fused_q4k_dot (PAR-001):
/// - 4 chunks of 64 values each (at offsets 0, 64, 128, 192)
/// - Each chunk: 32 low nibbles (scale is), then 32 high nibbles (scale is+1)
/// - NOT interleaved (L0, H0, L1, H1...) - that was the bug!
pub fn dequantize_q4_k(bytes: &[u8], num_elements: usize) -> Vec<f32> {
    const QK_K: usize = 256;
    const SUPER_BLOCK_BYTES: usize = 2 + 2 + 12 + 128; // 144 bytes

    let mut result = Vec::with_capacity(num_elements);
    let mut offset = 0;

    while result.len() < num_elements && offset + SUPER_BLOCK_BYTES <= bytes.len() {
        let d = f16_to_f32(u16::from_le_bytes([bytes[offset], bytes[offset + 1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([bytes[offset + 2], bytes[offset + 3]]));
        offset += 4;

        let mut scales = [0u8; 12];
        scales.copy_from_slice(&bytes[offset..offset + 12]);
        offset += 12;

        let qs = &bytes[offset..offset + 128];
        offset += 128;

        // PAR-001: 4 chunks of 64 values (low nibbles then high nibbles from 32 bytes)
        for j in (0..QK_K).step_by(64) {
            let q = &qs[j / 2..j / 2 + 32];
            let is = j / 32;

            let (sc1, m1) = extract_scale_min_q4k(&scales, is);
            push_q4k_nibbles(&mut result, num_elements, q, d * sc1, dmin * m1, 0);

            let (sc2, m2) = extract_scale_min_q4k(&scales, is + 1);
            push_q4k_nibbles(&mut result, num_elements, q, d * sc2, dmin * m2, 4);
        }
    }

    result.truncate(num_elements);
    result
}

/// Extract scale and min for Q4_K block index (0-7)
/// Matches fused_k.rs extract_scale_min exactly
#[inline]
fn extract_scale_min_q4k(scales: &[u8; 12], block_idx: usize) -> (f32, f32) {
    let j = block_idx;
    let (scale_bits, min_bits) = if j < 4 {
        // First 4 blocks: simple layout
        let d = scales[j] & 63;
        let m = scales[j + 4] & 63;
        (d, m)
    } else {
        // Last 4 blocks: packed layout using high bits from first 4 bytes
        let d = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
        let m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
        (d, m)
    };

    (f32::from(scale_bits), f32::from(min_bits))
}

/// Dequantize one Q6_K quadrant (32 values) using the provided bit extraction function
#[inline]
#[allow(clippy::cast_possible_wrap)]
fn dequantize_q6k_quadrant(
    result: &mut Vec<f32>,
    num_elements: usize,
    d: f32,
    sc: &[i8],
    sc_offset: usize,
    extract_q: impl Fn(usize) -> i32,
) {
    for l in 0..32 {
        if result.len() >= num_elements {
            break;
        }
        let is = l / 16;
        let q = extract_q(l);
        result.push(d * (sc[is + sc_offset] as f32) * (q as f32));
    }
}

/// Dequantize Q6_K format (GGUF K-quants)
/// Q6_K: super blocks of 256 elements
/// Each super block: ql (128 bytes) + qh (64 bytes) + scales (16 bytes) + d (f16) = 210 bytes
///
/// LAYOUT-001 FIX: Element ordering must match fused_q6k_dot (candle compatible):
/// - Process 128 values at a time (n=0, n=128)
/// - For each l in 0..32, extract 4 values at positions n+l, n+l+32, n+l+64, n+l+96
#[allow(clippy::cast_possible_wrap)]
pub fn dequantize_q6_k(bytes: &[u8], num_elements: usize) -> Vec<f32> {
    const QK_K: usize = 256;
    const SUPER_BLOCK_BYTES: usize = 128 + 64 + 16 + 2; // 210 bytes

    let mut result = Vec::with_capacity(num_elements);
    let mut offset = 0;

    while result.len() < num_elements && offset + SUPER_BLOCK_BYTES <= bytes.len() {
        // Q6_K layout: ql (128) + qh (64) + scales (16) + d (2)
        let ql = &bytes[offset..offset + 128];
        offset += 128;

        let qh = &bytes[offset..offset + 64];
        offset += 64;

        // Read scales (16 bytes, signed i8)
        let mut scales = [0i8; 16];
        for (i, scale) in scales.iter_mut().enumerate() {
            *scale = bytes[offset + i] as i8;
        }
        offset += 16;

        // Read d (f16 -> f32)
        let d = f16_to_f32(u16::from_le_bytes([bytes[offset], bytes[offset + 1]]));
        offset += 2;

        // Process 128 values at a time (n=0, n=128)
        for n in (0..QK_K).step_by(128) {
            let idx = n / 128;
            let sc = &scales[8 * idx..];
            let ql_slice = &ql[64 * idx..];
            let qh_slice = &qh[32 * idx..];

            // q1: positions n+0..n+31
            dequantize_q6k_quadrant(&mut result, num_elements, d, sc, 0, |l| {
                ((ql_slice[l] & 0xF) | ((qh_slice[l] & 3) << 4)) as i32 - 32
            });
            // q2: positions n+32..n+63
            dequantize_q6k_quadrant(&mut result, num_elements, d, sc, 2, |l| {
                ((ql_slice[l + 32] & 0xF) | (((qh_slice[l] >> 2) & 3) << 4)) as i32 - 32
            });
            // q3: positions n+64..n+95
            dequantize_q6k_quadrant(&mut result, num_elements, d, sc, 4, |l| {
                ((ql_slice[l] >> 4) | (((qh_slice[l] >> 4) & 3) << 4)) as i32 - 32
            });
            // q4: positions n+96..n+127
            dequantize_q6k_quadrant(&mut result, num_elements, d, sc, 6, |l| {
                ((ql_slice[l + 32] >> 4) | (((qh_slice[l] >> 6) & 3) << 4)) as i32 - 32
            });
        }
    }

    result.truncate(num_elements);
    result
}

/// Map APR dtype string to GGML quantization type ID.
///
/// These IDs are used by `load_quantized_weights_with_type()` to select
/// the correct GPU dequantization kernel (Q4K GEMV, Q6K GEMV, etc.).
///
/// NOTE: APR-native formats (q8, q4) are NOT GGML types and return None.
/// They use different binary layouts and must be dequantized on CPU via
/// `dequantize_apr_q8()` / `dequantize_apr_q4()`.
#[inline]
pub fn dtype_to_ggml_qtype(dtype: &str) -> Option<u32> {
    // GH-321: Use unified GgmlQuantType enum for GGML-compatible formats.
    // APR-native Q8/Q4 are NOT GGML — different binary layout — returns None.
    // F32/F16/BF16 are not quantized — returns None.
    crate::gguf::GgmlQuantType::from_str_lossy(dtype)
        .filter(|qt| {
            matches!(
                qt,
                crate::gguf::GgmlQuantType::Q4_0
                    | crate::gguf::GgmlQuantType::Q4_1
                    | crate::gguf::GgmlQuantType::Q5_0
                    | crate::gguf::GgmlQuantType::Q5_1
                    | crate::gguf::GgmlQuantType::Q8_0
                    | crate::gguf::GgmlQuantType::Q8_1
                    | crate::gguf::GgmlQuantType::Q2K
                    | crate::gguf::GgmlQuantType::Q3K
                    | crate::gguf::GgmlQuantType::Q4K
                    | crate::gguf::GgmlQuantType::Q5K
                    | crate::gguf::GgmlQuantType::Q6K
                    | crate::gguf::GgmlQuantType::IQ2XXS
                    | crate::gguf::GgmlQuantType::IQ2XS
            )
        })
        .map(crate::gguf::GgmlQuantType::as_id)
}

/// Check if dtype is a quantized format that can use GPU dequant kernels.
#[inline]
pub fn is_quantized_dtype(dtype: &str) -> bool {
    dtype_to_ggml_qtype(dtype).is_some()
}

// ============================================================================
// APR-native quantization formats (different from GGML!)
// ============================================================================

/// Dequantize APR Q8 format (NOT the same as GGML Q8_0!)
///
/// APR Q8: `[scale: f32 (4 bytes)] + [quantized: i8 × N]` (single scale for whole tensor)
/// GGML Q8_0: `[scale: f16 (2 bytes)] + [i8 × 32]` per block of 32
///
/// Dequant: `value = quantized_i8 * scale`
pub fn dequantize_apr_q8(bytes: &[u8], num_elements: usize) -> Vec<f32> {
    if bytes.len() < 4 {
        return vec![0.0; num_elements];
    }

    let scale = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    let quant_bytes = bytes
        .get(4..)
        .expect("APR Q8 buffer validated to have at least 4 bytes above");

    let mut result = Vec::with_capacity(num_elements);
    for i in 0..num_elements.min(quant_bytes.len()) {
        let q = quant_bytes[i] as i8;
        result.push(f32::from(q) * scale);
    }
    result
}

/// Dequantize APR Q4 format (NOT the same as GGML Q4_K!)
///
/// APR Q4: For each block of 32 values:
///   `[block_scale: f16 (2 bytes)] + [packed nibbles: 16 bytes]` = 18 bytes per block
///
/// Each nibble stores unsigned value (0-15), where:
///   `original = (nibble - 8) * scale`
///
/// Nibble packing: byte = low_nibble | (high_nibble << 4)
///   - Even index i: nibble = byte & 0x0F
///   - Odd index i:  nibble = (byte >> 4) & 0x0F
pub fn dequantize_apr_q4(bytes: &[u8], num_elements: usize) -> Vec<f32> {
    const BLOCK_SIZE: usize = 32;
    const BLOCK_BYTES: usize = 2 + 16; // f16 scale + 16 packed nibble bytes

    let mut result = Vec::with_capacity(num_elements);
    let mut offset = 0;

    while result.len() < num_elements && offset + BLOCK_BYTES <= bytes.len() {
        // Read block scale (f16)
        let scale_bits = u16::from_le_bytes([bytes[offset], bytes[offset + 1]]);
        let scale = f16_to_f32(scale_bits);
        offset += 2;

        // Unpack 32 nibbles from 16 bytes
        let packed = &bytes[offset..offset + 16];
        for i in 0..BLOCK_SIZE {
            if result.len() >= num_elements {
                break;
            }
            let byte = packed[i / 2];
            let nibble = if i % 2 == 0 {
                byte & 0x0F
            } else {
                (byte >> 4) & 0x0F
            };
            // Unsigned nibble (0-15) was stored as (original / scale + 8),
            // so original = (nibble - 8) * scale
            let value = (f32::from(nibble) - 8.0) * scale;
            result.push(value);
        }
        offset += 16;
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dequantize_apr_q8_round_trip() {
        // Simulate APR Q8 encoding: scale = max_abs / 127
        let original = vec![1.0f32, -0.5, 0.3, -0.8, 0.0, 0.7];
        let max_abs = original.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let scale = max_abs / 127.0;

        // Pack: [scale: f32 (4B)] + [quantized: i8 × N]
        let mut bytes = Vec::with_capacity(4 + original.len());
        bytes.extend_from_slice(&scale.to_le_bytes());
        for &v in &original {
            let q = (v / scale).round().clamp(-127.0, 127.0) as i8;
            bytes.push(q as u8);
        }

        let result = dequantize_apr_q8(&bytes, original.len());
        assert_eq!(result.len(), original.len());
        for (i, (&orig, &dequant)) in original.iter().zip(result.iter()).enumerate() {
            assert!(
                (orig - dequant).abs() < 0.02,
                "APR Q8 mismatch at {i}: orig={orig}, dequant={dequant}"
            );
        }
    }

    #[test]
    fn test_dequantize_apr_q8_zeros() {
        // All zeros: scale=1.0, all i8 zeros
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&1.0f32.to_le_bytes());
        bytes.extend_from_slice(&[0u8; 10]);

        let result = dequantize_apr_q8(&bytes, 10);
        assert_eq!(result.len(), 10);
        for &v in &result {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn test_dequantize_apr_q8_empty() {
        let result = dequantize_apr_q8(&[], 10);
        assert_eq!(result, vec![0.0; 10]);
    }

    /// f32_to_f16 helper (IEEE 754 conversion for test encoding)
    fn f32_to_f16_bits(value: f32) -> u16 {
        let bits = value.to_bits();
        let sign = (bits >> 31) & 1;
        let exp = ((bits >> 23) & 0xFF) as i32 - 127;
        let mant = bits & 0x7FFFFF;

        if exp > 15 {
            // Overflow → infinity
            ((sign << 15) | (0x1F << 10)) as u16
        } else if exp < -14 {
            // Underflow → zero
            (sign << 15) as u16
        } else {
            let f16_exp = (exp + 15) as u32;
            let f16_mant = mant >> 13;
            ((sign << 15) | (f16_exp << 10) | f16_mant) as u16
        }
    }

    #[test]
    fn test_dequantize_apr_q4_round_trip() {
        // Simulate APR Q4 encoding: block of 32, scale = max_abs / 7
        let original: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) / 16.0).collect();
        let max_abs = original.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 7.0 };

        // Pack: [scale: f16 (2B)] + [16 nibble bytes]
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&f32_to_f16_bits(scale).to_le_bytes());

        let mut packed = [0u8; 16];
        for (i, &v) in original.iter().enumerate() {
            let q = (v / scale).round().clamp(-8.0, 7.0) as i8;
            let nibble = ((q + 8) as u8) & 0x0F;
            if i % 2 == 0 {
                packed[i / 2] = nibble;
            } else {
                packed[i / 2] |= nibble << 4;
            }
        }
        bytes.extend_from_slice(&packed);

        let result = dequantize_apr_q4(&bytes, 32);
        assert_eq!(result.len(), 32);
        for (i, (&orig, &dequant)) in original.iter().zip(result.iter()).enumerate() {
            assert!(
                (orig - dequant).abs() < 0.25,
                "APR Q4 mismatch at {i}: orig={orig}, dequant={dequant}"
            );
        }
    }

    #[test]
    fn test_dequantize_apr_q4_empty() {
        let result = dequantize_apr_q4(&[], 10);
        assert!(result.is_empty());
    }

    #[test]
    fn test_apr_q8_not_ggml_q8_0() {
        // Prove that APR Q8 and GGML Q8_0 produce DIFFERENT results from same bytes.
        // This is the bug that GH-250 discovered.
        let mut apr_bytes = Vec::new();
        // APR Q8: 4-byte f32 scale + N i8 values
        apr_bytes.extend_from_slice(&0.01f32.to_le_bytes()); // scale = 0.01
        for i in 0..32 {
            apr_bytes.push((i as i8 - 16) as u8);
        }

        let apr_result = dequantize_apr_q8(&apr_bytes, 32);
        let ggml_result = dequantize_q8_0(&apr_bytes, 32);

        // They must be DIFFERENT (proving the format mismatch)
        assert_ne!(
            apr_result, ggml_result,
            "APR Q8 and GGML Q8_0 should produce different results from same bytes"
        );
    }

    #[test]
    fn test_dtype_to_ggml_qtype_apr_native_returns_none() {
        // APR-native Q8/Q4 should NOT map to GGML type IDs
        assert_eq!(dtype_to_ggml_qtype("q8"), None);
        assert_eq!(dtype_to_ggml_qtype("Q8"), None);
        assert_eq!(dtype_to_ggml_qtype("q4"), None);
        assert_eq!(dtype_to_ggml_qtype("Q4"), None);

        // GGML-compatible formats SHOULD map
        assert_eq!(dtype_to_ggml_qtype("Q8_0"), Some(8));
        assert_eq!(dtype_to_ggml_qtype("Q4_K"), Some(12));
        assert_eq!(dtype_to_ggml_qtype("Q6_K"), Some(14));
    }
}
