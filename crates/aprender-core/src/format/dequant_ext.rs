//! `AprV2DequantExt` — the dequantizing `get_tensor_as_f32` accessor, re-attached
//! in `aprender-core` (issue #2231).
//!
//! The sovereign `apr-format` leaf reader exposes only the raw container bytes
//! (`get_tensor_data`) and a trivial F32-only typed view (`get_f32_tensor`). The
//! full *dequantizing* accessor needs the GGUF Q4_K / Q6_K dequant kernels
//! (`crate::format::gguf::dequant`) and the f16-scaled APR-Q4 block path — all
//! quantization/physics concerns that must NOT live in the leaf. This module
//! re-attaches them as an extension trait over BOTH leaf reader types
//! (`AprV2Reader` and `AprV2ReaderRef`), so existing callers keep working by
//! adding `use aprender::format::AprV2DequantExt;` (or
//! `use crate::format::AprV2DequantExt;`).
//!
//! # f16 (issue #2231 / PMAT-905 class)
//!
//! f16→f32 here routes through the leaf's `apr_format::f16_to_f32` (the
//! IEEE-correct `half` crate). This is the READ direction and is exact for
//! normal f16; the documented byte-level change is on the WRITE side (the leaf's
//! `f32_to_f16` now uses IEEE round-to-nearest-even instead of the legacy
//! non-RNE `trueno::f32_to_f16`).

use crate::format::f16_safety::F16_MIN_NORMAL;
use crate::format::gguf::dequant::{dequantize_q4_k, dequantize_q6_k};
use crate::format::v2::{AprV2Reader, AprV2ReaderRef, TensorDType};

/// Dequantize APR-native Q4 block-quantized data to f32 (issue #2231).
///
/// Format: blocks of [scale: f16 (2 bytes)] + [packed nibbles: 16 bytes];
/// each block holds 32 values. Byte-identical to the pre-extraction
/// `v2::dequantize_q4` (the f16 scale read uses the leaf's IEEE f16 path; the
/// GH-186 NaN/Inf/subnormal clamp via [`F16_MIN_NORMAL`] is preserved).
#[must_use]
pub fn dequantize_q4(data: &[u8], element_count: usize) -> Vec<f32> {
    const BLOCK_SIZE: usize = 32;

    let mut result = Vec::with_capacity(element_count);
    let mut pos = 0;
    let mut remaining = element_count;

    while remaining > 0 && pos + 2 <= data.len() {
        // Read scale (f16). GH-186: clamp NaN/Inf/subnormal to prevent propagation.
        let scale_bits = u16::from_le_bytes([data[pos], data[pos + 1]]);
        let scale_raw = apr_format::f16_to_f32(scale_bits);
        let scale =
            if scale_raw.is_nan() || scale_raw.is_infinite() || scale_raw.abs() < F16_MIN_NORMAL {
                0.0
            } else {
                scale_raw
            };
        pos += 2;

        let values_in_block = remaining.min(BLOCK_SIZE);
        for i in 0..values_in_block {
            let byte_idx = pos + i / 2;
            if byte_idx >= data.len() {
                break;
            }
            let byte = data[byte_idx];
            let nibble = if i % 2 == 0 { byte & 0x0F } else { byte >> 4 };
            // Convert back from unsigned nibble (0-15) to signed (-8 to 7).
            let q = (nibble as i8) - 8;
            result.push(f32::from(q) * scale);
        }

        pos += 16;
        remaining = remaining.saturating_sub(BLOCK_SIZE);
    }

    result.resize(element_count, 0.0);
    result
}

/// Decode raw container bytes of a known dtype/element-count into f32.
///
/// Shared core of the extension so both reader types stay byte-identical.
fn decode_tensor_bytes(dtype: TensorDType, data: &[u8], element_count: usize) -> Option<Vec<f32>> {
    match dtype {
        TensorDType::F32 => {
            let floats: Vec<f32> = data
                .as_chunks::<4>()
                .0
                .iter()
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect();
            Some(floats)
        }
        TensorDType::F16 => {
            let floats: Vec<f32> = data
                .as_chunks::<2>()
                .0
                .iter()
                .map(|chunk| apr_format::f16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]])))
                .collect();
            Some(floats)
        }
        TensorDType::AprQ8 => {
            if data.len() < 4 {
                return None;
            }
            let scale = f32::from_le_bytes([data[0], data[1], data[2], data[3]]);
            let floats: Vec<f32> = data[4..]
                .iter()
                .map(|&b| f32::from(b as i8) * scale)
                .collect();
            Some(floats)
        }
        TensorDType::AprQ4 => Some(dequantize_q4(data, element_count)),
        TensorDType::BF16 => {
            let floats: Vec<f32> = data
                .as_chunks::<2>()
                .0
                .iter()
                .map(|chunk| {
                    let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                    f32::from_bits(u32::from(bits) << 16)
                })
                .collect();
            Some(floats)
        }
        TensorDType::Q4K => dequantize_q4_k(data, 0, element_count).ok(),
        TensorDType::Q6K => dequantize_q6_k(data, 0, element_count).ok(),
        _ => None, // Other types not yet supported
    }
}

/// Dequantizing accessor for the sovereign-leaf APR v2 readers (issue #2231).
///
/// Re-attaches `get_tensor_as_f32` (severed from the leaf) over both
/// [`AprV2Reader`] and [`AprV2ReaderRef`]. Supports F32, F16, BF16, APR-Q8,
/// APR-Q4, and raw GGUF Q4_K / Q6_K.
pub trait AprV2DequantExt {
    /// Get a tensor as an f32 `Vec`, dequantizing if necessary.
    ///
    /// Returns `None` for an unknown tensor name or an unsupported dtype.
    fn get_tensor_as_f32(&self, name: &str) -> Option<Vec<f32>>;
}

impl AprV2DequantExt for AprV2Reader {
    fn get_tensor_as_f32(&self, name: &str) -> Option<Vec<f32>> {
        let entry = self.get_tensor(name)?;
        let dtype = entry.dtype;
        let element_count = entry.element_count();
        let data = self.get_tensor_data(name)?;
        decode_tensor_bytes(dtype, data, element_count)
    }
}

impl AprV2DequantExt for AprV2ReaderRef<'_> {
    fn get_tensor_as_f32(&self, name: &str) -> Option<Vec<f32>> {
        let entry = self.get_tensor(name)?;
        let dtype = entry.dtype;
        let element_count = entry.element_count();
        let data = self.get_tensor_data(name)?;
        decode_tensor_bytes(dtype, data, element_count)
    }
}
