//! Fused Q4_K Matrix-Vector Multiply (F-GPU-130)
//!
//! This module implements fused quantized matrix-vector multiplication that operates
//! directly on Q4_K compressed weights without full dequantization.
//!
//! # Q4_K Format (llama.cpp compatible)
//!
//! Super-block layout (144 bytes per 256 elements):
//! - `d`: 2 bytes (f16 global scale)
//! - `dmin`: 2 bytes (f16 global min scale)
//! - `scales`: 12 bytes (packed 6-bit scales and mins for 8 sub-blocks)
//! - `qs`: 128 bytes (4-bit quantized values, interleaved low/high nibbles)
//!
//! # Golden Test Invariant (Section 12.4 of spec)
//!
//! For all Q4K weight W and input x:
//! ```text
//! matmul_q4k_f32(W, x) ≈ matmul(dequant_q4k_to_f32(W), x)  within ε = 1e-3
//! ```
//!
//! # Performance Targets
//!
//! - Baseline (dequant+matmul): 0.27 tok/s
//! - Target (fused): >5 tok/s CPU, >100 tok/s GPU
//!
//! # Example
//!
//! ```rust,ignore
//! use trueno::backends::q4k::matmul_q4k_f32;
//!
//! let q4k_weights = load_q4k_tensor("gate_proj.weight");
//! let input = vec![1.0f32; 896];
//! let output = matmul_q4k_f32(&q4k_weights, &input, 4864, 896);
//! ```

// Sub-modules
mod colmajor;
mod dequant;
mod gemv;

// Re-exports
#[allow(deprecated)]
pub use colmajor::{matmul_q4k_f32_colmajor, matmul_q4k_f32_colmajor_dispatch};
pub use dequant::dequantize_q4k_to_f32;
pub use gemv::{matmul_q4k_f32, matmul_q4k_f32_dispatch, matmul_q4k_f32_scalar};

// Constants (pub(crate) for submodule access)
pub(crate) const SUPER_BLOCK_SIZE: usize = 256;
pub(crate) const SUPER_BLOCK_BYTES: usize = 144;
pub(crate) const _SUB_BLOCK_SIZE: usize = 32; // Reserved for future sub-block optimizations

/// Convert f16 bits to f32.
///
/// NOTE: F16C hardware instruction (`_mm_cvtph_ps`) was tested (2026-04-05)
/// but the per-call `is_x86_feature_detected` overhead negated the gain.
/// The scalar path is already well-optimized by LLVM for typical Q4K scales
/// (normal f16 values that hit the fast path without subnormal branching).
/// The Q4K bottleneck is the FMA dependency chain, not header parsing.
#[inline(always)]
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exp = (bits >> 10) & 0x1F;
    let mantissa = (bits & 0x3FF) as u32;

    if exp == 0 {
        if mantissa == 0 {
            f32::from_bits(sign)
        } else {
            // Subnormal
            let mut m = mantissa;
            let mut e = 0i32;
            while (m & 0x400) == 0 {
                m <<= 1;
                e -= 1;
            }
            let new_exp = ((127 - 15 + 1 + e) as u32) << 23;
            let new_mantissa = (m & 0x3FF) << 13;
            f32::from_bits(sign | new_exp | new_mantissa)
        }
    } else if exp == 31 {
        f32::from_bits(sign | (0xFF << 23) | (mantissa << 13))
    } else {
        let new_exp = ((exp as i32 - 15 + 127) as u32) << 23;
        f32::from_bits(sign | new_exp | (mantissa << 13))
    }
}

/// Parse Q4_K super-block header and scales
///
/// Returns (d, dmin, scales[8], mins[8])
#[inline(always)]
pub(crate) fn parse_q4k_header(block: &[u8]) -> (f32, f32, [u8; 8], [u8; 8]) {
    debug_assert!(block.len() >= 16);

    // Read d and dmin (f16)
    let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));

    // Unpack scales and mins (llama.cpp format)
    let scales_bytes = block.get(4..16).expect("Q4_K: need ≥16 bytes for header");
    let mut scales = [0u8; 8];
    let mut mins = [0u8; 8];

    for i in 0..4 {
        // Blocks 0-3: lower 6 bits of bytes 0-3 and 4-7
        scales[i] = scales_bytes[i] & 0x3F;
        mins[i] = scales_bytes[i + 4] & 0x3F;
        // Blocks 4-7: lower 4 bits from bytes 8-11, upper 2 bits from bytes 0-3/4-7
        scales[i + 4] = (scales_bytes[i + 8] & 0x0F) | ((scales_bytes[i] >> 6) << 4);
        mins[i + 4] = (scales_bytes[i + 8] >> 4) | ((scales_bytes[i + 4] >> 6) << 4);
    }

    (d, dmin, scales, mins)
}

#[cfg(test)]
mod tests_core;
#[cfg(test)]
mod tests_coverage;
#[cfg(test)]
mod tests_golden;
