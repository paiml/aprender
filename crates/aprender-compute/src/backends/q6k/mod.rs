//! Fused Q6_K Matrix-Vector Multiply
//!
//! Q6_K format (210 bytes per 256 elements):
//! - `ql`: 128 bytes (lower 4 bits of each value)
//! - `qh`: 64 bytes (upper 2 bits, packed 4 values per byte)
//! - `scales`: 16 bytes (8-bit scales for 16 groups of 16 values)
//! - `d`: 2 bytes (f16 global scale)

// Sub-modules
mod colmajor;
mod gemv;

// Re-exports
#[allow(deprecated)]
pub use colmajor::{matmul_q6k_f32_colmajor, matmul_q6k_f32_colmajor_dispatch};
pub use gemv::{matmul_q6k_f32, matmul_q6k_f32_dispatch, matmul_q6k_f32_scalar};

// Constants (pub(crate) for submodule access)
pub(crate) const SUPER_BLOCK_SIZE: usize = 256;
pub(crate) const SUPER_BLOCK_BYTES: usize = 210;

/// Convert f16 bits to f32
#[inline(always)]
pub(crate) fn f16_to_f32(bits: u16) -> f32 {
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

#[cfg(test)]
mod tests_core;
#[cfg(test)]
mod tests_extended;
