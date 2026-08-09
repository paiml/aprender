//! Q5_K and Q6_K Quantization Operations (llama.cpp compatible)
//!
//! This module provides quantization formats and compute operations
//! for llama.cpp-compatible k-quant formats.
//!
//! # Formats
//!
//! - `BlockQ5K`: 5-bit quantization with super-blocks (256 values)
//! - `BlockQ6K`: 6-bit quantization with super-blocks (256 values)
//!
//! # Operations
//!
//! - `DotQ5KOp`: Dot product with Q5_K quantized weights
//! - `DotQ6KOp`: Dot product with Q6_K quantized weights
//!
//! # SIMD Optimization
//!
//! Both operations use AVX2/FMA when available for ~4x speedup.

use super::{Backend, ComputeOp};
use crate::error::TruenoError;

// ============================================================================
// Q5_K Block Format
// ============================================================================

/// Q5_K block format (5-bit with super-blocks).
///
/// Matches llama.cpp's block_q5_K format:
/// - Super-block of 256 values
/// - 5-bit quantization with k-quant scales
/// - Higher precision than Q4_K, lower than Q6_K
///
/// Memory layout:
/// ```text
/// | d (fp16) | dmin (fp16) | scales[12] | qh[32] | qs[128] |
/// ```
#[derive(Debug, Clone)]
pub struct BlockQ5K {
    /// Scale factor (half precision)
    pub d: f32,
    /// Minimum value scale (half precision)
    pub dmin: f32,
    /// Scales for each 32-value block (12 bytes packed)
    pub scales: [u8; 12],
    /// High bits for quantized values (32 bytes)
    pub qh: [u8; 32],
    /// Quantized values (128 bytes, 2 values per byte)
    pub qs: [u8; 128],
}

impl BlockQ5K {
    /// Block size in elements
    pub const BLOCK_SIZE: usize = 256;

    /// Dequantize a Q5_K block to f32.
    ///
    /// # Safety
    ///
    /// Output buffer must have at least BLOCK_SIZE elements.
    pub fn dequantize(&self, output: &mut [f32]) {
        debug_assert!(output.len() >= Self::BLOCK_SIZE);

        // Decode scales from packed format
        let mut scales = [0i8; 8];
        for i in 0..8 {
            let low = (self.scales[i] & 0x3F) as i8;
            scales[i] = low - 32;
        }

        // Dequantize each sub-block
        for block_idx in 0..8 {
            let scale = scales[block_idx] as f32;
            let base_idx = block_idx * 32;

            for i in 0..32 {
                let out_idx = base_idx + i;
                let byte_idx = base_idx / 2 + i / 2;

                // Extract 4-bit low value
                let q4 = if i % 2 == 0 { self.qs[byte_idx] & 0x0F } else { self.qs[byte_idx] >> 4 };

                // Extract 5th bit from qh
                let qh_bit = ((self.qh[i] >> block_idx) & 1) as u8;
                let q5 = q4 | (qh_bit << 4);

                // Dequantize: value = d * scale * (q5 - 16) + dmin
                output[out_idx] = self.d * scale * (q5 as f32 - 16.0) + self.dmin;
            }
        }
    }
}

// ============================================================================
// Q6_K Block Format
// ============================================================================

/// Q6_K block format (6-bit with super-blocks).
///
/// Matches llama.cpp's block_q6_K format:
/// - Super-block of 256 values
/// - 6-bit quantization with k-quant scales
/// - Highest precision k-quant format
///
/// Memory layout:
/// ```text
/// | ql[128] | qh[64] | scales[16] | d (fp16) |
/// ```
#[derive(Debug, Clone)]
pub struct BlockQ6K {
    /// Low 4 bits of quantized values (128 bytes)
    pub ql: [u8; 128],
    /// High 2 bits of quantized values (64 bytes)
    pub qh: [u8; 64],
    /// Scales for each 16-value block (16 bytes)
    pub scales: [i8; 16],
    /// Scale factor (half precision)
    pub d: f32,
}

impl BlockQ6K {
    /// Block size in elements
    pub const BLOCK_SIZE: usize = 256;

    /// Dequantize a Q6_K block to f32.
    ///
    /// # Safety
    ///
    /// Output buffer must have at least BLOCK_SIZE elements.
    pub fn dequantize(&self, output: &mut [f32]) {
        debug_assert!(output.len() >= Self::BLOCK_SIZE);

        // Dequantize each sub-block of 16 values
        for block_idx in 0..16 {
            let scale = self.scales[block_idx] as f32;
            let base_idx = block_idx * 16;

            for i in 0..16 {
                let out_idx = base_idx + i;
                let ql_idx = base_idx / 2 + i / 2;
                let qh_idx = base_idx / 4 + i / 4;

                // Extract 4-bit low value
                let ql_val = if i % 2 == 0 { self.ql[ql_idx] & 0x0F } else { self.ql[ql_idx] >> 4 };

                // Extract 2-bit high value
                let qh_shift = (i % 4) * 2;
                let qh_val = ((self.qh[qh_idx] >> qh_shift) & 0x03) as u8;

                // Combine to 6-bit value
                let q6 = ql_val | (qh_val << 4);

                // Dequantize: value = d * scale * (q6 - 32)
                output[out_idx] = self.d * scale * (q6 as f32 - 32.0);
            }
        }
    }
}

// ============================================================================
// Q5_K Dot Product Operation
// ============================================================================

/// Q5_K dot product operation.
///
/// Computes dot product between Q5_K quantized weights and f32 activations.
#[derive(Debug, Clone)]
pub struct DotQ5KOp {
    /// Number of blocks
    pub n_blocks: usize,
}

impl DotQ5KOp {
    /// Create a new Q5_K dot product operation.
    #[must_use]
    pub fn new(n_elements: usize) -> Self {
        Self { n_blocks: n_elements / BlockQ5K::BLOCK_SIZE }
    }

    /// Compute dot product with SIMD acceleration.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2", enable = "fma")]
    // SAFETY: caller verifies AVX2 support, input slices meet alignment/length requirements
    unsafe fn avx2_dot_block(block: &BlockQ5K, x: &[f32]) -> f32 {
        unsafe {
            use std::arch::x86_64::*;

            let mut acc = _mm256_setzero_ps();
            let mut dequant = [0.0f32; BlockQ5K::BLOCK_SIZE];
            block.dequantize(&mut dequant);

            let mut i = 0;
            while i + 8 <= BlockQ5K::BLOCK_SIZE {
                let vd = _mm256_loadu_ps(dequant.as_ptr().add(i));
                let vx = _mm256_loadu_ps(x.as_ptr().add(i));
                acc = _mm256_fmadd_ps(vd, vx, acc);
                i += 8;
            }

            // Horizontal sum
            let high = _mm256_extractf128_ps(acc, 1);
            let low = _mm256_castps256_ps128(acc);
            let sum128 = _mm_add_ps(high, low);
            let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
            let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
            _mm_cvtss_f32(sum32)
        }
    }
}

impl ComputeOp for DotQ5KOp {
    type Input = (Vec<BlockQ5K>, Vec<f32>);
    type Output = f32;

    fn name(&self) -> &'static str {
        "dot_q5k"
    }

    fn execute(&self, input: Self::Input, _backend: Backend) -> Result<Self::Output, TruenoError> {
        let (blocks, x) = input;

        if blocks.is_empty() || x.is_empty() {
            return Ok(0.0);
        }

        let mut sum = 0.0f32;

        #[cfg(target_arch = "x86_64")]
        {
            if matches!(backend, Backend::Avx2 | Backend::Auto) && is_x86_feature_detected!("avx2")
            {
                for (i, block) in blocks.iter().enumerate() {
                    let x_slice = &x[i * BlockQ5K::BLOCK_SIZE..];
                    // SAFETY: preconditions verified by caller
                    sum += unsafe { Self::avx2_dot_block(block, x_slice) };
                }
                return Ok(sum);
            }
        }

        // Scalar fallback
        let mut dequant = [0.0f32; BlockQ5K::BLOCK_SIZE];
        for (i, block) in blocks.iter().enumerate() {
            block.dequantize(&mut dequant);
            let x_slice = &x[i * BlockQ5K::BLOCK_SIZE..];
            for j in 0..BlockQ5K::BLOCK_SIZE {
                sum += dequant[j] * x_slice[j];
            }
        }

        Ok(sum)
    }

    fn tokens(&self, _input: &Self::Input) -> usize {
        self.n_blocks * BlockQ5K::BLOCK_SIZE
    }
}

// ============================================================================
// Q6_K Dot Product Operation
// ============================================================================

/// Q6_K dot product operation.
///
/// Computes dot product between Q6_K quantized weights and f32 activations.
#[derive(Debug, Clone)]
pub struct DotQ6KOp {
    /// Number of blocks
    pub n_blocks: usize,
}

impl DotQ6KOp {
    /// Create a new Q6_K dot product operation.
    #[must_use]
    pub fn new(n_elements: usize) -> Self {
        Self { n_blocks: n_elements / BlockQ6K::BLOCK_SIZE }
    }

    /// Compute dot product with SIMD acceleration.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2", enable = "fma")]
    // SAFETY: caller verifies AVX2 support, input slices meet alignment/length requirements
    unsafe fn avx2_dot_block(block: &BlockQ6K, x: &[f32]) -> f32 {
        unsafe {
            use std::arch::x86_64::*;

            let mut acc = _mm256_setzero_ps();
            let mut dequant = [0.0f32; BlockQ6K::BLOCK_SIZE];
            block.dequantize(&mut dequant);

            let mut i = 0;
            while i + 8 <= BlockQ6K::BLOCK_SIZE {
                let vd = _mm256_loadu_ps(dequant.as_ptr().add(i));
                let vx = _mm256_loadu_ps(x.as_ptr().add(i));
                acc = _mm256_fmadd_ps(vd, vx, acc);
                i += 8;
            }

            // Horizontal sum
            let high = _mm256_extractf128_ps(acc, 1);
            let low = _mm256_castps256_ps128(acc);
            let sum128 = _mm_add_ps(high, low);
            let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
            let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
            _mm_cvtss_f32(sum32)
        }
    }
}

impl ComputeOp for DotQ6KOp {
    type Input = (Vec<BlockQ6K>, Vec<f32>);
    type Output = f32;

    fn name(&self) -> &'static str {
        "dot_q6k"
    }

    fn execute(&self, input: Self::Input, _backend: Backend) -> Result<Self::Output, TruenoError> {
        let (blocks, x) = input;

        if blocks.is_empty() || x.is_empty() {
            return Ok(0.0);
        }

        let mut sum = 0.0f32;

        #[cfg(target_arch = "x86_64")]
        {
            if matches!(backend, Backend::Avx2 | Backend::Auto) && is_x86_feature_detected!("avx2")
            {
                for (i, block) in blocks.iter().enumerate() {
                    let x_slice = &x[i * BlockQ6K::BLOCK_SIZE..];
                    // SAFETY: preconditions verified by caller
                    sum += unsafe { Self::avx2_dot_block(block, x_slice) };
                }
                return Ok(sum);
            }
        }

        // Scalar fallback
        let mut dequant = [0.0f32; BlockQ6K::BLOCK_SIZE];
        for (i, block) in blocks.iter().enumerate() {
            block.dequantize(&mut dequant);
            let x_slice = &x[i * BlockQ6K::BLOCK_SIZE..];
            for j in 0..BlockQ6K::BLOCK_SIZE {
                sum += dequant[j] * x_slice[j];
            }
        }

        Ok(sum)
    }

    fn tokens(&self, _input: &Self::Input) -> usize {
        self.n_blocks * BlockQ6K::BLOCK_SIZE
    }
}

#[cfg(test)]
pub mod nf4;
mod tests;
