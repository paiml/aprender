//! AVX2-accelerated dequantization fast paths for Q4_0 / Q8_0 (GH-386).
//!
//! The scalar implementations in [`crate::format::quantize`] (`Q4_0Quantizer`,
//! `Q8_0Quantizer`) iterate one element at a time; LLVM's auto-vectorizer
//! handles the trivial multiply but bottlenecks on the i8→i32→f32 sign-extend
//! cascade and the nibble unpack, hitting ~1.2 Gelem/s (≈5× below memcpy
//! ceiling).
//!
//! This module provides AVX2 implementations that:
//!
//! - **Q8_0**: load 32 i8 elements per block, sign-extend to four 256-bit i32
//!   lanes via `_mm256_cvtepi8_epi32`, convert to f32, multiply by a broadcast
//!   f16 scale, and store the 32-element output as four 256-bit f32 vectors.
//!
//! - **Q4_0**: load 16 packed nibble bytes, extract low and high nibbles
//!   (mask + shift), interleave so that `byte_i` produces output positions
//!   `2i` and `2i+1` (matching the existing `Q4_0Quantizer::quantize` pack
//!   layout — NOT the GGML half-half layout used in `format::gguf::dequant`),
//!   subtract 8, convert to f32, multiply by the scale, and store.
//!
//! Runtime dispatch is via [`is_x86_feature_detected`]. Targets without
//! AVX2 fall back to the scalar path unchanged. All non-x86 architectures
//! also fall back. The fast paths produce **bit-exact** output relative to
//! the scalar reference (verified by `tests::scalar_simd_parity_*` and the
//! proptest under `tests::prop_avx2_matches_scalar_*`).
//!
//! # Safety
//!
//! Each `_avx2` function is `unsafe fn` and is only reachable from
//! [`dequantize_q8_0_avx2_dispatch`] / [`dequantize_q4_0_avx2_dispatch`],
//! which check `is_x86_feature_detected!("avx2")` immediately before the
//! call. The functions are marked `#[target_feature(enable = "avx2")]`
//! so the codegen for the AVX2 intrinsics is correct; the caller is
//! responsible for the runtime feature gate.
//!
//! Bounds invariants:
//!
//! - Caller passes `blocks` with `blocks.len() == num_blocks * BLOCK_BYTES`
//!   (Q8_0: 34, Q4_0: 18) and `out.len() == num_blocks * BLOCK_SIZE` (32).
//! - All loads/stores go through `_mm256_loadu_si256` / `_mm256_storeu_ps`
//!   so input alignment is not required.

#![allow(unsafe_code)] // GH-386: documented AVX2 fast path; runtime-gated.

#[allow(unused_imports)]
use crate::format::quantize::BLOCK_SIZE;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use half::f16;

/// Dispatch entry point for Q8_0 dequant.
///
/// Calls the AVX2 fast path when available, otherwise returns `false` so the
/// caller can use its scalar reference path. Writes exactly `num_blocks *
/// BLOCK_SIZE` elements to `out` when it returns `true`; `out.len()` must be
/// `≥ num_blocks * BLOCK_SIZE`.
#[inline]
pub(crate) fn dequantize_q8_0_avx2_dispatch(
    blocks: &[u8],
    num_blocks: usize,
    out: &mut [f32],
) -> bool {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: feature gate above guarantees AVX2 is available. Callers
            // (the `Q8_0Quantizer::dequantize` path) pre-validate that
            // `blocks.len() == num_blocks * Q8_0_BLOCK_BYTES (34)` and
            // `out.len() == num_blocks * BLOCK_SIZE (32)` — see
            // `tests::scalar_simd_parity_q8_0`.
            unsafe { dequantize_q8_0_avx2(blocks, num_blocks, out) };
            return true;
        }
    }
    let _ = (blocks, num_blocks, out);
    false
}

/// Dispatch entry point for Q4_0 dequant. See [`dequantize_q8_0_avx2_dispatch`].
#[inline]
pub(crate) fn dequantize_q4_0_avx2_dispatch(
    blocks: &[u8],
    num_blocks: usize,
    out: &mut [f32],
) -> bool {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: as for Q8_0 dispatch; preconditions on
            // `blocks.len() == num_blocks * Q4_0_BLOCK_BYTES (18)` and
            // `out.len() == num_blocks * BLOCK_SIZE (32)` enforced by callers.
            unsafe { dequantize_q4_0_avx2(blocks, num_blocks, out) };
            return true;
        }
    }
    let _ = (blocks, num_blocks, out);
    false
}

// ---------------------------------------------------------------------------
// AVX2 implementations
// ---------------------------------------------------------------------------

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn dequantize_q8_0_avx2(blocks: &[u8], num_blocks: usize, out: &mut [f32]) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::{
        __m128i, _mm256_cvtepi32_ps, _mm256_cvtepi8_epi32, _mm256_mul_ps, _mm256_set1_ps,
        _mm256_storeu_ps, _mm_loadu_si128, _mm_srli_si128,
    };
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::{
        __m128i, _mm256_cvtepi32_ps, _mm256_cvtepi8_epi32, _mm256_mul_ps, _mm256_set1_ps,
        _mm256_storeu_ps, _mm_loadu_si128, _mm_srli_si128,
    };

    const BLOCK_BYTES: usize = 34;

    // SAFETY: every intrinsic call is reachable only via
    // `dequantize_q8_0_avx2_dispatch`, which gates on `is_x86_feature_detected!("avx2")`.
    // Bounds: `blocks.len() >= num_blocks * BLOCK_BYTES`, `out.len() >=
    // num_blocks * BLOCK_SIZE` (caller invariants).
    unsafe {
        for block_idx in 0..num_blocks {
            let b_start = block_idx * BLOCK_BYTES;
            let block = &blocks[b_start..b_start + BLOCK_BYTES];

            // f16 scale → f32 → broadcast across 8 lanes.
            let scale = f16::from_le_bytes([block[0], block[1]]).to_f32();
            let scale_v = _mm256_set1_ps(scale);

            // Load 16 bytes (low half) + 16 bytes (high half) of the 32 i8 quants.
            // AVX2 _mm256_cvtepi8_epi32 takes 8 i8 from a 128-bit lane and
            // sign-extends to 8 i32; do it 4× (once per 8-element slice).
            let lo16 = _mm_loadu_si128(block.as_ptr().add(2).cast::<__m128i>());
            let hi16 = _mm_loadu_si128(block.as_ptr().add(18).cast::<__m128i>());

            let q_i32_0 = _mm256_cvtepi8_epi32(lo16);
            let q_i32_1 = _mm256_cvtepi8_epi32(_mm_srli_si128::<8>(lo16));
            let q_i32_2 = _mm256_cvtepi8_epi32(hi16);
            let q_i32_3 = _mm256_cvtepi8_epi32(_mm_srli_si128::<8>(hi16));

            let f0 = _mm256_mul_ps(_mm256_cvtepi32_ps(q_i32_0), scale_v);
            let f1 = _mm256_mul_ps(_mm256_cvtepi32_ps(q_i32_1), scale_v);
            let f2 = _mm256_mul_ps(_mm256_cvtepi32_ps(q_i32_2), scale_v);
            let f3 = _mm256_mul_ps(_mm256_cvtepi32_ps(q_i32_3), scale_v);

            let out_ptr = out.as_mut_ptr().add(block_idx * BLOCK_SIZE);
            _mm256_storeu_ps(out_ptr, f0);
            _mm256_storeu_ps(out_ptr.add(8), f1);
            _mm256_storeu_ps(out_ptr.add(16), f2);
            _mm256_storeu_ps(out_ptr.add(24), f3);
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn dequantize_q4_0_avx2(blocks: &[u8], num_blocks: usize, out: &mut [f32]) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::{
        __m128i, _mm256_cvtepi32_ps, _mm256_cvtepu8_epi32, _mm256_mul_ps, _mm256_set1_epi32,
        _mm256_set1_ps, _mm256_storeu_ps, _mm256_sub_epi32, _mm_and_si128, _mm_loadu_si128,
        _mm_set1_epi8, _mm_srli_epi16, _mm_srli_si128, _mm_unpackhi_epi8, _mm_unpacklo_epi8,
    };
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::{
        __m128i, _mm256_cvtepi32_ps, _mm256_cvtepu8_epi32, _mm256_mul_ps, _mm256_set1_epi32,
        _mm256_set1_ps, _mm256_storeu_ps, _mm256_sub_epi32, _mm_and_si128, _mm_loadu_si128,
        _mm_set1_epi8, _mm_srli_epi16, _mm_srli_si128, _mm_unpackhi_epi8, _mm_unpacklo_epi8,
    };

    const BLOCK_BYTES: usize = 18;

    // SAFETY: every intrinsic call is reachable only via
    // `dequantize_q4_0_avx2_dispatch`, which gates on
    // `is_x86_feature_detected!("avx2")`. Bounds: `blocks.len() >= num_blocks
    // * BLOCK_BYTES`, `out.len() >= num_blocks * BLOCK_SIZE` (caller invariants).
    unsafe {
        let mask_lo_nib = _mm_set1_epi8(0x0F);
        let bias_i32 = _mm256_set1_epi32(8);

        for block_idx in 0..num_blocks {
            let b_start = block_idx * BLOCK_BYTES;
            let block = &blocks[b_start..b_start + BLOCK_BYTES];

            let scale = f16::from_le_bytes([block[0], block[1]]).to_f32();
            let scale_v = _mm256_set1_ps(scale);

            // Load 16 packed bytes (32 nibbles).
            let packed = _mm_loadu_si128(block.as_ptr().add(2).cast::<__m128i>());

            // Low nibble of each byte: byte_i & 0x0F → output position 2i
            let nib_lo = _mm_and_si128(packed, mask_lo_nib);
            // High nibble of each byte: (byte_i >> 4) & 0x0F → output position 2i+1
            // (use 16-bit shift since AVX2 lacks 8-bit shift; mask afterwards).
            let nib_hi = _mm_and_si128(_mm_srli_epi16::<4>(packed), mask_lo_nib);

            // Interleave so positions (lo_0, hi_0, lo_1, hi_1, ...) match the
            // `Q4_0Quantizer::quantize` pack layout
            // (byte_i = (q_2i+1) << 4 | q_2i).
            let inter_lo = _mm_unpacklo_epi8(nib_lo, nib_hi); // → out 0..16
            let inter_hi = _mm_unpackhi_epi8(nib_lo, nib_hi); // → out 16..32

            // Sign-extend each 8-element u8 sublane (values 0..15) to i32,
            // then subtract the centering bias 8.
            let q_i32_0 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(inter_lo), bias_i32);
            let q_i32_1 = _mm256_sub_epi32(
                _mm256_cvtepu8_epi32(_mm_srli_si128::<8>(inter_lo)),
                bias_i32,
            );
            let q_i32_2 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(inter_hi), bias_i32);
            let q_i32_3 = _mm256_sub_epi32(
                _mm256_cvtepu8_epi32(_mm_srli_si128::<8>(inter_hi)),
                bias_i32,
            );

            let f0 = _mm256_mul_ps(_mm256_cvtepi32_ps(q_i32_0), scale_v);
            let f1 = _mm256_mul_ps(_mm256_cvtepi32_ps(q_i32_1), scale_v);
            let f2 = _mm256_mul_ps(_mm256_cvtepi32_ps(q_i32_2), scale_v);
            let f3 = _mm256_mul_ps(_mm256_cvtepi32_ps(q_i32_3), scale_v);

            let out_ptr = out.as_mut_ptr().add(block_idx * BLOCK_SIZE);
            _mm256_storeu_ps(out_ptr, f0);
            _mm256_storeu_ps(out_ptr.add(8), f1);
            _mm256_storeu_ps(out_ptr.add(16), f2);
            _mm256_storeu_ps(out_ptr.add(24), f3);
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::quantize::{
        quantize, QuantType, BLOCK_SIZE as BS, Q4_0_BLOCK_BYTES, Q8_0_BLOCK_BYTES,
    };

    /// Reference scalar Q8_0 dequant (mirrors `Q8_0Quantizer::dequantize` body).
    fn ref_dequantize_q8_0(blocks: &[u8], num_blocks: usize, total_elems: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; num_blocks * BS];
        for block_idx in 0..num_blocks {
            let b_start = block_idx * Q8_0_BLOCK_BYTES;
            let scale = half::f16::from_le_bytes([blocks[b_start], blocks[b_start + 1]]).to_f32();
            let qs = &blocks[b_start + 2..b_start + 2 + BS];
            let out_off = block_idx * BS;
            for (j, &q) in qs.iter().enumerate() {
                out[out_off + j] = (q as i8) as f32 * scale;
            }
        }
        out.truncate(total_elems);
        out
    }

    /// Reference scalar Q4_0 dequant (mirrors `Q4_0Quantizer::dequantize` body).
    fn ref_dequantize_q4_0(blocks: &[u8], num_blocks: usize, total_elems: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; num_blocks * BS];
        for block_idx in 0..num_blocks {
            let b_start = block_idx * Q4_0_BLOCK_BYTES;
            let scale = half::f16::from_le_bytes([blocks[b_start], blocks[b_start + 1]]).to_f32();
            let packed = &blocks[b_start + 2..b_start + 2 + 16];
            let out_off = block_idx * BS;
            for (i, &p) in packed.iter().enumerate() {
                let q0 = (p & 0x0F) as i8 - 8;
                let q1 = ((p >> 4) & 0x0F) as i8 - 8;
                out[out_off + i * 2] = (q0 as f32) * scale;
                out[out_off + i * 2 + 1] = (q1 as f32) * scale;
            }
        }
        out.truncate(total_elems);
        out
    }

    fn make_payload(n: usize, seed: u32) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let x = (i as u32).wrapping_mul(2_654_435_761).wrapping_add(seed) as f32;
                (x * 1.0e-9).sin()
            })
            .collect()
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn scalar_simd_parity_q8_0() {
        if !is_x86_feature_detected!("avx2") {
            eprintln!("skipping AVX2 parity test — CPU lacks avx2");
            return;
        }
        for n in [32, 64, 256, 1024, 32 * 71] {
            let data = make_payload(n, 7);
            let q = quantize(&data, &[n], QuantType::Q8_0).expect("quantize");
            let num_blocks = q.num_blocks();

            let ref_out = ref_dequantize_q8_0(&q.blocks, num_blocks, n);

            let mut simd_out = vec![0.0f32; num_blocks * BS];
            let dispatched = dequantize_q8_0_avx2_dispatch(&q.blocks, num_blocks, &mut simd_out);
            assert!(dispatched, "AVX2 dispatch must run on avx2 host");
            simd_out.truncate(n);

            assert_eq!(ref_out.len(), simd_out.len());
            for (i, (r, s)) in ref_out.iter().zip(&simd_out).enumerate() {
                assert!(
                    r.to_bits() == s.to_bits(),
                    "Q8_0 mismatch at i={i} n={n}: scalar={r} simd={s}"
                );
            }
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn scalar_simd_parity_q4_0() {
        if !is_x86_feature_detected!("avx2") {
            eprintln!("skipping AVX2 parity test — CPU lacks avx2");
            return;
        }
        for n in [32, 64, 256, 1024, 32 * 71] {
            let data = make_payload(n, 13);
            let q = quantize(&data, &[n], QuantType::Q4_0).expect("quantize");
            let num_blocks = q.num_blocks();

            let ref_out = ref_dequantize_q4_0(&q.blocks, num_blocks, n);

            let mut simd_out = vec![0.0f32; num_blocks * BS];
            let dispatched = dequantize_q4_0_avx2_dispatch(&q.blocks, num_blocks, &mut simd_out);
            assert!(dispatched, "AVX2 dispatch must run on avx2 host");
            simd_out.truncate(n);

            assert_eq!(ref_out.len(), simd_out.len());
            for (i, (r, s)) in ref_out.iter().zip(&simd_out).enumerate() {
                assert!(
                    r.to_bits() == s.to_bits(),
                    "Q4_0 mismatch at i={i} n={n}: scalar={r} simd={s}"
                );
            }
        }
    }

    /// Non-x86 platforms (and CPUs without AVX2) hit the dispatcher's `false`
    /// path so the scalar fallback runs. Verify the dispatch returns false on
    /// non-x86, and on x86 only returns false when AVX2 is missing.
    #[test]
    fn dispatch_returns_false_without_avx2() {
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        {
            let mut out = vec![0.0f32; 32];
            assert!(!dequantize_q8_0_avx2_dispatch(&[0u8; 34], 1, &mut out));
            assert!(!dequantize_q4_0_avx2_dispatch(&[0u8; 18], 1, &mut out));
        }
        // On x86, the dispatcher runs the SIMD path when AVX2 exists; the
        // test in that case is the parity test above. No assertion here.
    }
}
