//! AVX-512 SIMD Q4_K GEMV implementation.
//!
//! Contract: avx512-q4k-v1.yaml (C-AVX512-Q4K-001)
//! Processes 16 elements per iteration using zmm registers (2× throughput vs AVX2).
//! References: [46] GPTQ, [47] QuIP# AVX-512 dequant methodology.

use super::super::{parse_q4k_header, SUPER_BLOCK_BYTES, SUPER_BLOCK_SIZE};

/// Fused Q4_K matrix-vector multiply with AVX-512 SIMD (16-wide)
///
/// Contract: avx512-q4k-v1.yaml (C-AVX512-Q4K-001, C-AVX512-Q4K-002)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f", enable = "avx512bw", enable = "fma")]
pub(crate) unsafe fn matmul_q4k_f32_avx512(
    q4k_data: &[u8],
    input: &[f32],
    out_dim: usize,
    in_dim: usize,
) -> Vec<f32> {
    unsafe {
        use std::arch::x86_64::*;

        let num_blocks_per_row = (in_dim + SUPER_BLOCK_SIZE - 1) / SUPER_BLOCK_SIZE;
        let row_bytes = num_blocks_per_row * SUPER_BLOCK_BYTES;
        let low_mask = _mm512_set1_epi32(0x0F);

        // Uninit: output[out_idx] = hsum_avx512(acc) (SET) for every out_idx.
        let mut output: Vec<f32> = Vec::with_capacity(out_dim);
        // SAFETY: Each output[out_idx] is SET from local AVX-512 accumulator.
        output.set_len(out_dim);

        for out_idx in 0..out_dim {
            let row_start = out_idx * row_bytes;
            let mut acc = _mm512_setzero_ps();

            for sb_idx in 0..num_blocks_per_row {
                let sb_start = row_start + sb_idx * SUPER_BLOCK_BYTES;
                let sb_data = &q4k_data[sb_start..sb_start + SUPER_BLOCK_BYTES];
                let input_offset = sb_idx * SUPER_BLOCK_SIZE;
                process_q4k_superblock_avx512(
                    sb_data,
                    input,
                    input_offset,
                    in_dim,
                    low_mask,
                    &mut acc,
                );
            }

            output[out_idx] = hsum_avx512(acc);
        }

        contract_post_dequant!(output);
        output
    }
}

/// Bounds-aware 16-wide activation load for the Q4_K reduction tail.
///
/// Contract: avx512-q4k-v1.yaml (C-AVX512-Q4K-003).
/// Lanes whose activation index is ≥ `in_dim` are masked: they are NOT read
/// (no heap over-read past the activation buffer) and load as `0.0` (so the
/// downstream FMA contributes nothing for those positions). Used only on the
/// partial-chunk tail; the full-chunk hot path uses unguarded loads.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn load16_bounded(
    input_ptr: *const f32,
    base: usize,
    in_dim: usize,
) -> std::arch::x86_64::__m512 {
    use std::arch::x86_64::*;
    unsafe {
        if base + 16 <= in_dim {
            _mm512_loadu_ps(input_ptr.add(base))
        } else if base < in_dim {
            // 1..=15 valid low lanes; the rest are masked (unread, zeroed).
            let mask: __mmask16 = ((1u32 << (in_dim - base)) - 1) as __mmask16;
            _mm512_maskz_loadu_ps(mask, input_ptr.add(base))
        } else {
            _mm512_setzero_ps()
        }
    }
}

/// Process one Q4K super-block with AVX-512 (16-wide), fully unrolled.
///
/// Each super-block = 256 elements in 4 chunks of 64.
/// Each chunk: 32 low nibbles + 32 high nibbles.
/// AVX-512: 16 elements per iteration → 2 iterations per 32 nibbles.
///
/// Optimization (Phase 4, 2026-04-05):
/// - Fully unrolled inner loops (was while loop with 2 iterations)
/// - Bounds check hoisted out of hot loop (in_dim validated by caller)
/// - Software prefetch of next superblock's quantized data
///
/// NOTE: Dual-accumulator (low→acc0, high→acc1) was tested (2026-04-05)
/// but showed NO improvement. Zen 4's OOO engine already hides the FMA
/// dependency chain across iterations — adding a second accumulator just
/// adds merge overhead without helping the pipeline.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f", enable = "avx512bw", enable = "fma")]
unsafe fn process_q4k_superblock_avx512(
    sb_data: &[u8],
    input: &[f32],
    input_offset: usize,
    in_dim: usize,
    low_mask: std::arch::x86_64::__m512i,
    acc: &mut std::arch::x86_64::__m512,
) {
    unsafe {
        use std::arch::x86_64::*;

        let (d, dmin, scales, mins) = parse_q4k_header(sb_data);
        let qs = sb_data.get(16..144).expect("Q4_K: need ≥144 bytes for qs");
        let qs_ptr = qs.as_ptr();
        let input_ptr = input.as_ptr();

        // Software prefetch: next superblock's header + first quant bytes
        // Prefetch 2 cache lines ahead (128 bytes = most of next superblock)
        _mm_prefetch(sb_data.as_ptr().add(SUPER_BLOCK_BYTES) as *const i8, _MM_HINT_T0);
        _mm_prefetch(sb_data.as_ptr().add(SUPER_BLOCK_BYTES + 64) as *const i8, _MM_HINT_T0);

        for chunk_i in 0..4 {
            let chunk_start = chunk_i * 64;
            let q_start = chunk_i * 32;
            let input_base_lo0 = input_offset + chunk_start;

            // Skip chunks whose activation positions are entirely beyond the
            // reduction dim (the original code skipped these via its bounds
            // guard; we make it explicit so the tail branch below stays simple).
            if input_base_lo0 >= in_dim {
                continue;
            }

            let d1 = d * f32::from(scales[chunk_i * 2]);
            let dm1 = dmin * f32::from(mins[chunk_i * 2]);
            let d2 = d * f32::from(scales[chunk_i * 2 + 1]);
            let dm2 = dmin * f32::from(mins[chunk_i * 2 + 1]);

            let d1_vec = _mm512_set1_ps(d1);
            let dm1_vec = _mm512_set1_ps(dm1);
            let d2_vec = _mm512_set1_ps(d2);
            let dm2_vec = _mm512_set1_ps(dm2);

            // Dequant the 4×16 weight groups. `qs` is always fully in bounds (a
            // super-block is always SUPER_BLOCK_BYTES). Low nibbles map to
            // activation positions [base, base+32); high nibbles to
            // [base+32, base+64) — DIFFERENT positions, which is why the high
            // loads need their own bound (the bug this fixes).
            let q0 = _mm_loadu_si128(qs_ptr.add(q_start) as *const __m128i);
            let q0_i32 = _mm512_cvtepu8_epi32(q0);
            let q1 = _mm_loadu_si128(qs_ptr.add(q_start + 16) as *const __m128i);
            let q1_i32 = _mm512_cvtepu8_epi32(q1);
            let dq0 = _mm512_fmsub_ps(
                d1_vec,
                _mm512_cvtepi32_ps(_mm512_and_si512(q0_i32, low_mask)),
                dm1_vec,
            );
            let dq1 = _mm512_fmsub_ps(
                d1_vec,
                _mm512_cvtepi32_ps(_mm512_and_si512(q1_i32, low_mask)),
                dm1_vec,
            );
            let dqh0 =
                _mm512_fmsub_ps(d2_vec, _mm512_cvtepi32_ps(_mm512_srli_epi32(q0_i32, 4)), dm2_vec);
            let dqh1 =
                _mm512_fmsub_ps(d2_vec, _mm512_cvtepi32_ps(_mm512_srli_epi32(q1_i32, 4)), dm2_vec);

            let input_base_hi0 = input_base_lo0 + 32;

            if input_base_lo0 + 64 <= in_dim {
                // FAST PATH: the whole 64-element chunk is in bounds (the common
                // 256-aligned case) — unguarded 16-wide loads, hot-path perf
                // preserved.
                let x0 = _mm512_loadu_ps(input_ptr.add(input_base_lo0));
                let x1 = _mm512_loadu_ps(input_ptr.add(input_base_lo0 + 16));
                let xh0 = _mm512_loadu_ps(input_ptr.add(input_base_hi0));
                let xh1 = _mm512_loadu_ps(input_ptr.add(input_base_hi0 + 16));
                *acc = _mm512_fmadd_ps(dq0, x0, *acc);
                *acc = _mm512_fmadd_ps(dq1, x1, *acc);
                *acc = _mm512_fmadd_ps(dqh0, xh0, *acc);
                *acc = _mm512_fmadd_ps(dqh1, xh1, *acc);
            } else {
                // TAIL PATH: partial chunk — bounded masked loads. Lanes ≥ in_dim
                // are NOT read (no heap over-read) and load as 0.0 (0 FMA
                // contribution). Fixes C-AVX512-Q4K-003: the previous single
                // `input_base_lo0 + 32 <= in_dim` guard covered only the low
                // loads, so the high-nibble loads at [+32,+64) over-read up to
                // 128 bytes for any non-256-aligned in_dim (32/96/160/224 mod
                // 256). It also silently dropped the valid high-nibble tail —
                // masked loads compute it correctly.
                let x0 = load16_bounded(input_ptr, input_base_lo0, in_dim);
                let x1 = load16_bounded(input_ptr, input_base_lo0 + 16, in_dim);
                let xh0 = load16_bounded(input_ptr, input_base_hi0, in_dim);
                let xh1 = load16_bounded(input_ptr, input_base_hi0 + 16, in_dim);
                *acc = _mm512_fmadd_ps(dq0, x0, *acc);
                *acc = _mm512_fmadd_ps(dq1, x1, *acc);
                *acc = _mm512_fmadd_ps(dqh0, xh0, *acc);
                *acc = _mm512_fmadd_ps(dqh1, xh1, *acc);
            }
        }
    }
}

/// AVX-512 horizontal sum of 16 f32 lanes.
/// Uses avx512f-only intrinsics (no avx512dq dependency).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn hsum_avx512(v: std::arch::x86_64::__m512) -> f32 {
    use std::arch::x86_64::*;
    // Reduce 512→256 using shuffle instead of extractf32x8 (which needs avx512dq)
    let lo256 = _mm512_castps512_ps256(v);
    // Shift upper 256 bits down: use _mm512_shuffle_f32x4 to bring lanes 8-15 into 0-7
    let hi_shifted = _mm512_shuffle_f32x4(v, v, 0b_01_00_11_10); // swap upper and lower 256
    let hi256 = _mm512_castps512_ps256(hi_shifted);
    let sum256 = _mm256_add_ps(lo256, hi256);
    // Now reduce 256→scalar
    let hi128 = _mm256_extractf128_ps(sum256, 1);
    let lo128 = _mm256_castps256_ps128(sum256);
    let sum128 = _mm_add_ps(lo128, hi128);
    let hi64 = _mm_movehl_ps(sum128, sum128);
    let sum64 = _mm_add_ps(sum128, hi64);
    let hi32 = _mm_shuffle_ps(sum64, sum64, 1);
    let sum32 = _mm_add_ss(sum64, hi32);
    _mm_cvtss_f32(sum32)
}

/// AVX-512 chunk processor for parallel dispatch.
/// Contract: avx512-q4k-v1.yaml
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f", enable = "avx512bw", enable = "fma")]
pub(crate) unsafe fn compute_chunk_q4k_avx512(
    q4k_data: &[u8],
    input: &[f32],
    chunk: &mut [f32],
    start_row: usize,
    out_dim: usize,
    in_dim: usize,
    num_blocks_per_row: usize,
    row_bytes: usize,
) {
    unsafe {
        use std::arch::x86_64::*;

        let low_mask = _mm512_set1_epi32(0x0F);

        for (local_idx, out_val) in chunk.iter_mut().enumerate() {
            let out_idx = start_row + local_idx;
            if out_idx >= out_dim {
                break;
            }
            let row_start = out_idx * row_bytes;
            let mut acc = _mm512_setzero_ps();

            for sb_idx in 0..num_blocks_per_row {
                let sb_start = row_start + sb_idx * SUPER_BLOCK_BYTES;
                let sb_data = &q4k_data[sb_start..sb_start + SUPER_BLOCK_BYTES];
                let input_offset = sb_idx * SUPER_BLOCK_SIZE;
                process_q4k_superblock_avx512(
                    sb_data,
                    input,
                    input_offset,
                    in_dim,
                    low_mask,
                    &mut acc,
                );
            }

            *out_val = hsum_avx512(acc);
        }
    }
    contract_post_dequant!(chunk);
}
