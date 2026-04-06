//! Benchmark AVX2 vs AVX-VNNI for Q4_0×Q8_0 dot product
#![allow(unsafe_op_in_unsafe_fn)]
#![allow(clippy::needless_range_loop, clippy::cast_ptr_alignment)]

use std::arch::x86_64::*;
use std::time::Instant;

const ITERATIONS: usize = 100_000;
const DIM: usize = 2048; // Typical hidden dim
const Q4_BLOCK_SIZE: usize = 32;
const Q4_BLOCK_BYTES: usize = 18; // 2 bytes scale + 16 bytes quants

fn has_avx_vnni() -> bool {
    let result = unsafe { __cpuid_count(7, 1) };
    (result.eax & (1 << 4)) != 0
}

#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn dot_avx2(q4_data: &[u8], q8_scales: &[f32], q8_quants: &[i8]) -> f32 {
    let num_blocks = DIM / Q4_BLOCK_SIZE;
    let mut acc = _mm256_setzero_ps();
    let offset = _mm256_set1_epi8(8);
    let low_mask = _mm256_set1_epi8(0x0F);

    for block_idx in 0..num_blocks {
        let q4_ptr = q4_data.as_ptr().add(block_idx * Q4_BLOCK_BYTES);
        let q8_ptr = q8_quants.as_ptr().add(block_idx * Q4_BLOCK_SIZE);

        // Read Q4 scale (f16 stored as 2 bytes)
        let scale_bytes = std::ptr::read_unaligned(q4_ptr.cast::<u16>());
        let q4_scale = half::f16::from_bits(scale_bytes).to_f32();

        // Load Q4 nibbles and unpack
        let q4_packed = _mm_loadu_si128(q4_ptr.add(2).cast::<__m128i>());
        let q4_lo = _mm256_and_si256(_mm256_cvtepu8_epi16(q4_packed), low_mask);
        let q4_hi = _mm256_and_si256(_mm256_cvtepu8_epi16(_mm_srli_epi16(q4_packed, 4)), low_mask);

        // Interleave lo/hi to get full 32 values, then subtract offset
        let q4_vals = _mm256_sub_epi8(_mm256_packus_epi16(q4_lo, q4_hi), offset);

        // Load Q8 values
        let q8_vals = _mm256_loadu_si256(q8_ptr.cast::<__m256i>());

        // maddubs: pairs of (u8 × i8) -> i16, then horizontal add pairs
        let products = _mm256_maddubs_epi16(
            _mm256_sign_epi8(q4_vals, q8_vals),
            _mm256_sign_epi8(q8_vals, q8_vals),
        );

        // madd with 1s to sum pairs of i16 -> i32
        let sums = _mm256_madd_epi16(products, _mm256_set1_epi16(1));

        // Convert to float and accumulate with scale
        let scale_vec = _mm256_set1_ps(q4_scale * q8_scales[block_idx]);
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(sums), scale_vec, acc);
    }

    // Horizontal sum
    let hi = _mm256_extractf128_ps(acc, 1);
    let lo = _mm256_castps256_ps128(acc);
    let sum128 = _mm_add_ps(lo, hi);
    let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
    _mm_cvtss_f32(sum32)
}

#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn dot_avx_vnni(q4_data: &[u8], q8_scales: &[f32], q8_quants: &[i8]) -> f32 {
    use std::arch::asm;

    let num_blocks = DIM / Q4_BLOCK_SIZE;
    let mut acc = _mm256_setzero_ps();
    let offset = _mm256_set1_epi8(8);
    let low_mask = _mm256_set1_epi8(0x0F);

    for block_idx in 0..num_blocks {
        let q4_ptr = q4_data.as_ptr().add(block_idx * Q4_BLOCK_BYTES);
        let q8_ptr = q8_quants.as_ptr().add(block_idx * Q4_BLOCK_SIZE);

        let scale_bytes = std::ptr::read_unaligned(q4_ptr.cast::<u16>());
        let q4_scale = half::f16::from_bits(scale_bytes).to_f32();

        let q4_packed = _mm_loadu_si128(q4_ptr.add(2).cast::<__m128i>());
        let q4_lo = _mm256_and_si256(_mm256_cvtepu8_epi16(q4_packed), low_mask);
        let q4_hi = _mm256_and_si256(_mm256_cvtepu8_epi16(_mm_srli_epi16(q4_packed, 4)), low_mask);
        let q4_vals = _mm256_sub_epi8(_mm256_packus_epi16(q4_lo, q4_hi), offset);

        // Make unsigned by adding 8 back
        let q4_unsigned = _mm256_add_epi8(q4_vals, offset);
        let q8_vals = _mm256_loadu_si256(q8_ptr.cast::<__m256i>());

        // Use vpdpbusd: u8 × i8 -> i32 accumulate (VEX-encoded for AVX-VNNI)
        let mut int_acc = _mm256_setzero_si256();
        asm!(
            // VEX.256.66.0F38.W0 50 /r - VPDPBUSD ymm1, ymm2, ymm3/m256
            ".byte 0xc4, 0xe2, 0x6d, 0x50, 0xc1", // vpdpbusd ymm0, ymm2, ymm1
            inout("ymm0") int_acc,
            in("ymm1") q8_vals,
            in("ymm2") q4_unsigned,
            options(pure, nomem, nostack),
        );

        // Subtract bias: 8 * sum(q8) per lane - simplified for benchmark
        let scale_vec = _mm256_set1_ps(q4_scale * q8_scales[block_idx]);
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(int_acc), scale_vec, acc);
    }

    let hi = _mm256_extractf128_ps(acc, 1);
    let lo = _mm256_castps256_ps128(acc);
    let sum128 = _mm_add_ps(lo, hi);
    let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
    _mm_cvtss_f32(sum32)
}

fn main() {
    println!("=== SIMD Dot Product Benchmark ===\n");
    println!("CPU: Intel Core Ultra 7 155H");
    println!("AVX-VNNI available: {}", has_avx_vnni());
    println!("Dimension: {}", DIM);
    println!("Iterations: {}\n", ITERATIONS);

    // Setup test data
    let num_blocks = DIM / Q4_BLOCK_SIZE;
    let mut q4_data = vec![0u8; num_blocks * Q4_BLOCK_BYTES];
    let q8_scales = vec![0.1f32; num_blocks];
    let mut q8_quants = vec![0i8; DIM];

    // Fill with pseudo-random data
    for (i, b) in q4_data.iter_mut().enumerate() {
        *b = ((i * 17 + 3) % 256) as u8;
    }
    for (i, q) in q8_quants.iter_mut().enumerate() {
        *q = (((i * 13 + 7) % 256) as i8).wrapping_sub(64);
    }

    // Warmup
    for _ in 0..1000 {
        unsafe {
            let _ = dot_avx2(&q4_data, &q8_scales, &q8_quants);
        }
    }

    // Benchmark AVX2
    let start = Instant::now();
    let mut result_avx2 = 0.0f32;
    for _ in 0..ITERATIONS {
        unsafe {
            result_avx2 = dot_avx2(&q4_data, &q8_scales, &q8_quants);
        }
    }
    let avx2_time = start.elapsed();

    // Benchmark AVX-VNNI (only if available)
    let vnni_time;
    let result_vnni;
    if has_avx_vnni() {
        // Warmup
        for _ in 0..1000 {
            unsafe {
                let _ = dot_avx_vnni(&q4_data, &q8_scales, &q8_quants);
            }
        }

        let start = Instant::now();
        let mut r = 0.0f32;
        for _ in 0..ITERATIONS {
            unsafe {
                r = dot_avx_vnni(&q4_data, &q8_scales, &q8_quants);
            }
        }
        vnni_time = Some(start.elapsed());
        result_vnni = Some(r);
    } else {
        vnni_time = None;
        result_vnni = None;
    }

    // Results
    let avx2_ns = avx2_time.as_nanos() as f64 / ITERATIONS as f64;
    println!("AVX2 (maddubs+madd):");
    println!("  Time: {:.1} ns/dot", avx2_ns);
    println!("  Result: {:.4}", result_avx2);

    if let (Some(vt), Some(rv)) = (vnni_time, result_vnni) {
        let vnni_ns = vt.as_nanos() as f64 / ITERATIONS as f64;
        let speedup = avx2_ns / vnni_ns;
        println!("\nAVX-VNNI (vpdpbusd):");
        println!("  Time: {:.1} ns/dot", vnni_ns);
        println!("  Result: {:.4}", rv);
        println!("\nSpeedup: {:.2}x", speedup);

        if speedup > 1.1 {
            println!("✓ AVX-VNNI is faster - consider enabling in quantize.rs");
        } else if speedup < 0.9 {
            println!("✗ AVX-VNNI is slower - keep AVX2 path");
        } else {
            println!("≈ Similar performance - AVX2 path is fine");
        }
    } else {
        println!("\nAVX-VNNI: Not available on this CPU");
    }
}
