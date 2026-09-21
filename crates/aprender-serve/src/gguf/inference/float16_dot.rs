// #3076: F16 / BF16 weight-row · f32 activation dot products for `float16_matmul`.
//
// The row loop used to decode each weight through a `fn(u16) -> f32` pointer: one indirect
// call per element, which nothing could inline or vectorise. Measured on a Threadripper 7960X
// (Zen 4, 24 cores), Qwen2.5-0.5B-Instruct-f16 decoded at ~10 tok/s, and the 272 MB LM head
// read at ~27 GB/s. These kernels read the row's bytes directly: AVX2 + F16C + FMA on x86_64
// when the CPU has them (runtime-detected), and a chunked decode-then-dot everywhere else.

/// The 2-byte float format a weight row is stored in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Float16Kind {
    /// IEEE 754 binary16 (GGUF type 1).
    F16,
    /// bfloat16, the top half of an f32 (GGUF type 30).
    Bf16,
}

/// Dot product of a row of little-endian 2-byte floats with `x`.
///
/// Covers `min(row.len() / 2, x.len())` elements: a trailing odd byte is not an element, which
/// is exactly what the per-element `offset + 1 < data.len()` check it replaces admitted.
pub(super) fn float16_row_dot(kind: Float16Kind, row: &[u8], x: &[f32]) -> f32 {
    let n = (row.len() / 2).min(x.len());
    let (row, x) = (&row[..n * 2], &x[..n]);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            match kind {
                Float16Kind::F16 if is_x86_feature_detected!("f16c") => {
                    // SAFETY: avx2, fma and f16c verified at runtime; `row.len() == 2 * x.len()`.
                    return unsafe { f16_row_dot_avx2(row, x) };
                },
                Float16Kind::Bf16 => {
                    // SAFETY: avx2 and fma verified at runtime; `row.len() == 2 * x.len()`.
                    return unsafe { bf16_row_dot_avx2(row, x) };
                },
                Float16Kind::F16 => {},
            }
        }
    }
    float16_row_dot_portable(kind, row, x)
}

/// A bfloat16 is the top 16 bits of an f32.
#[inline]
fn bf16_bits_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

/// Decode one element (`bits` is the little-endian u16 of the weight).
#[inline]
fn decode_float16(kind: Float16Kind, bits: u16) -> f32 {
    match kind {
        Float16Kind::F16 => crate::quantize::f16_to_f32_lut(bits),
        Float16Kind::Bf16 => bf16_bits_to_f32(bits),
    }
}

/// The non-x86 path, and the x86 path on a CPU without the features above: decode a chunk into
/// a stack buffer, then accumulate it in eight independent lanes the compiler can vectorise.
pub(super) fn float16_row_dot_portable(kind: Float16Kind, row: &[u8], x: &[f32]) -> f32 {
    const CHUNK: usize = 64;
    let mut buf = [0.0f32; CHUNK];
    let mut lanes = [0.0f32; 8];
    let mut tail = 0.0f32;
    for (bytes, xs) in row.chunks(CHUNK * 2).zip(x.chunks(CHUNK)) {
        let m = xs.len();
        for (w, b) in buf[..m].iter_mut().zip(bytes.chunks_exact(2)) {
            *w = decode_float16(kind, u16::from_le_bytes([b[0], b[1]]));
        }
        let (ws, xs8) = (buf[..m].chunks_exact(8), xs.chunks_exact(8));
        let (w_rem, x_rem) = (ws.remainder(), xs8.remainder());
        for (w8, x8) in ws.zip(xs8) {
            for l in 0..8 {
                lanes[l] += w8[l] * x8[l];
            }
        }
        tail += w_rem.iter().zip(x_rem).map(|(w, x)| w * x).sum::<f32>();
    }
    lanes.iter().sum::<f32>() + tail
}

/// The shared AVX2 loop: four accumulators of fused multiply-adds over 32 elements, then
/// 8 at a time, then a scalar tail. A macro rather than a generic function, so that each
/// expansion sits inside a kernel whose target features include its `$load8`, and the 8-lane
/// load inlines. A closure would not inherit those features, and would cost a call per 8
/// weights.
#[cfg(target_arch = "x86_64")]
macro_rules! avx2_float16_row_dot {
    ($row:ident, $x:ident, $load8:ident, $decode1:expr) => {{
        use std::arch::x86_64::{
            _mm256_add_ps, _mm256_castps256_ps128, _mm256_extractf128_ps, _mm256_fmadd_ps,
            _mm256_loadu_ps, _mm256_setzero_ps, _mm_add_ps, _mm_cvtss_f32, _mm_hadd_ps,
        };
        let n = $x.len();
        let (rp, xp) = ($row.as_ptr(), $x.as_ptr());
        let (mut a0, mut a1, mut a2, mut a3) =
            (_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps());
        let mut i = 0;
        // `i + 32 <= n` (then `i + 8 <= n`) bounds every 16-byte `row` load at byte `2 * i`
        // and every 8-float `x` load at `i`, given `row.len() == 2 * x.len()`.
        while i + 32 <= n {
            a0 = _mm256_fmadd_ps($load8(rp.add(2 * i)), _mm256_loadu_ps(xp.add(i)), a0);
            a1 = _mm256_fmadd_ps($load8(rp.add(2 * i + 16)), _mm256_loadu_ps(xp.add(i + 8)), a1);
            a2 = _mm256_fmadd_ps($load8(rp.add(2 * i + 32)), _mm256_loadu_ps(xp.add(i + 16)), a2);
            a3 = _mm256_fmadd_ps($load8(rp.add(2 * i + 48)), _mm256_loadu_ps(xp.add(i + 24)), a3);
            i += 32;
        }
        while i + 8 <= n {
            a0 = _mm256_fmadd_ps($load8(rp.add(2 * i)), _mm256_loadu_ps(xp.add(i)), a0);
            i += 8;
        }
        let acc = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
        let s = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps::<1>(acc));
        let s = _mm_hadd_ps(s, s);
        let mut sum = _mm_cvtss_f32(_mm_hadd_ps(s, s));
        for j in i..n {
            sum += $decode1(u16::from_le_bytes([$row[2 * j], $row[2 * j + 1]])) * $x[j];
        }
        sum
    }};
}

/// 8 F16 weights at `p` as f32 lanes (`vcvtph2ps`).
///
/// # Safety
/// The CPU must support avx2 and f16c, and `p..p + 16` must be readable.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c")]
#[inline]
unsafe fn load8_f16(p: *const u8) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::{__m128i, _mm256_cvtph_ps, _mm_loadu_si128};
    // SAFETY: the caller guarantees 16 readable bytes at `p` and the target features.
    unsafe { _mm256_cvtph_ps(_mm_loadu_si128(p.cast::<__m128i>())) }
}

/// 8 BF16 weights at `p` as f32 lanes: widen u16 to u32, shift into the top half.
///
/// # Safety
/// The CPU must support avx2, and `p..p + 16` must be readable.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn load8_bf16(p: *const u8) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::{
        __m128i, _mm256_castsi256_ps, _mm256_cvtepu16_epi32, _mm256_slli_epi32, _mm_loadu_si128,
    };
    // SAFETY: the caller guarantees 16 readable bytes at `p` and the target feature.
    unsafe {
        let wide = _mm256_cvtepu16_epi32(_mm_loadu_si128(p.cast::<__m128i>()));
        _mm256_castsi256_ps(_mm256_slli_epi32::<16>(wide))
    }
}

/// F16 row dot on AVX2 + F16C + FMA.
///
/// # Safety
/// The CPU must support avx2, fma and f16c, and `row.len()` must be `2 * x.len()`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn f16_row_dot_avx2(row: &[u8], x: &[f32]) -> f32 {
    // SAFETY: loads are bounded as the macro documents; features are this fn's contract.
    unsafe { avx2_float16_row_dot!(row, x, load8_f16, crate::quantize::f16_to_f32_lut) }
}

/// BF16 row dot on AVX2 + FMA.
///
/// # Safety
/// The CPU must support avx2 and fma, and `row.len()` must be `2 * x.len()`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn bf16_row_dot_avx2(row: &[u8], x: &[f32]) -> f32 {
    // SAFETY: loads are bounded as the macro documents; features are this fn's contract.
    unsafe { avx2_float16_row_dot!(row, x, load8_bf16, bf16_bits_to_f32) }
}
