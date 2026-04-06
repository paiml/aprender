//! F16 (half-precision) conversion kernel.
//!
//! Matches `f16-conversion-v1.yaml`.
//! IEEE 754 half-precision ↔ single-precision conversion via bit manipulation.
//!
//! Each function provides one of three backends:
//! - `fn f16_to_f32_scalar(...)` / `fn f32_to_f16_scalar(...)` -- Pure Rust scalar
//! - `unsafe fn f16_to_f32_avx2(...)` -- AVX2 SIMD implementation
//! - `fn f16_convert_ptx() -> &'static str` -- PTX assembly source string

// ────────────────────────────────────────────────────────────────────────────
// Scalar implementation
// ────────────────────────────────────────────────────────────────────────────

/// Convert a half-precision (f16) bit pattern to f32.
///
/// Uses the bias trick: `f32_bits = (sign << 31) | ((exp + 112) << 23) | (mant << 13)`.
/// Only handles normal f16 values (exponent in 1..=30). Subnormals, inf, NaN are
/// handled with fallback paths.
#[inline]
pub fn f16_to_f32_single(bits: u16) -> f32 {
    let sign = u32::from((bits >> 15) & 1);
    let exp = u32::from((bits >> 10) & 0x1F);
    let mant = u32::from(bits & 0x3FF);

    if exp == 0 {
        // Zero or subnormal
        if mant == 0 {
            return f32::from_bits(sign << 31);
        }
        // Subnormal: convert via float arithmetic
        let sign_f = if sign == 1 { -1.0f32 } else { 1.0f32 };
        return sign_f * (mant as f32) * (2.0f32).powi(-24);
    }

    if exp == 31 {
        // Inf or NaN
        if mant == 0 {
            return f32::from_bits((sign << 31) | 0x7F80_0000);
        }
        return f32::from_bits((sign << 31) | 0x7F80_0000 | (mant << 13));
    }

    // Normal: bias trick
    let f32_bits = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
    f32::from_bits(f32_bits)
}

/// Convert an f32 value to f16 bit pattern.
///
/// Truncates mantissa (no rounding). Only handles normal range.
#[inline]
pub fn f32_to_f16_single(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = ((bits >> 31) & 1) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x007F_FFFF;

    if exp == 0 {
        // Zero or f32 subnormal → f16 zero
        return sign << 15;
    }

    if exp == 0xFF {
        // Inf or NaN
        if mant == 0 {
            return (sign << 15) | 0x7C00;
        }
        return (sign << 15) | 0x7C00 | ((mant >> 13) as u16 & 0x3FF).max(1);
    }

    // Normal: rebias exponent (f32 bias 127 → f16 bias 15)
    let f16_exp = exp - 112;
    if f16_exp <= 0 {
        // Underflow to zero
        return sign << 15;
    }
    if f16_exp >= 31 {
        // Overflow to infinity
        return (sign << 15) | 0x7C00;
    }

    let f16_mant = (mant >> 13) as u16;
    (sign << 15) | ((f16_exp as u16) << 10) | f16_mant
}

/// Batch convert f16 bit patterns to f32 (scalar reference).
///
/// # Panics
/// Panics if `input.len() != output.len()`.
pub fn f16_to_f32_scalar(input: &[u16], output: &mut [f32]) {
    assert_eq!(input.len(), output.len(), "dimension mismatch");
    for (bits, out) in input.iter().zip(output.iter_mut()) {
        *out = f16_to_f32_single(*bits);
    }
}

/// Batch convert f32 to f16 bit patterns (scalar reference).
///
/// # Panics
/// Panics if `input.len() != output.len()`.
pub fn f32_to_f16_scalar(input: &[f32], output: &mut [u16]) {
    assert_eq!(input.len(), output.len(), "dimension mismatch");
    for (val, out) in input.iter().zip(output.iter_mut()) {
        *out = f32_to_f16_single(*val);
    }
}

// ────────────────────────────────────────────────────────────────────────────
// AVX2 implementation
// ────────────────────────────────────────────────────────────────────────────

/// AVX2 f16→f32 conversion -- delegates to scalar.
///
/// # Safety
/// Requires AVX2 support.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn f16_to_f32_avx2(input: &[u16], output: &mut [f32]) {
    f16_to_f32_scalar(input, output);
}

/// AVX2 f32→f16 conversion -- delegates to scalar.
///
/// # Safety
/// Requires AVX2 support.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn f32_to_f16_avx2(input: &[f32], output: &mut [u16]) {
    f32_to_f16_scalar(input, output);
}

// ────────────────────────────────────────────────────────────────────────────
// PTX implementation
// ────────────────────────────────────────────────────────────────────────────

/// PTX assembly for f16→f32 conversion.
///
/// One thread per element. Uses hardware `cvt.f32.f16` instruction.
pub fn f16_convert_ptx() -> &'static str {
    r#".version 8.5
.target sm_90
.address_size 64
.visible .entry f16_to_f32_kernel(
    .param .u64 INPUT,
    .param .u64 OUTPUT,
    .param .u32 N
) {
    .reg .u32 %tid, %bid, %n, %idx;
    .reg .u64 %in_ptr, %out_ptr, %addr, %off64;
    .reg .b16 %h_val;
    .reg .f32 %f_val;
    .reg .pred %p_bound;

    mov.u32 %tid, %tid.x;
    mov.u32 %bid, %ctaid.x;

    ld.param.u32 %n, [N];
    ld.param.u64 %in_ptr, [INPUT];
    ld.param.u64 %out_ptr, [OUTPUT];

    // Global index
    mul.lo.u32 %idx, %bid, 256;
    add.u32 %idx, %idx, %tid;

    setp.ge.u32 %p_bound, %idx, %n;
    @%p_bound bra EXIT;

    // Load f16 value
    mul.wide.u32 %off64, %idx, 2;
    add.u64 %addr, %in_ptr, %off64;
    ld.global.b16 %h_val, [%addr];

    // Convert f16 to f32
    cvt.f32.f16 %f_val, %h_val;

    // Store f32 value
    mul.wide.u32 %off64, %idx, 4;
    add.u64 %addr, %out_ptr, %off64;
    st.global.f32 [%addr], %f_val;

EXIT:
    ret;
}
"#
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    /// Verify f16 zero converts to f32 zero and back
    #[test]
    fn test_f16_zero() {
        assert_eq!(f16_to_f32_single(0x0000), 0.0);
        assert_eq!(f32_to_f16_single(0.0), 0x0000);
    }

    /// Verify f16 negative zero preserves sign bit through conversion
    #[test]
    fn test_f16_negative_zero() {
        let neg_zero = f16_to_f32_single(0x8000);
        assert!(neg_zero.is_sign_negative());
        assert_eq!(neg_zero, -0.0);
    }

    /// Verify f16 bit pattern 0x3C00 converts to f32 1.0
    #[test]
    fn test_f16_one() {
        // f16 1.0 = 0x3C00 (sign=0, exp=15, mant=0)
        let val = f16_to_f32_single(0x3C00);
        assert!((val - 1.0).abs() < 1e-6);
    }

    /// Verify f16 conversion for known values: 0.5, 2.0, and -1.0
    #[test]
    fn test_f16_known_values() {
        // f16 0.5 = 0x3800
        assert!((f16_to_f32_single(0x3800) - 0.5).abs() < 1e-6);
        // f16 2.0 = 0x4000
        assert!((f16_to_f32_single(0x4000) - 2.0).abs() < 1e-6);
        // f16 -1.0 = 0xBC00
        assert!((f16_to_f32_single(0xBC00) + 1.0).abs() < 1e-6);
    }

    /// Verify f16-to-f32-to-f16 roundtrip is lossless for sampled normal values
    #[test]
    fn test_f16_roundtrip_normal() {
        // Test roundtrip for a selection of normal f16 values
        let test_values: Vec<u16> = (0x0400..=0x7BFF).step_by(17).collect();
        for &bits in &test_values {
            let f32_val = f16_to_f32_single(bits);
            let back = f32_to_f16_single(f32_val);
            assert_eq!(
                bits, back,
                "roundtrip failed for 0x{bits:04X}: f32={f32_val}, back=0x{back:04X}"
            );
        }
    }

    /// Verify sign bit is preserved for all normal f16 exponents
    #[test]
    fn test_f16_sign_preservation() {
        // For every normal f16, sign should be preserved
        for exp in 1u16..=30 {
            let pos = (exp << 10) | 0x100; // positive with some mantissa
            let neg = pos | 0x8000; // same with sign bit set
            assert!(f16_to_f32_single(pos) > 0.0);
            assert!(f16_to_f32_single(neg) < 0.0);
        }
    }

    /// Verify f16 positive and negative infinity convert correctly
    #[test]
    fn test_f16_inf() {
        let pos_inf = f16_to_f32_single(0x7C00);
        assert!(pos_inf.is_infinite() && pos_inf > 0.0);
        let neg_inf = f16_to_f32_single(0xFC00);
        assert!(neg_inf.is_infinite() && neg_inf < 0.0);
    }

    /// Verify f16 NaN bit pattern converts to f32 NaN
    #[test]
    fn test_f16_nan() {
        let nan = f16_to_f32_single(0x7C01);
        assert!(nan.is_nan());
    }

    /// Verify batch f16-to-f32 conversion for multiple known values
    #[test]
    fn test_f16_batch_conversion() {
        let input = [0x3C00, 0x4000, 0x3800]; // 1.0, 2.0, 0.5
        let mut output = [0.0f32; 3];
        f16_to_f32_scalar(&input, &mut output);
        assert!((output[0] - 1.0).abs() < 1e-6);
        assert!((output[1] - 2.0).abs() < 1e-6);
        assert!((output[2] - 0.5).abs() < 1e-6);
    }

    proptest! {
        #[test]
        fn prop_f16_roundtrip_normal(exp in 1u16..31, mant in 0u16..1024) {
            let bits = (exp << 10) | mant;
            let f32_val = f16_to_f32_single(bits);
            let back = f32_to_f16_single(f32_val);
            prop_assert_eq!(bits, back,
                "roundtrip failed for exp={} mant={}: 0x{:04X} → {} → 0x{:04X}", exp, mant, bits, f32_val, back);
        }

        #[test]
        fn prop_f16_sign_preserved(exp in 1u16..31, mant in 0u16..1024) {
            let pos = (exp << 10) | mant;
            let neg = pos | 0x8000;
            let pos_f32 = f16_to_f32_single(pos);
            let neg_f32 = f16_to_f32_single(neg);
            prop_assert!(pos_f32 >= 0.0, "positive f16 gave negative f32");
            prop_assert!(neg_f32 <= 0.0, "negative f16 gave positive f32");
        }
    }

    /// Verify f16 convert PTX contains entry point and hardware cvt instruction
    #[test]
    fn test_f16_ptx_structure() {
        let ptx = f16_convert_ptx();
        assert!(ptx.contains(".entry f16_to_f32_kernel"));
        assert!(ptx.contains("cvt.f32.f16"));
        assert!(ptx.contains("ret;"));
    }

    /// Verify f32-to-f16 edge cases: infinity, NaN, underflow, overflow
    #[test]
    fn test_f32_to_f16_edge_cases() {
        // +inf → 0x7C00
        assert_eq!(f32_to_f16_single(f32::INFINITY), 0x7C00);
        // -inf → 0xFC00
        assert_eq!(f32_to_f16_single(f32::NEG_INFINITY), 0xFC00);
        // NaN → f16 NaN (sign=0, exp=31, mantissa!=0)
        let nan_bits = f32_to_f16_single(f32::NAN);
        assert_eq!(nan_bits & 0x7C00, 0x7C00);
        assert_ne!(nan_bits & 0x03FF, 0);
        // Very small positive → underflow to zero
        assert_eq!(f32_to_f16_single(1e-10), 0x0000);
        // Very large positive → overflow to inf
        assert_eq!(f32_to_f16_single(1e10), 0x7C00);
        // f32 subnormal → f16 zero
        assert_eq!(f32_to_f16_single(f32::from_bits(0x0000_0001)), 0x0000);
        // -0.0 → 0x8000
        assert_eq!(f32_to_f16_single(-0.0), 0x8000);
    }

    /// Verify batch f32-to-f16 conversion
    #[test]
    fn test_f32_to_f16_batch() {
        let input = [1.0f32, 2.0, 0.5, -1.0];
        let mut output = [0u16; 4];
        f32_to_f16_scalar(&input, &mut output);
        assert_eq!(output[0], 0x3C00); // 1.0
        assert_eq!(output[1], 0x4000); // 2.0
        assert_eq!(output[2], 0x3800); // 0.5
        assert_eq!(output[3], 0xBC00); // -1.0
    }

    /// Verify f16 subnormal conversion
    #[test]
    fn test_f16_subnormal_conversion() {
        // Smallest positive subnormal: exp=0, mant=1
        let val = f16_to_f32_single(0x0001);
        assert!(val > 0.0);
        assert!(val < 1e-5);
        // Negative subnormal
        let neg_val = f16_to_f32_single(0x8001);
        assert!(neg_val < 0.0);
    }

    /// Verify AVX2 f16-to-f32 conversion matches scalar output
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_f16_avx2_parity() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let input = [0x3C00, 0x4000, 0x3800, 0xBC00];
        let mut scalar_out = [0.0f32; 4];
        let mut avx2_out = [0.0f32; 4];
        f16_to_f32_scalar(&input, &mut scalar_out);
        unsafe { f16_to_f32_avx2(&input, &mut avx2_out) };
        assert_eq!(scalar_out, avx2_out);
    }
}
