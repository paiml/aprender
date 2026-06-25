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

/// Convert an f32 value to f16 bit pattern using IEEE 754 round-to-nearest-even.
///
/// This is the **reference** f32→f16 encoder that [`f32_to_f16_scalar`] and the
/// AVX2 path wrap. It is **bit-identical to `half::f16::from_f32`** over the full
/// 2³² f32 domain: round-to-nearest-ties-to-even with a full sticky bit, correct
/// subnormal rounding (round-up into the smallest normal), overflow → ±Inf (e.g.
/// `255.99 → 0x5C00`, `65520.0 → 0x7C00`), and quiet-NaN payload preservation.
///
/// PMAT-905: previously this truncated (`mantissa >> 13`, round-toward-zero) which
/// diverged from IEEE RNE on >440M of the 2³² inputs (e.g. emitted `0x5BFF` for
/// `255.99`, `0x7BFF` for `65520.0`). Its falsifier was tautological (SIMD vs its
/// own truncating reference); the obligation is now oracle-vs-`half`.
#[inline]
pub fn f32_to_f16_single(val: f32) -> u16 {
    let bits = val.to_bits();
    // Sign bit shifted into f16 position (0x8000).
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x007F_FFFF;

    if exp == 0xFF {
        // Inf or NaN.
        if mant == 0 {
            return sign | 0x7C00;
        }
        // NaN: set the quiet bit and preserve the top mantissa bits (matches `half`).
        return sign | 0x7E00 | ((mant >> 13) as u16);
    }

    // Unbiased f32 exponent.
    let unbiased = exp - 127;

    // Overflow: anything that rounds to ≥ 2¹⁶ saturates to ±Inf.
    if unbiased > 15 {
        return sign | 0x7C00;
    }

    if unbiased >= -14 {
        // Normalized f16 range. Drop the low 13 mantissa bits with RNE.
        let half_exp = (unbiased + 15) as u16;
        let m = mant >> 13;
        let round_bit = (mant >> 12) & 1;
        let sticky = (mant & 0x0FFF) != 0;
        let mut h = (half_exp << 10) | (m as u16);
        if round_bit == 1 && (sticky || (m & 1) == 1) {
            // Carry propagates into the exponent; a max-mantissa carry yields the
            // 0x7C00 Inf encoding, exactly as IEEE requires.
            h += 1;
        }
        return sign | h;
    }

    // Subnormal / underflow range (unbiased < -14). Below 2⁻²⁵ rounds to ±0.
    if unbiased < -25 {
        return sign;
    }
    // Restore the implicit leading 1, then shift into f16-subnormal alignment.
    let mant_with_implicit = mant | 0x0080_0000;
    let shift = (-14 - unbiased) + 13;
    if shift >= 32 {
        return sign;
    }
    let m = mant_with_implicit >> shift;
    let round_bit = (mant_with_implicit >> (shift - 1)) & 1;
    let sticky = (mant_with_implicit & ((1u32 << (shift - 1)) - 1)) != 0;
    let mut h = m as u16;
    if round_bit == 1 && (sticky || (m & 1) == 1) {
        // May round up into the smallest normal — correct per IEEE.
        h += 1;
    }
    sign | h
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

    // ────────────────────────────────────────────────────────────────────────
    // PMAT-905: ORACLE-vs-`half` f32→f16 round-to-nearest-even falsifiers.
    //
    // The previous f32→f16 reference TRUNCATED the mantissa (round-toward-zero)
    // and its only falsifier asserted SIMD matched that same truncating
    // reference — a tautology (apr-vs-apr) that could never catch the rounding
    // bug. These tests assert BIT-IDENTITY against the trusted `half` crate
    // (IEEE 754) across the named hard cases plus a wide stride of the full
    // 2³² f32 domain. MUTATION CHECK: reverting `f32_to_f16_single` to
    // `mantissa >> 13` truncation makes `test_f32_to_f16_oracle_*` go RED.
    // ────────────────────────────────────────────────────────────────────────

    /// ORACLE falsifier: the named hard RNE cases that truncation gets wrong.
    /// 255.99 must round UP to 0x5C00 (truncation gives 0x5BFF); 65520.0 must
    /// overflow to Inf 0x7C00 (truncation gives the finite 0x7BFF); a tie with a
    /// non-zero discarded sticky bit must round up; the smallest f32 above the
    /// f16-subnormal midpoint must round to the smallest subnormal 0x0001.
    #[test]
    fn test_f32_to_f16_oracle_named_cases() {
        let cases: &[f32] = &[
            255.99,                      // → 0x5C00 (round up across exponent)
            65520.0,                     // → 0x7C00 (overflow to +Inf)
            -65520.0,                    // → 0xFC00 (overflow to -Inf)
            65504.0,                     // → 0x7BFF (largest finite f16, exact)
            f32::from_bits(0x3300_0001), // tiny → 0x0001 (subnormal round-up)
            f32::from_bits(0x3F80_2000), // exact 1 + 2⁻¹⁰ (representable f16)
        ];
        for &v in cases {
            let ours = f32_to_f16_single(v);
            let oracle = half::f16::from_f32(v).to_bits();
            assert_eq!(
                ours, oracle,
                "f32_to_f16({v}) = 0x{ours:04X}, half = 0x{oracle:04X}"
            );
        }
        // Spell out the two headline regressions explicitly.
        assert_eq!(
            f32_to_f16_single(255.99),
            0x5C00,
            "255.99 must RNE up to 0x5C00"
        );
        assert_eq!(
            f32_to_f16_single(65520.0),
            0x7C00,
            "65520 must overflow to Inf"
        );
    }

    /// ORACLE falsifier: ties-to-even with a non-zero discarded mantissa.
    /// A value exactly on a representable midpoint with sticky bits set must
    /// always round up (sticky beats round-half-to-even), and an exact midpoint
    /// with no sticky bits must round to the even neighbour.
    #[test]
    fn test_f32_to_f16_oracle_ties_to_even() {
        // 13-bit mantissa drop. Build a value whose dropped bits are exactly the
        // midpoint (0x1000) plus a sticky bit, then exactly the midpoint.
        let base = 1.0f32.to_bits(); // 0x3F80_0000
        let midpoint_sticky = f32::from_bits(base | 0x1001); // round bit + sticky → up
        let exact_midpoint = f32::from_bits(base | 0x1000); // tie → to even (down, mant LSB 0)
        for v in [midpoint_sticky, exact_midpoint] {
            assert_eq!(
                f32_to_f16_single(v),
                half::f16::from_f32(v).to_bits(),
                "tie-to-even mismatch for {v}"
            );
        }
    }

    /// ORACLE falsifier (wide grid): bit-identical to `half` over a deterministic
    /// stride across the full 2³² f32 domain — normals, subnormals, ties,
    /// overflow, Inf and NaN payloads. This is the de-tautologized replacement
    /// for FALSIFY-F16-002/004.
    #[test]
    fn test_f32_to_f16_oracle_wide_grid() {
        // Coprime-with-2 stride hits every exponent and a dense set of mantissas.
        let mut b: u32 = 0;
        let stride: u32 = 0x0001_0003;
        loop {
            let v = f32::from_bits(b);
            let ours = f32_to_f16_single(v);
            let oracle = half::f16::from_f32(v).to_bits();
            assert_eq!(
                ours, oracle,
                "f32 bits 0x{b:08X} (v={v}): ours=0x{ours:04X} half=0x{oracle:04X}"
            );
            let (next, overflow) = b.overflowing_add(stride);
            if overflow {
                break;
            }
            b = next;
        }
    }

    /// ORACLE falsifier: every subnormal and zero f16 target reached from f32.
    /// Exhaustively check that each f16 subnormal's exact f32 value, and the
    /// midpoints around it, round bit-identically to `half`.
    #[test]
    fn test_f32_to_f16_oracle_subnormals() {
        for h in 0u16..0x0400 {
            // h is a (positive) zero or subnormal f16 pattern.
            let exact = half::f16::from_bits(h).to_f32();
            assert_eq!(
                f32_to_f16_single(exact),
                half::f16::from_f32(exact).to_bits(),
                "subnormal exact roundtrip mismatch for h=0x{h:04X} (v={exact})"
            );
            // Negative twin.
            assert_eq!(
                f32_to_f16_single(-exact),
                half::f16::from_f32(-exact).to_bits(),
                "negative subnormal mismatch for h=0x{h:04X}"
            );
        }
    }

    proptest! {
        /// ORACLE property: f32_to_f16_single is bit-identical to `half` for
        /// every f32 bit pattern. Replaces the tautological SIMD-vs-self check.
        #[test]
        fn prop_f32_to_f16_matches_half(bits in any::<u32>()) {
            let v = f32::from_bits(bits);
            prop_assert_eq!(
                f32_to_f16_single(v),
                half::f16::from_f32(v).to_bits(),
                "mismatch at f32 bits 0x{:08X} (v={})", bits, v
            );
        }
    }

    /// CONSISTENCY (not oracle): the scalar batch path and the AVX2 path must
    /// agree with each other and with the single-value reference. This is the
    /// legitimate SIMD-vs-scalar check, kept distinct from the oracle gate.
    #[test]
    fn test_f32_to_f16_simd_scalar_consistency() {
        let input: Vec<f32> = vec![
            255.99,
            65520.0,
            1.0,
            2.0,
            0.5,
            -1.0,
            1e-8,
            1e8,
            f32::from_bits(0x3300_0001),
        ];
        let mut scalar_out = vec![0u16; input.len()];
        f32_to_f16_scalar(&input, &mut scalar_out);
        for (v, &got) in input.iter().zip(scalar_out.iter()) {
            assert_eq!(
                got,
                f32_to_f16_single(*v),
                "batch vs single mismatch for {v}"
            );
        }
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") {
            let mut avx2_out = vec![0u16; input.len()];
            unsafe { f32_to_f16_avx2(&input, &mut avx2_out) };
            assert_eq!(scalar_out, avx2_out, "AVX2 f32→f16 diverges from scalar");
        }
    }
}
