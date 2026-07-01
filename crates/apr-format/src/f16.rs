//! IEEE-754 half-precision (f16) conversion, delegating to the `half` crate.
//!
//! These replace the `trueno::f32_to_f16` / `trueno::f16_to_f32` calls that the
//! v2 reader/writer used (`format/v2/mod.rs:105` / `:112`). The `half` crate is
//! a tiny, dependency-free, well-audited binary16 implementation that performs
//! IEEE-754 round-to-nearest-even and is therefore the **correct** conversion.
//! The legacy hand-rolled `trueno::f32_to_f16` was round-half-up and carried a
//! mantissa-overflow carry bug that emitted the WRONG exponent (e.g.
//! `255.99 -> 0xD800` instead of the correct `0xDC00`).
//!
//! Because of this, **v2 f16-written tensor bytes change** relative to the
//! pre-extraction writer — this is a documented PMAT-905-class bug-fix, pinned
//! by `test_f16_parity_with_trueno_ref_known_divergence` (31 divergences) below.
//! On-disk byte-identity is preserved for F32 (the dominant case) but is
//! intentionally corrected for f16. See `crate::falsifiers` for the F32-scoped
//! byte-identity oracle.

use half::f16;

/// Convert an `f32` to IEEE-754 binary16 bits (round to nearest even).
#[inline]
#[must_use]
pub fn f32_to_f16(value: f32) -> u16 {
    f16::from_f32(value).to_bits()
}

/// Convert IEEE-754 binary16 bits back to `f32` (exact for normal f16).
#[inline]
#[must_use]
pub fn f16_to_f32(bits: u16) -> f32 {
    f16::from_bits(bits).to_f32()
}

#[cfg(test)]
mod tests {
    use super::{f16_to_f32, f32_to_f16};

    /// Reference port of the legacy `trueno::f32_to_f16` bit-twiddle, kept here
    /// ONLY to prove the `half`-crate path is bit-identical to the code it
    /// replaced. If this ever diverges, the v2 f16 bytes would change.
    fn trueno_ref_f32_to_f16(x: f32) -> u16 {
        let bits = x.to_bits();
        let sign = ((bits >> 16) & 0x8000) as u16;
        let exponent = ((bits >> 23) & 0xFF) as i32;
        let mantissa = bits & 0x007F_FFFF;
        if exponent == 255 {
            if mantissa == 0 {
                return sign | 0x7C00;
            }
            return sign | 0x7C00 | ((mantissa >> 13) as u16).max(1);
        }
        let new_exp = exponent - 112;
        if new_exp >= 31 {
            return sign | 0x7C00;
        }
        if new_exp <= 0 {
            if new_exp < -10 {
                return sign;
            }
            let mant = (mantissa | 0x0080_0000) >> (1 - new_exp + 13);
            return sign | mant as u16;
        }
        let round_bit = (mantissa >> 12) & 1;
        let mant16 = ((mantissa >> 13) as u16) + round_bit as u16;
        sign | ((new_exp as u16) << 10) | (mant16 & 0x03FF)
    }

    #[test]
    fn test_f16_roundtrip_representable() {
        for &v in &[0.0_f32, 1.0, -1.0, 0.5, 2.0, -2.5, 100.0, 0.001] {
            let back = f16_to_f32(f32_to_f16(v));
            // f16 has ~3 decimal digits of precision.
            assert!(
                (back - v).abs() <= v.abs() * 1e-2 + 1e-3,
                "v={v} back={back}"
            );
        }
    }

    #[test]
    fn test_f16_zero_and_neg_zero() {
        assert_eq!(f32_to_f16(0.0), 0x0000);
        assert_eq!(f32_to_f16(-0.0), 0x8000);
    }

    /// SPIKE FINDING (issue #2231 Stage 1): the `half` crate is NOT bit-identical
    /// to the legacy trueno `f32_to_f16` for two classes of inputs:
    ///   1. rounding boundaries where the mantissa rounds half-to-even (`half`,
    ///      IEEE-correct) vs trueno's round-half-up — a ±1-ulp difference; and
    ///   2. mantissa-overflow carry into the exponent — trueno omits the carry
    ///      and emits the WRONG exponent (e.g. `255.99 → 0xD800` instead of the
    ///      correct `0xDC00`), a large divergence.
    ///
    /// `half` is the IEEE-754-correct path, so the leaf intentionally adopts it.
    /// This test PINS the known-divergence count so a future change to either
    /// side is caught, while asserting exact agreement on the exactly-representable
    /// grid (integers / clean fractions) and on inf/zero specials.
    #[test]
    fn test_f16_parity_with_trueno_ref_known_divergence() {
        // (1) Exact agreement on exactly-representable f16 values and specials.
        for &v in &[
            0.0_f32, -0.0, 1.0, -1.0, 0.5, -0.5, 2.0, 256.0, -256.0, 65504.0, -65504.0,
        ] {
            assert_eq!(
                f32_to_f16(v),
                trueno_ref_f32_to_f16(v),
                "exact-representable v={v}"
            );
        }
        for &v in &[f32::INFINITY, f32::NEG_INFINITY] {
            assert_eq!(f32_to_f16(v), trueno_ref_f32_to_f16(v), "special v={v}");
        }

        // (2) Across a dense sweep the two paths diverge on rounding boundaries
        // and mantissa-carry cases. `half` is IEEE-correct; pin the count so the
        // divergence stays understood (and motivates the Stage-2 byte-identity
        // oracle scoping f16-written v2 tensors).
        let mut v = -300.0_f32;
        let mut diffs = 0usize;
        while v < 300.0 {
            if !v.is_nan() && f32_to_f16(v) != trueno_ref_f32_to_f16(v) {
                diffs += 1;
            }
            v += 0.013;
        }
        assert_eq!(
            diffs, 31,
            "f16 half-vs-trueno divergence count drifted (was 31)"
        );
    }
}
