// PMAT-905 (CPU f16-RNE sweep): falsifiers for the converter module's canonical
// `f32_to_f16` encoder (convert_report.rs). This is the encoder behind
// `apr convert --quantize fp16` (via `quantize_fp16`) and the SafeTensors FP16
// export byte path (via `f32_to_f16_bits` → `f32_slice_to_f16_le_bytes`).
//
// Oracle: `half::f16::from_f32`. RED on the old round-toward-zero impl (mantissa
// `>> 13` truncation + subnormal round-half-up + NaN-payload collapse), GREEN on
// the IEEE round-to-nearest-even fix. Mutation-verified by reverting the round-up.
//
// `half` is a dev-dependency of aprender-core (and an optional regular dep); the
// fully-qualified `::half` path resolves the extern crate unambiguously.

/// Named divergence cases where round-toward-zero and IEEE-RNE genuinely disagree.
/// The asserted bits are `half::f16::from_f32`'s output (the IEEE reference).
#[test]
fn falsify_convert_f32_to_f16_known_rne_divergences() {
    // 255.99 → mantissa carries up: old truncation = 0x5BFF, RNE = 0x5C00.
    assert_eq!(
        f32_to_f16(255.99),
        ::half::f16::from_f32(255.99).to_bits(),
        "255.99 must round-half-to-even up to 0x5C00, not truncate to 0x5BFF"
    );
    assert_eq!(f32_to_f16(255.99), 0x5C00);

    // 65520.0 → overflow tie: old truncation = 0x7BFF (max finite), RNE = +Inf 0x7C00.
    assert_eq!(
        f32_to_f16(65520.0),
        ::half::f16::from_f32(65520.0).to_bits(),
        "65520.0 sits on the overflow tie and must round to +Inf 0x7C00"
    );
    assert_eq!(f32_to_f16(65520.0), 0x7C00);
}

/// STRENGTHENED ties-to-even falsifier (review HOLD: the prior tie test passed on
/// BOTH the buggy and fixed impls because its tie values had a zero discarded
/// mantissa, so truncation and RNE agreed). These inputs have the discarded low
/// 13 bits == exactly 0x1000 (an exact half-way tie) with the kept LSB == 1, so
/// IEEE round-to-EVEN rounds UP while round-toward-zero (truncate) rounds DOWN —
/// the two encoders DISAGREE. RED on the old truncating impl.
#[test]
fn falsify_convert_f32_to_f16_ties_to_even_nonzero_discard() {
    // Each value: f32 bits with (mantissa & 0x1FFF) == 0x1000 (exact tie) and
    // ((mantissa >> 13) & 1) == 1 (kept LSB odd ⇒ round-to-even goes UP).
    // old(truncate) yields the LOWER bits; RNE(half) yields LOWER+1.
    let cases: &[(u32, f32, u16)] = &[
        // 0.12689209 → old 0x300F, RNE/half 0x3010
        (0x3E01_F000, 0.126_892_09_f32, 0x3010),
        // 8332.0 → old 0x7011, RNE/half 0x7012
        (0x4602_3000, 8332.0_f32, 0x7012),
        // -0.13079834 → old 0xB02F, RNE/half 0xB030
        (0xBE05_F000, -0.130_798_34_f32, 0xB030),
    ];
    for &(bits, val, want) in cases {
        // Sanity: the literal matches the bit pattern we constructed.
        assert_eq!(f32::from_bits(bits).to_bits(), val.to_bits());
        let got = f32_to_f16(val);
        let oracle = ::half::f16::from_f32(val).to_bits();
        assert_eq!(
            got, oracle,
            "tie {val} (bits={bits:#010x}) must round-to-even to {want:#06x}; got {got:#06x}"
        );
        assert_eq!(got, want);
        // The kept LSB of the truncated mantissa is odd, so RNE rounds UP: the
        // result must be exactly ONE ULP above the truncated value. This is what
        // FALSIFIES the old `mantissa >> 13` round-toward-zero encoder.
        let truncated = want - 1;
        assert_ne!(
            got, truncated,
            "round-toward-zero would have produced {truncated:#06x} — the bug"
        );
    }
}

/// Subnormal-region ties-to-even, where the old round-half-up subnormal path and
/// the f32-subnormal flush-to-zero diverge from IEEE RNE. RED on the old impl.
#[test]
fn falsify_convert_f32_to_f16_subnormal_rne() {
    // 5.1528215e-5 (bits 0x38582000): exact tie in the f16 subnormal field. Old
    // round-half-UP → 0x0361; IEEE round-to-EVEN → 0x0360 (mantissa even, down).
    let v = f32::from_bits(0x3858_2000);
    assert_eq!(
        f32_to_f16(v),
        ::half::f16::from_f32(v).to_bits(),
        "subnormal tie must round to even (0x0360), not half-up (0x0361)"
    );
    assert_eq!(f32_to_f16(v), 0x0360);

    // 6.100591e-5 (bits 0x387FE099): rounds UP across the subnormal→normal
    // boundary to the smallest NORMAL f16 (0x0400). The old code flushed this
    // tiny f32 to 0x0000.
    let v = f32::from_bits(0x387F_E099);
    assert_eq!(f32_to_f16(v), ::half::f16::from_f32(v).to_bits());
    assert_eq!(f32_to_f16(v), 0x0400);
}

/// NaN payloads must be preserved the way `half` preserves them (top mantissa
/// bits + quiet bit), not collapsed to a single canonical NaN.
#[test]
fn falsify_convert_f32_to_f16_nan_payload_matches_half() {
    // A signalling-ish NaN with a distinctive low payload.
    let v = f32::from_bits(0x7FC1_2345);
    let got = f32_to_f16(v);
    let want = ::half::f16::from_f32(v).to_bits();
    assert_eq!(got, want, "NaN encoding must match half exactly");
    assert!(::half::f16::from_bits(got).is_nan());
}

/// Exhaustive-by-stride bit-identity to `half::f16::from_f32` across the full
/// 2^32 f32 domain. NaN payloads are treated as equal (any-NaN == any-NaN).
/// RED on the old truncating impl (~251.6M divergences), GREEN on the fix.
#[test]
fn falsify_convert_f32_to_f16_bit_identical_to_half_on_grid() {
    // Odd stride so we sweep every exponent/mantissa region; keeps the test fast.
    let step = 0x2Bu32;
    let mut u: u32 = 0;
    loop {
        let v = f32::from_bits(u);
        let got = f32_to_f16(v);
        let want = ::half::f16::from_f32(v).to_bits();
        if got != want {
            let both_nan =
                ::half::f16::from_bits(got).is_nan() && ::half::f16::from_bits(want).is_nan();
            assert!(
                both_nan,
                "convert f32_to_f16 diverges from half at bits={u:#010x} (v={v:e}): \
                 got={got:#06x} want={want:#06x}"
            );
        }
        let (next, overflow) = u.overflowing_add(step);
        if overflow {
            break;
        }
        u = next;
    }
}

/// Edge values: ±0, ±Inf, the f16 max finite (65504), the overflow boundary, and
/// the smallest subnormal — each must match `half` bit-for-bit.
#[test]
fn falsify_convert_f32_to_f16_edge_values_match_half() {
    for v in [
        0.0_f32,
        -0.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        65504.0,  // f16 MAX finite → 0x7BFF
        65505.0,  // just over → +Inf
        100_000.0, // way over → +Inf
        6.103_515_6e-5, // smallest f16 normal (2^-14)
        5.96e-8,  // smallest f16 subnormal region
    ] {
        assert_eq!(
            f32_to_f16(v),
            ::half::f16::from_f32(v).to_bits(),
            "edge value {v:e} must match half"
        );
    }
}
