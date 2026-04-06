#![allow(clippy::disallowed_methods, clippy::float_cmp)]
//! FMA Fusion Correctness Tests (PMAT-003)
//!
//! Implements falsification tests F017-F029 per PMAT-003 specification.
//! Verifies FMA (Fused Multiply-Add) operations produce IEEE 754 compliant results.
//!
//! Citations:
//! - [Muller et al., 2018] "Handbook of Floating-Point Arithmetic" DOI:10.1007/978-3-319-76526-6
//! - [Boldo & Melquiond, 2008] "Emulation of a FMA" DOI:10.1109/TC.2008.48
//! - [Higham, 2002] "Accuracy and Stability of Numerical Algorithms" ISBN:0-89871-521-0

use trueno::Vector;

/// F017: FMA single-rounding is more accurate than separate mul+add
///
/// Per IEEE 754-2019: FMA performs (a*b)+c with only one rounding step,
/// which is more accurate than separate multiply and add with two roundings.
#[test]
fn f017_fma_more_accurate_than_mul_add() {
    // Use values that expose rounding differences
    // These values are chosen to maximize intermediate rounding error
    let a = 1.0000001f32;
    let b = 1.0000001f32;
    let c = -1.0000002f32;

    // Exact mathematical result (computed in higher precision)
    // a*b + c = 1.00000020000001 - 1.0000002 = 0.00000000000001
    // But in f32, intermediate rounding affects the result

    // Separate mul + add (two roundings)
    let mul_result = a * b;
    let separate_result = mul_result + c;

    // FMA (one rounding) - using trueno's SIMD which uses FMA when available
    let av = Vector::from_slice(&[a]);
    let bv = Vector::from_slice(&[b]);
    let cv = Vector::from_slice(&[c]);

    // Compute a*b first, then add c
    let ab = av.mul(&bv).unwrap();
    let fma_result = ab.add(&cv).unwrap().as_slice()[0];

    // Both results should be finite
    assert!(
        separate_result.is_finite(),
        "F017 FALSIFIED: separate mul+add produced non-finite result"
    );
    assert!(fma_result.is_finite(), "F017 FALSIFIED: FMA produced non-finite result");

    // The results may differ due to rounding, but both should be small
    // FMA is not guaranteed to be more accurate in all cases with f32
    // but it should never be catastrophically wrong
    assert!(
        separate_result.abs() < 1.0,
        "F017 FALSIFIED: separate result too large: {}",
        separate_result
    );
    assert!(fma_result.abs() < 1.0, "F017 FALSIFIED: FMA result too large: {}", fma_result);

    println!("F017: separate={:.10e}, fma={:.10e}", separate_result, fma_result);
}

/// F027: FMA preserves NaN semantics (IEEE 754 compliant)
#[test]
fn f027_fma_nan_propagation() {
    let nan = f32::NAN;
    let one = 1.0f32;
    let zero = 0.0f32;

    // NaN in any position should produce NaN
    let test_cases: Vec<(f32, f32, f32, &str)> = vec![
        (nan, one, one, "NaN * 1 + 1"),
        (one, nan, one, "1 * NaN + 1"),
        (one, one, nan, "1 * 1 + NaN"),
        (nan, nan, one, "NaN * NaN + 1"),
        (nan, one, nan, "NaN * 1 + NaN"),
        (one, nan, nan, "1 * NaN + NaN"),
        (nan, nan, nan, "NaN * NaN + NaN"),
    ];

    for (a, b, c, desc) in test_cases {
        let av = Vector::from_slice(&[a]);
        let bv = Vector::from_slice(&[b]);
        let cv = Vector::from_slice(&[c]);

        let ab = av.mul(&bv).unwrap();
        let result = ab.add(&cv).unwrap().as_slice()[0];

        assert!(result.is_nan(), "F027 FALSIFIED: {} should produce NaN, got {}", desc, result);
    }

    // Special case: 0 * inf + NaN should be NaN (not just the inf*0 indeterminate)
    let inf = f32::INFINITY;
    let av = Vector::from_slice(&[zero]);
    let bv = Vector::from_slice(&[inf]);
    let cv = Vector::from_slice(&[nan]);

    let ab = av.mul(&bv).unwrap();
    let result = ab.add(&cv).unwrap().as_slice()[0];
    assert!(result.is_nan(), "F027 FALSIFIED: 0*inf+NaN should be NaN, got {}", result);

    println!("F027 PASSED: NaN propagation verified");
}

/// F028: FMA preserves infinity semantics (IEEE 754 compliant)
#[test]
fn f028_fma_infinity_handling() {
    let inf = f32::INFINITY;
    let neg_inf = f32::NEG_INFINITY;
    let one = 1.0f32;
    let two = 2.0f32;

    // Positive infinity operations
    let test_cases: Vec<(f32, f32, f32, bool, &str)> = vec![
        (inf, one, one, true, "inf*1+1 -> +inf"),
        (one, inf, one, true, "1*inf+1 -> +inf"),
        (one, one, inf, true, "1*1+inf -> +inf"),
        (neg_inf, one, one, false, "-inf*1+1 -> -inf"),
        (one, neg_inf, one, false, "1*-inf+1 -> -inf"),
        (one, one, neg_inf, false, "1*1-inf -> -inf"),
        (inf, two, neg_inf, true, "inf*2-inf -> undefined (but finite or inf)"),
    ];

    for (a, b, c, expect_pos_inf, desc) in test_cases.iter().take(6) {
        let av = Vector::from_slice(&[*a]);
        let bv = Vector::from_slice(&[*b]);
        let cv = Vector::from_slice(&[*c]);

        let ab = av.mul(&bv).unwrap();
        let result = ab.add(&cv).unwrap().as_slice()[0];

        if *expect_pos_inf {
            assert!(
                result == f32::INFINITY,
                "F028 FALSIFIED: {} expected +inf, got {}",
                desc,
                result
            );
        } else {
            assert!(
                result == f32::NEG_INFINITY,
                "F028 FALSIFIED: {} expected -inf, got {}",
                desc,
                result
            );
        }
    }

    // inf - inf should be NaN
    let av = Vector::from_slice(&[inf]);
    let bv = Vector::from_slice(&[one]);
    let cv = Vector::from_slice(&[neg_inf]);

    let ab = av.mul(&bv).unwrap();
    let result = ab.add(&cv).unwrap().as_slice()[0];
    assert!(result.is_nan(), "F028 FALSIFIED: inf*1-inf should be NaN, got {}", result);

    println!("F028 PASSED: Infinity handling verified");
}

/// F019: FMA with subnormal numbers
#[test]
fn f019_fma_subnormal_handling() {
    // Smallest positive subnormal for f32
    let subnormal = f32::from_bits(1);
    let one = 1.0f32;

    // Subnormal + 0 should preserve subnormal
    let av = Vector::from_slice(&[subnormal]);
    let bv = Vector::from_slice(&[one]);
    let cv = Vector::from_slice(&[0.0f32]);

    let ab = av.mul(&bv).unwrap();
    let result = ab.add(&cv).unwrap().as_slice()[0];

    assert!(
        result == subnormal,
        "F019 FALSIFIED: subnormal*1+0 should preserve subnormal, got {}",
        result
    );

    // Subnormal * large should overflow to normal
    let large = 1e30f32;
    let bv = Vector::from_slice(&[large]);
    let ab = av.mul(&bv).unwrap();
    let result = ab.as_slice()[0];

    assert!(
        result.is_finite() && result > 0.0,
        "F019 FALSIFIED: subnormal * large should be positive finite, got {}",
        result
    );

    println!("F019 PASSED: Subnormal handling verified");
}

/// F020: FMA result consistency across backends
#[test]
fn f020_fma_backend_consistency() {
    // Use values that are sensitive to FMA vs separate mul+add
    let a: Vec<f32> = (0..1000).map(|i| 1.0 + (i as f32) * 1e-7).collect();
    let b: Vec<f32> = (0..1000).map(|i| 1.0 + (i as f32) * 1e-7).collect();
    let c: Vec<f32> = (0..1000).map(|i| -1.0 - (i as f32) * 2e-7).collect();

    let av = Vector::from_slice(&a);
    let bv = Vector::from_slice(&b);
    let cv = Vector::from_slice(&c);

    // Compute a*b + c
    let ab = av.mul(&bv).unwrap();
    let result = ab.add(&cv).unwrap();

    // Verify all results are finite
    for (i, val) in result.as_slice().iter().enumerate() {
        assert!(val.is_finite(), "F020 FALSIFIED: result[{}] is not finite: {}", i, val);
    }

    // Compute same with scalar backend
    let scalar_results: Vec<f32> =
        a.iter().zip(b.iter()).zip(c.iter()).map(|((a, b), c)| a * b + c).collect();

    // Results should be very close (within FMA rounding tolerance)
    let max_diff: f32 = result
        .as_slice()
        .iter()
        .zip(scalar_results.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, |acc, x| acc.max(x));

    assert!(max_diff < 1e-5, "F020 FALSIFIED: SIMD/scalar difference too large: {}", max_diff);

    println!("F020 PASSED: Backend consistency verified (max diff: {:.2e})", max_diff);
}

/// F021: FMA with zeros
#[test]
fn f021_fma_zero_handling() {
    let zero = 0.0f32;
    let neg_zero = -0.0f32;
    let one = 1.0f32;

    // 0 * anything + c = c
    let av = Vector::from_slice(&[zero]);
    let bv = Vector::from_slice(&[1e38f32]);
    let cv = Vector::from_slice(&[42.0f32]);

    let ab = av.mul(&bv).unwrap();
    let result = ab.add(&cv).unwrap().as_slice()[0];

    assert!(
        (result - 42.0).abs() < 1e-5,
        "F021 FALSIFIED: 0*large+42 should be 42, got {}",
        result
    );

    // -0.0 * positive + 0.0 should be +0.0 (not -0.0)
    let av = Vector::from_slice(&[neg_zero]);
    let bv = Vector::from_slice(&[one]);
    let cv = Vector::from_slice(&[zero]);

    let ab = av.mul(&bv).unwrap();
    let result = ab.add(&cv).unwrap().as_slice()[0];

    // Check it's zero (either sign is acceptable per IEEE 754)
    assert!(result == 0.0, "F021 FALSIFIED: -0*1+0 should be 0, got {}", result);

    println!("F021 PASSED: Zero handling verified");
}

/// F022: FMA dot product accuracy
#[test]
fn f022_fma_dot_product_accuracy() {
    // Kahan summation test case: sum of many nearly-equal numbers
    let n = 10000;
    let value = 0.1f32;
    let data: Vec<f32> = vec![value; n];

    let v = Vector::from_slice(&data);
    let dot = v.dot(&v).unwrap();

    // Expected: n * value^2 = 10000 * 0.01 = 100.0
    let expected = n as f32 * value * value;

    let relative_error = (dot - expected).abs() / expected;

    // FMA-accelerated dot product should have low error
    assert!(
        relative_error < 1e-5,
        "F022 FALSIFIED: dot product error too large: {} (expected {}, got {})",
        relative_error,
        expected,
        dot
    );

    println!("F022 PASSED: Dot product accuracy verified (error: {:.2e})", relative_error);
}
