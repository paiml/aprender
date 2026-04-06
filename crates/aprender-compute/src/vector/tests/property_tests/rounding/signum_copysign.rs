use super::super::super::super::*;
use proptest::prelude::*;

// Property test: signum correctness
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_signum_correctness(
        a in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.signum().unwrap();

        for (i, (&input, &output)) in a.iter()
            .zip(result.as_slice().iter())
            .enumerate() {
            let expected = input.signum();
            prop_assert!(
                (output - expected).abs() < 1e-5,
                "signum failed at {}: {} != {}",
                i, output, expected
            );
        }
    }
}

// Property test: signum range - always -1, 0, or 1
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_signum_range(
        a in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.signum().unwrap();

        for (i, &output) in result.as_slice().iter().enumerate() {
            prop_assert!(
                output == 1.0 || output == -1.0 || output.is_nan(),
                "signum should be 1.0, -1.0, or NaN at {}: signum({}) = {}",
                i, a[i], output
            );
        }
    }
}

// Property test: signum * abs = identity (for non-zero)
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_signum_abs_identity(
        a in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let signum_result = va.signum().unwrap();
        let abs_result = va.abs().unwrap();

        for (i, (&input, (&sign, &magnitude))) in a.iter()
            .zip(signum_result.as_slice().iter().zip(abs_result.as_slice().iter()))
            .enumerate() {
            // Skip zero values as they have special behavior
            if input.abs() > 1e-10 {
                let reconstructed = sign * magnitude;
                prop_assert!(
                    (reconstructed - input).abs() < 1e-5,
                    "signum*abs identity failed at {}: {} != signum({}) * abs({}) = {} * {} = {}",
                    i, input, input, input, sign, magnitude, reconstructed
                );
            }
        }
    }
}

// ========================================
// Property Tests: copysign()
// ========================================

// Property test: copysign correctness - matches f32::copysign
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_copysign_correctness(
        ab in prop::collection::vec((-100.0f32..100.0, -100.0f32..100.0), 1..100)
    ) {
        let a: Vec<f32> = ab.iter().map(|(x, _)| *x).collect();
        let b: Vec<f32> = ab.iter().map(|(_, y)| *y).collect();

        let va = Vector::from_slice(&a);
        let vb = Vector::from_slice(&b);
        let result = va.copysign(&vb).unwrap();

        for (i, (&mag, (&sgn, &output))) in a.iter()
            .zip(b.iter().zip(result.as_slice().iter()))
            .enumerate() {
            let expected = mag.copysign(sgn);
            prop_assert!(
                (output - expected).abs() < 1e-5 || (output.is_nan() && expected.is_nan()),
                "copysign failed at {}: copysign({}, {}) = {} != {}",
                i, mag, sgn, output, expected
            );
        }
    }
}

// Property test: magnitude preservation - abs(copysign(a, b)) = abs(a)
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_copysign_magnitude_preservation(
        ab in prop::collection::vec((-100.0f32..100.0, -100.0f32..100.0), 1..100)
    ) {
        let a: Vec<f32> = ab.iter().map(|(x, _)| *x).collect();
        let b: Vec<f32> = ab.iter().map(|(_, y)| *y).collect();

        let va = Vector::from_slice(&a);
        let vb = Vector::from_slice(&b);
        let result = va.copysign(&vb).unwrap();
        let abs_a = va.abs().unwrap();
        let abs_result = result.abs().unwrap();

        for (i, (&expected, &output)) in abs_a.as_slice().iter()
            .zip(abs_result.as_slice().iter())
            .enumerate() {
            prop_assert!(
                (output - expected).abs() < 1e-5,
                "magnitude not preserved at {}: abs(copysign({}, {})) = {} != abs({}) = {}",
                i, a[i], b[i], output, a[i], expected
            );
        }
    }
}

// Property test: sign copy - sign(copysign(a, b)) = sign(b) for non-zero b
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_copysign_sign_copy(
        ab in prop::collection::vec((-100.0f32..100.0, -100.0f32..100.0), 1..100)
    ) {
        let a: Vec<f32> = ab.iter().map(|(x, _)| *x).collect();
        let b: Vec<f32> = ab.iter().map(|(_, y)| *y).collect();

        let va = Vector::from_slice(&a);
        let vb = Vector::from_slice(&b);
        let result = va.copysign(&vb).unwrap();
        let signum_b = vb.signum().unwrap();
        let signum_result = result.signum().unwrap();

        for (i, (&sign_b, &sign_result)) in signum_b.as_slice().iter()
            .zip(signum_result.as_slice().iter())
            .enumerate() {
            // Skip NaN cases
            if !sign_b.is_nan() && !sign_result.is_nan() {
                prop_assert!(
                    (sign_result - sign_b).abs() < 1e-5,
                    "sign not copied at {}: sign(copysign({}, {})) = {} != sign({}) = {}",
                    i, a[i], b[i], sign_result, b[i], sign_b
                );
            }
        }
    }
}
