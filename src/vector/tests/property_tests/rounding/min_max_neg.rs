use super::super::super::super::*;
use proptest::prelude::*;

// ========================================
// Property Tests: minimum()
// ========================================

// Property test: minimum correctness - matches f32::min
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_minimum_correctness(
        ab in prop::collection::vec((-100.0f32..100.0, -100.0f32..100.0), 1..100)
    ) {
        let a: Vec<f32> = ab.iter().map(|(x, _)| *x).collect();
        let b: Vec<f32> = ab.iter().map(|(_, y)| *y).collect();

        let va = Vector::from_slice(&a);
        let vb = Vector::from_slice(&b);
        let result = va.minimum(&vb).unwrap();

        for (i, (&x, (&y, &output))) in a.iter()
            .zip(b.iter().zip(result.as_slice().iter()))
            .enumerate() {
            let expected = x.min(y);
            prop_assert!(
                (output - expected).abs() < 1e-5 || (output.is_nan() && expected.is_nan()),
                "minimum failed at {}: minimum({}, {}) = {} != {}",
                i, x, y, output, expected
            );
        }
    }
}

// Property test: commutativity - minimum(a, b) = minimum(b, a)
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_minimum_commutative(
        ab in prop::collection::vec((-100.0f32..100.0, -100.0f32..100.0), 1..100)
    ) {
        let a: Vec<f32> = ab.iter().map(|(x, _)| *x).collect();
        let b: Vec<f32> = ab.iter().map(|(_, y)| *y).collect();

        let va = Vector::from_slice(&a);
        let vb = Vector::from_slice(&b);
        let result1 = va.minimum(&vb).unwrap();
        let result2 = vb.minimum(&va).unwrap();

        for (i, (&r1, &r2)) in result1.as_slice().iter()
            .zip(result2.as_slice().iter())
            .enumerate() {
            prop_assert!(
                (r1 - r2).abs() < 1e-5 || (r1.is_nan() && r2.is_nan()),
                "commutativity failed at {}: minimum({}, {}) = {} != minimum({}, {}) = {}",
                i, a[i], b[i], r1, b[i], a[i], r2
            );
        }
    }
}

// Property test: idempotence - minimum(a, a) = a
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_minimum_idempotent(
        a in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.minimum(&va).unwrap();

        for (i, (&input, &output)) in a.iter()
            .zip(result.as_slice().iter())
            .enumerate() {
            prop_assert!(
                (output - input).abs() < 1e-5 || (output.is_nan() && input.is_nan()),
                "idempotence failed at {}: minimum({}, {}) = {} != {}",
                i, input, input, output, input
            );
        }
    }
}

// ========================================
// Property Tests: maximum()
// ========================================

// Property test: maximum correctness - matches f32::max
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_maximum_correctness(
        ab in prop::collection::vec((-100.0f32..100.0, -100.0f32..100.0), 1..100)
    ) {
        let a: Vec<f32> = ab.iter().map(|(x, _)| *x).collect();
        let b: Vec<f32> = ab.iter().map(|(_, y)| *y).collect();

        let va = Vector::from_slice(&a);
        let vb = Vector::from_slice(&b);
        let result = va.maximum(&vb).unwrap();

        for (i, (&x, (&y, &output))) in a.iter()
            .zip(b.iter().zip(result.as_slice().iter()))
            .enumerate() {
            let expected = x.max(y);
            prop_assert!(
                (output - expected).abs() < 1e-5 || (output.is_nan() && expected.is_nan()),
                "maximum failed at {}: maximum({}, {}) = {} != {}",
                i, x, y, output, expected
            );
        }
    }
}

// Property test: commutativity - maximum(a, b) = maximum(b, a)
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_maximum_commutative(
        ab in prop::collection::vec((-100.0f32..100.0, -100.0f32..100.0), 1..100)
    ) {
        let a: Vec<f32> = ab.iter().map(|(x, _)| *x).collect();
        let b: Vec<f32> = ab.iter().map(|(_, y)| *y).collect();

        let va = Vector::from_slice(&a);
        let vb = Vector::from_slice(&b);
        let result1 = va.maximum(&vb).unwrap();
        let result2 = vb.maximum(&va).unwrap();

        for (i, (&r1, &r2)) in result1.as_slice().iter()
            .zip(result2.as_slice().iter())
            .enumerate() {
            prop_assert!(
                (r1 - r2).abs() < 1e-5 || (r1.is_nan() && r2.is_nan()),
                "commutativity failed at {}: maximum({}, {}) = {} != maximum({}, {}) = {}",
                i, a[i], b[i], r1, b[i], a[i], r2
            );
        }
    }
}

// Property test: idempotence - maximum(a, a) = a
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_maximum_idempotent(
        a in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.maximum(&va).unwrap();

        for (i, (&input, &output)) in a.iter()
            .zip(result.as_slice().iter())
            .enumerate() {
            prop_assert!(
                (output - input).abs() < 1e-5 || (output.is_nan() && input.is_nan()),
                "idempotence failed at {}: maximum({}, {}) = {} != {}",
                i, input, input, output, input
            );
        }
    }
}

// ========================================
// Property Tests: neg()
// ========================================

// Property test: double negation is identity - -(-x) = x
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_neg_double_negation_property(
        a in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let neg_once = va.neg().unwrap();
        let neg_twice = neg_once.neg().unwrap();

        for (i, (&input, &output)) in a.iter()
            .zip(neg_twice.as_slice().iter())
            .enumerate() {
            prop_assert!(
                (output - input).abs() < 1e-5 || (output.is_nan() && input.is_nan()),
                "double negation failed at {}: -(-{}) = {} != {}",
                i, input, output, input
            );
        }
    }
}

// Property test: negation sign flip - sign(neg(x)) = -sign(x)
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_neg_sign_flip(
        a in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let neg_result = va.neg().unwrap();

        for (i, (&input, &output)) in a.iter()
            .zip(neg_result.as_slice().iter())
            .enumerate() {
            // Skip zero and NaN
            if input.abs() > 1e-10 && !input.is_nan() {
                prop_assert!(
                    (input.signum() + output.signum()).abs() < 1e-5,
                    "sign flip failed at {}: sign({}) + sign(-{}) = {} + {} != 0",
                    i, input, input, input.signum(), output.signum()
                );
            }
        }
    }
}

// Property test: negation preserves magnitude - abs(neg(x)) = abs(x)
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_neg_magnitude_preservation(
        a in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let neg_result = va.neg().unwrap();
        let abs_a = va.abs().unwrap();
        let abs_neg_a = neg_result.abs().unwrap();

        for (i, (&expected, &output)) in abs_a.as_slice().iter()
            .zip(abs_neg_a.as_slice().iter())
            .enumerate() {
            prop_assert!(
                (output - expected).abs() < 1e-5,
                "magnitude not preserved at {}: abs(-{}) = {} != abs({}) = {}",
                i, a[i], output, a[i], expected
            );
        }
    }
}
