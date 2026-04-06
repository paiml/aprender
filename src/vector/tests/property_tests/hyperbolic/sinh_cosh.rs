use super::super::super::super::*;
use proptest::prelude::*;

// Property test: sinh correctness
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_sinh_correctness(
        a in prop::collection::vec(-10.0f32..10.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.sinh().unwrap();

        for (i, (&input, &output)) in a.iter()
            .zip(result.as_slice().iter())
            .enumerate() {
            let expected = input.sinh();
            prop_assert!(
                (output - expected).abs() < 1e-5,
                "sinh failed at {}: {} != {}",
                i, output, expected
            );
        }
    }
}

// Property test: sinh is odd function - sinh(-x) = -sinh(x)
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_sinh_odd_function(
        a in prop::collection::vec(-10.0f32..10.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let sinh_pos = va.sinh().unwrap();

        let a_neg: Vec<f32> = a.iter().map(|x| -x).collect();
        let va_neg = Vector::from_slice(&a_neg);
        let sinh_neg = va_neg.sinh().unwrap();

        for (i, (&pos, &neg)) in sinh_pos.as_slice().iter()
            .zip(sinh_neg.as_slice().iter())
            .enumerate() {
            prop_assert!(
                (pos + neg).abs() < 1e-4,
                "sinh odd function failed at {}: sinh(-x)={} != -sinh(x)={}",
                i, neg, -pos
            );
        }
    }
}

// Property test: sinh definition - sinh(x) = (e^x - e^(-x)) / 2
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_sinh_definition(
        a in prop::collection::vec(-5.0f32..5.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.sinh().unwrap();

        for (i, (&input, &output)) in a.iter()
            .zip(result.as_slice().iter())
            .enumerate() {
            let expected = (input.exp() - (-input).exp()) / 2.0;
            let tolerance = if expected.abs() > 1.0 {
                expected.abs() * 1e-5
            } else {
                1e-5
            };
            prop_assert!(
                (output - expected).abs() < tolerance,
                "sinh definition failed at {}: {} != {} (input: {})",
                i, output, expected, input
            );
        }
    }
}

// Property test: cosh correctness
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_cosh_correctness(
        a in prop::collection::vec(-10.0f32..10.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.cosh().unwrap();

        for (i, (&input, &output)) in a.iter()
            .zip(result.as_slice().iter())
            .enumerate() {
            let expected = input.cosh();
            prop_assert!(
                (output - expected).abs() < 1e-5,
                "cosh failed at {}: {} != {}",
                i, output, expected
            );
        }
    }
}

// Property test: cosh is even function - cosh(-x) = cosh(x)
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_cosh_even_function(
        a in prop::collection::vec(-10.0f32..10.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let cosh_pos = va.cosh().unwrap();

        let a_neg: Vec<f32> = a.iter().map(|x| -x).collect();
        let va_neg = Vector::from_slice(&a_neg);
        let cosh_neg = va_neg.cosh().unwrap();

        for (i, (&pos, &neg)) in cosh_pos.as_slice().iter()
            .zip(cosh_neg.as_slice().iter())
            .enumerate() {
            prop_assert!(
                (pos - neg).abs() < 1e-4,
                "cosh even function failed at {}: cosh(-x)={} != cosh(x)={}",
                i, neg, pos
            );
        }
    }
}

// Property test: cosh definition - cosh(x) = (e^x + e^(-x)) / 2
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_cosh_definition(
        a in prop::collection::vec(-5.0f32..5.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.cosh().unwrap();

        for (i, (&input, &output)) in a.iter()
            .zip(result.as_slice().iter())
            .enumerate() {
            let expected = f32::midpoint(input.exp(), (-input).exp());
            let tolerance = if expected.abs() > 1.0 {
                expected.abs() * 1e-5
            } else {
                1e-5
            };
            prop_assert!(
                (output - expected).abs() < tolerance,
                "cosh definition failed at {}: {} != {} (input: {})",
                i, output, expected, input
            );
        }
    }
}

// Property test: hyperbolic identity - cosh^2(x) - sinh^2(x) = 1
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_cosh_sinh_identity(
        a in prop::collection::vec(-5.0f32..5.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let cosh_result = va.cosh().unwrap();
        let sinh_result = va.sinh().unwrap();

        for (i, (&c, &s)) in cosh_result.as_slice().iter()
            .zip(sinh_result.as_slice().iter())
            .enumerate() {
            let identity = c * c - s * s;
            let max_squared = c.abs().max(s.abs()).powi(2);
            let tolerance = if max_squared > 1.0 {
                max_squared * 1e-4
            } else {
                1e-5
            };
            prop_assert!(
                (identity - 1.0).abs() < tolerance,
                "Hyperbolic identity failed at {}: cosh^2({}) - sinh^2({}) = {} != 1",
                i, c, s, identity
            );
        }
    }
}
