use super::super::super::super::*;
use proptest::prelude::*;

// Property test: pow() correctness vs f32::powf()
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_pow_correctness(
        a in prop::collection::vec(0.1f32..100.0, 1..100),
        n in -3.0f32..3.0
    ) {
        let va = Vector::from_slice(&a);
        let result = va.pow(n).unwrap();

        for (i, (&a_val, &pow_val)) in a.iter()
            .zip(result.as_slice().iter())
            .enumerate() {
            let expected = a_val.powf(n);

            let tolerance = if expected.abs() > 1.0 {
                expected.abs() * 1e-4
            } else {
                1e-4
            };

            prop_assert!(
                (pow_val - expected).abs() < tolerance,
                "pow correctness failed at {}: {} != {}, diff = {}",
                i, pow_val, expected, (pow_val - expected).abs()
            );
        }
    }
}

// Property test: Power laws
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_pow_power_laws(
        a in prop::collection::vec(1.0f32..10.0, 1..50),
        n in 1.0f32..3.0,
        m in 1.0f32..3.0
    ) {
        // Test: (x^n)^m = x^(n*m)
        let va = Vector::from_slice(&a);
        let pow_n = va.pow(n).unwrap();
        let pow_n_then_m = pow_n.pow(m).unwrap();
        let pow_nm = va.pow(n * m).unwrap();

        for (i, (&expected, &actual)) in pow_nm.as_slice().iter()
            .zip(pow_n_then_m.as_slice().iter())
            .enumerate() {
            let tolerance = if expected.abs() > 1.0 {
                expected.abs() * 1e-3
            } else {
                1e-3
            };

            prop_assert!(
                (expected - actual).abs() < tolerance,
                "pow power law failed at {}: {} != {}, diff = {}",
                i, expected, actual, (expected - actual).abs()
            );
        }
    }
}

// Property test: pow() special cases
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_pow_special_cases(
        a in prop::collection::vec(0.1f32..100.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);

        // x^0 = 1
        let pow_zero = va.pow(0.0).unwrap();
        for &val in pow_zero.as_slice() {
            prop_assert!((val - 1.0).abs() < 1e-5, "x^0 should be 1");
        }

        // x^1 = x
        let pow_one = va.pow(1.0).unwrap();
        for (i, (&original, &pow_val)) in a.iter()
            .zip(pow_one.as_slice().iter())
            .enumerate() {
            prop_assert!(
                (original - pow_val).abs() < 1e-5,
                "x^1 failed at {}: {} != {}",
                i, original, pow_val
            );
        }

        // x^0.5 should equal sqrt(x)
        let pow_half = va.pow(0.5).unwrap();
        let sqrt_result = va.sqrt().unwrap();
        for (i, (&pow_val, &sqrt_val)) in pow_half.as_slice().iter()
            .zip(sqrt_result.as_slice().iter())
            .enumerate() {
            let tolerance = if sqrt_val.abs() > 1.0 {
                sqrt_val.abs() * 1e-5
            } else {
                1e-5
            };
            prop_assert!(
                (pow_val - sqrt_val).abs() < tolerance,
                "x^0.5 vs sqrt failed at {}: {} != {}",
                i, pow_val, sqrt_val
            );
        }
    }
}

// Property test: exp() correctness vs f32::exp()
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_exp_correctness(
        a in prop::collection::vec(-10.0f32..10.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.exp().unwrap();

        for (i, (&a_val, &exp_val)) in a.iter()
            .zip(result.as_slice().iter())
            .enumerate() {
            let expected = a_val.exp();

            let tolerance = if expected.abs() > 1.0 {
                expected.abs() * 1e-5
            } else {
                1e-5
            };

            prop_assert!(
                (exp_val - expected).abs() < tolerance,
                "exp correctness failed at {}: {} != {}, diff = {}",
                i, exp_val, expected, (exp_val - expected).abs()
            );
        }
    }
}

// Property test: exp() identity - exp(0) = 1
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_exp_zero_identity(
        len in 1usize..100
    ) {
        let zeros = vec![0.0f32; len];
        let va = Vector::from_slice(&zeros);
        let result = va.exp().unwrap();

        for (i, &val) in result.as_slice().iter().enumerate() {
            prop_assert!(
                (val - 1.0).abs() < 1e-5,
                "exp(0) identity failed at {}: {} != 1.0",
                i, val
            );
        }
    }
}

// Property test: exp() relation to addition - exp(a+b) = exp(a) * exp(b)
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_exp_addition_property(
        a in prop::collection::vec(-5.0f32..5.0, 1..50),
        b in prop::collection::vec(-5.0f32..5.0, 1..50)
    ) {
        let len = a.len().min(b.len());
        let a_vec: Vec<f32> = a.into_iter().take(len).collect();
        let b_vec: Vec<f32> = b.into_iter().take(len).collect();

        let va = Vector::from_slice(&a_vec);
        let vb = Vector::from_slice(&b_vec);

        // exp(a + b)
        let sum = va.add(&vb).unwrap();
        let exp_sum = sum.exp().unwrap();

        // exp(a) * exp(b)
        let exp_a = va.exp().unwrap();
        let exp_b = vb.exp().unwrap();
        let product = exp_a.mul(&exp_b).unwrap();

        for (i, (&exp_sum_val, &product_val)) in exp_sum.as_slice().iter()
            .zip(product.as_slice().iter())
            .enumerate() {
            let tolerance = if exp_sum_val.abs() > 1.0 {
                exp_sum_val.abs() * 1e-4
            } else {
                1e-4
            };

            prop_assert!(
                (exp_sum_val - product_val).abs() < tolerance,
                "exp(a+b) = exp(a)*exp(b) failed at {}: {} != {}, diff = {}",
                i, exp_sum_val, product_val, (exp_sum_val - product_val).abs()
            );
        }
    }
}
