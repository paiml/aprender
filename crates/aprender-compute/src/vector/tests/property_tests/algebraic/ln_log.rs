use super::super::super::super::*;
use proptest::prelude::*;

// Property test: ln() correctness vs f32::ln()
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_ln_correctness(
        a in prop::collection::vec(0.1f32..100.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.ln().unwrap();

        for (i, (&a_val, &ln_val)) in a.iter()
            .zip(result.as_slice().iter())
            .enumerate() {
            let expected = a_val.ln();

            let tolerance = if expected.abs() > 1.0 {
                expected.abs() * 1e-5
            } else {
                1e-5
            };

            prop_assert!(
                (ln_val - expected).abs() < tolerance,
                "ln correctness failed at {}: {} != {}, diff = {}",
                i, ln_val, expected, (ln_val - expected).abs()
            );
        }
    }
}

// Property test: ln() inverse of exp() - ln(exp(x)) = x
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_ln_inverse_exp_property(
        a in prop::collection::vec(-5.0f32..5.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let exp_result = va.exp().unwrap();
        let ln_result = exp_result.ln().unwrap();

        for (i, (&original, &recovered)) in a.iter()
            .zip(ln_result.as_slice().iter())
            .enumerate() {
            let tolerance = if original.abs() > 1.0 {
                original.abs() * 1e-4
            } else {
                1e-4
            };

            prop_assert!(
                (original - recovered).abs() < tolerance,
                "ln(exp(x)) != x at {}: {} != {}, diff = {}",
                i, original, recovered, (original - recovered).abs()
            );
        }
    }
}

// Property test: ln() product rule - ln(a*b) = ln(a) + ln(b)
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_ln_product_rule(
        a in prop::collection::vec(0.1f32..10.0, 1..50),
        b in prop::collection::vec(0.1f32..10.0, 1..50)
    ) {
        let len = a.len().min(b.len());
        let a_vec: Vec<f32> = a.into_iter().take(len).collect();
        let b_vec: Vec<f32> = b.into_iter().take(len).collect();

        let va = Vector::from_slice(&a_vec);
        let vb = Vector::from_slice(&b_vec);

        // ln(a * b)
        let product = va.mul(&vb).unwrap();
        let ln_product = product.ln().unwrap();

        // ln(a) + ln(b)
        let ln_a = va.ln().unwrap();
        let ln_b = vb.ln().unwrap();
        let sum = ln_a.add(&ln_b).unwrap();

        for (i, (&ln_prod_val, &sum_val)) in ln_product.as_slice().iter()
            .zip(sum.as_slice().iter())
            .enumerate() {
            let tolerance = if ln_prod_val.abs() > 1.0 {
                ln_prod_val.abs() * 1e-4
            } else {
                1e-4
            };

            prop_assert!(
                (ln_prod_val - sum_val).abs() < tolerance,
                "ln(a*b) = ln(a)+ln(b) failed at {}: {} != {}, diff = {}",
                i, ln_prod_val, sum_val, (ln_prod_val - sum_val).abs()
            );
        }
    }
}

// Property test: log2() correctness vs f32::log2()
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_log2_correctness(
        a in prop::collection::vec(0.001f32..1000.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.log2().unwrap();

        for (i, (&a_val, &log2_val)) in a.iter()
            .zip(result.as_slice().iter())
            .enumerate() {
            let expected = a_val.log2();

            prop_assert!(
                (log2_val - expected).abs() < 1e-4,
                "log2 correctness failed at {}: {} != {}, diff = {}",
                i, log2_val, expected, (log2_val - expected).abs()
            );
        }
    }
}

// Property test: log2(2^n) = n (power of 2 property)
proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    #[test]
    fn test_log2_power_property(
        n in prop::collection::vec(-10i32..10, 1..50)
    ) {
        let powers: Vec<f32> = n.iter().map(|&exp| 2.0f32.powi(exp)).collect();
        let va = Vector::from_slice(&powers);
        let result = va.log2().unwrap();

        for (&exp_val, &log2_val) in n.iter()
            .zip(result.as_slice().iter()) {
            let expected = exp_val as f32;

            prop_assert!(
                (log2_val - expected).abs() < 1e-4,
                "log2(2^{}) should be {}, got {}, diff = {}",
                exp_val, expected, log2_val, (log2_val - expected).abs()
            );
        }
    }
}

// Property test: log10() correctness vs f32::log10()
// Skipped on Windows due to upstream proptest float sampler bug
// (assertion in float_samplers.rs:462 with f32 range 0.001..1000.0)
#[cfg(not(target_os = "windows"))]
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_log10_correctness(
        a in prop::collection::vec(0.001f32..1000.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.log10().unwrap();

        for (i, (&a_val, &log10_val)) in a.iter()
            .zip(result.as_slice().iter())
            .enumerate() {
            let expected = a_val.log10();

            prop_assert!(
                (log10_val - expected).abs() < 1e-4,
                "log10 correctness failed at {}: {} != {}, diff = {}",
                i, log10_val, expected, (log10_val - expected).abs()
            );
        }
    }
}

// Property test: log10(10^n) = n (power of 10 property)
proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    #[test]
    fn test_log10_power_property(
        n in prop::collection::vec(-3i32..6, 1..30)
    ) {
        let powers: Vec<f32> = n.iter().map(|&exp| 10.0f32.powi(exp)).collect();
        let va = Vector::from_slice(&powers);
        let result = va.log10().unwrap();

        for (&exp_val, &log10_val) in n.iter()
            .zip(result.as_slice().iter()) {
            let expected = exp_val as f32;

            prop_assert!(
                (log10_val - expected).abs() < 1e-3,
                "log10(10^{}) should be {}, got {}, diff = {}",
                exp_val, expected, log10_val, (log10_val - expected).abs()
            );
        }
    }
}
