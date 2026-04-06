use super::super::super::*;
use proptest::prelude::*;

// Property test: sin() correctness vs f32::sin()
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_sin_correctness(
        a in prop::collection::vec(-10.0f32..10.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.sin().unwrap();

        for (i, (&a_val, &sin_val)) in a.iter()
            .zip(result.as_slice().iter())
            .enumerate() {
            let expected = a_val.sin();

            prop_assert!(
                (sin_val - expected).abs() < 1e-5,
                "sin correctness failed at {}: {} != {}, diff = {}",
                i, sin_val, expected, (sin_val - expected).abs()
            );
        }
    }
}

// Property test: sin() range [-1, 1]
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_sin_range(
        a in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.sin().unwrap();

        for (i, &sin_val) in result.as_slice().iter().enumerate() {
            prop_assert!(
                (-1.0..=1.0).contains(&sin_val),
                "sin range failed at {}: {} not in [-1, 1]",
                i, sin_val
            );
        }
    }
}

// Property test: sin() periodicity - sin(x + 2π) = sin(x)
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_sin_periodicity_property(
        a in prop::collection::vec(-5.0f32..5.0, 1..50)
    ) {
        use std::f32::consts::PI;

        let va = Vector::from_slice(&a);
        let sin_a = va.sin().unwrap();

        // Add 2π to each element
        let a_plus_2pi: Vec<f32> = a.iter().map(|&x| x + 2.0 * PI).collect();
        let va_plus_2pi = Vector::from_slice(&a_plus_2pi);
        let sin_a_plus_2pi = va_plus_2pi.sin().unwrap();

        for (i, (&sin_val, &sin_periodic_val)) in sin_a.as_slice().iter()
            .zip(sin_a_plus_2pi.as_slice().iter())
            .enumerate() {
            prop_assert!(
                (sin_val - sin_periodic_val).abs() < 1e-5,
                "sin periodicity failed at {}: {} != {}, diff = {}",
                i, sin_val, sin_periodic_val, (sin_val - sin_periodic_val).abs()
            );
        }
    }
}

// Property test: cos() correctness vs f32::cos()
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_cos_correctness(
        a in prop::collection::vec(-10.0f32..10.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.cos().unwrap();

        for (i, (&a_val, &cos_val)) in a.iter()
            .zip(result.as_slice().iter())
            .enumerate() {
            let expected = a_val.cos();

            prop_assert!(
                (cos_val - expected).abs() < 1e-5,
                "cos correctness failed at {}: {} != {}, diff = {}",
                i, cos_val, expected, (cos_val - expected).abs()
            );
        }
    }
}

// Property test: cos() range [-1, 1]
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_cos_range(
        a in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.cos().unwrap();

        for (i, &cos_val) in result.as_slice().iter().enumerate() {
            prop_assert!(
                (-1.0..=1.0).contains(&cos_val),
                "cos range failed at {}: {} not in [-1, 1]",
                i, cos_val
            );
        }
    }
}

// Property test: Pythagorean identity - sin²(x) + cos²(x) = 1
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_pythagorean_identity(
        a in prop::collection::vec(-10.0f32..10.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let sin_result = va.sin().unwrap();
        let cos_result = va.cos().unwrap();

        for (i, (&sin_val, &cos_val)) in sin_result.as_slice().iter()
            .zip(cos_result.as_slice().iter())
            .enumerate() {
            let sum_of_squares = sin_val * sin_val + cos_val * cos_val;

            prop_assert!(
                (sum_of_squares - 1.0).abs() < 1e-5,
                "Pythagorean identity failed at {}: sin²+cos² = {} != 1.0",
                i, sum_of_squares
            );
        }
    }
}

// Property test: tan() correctness
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_tan_correctness(
        a in prop::collection::vec(-1.5f32..1.5, 1..100)
    ) {
        // Use limited range to avoid tan asymptotes at ±π/2
        let va = Vector::from_slice(&a);
        let result = va.tan().unwrap();

        for (i, (&a_val, &tan_val)) in a.iter()
            .zip(result.as_slice().iter())
            .enumerate() {
            let expected = a_val.tan();

            prop_assert!(
                (tan_val - expected).abs() < 1e-5,
                "tan correctness failed at {}: {} != {}, diff = {}",
                i, tan_val, expected, (tan_val - expected).abs()
            );
        }
    }
}

// Property test: tan(x) = sin(x) / cos(x)
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_tan_sin_cos_identity(
        a in prop::collection::vec(-1.5f32..1.5, 1..100)
    ) {
        // Avoid asymptotes at ±π/2 where cos(x) ≈ 0
        let va = Vector::from_slice(&a);
        let tan_result = va.tan().unwrap();
        let sin_result = va.sin().unwrap();
        let cos_result = va.cos().unwrap();

        for (i, ((&tan_val, &sin_val), &cos_val)) in tan_result.as_slice().iter()
            .zip(sin_result.as_slice().iter())
            .zip(cos_result.as_slice().iter())
            .enumerate() {
            // Skip values where cos is very small (near asymptote)
            if cos_val.abs() > 1e-3 {
                let expected = sin_val / cos_val;
                prop_assert!(
                    (tan_val - expected).abs() < 1e-4,
                    "tan(x) != sin(x)/cos(x) at {}: {} != {}, cos={}",
                    i, tan_val, expected, cos_val
                );
            }
        }
    }
}

// Property test: tan is odd function - tan(-x) = -tan(x)
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_tan_odd_function(
        a in prop::collection::vec(-1.5f32..1.5, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let tan_pos = va.tan().unwrap();

        let a_neg: Vec<f32> = a.iter().map(|x| -x).collect();
        let va_neg = Vector::from_slice(&a_neg);
        let tan_neg = va_neg.tan().unwrap();

        for (i, (&pos, &neg)) in tan_pos.as_slice().iter()
            .zip(tan_neg.as_slice().iter())
            .enumerate() {
            prop_assert!(
                (pos + neg).abs() < 1e-5,
                "tan odd function failed at {}: tan(-x)={} != -tan(x)={}",
                i, neg, -pos
            );
        }
    }
}

// Property test: asin() correctness
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_asin_correctness(
        a in prop::collection::vec(-1.0f32..1.0, 1..100)
    ) {
        // Domain is [-1, 1] for asin
        let va = Vector::from_slice(&a);
        let result = va.asin().unwrap();

        for (i, (&a_val, &asin_val)) in a.iter()
            .zip(result.as_slice().iter())
            .enumerate() {
            let expected = a_val.asin();

            prop_assert!(
                (asin_val - expected).abs() < 1e-5,
                "asin correctness failed at {}: {} != {}, diff = {}",
                i, asin_val, expected, (asin_val - expected).abs()
            );
        }
    }
}

// Property test: asin(sin(x)) = x for x in [-π/2, π/2]
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_asin_sin_inverse(
        a in prop::collection::vec(-1.5f32..1.5, 1..100)
    ) {
        // Test range within [-π/2, π/2] to ensure inverse property
        let va = Vector::from_slice(&a);
        let sin_result = va.sin().unwrap();
        let asin_result = sin_result.asin().unwrap();

        for (i, (&original, &reconstructed)) in a.iter()
            .zip(asin_result.as_slice().iter())
            .enumerate() {
            prop_assert!(
                (original - reconstructed).abs() < 1e-5,
                "asin(sin(x)) != x at {}: {} != {}",
                i, reconstructed, original
            );
        }
    }
}

// Property test: asin is odd function - asin(-x) = -asin(x)
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_asin_odd_function(
        a in prop::collection::vec(-1.0f32..1.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let asin_pos = va.asin().unwrap();

        let a_neg: Vec<f32> = a.iter().map(|x| -x).collect();
        let va_neg = Vector::from_slice(&a_neg);
        let asin_neg = va_neg.asin().unwrap();

        for (i, (&pos, &neg)) in asin_pos.as_slice().iter()
            .zip(asin_neg.as_slice().iter())
            .enumerate() {
            prop_assert!(
                (pos + neg).abs() < 1e-5,
                "asin odd function failed at {}: asin(-x)={} != -asin(x)={}",
                i, neg, -pos
            );
        }
    }
}

// Property test: acos() correctness
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_acos_correctness(
        a in prop::collection::vec(-1.0f32..1.0, 1..100)
    ) {
        // Domain is [-1, 1] for acos
        let va = Vector::from_slice(&a);
        let result = va.acos().unwrap();

        for (i, (&a_val, &acos_val)) in a.iter()
            .zip(result.as_slice().iter())
            .enumerate() {
            let expected = a_val.acos();

            prop_assert!(
                (acos_val - expected).abs() < 1e-5,
                "acos correctness failed at {}: {} != {}, diff = {}",
                i, acos_val, expected, (acos_val - expected).abs()
            );
        }
    }
}

// Property test: acos(cos(x)) = x for x in [0, π]
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_acos_cos_inverse(
        a in prop::collection::vec(0.0f32..std::f32::consts::PI, 1..100)
    ) {
        // Test range within [0, π] to ensure inverse property
        let va = Vector::from_slice(&a);
        let cos_result = va.cos().unwrap();
        let acos_result = cos_result.acos().unwrap();

        for (i, (&original, &reconstructed)) in a.iter()
            .zip(acos_result.as_slice().iter())
            .enumerate() {
            prop_assert!(
                (original - reconstructed).abs() < 3e-4,
                "acos(cos(x)) != x at {}: {} != {}",
                i, reconstructed, original
            );
        }
    }
}

// Property test: acos symmetry - acos(-x) = π - acos(x)
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_acos_symmetry(
        a in prop::collection::vec(-1.0f32..1.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let acos_pos = va.acos().unwrap();

        let a_neg: Vec<f32> = a.iter().map(|x| -x).collect();
        let va_neg = Vector::from_slice(&a_neg);
        let acos_neg = va_neg.acos().unwrap();

        for (i, (&pos, &neg)) in acos_pos.as_slice().iter()
            .zip(acos_neg.as_slice().iter())
            .enumerate() {
            let expected = std::f32::consts::PI - pos;
            prop_assert!(
                (neg - expected).abs() < 1e-5,
                "acos symmetry failed at {}: acos(-x)={} != π - acos(x)={}",
                i, neg, expected
            );
        }
    }
}

// Property test: atan() correctness
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_atan_correctness(
        a in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        // atan accepts all real numbers
        let va = Vector::from_slice(&a);
        let result = va.atan().unwrap();

        for (i, (&a_val, &atan_val)) in a.iter()
            .zip(result.as_slice().iter())
            .enumerate() {
            let expected = a_val.atan();

            prop_assert!(
                (atan_val - expected).abs() < 1e-5,
                "atan correctness failed at {}: {} != {}, diff = {}",
                i, atan_val, expected, (atan_val - expected).abs()
            );
        }
    }
}

// Property test: atan(tan(x)) = x for x in (-π/2, π/2)
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_atan_tan_inverse(
        a in prop::collection::vec(-1.5f32..1.5, 1..100)
    ) {
        // Test range within (-π/2, π/2) to ensure inverse property
        let va = Vector::from_slice(&a);
        let tan_result = va.tan().unwrap();
        let atan_result = tan_result.atan().unwrap();

        for (i, (&original, &reconstructed)) in a.iter()
            .zip(atan_result.as_slice().iter())
            .enumerate() {
            prop_assert!(
                (original - reconstructed).abs() < 1e-5,
                "atan(tan(x)) != x at {}: {} != {}",
                i, reconstructed, original
            );
        }
    }
}

// Property test: atan is odd function - atan(-x) = -atan(x)
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_atan_odd_function(
        a in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let atan_pos = va.atan().unwrap();

        let a_neg: Vec<f32> = a.iter().map(|x| -x).collect();
        let va_neg = Vector::from_slice(&a_neg);
        let atan_neg = va_neg.atan().unwrap();

        for (i, (&pos, &neg)) in atan_pos.as_slice().iter()
            .zip(atan_neg.as_slice().iter())
            .enumerate() {
            prop_assert!(
                (pos + neg).abs() < 1e-5,
                "atan odd function failed at {}: atan(-x)={} != -atan(x)={}",
                i, neg, -pos
            );
        }
    }
}
