use super::super::super::super::*;
use proptest::prelude::*;

// Property test: sqrt() correctness
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_sqrt_correctness(
        a in prop::collection::vec(0.0f32..100.0, 1..100)  // Non-negative values only
    ) {
        let va = Vector::from_slice(&a);
        let result = va.sqrt().unwrap();

        // Verify: result[i] = sqrt(a[i])
        for (i, (&a_val, &result_val)) in a.iter()
            .zip(result.as_slice().iter())
            .enumerate() {
            let expected = a_val.sqrt();

            let tolerance = if expected.abs() > 1.0 {
                expected.abs() * 1e-6
            } else {
                1e-6
            };

            prop_assert!(
                (result_val - expected).abs() < tolerance,
                "sqrt correctness failed at {}: {} != {}, diff = {}",
                i, result_val, expected, (result_val - expected).abs()
            );
        }
    }
}

// Property test: sqrt() idempotence with squaring
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_sqrt_inverse_square(
        a in prop::collection::vec(0.0f32..100.0, 1..100)
    ) {
        // sqrt(a)^2 = a
        let va = Vector::from_slice(&a);
        let sqrt_result = va.sqrt().unwrap();
        let squared = sqrt_result.mul(&sqrt_result).unwrap();

        for (i, (&original, &recovered)) in a.iter()
            .zip(squared.as_slice().iter())
            .enumerate() {
            let tolerance = if original.abs() > 1.0 {
                original.abs() * 1e-5
            } else {
                1e-5
            };

            prop_assert!(
                (original - recovered).abs() < tolerance,
                "sqrt inverse failed at {}: {} != {}, diff = {}",
                i, original, recovered, (original - recovered).abs()
            );
        }
    }
}

// Property test: sqrt() monotonicity
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_sqrt_monotonic(
        a in prop::collection::vec(0.0f32..100.0, 2..100)
    ) {
        // If a[i] < a[j], then sqrt(a[i]) <= sqrt(a[j])
        let va = Vector::from_slice(&a);
        let result = va.sqrt().unwrap();
        let result_slice = result.as_slice();

        for i in 0..a.len()-1 {
            for j in i+1..a.len() {
                // Use a small epsilon to account for f32 precision
                let epsilon = 1e-6;
                if a[i] + epsilon < a[j] {
                    prop_assert!(
                        result_slice[i] <= result_slice[j],
                        "Monotonicity failed: sqrt({}) = {} should be <= sqrt({}) = {}",
                        a[i], result_slice[i], a[j], result_slice[j]
                    );
                } else if a[i] > a[j] + epsilon {
                    prop_assert!(
                        result_slice[i] >= result_slice[j],
                        "Monotonicity failed: sqrt({}) = {} should be >= sqrt({}) = {}",
                        a[i], result_slice[i], a[j], result_slice[j]
                    );
                }
            }
        }
    }
}

// Property test: recip() correctness
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_recip_correctness(
        a in prop::collection::vec(0.1f32..100.0, 1..100)  // Avoid zeros and very small values
    ) {
        let va = Vector::from_slice(&a);
        let result = va.recip().unwrap();

        // Verify: result[i] = 1 / a[i]
        for (i, (&a_val, &result_val)) in a.iter()
            .zip(result.as_slice().iter())
            .enumerate() {
            let expected = 1.0 / a_val;

            let tolerance = if expected.abs() > 1.0 {
                expected.abs() * 1e-6
            } else {
                1e-6
            };

            prop_assert!(
                (result_val - expected).abs() < tolerance,
                "recip correctness failed at {}: {} != {}, diff = {}",
                i, result_val, expected, (result_val - expected).abs()
            );
        }
    }
}

// Property test: recip() inverse property
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_recip_inverse(
        a in prop::collection::vec(0.1f32..100.0, 1..100)
    ) {
        // recip(recip(a)) = a
        let va = Vector::from_slice(&a);
        let recip_once = va.recip().unwrap();
        let recip_twice = recip_once.recip().unwrap();

        for (i, (&original, &recovered)) in a.iter()
            .zip(recip_twice.as_slice().iter())
            .enumerate() {
            let tolerance = if original.abs() > 1.0 {
                original.abs() * 1e-5
            } else {
                1e-5
            };

            prop_assert!(
                (original - recovered).abs() < tolerance,
                "recip inverse failed at {}: {} != {}, diff = {}",
                i, original, recovered, (original - recovered).abs()
            );
        }
    }
}

// Property test: recip() relation to division
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_recip_vs_division(
        a in prop::collection::vec(0.1f32..100.0, 1..100),
        scalar in 0.1f32..100.0
    ) {
        // scalar * recip(a) should equal scalar / a
        let va = Vector::from_slice(&a);
        let recip_result = va.recip().unwrap();
        let scaled = recip_result.scale(scalar).unwrap();

        for (i, (&a_val, &scaled_val)) in a.iter()
            .zip(scaled.as_slice().iter())
            .enumerate() {
            let expected = scalar / a_val;

            let tolerance = if expected.abs() > 1.0 {
                expected.abs() * 1e-5
            } else {
                1e-5
            };

            prop_assert!(
                (scaled_val - expected).abs() < tolerance,
                "recip vs division failed at {}: {} != {}, diff = {}",
                i, scaled_val, expected, (scaled_val - expected).abs()
            );
        }
    }
}
