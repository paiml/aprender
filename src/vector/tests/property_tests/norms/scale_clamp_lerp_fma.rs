use super::super::super::super::*;
use proptest::prelude::*;

// Property test: scale() distributivity over addition
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_scale_distributive(
        a in prop::collection::vec(-100.0f32..100.0, 1..100),
        scalar in -10.0f32..10.0
    ) {
        // scalar * (a + a) = (scalar * a) + (scalar * a)
        let va = Vector::from_slice(&a);
        let va_plus_va = va.add(&va).unwrap();
        let scaled_sum = va_plus_va.scale(scalar).unwrap();

        let scaled_a = va.scale(scalar).unwrap();
        let sum_of_scaled = scaled_a.add(&scaled_a).unwrap();

        for (i, (&val1, &val2)) in scaled_sum.as_slice().iter()
            .zip(sum_of_scaled.as_slice().iter())
            .enumerate() {
            let tolerance = if val1.abs() > 1.0 {
                val1.abs() * 1e-5
            } else {
                1e-3
            };
            prop_assert!(
                (val1 - val2).abs() < tolerance,
                "Distributivity failed at {}: {} != {}, diff = {}",
                i, val1, val2, (val1 - val2).abs()
            );
        }
    }
}

// Property test: scale() with 1.0 is identity
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_scale_identity(
        a in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.scale(1.0).unwrap();

        for (i, (&original, &scaled)) in a.iter()
            .zip(result.as_slice().iter())
            .enumerate() {
            prop_assert!(
                (original - scaled).abs() < 1e-5,
                "Identity failed at {}: {} != {}",
                i, original, scaled
            );
        }
    }
}

// Property test: scale() with 0.0 gives zeros
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_scale_zero(
        a in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.scale(0.0).unwrap();

        for (i, &val) in result.as_slice().iter().enumerate() {
            prop_assert!(
                val.abs() < 1e-10,
                "Zero scaling failed at {}: {} != 0.0",
                i, val
            );
        }
    }
}

// Property test: scale() associativity
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_scale_associative(
        a in prop::collection::vec(-100.0f32..100.0, 1..100),
        scalar1 in -10.0f32..10.0,
        scalar2 in -10.0f32..10.0
    ) {
        // (a * s1) * s2 = a * (s1 * s2)
        let va = Vector::from_slice(&a);
        let scaled_once = va.scale(scalar1).unwrap();
        let scaled_twice = scaled_once.scale(scalar2).unwrap();

        let combined_scalar = scalar1 * scalar2;
        let scaled_combined = va.scale(combined_scalar).unwrap();

        for (i, (&val1, &val2)) in scaled_twice.as_slice().iter()
            .zip(scaled_combined.as_slice().iter())
            .enumerate() {
            let tolerance = if val1.abs() > 1.0 {
                val1.abs() * 1e-4  // Slightly higher tolerance for double scaling
            } else {
                1e-3
            };
            prop_assert!(
                (val1 - val2).abs() < tolerance,
                "Associativity failed at {}: {} != {}, diff = {}",
                i, val1, val2, (val1 - val2).abs()
            );
        }
    }
}

// Property test: clamp() bounds enforcement
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_clamp_bounds(
        a in prop::collection::vec(-100.0f32..100.0, 1..100),
        min_val in -50.0f32..0.0,
        max_val in 0.0f32..50.0
    ) {
        let va = Vector::from_slice(&a);
        let result = va.clamp(min_val, max_val).unwrap();

        for (i, &val) in result.as_slice().iter().enumerate() {
            prop_assert!(
                val >= min_val && val <= max_val,
                "Value {} out of bounds [{}, {}] at index {}",
                val, min_val, max_val, i
            );
        }
    }
}

// Property test: clamp() idempotence
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_clamp_idempotent(
        a in prop::collection::vec(-100.0f32..100.0, 1..100),
        min_val in -50.0f32..0.0,
        max_val in 0.0f32..50.0
    ) {
        // clamp(clamp(v)) = clamp(v)
        let va = Vector::from_slice(&a);
        let clamped_once = va.clamp(min_val, max_val).unwrap();
        let clamped_twice = clamped_once.clamp(min_val, max_val).unwrap();

        for (i, (&val1, &val2)) in clamped_once.as_slice().iter()
            .zip(clamped_twice.as_slice().iter())
            .enumerate() {
            prop_assert!(
                (val1 - val2).abs() < 1e-10,
                "Idempotence failed at {}: {} != {}",
                i, val1, val2
            );
        }
    }
}

// Property test: clamp() monotonicity
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_clamp_monotonic(
        a in prop::collection::vec(-100.0f32..100.0, 2..100)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.clamp(-50.0, 50.0).unwrap();

        // For any i < j, if a[i] <= a[j], then clamp(a[i]) <= clamp(a[j])
        for i in 0..a.len() - 1 {
            for j in i + 1..a.len() {
                if a[i] <= a[j] {
                    prop_assert!(
                        result.as_slice()[i] <= result.as_slice()[j],
                        "Monotonicity violated: a[{}]={} <= a[{}]={} but clamp[{}]={} > clamp[{}]={}",
                        i, a[i], j, a[j], i, result.as_slice()[i], j, result.as_slice()[j]
                    );
                }
            }
        }
    }
}

// Property test: lerp() at endpoints
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_lerp_endpoints(
        a in prop::collection::vec(-100.0f32..100.0, 1..100),
        b in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        let len = a.len().min(b.len());
        let a_trimmed = &a[..len];
        let b_trimmed = &b[..len];

        let va = Vector::from_slice(a_trimmed);
        let vb = Vector::from_slice(b_trimmed);

        // t=0 should return a
        let result_zero = va.lerp(&vb, 0.0).unwrap();
        for (i, (&actual, &expected)) in result_zero.as_slice().iter()
            .zip(a_trimmed.iter())
            .enumerate() {
            prop_assert!(
                (actual - expected).abs() < 1e-5,
                "lerp(t=0) failed at {}: {} != {}",
                i, actual, expected
            );
        }

        // t=1 should return b
        let result_one = va.lerp(&vb, 1.0).unwrap();
        for (i, (&actual, &expected)) in result_one.as_slice().iter()
            .zip(b_trimmed.iter())
            .enumerate() {
            prop_assert!(
                (actual - expected).abs() < 1e-5,
                "lerp(t=1) failed at {}: {} != {}",
                i, actual, expected
            );
        }
    }
}

// Property test: lerp() linearity
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_lerp_linearity(
        a in prop::collection::vec(-100.0f32..100.0, 1..100),
        b in prop::collection::vec(-100.0f32..100.0, 1..100),
        t in 0.0f32..1.0
    ) {
        let len = a.len().min(b.len());
        let a_trimmed = &a[..len];
        let b_trimmed = &b[..len];

        let va = Vector::from_slice(a_trimmed);
        let vb = Vector::from_slice(b_trimmed);

        let result = va.lerp(&vb, t).unwrap();

        // Verify: result[i] = a[i] + t * (b[i] - a[i])
        // f32 lerp: a + t*(b-a) has 2 rounding points (subtraction + FMA).
        // Relative tolerance of 2e-5 covers worst-case f32 ULP accumulation.
        for (i, ((&a_val, &b_val), &result_val)) in a_trimmed.iter()
            .zip(b_trimmed.iter())
            .zip(result.as_slice().iter())
            .enumerate() {
            let expected = a_val + t * (b_val - a_val);

            let tolerance = if expected.abs() > 1.0 {
                expected.abs() * 2e-5
            } else {
                1e-4
            };

            prop_assert!(
                (result_val - expected).abs() < tolerance,
                "Linearity failed at {}: {} != {}, diff = {}",
                i, result_val, expected, (result_val - expected).abs()
            );
        }
    }
}

// Property test: lerp() symmetry
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_lerp_symmetry(
        a in prop::collection::vec(-100.0f32..100.0, 1..100),
        b in prop::collection::vec(-100.0f32..100.0, 1..100),
        t in 0.0f32..1.0
    ) {
        let len = a.len().min(b.len());
        let a_trimmed = &a[..len];
        let b_trimmed = &b[..len];

        let va = Vector::from_slice(a_trimmed);
        let vb = Vector::from_slice(b_trimmed);

        // lerp(a, b, t) should equal lerp(b, a, 1-t)
        let forward = va.lerp(&vb, t).unwrap();
        let reverse = vb.lerp(&va, 1.0 - t).unwrap();

        for (i, (&fwd, &rev)) in forward.as_slice().iter()
            .zip(reverse.as_slice().iter())
            .enumerate() {
            let tolerance = if fwd.abs() > 1.0 {
                fwd.abs() * 2e-5
            } else {
                1e-4
            };

            prop_assert!(
                (fwd - rev).abs() < tolerance,
                "Symmetry failed at {}: {} != {}, diff = {}",
                i, fwd, rev, (fwd - rev).abs()
            );
        }
    }
}

// Property test: fma() correctness
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_fma_correctness(
        a in prop::collection::vec(-100.0f32..100.0, 1..100),
        b in prop::collection::vec(-100.0f32..100.0, 1..100),
        c in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        let len = a.len().min(b.len()).min(c.len());
        let a_trimmed = &a[..len];
        let b_trimmed = &b[..len];
        let c_trimmed = &c[..len];

        let va = Vector::from_slice(a_trimmed);
        let vb = Vector::from_slice(b_trimmed);
        let vc = Vector::from_slice(c_trimmed);

        let result = va.fma(&vb, &vc).unwrap();

        // Verify: result[i] = a[i] * b[i] + c[i]
        for (i, ((&a_val, &b_val), (&c_val, &result_val))) in a_trimmed.iter()
            .zip(b_trimmed.iter())
            .zip(c_trimmed.iter().zip(result.as_slice().iter()))
            .enumerate() {
            let expected = a_val * b_val + c_val;

            let tolerance = if expected.abs() > 1.0 {
                expected.abs() * 1e-5
            } else {
                1e-4
            };

            prop_assert!(
                (result_val - expected).abs() < tolerance,
                "FMA correctness failed at {}: {} != {}, diff = {}",
                i, result_val, expected, (result_val - expected).abs()
            );
        }
    }
}

// Property test: fma() with zero multiplication
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_fma_zero_mul(
        a in prop::collection::vec(-100.0f32..100.0, 1..100),
        c in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        // fma(a, 0, c) = 0 * a + c = c
        let len = a.len().min(c.len());
        let a_trimmed = &a[..len];
        let c_trimmed = &c[..len];

        let va = Vector::from_slice(a_trimmed);
        let vc = Vector::from_slice(c_trimmed);
        let zeros = vec![0.0; len];
        let vzero = Vector::from_slice(&zeros);

        let result = va.fma(&vzero, &vc).unwrap();

        for (i, (&result_val, &c_val)) in result.as_slice().iter()
            .zip(c_trimmed.iter())
            .enumerate() {
            prop_assert!(
                (result_val - c_val).abs() < 1e-10,
                "Zero multiplication failed at {}: {} != {}, diff = {}",
                i, result_val, c_val, (result_val - c_val).abs()
            );
        }
    }
}

// Property test: fma() relation to mul and add
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_fma_vs_mul_add(
        a in prop::collection::vec(-100.0f32..100.0, 1..100),
        b in prop::collection::vec(-100.0f32..100.0, 1..100),
        c in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        let len = a.len().min(b.len()).min(c.len());
        let a_trimmed = &a[..len];
        let b_trimmed = &b[..len];
        let c_trimmed = &c[..len];

        let va = Vector::from_slice(a_trimmed);
        let vb = Vector::from_slice(b_trimmed);
        let vc = Vector::from_slice(c_trimmed);

        // fma(a, b, c) should approximately equal mul(a, b) + c
        let fma_result = va.fma(&vb, &vc).unwrap();
        let mul_result = va.mul(&vb).unwrap();
        let add_result = mul_result.add(&vc).unwrap();

        for (i, (&fma_val, &add_val)) in fma_result.as_slice().iter()
            .zip(add_result.as_slice().iter())
            .enumerate() {
            // FMA can have better accuracy, so use slightly higher tolerance
            let tolerance = if fma_val.abs() > 1.0 {
                fma_val.abs() * 1e-5
            } else {
                1e-4
            };

            prop_assert!(
                (fma_val - add_val).abs() < tolerance,
                "FMA vs mul+add failed at {}: {} != {}, diff = {}",
                i, fma_val, add_val, (fma_val - add_val).abs()
            );
        }
    }
}
