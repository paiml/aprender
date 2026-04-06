mod scale_clamp_lerp_fma;

use super::super::super::*;
use proptest::prelude::*;

// Property test: Dot product with self is non-negative (norm property)
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_dot_self_nonnegative(
        a in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.dot(&va).unwrap();

        // ||v||^2 = v·v >= 0 always
        prop_assert!(result >= 0.0);

        // If all zeros, should be exactly zero
        if a.iter().all(|&x| x == 0.0) {
            prop_assert_eq!(result, 0.0);
        } else {
            // If any non-zero element, result should be positive
            prop_assert!(result > 0.0);
        }
    }
}

// Property test: L2 norm is always non-negative
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_norm_l2_nonnegative(
        a in prop::collection::vec(-1000.0f32..1000.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let norm = va.norm_l2().unwrap();

        // ||v|| >= 0 always
        prop_assert!(norm >= 0.0);

        // If all zeros, norm should be exactly zero
        if a.iter().all(|&x| x.abs() < 1e-6) {
            prop_assert!(norm < 1e-5);
        }
    }
}

// Property test: L2 norm equals sqrt(dot(a, a))
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_norm_l2_equals_sqrt_dot(
        a in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let norm = va.norm_l2().unwrap();
        let dot_self = va.dot(&va).unwrap();

        // ||a|| = sqrt(a·a)
        // Use relative tolerance for large values
        let relative_error = if dot_self > 0.0 {
            ((norm * norm - dot_self) / dot_self).abs()
        } else {
            (norm * norm - dot_self).abs()
        };
        prop_assert!(relative_error < 1e-4 || (norm * norm - dot_self).abs() < 1e-2);
    }
}

// Property test: Scaling property ||c*a|| = |c| * ||a||
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_norm_l2_scaling(
        a in prop::collection::vec(-100.0f32..100.0, 1..100),
        c in -10.0f32..10.0
    ) {
        let va = Vector::from_slice(&a);
        let norm_a = va.norm_l2().unwrap();

        // Create c*a
        let scaled: Vec<f32> = a.iter().map(|&x| c * x).collect();
        let v_scaled = Vector::from_slice(&scaled);
        let norm_scaled = v_scaled.norm_l2().unwrap();

        // ||c*a|| = |c| * ||a||
        let expected = c.abs() * norm_a;
        prop_assert!((norm_scaled - expected).abs() < 1e-2);
    }
}

// Property test: Cauchy-Schwarz inequality |a·b| <= ||a|| * ||b||
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_cauchy_schwarz(
        a in prop::collection::vec(-100.0f32..100.0, 1..50),
        b in prop::collection::vec(-100.0f32..100.0, 1..50)
    ) {
        let len = a.len().min(b.len());
        let a_vec: Vec<f32> = a.into_iter().take(len).collect();
        let b_vec: Vec<f32> = b.into_iter().take(len).collect();

        let va = Vector::from_slice(&a_vec);
        let vb = Vector::from_slice(&b_vec);

        let dot_ab = va.dot(&vb).unwrap().abs();
        let norm_a = va.dot(&va).unwrap().sqrt();
        let norm_b = vb.dot(&vb).unwrap().sqrt();

        // |a·b| <= ||a|| * ||b||
        // Add small tolerance for floating point
        prop_assert!(dot_ab <= norm_a * norm_b + 1e-3);
    }
}

// Property test: Scaling property (multiply all by same constant)
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_scalar_multiplication(
        a in prop::collection::vec(-100.0f32..100.0, 1..100),
        scalar in -10.0f32..10.0
    ) {
        let va = Vector::from_slice(&a);

        // Create vector of all same scalar
        let scalars = vec![scalar; a.len()];
        let vs = Vector::from_slice(&scalars);

        let result = va.mul(&vs).unwrap();

        // Each element should be a[i] * scalar
        for (i, &val) in result.as_slice().iter().enumerate() {
            let expected = a[i] * scalar;
            prop_assert!((val - expected).abs() < 1e-3);
        }
    }
}

// Property test: Sum of scaled vector = scale * sum
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_sum_linearity(
        a in prop::collection::vec(-100.0f32..100.0, 1..100),
        scalar in -10.0f32..10.0
    ) {
        let va = Vector::from_slice(&a);

        // Create scaled version
        let scalars = vec![scalar; a.len()];
        let vs = Vector::from_slice(&scalars);
        let scaled = va.mul(&vs).unwrap();

        let sum_scaled = scaled.sum().unwrap();
        let sum_original = va.sum().unwrap();

        // sum(scalar * v) = scalar * sum(v)
        let expected = scalar * sum_original;
        prop_assert!((sum_scaled - expected).abs() < 1e-2);
    }
}

// Property test: Normalized vector has unit norm
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_normalize_unit_norm(
        a in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        // Skip if vector is too close to zero (would cause division by zero)
        let norm_squared: f32 = a.iter().map(|x| x * x).sum();
        prop_assume!(norm_squared > 1e-6);

        let va = Vector::from_slice(&a);
        let normalized = va.normalize().unwrap();

        // The normalized vector should have L2 norm = 1
        let norm = normalized.norm_l2().unwrap();
        prop_assert!((norm - 1.0).abs() < 1e-4, "norm = {}, expected 1.0", norm);
    }
}

// Property test: Normalization preserves direction (scaling invariance)
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_normalize_direction_invariant(
        a in prop::collection::vec(-100.0f32..100.0, 1..100),
        scale in 0.1f32..10.0
    ) {
        // Skip if vector is too close to zero
        let norm_squared: f32 = a.iter().map(|x| x * x).sum();
        prop_assume!(norm_squared > 1e-6);

        let va = Vector::from_slice(&a);

        // Scale the vector
        let scales = vec![scale; a.len()];
        let vs = Vector::from_slice(&scales);
        let scaled = va.mul(&vs).unwrap();

        // Both should normalize to the same direction
        let norm_a = va.normalize().unwrap();
        let norm_scaled = scaled.normalize().unwrap();

        // Check each element is close
        for (i, (&val_a, &val_scaled)) in norm_a.as_slice().iter()
            .zip(norm_scaled.as_slice().iter())
            .enumerate() {
            prop_assert!(
                (val_a - val_scaled).abs() < 1e-4,
                "Element {} differs: {} vs {}", i, val_a, val_scaled
            );
        }
    }
}

// Property test: L1 norm triangle inequality
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_norm_l1_triangle_inequality(
        len in 1usize..100,
        a in prop::collection::vec(-100.0f32..100.0, 1..100),
        b in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        // Triangle inequality: ||a + b||₁ <= ||a||₁ + ||b||₁
        // Use same length for both vectors
        let actual_len = len.min(a.len()).min(b.len());
        let a_trimmed = &a[..actual_len];
        let b_trimmed = &b[..actual_len];

        let va = Vector::from_slice(a_trimmed);
        let vb = Vector::from_slice(b_trimmed);

        let norm_a = va.norm_l1().unwrap();
        let norm_b = vb.norm_l1().unwrap();
        let sum = va.add(&vb).unwrap();
        let norm_sum = sum.norm_l1().unwrap();

        // Triangle inequality should hold
        prop_assert!(
            norm_sum <= norm_a + norm_b + 1e-3,
            "Triangle inequality violated: {} > {} + {}",
            norm_sum, norm_a, norm_b
        );
    }
}

// Property test: L1 norm absolute homogeneity
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_norm_l1_absolute_homogeneity(
        a in prop::collection::vec(-100.0f32..100.0, 1..100),
        scalar in -10.0f32..10.0
    ) {
        // Absolute homogeneity: ||c * v||₁ = |c| * ||v||₁
        let va = Vector::from_slice(&a);

        let norm_a = va.norm_l1().unwrap();

        // Scale the vector
        let scalars = vec![scalar; a.len()];
        let vs = Vector::from_slice(&scalars);
        let scaled = va.mul(&vs).unwrap();

        let norm_scaled = scaled.norm_l1().unwrap();

        // Should satisfy: ||c*v||₁ = |c| * ||v||₁
        let expected = scalar.abs() * norm_a;

        // Use relative tolerance for large values
        let tolerance = if expected.abs() > 1.0 {
            expected.abs() * 1e-5 // Relative tolerance
        } else {
            1e-2 // Absolute tolerance for small values
        };

        prop_assert!(
            (norm_scaled - expected).abs() < tolerance,
            "Homogeneity violated: {} != |{}| * {} = {}, diff = {}",
            norm_scaled, scalar, norm_a, expected, (norm_scaled - expected).abs()
        );
    }
}

// Property test: L1 norm equals sum of absolute values
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_norm_l1_definition(
        a in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let norm = va.norm_l1().unwrap();

        // Manual calculation of sum(|a[i]|)
        let manual_sum: f32 = a.iter().map(|x| x.abs()).sum();

        // Use relative tolerance for large values
        let tolerance = if manual_sum.abs() > 1.0 {
            manual_sum.abs() * 1e-5 // Relative tolerance
        } else {
            1e-3 // Absolute tolerance for small values
        };

        prop_assert!(
            (norm - manual_sum).abs() < tolerance,
            "L1 norm {} != manual sum {}, diff = {}",
            norm, manual_sum, (norm - manual_sum).abs()
        );
    }
}

// Property test: L∞ norm absolute homogeneity
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_norm_linf_absolute_homogeneity(
        a in prop::collection::vec(-100.0f32..100.0, 1..100),
        scalar in -10.0f32..10.0
    ) {
        prop_assume!(!a.is_empty());

        // Absolute homogeneity: ||c * v||∞ = |c| * ||v||∞
        let va = Vector::from_slice(&a);
        let norm_a = va.norm_linf().unwrap();

        // Scale the vector
        let scalars = vec![scalar; a.len()];
        let vs = Vector::from_slice(&scalars);
        let scaled = va.mul(&vs).unwrap();

        let norm_scaled = scaled.norm_linf().unwrap();

        // Should satisfy: ||c*v||∞ = |c| * ||v||∞
        let expected = scalar.abs() * norm_a;
        prop_assert!(
            (norm_scaled - expected).abs() < 1e-3,
            "Homogeneity violated: {} != |{}| * {} = {}",
            norm_scaled, scalar, norm_a, expected
        );
    }
}

// Property test: L∞ norm equals max of absolute values
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_norm_linf_definition(
        a in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        prop_assume!(!a.is_empty());

        let va = Vector::from_slice(&a);
        let norm = va.norm_linf().unwrap();

        // Manual calculation of max(|a[i]|)
        let manual_max = a.iter().map(|x| x.abs()).fold(0.0f32, f32::max);

        prop_assert!(
            (norm - manual_max).abs() < 1e-5,
            "L∞ norm {} != manual max {}",
            norm, manual_max
        );
    }
}

// Property test: L∞ norm submultiplicativity (Hölder's inequality special case)
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_norm_linf_submultiplicative(
        len in 1usize..100,
        a in prop::collection::vec(-100.0f32..100.0, 1..100),
        b in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        // For element-wise multiplication: ||a ⊙ b||∞ <= ||a||∞ * ||b||∞
        let actual_len = len.min(a.len()).min(b.len());
        let a_trimmed = &a[..actual_len];
        let b_trimmed = &b[..actual_len];

        let va = Vector::from_slice(a_trimmed);
        let vb = Vector::from_slice(b_trimmed);

        let norm_a = va.norm_linf().unwrap();
        let norm_b = vb.norm_linf().unwrap();
        let product = va.mul(&vb).unwrap();
        let norm_product = product.norm_linf().unwrap();

        // Submultiplicativity should hold
        prop_assert!(
            norm_product <= norm_a * norm_b + 1e-3,
            "Submultiplicativity violated: {} > {} * {}",
            norm_product, norm_a, norm_b
        );
    }
}

// Property test: abs() idempotence
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_abs_idempotent(
        a in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        // abs(abs(v)) = abs(v) - applying twice should be same as once
        let va = Vector::from_slice(&a);
        let abs_once = va.abs().unwrap();
        let abs_twice = abs_once.abs().unwrap();

        for (i, (&val_once, &val_twice)) in abs_once.as_slice().iter()
            .zip(abs_twice.as_slice().iter())
            .enumerate() {
            prop_assert!(
                (val_once - val_twice).abs() < 1e-5,
                "Idempotence failed at {}: {} != {}",
                i, val_once, val_twice
            );
        }
    }
}

// Property test: abs() is always non-negative
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_abs_non_negative(
        a in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.abs().unwrap();

        for (i, &val) in result.as_slice().iter().enumerate() {
            prop_assert!(
                val >= 0.0,
                "Negative value at {}: {}",
                i, val
            );
        }
    }
}

// Property test: abs() correctness
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_abs_correctness(
        a in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.abs().unwrap();

        for (i, (&input, &output)) in a.iter()
            .zip(result.as_slice().iter())
            .enumerate() {
            let expected = input.abs();
            prop_assert!(
                (output - expected).abs() < 1e-5,
                "Incorrect abs at {}: {} -> {}, expected {}",
                i, input, output, expected
            );
        }
    }
}
