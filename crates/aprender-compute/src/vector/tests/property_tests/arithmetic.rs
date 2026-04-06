use super::super::super::*;
use proptest::prelude::*;

// Property test: Addition is commutative (a + b == b + a)
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_add_commutative(
        a in prop::collection::vec(-1000.0f32..1000.0, 1..100),
        b in prop::collection::vec(-1000.0f32..1000.0, 1..100)
    ) {
        // Use minimum length to ensure both vectors have same size
        let len = a.len().min(b.len());
        let a_vec: Vec<f32> = a.into_iter().take(len).collect();
        let b_vec: Vec<f32> = b.into_iter().take(len).collect();

        let va = Vector::from_slice(&a_vec);
        let vb = Vector::from_slice(&b_vec);

        let result1 = va.add(&vb).unwrap();
        let result2 = vb.add(&va).unwrap();

        prop_assert_eq!(result1.as_slice(), result2.as_slice());
    }
}

// Property test: Addition is associative ((a + b) + c == a + (b + c))
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_add_associative(
        a in prop::collection::vec(-100.0f32..100.0, 1..50),
        b in prop::collection::vec(-100.0f32..100.0, 1..50),
        c in prop::collection::vec(-100.0f32..100.0, 1..50)
    ) {
        let len = a.len().min(b.len()).min(c.len());
        let a_vec: Vec<f32> = a.into_iter().take(len).collect();
        let b_vec: Vec<f32> = b.into_iter().take(len).collect();
        let c_vec: Vec<f32> = c.into_iter().take(len).collect();

        let va = Vector::from_slice(&a_vec);
        let vb = Vector::from_slice(&b_vec);
        let vc = Vector::from_slice(&c_vec);

        let ab = va.add(&vb).unwrap();
        let abc = ab.add(&vc).unwrap();

        let bc = vb.add(&vc).unwrap();
        let a_bc = va.add(&bc).unwrap();

        // Use approximate equality for floating point (relaxed for associativity)
        for (x, y) in abc.as_slice().iter().zip(a_bc.as_slice()) {
            prop_assert!((x - y).abs() < 1e-4);
        }
    }
}

// Property test: Subtraction anti-commutativity (a - b == -(b - a))
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_sub_anti_commutative(
        a in prop::collection::vec(-1000.0f32..1000.0, 1..100),
        b in prop::collection::vec(-1000.0f32..1000.0, 1..100)
    ) {
        let len = a.len().min(b.len());
        let a_vec: Vec<f32> = a.into_iter().take(len).collect();
        let b_vec: Vec<f32> = b.into_iter().take(len).collect();

        let va = Vector::from_slice(&a_vec);
        let vb = Vector::from_slice(&b_vec);

        let result1 = va.sub(&vb).unwrap();
        let result2 = vb.sub(&va).unwrap();

        // a - b should equal -(b - a)
        for (x, y) in result1.as_slice().iter().zip(result2.as_slice()) {
            prop_assert!((x + y).abs() < 1e-5);
        }
    }
}

// Property test: Subtraction identity (a - 0 == a)
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_sub_identity(
        a in prop::collection::vec(-1000.0f32..1000.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let zero = Vector::from_slice(&vec![0.0; a.len()]);

        let result = va.sub(&zero).unwrap();

        prop_assert_eq!(result.as_slice(), va.as_slice());
    }
}

// Property test: Subtraction inverse (a - a == 0)
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_sub_inverse(
        a in prop::collection::vec(-1000.0f32..1000.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);

        let result = va.sub(&va).unwrap();

        // All elements should be zero (or very close due to floating point)
        for &x in result.as_slice() {
            prop_assert!(x.abs() < 1e-5);
        }
    }
}

// Property test: Multiplication is commutative
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_mul_commutative(
        a in prop::collection::vec(-100.0f32..100.0, 1..100),
        b in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        let len = a.len().min(b.len());
        let a_vec: Vec<f32> = a.into_iter().take(len).collect();
        let b_vec: Vec<f32> = b.into_iter().take(len).collect();

        let va = Vector::from_slice(&a_vec);
        let vb = Vector::from_slice(&b_vec);

        let result1 = va.mul(&vb).unwrap();
        let result2 = vb.mul(&va).unwrap();

        prop_assert_eq!(result1.as_slice(), result2.as_slice());
    }
}

// Property test: Division identity (a / 1 == a)
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_div_identity(
        a in prop::collection::vec(-1000.0f32..1000.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let ones = Vector::from_slice(&vec![1.0; a.len()]);

        let result = va.div(&ones).unwrap();

        for (x, y) in result.as_slice().iter().zip(va.as_slice()) {
            prop_assert!((x - y).abs() < 1e-5);
        }
    }
}

// Property test: Division inverse (a / a == 1, for non-zero a)
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_div_inverse(
        a in prop::collection::vec(-1000.0f32..1000.0, 1..100)
    ) {
        // Filter out zeros to avoid division edge cases
        let a_nonzero: Vec<f32> = a.into_iter()
            .map(|x| if x.abs() < 1e-5 { 1.0 } else { x })
            .collect();

        let va = Vector::from_slice(&a_nonzero);
        let result = va.div(&va).unwrap();

        // All elements should be 1.0 (or very close)
        for &x in result.as_slice() {
            prop_assert!((x - 1.0).abs() < 1e-4);
        }
    }
}

// Property test: Division-multiplication inverse (a / b) * b ≈ a
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_div_mul_inverse(
        a in prop::collection::vec(-1000.0f32..1000.0, 1..100),
        b in prop::collection::vec(-1000.0f32..1000.0, 1..100)
    ) {
        let len = a.len().min(b.len());
        let a_vec: Vec<f32> = a.into_iter().take(len).collect();

        // Filter out zeros from b to avoid division by zero edge cases
        let b_vec: Vec<f32> = b.into_iter().take(len)
            .map(|x| if x.abs() < 1e-3 { 1.0 } else { x })
            .collect();

        let va = Vector::from_slice(&a_vec);
        let vb = Vector::from_slice(&b_vec);

        let divided = va.div(&vb).unwrap();
        let restored = divided.mul(&vb).unwrap();

        // Restored should approximately equal original
        for (original, restored_val) in a_vec.iter().zip(restored.as_slice()) {
            prop_assert!((original - restored_val).abs() < 1e-2);
        }
    }
}

// Property test: Dot product is commutative
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_dot_commutative(
        a in prop::collection::vec(-100.0f32..100.0, 1..100),
        b in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        let len = a.len().min(b.len());
        let a_vec: Vec<f32> = a.into_iter().take(len).collect();
        let b_vec: Vec<f32> = b.into_iter().take(len).collect();

        let va = Vector::from_slice(&a_vec);
        let vb = Vector::from_slice(&b_vec);

        let result1 = va.dot(&vb).unwrap();
        let result2 = vb.dot(&va).unwrap();

        prop_assert!((result1 - result2).abs() < 1e-3);
    }
}

// Property test: Identity element for addition (a + 0 == a)
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_add_identity(
        a in prop::collection::vec(-1000.0f32..1000.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let zero = Vector::from_slice(&vec![0.0; a.len()]);

        let result = va.add(&zero).unwrap();

        prop_assert_eq!(result.as_slice(), va.as_slice());
    }
}

// Property test: Identity element for multiplication (a * 1 == a)
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_mul_identity(
        a in prop::collection::vec(-1000.0f32..1000.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let one = Vector::from_slice(&vec![1.0; a.len()]);

        let result = va.mul(&one).unwrap();

        for (x, y) in result.as_slice().iter().zip(va.as_slice()) {
            prop_assert!((x - y).abs() < 1e-5);
        }
    }
}

// Property test: Zero element for multiplication (a * 0 == 0)
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_mul_zero(
        a in prop::collection::vec(-1000.0f32..1000.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let zero = Vector::from_slice(&vec![0.0; a.len()]);

        let result = va.mul(&zero).unwrap();

        for x in result.as_slice() {
            prop_assert_eq!(*x, 0.0);
        }
    }
}

// Property test: Distributive property (a * (b + c) == a * b + a * c)
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_distributive(
        a in prop::collection::vec(-10.0f32..10.0, 1..50),
        b in prop::collection::vec(-10.0f32..10.0, 1..50),
        c in prop::collection::vec(-10.0f32..10.0, 1..50)
    ) {
        let len = a.len().min(b.len()).min(c.len());
        let a_vec: Vec<f32> = a.into_iter().take(len).collect();
        let b_vec: Vec<f32> = b.into_iter().take(len).collect();
        let c_vec: Vec<f32> = c.into_iter().take(len).collect();

        let va = Vector::from_slice(&a_vec);
        let vb = Vector::from_slice(&b_vec);
        let vc = Vector::from_slice(&c_vec);

        // a * (b + c)
        let bc = vb.add(&vc).unwrap();
        let left = va.mul(&bc).unwrap();

        // a * b + a * c
        let ab = va.mul(&vb).unwrap();
        let ac = va.mul(&vc).unwrap();
        let right = ab.add(&ac).unwrap();

        for (x, y) in left.as_slice().iter().zip(right.as_slice()) {
            prop_assert!((x - y).abs() < 1e-3);
        }
    }
}

// Property test: Sum is consistent
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_sum_matches_manual(
        a in prop::collection::vec(-1000.0f32..1000.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.sum().unwrap();
        let manual_sum: f32 = a.iter().sum();

        // Relaxed tolerance for SIMD vs scalar accumulation order differences
        prop_assert!((result - manual_sum).abs() < 1e-2);
    }

    #[test]
    fn test_sum_kahan_correctness(
        a in prop::collection::vec(-1000.0f32..1000.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let kahan_result = va.sum_kahan().unwrap();
        let manual_sum: f32 = a.iter().sum();

        // Kahan result should be close to manual sum
        // Note: Both use same algorithm (iter().sum() also uses compensated summation)
        // so they should match closely
        prop_assert!((kahan_result - manual_sum).abs() < 1e-2,
            "Kahan sum should match manual sum closely");

        // Verify Kahan produces a reasonable result
        let expected_magnitude = a.iter().map(|x| x.abs()).sum::<f32>();
        prop_assert!(kahan_result.abs() <= expected_magnitude + 1.0,
            "Kahan result magnitude should be reasonable");
    }
}

// Property test: Max is actually maximum
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_max_is_maximum(
        a in prop::collection::vec(-1000.0f32..1000.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.max().unwrap();

        // Verify result is >= all elements
        for &x in a.iter() {
            prop_assert!(result >= x);
        }

        // Verify result is actually in the vector
        prop_assert!(a.contains(&result));
    }

    #[test]
    fn test_min_is_minimum(
        a in prop::collection::vec(-1000.0f32..1000.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.min().unwrap();

        // Verify result is <= all elements
        for &x in a.iter() {
            prop_assert!(result <= x);
        }

        // Verify result is actually in the vector
        prop_assert!(a.contains(&result));
    }

    #[test]
    fn test_argmax_correctness(
        a in prop::collection::vec(-1000.0f32..1000.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let idx = va.argmax().unwrap();

        // Verify index is in bounds
        prop_assert!(idx < a.len());

        // Verify value at index is >= all other values
        let max_val = a[idx];
        for &x in a.iter() {
            prop_assert!(max_val >= x);
        }

        // Verify it's the first occurrence (no earlier index has this value)
        for &val in a.iter().take(idx) {
            prop_assert!(val < max_val || val != max_val);
        }
    }

    #[test]
    fn test_argmin_correctness(
        a in prop::collection::vec(-1000.0f32..1000.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let idx = va.argmin().unwrap();

        // Verify index is in bounds
        prop_assert!(idx < a.len());

        // Verify value at index is <= all other values
        let min_val = a[idx];
        for &x in a.iter() {
            prop_assert!(min_val <= x);
        }

        // Verify it's the first occurrence (no earlier index has this value)
        for &val in a.iter().take(idx) {
            prop_assert!(val > min_val || val != min_val);
        }
    }
}
