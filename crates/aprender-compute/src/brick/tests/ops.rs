use super::super::*;

#[test]
fn test_token_budget_from_latency() {
    let budget = TokenBudget::from_latency(50.0);
    assert!((budget.us_per_token - 50.0).abs() < 0.001);
    assert!((budget.tokens_per_sec - 20_000.0).abs() < 1.0);
}

#[test]
fn test_token_budget_from_throughput() {
    let budget = TokenBudget::from_throughput(20_000.0);
    assert!((budget.us_per_token - 50.0).abs() < 0.001);
    assert!((budget.tokens_per_sec - 20_000.0).abs() < 1.0);
}

#[test]
fn test_token_budget_is_met() {
    let budget = TokenBudget::from_latency(50.0);
    assert!(budget.is_met(40.0)); // Under budget
    assert!(budget.is_met(50.0)); // Exactly at budget
    assert!(!budget.is_met(60.0)); // Over budget
}

#[test]
fn test_dot_op() {
    let op = DotOp::new(4);
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![5.0, 6.0, 7.0, 8.0];
    let result = op.execute((a, b), Backend::Scalar).unwrap();
    assert!((result - 70.0).abs() < 0.001); // 1*5 + 2*6 + 3*7 + 4*8 = 70
}

#[test]
fn test_add_op() {
    let op = AddOp::new(4);
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![5.0, 6.0, 7.0, 8.0];
    let result = op.execute((a, b), Backend::Scalar).unwrap();
    assert_eq!(result, vec![6.0, 8.0, 10.0, 12.0]);
}

#[test]
fn test_matmul_op() {
    let op = MatmulOp::new(2, 2, 2);
    // A = [[1, 2], [3, 4]]
    let a = vec![1.0, 2.0, 3.0, 4.0];
    // B = [[5, 6], [7, 8]]
    let b = vec![5.0, 6.0, 7.0, 8.0];
    let result = op.execute((a, b), Backend::Scalar).unwrap();
    // C = [[19, 22], [43, 50]]
    assert_eq!(result, vec![19.0, 22.0, 43.0, 50.0]);
}

// =========================================================================
// FALSIFY-MM: matmul-kernel-v1.yaml contract (trueno MatmulOp)
//
// Five-Whys (PMAT-354):
//   Why 1: trueno had 1 matmul unit test but zero FALSIFY-MM-* tests
//   Why 2: unit tests verify one 2x2 case, not mathematical invariants
//   Why 3: no mapping from matmul-kernel-v1.yaml to trueno test names
//   Why 4: trueno predates the provable-contracts YAML convention
//   Why 5: matmul was "obviously correct" (textbook GEMM)
//
// References:
//   - provable-contracts/contracts/matmul-kernel-v1.yaml
// =========================================================================

/// FALSIFY-MM-001: Shape correctness — output is [m, n]
#[test]
fn falsify_mm_001_shape_correctness() {
    for (m, k, n) in [(2, 3, 4), (1, 5, 1), (4, 4, 4), (3, 1, 2)] {
        let op = MatmulOp::new(m, k, n);
        let a = vec![1.0; m * k];
        let b = vec![1.0; k * n];
        let result = op.execute((a, b), Backend::Scalar).unwrap();
        assert_eq!(
            result.len(),
            m * n,
            "FALSIFIED MM-001: output len = {}, expected {} for ({m}x{k}) @ ({k}x{n})",
            result.len(),
            m * n
        );
    }
}

/// FALSIFY-MM-002: Numerical accuracy — known result for 2x3 @ 3x2
#[test]
fn falsify_mm_002_numerical_accuracy() {
    let op = MatmulOp::new(2, 3, 2);
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
    let result = op.execute((a, b), Backend::Scalar).unwrap();
    let expected = [58.0, 64.0, 139.0, 154.0];
    for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
        assert!((got - exp).abs() < 1e-4, "FALSIFIED MM-002: result[{i}] = {got}, expected {exp}");
    }
}

/// FALSIFY-MM-005: Identity matrix — A @ I = A and I @ B = B
#[test]
fn falsify_mm_005_identity_matrix() {
    // Test A @ I = A
    let m = 3;
    let k = 4;
    let op = MatmulOp::new(m, k, k);
    let a: Vec<f32> = (0..m * k).map(|i| (i as f32 + 1.0) * 0.5).collect();
    let mut identity = vec![0.0; k * k];
    for i in 0..k {
        identity[i * k + i] = 1.0;
    }
    let result = op.execute((a.clone(), identity), Backend::Scalar).unwrap();
    for (i, (&got, &exp)) in result.iter().zip(a.iter()).enumerate() {
        assert!((got - exp).abs() < 1e-5, "FALSIFIED MM-005: (A@I)[{i}] = {got}, expected {exp}");
    }

    // Test I @ B = B
    let op2 = MatmulOp::new(k, k, m);
    let b: Vec<f32> = (0..k * m).map(|i| (i as f32 + 1.0) * 0.3).collect();
    let mut identity2 = vec![0.0; k * k];
    for i in 0..k {
        identity2[i * k + i] = 1.0;
    }
    let result2 = op2.execute((identity2, b.clone()), Backend::Scalar).unwrap();
    for (i, (&got, &exp)) in result2.iter().zip(b.iter()).enumerate() {
        assert!((got - exp).abs() < 1e-5, "FALSIFIED MM-005: (I@B)[{i}] = {got}, expected {exp}");
    }
}

mod mm_proptest_falsify {
    use super::super::super::*;
    use proptest::prelude::*;

    // FALSIFY-MM-001-prop: Shape correctness for random dimensions
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn falsify_mm_001_prop_shape(
            m in 1..=8usize,
            k in 1..=8usize,
            n in 1..=8usize,
        ) {
            let op = MatmulOp::new(m, k, n);
            let a = vec![1.0; m * k];
            let b = vec![1.0; k * n];
            let result = op.execute((a, b), Backend::Scalar).unwrap();
            prop_assert_eq!(result.len(), m * n);
        }
    }

    // FALSIFY-MM-005-prop: Identity matrix for random dims
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(50))]

        #[test]
        fn falsify_mm_005_prop_identity(
            m in 1..=6usize,
            k in 1..=6usize,
            seed in 0..500u32,
        ) {
            let op = MatmulOp::new(m, k, k);
            let a: Vec<f32> = (0..m * k)
                .map(|i| ((i as f32 + seed as f32) * 0.37).sin())
                .collect();
            let mut identity = vec![0.0; k * k];
            for i in 0..k {
                identity[i * k + i] = 1.0;
            }
            let result = op.execute((a.clone(), identity), Backend::Scalar).unwrap();
            for (i, (&got, &exp)) in result.iter().zip(a.iter()).enumerate() {
                prop_assert!(
                    (got - exp).abs() < 1e-4,
                    "FALSIFIED MM-005-prop: (A@I)[{}] = {}, expected {}",
                    i, got, exp
                );
            }
        }
    }
}

#[test]
fn test_softmax_op() {
    let op = SoftmaxOp::new(3);
    let input = vec![1.0, 2.0, 3.0];
    let result = op.execute(input, Backend::Scalar).unwrap();
    // Sum should be 1.0
    let sum: f32 = result.iter().sum();
    assert!((sum - 1.0).abs() < 0.001);
    // Values should be increasing
    assert!(result[0] < result[1]);
    assert!(result[1] < result[2]);
}
