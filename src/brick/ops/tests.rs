use super::*;

#[test]
fn test_dot_op() {
    let op = DotOp::new(4);
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![5.0, 6.0, 7.0, 8.0];
    let result = op.execute((a, b), Backend::Scalar).unwrap();
    assert!((result - 70.0).abs() < 0.001); // 1*5 + 2*6 + 3*7 + 4*8 = 70
}

#[test]
fn test_dot_op_mismatch() {
    let op = DotOp::new(4);
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![4.0, 5.0, 6.0, 7.0];
    assert!(op.execute((a, b), Backend::Scalar).is_err());
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
fn test_add_op_mismatch() {
    let op = AddOp::new(4);
    let a = vec![1.0, 2.0];
    let b = vec![3.0, 4.0, 5.0];
    assert!(op.execute((a, b), Backend::Scalar).is_err());
}

#[test]
fn test_matmul_op() {
    // 2x3 * 3x2 = 2x2
    let op = MatmulOp::new(2, 3, 2);
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
    let result = op.execute((a, b), Backend::Scalar).unwrap();

    // Expected: [[1*7+2*9+3*11, 1*8+2*10+3*12], [4*7+5*9+6*11, 4*8+5*10+6*12]]
    // = [[7+18+33, 8+20+36], [28+45+66, 32+50+72]]
    // = [[58, 64], [139, 154]]
    assert!((result[0] - 58.0).abs() < 0.001);
    assert!((result[1] - 64.0).abs() < 0.001);
    assert!((result[2] - 139.0).abs() < 0.001);
    assert!((result[3] - 154.0).abs() < 0.001);
}

#[test]
fn test_softmax_op() {
    let op = SoftmaxOp::new(4);
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let result = op.execute(input, Backend::Scalar).unwrap();

    // Check sum = 1
    let sum: f32 = result.iter().sum();
    assert!((sum - 1.0).abs() < 0.001);

    // Check monotonicity
    assert!(result[0] < result[1]);
    assert!(result[1] < result[2]);
    assert!(result[2] < result[3]);
}

#[test]
fn test_softmax_op_empty() {
    let op = SoftmaxOp::new(0);
    let result = op.execute(vec![], Backend::Scalar).unwrap();
    assert!(result.is_empty());
}

#[test]
fn test_softmax_numerical_stability() {
    let op = SoftmaxOp::new(3);
    // Large values that would overflow naive exp
    let input = vec![1000.0, 1001.0, 1002.0];
    let result = op.execute(input, Backend::Scalar).unwrap();

    // Should still be valid probabilities
    let sum: f32 = result.iter().sum();
    assert!((sum - 1.0).abs() < 0.001);
    assert!(result.iter().all(|&x| x.is_finite()));
}

/// FALSIFICATION: Dot product is commutative
#[test]
fn test_falsify_dot_commutative() {
    let op = DotOp::new(5);
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let b = vec![5.0, 4.0, 3.0, 2.0, 1.0];

    let result1 = op.execute((a.clone(), b.clone()), Backend::Scalar).unwrap();
    let result2 = op.execute((b, a), Backend::Scalar).unwrap();

    assert!((result1 - result2).abs() < 1e-6, "FALSIFICATION FAILED: dot product not commutative");
}

/// FALSIFICATION: Softmax probabilities sum to 1
#[test]
fn test_falsify_softmax_sum_to_one() {
    for len in [1, 5, 10, 100] {
        let op = SoftmaxOp::new(len);
        let input: Vec<f32> = (0..len).map(|i| i as f32).collect();
        let result = op.execute(input, Backend::Scalar).unwrap();
        let sum: f32 = result.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "FALSIFICATION FAILED: softmax sum {} != 1.0 for len {}",
            sum,
            len
        );
    }
}
