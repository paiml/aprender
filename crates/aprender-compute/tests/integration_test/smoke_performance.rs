//! Smoke test and performance gate.

use trueno::{Matrix, Vector};

/// Smoke test: Every operation must execute without panic
#[test]
fn integration_smoke_test_all_operations() {
    let a = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let b = Vector::from_slice(&[5.0, 4.0, 3.0, 2.0, 1.0]);

    // Element-wise binary operations
    let _ = a.add(&b);
    let _ = a.sub(&b);
    let _ = a.mul(&b);
    let _ = a.div(&b);
    let _ = a.minimum(&b);
    let _ = a.maximum(&b);
    let _ = a.lerp(&b, 0.5);
    let _ = a.fma(&b, &a);

    // Element-wise unary operations
    let _ = a.abs();
    let _ = a.neg();
    let _ = a.sqrt();
    let _ = a.recip();
    let _ = a.pow(2.0);
    let _ = a.exp();
    let _ = a.ln();
    let _ = a.floor();
    let _ = a.ceil();
    let _ = a.round();
    let _ = a.trunc();
    let _ = a.fract();
    let _ = a.signum();

    // Trigonometric
    let _ = a.sin();
    let _ = a.cos();
    let _ = a.tan();
    let _ = a.sinh();
    let _ = a.cosh();
    let _ = a.tanh();
    let _ = a.asinh();

    // Reductions
    let _ = a.sum();
    let _ = a.sum_kahan();
    let _ = a.min();
    let _ = a.max();
    let _ = a.sum_of_squares();
    let _ = a.mean();
    let _ = a.variance();
    let _ = a.stddev();
    let _ = a.dot(&b);

    // Statistical
    let _ = a.covariance(&b);
    let _ = a.correlation(&b);

    // Activations
    let _ = a.relu();
    let _ = a.leaky_relu(0.01);
    let _ = a.elu(1.0);
    let _ = a.sigmoid();
    let _ = a.softmax();
    let _ = a.log_softmax();
    let _ = a.gelu();
    let _ = a.swish(); // Also known as SiLU

    // Preprocessing
    let _ = a.clip(-10.0, 10.0);
    let _ = a.minmax_normalize();
    let _ = a.zscore();

    // Norms
    let _ = a.norm_l1();
    let _ = a.norm_l2();
    let _ = a.norm_linf();
    let _ = a.normalize();

    // Index operations
    let _ = a.argmin();
    let _ = a.argmax();

    // Scalar operations
    let _ = a.scale(2.0);
    let _ = a.clamp(-10.0, 10.0);

    // Matrix operations
    let m1 = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let m2 = Matrix::identity(2);
    let _ = m1.matmul(&m2);
    let _ = m1.transpose();
    let _ = m1.rows();
    let _ = m1.cols();
    let _ = m1.shape();
    let _ = m1.get(0, 0);
    let _ = m1.as_slice();
}

// ============================================================================
// PERFORMANCE TEST (must complete under 30 seconds)
// ============================================================================

#[test]
fn integration_performance_gate() {
    use std::time::Instant;

    let start = Instant::now();

    // Run a subset of operations to verify performance
    for size in [100, 1000, 10000] {
        let data: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let v = Vector::from_slice(&data);

        let _ = v.sum();
        let _ = v.mean();
        let _ = v.dot(&v);
        let _ = v.add(&v);
        let _ = v.relu();
    }

    let elapsed = start.elapsed();

    // This specific test should be very fast (<1s)
    assert!(elapsed.as_secs() < 5, "Performance gate failed: took {:?}, expected <5s", elapsed);
}
