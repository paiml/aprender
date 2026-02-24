// =========================================================================
// FALSIFY-VE: Vector primitives contract (aprender primitives)
//
// Five-Whys (PMAT-354):
//   Why 1: aprender had no inline FALSIFY-VE-* tests for Vector
//   Why 2: vector tests exist but lack contract-mapped FALSIFY naming
//   Why 3: no YAML contract for vector primitives yet
//   Why 4: aprender predates the inline FALSIFY convention
//   Why 5: Vector arithmetic was "obviously correct" (basic operations)
//
// References:
//   - Cauchy-Schwarz inequality: |dot(u,v)| <= norm(u) * norm(v)
// =========================================================================

use super::*;

/// FALSIFY-VE-001: Dot product is commutative: dot(u,v) = dot(v,u)
#[test]
fn falsify_ve_001_dot_commutative() {
    let u = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let v = Vector::from_slice(&[4.0, 5.0, 6.0]);

    let uv = u.dot(&v);
    let vu = v.dot(&u);

    assert!(
        (uv - vu).abs() < 1e-6,
        "FALSIFIED VE-001: dot(u,v)={uv} != dot(v,u)={vu}"
    );
}

/// FALSIFY-VE-002: Norm is non-negative
#[test]
fn falsify_ve_002_norm_nonneg() {
    let v = Vector::from_slice(&[-3.0, 4.0]);
    let n = v.norm();

    assert!(n >= 0.0, "FALSIFIED VE-002: norm={n}, expected >= 0.0");
    assert!(
        (n - 5.0).abs() < 1e-5,
        "FALSIFIED VE-002: norm of [-3,4]={n}, expected 5.0"
    );
}

/// FALSIFY-VE-003: Cauchy-Schwarz: |dot(u,v)| <= norm(u) * norm(v)
#[test]
fn falsify_ve_003_cauchy_schwarz() {
    let u = Vector::from_slice(&[1.0, -2.0, 3.0, 0.5]);
    let v = Vector::from_slice(&[4.0, 0.0, -1.0, 2.0]);

    let dot = u.dot(&v).abs();
    let bound = u.norm() * v.norm();

    assert!(
        dot <= bound + 1e-5,
        "FALSIFIED VE-003: |dot|={dot} > norm(u)*norm(v)={bound}"
    );
}

/// FALSIFY-VE-004: Mean equals sum / length
#[test]
fn falsify_ve_004_mean_equals_sum_over_len() {
    let v = Vector::from_slice(&[2.0, 4.0, 6.0, 8.0, 10.0]);

    let mean = v.mean();
    let expected = v.sum() / v.len() as f32;

    assert!(
        (mean - expected).abs() < 1e-6,
        "FALSIFIED VE-004: mean={mean}, expected sum/len={expected}"
    );
    assert!(
        (mean - 6.0).abs() < 1e-6,
        "FALSIFIED VE-004: mean={mean}, expected 6.0"
    );
}

mod vector_proptest_falsify {
    use super::*;
    use proptest::prelude::*;

    /// FALSIFY-VE-001-prop: Dot product commutativity for random vectors
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(50))]

        #[test]
        fn falsify_ve_001_prop_dot_commutative(
            seed in 0..1000u32,
            n in 2..=16usize,
        ) {
            let u_data: Vec<f32> = (0..n)
                .map(|i| ((i as f32 + seed as f32) * 0.37).sin() * 10.0)
                .collect();
            let v_data: Vec<f32> = (0..n)
                .map(|i| ((i as f32 + seed as f32 + 50.0) * 0.53).cos() * 10.0)
                .collect();
            let u = Vector::from_vec(u_data);
            let v = Vector::from_vec(v_data);

            let uv = u.dot(&v);
            let vu = v.dot(&u);
            prop_assert!(
                (uv - vu).abs() < 1e-3,
                "FALSIFIED VE-001-prop: dot(u,v)={} != dot(v,u)={}",
                uv, vu
            );
        }
    }

    /// FALSIFY-VE-003-prop: Cauchy-Schwarz for random vectors
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(50))]

        #[test]
        fn falsify_ve_003_prop_cauchy_schwarz(
            seed in 0..1000u32,
            n in 2..=16usize,
        ) {
            let u_data: Vec<f32> = (0..n)
                .map(|i| ((i as f32 + seed as f32) * 0.37).sin() * 10.0)
                .collect();
            let v_data: Vec<f32> = (0..n)
                .map(|i| ((i as f32 + seed as f32 + 50.0) * 0.53).cos() * 10.0)
                .collect();
            let u = Vector::from_vec(u_data);
            let v = Vector::from_vec(v_data);

            let dot = u.dot(&v).abs();
            let bound = u.norm() * v.norm();
            prop_assert!(
                dot <= bound + 1e-2,
                "FALSIFIED VE-003-prop: |dot|={} > norm(u)*norm(v)={}",
                dot, bound
            );
        }
    }
}
