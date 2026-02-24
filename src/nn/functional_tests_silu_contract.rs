// =========================================================================
// FALSIFY-SI: silu-kernel-v1.yaml contract (aprender functional::silu)
//
// Five-Whys (PMAT-354):
//   Why 1: aprender had proptest SiLU tests but zero inline FALSIFY-SI-* tests
//   Why 2: proptests live in tests/contracts/, not near the implementation
//   Why 3: no mapping from silu-kernel-v1.yaml to inline test names
//   Why 4: aprender predates the inline FALSIFY convention
//   Why 5: SiLU was "obviously correct" (delegates to trueno::silu_scalar)
//
// References:
//   - provable-contracts/contracts/silu-kernel-v1.yaml
//   - Ramachandran et al. (2017) "Searching for Activation Functions"
// =========================================================================

use super::*;

/// FALSIFY-SI-001: Zero preservation — SiLU(0) = 0
#[test]
fn falsify_si_001_zero_preservation() {
    let result = silu_scalar(0.0);
    assert!(
        result.abs() < 1e-7,
        "FALSIFIED SI-001: SiLU(0) = {result}, expected 0"
    );

    let t = silu(&Tensor::new(&[0.0], &[1]));
    assert!(
        t.data()[0].abs() < 1e-7,
        "FALSIFIED SI-001: SiLU tensor(0) = {}",
        t.data()[0]
    );
}

/// FALSIFY-SI-002: Global lower bound — SiLU(x) > -0.279 for all x
///
/// The global minimum of SiLU is approximately -0.2784 at x ≈ -1.278.
#[test]
fn falsify_si_002_global_lower_bound() {
    let test_values: Vec<f32> = vec![
        -100.0, -50.0, -10.0, -5.0, -2.0, -1.278, -1.0, -0.5, 0.0, 0.5, 1.0, 5.0, 100.0,
    ];

    for &x in &test_values {
        let y = silu_scalar(x);
        assert!(
            y > -0.28,
            "FALSIFIED SI-002: SiLU({x}) = {y}, expected > -0.279"
        );
    }
}

/// FALSIFY-SI-003: Monotonic for positive inputs — x > y > 0 ⟹ SiLU(x) > SiLU(y)
#[test]
fn falsify_si_003_monotonic_positive() {
    let values: Vec<f32> = vec![0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0];

    for i in 1..values.len() {
        let y_prev = silu_scalar(values[i - 1]);
        let y_curr = silu_scalar(values[i]);
        assert!(
            y_curr > y_prev,
            "FALSIFIED SI-003: SiLU({}) = {y_curr} not > SiLU({}) = {y_prev}",
            values[i],
            values[i - 1]
        );
    }
}

/// FALSIFY-SI-004: Asymptotic linearity — |SiLU(x) - x| < 0.01 for x > 10
#[test]
fn falsify_si_004_asymptotic_linearity() {
    for &x in &[10.0f32, 20.0, 50.0, 100.0, 500.0] {
        let y = silu_scalar(x);
        assert!(
            (y - x).abs() < 0.01,
            "FALSIFIED SI-004: |SiLU({x}) - {x}| = {} >= 0.01",
            (y - x).abs()
        );
    }
}

/// FALSIFY-SI-005: Tensor API matches scalar API element-wise
#[test]
fn falsify_si_005_tensor_scalar_equivalence() {
    let values = vec![-5.0, -1.278, -0.5, 0.0, 0.5, 1.0, 5.0, 10.0];
    let t = silu(&Tensor::new(&values, &[values.len()]));

    for (i, &x) in values.iter().enumerate() {
        let expected = silu_scalar(x);
        assert!(
            (t.data()[i] - expected).abs() < 1e-7,
            "FALSIFIED SI-005: tensor[{i}]={} != scalar({x})={expected}",
            t.data()[i]
        );
    }
}

mod silu_proptest_falsify {
    use super::*;
    use proptest::prelude::*;

    /// FALSIFY-SI-002-prop: SiLU(x) > -0.28 for random x
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(50))]

        #[test]
        fn falsify_si_002_prop_lower_bound(
            x in -100.0f32..100.0,
        ) {
            let y = silu_scalar(x);
            prop_assert!(
                y > -0.28,
                "FALSIFIED SI-002-prop: SiLU({})={} <= -0.28",
                x, y
            );
        }
    }

    /// FALSIFY-SI-005-prop: Tensor matches scalar for random values
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(30))]

        #[test]
        fn falsify_si_005_prop_tensor_scalar(
            x in -50.0f32..50.0,
        ) {
            let t = silu(&Tensor::new(&[x], &[1]));
            let expected = silu_scalar(x);
            prop_assert!(
                (t.data()[0] - expected).abs() < 1e-6,
                "FALSIFIED SI-005-prop: tensor({})={} != scalar={}",
                x, t.data()[0], expected
            );
        }
    }
}
