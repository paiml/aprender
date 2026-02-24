// =========================================================================
// FALSIFY-CV: conv1d-kernel-v1.yaml contract (aprender Conv1d)
//
// Five-Whys (PMAT-354):
//   Why 1: aprender had 15+ Conv1d tests but zero FALSIFY-CV-* tests
//   Why 2: unit tests verify shapes/forward, not mathematical invariants
//   Why 3: no mapping from conv1d-kernel-v1.yaml to aprender test names
//   Why 4: aprender predates the provable-contracts YAML convention
//   Why 5: Conv1d was "obviously correct" (standard sliding window)
//
// References:
//   - provable-contracts/contracts/conv1d-kernel-v1.yaml
//   - LeCun et al. (1998) "Gradient-based learning applied to document recognition"
// =========================================================================

use super::*;

/// FALSIFY-CV-001: Output shape — L_out = floor((L + 2*pad - K) / stride) + 1
#[test]
fn falsify_cv_001_output_shape() {
    let test_cases: Vec<(usize, usize, usize, usize, usize, usize, usize)> = vec![
        // (in_ch, out_ch, K, stride, pad, L_in, expected_L_out)
        (1, 1, 3, 1, 0, 10, 8),  // (10 - 3)/1 + 1 = 8
        (4, 8, 5, 1, 0, 20, 16), // (20 - 5)/1 + 1 = 16
        (4, 8, 3, 2, 0, 10, 4),  // (10 - 3)/2 + 1 = 4
        (4, 8, 3, 1, 1, 10, 10), // (10 + 2 - 3)/1 + 1 = 10 (same padding)
        (1, 1, 1, 1, 0, 5, 5),   // K=1: (5 - 1)/1 + 1 = 5
    ];

    for (i, &(ic, oc, k, s, p, l_in, l_out)) in test_cases.iter().enumerate() {
        let conv = Conv1d::with_options(ic, oc, k, s, p, true);
        let x = Tensor::ones(&[2, ic, l_in]);
        let y = conv.forward(&x);
        assert_eq!(
            y.shape(),
            &[2, oc, l_out],
            "FALSIFIED CV-001 case {i}: shape {:?}, expected [2, {oc}, {l_out}]",
            y.shape()
        );
    }
}

/// FALSIFY-CV-001b: Shape formula — parametric sweep of K, stride, padding
#[test]
fn falsify_cv_001b_shape_formula() {
    for k in [1, 3, 5] {
        for s in [1, 2] {
            for p in [0, 1] {
                let l_in = 20;
                let expected_l_out = (l_in + 2 * p - k) / s + 1;
                let conv = Conv1d::with_options(2, 4, k, s, p, false);
                let x = Tensor::ones(&[1, 2, l_in]);
                let y = conv.forward(&x);
                assert_eq!(
                    y.shape()[2],
                    expected_l_out,
                    "FALSIFIED CV-001b: K={k}, S={s}, P={p}: L_out={}, expected {expected_l_out}",
                    y.shape()[2]
                );
            }
        }
    }
}

/// FALSIFY-CV-005: K=1 equivalence — conv1d(K=1) is a pointwise transform
///
/// With kernel_size=1, output length == input length (no spatial reduction).
#[test]
fn falsify_cv_005_kernel_one() {
    let conv = Conv1d::new(4, 8, 1);
    let x = Tensor::ones(&[2, 4, 15]);
    let y = conv.forward(&x);

    // K=1 should preserve length: L_out = (L - 1)/1 + 1 = L
    assert_eq!(
        y.shape(),
        &[2, 8, 15],
        "FALSIFIED CV-005: K=1 should preserve spatial dimension"
    );
}

/// FALSIFY-CV-002: Linearity — conv(a*x) = a*conv(x) without bias
#[test]
fn falsify_cv_002_linearity_no_bias() {
    let conv = Conv1d::with_options(2, 3, 3, 1, 0, false);
    let x = Tensor::new(
        &(0..2 * 2 * 8)
            .map(|i| (i as f32 * 0.1).sin())
            .collect::<Vec<_>>(),
        &[2, 2, 8],
    );
    let y_base = conv.forward(&x);

    for &alpha in &[2.0_f32, 0.5, -1.0, 0.1] {
        let scaled_data: Vec<f32> = x.data().iter().map(|&v| v * alpha).collect();
        let x_scaled = Tensor::new(&scaled_data, x.shape());
        let y_scaled = conv.forward(&x_scaled);

        for (i, (&ys, &yb)) in y_scaled.data().iter().zip(y_base.data().iter()).enumerate() {
            let expected = alpha * yb;
            let diff = (ys - expected).abs();
            let tol = 1e-3 * expected.abs().max(1.0);
            assert!(
                diff < tol,
                "FALSIFIED CV-002: conv({alpha}*x)[{i}] = {ys}, expected {expected}, diff = {diff}"
            );
        }
    }
}

mod conv1d_proptest_falsify {
    use super::*;
    use proptest::prelude::*;

    /// FALSIFY-CV-001-prop: Output shape formula for random parameters
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(50))]

        #[test]
        fn falsify_cv_001_prop_output_shape(
            k in prop::sample::select(vec![1usize, 3, 5, 7]),
            s in prop::sample::select(vec![1usize, 2]),
            p in prop::sample::select(vec![0usize, 1]),
            l_in in 10..=50usize,
        ) {
            let in_ch = 2;
            let out_ch = 4;
            // Ensure valid: L + 2*p >= K
            if l_in + 2 * p >= k {
                let expected = (l_in + 2 * p - k) / s + 1;
                let conv = Conv1d::with_options(in_ch, out_ch, k, s, p, false);
                let x = Tensor::ones(&[1, in_ch, l_in]);
                let y = conv.forward(&x);

                prop_assert_eq!(
                    y.shape()[2],
                    expected,
                    "FALSIFIED CV-001-prop: K={}, S={}, P={}, L_in={}: L_out={}, expected={}",
                    k, s, p, l_in, y.shape()[2], expected
                );
            }
        }
    }

    /// FALSIFY-CV-005-prop: K=1 preserves spatial dimension for random lengths
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(30))]

        #[test]
        fn falsify_cv_005_prop_kernel_one(
            l_in in 1..=100usize,
        ) {
            let conv = Conv1d::new(2, 4, 1);
            let x = Tensor::ones(&[1, 2, l_in]);
            let y = conv.forward(&x);

            prop_assert_eq!(
                y.shape()[2],
                l_in,
                "FALSIFIED CV-005-prop: K=1, L_in={}, L_out={}",
                l_in, y.shape()[2]
            );
        }
    }
}
