//! Tests for MSE, L1, and Huber losses

use super::*;
use crate::train::loss::LossFn;
use crate::Tensor;
use approx::assert_relative_eq;

#[test]
fn test_mse_accuracy_matches_reference() {
    let pred_vals = vec![1.0_f32, 2.0, 3.0, 4.0];
    let target_vals = vec![1.5_f32, 2.5, 2.5, 4.5];
    // Reference: mean((pred - target)^2) in f64
    let reference: f64 = pred_vals
        .iter()
        .zip(&target_vals)
        .map(|(p, t)| (f64::from(*p) - f64::from(*t)) * (f64::from(*p) - f64::from(*t)))
        .sum::<f64>()
        / pred_vals.len() as f64;
    let mse = MSELoss;
    let pred = Tensor::from_vec(pred_vals, false);
    let tgt = Tensor::from_vec(target_vals, false);
    let loss = mse.forward(&pred, &tgt);
    let actual = loss.data()[0];
    let diff = (f64::from(actual) - reference).abs();
    assert!(diff < 1e-6, "MSE accuracy: actual={actual}, ref={reference}, diff={diff}");
}

#[test]
fn test_mse_loss_basic() {
    let loss_fn = MSELoss;
    let pred = Tensor::from_vec(vec![1.0, 2.0, 3.0], true);
    let target = Tensor::from_vec(vec![1.5, 2.5, 3.5], false);

    let loss = loss_fn.forward(&pred, &target);

    // MSE = mean((0.5, 0.5, 0.5)^2) = 0.25
    assert_relative_eq!(loss.data()[0], 0.25, epsilon = 1e-5);
}

#[test]
fn test_mse_loss_zero_for_perfect() {
    let loss_fn = MSELoss;
    let pred = Tensor::from_vec(vec![1.0, 2.0, 3.0], true);
    let target = Tensor::from_vec(vec![1.0, 2.0, 3.0], false);

    let loss = loss_fn.forward(&pred, &target);

    assert_relative_eq!(loss.data()[0], 0.0, epsilon = 1e-5);
}

#[test]
fn test_mse_gradient() {
    let loss_fn = MSELoss;
    let pred = Tensor::from_vec(vec![1.0, 2.0, 3.0], true);
    let target = Tensor::from_vec(vec![0.0, 0.0, 0.0], false);

    let loss = loss_fn.forward(&pred, &target);

    // Trigger backward
    if let Some(backward_op) = loss.backward_op() {
        backward_op.backward();
    }

    // Check gradient: d(MSE)/d(pred) = 2*(pred - target)/n
    let grad = pred.grad().expect("gradient should be available");
    assert_relative_eq!(grad[0], 2.0 / 3.0, epsilon = 1e-5);
    assert_relative_eq!(grad[1], 4.0 / 3.0, epsilon = 1e-5);
    assert_relative_eq!(grad[2], 6.0 / 3.0, epsilon = 1e-5);
}

#[test]
#[should_panic(expected = "must have same length")]
fn test_mse_mismatched_lengths() {
    let loss_fn = MSELoss;
    let pred = Tensor::from_vec(vec![1.0, 2.0], true);
    let target = Tensor::from_vec(vec![1.0, 2.0, 3.0], false);

    loss_fn.forward(&pred, &target);
}

#[test]
fn test_mse_no_grad() {
    let loss_fn = MSELoss;
    let pred = Tensor::from_vec(vec![1.0, 2.0], false); // requires_grad = false
    let target = Tensor::from_vec(vec![1.5, 2.5], false);
    let loss = loss_fn.forward(&pred, &target);
    assert!(loss.data()[0] > 0.0);
}

#[test]
fn test_gradient_accumulation_mse() {
    let pred = Tensor::from_vec(vec![1.0, 2.0], true);
    let target = Tensor::from_vec(vec![0.0, 0.0], false);

    // First forward/backward
    let loss1 = MSELoss.forward(&pred, &target);
    if let Some(op) = loss1.backward_op() {
        op.backward();
    }

    // Second forward/backward - gradients should accumulate
    let loss2 = MSELoss.forward(&pred, &target);
    if let Some(op) = loss2.backward_op() {
        op.backward();
    }

    // Gradient should be 2x the single pass
    let grad = pred.grad().expect("gradient should be available");
    assert!(grad[0].abs() > 0.0);
    assert!(grad[1].abs() > 0.0);
}

#[test]
fn test_l1_loss_basic() {
    let loss_fn = L1Loss;
    let pred = Tensor::from_vec(vec![1.0, 2.0, 3.0], true);
    let target = Tensor::from_vec(vec![1.5, 2.5, 3.5], false);

    let loss = loss_fn.forward(&pred, &target);

    // L1 = mean(|0.5, 0.5, 0.5|) = 0.5
    assert_relative_eq!(loss.data()[0], 0.5, epsilon = 1e-5);
}

#[test]
fn test_l1_loss_zero_for_perfect() {
    let loss_fn = L1Loss;
    let pred = Tensor::from_vec(vec![1.0, 2.0, 3.0], true);
    let target = Tensor::from_vec(vec![1.0, 2.0, 3.0], false);

    let loss = loss_fn.forward(&pred, &target);
    assert_relative_eq!(loss.data()[0], 0.0, epsilon = 1e-5);
}

#[test]
fn test_l1_loss_gradient() {
    let loss_fn = L1Loss;
    let pred = Tensor::from_vec(vec![2.0, 0.0], true);
    let target = Tensor::from_vec(vec![0.0, 2.0], false);

    let loss = loss_fn.forward(&pred, &target);

    if let Some(backward_op) = loss.backward_op() {
        backward_op.backward();
    }

    let grad = pred.grad().expect("gradient should be available");
    // Gradient: sign(error) / n
    // First: sign(2) / 2 = 0.5
    // Second: sign(-2) / 2 = -0.5
    assert_relative_eq!(grad[0], 0.5, epsilon = 1e-5);
    assert_relative_eq!(grad[1], -0.5, epsilon = 1e-5);
}

#[test]
fn test_l1_robust_to_outliers() {
    let l1_loss = L1Loss;
    let mse_loss = MSELoss;

    // Normal data with one outlier
    let pred = Tensor::from_vec(vec![1.0, 2.0, 100.0], true);
    let target = Tensor::from_vec(vec![1.0, 2.0, 0.0], false);

    let l1 = l1_loss.forward(&pred, &target);
    let mse = mse_loss.forward(&pred.clone(), &target);

    // L1 should be much smaller than MSE due to outlier
    // L1 = (0 + 0 + 100) / 3 = 33.33
    // MSE = (0 + 0 + 10000) / 3 = 3333.33
    assert!(l1.data()[0] < mse.data()[0]);
}

#[test]
#[should_panic(expected = "must have same length")]
fn test_l1_mismatched_lengths() {
    let loss_fn = L1Loss;
    let pred = Tensor::from_vec(vec![1.0, 2.0], true);
    let target = Tensor::from_vec(vec![1.0, 2.0, 3.0], false);
    loss_fn.forward(&pred, &target);
}

#[test]
fn test_l1_no_grad() {
    let loss_fn = L1Loss;
    let pred = Tensor::from_vec(vec![1.0, 2.0], false);
    let target = Tensor::from_vec(vec![1.5, 2.5], false);
    let loss = loss_fn.forward(&pred, &target);
    assert!(loss.data()[0] > 0.0);
}

#[test]
fn test_gradient_accumulation_l1() {
    let pred = Tensor::from_vec(vec![1.0, 2.0], true);
    let target = Tensor::from_vec(vec![0.0, 0.0], false);

    let loss1 = L1Loss.forward(&pred, &target);
    if let Some(op) = loss1.backward_op() {
        op.backward();
    }

    let loss2 = L1Loss.forward(&pred, &target);
    if let Some(op) = loss2.backward_op() {
        op.backward();
    }

    let grad = pred.grad().expect("gradient should be available");
    assert!(grad[0].is_finite());
    assert!(grad[1].is_finite());
}

#[test]
fn test_huber_loss_small_error() {
    let loss_fn = HuberLoss::new(1.0);
    let pred = Tensor::from_vec(vec![1.0, 2.0, 3.0], true);
    let target = Tensor::from_vec(vec![1.5, 2.5, 3.5], false); // errors = 0.5

    let loss = loss_fn.forward(&pred, &target);

    // For small errors (|e| <= delta), Huber = 0.5 * e^2
    // = mean(0.5 * 0.25) = 0.125
    assert_relative_eq!(loss.data()[0], 0.125, epsilon = 1e-5);
}

#[test]
fn test_huber_loss_large_error() {
    let loss_fn = HuberLoss::new(1.0);
    let pred = Tensor::from_vec(vec![0.0], true);
    let target = Tensor::from_vec(vec![5.0], false); // error = 5 > delta

    let loss = loss_fn.forward(&pred, &target);

    // For large errors (|e| > delta), Huber = delta * (|e| - 0.5 * delta)
    // = 1 * (5 - 0.5) = 4.5
    assert_relative_eq!(loss.data()[0], 4.5, epsilon = 1e-5);
}

#[test]
fn test_huber_loss_mixed() {
    let loss_fn = HuberLoss::new(1.0);
    // One small error (0.5), one large error (3.0)
    let pred = Tensor::from_vec(vec![0.0, 0.0], true);
    let target = Tensor::from_vec(vec![0.5, 3.0], false);

    let loss = loss_fn.forward(&pred, &target);

    // Small: 0.5 * 0.25 = 0.125
    // Large: 1 * (3 - 0.5) = 2.5
    // Mean: (0.125 + 2.5) / 2 = 1.3125
    assert_relative_eq!(loss.data()[0], 1.3125, epsilon = 1e-5);
}

#[test]
fn test_huber_loss_gradient() {
    let loss_fn = HuberLoss::new(1.0);
    let pred = Tensor::from_vec(vec![0.0, 0.0], true);
    let target = Tensor::from_vec(vec![0.5, 3.0], false);

    let loss = loss_fn.forward(&pred, &target);

    if let Some(backward_op) = loss.backward_op() {
        backward_op.backward();
    }

    let grad = pred.grad().expect("gradient should be available");
    // Small error: grad = error / n = -0.5 / 2 = -0.25
    // Large error: grad = delta * sign(error) / n = 1 * (-1) / 2 = -0.5
    assert_relative_eq!(grad[0], -0.25, epsilon = 1e-5);
    assert_relative_eq!(grad[1], -0.5, epsilon = 1e-5);
}

#[test]
fn test_huber_default() {
    let loss_fn = HuberLoss::default();
    let pred = Tensor::from_vec(vec![1.0], true);
    let target = Tensor::from_vec(vec![2.0], false);

    let loss = loss_fn.forward(&pred, &target);
    assert!(loss.data()[0] > 0.0);
}

#[test]
fn test_smooth_l1_is_huber() {
    // SmoothL1Loss is type alias for HuberLoss
    let loss_fn: SmoothL1Loss = HuberLoss::new(1.0);
    let pred = Tensor::from_vec(vec![1.0], true);
    let target = Tensor::from_vec(vec![2.0], false);

    let loss = loss_fn.forward(&pred, &target);
    assert!(loss.data()[0] > 0.0);
}

#[test]
fn test_huber_default_delta() {
    let loss_fn = HuberLoss::default_delta();
    let pred = Tensor::from_vec(vec![1.0, 2.0], true);
    let target = Tensor::from_vec(vec![1.5, 2.5], false);
    let loss = loss_fn.forward(&pred, &target);
    assert!(loss.data()[0] > 0.0);
}

#[test]
#[should_panic(expected = "must have same length")]
fn test_huber_mismatched_lengths() {
    let loss_fn = HuberLoss::new(1.0);
    let pred = Tensor::from_vec(vec![1.0, 2.0], true);
    let target = Tensor::from_vec(vec![1.0, 2.0, 3.0], false);
    loss_fn.forward(&pred, &target);
}

#[test]
#[should_panic(expected = "delta must be positive")]
fn test_huber_negative_delta() {
    HuberLoss::new(-1.0);
}

#[test]
fn test_huber_no_grad() {
    let loss_fn = HuberLoss::new(1.0);
    let pred = Tensor::from_vec(vec![1.0, 2.0], false);
    let target = Tensor::from_vec(vec![1.5, 2.5], false);
    let loss = loss_fn.forward(&pred, &target);
    assert!(loss.data()[0] > 0.0);
}

#[test]
fn test_gradient_accumulation_huber() {
    let pred = Tensor::from_vec(vec![1.0, 5.0], true);
    let target = Tensor::from_vec(vec![0.0, 0.0], false);

    let loss1 = HuberLoss::new(1.0).forward(&pred, &target);
    if let Some(op) = loss1.backward_op() {
        op.backward();
    }

    let loss2 = HuberLoss::new(1.0).forward(&pred, &target);
    if let Some(op) = loss2.backward_op() {
        op.backward();
    }

    let grad = pred.grad().expect("gradient should be available");
    assert!(grad[0].is_finite());
    assert!(grad[1].is_finite());
}

// =========================================================================
// FALSIFY-LF: loss-functions-v1.yaml contract (entrenar MSE, L1, Huber)
//
// Five-Whys (PMAT-354):
//   Why 1: entrenar had 20+ loss tests but zero FALSIFY-LF-* tests
//   Why 2: tests verify specific values, not mathematical invariants
//   Why 3: no mapping from loss-functions-v1.yaml to entrenar test names
//   Why 4: entrenar predates the provable-contracts YAML convention
//   Why 5: losses were "obviously correct" (textbook formulas)
//
// References:
//   - provable-contracts/contracts/loss-functions-v1.yaml
// =========================================================================

/// FALSIFY-LF-001e: MSE is non-negative
#[test]
fn falsify_lf_001e_mse_non_negative() {
    for vals in [
        (vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]),
        (vec![-10.0, 0.0, 10.0], vec![5.0, -5.0, 0.0]),
        (vec![0.0, 0.0], vec![100.0, -100.0]),
    ] {
        let pred = Tensor::from_vec(vals.0.clone(), true);
        let target = Tensor::from_vec(vals.1, false);
        let loss = MSELoss.forward(&pred, &target);
        assert!(
            loss.data()[0] >= 0.0,
            "FALSIFIED LF-001e: MSE = {} < 0 for pred={:?}",
            loss.data()[0],
            vals.0
        );
    }
}

/// FALSIFY-LF-002e: Loss = 0 when pred == target (MSE, L1, Huber)
#[test]
fn falsify_lf_002e_zero_at_perfect() {
    let data = vec![1.0, -2.0, 3.5, 0.0];
    let pred = Tensor::from_vec(data.clone(), true);
    let target = Tensor::from_vec(data, false);

    let mse = MSELoss.forward(&pred, &target);
    assert!(mse.data()[0].abs() < 1e-6, "FALSIFIED LF-002e: MSE(x,x) = {}", mse.data()[0]);

    let pred2 = Tensor::from_vec(vec![1.0, -2.0, 3.5, 0.0], true);
    let target2 = Tensor::from_vec(vec![1.0, -2.0, 3.5, 0.0], false);
    let l1 = L1Loss.forward(&pred2, &target2);
    assert!(l1.data()[0].abs() < 1e-6, "FALSIFIED LF-002e: L1(x,x) = {}", l1.data()[0]);

    let pred3 = Tensor::from_vec(vec![1.0, -2.0, 3.5, 0.0], true);
    let target3 = Tensor::from_vec(vec![1.0, -2.0, 3.5, 0.0], false);
    let huber = HuberLoss::new(1.0).forward(&pred3, &target3);
    assert!(huber.data()[0].abs() < 1e-6, "FALSIFIED LF-002e: Huber(x,x) = {}", huber.data()[0]);
}

/// FALSIFY-LF-004e: Huber continuity at transition point
#[test]
fn falsify_lf_004e_huber_transition_continuity() {
    let delta = 1.0;
    let eps = 0.001;

    // Just below delta: quadratic region, 0.5 * (delta-eps)^2
    let pred_below = Tensor::from_vec(vec![0.0], true);
    let target_below = Tensor::from_vec(vec![delta - eps], false);
    let loss_below = HuberLoss::new(delta).forward(&pred_below, &target_below);

    // Just above delta: linear region, delta * (|delta+eps| - 0.5*delta)
    let pred_above = Tensor::from_vec(vec![0.0], true);
    let target_above = Tensor::from_vec(vec![delta + eps], false);
    let loss_above = HuberLoss::new(delta).forward(&pred_above, &target_above);

    // The two should be close (continuous transition)
    let diff = (loss_above.data()[0] - loss_below.data()[0]).abs();
    assert!(
        diff < 2.0 * eps,
        "FALSIFIED LF-004e: Huber discontinuous at delta: |L(δ+ε) - L(δ-ε)| = {diff}"
    );
}

/// FALSIFY-LF-005e: L1(a, b) == L1(b, a)
#[test]
fn falsify_lf_005e_l1_symmetric() {
    let a = Tensor::from_vec(vec![1.0, -3.0, 5.0], true);
    let b = Tensor::from_vec(vec![4.0, 2.0, -1.0], false);
    let loss_ab = L1Loss.forward(&a, &b);

    let a2 = Tensor::from_vec(vec![4.0, 2.0, -1.0], true);
    let b2 = Tensor::from_vec(vec![1.0, -3.0, 5.0], false);
    let loss_ba = L1Loss.forward(&a2, &b2);

    assert!(
        (loss_ab.data()[0] - loss_ba.data()[0]).abs() < 1e-6,
        "FALSIFIED LF-005e: L1(a,b)={} != L1(b,a)={}",
        loss_ab.data()[0],
        loss_ba.data()[0]
    );
}

mod lf_proptest_falsify {
    use super::*;
    use proptest::prelude::*;

    // FALSIFY-LF-001e-prop: MSE non-negative for random inputs
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn falsify_lf_001e_prop_mse_non_negative(
            seed in 0..1000u32,
            n in 2..=16usize,
        ) {
            let pred_data: Vec<f32> = (0..n)
                .map(|i| ((i as f32 + seed as f32) * 0.37).sin() * 10.0)
                .collect();
            let target_data: Vec<f32> = (0..n)
                .map(|i| ((i as f32 + seed as f32) * 0.73).cos() * 10.0)
                .collect();

            let pred = Tensor::from_vec(pred_data, true);
            let target = Tensor::from_vec(target_data, false);
            let loss = MSELoss.forward(&pred, &target);
            prop_assert!(
                loss.data()[0] >= 0.0,
                "FALSIFIED LF-001e-prop: MSE = {} < 0",
                loss.data()[0]
            );
        }
    }

    // FALSIFY-LF-005e-prop: L1 symmetry for random inputs
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn falsify_lf_005e_prop_l1_symmetric(
            seed in 0..1000u32,
            n in 2..=16usize,
        ) {
            let a_data: Vec<f32> = (0..n)
                .map(|i| ((i as f32 + seed as f32) * 0.37).sin() * 10.0)
                .collect();
            let b_data: Vec<f32> = (0..n)
                .map(|i| ((i as f32 + seed as f32) * 0.73).cos() * 10.0)
                .collect();

            let loss_ab = L1Loss.forward(
                &Tensor::from_vec(a_data.clone(), true),
                &Tensor::from_vec(b_data.clone(), false),
            );
            let loss_ba = L1Loss.forward(
                &Tensor::from_vec(b_data, true),
                &Tensor::from_vec(a_data, false),
            );
            prop_assert!(
                (loss_ab.data()[0] - loss_ba.data()[0]).abs() < 1e-5,
                "FALSIFIED LF-005e-prop: L1(a,b)={} != L1(b,a)={}",
                loss_ab.data()[0], loss_ba.data()[0]
            );
        }
    }
}
