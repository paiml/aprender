//! Cross Entropy Loss for classification

use crate::Tensor;
use ndarray::Array1;

use super::LossFn;

/// Cross Entropy Loss (for classification)
///
/// L = -sum(targets * log(softmax(predictions)))
///
/// # Example
///
/// ```
/// use entrenar::train::{CrossEntropyLoss, LossFn};
/// use entrenar::Tensor;
///
/// let loss_fn = CrossEntropyLoss;
/// let logits = Tensor::from_vec(vec![2.0, 1.0, 0.5], true);
/// let targets = Tensor::from_vec(vec![1.0, 0.0, 0.0], false); // one-hot
///
/// let loss = loss_fn.forward(&logits, &targets);
/// assert!(loss.data()[0] > 0.0);
/// ```
pub struct CrossEntropyLoss;

impl CrossEntropyLoss {
    /// Compute softmax: exp(x_i) / sum(exp(x_j))
    pub(crate) fn softmax(x: &Array1<f32>) -> Array1<f32> {
        let max = x.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_x: Array1<f32> = x.mapv(|v| (v - max).exp());
        let sum: f32 = exp_x.sum();
        exp_x / sum
    }
}

impl LossFn for CrossEntropyLoss {
    fn forward(&self, predictions: &Tensor, targets: &Tensor) -> Tensor {
        assert_eq!(
            predictions.len(),
            targets.len(),
            "Predictions and targets must have same length"
        );

        // Compute softmax
        let probs = Self::softmax(predictions.data());

        // Compute cross entropy: -sum(targets * log(probs))
        let ce: f32 = targets
            .data()
            .iter()
            .zip(probs.iter())
            .map(|(&t, &p)| -t * (p + 1e-10).max(f32::MIN_POSITIVE).ln())
            .sum();

        // Create loss tensor
        let mut loss = Tensor::from_vec(vec![ce], true);

        // Set up gradient: d(CE)/d(logits) = probs - targets
        let grad = &probs - targets.data();

        use crate::autograd::BackwardOp;
        use std::rc::Rc;

        struct CEBackward {
            pred_grad_cell: Rc<std::cell::RefCell<Option<Array1<f32>>>>,
            grad: Array1<f32>,
        }

        impl BackwardOp for CEBackward {
            fn backward(&self) {
                let mut pred_grad = self.pred_grad_cell.borrow_mut();
                if let Some(existing) = pred_grad.as_mut() {
                    *existing = &*existing + &self.grad;
                } else {
                    *pred_grad = Some(self.grad.clone());
                }
            }
        }

        if predictions.requires_grad() {
            loss.set_backward_op(Rc::new(CEBackward {
                pred_grad_cell: predictions.grad_cell(),
                grad,
            }));
        }

        loss
    }

    fn name(&self) -> &'static str {
        "CrossEntropy"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Reference softmax (f64 precision) for accuracy verification
    fn reference_softmax_f64(logits: &[f32]) -> Vec<f64> {
        let logits_f64: Vec<f64> = logits.iter().map(|&x| f64::from(x)).collect();
        let max = logits_f64.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let exp_vals: Vec<f64> = logits_f64.iter().map(|&x| (x - max).exp()).collect();
        let sum: f64 = exp_vals.iter().sum();
        exp_vals.iter().map(|&e| e / sum).collect()
    }

    /// Reference cross-entropy (f64 precision) for accuracy verification
    fn reference_cross_entropy_f64(logits: &[f32], target_idx: usize) -> f64 {
        let probs = reference_softmax_f64(logits);
        -probs[target_idx].max(1e-30).ln()
    }

    #[test]
    fn test_cross_entropy_accuracy_matches_reference() {
        let logits = vec![2.0_f32, 1.0, 0.5];
        let target_idx = 0;
        let reference = reference_cross_entropy_f64(&logits, target_idx) as f32;
        let ce = CrossEntropyLoss;
        let pred = Tensor::from_vec(logits, false);
        let mut one_hot = vec![0.0_f32; 3];
        one_hot[target_idx] = 1.0;
        let tgt = Tensor::from_vec(one_hot, false);
        let loss = ce.forward(&pred, &tgt);
        let actual = loss.data()[0];
        let diff = (actual - reference).abs();
        assert!(diff < 1e-5, "CE accuracy: actual={actual}, ref={reference}, diff={diff}");
    }

    #[test]
    fn test_cross_entropy_accuracy_10class() {
        let logits: Vec<f32> = (0..10).map(|i| (i as f32 - 5.0) * 0.5).collect();
        for target_idx in 0..10 {
            let reference = reference_cross_entropy_f64(&logits, target_idx) as f32;
            let ce = CrossEntropyLoss;
            let pred = Tensor::from_vec(logits.clone(), false);
            let mut one_hot = vec![0.0_f32; 10];
            one_hot[target_idx] = 1.0;
            let tgt = Tensor::from_vec(one_hot, false);
            let loss = ce.forward(&pred, &tgt);
            let actual = loss.data()[0];
            let diff = (actual - reference).abs();
            assert!(diff < 1e-4, "CE accuracy 10-class[{target_idx}]: diff={diff}");
        }
    }

    #[test]
    fn test_cross_entropy_loss() {
        let loss_fn = CrossEntropyLoss;
        let logits = Tensor::from_vec(vec![2.0, 1.0, 0.5], true);
        let targets = Tensor::from_vec(vec![1.0, 0.0, 0.0], false);

        let loss = loss_fn.forward(&logits, &targets);

        // Loss should be positive
        assert!(loss.data()[0] > 0.0);
        assert!(loss.data()[0].is_finite());
    }

    #[test]
    fn test_softmax() {
        let x = Array1::from(vec![1.0, 2.0, 3.0]);
        let probs = CrossEntropyLoss::softmax(&x);

        // Probabilities should sum to 1
        let sum: f32 = probs.sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-5);

        // All probabilities should be in [0, 1]
        for &p in &probs {
            assert!((0.0..=1.0).contains(&p));
        }
    }

    #[test]
    fn test_cross_entropy_gradient() {
        let loss_fn = CrossEntropyLoss;
        let logits = Tensor::from_vec(vec![2.0, 1.0, 0.5], true);
        let targets = Tensor::from_vec(vec![1.0, 0.0, 0.0], false);

        let loss = loss_fn.forward(&logits, &targets);

        if let Some(backward_op) = loss.backward_op() {
            backward_op.backward();
        }

        let grad = logits.grad().expect("gradient should be available");
        // Gradient should exist and be finite
        for g in &grad {
            assert!(g.is_finite());
        }
        // For CE with target at index 0, grad[0] should be negative
        // (pred - target where target=1)
        assert!(grad[0] < 0.0);
    }

    #[test]
    #[should_panic(expected = "must have same length")]
    fn test_cross_entropy_mismatched_lengths() {
        let loss_fn = CrossEntropyLoss;
        let pred = Tensor::from_vec(vec![1.0, 2.0], true);
        let target = Tensor::from_vec(vec![1.0, 2.0, 3.0], false);
        loss_fn.forward(&pred, &target);
    }

    #[test]
    fn test_cross_entropy_no_grad() {
        let loss_fn = CrossEntropyLoss;
        let pred = Tensor::from_vec(vec![2.0, 1.0], false);
        let target = Tensor::from_vec(vec![1.0, 0.0], false);
        let loss = loss_fn.forward(&pred, &target);
        assert!(loss.data()[0] > 0.0);
    }

    #[test]
    fn test_softmax_numerical_stability() {
        // Large values that could cause overflow without max subtraction
        let x = Array1::from(vec![1000.0, 1001.0, 1002.0]);
        let probs = CrossEntropyLoss::softmax(&x);

        // Should still sum to 1.0
        let sum: f32 = probs.sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-5);

        // All values should be valid
        for &p in &probs {
            assert!(p.is_finite());
            assert!(p >= 0.0);
        }
    }

    #[test]
    fn test_gradient_accumulation_cross_entropy() {
        let logits = Tensor::from_vec(vec![2.0, 1.0], true);
        let targets = Tensor::from_vec(vec![1.0, 0.0], false);

        let loss1 = CrossEntropyLoss.forward(&logits, &targets);
        if let Some(op) = loss1.backward_op() {
            op.backward();
        }

        let loss2 = CrossEntropyLoss.forward(&logits, &targets);
        if let Some(op) = loss2.backward_op() {
            op.backward();
        }

        let grad = logits.grad().expect("gradient should be available");
        assert!(grad[0].is_finite());
        assert!(grad[1].is_finite());
    }
}

// =========================================================================
// FALSIFY-CE: cross-entropy-kernel-v1.yaml contract (entrenar CrossEntropyLoss)
//
// Five-Whys (PMAT-354):
//   Why 1: entrenar had 7 CE tests but zero FALSIFY-CE-* contract tests
//   Why 2: existing tests verify API shape, not mathematical invariants
//   Why 3: no mapping from cross-entropy-kernel-v1.yaml claims to test names
//   Why 4: entrenar CE predates the provable-contracts YAML convention
//   Why 5: CE was "obviously correct" (standard softmax + NLL)
//
// References:
//   - provable-contracts/contracts/cross-entropy-kernel-v1.yaml
//   - Shannon (1948) "A Mathematical Theory of Communication"
// =========================================================================
#[cfg(test)]
mod ce_contract_tests {
    use super::*;
    use ndarray::Array1;

    /// Helper: create one-hot targets
    fn one_hot(idx: usize, len: usize) -> Vec<f32> {
        let mut v = vec![0.0; len];
        v[idx] = 1.0;
        v
    }

    /// FALSIFY-CE-001: Non-negativity — CE(targets, logits) >= 0
    ///
    /// Contract: Cross-entropy of valid probability targets is always non-negative.
    #[test]
    fn falsify_ce_001_non_negativity() {
        let ce = CrossEntropyLoss;

        let cases: Vec<(Vec<f32>, Vec<f32>)> = vec![
            (vec![2.0, 1.0, 0.5], one_hot(0, 3)),
            (vec![0.0, 0.0, 0.0], one_hot(1, 3)),
            (vec![-10.0, 10.0], one_hot(0, 2)),
            (vec![100.0, -100.0, 0.0], one_hot(2, 3)),
            (vec![0.1, 0.2, 0.3, 0.4], one_hot(3, 4)),
        ];

        for (i, (logits, targets)) in cases.iter().enumerate() {
            let pred = Tensor::from_vec(logits.clone(), false);
            let tgt = Tensor::from_vec(targets.clone(), false);
            let loss = ce.forward(&pred, &tgt);
            let val = loss.data()[0];
            assert!(val >= -1e-6, "FALSIFIED CE-001 case {i}: CE = {val} < 0");
        }
    }

    /// FALSIFY-CE-002: Log-softmax upper bound — log_softmax(x)_i <= 0
    ///
    /// Contract: All log-softmax values must be non-positive.
    #[test]
    fn falsify_ce_002_log_softmax_upper_bound() {
        let cases: Vec<Vec<f32>> = vec![
            vec![1.0, 2.0, 3.0],
            vec![0.0, 0.0, 0.0],
            vec![-100.0, 100.0],
            vec![1000.0, 1001.0, 999.0],
            vec![-500.0, -500.0, -500.0, -500.0],
        ];

        for (i, logits) in cases.iter().enumerate() {
            let x = Array1::from(logits.clone());
            let probs = CrossEntropyLoss::softmax(&x);
            for (j, &p) in probs.iter().enumerate() {
                let log_p = p.ln();
                assert!(log_p <= 1e-6, "FALSIFIED CE-002 case {i}[{j}]: log_softmax = {log_p} > 0");
            }
        }
    }

    /// FALSIFY-CE-003: Numerical stability — no NaN/Inf for finite logits
    ///
    /// Contract: CE must produce finite output for all finite inputs.
    #[test]
    fn falsify_ce_003_numerical_stability() {
        let ce = CrossEntropyLoss;

        let extreme_cases: Vec<(Vec<f32>, Vec<f32>)> = vec![
            (vec![500.0, -500.0, 0.0], one_hot(0, 3)),
            (vec![-1000.0, -1000.0, -1000.0], one_hot(1, 3)),
            (vec![88.0, 88.0], one_hot(0, 2)), // near f32 exp overflow
            (vec![-88.0, -88.0, -88.0], one_hot(2, 3)), // near f32 exp underflow
        ];

        for (i, (logits, targets)) in extreme_cases.iter().enumerate() {
            let pred = Tensor::from_vec(logits.clone(), false);
            let tgt = Tensor::from_vec(targets.clone(), false);
            let loss = ce.forward(&pred, &tgt);
            let val = loss.data()[0];
            assert!(val.is_finite(), "FALSIFIED CE-003 case {i}: CE = {val} (not finite)");
        }
    }

    /// FALSIFY-CE-006: Perfect prediction — CE approaches 0 as dominant logit grows
    ///
    /// Contract: CE(one_hot(k), logits) → 0 when logits_k >> logits_j for j≠k
    #[test]
    fn falsify_ce_006_perfect_prediction() {
        let ce = CrossEntropyLoss;

        for &target in &[0, 1, 2] {
            let mut logits = vec![-50.0; 3];
            logits[target] = 50.0;
            let pred = Tensor::from_vec(logits, false);
            let tgt = Tensor::from_vec(one_hot(target, 3), false);
            let loss = ce.forward(&pred, &tgt);
            let val = loss.data()[0];
            assert!(
                val < 1e-3,
                "FALSIFIED CE-006: CE(one_hot({target}), dominant) = {val}, expected ≈ 0"
            );
        }
    }

    /// FALSIFY-CE-001b: Uniform logits — CE = log(C)
    ///
    /// Contract: When all logits are equal, softmax is uniform 1/C,
    /// so CE = -log(1/C) = log(C).
    #[test]
    fn falsify_ce_001b_uniform_logits() {
        let ce = CrossEntropyLoss;

        for &nc in &[2_usize, 3, 5, 10] {
            let logits = vec![1.0; nc];
            let targets = one_hot(0, nc);
            let pred = Tensor::from_vec(logits, false);
            let tgt = Tensor::from_vec(targets, false);
            let loss = ce.forward(&pred, &tgt);
            let val = loss.data()[0];
            let expected = (nc as f32).ln();
            let diff = (val - expected).abs();
            assert!(
                diff < 1e-4,
                "FALSIFIED CE-001b: CE(uniform, C={nc}) = {val}, expected log({nc}) = {expected}"
            );
        }
    }

    mod ce_proptest_falsify {
        use super::*;
        use proptest::prelude::*;

        // FALSIFY-CE-001-prop: Non-negativity for random one-hot targets
        proptest! {
            #![proptest_config(ProptestConfig::with_cases(200))]

            #[test]
            fn falsify_ce_001_prop_non_negativity(
                nc in 2..=10usize,
                target in 0..10usize,
                seed in 0..1000u32,
            ) {
                let target = target % nc;
                let logits: Vec<f32> = (0..nc)
                    .map(|i| ((i as f32 + seed as f32) * 0.37).sin() * 10.0)
                    .collect();

                let ce = CrossEntropyLoss;
                let pred = Tensor::from_vec(logits, false);
                let tgt = Tensor::from_vec(one_hot(target, nc), false);
                let loss = ce.forward(&pred, &tgt);
                let val = loss.data()[0];
                prop_assert!(
                    val >= -1e-6,
                    "FALSIFIED CE-001-prop: CE = {} < 0 (nc={}, target={})",
                    val, nc, target
                );
            }
        }

        // FALSIFY-CE-003-prop: Numerical stability for random logits
        proptest! {
            #![proptest_config(ProptestConfig::with_cases(200))]

            #[test]
            fn falsify_ce_003_prop_finite_output(
                nc in 2..=10usize,
                target in 0..10usize,
                scale in 0.1f32..100.0,
                seed in 0..1000u32,
            ) {
                let target = target % nc;
                let logits: Vec<f32> = (0..nc)
                    .map(|i| ((i as f32 + seed as f32) * 0.73).cos() * scale)
                    .collect();

                let ce = CrossEntropyLoss;
                let pred = Tensor::from_vec(logits, false);
                let tgt = Tensor::from_vec(one_hot(target, nc), false);
                let loss = ce.forward(&pred, &tgt);
                let val = loss.data()[0];
                prop_assert!(
                    val.is_finite(),
                    "FALSIFIED CE-003-prop: CE = {} (not finite) for nc={}, scale={}",
                    val, nc, scale
                );
            }
        }

        // FALSIFY-CE-002-prop: Log-softmax upper bound for random inputs
        proptest! {
            #![proptest_config(ProptestConfig::with_cases(200))]

            #[test]
            fn falsify_ce_002_prop_log_softmax_bound(
                nc in 2..=10usize,
                scale in 0.1f32..100.0,
                seed in 0..1000u32,
            ) {
                let logits: Vec<f32> = (0..nc)
                    .map(|i| ((i as f32 + seed as f32) * 0.37).sin() * scale)
                    .collect();
                let x = Array1::from(logits);
                let probs = CrossEntropyLoss::softmax(&x);
                for (j, &p) in probs.iter().enumerate() {
                    prop_assert!(
                        (0.0..=1.0 + 1e-6).contains(&p),
                        "FALSIFIED CE-002-prop: softmax[{}] = {} outside [0,1]",
                        j, p
                    );
                    let log_p = p.ln();
                    prop_assert!(
                        log_p <= 1e-6,
                        "FALSIFIED CE-002-prop: log(softmax[{}]) = {} > 0",
                        j, log_p
                    );
                }
            }
        }
    }
}
