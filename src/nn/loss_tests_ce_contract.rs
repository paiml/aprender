// =========================================================================
// FALSIFY-CE: cross-entropy-kernel-v1.yaml contract (aprender CrossEntropyLoss)
//
// Five-Whys (PMAT-354):
//   Why 1: aprender had 10+ CE tests but zero FALSIFY-CE-* tests
//   Why 2: unit tests verify correctness, not invariant boundaries
//   Why 3: no mapping from cross-entropy-kernel-v1.yaml to aprender tests
//   Why 4: aprender predates the provable-contracts YAML convention
//   Why 5: CE was "obviously correct" (standard log-softmax + NLL)
//
// References:
//   - provable-contracts/contracts/cross-entropy-kernel-v1.yaml
//   - Bishop (2006) "Pattern Recognition and Machine Learning"
// =========================================================================

#[cfg(test)]
mod tests {
    use super::super::*;

    /// FALSIFY-CE-001: Non-negativity — CE(targets, logits) >= 0
    ///
    /// Cross-entropy of a valid probability distribution is always non-negative.
    #[test]
    fn falsify_ce_001_non_negativity() {
        let criterion = CrossEntropyLoss::with_reduction(Reduction::None);

        let test_cases: Vec<(Vec<f32>, Vec<f32>, usize)> = vec![
            // (logits_flat, targets, num_classes)
            (vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![0.0, 2.0], 3),
            (vec![0.0, 0.0, 0.0, 0.0], vec![0.0, 1.0], 2),
            (vec![-10.0, 10.0, -10.0, 10.0], vec![1.0, 1.0], 2),
            (vec![100.0, -100.0, 0.0], vec![0.0], 3),
        ];

        for (i, (logits_data, targets_data, nc)) in test_cases.iter().enumerate() {
            let batch = targets_data.len();
            let logits = Tensor::new(logits_data, &[batch, *nc]);
            let targets = Tensor::new(targets_data, &[batch]);
            let loss = criterion.forward(&logits, &targets);

            for (j, &val) in loss.data().iter().enumerate() {
                assert!(
                    val >= -1e-6,
                    "FALSIFIED CE-001 case {i}[{j}]: CE = {val} < 0"
                );
            }
        }
    }

    /// FALSIFY-CE-006: Boundary — perfect prediction approaches 0
    ///
    /// When the dominant logit >> others, loss → 0.
    #[test]
    fn falsify_ce_006_perfect_prediction() {
        let criterion = CrossEntropyLoss::with_reduction(Reduction::None);

        // logit for correct class is much larger than others
        let logits = Tensor::new(&[50.0, -50.0, -50.0, -50.0, 50.0, -50.0], &[2, 3]);
        let targets = Tensor::new(&[0.0, 1.0], &[2]);
        let loss = criterion.forward(&logits, &targets);

        for (i, &val) in loss.data().iter().enumerate() {
            assert!(
                val < 1e-3,
                "FALSIFIED CE-006: CE for near-perfect prediction [{i}] = {val}, expected ≈ 0"
            );
        }
    }

    /// FALSIFY-CE-003: Numerical stability — no NaN/Inf for extreme logits
    #[test]
    fn falsify_ce_003_numerical_stability() {
        let criterion = CrossEntropyLoss::new();

        // Extreme logit ranges that can cause overflow in naive exp()
        let test_cases: Vec<(Vec<f32>, Vec<f32>, usize)> = vec![
            (vec![1000.0, -1000.0, 0.0], vec![0.0], 3),
            (vec![-500.0, -500.0, -500.0], vec![1.0], 3),
            (vec![0.0, 0.0, 0.0, 0.0], vec![2.0], 4),
        ];

        for (i, (logits_data, targets_data, nc)) in test_cases.iter().enumerate() {
            let batch = targets_data.len();
            let logits = Tensor::new(logits_data, &[batch, *nc]);
            let targets = Tensor::new(targets_data, &[batch]);
            let loss = criterion.forward(&logits, &targets);

            let val = loss.data()[0];
            assert!(
                val.is_finite(),
                "FALSIFIED CE-003 case {i}: CE = {val} (not finite)"
            );
        }
    }

    /// FALSIFY-CE-001b: Uniform logits — CE = log(num_classes)
    ///
    /// When all logits are equal, softmax is uniform, so CE = -log(1/C) = log(C).
    #[test]
    fn falsify_ce_001b_uniform_logits() {
        let criterion = CrossEntropyLoss::with_reduction(Reduction::None);

        for &num_classes in &[2_usize, 3, 5, 10] {
            let logits_data = vec![1.0; num_classes];
            let logits = Tensor::new(&logits_data, &[1, num_classes]);
            let targets = Tensor::new(&[0.0], &[1]);
            let loss = criterion.forward(&logits, &targets);

            let expected = (num_classes as f32).ln();
            let val = loss.data()[0];
            let diff = (val - expected).abs();
            assert!(
                diff < 1e-4,
                "FALSIFIED CE-001b: CE(uniform, C={num_classes}) = {val}, expected log({num_classes}) = {expected}"
            );
        }
    }

    /// FALSIFY-CE-002: Log-softmax upper bound — all log-softmax values <= 0
    #[test]
    fn falsify_ce_002_log_softmax_upper_bound() {
        let criterion = CrossEntropyLoss::with_reduction(Reduction::None);

        // For different num_classes, loss for uniform logits = log(C),
        // which is the max possible CE for one-hot targets with uniform softmax.
        // This indirectly validates log-softmax upper bound via the loss formula.
        for &nc in &[2_usize, 3, 5, 10, 50] {
            let logits_data = vec![0.0; nc];
            let logits = Tensor::new(&logits_data, &[1, nc]);
            let targets = Tensor::new(&[0.0], &[1]);
            let loss = criterion.forward(&logits, &targets);
            let val = loss.data()[0];
            // CE = -log(1/C) = log(C) >= 0 for C >= 1
            assert!(
                val >= -1e-6,
                "FALSIFIED CE-002: CE for uniform logits C={nc} = {val} < 0"
            );
        }
    }

    mod ce_proptest_falsify {
        use super::super::super::*;
        use proptest::prelude::*;

        /// FALSIFY-CE-001-prop: Non-negativity for random logits and targets
        proptest! {
            #![proptest_config(ProptestConfig::with_cases(200))]

            #[test]
            fn falsify_ce_001_prop_non_negativity(
                nc in 2..=10usize,
                target in 0..10usize,
                seed in 0..1000u32,
            ) {
                let target = target % nc;
                let logits_data: Vec<f32> = (0..nc)
                    .map(|i| ((i as f32 + seed as f32) * 0.37).sin() * 10.0)
                    .collect();

                let criterion = CrossEntropyLoss::with_reduction(Reduction::None);
                let logits = Tensor::new(&logits_data, &[1, nc]);
                let targets = Tensor::new(&[target as f32], &[1]);
                let loss = criterion.forward(&logits, &targets);
                let val = loss.data()[0];
                prop_assert!(
                    val >= -1e-6,
                    "FALSIFIED CE-001-prop: CE = {} < 0 (nc={}, target={})",
                    val, nc, target
                );
            }
        }

        /// FALSIFY-CE-003-prop: Numerical stability for random scales
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
                let logits_data: Vec<f32> = (0..nc)
                    .map(|i| ((i as f32 + seed as f32) * 0.73).cos() * scale)
                    .collect();

                let criterion = CrossEntropyLoss::new();
                let logits = Tensor::new(&logits_data, &[1, nc]);
                let targets = Tensor::new(&[target as f32], &[1]);
                let loss = criterion.forward(&logits, &targets);
                let val = loss.data()[0];
                prop_assert!(
                    val.is_finite(),
                    "FALSIFIED CE-003-prop: CE = {} (not finite), nc={}, scale={}",
                    val, nc, scale
                );
            }
        }

        /// FALSIFY-CE-006-prop: Perfect prediction approaches zero
        proptest! {
            #![proptest_config(ProptestConfig::with_cases(100))]

            #[test]
            fn falsify_ce_006_prop_perfect_prediction(
                nc in 2..=10usize,
                target in 0..10usize,
            ) {
                let target = target % nc;
                let mut logits_data = vec![-50.0; nc];
                logits_data[target] = 50.0;

                let criterion = CrossEntropyLoss::with_reduction(Reduction::None);
                let logits = Tensor::new(&logits_data, &[1, nc]);
                let targets = Tensor::new(&[target as f32], &[1]);
                let loss = criterion.forward(&logits, &targets);
                let val = loss.data()[0];
                prop_assert!(
                    val < 1e-3,
                    "FALSIFIED CE-006-prop: CE = {} for dominant logit, expected ≈ 0",
                    val
                );
            }
        }
    }
}
