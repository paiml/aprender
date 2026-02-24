// =========================================================================
// FALSIFY-BN: batchnorm-kernel-v1.yaml contract (aprender BatchNorm1d)
//
// Five-Whys (PMAT-354):
//   Why 1: aprender had 10+ BatchNorm tests but zero FALSIFY-BN-* tests
//   Why 2: unit tests verify shapes/parameters, not mathematical invariants
//   Why 3: no mapping from batchnorm-kernel-v1.yaml to aprender test names
//   Why 4: aprender predates the provable-contracts YAML convention
//   Why 5: BatchNorm was "obviously correct" (standard Ioffe & Szegedy)
//
// References:
//   - provable-contracts/contracts/batchnorm-kernel-v1.yaml
//   - Ioffe & Szegedy (2015) "Batch Normalization"
// =========================================================================

use super::*;

/// FALSIFY-BN-001: Training standardization — per-channel mean ≈ 0
///
/// With gamma=1, beta=0 (defaults), each channel's batch mean should be ≈ 0.
#[test]
fn falsify_bn_001_training_standardization() {
    let norm = BatchNorm1d::new(3);
    // Input: [batch=4, features=3]
    let x = Tensor::new(
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        &[4, 3],
    );
    let y = norm.forward(&x);
    let y_data = y.data();

    // Check per-channel (column) means
    for c in 0..3 {
        let channel_mean: f32 = (0..4).map(|b| y_data[b * 3 + c]).sum::<f32>() / 4.0;
        assert!(
            channel_mean.abs() < 1e-4,
            "FALSIFIED BN-001: channel {c} mean = {channel_mean}, expected ≈ 0"
        );
    }
}

/// FALSIFY-BN-002: Denominator safety — no NaN/Inf for constant channel
///
/// When all values in a channel are equal (zero variance), eps prevents div-by-zero.
#[test]
fn falsify_bn_002_denominator_safety() {
    let norm = BatchNorm1d::new(2);
    // Channel 0: all 5.0 (zero variance), Channel 1: varying
    let x = Tensor::new(&[5.0, 1.0, 5.0, 2.0, 5.0, 3.0, 5.0, 4.0], &[4, 2]);
    let y = norm.forward(&x);

    for (i, &val) in y.data().iter().enumerate() {
        assert!(
            val.is_finite(),
            "FALSIFIED BN-002: output[{i}] = {val} (NaN/Inf for constant channel)"
        );
    }
}

/// FALSIFY-BN-004: Eval uses running stats — BN_eval(x) != BN_train(x)
///
/// After updating running stats, eval mode should produce different output
/// than training mode.
#[test]
fn falsify_bn_004_eval_uses_running_stats() {
    let mut norm = BatchNorm1d::new(2);

    // First forward pass in training mode to update running stats
    let x = Tensor::new(&[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0], &[4, 2]);
    let y_train = norm.forward(&x);

    // Switch to eval mode
    norm.eval();
    let y_eval = norm.forward(&x);

    // Training and eval should differ (running stats != batch stats after 1 update)
    let any_differ = y_train
        .data()
        .iter()
        .zip(y_eval.data().iter())
        .any(|(&a, &b)| (a - b).abs() > 1e-4);
    assert!(
        any_differ,
        "FALSIFIED BN-004: eval output matches training output exactly"
    );
}

/// FALSIFY-BN-006: Boundary batch_size=1 — zero variance yields beta
///
/// With N=1, variance is 0, so output = gamma * 0 + beta = beta (= 0 by default).
#[test]
fn falsify_bn_006_batch_size_one() {
    let norm = BatchNorm1d::new(3);
    let x = Tensor::new(&[5.0, -3.0, 7.0], &[1, 3]);
    let y = norm.forward(&x);

    for (i, &val) in y.data().iter().enumerate() {
        assert!(
            val.is_finite(),
            "FALSIFIED BN-006: output[{i}] = {val} (not finite for batch=1)"
        );
        // With batch=1, variance=0, normalized = 0/(0+eps).sqrt() ≈ 0, so output ≈ beta = 0
        assert!(
            val.abs() < 1e-2,
            "FALSIFIED BN-006: output[{i}] = {val}, expected ≈ 0 for batch_size=1"
        );
    }
}

mod bn_proptest_falsify {
    use super::*;
    use proptest::prelude::*;

    /// FALSIFY-BN-001-prop: Training standardization — per-channel mean ≈ 0
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(30))]

        #[test]
        fn falsify_bn_001_prop_training_standardization(
            n_features in 1..=8usize,
            seed in 0..500u32,
        ) {
            let batch = 8;
            let norm = BatchNorm1d::new(n_features);
            let data: Vec<f32> = (0..batch * n_features)
                .map(|i| ((i as f32 + seed as f32) * 0.37).sin() * 10.0)
                .collect();
            let x = Tensor::new(&data, &[batch, n_features]);
            let y = norm.forward(&x);
            let y_data = y.data();

            for c in 0..n_features {
                let mean: f32 = (0..batch).map(|b| y_data[b * n_features + c]).sum::<f32>()
                    / batch as f32;
                prop_assert!(
                    mean.abs() < 0.1,
                    "FALSIFIED BN-001-prop: channel {} mean={} for n_features={}, seed={}",
                    c, mean, n_features, seed
                );
            }
        }
    }

    /// FALSIFY-BN-002-prop: Denominator safety — finite output for constant channels
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(30))]

        #[test]
        fn falsify_bn_002_prop_denominator_safety(
            constant_val in -100.0f32..100.0,
            n_features in 1..=4usize,
        ) {
            let batch = 4;
            let norm = BatchNorm1d::new(n_features);
            // All channels have constant value
            let data: Vec<f32> = vec![constant_val; batch * n_features];
            let x = Tensor::new(&data, &[batch, n_features]);
            let y = norm.forward(&x);

            for (i, &val) in y.data().iter().enumerate() {
                prop_assert!(
                    val.is_finite(),
                    "FALSIFIED BN-002-prop: output[{}]={} for constant={}",
                    i, val, constant_val
                );
            }
        }
    }
}
