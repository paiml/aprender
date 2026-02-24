// =========================================================================
// FALSIFY-CAL: calibration-v1.yaml contract (aprender calibration)
//
// Five-Whys (PMAT-354):
//   Why 1: aprender had no inline FALSIFY-CAL-* tests
//   Why 2: calibration tests exist but lack contract-mapped FALSIFY naming
//   Why 3: no mapping from calibration-v1.yaml to inline test names
//   Why 4: aprender predates the inline FALSIFY convention
//   Why 5: Temperature scaling was "obviously correct" (divide by T)
//
// References:
//   - provable-contracts/contracts/calibration-v1.yaml
//   - Guo et al. (2017) "On Calibration of Modern Neural Networks"
// =========================================================================

use super::*;
use crate::primitives::Vector;

/// FALSIFY-CAL-001: predict_proba outputs sum to ≈1 (softmax over scaled logits)
#[test]
fn falsify_cal_001_proba_sums_to_one() {
    let cal = TemperatureScaling::new();
    let logits = Vector::from_slice(&[2.0, 1.0, 0.5]);

    let probs = cal.predict_proba(&logits);
    let sum: f32 = probs.as_slice().iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-5,
        "FALSIFIED CAL-001: proba sum={sum}, expected ≈ 1.0"
    );
}

/// FALSIFY-CAL-002: predict_proba outputs are non-negative
#[test]
fn falsify_cal_002_non_negative() {
    let cal = TemperatureScaling::new();
    let logits = Vector::from_slice(&[-1.0, 0.0, 1.0, 2.0]);

    let probs = cal.predict_proba(&logits);
    for (i, &p) in probs.as_slice().iter().enumerate() {
        assert!(p >= 0.0, "FALSIFIED CAL-002: proba[{i}] = {p} < 0");
    }
}

/// FALSIFY-CAL-003: Calibrated output length matches input length
#[test]
fn falsify_cal_003_output_length() {
    let cal = TemperatureScaling::new();
    let logits = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);

    let calibrated = cal.calibrate(&logits);
    assert_eq!(
        calibrated.len(),
        5,
        "FALSIFIED CAL-003: {} outputs for 5 inputs",
        calibrated.len()
    );
}

/// FALSIFY-CAL-004: Default temperature is 1.0
#[test]
fn falsify_cal_004_default_temperature() {
    let cal = TemperatureScaling::new();
    assert!(
        (cal.temperature() - 1.0).abs() < 1e-6,
        "FALSIFIED CAL-004: default temperature={}, expected 1.0",
        cal.temperature()
    );
}

mod cal_proptest_falsify {
    use super::*;
    use proptest::prelude::*;

    /// FALSIFY-CAL-001-prop: predict_proba sums to ~1 for random logits
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(30))]

        #[test]
        fn falsify_cal_001_prop_proba_sum(
            n in 2..=6usize,
            seed in 0..500u32,
        ) {
            let logits_data: Vec<f32> = (0..n)
                .map(|i| ((i as f32 + seed as f32) * 0.37).sin() * 3.0)
                .collect();
            let logits = Vector::from_vec(logits_data);

            let cal = TemperatureScaling::new();
            let probs = cal.predict_proba(&logits);
            let sum: f32 = probs.as_slice().iter().sum();
            prop_assert!(
                (sum - 1.0).abs() < 1e-4,
                "FALSIFIED CAL-001-prop: proba sum={}, expected ~1.0",
                sum
            );
        }
    }

    /// FALSIFY-CAL-002-prop: predict_proba outputs non-negative
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(30))]

        #[test]
        fn falsify_cal_002_prop_non_negative(
            n in 2..=6usize,
            seed in 0..500u32,
        ) {
            let logits_data: Vec<f32> = (0..n)
                .map(|i| ((i as f32 + seed as f32) * 0.37).sin() * 5.0)
                .collect();
            let logits = Vector::from_vec(logits_data);

            let cal = TemperatureScaling::new();
            let probs = cal.predict_proba(&logits);
            for (i, &p) in probs.as_slice().iter().enumerate() {
                prop_assert!(
                    p >= 0.0,
                    "FALSIFIED CAL-002-prop: proba[{}]={} < 0",
                    i, p
                );
            }
        }
    }
}
