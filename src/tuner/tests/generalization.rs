//! Generalization and robustness tests (F081-F100).

use super::super::*;

#[test]
fn f085_adversarial_inputs_handled() {
    // Extreme values should not crash
    let features = TunerFeatures::builder()
        .model_params_b(0.001) // Very small
        .hidden_dim(1)
        .batch_size(1000) // Very large (will be clamped)
        .build();

    let tuner = BrickTuner::new();
    let rec = tuner.recommend(&features);

    // Should produce some recommendation without crashing
    assert!(rec.throughput.predicted_tps > 0.0);
}

#[test]
fn f091_cold_start_handling() {
    // Tuner should work with default (untrained) model
    let tuner = BrickTuner::new();
    assert_eq!(tuner.sample_count, 0);

    let features = TunerFeatures::builder().batch_size(4).build();
    let rec = tuner.recommend(&features);

    // Should still produce reasonable recommendations
    assert!(rec.confidence_overall > 0.0);
}

#[test]
fn f096_extreme_values_clipped() {
    let features = TunerFeatures::builder()
        .model_params_b(1000.0) // Way over max
        .hidden_dim(100000) // Way over max
        .batch_size(1000) // Way over max
        .build();

    // All values should be clipped to [0, 1]
    let v = features.to_vector();
    assert!(v.iter().all(|x| *x >= 0.0 && *x <= 1.0));
}
