//! Model accuracy tests (F001-F020).

use super::super::*;
use std::time::Instant;

#[test]
fn f001_throughput_prediction_reasonable() {
    let features = TunerFeatures::builder()
        .model_params_b(1.5)
        .hidden_dim(1536)
        .batch_size(4)
        .quant_type(QuantType::Q4K)
        .cuda_graphs(true)
        .build();

    let regressor = ThroughputRegressor::new();
    let prediction = regressor.predict(&features);

    // Prediction should be positive and reasonable
    assert!(prediction.predicted_tps > 0.0);
    assert!(prediction.predicted_tps < 10000.0);
}

#[test]
fn f010_prediction_latency_under_1ms() {
    let features = TunerFeatures::builder().model_params_b(1.5).batch_size(4).build();

    let tuner = BrickTuner::new();
    let start = Instant::now();
    let _rec = tuner.recommend(&features);
    let elapsed = start.elapsed();

    assert!(elapsed.as_millis() < 1, "Prediction took {}ms", elapsed.as_millis());
}

#[test]
fn f015_batch_size_monotonic() {
    let regressor = ThroughputRegressor::new();

    let pred_m1 = regressor.predict(&TunerFeatures::builder().batch_size(1).build());
    let pred_m4 = regressor.predict(&TunerFeatures::builder().batch_size(4).build());
    let pred_m8 = regressor.predict(&TunerFeatures::builder().batch_size(8).build());

    // Higher batch size should predict higher throughput
    assert!(
        pred_m4.predicted_tps >= pred_m1.predicted_tps,
        "M=4 ({}) should be >= M=1 ({})",
        pred_m4.predicted_tps,
        pred_m1.predicted_tps
    );
    assert!(
        pred_m8.predicted_tps >= pred_m4.predicted_tps,
        "M=8 ({}) should be >= M=4 ({})",
        pred_m8.predicted_tps,
        pred_m4.predicted_tps
    );
}

#[test]
fn f019_cuda_graphs_benefit_predicted() {
    let regressor = ThroughputRegressor::new();

    let pred_no_graph =
        regressor.predict(&TunerFeatures::builder().batch_size(1).cuda_graphs(false).build());
    let pred_with_graph =
        regressor.predict(&TunerFeatures::builder().batch_size(1).cuda_graphs(true).build());

    // CUDA graphs should predict higher throughput
    assert!(
        pred_with_graph.predicted_tps >= pred_no_graph.predicted_tps,
        "With graphs ({}) should be >= without ({})",
        pred_with_graph.predicted_tps,
        pred_no_graph.predicted_tps
    );
}
