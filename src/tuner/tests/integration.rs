//! Integration correctness tests (F061-F080).

use super::super::*;
use crate::brick::BrickProfiler;

#[test]
fn f066_recommendations_json_valid() {
    let features = TunerFeatures::builder().model_params_b(1.5).batch_size(4).build();

    let tuner = BrickTuner::new();
    let rec = tuner.recommend(&features);

    let json = serde_json::to_string(&rec);
    assert!(json.is_ok(), "Failed to serialize recommendation");
}

#[test]
fn f070_safetensors_roundtrip() {
    let tuner = BrickTuner::new();

    // Serialize
    let json = tuner.to_json();
    assert!(json.is_ok());

    // Deserialize
    let loaded = BrickTuner::from_json(&json.unwrap());
    assert!(loaded.is_ok());
    assert_eq!(loaded.unwrap().version, tuner.version);
}

#[test]
fn f071_feature_extractor_deterministic() {
    let config = RunConfig::default();
    let profiler = BrickProfiler::new();
    let extractor = FeatureExtractor::new();

    let f1 = extractor.extract(&profiler, &config);
    let f2 = extractor.extract(&profiler, &config);

    assert_eq!(f1.to_vector(), f2.to_vector());
}

#[test]
fn f072_prediction_deterministic() {
    let features = TunerFeatures::builder().model_params_b(1.5).batch_size(4).build();

    let tuner = BrickTuner::new();
    let rec1 = tuner.recommend(&features);
    let rec2 = tuner.recommend(&features);

    assert_eq!(rec1.throughput.predicted_tps, rec2.throughput.predicted_tps);
    assert_eq!(rec1.kernel.top_kernel, rec2.kernel.top_kernel);
}

#[test]
fn f075_error_handling_graceful() {
    // Invalid features should not panic
    let features = TunerFeatures { model_params_b: f32::NAN, ..Default::default() };

    let result = features.validate();
    assert!(result.is_err());
}
