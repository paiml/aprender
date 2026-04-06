//! F061-F080: Integration Correctness (20 points)

use trueno::tuner::{BrickTuner, QuantType, ThroughputRegressor, TunerFeatures};

/// F061: BrickProfiler integration compile check
#[test]
fn f061_brick_profiler_exists() {
    use trueno::brick::BrickProfiler;
    let _profiler = BrickProfiler::new();
}

/// F062: HardwareCapability integration
#[test]
fn f062_hardware_capability_exists() {
    use trueno::hardware::HardwareCapability;
    let _cap = HardwareCapability::detect();
}

/// F063: TunerFeatures can be built from HardwareCapability
#[test]
fn f063_features_from_hardware() {
    use trueno::hardware::HardwareCapability;

    let hw = HardwareCapability::detect();
    let features = TunerFeatures::builder()
        .gpu_mem_bw_gbs(hw.gpu.as_ref().map(|g| g.memory_bw_gbps as f32).unwrap_or(500.0))
        .build();

    assert!(features.validate().is_ok(), "F063 FALSIFIED: hardware-based features invalid");
}

/// F064: Tuner creation is fast
#[test]
fn f064_tuner_creation_fast() {
    let start = std::time::Instant::now();
    for _ in 0..100 {
        let _tuner = BrickTuner::new();
    }
    let elapsed = start.elapsed();
    let avg_us = elapsed.as_micros() / 100;

    assert!(avg_us < 1000, "F064 FALSIFIED: tuner creation {} us >= 1ms", avg_us);
}

/// F065: Model load time < 100ms (placeholder for persistence)
#[test]
fn f065_model_load_fast() {
    // Will be implemented with T-TUNER-004
    let start = std::time::Instant::now();
    let _tuner = BrickTuner::new();
    let elapsed = start.elapsed();

    assert!(
        elapsed.as_millis() < 100,
        "F065 FALSIFIED: tuner creation {} ms >= 100ms",
        elapsed.as_millis()
    );
}

/// F066: Feature extraction is fast
#[test]
fn f066_feature_extraction_fast() {
    let features = TunerFeatures::builder().model_params_b(1.5).batch_size(4).build();

    let start = std::time::Instant::now();
    for _ in 0..1000 {
        let _vec = features.to_vector();
    }
    let elapsed = start.elapsed();
    let avg_ns = elapsed.as_nanos() / 1000;

    assert!(avg_ns < 1000, "F066 FALSIFIED: feature extraction {} ns >= 1us", avg_ns);
}

/// F067: Recommendation is fast
#[test]
fn f067_recommendation_fast() {
    let tuner = BrickTuner::new();
    let features = TunerFeatures::builder().model_params_b(1.5).build();

    let start = std::time::Instant::now();
    for _ in 0..100 {
        let _rec = tuner.recommend(&features);
    }
    let elapsed = start.elapsed();
    let avg_us = elapsed.as_micros() / 100;

    assert!(avg_us < 1000, "F067 FALSIFIED: recommendation {} us >= 1ms", avg_us);
}

/// F068: Thread safety - concurrent predictions
#[test]
fn f068_thread_safety() {
    use std::sync::Arc;
    use std::thread;

    let regressor = Arc::new(ThroughputRegressor::new());
    let features = TunerFeatures::builder().model_params_b(1.5).build();

    let handles: Vec<_> = (0..4)
        .map(|_| {
            let r = Arc::clone(&regressor);
            let f = features.clone();
            thread::spawn(move || {
                for _ in 0..100 {
                    let _ = r.predict(&f);
                }
            })
        })
        .collect();

    for h in handles {
        h.join().expect("F068 FALSIFIED: thread panicked");
    }
}

/// F069: Clone works correctly
#[test]
fn f069_clone_correct() {
    let features = TunerFeatures::builder().model_params_b(1.5).batch_size(4).build();

    let cloned = features.clone();
    let orig_vec = features.to_vector();
    let clone_vec = cloned.to_vector();

    assert_eq!(orig_vec, clone_vec, "F069 FALSIFIED: clone produced different vector");
}

/// F070: Serialization round-trip (placeholder for SafeTensors)
#[test]
fn f070_serialize_roundtrip() {
    let tuner = BrickTuner::new();
    let json = serde_json::to_string(&tuner).expect("serialize");
    let restored: BrickTuner = serde_json::from_str(&json).expect("deserialize");

    // Both should produce same recommendations
    let features = TunerFeatures::builder().model_params_b(1.5).build();
    let rec1 = tuner.recommend(&features);
    let rec2 = restored.recommend(&features);

    assert_eq!(
        rec1.kernel.top_kernel, rec2.kernel.top_kernel,
        "F070 FALSIFIED: restored tuner differs"
    );
}

/// F071: Feature extractor deterministic
#[test]
fn f071_extractor_deterministic() {
    let features = TunerFeatures::builder()
        .model_params_b(1.5)
        .batch_size(4)
        .quant_type(QuantType::Q4K)
        .build();

    let vec1 = features.to_vector();
    let vec2 = features.to_vector();

    assert_eq!(vec1, vec2, "F071 FALSIFIED: feature extraction not deterministic");
}

/// F072: Prediction deterministic across instances
#[test]
fn f072_prediction_deterministic_instances() {
    let features = TunerFeatures::builder().model_params_b(1.5).build();

    let r1 = ThroughputRegressor::new();
    let r2 = ThroughputRegressor::new();

    let p1 = r1.predict(&features);
    let p2 = r2.predict(&features);

    assert!(
        (p1.predicted_tps - p2.predicted_tps).abs() < 0.001,
        "F072 FALSIFIED: different instances produce different predictions"
    );
}

/// F073: Default values are sensible
#[test]
fn f073_defaults_sensible() {
    let features = TunerFeatures::default();
    assert!(features.validate().is_ok(), "F073 FALSIFIED: default features invalid");
}

/// F074: Builder chain works
#[test]
fn f074_builder_chain() {
    let features = TunerFeatures::builder()
        .model_params_b(1.5)
        .hidden_dim(1536)
        .num_layers(28)
        .num_heads(12)
        .batch_size(4)
        .seq_len(512)
        .quant_type(QuantType::Q4K)
        .gpu_mem_bw_gbs(1000.0)
        .gpu_sm_count(128)
        .cuda_graphs(true)
        .build();

    assert!(features.validate().is_ok());
}

/// F075: Error messages are helpful
#[test]
fn f075_error_messages() {
    // F024-aware: try_build() rejects negative model_params_b at construction time
    let result = TunerFeatures::builder().model_params_b(-1.0).try_build();

    match result {
        Err(e) => {
            let msg = format!("{}", e);
            assert!(
                msg.contains("model_params")
                    || msg.contains("negative")
                    || msg.contains("non-negative")
                    || msg.contains("invalid")
                    || msg.contains("Invalid"),
                "F075 FALSIFIED: error message not helpful: {}",
                msg
            );
        }
        Ok(_) => panic!("F075 FALSIFIED: negative model_params_b should have been rejected"),
    }
}

/// F076-F080: Reserved for integration tests
#[test]
fn f076_to_f080_reserved() {
    // Reserved for:
    // F076: BrickProfiler data collection
    // F077: cbtop integration
    // F078: Logging integration
    // F079: Metrics export
    // F080: Backward compatibility
}
