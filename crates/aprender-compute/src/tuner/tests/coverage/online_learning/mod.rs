//! Online learning, concept drift, error types, classifiers, builder tests.

mod feature_extractor;

use crate::tuner::*;

// =========================================================================
// TunerDataCollector: online learning and concept drift
// =========================================================================

#[test]
fn test_collector_record_feedback() {
    let mut collector = TunerDataCollector::new();
    collector.record_feedback(0, UserFeedback::Accepted);
    collector.record_feedback(2, UserFeedback::Rejected);

    assert_eq!(collector.get_feedback(0), UserFeedback::Accepted);
    assert_eq!(collector.get_feedback(1), UserFeedback::None); // Filled with None
    assert_eq!(collector.get_feedback(2), UserFeedback::Rejected);
    assert_eq!(collector.get_feedback(100), UserFeedback::None); // Out of range
}

#[test]
fn test_collector_record_prediction_error_disabled() {
    let mut collector = TunerDataCollector::new();
    // Online learning disabled by default
    collector.record_prediction_error(100.0, 80.0);
    // Error window should remain empty
    assert!(collector.error_window.is_empty());
}

#[test]
fn test_collector_record_prediction_error_enabled() {
    let mut collector = TunerDataCollector::with_online_learning();
    collector.record_prediction_error(100.0, 80.0);
    assert_eq!(collector.error_window.len(), 1);
    // Error = |100-80|/80 = 0.25
    assert!((collector.error_window[0] - 0.25).abs() < 0.01);
}

#[test]
fn test_collector_record_prediction_error_zero_actual() {
    let mut collector = TunerDataCollector::with_online_learning();
    collector.record_prediction_error(100.0, 0.0);
    assert_eq!(collector.error_window.len(), 1);
    // When actual=0, error should be 1.0
    assert_eq!(collector.error_window[0], 1.0);
}

#[test]
fn test_collector_record_prediction_error_window_trimming() {
    let mut collector = TunerDataCollector::with_online_learning();
    // Default window size is 50
    for i in 0..60 {
        collector.record_prediction_error(100.0 + i as f32, 100.0);
    }
    assert_eq!(collector.error_window.len(), 50, "Window should be trimmed to max size");
}

#[test]
fn test_collector_detect_concept_drift_insufficient_data() {
    let collector = TunerDataCollector::new();
    let status = collector.detect_concept_drift();
    assert!(!status.drift_detected);
    assert!(!status.recommend_retrain);
    assert!(status.explanation.contains("Insufficient"));
}

#[test]
fn test_collector_detect_concept_drift_with_high_error() {
    let mut collector = TunerDataCollector::with_online_learning();
    // Add enough errors that exceed threshold (0.15)
    for _ in 0..20 {
        collector.error_window.push(0.30); // 30% error > 15% threshold
    }
    let status = collector.detect_concept_drift();
    assert!(status.drift_detected);
    assert!(status.recommend_retrain);
}

#[test]
fn test_collector_detect_concept_drift_fresh_model() {
    let mut collector = TunerDataCollector::with_online_learning();
    // Add low errors
    for _ in 0..20 {
        collector.error_window.push(0.05); // 5% error < 15% threshold
    }
    collector.samples_at_last_train = collector.samples.len(); // Just trained
    let status = collector.detect_concept_drift();
    assert!(!status.drift_detected);
    assert!(!status.recommend_retrain);
    assert!(status.explanation.contains("fresh"));
}

#[test]
fn test_collector_detect_concept_drift_stale() {
    let mut collector = TunerDataCollector::with_online_learning();
    // Low errors but high staleness
    for _ in 0..20 {
        collector.error_window.push(0.05);
    }
    // Add many samples since last training
    for i in 0..120 {
        let features = TunerFeatures::builder().model_params_b(1.0 + (i as f32) * 0.1).build();
        collector.samples.push(TrainingSample {
            features,
            throughput_tps: 100.0,
            best_kernel: KernelType::TiledQ4K,
            bottleneck: BottleneckClass::MemoryBound,
            timestamp: format!("{}", i),
            hardware_id: "test".to_string(),
        });
    }
    collector.samples_at_last_train = 0; // Trained at the beginning

    let status = collector.detect_concept_drift();
    assert!(status.staleness_score > 0.8);
    assert!(status.recommend_retrain);
    assert!(status.explanation.contains("stale") || status.explanation.contains("Model stale"));
}

#[test]
fn test_collector_should_retrain_disabled() {
    let collector = TunerDataCollector::new();
    assert!(!collector.should_retrain(), "Should not retrain when online learning disabled");
}

#[test]
fn test_collector_mark_trained() {
    let mut collector = TunerDataCollector::with_online_learning();
    for _ in 0..20 {
        collector.error_window.push(0.20);
    }
    collector.mark_trained();
    assert!(collector.error_window.is_empty(), "Error window should be cleared after training");
    assert_eq!(collector.samples_at_last_train, collector.samples.len());
}

#[test]
fn test_collector_merge() {
    let mut collector1 = TunerDataCollector::new();
    let mut collector2 = TunerDataCollector::new();

    let features = TunerFeatures::builder().build();
    collector1.samples.push(TrainingSample {
        features: features.clone(),
        throughput_tps: 100.0,
        best_kernel: KernelType::TiledQ4K,
        bottleneck: BottleneckClass::MemoryBound,
        timestamp: "1".to_string(),
        hardware_id: "hw1".to_string(),
    });
    collector2.samples.push(TrainingSample {
        features,
        throughput_tps: 200.0,
        best_kernel: KernelType::VectorizedQ4K,
        bottleneck: BottleneckClass::ComputeBound,
        timestamp: "2".to_string(),
        hardware_id: "hw2".to_string(),
    });

    collector1.merge(&collector2);
    assert_eq!(collector1.samples.len(), 2);
}

#[test]
fn test_collector_training_stats() {
    let mut collector = TunerDataCollector::with_online_learning();
    collector.record_feedback(0, UserFeedback::Accepted);
    collector.record_feedback(1, UserFeedback::Rejected);
    collector.record_feedback(2, UserFeedback::Alternative);

    let stats = collector.training_stats();
    assert_eq!(stats.accepted_count, 1);
    assert_eq!(stats.rejected_count, 1);
    assert_eq!(stats.alternative_count, 1);
    assert!(stats.online_learning_enabled);
}

#[test]
fn test_collector_enable_disable_online_learning() {
    let mut collector = TunerDataCollector::new();
    assert!(!collector.is_online_learning_enabled());
    collector.enable_online_learning();
    assert!(collector.is_online_learning_enabled());
    collector.disable_online_learning();
    assert!(!collector.is_online_learning_enabled());
}

// =========================================================================
// TunerError variants coverage
// =========================================================================

#[test]
fn test_tuner_error_training_failed() {
    let err = TunerError::TrainingFailed("gradient explosion".to_string());
    assert!(format!("{}", err).contains("Training failed"));
    assert!(format!("{}", err).contains("gradient explosion"));
}

#[test]
fn test_tuner_error_io() {
    let err = TunerError::Io("file not found".to_string());
    assert!(format!("{}", err).contains("I/O error"));
}

#[test]
fn test_tuner_error_invalid_format() {
    let err = TunerError::InvalidFormat("bad magic".to_string());
    assert!(format!("{}", err).contains("Invalid format"));
}

#[test]
fn test_tuner_error_is_std_error() {
    let err = TunerError::ModelNotFound;
    // Verify it implements std::error::Error
    let _: &dyn std::error::Error = &err;
}

// =========================================================================
// Types coverage: QuantType, KernelType, BottleneckClass
// =========================================================================

#[test]
fn test_quant_type_to_index_all_variants() {
    assert_eq!(QuantType::Q4_0.to_index(), 0);
    assert_eq!(QuantType::Q4_1.to_index(), 1);
    assert_eq!(QuantType::Q4K.to_index(), 2);
    assert_eq!(QuantType::Q5K.to_index(), 3);
    assert_eq!(QuantType::Q6K.to_index(), 4);
    assert_eq!(QuantType::Q8_0.to_index(), 5);
    assert_eq!(QuantType::F16.to_index(), 6);
    assert_eq!(QuantType::F32.to_index(), 7);
}

#[test]
fn test_quant_type_bytes_per_param() {
    assert!((QuantType::Q4_0.bytes_per_param() - 0.5625).abs() < 0.001);
    assert!((QuantType::Q4_1.bytes_per_param() - 0.5625).abs() < 0.001);
    assert!((QuantType::Q4K.bytes_per_param() - 0.5625).abs() < 0.001);
    assert!((QuantType::Q5K.bytes_per_param() - 0.6875).abs() < 0.001);
    assert!((QuantType::Q6K.bytes_per_param() - 0.8125).abs() < 0.001);
    assert!((QuantType::Q8_0.bytes_per_param() - 1.0).abs() < 0.001);
    assert!((QuantType::F16.bytes_per_param() - 2.0).abs() < 0.001);
    assert!((QuantType::F32.bytes_per_param() - 4.0).abs() < 0.001);
}

#[test]
fn test_kernel_type_from_index_roundtrip() {
    for idx in 0..15 {
        let kt = KernelType::from_index(idx);
        assert_eq!(kt.to_index(), idx, "Round-trip failed for index {}", idx);
    }
    // Out of range should return Unknown
    assert_eq!(KernelType::from_index(100), KernelType::Unknown);
    assert_eq!(KernelType::from_index(15), KernelType::Unknown);
}

#[test]
fn test_bottleneck_class_from_brick_bottleneck() {
    use crate::brick::BrickBottleneck;
    assert_eq!(
        BottleneckClass::from_brick_bottleneck(BrickBottleneck::Memory),
        BottleneckClass::MemoryBound
    );
    assert_eq!(
        BottleneckClass::from_brick_bottleneck(BrickBottleneck::Compute),
        BottleneckClass::ComputeBound
    );
    assert_eq!(
        BottleneckClass::from_brick_bottleneck(BrickBottleneck::Unknown),
        BottleneckClass::Unknown
    );
}

#[test]
fn test_bottleneck_class_recommended_action() {
    let action = BottleneckClass::MemoryBound.recommended_action();
    assert!(action.contains("batch size"));
    let action = BottleneckClass::ComputeBound.recommended_action();
    assert!(action.contains("tensor cores") || action.contains("Rare"));
    let action = BottleneckClass::LaunchBound.recommended_action();
    assert!(action.contains("CUDA graphs"));
    let action = BottleneckClass::AttentionBound.recommended_action();
    assert!(action.contains("Flash") || action.contains("sequence"));
    let action = BottleneckClass::Unknown.recommended_action();
    assert!(action.contains("profiling"));
}

#[test]
fn test_bottleneck_class_to_index() {
    assert_eq!(BottleneckClass::Unknown.to_index(), 0);
    assert_eq!(BottleneckClass::MemoryBound.to_index(), 1);
    assert_eq!(BottleneckClass::ComputeBound.to_index(), 2);
    assert_eq!(BottleneckClass::LaunchBound.to_index(), 3);
    assert_eq!(BottleneckClass::AttentionBound.to_index(), 4);
}

#[test]
fn test_bottleneck_class_display() {
    assert_eq!(format!("{}", BottleneckClass::Unknown), "Unknown");
    assert_eq!(format!("{}", BottleneckClass::MemoryBound), "MemoryBound");
    assert_eq!(format!("{}", BottleneckClass::ComputeBound), "ComputeBound");
    assert_eq!(format!("{}", BottleneckClass::LaunchBound), "LaunchBound");
    assert_eq!(format!("{}", BottleneckClass::AttentionBound), "AttentionBound");
}

// =========================================================================
// Throughput Regressor: roofline clamping
// =========================================================================

#[test]
fn test_compute_roofline_bound() {
    let features = TunerFeatures::builder()
        .model_params_b(7.0)
        .quant_type(QuantType::Q4K)
        .gpu_mem_bw_gbs(1000.0)
        .batch_size(1)
        .build();
    let bound = ThroughputRegressor::compute_roofline_bound(&features);
    assert!(bound > 0.0, "Roofline bound should be positive");
    assert!(bound <= 10000.0, "Roofline bound should be <= 10000");
}

#[test]
fn test_bytes_per_param_from_onehot() {
    // Test each quant type
    for idx in 0..8 {
        let mut onehot = [0.0f32; 8];
        onehot[idx] = 1.0;
        let bpp = ThroughputRegressor::bytes_per_param_from_onehot(&onehot);
        assert!(bpp > 0.0, "Bytes per param should be positive for index {}", idx);
    }

    // All zeros: max_by returns last of equal elements (index 7) -> F32 -> 4.0
    let onehot_zeros = [0.0f32; 8];
    let bpp = ThroughputRegressor::bytes_per_param_from_onehot(&onehot_zeros);
    assert!(
        (bpp - 4.0).abs() < 0.001,
        "All-zero onehot should map to last index (F32=4.0), got {}",
        bpp
    );
}

// =========================================================================
// BottleneckClassifier with profiler bottleneck class preset
// =========================================================================

#[test]
fn test_bottleneck_classifier_with_preset_class() {
    let classifier = BottleneckClassifier::new();
    let mut features = TunerFeatures::builder().build();
    features.bottleneck_class = Some(BottleneckClass::AttentionBound);

    let prediction = classifier.predict(&features);
    assert_eq!(prediction.class, BottleneckClass::AttentionBound);
    assert_eq!(prediction.confidence, 0.95);
}

#[test]
fn test_bottleneck_classifier_heuristic_launch_bound() {
    let classifier = BottleneckClassifier::new();
    // batch_size=1 and no CUDA graphs -> launch bound
    let features = TunerFeatures::builder().batch_size(1).cuda_graphs(false).build();
    let prediction = classifier.predict(&features);
    assert_eq!(prediction.class, BottleneckClass::LaunchBound);
}

#[test]
fn test_bottleneck_classifier_heuristic_attention_bound() {
    let classifier = BottleneckClassifier::new();
    // Long sequence -> attention bound
    let features = TunerFeatures::builder().batch_size(4).seq_len(2048).build();
    let prediction = classifier.predict(&features);
    assert_eq!(prediction.class, BottleneckClass::AttentionBound);
}

#[test]
fn test_bottleneck_classifier_heuristic_memory_bound() {
    let classifier = BottleneckClassifier::new();
    // Normal batch size, short sequence -> memory bound
    let features = TunerFeatures::builder().batch_size(4).seq_len(64).build();
    let prediction = classifier.predict(&features);
    assert_eq!(prediction.class, BottleneckClass::MemoryBound);
}

// =========================================================================
// KernelClassifier various batch sizes
// =========================================================================

#[test]
fn test_kernel_classifier_large_batch() {
    let classifier = KernelClassifier::new();
    let features = TunerFeatures::builder().batch_size(8).build();
    let rec = classifier.predict(&features);
    assert_eq!(rec.top_kernel, KernelType::BatchedQ4K);
}

#[test]
fn test_kernel_classifier_medium_batch() {
    let classifier = KernelClassifier::new();
    let features = TunerFeatures::builder().batch_size(2).build();
    let rec = classifier.predict(&features);
    assert_eq!(rec.top_kernel, KernelType::VectorizedQ4K);
}

#[test]
fn test_kernel_classifier_single_no_cuda_graphs() {
    let classifier = KernelClassifier::new();
    let features = TunerFeatures::builder().batch_size(1).cuda_graphs(false).build();
    let rec = classifier.predict(&features);
    assert_eq!(rec.top_kernel, KernelType::CoalescedQ4K);
}

#[test]
fn test_kernel_classifier_single_with_cuda_graphs() {
    let classifier = KernelClassifier::new();
    let features = TunerFeatures::builder().batch_size(1).cuda_graphs(true).build();
    let rec = classifier.predict(&features);
    assert_eq!(rec.top_kernel, KernelType::VectorizedQ4K);
}

// =========================================================================
// TunerFeatures builder all options
// =========================================================================

#[test]
fn test_builder_all_options() {
    let features = TunerFeatures::builder()
        .model_params_b(7.0)
        .hidden_dim(4096)
        .num_layers(32)
        .num_heads(32)
        .head_dim(128)
        .vocab_size(32000)
        .batch_size(4)
        .seq_len(2048)
        .cuda_graphs(true)
        .kv_caches(4)
        .is_prefill(true)
        .quant_type(QuantType::Q6K)
        .kernel_type(KernelType::CoalescedQ6K)
        .gpu_mem_bw_gbs(1000.0)
        .gpu_compute_tflops(100.0)
        .gpu_sm_count(128)
        .gpu_l2_cache_mb(64.0)
        .is_zero_copy(true)
        .measured_tps(150.0)
        .build();

    assert!(features.validate().is_ok());
    assert_eq!(features.cuda_graphs, 1.0);
    assert_eq!(features.is_prefill, 1.0);
    assert_eq!(features.is_zero_copy, 1.0);
    assert!(features.measured_tps.is_some());
    assert_eq!(features.measured_tps.unwrap(), 150.0);
    assert_eq!(features.quant_type_onehot[4], 1.0); // Q6K index
    assert_eq!(features.kernel_type_onehot[6], 1.0); // CoalescedQ6K index

    let vec = features.to_vector();
    assert_eq!(vec.len(), TunerFeatures::DIM);
}
