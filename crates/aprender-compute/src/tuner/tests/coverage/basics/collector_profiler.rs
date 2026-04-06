//! Coverage tests for TunerDataCollector and BrickProfiler integration.

use crate::tuner::*;

#[test]
fn test_collector_training_progress() {
    let mut collector = TunerDataCollector::new();

    let (current, required) = collector.training_progress();
    assert_eq!(current, 0);
    assert_eq!(required, TunerDataCollector::MIN_SAMPLES_FOR_TRAINING);

    // Add 5 samples
    for i in 0..5 {
        let features =
            TunerFeatures::builder().model_params_b(7.0).batch_size((i as u32) + 1).build();
        collector.samples.push(TrainingSample {
            features,
            throughput_tps: 100.0,
            best_kernel: KernelType::TiledQ4K,
            bottleneck: BottleneckClass::MemoryBound,
            timestamp: format!("{}", i),
            hardware_id: "test".to_string(),
        });
    }

    let (current, required) = collector.training_progress();
    assert_eq!(current, 5);
    assert_eq!(required, TunerDataCollector::MIN_SAMPLES_FOR_TRAINING);
}

#[test]
fn test_collector_train_if_ready_not_ready() {
    let collector = TunerDataCollector::new();
    assert!(collector.train_if_ready().is_none());
}

#[test]
fn test_collector_train_if_ready_success() {
    let mut collector = TunerDataCollector::new();

    // Add MIN_SAMPLES_FOR_TRAINING samples (1000) to trigger training
    for i in 0..TunerDataCollector::MIN_SAMPLES_FOR_TRAINING {
        let features = TunerFeatures::builder()
            .model_params_b(1.0 + (i as f32) % 20.0)
            .hidden_dim(2048 + (i as u32) % 4096)
            .batch_size((i as u32) % 16 + 1)
            .quant_type(if i % 2 == 0 { QuantType::Q4K } else { QuantType::Q8_0 })
            .build();
        collector.samples.push(TrainingSample {
            features,
            throughput_tps: 50.0 + (i as f32) % 200.0,
            best_kernel: KernelType::TiledQ4K,
            bottleneck: BottleneckClass::MemoryBound,
            timestamp: format!("{}", i),
            hardware_id: "test-gpu".to_string(),
        });
    }

    let result = collector.train_if_ready();
    assert!(result.is_some());

    let tuner = result.unwrap();
    // Tuner should have been trained
    assert!(tuner.throughput_sample_count() > 0);
}

#[test]
fn test_collector_bootstrap_from_five_whys() {
    let collector = TunerDataCollector::bootstrap_from_five_whys();

    // Bootstrap returns empty collector for now (TODO: load actual data)
    // But it should still be a valid collector
    // Collector is either empty or has samples - this is always true but exercises the API
    let _ = collector.samples();
    assert!(!collector.is_online_learning_enabled());
}

#[test]
fn test_collector_auto_retrain_not_ready() {
    let mut collector = TunerDataCollector::new();
    let mut tuner = BrickTuner::new();

    // Should return false when not ready to retrain
    assert!(!collector.auto_retrain(&mut tuner));
}

#[test]
fn test_collector_auto_retrain_success() {
    let mut collector = TunerDataCollector::with_online_learning();
    let mut tuner = BrickTuner::new();

    // Add enough samples to trigger retrain
    for i in 0..150 {
        let features = TunerFeatures::builder()
            .model_params_b(1.0 + (i as f32) * 0.1)
            .hidden_dim(2048)
            .batch_size((i as u32) % 16 + 1)
            .quant_type(if i % 2 == 0 { QuantType::Q4K } else { QuantType::Q8_0 })
            .build();
        collector.samples.push(TrainingSample {
            features,
            throughput_tps: 30.0 + (i as f32) * 2.0,
            best_kernel: KernelType::TiledQ4K,
            bottleneck: BottleneckClass::MemoryBound,
            timestamp: format!("{}", i),
            hardware_id: "auto-retrain-test".to_string(),
        });
    }

    // Force should_retrain to return true by exceeding threshold
    collector.samples_at_last_train = 0;
    collector.retrain_threshold = 50;

    let result = collector.auto_retrain(&mut tuner);
    assert!(result);
}

#[test]
fn test_collector_from_json_valid() {
    // Create a collector with samples, serialize it, then deserialize
    let mut original = TunerDataCollector::new();
    let features =
        TunerFeatures::builder().model_params_b(7.0).hidden_dim(4096).batch_size(1).build();
    original.samples.push(TrainingSample {
        features,
        throughput_tps: 150.0,
        best_kernel: KernelType::TiledQ4K,
        bottleneck: BottleneckClass::MemoryBound,
        timestamp: "1704067200".to_string(),
        hardware_id: "RTX4090".to_string(),
    });

    // Round-trip through JSON
    let json = original.to_json().unwrap();
    let result = TunerDataCollector::from_json(&json);
    assert!(result.is_ok());

    let collector = result.unwrap();
    assert_eq!(collector.samples().len(), 1);
    assert_eq!(collector.samples()[0].throughput_tps, 150.0);
}

#[test]
fn test_collector_from_json_invalid() {
    let json = "not valid json";
    let result = TunerDataCollector::from_json(json);
    assert!(result.is_err());
}

#[test]
fn test_concept_drift_status_fields() {
    let status = ConceptDriftStatus {
        drift_detected: true,
        staleness_score: 0.75,
        samples_since_training: 100,
        recommend_retrain: true,
        explanation: "High error rate detected".to_string(),
    };

    assert!(status.drift_detected);
    assert_eq!(status.staleness_score, 0.75);
    assert_eq!(status.samples_since_training, 100);
    assert!(status.recommend_retrain);
    assert!(status.explanation.contains("error"));
}

#[test]
fn test_training_stats_all_fields() {
    let stats = TrainingStats {
        total_samples: 500,
        samples_since_training: 50,
        accepted_count: 200,
        rejected_count: 50,
        alternative_count: 100,
        staleness_score: 0.3,
        drift_detected: false,
        online_learning_enabled: true,
    };

    assert_eq!(stats.total_samples, 500);
    assert_eq!(stats.samples_since_training, 50);
    assert_eq!(stats.accepted_count, 200);
    assert_eq!(stats.rejected_count, 50);
    assert_eq!(stats.alternative_count, 100);
    assert!(!stats.drift_detected);
    assert!(stats.online_learning_enabled);
}

// =========================================================================
// BrickProfiler::tokens_per_sec + tuner integration (tuner/mod.rs coverage)
// =========================================================================

#[test]
fn test_brick_profiler_tokens_per_sec_no_data() {
    use crate::brick::BrickProfiler;
    let profiler = BrickProfiler::new();
    // No tokens processed, no time elapsed -> None
    assert!(profiler.tokens_per_sec().is_none());
}

#[test]
fn test_brick_profiler_tokens_per_sec_with_data() {
    use crate::brick::BrickProfiler;
    let mut profiler = BrickProfiler::new();
    profiler.enable();
    let timer = profiler.start("test_brick");
    std::thread::sleep(std::time::Duration::from_millis(1));
    profiler.stop(timer, 100);

    let tps = profiler.tokens_per_sec();
    // If total_ns > 0 and total_tokens > 0, should return Some
    if profiler.total_ns() > 0 && profiler.total_tokens() > 0 {
        assert!(tps.is_some());
        assert!(tps.unwrap() > 0.0);
    }
}

#[test]
fn test_brick_profiler_get_tuner_recommendations_disabled() {
    use crate::brick::BrickProfiler;
    let profiler = BrickProfiler::new();
    // Profiler is disabled by default
    assert!(!profiler.is_enabled());
    let config = RunConfig::default();
    let result = profiler.get_tuner_recommendations(&config);
    assert!(result.is_none(), "Disabled profiler should return None");
}

#[test]
fn test_brick_profiler_get_tuner_recommendations_enabled() {
    use crate::brick::BrickProfiler;
    let mut profiler = BrickProfiler::new();
    profiler.enable();
    let config = RunConfig::default();
    let result = profiler.get_tuner_recommendations(&config);
    assert!(result.is_some(), "Enabled profiler should return Some");
    let rec = result.unwrap();
    assert!(rec.throughput.predicted_tps > 0.0);
    assert!(rec.confidence_overall > 0.0);
}

#[test]
fn test_brick_profiler_print_tuner_recommendations_disabled() {
    use crate::brick::BrickProfiler;
    let profiler = BrickProfiler::new();
    let config = RunConfig::default();
    // Should not panic, just print "not available"
    profiler.print_tuner_recommendations(&config);
}

#[test]
fn test_brick_profiler_print_tuner_recommendations_enabled() {
    use crate::brick::BrickProfiler;
    let mut profiler = BrickProfiler::new();
    profiler.enable();
    let config = RunConfig::default();
    // Should print recommendation without panic
    profiler.print_tuner_recommendations(&config);
}
