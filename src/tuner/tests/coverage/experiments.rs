//! BrickTuner suggest_experiments, render, and serialization tests.

use crate::tuner::*;

// =========================================================================
// BrickTuner: suggest_experiments coverage (various bottleneck types)
// =========================================================================

#[test]
fn test_suggest_experiments_memory_bound_small_batch() {
    let tuner = BrickTuner::new();
    let features = TunerFeatures::builder().batch_size(1).build();

    let bottleneck = BottleneckPrediction {
        class: BottleneckClass::MemoryBound,
        confidence: 0.85,
        explanation: "Memory bound".to_string(),
        recommended_action: "Increase batch size".to_string(),
    };

    let suggestions = tuner.suggest_experiments(&features, &bottleneck);
    // Should suggest increasing batch size since batch_size < 8
    assert!(
        suggestions.iter().any(|s| matches!(s, ExperimentSuggestion::IncreaseBatchSize { .. })),
        "Should suggest increasing batch size for memory-bound with small batch"
    );
    // Should suggest trying BatchedQ4K kernel
    assert!(
        suggestions.iter().any(|s| matches!(s, ExperimentSuggestion::TryKernel { .. })),
        "Should suggest trying a kernel"
    );
}

#[test]
fn test_suggest_experiments_memory_bound_large_batch() {
    let tuner = BrickTuner::new();
    // batch_size_norm = 8/64 = 0.125, so batch_size = round(0.125*64) = 8
    let features = TunerFeatures::builder().batch_size(8).build();

    let bottleneck = BottleneckPrediction {
        class: BottleneckClass::MemoryBound,
        confidence: 0.85,
        explanation: "Memory bound".to_string(),
        recommended_action: "Increase batch size".to_string(),
    };

    let suggestions = tuner.suggest_experiments(&features, &bottleneck);
    // batch_size >= 8, so no batch size increase suggestion
    assert!(
        !suggestions.iter().any(|s| matches!(s, ExperimentSuggestion::IncreaseBatchSize { .. })),
        "Should NOT suggest increasing batch size when already >= 8"
    );
    // But should suggest EnableMultiKvCache since batch_size > 1
    assert!(
        suggestions.iter().any(|s| matches!(s, ExperimentSuggestion::EnableMultiKvCache { .. })),
        "Should suggest multi-KV cache for batch_size > 1"
    );
}

#[test]
fn test_suggest_experiments_launch_bound_no_cuda_graphs() {
    let tuner = BrickTuner::new();
    let features = TunerFeatures::builder().batch_size(1).cuda_graphs(false).build();

    let bottleneck = BottleneckPrediction {
        class: BottleneckClass::LaunchBound,
        confidence: 0.75,
        explanation: "Launch bound".to_string(),
        recommended_action: "Enable CUDA graphs".to_string(),
    };

    let suggestions = tuner.suggest_experiments(&features, &bottleneck);
    assert!(
        suggestions.iter().any(|s| matches!(s, ExperimentSuggestion::EnableCudaGraphs)),
        "Should suggest enabling CUDA graphs"
    );
    assert!(
        suggestions.iter().any(|s| matches!(
            s,
            ExperimentSuggestion::TryKernel { kernel: KernelType::FusedRmsNormQ4K }
        )),
        "Should suggest fused kernel"
    );
}

#[test]
fn test_suggest_experiments_launch_bound_with_cuda_graphs() {
    let tuner = BrickTuner::new();
    let features = TunerFeatures::builder().batch_size(1).cuda_graphs(true).build();

    let bottleneck = BottleneckPrediction {
        class: BottleneckClass::LaunchBound,
        confidence: 0.75,
        explanation: "Launch bound".to_string(),
        recommended_action: "Enable CUDA graphs".to_string(),
    };

    let suggestions = tuner.suggest_experiments(&features, &bottleneck);
    // Should NOT suggest CUDA graphs since already enabled
    assert!(
        !suggestions.iter().any(|s| matches!(s, ExperimentSuggestion::EnableCudaGraphs)),
        "Should NOT suggest CUDA graphs when already enabled"
    );
}

#[test]
fn test_suggest_experiments_attention_bound() {
    let tuner = BrickTuner::new();
    let features = TunerFeatures::builder().batch_size(1).seq_len(1024).build();

    let bottleneck = BottleneckPrediction {
        class: BottleneckClass::AttentionBound,
        confidence: 0.80,
        explanation: "Attention bound".to_string(),
        recommended_action: "Use Flash Decoding".to_string(),
    };

    let suggestions = tuner.suggest_experiments(&features, &bottleneck);
    assert!(
        suggestions.iter().any(|s| matches!(
            s,
            ExperimentSuggestion::TryKernel { kernel: KernelType::BatchedAttention }
        )),
        "Should suggest batched attention kernel"
    );
    assert!(
        suggestions.iter().any(|s| matches!(s, ExperimentSuggestion::ReduceSequenceLength { .. })),
        "Should suggest reducing sequence length"
    );
}

#[test]
fn test_suggest_experiments_compute_bound() {
    let tuner = BrickTuner::new();
    let features = TunerFeatures::builder().batch_size(1).build();

    let bottleneck = BottleneckPrediction {
        class: BottleneckClass::ComputeBound,
        confidence: 0.70,
        explanation: "Compute bound".to_string(),
        recommended_action: "Check for redundant computation".to_string(),
    };

    let suggestions = tuner.suggest_experiments(&features, &bottleneck);
    // ComputeBound falls into the default arm with batch_size < 4
    assert!(
        suggestions.iter().any(|s| matches!(s, ExperimentSuggestion::IncreaseBatchSize { .. })),
        "Default arm should suggest increasing batch size when < 4"
    );
}

#[test]
fn test_suggest_experiments_unknown_large_batch() {
    let tuner = BrickTuner::new();
    let features = TunerFeatures::builder().batch_size(4).build();

    let bottleneck = BottleneckPrediction {
        class: BottleneckClass::Unknown,
        confidence: 0.50,
        explanation: "Unknown".to_string(),
        recommended_action: "Run profiling".to_string(),
    };

    let suggestions = tuner.suggest_experiments(&features, &bottleneck);
    // batch_size >= 4, so default arm should NOT suggest increasing
    assert!(suggestions.is_empty(), "No suggestions when unknown bottleneck and batch_size >= 4");
}

// =========================================================================
// BrickTuner: render_panel, render_compact, render_comparison
// =========================================================================

#[test]
fn test_brick_tuner_render_panel() {
    let tuner = BrickTuner::new();
    let features = TunerFeatures::builder().batch_size(2).model_params_b(1.5).build();
    let rec = tuner.recommend(&features);
    let lines = tuner.render_panel(&rec);
    assert!(!lines.is_empty(), "Panel should have lines");
    // Should contain version info
    assert!(lines[0].contains("BrickTuner"), "First line should mention BrickTuner");
}

#[test]
fn test_brick_tuner_render_panel_few_suggestions() {
    let tuner = BrickTuner::new();
    let _features = TunerFeatures::builder().batch_size(4).build();

    let bottleneck = BottleneckPrediction {
        class: BottleneckClass::Unknown,
        confidence: 0.50,
        explanation: "Unknown".to_string(),
        recommended_action: "Run profiling".to_string(),
    };

    // Create a recommendation with zero suggestions
    let rec = TunerRecommendation {
        throughput: ThroughputPrediction {
            predicted_tps: 100.0,
            confidence: 0.85,
            top_features: vec![],
        },
        kernel: KernelRecommendation {
            top_kernel: KernelType::TiledQ4K,
            confidence: 0.90,
            alternatives: vec![],
        },
        bottleneck,
        model_version: "1.0.0".to_string(),
        confidence_overall: 0.75,
        suggested_experiments: vec![], // No suggestions
    };

    let lines = tuner.render_panel(&rec);
    // Should pad to 3 empty suggestion lines
    assert!(lines.len() >= 10, "Panel should have padding for missing suggestions");
}

#[test]
fn test_brick_tuner_render_compact() {
    let tuner = BrickTuner::new();
    let features = TunerFeatures::builder().batch_size(2).model_params_b(1.5).build();
    let rec = tuner.recommend(&features);
    let compact = tuner.render_compact(&rec);
    assert!(compact.contains("Tuner:"), "Compact should start with 'Tuner:'");
    assert!(compact.contains("tok/s"), "Compact should mention tok/s");
}

#[test]
fn test_brick_tuner_render_comparison_excellent() {
    let tuner = BrickTuner::new();
    let features = TunerFeatures::builder().batch_size(2).model_params_b(1.5).build();
    let rec = tuner.recommend(&features);
    // Give actual_tps close to predicted for "Excellent"
    let actual_tps = rec.throughput.predicted_tps * 0.98;
    let comparison = tuner.render_comparison(&rec, actual_tps);
    assert_eq!(comparison.len(), 2);
    assert!(comparison[0].contains("Predicted"));
    assert!(comparison[0].contains("Actual"));
}

#[test]
fn test_brick_tuner_render_comparison_poor() {
    let tuner = BrickTuner::new();
    let features = TunerFeatures::builder().batch_size(2).model_params_b(1.5).build();
    let rec = tuner.recommend(&features);
    // Give very different actual_tps for "Poor"
    let actual_tps = rec.throughput.predicted_tps * 0.5;
    let comparison = tuner.render_comparison(&rec, actual_tps);
    assert_eq!(comparison.len(), 2);
    assert!(comparison[1].contains("Poor"));
}

#[test]
fn test_brick_tuner_render_comparison_zero_actual() {
    let tuner = BrickTuner::new();
    let features = TunerFeatures::builder().batch_size(2).build();
    let rec = tuner.recommend(&features);
    let comparison = tuner.render_comparison(&rec, 0.0);
    assert_eq!(comparison.len(), 2);
    // With actual=0, error_pct=0.0, so "Excellent"
    assert!(comparison[1].contains("Excellent"));
}

// =========================================================================
// BrickTuner: JSON serialization round-trip
// =========================================================================

#[test]
fn test_brick_tuner_json_roundtrip() {
    let tuner = BrickTuner::new();
    let json = tuner.to_json().expect("Serialization should succeed");
    let loaded = BrickTuner::from_json(&json).expect("Deserialization should succeed");
    assert_eq!(tuner.version(), loaded.version());
    assert_eq!(tuner.throughput_mape(), loaded.throughput_mape());
}

#[test]
fn test_brick_tuner_from_json_invalid() {
    let result = BrickTuner::from_json("not valid json");
    assert!(result.is_err());
}

// =========================================================================
// BrickTuner: APR save/load round-trip
// =========================================================================

#[test]
fn test_brick_tuner_apr_roundtrip() {
    let tuner = BrickTuner::new();
    let dir = std::env::temp_dir().join("trueno_test_apr_roundtrip");
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join("test_tuner.apr");

    tuner.save_apr(&path).expect("Save should succeed");
    let loaded = BrickTuner::load_apr(&path).expect("Load should succeed");
    assert_eq!(tuner.version(), loaded.version());
    assert_eq!(tuner.sample_count, loaded.sample_count);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_brick_tuner_load_apr_bad_magic() {
    let dir = std::env::temp_dir().join("trueno_test_bad_magic");
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join("bad_magic.apr");

    // Write file with wrong magic bytes
    std::fs::write(&path, b"BAD1xxxxxxxxxxxx").expect("write");
    let result = BrickTuner::load_apr(&path);
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("Invalid") || err_msg.contains("magic"));

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_brick_tuner_load_apr_crc_mismatch() {
    let tuner = BrickTuner::new();
    let dir = std::env::temp_dir().join("trueno_test_crc_mismatch");
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join("bad_crc.apr");

    tuner.save_apr(&path).expect("Save should succeed");

    // Corrupt one byte in the file (after the magic and length)
    let mut data = std::fs::read(&path).expect("read");
    if data.len() > 12 {
        data[12] ^= 0xFF; // Flip bits in JSON payload
    }
    std::fs::write(&path, &data).expect("write");

    let result = BrickTuner::load_apr(&path);
    assert!(result.is_err());

    let _ = std::fs::remove_dir_all(&dir);
}

// =========================================================================
// TunerDataCollector: APR save/load round-trip
// =========================================================================

#[test]
fn test_data_collector_apr_roundtrip() {
    let mut collector = TunerDataCollector::new();
    let features = TunerFeatures::builder().model_params_b(7.0).batch_size(4).build();
    collector.samples.push(TrainingSample {
        features,
        throughput_tps: 200.0,
        best_kernel: KernelType::VectorizedQ4K,
        bottleneck: BottleneckClass::MemoryBound,
        timestamp: "12345".to_string(),
        hardware_id: "test-hw".to_string(),
    });

    let dir = std::env::temp_dir().join("trueno_test_collector_apr");
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join("test_collector.apr");

    collector.save_apr(&path).expect("Save should succeed");
    let loaded = TunerDataCollector::load_apr(&path).expect("Load should succeed");
    assert_eq!(loaded.samples().len(), 1);
    assert_eq!(loaded.samples()[0].throughput_tps, 200.0);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_data_collector_load_apr_bad_magic() {
    let dir = std::env::temp_dir().join("trueno_test_collector_bad_magic");
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join("bad.apr");
    std::fs::write(&path, b"XXXXABCD").expect("write");

    let result = TunerDataCollector::load_apr(&path);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("APR2"));

    let _ = std::fs::remove_dir_all(&dir);
}
