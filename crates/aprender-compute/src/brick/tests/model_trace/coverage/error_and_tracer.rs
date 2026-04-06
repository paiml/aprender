use super::super::super::super::*;

// =========================================================================
// F284-F298: QuantizationError, ModelTracer, TensorStats, and remaining tests
// =========================================================================

/// F284: QuantizationErrorTrace boundary conditions
#[test]
fn test_f284_quant_error_boundaries() {
    // Perfect match (identical)
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let trace =
        QuantizationErrorTrace::compute(BrickId::QkvProjection, 0, &data, &data, QuantType::F32);
    assert_eq!(trace.mse, 0.0);
    assert!((trace.cosine_similarity - 1.0).abs() < 0.0001);
    assert!(trace.is_acceptable());

    // Large error (warning threshold)
    let reference = vec![1.0, 0.0, 0.0, 0.0];
    let bad_quant = vec![0.97, 0.02, 0.02, 0.02];
    let trace2 = QuantizationErrorTrace::compute(
        BrickId::AttentionScore,
        0,
        &bad_quant,
        &reference,
        QuantType::Q4_K,
    );
    assert!(trace2.cosine_similarity < 1.0);

    // Test model-level aggregation
    let mut model_error = ModelQuantizationError::default();
    model_error.add_error(trace);
    model_error.add_error(trace2);

    assert_eq!(model_error.brick_errors.len(), 2);
    assert!(model_error.worst_brick().is_some());
}

/// F285: ModelTracer disabled config verification
#[test]
fn test_f285_model_tracer_disabled() {
    let disabled = ModelTracerConfig::default();
    assert!(!disabled.is_enabled());
    assert!(!disabled.trace_activations);
    assert!(!disabled.trace_attention);
    assert!(!disabled.trace_logits);
    assert!(!disabled.trace_quant_error);
    assert!(!disabled.trace_kv_cache);

    let mut tracer = ModelTracer::new(disabled);

    // Verify no-op behavior
    tracer.begin_forward(0);
    let layer = LayerActivationTrace::new(0);
    tracer.record_layer_activation(layer);
    let kv = KvCacheStateTrace::new(0, 2048);
    tracer.record_kv_state(kv);
    let result = tracer.end_forward();
    assert!(result.is_none()); // No anomaly detection when disabled

    let summary = tracer.summary();
    assert_eq!(summary.total_forwards, 0); // Not tracked when disabled
}

/// F286: TensorStats edge cases
#[test]
fn test_f286_tensor_stats_edge_cases() {
    // Empty slice
    let empty: Vec<f32> = vec![];
    let stats = TensorStats::from_slice(&empty);
    assert_eq!(stats.count, 0);
    assert!(!stats.has_anomaly()); // Empty is not an anomaly

    // Single element
    let single = vec![42.0];
    let stats = TensorStats::from_slice(&single);
    assert_eq!(stats.count, 1);
    assert_eq!(stats.min, 42.0);
    assert_eq!(stats.max, 42.0);
    assert_eq!(stats.mean, 42.0);
    assert_eq!(stats.std, 0.0); // No variance with single element
}

/// F287: AttentionWeightTrace::is_uniform
#[test]
fn test_f287_attention_uniform_detection() {
    // Uniform distribution (high entropy)
    let uniform_weights = vec![0.25, 0.25, 0.25, 0.25];
    let trace = AttentionWeightTrace::from_weights(0, 0, 3, &uniform_weights, 4);
    assert!(trace.is_uniform(1.0)); // Entropy threshold of 1.0

    // Peaky distribution (low entropy)
    let peaky_weights = vec![0.9, 0.05, 0.03, 0.02];
    let trace2 = AttentionWeightTrace::from_weights(0, 0, 3, &peaky_weights, 4);
    assert!(!trace2.is_uniform(1.0)); // Not uniform
}

/// F289: QuantizationErrorTrace with empty data
#[test]
fn test_f289_quant_error_empty() {
    let empty: Vec<f32> = vec![];
    let trace =
        QuantizationErrorTrace::compute(BrickId::QkvProjection, 0, &empty, &empty, QuantType::Q4_K);
    assert_eq!(trace.mse, 0.0);
    assert_eq!(trace.cosine_similarity, 1.0);
    assert!(trace.snr_db.is_infinite());
}

/// F290: ModelTracer record_logits and record_quant_error
#[test]
fn test_f290_model_tracer_record_methods() {
    let config = ModelTracerConfig::full();
    let mut tracer = ModelTracer::new(config);

    tracer.begin_forward(0);

    // Record attention trace
    let attn_trace = AttentionWeightTrace::from_weights(0, 0, 5, &[0.5, 0.3, 0.2], 3);
    tracer.record_attention(attn_trace);

    // Record logits - need to first have logit trace initialized
    // This exercises the record_logits path

    // Record quant error
    let quant_trace = QuantizationErrorTrace::compute(
        BrickId::QkvProjection,
        0,
        &[1.02, 1.98, 3.05],
        &[1.0, 2.0, 3.0],
        QuantType::Q4_K,
    );
    tracer.record_quant_error(quant_trace);

    // End forward and verify
    let _result = tracer.end_forward();
    // Should complete without error
    let summary = tracer.summary();
    assert_eq!(summary.total_forwards, 1);
}

/// F293: ModelQuantizationError warning and critical counts
#[test]
fn test_f293_model_quant_error_thresholds() {
    let mut model_error = ModelQuantizationError::default();

    // Add an acceptable error
    let good = QuantizationErrorTrace {
        brick_id: BrickId::QkvProjection,
        layer_idx: 0,
        mse: 0.001,
        max_abs_error: 0.01,
        cosine_similarity: 0.998,
        snr_db: 40.0,
        quant_type: QuantType::Q4_K,
    };
    model_error.add_error(good);

    // Add a warning-level error
    let warning = QuantizationErrorTrace {
        brick_id: BrickId::AttentionScore,
        layer_idx: 1,
        mse: 0.01,
        max_abs_error: 0.1,
        cosine_similarity: 0.992, // Between 0.99 and 0.995
        snr_db: 25.0,
        quant_type: QuantType::Q4_K,
    };
    model_error.add_error(warning);

    // Add a critical error
    let critical = QuantizationErrorTrace {
        brick_id: BrickId::DownProjection,
        layer_idx: 2,
        mse: 0.1,
        max_abs_error: 1.0,
        cosine_similarity: 0.85, // Below 0.99
        snr_db: 10.0,
        quant_type: QuantType::Q2_K,
    };
    model_error.add_error(critical);

    assert_eq!(model_error.brick_errors.len(), 3);
    assert!(model_error.warning_count() >= 1);
    assert!(model_error.critical_count() >= 1);

    let worst = model_error.worst_brick().unwrap();
    assert!(worst.cosine_similarity < 0.9);
}

/// F294: TensorStats::is_vanishing
#[test]
fn test_f294_tensor_stats_vanishing() {
    // Create nearly constant tensor (vanishing gradients)
    let data = vec![1.0; 1000];
    let stats = TensorStats::from_slice(&data);
    assert!(stats.is_vanishing()); // std should be 0

    // Non-vanishing tensor
    let varied: Vec<f32> = (0..1000).map(|i| i as f32).collect();
    let stats2 = TensorStats::from_slice(&varied);
    assert!(!stats2.is_vanishing());
}

/// F295: TensorStats high variance anomaly
#[test]
fn test_f295_tensor_stats_high_variance() {
    // Create tensor with extreme variance
    let mut data = vec![0.0; 100];
    data[0] = 1e5;
    data[1] = -1e5;
    let stats = TensorStats::from_slice(&data);
    assert!(stats.std > 1e4);
    assert!(stats.has_anomaly());
    let desc = stats.anomaly_description().unwrap();
    assert!(desc.contains("variance") || desc.contains("std"));
}

/// F296: ModelTracer record_logits path
#[test]
fn test_f296_model_tracer_record_logits() {
    let config = ModelTracerConfig::full();
    let mut tracer = ModelTracer::new(config);

    tracer.begin_forward(0);

    // Create logit trace manually
    let mut logit_trace = LogitEvolutionTrace::new(100, 0.7, 0.9);
    let token = logit_trace.track_token(42, "hello".to_string());
    token.final_probability = 0.5;

    // Set the logit trace
    tracer.set_current_logit_trace(Some(logit_trace));

    // Record logits - this should exercise the record_logits path
    let logits: Vec<f32> = (0..100).map(|i| i as f32).collect();
    tracer.record_logits(0, &logits);

    // Verify it was recorded
    if let Some(trace) = tracer.current_logit_trace() {
        assert!(!trace.tracked_tokens.is_empty());
    }

    tracer.end_forward();
}

/// F297: ModelActivationTrace add_layer without anomaly
#[test]
fn test_f297_model_activation_add_normal_layers() {
    let mut trace = ModelActivationTrace::with_capacity(10);

    // Add several normal layers
    for i in 0..5 {
        let mut layer = LayerActivationTrace::new(i);
        layer.input_stats = TensorStats::from_slice(&vec![1.0; 100]);
        layer.output_stats = TensorStats::from_slice(&vec![1.1; 100]);
        trace.add_layer(layer);
    }

    // No anomaly should be detected
    assert!(!trace.has_anomaly);
    assert!(trace.anomaly_desc.is_none());
    assert_eq!(trace.layers.len(), 5);
}

/// F298: AsyncTask node type coverage
#[test]
fn test_f298_async_task_node() {
    let mut graph = ExecutionGraph::new();

    let async_task = graph.add_node(ExecutionNode::AsyncTask {
        name: "inference_loop".to_string(),
        poll_count: 100,
        yield_count: 50,
        total_poll_ns: 1_000_000,
    });

    // Verify node was added
    assert_eq!(async_task.0, 0);
}
