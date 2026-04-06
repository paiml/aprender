
#[test]
fn test_pipeline_parallel_invalid_stage() {
    let result = PipelineParallel::new(4, 10, 24, 4);
    assert!(result.is_err());
}

#[test]
fn test_pipeline_parallel_bubble_ratio_zero_microbatches() {
    let pp = PipelineParallel::new(4, 0, 24, 4).expect("test");
    let ratio = pp.bubble_ratio(0);
    assert_eq!(ratio, 1.0);
}

#[test]
fn test_pipeline_parallel_bubble_ratio_many_microbatches() {
    let pp = PipelineParallel::new(4, 0, 24, 4).expect("test");
    // With many micro-batches, bubble ratio approaches 0
    let ratio = pp.bubble_ratio(1000);
    assert!(ratio < 0.01);
}

#[test]
fn test_pipeline_parallel_record_multiple_microbatches() {
    let mut pp = PipelineParallel::new(4, 0, 24, 4).expect("test");

    // Record several micro-batches
    pp.record_micro_batch(10.0);
    pp.record_micro_batch(20.0);
    pp.record_micro_batch(30.0);

    let stats = pp.stats();
    assert_eq!(stats.micro_batches_processed, 3);
    assert_eq!(stats.forward_passes, 3);
    // Running average: (10 + 20 + 30) / 3 = 20
    assert!((stats.avg_stage_latency_ms - 20.0).abs() < 0.1);
}

#[test]
fn test_pipeline_stage_clone_serialize() {
    let stage = PipelineStage {
        index: 1,
        start_layer: 6,
        end_layer: 12,
        num_layers: 6,
    };
    let cloned = stage.clone();
    assert_eq!(cloned.index, 1);

    let json = serde_json::to_string(&stage).expect("serialize");
    let deserialized: PipelineStage = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(deserialized.num_layers, 6);
}

// =========================================================================
// Additional Coverage Tests: ParallelConfig Ranks
// =========================================================================

#[test]
fn test_parallel_config_is_tp_last() {
    // tp_size=2, last TP rank is 1
    let config = ParallelConfig::new(2, 1, 1, 1).expect("test");
    assert!(config.is_tp_last());
    assert!(!config.is_tp_first());
}

#[test]
fn test_parallel_config_is_pp_last() {
    // pp_size=2, last PP stage is at rank 1 (with tp_size=1)
    let config = ParallelConfig::new(1, 2, 1, 1).expect("test");
    assert!(config.is_pp_last());
    assert!(!config.is_pp_first());
}

#[test]
fn test_parallel_config_complex_ranks() {
    // World: tp=2, pp=3, dp=2 => 12 ranks
    // Rank 7: tp_rank = 7 % 2 = 1, pp_stage = (7/2) % 3 = 0, dp_rank = 7/(2*3) = 1
    let config = ParallelConfig::new(2, 3, 2, 7).expect("test");
    assert_eq!(config.tp_rank(), 1);
    assert_eq!(config.pp_stage(), 0);
    assert_eq!(config.dp_rank(), 1);
}

#[test]
fn test_parallel_config_default() {
    let config = ParallelConfig::default();
    assert_eq!(config.tp_size, 1);
    assert_eq!(config.pp_size, 1);
    assert_eq!(config.dp_size, 1);
    assert_eq!(config.rank, 0);
}

#[test]
fn test_parallel_config_serialization() {
    let config = ParallelConfig::new(2, 2, 2, 3).expect("test");
    let json = serde_json::to_string(&config).expect("serialize");
    let deserialized: ParallelConfig = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(deserialized.tp_size, 2);
    assert_eq!(deserialized.rank, 3);
}

// =========================================================================
// Additional Coverage Tests: DistributedContext
// =========================================================================

#[test]
fn test_distributed_context_pipeline_parallel_mut() {
    let config = ParallelConfig::new(1, 4, 1, 0).expect("test");
    let mut ctx = DistributedContext::new(config).expect("test");

    ctx.init_pipeline(24, 4).expect("test");

    // Use pipeline_parallel_mut
    if let Some(pp) = ctx.pipeline_parallel_mut() {
        pp.record_micro_batch(15.0);
    }

    let stats = ctx.pipeline_parallel().expect("pp").stats();
    assert_eq!(stats.micro_batches_processed, 1);
}

#[test]
fn test_distributed_context_config_accessor() {
    let config = ParallelConfig::new(2, 2, 2, 5).expect("test");
    let ctx = DistributedContext::new(config).expect("test");

    let cfg = ctx.config();
    assert_eq!(cfg.tp_size, 2);
    assert_eq!(cfg.rank, 5);
}

#[test]
fn test_distributed_context_init_pipeline_single_pp() {
    // When pp_size = 1, init_pipeline should not create PipelineParallel
    let config = ParallelConfig::new(2, 1, 1, 0).expect("test");
    let mut ctx = DistributedContext::new(config).expect("test");

    ctx.init_pipeline(24, 4).expect("test");
    assert!(ctx.pipeline_parallel().is_none());
}

#[test]
fn test_distributed_context_debug() {
    let config = ParallelConfig::single();
    let ctx = DistributedContext::new(config).expect("test");
    let debug_str = format!("{:?}", ctx);
    assert!(debug_str.contains("config"));
    assert!(debug_str.contains("initialized"));
}

// =========================================================================
// Additional Coverage Tests: ZeroOffload Serialization
// =========================================================================

#[test]
fn test_zero_offload_serialization() {
    let zero = ZeroOffload::inference();
    let json = serde_json::to_string(&zero).expect("serialize");
    let deserialized: ZeroOffload = serde_json::from_str(&json).expect("deserialize");
    assert!(deserialized.offload_params);
    assert!(deserialized.offload_activations);
}

// =========================================================================
// Additional Coverage Tests: ParallelError NotInitialized
// =========================================================================

#[test]
fn test_parallel_error_not_initialized_display() {
    let err = ParallelError::NotInitialized;
    let msg = err.to_string();
    assert!(msg.contains("not initialized"));
}

#[test]
fn test_parallel_error_pipeline_error_display() {
    let err = ParallelError::PipelineError("stage mismatch".to_string());
    let msg = err.to_string();
    assert!(msg.contains("stage mismatch"));
}

#[test]
fn test_parallel_error_invalid_world_size_display() {
    let err = ParallelError::InvalidWorldSize(0);
    let msg = err.to_string();
    assert!(msg.contains("0"));
}

// =========================================================================
// Additional Coverage Tests: ParallelTensor Operations
// =========================================================================

#[test]
fn test_parallel_tensor_zeros_various_shapes() {
    let t1 = ParallelTensor::zeros(vec![1]);
    assert_eq!(t1.numel(), 1);

    let t2 = ParallelTensor::zeros(vec![2, 3, 4]);
    assert_eq!(t2.numel(), 24);
    assert_eq!(t2.sum(), 0.0);
}

#[test]
fn test_parallel_tensor_sum_negative_values() {
    let tensor = ParallelTensor::new(vec![4], vec![-1.0, 2.0, -3.0, 4.0]).expect("test");
    assert_eq!(tensor.sum(), 2.0);
}

#[test]
#[allow(clippy::many_single_char_names)]
fn test_parallel_tensor_large_matmul() {
    // Larger matrix multiplication
    let m = 4;
    let k = 8;
    let n = 4;
    let a = ParallelTensor::new(vec![m, k], vec![1.0; m * k]).expect("test");
    let b = ParallelTensor::new(vec![k, n], vec![1.0; k * n]).expect("test");
    let c = a.matmul(&b).expect("test");

    assert_eq!(c.shape, vec![m, n]);
    // Each element should be k (sum of 1.0 * 1.0 k times)
    assert_eq!(c.data[0], k as f32);
}

#[test]
fn test_parallel_tensor_clone() {
    let tensor = ParallelTensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).expect("test");
    let cloned = tensor.clone();
    assert_eq!(cloned.shape, tensor.shape);
    assert_eq!(cloned.data, tensor.data);
}

#[test]
fn test_parallel_tensor_debug() {
    let tensor = ParallelTensor::new(vec![2], vec![1.0, 2.0]).expect("test");
    let debug_str = format!("{:?}", tensor);
    assert!(debug_str.contains("shape"));
    assert!(debug_str.contains("data"));
}

// =========================================================================
// Additional Coverage Tests: PipelineStats Default
// =========================================================================

#[test]
fn test_pipeline_stats_default() {
    let stats = PipelineStats::default();
    assert_eq!(stats.micro_batches_processed, 0);
    assert_eq!(stats.forward_passes, 0);
    assert_eq!(stats.bubble_time_ms, 0.0);
    assert_eq!(stats.avg_stage_latency_ms, 0.0);
}

#[test]
fn test_pipeline_stats_serialization() {
    let stats = PipelineStats {
        micro_batches_processed: 50,
        forward_passes: 50,
        bubble_time_ms: 2.5,
        avg_stage_latency_ms: 8.0,
    };
    let json = serde_json::to_string(&stats).expect("serialize");
    let deserialized: PipelineStats = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(deserialized.micro_batches_processed, 50);
    assert!((deserialized.avg_stage_latency_ms - 8.0).abs() < 0.01);
}
