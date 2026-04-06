use super::*;

// =========================================================================
// ParallelConfig Tests
// =========================================================================

#[test]
fn test_parallel_config_new() {
    let config = ParallelConfig::new(2, 2, 2, 0).expect("test");
    assert_eq!(config.tp_size, 2);
    assert_eq!(config.pp_size, 2);
    assert_eq!(config.dp_size, 2);
    assert_eq!(config.world_size, 8);
    assert_eq!(config.rank, 0);
}

#[test]
fn test_parallel_config_single() {
    let config = ParallelConfig::single();
    assert_eq!(config.tp_size, 1);
    assert_eq!(config.pp_size, 1);
    assert_eq!(config.dp_size, 1);
    assert_eq!(config.world_size, 1);
    assert_eq!(config.rank, 0);
}

#[test]
fn test_parallel_config_invalid_rank() {
    let result = ParallelConfig::new(2, 2, 2, 100);
    assert!(result.is_err());
}

#[test]
fn test_parallel_config_invalid_world_size() {
    let result = ParallelConfig::new(0, 0, 0, 0);
    assert!(result.is_err());
}

#[test]
fn test_parallel_config_ranks() {
    // World size = 2 * 2 * 2 = 8
    // Rank 5: tp_rank=1, pp_stage=0, dp_rank=1
    let config = ParallelConfig::new(2, 2, 2, 5).expect("test");
    assert_eq!(config.tp_rank(), 1);
    assert_eq!(config.pp_stage(), 0);
    assert_eq!(config.dp_rank(), 1);
}

#[test]
fn test_parallel_config_first_last_checks() {
    let config = ParallelConfig::new(2, 2, 1, 0).expect("test");
    assert!(config.is_tp_first());
    assert!(!config.is_tp_last());
    assert!(config.is_pp_first());
    assert!(!config.is_pp_last());
}

// =========================================================================
// ParallelTensor Tests
// =========================================================================

#[test]
fn test_parallel_tensor_new() {
    let tensor = ParallelTensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("test");
    assert_eq!(tensor.shape, vec![2, 3]);
    assert_eq!(tensor.numel(), 6);
}

#[test]
fn test_parallel_tensor_zeros() {
    let tensor = ParallelTensor::zeros(vec![2, 3]);
    assert_eq!(tensor.sum(), 0.0);
}

#[test]
fn test_parallel_tensor_narrow_rows() {
    let tensor = ParallelTensor::new(vec![4, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        .expect("test");
    let narrowed = tensor.narrow(0, 1, 2).expect("test");
    assert_eq!(narrowed.shape, vec![2, 2]);
    assert_eq!(narrowed.data, vec![3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn test_parallel_tensor_narrow_cols() {
    let tensor = ParallelTensor::new(vec![2, 4], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        .expect("test");
    let narrowed = tensor.narrow(1, 1, 2).expect("test");
    assert_eq!(narrowed.shape, vec![2, 2]);
    assert_eq!(narrowed.data, vec![2.0, 3.0, 6.0, 7.0]);
}

#[test]
fn test_parallel_tensor_transpose() {
    let tensor = ParallelTensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("test");
    let transposed = tensor.transpose().expect("test");
    assert_eq!(transposed.shape, vec![3, 2]);
    assert_eq!(transposed.data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

#[test]
fn test_parallel_tensor_matmul() {
    // [1, 2] @ [[1, 2], [3, 4]] = [7, 10]
    let a = ParallelTensor::new(vec![1, 2], vec![1.0, 2.0]).expect("test");
    let b = ParallelTensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).expect("test");
    let c = a.matmul(&b).expect("test");
    assert_eq!(c.shape, vec![1, 2]);
    assert_eq!(c.data, vec![7.0, 10.0]);
}

#[test]
fn test_parallel_tensor_add() {
    let a = ParallelTensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).expect("test");
    let b = ParallelTensor::new(vec![2, 2], vec![5.0, 6.0, 7.0, 8.0]).expect("test");
    let c = a.add(&b).expect("test");
    assert_eq!(c.data, vec![6.0, 8.0, 10.0, 12.0]);
}

// =========================================================================
// Communicator Tests
// =========================================================================

#[test]
fn test_communicator_new() {
    let comm = Communicator::new(4, 2).expect("test");
    assert_eq!(comm.world_size(), 4);
    assert_eq!(comm.rank(), 2);
}

#[test]
fn test_communicator_invalid_rank() {
    let result = Communicator::new(4, 10);
    assert!(result.is_err());
}

#[test]
fn test_communicator_all_reduce_sum() {
    let comm = Communicator::new(4, 0).expect("test");
    let tensor = ParallelTensor::new(vec![2], vec![1.0, 2.0]).expect("test");
    let result = comm.all_reduce(&tensor, ReduceOp::Sum).expect("test");
    // test: multiply by world_size
    assert_eq!(result.data, vec![4.0, 8.0]);
}

#[test]
fn test_communicator_all_reduce_avg() {
    let comm = Communicator::new(4, 0).expect("test");
    let tensor = ParallelTensor::new(vec![2], vec![1.0, 2.0]).expect("test");
    let result = comm.all_reduce(&tensor, ReduceOp::Avg).expect("test");
    assert_eq!(result.data, vec![1.0, 2.0]);
}

#[test]
fn test_communicator_all_gather() {
    let comm = Communicator::new(2, 0).expect("test");
    let tensor = ParallelTensor::new(vec![2], vec![1.0, 2.0]).expect("test");
    let result = comm.all_gather(&tensor).expect("test");
    assert_eq!(result.shape, vec![4]);
    assert_eq!(result.data, vec![1.0, 2.0, 1.0, 2.0]);
}

#[test]
fn test_communicator_barrier() {
    let comm = Communicator::new(4, 0).expect("test");
    assert!(comm.barrier().is_ok());
}

// =========================================================================
// TensorParallel Tests
// =========================================================================

#[test]
fn test_tensor_parallel_new() {
    let tp = TensorParallel::new(4, 2).expect("test");
    assert_eq!(tp.tp_size(), 4);
    assert_eq!(tp.rank(), 2);
}

#[test]
fn test_tensor_parallel_invalid_rank() {
    let result = TensorParallel::new(4, 10);
    assert!(result.is_err());
}

#[test]
fn test_tensor_parallel_chunk_size() {
    let tp = TensorParallel::new(4, 0).expect("test");
    assert_eq!(tp.chunk_size(100), 25);
    assert_eq!(tp.chunk_size(16), 4);
}

#[test]
fn test_tensor_parallel_column_linear() {
    let tp = TensorParallel::new(2, 0).expect("test");

    // Input: (1, 4), Weight: (8, 4) split to (4, 4) per rank
    let input = ParallelTensor::new(vec![1, 4], vec![1.0, 1.0, 1.0, 1.0]).expect("test");
    let weight =
        ParallelTensor::new(vec![8, 4], (0..32).map(|i| i as f32).collect()).expect("test");

    let output = tp
        .column_parallel_linear(&input, &weight, None)
        .expect("test");
    // Output should be (1, 4) - chunk of full output
    assert_eq!(output.shape, vec![1, 4]);
}

#[test]
fn test_tensor_parallel_row_linear() {
    let tp = TensorParallel::new(2, 0).expect("test");

    // Row parallel: Weight (4, 8) split to (2, 8) per rank
    // After transpose: (8, 2)
    // Input needs to be (batch, 8) to matmul with (8, 2) -> output (batch, 2)
    let input = ParallelTensor::new(vec![1, 8], vec![1.0; 8]).expect("test");
    let weight =
        ParallelTensor::new(vec![4, 8], (0..32).map(|i| i as f32).collect()).expect("test");

    let output = tp.row_parallel_linear(&input, &weight, None).expect("test");
    // Output shape after row parallel
    assert!(!output.data.is_empty());
    // After all-reduce, output shape is (1, 2)
    assert_eq!(output.shape[0], 1);
}

// =========================================================================
// PipelineParallel Tests
// =========================================================================

#[test]
fn test_pipeline_parallel_new() {
    let pp = PipelineParallel::new(4, 1, 24, 4).expect("test");
    assert_eq!(pp.num_stages(), 4);
    assert_eq!(pp.stage(), 1);
    assert_eq!(pp.micro_batch_size(), 4);
}

#[test]
fn test_pipeline_parallel_layer_distribution() {
    // 24 layers across 4 stages = 6 layers each
    let pp = PipelineParallel::new(4, 0, 24, 4).expect("test");
    let info = pp.stage_info();
    assert_eq!(info.start_layer, 0);
    assert_eq!(info.end_layer, 6);
    assert_eq!(info.num_layers, 6);

    let pp2 = PipelineParallel::new(4, 3, 24, 4).expect("test");
    let info2 = pp2.stage_info();
    assert_eq!(info2.start_layer, 18);
    assert_eq!(info2.end_layer, 24);
}

#[test]
fn test_pipeline_parallel_uneven_layers() {
    // 25 layers across 4 stages: 7, 6, 6, 6
    let pp = PipelineParallel::new(4, 0, 25, 4).expect("test");
    assert_eq!(pp.stage_info().num_layers, 7);

    let pp1 = PipelineParallel::new(4, 1, 25, 4).expect("test");
    assert_eq!(pp1.stage_info().num_layers, 6);
}

#[test]
fn test_pipeline_parallel_first_last() {
    let first = PipelineParallel::new(4, 0, 24, 4).expect("test");
    assert!(first.is_first_stage());
    assert!(!first.is_last_stage());

    let last = PipelineParallel::new(4, 3, 24, 4).expect("test");
    assert!(!last.is_first_stage());
    assert!(last.is_last_stage());
}

#[test]
fn test_pipeline_parallel_bubble_ratio() {
    let pp = PipelineParallel::new(4, 0, 24, 4).expect("test");
    // Bubble = (4-1) / (4 + 8 - 1) = 3/11 ≈ 0.27
    let ratio = pp.bubble_ratio(8);
    assert!(ratio > 0.2 && ratio < 0.4);
}

#[test]
fn test_pipeline_parallel_stats() {
    let mut pp = PipelineParallel::new(4, 0, 24, 4).expect("test");
    pp.record_micro_batch(10.0);
    pp.record_micro_batch(12.0);

    let stats = pp.stats();
    assert_eq!(stats.micro_batches_processed, 2);
    assert_eq!(stats.forward_passes, 2);
    assert!((stats.avg_stage_latency_ms - 11.0).abs() < 0.1);
}

// =========================================================================
// ZeroOffload Tests
// =========================================================================

#[test]
fn test_zero_offload_default() {
    let zero = ZeroOffload::default();
    assert!(zero.offload_optimizer);
    assert!(!zero.offload_params);
    assert!(zero.pin_memory);
}

#[test]
fn test_zero_offload_inference() {
    let zero = ZeroOffload::inference();
    assert!(!zero.offload_optimizer);
    assert!(zero.offload_params);
    assert!(zero.offload_activations);
}

#[test]
fn test_zero_offload_memory_savings() {
    let zero = ZeroOffload::default();
    let savings = zero.memory_savings_ratio();
    assert!(savings >= 0.0 && savings <= 1.0);

    let zero_inference = ZeroOffload::inference();
    let savings_inference = zero_inference.memory_savings_ratio();
    assert!(savings_inference > savings);
}

// =========================================================================
// DistributedContext Tests
// =========================================================================

#[test]
fn test_distributed_context_single() {
    let config = ParallelConfig::single();
    let ctx = DistributedContext::new(config).expect("test");

    assert!(!ctx.is_distributed());
    assert!(ctx.is_initialized());
    assert!(ctx.tensor_parallel().is_none());
    assert!(ctx.pipeline_parallel().is_none());
}

#[test]
fn test_distributed_context_with_tp() {
    let config = ParallelConfig::new(4, 1, 1, 0).expect("test");
    let ctx = DistributedContext::new(config).expect("test");

    assert!(ctx.is_distributed());
    assert!(ctx.tensor_parallel().is_some());
    assert_eq!(ctx.tensor_parallel().expect("test").tp_size(), 4);
}

#[test]
fn test_distributed_context_init_pipeline() {
    let config = ParallelConfig::new(1, 4, 1, 0).expect("test");
    let mut ctx = DistributedContext::new(config).expect("test");

    ctx.init_pipeline(24, 4).expect("test");
    assert!(ctx.pipeline_parallel().is_some());
    assert_eq!(ctx.pipeline_parallel().expect("test").num_stages(), 4);
}

#[test]
fn test_distributed_context_zero_offload() {
    let config = ParallelConfig::single();
    let mut ctx = DistributedContext::new(config).expect("test");

    ctx.set_zero_offload(ZeroOffload::inference());
    assert!(ctx.zero_offload().offload_params);
}

// =========================================================================
// ReduceOp Tests
// =========================================================================

#[test]
fn test_reduce_op_serialization() {
    let op = ReduceOp::Sum;
    let json = serde_json::to_string(&op).expect("test");
    let deserialized: ReduceOp = serde_json::from_str(&json).expect("test");
    assert_eq!(op, deserialized);
}

// =========================================================================
// Error Tests
// =========================================================================

#[test]
fn test_parallel_error_display() {
    let err = ParallelError::InvalidRank {
        rank: 10,
        world_size: 4,
    };
    assert!(err.to_string().contains("10"));
    assert!(err.to_string().contains("4"));

    let err2 = ParallelError::CommunicationError("timeout".to_string());
    assert!(err2.to_string().contains("timeout"));
}

// =========================================================================
// Extended Coverage Tests: ParallelConfig
// =========================================================================

#[test]
fn test_parallel_config_world_size_calculation_ext_cov() {
    let config = ParallelConfig::new(2, 2, 2, 0).expect("test");
    assert_eq!(config.world_size, 8);
}

#[test]
fn test_parallel_config_single_debug_ext_cov() {
    let config = ParallelConfig::single();
    let debug_str = format!("{:?}", config);
    assert!(debug_str.contains("tp_size"));
    assert!(debug_str.contains("pp_size"));
}

#[test]
fn test_parallel_config_invalid_zero_tp_ext_cov() {
    let result = ParallelConfig::new(0, 1, 1, 0);
    assert!(result.is_err());
}

#[test]
fn test_parallel_config_invalid_zero_pp_ext_cov() {
    let result = ParallelConfig::new(1, 0, 1, 0);
    assert!(result.is_err());
}

#[test]
fn test_parallel_config_invalid_zero_dp_ext_cov() {
    let result = ParallelConfig::new(1, 1, 0, 0);
    assert!(result.is_err());
}

// =========================================================================
// Extended Coverage Tests: ReduceOp
// =========================================================================

#[test]
fn test_reduce_op_all_variants_ext_cov() {
    let ops = [ReduceOp::Sum, ReduceOp::Min, ReduceOp::Max, ReduceOp::Avg];
    for op in ops {
        let json = serde_json::to_string(&op).expect("serialize");
        let _: ReduceOp = serde_json::from_str(&json).expect("deserialize");
    }
}

include!("tests_reduce.rs");
include!("tests_pipeline_parallel.rs");
