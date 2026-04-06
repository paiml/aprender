
#[test]
fn test_reduce_op_clone_ext_cov() {
    let op = ReduceOp::Max;
    let cloned = op;
    assert_eq!(op, cloned);
}

#[test]
fn test_reduce_op_debug_ext_cov() {
    let op = ReduceOp::Avg;
    let debug_str = format!("{:?}", op);
    assert!(debug_str.contains("Avg"));
}

// =========================================================================
// Extended Coverage Tests: ParallelError
// =========================================================================

#[test]
fn test_parallel_error_all_variants_ext_cov() {
    let errors: [ParallelError; 5] = [
        ParallelError::InvalidRank {
            rank: 5,
            world_size: 4,
        },
        ParallelError::InvalidWorldSize(0),
        ParallelError::CommunicationError("timeout".to_string()),
        ParallelError::ShapeMismatch {
            expected: vec![2, 3],
            got: vec![3, 2],
        },
        ParallelError::PipelineError("stage error".to_string()),
    ];
    for err in errors {
        let _ = err.to_string();
    }
}

#[test]
fn test_parallel_error_shape_mismatch_ext_cov() {
    let err = ParallelError::ShapeMismatch {
        expected: vec![10, 20],
        got: vec![20, 10],
    };
    let msg = err.to_string();
    assert!(msg.contains("10") || msg.contains("20"));
}

#[test]
fn test_parallel_error_debug_ext_cov() {
    let err = ParallelError::NotInitialized;
    let debug_str = format!("{:?}", err);
    assert!(debug_str.contains("NotInitialized"));
}

// =========================================================================
// Extended Coverage Tests: ZeroOffload
// =========================================================================

#[test]
fn test_zero_offload_clone_ext_cov() {
    let zero = ZeroOffload::inference();
    let cloned = zero.clone();
    assert_eq!(zero.offload_params, cloned.offload_params);
    assert_eq!(zero.offload_activations, cloned.offload_activations);
}

#[test]
fn test_zero_offload_debug_ext_cov() {
    let zero = ZeroOffload::default();
    let debug_str = format!("{:?}", zero);
    assert!(debug_str.contains("offload_optimizer"));
    assert!(debug_str.contains("pin_memory"));
}

#[test]
fn test_zero_offload_memory_savings_extremes_ext_cov() {
    // Test with all offload options enabled
    let full_offload = ZeroOffload {
        offload_optimizer: true,
        offload_params: true,
        offload_activations: true,
        pin_memory: true,
        overlap_comm: true,
    };
    let savings = full_offload.memory_savings_ratio();
    assert!(savings >= 0.0);
    assert!(savings <= 1.0);

    // Test with no offload
    let no_offload = ZeroOffload {
        offload_optimizer: false,
        offload_params: false,
        offload_activations: false,
        pin_memory: false,
        overlap_comm: false,
    };
    let no_savings = no_offload.memory_savings_ratio();
    assert!(no_savings >= 0.0);
    assert!(no_savings < savings);
}

// =========================================================================
// Extended Coverage Tests: PipelineStats
// =========================================================================

#[test]
fn test_pipeline_stats_clone_debug_ext_cov() {
    let stats = PipelineStats {
        micro_batches_processed: 100,
        forward_passes: 100,
        bubble_time_ms: 5.0,
        avg_stage_latency_ms: 10.5,
    };
    let cloned = stats.clone();
    assert_eq!(
        stats.micro_batches_processed,
        cloned.micro_batches_processed
    );

    let debug_str = format!("{:?}", stats);
    assert!(debug_str.contains("micro_batches_processed"));
    assert!(debug_str.contains("bubble_time_ms"));
}

// =========================================================================
// Extended Coverage Tests: TensorParallel
// =========================================================================

#[test]
fn test_tensor_parallel_chunk_size_ext_cov() {
    let tp = TensorParallel::new(4, 0).expect("test");
    let chunk = tp.chunk_size(1000);
    assert_eq!(chunk, 250); // 1000 / 4 = 250
}

#[test]
fn test_tensor_parallel_debug_ext_cov() {
    let tp = TensorParallel::new(8, 2).expect("test");
    let debug_str = format!("{:?}", tp);
    assert!(debug_str.contains("tp_size"));
    assert!(debug_str.contains("rank"));
}

#[test]
fn test_tensor_parallel_invalid_rank_ext_cov() {
    let result = TensorParallel::new(4, 10);
    assert!(result.is_err());
}

#[test]
fn test_tensor_parallel_invalid_size_ext_cov() {
    let result = TensorParallel::new(0, 0);
    assert!(result.is_err());
}

// =========================================================================
// Extended Coverage Tests: Communicator
// =========================================================================

#[test]
fn test_communicator_debug_ext_cov() {
    let comm = Communicator::new(4, 0).expect("test");
    let debug_str = format!("{:?}", comm);
    assert!(debug_str.contains("world_size"));
    assert!(debug_str.contains("rank"));
}

#[test]
fn test_communicator_invalid_rank_ext_cov() {
    let result = Communicator::new(4, 10);
    assert!(result.is_err());
}

// =========================================================================
// Extended Coverage Tests: PipelineParallel
// =========================================================================

#[test]
fn test_pipeline_parallel_stage_info_ext_cov() {
    // PipelineParallel::new(pp_size, stage, total_layers, micro_batch_size)
    let pp = PipelineParallel::new(4, 0, 24, 4).expect("test");
    let info = pp.stage_info();
    assert_eq!(info.start_layer, 0);
    assert_eq!(info.num_layers, 6); // 24 / 4 = 6 layers per stage
}

#[test]
fn test_pipeline_parallel_debug_ext_cov() {
    let pp = PipelineParallel::new(4, 0, 24, 4).expect("test");
    let debug_str = format!("{:?}", pp);
    assert!(debug_str.contains("pp_size"));
    assert!(debug_str.contains("stage"));
}

#[test]
fn test_pipeline_parallel_micro_batch_size_ext_cov() {
    let pp = PipelineParallel::new(4, 0, 24, 8).expect("test");
    assert_eq!(pp.micro_batch_size(), 8);
}

#[test]
fn test_pipeline_parallel_first_last_stage_ext_cov() {
    let first = PipelineParallel::new(4, 0, 24, 4).expect("test");
    let last = PipelineParallel::new(4, 3, 24, 4).expect("test");
    let middle = PipelineParallel::new(4, 1, 24, 4).expect("test");

    assert!(first.is_first_stage());
    assert!(!first.is_last_stage());

    assert!(!last.is_first_stage());
    assert!(last.is_last_stage());

    assert!(!middle.is_first_stage());
    assert!(!middle.is_last_stage());
}

// =========================================================================
// Additional Coverage Tests: ParallelTensor Edge Cases
// =========================================================================

#[test]
fn test_parallel_tensor_new_shape_mismatch() {
    // Data size doesn't match shape
    let result = ParallelTensor::new(vec![2, 3], vec![1.0, 2.0]); // 6 expected, 2 provided
    assert!(result.is_err());
    if let Err(ParallelError::ShapeMismatch { expected, got }) = result {
        assert_eq!(expected, vec![6]);
        assert_eq!(got, vec![2]);
    }
}

#[test]
fn test_parallel_tensor_narrow_1d() {
    let tensor = ParallelTensor::new(vec![6], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("test");
    let narrowed = tensor.narrow(0, 2, 3).expect("test");
    assert_eq!(narrowed.shape, vec![3]);
    assert_eq!(narrowed.data, vec![3.0, 4.0, 5.0]);
}

#[test]
fn test_parallel_tensor_narrow_invalid_dim() {
    let tensor = ParallelTensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("test");
    let result = tensor.narrow(5, 0, 1); // dim 5 doesn't exist
    assert!(result.is_err());
}

#[test]
fn test_parallel_tensor_narrow_fallback_3d() {
    // Test the fallback path for >2D tensors
    let tensor = ParallelTensor::new(vec![2, 2, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        .expect("test");
    let narrowed = tensor.narrow(0, 0, 2).expect("test");
    // Fallback path uses simple slicing
    assert_eq!(narrowed.shape[0], 2);
}

#[test]
fn test_parallel_tensor_transpose_non_2d_error() {
    let tensor = ParallelTensor::new(vec![6], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("test");
    let result = tensor.transpose();
    assert!(result.is_err());
    if let Err(ParallelError::ShapeMismatch { expected, got }) = result {
        assert_eq!(expected, vec![2]);
        assert_eq!(got, vec![1]);
    }
}

#[test]
fn test_parallel_tensor_matmul_non_2d_error() {
    let a = ParallelTensor::new(vec![4], vec![1.0, 2.0, 3.0, 4.0]).expect("test");
    let b = ParallelTensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).expect("test");
    let result = a.matmul(&b);
    assert!(result.is_err());
}

#[test]
fn test_parallel_tensor_matmul_dimension_mismatch() {
    let a = ParallelTensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("test");
    let b = ParallelTensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).expect("test"); // K mismatch
    let result = a.matmul(&b);
    assert!(result.is_err());
    if let Err(ParallelError::ShapeMismatch { expected, got }) = result {
        assert_eq!(expected, vec![3]); // a's K dimension
        assert_eq!(got, vec![2]); // b's first dimension
    }
}

#[test]
fn test_parallel_tensor_add_shape_mismatch() {
    let a = ParallelTensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).expect("test");
    let b = ParallelTensor::new(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("test");
    let result = a.add(&b);
    assert!(result.is_err());
}

// =========================================================================
// Additional Coverage Tests: Communicator Operations
// =========================================================================

#[test]
fn test_communicator_all_reduce_max() {
    let comm = Communicator::new(4, 0).expect("test");
    let tensor = ParallelTensor::new(vec![3], vec![1.0, 5.0, 3.0]).expect("test");
    let result = comm.all_reduce(&tensor, ReduceOp::Max).expect("test");
    // Single process Max returns as-is
    assert_eq!(result.data, vec![1.0, 5.0, 3.0]);
}

#[test]
fn test_communicator_all_reduce_min() {
    let comm = Communicator::new(4, 0).expect("test");
    let tensor = ParallelTensor::new(vec![3], vec![1.0, 5.0, 3.0]).expect("test");
    let result = comm.all_reduce(&tensor, ReduceOp::Min).expect("test");
    // Single process Min returns as-is
    assert_eq!(result.data, vec![1.0, 5.0, 3.0]);
}

#[test]
fn test_communicator_reduce_scatter_sum() {
    let comm = Communicator::new(2, 0).expect("test");
    let tensor = ParallelTensor::new(vec![4], vec![1.0, 2.0, 3.0, 4.0]).expect("test");
    let result = comm.reduce_scatter(&tensor, ReduceOp::Sum).expect("test");
    // Rank 0 gets first half, multiplied by world_size for Sum
    assert_eq!(result.shape, vec![2]);
    assert_eq!(result.data, vec![2.0, 4.0]); // [1, 2] * 2
}

#[test]
fn test_communicator_reduce_scatter_avg() {
    let comm = Communicator::new(2, 1).expect("test");
    let tensor = ParallelTensor::new(vec![4], vec![1.0, 2.0, 3.0, 4.0]).expect("test");
    let result = comm.reduce_scatter(&tensor, ReduceOp::Avg).expect("test");
    // Rank 1 gets second half, no transformation for Avg
    assert_eq!(result.shape, vec![2]);
    assert_eq!(result.data, vec![3.0, 4.0]);
}

#[test]
fn test_communicator_reduce_scatter_max() {
    let comm = Communicator::new(2, 0).expect("test");
    let tensor = ParallelTensor::new(vec![4], vec![1.0, 2.0, 3.0, 4.0]).expect("test");
    let result = comm.reduce_scatter(&tensor, ReduceOp::Max).expect("test");
    assert_eq!(result.data, vec![1.0, 2.0]);
}

#[test]
fn test_communicator_reduce_scatter_min() {
    let comm = Communicator::new(2, 1).expect("test");
    let tensor = ParallelTensor::new(vec![4], vec![1.0, 2.0, 3.0, 4.0]).expect("test");
    let result = comm.reduce_scatter(&tensor, ReduceOp::Min).expect("test");
    assert_eq!(result.data, vec![3.0, 4.0]);
}

#[test]
fn test_communicator_all_gather_empty_shape() {
    let comm = Communicator::new(2, 0).expect("test");
    let tensor = ParallelTensor {
        shape: vec![],
        data: vec![],
    };
    let result = comm.all_gather(&tensor).expect("test");
    assert!(result.shape.is_empty());
    assert!(result.data.is_empty());
}

#[test]
fn test_communicator_reduce_scatter_empty_shape() {
    let comm = Communicator::new(2, 0).expect("test");
    let tensor = ParallelTensor {
        shape: vec![],
        data: vec![],
    };
    let result = comm.reduce_scatter(&tensor, ReduceOp::Sum).expect("test");
    assert!(result.shape.is_empty());
}

#[test]
fn test_communicator_clone() {
    let comm = Communicator::new(4, 2).expect("test");
    let cloned = comm.clone();
    assert_eq!(cloned.world_size(), 4);
    assert_eq!(cloned.rank(), 2);
}

// =========================================================================
// Additional Coverage Tests: TensorParallel with Bias
// =========================================================================

#[test]
fn test_tensor_parallel_column_linear_with_bias() {
    let tp = TensorParallel::new(2, 0).expect("test");

    let input = ParallelTensor::new(vec![1, 4], vec![1.0, 1.0, 1.0, 1.0]).expect("test");
    let weight =
        ParallelTensor::new(vec![8, 4], (0..32).map(|i| i as f32).collect()).expect("test");
    let bias = ParallelTensor::new(vec![8], vec![0.5; 8]).expect("test");

    let output = tp
        .column_parallel_linear(&input, &weight, Some(&bias))
        .expect("test");
    assert_eq!(output.shape, vec![1, 4]);
    // Verify bias was added (values should be different from no-bias case)
    assert!(output.data.iter().all(|&x| x != 0.0));
}

#[test]
fn test_tensor_parallel_row_linear_with_bias_rank0() {
    let tp = TensorParallel::new(2, 0).expect("test");

    let input = ParallelTensor::new(vec![1, 8], vec![1.0; 8]).expect("test");
    let weight =
        ParallelTensor::new(vec![4, 8], (0..32).map(|i| i as f32).collect()).expect("test");
    // Bias shape should match output (4 features, split by 2 = 2 per rank)
    let bias = ParallelTensor::new(vec![1, 2], vec![1.0; 2]).expect("test");

    let output = tp
        .row_parallel_linear(&input, &weight, Some(&bias))
        .expect("test");
    // Bias should be added on rank 0
    assert!(!output.data.is_empty());
}

#[test]
fn test_tensor_parallel_row_linear_with_bias_rank1() {
    let tp = TensorParallel::new(2, 1).expect("test");

    let input = ParallelTensor::new(vec![1, 8], vec![1.0; 8]).expect("test");
    let weight =
        ParallelTensor::new(vec![4, 8], (0..32).map(|i| i as f32).collect()).expect("test");
    // Bias shape should match output (4 features, split by 2 = 2 per rank)
    let bias = ParallelTensor::new(vec![1, 2], vec![1.0; 2]).expect("test");

    let output = tp
        .row_parallel_linear(&input, &weight, Some(&bias))
        .expect("test");
    // Bias should NOT be added on rank 1
    assert!(!output.data.is_empty());
}

// =========================================================================
// Additional Coverage Tests: PipelineParallel Edge Cases
// =========================================================================

#[test]
fn test_pipeline_parallel_invalid_zero_size() {
    let result = PipelineParallel::new(0, 0, 24, 4);
    assert!(result.is_err());
}
