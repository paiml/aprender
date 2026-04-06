//! grammar_f701 - Part 2

use cbtop::{
    CompositionMode, ComputeBlock, DataTransform, DataType, Dimensions, ExecutionContext,
    ExecutionPolicy, ExecutionStrategy, GpuDevice, GrammarError, LinearResourceScale, Operation,
    QosLevel, ResourceMapping, ResourceScale, SimdWidth, StrategyLayer, TensorSpec, WorkloadSpec,
};

// ============================================================================
// Workload Specification Tests
// ============================================================================

#[test]
fn test_workload_dot_product() {
    let workload = WorkloadSpec::dot(10000);
    assert_eq!(workload.operation, Operation::Dot);
    assert_eq!(workload.dimensions.n, 10000);
    assert_eq!(workload.dtype, DataType::F32);
    assert_eq!(workload.inputs.len(), 2);
    assert_eq!(workload.outputs.len(), 1);
}

#[test]
fn test_workload_matmul() {
    let workload = WorkloadSpec::matmul(256, 256, 256);
    assert_eq!(workload.operation, Operation::Matmul);
    assert_eq!(workload.dimensions.m, 256);
    assert_eq!(workload.dimensions.n, 256);
    assert_eq!(workload.dimensions.k, 256);
}

#[test]
fn test_workload_attention() {
    let workload = WorkloadSpec::attention(2, 512, 8, 64);
    assert_eq!(workload.operation, Operation::Attention);
    assert_eq!(workload.dimensions.batch, 2);
    assert_eq!(workload.dimensions.seq_len, 512);
    assert_eq!(workload.dimensions.num_heads, 8);
    assert_eq!(workload.dimensions.head_dim, 64);
}

#[test]
fn test_workload_flop_count_dot() {
    let workload = WorkloadSpec::dot(1000);
    // Dot: n * 2 (mul + add)
    assert_eq!(workload.flop_count(), 2000);
}

#[test]
fn test_workload_flop_count_matmul() {
    let workload = WorkloadSpec::matmul(64, 64, 64);
    // Matmul: M * N * K * 2
    assert_eq!(workload.flop_count(), 64 * 64 * 64 * 2);
}

// ============================================================================
// Data Type Tests
// ============================================================================

#[test]
fn test_data_type_sizes() {
    assert_eq!(DataType::F32.byte_size(), 4);
    assert_eq!(DataType::F16.byte_size(), 2);
    assert_eq!(DataType::Bf16.byte_size(), 2);
    assert_eq!(DataType::I8.byte_size(), 1);
    assert_eq!(DataType::U8.byte_size(), 1);
    assert_eq!(DataType::Q4.byte_size(), 1);
}

// ============================================================================
// Tensor Specification Tests
// ============================================================================

#[test]
fn test_tensor_spec_numel() {
    let spec = TensorSpec::new("test", vec![10, 20, 30], DataType::F32);
    assert_eq!(spec.numel(), 6000);
}

#[test]
fn test_tensor_spec_byte_size() {
    let spec = TensorSpec::new("test", vec![100, 100], DataType::F32);
    assert_eq!(spec.byte_size(), 100 * 100 * 4);
}

// ============================================================================
// Execution Strategy Tests
// ============================================================================

#[test]
fn test_strategy_simd_auto() {
    let strategy = ExecutionStrategy::simd_auto();
    assert!(matches!(strategy, ExecutionStrategy::Simd { width: SimdWidth::Auto }));
}

#[test]
fn test_strategy_simd_avx2() {
    let strategy = ExecutionStrategy::simd(SimdWidth::Avx2);
    assert!(matches!(strategy, ExecutionStrategy::Simd { width: SimdWidth::Avx2 }));
}

#[test]
fn test_strategy_parallel() {
    let strategy = ExecutionStrategy::parallel(8);
    assert!(matches!(strategy, ExecutionStrategy::Parallel { threads: 8, .. }));
}

#[test]
fn test_strategy_gpu_auto() {
    let strategy = ExecutionStrategy::gpu_auto();
    assert!(matches!(strategy, ExecutionStrategy::Gpu { device: GpuDevice::Auto, .. }));
}

// ============================================================================
// Execution Policy Tests
// ============================================================================

#[test]
fn test_policy_realtime() {
    let policy = ExecutionPolicy::realtime();
    assert_eq!(policy.qos, QosLevel::Realtime);
    assert!(!policy.preemptible);
    assert!(policy.timeout.is_some());
}

#[test]
fn test_policy_batch() {
    let policy = ExecutionPolicy::batch();
    assert_eq!(policy.qos, QosLevel::BestEffort);
    assert!(policy.preemptible);
    assert!(policy.timeout.is_none());
}

#[test]
fn test_policy_interactive() {
    let policy = ExecutionPolicy::interactive();
    assert_eq!(policy.qos, QosLevel::Interactive);
    assert!(policy.preemptible);
}

#[test]
fn test_policy_debug() {
    let policy = ExecutionPolicy::debug();
    assert!(policy.observability.tracing);
    assert!(policy.observability.metrics);
    assert_eq!(policy.observability.sampling_rate, 1.0);
}

// ============================================================================
// Data Transform Tests
// ============================================================================

#[test]
fn test_transform_tile() {
    let t = DataTransform::tile(64);
    assert!(matches!(t, DataTransform::Tile { tile_size: 64 }));
}

#[test]
fn test_transform_quantize() {
    let t = DataTransform::quantize(4);
    assert!(matches!(t, DataTransform::Quantize { bits: 4, .. }));
}

#[test]
fn test_transform_pad() {
    let t = DataTransform::pad(64);
    assert!(matches!(t, DataTransform::Pad { alignment: 64 }));
}

// ============================================================================
// Resource Scale Tests
// ============================================================================

#[test]
fn test_linear_scale_interpolation() {
    let scale = LinearResourceScale::new((0.0, 100.0), (0.0, 8.0)).unwrap();
    assert_eq!(scale.scale(0.0), 0.0);
    assert_eq!(scale.scale(50.0), 4.0);
    assert_eq!(scale.scale(100.0), 8.0);
}

#[test]
fn test_linear_scale_domain_range() {
    let scale = LinearResourceScale::new((10.0, 100.0), (1.0, 10.0)).unwrap();
    assert_eq!(scale.domain(), (10.0, 100.0));
    assert_eq!(scale.range(), (1.0, 10.0));
}

// ============================================================================
// Strategy Layer Tests
// ============================================================================

#[test]
fn test_strategy_layer_new() {
    let layer = StrategyLayer::new(ExecutionStrategy::simd_auto());
    assert_eq!(layer.priority, 0);
    assert!(layer.workload.is_none());
}

#[test]
fn test_strategy_layer_priority() {
    let layer = StrategyLayer::new(ExecutionStrategy::gpu_auto()).priority(10);
    assert_eq!(layer.priority, 10);
}

// ============================================================================
// Execution Context Tests
// ============================================================================

#[test]
fn test_context_cpu() {
    let ctx = ExecutionContext::cpu();
    assert!(matches!(ctx, ExecutionContext::Cpu { .. }));
}

#[test]
fn test_context_gpu() {
    let ctx = ExecutionContext::gpu(0);
    assert!(matches!(ctx, ExecutionContext::Gpu { device_id: 0, .. }));
}

// ============================================================================
// Integration Tests
// ============================================================================

#[test]
fn test_full_pipeline() {
    let result = ComputeBlock::builder()
        .workload(WorkloadSpec::matmul(1024, 1024, 1024))
        .resources(ResourceMapping::new().cores_value(8))
        .strategy(ExecutionStrategy::simd_auto())
        .transform(DataTransform::tile(64))
        .context(ExecutionContext::cpu())
        .composition(CompositionMode::batch(32))
        .policy(ExecutionPolicy::interactive())
        .build()
        .unwrap()
        .execute();

    assert!(result.is_ok());
    let exec_result = result.unwrap();
    assert!(exec_result.gflops >= 0.0);
}

#[test]
fn test_dimensions_helpers() {
    let vec_dims = Dimensions::vector(1000);
    assert_eq!(vec_dims.n, 1000);

    let mat_dims = Dimensions::matmul(64, 128, 256);
    assert_eq!(mat_dims.m, 64);
    assert_eq!(mat_dims.n, 128);
    assert_eq!(mat_dims.k, 256);

    let attn_dims = Dimensions::attention(2, 512, 8, 64);
    assert_eq!(attn_dims.batch, 2);
    assert_eq!(attn_dims.seq_len, 512);
}

#[test]
fn test_error_display() {
    let err = GrammarError::MissingWorkload;
    assert!(!err.to_string().is_empty());

    let err = GrammarError::InvalidDimensions("test".to_string());
    assert!(err.to_string().contains("test"));

    let err = GrammarError::InvalidScaleDomain { min: 10.0, max: 0.0 };
    assert!(err.to_string().contains("10"));
}
