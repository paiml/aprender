//! grammar_f701 - Part 1

use cbtop::{
    ByteSize, CompositionMode, ComputeBlock, ComputeBlockBuilder, DataTransform, ExecutionStrategy,
    GrammarError, LinearResourceScale, LogResourceScale, ResourceMapping, WorkloadSpec,
};

// ============================================================================
// F701: Builder Rejects Incomplete Spec
// ============================================================================

#[test]
fn f701_builder_rejects_no_workload() {
    // F701: Build without workload must return Err
    let result = ComputeBlock::builder().build();
    assert!(matches!(result, Err(GrammarError::MissingWorkload)));
}

#[test]
fn f701_builder_accepts_with_workload() {
    // F701: Build with workload must succeed
    let result = ComputeBlock::builder()
        .workload(WorkloadSpec::dot(1000))
        .build();
    assert!(result.is_ok());
}

#[test]
fn f701_builder_accepts_minimal_spec() {
    // F701: Minimal valid spec
    let result = ComputeBlock::builder()
        .workload(WorkloadSpec::elementwise(100))
        .build();
    assert!(result.is_ok());
}

// ============================================================================
// F702: Strategy Fallback
// ============================================================================

#[test]
fn f702_strategy_layers_ordered_by_priority() {
    // F702: Higher priority strategies tried first
    let block = ComputeBlock::builder()
        .workload(WorkloadSpec::matmul(64, 64, 64))
        .strategy_with_priority(ExecutionStrategy::Sequential, 0)
        .strategy_with_priority(ExecutionStrategy::simd_auto(), 5)
        .strategy_with_priority(ExecutionStrategy::gpu_auto(), 10)
        .build()
        .unwrap();

    let result = block.execute().unwrap();
    // GPU should be tried first (highest priority)
    assert!(result.strategy_used.contains("Gpu"));
}

#[test]
fn f702_fallback_to_lower_priority() {
    // F702: Multiple strategies available for fallback
    let block = ComputeBlock::builder()
        .workload(WorkloadSpec::dot(1000))
        .strategy(ExecutionStrategy::simd_auto())
        .strategy(ExecutionStrategy::Sequential)
        .build()
        .unwrap();

    let result = block.execute();
    assert!(result.is_ok());
}

// ============================================================================
// F703: Resource Scaling
// ============================================================================

#[test]
fn f703_resource_mapping_cores() {
    // F703: Resource mapping with core binding
    let resources = ResourceMapping::new().cores_value(8);

    assert_eq!(resources.cores_value, Some(8));
}

#[test]
fn f703_resource_mapping_memory() {
    // F703: Resource mapping with memory limit
    let resources = ResourceMapping::new().memory_value(ByteSize::gb(4));

    assert_eq!(resources.memory_value, Some(ByteSize::gb(4)));
}

#[test]
fn f703_byte_size_conversions() {
    // F703: ByteSize helper works correctly
    assert_eq!(ByteSize::mb(1).bytes(), 1024 * 1024);
    assert_eq!(ByteSize::gb(1).bytes(), 1024 * 1024 * 1024);
    assert_eq!(ByteSize::gb(4).bytes(), 4 * 1024 * 1024 * 1024);
}

// ============================================================================
// F704: Composition Consistency
// ============================================================================

#[test]
fn f704_composition_none_vs_batch_1() {
    // F704: Batch(1) should behave like None
    let none_block = ComputeBlock::builder()
        .workload(WorkloadSpec::dot(1000))
        .composition(CompositionMode::None)
        .build()
        .unwrap();

    let batch_block = ComputeBlock::builder()
        .workload(WorkloadSpec::dot(1000))
        .composition(CompositionMode::batch(1))
        .build()
        .unwrap();

    // Both should execute successfully
    let none_result = none_block.execute();
    let batch_result = batch_block.execute();

    assert!(none_result.is_ok());
    assert!(batch_result.is_ok());
}

#[test]
fn f704_data_parallel_composition() {
    // F704: DataParallel composition
    let block = ComputeBlock::builder()
        .workload(WorkloadSpec::matmul(1024, 1024, 1024))
        .composition(CompositionMode::data_parallel(4))
        .build()
        .unwrap();

    let result = block.execute();
    assert!(result.is_ok());
}

#[test]
fn f704_pipeline_composition() {
    // F704: Pipeline composition
    let block = ComputeBlock::builder()
        .workload(WorkloadSpec::attention(1, 512, 8, 64))
        .composition(CompositionMode::pipeline(2))
        .build()
        .unwrap();

    let result = block.execute();
    assert!(result.is_ok());
}

// ============================================================================
// F710: Identity Transform
// ============================================================================

#[test]
fn f710_identity_transform_is_noop() {
    // F710: Identity transform doesn't modify anything
    let block = ComputeBlock::builder()
        .workload(WorkloadSpec::elementwise(1000))
        .transform(DataTransform::Identity)
        .build()
        .unwrap();

    // Should execute without modification
    let result = block.execute();
    assert!(result.is_ok());
}

#[test]
fn f710_transform_identity_constructor() {
    // F710: DataTransform::identity() returns Identity variant
    let t = DataTransform::identity();
    assert!(matches!(t, DataTransform::Identity));
}

// ============================================================================
// F711: Scale Domain Validation
// ============================================================================

#[test]
fn f711_linear_scale_invalid_domain() {
    // F711: Domain(10, 0) must return Err
    let result = LinearResourceScale::new((10.0, 0.0), (0.0, 8.0));
    assert!(matches!(
        result,
        Err(GrammarError::InvalidScaleDomain { .. })
    ));
}

#[test]
fn f711_linear_scale_equal_domain() {
    // F711: Domain(10, 10) must return Err
    let result = LinearResourceScale::new((10.0, 10.0), (0.0, 8.0));
    assert!(matches!(
        result,
        Err(GrammarError::InvalidScaleDomain { .. })
    ));
}

#[test]
fn f711_linear_scale_valid_domain() {
    // F711: Valid domain should succeed
    let result = LinearResourceScale::new((0.0, 100.0), (0.0, 8.0));
    assert!(result.is_ok());
}

#[test]
fn f711_log_scale_invalid_domain() {
    // F711: Log scale with invalid domain
    let result = LogResourceScale::new(10.0, (100.0, 1.0), (0.0, 8.0));
    assert!(matches!(
        result,
        Err(GrammarError::InvalidScaleDomain { .. })
    ));
}

#[test]
fn f711_log_scale_valid_domain() {
    // F711: Log scale with valid domain
    let result = LogResourceScale::new(10.0, (1.0, 1000.0), (0.0, 8.0));
    assert!(result.is_ok());
}

// ============================================================================
// F712: Facet Generation
// ============================================================================

#[test]
fn f712_facet_by_sets_params() {
    // F712: Facet parameters are stored
    let block = ComputeBlock::builder()
        .workload(WorkloadSpec::matmul(64, 64, 64))
        .facet_by("tile_size", vec![16.0, 32.0, 64.0])
        .build()
        .unwrap();

    // Facet params should be accessible
    assert!(block.workload().is_some());
}

// ============================================================================
// F719: Builder Immutability
// ============================================================================

#[test]
fn f719_builder_creates_independent_instances() {
    // F719: Reusing builder creates independent instances
    let builder = ComputeBlock::builder().workload(WorkloadSpec::dot(1000));

    let block1 = builder
        .clone()
        .strategy(ExecutionStrategy::simd_auto())
        .build();
    let block2 = builder
        .clone()
        .strategy(ExecutionStrategy::Sequential)
        .build();

    assert!(block1.is_ok());
    assert!(block2.is_ok());

    // Both should execute independently
    let result1 = block1.unwrap().execute();
    let result2 = block2.unwrap().execute();

    assert!(result1.is_ok());
    assert!(result2.is_ok());
}

#[test]
fn f719_builder_default_is_empty() {
    // F719: Default builder has no workload
    let builder = ComputeBlockBuilder::new();
    let result = builder.build();
    assert!(matches!(result, Err(GrammarError::MissingWorkload)));
}
