//! Grammar of ComputeBlock (§32)
//!
//! A declarative, composable framework for specifying compute workloads,
//! inspired by Wilkinson's Grammar of Graphics (2005).
//!
//! # Conceptual Foundation
//!
//! Just as the Grammar of Graphics decomposes visualization:
//! ```text
//! Data + Aesthetics + Geometry + Statistics + Scales + Coordinates + Facets + Theme → Visualization
//! ```
//!
//! The Grammar of ComputeBlock decomposes computation:
//! ```text
//! Workload + Resources + Strategy + Transform + Scales + Context + Composition + Policy → Execution
//! ```
//!
//! # Example
//!
//! ```ignore
//! use cbtop::grammar::*;
//!
//! let result = ComputeBlock::builder()
//!     .workload(Workload::matmul(1024, 1024, 1024))
//!     .strategy(Strategy::Gpu(GpuDevice::Auto))
//!     .strategy(Strategy::Simd(SimdWidth::Avx2))  // Fallback
//!     .transform(Transform::Tile { tile_size: 64 })
//!     .policy(Policy::realtime())
//!     .build()?
//!     .execute()?;
//! ```
//!
//! # References
//!
//! - [Wilkinson 2005] "The Grammar of Graphics" Springer
//! - [Wickham 2010] "A Layered Grammar of Graphics" JCGS
//! - [Halide 2013] "Halide: Optimizing Parallelism" PLDI
//! - [TVM 2018] "TVM: End-to-End Optimizing Compiler" OSDI

mod composition;
mod compute_block;
mod context;
mod error;
mod policy;
mod resources;
mod scales;
mod strategy;
mod transform;
mod workload;

// Re-export all public types for backwards compatibility.
pub use composition::CompositionMode;
pub use compute_block::{BuiltComputeBlock, ComputeBlock, ComputeBlockBuilder, ExecutionResult};
pub use context::{CpuAffinity, ExecutionContext};
pub use error::{GrammarError, GrammarResult};
pub use policy::{ExecutionPolicy, ObservabilityConfig, QosLevel, ResourceLimits, RetryPolicy};
pub use resources::{ByteSize, ResourceMapping, ScaleBinding};
pub use scales::{LinearResourceScale, LogResourceScale, ResourceScale};
pub use strategy::{ExecutionStrategy, GpuDevice, KernelSpec, SimdWidth, StrategyLayer};
pub use transform::{DataTransform, QuantScheme};
pub use workload::{DataType, Dimensions, Operation, TensorSpec, WorkloadSpec};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_workload_spec_dot() {
        let workload = WorkloadSpec::dot(1000);
        assert_eq!(workload.operation, Operation::Dot);
        assert_eq!(workload.dimensions.n, 1000);
        assert_eq!(workload.flop_count(), 2000);
    }

    #[test]
    fn test_workload_spec_matmul() {
        let workload = WorkloadSpec::matmul(64, 64, 64);
        assert_eq!(workload.operation, Operation::Matmul);
        assert_eq!(workload.dimensions.m, 64);
        assert_eq!(workload.dimensions.n, 64);
        assert_eq!(workload.dimensions.k, 64);
    }

    #[test]
    fn test_builder_missing_workload() {
        let result = ComputeBlock::builder().build();
        assert!(matches!(result, Err(GrammarError::MissingWorkload)));
    }

    #[test]
    fn test_builder_with_workload() {
        let result = ComputeBlock::builder().workload(WorkloadSpec::dot(1000)).build();
        assert!(result.is_ok());
    }

    #[test]
    fn test_strategy_fallback() {
        let block = ComputeBlock::builder()
            .workload(WorkloadSpec::matmul(1024, 1024, 1024))
            .strategy_with_priority(ExecutionStrategy::gpu_auto(), 10)
            .strategy_with_priority(ExecutionStrategy::simd_auto(), 5)
            .strategy_with_priority(ExecutionStrategy::Sequential, 0)
            .build()
            .unwrap();

        let result = block.execute().unwrap();
        assert!(result.strategy_used.contains("Gpu"));
    }

    #[test]
    fn test_execution_policy_realtime() {
        let policy = ExecutionPolicy::realtime();
        assert_eq!(policy.qos, QosLevel::Realtime);
        assert!(!policy.preemptible);
        assert!(policy.timeout.is_some());
    }

    #[test]
    fn test_linear_scale() {
        let scale = LinearResourceScale::new((0.0, 100.0), (0.0, 8.0)).unwrap();
        assert_eq!(scale.scale(50.0), 4.0);
        assert_eq!(scale.scale(0.0), 0.0);
        assert_eq!(scale.scale(100.0), 8.0);
    }

    #[test]
    fn test_linear_scale_invalid_domain() {
        let result = LinearResourceScale::new((100.0, 0.0), (0.0, 8.0));
        assert!(matches!(result, Err(GrammarError::InvalidScaleDomain { .. })));
    }

    #[test]
    fn test_data_type_byte_size() {
        assert_eq!(DataType::F32.byte_size(), 4);
        assert_eq!(DataType::F16.byte_size(), 2);
        assert_eq!(DataType::I8.byte_size(), 1);
    }

    #[test]
    fn test_tensor_spec() {
        let spec = TensorSpec::new("test", vec![10, 20, 30], DataType::F32);
        assert_eq!(spec.numel(), 6000);
        assert_eq!(spec.byte_size(), 24000);
    }

    #[test]
    fn test_composition_modes() {
        let batch = CompositionMode::batch(32);
        assert!(matches!(batch, CompositionMode::Batch { batch_size: 32, .. }));

        let dp = CompositionMode::data_parallel(4);
        assert!(matches!(dp, CompositionMode::DataParallel { shards: 4 }));
    }

    #[test]
    fn test_transform_identity() {
        let t = DataTransform::identity();
        assert!(matches!(t, DataTransform::Identity));
    }

    #[test]
    fn test_facet_by() {
        let block = ComputeBlock::builder()
            .workload(WorkloadSpec::matmul(64, 64, 64))
            .facet_by("tile_size", vec![16.0, 32.0, 64.0])
            .build()
            .unwrap();

        // Facet params should be set
        assert!(block.inner.facet_params.is_some());
    }
}
