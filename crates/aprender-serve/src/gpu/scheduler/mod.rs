//! GPU Scheduler Module (PMAT-802)
//!
//! Shattered from gpu/cuda_scheduler.rs
//! Types extracted to types.rs for file health (PMAT-COMPLY)

pub mod batch;
mod core;
mod kv;
pub mod linear_attn;
mod loading;
mod model;
pub mod moe_dispatch;
mod ops;
mod types;

#[cfg(feature = "cuda")]
pub use core::CudaScheduler;
pub use model::GpuModel;
pub use types::{
    AttentionBuffers, BlockWeights, GpuGenerateConfig, GpuModelConfig, GpuWeightError,
    LinearAttnWeights, LmHeadWeight, LmHeadWeightTransposed, MoeExpertWeights, ValidatedGpuWeights,
    WeightType,
};

// Re-export linear attention state
pub use linear_attn::LinearAttnState;

// Re-export KV cache functions for external use (public API)
#[allow(unused_imports)]
pub use kv::{forward_gpu_incremental, forward_gpu_with_cache, generate_with_cache};

// Re-export ops for crate-internal use (apr::helpers delegation)
pub(crate) use ops::layer_norm_static;
