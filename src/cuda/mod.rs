//! CUDA PTX Generation and Execution Module
//!
//! Provides NVIDIA CUDA-specific PTX code generation and execution via `trueno-gpu`.
//! This is an optional backend for maximum performance on NVIDIA hardware.
//!
//! ## Architecture
//!
//! ```text
//! +-----------------------+
//! |   CudaExecutor API    |  <- High-level execution API
//! +-----------------------+
//! |   CudaKernels API     |  <- PTX generation
//! +-----------------------+
//! |   trueno_gpu::driver  |  <- CUDA runtime (context, stream, memory)
//! +-----------------------+
//! |   trueno_gpu::kernels |  <- Hand-optimized PTX kernels
//! +-----------------------+
//! |   trueno_gpu::ptx     |  <- Pure Rust PTX generation
//! +-----------------------+
//! ```
//!
//! ## Module Organization
//!
//! - `kernels`: Kernel type definitions and PTX generation
//! - `memory`: GPU memory pool and staging buffers
//! - `types`: Weight loading types and transformer workspace
//! - `pipeline`: Async pipeline and PTX optimization
//! - `executor`: CUDA execution engine (split into submodules)
//!
//! ## Available Kernels
//!
//! - **GEMM**: Matrix multiplication (naive, tiled, tensor core)
//! - **Softmax**: Numerically stable softmax with warp shuffle
//! - **LayerNorm**: Fused layer normalization
//! - **Attention**: FlashAttention-style tiled attention
//! - **Quantize**: Q4_K/Q5_K/Q6_K dequantization-fused GEMM/GEMV

// Submodules
pub mod kernels;
pub mod memory;
pub mod pipeline;
pub mod streaming;
pub mod types;

// Re-export everything for backwards compatibility
pub use kernels::{CudaKernels, KernelType};
pub use memory::{
    GpuBufferHandle, GpuMemoryPool, PinnedHostBuffer, PoolStats, SizeClass, StagingBufferPool,
    StagingPoolStats, TransferMode,
};
pub use pipeline::{
    presets, AsyncPipeline, BankConflictStrategy, MemoryPattern, PtxOptimizationHints,
    PtxOptimizer, RegisterTiling,
};
pub use streaming::{check_vram_sufficient, should_use_streaming, StreamingConfig, StreamingMode};
pub use types::{
    BoundLayerWeights, BoundWeight, GemvKernel, IndexedLayerWeights, TransformerWorkspace,
    WeightQuantType,
};

// The executor module (21K lines) - future work to split into submodules:
// - executor/core.rs: Basic context and profiling
// - executor/weights.rs: Weight loading and caching
// - executor/gemm.rs: GEMM/GEMV operations
// - executor/quantized.rs: Quantized GEMV operations
// - executor/activations.rs: GELU, SiLU, RMSNorm, RoPE
// - executor/attention.rs: Flash attention, incremental attention
// - executor/layer.rs: Transformer layer operations
// - executor/forward.rs: Forward pass methods
// - executor/graph.rs: CUDA graph capture and replay
// - executor/kv_cache.rs: KV cache management
mod executor;
pub mod gpu_profile;
pub use executor::CudaExecutor;
pub use gpu_profile::GpuProfile;
