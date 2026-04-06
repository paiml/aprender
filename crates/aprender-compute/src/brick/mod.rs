//! ComputeBrick: Token-Centric Compute Units
//!
//! A **ComputeBrick** is a self-verifying, token-centric compute unit that bundles:
//! - **Operation**: The compute operation (matmul, dot, softmax, etc.)
//! - **Assertions**: Falsifiable claims about the output (equivalence, bounds)
//! - **Budget**: Performance target in µs/token or tokens/sec
//! - **Backend**: Execution target (Scalar, AVX2, CUDA, etc.)
//!
//! # Core Insight
//!
//! A **token** is the unit of data; a **ComputeBrick** is the unit of compute.
//!
//! ```text
//! Token ──▶ [ComputeBrick] ──▶ Token
//!            (matmul, softmax, attention)
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use trueno::brick::{ComputeBrick, ComputeBackend, MatmulOp};
//!
//! let matmul = ComputeBrick::new(MatmulOp::new(1024, 1024, 1024))
//!     .assert_equiv(ComputeBackend::Scalar)
//!     .budget_tok_per_sec(50_000.0)
//!     .backend(ComputeBackend::Avx2);
//!
//! let result = matmul.run((a, b))?;
//! println!("Throughput: {:.0} tok/s", result.tokens_per_sec);
//! ```
//!
//! # Scientific Basis
//!
//! Per Popper (1959), a theory that makes no falsifiable predictions is not scientific.
//! A ComputeBrick with no assertions makes no testable claims and is therefore invalid.

// Submodules
mod batch;
mod buffer;
mod circuit;
mod connection;
mod memory;
mod perf_metrics;
mod profiling;
mod rate_limit;
mod resource_pool;
mod shutdown;

// Re-export profiling functions
pub use profiling::{
    cached_nanos, cached_nanos_or_now, cpu_cycles, get_page_faults, init_time_service,
    with_page_fault_tracking,
};

// Re-export perf_metrics types
pub use perf_metrics::{InferencePhase, PerfMetrics};

// Re-export memory types
#[cfg(not(target_arch = "wasm32"))]
pub use memory::AlignedBuffer;
pub use memory::{
    is_direct_io_aligned, madvise_region, prefetch_for_inference, prefetch_ptr, prefetch_slice,
    CacheAligned, MemoryAdvice, PrefetchLocality, CACHE_LINE_SIZE, CACHE_LINE_SIZE_F32,
    DIRECT_IO_ALIGNMENT,
};

// Re-export buffer types
pub use buffer::{BufferWatermarks, WatermarkedBuffer};

// Re-export circuit breaker types
pub use circuit::{CircuitBreaker, CircuitState};

// Re-export shutdown types
pub use shutdown::{GracefulShutdown, ShutdownGuard, ShutdownResult};

// Re-export resource pool types
pub use resource_pool::{PooledResource, ResourcePool};

// Re-export rate limiting types
pub use rate_limit::{LimitError, ServeLimits};

// Re-export connection types
pub use connection::{ConnectionState, KeepAliveConfig, ManagedConnection};

// Re-export batch types
pub use batch::{balance211, split_batch, Balance211Iter, BatchSplitStrategy};

// KV cache management
mod kv_cache;
pub use kv_cache::{KvCacheManager, KvCacheSlotInfo, SequentialBatchOrderer};

// SIMD configuration
mod simd_config;
pub use simd_config::{
    unroll_tail_process, AmxTileConfig, LazySimdConfig, SimdBackendState, UnrollFactor,
    UnrollTailIterator,
};

// Execution graph and brick profiling types (PAR-073, PAR-200, PAR-201)
mod exec_graph;
pub use exec_graph::{
    BrickBottleneck, BrickCategory, BrickId, BrickSample, BrickStats, CategoryStats, EdgeType,
    ExecutionEdge, ExecutionGraph, ExecutionNode, ExecutionNodeId, PtxRegistry, SyncMode,
    TransferDirection,
};

// BrickProfiler and tile profiling (TILING-SPEC-001)
mod profiler;
pub use profiler::{
    fnv1a_f32_checksum, BrickIdTimer, BrickProfiler, BrickTimer, DivergenceInfo, KernelChecksum,
    TileLevel, TileStats, TileTimer,
};

// Model-level inference tracing (Phase 13, E.11)
mod tracing;
pub use tracing::{
    AttentionTraceConfig, AttentionWeightTrace, KvCacheSessionTrace, KvCacheStateTrace,
    LayerActivationTrace, LogitEvolutionTrace, ModelActivationTrace, ModelQuantizationError,
    ModelTracer, ModelTracerConfig, ModelTracerSummary, QuantType, QuantizationErrorTrace,
    TensorStats, TokenLogitEvolution,
};

// Async and buffer patterns (Phase 12, E.10)
mod patterns;
pub use patterns::{
    reserve_capacity, AsyncResult, BoundedQueue, DualWakerState, FlowControlError,
    GraphReuseCounter, ReserveStrategy, StrategicBuffer, StreamCapacity, WakeDecision,
    WakeSkipState,
};

// Built-in compute operations
mod ops;
pub use ops::{AddOp, DotOp, MatmulOp, SoftmaxOp};

// Fused operations for transformer inference (PMAT-PERF-009)
mod fused_ops;
pub use fused_ops::{FusedGateUpOp, FusedGateUpWeights, FusedQKVOp, FusedQKVWeights};

// SIMD-optimized attention operation (PMAT-017)
mod attention;
pub use attention::AttentionOp;

// Q5_K and Q6_K quantization operations (llama.cpp compatible)
mod quant_ops;
pub use quant_ops::{BlockQ5K, BlockQ6K, DotQ5KOp, DotQ6KOp};

// Tests (7,400+ lines extracted for TDG compliance)
#[cfg(test)]
mod tests;

// Async task profiler (Phase 11, E.9.4)
mod async_profiler;
pub use async_profiler::AsyncTaskProfiler;

// Budget types (TokenBudget, ByteBudget, TokenResult)
mod budget;
pub use budget::{ByteBudget, TokenBudget, TokenResult};

// Core brick types (ComputeBackend, BrickError, ComputeAssertion, ComputeOp, etc.)
mod types;
pub use types::{
    AssertionResult, Backend, BrickError, BrickVerification, ComputeAssertion, ComputeBackend,
    ComputeOp,
};

// ComputeBrick struct, builder methods, and BrickLayer composition
mod compute_brick;
pub use compute_brick::{BrickLayer, ComputeBrick};
