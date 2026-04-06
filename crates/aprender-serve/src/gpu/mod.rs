//! GPU Acceleration Module (Phase 4)
//!
//! Provides GPU-accelerated compute primitives via trueno's wgpu backend.
//!
//! ## Architecture
//!
//! ```text
//! +-----------------------+
//! |    GpuCompute API     |  <- Safe public API
//! +-----------------------+
//! |   trueno::GpuBackend  |  <- wgpu-based GPU compute
//! +-----------------------+
//! |   wgpu Device/Queue   |  <- WebGPU abstraction
//! +-----------------------+
//! ```
//!
//! ## Usage
//!
//! ```rust,ignore
//! use realizar::gpu::{GpuCompute, ComputeBackend};
//!
//! // Auto-select best backend
//! let compute = GpuCompute::auto()?;
//!
//! // GPU matmul
//! let c = compute.matmul(&a, &b, m, k, n)?;
//! ```
//!
//! ## Performance Targets (Refs REALIZAR-PERF-SPEC-001)
//!
//! | Operation | GPU Target | CPU Baseline |
//! |-----------|------------|--------------|
//! | matmul    | 20x faster | 1x           |
//! | tok/s     | ≥100       | ≥25          |

use crate::error::Result;
use std::collections::HashMap;
use std::time::Duration;

// PMAT-802: Extracted modules
pub mod adapters;
mod allocator;
pub mod backend;
mod batch_scheduling;
mod diagnostics;
pub mod executor;
mod metrics;
pub mod mock_backend;
pub mod planner; // Phase 47: Plan/Execute separation
mod resilience;
pub mod scheduler;
mod simd_ops;
mod streaming_kv;

// Types available without cuda feature
#[cfg(feature = "cuda")]
pub use scheduler::CudaScheduler;
pub use scheduler::{
    AttentionBuffers, BlockWeights, GpuGenerateConfig, GpuModel, GpuModelConfig, WeightType,
};

// Planner exports (Phase 47: Plan/Execute separation)
pub use planner::{
    plan_lm_head_path, plan_sampling, BatchPlanner, BlockForwardPlan, GenerationConfig,
    GenerationStep, LmHeadPath, SamplingStrategy,
};

// Allocator exports (M21, M22)
pub use allocator::{
    blocked_matmul, naive_matmul, prefetch_read, sequential_sum, sum_with_prefetch,
    CacheAlignedBuffer, ForwardArena, ScratchBuffer, TensorPool,
};

// Diagnostics exports (M32)
pub use diagnostics::{
    DebugMode, DiagnosticsCollector, DiagnosticsSummary, LogConfig, LogEntry, LogLevel, Logger,
    MemoryReport, MemoryTracker, PhaseTimer, RequestCapture, StateDump,
};

// Resilience exports (M31)
pub use resilience::{
    BulkheadConfig, BulkheadManager, BulkheadPermit, BulkheadStats, CircuitBreaker, CircuitConfig,
    CircuitState, ErrorCategory, RequestType, RetryConfig, RetryDecision, RetryPolicy,
};

// SIMD ops exports (M18)
pub use simd_ops::{scalar_rope, scalar_softmax, simd_rope, simd_softmax};

// Streaming KV cache exports (M6)
pub use streaming_kv::{StreamingKVCache, StreamingKVCacheFp16};

// Metrics exports (M28)
pub use metrics::{
    AsyncGpuResult, ComputeBackend, GpuBufferPool, GpuCompute, GpuPoolStats, HealthChecker,
    HybridScheduler, InferenceMetrics, ShutdownCoordinator,
};
// Internal-only matmul functions (used by scheduler module and apr::helpers)
pub(crate) use metrics::{cpu_matmul, cpu_matmul_transpose_b, cpu_matmul_transposed_simd};

// Internal-only RMSNorm (canonical impl, used by apr::helpers delegation)
pub(crate) use scheduler::layer_norm_static;

// Batch scheduling exports (M25, M26, M27)
pub use batch_scheduling::{
    AllocationId, AsyncRequestQueue, BatchId, InferenceBatchScheduler, InferenceCompletionHandler,
    InferenceEventNotifier, Priority, PriorityRequest, PriorityRequestQueue, RequestId,
    ResourceTracker, SpeculativeBuffer, TimeoutManager, TokenBatch, TokenRateLimiter,
};

// ============================================================================
// GPU Buffer Limits (IMP-090, IMP-091)
// ============================================================================

/// Maximum GPU buffer size in bytes (wgpu limit: 256MB)
const MAX_GPU_BUFFER_BYTES: usize = 256 * 1024 * 1024;

/// Vocab size threshold above which we use CPU for embedding/lm_head
/// This is calculated as: MAX_GPU_BUFFER_BYTES / (hidden_dim * sizeof(f32))
/// For hidden_dim=1536: 256MB / (1536 * 4) = 43,690 tokens
/// We use 65536 as a round threshold that works for most models
pub const LARGE_VOCAB_THRESHOLD: usize = 65536;

/// Check if a matrix operation would exceed GPU buffer limits (IMP-090)
///
/// Returns true if the operation should use CPU fallback
#[inline]
#[must_use]
pub fn exceeds_gpu_buffer_limit(elements: usize) -> bool {
    elements * std::mem::size_of::<f32>() > MAX_GPU_BUFFER_BYTES
}

/// Matmul batch operation: (A matrix, B matrix, m rows, k cols, n cols)
pub type MatmulOp = (Vec<f32>, Vec<f32>, usize, usize, usize);

// ============================================================================
// Contiguous Attention Buffer (M19 - IMP-040)
// ============================================================================

/// Contiguous memory buffer for attention tensors (M19 - IMP-040)
///
/// Pre-allocates a single contiguous block for Q, K, V, O tensors
/// to reduce memory fragmentation during attention computation.
#[derive(Debug)]
pub struct ContiguousAttentionBuffer {
    /// Single contiguous allocation for all tensors
    data: Vec<f32>,
    /// Maximum sequence length
    max_seq_len: usize,
    /// Number of attention heads (stored for future use)
    #[allow(dead_code)]
    num_heads: usize,
    /// Dimension per head (stored for future use)
    #[allow(dead_code)]
    head_dim: usize,
    /// Size of each tensor (Q, K, V, O have same size)
    tensor_size: usize,
}

impl ContiguousAttentionBuffer {
    /// Create a new contiguous attention buffer
    #[must_use]
    pub fn new(max_seq_len: usize, num_heads: usize, head_dim: usize) -> Self {
        let tensor_size = max_seq_len * num_heads * head_dim;
        // Allocate 4x for Q, K, V, O in single contiguous block
        let data = vec![0.0f32; tensor_size * 4];

        Self {
            data,
            max_seq_len,
            num_heads,
            head_dim,
            tensor_size,
        }
    }

    /// Check if buffer is contiguous (always true for this implementation)
    #[must_use]
    pub fn is_contiguous(&self) -> bool {
        // Buffer is contiguous by construction
        true
    }

    /// Get views into Q, K, V, O tensors
    #[must_use]
    pub fn get_views(&self) -> (&[f32], &[f32], &[f32], &[f32]) {
        let q_start = 0;
        let k_start = self.tensor_size;
        let v_start = self.tensor_size * 2;
        let o_start = self.tensor_size * 3;

        (
            &self.data[q_start..k_start],
            &self.data[k_start..v_start],
            &self.data[v_start..o_start],
            &self.data[o_start..],
        )
    }

    /// Get mutable views into Q, K, V, O tensors
    pub fn get_views_mut(&mut self) -> (&mut [f32], &mut [f32], &mut [f32], &mut [f32]) {
        let tensor_size = self.tensor_size;

        // Split the data into 4 mutable slices
        let (q, rest) = self.data.split_at_mut(tensor_size);
        let (k, rest) = rest.split_at_mut(tensor_size);
        let (v, o) = rest.split_at_mut(tensor_size);

        (q, k, v, o)
    }

    /// Reset buffer to zeros for reuse
    pub fn reset(&mut self) {
        self.data.fill(0.0);
    }

    /// Get maximum sequence length
    #[must_use]
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }
}

// ============================================================================
// Batch Processing & Parallel Execution (M20 - IMP-043/044/045)
// ============================================================================

/// Batch token embedding lookup (M20 - IMP-043)
///
/// Performs vectorized embedding lookup for multiple tokens at once.
/// More efficient than individual lookups due to better memory access patterns.
#[must_use]
pub fn batch_embed(embedding_table: &[f32], tokens: &[usize], hidden_dim: usize) -> Vec<f32> {
    if tokens.is_empty() || embedding_table.is_empty() {
        return Vec::new();
    }

    let mut result = Vec::with_capacity(tokens.len() * hidden_dim);

    // Batch copy embeddings
    for &token in tokens {
        let start_idx = token * hidden_dim;
        let end_idx = start_idx + hidden_dim;
        if end_idx <= embedding_table.len() {
            result.extend_from_slice(&embedding_table[start_idx..end_idx]);
        } else {
            // N-09: OOB token → zeros. Contract: embedding-lookup-v1.yaml
            // requires token_ids[i] < vocab_size.
            eprintln!(
                "Warning: batch_embed token {} OOB (offset={start_idx}, table_len={}). N-09 escape.",
                token, embedding_table.len()
            );
            result.extend(std::iter::repeat_n(0.0, hidden_dim));
        }
    }

    result
}

/// Sequential FFN computation (baseline for comparison)
///
/// Standard two-layer feed-forward network: up projection -> activation -> down projection
#[must_use]
pub fn sequential_ffn(
    input: &[f32],
    w_up: &[f32],
    w_down: &[f32],
    hidden_dim: usize,
    intermediate_dim: usize,
) -> Vec<f32> {
    if input.is_empty() {
        return Vec::new();
    }

    // Up projection: (hidden_dim,) @ (hidden_dim, intermediate_dim) -> (intermediate_dim,)
    let mut intermediate = vec![0.0f32; intermediate_dim];
    for i in 0..intermediate_dim {
        let mut sum = 0.0f32;
        for j in 0..hidden_dim {
            sum += input[j] * w_up[j * intermediate_dim + i];
        }
        // GELU activation
        intermediate[i] =
            sum * 0.5 * (1.0 + (sum * 0.797_884_5 * (1.0 + 0.044_715 * sum * sum)).tanh());
    }

    // Down projection: (intermediate_dim,) @ (intermediate_dim, hidden_dim) -> (hidden_dim,)
    let mut output = vec![0.0f32; hidden_dim];
    for i in 0..hidden_dim {
        let mut sum = 0.0f32;
        for j in 0..intermediate_dim {
            sum += intermediate[j] * w_down[j * hidden_dim + i];
        }
        output[i] = sum;
    }

    output
}

/// Parallel FFN computation (M20 - IMP-044)
///
/// Uses rayon parallelism for the down projection matmul.
#[must_use]
pub fn parallel_ffn(
    input: &[f32],
    w_up: &[f32],
    w_down: &[f32],
    hidden_dim: usize,
    intermediate_dim: usize,
) -> Vec<f32> {
    use rayon::prelude::*;

    if input.is_empty() {
        return Vec::new();
    }

    // Up projection with GELU (sequential - typically smaller)
    let intermediate: Vec<f32> = (0..intermediate_dim)
        .map(|i| {
            let sum: f32 = (0..hidden_dim)
                .map(|j| input[j] * w_up[j * intermediate_dim + i])
                .sum();
            // GELU activation
            sum * 0.5 * (1.0 + (sum * 0.797_884_5 * (1.0 + 0.044_715 * sum * sum)).tanh())
        })
        .collect();

    // Down projection with rayon parallelism
    let output: Vec<f32> = (0..hidden_dim)
        .into_par_iter()
        .map(|i| {
            (0..intermediate_dim)
                .map(|j| intermediate[j] * w_down[j * hidden_dim + i])
                .sum()
        })
        .collect();

    output
}

/// Standard two-pass layer normalization (baseline for comparison)
///
/// First pass computes mean, second pass computes variance.
#[must_use]
pub fn standard_layernorm(input: &[f32], gamma: &[f32], beta: &[f32], eps: f32) -> Vec<f32> {
    if input.is_empty() {
        return Vec::new();
    }

    let n = input.len() as f32;

    // First pass: compute mean
    let mean: f32 = input.iter().sum::<f32>() / n;

    // Second pass: compute variance
    let variance: f32 = input.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / n;

    let std_dev = (variance + eps).sqrt();

    // Normalize and apply gamma/beta
    input
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            let normalized = (x - mean) / std_dev;
            normalized * gamma.get(i).copied().unwrap_or(1.0) + beta.get(i).copied().unwrap_or(0.0)
        })
        .collect()
}

/// Fused single-pass layer normalization using Welford's algorithm (M20 - IMP-045)
///
/// Computes mean and variance in a single pass, reducing memory bandwidth.
#[must_use]
pub fn fused_layernorm(input: &[f32], gamma: &[f32], beta: &[f32], eps: f32) -> Vec<f32> {
    if input.is_empty() {
        return Vec::new();
    }

    let n = input.len();

    // Welford's online algorithm for mean and variance
    let mut mean = 0.0f32;
    let mut m2 = 0.0f32;

    for (i, &x) in input.iter().enumerate() {
        let delta = x - mean;
        mean += delta / (i + 1) as f32;
        let delta2 = x - mean;
        m2 += delta * delta2;
    }

    let variance = m2 / n as f32;
    let std_dev = (variance + eps).sqrt();
    let inv_std = 1.0 / std_dev;

    // Normalize and apply gamma/beta in single pass
    input
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            let normalized = (x - mean) * inv_std;
            normalized * gamma.get(i).copied().unwrap_or(1.0) + beta.get(i).copied().unwrap_or(0.0)
        })
        .collect()
}

// ============================================================================
// Phase 14: Quantized Compute Kernels (M23)
// ============================================================================

/// Quantized dot product for Q4_0 blocks (M23 - IMP-052)
///
/// Computes dot product directly on Q4_0 quantized data without full dequantization.
/// Q4_0 format: 2 bytes (f16 scale) + 16 bytes (32 x 4-bit values)
#[must_use]
pub fn quantized_dot_q4(block_a: &[u8], block_b: &[u8]) -> f32 {
    if block_a.len() < 18 || block_b.len() < 18 {
        return 0.0;
    }

    // Extract scales (f16 little-endian)
    let scale_a = half::f16::from_le_bytes([block_a[0], block_a[1]]).to_f32();
    let scale_b = half::f16::from_le_bytes([block_b[0], block_b[1]]).to_f32();

    // Accumulate dot product over packed 4-bit values
    let mut acc = 0i32;

    for i in 0..16 {
        let byte_a = block_a[2 + i];
        let byte_b = block_b[2 + i];

        // Extract low and high nibbles, center at 8
        let a_lo = (byte_a & 0x0F) as i32 - 8;
        let a_hi = ((byte_a >> 4) & 0x0F) as i32 - 8;
        let b_lo = (byte_b & 0x0F) as i32 - 8;
        let b_hi = ((byte_b >> 4) & 0x0F) as i32 - 8;

        acc += a_lo * b_lo + a_hi * b_hi;
    }

    // Apply combined scale
    (acc as f32) * scale_a * scale_b
}

include!("mod_quantized_dot.rs");
include!("mod_max_error_recovery.rs");
include!("mod_connection_state.rs");
include!("resource_monitor.rs");
