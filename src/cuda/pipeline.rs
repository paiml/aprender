//! Async Pipeline and PTX Optimization
//!
//! This module provides:
//! - `AsyncPipeline`: Multi-stream async execution for overlapping compute and transfer
//! - `PtxOptimizer`: PTX optimization hints for kernel generation
//! - Kernel presets for common LLM inference patterns

use trueno_gpu::driver::{CudaContext, CudaStream};
use trueno_gpu::GpuError;

use crate::cuda::kernels::KernelType;

// ============================================================================
// Async Pipeline (IMP-1000c)
// ============================================================================

/// Multi-stream async execution pipeline for overlapping compute and transfer
///
/// Uses separate streams for:
/// - Compute: kernel execution
/// - Transfer: H2D and D2H memory copies
///
/// This enables hiding PCIe transfer latency by overlapping with computation.
pub struct AsyncPipeline {
    /// Stream for compute operations (kernel launches)
    compute_stream: CudaStream,
    /// Stream for memory transfers (H2D, D2H)
    transfer_stream: CudaStream,
    /// Number of layers queued
    layers_queued: usize,
    /// Whether pipeline is active
    active: bool,
}

impl AsyncPipeline {
    /// Create a new async pipeline with separate compute and transfer streams
    ///
    /// # Errors
    ///
    /// Returns error if stream creation fails.
    pub fn new(context: &CudaContext) -> Result<Self, GpuError> {
        let compute_stream = CudaStream::new(context)?;
        let transfer_stream = CudaStream::new(context)?;

        Ok(Self {
            compute_stream,
            transfer_stream,
            layers_queued: 0,
            active: false,
        })
    }

    /// Start the pipeline
    pub fn begin(&mut self) {
        self.active = true;
        self.layers_queued = 0;
    }

    /// Enqueue a layer for async execution
    ///
    /// Returns the layer index for tracking.
    pub fn enqueue_layer(&mut self) -> usize {
        let layer_idx = self.layers_queued;
        self.layers_queued += 1;
        layer_idx
    }

    /// Get the compute stream for kernel launches
    #[must_use]
    pub fn compute_stream(&self) -> &CudaStream {
        &self.compute_stream
    }

    /// Get the transfer stream for memory operations
    #[must_use]
    pub fn transfer_stream(&self) -> &CudaStream {
        &self.transfer_stream
    }

    /// Synchronize both streams (wait for all operations to complete)
    ///
    /// # Errors
    ///
    /// Returns error if synchronization fails.
    pub fn sync(&self) -> Result<(), GpuError> {
        self.compute_stream.synchronize()?;
        self.transfer_stream.synchronize()?;
        Ok(())
    }

    /// End the pipeline and synchronize
    ///
    /// # Errors
    ///
    /// Returns error if synchronization fails.
    pub fn end(&mut self) -> Result<(), GpuError> {
        self.sync()?;
        self.active = false;
        Ok(())
    }

    /// Check if pipeline is active
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.active
    }

    /// Get number of layers queued
    #[must_use]
    pub fn layers_queued(&self) -> usize {
        self.layers_queued
    }
}

// ============================================================================
// PTX Micro-optimization (IMP-1000d)
// ============================================================================

/// Memory access pattern hints for PTX optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MemoryPattern {
    /// Scalar loads (ld.global.f32)
    #[default]
    Scalar,
    /// Vectorized 2-element loads (ld.global.v2.f32)
    Vector2,
    /// Vectorized 4-element loads (ld.global.v4.f32)
    Vector4,
}

/// Register tiling configuration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RegisterTiling {
    /// Tile width per thread
    pub width: u32,
    /// Tile height per thread
    pub height: u32,
}

impl Default for RegisterTiling {
    fn default() -> Self {
        Self {
            width: 4,
            height: 4,
        }
    }
}

impl RegisterTiling {
    /// Create 8x8 register tiling (optimal for A100/H100)
    #[must_use]
    pub const fn large() -> Self {
        Self {
            width: 8,
            height: 8,
        }
    }

    /// Create 4x4 register tiling (balanced)
    #[must_use]
    pub const fn medium() -> Self {
        Self {
            width: 4,
            height: 4,
        }
    }

    /// Create 2x2 register tiling (low register pressure)
    #[must_use]
    pub const fn small() -> Self {
        Self {
            width: 2,
            height: 2,
        }
    }

    /// Calculate registers needed for this tiling
    #[must_use]
    pub const fn registers_needed(&self) -> u32 {
        self.width * self.height
    }
}

/// Shared memory bank conflict avoidance strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BankConflictStrategy {
    /// No conflict avoidance
    #[default]
    None,
    /// Padding to avoid conflicts (adds +1 element per row)
    Padding,
    /// XOR-based conflict avoidance
    Xor,
}

/// PTX optimization hints for kernel generation
///
/// These hints guide PTX code generation for optimal performance.
/// Not all hints are applicable to all kernels.
#[derive(Debug, Clone, Default)]
pub struct PtxOptimizationHints {
    /// Memory access pattern for global loads/stores
    pub memory_pattern: MemoryPattern,
    /// Register tiling configuration
    pub register_tiling: RegisterTiling,
    /// Bank conflict avoidance strategy
    pub bank_conflict_strategy: BankConflictStrategy,
    /// Target occupancy (0.0-1.0, 0 = auto)
    pub target_occupancy: f32,
    /// Enable instruction-level parallelism hints
    pub enable_ilp: bool,
    /// Preferred shared memory size (0 = default)
    pub shared_mem_preference: u32,
}

impl PtxOptimizationHints {
    /// Create optimization hints for maximum throughput
    #[must_use]
    pub fn max_throughput() -> Self {
        Self {
            memory_pattern: MemoryPattern::Vector4,
            register_tiling: RegisterTiling::large(),
            bank_conflict_strategy: BankConflictStrategy::Padding,
            target_occupancy: 0.75,
            enable_ilp: true,
            shared_mem_preference: 0,
        }
    }

    /// Create optimization hints for low latency
    #[must_use]
    pub fn low_latency() -> Self {
        Self {
            memory_pattern: MemoryPattern::Scalar,
            register_tiling: RegisterTiling::small(),
            bank_conflict_strategy: BankConflictStrategy::None,
            target_occupancy: 1.0,
            enable_ilp: false,
            shared_mem_preference: 0,
        }
    }

    /// Create balanced optimization hints
    #[must_use]
    pub fn balanced() -> Self {
        Self {
            memory_pattern: MemoryPattern::Vector2,
            register_tiling: RegisterTiling::medium(),
            bank_conflict_strategy: BankConflictStrategy::Padding,
            target_occupancy: 0.5,
            enable_ilp: true,
            shared_mem_preference: 0,
        }
    }

    /// Check if vectorized loads are enabled
    #[must_use]
    pub const fn uses_vectorized_loads(&self) -> bool {
        matches!(
            self.memory_pattern,
            MemoryPattern::Vector2 | MemoryPattern::Vector4
        )
    }

    /// Get the vector width for loads (1, 2, or 4)
    #[must_use]
    pub const fn vector_width(&self) -> u32 {
        match self.memory_pattern {
            MemoryPattern::Scalar => 1,
            MemoryPattern::Vector2 => 2,
            MemoryPattern::Vector4 => 4,
        }
    }

    /// Calculate recommended shared memory padding per row
    ///
    /// Returns 0 if no padding, 1 if padding enabled.
    #[must_use]
    pub const fn shared_mem_padding(&self) -> u32 {
        match self.bank_conflict_strategy {
            BankConflictStrategy::Padding => 1,
            _ => 0,
        }
    }
}

/// PTX optimizer that applies optimization hints
///
/// This struct provides methods to transform PTX code based on
/// optimization hints. Currently tracks hints for future use
/// when trueno-gpu adds vectorized load support.
pub struct PtxOptimizer {
    hints: PtxOptimizationHints,
}

impl PtxOptimizer {
    /// Create a new PTX optimizer with the given hints
    #[must_use]
    pub const fn new(hints: PtxOptimizationHints) -> Self {
        Self { hints }
    }

    /// Get the optimization hints
    #[must_use]
    pub const fn hints(&self) -> &PtxOptimizationHints {
        &self.hints
    }

    /// Generate optimization summary for debugging
    #[must_use]
    pub fn summary(&self) -> String {
        format!(
            "PtxOptimizer[vec={}, tile={}x{}, bank={:?}, ilp={}]",
            self.hints.vector_width(),
            self.hints.register_tiling.width,
            self.hints.register_tiling.height,
            self.hints.bank_conflict_strategy,
            self.hints.enable_ilp
        )
    }

    /// Calculate shared memory size with padding applied
    #[must_use]
    pub const fn padded_shared_mem_row(&self, row_elements: u32) -> u32 {
        row_elements + self.hints.shared_mem_padding()
    }

    /// Estimate register usage for the tiling configuration
    #[must_use]
    pub const fn estimated_registers(&self) -> u32 {
        // Base registers: thread ID, indices, etc
        let base = 16;
        // Accumulator registers for tiling
        let accum = self.hints.register_tiling.registers_needed();
        // Extra for ILP (double buffering)
        let ilp_extra = if self.hints.enable_ilp { accum } else { 0 };
        base + accum + ilp_extra
    }

    /// Check if optimization hints suggest high register pressure
    #[must_use]
    pub const fn is_high_register_pressure(&self) -> bool {
        self.estimated_registers() > 64
    }
}

/// Pre-configured kernel configurations for common LLM inference patterns
pub mod presets {
    use super::KernelType;

    /// Kernel preset for Llama-style attention
    pub fn llama_attention(seq_len: u32, head_dim: u32) -> KernelType {
        KernelType::Attention {
            seq_len,
            head_dim,
            causal: true,
        }
    }

    /// Kernel preset for feed-forward network GEMM
    pub fn ffn_gemm(batch: u32, hidden: u32, intermediate: u32) -> KernelType {
        KernelType::GemmTiled {
            m: batch,
            n: intermediate,
            k: hidden,
            tile_size: 32,
        }
    }

    /// Kernel preset for Q4_K quantized model (simplified format)
    pub fn q4k_inference(batch: u32, hidden: u32, k: u32) -> KernelType {
        KernelType::QuantizedGemm {
            m: batch,
            n: hidden,
            k,
        }
    }

    /// Kernel preset for Q4_K quantized model (GGML super-block format) - PARITY-041
    /// Uses real GGML Q4_K layout: 256 values per super-block, 144 bytes each
    /// k must be divisible by 256 (super-block size)
    pub fn q4k_ggml_inference(batch: u32, hidden: u32, k: u32) -> KernelType {
        debug_assert!(
            k.is_multiple_of(256),
            "k must be divisible by 256 for GGML super-blocks"
        );
        KernelType::QuantizedGemmGgml {
            m: batch,
            n: hidden,
            k,
        }
    }

    /// Kernel preset for RMSNorm (LayerNorm variant)
    pub fn rmsnorm(hidden_size: u32) -> KernelType {
        KernelType::LayerNorm {
            hidden_size,
            epsilon: 1e-6,
            affine: false,
        }
    }

    /// Kernel preset for multi-head attention (PARITY-043)
    /// Processes all heads in parallel for maximum GPU occupancy
    pub fn multi_head_attention(seq_len: u32, head_dim: u32, n_heads: u32) -> KernelType {
        KernelType::MultiHeadAttention {
            seq_len,
            head_dim,
            n_heads,
            causal: true, // Default to autoregressive/causal
        }
    }

    /// Kernel preset for phi-2 model multi-head attention (PARITY-043)
    /// phi-2: 32 heads, 80 head_dim (2560/32)
    pub fn phi2_multi_head_attention(seq_len: u32) -> KernelType {
        KernelType::MultiHeadAttention {
            seq_len,
            head_dim: 80,
            n_heads: 32,
            causal: true,
        }
    }

    /// Kernel preset for Tensor Core multi-head attention (REALIZAR-PARITY-001.3)
    /// Uses FP16 WMMA for ~40x speedup over FP32 baseline
    /// Requires sm_70+ (Volta, Turing, Ampere, Ada Lovelace, Hopper)
    pub fn tensor_core_attention(seq_len: u32, head_dim: u32, n_heads: u32) -> KernelType {
        KernelType::AttentionTensorCore {
            seq_len,
            head_dim,
            n_heads,
            causal: true, // Default to autoregressive/causal for LLM inference
        }
    }

    /// Kernel preset for Llama-style Tensor Core attention
    /// Llama: 32 heads, 128 head_dim (4096/32)
    pub fn llama_tensor_core_attention(seq_len: u32) -> KernelType {
        KernelType::AttentionTensorCore {
            seq_len,
            head_dim: 128,
            n_heads: 32,
            causal: true,
        }
    }
}

include!("pipeline_memory_pattern.rs");
