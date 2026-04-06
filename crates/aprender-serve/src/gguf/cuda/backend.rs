//! CUDA backend for kernel configuration and PTX generation
//!
//! This module provides the `CudaBackend` struct for configuring CUDA kernels
//! and generating PTX code using trueno-gpu primitives.
//!
//! # Features
//!
//! - **IMP-312**: Q4_K quantized GEMM kernel (dequant + matmul fusion)
//! - **IMP-313**: FlashAttention-style tiled attention
//! - **IMP-314**: Paged KV cache memory management
//! - **IMP-315**: CUDA graph capture helpers
//!
//! # Example
//!
//! ```rust,ignore
//! use realizar::gguf::CudaBackend;
//!
//! let cuda = CudaBackend::new(1024, 1024, 4096, 64);
//! let ptx = cuda.q4k_gemm_ptx();  // Get PTX for Q4_K GEMM kernel
//! let attention_ptx = cuda.flash_attention_ptx(2048, 64, true);  // Causal attention
//! ```

use trueno_gpu::kernels::{AttentionKernel, Kernel, QuantizeKernel};

/// CUDA backend for kernel configuration and PTX generation
///
/// Provides dimension-aware kernel generation and launch configuration
/// for CUDA-accelerated inference operations.
///
/// # Example
///
/// ```rust,ignore
/// use realizar::gguf::CudaBackend;
///
/// let cuda = CudaBackend::new(1024, 1024, 4096, 64);
/// let ptx = cuda.q4k_gemm_ptx();  // Get PTX for Q4_K GEMM kernel
/// let attention_ptx = cuda.flash_attention_ptx(2048, 64, true);  // Causal attention
/// ```
#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
pub struct CudaBackend {
    /// Output rows (M) for GEMM operations
    pub m: u32,
    /// Output columns (N) for GEMM operations
    pub n: u32,
    /// Inner dimension (K) - must be divisible by Q4_K block size (32)
    pub k: u32,
    /// Head dimension for attention (typically 64 or 128)
    pub head_dim: u32,
    /// Number of attention heads
    pub num_heads: u32,
    /// Maximum sequence length for KV cache
    pub max_seq_len: u32,
    /// Cached PTX for Q4_K GEMM kernel (IMP-312)
    q4k_gemm_ptx_cache: std::cell::RefCell<Option<String>>,
    /// Cached PTX for FlashAttention kernel (IMP-313)
    flash_attention_ptx_cache: std::cell::RefCell<Option<String>>,
}

#[cfg(feature = "cuda")]
impl CudaBackend {
    /// Create a new CUDA backend with specified dimensions
    ///
    /// # Arguments
    /// * `m` - Output rows for GEMM
    /// * `n` - Output columns for GEMM
    /// * `k` - Inner dimension (should be divisible by 32 for Q4_K)
    /// * `head_dim` - Head dimension for attention (typically 64)
    #[must_use]
    pub fn new(m: u32, n: u32, k: u32, head_dim: u32) -> Self {
        Self {
            m,
            n,
            k,
            head_dim,
            num_heads: 32,     // Default for many models
            max_seq_len: 2048, // Default context length
            q4k_gemm_ptx_cache: std::cell::RefCell::new(None),
            flash_attention_ptx_cache: std::cell::RefCell::new(None),
        }
    }

    /// Set the number of attention heads
    #[must_use]
    pub const fn with_num_heads(mut self, num_heads: u32) -> Self {
        self.num_heads = num_heads;
        self
    }

    /// Set the maximum sequence length for KV cache
    #[must_use]
    pub const fn with_max_seq_len(mut self, max_seq_len: u32) -> Self {
        self.max_seq_len = max_seq_len;
        self
    }

    // ========================================================================
    // IMP-312: CUDA Q4_K Dequant+Matmul Kernel
    // ========================================================================

    /// Generate PTX for Q4_K quantized GEMM kernel (IMP-312)
    ///
    /// The kernel fuses dequantization with matrix multiplication:
    /// - Dequantization: val = scale * quant + min (per Q4_K block)
    /// - Matrix multiply: C = A × dequant(B)
    ///
    /// # Performance
    /// - Uses warp shuffle for efficient reduction
    /// - Shared memory for dequantized tiles
    /// - Coalesced memory access patterns
    #[must_use]
    pub fn q4k_gemm_ptx(&self) -> String {
        // Check cache first
        if let Some(cached) = self.q4k_gemm_ptx_cache.borrow().as_ref() {
            return cached.clone();
        }

        // Generate PTX using trueno-gpu
        let kernel = QuantizeKernel::new(self.m, self.n, self.k);
        let ptx = kernel.emit_ptx();

        // Cache the result
        *self.q4k_gemm_ptx_cache.borrow_mut() = Some(ptx.clone());
        ptx
    }

    /// Get kernel name for Q4_K GEMM
    #[must_use]
    pub fn q4k_gemm_kernel_name(&self) -> &'static str {
        "q4k_gemm_fused"
    }

    /// Get number of Q4_K blocks per row (K / 32)
    #[must_use]
    pub const fn q4k_blocks_per_row(&self) -> u32 {
        self.k / 32
    }

    /// Estimate Q4_K weight memory size in bytes
    /// Each block: 2 bytes header (scale+min) + 16 bytes data = 18 bytes for 32 weights
    #[must_use]
    pub const fn q4k_weight_bytes(&self) -> usize {
        let blocks_per_row = self.k / 32;
        let bytes_per_row = blocks_per_row * 18;
        (self.n as usize) * (bytes_per_row as usize)
    }

    // ========================================================================
    // IMP-313: CUDA FlashAttention Kernel
    // ========================================================================

    /// Generate PTX for FlashAttention-style tiled attention (IMP-313)
    ///
    /// Implements IO-aware attention per Dao et al. [16]:
    /// - Never materializes the full N×N attention matrix
    /// - Online softmax with running max and sum
    /// - O(N × d) memory instead of O(N²)
    ///
    /// # Arguments
    /// * `seq_len` - Sequence length (N)
    /// * `head_dim` - Head dimension (d)
    /// * `causal` - Enable causal masking for autoregressive models
    #[must_use]
    pub fn flash_attention_ptx(&self, seq_len: u32, head_dim: u32, causal: bool) -> String {
        let kernel = if causal {
            AttentionKernel::new(seq_len, head_dim).with_causal()
        } else {
            AttentionKernel::new(seq_len, head_dim)
        };
        kernel.emit_ptx()
    }

    /// Generate PTX for causal FlashAttention (cached version)
    #[must_use]
    pub fn flash_attention_causal_ptx(&self) -> String {
        // Check cache first
        if let Some(cached) = self.flash_attention_ptx_cache.borrow().as_ref() {
            return cached.clone();
        }

        // Generate causal attention PTX
        let ptx = self.flash_attention_ptx(self.max_seq_len, self.head_dim, true);

        // Cache the result
        *self.flash_attention_ptx_cache.borrow_mut() = Some(ptx.clone());
        ptx
    }

    /// Get kernel name for FlashAttention
    #[must_use]
    pub const fn flash_attention_kernel_name(&self, causal: bool) -> &'static str {
        if causal {
            "flash_attention_causal"
        } else {
            "flash_attention"
        }
    }

    /// Estimate shared memory size for FlashAttention (in bytes)
    /// Uses tiles of Q (B_r × d) and KV (B_c × d × 2)
    #[must_use]
    pub const fn flash_attention_smem_bytes(&self) -> usize {
        let tile_q = 64_u32;
        let tile_kv = 64_u32;
        let d = self.head_dim;
        // Q tile + K tile + V tile, all f32
        ((tile_q * d + tile_kv * d * 2) * 4) as usize
    }

    // ========================================================================
    // IMP-314: CUDA KV Cache with Paged Memory
    // ========================================================================

    /// Calculate KV cache memory size per layer in bytes
    ///
    /// KV cache stores Key and Value tensors for attention:
    /// - K: [num_heads, seq_len, head_dim] × sizeof(f32)
    /// - V: [num_heads, seq_len, head_dim] × sizeof(f32)
    #[must_use]
    pub const fn kv_cache_bytes_per_layer(&self) -> usize {
        let k_size = self.num_heads * self.max_seq_len * self.head_dim * 4;
        let v_size = self.num_heads * self.max_seq_len * self.head_dim * 4;
        (k_size + v_size) as usize
    }

    /// Calculate total KV cache memory for all layers
    #[must_use]
    pub const fn kv_cache_total_bytes(&self, num_layers: u32) -> usize {
        self.kv_cache_bytes_per_layer() * (num_layers as usize)
    }

    /// Get page size for paged KV cache (IMP-314)
    /// Default: 64 tokens per page to balance memory efficiency and fragmentation
    #[must_use]
    pub const fn kv_cache_page_tokens(&self) -> u32 {
        64
    }

    /// Calculate number of pages needed for given sequence length
    #[must_use]
    pub const fn kv_cache_pages_needed(&self, seq_len: u32) -> u32 {
        let page_size = self.kv_cache_page_tokens();
        seq_len.div_ceil(page_size)
    }

    // ========================================================================
    // IMP-315: CUDA Graph Capture Helpers
    // ========================================================================

    /// Get CUDA launch configuration for Q4_K GEMM kernel
    ///
    /// Returns (grid_dim, block_dim) tuple for kernel launch
    #[must_use]
    pub const fn q4k_gemm_launch_config(&self) -> ((u32, u32, u32), (u32, u32, u32)) {
        let tile_size = 32_u32;
        let grid_x = self.n.div_ceil(tile_size);
        let grid_y = self.m.div_ceil(tile_size);
        let grid = (grid_x, grid_y, 1);
        let block = (tile_size * tile_size, 1, 1);
        (grid, block)
    }

    /// Get CUDA launch configuration for FlashAttention kernel
    #[must_use]
    pub const fn flash_attention_launch_config(
        &self,
        seq_len: u32,
    ) -> ((u32, u32, u32), (u32, u32, u32)) {
        let tile_q = 64_u32;
        let num_q_blocks = seq_len.div_ceil(tile_q);
        let grid = (num_q_blocks, self.num_heads, 1);
        let block = (tile_q * self.head_dim, 1, 1);
        (grid, block)
    }

    /// Check if dimensions are valid for CUDA kernels
    #[must_use]
    pub const fn validate_dimensions(&self) -> bool {
        // K must be divisible by Q4_K block size (32)
        let k_valid = self.k.is_multiple_of(32);
        // Head dim should be power of 2 for efficient memory access
        let head_dim_valid = self.head_dim.is_power_of_two();
        // Dimensions must be non-zero
        let non_zero = self.m > 0 && self.n > 0 && self.k > 0 && self.head_dim > 0;
        k_valid && head_dim_valid && non_zero
    }

    /// Get PTX target SM version (default: sm_89 for Ada Lovelace/RTX 4090)
    #[must_use]
    pub const fn ptx_target(&self) -> &'static str {
        "sm_89"
    }

    /// Get PTX version (default: 8.0)
    #[must_use]
    pub const fn ptx_version(&self) -> (u32, u32) {
        (8, 0)
    }
}

include!("backend_cuda.rs");
