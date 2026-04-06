//! CUDA Test Fixtures - Phase 54 (Five-Whys Root Cause Fix)
//!
//! This module provides lightweight test fixtures for isolated kernel testing.
//! It addresses the root cause identified in Five-Whys analysis: missing Layer 2
//! (kernel parity) test infrastructure.
//!
//! ## Problem Statement
//!
//! BUG-GGUF-001 (Q4_0 layout mismatch) and the Q5_0 GQA bug were caught late because:
//! 1. End-to-end tests require full model files
//! 2. No synthetic weight generators for isolated testing
//! 3. No CPU reference paths exposed for parity comparison
//!
//! ## Solution: ModelFixture Pattern
//!
//! This module provides:
//! - `SyntheticWeights`: Generate quantized weights for any format
//! - `KernelParity`: Compare CPU dequantize+matmul vs GPU fused GEMV
//! - `MinimalExecutor`: Lightweight CudaExecutor for single-kernel tests
//!
//! ## Test Layers (Probar-style)
//!
//! - **Layer 1**: Unit tests (pure functions, no GPU)
//! - **Layer 2**: Kernel parity tests (CPU vs GPU for single ops) ← THIS MODULE
//! - **Layer 3**: Component tests (attention, FFN, etc.)
//! - **Layer 4**: Integration tests (full model inference)

#![cfg(feature = "cuda")]

use half::f16;

// ============================================================================
// Synthetic Weight Generators
// ============================================================================

/// Generate synthetic Q4_0 quantized weights for testing.
///
/// Q4_0 format: 18 bytes per block
/// - 2 bytes: f16 scale
/// - 16 bytes: 32 x 4-bit quantized values (packed as nibbles)
///
/// # Arguments
/// * `num_blocks` - Number of Q4_0 blocks to generate
///
/// # Returns
/// Vec<u8> of quantized weight data
pub fn generate_q4_0_weights(num_blocks: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(num_blocks * 18);

    for block_idx in 0..num_blocks {
        // Scale: predictable value for verification
        let scale = 0.1 * (block_idx as f32 + 1.0);
        let scale_f16 = f16::from_f32(scale);
        data.extend_from_slice(&scale_f16.to_le_bytes());

        // Quants: 16 bytes, each byte has 2 nibbles (low, high)
        for j in 0..16 {
            let low = ((block_idx + j) % 16) as u8;
            let high = ((block_idx + j + 1) % 16) as u8;
            data.push(low | (high << 4));
        }
    }

    data
}

/// Generate synthetic Q5_0 quantized weights for testing.
///
/// Q5_0 format: 22 bytes per block
/// - 2 bytes: f16 scale
/// - 4 bytes: qh (high bits for 32 values, packed as u32)
/// - 16 bytes: qs (low 4 bits for 32 values, packed as nibbles)
///
/// # Arguments
/// * `num_blocks` - Number of Q5_0 blocks to generate
///
/// # Returns
/// Vec<u8> of quantized weight data
pub fn generate_q5_0_weights(num_blocks: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(num_blocks * 22);

    for block_idx in 0..num_blocks {
        // Scale: predictable value for verification
        let scale = 0.1 * (block_idx as f32 + 1.0);
        let scale_f16 = f16::from_f32(scale);
        data.extend_from_slice(&scale_f16.to_le_bytes());

        // qh: 4 bytes containing high bits for 32 values
        // Use alternating pattern for predictable verification
        let qh: u32 = 0xAAAA_5555;
        data.extend_from_slice(&qh.to_le_bytes());

        // qs: 16 bytes, each byte has 2 nibbles (low 4 bits)
        for j in 0..16 {
            let low = ((block_idx + j) % 16) as u8;
            let high = ((block_idx + j + 1) % 16) as u8;
            data.push(low | (high << 4));
        }
    }

    data
}

/// Generate synthetic Q4_1 quantized weights for testing.
///
/// Q4_1 format: 20 bytes per block
/// - 2 bytes: f16 scale (d)
/// - 2 bytes: f16 min (m)
/// - 16 bytes: 32 x 4-bit quantized values (packed as nibbles)
///
/// # Arguments
/// * `num_blocks` - Number of Q4_1 blocks to generate
///
/// # Returns
/// Vec<u8> of quantized weight data
pub fn generate_q4_1_weights(num_blocks: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(num_blocks * 20);

    for block_idx in 0..num_blocks {
        // Scale (d): predictable value
        let scale = 0.1 * (block_idx as f32 + 1.0);
        let scale_f16 = f16::from_f32(scale);
        data.extend_from_slice(&scale_f16.to_le_bytes());

        // Min (m): offset for this block
        let min = -0.5 * (block_idx as f32 + 1.0);
        let min_f16 = f16::from_f32(min);
        data.extend_from_slice(&min_f16.to_le_bytes());

        // Quants: 16 bytes, each byte has 2 nibbles
        for j in 0..16 {
            let low = ((block_idx + j) % 16) as u8;
            let high = ((block_idx + j + 1) % 16) as u8;
            data.push(low | (high << 4));
        }
    }

    data
}

/// Generate synthetic Q8_0 quantized weights for testing.
///
/// Q8_0 format: 34 bytes per block
/// - 2 bytes: f16 scale
/// - 32 bytes: 32 x 8-bit quantized values
///
/// # Arguments
/// * `num_blocks` - Number of Q8_0 blocks to generate
///
/// # Returns
/// Vec<u8> of quantized weight data
pub fn generate_q8_0_weights(num_blocks: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(num_blocks * 34);

    for block_idx in 0..num_blocks {
        // Scale: predictable value
        let scale = 0.01 * (block_idx as f32 + 1.0);
        let scale_f16 = f16::from_f32(scale);
        data.extend_from_slice(&scale_f16.to_le_bytes());

        // Quants: 32 x i8 values (stored as u8)
        for j in 0..32 {
            // Range: -128 to 127, mapped to pattern
            let val = (((block_idx + j) % 256) as i8) as u8;
            data.push(val);
        }
    }

    data
}

// ============================================================================
// GQA Test Configurations
// ============================================================================

/// Standard GQA configurations for testing
pub struct GqaConfig {
    pub hidden_dim: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub intermediate_dim: usize,
    pub name: &'static str,
}

impl GqaConfig {
    /// Qwen 0.5B configuration (14 Q heads, 2 KV heads)
    pub const QWEN_0_5B: Self = Self {
        hidden_dim: 896,
        num_heads: 14,
        num_kv_heads: 2,
        head_dim: 64,
        intermediate_dim: 4864,
        name: "Qwen-0.5B",
    };

    /// TinyLlama configuration (32 Q heads, 4 KV heads)
    pub const TINY_LLAMA: Self = Self {
        hidden_dim: 2048,
        num_heads: 32,
        num_kv_heads: 4,
        head_dim: 64,
        intermediate_dim: 5632,
        name: "TinyLlama",
    };

    /// Qwen 1.5B configuration (12 Q heads, 2 KV heads)
    pub const QWEN_1_5B: Self = Self {
        hidden_dim: 1536,
        num_heads: 12,
        num_kv_heads: 2,
        head_dim: 128,
        intermediate_dim: 8960,
        name: "Qwen-1.5B",
    };

    /// Llama 7B configuration (32 Q heads, 32 KV heads - MHA, not GQA)
    pub const LLAMA_7B_MHA: Self = Self {
        hidden_dim: 4096,
        num_heads: 32,
        num_kv_heads: 32,
        head_dim: 128,
        intermediate_dim: 11008,
        name: "Llama-7B-MHA",
    };

    /// Q dimension (full attention heads)
    #[must_use]
    pub const fn q_dim(&self) -> usize {
        self.num_heads * self.head_dim
    }

    /// KV dimension (grouped query attention)
    #[must_use]
    pub const fn kv_dim(&self) -> usize {
        self.num_kv_heads * self.head_dim
    }

    /// GQA group size (how many Q heads share each KV head)
    #[must_use]
    pub const fn gqa_group_size(&self) -> usize {
        self.num_heads / self.num_kv_heads
    }

    /// Whether this is true GQA (kv_heads < q_heads)
    #[must_use]
    pub const fn is_gqa(&self) -> bool {
        self.num_kv_heads < self.num_heads
    }
}

// ============================================================================
// Parity Test Helpers
// ============================================================================

/// Compute relative difference between two values
#[must_use]
pub fn relative_diff(a: f32, b: f32) -> f32 {
    let diff = (a - b).abs();
    let denom = a.abs().max(b.abs()).max(1e-6);
    diff / denom
}

/// Compute max element-wise difference between two vectors
#[must_use]
pub fn max_element_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vector lengths must match");
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// Compute sum relative difference between two vectors
#[must_use]
pub fn sum_relative_diff(a: &[f32], b: &[f32]) -> f32 {
    let sum_a: f32 = a.iter().sum();
    let sum_b: f32 = b.iter().sum();
    relative_diff(sum_a, sum_b)
}

/// Check if vectors are within tolerance (element-wise max diff)
#[must_use]
pub fn vectors_match(a: &[f32], b: &[f32], tolerance: f32) -> bool {
    max_element_diff(a, b) <= tolerance
}

// ============================================================================
// ModelHarness: Setup complete executor state for integration testing
// ============================================================================

/// Configuration for ModelHarness
pub struct HarnessConfig {
    pub hidden_dim: usize,
    pub intermediate_dim: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub max_seq_len: usize,
}

impl Default for HarnessConfig {
    fn default() -> Self {
        // Minimal config for fast tests
        Self {
            hidden_dim: 256,
            intermediate_dim: 512,
            num_layers: 2,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 64,
            vocab_size: 1024,
            max_seq_len: 128,
        }
    }
}

impl HarnessConfig {
    /// Tiny config for fastest tests
    pub fn tiny() -> Self {
        Self {
            hidden_dim: 64,
            intermediate_dim: 128,
            num_layers: 1,
            num_heads: 2,
            num_kv_heads: 1,
            head_dim: 32,
            vocab_size: 256,
            max_seq_len: 32,
        }
    }

    /// Config matching Qwen 0.5B dimensions (scaled down)
    pub fn qwen_like() -> Self {
        Self {
            hidden_dim: 256,       // 896 scaled down
            intermediate_dim: 512, // 4864 scaled down
            num_layers: 2,
            num_heads: 8,
            num_kv_heads: 2,
            head_dim: 32,
            vocab_size: 512,
            max_seq_len: 64,
        }
    }
}

/// Compute Q4_K weight size: rows * cols / 256 elements per superblock * 144 bytes per superblock
#[inline]
fn q4k_weight_size(rows: usize, cols: usize) -> usize {
    rows * cols / 256 * 144
}

/// Load zero-initialized Q4_K weights into executor cache
fn load_zero_weights(
    exec: &mut crate::cuda::executor::CudaExecutor,
    name: &str,
    rows: usize,
    cols: usize,
) -> Result<(), crate::cuda::executor::GpuError> {
    let weights = vec![0u8; q4k_weight_size(rows, cols)];
    exec.load_quantized_weights(name, &weights)?;
    Ok(())
}

/// Load attention weights (Q/K/V/O projections) for one layer
fn load_layer_attn_weights(
    exec: &mut crate::cuda::executor::CudaExecutor,
    prefix: &str,
    config: &HarnessConfig,
) -> Result<(), crate::cuda::executor::GpuError> {
    let q_dim = config.num_heads * config.head_dim;
    let kv_dim = config.num_kv_heads * config.head_dim;

    load_zero_weights(
        exec,
        &format!("{prefix}.attn_q.weight"),
        q_dim,
        config.hidden_dim,
    )?;
    load_zero_weights(
        exec,
        &format!("{prefix}.attn_k.weight"),
        kv_dim,
        config.hidden_dim,
    )?;
    load_zero_weights(
        exec,
        &format!("{prefix}.attn_v.weight"),
        kv_dim,
        config.hidden_dim,
    )?;
    load_zero_weights(
        exec,
        &format!("{prefix}.attn_output.weight"),
        config.hidden_dim,
        q_dim,
    )
}

/// Load FFN weights (gate/up/down projections) for one layer
fn load_layer_ffn_weights(
    exec: &mut crate::cuda::executor::CudaExecutor,
    prefix: &str,
    config: &HarnessConfig,
) -> Result<(), crate::cuda::executor::GpuError> {
    load_zero_weights(
        exec,
        &format!("{prefix}.ffn_gate.weight"),
        config.intermediate_dim,
        config.hidden_dim,
    )?;
    load_zero_weights(
        exec,
        &format!("{prefix}.ffn_up.weight"),
        config.intermediate_dim,
        config.hidden_dim,
    )?;
    load_zero_weights(
        exec,
        &format!("{prefix}.ffn_down.weight"),
        config.hidden_dim,
        config.intermediate_dim,
    )
}

include!("executor_harness.rs");
