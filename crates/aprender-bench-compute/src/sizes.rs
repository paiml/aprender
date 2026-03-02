//! Canonical LLM sizes for compute benchmarks.
//!
//! Each size represents a real model configuration to ensure benchmarks
//! measure performance at production-relevant dimensions.

/// Matrix multiplication sizes: (M, K, N) for C = A[M×K] × B[K×N]
///
/// Single-token inference: M=1, K=hidden_dim, N=output_dim.
/// For matmul benchmarks we also include square sizes.
pub const MATMUL_SIZES: &[(usize, usize, usize)] = &[
    (1, 1536, 1536),    // 1.5B single-token QKV projection
    (1, 4096, 4096),    // 7B single-token QKV projection
    (1, 4096, 11008),   // 7B single-token FFN up_proj
    (512, 512, 512),    // Small square (cache/SIMD baseline)
    (1024, 1024, 1024), // Medium square
];

/// Single-token matvec sizes: (hidden_dim, output_dim)
///
/// The hot path in autoregressive generation: one input vector × weight matrix.
pub const MATVEC_SIZES: &[(usize, usize)] = &[
    (1536, 1536),   // 1.5B hidden→hidden
    (1536, 8960),   // 1.5B hidden→FFN
    (4096, 4096),   // 7B hidden→hidden
    (4096, 11008),  // 7B hidden→FFN
];

/// Activation sizes (number of elements).
///
/// Matches FFN intermediate dimensions from real models.
pub const ACTIVATION_SIZES: &[usize] = &[1536, 4096, 8960, 11008, 16384];

/// Norm sizes (hidden dimensions).
pub const NORM_SIZES: &[usize] = &[1536, 4096, 5120, 8192];

/// Quantization sizes (number of f32 elements).
///
/// Must be multiples of 32 (Q4_0/Q8_0 block size).
pub const QUANT_SIZES: &[usize] = &[1024, 4096, 16384, 65536, 262144];

/// Softmax sizes (vocabulary dimension — the logits vector).
pub const SOFTMAX_SIZES: &[usize] = &[32000, 32768, 128256, 151936];

/// Scaling analysis sizes for O(n) verification.
pub const SCALING_SIZES: &[usize] = &[256, 1024, 4096, 16384, 65536, 262144];

/// Transpose sizes: (rows, cols) representing attention-relevant shapes.
///
/// Transposing K^T in attention: (seq_len, head_dim) or (head_dim, seq_len).
/// At LLM scale these are cache-unfriendly for naive implementations.
pub const TRANSPOSE_SIZES: &[(usize, usize)] = &[
    (128, 128),   // head_dim × head_dim (7B: 128)
    (512, 128),   // seq_len × head_dim (typical prefill)
    (2048, 128),  // long context × head_dim
    (4096, 4096), // square (weight matrix)
];

/// Element-wise op sizes (hidden dimensions for add, mul_scalar).
pub const ELEMENTWISE_SIZES: &[usize] = &[1536, 4096, 8192, 16384];

/// RoPE sizes: (seq_len, num_heads, head_dim)
///
/// Rotary position embedding applied per-head per-token.
/// 7B: 32 heads × 128 head_dim, 1.5B: 12 heads × 128 head_dim.
pub const ROPE_SIZES: &[(usize, usize, usize)] = &[
    (1, 12, 128),    // 1.5B single-token
    (1, 32, 128),    // 7B single-token
    (128, 32, 128),  // 7B prefill 128 tokens
    (512, 32, 128),  // 7B prefill 512 tokens
];
