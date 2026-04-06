//! Fused Operations for Transformer Inference
//!
//! This module contains fused compute operations that combine multiple
//! operations into single passes for improved performance.
//!
//! # Operations
//!
//! - `FusedQKVOp`: Fused Query/Key/Value projection (3 matmuls → 1)
//! - `FusedGateUpOp`: Fused Gate+Up FFN projection with SiLU (SwiGLU)
//!
//! # Performance Impact
//!
//! Fusing operations provides:
//! - Reduced kernel launches (GPU)
//! - Better cache utilization (data loaded once)
//! - Eliminated intermediate memory traffic

use super::{Backend, ComputeOp};
use crate::error::TruenoError;

// ============================================================================
// Fused Q/K/V Projection (PMAT-PERF-009)
// ============================================================================

/// Weights for fused QKV projection
#[derive(Debug, Clone)]
pub struct FusedQKVWeights {
    /// Q projection weights [hidden_size, hidden_size]
    pub q_weight: Vec<f32>,
    /// K projection weights [hidden_size, kv_dim]
    pub k_weight: Vec<f32>,
    /// V projection weights [hidden_size, kv_dim]
    pub v_weight: Vec<f32>,
}

/// Fused Q/K/V projection operation for transformer attention.
///
/// Computes Q, K, V projections in a single pass over the input:
/// - Q = x * W_q (hidden_size → hidden_size)
/// - K = x * W_k (hidden_size → kv_dim)
/// - V = x * W_v (hidden_size → kv_dim)
///
/// # Performance Impact
///
/// Fusing 3 separate matmuls into 1 operation provides:
/// - 3x reduction in kernel launches (GPU)
/// - Better cache utilization (input x loaded once)
/// - Expected speedup: 2-3x for decode phase
///
/// # Five-Whys Root Cause (PMAT-PERF-009)
///
/// ```text
/// Why 1: Why is decode throughput 131 tok/s vs 400 tok/s target?
/// → 280+ kernel launches per token (10+ per layer × 28 layers)
///
/// Why 2: Why so many kernel launches?
/// → Q, K, V computed as 3 separate GEMV operations
///
/// Why 3: Why separate operations?
/// → Original implementation didn't consider launch overhead
///
/// Why 4: Why does launch overhead matter?
/// → GPU kernel launch: ~5-10µs, 280 launches = 1.4-2.8ms overhead/token
///
/// Why 5: ROOT CAUSE
/// → Kernel launch overhead (2.8ms) exceeds compute time for small batch decode
/// → FIX: Fuse Q/K/V into single kernel, reducing launches by 2/3
/// ```
#[derive(Debug, Clone)]
pub struct FusedQKVOp {
    /// Hidden dimension size
    pub hidden_size: usize,
    /// KV dimension (num_kv_heads * head_dim, may differ from hidden_size for GQA)
    pub kv_dim: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Head dimension
    pub head_dim: usize,
}

impl FusedQKVOp {
    /// Create a new fused QKV operation.
    ///
    /// # Arguments
    /// * `hidden_size` - Hidden dimension (e.g., 3584 for Qwen 3B)
    /// * `num_heads` - Number of attention heads
    /// * `num_kv_heads` - Number of KV heads (may differ for GQA)
    pub fn new(hidden_size: usize, num_heads: usize, num_kv_heads: usize) -> Self {
        let head_dim = hidden_size / num_heads;
        let kv_dim = num_kv_heads * head_dim;
        Self { hidden_size, kv_dim, num_heads, head_dim }
    }
}

#[allow(clippy::needless_range_loop)] // Matrix indexing is clearer with explicit loops
impl ComputeOp for FusedQKVOp {
    type Input = (Vec<f32>, FusedQKVWeights);
    type Output = (Vec<f32>, Vec<f32>, Vec<f32>); // (Q, K, V)

    fn name(&self) -> &'static str {
        "fused_qkv"
    }

    fn execute(&self, input: Self::Input, _backend: Backend) -> Result<Self::Output, TruenoError> {
        let (x, weights) = input;

        // Validate input dimensions
        if x.len() != self.hidden_size {
            return Err(TruenoError::SizeMismatch { expected: self.hidden_size, actual: x.len() });
        }

        let h = self.hidden_size;

        // Q projection: x @ W_q^T -> [hidden_size]
        // CGP-DBUF: SIMD dot product per row (was scalar nested loop).
        let mut q: Vec<f32> = Vec::with_capacity(h);
        // SAFETY: Each q[i] is SET to simd_dot result before any read.
        unsafe {
            q.set_len(h);
        }
        for i in 0..h {
            q[i] =
                super::attention::AttentionOp::simd_dot(&x, &weights.q_weight[i * h..(i + 1) * h]);
        }

        // K projection: x @ W_k^T -> [kv_dim]
        let mut k: Vec<f32> = Vec::with_capacity(self.kv_dim);
        // SAFETY: Each k[i] is SET to simd_dot result before any read.
        unsafe {
            k.set_len(self.kv_dim);
        }
        for i in 0..self.kv_dim {
            k[i] =
                super::attention::AttentionOp::simd_dot(&x, &weights.k_weight[i * h..(i + 1) * h]);
        }

        // V projection: x @ W_v^T -> [kv_dim]
        let mut v: Vec<f32> = Vec::with_capacity(self.kv_dim);
        // SAFETY: Each v[i] is SET to simd_dot result before any read.
        unsafe {
            v.set_len(self.kv_dim);
        }
        for i in 0..self.kv_dim {
            v[i] =
                super::attention::AttentionOp::simd_dot(&x, &weights.v_weight[i * h..(i + 1) * h]);
        }

        Ok((q, k, v))
    }

    fn tokens(&self, _input: &Self::Input) -> usize {
        // Output tokens = Q + K + V dimensions
        self.hidden_size + 2 * self.kv_dim
    }
}

// ============================================================================
// Fused Gate+Up FFN Projection (PMAT-PERF-009)
// ============================================================================

/// Weights for fused gate+up FFN projection
#[derive(Debug, Clone)]
pub struct FusedGateUpWeights {
    /// Gate projection weights [hidden_size, intermediate_size]
    pub gate_weight: Vec<f32>,
    /// Up projection weights [hidden_size, intermediate_size]
    pub up_weight: Vec<f32>,
}

/// Fused Gate+Up FFN projection with SiLU activation.
///
/// Computes gate and up projections in a single pass:
/// - gate = x * W_gate
/// - up = x * W_up
/// - output = SiLU(gate) * up (SwiGLU activation)
///
/// # Performance Impact
///
/// Fusing 2 separate matmuls + activation provides:
/// - 2x reduction in kernel launches (GPU)
/// - Fused SiLU avoids intermediate memory traffic
/// - Expected speedup: 1.5-2x for decode phase
///
/// # Five-Whys Root Cause (PMAT-PERF-009)
///
/// ```text
/// Why 1: Why is FFN phase slow?
/// → 3 kernel launches: gate_proj, up_proj, SiLU activation
///
/// Why 2: Why separate kernels?
/// → Traditional implementation pattern from training frameworks
///
/// Why 3: Why does this matter for inference?
/// → Inference is memory-bound; kernel launch overhead dominates
///
/// Why 4: Why not fuse earlier?
/// → Requires custom kernel development
///
/// Why 5: ROOT CAUSE
/// → SwiGLU requires gate*up pattern that naturally fuses
/// → FIX: Fuse gate+up+SiLU into single operation
/// ```
#[derive(Debug, Clone)]
pub struct FusedGateUpOp {
    /// Hidden dimension size
    pub hidden_size: usize,
    /// Intermediate FFN dimension
    pub intermediate_size: usize,
}

impl FusedGateUpOp {
    /// Create a new fused gate+up operation.
    ///
    /// # Arguments
    /// * `hidden_size` - Hidden dimension (e.g., 3584 for Qwen 3B)
    /// * `intermediate_size` - FFN intermediate dimension (e.g., 18944)
    pub fn new(hidden_size: usize, intermediate_size: usize) -> Self {
        Self { hidden_size, intermediate_size }
    }

    /// SiLU activation: x * sigmoid(x)
    ///
    /// ONE PATH: Delegates to `crate::activations::silu_scalar` (UCBD §4).
    #[inline]
    pub fn silu(x: f32) -> f32 {
        contract_pre_silu!();
        let result = crate::activations::silu_scalar(x);
        contract_post_silu!(&[result]);
        result
    }
}

impl ComputeOp for FusedGateUpOp {
    type Input = (Vec<f32>, FusedGateUpWeights);
    type Output = Vec<f32>; // SwiGLU output [intermediate_size]

    fn name(&self) -> &'static str {
        "fused_gate_up"
    }

    fn execute(&self, input: Self::Input, _backend: Backend) -> Result<Self::Output, TruenoError> {
        let (x, weights) = input;

        // Validate input dimensions
        if x.len() != self.hidden_size {
            return Err(TruenoError::SizeMismatch { expected: self.hidden_size, actual: x.len() });
        }

        // SIMD-optimized fused gate + up + SwiGLU
        // CGP-DBUF: Use slice-based SIMD dot directly — eliminates 2×intermediate_size
        // Vector allocations per call (was ~38K allocs for Qwen 3B).
        // Uninit: output[i] = silu(gate) * up (SET) for every i.
        let mut output: Vec<f32> = Vec::with_capacity(self.intermediate_size);
        // SAFETY: Loop writes output[i] = silu(gate_sum) * up_sum for all i.
        unsafe {
            output.set_len(self.intermediate_size);
        }

        let h = self.hidden_size;
        for i in 0..self.intermediate_size {
            let row_start = i * h;
            let row_end = row_start + h;

            // Direct SIMD dot on slices — no Vector allocation
            let gate_sum = super::attention::AttentionOp::simd_dot(
                &x,
                &weights.gate_weight[row_start..row_end],
            );
            let up_sum =
                super::attention::AttentionOp::simd_dot(&x, &weights.up_weight[row_start..row_end]);

            // SwiGLU: SiLU(gate) * up
            output[i] = Self::silu(gate_sum) * up_sum;
        }

        Ok(output)
    }

    fn tokens(&self, _input: &Self::Input) -> usize {
        self.intermediate_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fused_qkv_basic() {
        // hidden=4, num_heads=2, kv_heads=1 → head_dim=2, kv_dim=2
        let op = FusedQKVOp::new(4, 2, 1);

        let x = vec![1.0, 2.0, 3.0, 4.0];
        let weights = FusedQKVWeights {
            q_weight: vec![1.0; 16], // hidden_size x hidden_size = 4x4 = 16
            k_weight: vec![1.0; 8],  // kv_dim x hidden_size = 2x4 = 8
            v_weight: vec![1.0; 8],  // kv_dim x hidden_size = 2x4 = 8
        };

        let (q, k, v) = op.execute((x, weights), Backend::Scalar).unwrap();

        assert_eq!(q.len(), 4);
        assert_eq!(k.len(), 2);
        assert_eq!(v.len(), 2);
    }

    #[test]
    fn test_fused_qkv_dimension_mismatch() {
        let op = FusedQKVOp::new(4, 2, 2);
        let x = vec![1.0, 2.0]; // Wrong size
        let weights = FusedQKVWeights {
            q_weight: vec![1.0; 16],
            k_weight: vec![1.0; 8],
            v_weight: vec![1.0; 8],
        };

        let result = op.execute((x, weights), Backend::Scalar);
        assert!(result.is_err());
    }

    #[test]
    fn test_fused_gate_up_basic() {
        let op = FusedGateUpOp::new(4, 2);

        let x = vec![1.0, 2.0, 3.0, 4.0];
        let weights = FusedGateUpWeights {
            gate_weight: vec![1.0; 8], // 2x4
            up_weight: vec![1.0; 8],   // 2x4
        };

        let output = op.execute((x, weights), Backend::Scalar).unwrap();
        assert_eq!(output.len(), 2);

        // Output should be SiLU(gate_sum) * up_sum
        // gate_sum = up_sum = 1+2+3+4 = 10
        // SiLU(10) ≈ 10 * sigmoid(10) ≈ 10 * 0.99995 ≈ 10
        // output ≈ 10 * 10 = 100
        assert!(output[0] > 90.0 && output[0] < 110.0);
    }

    #[test]
    fn test_fused_gate_up_dimension_mismatch() {
        let op = FusedGateUpOp::new(4, 2);
        let x = vec![1.0, 2.0]; // Wrong size
        let weights = FusedGateUpWeights { gate_weight: vec![1.0; 8], up_weight: vec![1.0; 8] };

        let result = op.execute((x, weights), Backend::Scalar);
        assert!(result.is_err());
    }

    #[test]
    fn test_silu_values() {
        // SiLU(0) = 0
        assert!((FusedGateUpOp::silu(0.0) - 0.0).abs() < 1e-6);

        // SiLU(x) → x for large positive x
        assert!((FusedGateUpOp::silu(10.0) - 10.0).abs() < 0.01);

        // SiLU(x) → 0 for large negative x
        assert!(FusedGateUpOp::silu(-10.0).abs() < 0.01);
    }

    #[test]
    fn test_fused_qkv_tokens() {
        // hidden=128, heads=8, kv_heads=4 → head_dim=16, kv_dim=64
        let op = FusedQKVOp::new(128, 8, 4);
        let weights = FusedQKVWeights { q_weight: vec![], k_weight: vec![], v_weight: vec![] };
        // tokens = hidden + 2 * kv_dim = 128 + 2 * 64 = 256
        assert_eq!(op.tokens(&(vec![], weights)), 256);
    }

    #[test]
    fn test_fused_gate_up_tokens() {
        let op = FusedGateUpOp::new(128, 256);
        let weights = FusedGateUpWeights { gate_weight: vec![], up_weight: vec![] };
        assert_eq!(op.tokens(&(vec![], weights)), 256);
    }
}
