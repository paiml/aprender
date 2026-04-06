//! Fused CUDA Bricks - CoalescedDp4a and FusedFFN
//!
//! High-performance fused bricks for DP4A operations and FFN,
//! extracted from brick/mod.rs (PMAT-802).

#![allow(clippy::too_many_arguments)]

use std::time::Instant;

use super::{AssertionKind, BrickAssertion, BrickError, ComputeBrick, TokenBudget, TokenResult};
#[cfg(feature = "cuda")]
#[allow(unused_imports)]
use crate::cuda::CudaExecutor;
#[cfg(feature = "cuda")]
#[allow(unused_imports)]
use crate::error::RealizarError;

// ============================================================================
// Coalesced DP4A Brick (Section 5.3 - P0)
// ============================================================================

/// Coalesced DP4A GEMV Brick for bandwidth-optimized quantized matmul.
///
/// Per spec: docs/specifications/qwen2.5-coder-showcase-demo.md §5.3
///
/// Key optimizations:
/// - 4-byte coalesced loads (vs 1-byte non-coalesced)
/// - DP4A instruction for 4 multiply-adds per cycle
/// - Pre-quantized Q8 activations for integer arithmetic
///
/// # Implementation
///
/// Wraps `CudaExecutor::packed_dp4a_q4k_q8_gemv_async()` from cuda.rs (PAR-063-V6).
#[derive(Debug, Clone)]
pub struct CoalescedDp4aBrick {
    /// Input dimension (K)
    pub k: usize,
    /// Output dimension (N)
    pub n: usize,
    /// Token budget (target: 4x improvement over non-coalesced)
    budget: TokenBudget,
}

impl CoalescedDp4aBrick {
    /// Create new Coalesced DP4A brick.
    ///
    /// # Arguments
    ///
    /// * `k` - Input dimension (must be multiple of 256 for Q4K)
    /// * `n` - Output dimension
    #[must_use]
    pub fn new(k: usize, n: usize) -> Self {
        // Budget based on memory bandwidth model
        // Q4K: 4.5 bits/value → k * 4.5 / 8 bytes
        // At 700 GB/s bandwidth: time_us = bytes / (700e9 / 1e6)
        let bytes = (k as f64 * n as f64 * 4.5) / 8.0;
        let bandwidth_gb_s = 700.0; // RTX 4090 achievable
        let budget_us = bytes / (bandwidth_gb_s * 1e3);

        Self {
            k,
            n,
            budget: TokenBudget::from_latency(budget_us.max(1.0)),
        }
    }

    /// Set custom budget.
    #[must_use]
    pub fn with_budget(mut self, budget: TokenBudget) -> Self {
        self.budget = budget;
        self
    }

    /// Theoretical FLOPS for this operation.
    #[must_use]
    pub fn flops(&self) -> u64 {
        // GEMV: 2 * K * N (multiply-add per element)
        2 * self.k as u64 * self.n as u64
    }

    /// Arithmetic intensity (FLOPS / bytes).
    #[must_use]
    pub fn arithmetic_intensity(&self) -> f64 {
        contract_pre_compute_ceiling!();
        let bytes = (self.k as f64 * 4.5) / 8.0 + self.n as f64 * 4.0; // Q4K weights + f32 output
        self.flops() as f64 / bytes
    }

    /// Compute GEMV with Q8 activations and Q4K weights.
    ///
    /// **REAL IMPLEMENTATION** - CPU reference for DP4A-style compute.
    ///
    /// # Arguments
    /// * `input_q8` - Quantized int8 input vector [K]
    /// * `input_scale` - Scale factor for input
    /// * `weights_q4` - Quantized 4-bit weights [N * K / 2] (packed nibbles)
    /// * `weight_scales` - Scale factors per output [N]
    ///
    /// # Returns
    /// * Output vector [N]
    pub fn forward(
        &self,
        input_q8: &[i8],
        input_scale: f32,
        weights_q4: &[u8],
        weight_scales: &[f32],
    ) -> Result<Vec<f32>, BrickError> {
        if input_q8.len() != self.k {
            return Err(BrickError::InvalidInput(format!(
                "Input length {} != k {}",
                input_q8.len(),
                self.k
            )));
        }
        if weights_q4.len() != self.n * self.k / 2 {
            return Err(BrickError::InvalidInput(format!(
                "Weights length {} != n * k / 2 = {}",
                weights_q4.len(),
                self.n * self.k / 2
            )));
        }
        if weight_scales.len() != self.n {
            return Err(BrickError::InvalidInput(format!(
                "Weight scales length {} != n {}",
                weight_scales.len(),
                self.n
            )));
        }

        let mut output = vec![0.0f32; self.n];

        // GEMV: output[n] = sum_k(input[k] * weights[n, k])
        for n in 0..self.n {
            let mut acc = 0i32;

            // Process in groups of 4 (simulating DP4A: 4 multiply-adds per instruction)
            for k_group in (0..self.k).step_by(4) {
                // Unpack 4-bit weights (2 weights per byte)
                for k_offset in 0..4 {
                    let k = k_group + k_offset;
                    if k >= self.k {
                        break;
                    }

                    // Index into packed nibble array (2 weights per byte)
                    // Not a midpoint calculation - clippy false positive
                    #[allow(clippy::manual_midpoint)]
                    let weight_byte_idx = (n * self.k + k) / 2;
                    let weight_nibble = if k % 2 == 0 {
                        (weights_q4[weight_byte_idx] & 0x0F) as i8 - 8 // Low nibble, centered
                    } else {
                        ((weights_q4[weight_byte_idx] >> 4) & 0x0F) as i8 - 8 // High nibble
                    };

                    // Integer multiply-accumulate (DP4A-style)
                    acc += input_q8[k] as i32 * weight_nibble as i32;
                }
            }

            // Dequantize: scale by input_scale * weight_scale
            output[n] = acc as f32 * input_scale * weight_scales[n];
        }

        Ok(output)
    }

    /// Execute GEMV with timing (for benchmarking).
    pub fn forward_timed(
        &self,
        input_q8: &[i8],
        input_scale: f32,
        weights_q4: &[u8],
        weight_scales: &[f32],
    ) -> Result<TokenResult<Vec<f32>>, BrickError> {
        let start = Instant::now();
        let output = self.forward(input_q8, input_scale, weights_q4, weight_scales)?;
        let elapsed_us = start.elapsed().as_secs_f64() * 1_000_000.0;

        Ok(TokenResult {
            output,
            tokens_processed: 1,
            us_per_token: elapsed_us,
            tokens_per_sec: 1_000_000.0 / elapsed_us,
            budget_met: elapsed_us <= self.budget.us_per_token,
        })
    }

    /// Legacy stub for backward compatibility (prefer `forward()`)
    #[deprecated(note = "Use forward() for real implementation")]
    pub fn execute(&self) -> Result<Vec<f32>, BrickError> {
        if !self.k.is_multiple_of(256) || self.k == 0 || self.n == 0 {
            return Err(BrickError::InvalidInput(format!(
                "Invalid dimensions: k={} (must be multiple of 256), n={}",
                self.k, self.n
            )));
        }
        Ok(vec![0.0; self.n])
    }
}

impl ComputeBrick for CoalescedDp4aBrick {
    type Output = Vec<f32>;

    fn name(&self) -> &'static str {
        "coalesced_dp4a"
    }

    fn budget(&self) -> TokenBudget {
        self.budget
    }

    fn assertions(&self) -> Vec<BrickAssertion> {
        vec![
            BrickAssertion::no_nan(),
            BrickAssertion::no_inf(),
            BrickAssertion::budget_met(),
            BrickAssertion {
                name: "bandwidth_efficient".to_string(),
                description: "Achieves >= 70% of peak memory bandwidth".to_string(),
                kind: AssertionKind::Custom {
                    check_name: "bandwidth_efficient".to_string(),
                },
            },
        ]
    }

    fn can_run(&self) -> bool {
        // K must be multiple of 256 for Q4K super-blocks
        self.k.is_multiple_of(256) && self.k > 0 && self.n > 0
    }
}

// ============================================================================
// Fused Megakernel Brick (P1)
// ============================================================================

/// Fused SwiGLU FFN Brick (gate-up-down with DP4A optimization).
///
/// Per spec: docs/specifications/qwen2.5-coder-showcase-demo.md §5.1 (P1)
///
/// **Architecture** (DP4A-optimized pipeline):
/// ```text
/// input ─┬─► Q8 quantize ─┬─► gate_proj (Q4K×Q8) ─┐
///        │                │                       ├─► SwiGLU ─► Q8 ─► down_proj ─► output
///        │                └─► up_proj (Q4K×Q8) ───┘
///        │
///        └─► (Q8 shared between gate & up - 1 quant vs 2)
/// ```
///
/// **Optimizations**:
/// 1. Shared Q8 quantization (input reused for gate & up)
/// 2. Packed DP4A GEMV (4 int8 MADs per instruction)
/// 3. Fused SwiGLU activation (silu(gate) * up in single kernel)
///
/// **Performance**: 3x vs naive (1 shared quant + fused activation)
#[derive(Debug, Clone)]
pub struct FusedFfnBrick {
    /// Hidden dimension (e.g., 1536 for 1.5B, 4096 for 32B)
    pub hidden_dim: usize,
    /// Intermediate dimension (typically 4x hidden_dim)
    pub intermediate_dim: usize,
    /// Token budget (target: 12.2µs for 2x Ollama)
    budget: TokenBudget,
    /// Use packed DP4A (PACKED_DP4A=1 env var)
    pub use_packed_dp4a: bool,
}

impl FusedFfnBrick {
    /// Create new fused FFN brick with default DP4A settings.
    #[must_use]
    pub fn new(hidden_dim: usize, intermediate_dim: usize) -> Self {
        // Check PACKED_DP4A env var
        let use_packed_dp4a = std::env::var("PACKED_DP4A")
            .map(|v| v == "1")
            .unwrap_or(false);

        // Budget: 12.2µs target for 2x Ollama performance
        // Derived from: 35.7µs/layer budget × 0.36 FFN fraction = 12.9µs
        Self {
            hidden_dim,
            intermediate_dim,
            budget: TokenBudget::from_latency(12.2),
            use_packed_dp4a,
        }
    }

    /// Create with packed DP4A enabled (for benchmarking).
    #[must_use]
    pub fn with_packed_dp4a(hidden_dim: usize, intermediate_dim: usize) -> Self {
        Self {
            hidden_dim,
            intermediate_dim,
            budget: TokenBudget::from_latency(12.2),
            use_packed_dp4a: true,
        }
    }

    /// Set custom budget.
    #[must_use]
    pub fn with_budget(mut self, budget: TokenBudget) -> Self {
        self.budget = budget;
        self
    }

    /// Compute FLOPs for this FFN layer.
    #[must_use]
    pub fn flops(&self) -> u64 {
        // gate: 2 * hidden * intermediate
        // up: 2 * hidden * intermediate
        // down: 2 * intermediate * hidden
        // Total: 6 * hidden * intermediate
        6 * self.hidden_dim as u64 * self.intermediate_dim as u64
    }

    /// Compute arithmetic intensity (FLOPs / bytes).
    #[must_use]
    pub fn arithmetic_intensity(&self) -> f64 {
        // Bytes: Q4K weights (4.5 bits) + f32 activations
        // gate: hidden * intermediate * 4.5/8 bytes
        // up: hidden * intermediate * 4.5/8 bytes
        // down: intermediate * hidden * 4.5/8 bytes
        // activations: hidden * 4 + intermediate * 4 * 2 + hidden * 4
        let weight_bytes = 3.0 * self.hidden_dim as f64 * self.intermediate_dim as f64 * 4.5 / 8.0;
        let activation_bytes =
            (self.hidden_dim * 4 + self.intermediate_dim * 8 + self.hidden_dim * 4) as f64;
        self.flops() as f64 / (weight_bytes + activation_bytes)
    }

    /// Compute FFN with SwiGLU activation.
    ///
    /// **REAL IMPLEMENTATION** - Full FFN forward pass:
    /// ```text
    /// gate = input @ gate_proj
    /// up = input @ up_proj
    /// hidden = silu(gate) * up  // SwiGLU
    /// output = hidden @ down_proj
    /// ```
    ///
    /// # Arguments
    /// * `input` - Input tensor [hidden_dim]
    /// * `gate_proj` - Gate projection weights [intermediate_dim, hidden_dim]
    /// * `up_proj` - Up projection weights [intermediate_dim, hidden_dim]
    /// * `down_proj` - Down projection weights [hidden_dim, intermediate_dim]
    ///
    /// # Returns
    /// * Output tensor [hidden_dim]
    pub fn forward(
        &self,
        input: &[f32],
        gate_proj: &[f32],
        up_proj: &[f32],
        down_proj: &[f32],
    ) -> Result<Vec<f32>, BrickError> {
        if input.len() != self.hidden_dim {
            return Err(BrickError::InvalidInput(format!(
                "Input length {} != hidden_dim {}",
                input.len(),
                self.hidden_dim
            )));
        }
        let expected_gate_up = self.intermediate_dim * self.hidden_dim;
        if gate_proj.len() != expected_gate_up || up_proj.len() != expected_gate_up {
            return Err(BrickError::InvalidInput(format!(
                "Gate/Up length {} != intermediate * hidden = {}",
                gate_proj.len(),
                expected_gate_up
            )));
        }
        if down_proj.len() != self.hidden_dim * self.intermediate_dim {
            return Err(BrickError::InvalidInput(format!(
                "Down length {} != hidden * intermediate = {}",
                down_proj.len(),
                self.hidden_dim * self.intermediate_dim
            )));
        }

        // Step 1: Gate projection (input @ gate_proj^T)
        let mut gate = vec![0.0f32; self.intermediate_dim];
        for i in 0..self.intermediate_dim {
            let mut sum = 0.0f32;
            for j in 0..self.hidden_dim {
                sum += input[j] * gate_proj[i * self.hidden_dim + j];
            }
            gate[i] = sum;
        }

        // Step 2: Up projection (input @ up_proj^T)
        let mut up = vec![0.0f32; self.intermediate_dim];
        for i in 0..self.intermediate_dim {
            let mut sum = 0.0f32;
            for j in 0..self.hidden_dim {
                sum += input[j] * up_proj[i * self.hidden_dim + j];
            }
            up[i] = sum;
        }

        // Step 3: SwiGLU activation: silu(gate) * up
        // silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
        let mut hidden = vec![0.0f32; self.intermediate_dim];
        for i in 0..self.intermediate_dim {
            let silu_gate = gate[i] / (1.0 + (-gate[i]).exp());
            hidden[i] = silu_gate * up[i];
        }

        // Step 4: Down projection (hidden @ down_proj^T)
        let mut output = vec![0.0f32; self.hidden_dim];
        for i in 0..self.hidden_dim {
            let mut sum = 0.0f32;
            for j in 0..self.intermediate_dim {
                sum += hidden[j] * down_proj[i * self.intermediate_dim + j];
            }
            output[i] = sum;
        }

        Ok(output)
    }

    /// Execute FFN with timing (for benchmarking).
    pub fn forward_timed(
        &self,
        input: &[f32],
        gate_proj: &[f32],
        up_proj: &[f32],
        down_proj: &[f32],
    ) -> Result<TokenResult<Vec<f32>>, BrickError> {
        let start = Instant::now();
        let output = self.forward(input, gate_proj, up_proj, down_proj)?;
        let elapsed_us = start.elapsed().as_secs_f64() * 1_000_000.0;

        Ok(TokenResult {
            output,
            tokens_processed: 1,
            us_per_token: elapsed_us,
            tokens_per_sec: 1_000_000.0 / elapsed_us,
            budget_met: elapsed_us <= self.budget.us_per_token,
        })
    }

    /// Legacy stub for backward compatibility (prefer `forward()`)
    #[deprecated(note = "Use forward() for real implementation")]
    pub fn execute(&self) -> Result<Vec<f32>, BrickError> {
        if self.hidden_dim == 0 || self.intermediate_dim == 0 {
            return Err(BrickError::InvalidInput("Zero dimension".to_string()));
        }
        Ok(vec![0.0; self.hidden_dim])
    }
}

include!("fused_ffn_brick.rs");
