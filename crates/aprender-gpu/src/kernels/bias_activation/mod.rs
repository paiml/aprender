//! Bias + Activation Epilogue Kernel
//!
//! Element-wise kernel for applying bias and activation functions to GEMM output.
//! Used as an epilogue after matrix multiplication.
//!
//! # Operations
//! - Add bias: `output[i] += bias[i % bias_size]`
//! - ReLU: `output[i] = max(0, output[i])`
//! - GELU: `output[i] = x * sigmoid(1.702 * x)`
//!
//! # Example
//! ```
//! use trueno_gpu::kernels::{BiasActivationKernel, Kernel};
//!
//! let kernel = BiasActivationKernel::new(1024, 64)
//!     .with_relu();
//! let ptx = kernel.emit_ptx();
//! assert!(ptx.contains("bias_activation"));
//! ```

use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl};
use crate::ptx::{PtxKernel, PtxType};

/// Activation function type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Activation {
    /// No activation (identity)
    #[default]
    None,
    /// ReLU: max(0, x)
    ReLU,
    /// GELU approximation: x * sigmoid(1.702 * x)
    GELU,
}

/// Bias + Activation kernel configuration
#[derive(Debug, Clone)]
pub struct BiasActivationKernel {
    /// Total number of elements
    n: u32,
    /// Size of bias vector (output is bias[i % bias_size])
    bias_size: u32,
    /// Activation function
    activation: Activation,
}

impl BiasActivationKernel {
    /// Create a new bias + activation kernel
    ///
    /// # Arguments
    /// * `n` - Total number of output elements
    /// * `bias_size` - Size of bias vector (typically output dimension)
    #[must_use]
    pub fn new(n: u32, bias_size: u32) -> Self {
        Self {
            n,
            bias_size,
            activation: Activation::None,
        }
    }

    /// Add ReLU activation
    #[must_use]
    pub fn with_relu(mut self) -> Self {
        self.activation = Activation::ReLU;
        self
    }

    /// Add GELU activation
    #[must_use]
    pub fn with_gelu(mut self) -> Self {
        self.activation = Activation::GELU;
        self
    }

    /// Set activation function
    #[must_use]
    pub fn with_activation(mut self, activation: Activation) -> Self {
        self.activation = activation;
        self
    }
}

impl super::Kernel for BiasActivationKernel {
    fn name(&self) -> &str {
        "bias_activation"
    }

    fn build_ptx(&self) -> PtxKernel {
        let activation = self.activation;
        let bias_size = self.bias_size;

        PtxKernel::new("bias_activation")
            .param(PtxType::U64, "output")
            .param(PtxType::U64, "bias")
            .param(PtxType::U32, "n")
            .build(|ctx| {
                // Thread index calculation: global_id = blockIdx.x * blockDim.x + threadIdx.x
                let ctaid_x = ctx.special_reg(crate::ptx::PtxReg::CtaIdX);
                let ntid_x = ctx.special_reg(crate::ptx::PtxReg::NtidX);
                let tid_x = ctx.special_reg(crate::ptx::PtxReg::TidX);
                let global_id = ctx.mad_lo_u32(ctaid_x, ntid_x, tid_x);

                // Bounds check: if (global_id >= n) return
                let n_param = ctx.load_param_u32("n");
                let out_of_bounds = ctx.setp_ge_u32(global_id, n_param);
                ctx.branch_if(out_of_bounds, "exit");

                // Load output[global_id]
                let output_ptr = ctx.load_param_u64("output");
                let offset = ctx.mul_wide_u32(global_id, 4); // sizeof(f32)
                let addr = ctx.add_u64(output_ptr, offset);
                let value = ctx.ld_global_f32(addr);

                // Load bias[global_id % bias_size] - bias_size is baked into kernel
                let bias_ptr = ctx.load_param_u64("bias");
                let bias_idx = ctx.rem_u32(global_id, bias_size);
                let bias_offset = ctx.mul_wide_u32(bias_idx, 4);
                let bias_addr = ctx.add_u64(bias_ptr, bias_offset);
                let bias_val = ctx.ld_global_f32(bias_addr);

                // Add bias: result = value + bias
                let result = ctx.add_f32(value, bias_val);

                // Apply activation function
                let activated = match activation {
                    Activation::None => result,
                    Activation::ReLU => {
                        // ReLU: max(0, x)
                        let zero = ctx.mov_f32_imm(0.0);
                        ctx.max_f32(result, zero)
                    }
                    Activation::GELU => {
                        // GELU approximation: x * sigmoid(1.702 * x)
                        // sigmoid(y) = 1 / (1 + exp(-y))
                        // exp(-y) ≈ 2^(-y * log2(e))
                        let coeff = ctx.mov_f32_imm(1.702);
                        let scaled = ctx.mul_f32(result, coeff);

                        // Compute exp(-scaled) via ex2: negate by subtracting from 0
                        let zero = ctx.mov_f32_imm(0.0);
                        let neg_scaled = ctx.sub_f32(zero, scaled);
                        let log2_e = ctx.mov_f32_imm(std::f32::consts::LOG2_E);
                        let scaled_for_ex2 = ctx.mul_f32(neg_scaled, log2_e);
                        let exp_val = ctx.ex2_f32(scaled_for_ex2);

                        // sigmoid = 1 / (1 + exp(-scaled))
                        let one = ctx.mov_f32_imm(1.0);
                        let denom = ctx.add_f32(one, exp_val);
                        let sigmoid = ctx.div_f32(one, denom);

                        // GELU = x * sigmoid(1.702 * x)
                        ctx.mul_f32(result, sigmoid)
                    }
                };

                // Store result back to output[global_id]
                ctx.st_global_f32(addr, activated);

                ctx.label("exit");
                ctx.ret();
            })
    }
}

#[cfg(test)]
mod tests;
