//! SwiGLU Activation Kernels
//!
//! Fused SwiGLU kernels for transformer FFN blocks.
//!
//! - `FusedSwigluKernel`: Single-sequence fused SiLU + multiply
//! - `BatchedSwigluKernel`: Multi-sequence batched version

#![allow(clippy::similar_names)]

use crate::kernels::Kernel;
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl};
use crate::ptx::{PtxKernel, PtxReg, PtxType};

/// Fused SwiGLU Kernel: output = silu(gate) * up
///
/// Combines SiLU activation and element-wise multiply in one pass.
/// This is the gated activation used in LLaMA FFN.
///
/// gate_proj = x @ W_gate
/// up_proj = x @ W_up
/// output = silu(gate_proj) * up_proj
///
/// # PAR-023: Fused kernel eliminates one memory round-trip
#[derive(Debug, Clone)]
pub struct FusedSwigluKernel {
    /// Number of elements
    pub n: u32,
}

impl FusedSwigluKernel {
    /// Create a new fused SwiGLU kernel
    #[must_use]
    pub const fn new(n: u32) -> Self {
        Self { n }
    }
}

impl Kernel for FusedSwigluKernel {
    fn name(&self) -> &str {
        "fused_swiglu"
    }

    fn build_ptx(&self) -> PtxKernel {
        PtxKernel::new("fused_swiglu")
            .param(PtxType::U64, "gate_ptr") // gate_proj
            .param(PtxType::U64, "up_ptr") // up_proj
            .param(PtxType::U64, "output_ptr")
            .param(PtxType::U32, "n")
            .build(|ctx| {
                // Global thread ID
                let tid = ctx.special_reg(PtxReg::TidX);
                let ctaid = ctx.special_reg(PtxReg::CtaIdX);
                let ntid = ctx.special_reg(PtxReg::NtidX);
                let gid = ctx.mad_lo_u32(ctaid, ntid, tid);

                // Load parameters
                let n = ctx.load_param_u32("n");
                let gate_ptr = ctx.load_param_u64("gate_ptr");
                let up_ptr = ctx.load_param_u64("up_ptr");
                let output_ptr = ctx.load_param_u64("output_ptr");

                // Bounds check
                let in_bounds = ctx.setp_lt_u32(gid, n);
                ctx.branch_if_not(in_bounds, "exit");

                // Calculate address
                let four = ctx.mov_u32_imm(4);
                let offset = ctx.mul_wide_u32_reg(gid, four);
                let gate_addr = ctx.add_u64(gate_ptr, offset);
                let up_addr = ctx.add_u64(up_ptr, offset);
                let out_addr = ctx.add_u64(output_ptr, offset);

                // Load gate and up
                let gate = ctx.ld_global_f32(gate_addr);
                let up = ctx.ld_global_f32(up_addr);

                // Compute SiLU(gate): gate * sigmoid(gate)
                let zero = ctx.mov_f32_imm(0.0);
                let neg_gate = ctx.sub_f32(zero, gate);
                let log2_e = ctx.mov_f32_imm(std::f32::consts::LOG2_E);
                let scaled = ctx.mul_f32(neg_gate, log2_e);
                let exp_neg = ctx.ex2_f32(scaled);
                let one = ctx.mov_f32_imm(1.0);
                let denom = ctx.add_f32(one, exp_neg);
                let sigmoid = ctx.div_f32(one, denom);
                let silu_gate = ctx.mul_f32(gate, sigmoid);

                // Multiply: silu(gate) * up
                let result = ctx.mul_f32(silu_gate, up);

                // Store
                ctx.st_global_f32(out_addr, result);

                ctx.label("exit");
                ctx.ret();
            })
    }
}

// ============================================================================
// PAR-114: Batched SwiGLU Kernel (processes M sequences in parallel)
// ============================================================================

/// Batched SwiGLU: output[m] = silu(gate[m]) * up[m] for m in 0..M
///
/// Processes M sequences in parallel using Grid.y for batch index.
///
/// # Parameters
///
/// - `gate_ptr`: Packed gate values [M × n]
/// - `up_ptr`: Packed up values [M × n]
/// - `output_ptr`: Output [M × n]
///
/// # Grid Configuration
///
/// - Grid: (ceil(n/256), batch_size, 1)
/// - Block: (256, 1, 1)
#[derive(Debug, Clone)]
pub struct BatchedSwigluKernel {
    /// Elements per sequence
    pub n: u32,
    /// Batch size (M)
    pub batch_size: u32,
}

impl BatchedSwigluKernel {
    /// Create a new batched SwiGLU kernel
    #[must_use]
    pub const fn new(n: u32, batch_size: u32) -> Self {
        Self { n, batch_size }
    }
}

impl Kernel for BatchedSwigluKernel {
    fn name(&self) -> &str {
        "batched_swiglu"
    }

    fn build_ptx(&self) -> PtxKernel {
        let n = self.n;

        PtxKernel::new("batched_swiglu")
            .param(PtxType::U64, "gate_ptr")
            .param(PtxType::U64, "up_ptr")
            .param(PtxType::U64, "output_ptr")
            .build(move |ctx| {
                // Global thread ID within the sequence
                let tid = ctx.special_reg(PtxReg::TidX);
                let ctaid_x = ctx.special_reg(PtxReg::CtaIdX);
                let batch_idx = ctx.special_reg(PtxReg::CtaIdY); // Grid.y = sequence index
                let ntid = ctx.special_reg(PtxReg::NtidX);
                let local_gid = ctx.mad_lo_u32(ctaid_x, ntid, tid);

                // Load parameters
                let gate_ptr = ctx.load_param_u64("gate_ptr");
                let up_ptr = ctx.load_param_u64("up_ptr");
                let output_ptr = ctx.load_param_u64("output_ptr");

                // Bounds check within sequence
                let n_val = ctx.mov_u32_imm(n);
                let in_bounds = ctx.setp_lt_u32(local_gid, n_val);
                ctx.branch_if_not(in_bounds, "exit");

                // Calculate global element index: batch_idx × n + local_gid
                let batch_offset = ctx.mul_lo_u32(batch_idx, n_val);
                let gid = ctx.add_u32_reg(batch_offset, local_gid);

                // Calculate byte address (gid × 4 bytes)
                let four = ctx.mov_u32_imm(4);
                let offset = ctx.mul_wide_u32_reg(gid, four);
                let gate_addr = ctx.add_u64(gate_ptr, offset);
                let up_addr = ctx.add_u64(up_ptr, offset);
                let out_addr = ctx.add_u64(output_ptr, offset);

                // Load gate and up values
                let gate = ctx.ld_global_f32(gate_addr);
                let up = ctx.ld_global_f32(up_addr);

                // Compute SiLU(gate): gate × sigmoid(gate)
                let zero = ctx.mov_f32_imm(0.0);
                let neg_gate = ctx.sub_f32(zero, gate);
                let log2_e = ctx.mov_f32_imm(std::f32::consts::LOG2_E);
                let scaled = ctx.mul_f32(neg_gate, log2_e);
                let exp_neg = ctx.ex2_f32(scaled);
                let one = ctx.mov_f32_imm(1.0);
                let denom = ctx.add_f32(one, exp_neg);
                let sigmoid = ctx.div_f32(one, denom);
                let silu_gate = ctx.mul_f32(gate, sigmoid);

                // Multiply: silu(gate) × up
                let result = ctx.mul_f32(silu_gate, up);

                // Store
                ctx.st_global_f32(out_addr, result);

                ctx.label("exit");
                ctx.ret();
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fused_swiglu_kernel_name() {
        let kernel = FusedSwigluKernel::new(2048);
        assert_eq!(kernel.name(), "fused_swiglu");
    }

    #[test]
    fn test_fused_swiglu_ptx_generation() {
        let kernel = FusedSwigluKernel::new(2048);
        let ptx = kernel.emit_ptx();

        // Verify entry point
        assert!(ptx.contains(".entry fused_swiglu"));

        // Verify gate_ptr and up_ptr params
        assert!(ptx.contains(".param .u64 gate_ptr"));
        assert!(ptx.contains(".param .u64 up_ptr"));

        // Verify SiLU computation (exp and division)
        assert!(ptx.contains("ex2.approx.f32"));
        assert!(ptx.contains("div.rn.f32"));
    }

    #[test]
    fn test_batched_swiglu_kernel_name() {
        let kernel = BatchedSwigluKernel::new(2048, 4);
        assert_eq!(kernel.name(), "batched_swiglu");
    }

    #[test]
    fn test_batched_swiglu_ptx_generation() {
        let kernel = BatchedSwigluKernel::new(2048, 4);
        let ptx = kernel.emit_ptx();

        assert!(ptx.contains(".entry batched_swiglu"));
        assert!(ptx.contains("ex2.approx.f32"));
        assert!(ptx.contains("div.rn.f32"));
    }
}
