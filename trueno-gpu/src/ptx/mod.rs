//! PTX Code Generation Module
//!
//! Pure Rust PTX generation - no LLVM, no nvcc, no external dependencies.
//!
//! ## Architecture
//!
//! ```text
//! Rust Kernel (builder) → PTX IR → PTX Text → CUDA Driver → GPU
//! ```
//!
//! ## Example
//!
//! ```rust
//! use trueno_gpu::ptx::{PtxModule, PtxKernel, PtxType};
//!
//! let module = PtxModule::new()
//!     .version(8, 0)
//!     .target("sm_70")
//!     .address_size(64);
//!
//! let ptx = module.emit();
//! assert!(ptx.contains(".version 8.0"));
//! ```

pub mod builder;
mod emit;
mod instructions;
pub mod optimize;
pub mod parity;
mod registers;
mod types;

pub use builder::{
    KernelBuilder, KernelBuilderCore, PtxArithmetic, PtxAtomic, PtxComparison, PtxControl,
    PtxKernel, PtxMemory, PtxModule, PtxSync,
};
pub use instructions::{CmpOp, Operand, PtxInstruction, PtxOp, WmmaLayout, WmmaShape};
pub use registers::{LiveRange, PhysicalReg, PtxReg, RegisterAllocator, VirtualReg};
pub use types::{PtxStateSpace, PtxType};

use crate::error::{GpuError, Result};

/// Minimum supported PTX version (7.0 for SM 7.0+)
pub const MIN_PTX_VERSION: (u32, u32) = (7, 0);

/// Validate PTX version
pub fn validate_version(major: u32, minor: u32) -> Result<()> {
    if major < MIN_PTX_VERSION.0 || (major == MIN_PTX_VERSION.0 && minor < MIN_PTX_VERSION.1) {
        return Err(GpuError::InvalidPtxVersion { major, minor });
    }
    Ok(())
}

/// Validate compute capability target
pub fn validate_target(target: &str) -> Result<()> {
    // Must be sm_XX format where XX >= 70
    if !target.starts_with("sm_") {
        return Err(GpuError::InvalidTarget(target.to_string()));
    }

    let version_str = &target[3..];
    let version: u32 =
        version_str.parse().map_err(|_| GpuError::InvalidTarget(target.to_string()))?;

    if version < 70 {
        return Err(GpuError::InvalidTarget(target.to_string()));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========== EXTREME TDD: Comprehensive PTX Module Tests ==========

    #[test]
    fn test_validate_version_valid() {
        assert!(validate_version(7, 0).is_ok());
        assert!(validate_version(8, 0).is_ok());
        assert!(validate_version(8, 5).is_ok());
        assert!(validate_version(9, 0).is_ok());
    }

    #[test]
    fn test_validate_version_invalid() {
        assert!(validate_version(6, 5).is_err());
        assert!(validate_version(6, 0).is_err());
        assert!(validate_version(5, 0).is_err());
    }

    #[test]
    fn test_validate_target_valid() {
        assert!(validate_target("sm_70").is_ok());
        assert!(validate_target("sm_75").is_ok());
        assert!(validate_target("sm_80").is_ok());
        assert!(validate_target("sm_86").is_ok());
        assert!(validate_target("sm_89").is_ok());
        assert!(validate_target("sm_90").is_ok());
        // trueno#188: Blackwell support
        assert!(validate_target("sm_100").is_ok());
        assert!(validate_target("sm_120").is_ok());
        assert!(validate_target("sm_121").is_ok());
    }

    #[test]
    fn test_validate_target_invalid() {
        assert!(validate_target("sm_50").is_err());
        assert!(validate_target("sm_60").is_err());
        assert!(validate_target("sm_61").is_err());
        assert!(validate_target("compute_70").is_err());
        assert!(validate_target("70").is_err());
        assert!(validate_target("").is_err());
    }

    #[test]
    fn test_module_creation() {
        let module = PtxModule::new();
        assert_eq!(module.get_version(), (8, 0)); // Default
    }

    #[test]
    fn test_module_version_builder() {
        let module = PtxModule::new().version(8, 5);
        assert_eq!(module.get_version(), (8, 5));
    }

    #[test]
    fn test_module_target_builder() {
        let module = PtxModule::new().target("sm_86");
        assert_eq!(module.get_target(), "sm_86");
    }

    #[test]
    fn test_module_address_size() {
        let module = PtxModule::new().address_size(64);
        assert_eq!(module.get_address_size(), 64);
    }

    #[test]
    fn test_module_emit_header() {
        let module = PtxModule::new().version(8, 0).target("sm_70").address_size(64);

        let ptx = module.emit();

        assert!(ptx.contains(".version 8.0"));
        assert!(ptx.contains(".target sm_70"));
        assert!(ptx.contains(".address_size 64"));
    }

    #[test]
    fn test_module_emit_with_kernel() {
        let kernel = PtxKernel::new("vector_add")
            .param(PtxType::U64, "a_ptr")
            .param(PtxType::U64, "b_ptr")
            .param(PtxType::U64, "c_ptr")
            .param(PtxType::U32, "n");

        let module =
            PtxModule::new().version(8, 0).target("sm_70").address_size(64).add_kernel(kernel);

        let ptx = module.emit();

        assert!(ptx.contains(".visible .entry vector_add"));
        assert!(ptx.contains(".param .u64 a_ptr"));
        assert!(ptx.contains(".param .u64 b_ptr"));
        assert!(ptx.contains(".param .u64 c_ptr"));
        assert!(ptx.contains(".param .u32 n"));
    }

    #[test]
    fn test_kernel_with_shared_memory() {
        let kernel = PtxKernel::new("gemm_tiled").shared_memory(4096); // 4KB shared memory

        assert_eq!(kernel.shared_memory_bytes(), 4096);
    }

    #[test]
    fn test_ptx_type_sizes() {
        assert_eq!(PtxType::U8.size_bytes(), 1);
        assert_eq!(PtxType::U16.size_bytes(), 2);
        assert_eq!(PtxType::U32.size_bytes(), 4);
        assert_eq!(PtxType::U64.size_bytes(), 8);
        assert_eq!(PtxType::F16.size_bytes(), 2);
        assert_eq!(PtxType::F32.size_bytes(), 4);
        assert_eq!(PtxType::F64.size_bytes(), 8);
    }

    #[test]
    fn test_ptx_type_to_string() {
        assert_eq!(PtxType::U32.to_ptx_string(), ".u32");
        assert_eq!(PtxType::U64.to_ptx_string(), ".u64");
        assert_eq!(PtxType::F32.to_ptx_string(), ".f32");
        assert_eq!(PtxType::F16.to_ptx_string(), ".f16");
        assert_eq!(PtxType::Pred.to_ptx_string(), ".pred");
    }

    #[test]
    fn test_special_registers() {
        assert_eq!(PtxReg::TidX.to_ptx_string(), "%tid.x");
        assert_eq!(PtxReg::TidY.to_ptx_string(), "%tid.y");
        assert_eq!(PtxReg::TidZ.to_ptx_string(), "%tid.z");
        assert_eq!(PtxReg::CtaIdX.to_ptx_string(), "%ctaid.x");
        assert_eq!(PtxReg::NtidX.to_ptx_string(), "%ntid.x");
    }

    #[test]
    fn test_virtual_register_allocation() {
        let mut allocator = RegisterAllocator::new();

        let r1 = allocator.allocate_virtual(PtxType::F32);
        let r2 = allocator.allocate_virtual(PtxType::F32);
        let r3 = allocator.allocate_virtual(PtxType::U32);

        assert_ne!(r1.id(), r2.id());
        assert_ne!(r2.id(), r3.id());
    }

    #[test]
    fn test_register_pressure_tracking() {
        let mut allocator = RegisterAllocator::new();

        // Allocate some registers
        let _r1 = allocator.allocate_virtual(PtxType::F32);
        let _r2 = allocator.allocate_virtual(PtxType::F32);
        let _r3 = allocator.allocate_virtual(PtxType::F32);

        let pressure = allocator.pressure_report();
        assert_eq!(pressure.max_live, 3);
        assert_eq!(pressure.spill_count, 0);
    }

    #[test]
    fn test_emit_vector_add_kernel() {
        // This is the acceptance test from spec TG-001
        let kernel = PtxKernel::new("vector_add")
            .param(PtxType::U64, "a_ptr")
            .param(PtxType::U64, "b_ptr")
            .param(PtxType::U64, "c_ptr")
            .param(PtxType::U32, "n")
            .build(|ctx| {
                // Calculate global thread index
                let tid = ctx.special_reg(PtxReg::TidX);
                let ctaid = ctx.special_reg(PtxReg::CtaIdX);
                let ntid = ctx.special_reg(PtxReg::NtidX);

                let idx = ctx.mad_lo_u32(ctaid, ntid, tid);

                // Bounds check
                let n = ctx.load_param_u32("n");
                let pred = ctx.setp_ge_u32(idx, n);
                ctx.branch_if(pred, "exit");

                // Load, add, store
                let a_ptr = ctx.load_param_u64("a_ptr");
                let b_ptr = ctx.load_param_u64("b_ptr");
                let c_ptr = ctx.load_param_u64("c_ptr");

                let offset = ctx.mul_wide_u32(idx, 4);
                let a_addr = ctx.add_u64(a_ptr, offset);
                let b_addr = ctx.add_u64(b_ptr, offset);
                let c_addr = ctx.add_u64(c_ptr, offset);

                let a_val = ctx.ld_global_f32(a_addr);
                let b_val = ctx.ld_global_f32(b_addr);
                let c_val = ctx.add_f32(a_val, b_val);
                ctx.st_global_f32(c_addr, c_val);

                ctx.label("exit");
                ctx.ret();
            });

        let module =
            PtxModule::new().version(8, 0).target("sm_70").address_size(64).add_kernel(kernel);

        let ptx = module.emit();

        // Verify key PTX instructions are present
        assert!(ptx.contains("mov.u32"));
        assert!(ptx.contains("%tid.x"));
        assert!(ptx.contains("mad.lo"));
        assert!(ptx.contains("setp.ge"));
        assert!(ptx.contains("ld.global.f32"));
        assert!(ptx.contains("add.f32"));
        assert!(ptx.contains("st.global.f32"));
        assert!(ptx.contains("ret;"));
    }
}
