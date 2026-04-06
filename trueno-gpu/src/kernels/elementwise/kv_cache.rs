//! KV Cache Kernels
//!
//! Kernels for efficient KV cache updates in autoregressive inference.
//!
//! - `KvCacheScatterKernel`: Scatter K/V vectors to cache positions
//! - `KvCacheScatterIndirectKernel`: CUDA Graph compatible version

#![allow(clippy::similar_names)]

use crate::kernels::Kernel;
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl};
use crate::ptx::{PtxKernel, PtxReg, PtxType};

// ============================================================================
// PAR-052: KV Cache Scatter Kernels
// ============================================================================

/// KV Cache Scatter Kernel: Scatter K/V vectors to strided KV cache positions
///
/// Used to update KV cache at specific positions without full D2D copies.
/// Replaces 672+ D2D copies per token with two kernel launches.
#[derive(Debug, Clone)]
pub struct KvCacheScatterKernel {
    /// Number of KV heads
    pub num_kv_heads: u32,
    /// Head dimension
    pub head_dim: u32,
    /// Maximum sequence length
    pub max_len: u32,
}

impl KvCacheScatterKernel {
    /// Create a new KV cache scatter kernel
    #[must_use]
    pub const fn new(num_kv_heads: u32, head_dim: u32, max_len: u32) -> Self {
        Self { num_kv_heads, head_dim, max_len }
    }
}

impl Kernel for KvCacheScatterKernel {
    fn name(&self) -> &str {
        "kv_cache_scatter"
    }

    fn build_ptx(&self) -> PtxKernel {
        PtxKernel::new("kv_cache_scatter")
            .param(PtxType::U64, "src_ptr")
            .param(PtxType::U64, "cache_ptr")
            .param(PtxType::U32, "pos")
            .param(PtxType::U32, "head_dim")
            .param(PtxType::U32, "max_len")
            .build(|ctx| {
                let tid = ctx.special_reg(PtxReg::TidX);
                let ctaid = ctx.special_reg(PtxReg::CtaIdX);
                let src_ptr = ctx.load_param_u64("src_ptr");
                let cache_ptr = ctx.load_param_u64("cache_ptr");
                let pos = ctx.load_param_u32("pos");
                let head_dim = ctx.load_param_u32("head_dim");
                let max_len = ctx.load_param_u32("max_len");

                // Each block handles one head, each thread one element
                let head_idx = ctaid;
                let elem_idx = tid;

                // Bounds check
                let in_bounds = ctx.setp_lt_u32(elem_idx, head_dim);
                ctx.branch_if_not(in_bounds, "exit");

                // Source offset: head_idx * head_dim + elem_idx
                let src_head_offset = ctx.mul_lo_u32(head_idx, head_dim);
                let src_offset = ctx.add_u32_reg(src_head_offset, elem_idx);
                let four = ctx.mov_u32_imm(4);
                let src_bytes = ctx.mul_lo_u32(src_offset, four);
                let src_bytes_64 = ctx.cvt_u64_u32(src_bytes);
                let src_addr = ctx.add_u64(src_ptr, src_bytes_64);

                // Cache offset: (head_idx * max_len + pos) * head_dim + elem_idx
                let cache_head_stride = ctx.mul_lo_u32(head_idx, max_len);
                let cache_pos_offset = ctx.add_u32_reg(cache_head_stride, pos);
                let cache_elem_stride = ctx.mul_lo_u32(cache_pos_offset, head_dim);
                let cache_offset = ctx.add_u32_reg(cache_elem_stride, elem_idx);
                let cache_bytes = ctx.mul_lo_u32(cache_offset, four);
                let cache_bytes_64 = ctx.cvt_u64_u32(cache_bytes);
                let cache_addr = ctx.add_u64(cache_ptr, cache_bytes_64);

                // Load from source and store to cache
                let val = ctx.ld_global_f32(src_addr);
                ctx.st_global_f32(cache_addr, val);

                ctx.label("exit");
                ctx.ret();
            })
    }
}

/// KV Cache Scatter Indirect Kernel: CUDA Graph compatible version
///
/// Reads position from device memory instead of kernel parameter.
#[derive(Debug, Clone)]
pub struct KvCacheScatterIndirectKernel {
    /// Number of KV heads
    pub num_kv_heads: u32,
    /// Head dimension
    pub head_dim: u32,
    /// Maximum sequence length
    pub max_len: u32,
}

impl KvCacheScatterIndirectKernel {
    /// Create a new indirect KV cache scatter kernel
    #[must_use]
    pub const fn new(num_kv_heads: u32, head_dim: u32, max_len: u32) -> Self {
        Self { num_kv_heads, head_dim, max_len }
    }
}

impl Kernel for KvCacheScatterIndirectKernel {
    fn name(&self) -> &str {
        "kv_cache_scatter_indirect"
    }

    fn build_ptx(&self) -> PtxKernel {
        PtxKernel::new("kv_cache_scatter_indirect")
            .param(PtxType::U64, "src_ptr")
            .param(PtxType::U64, "cache_ptr")
            .param(PtxType::U64, "pos_ptr") // Indirect: read from device memory
            .param(PtxType::U32, "head_dim")
            .param(PtxType::U32, "max_len")
            .build(|ctx| {
                let tid = ctx.special_reg(PtxReg::TidX);
                let ctaid = ctx.special_reg(PtxReg::CtaIdX);
                let src_ptr = ctx.load_param_u64("src_ptr");
                let cache_ptr = ctx.load_param_u64("cache_ptr");
                let pos_ptr = ctx.load_param_u64("pos_ptr");
                let head_dim = ctx.load_param_u32("head_dim");
                let max_len = ctx.load_param_u32("max_len");

                // Read position from device memory (indirect)
                let pos = ctx.ld_global_u32(pos_ptr);

                let head_idx = ctaid;
                let elem_idx = tid;

                let in_bounds = ctx.setp_lt_u32(elem_idx, head_dim);
                ctx.branch_if_not(in_bounds, "exit");

                let src_head_offset = ctx.mul_lo_u32(head_idx, head_dim);
                let src_offset = ctx.add_u32_reg(src_head_offset, elem_idx);
                let four = ctx.mov_u32_imm(4);
                let src_bytes = ctx.mul_lo_u32(src_offset, four);
                let src_bytes_64 = ctx.cvt_u64_u32(src_bytes);
                let src_addr = ctx.add_u64(src_ptr, src_bytes_64);

                let cache_head_stride = ctx.mul_lo_u32(head_idx, max_len);
                let cache_pos_offset = ctx.add_u32_reg(cache_head_stride, pos);
                let cache_elem_stride = ctx.mul_lo_u32(cache_pos_offset, head_dim);
                let cache_offset = ctx.add_u32_reg(cache_elem_stride, elem_idx);
                let cache_bytes = ctx.mul_lo_u32(cache_offset, four);
                let cache_bytes_64 = ctx.cvt_u64_u32(cache_bytes);
                let cache_addr = ctx.add_u64(cache_ptr, cache_bytes_64);

                let val = ctx.ld_global_f32(src_addr);
                ctx.st_global_f32(cache_addr, val);

                ctx.label("exit");
                ctx.ret();
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kv_cache_scatter_kernel_name() {
        let kernel = KvCacheScatterKernel::new(8, 64, 2048);
        assert_eq!(kernel.name(), "kv_cache_scatter");
    }

    #[test]
    fn test_kv_cache_scatter_ptx_generation() {
        let kernel = KvCacheScatterKernel::new(8, 64, 2048);
        let ptx = kernel.emit_ptx();

        assert!(ptx.contains(".entry kv_cache_scatter"));
        assert!(ptx.contains(".param .u64 src_ptr"));
        assert!(ptx.contains(".param .u64 cache_ptr"));
        assert!(ptx.contains(".param .u32 pos"));
    }

    #[test]
    fn test_kv_cache_scatter_indirect_kernel_name() {
        let kernel = KvCacheScatterIndirectKernel::new(8, 64, 2048);
        assert_eq!(kernel.name(), "kv_cache_scatter_indirect");
    }

    #[test]
    fn test_kv_cache_scatter_indirect_ptx_generation() {
        let kernel = KvCacheScatterIndirectKernel::new(8, 64, 2048);
        let ptx = kernel.emit_ptx();

        // Verify indirect position read (u64 pointer instead of u32 value)
        assert!(ptx.contains(".entry kv_cache_scatter_indirect"));
        assert!(ptx.contains(".param .u64 pos_ptr"));
        assert!(ptx.contains("ld.global.u32"));
    }
}
