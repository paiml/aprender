//! GPU f32↔bf16 cast kernels (R-002: BF16 mixed precision foundation)
//!
//! Provides element-wise conversion between f32 and bf16 on GPU.
//! BF16 uses the same 8-bit exponent as f32 but only 7 mantissa bits,
//! so conversion is a simple truncation (f32→bf16) or zero-extension (bf16→f32).
//!
//! # Contract (C-BF16CAST-001)
//!
//! - `cast_f32_to_bf16`: output[i] == truncate(input[i]) for all i in [0, n)
//! - `cast_bf16_to_f32`: output[i] == extend(input[i]) for all i in [0, n)
//! - Round-trip: `cast_bf16_to_f32(cast_f32_to_bf16(x))` preserves f32 values
//!   within BF16 representable range (7-bit mantissa precision)
//! - NaN/Inf preserved through both conversions

#![allow(unsafe_code)]
#![allow(trivial_casts)]
#![allow(clippy::borrow_as_ptr)]
#![allow(clippy::ref_as_ptr)]

#[cfg(feature = "cuda")]
use trueno_gpu::driver::{CudaStream, GpuBuffer, LaunchConfig};
#[cfg(feature = "cuda")]
use trueno_gpu::ptx::{PtxArithmetic, PtxComparison, PtxControl, PtxKernel, PtxModule, PtxType};

use crate::autograd::cuda_tensor::{CudaTensorError, Result};

#[cfg(feature = "cuda")]
use super::cache::FORWARD_KERNEL_CACHE;

/// Build PTX kernel for f32 → bf16 cast.
///
/// Each thread converts one element: loads f32 as u32 bits, takes upper 16 bits
/// (sign + 8-bit exponent + 7-bit mantissa), stores as u16.
#[cfg(feature = "cuda")]
fn build_cast_f32_to_bf16_ptx(_n: u32) -> String {
    let kernel = PtxKernel::new("cast_f32_to_bf16")
        .param(PtxType::U64, "src_ptr")
        .param(PtxType::U64, "dst_ptr")
        .param(PtxType::U32, "n")
        .build(|ctx| {
            let ctaid_x = ctx.special_reg(trueno_gpu::ptx::PtxReg::CtaIdX);
            let ntid_x = ctx.special_reg(trueno_gpu::ptx::PtxReg::NtidX);
            let tid_x = ctx.special_reg(trueno_gpu::ptx::PtxReg::TidX);

            let idx = ctx.mad_lo_u32(ctaid_x, ntid_x, tid_x);
            let n_param = ctx.load_param_u32("n");
            let pred = ctx.setp_ge_u32(idx, n_param);
            ctx.branch_if(pred, "exit");

            let src_ptr = ctx.load_param_u64("src_ptr");
            let dst_ptr = ctx.load_param_u64("dst_ptr");

            // Load f32 as raw u32 bits: src_ptr + idx * 4
            let offset = ctx.mul_wide_u32(idx, 4);
            let addr = ctx.add_u64(src_ptr, offset);
            let bits = ctx.ld_global_u32(addr);

            // Right-shift by 16 to get upper 16 bits (bf16 = truncated f32)
            let bf16_bits = ctx.shr_u32_imm(bits, 16);

            // Store as u16: dst_ptr + idx * 2
            let dst_offset = ctx.mul_wide_u32(idx, 2);
            let dst_addr = ctx.add_u64(dst_ptr, dst_offset);
            ctx.st_global_u16(dst_addr, bf16_bits);

            ctx.label("exit");
            ctx.ret();
        });
    PtxModule::new().target("sm_70").add_kernel(kernel).emit()
}

/// Build PTX kernel for bf16 → f32 cast.
///
/// Each thread converts one element: loads bf16 as u16, left-shifts to upper 16 bits
/// of a u32 (zero-extending the mantissa), stores as u32 (f32 bits).
#[cfg(feature = "cuda")]
fn build_cast_bf16_to_f32_ptx(_n: u32) -> String {
    let kernel = PtxKernel::new("cast_bf16_to_f32")
        .param(PtxType::U64, "src_ptr")
        .param(PtxType::U64, "dst_ptr")
        .param(PtxType::U32, "n")
        .build(|ctx| {
            let ctaid_x = ctx.special_reg(trueno_gpu::ptx::PtxReg::CtaIdX);
            let ntid_x = ctx.special_reg(trueno_gpu::ptx::PtxReg::NtidX);
            let tid_x = ctx.special_reg(trueno_gpu::ptx::PtxReg::TidX);

            let idx = ctx.mad_lo_u32(ctaid_x, ntid_x, tid_x);
            let n_param = ctx.load_param_u32("n");
            let pred = ctx.setp_ge_u32(idx, n_param);
            ctx.branch_if(pred, "exit");

            let src_ptr = ctx.load_param_u64("src_ptr");
            let dst_ptr = ctx.load_param_u64("dst_ptr");

            // Load bf16 as u16: src_ptr + idx * 2
            let src_offset = ctx.mul_wide_u32(idx, 2);
            let src_addr = ctx.add_u64(src_ptr, src_offset);
            let bf16_bits = ctx.ld_global_u16(src_addr);

            // Left-shift by 16 to place in upper 16 bits (zero-extend mantissa)
            let f32_bits = ctx.shl_u32_imm(bf16_bits, 16);

            // Store as u32 (which is f32 bits): dst_ptr + idx * 4
            let dst_offset = ctx.mul_wide_u32(idx, 4);
            let dst_addr = ctx.add_u64(dst_ptr, dst_offset);
            ctx.st_global_u32(dst_addr, f32_bits);

            ctx.label("exit");
            ctx.ret();
        });
    PtxModule::new().target("sm_70").add_kernel(kernel).emit()
}

/// Cast f32 GPU buffer to bf16 on GPU.
///
/// # Contract (C-BF16CAST-001)
///
/// - **Precondition**: `src.len() >= n`, `dst.len() >= n`, `n > 0`
/// - **Postcondition**: `dst[i]` contains bf16 representation of `src[i]` (truncated mantissa)
/// - **Invariant**: No CPU-side data transfers
#[cfg(feature = "cuda")]
pub fn cast_f32_to_bf16_gpu(
    src: &GpuBuffer<f32>,
    dst: &mut GpuBuffer<u16>,
    n: u32,
    stream: &CudaStream,
) -> Result<()> {
    let cache = FORWARD_KERNEL_CACHE.get().ok_or(CudaTensorError::DeviceNotInitialized)?;
    let mut cache = cache.lock().map_err(|_err| {
        CudaTensorError::KernelError("Failed to acquire kernel cache lock".to_string())
    })?;

    let key = "cast_f32_to_bf16";
    let module = match cache.get_cached(key) {
        Some(m) => m,
        None => {
            let ptx = build_cast_f32_to_bf16_ptx(0);
            cache.get_or_compile(key, &ptx)?
        }
    };

    let config = LaunchConfig { grid: (n.div_ceil(256), 1, 1), block: (256, 1, 1), shared_mem: 0 };

    let src_ptr = src.as_ptr();
    let dst_ptr = dst.as_ptr();

    let mut args: [*mut std::ffi::c_void; 3] =
        [&src_ptr as *const _ as *mut _, &dst_ptr as *const _ as *mut _, &n as *const _ as *mut _];

    // SAFETY: Kernel launch requires FFI. src and dst are valid GPU allocations,
    // src has n*4 bytes readable, dst has n*2 bytes writable.
    unsafe {
        stream.launch_kernel(module, "cast_f32_to_bf16", &config, &mut args).map_err(|e| {
            CudaTensorError::KernelError(format!("cast_f32_to_bf16 launch failed: {e:?}"))
        })?;
    }

    Ok(())
}

/// Cast bf16 GPU buffer to f32 on GPU.
///
/// # Contract (C-BF16CAST-001)
///
/// - **Precondition**: `src.len() >= n`, `dst.len() >= n`, `n > 0`
/// - **Postcondition**: `dst[i]` contains f32 representation of `src[i]` (zero-extended mantissa)
/// - **Invariant**: No CPU-side data transfers
#[cfg(feature = "cuda")]
pub fn cast_bf16_to_f32_gpu(
    src: &GpuBuffer<u16>,
    dst: &mut GpuBuffer<f32>,
    n: u32,
    stream: &CudaStream,
) -> Result<()> {
    let cache = FORWARD_KERNEL_CACHE.get().ok_or(CudaTensorError::DeviceNotInitialized)?;
    let mut cache = cache.lock().map_err(|_err| {
        CudaTensorError::KernelError("Failed to acquire kernel cache lock".to_string())
    })?;

    let key = "cast_bf16_to_f32";
    let module = match cache.get_cached(key) {
        Some(m) => m,
        None => {
            let ptx = build_cast_bf16_to_f32_ptx(0);
            cache.get_or_compile(key, &ptx)?
        }
    };

    let config = LaunchConfig { grid: (n.div_ceil(256), 1, 1), block: (256, 1, 1), shared_mem: 0 };

    let src_ptr = src.as_ptr();
    let dst_ptr = dst.as_ptr();

    let mut args: [*mut std::ffi::c_void; 3] =
        [&src_ptr as *const _ as *mut _, &dst_ptr as *const _ as *mut _, &n as *const _ as *mut _];

    // SAFETY: src and dst are valid GPU allocations, src has n*2 bytes, dst has n*4 bytes.
    unsafe {
        stream.launch_kernel(module, "cast_bf16_to_f32", &config, &mut args).map_err(|e| {
            CudaTensorError::KernelError(format!("cast_bf16_to_f32 launch failed: {e:?}"))
        })?;
    }

    Ok(())
}

/// CPU-side f32 to bf16 conversion for a slice (uses `half` crate).
///
/// Useful for pre-converting weights before GPU upload.
pub fn f32_slice_to_bf16(src: &[f32]) -> Vec<half::bf16> {
    src.iter().map(|&v| half::bf16::from_f32(v)).collect()
}

/// CPU-side bf16 to f32 conversion for a slice (uses `half` crate).
pub fn bf16_slice_to_f32(src: &[half::bf16]) -> Vec<f32> {
    src.iter().map(|v| v.to_f32()).collect()
}

/// GPU f32 → f16 cast using trueno's CastF32ToF16Kernel (PTX cvt.rn.f16.f32).
///
/// Enables FP16 GEMM dispatch by casting fp32 activations to fp16.
/// Overhead: ~0.02ms for 512×1536 elements at 256 GB/s — negligible vs GEMM savings.
#[cfg(feature = "cuda")]
pub fn cast_f32_to_f16_gpu(
    src: &GpuBuffer<f32>,
    dst: &mut GpuBuffer<u16>,
    n: u32,
    stream: &CudaStream,
) -> Result<()> {
    use trueno_gpu::kernels::{CastF32ToF16Kernel, Kernel};

    let cache = FORWARD_KERNEL_CACHE.get().ok_or(CudaTensorError::DeviceNotInitialized)?;
    let mut cache = cache.lock().map_err(|_err| {
        CudaTensorError::KernelError("Failed to acquire kernel cache lock".to_string())
    })?;

    let key = "cast_f32_to_f16";
    let module = match cache.get_cached(key) {
        Some(m) => m,
        None => {
            let kernel = CastF32ToF16Kernel;
            let ptx = kernel.emit_ptx_for_target(cache.sm_target());
            cache.get_or_compile(key, &ptx)?
        }
    };

    let config = LaunchConfig { grid: (n.div_ceil(256), 1, 1), block: (256, 1, 1), shared_mem: 0 };

    let src_ptr = src.as_ptr();
    let dst_ptr = dst.as_ptr();

    let mut args: [*mut std::ffi::c_void; 3] =
        [&src_ptr as *const _ as *mut _, &dst_ptr as *const _ as *mut _, &n as *const _ as *mut _];

    unsafe {
        stream
            .launch_kernel(module, "cast_f32_to_f16", &config, &mut args)
            .map_err(|e| CudaTensorError::KernelError(format!("f32→f16 cast failed: {e:?}")))?;
    }

    Ok(())
}
