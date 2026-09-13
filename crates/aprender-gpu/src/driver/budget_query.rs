//! CUDA-backed queries that feed [`crate::launch_budget`].
//!
//! Split deliberately: the *policy* (`crate::launch_budget::validate_launch`) is pure and
//! ungated so its case table runs in the required check; only these *queries* need a
//! device. Putting the policy here would have made it untestable in CI, because the whole
//! `driver` module is `#[cfg(feature = "cuda")]` — the same trap that leaves
//! `driver::ptx_patch`'s GH-480 tests dark despite its comment claiming otherwise.

use super::sys::{
    CUdevice, CUfunction, CudaDriver, CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
    CU_DEVICE_ATTRIBUTE_WARP_SIZE, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
    CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, CU_FUNC_ATTRIBUTE_NUM_REGS,
    CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
};
use crate::error::GpuError;
use crate::launch_budget::{DeviceLimits, KernelAttributes};
use std::os::raw::c_int;

fn driver() -> Result<&'static CudaDriver, GpuError> {
    CudaDriver::load().ok_or_else(|| {
        GpuError::CudaDriver("libcuda not loadable for budget query".to_string(), -1)
    })
}

fn func_attr(d: &CudaDriver, attrib: c_int, func: CUfunction) -> Result<u32, GpuError> {
    let mut v: c_int = 0;
    // SAFETY: `func` is a live CUfunction from cuModuleGetFunction, `attrib` is one of the
    // CU_FUNC_ATTRIBUTE_* constants, and `v` is a valid out-pointer for the call's lifetime.
    CudaDriver::check(unsafe { (d.cuFuncGetAttribute)(&mut v, attrib, func) })?;
    u32::try_from(v)
        .map_err(|_| GpuError::CudaDriver(format!("negative function attribute {attrib}: {v}"), -1))
}

fn dev_attr(d: &CudaDriver, attrib: c_int, device: CUdevice) -> Result<u32, GpuError> {
    let mut v: c_int = 0;
    // SAFETY: `device` is a valid CUdevice ordinal handle and `v` is a valid out-pointer.
    CudaDriver::check(unsafe { (d.cuDeviceGetAttribute)(&mut v, attrib, device) })?;
    u32::try_from(v)
        .map_err(|_| GpuError::CudaDriver(format!("negative device attribute {attrib}: {v}"), -1))
}

/// Read a compiled kernel's resource usage back from the driver.
///
/// These describe the **cubin the JIT actually produced**, not the PTX we emitted — which
/// is the whole point: a codegen change that inflates register pressure is invisible in the
/// PTX text and shows up here.
///
/// # Safety
///
/// `func` must be a live `CUfunction` obtained from `cuModuleGetFunction` on a module that
/// is still loaded. The driver dereferences it.
///
/// # Errors
///
/// Returns [`GpuError::CudaDriver`] if libcuda is unavailable or any attribute query fails.
pub unsafe fn kernel_attributes(func: CUfunction) -> Result<KernelAttributes, GpuError> {
    let d = driver()?;
    Ok(KernelAttributes {
        max_threads_per_block: func_attr(d, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, func)?,
        num_regs: func_attr(d, CU_FUNC_ATTRIBUTE_NUM_REGS, func)?,
        static_shared_bytes: func_attr(d, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, func)?,
        local_bytes: func_attr(d, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, func)?,
    })
}

/// Query a device's limits instead of looking them up in a per-SM table.
///
/// The hand-written `CU_TARGET_COMPUTE_*` table stops at 90 and the contract's own `domain`
/// stops at 90. Querying means sm_100 / sm_120 / sm_121 and whatever ships next need no
/// edit here.
///
/// # Errors
///
/// Returns [`GpuError::CudaDriver`] if libcuda is unavailable or any attribute query fails.
pub fn device_limits(device: CUdevice) -> Result<DeviceLimits, GpuError> {
    let d = driver()?;
    Ok(DeviceLimits {
        max_threads_per_block: dev_attr(d, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, device)?,
        max_shared_per_block: dev_attr(d, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, device)?,
        max_regs_per_block: dev_attr(d, CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK, device)?,
        warp_size: dev_attr(d, CU_DEVICE_ATTRIBUTE_WARP_SIZE, device)?,
    })
}

/// The contract's own postcondition: `cuOccupancyMaxActiveBlocksPerMultiprocessor > 0`.
///
/// Zero means the configuration cannot be resident on any SM — the launch is impossible,
/// not merely slow.
///
/// # Safety
///
/// `func` must be a live `CUfunction` from a still-loaded module.
///
/// # Errors
///
/// Returns [`GpuError::CudaDriver`] if libcuda is unavailable or the query fails.
pub unsafe fn occupancy_max_active_blocks_per_sm(
    func: CUfunction,
    block_size: u32,
    dynamic_shared_bytes: u32,
) -> Result<u32, GpuError> {
    let d = driver()?;
    let mut blocks: c_int = 0;
    let bs = c_int::try_from(block_size).map_err(|_| {
        GpuError::InvalidParameter(format!("block_size {block_size} exceeds c_int"))
    })?;
    // SAFETY: `func` is a live CUfunction, `blocks` is a valid out-pointer, and the block
    // size / dynamic shared size are plain scalars the driver validates itself.
    CudaDriver::check(unsafe {
        (d.cuOccupancyMaxActiveBlocksPerMultiprocessor)(
            &mut blocks,
            func,
            bs,
            dynamic_shared_bytes as usize,
        )
    })?;
    u32::try_from(blocks)
        .map_err(|_| GpuError::CudaDriver(format!("negative occupancy: {blocks}"), -1))
}

/// Full `register_budget` check for one kernel on one device.
///
/// Runs the pure policy, then asserts the contract's postcondition against the driver.
/// This is the equation `contracts/trueno/ptx-codegen-safety-v1.yaml` has declared since
/// 2026-04-06 and never enforced.
///
/// # Safety
///
/// `func` must be a live `CUfunction` from a still-loaded module, and `device` the ordinal
/// of the device that module was loaded on.
///
/// # Errors
///
/// Returns [`GpuError::InvalidParameter`] naming the violated budget, or
/// [`GpuError::CudaDriver`] if a query fails.
pub unsafe fn enforce_register_budget(
    func: CUfunction,
    device: CUdevice,
    block_size: u32,
    dynamic_shared_bytes: u32,
) -> Result<(), GpuError> {
    // SAFETY: forwarded from this fn's own contract on `func` and `device`.
    let attrs = unsafe { kernel_attributes(func) }?;
    let limits = device_limits(device)?;
    crate::launch_budget::validate_launch(&attrs, &limits, block_size, dynamic_shared_bytes)
        .map_err(|v| GpuError::InvalidParameter(v.to_string()))?;
    // SAFETY: same contract as above.
    let blocks =
        unsafe { occupancy_max_active_blocks_per_sm(func, block_size, dynamic_shared_bytes) }?;
    if blocks == 0 {
        return Err(GpuError::InvalidParameter(format!(
            "cuOccupancyMaxActiveBlocksPerMultiprocessor == 0 for block_size={block_size}, \
             dynamic_shared={dynamic_shared_bytes}: the launch cannot be resident on any SM \
             (contract ptx-codegen-safety-v1 register_budget postcondition)"
        )));
    }
    Ok(())
}
