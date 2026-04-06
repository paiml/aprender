#![allow(unsafe_code)]
#![allow(trivial_casts)]
#![allow(clippy::borrow_as_ptr)]
#![allow(clippy::ref_as_ptr)]

#[cfg(feature = "cuda")]
use trueno_gpu::driver::{CudaStream, GpuBuffer, LaunchConfig};
#[cfg(feature = "cuda")]
use trueno_gpu::kernels::backward::{GeluBackwardKernel, ReluBackwardKernel, SiluBackwardKernel};
#[cfg(feature = "cuda")]
use trueno_gpu::kernels::Kernel;

use super::super::cuda_tensor::{CudaTensorError, Result};
#[cfg(feature = "cuda")]
use super::cache::KERNEL_CACHE;

/// ReLU backward pass on GPU
///
/// Computes: grad_input = grad_output * (input > 0 ? 1 : 0)
///
/// # Arguments
/// * `input` - Original input to forward pass
/// * `grad_output` - Gradient from upstream
/// * `grad_input` - Output buffer for computed gradient
/// * `stream` - CUDA stream for async execution
#[cfg(feature = "cuda")]
pub fn relu_backward(
    input: &GpuBuffer<f32>,
    grad_output: &GpuBuffer<f32>,
    grad_input: &mut GpuBuffer<f32>,
    stream: &CudaStream,
) -> Result<()> {
    let n = input.len() as u32;

    let cache = KERNEL_CACHE.get().ok_or(CudaTensorError::DeviceNotInitialized)?;
    let mut cache = cache.lock().map_err(|_err| {
        CudaTensorError::KernelError("Failed to acquire kernel cache lock".to_string())
    })?;

    let module = match cache.get_cached("relu_backward") {
        Some(m) => m,
        None => {
            let kernel = ReluBackwardKernel::new(n);
            let ptx = kernel.emit_ptx_for_target(cache.sm_target());
            cache.get_or_compile("relu_backward", &ptx)?
        }
    };

    let config = LaunchConfig { grid: (n.div_ceil(256), 1, 1), block: (256, 1, 1), shared_mem: 0 };

    let input_ptr = input.as_ptr();
    let grad_out_ptr = grad_output.as_ptr();
    let grad_in_ptr = grad_input.as_ptr();

    let mut args: [*mut std::ffi::c_void; 4] = [
        &input_ptr as *const _ as *mut _,
        &grad_out_ptr as *const _ as *mut _,
        &grad_in_ptr as *const _ as *mut _,
        &n as *const _ as *mut _,
    ];

    // SAFETY: Kernel launch requires FFI. All buffers are valid GPU allocations with
    // matching sizes, and the kernel parameters match the expected PTX signature.
    unsafe {
        stream.launch_kernel(module, "relu_backward", &config, &mut args).map_err(|e| {
            CudaTensorError::KernelError(format!("ReLU backward launch failed: {e:?}"))
        })?;
    }

    Ok(())
}

/// GELU backward pass on GPU
///
/// Computes gradient using tanh approximation derivative
#[cfg(feature = "cuda")]
pub fn gelu_backward(
    input: &GpuBuffer<f32>,
    grad_output: &GpuBuffer<f32>,
    grad_input: &mut GpuBuffer<f32>,
    stream: &CudaStream,
) -> Result<()> {
    let n = input.len() as u32;

    let cache = KERNEL_CACHE.get().ok_or(CudaTensorError::DeviceNotInitialized)?;
    let mut cache = cache.lock().map_err(|_err| {
        CudaTensorError::KernelError("Failed to acquire kernel cache lock".to_string())
    })?;

    let module = match cache.get_cached("gelu_backward") {
        Some(m) => m,
        None => {
            let kernel = GeluBackwardKernel::new(n);
            let ptx = kernel.emit_ptx_for_target(cache.sm_target());
            cache.get_or_compile("gelu_backward", &ptx)?
        }
    };

    let config = LaunchConfig { grid: (n.div_ceil(256), 1, 1), block: (256, 1, 1), shared_mem: 0 };

    let input_ptr = input.as_ptr();
    let grad_out_ptr = grad_output.as_ptr();
    let grad_in_ptr = grad_input.as_ptr();

    let mut args: [*mut std::ffi::c_void; 4] = [
        &input_ptr as *const _ as *mut _,
        &grad_out_ptr as *const _ as *mut _,
        &grad_in_ptr as *const _ as *mut _,
        &n as *const _ as *mut _,
    ];

    // SAFETY: Kernel launch requires FFI. All buffers are valid GPU allocations with
    // matching sizes, and the kernel parameters match the expected PTX signature.
    unsafe {
        stream.launch_kernel(module, "gelu_backward", &config, &mut args).map_err(|e| {
            CudaTensorError::KernelError(format!("GELU backward launch failed: {e:?}"))
        })?;
    }

    Ok(())
}

/// SiLU/Swish backward pass on GPU
///
/// Computes: grad_input = grad_output * (sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x)))
#[cfg(feature = "cuda")]
pub fn silu_backward(
    input: &GpuBuffer<f32>,
    grad_output: &GpuBuffer<f32>,
    grad_input: &mut GpuBuffer<f32>,
    stream: &CudaStream,
) -> Result<()> {
    let n = input.len() as u32;

    let cache = KERNEL_CACHE.get().ok_or(CudaTensorError::DeviceNotInitialized)?;
    let mut cache = cache.lock().map_err(|_err| {
        CudaTensorError::KernelError("Failed to acquire kernel cache lock".to_string())
    })?;

    let module = match cache.get_cached("silu_backward") {
        Some(m) => m,
        None => {
            let kernel = SiluBackwardKernel::new(n);
            let ptx = kernel.emit_ptx_for_target(cache.sm_target());
            cache.get_or_compile("silu_backward", &ptx)?
        }
    };

    let config = LaunchConfig { grid: (n.div_ceil(256), 1, 1), block: (256, 1, 1), shared_mem: 0 };

    let input_ptr = input.as_ptr();
    let grad_out_ptr = grad_output.as_ptr();
    let grad_in_ptr = grad_input.as_ptr();

    let mut args: [*mut std::ffi::c_void; 4] = [
        &input_ptr as *const _ as *mut _,
        &grad_out_ptr as *const _ as *mut _,
        &grad_in_ptr as *const _ as *mut _,
        &n as *const _ as *mut _,
    ];

    // SAFETY: Kernel launch requires FFI. All buffers are valid GPU allocations with
    // matching sizes, and the kernel parameters match the expected PTX signature.
    unsafe {
        stream.launch_kernel(module, "silu_backward", &config, &mut args).map_err(|e| {
            CudaTensorError::KernelError(format!("SiLU backward launch failed: {e:?}"))
        })?;
    }

    Ok(())
}
