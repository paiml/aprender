//! CUDA-accelerated training utilities
//!
//! This module provides high-level training primitives that use CUDA kernels
//! when available, with automatic CPU fallback.
//!
//! # Architecture (SPEC-FT-001 v3.2.0)
//!
//! ```text
//! CudaTrainer
//!   ├── device: CudaDevice
//!   ├── forward: gemm_forward kernel
//!   ├── backward: gemm_backward_a/b kernels
//!   └── optimizer: adamw_step_cuda kernel
//! ```
//!
//! # Example
//!
//! ```ignore
//! use entrenar::autograd::cuda_training::CudaTrainer;
//!
//! let trainer = CudaTrainer::new()?;
//! let logits = trainer.matmul_forward(&hidden, &weights, m, k, n)?;
//! trainer.adamw_step(&mut weights, &grads, lr, step)?;
//! ```

#[cfg(feature = "cuda")]
use std::sync::Arc;

#[cfg(feature = "cuda")]
use trueno_gpu::driver::{cuda_available, CudaContext, CudaStream, GpuBuffer};

use super::cuda_tensor::{CudaTensorError, Result};
#[cfg(feature = "cuda")]
use provable_contracts_macros::requires;

#[cfg(feature = "cuda")]
use super::cuda_backward::{gemm_backward_a, gemm_backward_b, init_kernel_cache};
#[cfg(feature = "cuda")]
use super::cuda_forward::{gemm_forward, init_forward_kernel_cache};
#[cfg(feature = "cuda")]
use super::cuda_optim::{adamw_step_cuda, gradient_clip_cuda, init_optim_kernel_cache};

/// CUDA-accelerated training context
///
/// Manages GPU resources and provides high-level training operations.
#[cfg(feature = "cuda")]
pub struct CudaTrainer {
    ctx: Arc<CudaContext>,
    stream: CudaStream,
    step: u32,
}

#[cfg(feature = "cuda")]
impl CudaTrainer {
    /// Create a new CUDA trainer on the default GPU
    pub fn new() -> Result<Self> {
        Self::with_device(0)
    }

    /// Create a new CUDA trainer on the specified GPU
    pub fn with_device(device_id: i32) -> Result<Self> {
        if !cuda_available() {
            return Err(CudaTensorError::CudaNotAvailable("No CUDA driver found".into()));
        }

        let ctx = Arc::new(
            CudaContext::new(device_id)
                .map_err(|e| CudaTensorError::CudaNotAvailable(format!("{e:?}")))?,
        );
        let stream = CudaStream::new(&ctx)
            .map_err(|e| CudaTensorError::AllocationFailed(format!("{e:?}")))?;

        // Initialize all kernel caches
        init_forward_kernel_cache(ctx.clone())?;
        init_kernel_cache(ctx.clone())?;
        init_optim_kernel_cache(ctx.clone())?;

        Ok(Self { ctx, stream, step: 0 })
    }

    /// Get the CUDA context
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    /// Get the CUDA stream
    pub fn stream(&self) -> &CudaStream {
        &self.stream
    }

    /// Synchronize the stream (wait for all operations to complete)
    pub fn synchronize(&self) -> Result<()> {
        self.stream.synchronize().map_err(|e| CudaTensorError::KernelError(format!("{e:?}")))
    }

    /// Allocate a GPU buffer from host data
    pub fn upload(&self, data: &[f32]) -> Result<GpuBuffer<f32>> {
        let mut buf = GpuBuffer::from_host(&self.ctx, data)
            .map_err(|e| CudaTensorError::AllocationFailed(format!("{e:?}")))?;
        // PMAT-420: Set context for thread-safe transfers
        buf.set_context(&self.ctx);
        Ok(buf)
    }

    /// Allocate a zero-initialized GPU buffer
    pub fn zeros(&self, len: usize) -> Result<GpuBuffer<f32>> {
        let data = vec![0.0f32; len];
        self.upload(&data)
    }

    /// Query free VRAM in MB (via cuMemGetInfo).
    /// Returns None if query fails.
    pub fn free_memory_mb(&self) -> Option<u64> {
        self.ctx.memory_info().map(|(free, _total)| (free / (1024 * 1024)) as u64).ok()
    }

    /// Download GPU buffer to host
    pub fn download(&self, buffer: &GpuBuffer<f32>) -> Result<Vec<f32>> {
        let mut result = vec![0.0f32; buffer.len()];
        buffer
            .copy_to_host(&mut result)
            .map_err(|e| CudaTensorError::TransferFailed(format!("{e:?}")))?;
        Ok(result)
    }

    /// Matrix multiply forward pass: C = A @ B
    ///
    /// # Arguments
    /// - `a`: Input matrix (m × k)
    /// - `b`: Weight matrix (k × n)
    /// - `c`: Output matrix (m × n)
    /// - `m`, `k`, `n`: Matrix dimensions
    pub fn matmul_forward(
        &self,
        a: &GpuBuffer<f32>,
        b: &GpuBuffer<f32>,
        c: &mut GpuBuffer<f32>,
        m: u32,
        k: u32,
        n: u32,
    ) -> Result<()> {
        gemm_forward(a, b, c, m, k, n, &self.stream)
    }

    /// Matrix multiply backward pass for weight gradients
    ///
    /// Given C = A @ B, computes:
    /// - grad_A = grad_C @ B^T
    /// - grad_B = A^T @ grad_C
    // Contract: backward-pass-v1 / matmul_backward
    #[requires(m > 0 && k > 0 && n > 0)]
    pub fn matmul_backward(
        &self,
        a: &GpuBuffer<f32>,
        b: &GpuBuffer<f32>,
        grad_c: &GpuBuffer<f32>,
        grad_a: &mut GpuBuffer<f32>,
        grad_b: &mut GpuBuffer<f32>,
        m: u32,
        k: u32,
        n: u32,
    ) -> Result<()> {
        gemm_backward_a(grad_c, b, grad_a, m, k, n, &self.stream)?;
        gemm_backward_b(a, grad_c, grad_b, m, k, n, &self.stream)?;
        Ok(())
    }

    /// AdamW optimizer step on GPU
    ///
    /// Updates weights in-place using the AdamW algorithm.
    pub fn adamw_step(
        &mut self,
        params: &mut GpuBuffer<f32>,
        grads: &GpuBuffer<f32>,
        m_state: &mut GpuBuffer<f32>,
        v_state: &mut GpuBuffer<f32>,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
    ) -> Result<()> {
        self.step += 1;
        let n = params.len() as u32;
        adamw_step_cuda(
            params,
            grads,
            m_state,
            v_state,
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            self.step,
            n,
            &self.stream,
        )
    }

    /// Apply gradient clipping
    pub fn clip_gradients(&self, grads: &mut GpuBuffer<f32>, max_norm: f32) -> Result<()> {
        // Compute gradient norm on CPU (requires download)
        let grad_data = self.download(grads)?;
        let grad_norm: f32 = grad_data.iter().map(|x| x * x).sum::<f32>().sqrt();

        // Compute scale factor
        let scale = if grad_norm > max_norm { max_norm / grad_norm } else { 1.0 };

        // Apply clipping on GPU
        gradient_clip_cuda(grads, scale, grads.len() as u32, &self.stream)
    }

    /// Get current optimizer step count
    pub fn step_count(&self) -> u32 {
        self.step
    }

    /// Reset optimizer step count (for new training run)
    pub fn reset_step(&mut self) {
        self.step = 0;
    }

    /// Get device name
    pub fn device_name(&self) -> String {
        self.ctx.device_name().unwrap_or_else(|_err| "Unknown GPU".to_string())
    }

    /// Get total GPU memory in bytes
    pub fn total_memory(&self) -> usize {
        self.ctx.total_memory().unwrap_or(0)
    }
}

#[cfg(feature = "cuda")]
#[allow(clippy::missing_fields_in_debug)]
impl std::fmt::Debug for CudaTrainer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaTrainer")
            .field("device", &self.device_name())
            .field("memory_gb", &(self.total_memory() as f64 / 1e9))
            .field("step", &self.step)
            .finish()
    }
}

// CPU fallback when CUDA is not available
#[cfg(not(feature = "cuda"))]
pub struct CudaTrainer;

#[cfg(not(feature = "cuda"))]
impl CudaTrainer {
    pub fn new() -> Result<Self> {
        Err(CudaTensorError::CudaNotAvailable("Compiled without CUDA support".into()))
    }
}

/// Check if CUDA training is available
pub fn cuda_training_available() -> bool {
    #[cfg(feature = "cuda")]
    {
        trueno_gpu::driver::cuda_available()
    }
    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_training_available() {
        // Just verify the function compiles and runs
        let _ = cuda_training_available();
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_cuda_trainer_creation() {
        if !cuda_training_available() {
            return;
        }

        let trainer = CudaTrainer::new();
        assert!(trainer.is_ok());

        let trainer = trainer.expect("operation should succeed");
        assert!(!trainer.device_name().is_empty());
        assert!(trainer.total_memory() > 0);
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_cuda_trainer_upload_download() {
        if !cuda_training_available() {
            return;
        }

        let trainer = CudaTrainer::new().expect("operation should succeed");
        let data = vec![1.0, 2.0, 3.0, 4.0];

        let gpu_buffer = trainer.upload(&data).expect("load should succeed");
        let result = trainer.download(&gpu_buffer).expect("load should succeed");

        assert_eq!(data, result);
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_cuda_trainer_zeros() {
        if !cuda_training_available() {
            return;
        }

        let trainer = CudaTrainer::new().expect("operation should succeed");
        let gpu_buffer = trainer.zeros(100).expect("operation should succeed");
        let result = trainer.download(&gpu_buffer).expect("load should succeed");

        assert_eq!(result.len(), 100);
        assert!(result.iter().all(|&x| x == 0.0));
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_cuda_trainer_synchronize() {
        if !cuda_training_available() {
            return;
        }

        let trainer = CudaTrainer::new().expect("operation should succeed");
        // Synchronize should succeed
        assert!(trainer.synchronize().is_ok());
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_cuda_trainer_context_and_stream() {
        if !cuda_training_available() {
            return;
        }

        let trainer = CudaTrainer::new().expect("operation should succeed");
        // Accessing context and stream should not panic
        let _ctx = trainer.context();
        let _stream = trainer.stream();
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_cuda_trainer_step_count() {
        if !cuda_training_available() {
            return;
        }

        let mut trainer = CudaTrainer::new().expect("operation should succeed");
        assert_eq!(trainer.step_count(), 0);

        // Simulate an optimizer step by calling adamw_step
        let mut params = trainer.upload(&[1.0, 2.0, 3.0]).expect("load should succeed");
        let grads = trainer.upload(&[0.1, 0.1, 0.1]).expect("load should succeed");
        let mut m_state = trainer.zeros(3).expect("operation should succeed");
        let mut v_state = trainer.zeros(3).expect("operation should succeed");

        trainer
            .adamw_step(
                &mut params,
                &grads,
                &mut m_state,
                &mut v_state,
                0.001,
                0.9,
                0.999,
                1e-8,
                0.0,
            )
            .expect("operation should succeed");

        assert_eq!(trainer.step_count(), 1);

        trainer.reset_step();
        assert_eq!(trainer.step_count(), 0);
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_cuda_trainer_matmul_forward() {
        if !cuda_training_available() {
            return;
        }

        let trainer = CudaTrainer::new().expect("operation should succeed");

        // 2x3 @ 3x2 = 2x2
        let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
        let b_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 3x2
        let c_data: Vec<f32> = vec![0.0; 4]; // 2x2

        let a = trainer.upload(&a_data).expect("load should succeed");
        let b = trainer.upload(&b_data).expect("load should succeed");
        let mut c = trainer.upload(&c_data).expect("load should succeed");

        trainer.matmul_forward(&a, &b, &mut c, 2, 3, 2).expect("operation should succeed");
        trainer.synchronize().expect("operation should succeed");

        let result = trainer.download(&c).expect("load should succeed");
        // Verify result is not all zeros (matmul should produce non-zero output)
        assert!(!result.iter().all(|&x| x == 0.0));
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_cuda_trainer_clip_gradients() {
        if !cuda_training_available() {
            return;
        }

        let trainer = CudaTrainer::new().expect("operation should succeed");

        // Create large gradients that should be clipped
        let grad_data: Vec<f32> = vec![10.0, 10.0, 10.0, 10.0]; // norm = 20
        let mut grads = trainer.upload(&grad_data).expect("load should succeed");

        // Clip to max_norm = 1.0
        trainer.clip_gradients(&mut grads, 1.0).expect("operation should succeed");
        trainer.synchronize().expect("operation should succeed");

        let result = trainer.download(&grads).expect("load should succeed");
        // Gradients should be scaled down
        let norm: f32 = result.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(norm <= 1.1, "Gradient norm should be clipped to ~1.0, got {norm}");
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_cuda_trainer_debug_impl() {
        if !cuda_training_available() {
            return;
        }

        let trainer = CudaTrainer::new().expect("operation should succeed");
        let debug_str = format!("{trainer:?}");
        assert!(debug_str.contains("CudaTrainer"));
        assert!(debug_str.contains("device"));
        assert!(debug_str.contains("step"));
    }
}
