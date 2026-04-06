//! CUDA Scheduler Core (PMAT-802)
//!
//! Core CudaScheduler implementation for direct CUDA execution.

#[cfg(feature = "cuda")]
#[allow(unused_imports)]
use crate::cuda::CudaExecutor;
#[cfg(feature = "cuda")]
#[allow(unused_imports)]
use crate::error::{RealizarError, Result};

// Unlike HybridScheduler, this ALWAYS uses CUDA (even for m=1)
// ============================================================================

/// CUDA-native scheduler for GpuModel inference
///
/// Key difference from HybridScheduler:
/// - ALWAYS uses CudaExecutor for matmul (no CPU fallback for m=1)
/// - Direct CUDA kernel execution instead of trueno wgpu
/// - Eliminates the m=1 CPU restriction that causes 1090x slowdown
///
/// ## Performance Impact
///
/// The HybridScheduler forces CPU for m=1 (single-token generation), which
/// causes token-by-token inference to run on CPU. CudaScheduler eliminates
/// this restriction, enabling full GPU acceleration for generation.
#[cfg(feature = "cuda")]
pub struct CudaScheduler {
    executor: crate::cuda::CudaExecutor,
}

#[cfg(feature = "cuda")]
impl CudaScheduler {
    /// Create a new CUDA scheduler
    ///
    /// # Errors
    ///
    /// Returns error if CUDA is not available or initialization fails.
    pub fn new() -> Result<Self> {
        let executor = crate::cuda::CudaExecutor::new(0).map_err(|e| RealizarError::GpuError {
            reason: format!("Failed to create CudaExecutor: {}", e),
        })?;

        Ok(Self { executor })
    }

    /// Check if CUDA is available
    #[must_use]
    pub fn has_cuda(&self) -> bool {
        true // If we have a CudaScheduler, CUDA is available
    }

    /// Check if this scheduler uses CUDA for the given matrix dimensions
    ///
    /// Unlike HybridScheduler, CudaScheduler ALWAYS returns true (uses CUDA).
    #[must_use]
    #[allow(clippy::unused_self)]
    pub fn uses_cuda_for(&self, _m: usize, _k: usize, _n: usize) -> bool {
        true // Always use CUDA - this is the key difference from HybridScheduler
    }

    /// Execute matmul using CUDA
    ///
    /// Same interface as HybridScheduler::matmul for drop-in replacement.
    ///
    /// # Errors
    ///
    /// Returns error if CUDA execution fails.
    #[allow(clippy::many_single_char_names)]
    pub fn matmul(
        &mut self,
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>> {
        let mut output = vec![0.0f32; m * n];

        self.executor
            .gemm(a, b, &mut output, m as u32, n as u32, k as u32)
            .map_err(|e| RealizarError::GpuError {
                reason: format!("CUDA GEMM failed: {}", e),
            })?;

        Ok(output)
    }

    /// Get device name
    pub fn device_name(&self) -> Result<String> {
        self.executor
            .device_name()
            .map_err(|e| RealizarError::GpuError {
                reason: format!("Failed to get device name: {}", e),
            })
    }

    /// Cache a weight matrix on GPU (PARITY-120: 10x speedup)
    ///
    /// Weights stay on GPU and are reused for all forward passes.
    ///
    /// # Errors
    ///
    /// Returns error if GPU memory allocation fails.
    pub fn cache_weight(&mut self, name: &str, weight: &[f32]) -> Result<()> {
        self.executor
            .load_weights(name, weight)
            .map(|_| ())
            .map_err(|e| RealizarError::GpuError {
                reason: format!("Failed to cache weight '{}': {}", name, e),
            })
    }

    /// Check if a weight is cached
    #[must_use]
    pub fn has_cached_weight(&self, name: &str) -> bool {
        self.executor.has_weights(name)
    }

    /// Get number of cached weights
    #[must_use]
    pub fn cached_weight_count(&self) -> usize {
        self.executor.cached_weight_count()
    }

    /// Execute matmul using cached weight (PARITY-120: 10x speedup)
    ///
    /// Uses pre-loaded weight on GPU, only transfers input/output.
    /// This is the fast path for single-token generation.
    ///
    /// # Arguments
    ///
    /// * `weight_name` - Name of cached weight (from `cache_weight`)
    /// * `x` - Input vector (k elements)
    /// * `k` - Input dimension
    /// * `n` - Output dimension
    ///
    /// # Errors
    ///
    /// Returns error if weight not cached or CUDA fails.
    pub fn matmul_cached(
        &mut self,
        weight_name: &str,
        x: &[f32],
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>> {
        let mut output = vec![0.0f32; n];

        self.executor
            .gemv_cached(weight_name, x, &mut output, k as u32, n as u32)
            .map_err(|e| RealizarError::GpuError {
                reason: format!("CUDA cached GEMV failed: {}", e),
            })?;

        Ok(output)
    }
}

#[cfg(test)]
#[cfg(feature = "cuda")]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_scheduler_new() {
        // CudaScheduler::new() should succeed on RTX 4090
        let scheduler = CudaScheduler::new();
        assert!(
            scheduler.is_ok(),
            "CudaScheduler::new() failed: {:?}",
            scheduler.err()
        );
    }

    #[test]
    fn test_cuda_scheduler_has_cuda() {
        let scheduler = CudaScheduler::new().expect("scheduler");
        assert!(scheduler.has_cuda());
    }

    #[test]
    fn test_cuda_scheduler_uses_cuda_for_all_dims() {
        let scheduler = CudaScheduler::new().expect("scheduler");
        // CudaScheduler ALWAYS uses CUDA, unlike HybridScheduler
        assert!(scheduler.uses_cuda_for(1, 64, 64)); // m=1 (single token)
        assert!(scheduler.uses_cuda_for(8, 256, 256)); // batch
        assert!(scheduler.uses_cuda_for(1, 1, 1)); // tiny
        assert!(scheduler.uses_cuda_for(1024, 4096, 4096)); // large
    }

    #[test]
    fn test_cuda_scheduler_device_name() {
        let scheduler = CudaScheduler::new().expect("scheduler");
        let name = scheduler.device_name();
        assert!(name.is_ok());
        let name = name.expect("name");
        // Should contain GPU name
        assert!(!name.is_empty());
    }

    #[test]
    fn test_cuda_scheduler_matmul_basic() {
        let mut scheduler = CudaScheduler::new().expect("scheduler");

        // 2x3 @ 3x2 = 2x2
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let result = scheduler.matmul(&a, &b, 2, 3, 2);
        assert!(result.is_ok());
        let output = result.expect("output");
        assert_eq!(output.len(), 4);
    }

    #[test]
    fn test_cuda_scheduler_matmul_single_element() {
        let mut scheduler = CudaScheduler::new().expect("scheduler");

        // 1x1 @ 1x1 = 1x1
        let a = vec![3.0];
        let b = vec![4.0];
        let result = scheduler.matmul(&a, &b, 1, 1, 1);
        assert!(result.is_ok());
        let output = result.expect("output");
        assert_eq!(output.len(), 1);
        // 3 * 4 = 12
        assert!((output[0] - 12.0).abs() < 0.1);
    }

    #[test]
    fn test_cuda_scheduler_matmul_larger() {
        let mut scheduler = CudaScheduler::new().expect("scheduler");

        // 4x64 @ 64x32 = 4x32
        let a: Vec<f32> = (0..256).map(|i| (i as f32) * 0.01).collect();
        let b: Vec<f32> = (0..2048).map(|i| (i as f32) * 0.001).collect();
        let result = scheduler.matmul(&a, &b, 4, 64, 32);
        assert!(result.is_ok());
        let output = result.expect("output");
        assert_eq!(output.len(), 128);
    }

    #[test]
    fn test_cuda_scheduler_cache_weight() {
        let mut scheduler = CudaScheduler::new().expect("scheduler");

        let weight = vec![1.0f32; 256 * 128];
        let result = scheduler.cache_weight("test_weight", &weight);
        assert!(result.is_ok());
        assert!(scheduler.has_cached_weight("test_weight"));
        assert!(!scheduler.has_cached_weight("nonexistent"));
    }

    #[test]
    fn test_cuda_scheduler_cached_weight_count() {
        let mut scheduler = CudaScheduler::new().expect("scheduler");

        let initial_count = scheduler.cached_weight_count();
        let weight = vec![1.0f32; 64 * 64];
        scheduler
            .cache_weight("weight_1", &weight)
            .expect("cache_weight");
        assert_eq!(scheduler.cached_weight_count(), initial_count + 1);

        scheduler
            .cache_weight("weight_2", &weight)
            .expect("cache_weight");
        assert_eq!(scheduler.cached_weight_count(), initial_count + 2);
    }

    #[test]
    fn test_cuda_scheduler_matmul_cached() {
        let mut scheduler = CudaScheduler::new().expect("scheduler");

        // Cache a 64x32 weight matrix
        let weight: Vec<f32> = (0..2048).map(|i| (i as f32) * 0.001).collect();
        scheduler
            .cache_weight("cached_test", &weight)
            .expect("cache_weight");

        // GEMV: 1x64 @ 64x32 = 1x32
        let input: Vec<f32> = (0..64).map(|i| (i as f32) * 0.1).collect();
        let result = scheduler.matmul_cached("cached_test", &input, 64, 32);
        assert!(result.is_ok());
        let output = result.expect("output");
        assert_eq!(output.len(), 32);
    }

    #[test]
    fn test_cuda_scheduler_matmul_identity() {
        let mut scheduler = CudaScheduler::new().expect("scheduler");

        // Identity matrix test: I @ v = v
        // 4x4 identity @ 4x1 = 4x1
        #[rustfmt::skip]
        let identity = vec![
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ];
        let v = vec![1.0, 2.0, 3.0, 4.0];
        let result = scheduler.matmul(&identity, &v, 4, 4, 1);
        assert!(result.is_ok());
        let output = result.expect("output");
        for (i, &expected) in v.iter().enumerate() {
            assert!(
                (output[i] - expected).abs() < 0.01,
                "idx={} got={} expected={}",
                i,
                output[i],
                expected
            );
        }
    }
}
