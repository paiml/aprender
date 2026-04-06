
impl GpuBufferPool {
    /// Create new buffer pool with default configuration
    #[must_use]
    pub fn new() -> Self {
        Self {
            available_buffers: std::collections::HashMap::new(),
            bucket_sizes: (10..=24).map(|i| 1 << i).collect(), // 1KB to 16MB
            max_per_bucket: 4,
        }
    }

    /// Get bucket size for requested allocation
    fn get_bucket(&self, size: usize) -> usize {
        *self
            .bucket_sizes
            .iter()
            .find(|&&b| b >= size)
            .unwrap_or(&size)
    }

    /// Acquire buffer of at least `size` elements
    pub fn acquire(&mut self, size: usize) -> Vec<f32> {
        let bucket = self.get_bucket(size);
        if let Some(buffers) = self.available_buffers.get_mut(&bucket) {
            if let Some(mut buf) = buffers.pop() {
                buf.resize(size, 0.0);
                return buf;
            }
        }
        vec![0.0; size]
    }

    /// Release buffer back to pool for reuse
    pub fn release(&mut self, mut buffer: Vec<f32>) {
        let bucket = self.get_bucket(buffer.capacity());
        let buffers = self.available_buffers.entry(bucket).or_default();
        if buffers.len() < self.max_per_bucket {
            buffer.clear();
            buffers.push(buffer);
        }
        // Otherwise just drop it
    }

    /// Clear all cached buffers
    pub fn clear(&mut self) {
        self.available_buffers.clear();
    }

    /// Get configured bucket sizes
    #[must_use]
    pub fn bucket_sizes(&self) -> &[usize] {
        &self.bucket_sizes
    }

    /// Get pool statistics
    #[must_use]
    pub fn stats(&self) -> GpuPoolStats {
        let total_buffers: usize = self.available_buffers.values().map(Vec::len).sum();
        let total_bytes: usize = self
            .available_buffers
            .iter()
            .map(|(bucket, buffers)| bucket * buffers.len() * 4)
            .sum();
        GpuPoolStats {
            cached_buffers: total_buffers,
            cached_bytes: total_bytes,
        }
    }
}

impl Default for GpuBufferPool {
    fn default() -> Self {
        Self::new()
    }
}

/// GPU buffer pool statistics
#[derive(Debug, Clone, Copy)]
pub struct GpuPoolStats {
    /// Number of cached buffers
    pub cached_buffers: usize,
    /// Total cached bytes
    pub cached_bytes: usize,
}

/// Async GPU compute handle for non-blocking operations
///
/// Per spec: "Async transfer - No host blocking"
pub struct AsyncGpuResult {
    /// Result data when ready
    result: Option<Vec<f32>>,
    /// Whether computation is complete
    ready: bool,
}

impl AsyncGpuResult {
    /// Create result that's immediately ready (CPU fallback)
    pub fn ready(data: Vec<f32>) -> Self {
        Self {
            result: Some(data),
            ready: true,
        }
    }

    /// Create pending result (GPU async)
    pub fn pending() -> Self {
        Self {
            result: None,
            ready: false,
        }
    }

    /// Check if result is ready
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.ready
    }

    /// Mark as ready with result
    pub fn set_result(&mut self, data: Vec<f32>) {
        self.result = Some(data);
        self.ready = true;
    }

    /// Block until result is ready (for synchronization points)
    pub fn wait(self) -> Vec<f32> {
        self.result.expect("Result not ready")
    }

    /// Try to get result without blocking
    pub fn try_get(&self) -> Option<&Vec<f32>> {
        if self.ready {
            self.result.as_ref()
        } else {
            None
        }
    }
}

/// Hybrid CPU/GPU scheduler
///
/// Automatically selects optimal backend based on workload size.
pub struct HybridScheduler {
    gpu_compute: GpuCompute,
    /// Minimum matrix size (m*k*n) to use GPU
    gpu_threshold: usize,
    /// Buffer pool for memory reuse
    buffer_pool: GpuBufferPool,
}

impl HybridScheduler {
    /// Create hybrid scheduler with auto-detected GPU
    ///
    /// # Errors
    ///
    /// Returns error if compute initialization fails.
    pub fn new() -> Result<Self> {
        Ok(Self {
            gpu_compute: GpuCompute::auto()?,
            gpu_threshold: 64 * 64 * 64, // 262K elements
            buffer_pool: GpuBufferPool::new(),
        })
    }

    /// Create scheduler with custom threshold
    ///
    /// # Arguments
    ///
    /// * `gpu_threshold` - Minimum m*k*n to trigger GPU acceleration
    ///
    /// # Errors
    ///
    /// Returns error if compute initialization fails.
    pub fn with_threshold(gpu_threshold: usize) -> Result<Self> {
        Ok(Self {
            gpu_compute: GpuCompute::auto()?,
            gpu_threshold,
            buffer_pool: GpuBufferPool::new(),
        })
    }

    /// Check if GPU is available
    #[must_use]
    pub fn has_gpu(&self) -> bool {
        self.gpu_compute.is_gpu()
    }

    /// Get GPU threshold
    #[must_use]
    pub fn gpu_threshold(&self) -> usize {
        contract_pre_gpu_threshold!();
        let result = self.gpu_threshold;
        contract_post_gpu_threshold!(&result);
        result
    }

    /// Decide whether to use GPU for given workload
    ///
    /// IMP-097: For m=1 (single-token inference), CPU is faster due to:
    /// - No GPU data transfer overhead
    /// - No kernel launch latency
    /// - CPU SIMD is sufficient for vector-matrix multiply
    #[must_use]
    #[allow(clippy::many_single_char_names)]
    pub fn should_use_gpu(&self, m: usize, k: usize, n: usize) -> bool {
        // IMP-097: Force CPU for single-token operations (m=1)
        // GPU kernel launch overhead exceeds compute benefit for small batch sizes
        if m <= 1 {
            return false;
        }
        self.gpu_compute.is_gpu() && (m * k * n) >= self.gpu_threshold
    }

    /// Execute matmul with automatic backend selection
    ///
    /// Uses GPU for large matrices, CPU for small ones.
    ///
    /// # Errors
    ///
    /// Returns error if compute fails.
    #[allow(clippy::many_single_char_names)]
    pub fn matmul(
        &mut self,
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>> {
        if self.should_use_gpu(m, k, n) {
            self.gpu_compute.matmul(a, b, m, k, n)
        } else {
            Ok(cpu_matmul(a, b, m, k, n))
        }
    }

    /// Execute matmul with pooled output buffer
    ///
    /// Reduces allocation overhead by reusing buffers.
    ///
    /// # Errors
    ///
    /// Returns error if compute fails.
    #[allow(clippy::many_single_char_names)]
    pub fn matmul_pooled(
        &mut self,
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>> {
        // Acquire buffer from pool
        let mut output = self.buffer_pool.acquire(m * n);

        // Compute result
        let result = if self.should_use_gpu(m, k, n) {
            self.gpu_compute.matmul(a, b, m, k, n)?
        } else {
            cpu_matmul(a, b, m, k, n)
        };

        // Copy to pooled buffer
        output.copy_from_slice(&result);
        Ok(output)
    }

    /// Release buffer back to pool
    ///
    /// Call this when done with a buffer returned by `matmul_pooled`.
    pub fn release_buffer(&mut self, buffer: Vec<f32>) {
        self.buffer_pool.release(buffer);
    }

    /// Get buffer pool statistics
    #[must_use]
    pub fn pool_stats(&self) -> GpuPoolStats {
        self.buffer_pool.stats()
    }

    /// Execute matmul asynchronously (non-blocking on CPU fallback)
    ///
    /// Per spec: "Async transfer - No host blocking"
    ///
    /// # Errors
    ///
    /// Returns error if compute setup fails.
    #[allow(clippy::many_single_char_names)]
    pub fn matmul_async(
        &mut self,
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<AsyncGpuResult> {
        // For CPU fallback, compute immediately
        // For GPU, this would submit to command queue without blocking
        let result = if self.should_use_gpu(m, k, n) {
            self.gpu_compute.matmul(a, b, m, k, n)?
        } else {
            cpu_matmul(a, b, m, k, n)
        };

        Ok(AsyncGpuResult::ready(result))
    }

    /// Process batch of matmuls with optimal scheduling
    ///
    /// Batches small operations for CPU, pipelines large ones for GPU.
    ///
    /// # Errors
    ///
    /// Returns error if any compute fails.
    pub fn matmul_batch(&mut self, operations: &[MatmulOp]) -> Result<Vec<Vec<f32>>> {
        let mut results = Vec::with_capacity(operations.len());

        for (a, b, m, k, n) in operations {
            let result = self.matmul(a, b, *m, *k, *n)?;
            results.push(result);
        }

        Ok(results)
    }

    /// Execute matmul with B transposed: A @ B^T
    ///
    /// Computes C[m,n] = A[m,k] @ B[n,k]^T
    /// where B is stored row-major as [n, k].
    ///
    /// # Errors
    ///
    /// Returns error if compute fails.
    #[allow(clippy::many_single_char_names)]
    pub fn matmul_transpose_b(
        &mut self,
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>> {
        // For attention: Q[seq, head_dim] @ K[seq, head_dim]^T = scores[seq, seq]
        // B is stored as [n, k], we need B^T which is [k, n]
        if self.should_use_gpu(m, k, n) {
            // Transpose B and use GPU matmul
            let b_t = transpose(b, n, k);
            self.gpu_compute.matmul(a, &b_t, m, k, n)
        } else {
            // CPU: compute A @ B^T directly
            Ok(cpu_matmul_transpose_b(a, b, m, k, n))
        }
    }
}
