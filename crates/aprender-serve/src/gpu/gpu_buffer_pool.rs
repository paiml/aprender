
impl GpuCompute {
    /// Create GPU compute context with auto-detected backend
    ///
    /// Attempts to initialize GPU backend, falls back to CPU if unavailable.
    ///
    /// # Errors
    ///
    /// Returns error if both GPU and CPU initialization fail (should not happen).
    pub fn auto() -> Result<Self> {
        Self::new(ComputeBackend::Auto)
    }

    /// Create GPU compute context with specified backend
    ///
    /// # Arguments
    ///
    /// * `backend` - Backend selection (Gpu, Cpu, or Auto)
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - `Gpu` backend requested but GPU is not available
    /// - Backend initialization fails
    pub fn new(backend: ComputeBackend) -> Result<Self> {
        match backend {
            ComputeBackend::Gpu => {
                if trueno::backends::gpu::GpuBackend::is_available() {
                    Ok(Self {
                        backend: ComputeBackend::Gpu,
                        gpu: Some(trueno::backends::gpu::GpuBackend::new()),
                    })
                } else {
                    Err(RealizarError::GpuError {
                        reason: "GPU not available".to_string(),
                    })
                }
            },
            ComputeBackend::Cpu => Ok(Self {
                backend: ComputeBackend::Cpu,
                gpu: None,
            }),
            ComputeBackend::Auto => {
                if trueno::backends::gpu::GpuBackend::is_available() {
                    Ok(Self {
                        backend: ComputeBackend::Gpu,
                        gpu: Some(trueno::backends::gpu::GpuBackend::new()),
                    })
                } else {
                    Ok(Self {
                        backend: ComputeBackend::Cpu,
                        gpu: None,
                    })
                }
            },
        }
    }

    /// Check if GPU backend is active
    #[must_use]
    pub fn is_gpu(&self) -> bool {
        self.backend == ComputeBackend::Gpu && self.gpu.is_some()
    }

    /// Get active backend type
    #[must_use]
    pub fn backend(&self) -> ComputeBackend {
        self.backend
    }

    /// GPU-accelerated matrix multiplication
    ///
    /// Computes `C = A @ B` where:
    /// - A is `[m, k]`
    /// - B is `[k, n]`
    /// - C is `[m, n]`
    ///
    /// # Arguments
    ///
    /// * `a` - Left matrix as flat f32 slice, row-major `[m, k]`
    /// * `b` - Right matrix as flat f32 slice, row-major `[k, n]`
    /// * `m` - Rows in A and C
    /// * `k` - Cols in A, rows in B
    /// * `n` - Cols in B and C
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Input dimensions don't match
    /// - GPU compute fails
    #[allow(clippy::many_single_char_names)]
    pub fn matmul(
        &mut self,
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>> {
        // Validate dimensions
        if a.len() != m * k {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "Matrix A size {} doesn't match m*k={}*{}={}",
                    a.len(),
                    m,
                    k,
                    m * k
                ),
            });
        }
        if b.len() != k * n {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "Matrix B size {} doesn't match k*n={}*{}={}",
                    b.len(),
                    k,
                    n,
                    k * n
                ),
            });
        }

        if let Some(gpu) = &mut self.gpu {
            // GPU path
            #[allow(clippy::implicit_clone)]
            gpu.matmul(a, b, m, k, n)
                .map_err(|e| RealizarError::GpuError {
                    reason: e.to_string(),
                })
        } else {
            // CPU fallback: naive matmul
            Ok(cpu_matmul(a, b, m, k, n))
        }
    }

    /// GPU-accelerated matrix multiplication with Tensor input/output
    ///
    /// # Arguments
    ///
    /// * `a` - Left tensor `[m, k]`
    /// * `b` - Right tensor `[k, n]`
    ///
    /// # Errors
    ///
    /// Returns error if tensors are not 2D or dimensions don't match.
    #[allow(clippy::many_single_char_names)]
    pub fn matmul_tensor(&mut self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>> {
        let a_shape = a.shape();
        let b_shape = b.shape();

        if a_shape.len() != 2 || b_shape.len() != 2 {
            return Err(RealizarError::InvalidShape {
                reason: "matmul_tensor requires 2D tensors".to_string(),
            });
        }

        let m = a_shape[0];
        let k = a_shape[1];
        let k2 = b_shape[0];
        let n = b_shape[1];

        if k != k2 {
            return Err(RealizarError::InvalidShape {
                reason: format!("Inner dimensions don't match: A[{m},{k}] @ B[{k2},{n}]"),
            });
        }

        let result = self.matmul(a.data(), b.data(), m, k, n)?;
        Tensor::from_vec(vec![m, n], result)
    }

    /// GPU-accelerated vector dot product
    ///
    /// # Errors
    ///
    /// Returns error if vectors have different lengths or GPU compute fails.
    pub fn dot(&mut self, a: &[f32], b: &[f32]) -> Result<f32> {
        if a.len() != b.len() {
            return Err(RealizarError::InvalidShape {
                reason: format!("Vector lengths don't match: {} vs {}", a.len(), b.len()),
            });
        }

        if let Some(gpu) = &mut self.gpu {
            #[allow(clippy::implicit_clone)]
            gpu.dot(a, b).map_err(|e| RealizarError::GpuError {
                reason: e.to_string(),
            })
        } else {
            // CPU fallback
            Ok(a.iter().zip(b.iter()).map(|(x, y)| x * y).sum())
        }
    }

    /// GPU-accelerated ReLU activation
    ///
    /// # Errors
    ///
    /// Returns error if GPU compute fails.
    pub fn relu(&mut self, input: &[f32]) -> Result<Vec<f32>> {
        if let Some(gpu) = &mut self.gpu {
            #[allow(clippy::implicit_clone)]
            gpu.relu(input).map_err(|e| RealizarError::GpuError {
                reason: e.to_string(),
            })
        } else {
            Ok(input.iter().map(|&x| x.max(0.0)).collect())
        }
    }

    /// GPU-accelerated sigmoid activation
    ///
    /// # Errors
    ///
    /// Returns error if GPU compute fails.
    pub fn sigmoid(&mut self, input: &[f32]) -> Result<Vec<f32>> {
        if let Some(gpu) = &mut self.gpu {
            #[allow(clippy::implicit_clone)]
            gpu.sigmoid(input).map_err(|e| RealizarError::GpuError {
                reason: e.to_string(),
            })
        } else {
            Ok(input.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect())
        }
    }
}

/// CPU fallback matmul implementation
#[allow(clippy::many_single_char_names)]
pub(crate) fn cpu_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    // For m=1 (vector-matrix multiply), use optimized path
    if m == 1 {
        return cpu_vector_matmul(a, b, k, n);
    }

    let mut c = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = sum;
        }
    }
    c
}

/// IMP-098: Parallelized vector-matrix multiply: a[1,k] @ b[k,n] -> c[1,n]
///
/// Uses parallel output chunks for multi-core utilization.
/// Each thread accumulates its chunk of outputs independently.
#[allow(clippy::many_single_char_names)]
fn cpu_vector_matmul(a: &[f32], b: &[f32], k: usize, n: usize) -> Vec<f32> {
    use rayon::prelude::*;

    // For small n, use sequential (avoids rayon overhead)
    if n < 2048 {
        return cpu_vector_matmul_seq(a, b, k, n);
    }

    // Parallel over output chunks
    const CHUNK_SIZE: usize = 1024;
    let num_chunks = n.div_ceil(CHUNK_SIZE);

    let chunks: Vec<Vec<f32>> = (0..num_chunks)
        .into_par_iter()
        .map(|chunk_idx| {
            let start = chunk_idx * CHUNK_SIZE;
            let end = (start + CHUNK_SIZE).min(n);
            let chunk_len = end - start;
            let mut chunk_c = vec![0.0f32; chunk_len];

            // Accumulate this chunk of outputs
            for (p, &a_val) in a.iter().enumerate() {
                let row_start = p * n + start;
                let row = &b[row_start..row_start + chunk_len];
                for (j, &b_val) in row.iter().enumerate() {
                    chunk_c[j] += a_val * b_val;
                }
            }
            chunk_c
        })
        .collect();

    // Flatten chunks into result
    chunks.into_iter().flatten().collect()
}

/// Sequential fallback for small outputs
#[allow(clippy::many_single_char_names)]
fn cpu_vector_matmul_seq(a: &[f32], b: &[f32], _k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; n];

    // Row-major accumulation: for each row of B, scale by corresponding a[p]
    for (p, &a_val) in a.iter().enumerate() {
        let row = &b[p * n..(p + 1) * n];
        for (j, &b_val) in row.iter().enumerate() {
            c[j] += a_val * b_val;
        }
    }

    c
}

/// CPU matmul with B transposed: A @ B^T
/// a[m,k] @ b[n,k]^T -> c[m,n]
#[allow(clippy::many_single_char_names)]
pub(crate) fn cpu_matmul_transpose_b(
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Vec<f32> {
    let mut c = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for p in 0..k {
                // a[i,p] * b[j,p] (b is stored row-major as [n,k])
                sum += a[i * k + p] * b[j * k + p];
            }
            c[i * n + j] = sum;
        }
    }
    c
}

/// Transpose a matrix from [rows, cols] to [cols, rows].
///
/// PMAT-285: Delegates to `contract_gate::transpose_f32` (single source of truth).
pub(crate) fn transpose(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    crate::contract_gate::transpose_f32(data, rows, cols)
}

/// IMP-096: Parallel SIMD vector-matrix multiply using transposed weights
///
/// Computes a[1,k] @ weight_t[n,k]^T + bias[n] -> c[n]
/// Each output c[j] = dot(a, weight_t[j,:]) + bias[j]
///
/// Uses transposed weights for row-major access pattern (contiguous dot products).
/// Parallelized with rayon. Compiler auto-vectorizes the inner dot product.
#[allow(clippy::many_single_char_names)]
pub(crate) fn cpu_matmul_transposed_simd(
    a: &[f32],        // Input vector: [k]
    weight_t: &[f32], // Transposed weights: [n, k] (row-major)
    bias: &[f32],     // Bias: [n]
    k: usize,
    n: usize,
) -> Vec<f32> {
    use rayon::prelude::*;

    // Process in chunks for better parallelism and cache locality
    const CHUNK_SIZE: usize = 4096;

    (0..n)
        .into_par_iter()
        .step_by(CHUNK_SIZE)
        .flat_map(|chunk_start| {
            let chunk_end = (chunk_start + CHUNK_SIZE).min(n);
            (chunk_start..chunk_end)
                .map(|j| {
                    // Row-major access: weight_t[j, :] is contiguous in memory
                    let row = &weight_t[j * k..(j + 1) * k];

                    // Compiler auto-vectorizes this dot product pattern
                    let dot: f32 = row.iter().zip(a.iter()).map(|(&w, &h)| w * h).sum();
                    dot + bias[j]
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

/// GPU buffer pool for memory reuse and reduced allocation overhead
pub struct GpuBufferPool {
    /// Available buffers indexed by size bucket
    available_buffers: std::collections::HashMap<usize, Vec<Vec<f32>>>,
    /// Size buckets for efficient pooling (powers of 2)
    bucket_sizes: Vec<usize>,
    /// Maximum cached buffers per bucket
    max_per_bucket: usize,
}
