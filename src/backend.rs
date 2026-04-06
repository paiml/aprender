/// Backend selection and cost model (per spec section 2.2)
///
/// # Mixture-of-Experts (MoE) Routing
///
/// This module implements adaptive backend selection using a Mixture-of-Experts approach.
/// The MoE router analyzes operation complexity and data size to select the optimal
/// compute backend (Scalar/SIMD/GPU).
///
/// ## Operation Complexity Levels
///
/// - **Low**: Element-wise operations (add, multiply, etc.) - Memory-bound, GPU rarely beneficial
/// - **Medium**: Reductions (dot product, sum, etc.) - Moderate compute, GPU at 100K+ elements
/// - **High**: Matrix operations (matmul, convolution) - Compute-intensive O(n²) or O(n³), GPU at 10K+ elements
///
/// ## Usage Example
///
/// ```rust
/// use batuta::backend::{BackendSelector, OpComplexity};
///
/// let selector = BackendSelector::new();
///
/// // Element-wise operation
/// let backend = selector.select_with_moe(OpComplexity::Low, 500_000);
/// // Returns: Scalar (below 1M threshold, memory-bound)
///
/// // Matrix multiplication
/// let backend = selector.select_with_moe(OpComplexity::High, 50_000);
/// // Returns: GPU (above 10K threshold for O(n²) ops)
/// ```
///
/// ## Performance Thresholds
///
/// Based on empirical analysis and the 5× PCIe rule (Gregg & Hazelwood 2011):
///
/// | Complexity | SIMD Threshold | GPU Threshold | Rationale |
/// |------------|---------------|---------------|-----------|
/// | Low | 1M elements | Never | Memory-bound, PCIe overhead dominates |
/// | Medium | 10K elements | 100K elements | Moderate compute/transfer ratio |
/// | High | 1K elements | 10K elements | O(n²/n³) complexity favors GPU |
///
use serde::{Deserialize, Serialize};

#[cfg(feature = "trueno-integration")]
use trueno::{Matrix, Vector};

/// Compute backend options
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[allow(clippy::upper_case_acronyms)]
pub enum Backend {
    /// Scalar operations (baseline)
    Scalar,
    /// SIMD vectorization (AVX2, NEON)
    SIMD,
    /// GPU acceleration (WebGPU/Vulkan)
    GPU,
}

impl std::fmt::Display for Backend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Backend::Scalar => write!(f, "Scalar"),
            Backend::SIMD => write!(f, "SIMD"),
            Backend::GPU => write!(f, "GPU"),
        }
    }
}

/// Operation complexity for MoE (Mixture-of-Experts) routing
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum OpComplexity {
    /// Simple operations (add, mul) - O(n), prefer SIMD unless very large
    Low,
    /// Moderate operations (dot, reduce) - O(n), GPU beneficial at 100K+ elements
    Medium,
    /// Complex operations (matmul, convolution) - O(n²) or O(n³), GPU beneficial at 10K+ elements
    High,
}

/// Cost model for backend selection
/// Based on spec section 2.2 lines 191-204
pub struct BackendSelector {
    /// PCIe bandwidth in bytes/sec (default: 32 GB/s for PCIe 4.0 x16)
    pcie_bandwidth: f64,

    /// GPU compute throughput in FLOPS (default: 20 TFLOPS for A100)
    gpu_gflops: f64,

    /// Minimum dispatch ratio (default: 5× per Gregg & Hazelwood 2011)
    min_dispatch_ratio: f64,
}

impl Default for BackendSelector {
    fn default() -> Self {
        Self {
            pcie_bandwidth: 32e9,    // 32 GB/s
            gpu_gflops: 20e12,       // 20 TFLOPS
            min_dispatch_ratio: 5.0, // 5× rule
        }
    }
}

impl BackendSelector {
    pub fn new() -> Self {
        Self::default()
    }

    /// Configure custom PCIe bandwidth (must be > 0).
    pub fn with_pcie_bandwidth(mut self, bandwidth: f64) -> Self {
        assert!(bandwidth > 0.0, "PCIe bandwidth must be > 0");
        self.pcie_bandwidth = bandwidth;
        self
    }

    /// Configure custom GPU throughput (must be > 0).
    pub fn with_gpu_gflops(mut self, gflops: f64) -> Self {
        assert!(gflops > 0.0, "GPU GFLOPS must be > 0");
        self.gpu_gflops = gflops;
        self
    }

    /// Configure custom dispatch ratio threshold
    pub fn with_min_dispatch_ratio(mut self, ratio: f64) -> Self {
        self.min_dispatch_ratio = ratio;
        self
    }

    /// Select optimal backend based on workload characteristics
    ///
    /// # Arguments
    /// * `data_bytes` - Amount of data to transfer (host → device)
    /// * `flops` - Floating point operations required
    ///
    /// # Returns
    /// Recommended backend based on cost model
    ///
    /// # Cost Model
    /// GPU dispatch is beneficial when:
    /// ```text
    /// compute_time > min_dispatch_ratio × transfer_time
    /// ```
    ///
    /// Per Gregg & Hazelwood (2011), the 5× rule accounts for:
    /// - Host→Device transfer (PCIe overhead)
    /// - Kernel launch latency
    /// - Device→Host transfer
    /// - CPU-GPU synchronization
    pub fn select_backend(&self, data_bytes: usize, flops: u64) -> Backend {
        // Calculate transfer time (seconds)
        let transfer_s = data_bytes as f64 / self.pcie_bandwidth;

        // Calculate compute time (seconds)
        let compute_s = flops as f64 / self.gpu_gflops;

        // Apply 5× dispatch rule
        if compute_s > self.min_dispatch_ratio * transfer_s {
            Backend::GPU
        } else {
            // Fallback to SIMD for intermediate workloads
            Backend::SIMD
        }
    }

    /// Select backend for matrix multiplication
    ///
    /// # Arguments
    /// * `m`, `n`, `k` - Matrix dimensions (M×K) × (K×N) = (M×N)
    ///
    /// # Complexity
    /// - Data: O(mk + kn + mn) = O(mk + kn + mn) bytes
    /// - FLOPs: O(2mnk) = O(mnk) operations
    pub fn select_for_matmul(&self, m: usize, n: usize, k: usize) -> Backend {
        // Data size: two input matrices + output (f32 = 4 bytes)
        let data_bytes = (m * k + k * n + m * n) * 4;

        // FLOPs: 2mnk (multiply-add per element)
        let flops = (2 * m * n * k) as u64;

        self.select_backend(data_bytes, flops)
    }

    /// Select backend for vector operations
    ///
    /// # Arguments
    /// * `n` - Vector length
    /// * `ops_per_element` - Operations per element (e.g., 2 for dot product)
    pub fn select_for_vector_op(&self, n: usize, ops_per_element: u64) -> Backend {
        // Data size: typically two input vectors + output (f32 = 4 bytes)
        let data_bytes = n * 3 * 4;

        // FLOPs
        let flops = n as u64 * ops_per_element;

        self.select_backend(data_bytes, flops)
    }

    /// Select backend for element-wise operations
    ///
    /// Element-wise ops are memory-bound, so GPU is rarely beneficial
    pub fn select_for_elementwise(&self, n: usize) -> Backend {
        // Element-wise ops: 1 FLOP per element, memory-bound
        // GPU overhead rarely justified
        if n > 1_000_000 {
            Backend::SIMD
        } else {
            Backend::Scalar
        }
    }

    /// MoE (Mixture-of-Experts) routing: select backend based on operation complexity
    ///
    /// # Arguments
    /// * `complexity` - Operation complexity (Low/Medium/High)
    /// * `data_size` - Number of elements in the operation
    ///
    /// # Returns
    /// Recommended backend using adaptive thresholds per complexity level
    ///
    /// # MoE Thresholds (per empirical performance analysis)
    /// - **Low complexity** (element-wise): SIMD at 1M+ elements, never GPU
    /// - **Medium complexity** (reductions): SIMD at 10K+, GPU at 100K+ elements
    /// - **High complexity** (matmul): SIMD at 1K+, GPU at 10K+ elements
    pub fn select_with_moe(&self, complexity: OpComplexity, data_size: usize) -> Backend {
        match complexity {
            OpComplexity::Low => {
                // Element-wise: memory-bound, GPU overhead not justified
                if data_size > 1_000_000 {
                    Backend::SIMD
                } else {
                    Backend::Scalar
                }
            }
            OpComplexity::Medium => {
                // Reductions (dot product, sum): moderate compute
                if data_size > 100_000 {
                    Backend::GPU
                } else if data_size > 10_000 {
                    Backend::SIMD
                } else {
                    Backend::Scalar
                }
            }
            OpComplexity::High => {
                // Matrix operations: compute-intensive, O(n²) or O(n³)
                if data_size > 10_000 {
                    Backend::GPU
                } else if data_size > 1_000 {
                    Backend::SIMD
                } else {
                    Backend::Scalar
                }
            }
        }
    }

    /// Map Batuta Backend to Trueno Backend
    #[cfg(feature = "trueno-integration")]
    pub fn to_trueno_backend(backend: Backend) -> trueno::Backend {
        match backend {
            Backend::Scalar => trueno::Backend::Scalar,
            Backend::SIMD => trueno::Backend::Auto, // Let Trueno pick best SIMD (AVX2/NEON)
            Backend::GPU => trueno::Backend::GPU,
        }
    }

    /// Perform vector addition using Trueno with selected backend
    #[cfg(feature = "trueno-integration")]
    pub fn vector_add(&self, a: &[f32], b: &[f32]) -> Result<Vec<f32>, String> {
        if a.len() != b.len() {
            return Err("Vector lengths must match".to_string());
        }

        let _backend = self.select_with_moe(OpComplexity::Low, a.len());

        let vec_a: Vector<f32> = Vector::from_slice(a);
        let vec_b: Vector<f32> = Vector::from_slice(b);

        match vec_a.add(&vec_b) {
            Ok(result) => Ok(result.as_slice().to_vec()),
            Err(e) => Err(format!("Trueno error: {}", e)),
        }
    }

    /// Perform matrix multiplication using Trueno with selected backend
    #[cfg(feature = "trueno-integration")]
    pub fn matrix_multiply(
        &self,
        a: &[f32],
        b: &[f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<Vec<f32>, String> {
        // Matrix A: m×k, Matrix B: k×n, Result: m×n
        if a.len() != m * k {
            return Err(format!("Matrix A size mismatch: expected {}, got {}", m * k, a.len()));
        }
        if b.len() != k * n {
            return Err(format!("Matrix B size mismatch: expected {}, got {}", k * n, b.len()));
        }

        let _backend = self.select_for_matmul(m, n, k);

        // Create matrices using from_vec (Trueno API)
        let mat_a: Matrix<f32> = match Matrix::from_vec(m, k, a.to_vec()) {
            Ok(m) => m,
            Err(e) => return Err(format!("Trueno error creating matrix A: {}", e)),
        };

        let mat_b: Matrix<f32> = match Matrix::from_vec(k, n, b.to_vec()) {
            Ok(m) => m,
            Err(e) => return Err(format!("Trueno error creating matrix B: {}", e)),
        };

        match mat_a.matmul(&mat_b) {
            Ok(result) => Ok(result.as_slice().to_vec()),
            Err(e) => Err(format!("Trueno error in matmul: {}", e)),
        }
    }
}

#[cfg(test)]
#[path = "backend_tests.rs"]
mod tests;
