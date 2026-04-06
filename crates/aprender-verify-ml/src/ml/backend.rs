//! Backend selection and cost model for GPU-accelerated generation
//!
//! Implements adaptive backend selection using a Mixture-of-Experts approach.
//! The MoE router analyzes operation complexity and data size to select the optimal
//! compute backend (Scalar/SIMD/GPU).
//!
//! # Operation Complexity Levels
//!
//! - **Low**: Element-wise operations (add, multiply, etc.) - Memory-bound, GPU rarely beneficial
//! - **Medium**: Reductions (dot product, sum, etc.) - Moderate compute, GPU at 100K+ elements
//! - **High**: Matrix operations (matmul, convolution) - Compute-intensive O(n²) or O(n³), GPU at 10K+ elements
//!
//! # Usage Example
//!
//! ```
//! use verificar::ml::{BackendSelector, OpComplexity};
//!
//! let selector = BackendSelector::new();
//!
//! // Element-wise operation
//! let backend = selector.select_with_moe(OpComplexity::Low, 500_000);
//! // Returns: Scalar (below 1M threshold, memory-bound)
//!
//! // Matrix multiplication
//! let backend = selector.select_with_moe(OpComplexity::High, 50_000);
//! // Returns: GPU (above 10K threshold for O(n²) ops)
//! ```
//!
//! # Performance Thresholds
//!
//! Based on empirical analysis and the 5× PCIe rule (Gregg & Hazelwood 2011):
//!
//! | Complexity | SIMD Threshold | GPU Threshold | Rationale |
//! |------------|----------------|---------------|-----------|
//! | Low        | 1M elements    | Never         | Memory-bound, PCIe overhead dominates |
//! | Medium     | 10K elements   | 100K elements | Moderate compute/transfer ratio |
//! | High       | 1K elements    | 10K elements  | O(n²/n³) complexity favors GPU |

use serde::{Deserialize, Serialize};
use std::fmt;

/// Compute backend options
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Backend {
    /// Scalar operations (baseline)
    Scalar,
    /// SIMD vectorization (AVX2, NEON)
    Simd,
    /// GPU acceleration (WebGPU/Vulkan via trueno)
    Gpu,
}

impl fmt::Display for Backend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Backend::Scalar => write!(f, "Scalar"),
            Backend::Simd => write!(f, "SIMD"),
            Backend::Gpu => write!(f, "GPU"),
        }
    }
}

impl Default for Backend {
    fn default() -> Self {
        Backend::Scalar
    }
}

/// Operation complexity for MoE (Mixture-of-Experts) routing
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum OpComplexity {
    /// Simple operations (add, mul) - O(n), prefer SIMD unless very large
    Low,
    /// Moderate operations (dot, reduce) - O(n), GPU beneficial at 100K+ elements
    Medium,
    /// Complex operations (matmul, convolution) - O(n²) or O(n³), GPU beneficial at 10K+ elements
    High,
}

impl Default for OpComplexity {
    fn default() -> Self {
        OpComplexity::Low
    }
}

/// Cost model for backend selection
///
/// Based on Gregg & Hazelwood (2011) 5× PCIe rule for GPU dispatch decisions.
#[derive(Debug, Clone)]
pub struct BackendSelector {
    /// PCIe bandwidth in bytes/sec (default: 32 GB/s for PCIe 4.0 x16)
    pcie_bandwidth: f64,
    /// GPU compute throughput in FLOPS (default: 20 TFLOPS for A100)
    gpu_gflops: f64,
    /// Minimum dispatch ratio (default: 5× per Gregg & Hazelwood 2011)
    min_dispatch_ratio: f64,
    /// SIMD threshold for low complexity ops
    simd_threshold_low: usize,
    /// SIMD threshold for medium complexity ops
    simd_threshold_medium: usize,
    /// GPU threshold for medium complexity ops
    gpu_threshold_medium: usize,
    /// SIMD threshold for high complexity ops
    simd_threshold_high: usize,
    /// GPU threshold for high complexity ops
    gpu_threshold_high: usize,
}

impl Default for BackendSelector {
    fn default() -> Self {
        Self {
            pcie_bandwidth: 32e9,    // 32 GB/s
            gpu_gflops: 20e12,       // 20 TFLOPS
            min_dispatch_ratio: 5.0, // 5× rule
            simd_threshold_low: 1_000_000,
            simd_threshold_medium: 10_000,
            gpu_threshold_medium: 100_000,
            simd_threshold_high: 1_000,
            gpu_threshold_high: 10_000,
        }
    }
}

impl BackendSelector {
    /// Create a new backend selector with default thresholds
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Configure custom PCIe bandwidth
    #[must_use]
    pub fn with_pcie_bandwidth(mut self, bandwidth: f64) -> Self {
        self.pcie_bandwidth = bandwidth;
        self
    }

    /// Configure custom GPU throughput
    #[must_use]
    pub fn with_gpu_gflops(mut self, gflops: f64) -> Self {
        self.gpu_gflops = gflops;
        self
    }

    /// Configure custom dispatch ratio threshold
    #[must_use]
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
    #[must_use]
    pub fn select_backend(&self, data_bytes: usize, flops: u64) -> Backend {
        // Calculate transfer time (seconds)
        let transfer_s = data_bytes as f64 / self.pcie_bandwidth;

        // Calculate compute time (seconds)
        let compute_s = flops as f64 / self.gpu_gflops;

        // Apply 5× dispatch rule
        if compute_s > self.min_dispatch_ratio * transfer_s {
            Backend::Gpu
        } else {
            // Fallback to SIMD for intermediate workloads
            Backend::Simd
        }
    }

    /// Select backend for matrix multiplication
    ///
    /// # Arguments
    /// * `m`, `n`, `k` - Matrix dimensions (M×K) × (K×N) = (M×N)
    ///
    /// # Complexity
    /// - Data: O(mk + kn + mn) bytes
    /// - FLOPs: O(2mnk) operations
    #[must_use]
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
    #[must_use]
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
    #[must_use]
    pub fn select_for_elementwise(&self, n: usize) -> Backend {
        // Element-wise ops: 1 FLOP per element, memory-bound
        // GPU overhead rarely justified
        if n > self.simd_threshold_low {
            Backend::Simd
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
    #[must_use]
    pub fn select_with_moe(&self, complexity: OpComplexity, data_size: usize) -> Backend {
        match complexity {
            OpComplexity::Low => {
                // Element-wise: memory-bound, GPU overhead not justified
                if data_size > self.simd_threshold_low {
                    Backend::Simd
                } else {
                    Backend::Scalar
                }
            }
            OpComplexity::Medium => {
                // Reductions (dot product, sum): moderate compute
                if data_size > self.gpu_threshold_medium {
                    Backend::Gpu
                } else if data_size > self.simd_threshold_medium {
                    Backend::Simd
                } else {
                    Backend::Scalar
                }
            }
            OpComplexity::High => {
                // Matrix operations: compute-intensive, O(n²) or O(n³)
                if data_size > self.gpu_threshold_high {
                    Backend::Gpu
                } else if data_size > self.simd_threshold_high {
                    Backend::Simd
                } else {
                    Backend::Scalar
                }
            }
        }
    }

    /// Get selection statistics for profiling
    #[must_use]
    pub fn selection_stats(&self, complexity: OpComplexity, data_size: usize) -> SelectionStats {
        let backend = self.select_with_moe(complexity, data_size);

        // Estimate performance multiplier
        let speedup = match backend {
            Backend::Scalar => 1.0,
            Backend::Simd => {
                // AVX2 gives ~4-8x for aligned data
                match complexity {
                    OpComplexity::Low => 4.0,
                    OpComplexity::Medium => 6.0,
                    OpComplexity::High => 8.0,
                }
            }
            Backend::Gpu => {
                // GPU speedup depends heavily on problem size
                let base = match complexity {
                    OpComplexity::Low => 1.0, // Never selected for Low
                    OpComplexity::Medium => 10.0,
                    OpComplexity::High => 50.0,
                };
                // Scale with data size (diminishing returns)
                base * (data_size as f64 / 10_000.0).min(10.0)
            }
        };

        SelectionStats {
            backend,
            complexity,
            data_size,
            estimated_speedup: speedup,
        }
    }
}

/// Statistics about backend selection decision
#[derive(Debug, Clone)]
pub struct SelectionStats {
    /// Selected backend
    pub backend: Backend,
    /// Operation complexity
    pub complexity: OpComplexity,
    /// Data size (elements)
    pub data_size: usize,
    /// Estimated speedup vs scalar
    pub estimated_speedup: f64,
}

impl fmt::Display for SelectionStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} backend for {:?} complexity ({} elements) - ~{:.1}x speedup",
            self.backend, self.complexity, self.data_size, self.estimated_speedup
        )
    }
}

/// Batch configuration for parallel generation
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// Backend selector
    selector: BackendSelector,
    /// Batch size for generation
    pub batch_size: usize,
    /// Operation complexity hint
    pub complexity: OpComplexity,
}

impl BatchConfig {
    /// Create a new batch configuration
    #[must_use]
    pub fn new(batch_size: usize) -> Self {
        Self {
            selector: BackendSelector::new(),
            batch_size,
            complexity: OpComplexity::Low,
        }
    }

    /// Set operation complexity
    #[must_use]
    pub fn with_complexity(mut self, complexity: OpComplexity) -> Self {
        self.complexity = complexity;
        self
    }

    /// Get recommended backend for this batch
    #[must_use]
    pub fn recommended_backend(&self) -> Backend {
        self.selector
            .select_with_moe(self.complexity, self.batch_size)
    }

    /// Check if GPU acceleration is recommended
    #[must_use]
    pub fn should_use_gpu(&self) -> bool {
        self.recommended_backend() == Backend::Gpu
    }

    /// Check if SIMD acceleration is recommended
    #[must_use]
    pub fn should_use_simd(&self) -> bool {
        matches!(self.recommended_backend(), Backend::Simd | Backend::Gpu)
    }
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self::new(1000)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_display() {
        assert_eq!(format!("{}", Backend::Scalar), "Scalar");
        assert_eq!(format!("{}", Backend::Simd), "SIMD");
        assert_eq!(format!("{}", Backend::Gpu), "GPU");
    }

    #[test]
    fn test_backend_default() {
        assert_eq!(Backend::default(), Backend::Scalar);
    }

    #[test]
    fn test_op_complexity_ordering() {
        assert!(OpComplexity::Low < OpComplexity::Medium);
        assert!(OpComplexity::Medium < OpComplexity::High);
    }

    #[test]
    fn test_selector_default() {
        let selector = BackendSelector::new();
        assert_eq!(selector.min_dispatch_ratio, 5.0);
    }

    #[test]
    fn test_select_elementwise_small() {
        let selector = BackendSelector::new();
        let backend = selector.select_for_elementwise(100);
        assert_eq!(backend, Backend::Scalar);
    }

    #[test]
    fn test_select_elementwise_large() {
        let selector = BackendSelector::new();
        let backend = selector.select_for_elementwise(10_000_000);
        assert_eq!(backend, Backend::Simd);
    }

    #[test]
    fn test_moe_low_complexity() {
        let selector = BackendSelector::new();

        // Small: Scalar
        assert_eq!(
            selector.select_with_moe(OpComplexity::Low, 100),
            Backend::Scalar
        );

        // Large: SIMD (never GPU for low complexity)
        assert_eq!(
            selector.select_with_moe(OpComplexity::Low, 10_000_000),
            Backend::Simd
        );
    }

    #[test]
    fn test_moe_medium_complexity() {
        let selector = BackendSelector::new();

        // Small: Scalar
        assert_eq!(
            selector.select_with_moe(OpComplexity::Medium, 100),
            Backend::Scalar
        );

        // Medium: SIMD
        assert_eq!(
            selector.select_with_moe(OpComplexity::Medium, 50_000),
            Backend::Simd
        );

        // Large: GPU
        assert_eq!(
            selector.select_with_moe(OpComplexity::Medium, 500_000),
            Backend::Gpu
        );
    }

    #[test]
    fn test_moe_high_complexity() {
        let selector = BackendSelector::new();

        // Small: Scalar
        assert_eq!(
            selector.select_with_moe(OpComplexity::High, 100),
            Backend::Scalar
        );

        // Medium: SIMD
        assert_eq!(
            selector.select_with_moe(OpComplexity::High, 5_000),
            Backend::Simd
        );

        // Large: GPU
        assert_eq!(
            selector.select_with_moe(OpComplexity::High, 50_000),
            Backend::Gpu
        );
    }

    #[test]
    fn test_select_matmul_small() {
        let selector = BackendSelector::new();
        // 10x10 matmul: very small
        let backend = selector.select_for_matmul(10, 10, 10);
        assert_eq!(backend, Backend::Simd);
    }

    #[test]
    fn test_select_matmul_large() {
        let selector = BackendSelector::new();
        // 1000x1000 matmul: large but 5× rule still favors SIMD for this size
        // (compute time doesn't exceed 5× transfer time with default params)
        let backend = selector.select_for_matmul(1000, 1000, 1000);
        assert_eq!(backend, Backend::Simd);

        // For extremely large matmul where O(n³) compute dominates O(n²) transfer
        // Need n > 7500 for GPU to be beneficial with 2× rule
        // Using n=10000 to ensure GPU selection
        let fast_gpu_selector = BackendSelector::new().with_min_dispatch_ratio(2.0);
        let backend = fast_gpu_selector.select_for_matmul(10000, 10000, 10000);
        assert_eq!(backend, Backend::Gpu);
    }

    #[test]
    fn test_selection_stats() {
        let selector = BackendSelector::new();
        let stats = selector.selection_stats(OpComplexity::High, 100_000);

        assert_eq!(stats.backend, Backend::Gpu);
        assert!(stats.estimated_speedup > 1.0);
        assert!(format!("{}", stats).contains("GPU"));
    }

    #[test]
    fn test_batch_config() {
        let config = BatchConfig::new(50_000).with_complexity(OpComplexity::Medium);

        assert_eq!(config.batch_size, 50_000);
        assert!(config.should_use_simd());
        assert!(!config.should_use_gpu());
    }

    #[test]
    fn test_batch_config_gpu() {
        let config = BatchConfig::new(500_000).with_complexity(OpComplexity::Medium);

        assert!(config.should_use_gpu());
    }

    #[test]
    fn test_batch_config_default() {
        let config = BatchConfig::default();
        assert_eq!(config.batch_size, 1000);
    }

    #[test]
    fn test_custom_thresholds() {
        let selector = BackendSelector::new()
            .with_pcie_bandwidth(64e9)
            .with_gpu_gflops(80e12)
            .with_min_dispatch_ratio(3.0);

        // With faster GPU, smaller workloads become viable
        assert!(selector.pcie_bandwidth > 32e9);
        assert!(selector.gpu_gflops > 20e12);
    }

    #[test]
    fn test_vector_op_selection() {
        let selector = BackendSelector::new();

        // Small vector dot product
        let backend = selector.select_for_vector_op(100, 2);
        assert_eq!(backend, Backend::Simd);

        // Large vector ops - 5× rule still favors SIMD for memory-bound ops
        // (vector ops are inherently memory-bound with low compute intensity)
        let backend = selector.select_for_vector_op(10_000_000, 2);
        assert_eq!(backend, Backend::Simd);

        // With very high flops per element and lower dispatch ratio, GPU is viable
        // Need flops/element to overcome the 3:1 data/compute ratio
        let fast_gpu_selector = BackendSelector::new()
            .with_min_dispatch_ratio(0.1) // Very aggressive GPU dispatch
            .with_gpu_gflops(1e12); // Model a slower GPU
        let backend = fast_gpu_selector.select_for_vector_op(10_000_000, 100);
        assert_eq!(backend, Backend::Gpu);
    }

    #[test]
    fn test_op_complexity_default() {
        assert_eq!(OpComplexity::default(), OpComplexity::Low);
    }

    #[test]
    fn test_backend_serialization() {
        let backend = Backend::Gpu;
        let json = serde_json::to_string(&backend).expect("serialize");
        assert_eq!(json, "\"Gpu\"");

        let parsed: Backend = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed, Backend::Gpu);
    }

    #[test]
    fn test_complexity_serialization() {
        let complexity = OpComplexity::High;
        let json = serde_json::to_string(&complexity).expect("serialize");

        let parsed: OpComplexity = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed, OpComplexity::High);
    }
}
