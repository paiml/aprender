//! ComputeBackend trait for abstracting GPU operations (Phase 41)
//!
//! Enables mocking GPU host code for coverage testing.

use std::error::Error;

/// Result type for backend operations
pub type BackendResult<T> = Result<T, Box<dyn Error + Send + Sync>>;

/// Abstraction over GPU compute backends (CUDA, Mock, future: Metal, HIP)
///
/// This trait allows swapping out the actual GPU implementation with a mock
/// for testing purposes, enabling coverage of GPU host code without requiring
/// actual GPU hardware.
///
/// # Example
///
/// ```rust,ignore
/// use realizar::gpu::backend::{ComputeBackend, BackendResult};
///
/// struct MockBackend {
///     weights: std::collections::HashMap<String, Vec<f32>>,
/// }
///
/// impl ComputeBackend for MockBackend {
///     fn is_available() -> bool { true }
///     fn new(_device_id: u32) -> BackendResult<Self> {
///         Ok(Self { weights: Default::default() })
///     }
///     // ... implement other methods
/// }
/// ```
pub trait ComputeBackend: Send {
    /// Check if this backend type is available on the system
    fn is_available() -> bool
    where
        Self: Sized;

    /// Create a new backend instance for the given device
    ///
    /// # Errors
    ///
    /// Returns error if device initialization fails.
    fn new(device_id: u32) -> BackendResult<Self>
    where
        Self: Sized;

    /// Get device name
    fn device_name(&self) -> String;

    // =========================================================================
    // Weight Management
    // =========================================================================

    /// Load f32 weights to device memory
    ///
    /// Returns a handle/index for the loaded weights.
    ///
    /// # Errors
    ///
    /// Returns error if memory allocation or transfer fails.
    fn load_weights(&mut self, name: &str, data: &[f32]) -> BackendResult<usize>;

    /// Load quantized weights to device memory
    ///
    /// # Arguments
    ///
    /// * `name` - Unique identifier for the weights
    /// * `data` - Raw quantized bytes
    /// * `qtype` - Quantization type (e.g., 2 for Q4_0, 3 for Q4_K)
    ///
    /// # Errors
    ///
    /// Returns error if memory allocation or transfer fails.
    fn load_quantized_weights(
        &mut self,
        name: &str,
        data: &[u8],
        qtype: u32,
    ) -> BackendResult<usize>;

    /// Check if weights are loaded
    fn has_weights(&self, name: &str) -> bool;

    /// Clear all cached weights from device memory
    fn clear_weights(&mut self);

    /// Get count of cached weights
    fn cached_weight_count(&self) -> usize;

    // =========================================================================
    // Core Compute
    // =========================================================================

    /// Matrix multiplication: C = A @ B
    ///
    /// # Arguments
    ///
    /// * `a` - Left matrix, row-major [m, k]
    /// * `b` - Right matrix, row-major [k, n]
    /// * `m` - Rows in A
    /// * `k` - Cols in A / Rows in B
    /// * `n` - Cols in B
    ///
    /// # Errors
    ///
    /// Returns error if dimensions are invalid or compute fails.
    fn matmul(&mut self, a: &[f32], b: &[f32], m: u32, k: u32, n: u32) -> BackendResult<Vec<f32>>;

    /// Matrix multiplication using cached weights
    ///
    /// # Arguments
    ///
    /// * `weight_name` - Name of previously loaded weights
    /// * `x` - Input vector/matrix
    /// * `m` - Batch size (rows in input)
    /// * `k` - Input dimension
    /// * `n` - Output dimension
    ///
    /// # Errors
    ///
    /// Returns error if weights not found or compute fails.
    fn matmul_cached(
        &mut self,
        weight_name: &str,
        x: &[f32],
        m: u32,
        k: u32,
        n: u32,
    ) -> BackendResult<Vec<f32>>;

    // =========================================================================
    // Quantized Operations
    // =========================================================================

    /// Q4_K quantized GEMV using cached weights
    ///
    /// Computes y = dequant(W_q4k) @ x where W is Q4_K quantized.
    ///
    /// # Arguments
    ///
    /// * `weight_name` - Name of previously loaded Q4_K weights
    /// * `input` - Input vector [k]
    /// * `n` - Output dimension (rows in weight matrix)
    /// * `k` - Input dimension (cols in weight matrix)
    ///
    /// # Errors
    ///
    /// Returns error if weights not found, wrong quantization type, or compute fails.
    fn q4k_gemv_cached(
        &mut self,
        weight_name: &str,
        input: &[f32],
        n: u32,
        k: u32,
    ) -> BackendResult<Vec<f32>>;

    // =========================================================================
    // Synchronization
    // =========================================================================

    /// Synchronize device execution
    ///
    /// Blocks until all queued operations complete.
    ///
    /// # Errors
    ///
    /// Returns error if synchronization fails.
    fn synchronize(&self) -> BackendResult<()>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// Mock backend for testing trait compliance
    struct MockBackend {
        device_name: String,
        weights: HashMap<String, Vec<f32>>,
        quantized_weights: HashMap<String, (Vec<u8>, u32)>,
    }

    impl ComputeBackend for MockBackend {
        fn is_available() -> bool {
            true
        }

        fn new(device_id: u32) -> BackendResult<Self> {
            Ok(Self {
                device_name: format!("MockDevice:{}", device_id),
                weights: HashMap::new(),
                quantized_weights: HashMap::new(),
            })
        }

        fn device_name(&self) -> String {
            self.device_name.clone()
        }

        fn load_weights(&mut self, name: &str, data: &[f32]) -> BackendResult<usize> {
            let len = data.len();
            self.weights.insert(name.to_string(), data.to_vec());
            Ok(len)
        }

        fn load_quantized_weights(
            &mut self,
            name: &str,
            data: &[u8],
            qtype: u32,
        ) -> BackendResult<usize> {
            let len = data.len();
            self.quantized_weights
                .insert(name.to_string(), (data.to_vec(), qtype));
            Ok(len)
        }

        fn has_weights(&self, name: &str) -> bool {
            self.weights.contains_key(name) || self.quantized_weights.contains_key(name)
        }

        fn clear_weights(&mut self) {
            self.weights.clear();
            self.quantized_weights.clear();
        }

        fn cached_weight_count(&self) -> usize {
            self.weights.len() + self.quantized_weights.len()
        }

        #[allow(clippy::many_single_char_names)]
        fn matmul(
            &mut self,
            a: &[f32],
            b: &[f32],
            m: u32,
            k: u32,
            n: u32,
        ) -> BackendResult<Vec<f32>> {
            let m = m as usize;
            let k = k as usize;
            let n = n as usize;

            if a.len() != m * k {
                return Err(format!("A size {} != m*k {}", a.len(), m * k).into());
            }
            if b.len() != k * n {
                return Err(format!("B size {} != k*n {}", b.len(), k * n).into());
            }

            let mut c = vec![0.0f32; m * n];
            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0f32;
                    for p in 0..k {
                        sum += a[i * k + p] * b[p * n + j];
                    }
                    c[i * n + j] = sum;
                }
            }
            Ok(c)
        }

        fn matmul_cached(
            &mut self,
            weight_name: &str,
            x: &[f32],
            m: u32,
            k: u32,
            n: u32,
        ) -> BackendResult<Vec<f32>> {
            let weights = self
                .weights
                .get(weight_name)
                .ok_or_else(|| format!("Weights '{}' not found", weight_name))?
                .clone();
            self.matmul(x, &weights, m, k, n)
        }

        fn q4k_gemv_cached(
            &mut self,
            weight_name: &str,
            input: &[f32],
            n: u32,
            k: u32,
        ) -> BackendResult<Vec<f32>> {
            // Mock: just return zeros of correct size
            if !self.quantized_weights.contains_key(weight_name) {
                return Err(format!("Quantized weights '{}' not found", weight_name).into());
            }
            let _ = input;
            let _ = k;
            Ok(vec![0.0f32; n as usize])
        }

        fn synchronize(&self) -> BackendResult<()> {
            Ok(())
        }
    }

    #[test]
    fn test_mock_backend_creation() {
        assert!(MockBackend::is_available());
        let backend = MockBackend::new(0).expect("backend");
        assert_eq!(backend.device_name(), "MockDevice:0");
    }

    #[test]
    fn test_mock_backend_weight_management() {
        let mut backend = MockBackend::new(0).expect("backend");

        // Initially empty
        assert_eq!(backend.cached_weight_count(), 0);
        assert!(!backend.has_weights("test"));

        // Load weights
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let handle = backend.load_weights("test", &data).expect("handle");
        assert_eq!(handle, 4);
        assert!(backend.has_weights("test"));
        assert_eq!(backend.cached_weight_count(), 1);

        // Load quantized weights
        let qdata = vec![0u8; 18]; // Q4_0 block
        let qhandle = backend
            .load_quantized_weights("test_q4", &qdata, 2)
            .expect("expected value");
        assert_eq!(qhandle, 18);
        assert!(backend.has_weights("test_q4"));
        assert_eq!(backend.cached_weight_count(), 2);

        // Clear
        backend.clear_weights();
        assert_eq!(backend.cached_weight_count(), 0);
        assert!(!backend.has_weights("test"));
    }

    #[test]
    fn test_mock_backend_matmul() {
        let mut backend = MockBackend::new(0).expect("backend");

        // 2x3 @ 3x2 = 2x2
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];

        let c = backend.matmul(&a, &b, 2, 3, 2).expect("c");

        // Expected: [[1*7+2*9+3*11, 1*8+2*10+3*12], [4*7+5*9+6*11, 4*8+5*10+6*12]]
        //         = [[7+18+33, 8+20+36], [28+45+66, 32+50+72]]
        //         = [[58, 64], [139, 154]]
        assert_eq!(c.len(), 4);
        assert!((c[0] - 58.0).abs() < 1e-5);
        assert!((c[1] - 64.0).abs() < 1e-5);
        assert!((c[2] - 139.0).abs() < 1e-5);
        assert!((c[3] - 154.0).abs() < 1e-5);
    }

    #[test]
    fn test_mock_backend_matmul_cached() {
        let mut backend = MockBackend::new(0).expect("backend");

        // Load weights: 3x2 matrix
        let weights = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        backend.load_weights("W", &weights).expect("load_weights");

        // Input: 2x3
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let c = backend.matmul_cached("W", &x, 2, 3, 2).expect("c");
        assert_eq!(c.len(), 4);
    }

    #[test]
    fn test_mock_backend_sync() {
        let backend = MockBackend::new(0).expect("backend");
        assert!(backend.synchronize().is_ok());
    }

    #[test]
    fn test_mock_backend_matmul_dimension_error_a() {
        let mut backend = MockBackend::new(0).expect("backend");
        // A should be 2x3 but we provide wrong size
        let a = vec![1.0, 2.0, 3.0]; // only 3 elements, not 6
        let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]; // 3x2

        let result = backend.matmul(&a, &b, 2, 3, 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_mock_backend_matmul_dimension_error_b() {
        let mut backend = MockBackend::new(0).expect("backend");
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
                                                    // B should be 3x2 but we provide wrong size
        let b = vec![7.0, 8.0]; // only 2 elements, not 6

        let result = backend.matmul(&a, &b, 2, 3, 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_mock_backend_matmul_cached_weight_not_found() {
        let mut backend = MockBackend::new(0).expect("backend");
        let x = vec![1.0, 2.0, 3.0];
        let result = backend.matmul_cached("nonexistent", &x, 1, 3, 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_mock_backend_q4k_gemv_cached() {
        let mut backend = MockBackend::new(0).expect("backend");

        // Load quantized weights
        let qdata = vec![0u8; 256]; // Q4_K data
        backend
            .load_quantized_weights("q4k_test", &qdata, 3)
            .expect("expected value");

        let input = vec![1.0, 2.0, 3.0, 4.0];
        let result = backend
            .q4k_gemv_cached("q4k_test", &input, 8, 4)
            .expect("result");

        assert_eq!(result.len(), 8);
    }

    #[test]
    fn test_mock_backend_q4k_gemv_cached_weight_not_found() {
        let mut backend = MockBackend::new(0).expect("backend");
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let result = backend.q4k_gemv_cached("nonexistent", &input, 8, 4);
        assert!(result.is_err());
    }

    #[test]
    fn test_mock_backend_different_device_ids() {
        let backend0 = MockBackend::new(0).expect("backend0");
        let backend1 = MockBackend::new(1).expect("backend1");
        let backend2 = MockBackend::new(42).expect("backend2");

        assert_eq!(backend0.device_name(), "MockDevice:0");
        assert_eq!(backend1.device_name(), "MockDevice:1");
        assert_eq!(backend2.device_name(), "MockDevice:42");
    }

    #[test]
    fn test_mock_backend_load_multiple_weights() {
        let mut backend = MockBackend::new(0).expect("backend");

        backend
            .load_weights("w1", &[1.0, 2.0])
            .expect("load_weights");
        backend
            .load_weights("w2", &[3.0, 4.0])
            .expect("load_weights");
        backend
            .load_weights("w3", &[5.0, 6.0])
            .expect("load_weights");
        backend
            .load_quantized_weights("q1", &[0u8; 10], 2)
            .expect("load_quantized_weights");
        backend
            .load_quantized_weights("q2", &[0u8; 20], 3)
            .expect("load_quantized_weights");

        assert_eq!(backend.cached_weight_count(), 5);
        assert!(backend.has_weights("w1"));
        assert!(backend.has_weights("w2"));
        assert!(backend.has_weights("w3"));
        assert!(backend.has_weights("q1"));
        assert!(backend.has_weights("q2"));
        assert!(!backend.has_weights("w4"));
    }

    #[test]
    fn test_mock_backend_matmul_1x1() {
        let mut backend = MockBackend::new(0).expect("backend");
        // Simple scalar multiplication via matmul
        let a = vec![2.0];
        let b = vec![3.0];
        let c = backend.matmul(&a, &b, 1, 1, 1).expect("c");
        assert_eq!(c.len(), 1);
        assert!((c[0] - 6.0).abs() < 1e-5);
    }

    #[test]
    fn test_mock_backend_matmul_identity() {
        let mut backend = MockBackend::new(0).expect("backend");
        // Identity matrix multiplication
        let a = vec![1.0, 2.0, 3.0, 4.0]; // 2x2
        let identity = vec![1.0, 0.0, 0.0, 1.0]; // 2x2 identity
        let c = backend.matmul(&a, &identity, 2, 2, 2).expect("c");
        assert_eq!(c.len(), 4);
        assert!((c[0] - 1.0).abs() < 1e-5);
        assert!((c[1] - 2.0).abs() < 1e-5);
        assert!((c[2] - 3.0).abs() < 1e-5);
        assert!((c[3] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn test_mock_backend_load_empty_weights() {
        let mut backend = MockBackend::new(0).expect("backend");
        let result = backend.load_weights("empty", &[]);
        assert!(result.is_ok());
        assert_eq!(result.expect("result"), 0);
        assert!(backend.has_weights("empty"));
    }
}
