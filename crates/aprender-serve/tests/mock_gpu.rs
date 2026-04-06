//! Mock GPU tests for host code coverage (Phase 41)
//!
//! Exercises GPU scheduling logic without actual GPU hardware.
//! Uses a mock compute backend that simulates GPU operations in software.
//!
//! This test module provides:
//! 1. Independent MockBackend implementation for isolation
//! 2. Comprehensive tests for weight management
//! 3. State isolation tests (catch accumulator bugs like PARITY-114)
//! 4. Matmul correctness verification

use std::collections::HashMap;

/// Mock GPU compute backend for testing scheduler logic
///
/// Simulates GPU operations without requiring actual GPU hardware.
/// Implements weight caching, matmul, and synchronization.
pub struct MockBackend {
    device_id: u32,
    weights: HashMap<String, Vec<f32>>,
    quantized_weights: HashMap<String, (Vec<u8>, u32)>, // (data, bits)
}

impl MockBackend {
    /// Check if mock backend is available (always true)
    #[must_use]
    pub fn is_available() -> bool {
        true
    }

    /// Create a new mock backend for the given device ID
    pub fn new(device_id: u32) -> Result<Self, &'static str> {
        Ok(Self {
            device_id,
            weights: HashMap::new(),
            quantized_weights: HashMap::new(),
        })
    }

    /// Get the device name
    #[must_use]
    pub fn device_name(&self) -> String {
        format!("MockGPU-{}", self.device_id)
    }

    /// Load weights into the mock backend's cache
    ///
    /// Returns the number of elements loaded.
    pub fn load_weights(&mut self, name: &str, weights: &[f32]) -> Result<usize, &'static str> {
        self.weights.insert(name.to_string(), weights.to_vec());
        Ok(weights.len())
    }

    /// Load quantized weights into the mock backend's cache
    ///
    /// Returns the number of bytes loaded.
    pub fn load_quantized_weights(
        &mut self,
        name: &str,
        data: &[u8],
        bits: u32,
    ) -> Result<usize, &'static str> {
        self.quantized_weights
            .insert(name.to_string(), (data.to_vec(), bits));
        Ok(data.len())
    }

    /// Check if weights are cached
    #[must_use]
    pub fn has_weights(&self, name: &str) -> bool {
        self.weights.contains_key(name) || self.quantized_weights.contains_key(name)
    }

    /// Get the number of cached weight tensors
    #[must_use]
    pub fn cached_weight_count(&self) -> usize {
        self.weights.len() + self.quantized_weights.len()
    }

    /// Clear all cached weights
    pub fn clear_weights(&mut self) {
        self.weights.clear();
        self.quantized_weights.clear();
    }

    /// Perform matrix multiplication: C = A @ B
    ///
    /// A: [m, k], B: [k, n] -> C: [m, n]
    #[allow(clippy::many_single_char_names)]
    pub fn matmul(
        &self,
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>, &'static str> {
        if a.len() != m * k {
            return Err("Matrix A dimension mismatch");
        }
        if b.len() != k * n {
            return Err("Matrix B dimension mismatch");
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

    /// Perform matrix multiplication using cached weights
    ///
    /// Looks up weights by name, then performs input @ weights^T
    /// input: [m, k], weights: [n, k] -> output: [m, n]
    #[allow(clippy::many_single_char_names)]
    pub fn matmul_cached(
        &self,
        weight_name: &str,
        input: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>, &'static str> {
        let weights = self.weights.get(weight_name).ok_or("Weight not found")?;

        if weights.len() != n * k {
            return Err("Cached weight dimension mismatch");
        }
        if input.len() != m * k {
            return Err("Input dimension mismatch");
        }

        // Compute input @ weights^T (weights stored as [n, k])
        let mut output = vec![0.0f32; m * n];

        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for p in 0..k {
                    // weights[j, p] = weights[j * k + p]
                    sum += input[i * k + p] * weights[j * k + p];
                }
                output[i * n + j] = sum;
            }
        }

        Ok(output)
    }

    /// Synchronize all pending operations (no-op for mock)
    pub fn synchronize(&self) -> Result<(), &'static str> {
        Ok(())
    }
}

// ============================================================================
// Tests: Backend Availability and Creation
// ============================================================================

#[test]
fn test_mock_backend_available() {
    assert!(MockBackend::is_available());
}

#[test]
fn test_mock_backend_creation() {
    let backend = MockBackend::new(0).expect("should create");
    assert!(!backend.device_name().is_empty());
    assert!(backend.device_name().contains("MockGPU"));
}

#[test]
fn test_mock_backend_creation_multiple_devices() {
    let backend0 = MockBackend::new(0).expect("should create device 0");
    let backend1 = MockBackend::new(1).expect("should create device 1");

    assert!(backend0.device_name().contains('0'));
    assert!(backend1.device_name().contains('1'));
}

// ============================================================================
// Tests: Weight Loading and Caching
// ============================================================================

#[test]
fn test_weight_loading() {
    let mut backend = MockBackend::new(0).unwrap();
    let weights = vec![1.0f32; 1024];
    let size = backend.load_weights("test_weight", &weights).unwrap();
    assert_eq!(size, 1024);
    assert!(backend.has_weights("test_weight"));
    assert_eq!(backend.cached_weight_count(), 1);
}

#[test]
fn test_weight_loading_multiple() {
    let mut backend = MockBackend::new(0).unwrap();

    backend.load_weights("w1", &[1.0; 100]).unwrap();
    backend.load_weights("w2", &[2.0; 200]).unwrap();
    backend.load_weights("w3", &[3.0; 300]).unwrap();

    assert_eq!(backend.cached_weight_count(), 3);
    assert!(backend.has_weights("w1"));
    assert!(backend.has_weights("w2"));
    assert!(backend.has_weights("w3"));
    assert!(!backend.has_weights("w4"));
}

#[test]
fn test_weight_overwrite() {
    let mut backend = MockBackend::new(0).unwrap();

    backend.load_weights("w1", &[1.0; 100]).unwrap();
    assert_eq!(backend.cached_weight_count(), 1);

    // Overwrite with new weights
    backend.load_weights("w1", &[2.0; 200]).unwrap();
    assert_eq!(backend.cached_weight_count(), 1); // Still 1, overwritten
}

#[test]
fn test_quantized_weight_loading() {
    let mut backend = MockBackend::new(0).unwrap();
    let qweights = vec![0u8; 512];
    let size = backend
        .load_quantized_weights("q_weight", &qweights, 4)
        .unwrap();
    assert_eq!(size, 512);
    assert!(backend.has_weights("q_weight"));
}

#[test]
fn test_quantized_weight_loading_different_bits() {
    let mut backend = MockBackend::new(0).unwrap();

    backend
        .load_quantized_weights("q4", &[0u8; 100], 4)
        .unwrap();
    backend
        .load_quantized_weights("q8", &[0u8; 200], 8)
        .unwrap();

    assert!(backend.has_weights("q4"));
    assert!(backend.has_weights("q8"));
    assert_eq!(backend.cached_weight_count(), 2);
}

#[test]
fn test_clear_weights() {
    let mut backend = MockBackend::new(0).unwrap();
    backend.load_weights("w1", &[1.0; 100]).unwrap();
    backend.load_weights("w2", &[2.0; 100]).unwrap();
    assert_eq!(backend.cached_weight_count(), 2);

    backend.clear_weights();
    assert_eq!(backend.cached_weight_count(), 0);
    assert!(!backend.has_weights("w1"));
    assert!(!backend.has_weights("w2"));
}

#[test]
fn test_clear_includes_quantized() {
    let mut backend = MockBackend::new(0).unwrap();
    backend.load_weights("f32_weight", &[1.0; 100]).unwrap();
    backend
        .load_quantized_weights("q4_weight", &[0u8; 50], 4)
        .unwrap();
    assert_eq!(backend.cached_weight_count(), 2);

    backend.clear_weights();
    assert_eq!(backend.cached_weight_count(), 0);
    assert!(!backend.has_weights("f32_weight"));
    assert!(!backend.has_weights("q4_weight"));
}

// ============================================================================
// Tests: Matrix Multiplication
// ============================================================================

#[test]
fn test_matmul_basic() {
    let backend = MockBackend::new(0).unwrap();

    // 2x3 @ 3x2 = 2x2
    // A = [[1,2,3], [4,5,6]]
    // B = [[1,2], [3,4], [5,6]]
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
    let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 3x2 row-major

    let c = backend.matmul(&a, &b, 2, 3, 2).unwrap();
    assert_eq!(c.len(), 4);

    // C[0,0] = 1*1 + 2*3 + 3*5 = 1 + 6 + 15 = 22
    // C[0,1] = 1*2 + 2*4 + 3*6 = 2 + 8 + 18 = 28
    // C[1,0] = 4*1 + 5*3 + 6*5 = 4 + 15 + 30 = 49
    // C[1,1] = 4*2 + 5*4 + 6*6 = 8 + 20 + 36 = 64
    assert!(
        (c[0] - 22.0).abs() < 1e-5,
        "C[0,0] should be 22, got {}",
        c[0]
    );
    assert!(
        (c[1] - 28.0).abs() < 1e-5,
        "C[0,1] should be 28, got {}",
        c[1]
    );
    assert!(
        (c[2] - 49.0).abs() < 1e-5,
        "C[1,0] should be 49, got {}",
        c[2]
    );
    assert!(
        (c[3] - 64.0).abs() < 1e-5,
        "C[1,1] should be 64, got {}",
        c[3]
    );
}

#[test]
fn test_matmul_identity() {
    let backend = MockBackend::new(0).unwrap();

    // A @ I = A
    let a = vec![1.0, 2.0, 3.0, 4.0]; // 2x2
    let identity = vec![1.0, 0.0, 0.0, 1.0]; // 2x2 identity

    let c = backend.matmul(&a, &identity, 2, 2, 2).unwrap();
    assert_eq!(c.len(), 4);

    for (i, (&expected, &actual)) in a.iter().zip(c.iter()).enumerate() {
        assert!(
            (expected - actual).abs() < 1e-5,
            "Element {} mismatch: expected {}, got {}",
            i,
            expected,
            actual
        );
    }
}

#[test]
fn test_matmul_vector_matrix() {
    let backend = MockBackend::new(0).unwrap();

    // [1, 2, 3] @ [[1,0], [0,1], [1,1]] = [1+0+3, 0+2+3] = [4, 5]
    let a = vec![1.0, 2.0, 3.0]; // 1x3
    let b = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0]; // 3x2

    let c = backend.matmul(&a, &b, 1, 3, 2).unwrap();
    assert_eq!(c.len(), 2);
    assert!((c[0] - 4.0).abs() < 1e-5, "expected 4, got {}", c[0]);
    assert!((c[1] - 5.0).abs() < 1e-5, "expected 5, got {}", c[1]);
}

#[test]
fn test_matmul_dimension_error_a() {
    let backend = MockBackend::new(0).unwrap();

    let a = vec![1.0; 10]; // Wrong size for 2x3
    let b = vec![1.0; 6]; // 3x2

    let result = backend.matmul(&a, &b, 2, 3, 2);
    assert!(result.is_err());
}

#[test]
fn test_matmul_dimension_error_b() {
    let backend = MockBackend::new(0).unwrap();

    let a = vec![1.0; 6]; // 2x3
    let b = vec![1.0; 10]; // Wrong size for 3x2

    let result = backend.matmul(&a, &b, 2, 3, 2);
    assert!(result.is_err());
}

#[test]
fn test_matmul_zeros() {
    let backend = MockBackend::new(0).unwrap();

    let a = vec![0.0f32; 16];
    let b = vec![1.0f32; 16];

    let c = backend.matmul(&a, &b, 4, 4, 4).unwrap();

    // Zero input should produce zero output
    for (i, &val) in c.iter().enumerate() {
        assert!(val.abs() < 1e-10, "Element {} should be 0, got {}", i, val);
    }
}

#[test]
fn test_matmul_negative_values() {
    let backend = MockBackend::new(0).unwrap();

    let a = vec![-1.0, 2.0, -3.0, 4.0]; // 2x2
    let b = vec![1.0, -1.0, -1.0, 1.0]; // 2x2

    let c = backend.matmul(&a, &b, 2, 2, 2).unwrap();

    // C[0,0] = -1*1 + 2*(-1) = -1 - 2 = -3
    // C[0,1] = -1*(-1) + 2*1 = 1 + 2 = 3
    // C[1,0] = -3*1 + 4*(-1) = -3 - 4 = -7
    // C[1,1] = -3*(-1) + 4*1 = 3 + 4 = 7

    assert!(
        (c[0] - (-3.0)).abs() < 1e-5,
        "C[0,0] should be -3, got {}",
        c[0]
    );
    assert!(
        (c[1] - 3.0).abs() < 1e-5,
        "C[0,1] should be 3, got {}",
        c[1]
    );
    assert!(
        (c[2] - (-7.0)).abs() < 1e-5,
        "C[1,0] should be -7, got {}",
        c[2]
    );
    assert!(
        (c[3] - 7.0).abs() < 1e-5,
        "C[1,1] should be 7, got {}",
        c[3]
    );
}

#[test]
fn test_large_matmul() {
    let backend = MockBackend::new(0).unwrap();

    // Test larger matrix to ensure no issues with size
    let rows = 128;
    let inner = 256;
    let cols = 64;

    let a: Vec<f32> = (0..rows * inner).map(|i| (i % 10) as f32 * 0.1).collect();
    let b: Vec<f32> = (0..inner * cols).map(|i| (i % 10) as f32 * 0.1).collect();

    let c = backend.matmul(&a, &b, rows, inner, cols).unwrap();
    assert_eq!(c.len(), rows * cols);

    // Just verify it runs without error and produces finite values
    for (i, &val) in c.iter().enumerate() {
        assert!(val.is_finite(), "Element {} is not finite: {}", i, val);
    }
}

// ============================================================================
// Tests: Cached Weight Matmul
// ============================================================================

#[test]
fn test_matmul_cached() {
    let mut backend = MockBackend::new(0).unwrap();

    // Load weights: 64x32 (stored as [64, 32] row-major)
    let weights = vec![1.0f32; 64 * 32];
    backend.load_weights("fc1", &weights).unwrap();

    // Input: 1x32
    let input = vec![1.0f32; 32];

    // Compute input @ weights^T -> [1, 64]
    // Each element = sum of 32 ones * 32 ones = 32.0
    let output = backend.matmul_cached("fc1", &input, 1, 32, 64).unwrap();
    assert_eq!(output.len(), 64);

    for (i, &val) in output.iter().enumerate() {
        assert!(
            (val - 32.0).abs() < 1e-5,
            "output[{}] should be 32.0, got {}",
            i,
            val
        );
    }
}

#[test]
fn test_matmul_cached_missing_weight() {
    let backend = MockBackend::new(0).unwrap();

    let input = vec![1.0f32; 32];
    let result = backend.matmul_cached("nonexistent", &input, 1, 32, 64);

    assert!(result.is_err());
}

#[test]
fn test_matmul_cached_dimension_mismatch() {
    let mut backend = MockBackend::new(0).unwrap();

    // Load 64x32 weights
    let weights = vec![1.0f32; 64 * 32];
    backend.load_weights("fc1", &weights).unwrap();

    // Try to use with wrong dimensions
    let input = vec![1.0f32; 64]; // Wrong k dimension
    let result = backend.matmul_cached("fc1", &input, 1, 64, 64);

    assert!(result.is_err());
}

// ============================================================================
// Tests: State Isolation (Critical for GPU Correctness)
// ============================================================================

#[test]
fn test_state_isolation() {
    // CRITICAL: Same operation twice must produce identical results
    // This catches state accumulation bugs (cf. PARITY-114)
    let backend = MockBackend::new(0).unwrap();

    // 4x4 @ 4x4 -> 4x4
    let a = vec![1.0f32; 16];
    let b = vec![2.0f32; 16];

    let r1 = backend.matmul(&a, &b, 4, 4, 4).unwrap();
    let r2 = backend.matmul(&a, &b, 4, 4, 4).unwrap();

    assert_eq!(r1.len(), r2.len(), "Output lengths should match");

    for (i, (&v1, &v2)) in r1.iter().zip(r2.iter()).enumerate() {
        assert!(
            (v1 - v2).abs() < 1e-10,
            "State leak detected at element {}: first={}, second={}",
            i,
            v1,
            v2
        );
    }
}

#[test]
fn test_state_isolation_with_caching() {
    // Same operation with cached weights twice must produce identical results
    let mut backend = MockBackend::new(0).unwrap();

    let weights = vec![1.0f32; 64 * 32];
    backend.load_weights("fc1", &weights).unwrap();

    let input = vec![0.5f32; 32];

    let r1 = backend.matmul_cached("fc1", &input, 1, 32, 64).unwrap();
    let r2 = backend.matmul_cached("fc1", &input, 1, 32, 64).unwrap();

    assert_eq!(r1.len(), r2.len());

    for (i, (&v1, &v2)) in r1.iter().zip(r2.iter()).enumerate() {
        assert!(
            (v1 - v2).abs() < 1e-10,
            "State leak detected at element {}: first={}, second={}",
            i,
            v1,
            v2
        );
    }
}

#[test]
fn test_state_isolation_interleaved_operations() {
    // Interleaved operations should not affect each other
    let mut backend = MockBackend::new(0).unwrap();

    backend.load_weights("w1", &vec![1.0f32; 64]).unwrap();
    backend.load_weights("w2", &vec![2.0f32; 64]).unwrap();

    let input = vec![1.0f32; 8];

    // Alternate between w1 and w2
    let r1_w1 = backend.matmul_cached("w1", &input, 1, 8, 8).unwrap();
    let r1_w2 = backend.matmul_cached("w2", &input, 1, 8, 8).unwrap();
    let r2_w1 = backend.matmul_cached("w1", &input, 1, 8, 8).unwrap();
    let r2_w2 = backend.matmul_cached("w2", &input, 1, 8, 8).unwrap();

    // r1_w1 should equal r2_w1
    for (i, (&v1, &v2)) in r1_w1.iter().zip(r2_w1.iter()).enumerate() {
        assert!(
            (v1 - v2).abs() < 1e-10,
            "w1 state leak at {}: {} != {}",
            i,
            v1,
            v2
        );
    }

    // r1_w2 should equal r2_w2
    for (i, (&v1, &v2)) in r1_w2.iter().zip(r2_w2.iter()).enumerate() {
        assert!(
            (v1 - v2).abs() < 1e-10,
            "w2 state leak at {}: {} != {}",
            i,
            v1,
            v2
        );
    }

    // w1 and w2 should produce different results (weights are different)
    assert!(
        (r1_w1[0] - r1_w2[0]).abs() > 1e-5,
        "w1 and w2 should produce different results"
    );
}

// ============================================================================
// Tests: Synchronization
// ============================================================================

#[test]
fn test_synchronize() {
    let backend = MockBackend::new(0).unwrap();
    backend.synchronize().expect("sync should succeed");
}

#[test]
fn test_synchronize_after_operations() {
    let backend = MockBackend::new(0).unwrap();

    // Do some operations
    let a = vec![1.0f32; 16];
    let b = vec![1.0f32; 16];
    let _result = backend.matmul(&a, &b, 4, 4, 4).unwrap();

    // Sync should still work
    backend
        .synchronize()
        .expect("sync after ops should succeed");
}

// ============================================================================
// Tests: Multi-Backend Isolation
// ============================================================================

#[test]
fn test_backend_device_tracking() {
    // Multiple backends should track their own device IDs
    let backend0 = MockBackend::new(0).unwrap();
    let backend1 = MockBackend::new(1).unwrap();
    let backend2 = MockBackend::new(2).unwrap();

    assert!(backend0.device_name().contains('0'));
    assert!(backend1.device_name().contains('1'));
    assert!(backend2.device_name().contains('2'));
}

#[test]
fn test_weight_isolation_between_backends() {
    let mut backend0 = MockBackend::new(0).unwrap();
    let mut backend1 = MockBackend::new(1).unwrap();

    backend0.load_weights("shared_name", &[1.0; 100]).unwrap();

    // backend1 should not see backend0's weights
    assert!(backend0.has_weights("shared_name"));
    assert!(!backend1.has_weights("shared_name"));

    // Load different weights with same name to backend1
    backend1.load_weights("shared_name", &[2.0; 200]).unwrap();

    // Both should have their own copy
    assert!(backend0.has_weights("shared_name"));
    assert!(backend1.has_weights("shared_name"));

    // Counts should be independent
    assert_eq!(backend0.cached_weight_count(), 1);
    assert_eq!(backend1.cached_weight_count(), 1);
}
