//! Mock compute backend for coverage testing (Phase 41)
//!
//! Allows GPU scheduling/orchestration code to run without actual GPU.
//! This enables comprehensive testing of GPU host-side logic on CI systems
//! that may not have GPUs available.
//!
//! # Example
//!
//! ```rust,ignore
//! use realizar::gpu::mock_backend::MockBackend;
//! use realizar::gpu::backend::ComputeBackend;
//!
//! // Create mock backend (always succeeds)
//! let mut backend = MockBackend::new(0).expect("backend");
//!
//! // Load weights (stored in CPU memory)
//! backend.load_weights("layer0.weight", &weights)?;
//!
//! // Execute matmul (runs on CPU with naive O(n^3) algorithm)
//! let result = backend.matmul(&a, &b, m, k, n)?;
//! ```

#![allow(clippy::many_single_char_names)]
#![allow(clippy::missing_fields_in_debug)]

use super::backend::{BackendResult, ComputeBackend};
use std::collections::HashMap;

/// Mock backend that runs on CPU for testing GPU host code
///
/// This backend stores weights in CPU memory and executes all operations
/// using naive CPU algorithms. It's designed for correctness (not speed),
/// making it ideal for testing GPU orchestration logic without GPU hardware.
///
/// ## Features
///
/// - Always available (no GPU required)
/// - Stores F32 and quantized weights in `HashMap`
/// - Naive O(n^3) matmul for correctness verification
///
/// ## Usage
///
/// ```rust,ignore
/// use realizar::gpu::mock_backend::MockBackend;
/// use realizar::gpu::backend::ComputeBackend;
///
/// let mut backend = MockBackend::new(0)?;
///
/// // Load weights
/// backend.load_weights("fc1", &weights)?;
///
/// // Compute
/// let output = backend.matmul_cached("fc1", &input, 1, 768, 3072)?;
/// ```
pub struct MockBackend {
    /// F32 weight storage
    weights: HashMap<String, Vec<f32>>,
    /// Quantized weight storage (raw bytes)
    quantized_weights: HashMap<String, Vec<u8>>,
    /// Quantization types for each quantized weight
    quantized_types: HashMap<String, u32>,
    /// Mock device name
    device_name: String,
}

impl MockBackend {
    /// Create a new mock backend with default configuration
    #[must_use]
    pub fn new_mock() -> Self {
        Self {
            weights: HashMap::new(),
            quantized_weights: HashMap::new(),
            quantized_types: HashMap::new(),
            device_name: "MockGPU (CPU fallback)".to_string(),
        }
    }

    /// Get a reference to stored F32 weights
    #[must_use]
    pub fn get_weights(&self, name: &str) -> Option<&Vec<f32>> {
        self.weights.get(name)
    }

    /// Get a reference to stored quantized weights
    #[must_use]
    pub fn get_quantized_weights(&self, name: &str) -> Option<&Vec<u8>> {
        self.quantized_weights.get(name)
    }

    /// Get the quantization type for a weight
    #[must_use]
    pub fn get_quant_type(&self, name: &str) -> Option<u32> {
        self.quantized_types.get(name).copied()
    }
}

impl ComputeBackend for MockBackend {
    fn is_available() -> bool
    where
        Self: Sized,
    {
        true // Mock backend is always available
    }

    fn new(_device_id: u32) -> BackendResult<Self>
    where
        Self: Sized,
    {
        Ok(Self::new_mock())
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
            .insert(name.to_string(), data.to_vec());
        self.quantized_types.insert(name.to_string(), qtype);
        Ok(len)
    }

    fn has_weights(&self, name: &str) -> bool {
        self.weights.contains_key(name) || self.quantized_weights.contains_key(name)
    }

    fn clear_weights(&mut self) {
        self.weights.clear();
        self.quantized_weights.clear();
        self.quantized_types.clear();
    }

    fn cached_weight_count(&self) -> usize {
        self.weights.len() + self.quantized_weights.len()
    }

    fn matmul(&mut self, a: &[f32], b: &[f32], m: u32, k: u32, n: u32) -> BackendResult<Vec<f32>> {
        let (m, k, n) = (m as usize, k as usize, n as usize);

        // Validate dimensions
        if a.len() != m * k {
            return Err(format!(
                "Matrix A has {} elements, expected {}*{}={}",
                a.len(),
                m,
                k,
                m * k
            )
            .into());
        }
        if b.len() != k * n {
            return Err(format!(
                "Matrix B has {} elements, expected {}*{}={}",
                b.len(),
                k,
                n,
                k * n
            )
            .into());
        }

        // Naive O(n^3) matmul for correctness
        let mut c = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for l in 0..k {
                    sum += a[i * k + l] * b[l * n + j];
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
        let (m, k, n) = (m as usize, k as usize, n as usize);

        // Get cached weight
        let weight = self
            .weights
            .get(weight_name)
            .ok_or_else(|| format!("Weight '{}' not found", weight_name))?
            .clone();

        // Validate input dimensions
        if x.len() != m * k {
            return Err(format!(
                "Input has {} elements, expected {}*{}={}",
                x.len(),
                m,
                k,
                m * k
            )
            .into());
        }

        // Weight is [k, n] stored row-major
        // Compute C = X @ W where X is [m, k], W is [k, n]
        let mut c = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for p in 0..k {
                    sum += x[i * k + p] * weight[p * n + j];
                }
                c[i * n + j] = sum;
            }
        }

        Ok(c)
    }

    fn q4k_gemv_cached(
        &mut self,
        weight_name: &str,
        input: &[f32],
        n: u32,
        k: u32,
    ) -> BackendResult<Vec<f32>> {
        let (n, k) = (n as usize, k as usize);

        // Verify weight exists and is quantized
        if !self.quantized_weights.contains_key(weight_name) {
            return Err(format!("Quantized weight '{}' not found", weight_name).into());
        }

        // Verify quantization type (Q4_K = 3 in GGML, but we accept various types for mock)
        let _qtype = self
            .quantized_types
            .get(weight_name)
            .ok_or_else(|| format!("Quantization type not found for weight '{}'", weight_name))?;

        // Validate input dimensions
        if input.len() != k {
            return Err(format!("Input has {} elements, expected {}", input.len(), k).into());
        }

        // Mock implementation: return zeros of correct size
        // A real implementation would dequantize and compute
        // For testing purposes, we return realistic-looking values
        // by computing a simple weighted sum
        let mut output = vec![0.0f32; n];

        // Generate deterministic but non-trivial output for testing
        // This helps catch bugs where outputs are checked but not computed
        let input_sum: f32 = input.iter().sum();
        let scale = input_sum / (k as f32).max(1.0);

        for i in 0..n {
            // Deterministic pattern based on index and input
            output[i] = scale * ((i % 7) as f32 - 3.0) * 0.1;
        }

        Ok(output)
    }

    fn synchronize(&self) -> BackendResult<()> {
        // Mock backend is synchronous, nothing to wait for
        Ok(())
    }
}

// Implement Send for thread-safety (required by trait)
// SAFETY: MockBackend only contains thread-safe types (HashMap, String, Vec)
unsafe impl Send for MockBackend {}

impl std::fmt::Debug for MockBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MockBackend")
            .field("device_name", &self.device_name)
            .field("num_weights", &self.weights.len())
            .field("num_quantized_weights", &self.quantized_weights.len())
            .finish()
    }
}

include!("mock_backend_load.rs");
