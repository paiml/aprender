//! Phase 43 - GpuExecutorTrait for Dependency Injection
//!
//! Abstracts GPU execution to enable testing without actual CUDA hardware.
//!
//! # Architecture
//!
//! ```text
//! GpuModel
//!    │
//!    └─► Box<dyn GpuExecutorTrait>
//!              │
//!              ├─► CudaExecutorAdapter (production)
//!              └─► MockExecutor (testing)
//! ```
//!
//! # Coverage Impact
//!
//! This trait enables testing of:
//! - `GpuModel::forward()` - Full forward pass logic
//! - `GpuModel::generate()` - Token generation flow
//! - Layer-by-layer computation verification

use crate::error::Result;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

/// Trait for GPU execution backends
///
/// Implementations must be Send + Sync to allow model transfer between threads
/// and safe access in `Arc<RwLock<GpuModel>>` contexts.
pub trait GpuExecutorTrait: Send + Sync {
    /// Perform matrix multiplication: C = A @ B
    ///
    /// # Arguments
    ///
    /// * `a` - Left matrix [m, k]
    /// * `b` - Right matrix [k, n]
    /// * `m` - Rows in A
    /// * `k` - Columns in A / Rows in B
    /// * `n` - Columns in B
    ///
    /// # Returns
    ///
    /// Result matrix [m, n]
    fn matmul(&mut self, a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Result<Vec<f32>>;

    /// Check if GPU backend is available
    fn is_available(&self) -> bool;

    /// Get backend name for debugging
    fn name(&self) -> &str;

    /// Synchronize execution (wait for pending operations)
    fn synchronize(&self) -> Result<()>;

    /// Perform matrix multiplication with transposed B: C = A @ B^T
    ///
    /// # Arguments
    ///
    /// * `a` - Left matrix [m, k]
    /// * `b` - Right matrix [n, k] (will be transposed to [k, n])
    /// * `m` - Rows in A
    /// * `k` - Columns in A / Columns in B (before transpose)
    /// * `n` - Rows in B (becomes columns after transpose)
    ///
    /// # Returns
    ///
    /// Result matrix [m, n]
    fn matmul_transpose_b(
        &mut self,
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>>;
}

/// Call record for MockExecutor
#[derive(Debug, Clone, PartialEq)]
pub enum ExecutorCall {
    /// Matrix multiplication call
    Matmul {
        /// Input A dimensions
        a_len: usize,
        /// Input B dimensions
        b_len: usize,
        /// M dimension
        m: usize,
        /// K dimension
        k: usize,
        /// N dimension
        n: usize,
    },
    /// Matrix multiplication with transposed B call
    MatmulTransposeB {
        /// Input A dimensions
        a_len: usize,
        /// Input B dimensions (before transpose)
        b_len: usize,
        /// M dimension
        m: usize,
        /// K dimension
        k: usize,
        /// N dimension
        n: usize,
    },
    /// Synchronize call
    Synchronize,
}

/// Mock executor for testing GpuModel without CUDA
///
/// Records all calls for verification and returns configurable results.
/// Uses interior mutability (Mutex) to allow Sync trait implementation.
pub struct MockExecutor {
    /// Name of this mock
    name: String,
    /// Recorded calls (protected by mutex for thread-safety)
    calls: Mutex<Vec<ExecutorCall>>,
    /// Counter for unique call IDs
    call_counter: AtomicUsize,
    /// Whether to simulate availability
    available: bool,
    /// Custom matmul result (if None, returns zeros)
    matmul_result: Option<Vec<f32>>,
    /// Whether matmul should fail
    matmul_should_fail: bool,
}

impl MockExecutor {
    /// Create a new mock executor
    #[must_use]
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            calls: Mutex::new(Vec::new()),
            call_counter: AtomicUsize::new(0),
            available: true,
            matmul_result: None,
            matmul_should_fail: false,
        }
    }

    /// Create mock that simulates unavailability
    #[must_use]
    pub fn unavailable(name: &str) -> Self {
        Self {
            name: name.to_string(),
            calls: Mutex::new(Vec::new()),
            call_counter: AtomicUsize::new(0),
            available: false,
            matmul_result: None,
            matmul_should_fail: false,
        }
    }

    /// Set custom matmul result
    #[must_use]
    pub fn with_matmul_result(mut self, result: Vec<f32>) -> Self {
        self.matmul_result = Some(result);
        self
    }

    /// Configure matmul to fail
    #[must_use]
    pub fn with_matmul_failure(mut self) -> Self {
        self.matmul_should_fail = true;
        self
    }

    /// Acquire the calls lock, recovering from poison if needed.
    ///
    /// Centralizes the resource-acquire pattern for the `calls` mutex.
    /// All call-recording and call-querying methods delegate here.
    fn lock_calls(&self) -> std::sync::MutexGuard<'_, Vec<ExecutorCall>> {
        self.calls
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Record a call and return the configured matmul response.
    ///
    /// Consolidates the record → check-failure → respond pattern
    /// shared by `matmul` and `matmul_transpose_b`.
    fn record_call_and_respond(&self, call: ExecutorCall, output_len: usize) -> Result<Vec<f32>> {
        self.lock_calls().push(call);
        self.call_counter.fetch_add(1, Ordering::SeqCst);

        if self.matmul_should_fail {
            return Err(crate::error::RealizarError::GpuError {
                reason: "MockExecutor configured to fail".to_string(),
            });
        }

        if let Some(ref result) = self.matmul_result {
            Ok(result.clone())
        } else {
            Ok(vec![0.0f32; output_len])
        }
    }

    /// Get all recorded calls (cloned for thread-safety)
    #[must_use]
    pub fn calls(&self) -> Vec<ExecutorCall> {
        self.lock_calls().clone()
    }

    /// Get number of calls
    #[must_use]
    pub fn call_count(&self) -> usize {
        self.lock_calls().len()
    }

    /// Get number of matmul calls
    #[must_use]
    pub fn matmul_count(&self) -> usize {
        self.lock_calls()
            .iter()
            .filter(|c| matches!(c, ExecutorCall::Matmul { .. }))
            .count()
    }

    /// Clear recorded calls
    pub fn clear_calls(&self) {
        self.lock_calls().clear();
        self.call_counter.store(0, Ordering::SeqCst);
    }

    /// Check if specific call was made
    #[must_use]
    pub fn has_call(&self, call: &ExecutorCall) -> bool {
        self.lock_calls().contains(call)
    }

    /// Get last call (cloned for thread-safety)
    #[must_use]
    pub fn last_call(&self) -> Option<ExecutorCall> {
        self.lock_calls().last().cloned()
    }
}

impl GpuExecutorTrait for MockExecutor {
    #[allow(clippy::many_single_char_names)]
    fn matmul(&mut self, a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Result<Vec<f32>> {
        self.record_call_and_respond(
            ExecutorCall::Matmul {
                a_len: a.len(),
                b_len: b.len(),
                m,
                k,
                n,
            },
            m * n,
        )
    }

    fn is_available(&self) -> bool {
        self.available
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn synchronize(&self) -> Result<()> {
        // Record sync call (need interior mutability for &self)
        // For testing, we skip recording in synchronize since it takes &self
        Ok(())
    }

    #[allow(clippy::many_single_char_names)]
    fn matmul_transpose_b(
        &mut self,
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>> {
        self.record_call_and_respond(
            ExecutorCall::MatmulTransposeB {
                a_len: a.len(),
                b_len: b.len(),
                m,
                k,
                n,
            },
            m * n,
        )
    }
}

impl std::fmt::Debug for MockExecutor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MockExecutor")
            .field("name", &self.name)
            .field("calls", &self.lock_calls().len())
            .field("call_counter", &self.call_counter.load(Ordering::SeqCst))
            .field("available", &self.available)
            .field("matmul_result", &self.matmul_result.is_some())
            .field("matmul_should_fail", &self.matmul_should_fail)
            .finish()
    }
}

/// CPU-based executor for testing and fallback
///
/// Implements actual matrix multiplication on CPU.
pub struct CpuExecutor {
    name: String,
}

impl CpuExecutor {
    /// Create a new CPU executor
    #[must_use]
    pub fn new() -> Self {
        Self {
            name: "CpuExecutor".to_string(),
        }
    }
}

impl Default for CpuExecutor {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuExecutorTrait for CpuExecutor {
    #[allow(clippy::many_single_char_names)]
    fn matmul(&mut self, a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Result<Vec<f32>> {
        // Validate dimensions
        if a.len() != m * k {
            return Err(crate::error::RealizarError::InvalidShape {
                reason: format!("A size {} != m*k {}", a.len(), m * k),
            });
        }
        if b.len() != k * n {
            return Err(crate::error::RealizarError::InvalidShape {
                reason: format!("B size {} != k*n {}", b.len(), k * n),
            });
        }

        // Naive matmul (for correctness, not performance)
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

    fn is_available(&self) -> bool {
        true // CPU is always available
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn synchronize(&self) -> Result<()> {
        Ok(()) // No-op for CPU
    }

    #[allow(clippy::many_single_char_names)]
    fn matmul_transpose_b(
        &mut self,
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>> {
        // Validate dimensions: A is [m, k], B is [n, k] (to be transposed to [k, n])
        if a.len() != m * k {
            return Err(crate::error::RealizarError::InvalidShape {
                reason: format!("A size {} != m*k {}", a.len(), m * k),
            });
        }
        if b.len() != n * k {
            return Err(crate::error::RealizarError::InvalidShape {
                reason: format!("B size {} != n*k {}", b.len(), n * k),
            });
        }

        // Naive matmul with B transposed: C = A @ B^T
        // A is [m, k], B is [n, k], C is [m, n]
        // B^T is [k, n], so C[i,j] = sum over p of A[i,p] * B[j,p]
        let mut c = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for p in 0..k {
                    sum += a[i * k + p] * b[j * k + p]; // Note: b[j, p] not b[p, j]
                }
                c[i * n + j] = sum;
            }
        }
        Ok(c)
    }
}

impl std::fmt::Debug for CpuExecutor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CpuExecutor")
            .field("name", &self.name)
            .finish()
    }
}

include!("executor_mock.rs");
