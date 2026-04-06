//! SIMD execution backend.
//!
//! Executes SIMD-accelerated vector operations using pepita's SIMD primitives.
//!
//! ## Supported Operations
//!
//! - Vector addition (f32, f64)
//! - Vector multiplication (f32)
//! - Dot product (f32)
//! - Matrix multiplication (f32)
//!
//! ## Example
//!
//! ```rust,ignore
//! use repartir::executor::simd::{SimdExecutor, SimdOperation, SimdTask};
//!
//! let executor = SimdExecutor::new();
//!
//! // Create a vector addition task
//! let task = SimdTask::vadd_f32(
//!     vec![1.0, 2.0, 3.0, 4.0],
//!     vec![5.0, 6.0, 7.0, 8.0],
//! );
//!
//! let result = executor.execute_simd(task).await?;
//! assert_eq!(result.output_f32(), &[6.0, 8.0, 10.0, 12.0]);
//! ```

use crate::error::{RepartirError, Result};
use crate::executor::{BoxFuture, Executor};
use crate::task::{ExecutionResult, Task, TaskId};
use pepita::simd::{MatrixOps, SimdCapabilities, SimdOps};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{debug, info};

// ============================================================================
// SIMD OPERATION TYPES
// ============================================================================

/// SIMD operation type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdOperation {
    /// Vector addition (f32).
    VaddF32,
    /// Vector addition (f64).
    VaddF64,
    /// Vector multiplication (f32).
    VmulF32,
    /// Dot product (f32).
    DotF32,
    /// Matrix multiplication (f32).
    MatMulF32,
}

impl std::fmt::Display for SimdOperation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::VaddF32 => write!(f, "vadd_f32"),
            Self::VaddF64 => write!(f, "vadd_f64"),
            Self::VmulF32 => write!(f, "vmul_f32"),
            Self::DotF32 => write!(f, "dot_f32"),
            Self::MatMulF32 => write!(f, "matmul_f32"),
        }
    }
}

// ============================================================================
// SIMD TASK
// ============================================================================

/// SIMD task configuration.
#[derive(Debug, Clone)]
pub struct SimdTask {
    /// Task ID.
    id: TaskId,
    /// Operation to perform.
    operation: SimdOperation,
    /// First input (f32).
    input_a_f32: Vec<f32>,
    /// Second input (f32).
    input_b_f32: Vec<f32>,
    /// First input (f64).
    input_a_f64: Vec<f64>,
    /// Second input (f64).
    input_b_f64: Vec<f64>,
    /// Matrix dimensions (`rows_a`, `cols_a`, `cols_b`) for matmul.
    matrix_dims: Option<(usize, usize, usize)>,
}

impl SimdTask {
    /// Create a vector addition task (f32).
    #[must_use]
    pub fn vadd_f32(a: Vec<f32>, b: Vec<f32>) -> Self {
        Self {
            id: TaskId::new(),
            operation: SimdOperation::VaddF32,
            input_a_f32: a,
            input_b_f32: b,
            input_a_f64: Vec::new(),
            input_b_f64: Vec::new(),
            matrix_dims: None,
        }
    }

    /// Create a vector addition task (f64).
    #[must_use]
    pub fn vadd_f64(a: Vec<f64>, b: Vec<f64>) -> Self {
        Self {
            id: TaskId::new(),
            operation: SimdOperation::VaddF64,
            input_a_f32: Vec::new(),
            input_b_f32: Vec::new(),
            input_a_f64: a,
            input_b_f64: b,
            matrix_dims: None,
        }
    }

    /// Create a vector multiplication task (f32).
    #[must_use]
    pub fn vmul_f32(a: Vec<f32>, b: Vec<f32>) -> Self {
        Self {
            id: TaskId::new(),
            operation: SimdOperation::VmulF32,
            input_a_f32: a,
            input_b_f32: b,
            input_a_f64: Vec::new(),
            input_b_f64: Vec::new(),
            matrix_dims: None,
        }
    }

    /// Create a dot product task (f32).
    #[must_use]
    pub fn dot_f32(a: Vec<f32>, b: Vec<f32>) -> Self {
        Self {
            id: TaskId::new(),
            operation: SimdOperation::DotF32,
            input_a_f32: a,
            input_b_f32: b,
            input_a_f64: Vec::new(),
            input_b_f64: Vec::new(),
            matrix_dims: None,
        }
    }

    /// Create a matrix multiplication task (f32).
    ///
    /// Matrix A is `rows_a` x `cols_a`, Matrix B is `cols_a` x `cols_b`.
    #[must_use]
    pub fn matmul_f32(
        a: Vec<f32>,
        b: Vec<f32>,
        rows_a: usize,
        cols_a: usize,
        cols_b: usize,
    ) -> Self {
        Self {
            id: TaskId::new(),
            operation: SimdOperation::MatMulF32,
            input_a_f32: a,
            input_b_f32: b,
            input_a_f64: Vec::new(),
            input_b_f64: Vec::new(),
            matrix_dims: Some((rows_a, cols_a, cols_b)),
        }
    }

    /// Get task ID.
    #[must_use]
    pub const fn id(&self) -> TaskId {
        self.id
    }

    /// Get operation type.
    #[must_use]
    pub const fn operation(&self) -> SimdOperation {
        self.operation
    }

    /// Validate task inputs.
    fn validate(&self) -> Result<()> {
        match self.operation {
            SimdOperation::VaddF32 | SimdOperation::VmulF32 | SimdOperation::DotF32 => {
                if self.input_a_f32.len() != self.input_b_f32.len() {
                    return Err(RepartirError::InvalidTask {
                        reason: format!(
                            "Input vectors must have same length: {} != {}",
                            self.input_a_f32.len(),
                            self.input_b_f32.len()
                        ),
                    });
                }
                if self.input_a_f32.is_empty() {
                    return Err(RepartirError::InvalidTask {
                        reason: "Input vectors cannot be empty".to_string(),
                    });
                }
            }
            SimdOperation::VaddF64 => {
                if self.input_a_f64.len() != self.input_b_f64.len() {
                    return Err(RepartirError::InvalidTask {
                        reason: format!(
                            "Input vectors must have same length: {} != {}",
                            self.input_a_f64.len(),
                            self.input_b_f64.len()
                        ),
                    });
                }
                if self.input_a_f64.is_empty() {
                    return Err(RepartirError::InvalidTask {
                        reason: "Input vectors cannot be empty".to_string(),
                    });
                }
            }
            SimdOperation::MatMulF32 => {
                let Some((rows_a, cols_a, cols_b)) = self.matrix_dims else {
                    return Err(RepartirError::InvalidTask {
                        reason: "Matrix dimensions not specified".to_string(),
                    });
                };
                if self.input_a_f32.len() != rows_a * cols_a {
                    return Err(RepartirError::InvalidTask {
                        reason: format!(
                            "Matrix A size mismatch: {} != {} * {}",
                            self.input_a_f32.len(),
                            rows_a,
                            cols_a
                        ),
                    });
                }
                if self.input_b_f32.len() != cols_a * cols_b {
                    return Err(RepartirError::InvalidTask {
                        reason: format!(
                            "Matrix B size mismatch: {} != {} * {}",
                            self.input_b_f32.len(),
                            cols_a,
                            cols_b
                        ),
                    });
                }
            }
        }
        Ok(())
    }
}

// ============================================================================
// SIMD RESULT
// ============================================================================

/// Result of a SIMD operation.
#[derive(Debug, Clone)]
pub struct SimdResult {
    /// Task ID.
    task_id: TaskId,
    /// Output (f32).
    output_f32: Vec<f32>,
    /// Output (f64).
    output_f64: Vec<f64>,
    /// Scalar result (for dot product).
    scalar_f32: Option<f32>,
    /// Execution duration.
    duration: Duration,
    /// Elements processed.
    elements: usize,
    /// Throughput (elements/second).
    throughput: f64,
}

impl SimdResult {
    /// Get task ID.
    #[must_use]
    pub const fn task_id(&self) -> TaskId {
        self.task_id
    }

    /// Get f32 output vector.
    #[must_use]
    pub fn output_f32(&self) -> &[f32] {
        &self.output_f32
    }

    /// Get f64 output vector.
    #[must_use]
    pub fn output_f64(&self) -> &[f64] {
        &self.output_f64
    }

    /// Get scalar result (dot product).
    #[must_use]
    pub const fn scalar_f32(&self) -> Option<f32> {
        self.scalar_f32
    }

    /// Get execution duration.
    #[must_use]
    pub const fn duration(&self) -> Duration {
        self.duration
    }

    /// Get elements processed.
    #[must_use]
    pub const fn elements(&self) -> usize {
        self.elements
    }

    /// Get throughput in elements/second.
    #[must_use]
    pub const fn throughput(&self) -> f64 {
        self.throughput
    }
}

// ============================================================================
// SIMD EXECUTOR
// ============================================================================

/// SIMD executor metrics.
#[derive(Debug, Default)]
pub struct SimdMetrics {
    /// Total operations executed.
    pub operations: AtomicU64,
    /// Total elements processed.
    pub elements_processed: AtomicU64,
    /// Total execution time (nanoseconds).
    pub total_time_ns: AtomicU64,
}

impl SimdMetrics {
    /// Get operations count.
    #[must_use]
    pub fn operations(&self) -> u64 {
        self.operations.load(Ordering::Relaxed)
    }

    /// Get elements processed.
    #[must_use]
    pub fn elements_processed(&self) -> u64 {
        self.elements_processed.load(Ordering::Relaxed)
    }

    /// Get average throughput.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn avg_throughput(&self) -> f64 {
        let elements = self.elements_processed.load(Ordering::Relaxed) as f64;
        let time_ns = self.total_time_ns.load(Ordering::Relaxed) as f64;
        if time_ns > 0.0 {
            elements / (time_ns / 1_000_000_000.0)
        } else {
            0.0
        }
    }
}

/// SIMD executor for vectorized operations.
///
/// Uses pepita's SIMD primitives for hardware-accelerated vector operations.
pub struct SimdExecutor {
    /// SIMD operations provider.
    ops: SimdOps,
    /// Matrix operations provider.
    matrix_ops: MatrixOps,
    /// Detected capabilities.
    caps: SimdCapabilities,
    /// Execution metrics.
    metrics: Arc<SimdMetrics>,
}

impl SimdExecutor {
    /// Create a new SIMD executor.
    #[must_use]
    pub fn new() -> Self {
        let caps = SimdCapabilities::detect();
        info!(
            "SimdExecutor initialized: {} ({}-bit vectors)",
            caps.description(),
            caps.best_vector_width()
        );
        Self {
            ops: SimdOps::new(),
            matrix_ops: MatrixOps::new(),
            caps,
            metrics: Arc::new(SimdMetrics::default()),
        }
    }

    /// Get SIMD capabilities.
    #[must_use]
    pub const fn capabilities(&self) -> &SimdCapabilities {
        &self.caps
    }

    /// Get execution metrics.
    #[must_use]
    pub fn metrics(&self) -> Arc<SimdMetrics> {
        Arc::clone(&self.metrics)
    }

    /// Get best vector width in bits.
    #[must_use]
    pub const fn vector_width(&self) -> u32 {
        self.caps.best_vector_width()
    }

    /// Check if SIMD is available.
    #[must_use]
    pub const fn has_simd(&self) -> bool {
        self.caps.has_simd()
    }

    /// Execute a SIMD task.
    ///
    /// # Errors
    ///
    /// Returns an error if task validation fails.
    ///
    /// # Panics
    ///
    /// Panics if `matrix_dims` is `None` for a `MatMulF32` operation (should
    /// not happen after validation).
    #[allow(clippy::unused_async)]
    pub async fn execute_simd(&self, task: SimdTask) -> Result<SimdResult> {
        task.validate()?;

        let start = Instant::now();
        let task_id = task.id();
        let operation = task.operation();

        debug!("Executing SIMD task {}: {}", task_id, operation);

        let (output_f32, output_f64, scalar_f32, elements) = match operation {
            SimdOperation::VaddF32 => {
                let len = task.input_a_f32.len();
                let mut output = vec![0.0f32; len];
                self.ops.vadd_f32(&task.input_a_f32, &task.input_b_f32, &mut output);
                (output, Vec::new(), None, len)
            }
            SimdOperation::VaddF64 => {
                let len = task.input_a_f64.len();
                let mut output = vec![0.0f64; len];
                self.ops.vadd_f64(&task.input_a_f64, &task.input_b_f64, &mut output);
                (Vec::new(), output, None, len)
            }
            SimdOperation::VmulF32 => {
                let len = task.input_a_f32.len();
                let mut output = vec![0.0f32; len];
                self.ops.vmul_f32(&task.input_a_f32, &task.input_b_f32, &mut output);
                (output, Vec::new(), None, len)
            }
            SimdOperation::DotF32 => {
                let len = task.input_a_f32.len();
                let result = self.ops.dot_f32(&task.input_a_f32, &task.input_b_f32);
                (Vec::new(), Vec::new(), Some(result), len)
            }
            SimdOperation::MatMulF32 => {
                let (rows_a, cols_a, cols_b) =
                    task.matrix_dims.ok_or_else(|| RepartirError::InvalidTask {
                        reason: "matrix_dims required for MatMulF32".to_string(),
                    })?;
                let mut output = vec![0.0f32; rows_a * cols_b];
                self.matrix_ops.matmul_f32(
                    &task.input_a_f32,
                    &task.input_b_f32,
                    &mut output,
                    rows_a,
                    cols_a,
                    cols_b,
                );
                let elements = rows_a * cols_a + cols_a * cols_b;
                (output, Vec::new(), None, elements)
            }
        };

        let duration = start.elapsed();
        #[allow(clippy::cast_precision_loss)]
        let throughput =
            if duration.as_nanos() > 0 { elements as f64 / duration.as_secs_f64() } else { 0.0 };

        // Update metrics
        self.metrics.operations.fetch_add(1, Ordering::Relaxed);
        #[allow(clippy::cast_possible_truncation)]
        self.metrics.elements_processed.fetch_add(elements as u64, Ordering::Relaxed);
        #[allow(clippy::cast_possible_truncation)]
        self.metrics.total_time_ns.fetch_add(duration.as_nanos() as u64, Ordering::Relaxed);

        debug!(
            "SIMD task {} completed in {:?} ({:.2}M elem/s)",
            task_id,
            duration,
            throughput / 1_000_000.0
        );

        Ok(SimdResult {
            task_id,
            output_f32,
            output_f64,
            scalar_f32,
            duration,
            elements,
            throughput,
        })
    }
}

impl Default for SimdExecutor {
    fn default() -> Self {
        Self::new()
    }
}

impl Executor for SimdExecutor {
    fn execute(&self, _task: Task) -> BoxFuture<'_, Result<ExecutionResult>> {
        Box::pin(async move {
            // SIMD executor only handles SIMD backend tasks
            // For general tasks, return an error suggesting CPU executor
            Err(RepartirError::InvalidTask {
                reason: "SimdExecutor only handles SIMD operations. Use execute_simd() method or CpuExecutor for binary tasks.".to_string(),
            })
        })
    }

    fn capacity(&self) -> usize {
        // SIMD operates on the local CPU, capacity is based on vector width
        self.caps.best_f32_width()
    }

    fn name(&self) -> &'static str {
        "SIMD"
    }
}

#[allow(clippy::missing_fields_in_debug)]
impl std::fmt::Debug for SimdExecutor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SimdExecutor")
            .field("capabilities", &self.caps.description())
            .field("vector_width", &self.caps.best_vector_width())
            .field("operations", &self.metrics.operations())
            .finish()
    }
}

// ============================================================================
// TESTS - EXTREME TDD
// ============================================================================

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::disallowed_methods,
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::uninlined_format_args,
    clippy::unreadable_literal,
    clippy::panic
)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------------
    // SimdOperation Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_simd_operation_display() {
        assert_eq!(format!("{}", SimdOperation::VaddF32), "vadd_f32");
        assert_eq!(format!("{}", SimdOperation::VaddF64), "vadd_f64");
        assert_eq!(format!("{}", SimdOperation::VmulF32), "vmul_f32");
        assert_eq!(format!("{}", SimdOperation::DotF32), "dot_f32");
        assert_eq!(format!("{}", SimdOperation::MatMulF32), "matmul_f32");
    }

    #[test]
    fn test_simd_operation_eq() {
        assert_eq!(SimdOperation::VaddF32, SimdOperation::VaddF32);
        assert_ne!(SimdOperation::VaddF32, SimdOperation::VmulF32);
    }

    #[test]
    fn test_simd_operation_copy() {
        let op = SimdOperation::DotF32;
        let op2 = op;
        assert_eq!(op, op2);
    }

    // ------------------------------------------------------------------------
    // SimdTask Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_simd_task_vadd_f32() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let task = SimdTask::vadd_f32(a.clone(), b.clone());

        assert_eq!(task.operation(), SimdOperation::VaddF32);
        assert_eq!(task.input_a_f32, a);
        assert_eq!(task.input_b_f32, b);
        assert!(task.validate().is_ok());
    }

    #[test]
    fn test_simd_task_vadd_f64() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let task = SimdTask::vadd_f64(a.clone(), b.clone());

        assert_eq!(task.operation(), SimdOperation::VaddF64);
        assert_eq!(task.input_a_f64, a);
        assert_eq!(task.input_b_f64, b);
        assert!(task.validate().is_ok());
    }

    #[test]
    fn test_simd_task_vmul_f32() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let task = SimdTask::vmul_f32(a, b);

        assert_eq!(task.operation(), SimdOperation::VmulF32);
        assert!(task.validate().is_ok());
    }

    #[test]
    fn test_simd_task_dot_f32() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let task = SimdTask::dot_f32(a, b);

        assert_eq!(task.operation(), SimdOperation::DotF32);
        assert!(task.validate().is_ok());
    }

    #[test]
    fn test_simd_task_matmul_f32() {
        // 2x3 * 3x2 = 2x2
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let task = SimdTask::matmul_f32(a, b, 2, 3, 2);

        assert_eq!(task.operation(), SimdOperation::MatMulF32);
        assert_eq!(task.matrix_dims, Some((2, 3, 2)));
        assert!(task.validate().is_ok());
    }

    #[test]
    fn test_simd_task_validate_empty() {
        let task = SimdTask::vadd_f32(vec![], vec![]);
        assert!(task.validate().is_err());
    }

    #[test]
    fn test_simd_task_validate_length_mismatch() {
        let task = SimdTask::vadd_f32(vec![1.0, 2.0], vec![1.0]);
        assert!(task.validate().is_err());
    }

    #[test]
    fn test_simd_task_validate_f64_empty() {
        let task = SimdTask::vadd_f64(vec![], vec![]);
        assert!(task.validate().is_err());
    }

    #[test]
    fn test_simd_task_validate_f64_length_mismatch() {
        let task = SimdTask::vadd_f64(vec![1.0, 2.0], vec![1.0]);
        assert!(task.validate().is_err());
    }

    #[test]
    fn test_simd_task_validate_matmul_dimensions() {
        // Wrong size for matrix A
        let task = SimdTask::matmul_f32(vec![1.0, 2.0], vec![1.0, 2.0, 3.0, 4.0], 2, 2, 2);
        assert!(task.validate().is_err());
    }

    #[test]
    fn test_simd_task_validate_matmul_b_dimensions() {
        // Wrong size for matrix B
        let task = SimdTask::matmul_f32(vec![1.0, 2.0, 3.0, 4.0], vec![1.0, 2.0], 2, 2, 2);
        assert!(task.validate().is_err());
    }

    #[test]
    fn test_simd_task_id() {
        let task = SimdTask::vadd_f32(vec![1.0], vec![1.0]);
        let id = task.id();
        assert!(!id.to_string().is_empty());
    }

    // ------------------------------------------------------------------------
    // SimdResult Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_simd_result_output_f32() {
        let result = SimdResult {
            task_id: TaskId::new(),
            output_f32: vec![1.0, 2.0, 3.0],
            output_f64: vec![],
            scalar_f32: None,
            duration: Duration::from_millis(10),
            elements: 3,
            throughput: 300.0,
        };

        assert_eq!(result.output_f32(), &[1.0, 2.0, 3.0]);
        assert!(result.output_f64().is_empty());
        assert!(result.scalar_f32().is_none());
        assert_eq!(result.elements(), 3);
    }

    #[test]
    fn test_simd_result_scalar() {
        let result = SimdResult {
            task_id: TaskId::new(),
            output_f32: vec![],
            output_f64: vec![],
            scalar_f32: Some(42.0),
            duration: Duration::from_millis(5),
            elements: 10,
            throughput: 2000.0,
        };

        assert_eq!(result.scalar_f32(), Some(42.0));
        assert!(result.output_f32().is_empty());
    }

    #[test]
    fn test_simd_result_duration() {
        let result = SimdResult {
            task_id: TaskId::new(),
            output_f32: vec![],
            output_f64: vec![],
            scalar_f32: None,
            duration: Duration::from_secs(1),
            elements: 1000,
            throughput: 1000.0,
        };

        assert_eq!(result.duration(), Duration::from_secs(1));
        assert_eq!(result.throughput(), 1000.0);
    }

    // ------------------------------------------------------------------------
    // SimdMetrics Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_simd_metrics_default() {
        let metrics = SimdMetrics::default();
        assert_eq!(metrics.operations(), 0);
        assert_eq!(metrics.elements_processed(), 0);
        assert_eq!(metrics.avg_throughput(), 0.0);
    }

    #[test]
    fn test_simd_metrics_update() {
        let metrics = SimdMetrics::default();
        metrics.operations.fetch_add(1, Ordering::Relaxed);
        metrics.elements_processed.fetch_add(1000, Ordering::Relaxed);
        metrics.total_time_ns.fetch_add(1_000_000, Ordering::Relaxed);

        assert_eq!(metrics.operations(), 1);
        assert_eq!(metrics.elements_processed(), 1000);
        assert!(metrics.avg_throughput() > 0.0);
    }

    // ------------------------------------------------------------------------
    // SimdExecutor Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_simd_executor_new() {
        let executor = SimdExecutor::new();
        assert!(executor.vector_width() >= 64);
        assert!(!executor.name().is_empty());
    }

    #[test]
    fn test_simd_executor_default() {
        let executor = SimdExecutor::default();
        assert!(executor.capacity() > 0);
    }

    #[test]
    fn test_simd_executor_capabilities() {
        let executor = SimdExecutor::new();
        let caps = executor.capabilities();
        assert!(caps.best_vector_width() >= 64);
    }

    #[test]
    fn test_simd_executor_metrics() {
        let executor = SimdExecutor::new();
        let metrics = executor.metrics();
        assert_eq!(metrics.operations(), 0);
    }

    #[test]
    fn test_simd_executor_name() {
        let executor = SimdExecutor::new();
        assert_eq!(executor.name(), "SIMD");
    }

    #[test]
    fn test_simd_executor_debug() {
        let executor = SimdExecutor::new();
        let debug = format!("{:?}", executor);
        assert!(debug.contains("SimdExecutor"));
        assert!(debug.contains("vector_width"));
    }

    #[tokio::test]
    async fn test_simd_executor_vadd_f32() {
        let executor = SimdExecutor::new();
        let task = SimdTask::vadd_f32(vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0]);

        let result = executor.execute_simd(task).await.unwrap();

        assert_eq!(result.output_f32(), &[6.0, 8.0, 10.0, 12.0]);
        assert_eq!(result.elements(), 4);
    }

    #[tokio::test]
    async fn test_simd_executor_vadd_f64() {
        let executor = SimdExecutor::new();
        let task = SimdTask::vadd_f64(vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0]);

        let result = executor.execute_simd(task).await.unwrap();

        assert_eq!(result.output_f64(), &[6.0, 8.0, 10.0, 12.0]);
        assert_eq!(result.elements(), 4);
    }

    #[tokio::test]
    async fn test_simd_executor_vmul_f32() {
        let executor = SimdExecutor::new();
        let task = SimdTask::vmul_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2.0, 3.0, 4.0, 5.0]);

        let result = executor.execute_simd(task).await.unwrap();

        assert_eq!(result.output_f32(), &[2.0, 6.0, 12.0, 20.0]);
    }

    #[tokio::test]
    async fn test_simd_executor_dot_f32() {
        let executor = SimdExecutor::new();
        // dot([1,2,3], [4,5,6]) = 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        let task = SimdTask::dot_f32(vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]);

        let result = executor.execute_simd(task).await.unwrap();

        assert_eq!(result.scalar_f32(), Some(32.0));
    }

    #[tokio::test]
    async fn test_simd_executor_matmul_f32() {
        let executor = SimdExecutor::new();
        // 2x2 identity * 2x2 identity = 2x2 identity
        let identity = vec![1.0, 0.0, 0.0, 1.0];
        let task = SimdTask::matmul_f32(identity.clone(), identity.clone(), 2, 2, 2);

        let result = executor.execute_simd(task).await.unwrap();

        assert_eq!(result.output_f32(), &[1.0, 0.0, 0.0, 1.0]);
    }

    #[tokio::test]
    async fn test_simd_executor_matmul_simple() {
        let executor = SimdExecutor::new();
        // [[1,2],[3,4]] * [[5,6],[7,8]] = [[19,22],[43,50]]
        // (1*5+2*7, 1*6+2*8) = (19, 22)
        // (3*5+4*7, 3*6+4*8) = (43, 50)
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let task = SimdTask::matmul_f32(a, b, 2, 2, 2);

        let result = executor.execute_simd(task).await.unwrap();

        assert_eq!(result.output_f32(), &[19.0, 22.0, 43.0, 50.0]);
    }

    #[tokio::test]
    async fn test_simd_executor_invalid_task() {
        let executor = SimdExecutor::new();
        let task = SimdTask::vadd_f32(vec![1.0], vec![1.0, 2.0]);

        let result = executor.execute_simd(task).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_simd_executor_metrics_updated() {
        let executor = SimdExecutor::new();
        let task = SimdTask::vadd_f32(vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0]);

        let _ = executor.execute_simd(task).await.unwrap();

        let metrics = executor.metrics();
        assert_eq!(metrics.operations(), 1);
        assert_eq!(metrics.elements_processed(), 4);
    }

    #[tokio::test]
    async fn test_simd_executor_large_vectors() {
        let executor = SimdExecutor::new();
        let size = 10000;
        let a: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..size).map(|i| i as f32 * 2.0).collect();

        let task = SimdTask::vadd_f32(a, b);
        let result = executor.execute_simd(task).await.unwrap();

        assert_eq!(result.elements(), size);
        assert_eq!(result.output_f32()[0], 0.0);
        assert_eq!(result.output_f32()[1], 3.0); // 1 + 2
        assert_eq!(result.output_f32()[9999], 29997.0); // 9999 + 19998
    }

    #[tokio::test]
    async fn test_simd_executor_throughput() {
        let executor = SimdExecutor::new();
        let size = 100000;
        let a: Vec<f32> = vec![1.0; size];
        let b: Vec<f32> = vec![2.0; size];

        let task = SimdTask::vadd_f32(a, b);
        let result = executor.execute_simd(task).await.unwrap();

        assert!(result.throughput() > 0.0);
        assert!(result.duration().as_nanos() > 0);
    }

    #[tokio::test]
    async fn test_simd_executor_execute_trait_method() {
        let executor = SimdExecutor::new();
        let task =
            Task::builder().binary("/bin/echo").backend(crate::task::Backend::Cpu).build().unwrap();

        let result = executor.execute(task).await;
        assert!(result.is_err());
    }

    #[test]
    fn test_simd_executor_has_simd() {
        let executor = SimdExecutor::new();
        // On x86_64 or aarch64, we should have some SIMD
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
        assert!(executor.has_simd());
    }

    #[tokio::test]
    async fn test_simd_executor_multiple_operations() {
        let executor = SimdExecutor::new();

        // Execute multiple operations
        for _ in 0..5 {
            let task = SimdTask::vadd_f32(vec![1.0, 2.0], vec![3.0, 4.0]);
            let _ = executor.execute_simd(task).await.unwrap();
        }

        let metrics = executor.metrics();
        assert_eq!(metrics.operations(), 5);
        assert_eq!(metrics.elements_processed(), 10);
    }

    #[tokio::test]
    async fn test_simd_executor_dot_product_large() {
        let executor = SimdExecutor::new();
        let size = 1024;
        let a: Vec<f32> = vec![1.0; size];
        let b: Vec<f32> = vec![1.0; size];

        let task = SimdTask::dot_f32(a, b);
        let result = executor.execute_simd(task).await.unwrap();

        assert_eq!(result.scalar_f32(), Some(1024.0));
    }

    #[tokio::test]
    async fn test_simd_executor_matmul_rectangular() {
        let executor = SimdExecutor::new();
        // 2x3 * 3x1 = 2x1
        // [[1,2,3],[4,5,6]] * [[1],[2],[3]] = [[14],[32]]
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![1.0, 2.0, 3.0];
        let task = SimdTask::matmul_f32(a, b, 2, 3, 1);

        let result = executor.execute_simd(task).await.unwrap();

        assert_eq!(result.output_f32(), &[14.0, 32.0]);
    }
}
