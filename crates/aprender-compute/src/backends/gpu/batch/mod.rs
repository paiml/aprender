//! Async GPU command batching for reduced transfer overhead
//!
//! This module provides an async API for GPU operations that batches multiple
//! operations together to minimize CPU↔GPU data transfers.
//!
//! # Motivation
//!
//! The synchronous GPU API transfers data for each operation:
//! ```text
//! vec.relu()      // Upload → GPU compute → Download
//! vec.scale(2.0)  // Upload → GPU compute → Download
//! vec.add(&other) // Upload → GPU compute → Download
//! Total: 6 transfers (3 up, 3 down)
//! ```
//!
//! The async batch API queues operations and executes them together:
//! ```text
//! batch.relu(input)
//! batch.scale(relu_out, 2.0)
//! batch.add(scaled, other)
//! batch.execute()  // Upload once → 3 GPU computes → Download once
//! Total: 2 transfers (1 up, 1 down)  // 3x reduction!
//! ```
//!
//! # Example
//!
//! ```rust,no_run
//! use trueno::backends::gpu::{GpuDevice, GpuCommandBatch};
//!
//! # async fn example() -> Result<(), String> {
//! let device = GpuDevice::new()?;
//! let mut batch = GpuCommandBatch::new(device);
//!
//! // Queue operations (no GPU execution yet)
//! let input = batch.upload(&[1.0, 2.0, -3.0, 4.0]);
//! let relu_out = batch.relu(input);
//! let scaled = batch.scale(relu_out, 2.0);
//! let other = batch.upload(&[0.5, 0.5, 0.5, 0.5]);
//! let final_out = batch.add(scaled, other);
//!
//! // Execute all operations in single batch
//! batch.execute().await?;
//!
//! // Read final result
//! let result = batch.read(final_out).await?;
//! assert_eq!(result, vec![2.5, 4.5, 0.5, 8.5]);
//! # Ok(())
//! # }
//! ```

mod execute;

pub use execute::dispatch::PipelineCache;

#[cfg(test)]
mod tests;

use super::GpuDevice;
use std::collections::HashMap;
use std::sync::Arc;
use wgpu;

/// Unique identifier for a buffer in a batch
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferId(pub(crate) usize);

/// GPU operation to be executed in a batch
#[derive(Debug)]
pub(crate) enum GpuOp {
    /// ReLU activation: max(0, x)
    Relu { input: BufferId, output: BufferId },

    /// Scalar multiplication: x * scalar
    Scale { input: BufferId, output: BufferId, scalar: f32 },

    /// Element-wise addition: a + b
    Add { a: BufferId, b: BufferId, output: BufferId },

    /// Element-wise multiplication: a * b
    Mul { a: BufferId, b: BufferId, output: BufferId },

    /// Dot product: sum(a[i] * b[i])
    Dot {
        a: BufferId,
        b: BufferId,
        output: BufferId, // Single-element buffer for result
    },

    /// Sigmoid activation: 1 / (1 + exp(-x))
    Sigmoid { input: BufferId, output: BufferId },

    /// Hyperbolic tangent: tanh(x)
    Tanh { input: BufferId, output: BufferId },

    /// Swish activation: x * sigmoid(x)
    Swish { input: BufferId, output: BufferId },

    /// GELU activation: x * Φ(x) where Φ is cumulative distribution function
    Gelu { input: BufferId, output: BufferId },

    /// Element-wise subtraction: a - b
    Sub { a: BufferId, b: BufferId, output: BufferId },

    /// Matrix multiplication: C = A × B
    /// A is M×K, B is K×N, C is M×N (all row-major)
    Matmul { a: BufferId, b: BufferId, output: BufferId, m: u32, k: u32, n: u32 },
}

/// Command batch for async GPU execution
///
/// Accumulates GPU operations and executes them together to minimize
/// CPU↔GPU data transfers.
pub struct GpuCommandBatch {
    pub(crate) device: Arc<GpuDevice>,
    pub(crate) operations: Vec<GpuOp>,
    pub(crate) buffers: HashMap<BufferId, BufferInfo>,
    pub(crate) next_buffer_id: usize,
}

/// Information about a buffer in the batch
#[derive(Debug)]
pub(crate) struct BufferInfo {
    /// Size in elements (f32)
    pub(crate) size: usize,

    /// Initial data to upload (if any)
    pub(crate) data: Option<Vec<f32>>,

    /// GPU buffer (created during execute(), or pre-existing for imported buffers).
    /// Wrapped in `Arc` to allow sharing across multiple batch executions (KAIZEN-015).
    /// When `Some`, execute() skips buffer creation (already GPU-resident).
    pub(crate) gpu_buffer: Option<Arc<wgpu::Buffer>>,
}

impl GpuCommandBatch {
    /// Create a new command batch
    pub fn new(device: GpuDevice) -> Self {
        Self {
            device: Arc::new(device),
            operations: Vec::new(),
            buffers: HashMap::new(),
            next_buffer_id: 0,
        }
    }

    /// Allocate a new buffer ID
    fn alloc_buffer(&mut self, size: usize, data: Option<Vec<f32>>) -> BufferId {
        let id = BufferId(self.next_buffer_id);
        self.next_buffer_id += 1;

        self.buffers.insert(id, BufferInfo { size, data, gpu_buffer: None });

        id
    }

    /// Upload data to GPU (queued for batch execution)
    ///
    /// Returns a buffer ID that can be used in subsequent operations.
    pub fn upload(&mut self, data: &[f32]) -> BufferId {
        self.alloc_buffer(data.len(), Some(data.to_vec()))
    }

    /// Allocate an output buffer for an operation
    fn alloc_output(&mut self, size: usize) -> BufferId {
        self.alloc_buffer(size, None)
    }

    /// Queue ReLU operation: max(0, x)
    ///
    /// Returns buffer ID for the output.
    pub fn relu(&mut self, input: BufferId) -> BufferId {
        let size = self.buffers.get(&input).expect("Invalid buffer ID").size;

        let output = self.alloc_output(size);

        self.operations.push(GpuOp::Relu { input, output });

        output
    }

    /// Queue scalar multiplication: x * scalar
    ///
    /// Returns buffer ID for the output.
    pub fn scale(&mut self, input: BufferId, scalar: f32) -> BufferId {
        let size = self.buffers.get(&input).expect("Invalid buffer ID").size;

        let output = self.alloc_output(size);

        self.operations.push(GpuOp::Scale { input, output, scalar });

        output
    }

    /// Queue element-wise addition: a + b
    ///
    /// Returns buffer ID for the output.
    ///
    /// # Panics
    ///
    /// Panics if buffers have different sizes.
    pub fn add(&mut self, a: BufferId, b: BufferId) -> BufferId {
        let size_a = self.buffers.get(&a).expect("Invalid buffer ID").size;
        let size_b = self.buffers.get(&b).expect("Invalid buffer ID").size;

        assert_eq!(size_a, size_b, "Buffer size mismatch: {} vs {}", size_a, size_b);

        let output = self.alloc_output(size_a);

        self.operations.push(GpuOp::Add { a, b, output });

        output
    }

    /// Queue element-wise multiplication: a * b
    ///
    /// Returns buffer ID for the output.
    ///
    /// # Panics
    ///
    /// Panics if buffers have different sizes.
    pub fn mul(&mut self, a: BufferId, b: BufferId) -> BufferId {
        let size_a = self.buffers.get(&a).expect("Invalid buffer ID").size;
        let size_b = self.buffers.get(&b).expect("Invalid buffer ID").size;

        assert_eq!(size_a, size_b, "Buffer size mismatch: {} vs {}", size_a, size_b);

        let output = self.alloc_output(size_a);

        self.operations.push(GpuOp::Mul { a, b, output });

        output
    }

    /// Queue dot product: sum(a[i] * b[i])
    ///
    /// Returns buffer ID for a single-element output buffer.
    ///
    /// # Panics
    ///
    /// Panics if buffers have different sizes.
    pub fn dot(&mut self, a: BufferId, b: BufferId) -> BufferId {
        let size_a = self.buffers.get(&a).expect("Invalid buffer ID").size;
        let size_b = self.buffers.get(&b).expect("Invalid buffer ID").size;

        assert_eq!(size_a, size_b, "Buffer size mismatch: {} vs {}", size_a, size_b);

        let output = self.alloc_output(1); // Dot product returns scalar

        self.operations.push(GpuOp::Dot { a, b, output });

        output
    }

    /// Queue sigmoid activation: 1 / (1 + exp(-x))
    ///
    /// Returns buffer ID for the output.
    pub fn sigmoid(&mut self, input: BufferId) -> BufferId {
        let size = self.buffers.get(&input).expect("Invalid buffer ID").size;

        let output = self.alloc_output(size);

        self.operations.push(GpuOp::Sigmoid { input, output });

        output
    }

    /// Queue hyperbolic tangent: tanh(x)
    ///
    /// Returns buffer ID for the output.
    pub fn tanh(&mut self, input: BufferId) -> BufferId {
        let size = self.buffers.get(&input).expect("Invalid buffer ID").size;

        let output = self.alloc_output(size);

        self.operations.push(GpuOp::Tanh { input, output });

        output
    }

    /// Queue Swish activation: x * sigmoid(x)
    ///
    /// Returns buffer ID for the output.
    pub fn swish(&mut self, input: BufferId) -> BufferId {
        let size = self.buffers.get(&input).expect("Invalid buffer ID").size;

        let output = self.alloc_output(size);

        self.operations.push(GpuOp::Swish { input, output });

        output
    }

    /// Queue GELU activation: x * Φ(x)
    ///
    /// Returns buffer ID for the output.
    pub fn gelu(&mut self, input: BufferId) -> BufferId {
        let size = self.buffers.get(&input).expect("Invalid buffer ID").size;

        let output = self.alloc_output(size);

        self.operations.push(GpuOp::Gelu { input, output });

        output
    }

    /// Queue element-wise subtraction: a - b
    ///
    /// Returns buffer ID for the output.
    ///
    /// # Panics
    ///
    /// Panics if buffers have different sizes.
    pub fn sub(&mut self, a: BufferId, b: BufferId) -> BufferId {
        let size_a = self.buffers.get(&a).expect("Invalid buffer ID").size;
        let size_b = self.buffers.get(&b).expect("Invalid buffer ID").size;

        assert_eq!(size_a, size_b, "Buffer size mismatch: {} vs {}", size_a, size_b);

        let output = self.alloc_output(size_a);

        self.operations.push(GpuOp::Sub { a, b, output });

        output
    }

    /// Queue matrix multiplication: C = A × B
    ///
    /// A is M×K elements, B is K×N elements, output is M×N elements.
    /// All matrices are row-major flat arrays.
    ///
    /// Returns buffer ID for the M×N output.
    ///
    /// # Panics
    ///
    /// Panics if buffer sizes don't match the declared dimensions.
    pub fn matmul(&mut self, a: BufferId, b: BufferId, m: u32, k: u32, n: u32) -> BufferId {
        let size_a = self.buffers.get(&a).expect("Invalid buffer A ID").size;
        let size_b = self.buffers.get(&b).expect("Invalid buffer B ID").size;

        assert_eq!(
            size_a,
            (m * k) as usize,
            "Buffer A size {} doesn't match M×K = {}",
            size_a,
            m * k
        );
        assert_eq!(
            size_b,
            (k * n) as usize,
            "Buffer B size {} doesn't match K×N = {}",
            size_b,
            k * n
        );

        let output = self.alloc_output((m * n) as usize);

        self.operations.push(GpuOp::Matmul { a, b, output, m, k, n });

        output
    }

    /// Import a pre-existing GPU buffer for use in batch operations.
    ///
    /// Unlike `upload()` which copies host data to GPU during `execute()`,
    /// imported buffers are already GPU-resident and skip the upload step.
    /// The `Arc` wrapper allows the same buffer to be shared across multiple
    /// batch executions without re-uploading (KAIZEN-015: GPU-resident weights).
    ///
    /// # Contract (C-BATCH-IMPORT-001)
    ///
    /// - **Precondition**: `buffer` is a valid `wgpu::Buffer` with STORAGE | COPY_SRC usage
    /// - **Postcondition**: Returned `BufferId` can be used in all batch operations (matmul, etc.)
    /// - **Invariant**: Imported buffer is NOT destroyed when the batch is dropped —
    ///   the `Arc` keeps it alive as long as the caller retains a clone
    pub fn import_buffer(&mut self, buffer: Arc<wgpu::Buffer>, size: usize) -> BufferId {
        let id = BufferId(self.next_buffer_id);
        self.next_buffer_id += 1;
        self.buffers.insert(id, BufferInfo { size, data: None, gpu_buffer: Some(buffer) });
        id
    }

    /// Get the underlying wgpu device for creating persistent buffers.
    ///
    /// Used to create `wgpu::Buffer` instances that outlive individual batch executions.
    /// Created buffers can be registered via `import_buffer()`.
    pub fn wgpu_device(&self) -> &wgpu::Device {
        &self.device.device
    }

    /// Get the underlying wgpu queue for writing to persistent buffers.
    pub fn wgpu_queue(&self) -> &wgpu::Queue {
        &self.device.queue
    }

    /// Get number of queued operations
    pub fn num_operations(&self) -> usize {
        self.operations.len()
    }

    /// Get number of buffers
    pub fn num_buffers(&self) -> usize {
        self.buffers.len()
    }
}
