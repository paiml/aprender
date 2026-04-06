//! GPU Memory Allocator Module (PMAT-802)
//!
//! Extracted from gpu/mod.rs - Memory pooling and arena allocation for inference.
//!
//! ## Contents
//! - `CacheAlignedBuffer` - Cache-aligned tensor storage (IMP-046)
//! - `TensorPool` - Buffer reuse pool (IMP-049)
//! - `ForwardArena` - Bump allocator for forward pass (IMP-050)
//! - `ScratchBuffer` - Per-layer scratch space (IMP-051)
//! - Prefetch and blocked matmul utilities

// ============================================================================
// Cache Efficiency & Prefetch (M21 - IMP-046/047/048)
// ============================================================================

/// Cache line size in bytes (typical x86-64)
const CACHE_LINE_SIZE: usize = 64;

/// Cache-aligned buffer for tensor storage (M21 - IMP-046)
///
/// Ensures data is aligned to cache line boundaries (64 bytes) for optimal
/// memory access patterns and avoiding false sharing.
#[derive(Debug)]
pub struct CacheAlignedBuffer {
    /// Underlying storage with extra space for alignment
    data: Vec<f32>,
    /// Offset to aligned start within data
    offset: usize,
    /// Logical length of the buffer
    len: usize,
}

impl CacheAlignedBuffer {
    /// Create a new cache-aligned buffer of the given size
    #[must_use]
    pub fn new(len: usize) -> Self {
        // Allocate extra space to ensure we can align
        // 64 bytes = 16 f32 values
        let align_elements = CACHE_LINE_SIZE / std::mem::size_of::<f32>();
        let extra = align_elements - 1;
        let data = vec![0.0f32; len + extra];

        // Find the aligned offset
        let ptr = data.as_ptr() as usize;
        let misalignment = ptr % CACHE_LINE_SIZE;
        let offset = if misalignment == 0 {
            0
        } else {
            (CACHE_LINE_SIZE - misalignment) / std::mem::size_of::<f32>()
        };

        Self { data, offset, len }
    }

    /// Check if the buffer is aligned to the given boundary
    #[must_use]
    pub fn is_aligned(&self, alignment: usize) -> bool {
        let ptr = self.as_slice().as_ptr() as usize;
        ptr.is_multiple_of(alignment)
    }

    /// Get the logical length of the buffer
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if the buffer is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get an immutable slice of the aligned data
    #[must_use]
    pub fn as_slice(&self) -> &[f32] {
        &self.data[self.offset..self.offset + self.len]
    }

    /// Get a mutable slice of the aligned data
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        let offset = self.offset;
        let len = self.len;
        &mut self.data[offset..offset + len]
    }
}

/// Software prefetch hint for read access (M21 - IMP-047)
///
/// Hints to the CPU that data at the given position will be needed soon.
/// This is a no-op on platforms without prefetch support.
#[inline]
pub fn prefetch_read(data: &[f32], position: usize, distance: usize) {
    let prefetch_pos = position + distance;
    if prefetch_pos < data.len() {
        // Use a volatile read to hint the prefetch without actual side effects
        // This is a simplified portable approach; real prefetch uses intrinsics
        // SAFETY: We've verified prefetch_pos is in bounds
        let _ = unsafe { std::ptr::read_volatile(&raw const data[prefetch_pos]) };
    }
}

/// Sequential sum without prefetch (baseline for comparison)
#[must_use]
pub fn sequential_sum(data: &[f32]) -> f32 {
    data.iter().sum()
}

/// Sum with software prefetch hints (M21 - IMP-047)
///
/// Uses prefetch hints to reduce cache miss latency for sequential access.
#[must_use]
pub fn sum_with_prefetch(data: &[f32], prefetch_distance: usize) -> f32 {
    let mut sum = 0.0f32;
    let len = data.len();

    for i in 0..len {
        // Prefetch ahead
        if i + prefetch_distance < len {
            prefetch_read(data, i, prefetch_distance);
        }
        sum += data[i];
    }

    sum
}

/// Naive matrix multiplication (baseline for comparison)
///
/// Computes C = A @ B where A is (rows x inner) and B is (inner x cols)
#[must_use]
pub fn naive_matmul(
    mat_a: &[f32],
    mat_b: &[f32],
    rows: usize,
    inner: usize,
    cols: usize,
) -> Vec<f32> {
    let mut result = vec![0.0f32; rows * cols];

    for row in 0..rows {
        for col in 0..cols {
            let mut sum = 0.0f32;
            for idx in 0..inner {
                sum += mat_a[row * inner + idx] * mat_b[idx * cols + col];
            }
            result[row * cols + col] = sum;
        }
    }

    result
}

/// Cache-blocked matrix multiplication (M21 - IMP-048)
///
/// Compute inner block of matrix multiplication
#[inline]
fn compute_block(
    result: &mut [f32],
    mat_a: &[f32],
    mat_b: &[f32],
    row_range: std::ops::Range<usize>,
    col_range: std::ops::Range<usize>,
    inner_range: std::ops::Range<usize>,
    inner: usize,
    cols: usize,
) {
    for row in row_range {
        for col in col_range.clone() {
            let mut sum = result[row * cols + col];
            for idx in inner_range.clone() {
                sum += mat_a[row * inner + idx] * mat_b[idx * cols + col];
            }
            result[row * cols + col] = sum;
        }
    }
}

/// Uses tiling/blocking to improve cache locality for large matrices.
/// Block size should be chosen to fit in L1/L2 cache.
#[must_use]
#[allow(clippy::many_single_char_names)] // Matrix indices are standard notation
pub fn blocked_matmul(
    mat_a: &[f32],
    mat_b: &[f32],
    rows: usize,
    inner: usize,
    cols: usize,
    block_size: usize,
) -> Vec<f32> {
    let mut result = vec![0.0f32; rows * cols];

    // Process in blocks for better cache utilization
    for row_blk in (0..rows).step_by(block_size) {
        let row_end = (row_blk + block_size).min(rows);
        for col_blk in (0..cols).step_by(block_size) {
            let col_end = (col_blk + block_size).min(cols);
            for inner_blk in (0..inner).step_by(block_size) {
                let inner_end = (inner_blk + block_size).min(inner);
                compute_block(
                    &mut result,
                    mat_a,
                    mat_b,
                    row_blk..row_end,
                    col_blk..col_end,
                    inner_blk..inner_end,
                    inner,
                    cols,
                );
            }
        }
    }

    result
}

// ============================================================================
// Phase 13: Memory Pooling & Arena Allocation (M22)
// ============================================================================

/// Tensor memory pool for reusing buffers during inference (M22 - IMP-049)
///
/// Maintains a pool of pre-allocated buffers organized by size class
/// to reduce allocation overhead during token generation.
#[derive(Debug)]
pub struct TensorPool {
    /// Maximum number of buffers to keep in pool
    capacity: usize,
    /// Available buffers organized by size
    buffers: Vec<Vec<f32>>,
}

impl TensorPool {
    /// Create a new tensor pool with the given capacity
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            buffers: Vec::with_capacity(capacity),
        }
    }

    /// Get the maximum capacity of the pool
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get the number of available buffers in the pool
    #[must_use]
    pub fn available(&self) -> usize {
        self.buffers.len()
    }

    /// Acquire a buffer of the given size
    ///
    /// If a suitable buffer exists in the pool, it will be reused.
    /// Otherwise, a new buffer is allocated.
    pub fn acquire(&mut self, size: usize) -> Vec<f32> {
        // Look for a buffer of sufficient size
        if let Some(idx) = self.buffers.iter().position(|b| b.capacity() >= size) {
            let mut buffer = self.buffers.swap_remove(idx);
            buffer.resize(size, 0.0);
            buffer
        } else {
            // Allocate new buffer
            vec![0.0f32; size]
        }
    }

    /// Release a buffer back to the pool
    ///
    /// The buffer will be kept for reuse if the pool has capacity.
    pub fn release(&mut self, buffer: Vec<f32>) {
        if self.buffers.len() < self.capacity {
            self.buffers.push(buffer);
        }
        // If at capacity, buffer is dropped
    }

    /// Clear all buffers from the pool
    pub fn clear(&mut self) {
        self.buffers.clear();
    }
}

/// Arena allocator for forward pass temporaries (M22 - IMP-050)
///
/// Uses bump allocation for fast, contiguous allocation of tensors
/// during a single forward pass. Reset between passes for reuse.
#[derive(Debug)]
pub struct ForwardArena {
    /// Backing storage
    data: Vec<f32>,
    /// Current allocation offset
    offset: usize,
}

impl ForwardArena {
    /// Create a new arena with the given capacity (in f32 elements)
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            data: vec![0.0f32; capacity],
            offset: 0,
        }
    }

    /// Get the total capacity of the arena
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.data.len()
    }

    /// Get the current used amount
    #[must_use]
    pub fn used(&self) -> usize {
        self.offset
    }

    /// Allocate a slice of the given size from the arena
    ///
    /// Returns a mutable slice into the arena's backing storage.
    /// Panics if there is insufficient capacity.
    pub fn alloc(&mut self, size: usize) -> &mut [f32] {
        let start = self.offset;
        let end = start + size;

        assert!(
            end <= self.data.len(),
            "ForwardArena: insufficient capacity (need {}, have {})",
            end,
            self.data.len()
        );

        self.offset = end;
        &mut self.data[start..end]
    }

    /// Reset the arena for reuse
    ///
    /// This does not deallocate memory, just resets the offset.
    pub fn reset(&mut self) {
        self.offset = 0;
    }
}

/// Scratch buffer for layer-wise intermediate computations (M22 - IMP-051)
///
/// Provides pre-allocated scratch space for each transformer layer,
/// avoiding repeated allocations during inference.
#[derive(Debug)]
pub struct ScratchBuffer {
    /// Number of layers
    num_layers: usize,
    /// Size per layer (in f32 elements)
    layer_size: usize,
    /// Backing storage (contiguous for all layers)
    data: Vec<f32>,
}

impl ScratchBuffer {
    /// Create scratch buffers for the given number of layers
    #[must_use]
    pub fn new(num_layers: usize, layer_size: usize) -> Self {
        Self {
            num_layers,
            layer_size,
            data: vec![0.0f32; num_layers * layer_size],
        }
    }

    /// Get the number of layers
    #[must_use]
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    /// Get the size per layer
    #[must_use]
    pub fn layer_size(&self) -> usize {
        self.layer_size
    }

    /// Get the total size of all scratch buffers
    #[must_use]
    pub fn total_size(&self) -> usize {
        self.num_layers * self.layer_size
    }

    /// Get immutable scratch space for a specific layer
    ///
    /// # Panics
    /// Panics if layer_idx >= num_layers
    #[must_use]
    pub fn get_layer(&self, layer_idx: usize) -> &[f32] {
        assert!(
            layer_idx < self.num_layers,
            "ScratchBuffer: layer index {} out of bounds (max {})",
            layer_idx,
            self.num_layers
        );
        let start = layer_idx * self.layer_size;
        let end = start + self.layer_size;
        &self.data[start..end]
    }

    /// Get mutable scratch space for a specific layer
    ///
    /// # Panics
    /// Panics if layer_idx >= num_layers
    pub fn get_layer_mut(&mut self, layer_idx: usize) -> &mut [f32] {
        assert!(
            layer_idx < self.num_layers,
            "ScratchBuffer: layer index {} out of bounds (max {})",
            layer_idx,
            self.num_layers
        );
        let start = layer_idx * self.layer_size;
        let end = start + self.layer_size;
        &mut self.data[start..end]
    }

    /// Reset all scratch buffers to zero
    pub fn reset(&mut self) {
        self.data.fill(0.0);
    }
}

include!("allocator_cache_aligned.rs");
