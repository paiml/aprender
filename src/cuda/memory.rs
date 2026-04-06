//! GPU Memory Pool and Staging Buffers
//!
//! This module provides efficient memory management for GPU operations:
//! - `GpuMemoryPool`: Reduces cudaMalloc/cudaFree overhead
//! - `StagingBufferPool`: Pinned host memory for fast transfers
//! - `TransferMode`: Configuration for host-device transfers

use std::collections::BTreeMap;

// ============================================================================
// GPU Memory Pool (IMP-900d)
// ============================================================================

/// Size class for memory pool allocation
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct SizeClass(usize);

impl SizeClass {
    /// Standard size classes (powers of 2 from 4KB to 256MB)
    pub const CLASSES: [usize; 9] = [
        4096,        // 4 KB
        16384,       // 16 KB
        65536,       // 64 KB
        262_144,     // 256 KB
        1_048_576,   // 1 MB
        4_194_304,   // 4 MB
        16_777_216,  // 16 MB
        67_108_864,  // 64 MB
        268_435_456, // 256 MB
    ];

    /// Find the smallest size class that fits the requested size
    #[must_use]
    pub fn for_size(size: usize) -> Option<Self> {
        Self::CLASSES
            .iter()
            .find(|&&class| class >= size)
            .map(|&class| SizeClass(class))
    }

    /// Get the size in bytes
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.0
    }
}

/// GPU memory pool for efficient buffer allocation (IMP-900d)
///
/// Reduces cudaMalloc/cudaFree overhead by reusing allocated buffers.
/// Buffers are organized by size class for O(1) allocation when a
/// matching buffer is available.
///
/// # Performance Impact
///
/// - Without pool: ~50-100μs per cudaMalloc/cudaFree pair
/// - With pool: ~1-5μs for buffer reuse
/// - Expected improvement: 1.5-2x for memory-bound workloads
#[derive(Debug)]
pub struct GpuMemoryPool {
    /// Free buffers organized by size class
    free_buffers: BTreeMap<usize, Vec<GpuBufferHandle>>,
    /// Total bytes currently allocated
    total_allocated: usize,
    /// Peak memory usage
    peak_usage: usize,
    /// Number of allocations served from pool
    pool_hits: usize,
    /// Number of allocations requiring new cudaMalloc
    pool_misses: usize,
    /// Maximum pool size (bytes)
    max_size: usize,
}

/// Handle to a GPU buffer (stores raw pointer and size)
#[derive(Debug)]
pub struct GpuBufferHandle {
    /// Size in bytes
    pub size: usize,
    /// Whether this buffer is currently in use
    pub in_use: bool,
}

impl Default for GpuMemoryPool {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuMemoryPool {
    /// Create a new GPU memory pool
    #[must_use]
    pub fn new() -> Self {
        Self {
            free_buffers: BTreeMap::new(),
            total_allocated: 0,
            peak_usage: 0,
            pool_hits: 0,
            pool_misses: 0,
            max_size: 2 * 1024 * 1024 * 1024, // 2 GB default
        }
    }

    /// Create a pool with custom max size
    #[must_use]
    pub fn with_max_size(max_size: usize) -> Self {
        Self {
            max_size,
            ..Self::new()
        }
    }

    /// Try to get a buffer from the pool
    ///
    /// Returns a buffer handle if one of suitable size is available.
    pub fn try_get(&mut self, size: usize) -> Option<GpuBufferHandle> {
        // Find the smallest size class that fits
        let size_class = SizeClass::for_size(size)?;
        let class_size = size_class.bytes();

        // Check if we have a free buffer in this size class
        if let Some(buffers) = self.free_buffers.get_mut(&class_size) {
            if let Some(mut handle) = buffers.pop() {
                handle.in_use = true;
                self.pool_hits += 1;
                return Some(handle);
            }
        }

        // No buffer available, will need to allocate
        self.pool_misses += 1;
        None
    }

    /// Return a buffer to the pool for reuse
    pub fn return_buffer(&mut self, mut handle: GpuBufferHandle) {
        handle.in_use = false;
        let size_class = SizeClass::for_size(handle.size).map_or(handle.size, |s| s.bytes());

        self.free_buffers
            .entry(size_class)
            .or_default()
            .push(handle);
    }

    /// Record an allocation (for tracking)
    pub fn record_allocation(&mut self, size: usize) {
        self.total_allocated += size;
        if self.total_allocated > self.peak_usage {
            self.peak_usage = self.total_allocated;
        }
    }

    /// Record a deallocation (for tracking)
    pub fn record_deallocation(&mut self, size: usize) {
        self.total_allocated = self.total_allocated.saturating_sub(size);
    }

    /// Check if pool has capacity for additional allocation
    #[must_use]
    pub fn has_capacity(&self, size: usize) -> bool {
        self.total_allocated + size <= self.max_size
    }

    /// Get maximum pool size
    #[must_use]
    pub fn max_size(&self) -> usize {
        self.max_size
    }

    /// Get pool statistics
    #[must_use]
    pub fn stats(&self) -> PoolStats {
        PoolStats {
            total_allocated: self.total_allocated,
            peak_usage: self.peak_usage,
            pool_hits: self.pool_hits,
            pool_misses: self.pool_misses,
            hit_rate: if self.pool_hits + self.pool_misses > 0 {
                self.pool_hits as f64 / (self.pool_hits + self.pool_misses) as f64
            } else {
                0.0
            },
            free_buffers: self.free_buffers.values().map(Vec::len).sum(),
        }
    }

    /// Clear all free buffers (releases GPU memory)
    pub fn clear(&mut self) {
        self.free_buffers.clear();
    }
}

/// Memory pool statistics
#[derive(Debug, Clone)]
pub struct PoolStats {
    /// Total bytes currently allocated
    pub total_allocated: usize,
    /// Peak memory usage (bytes)
    pub peak_usage: usize,
    /// Number of allocations served from pool
    pub pool_hits: usize,
    /// Number of allocations requiring new cudaMalloc
    pub pool_misses: usize,
    /// Hit rate (0.0 to 1.0)
    pub hit_rate: f64,
    /// Number of free buffers in pool
    pub free_buffers: usize,
}

impl PoolStats {
    /// Calculate memory savings from pooling
    #[must_use]
    pub fn estimated_savings_bytes(&self) -> usize {
        // Each pool hit saves ~100μs of cudaMalloc time
        // Estimate average allocation size from peak/total ratio
        if self.pool_hits > 0 {
            self.pool_hits * 1024 * 1024 // Assume average 1MB allocation
        } else {
            0
        }
    }
}

// ============================================================================
// PARITY-042: Pinned Host Memory for Zero-Copy Transfers
// ============================================================================

/// Pinned (page-locked) host memory buffer for faster GPU transfers
///
/// Pinned memory provides several benefits:
/// - DMA transfers without CPU involvement (~2x faster H2D/D2H)
/// - Zero-copy access where GPU can directly read host memory
/// - Async transfer overlap with kernel execution
///
/// # Memory Model
///
/// ```text
/// Regular Memory:        Pinned Memory:
/// ┌─────────────┐       ┌─────────────┐
/// │ Host Memory │       │ Host Memory │ (page-locked)
/// └──────┬──────┘       └──────┬──────┘
///        │ copy                 │ DMA
/// ┌──────▼──────┐       ┌──────▼──────┐
/// │ Page Cache  │       │   (skip)    │
/// └──────┬──────┘       └─────────────┘
///        │ DMA                  │
/// ┌──────▼──────┐       ┌──────▼──────┐
/// │ GPU Memory  │       │ GPU Memory  │
/// └─────────────┘       └─────────────┘
/// ```
///
/// # CUDA Implementation
///
/// When trueno-gpu adds `cuMemAllocHost` support, this will use true
/// page-locked memory. Currently uses aligned allocation as fallback.
#[derive(Debug)]
pub struct PinnedHostBuffer<T> {
    /// Aligned data storage
    data: Vec<T>,
    /// Whether this is truly pinned (requires CUDA driver support)
    is_pinned: bool,
}

impl<T: Copy + Default> PinnedHostBuffer<T> {
    /// Allocate a pinned host buffer
    ///
    /// # Arguments
    ///
    /// * `len` - Number of elements
    ///
    /// # Note
    ///
    /// Currently falls back to aligned allocation. True pinned memory
    /// requires trueno-gpu CUDA driver support for `cuMemAllocHost`.
    #[must_use]
    pub fn new(len: usize) -> Self {
        // Allocate with alignment for cache-line efficiency (64 bytes)
        // Note: Currently uses standard allocation. True CUDA pinned memory
        // (cuMemAllocHost) requires trueno-gpu driver support - tracked in PARITY-042.
        let data = vec![T::default(); len];

        Self {
            data,
            is_pinned: false, // Will be true when CUDA support added
        }
    }

    /// Get slice of data
    #[must_use]
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// Get mutable slice of data
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Get length in elements
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Check if truly pinned (page-locked)
    #[must_use]
    pub fn is_pinned(&self) -> bool {
        self.is_pinned
    }

    /// Size in bytes
    #[must_use]
    pub fn size_bytes(&self) -> usize {
        self.len() * std::mem::size_of::<T>()
    }

    /// Copy from slice
    pub fn copy_from_slice(&mut self, src: &[T]) {
        self.data.copy_from_slice(src);
    }
}

/// Pool of staging buffers for efficient H2D/D2H transfers (PARITY-042)
///
/// Maintains reusable pinned buffers to avoid allocation overhead.
/// Staging buffers are used to:
/// 1. Copy data to pinned memory
/// 2. Async transfer to GPU
/// 3. Overlap with kernel execution
///
/// # Performance Impact
///
/// - Without staging: allocate → copy → free each transfer
/// - With staging: reuse pre-allocated pinned buffers
/// - Expected improvement: 1.3-1.5x for memory-bound workloads
#[derive(Debug)]
pub struct StagingBufferPool {
    /// Free staging buffers by size class
    free_buffers: BTreeMap<usize, Vec<PinnedHostBuffer<f32>>>,
    /// Total bytes allocated
    total_allocated: usize,
    /// Peak memory usage
    peak_usage: usize,
    /// Pool hits (buffer reuse)
    pool_hits: usize,
    /// Pool misses (new allocation)
    pool_misses: usize,
    /// Maximum pool size in bytes
    max_size: usize,
}

impl Default for StagingBufferPool {
    fn default() -> Self {
        Self::new()
    }
}

impl StagingBufferPool {
    /// Create a new staging buffer pool
    #[must_use]
    pub fn new() -> Self {
        Self {
            free_buffers: BTreeMap::new(),
            total_allocated: 0,
            peak_usage: 0,
            pool_hits: 0,
            pool_misses: 0,
            max_size: 512 * 1024 * 1024, // 512 MB default for staging
        }
    }

    /// Create pool with custom max size
    #[must_use]
    pub fn with_max_size(max_size: usize) -> Self {
        Self {
            max_size,
            ..Self::new()
        }
    }

    /// Get a staging buffer of at least `size` elements
    ///
    /// Returns a buffer from the pool if available, otherwise allocates new.
    pub fn get(&mut self, size: usize) -> PinnedHostBuffer<f32> {
        let size_bytes = size * std::mem::size_of::<f32>();
        let size_class = SizeClass::for_size(size_bytes).map_or(size_bytes, |c| c.bytes());
        let elements = size_class / std::mem::size_of::<f32>();

        // Try to get from pool
        if let Some(buffers) = self.free_buffers.get_mut(&size_class) {
            if let Some(buf) = buffers.pop() {
                self.pool_hits += 1;
                return buf;
            }
        }

        // Allocate new buffer
        self.pool_misses += 1;
        let buf = PinnedHostBuffer::new(elements);
        self.total_allocated += buf.size_bytes();
        self.peak_usage = self.peak_usage.max(self.total_allocated);
        buf
    }

    /// Return a buffer to the pool
    pub fn put(&mut self, buf: PinnedHostBuffer<f32>) {
        let size_class = buf.size_bytes();

        // Don't pool if over max size
        if self.total_allocated > self.max_size {
            self.total_allocated = self.total_allocated.saturating_sub(size_class);
            return; // Drop buffer
        }

        self.free_buffers.entry(size_class).or_default().push(buf);
    }

    /// Get pool statistics
    #[must_use]
    pub fn stats(&self) -> StagingPoolStats {
        let free_count: usize = self.free_buffers.values().map(Vec::len).sum();
        StagingPoolStats {
            total_allocated: self.total_allocated,
            peak_usage: self.peak_usage,
            pool_hits: self.pool_hits,
            pool_misses: self.pool_misses,
            free_buffers: free_count,
            hit_rate: if self.pool_hits + self.pool_misses > 0 {
                self.pool_hits as f64 / (self.pool_hits + self.pool_misses) as f64
            } else {
                0.0
            },
        }
    }

    /// Clear all buffers from pool
    pub fn clear(&mut self) {
        self.free_buffers.clear();
        self.total_allocated = 0;
    }
}

include!("transfer.rs");
include!("memory_transfer_mode.rs");
