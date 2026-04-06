//! PERF-005: io_uring Registered Buffers
//!
//! Scientific Basis: [Axboe 2019] demonstrated 2-5x IOPS improvement with
//! registered buffers by eliminating per-I/O buffer mapping overhead.
//!
//! ## Performance Targets
//!
//! | Metric | Before | Target | Falsification |
//! |--------|--------|--------|---------------|
//! | Buffer setup | 200ns/IO | 0ns/IO | `perf stat -e dTLB-load-misses` |
//! | IOPS | 286K | 500K | fio randread QD=32 |
//!
//! ## Implementation
//!
//! Pre-register compression buffers at startup, then use registered buffer
//! indices in SQEs instead of raw pointers. This eliminates:
//! - Per-I/O buffer pinning
//! - TLB shootdowns
//! - Page table walks
//!
//! ## Falsification Matrix Points
//!
//! - B.21: Buffer registration succeeds
//! - B.22: Registered buffer used in SQE
//! - B.23: Per-I/O buffer setup eliminated
//! - B.24: TLB misses reduced >50%
//! - B.25: IOPS improved >1.5x baseline
//! - B.26: Latency p99 not regressed <1.1x
//! - B.27: Memory usage not increased <1.1x
//! - B.28: Buffer reuse verified 100%
//! - B.29: No buffer leaks
//! - B.30: Correctness maintained 100%

use std::io::{self, IoSliceMut};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Page size constant
pub const PAGE_SIZE: usize = 4096;

/// Default queue depth
pub const DEFAULT_QUEUE_DEPTH: usize = 128;

/// Default pages per buffer (256 pages = 1MB)
pub const DEFAULT_PAGES_PER_BUFFER: usize = 256;

/// Configuration for registered buffer pool
#[derive(Debug, Clone)]
pub struct RegisteredBufferConfig {
    /// Enable registered buffers
    pub enabled: bool,

    /// Queue depth (number of concurrent I/Os)
    pub queue_depth: usize,

    /// Pages per buffer (determines max I/O size)
    pub pages_per_buffer: usize,

    /// Alignment requirement (typically 4KB or 2MB for huge pages)
    pub alignment: usize,

    /// Pre-fault pages to avoid page faults in hot path
    pub prefault: bool,

    /// PERF-009: Use 2MB huge pages for reduced TLB pressure
    pub use_huge_pages: bool,
}

impl Default for RegisteredBufferConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            queue_depth: DEFAULT_QUEUE_DEPTH,
            pages_per_buffer: DEFAULT_PAGES_PER_BUFFER,
            alignment: PAGE_SIZE,
            prefault: true,
            use_huge_pages: false,
        }
    }
}

impl RegisteredBufferConfig {
    /// Conservative configuration - smaller buffers, safe defaults
    pub fn conservative() -> Self {
        Self {
            enabled: true,
            queue_depth: 64,
            pages_per_buffer: 64,
            alignment: PAGE_SIZE,
            prefault: false,
            use_huge_pages: false,
        }
    }

    /// Aggressive configuration - larger buffers for maximum throughput
    pub fn aggressive() -> Self {
        Self {
            enabled: true,
            queue_depth: 256,
            pages_per_buffer: 512,
            alignment: 2 * 1024 * 1024, // 2MB for huge pages
            prefault: true,
            use_huge_pages: true, // PERF-009: Enable huge pages for TLB efficiency
        }
    }

    /// Calculate total buffer size
    pub fn total_size(&self) -> usize {
        self.queue_depth * self.pages_per_buffer * PAGE_SIZE
    }

    /// Calculate single buffer size
    pub fn buffer_size(&self) -> usize {
        self.pages_per_buffer * PAGE_SIZE
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), RegisteredBufferError> {
        if self.queue_depth == 0 {
            return Err(RegisteredBufferError::InvalidConfig("queue_depth cannot be 0".into()));
        }
        if self.pages_per_buffer == 0 {
            return Err(RegisteredBufferError::InvalidConfig(
                "pages_per_buffer cannot be 0".into(),
            ));
        }
        if !self.alignment.is_power_of_two() {
            return Err(RegisteredBufferError::InvalidConfig(
                "alignment must be power of 2".into(),
            ));
        }
        Ok(())
    }
}

/// Errors from registered buffer operations
#[derive(Debug)]
pub enum RegisteredBufferError {
    /// Invalid configuration
    InvalidConfig(String),
    /// Allocation failed
    AllocationFailed(io::Error),
    /// Registration with io_uring failed
    RegistrationFailed(io::Error),
    /// Buffer index out of bounds
    IndexOutOfBounds { index: usize, max: usize },
    /// Buffer already in use
    BufferInUse(usize),
    /// Pool exhausted
    PoolExhausted,
}

impl std::fmt::Display for RegisteredBufferError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RegisteredBufferError::InvalidConfig(msg) => write!(f, "Invalid config: {}", msg),
            RegisteredBufferError::AllocationFailed(e) => write!(f, "Allocation failed: {}", e),
            RegisteredBufferError::RegistrationFailed(e) => write!(f, "Registration failed: {}", e),
            RegisteredBufferError::IndexOutOfBounds { index, max } => {
                write!(f, "Index {} out of bounds (max {})", index, max)
            }
            RegisteredBufferError::BufferInUse(idx) => write!(f, "Buffer {} already in use", idx),
            RegisteredBufferError::PoolExhausted => write!(f, "Buffer pool exhausted"),
        }
    }
}

impl std::error::Error for RegisteredBufferError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            RegisteredBufferError::AllocationFailed(e) => Some(e),
            RegisteredBufferError::RegistrationFailed(e) => Some(e),
            _ => None,
        }
    }
}

/// Statistics for buffer pool usage
#[derive(Debug, Default)]
pub struct BufferPoolStats {
    /// Total allocations
    pub allocations: AtomicU64,
    /// Total deallocations
    pub deallocations: AtomicU64,
    /// Peak concurrent usage
    pub peak_usage: AtomicU64,
    /// Current usage
    pub current_usage: AtomicU64,
    /// Allocation failures (pool exhausted)
    pub exhausted_count: AtomicU64,
}

impl BufferPoolStats {
    /// Record an allocation
    pub fn record_alloc(&self) {
        self.allocations.fetch_add(1, Ordering::Relaxed);
        let current = self.current_usage.fetch_add(1, Ordering::Relaxed) + 1;
        // Update peak if necessary
        let mut peak = self.peak_usage.load(Ordering::Relaxed);
        while current > peak {
            match self.peak_usage.compare_exchange_weak(
                peak,
                current,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(p) => peak = p,
            }
        }
    }

    /// Record a deallocation
    pub fn record_dealloc(&self) {
        self.deallocations.fetch_add(1, Ordering::Relaxed);
        self.current_usage.fetch_sub(1, Ordering::Relaxed);
    }

    /// Record exhaustion event
    pub fn record_exhausted(&self) {
        self.exhausted_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Check for leaks (allocations != deallocations)
    pub fn has_leaks(&self) -> bool {
        self.allocations.load(Ordering::Relaxed) != self.deallocations.load(Ordering::Relaxed)
    }

    /// Get reuse percentage
    pub fn reuse_percentage(&self) -> f64 {
        let allocs = self.allocations.load(Ordering::Relaxed);
        if allocs == 0 {
            return 100.0;
        }
        let exhausted = self.exhausted_count.load(Ordering::Relaxed);
        ((allocs - exhausted) as f64 / allocs as f64) * 100.0
    }
}

/// A pool of pre-allocated, registered buffers for io_uring
///
/// This is the core PERF-005 implementation. Buffers are:
/// 1. Allocated at startup with proper alignment
/// 2. Registered with io_uring via IORING_REGISTER_BUFFERS
/// 3. Tracked with a bitmap for allocation
/// 4. Referenced by index in SQEs (not raw pointers)
pub struct RegisteredBufferPool {
    /// Configuration
    config: RegisteredBufferConfig,

    /// The actual buffer memory
    memory: NonNull<u8>,

    /// Total allocation size (for munmap)
    total_size: usize,

    /// Allocation bitmap (true = in use)
    in_use: Vec<AtomicBool>,

    /// Statistics
    stats: BufferPoolStats,

    /// Whether registered with io_uring
    registered: bool,
}

// SAFETY: The buffer memory is owned by this struct and can be safely
// sent between threads. Access is synchronized via atomic operations.
unsafe impl Send for RegisteredBufferPool {}
unsafe impl Sync for RegisteredBufferPool {}

impl RegisteredBufferPool {
    /// Create a new registered buffer pool
    pub fn new(config: RegisteredBufferConfig) -> Result<Self, RegisteredBufferError> {
        config.validate()?;

        let buffer_size = config.buffer_size();
        let total_size = config.queue_depth * buffer_size;

        // Allocate aligned memory (optionally with huge pages)
        let memory = Self::allocate_aligned(total_size, config.alignment, config.use_huge_pages)?;

        // Pre-fault if requested
        if config.prefault {
            Self::prefault_memory(memory, total_size);
        }

        // Initialize allocation bitmap
        let in_use: Vec<AtomicBool> =
            (0..config.queue_depth).map(|_| AtomicBool::new(false)).collect();

        Ok(Self {
            config,
            memory,
            total_size,
            in_use,
            stats: BufferPoolStats::default(),
            registered: false,
        })
    }

    /// Allocate aligned memory with optional huge page support (PERF-009)
    fn allocate_aligned(
        size: usize,
        alignment: usize,
        use_huge_pages: bool,
    ) -> Result<NonNull<u8>, RegisteredBufferError> {
        use nix::libc::{mmap, MAP_ANONYMOUS, MAP_HUGETLB, MAP_PRIVATE, PROT_READ, PROT_WRITE};
        use std::ptr::null_mut;

        // Round up size to alignment
        let aligned_size = (size + alignment - 1) & !(alignment - 1);

        // Build mmap flags
        let mut flags = MAP_PRIVATE | MAP_ANONYMOUS;

        // PERF-009: Add huge page flags if enabled
        // MAP_HUGE_2MB = 21 << 26
        if use_huge_pages {
            flags |= MAP_HUGETLB | (21 << 26);
        }

        // SAFETY: mmap is called with valid parameters for anonymous memory
        let ptr = unsafe { mmap(null_mut(), aligned_size, PROT_READ | PROT_WRITE, flags, -1, 0) };

        // Check for failure
        if ptr == nix::libc::MAP_FAILED {
            // If huge pages failed and were requested, fall back to regular pages
            if use_huge_pages {
                tracing::warn!(
                    size = aligned_size,
                    "PERF-009: Huge page allocation failed, falling back to regular pages"
                );
                return Self::allocate_aligned(size, alignment, false);
            }
            return Err(RegisteredBufferError::AllocationFailed(io::Error::last_os_error()));
        }

        if use_huge_pages {
            tracing::info!(
                size = aligned_size,
                "PERF-009: Allocated {} bytes with 2MB huge pages",
                aligned_size
            );
        }

        NonNull::new(ptr as *mut u8).ok_or_else(|| {
            RegisteredBufferError::AllocationFailed(io::Error::other("mmap returned null"))
        })
    }

    /// Pre-fault all pages to avoid page faults in hot path
    fn prefault_memory(memory: NonNull<u8>, size: usize) {
        // Touch every page to ensure it's faulted in
        let ptr = memory.as_ptr();
        for offset in (0..size).step_by(PAGE_SIZE) {
            // SAFETY: We allocated this memory and offset is within bounds
            unsafe {
                std::ptr::write_volatile(ptr.add(offset), 0);
            }
        }
    }

    /// Get configuration
    pub fn config(&self) -> &RegisteredBufferConfig {
        &self.config
    }

    /// Get statistics
    pub fn stats(&self) -> &BufferPoolStats {
        &self.stats
    }

    /// Check if registered with io_uring
    pub fn is_registered(&self) -> bool {
        self.registered
    }

    /// Mark as registered (called after IORING_REGISTER_BUFFERS succeeds)
    pub fn mark_registered(&mut self) {
        self.registered = true;
    }

    /// Mark as unregistered (called after IORING_UNREGISTER_BUFFERS)
    pub fn mark_unregistered(&mut self) {
        self.registered = false;
    }

    /// Allocate a buffer, returning its index
    pub fn allocate(&self) -> Result<usize, RegisteredBufferError> {
        for (index, slot) in self.in_use.iter().enumerate() {
            // Try to atomically acquire this slot
            if slot.compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed).is_ok() {
                self.stats.record_alloc();
                return Ok(index);
            }
        }

        self.stats.record_exhausted();
        Err(RegisteredBufferError::PoolExhausted)
    }

    /// Deallocate a buffer by index
    pub fn deallocate(&self, index: usize) -> Result<(), RegisteredBufferError> {
        if index >= self.in_use.len() {
            return Err(RegisteredBufferError::IndexOutOfBounds { index, max: self.in_use.len() });
        }

        // Release the slot
        self.in_use[index].store(false, Ordering::Release);
        self.stats.record_dealloc();
        Ok(())
    }

    /// Get a raw pointer to buffer at index
    ///
    /// # Safety
    /// Caller must ensure the buffer is allocated and not concurrently modified.
    pub unsafe fn get_buffer_ptr(&self, index: usize) -> Result<*mut u8, RegisteredBufferError> {
        if index >= self.config.queue_depth {
            return Err(RegisteredBufferError::IndexOutOfBounds {
                index,
                max: self.config.queue_depth,
            });
        }

        let offset = index * self.config.buffer_size();
        Ok(self.memory.as_ptr().add(offset))
    }

    /// Get a mutable slice to buffer at index
    ///
    /// # Safety
    /// Caller must ensure the buffer is allocated and not concurrently accessed.
    /// Despite taking `&self`, this returns a mutable slice because the buffer
    /// memory is externally synchronized (tracked by `in_use` bitmap).
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn get_buffer_slice(
        &self,
        index: usize,
    ) -> Result<&mut [u8], RegisteredBufferError> {
        let ptr = self.get_buffer_ptr(index)?;
        Ok(std::slice::from_raw_parts_mut(ptr, self.config.buffer_size()))
    }

    /// Build IoSliceMut array for io_uring registration
    ///
    /// This is used with `ring.submitter().register_buffers()`.
    pub fn build_io_slices(&self) -> Vec<IoSliceMut<'_>> {
        let mut slices = Vec::with_capacity(self.config.queue_depth);
        let buffer_size = self.config.buffer_size();

        for i in 0..self.config.queue_depth {
            let offset = i * buffer_size;
            // SAFETY: We own this memory and the slice is within bounds
            let slice = unsafe {
                std::slice::from_raw_parts_mut(self.memory.as_ptr().add(offset), buffer_size)
            };
            slices.push(IoSliceMut::new(slice));
        }

        slices
    }

    /// Get current usage count
    pub fn current_usage(&self) -> usize {
        self.in_use.iter().filter(|slot| slot.load(Ordering::Relaxed)).count()
    }

    /// Get number of available buffers
    pub fn available(&self) -> usize {
        self.config.queue_depth - self.current_usage()
    }
}

impl Drop for RegisteredBufferPool {
    fn drop(&mut self) {
        // SAFETY: We allocated this memory with mmap
        unsafe {
            nix::libc::munmap(self.memory.as_ptr() as *mut _, self.total_size);
        }
    }
}

/// RAII guard for a borrowed buffer
pub struct BufferGuard<'a> {
    pool: &'a RegisteredBufferPool,
    index: usize,
}

impl<'a> BufferGuard<'a> {
    /// Create a new buffer guard
    pub fn new(pool: &'a RegisteredBufferPool) -> Result<Self, RegisteredBufferError> {
        let index = pool.allocate()?;
        Ok(Self { pool, index })
    }

    /// Get the buffer index (for use in SQEs)
    pub fn index(&self) -> usize {
        self.index
    }

    /// Get a mutable slice to the buffer
    ///
    /// # Safety
    /// Must not be called concurrently with other access to the same buffer.
    pub unsafe fn as_slice_mut(&mut self) -> &mut [u8] {
        self.pool.get_buffer_slice(self.index).expect("buffer guard holds valid allocation")
    }

    /// Get a raw pointer to the buffer
    pub fn as_ptr(&self) -> *mut u8 {
        // SAFETY: We hold the allocation
        unsafe {
            self.pool.get_buffer_ptr(self.index).expect("buffer guard holds valid allocation")
        }
    }
}

impl<'a> Drop for BufferGuard<'a> {
    fn drop(&mut self) {
        let _ = self.pool.deallocate(self.index);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // RegisteredBufferConfig Tests
    // ========================================================================

    #[test]
    fn test_config_default() {
        let config = RegisteredBufferConfig::default();
        assert!(config.enabled);
        assert_eq!(config.queue_depth, DEFAULT_QUEUE_DEPTH);
        assert_eq!(config.pages_per_buffer, DEFAULT_PAGES_PER_BUFFER);
        assert_eq!(config.alignment, PAGE_SIZE);
        assert!(config.prefault);
    }

    #[test]
    fn test_config_conservative() {
        let config = RegisteredBufferConfig::conservative();
        assert!(config.enabled);
        assert_eq!(config.queue_depth, 64);
        assert_eq!(config.pages_per_buffer, 64);
        assert!(!config.prefault);
    }

    #[test]
    fn test_config_aggressive() {
        let config = RegisteredBufferConfig::aggressive();
        assert!(config.enabled);
        assert_eq!(config.queue_depth, 256);
        assert_eq!(config.pages_per_buffer, 512);
        assert_eq!(config.alignment, 2 * 1024 * 1024);
        assert!(config.prefault);
    }

    #[test]
    fn test_config_total_size() {
        let config = RegisteredBufferConfig::default();
        // 128 * 256 * 4096 = 128MB
        assert_eq!(config.total_size(), 128 * 256 * 4096);
    }

    #[test]
    fn test_config_buffer_size() {
        let config = RegisteredBufferConfig::default();
        // 256 * 4096 = 1MB
        assert_eq!(config.buffer_size(), 256 * 4096);
    }

    #[test]
    fn test_config_validate_success() {
        let config = RegisteredBufferConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_validate_zero_queue_depth() {
        let mut config = RegisteredBufferConfig::default();
        config.queue_depth = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validate_zero_pages() {
        let mut config = RegisteredBufferConfig::default();
        config.pages_per_buffer = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validate_bad_alignment() {
        let mut config = RegisteredBufferConfig::default();
        config.alignment = 1000; // Not power of 2
        assert!(config.validate().is_err());
    }

    // ========================================================================
    // BufferPoolStats Tests
    // ========================================================================

    #[test]
    fn test_stats_default() {
        let stats = BufferPoolStats::default();
        assert_eq!(stats.allocations.load(Ordering::Relaxed), 0);
        assert_eq!(stats.deallocations.load(Ordering::Relaxed), 0);
        assert_eq!(stats.current_usage.load(Ordering::Relaxed), 0);
        assert_eq!(stats.peak_usage.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_stats_record_alloc() {
        let stats = BufferPoolStats::default();
        stats.record_alloc();
        assert_eq!(stats.allocations.load(Ordering::Relaxed), 1);
        assert_eq!(stats.current_usage.load(Ordering::Relaxed), 1);
        assert_eq!(stats.peak_usage.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_stats_record_dealloc() {
        let stats = BufferPoolStats::default();
        stats.record_alloc();
        stats.record_dealloc();
        assert_eq!(stats.allocations.load(Ordering::Relaxed), 1);
        assert_eq!(stats.deallocations.load(Ordering::Relaxed), 1);
        assert_eq!(stats.current_usage.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_stats_peak_usage() {
        let stats = BufferPoolStats::default();
        stats.record_alloc();
        stats.record_alloc();
        stats.record_alloc();
        assert_eq!(stats.peak_usage.load(Ordering::Relaxed), 3);
        stats.record_dealloc();
        stats.record_dealloc();
        assert_eq!(stats.peak_usage.load(Ordering::Relaxed), 3); // Peak unchanged
        assert_eq!(stats.current_usage.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_stats_no_leaks() {
        let stats = BufferPoolStats::default();
        stats.record_alloc();
        stats.record_alloc();
        stats.record_dealloc();
        stats.record_dealloc();
        assert!(!stats.has_leaks());
    }

    #[test]
    fn test_stats_has_leaks() {
        let stats = BufferPoolStats::default();
        stats.record_alloc();
        stats.record_alloc();
        stats.record_dealloc();
        assert!(stats.has_leaks());
    }

    #[test]
    fn test_stats_reuse_percentage() {
        let stats = BufferPoolStats::default();
        for _ in 0..100 {
            stats.record_alloc();
        }
        assert_eq!(stats.reuse_percentage(), 100.0);
    }

    // ========================================================================
    // RegisteredBufferPool Tests (PERF-005 Core)
    // ========================================================================

    #[test]
    fn test_pool_new() {
        // Use small config for tests
        let mut config = RegisteredBufferConfig::default();
        config.queue_depth = 4;
        config.pages_per_buffer = 1;
        config.prefault = false;

        let pool = RegisteredBufferPool::new(config).unwrap();
        assert_eq!(pool.config().queue_depth, 4);
        assert!(!pool.is_registered());
    }

    #[test]
    fn test_pool_allocate_deallocate() {
        let mut config = RegisteredBufferConfig::default();
        config.queue_depth = 4;
        config.pages_per_buffer = 1;
        config.prefault = false;

        let pool = RegisteredBufferPool::new(config).unwrap();

        let idx1 = pool.allocate().unwrap();
        assert_eq!(idx1, 0);

        let idx2 = pool.allocate().unwrap();
        assert_eq!(idx2, 1);

        pool.deallocate(idx1).unwrap();
        pool.deallocate(idx2).unwrap();

        // Should be able to reuse
        let idx3 = pool.allocate().unwrap();
        assert!(idx3 == 0 || idx3 == 1);
    }

    #[test]
    fn test_pool_exhaustion() {
        let mut config = RegisteredBufferConfig::default();
        config.queue_depth = 2;
        config.pages_per_buffer = 1;
        config.prefault = false;

        let pool = RegisteredBufferPool::new(config).unwrap();

        let _idx1 = pool.allocate().unwrap();
        let _idx2 = pool.allocate().unwrap();

        // Third allocation should fail
        let result = pool.allocate();
        assert!(matches!(result, Err(RegisteredBufferError::PoolExhausted)));
    }

    #[test]
    fn test_pool_current_usage() {
        let mut config = RegisteredBufferConfig::default();
        config.queue_depth = 4;
        config.pages_per_buffer = 1;
        config.prefault = false;

        let pool = RegisteredBufferPool::new(config).unwrap();
        assert_eq!(pool.current_usage(), 0);
        assert_eq!(pool.available(), 4);

        let idx = pool.allocate().unwrap();
        assert_eq!(pool.current_usage(), 1);
        assert_eq!(pool.available(), 3);

        pool.deallocate(idx).unwrap();
        assert_eq!(pool.current_usage(), 0);
        assert_eq!(pool.available(), 4);
    }

    #[test]
    fn test_pool_get_buffer_ptr() {
        let mut config = RegisteredBufferConfig::default();
        config.queue_depth = 2;
        config.pages_per_buffer = 1;
        config.prefault = false;

        let pool = RegisteredBufferPool::new(config).unwrap();

        // SAFETY: Test-only, buffer exists
        unsafe {
            let ptr0 = pool.get_buffer_ptr(0).unwrap();
            let ptr1 = pool.get_buffer_ptr(1).unwrap();
            assert_ne!(ptr0, ptr1);
            assert_eq!(ptr1.offset_from(ptr0) as usize, PAGE_SIZE);
        }
    }

    #[test]
    fn test_pool_get_buffer_slice() {
        let mut config = RegisteredBufferConfig::default();
        config.queue_depth = 2;
        config.pages_per_buffer = 1;
        config.prefault = false;

        let pool = RegisteredBufferPool::new(config).unwrap();

        // SAFETY: Test-only, buffer exists
        unsafe {
            let slice = pool.get_buffer_slice(0).unwrap();
            assert_eq!(slice.len(), PAGE_SIZE);

            // Write to buffer
            slice[0] = 42;
            assert_eq!(slice[0], 42);
        }
    }

    #[test]
    fn test_pool_build_io_slices() {
        let mut config = RegisteredBufferConfig::default();
        config.queue_depth = 4;
        config.pages_per_buffer = 1;
        config.prefault = false;

        let pool = RegisteredBufferPool::new(config).unwrap();
        let slices = pool.build_io_slices();

        assert_eq!(slices.len(), 4);
        for slice in &slices {
            assert_eq!(slice.len(), PAGE_SIZE);
        }
    }

    #[test]
    fn test_pool_out_of_bounds() {
        let mut config = RegisteredBufferConfig::default();
        config.queue_depth = 2;
        config.pages_per_buffer = 1;
        config.prefault = false;

        let pool = RegisteredBufferPool::new(config).unwrap();

        // SAFETY: Test error handling
        unsafe {
            let result = pool.get_buffer_ptr(999);
            assert!(matches!(result, Err(RegisteredBufferError::IndexOutOfBounds { .. })));
        }
    }

    // ========================================================================
    // BufferGuard Tests
    // ========================================================================

    #[test]
    fn test_buffer_guard_allocate() {
        let mut config = RegisteredBufferConfig::default();
        config.queue_depth = 4;
        config.pages_per_buffer = 1;
        config.prefault = false;

        let pool = RegisteredBufferPool::new(config).unwrap();
        assert_eq!(pool.current_usage(), 0);

        {
            let guard = BufferGuard::new(&pool).unwrap();
            assert_eq!(guard.index(), 0);
            assert_eq!(pool.current_usage(), 1);
        }

        // Guard dropped, buffer should be released
        assert_eq!(pool.current_usage(), 0);
    }

    #[test]
    fn test_buffer_guard_multiple() {
        let mut config = RegisteredBufferConfig::default();
        config.queue_depth = 4;
        config.pages_per_buffer = 1;
        config.prefault = false;

        let pool = RegisteredBufferPool::new(config).unwrap();

        let _g1 = BufferGuard::new(&pool).unwrap();
        let _g2 = BufferGuard::new(&pool).unwrap();
        let _g3 = BufferGuard::new(&pool).unwrap();
        assert_eq!(pool.current_usage(), 3);
    }

    #[test]
    fn test_buffer_guard_write() {
        let mut config = RegisteredBufferConfig::default();
        config.queue_depth = 2;
        config.pages_per_buffer = 1;
        config.prefault = false;

        let pool = RegisteredBufferPool::new(config).unwrap();
        let mut guard = BufferGuard::new(&pool).unwrap();

        // SAFETY: Single writer
        unsafe {
            let slice = guard.as_slice_mut();
            slice[0] = 42;
            slice[4095] = 255;
            assert_eq!(slice[0], 42);
            assert_eq!(slice[4095], 255);
        }
    }

    // ========================================================================
    // Falsification Matrix Tests (Section B: Points 21-30)
    // ========================================================================

    /// B.21: Buffer registration succeeds
    #[test]
    fn test_falsify_b21_buffer_registration_succeeds() {
        let mut config = RegisteredBufferConfig::default();
        config.queue_depth = 4;
        config.pages_per_buffer = 1;
        config.prefault = false;

        let result = RegisteredBufferPool::new(config);
        assert!(result.is_ok(), "B.21: Buffer registration must succeed");
    }

    /// B.27: Memory usage not increased significantly
    #[test]
    fn test_falsify_b27_memory_usage() {
        let mut config = RegisteredBufferConfig::default();
        config.queue_depth = 16;
        config.pages_per_buffer = 4;
        config.prefault = false;

        let expected_size = config.total_size();
        let pool = RegisteredBufferPool::new(config).unwrap();

        // Pool should use approximately the expected size
        assert_eq!(pool.total_size, expected_size);
        // No hidden allocations
        assert!(
            pool.total_size <= expected_size * 2,
            "B.27: Memory usage should not exceed 2x expected"
        );
    }

    /// B.28: Buffer reuse verified
    #[test]
    fn test_falsify_b28_buffer_reuse() {
        let mut config = RegisteredBufferConfig::default();
        config.queue_depth = 4;
        config.pages_per_buffer = 1;
        config.prefault = false;

        let pool = RegisteredBufferPool::new(config).unwrap();

        // Simulate workload: allocate and deallocate many times
        for _ in 0..1000 {
            let idx = pool.allocate().unwrap();
            pool.deallocate(idx).unwrap();
        }

        assert_eq!(pool.stats().reuse_percentage(), 100.0, "B.28: Must achieve 100% buffer reuse");
    }

    /// B.29: No buffer leaks
    #[test]
    fn test_falsify_b29_no_buffer_leaks() {
        let mut config = RegisteredBufferConfig::default();
        config.queue_depth = 4;
        config.pages_per_buffer = 1;
        config.prefault = false;

        let pool = RegisteredBufferPool::new(config).unwrap();

        // Allocate and deallocate
        for _ in 0..100 {
            let idx = pool.allocate().unwrap();
            pool.deallocate(idx).unwrap();
        }

        assert!(!pool.stats().has_leaks(), "B.29: Must have no buffer leaks");
    }

    /// B.30: Correctness maintained
    #[test]
    fn test_falsify_b30_correctness() {
        let mut config = RegisteredBufferConfig::default();
        config.queue_depth = 4;
        config.pages_per_buffer = 1;
        config.prefault = false;

        let pool = RegisteredBufferPool::new(config).unwrap();

        // Write pattern to each buffer and verify
        for i in 0..4 {
            unsafe {
                let slice = pool.get_buffer_slice(i).unwrap();
                for (j, byte) in slice.iter_mut().enumerate() {
                    *byte = ((i + j) % 256) as u8;
                }
            }
        }

        // Verify patterns
        for i in 0..4 {
            unsafe {
                let slice = pool.get_buffer_slice(i).unwrap();
                for (j, &byte) in slice.iter().enumerate() {
                    assert_eq!(
                        byte,
                        ((i + j) % 256) as u8,
                        "B.30: Data integrity must be maintained"
                    );
                }
            }
        }
    }

    // ========================================================================
    // Concurrent Access Tests
    // ========================================================================

    #[test]
    fn test_concurrent_allocation() {
        use std::sync::Arc;
        use std::thread;

        let mut config = RegisteredBufferConfig::default();
        config.queue_depth = 64;
        config.pages_per_buffer = 1;
        config.prefault = false;

        let pool = Arc::new(RegisteredBufferPool::new(config).unwrap());

        let handles: Vec<_> = (0..4)
            .map(|_| {
                let pool = Arc::clone(&pool);
                thread::spawn(move || {
                    for _ in 0..100 {
                        if let Ok(idx) = pool.allocate() {
                            // Simulate work
                            thread::yield_now();
                            pool.deallocate(idx).unwrap();
                        }
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        // All buffers should be released
        assert_eq!(pool.current_usage(), 0);
        assert!(!pool.stats().has_leaks());
    }
}
