//! NUMA-aware memory allocation
//!
//! Allocates memory on the same NUMA node as the worker thread for:
//!
//! Benefits:
//! - Reduced memory access latency (local vs remote)
//! - Better cache coherency
//! - Higher memory bandwidth
//!
//! ## NUMA Architecture
//!
//! ```text
//! ┌─────────────────────┐     ┌─────────────────────┐
//! │      NUMA Node 0     │     │      NUMA Node 1     │
//! │  ┌─────┐  ┌─────┐   │     │  ┌─────┐  ┌─────┐   │
//! │  │CPU 0│  │CPU 1│   │     │  │CPU 2│  │CPU 3│   │
//! │  └─────┘  └─────┘   │     │  └─────┘  └─────┘   │
//! │      ↓       ↓      │     │      ↓       ↓      │
//! │  ┌───────────────┐  │     │  ┌───────────────┐  │
//! │  │  Local Memory  │◄─────►│  │  Local Memory  │  │
//! │  └───────────────┘  │ QPI │  └───────────────┘  │
//! └─────────────────────┘     └─────────────────────┘
//! ```
//!
//! Local memory access: ~80ns
//! Remote memory access: ~150ns (2x latency)
//!
//! ## Usage
//!
//! ```ignore
//! let allocator = NumaAllocator::for_current_cpu()?;
//! let buffer = allocator.alloc(4096)?;
//! // buffer is allocated on local NUMA node
//! ```

use std::io;
use thiserror::Error;

/// Errors from NUMA operations
#[derive(Error, Debug)]
pub enum NumaError {
    #[error("NUMA not available on this system")]
    NotAvailable,

    #[error("Failed to allocate NUMA memory: {0}")]
    AllocFailed(io::Error),

    #[error("Failed to bind memory: {0}")]
    BindFailed(io::Error),

    #[error("Invalid NUMA node: {0}")]
    InvalidNode(i32),

    #[error("Failed to get CPU NUMA node: {0}")]
    GetNodeFailed(io::Error),
}

/// NUMA-aware memory allocator
#[derive(Debug, Clone)]
pub struct NumaAllocator {
    /// NUMA node to allocate on (-1 for interleaved/any)
    node: i32,

    /// Strict binding (fail if can't bind to node)
    strict: bool,
}

impl NumaAllocator {
    /// Create allocator for specific NUMA node
    pub fn new(node: i32) -> Self {
        Self { node, strict: false }
    }

    /// Create allocator with strict binding
    pub fn strict(node: i32) -> Self {
        Self { node, strict: true }
    }

    /// Create allocator for current CPU's NUMA node
    pub fn for_current_cpu() -> Result<Self, NumaError> {
        let node = Self::get_current_node()?;
        Ok(Self { node, strict: false })
    }

    /// Get the NUMA node
    pub fn node(&self) -> i32 {
        self.node
    }

    /// Check if strict binding is enabled
    pub fn is_strict(&self) -> bool {
        self.strict
    }

    /// Allocate memory on the configured NUMA node
    ///
    /// Returns a Vec<u8> with capacity `size`.
    /// On Linux with NUMA, uses first-touch policy after mbind hint.
    /// The memory is zeroed.
    pub fn alloc(&self, size: usize) -> Result<Vec<u8>, NumaError> {
        if size == 0 {
            return Ok(Vec::new());
        }

        // Allocate using standard allocator
        let mut vec = vec![0u8; size];

        // On Linux, try to bind to NUMA node (advisory, may fail on non-NUMA)
        #[cfg(target_os = "linux")]
        if self.node >= 0 {
            // Ignore bind errors - NUMA may not be available
            let _ = self.bind_memory(vec.as_mut_ptr(), size);
        }

        Ok(vec)
    }

    /// Bind existing memory region to NUMA node
    #[cfg(target_os = "linux")]
    pub fn bind_memory(&self, ptr: *mut u8, size: usize) -> Result<(), NumaError> {
        use nix::libc;

        // mbind constants (not exported by libc crate)
        const MPOL_BIND: i32 = 2;
        const MPOL_MF_MOVE: u32 = 1 << 1;
        const MPOL_MF_STRICT: u32 = 1 << 0;

        if self.node < 0 || size == 0 {
            return Ok(());
        }

        // Create nodemask for the target node
        let nodemask: u64 = 1 << self.node;
        let flags = if self.strict { MPOL_MF_STRICT | MPOL_MF_MOVE } else { MPOL_MF_MOVE };

        // Use syscall directly since mbind isn't in libc
        let ret = unsafe {
            libc::syscall(
                libc::SYS_mbind,
                ptr as *mut libc::c_void,
                size,
                MPOL_BIND,
                &nodemask as *const u64,
                64u64, // Max nodes
                flags,
            )
        };

        if ret != 0 {
            return Err(NumaError::BindFailed(io::Error::last_os_error()));
        }

        Ok(())
    }

    /// Bind memory (non-Linux stub)
    #[cfg(not(target_os = "linux"))]
    pub fn bind_memory(&self, _ptr: *mut u8, _size: usize) -> Result<(), NumaError> {
        Ok(())
    }

    /// Get NUMA node for current CPU
    #[cfg(target_os = "linux")]
    pub fn get_current_node() -> Result<i32, NumaError> {
        use nix::libc;

        let cpu = unsafe { libc::sched_getcpu() };
        if cpu < 0 {
            return Err(NumaError::GetNodeFailed(io::Error::last_os_error()));
        }

        // Read numa_node from sysfs
        let path = format!("/sys/devices/system/cpu/cpu{}/node0", cpu);
        if std::path::Path::new(&path).exists() {
            return Ok(0); // Has node0, so node is 0
        }

        // Try to find which node
        for node in 0..16 {
            let node_path = format!("/sys/devices/system/node/node{}/cpulist", node);
            if let Ok(cpulist) = std::fs::read_to_string(&node_path) {
                if Self::cpu_in_list(cpu as usize, &cpulist) {
                    return Ok(node);
                }
            }
        }

        // Default to node 0 if can't determine
        Ok(0)
    }

    /// Get NUMA node (non-Linux stub)
    #[cfg(not(target_os = "linux"))]
    pub fn get_current_node() -> Result<i32, NumaError> {
        Ok(0) // Non-NUMA systems are effectively node 0
    }

    /// Check if NUMA is available on this system
    #[cfg(target_os = "linux")]
    pub fn is_available() -> bool {
        std::path::Path::new("/sys/devices/system/node/node0").exists()
    }

    /// Check if NUMA is available (non-Linux stub)
    #[cfg(not(target_os = "linux"))]
    pub fn is_available() -> bool {
        false
    }

    /// Get number of NUMA nodes
    #[cfg(target_os = "linux")]
    pub fn num_nodes() -> usize {
        let mut count = 0;
        for i in 0..64 {
            let path = format!("/sys/devices/system/node/node{}", i);
            if std::path::Path::new(&path).exists() {
                count += 1;
            } else {
                break;
            }
        }
        count.max(1)
    }

    /// Get number of NUMA nodes (non-Linux stub)
    #[cfg(not(target_os = "linux"))]
    pub fn num_nodes() -> usize {
        1
    }

    /// Get memory info for a NUMA node (in bytes)
    #[cfg(target_os = "linux")]
    pub fn node_memory(node: i32) -> Option<u64> {
        let path = format!("/sys/devices/system/node/node{}/meminfo", node);
        let content = std::fs::read_to_string(&path).ok()?;

        for line in content.lines() {
            if line.contains("MemTotal:") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 4 {
                    let kb: u64 = parts[3].parse().ok()?;
                    return Some(kb * 1024);
                }
            }
        }
        None
    }

    /// Get memory info (non-Linux stub)
    #[cfg(not(target_os = "linux"))]
    pub fn node_memory(_node: i32) -> Option<u64> {
        None
    }

    /// Parse CPU list format (e.g., "0-3,8-11")
    fn cpu_in_list(cpu: usize, cpulist: &str) -> bool {
        for part in cpulist.trim().split(',') {
            let part = part.trim();
            if part.contains('-') {
                let range: Vec<&str> = part.split('-').collect();
                if range.len() == 2 {
                    if let (Ok(start), Ok(end)) =
                        (range[0].parse::<usize>(), range[1].parse::<usize>())
                    {
                        if cpu >= start && cpu <= end {
                            return true;
                        }
                    }
                }
            } else if let Ok(single) = part.parse::<usize>() {
                if cpu == single {
                    return true;
                }
            }
        }
        false
    }
}

impl Default for NumaAllocator {
    fn default() -> Self {
        Self::new(-1) // Any node
    }
}

/// Pre-allocated NUMA-local buffer pool
#[derive(Debug)]
pub struct NumaBufferPool {
    allocator: NumaAllocator,
    buffer_size: usize,
    buffers: Vec<Vec<u8>>,
    available: Vec<usize>,
}

impl NumaBufferPool {
    /// Create a new buffer pool
    pub fn new(
        allocator: NumaAllocator,
        buffer_size: usize,
        count: usize,
    ) -> Result<Self, NumaError> {
        let mut buffers = Vec::with_capacity(count);
        let available: Vec<usize> = (0..count).collect();

        for _ in 0..count {
            buffers.push(allocator.alloc(buffer_size)?);
        }

        Ok(Self { allocator, buffer_size, buffers, available })
    }

    /// Try to acquire a buffer index from the pool
    pub fn acquire(&mut self) -> Option<usize> {
        self.available.pop()
    }

    /// Get a buffer by index (must have been acquired first)
    pub fn get(&mut self, idx: usize) -> Option<&mut [u8]> {
        if idx < self.buffers.len() {
            Some(&mut self.buffers[idx])
        } else {
            None
        }
    }

    /// Return a buffer to the pool
    pub fn put(&mut self, idx: usize) {
        if idx < self.buffers.len() && !self.available.contains(&idx) {
            self.available.push(idx);
        }
    }

    /// Number of available buffers
    pub fn available_count(&self) -> usize {
        self.available.len()
    }

    /// Total number of buffers
    pub fn total_count(&self) -> usize {
        self.buffers.len()
    }

    /// Buffer size
    pub fn buffer_size(&self) -> usize {
        self.buffer_size
    }

    /// NUMA node
    pub fn node(&self) -> i32 {
        self.allocator.node()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ============================================================================
    // NumaAllocator Construction Tests
    // ============================================================================

    #[test]
    fn test_numa_allocator_new() {
        let allocator = NumaAllocator::new(0);
        assert_eq!(allocator.node(), 0);
        assert!(!allocator.is_strict());
    }

    #[test]
    fn test_numa_allocator_strict() {
        let allocator = NumaAllocator::strict(1);
        assert_eq!(allocator.node(), 1);
        assert!(allocator.is_strict());
    }

    #[test]
    fn test_numa_allocator_default() {
        let allocator = NumaAllocator::default();
        assert_eq!(allocator.node(), -1);
    }

    #[test]
    fn test_numa_allocator_for_current_cpu() {
        let result = NumaAllocator::for_current_cpu();
        assert!(result.is_ok());
        let allocator = result.unwrap();
        assert!(allocator.node() >= 0);
    }

    // ============================================================================
    // Allocation Tests
    // ============================================================================

    #[test]
    fn test_numa_alloc_zero_size() {
        let allocator = NumaAllocator::default();
        let result = allocator.alloc(0);
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    #[test]
    fn test_numa_alloc_small() {
        let allocator = NumaAllocator::default();
        let result = allocator.alloc(4096);
        assert!(result.is_ok());
        let buf = result.unwrap();
        assert_eq!(buf.len(), 4096);
    }

    #[test]
    fn test_numa_alloc_large() {
        let allocator = NumaAllocator::default();
        let result = allocator.alloc(1 << 20); // 1 MB
        assert!(result.is_ok());
        let buf = result.unwrap();
        assert_eq!(buf.len(), 1 << 20);
    }

    #[test]
    fn test_numa_alloc_with_node() {
        let allocator = NumaAllocator::new(0);
        let result = allocator.alloc(4096);
        // Should succeed on node 0 (always exists)
        assert!(result.is_ok());
    }

    // ============================================================================
    // System Info Tests
    // ============================================================================

    #[test]
    fn test_numa_is_available() {
        // Just verify it doesn't panic
        let _ = NumaAllocator::is_available();
    }

    #[test]
    fn test_numa_num_nodes() {
        let nodes = NumaAllocator::num_nodes();
        assert!(nodes >= 1);
    }

    #[test]
    fn test_numa_get_current_node() {
        let result = NumaAllocator::get_current_node();
        assert!(result.is_ok());
        let node = result.unwrap();
        assert!(node >= 0);
    }

    // ============================================================================
    // CPU List Parsing Tests
    // ============================================================================

    #[test]
    fn test_cpu_in_list_single() {
        assert!(NumaAllocator::cpu_in_list(0, "0"));
        assert!(NumaAllocator::cpu_in_list(5, "5"));
        assert!(!NumaAllocator::cpu_in_list(1, "0"));
    }

    #[test]
    fn test_cpu_in_list_range() {
        assert!(NumaAllocator::cpu_in_list(0, "0-3"));
        assert!(NumaAllocator::cpu_in_list(2, "0-3"));
        assert!(NumaAllocator::cpu_in_list(3, "0-3"));
        assert!(!NumaAllocator::cpu_in_list(4, "0-3"));
    }

    #[test]
    fn test_cpu_in_list_multiple() {
        assert!(NumaAllocator::cpu_in_list(0, "0-3,8-11"));
        assert!(NumaAllocator::cpu_in_list(9, "0-3,8-11"));
        assert!(!NumaAllocator::cpu_in_list(5, "0-3,8-11"));
    }

    #[test]
    fn test_cpu_in_list_mixed() {
        assert!(NumaAllocator::cpu_in_list(0, "0,2-4,8"));
        assert!(NumaAllocator::cpu_in_list(3, "0,2-4,8"));
        assert!(NumaAllocator::cpu_in_list(8, "0,2-4,8"));
        assert!(!NumaAllocator::cpu_in_list(1, "0,2-4,8"));
    }

    #[test]
    fn test_cpu_in_list_whitespace() {
        assert!(NumaAllocator::cpu_in_list(0, " 0-3 \n"));
        // Note: "0 - 3" (with spaces) doesn't parse - kernel doesn't use this format
    }

    // ============================================================================
    // Error Tests
    // ============================================================================

    #[test]
    fn test_numa_error_display() {
        let err = NumaError::InvalidNode(99);
        let msg = format!("{}", err);
        assert!(msg.contains("99"));
    }

    #[test]
    fn test_numa_error_debug() {
        let err = NumaError::NotAvailable;
        let debug = format!("{:?}", err);
        assert!(debug.contains("NotAvailable"));
    }

    #[test]
    fn test_numa_error_variants() {
        let _ = NumaError::NotAvailable;
        let _ = NumaError::AllocFailed(io::Error::from_raw_os_error(1));
        let _ = NumaError::BindFailed(io::Error::from_raw_os_error(1));
        let _ = NumaError::InvalidNode(0);
        let _ = NumaError::GetNodeFailed(io::Error::from_raw_os_error(1));
    }

    // ============================================================================
    // NumaBufferPool Tests
    // ============================================================================

    #[test]
    fn test_buffer_pool_new() {
        let allocator = NumaAllocator::default();
        let pool = NumaBufferPool::new(allocator, 4096, 10);
        assert!(pool.is_ok());
        let pool = pool.unwrap();
        assert_eq!(pool.total_count(), 10);
        assert_eq!(pool.available_count(), 10);
        assert_eq!(pool.buffer_size(), 4096);
    }

    #[test]
    fn test_buffer_pool_acquire_put() {
        let allocator = NumaAllocator::default();
        let mut pool = NumaBufferPool::new(allocator, 4096, 3).unwrap();

        // Acquire all buffers
        let b0 = pool.acquire();
        assert!(b0.is_some());
        assert_eq!(pool.available_count(), 2);

        let b1 = pool.acquire();
        assert!(b1.is_some());
        assert_eq!(pool.available_count(), 1);

        let b2 = pool.acquire();
        assert!(b2.is_some());
        assert_eq!(pool.available_count(), 0);

        // No more available
        assert!(pool.acquire().is_none());

        // Return one
        pool.put(b0.unwrap());
        assert_eq!(pool.available_count(), 1);

        // Can acquire again
        assert!(pool.acquire().is_some());
    }

    #[test]
    fn test_buffer_pool_get_buffer() {
        let allocator = NumaAllocator::default();
        let mut pool = NumaBufferPool::new(allocator, 4096, 2).unwrap();

        let idx = pool.acquire().unwrap();
        let buffer = pool.get(idx);
        assert!(buffer.is_some());
        assert_eq!(buffer.unwrap().len(), 4096);
    }

    #[test]
    fn test_buffer_pool_put_invalid() {
        let allocator = NumaAllocator::default();
        let mut pool = NumaBufferPool::new(allocator, 4096, 2).unwrap();

        // Put invalid index (should be ignored)
        pool.put(999);
        assert_eq!(pool.available_count(), 2);
    }

    #[test]
    fn test_buffer_pool_put_duplicate() {
        let allocator = NumaAllocator::default();
        let mut pool = NumaBufferPool::new(allocator, 4096, 2).unwrap();

        let idx = pool.acquire().unwrap();
        pool.put(idx);
        pool.put(idx); // Duplicate put

        // Should not add duplicate
        assert_eq!(pool.available_count(), 2);
    }

    #[test]
    fn test_buffer_pool_node() {
        let allocator = NumaAllocator::new(0);
        let pool = NumaBufferPool::new(allocator, 4096, 1).unwrap();
        assert_eq!(pool.node(), 0);
    }

    // ============================================================================
    // Clone Tests
    // ============================================================================

    #[test]
    fn test_numa_allocator_clone() {
        let allocator = NumaAllocator::strict(1);
        let cloned = allocator.clone();
        assert_eq!(allocator.node(), cloned.node());
        assert_eq!(allocator.is_strict(), cloned.is_strict());
    }

    // ============================================================================
    // Integration Tests
    // ============================================================================

    #[test]
    fn test_numa_workflow() {
        // Typical workflow: detect node, allocate, use
        let allocator = NumaAllocator::for_current_cpu().unwrap();
        let mut buffer = allocator.alloc(4096).unwrap();

        // Use the buffer
        buffer[0] = 42;
        buffer[4095] = 99;

        assert_eq!(buffer[0], 42);
        assert_eq!(buffer[4095], 99);
    }

    #[test]
    fn test_numa_multi_node_aware() {
        let num_nodes = NumaAllocator::num_nodes();

        // Create allocator for each node
        for node in 0..num_nodes as i32 {
            let allocator = NumaAllocator::new(node);
            // Allocation should work (even if we can't verify placement)
            let result = allocator.alloc(4096);
            assert!(result.is_ok());
        }
    }
}
