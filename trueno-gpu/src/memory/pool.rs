//! GPU Memory Pool Allocator
//!
//! Implements a page-based memory pool to reduce allocation overhead
//! and track fragmentation per PagedAttention [12].

use std::collections::HashMap;
use std::time::Instant;

/// Memory pool configuration
#[derive(Debug, Clone)]
pub struct PoolConfig {
    /// Total pool size in bytes
    pub total_bytes: u64,
    /// Page size (default: 256KB per PagedAttention)
    pub page_size: u64,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            total_bytes: 1024 * 1024 * 1024, // 1GB default
            page_size: 256 * 1024,           // 256KB pages
        }
    }
}

/// Allocation metadata
#[derive(Debug, Clone)]
pub struct AllocationInfo {
    /// Start page index
    pub start_page: u64,
    /// Number of pages
    pub num_pages: u64,
    /// Size in bytes
    pub size: u64,
    /// Allocation timestamp
    pub timestamp: Instant,
}

/// Unique allocation identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AllocationId(u64);

impl AllocationId {
    /// Create a new unique ID
    #[must_use]
    pub fn new() -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        Self(COUNTER.fetch_add(1, Ordering::Relaxed))
    }
}

impl Default for AllocationId {
    fn default() -> Self {
        Self::new()
    }
}

/// GPU memory pool with fragmentation tracking
pub struct MemoryPool {
    /// Configuration
    config: PoolConfig,
    /// Free page bitmap (true = free)
    free_pages: Vec<bool>,
    /// Allocation metadata
    allocations: HashMap<AllocationId, AllocationInfo>,
    /// Total allocations made
    total_allocations: u64,
}

impl MemoryPool {
    /// Create a new memory pool
    #[must_use]
    pub fn new(config: PoolConfig) -> Self {
        let num_pages = config.total_bytes / config.page_size;
        Self {
            config,
            free_pages: vec![true; num_pages as usize],
            allocations: HashMap::new(),
            total_allocations: 0,
        }
    }

    /// Allocate pages
    pub fn allocate(&mut self, size: u64) -> Option<AllocationId> {
        let pages_needed = (size + self.config.page_size - 1) / self.config.page_size;

        // Find contiguous free pages
        let start = self.find_contiguous_pages(pages_needed)?;

        // Mark as allocated
        for i in start..(start + pages_needed) {
            self.free_pages[i as usize] = false;
        }

        let id = AllocationId::new();
        self.allocations.insert(
            id,
            AllocationInfo {
                start_page: start,
                num_pages: pages_needed,
                size,
                timestamp: Instant::now(),
            },
        );
        self.total_allocations += 1;

        Some(id)
    }

    /// Free an allocation
    pub fn free(&mut self, id: AllocationId) -> bool {
        if let Some(info) = self.allocations.remove(&id) {
            for i in info.start_page..(info.start_page + info.num_pages) {
                self.free_pages[i as usize] = true;
            }
            true
        } else {
            false
        }
    }

    /// Find contiguous free pages
    fn find_contiguous_pages(&self, count: u64) -> Option<u64> {
        let mut consecutive = 0u64;
        let mut start = 0u64;

        for (i, &is_free) in self.free_pages.iter().enumerate() {
            if is_free {
                if consecutive == 0 {
                    start = i as u64;
                }
                consecutive += 1;
                if consecutive >= count {
                    return Some(start);
                }
            } else {
                consecutive = 0;
            }
        }

        None
    }

    /// Calculate fragmentation percentage
    #[must_use]
    pub fn fragmentation_pct(&self) -> f64 {
        let free_count = self.free_pages.iter().filter(|&&f| f).count();
        if free_count == 0 {
            return 0.0;
        }

        let largest_free = self.largest_contiguous_free();
        let fragmentation = 1.0 - (largest_free as f64 / free_count as f64);
        fragmentation * 100.0
    }

    /// Find largest contiguous free region
    fn largest_contiguous_free(&self) -> usize {
        let mut max_consecutive = 0;
        let mut current = 0;

        for &is_free in &self.free_pages {
            if is_free {
                current += 1;
                max_consecutive = max_consecutive.max(current);
            } else {
                current = 0;
            }
        }

        max_consecutive
    }

    /// Get pool statistics
    #[must_use]
    pub fn stats(&self) -> PoolStats {
        let free_pages = self.free_pages.iter().filter(|&&f| f).count();
        let used_pages = self.free_pages.len() - free_pages;

        PoolStats {
            total_pages: self.free_pages.len(),
            free_pages,
            used_pages,
            fragmentation_pct: self.fragmentation_pct(),
            total_allocations: self.total_allocations,
            active_allocations: self.allocations.len(),
        }
    }
}

/// Pool statistics
#[derive(Debug, Clone)]
pub struct PoolStats {
    /// Total pages in pool
    pub total_pages: usize,
    /// Free pages
    pub free_pages: usize,
    /// Used pages
    pub used_pages: usize,
    /// Fragmentation percentage
    pub fragmentation_pct: f64,
    /// Total allocations made
    pub total_allocations: u64,
    /// Currently active allocations
    pub active_allocations: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pool_creation() {
        let config = PoolConfig {
            total_bytes: 1024 * 1024, // 1MB
            page_size: 4096,          // 4KB pages
        };
        let pool = MemoryPool::new(config);

        assert_eq!(pool.free_pages.len(), 256); // 1MB / 4KB = 256 pages
    }

    #[test]
    fn test_allocation() {
        let config = PoolConfig { total_bytes: 1024 * 1024, page_size: 4096 };
        let mut pool = MemoryPool::new(config);

        let id = pool.allocate(8192); // 2 pages
        assert!(id.is_some());

        let stats = pool.stats();
        assert_eq!(stats.used_pages, 2);
        assert_eq!(stats.active_allocations, 1);
    }

    #[test]
    fn test_free() {
        let config = PoolConfig { total_bytes: 1024 * 1024, page_size: 4096 };
        let mut pool = MemoryPool::new(config);

        let id = pool.allocate(8192).expect("test");
        assert!(pool.free(id));

        let stats = pool.stats();
        assert_eq!(stats.used_pages, 0);
        assert_eq!(stats.active_allocations, 0);
    }

    #[test]
    fn test_fragmentation() {
        let config = PoolConfig {
            total_bytes: 40960, // 10 pages
            page_size: 4096,
        };
        let mut pool = MemoryPool::new(config);

        // Allocate every other page to create fragmentation
        let id1 = pool.allocate(4096).expect("test");
        let _id2 = pool.allocate(4096).expect("test");
        let id3 = pool.allocate(4096).expect("test");
        let _id4 = pool.allocate(4096).expect("test");
        let id5 = pool.allocate(4096).expect("test");

        // Free alternating pages
        pool.free(id1);
        pool.free(id3);
        pool.free(id5);

        let stats = pool.stats();
        // 3 free pages, but scattered (high fragmentation)
        assert!(stats.fragmentation_pct > 0.0);
    }

    #[test]
    fn test_allocation_fails_when_full() {
        let config = PoolConfig {
            total_bytes: 4096, // 1 page
            page_size: 4096,
        };
        let mut pool = MemoryPool::new(config);

        let _id1 = pool.allocate(4096).expect("test");
        let id2 = pool.allocate(4096);
        assert!(id2.is_none());
    }
}
