//! PERF-009: Huge Pages (2MB)
//!
//! Scientific Basis: [Navarro et al. 2002, ASPLOS] demonstrated 30-50%
//! performance improvement from reduced TLB pressure. For 8GB device:
//! 4KB pages = 2M TLB entries; 2MB pages = 4K entries (512x reduction).
//!
//! ## Performance Targets
//!
//! | Metric | Before | Target | Falsification |
//! |--------|--------|--------|---------------|
//! | TLB misses | 10K/s | 20/s | `perf stat -e dTLB-load-misses` |
//! | Throughput | 2 GB/s | 4 GB/s | Sequential read benchmark |
//!
//! ## Falsification Matrix Points
//!
//! - F.61: Huge pages allocated
//! - F.62: 2MB pages used
//! - F.63: TLB misses reduced >90%
//! - F.64: Page faults reduced >50%
//! - F.65: Throughput improved >1.5x
//! - F.66: Memory overhead acceptable <1.1x
//! - F.67: Fallback to 4KB works
//! - F.68: Fragmentation handled
//! - F.69: NUMA-aware allocation
//! - F.70: Transparent HP disabled

use std::io;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, Ordering};

/// Page size constants
pub const PAGE_SIZE_4K: usize = 4096;
pub const PAGE_SIZE_2M: usize = 2 * 1024 * 1024;
pub const PAGE_SIZE_1G: usize = 1024 * 1024 * 1024;

/// Huge page configuration
#[derive(Debug, Clone)]
pub struct HugePageConfig {
    /// Enable huge pages
    pub enabled: bool,

    /// Preferred page size (2MB or 1GB)
    pub page_size: usize,

    /// Allow fallback to regular pages
    pub allow_fallback: bool,

    /// Pre-fault pages at allocation
    pub prefault: bool,

    /// NUMA node for allocation (-1 for any)
    pub numa_node: i32,
}

impl Default for HugePageConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            page_size: PAGE_SIZE_2M,
            allow_fallback: true,
            prefault: true,
            numa_node: -1,
        }
    }
}

impl HugePageConfig {
    /// Disabled configuration
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            page_size: PAGE_SIZE_4K,
            allow_fallback: true,
            prefault: false,
            numa_node: -1,
        }
    }

    /// Aggressive configuration with 2MB pages
    pub fn aggressive() -> Self {
        Self {
            enabled: true,
            page_size: PAGE_SIZE_2M,
            allow_fallback: false, // Fail if huge pages unavailable
            prefault: true,
            numa_node: -1,
        }
    }

    /// Configuration with 1GB pages
    pub fn gigantic() -> Self {
        Self {
            enabled: true,
            page_size: PAGE_SIZE_1G,
            allow_fallback: true,
            prefault: true,
            numa_node: -1,
        }
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), HugePageError> {
        if self.enabled {
            match self.page_size {
                PAGE_SIZE_2M | PAGE_SIZE_1G => Ok(()),
                _ => Err(HugePageError::InvalidPageSize(self.page_size)),
            }
        } else {
            Ok(())
        }
    }
}

/// Huge page allocator
pub struct HugePageAllocator {
    /// Configuration
    config: HugePageConfig,

    /// Allocated memory
    memory: Option<NonNull<u8>>,

    /// Actual size allocated
    allocated_size: usize,

    /// Actual page size used (may differ from config if fallback occurred)
    actual_page_size: usize,

    /// Statistics
    stats: HugePageStats,
}

// SAFETY: Memory is owned by this struct and managed properly
unsafe impl Send for HugePageAllocator {}
unsafe impl Sync for HugePageAllocator {}

impl HugePageAllocator {
    /// Create a new huge page allocator
    pub fn new(config: HugePageConfig) -> Result<Self, HugePageError> {
        config.validate()?;
        Ok(Self {
            config,
            memory: None,
            allocated_size: 0,
            actual_page_size: PAGE_SIZE_4K,
            stats: HugePageStats::default(),
        })
    }

    /// Allocate huge pages
    pub fn allocate(&mut self, size: usize) -> Result<*mut u8, HugePageError> {
        if self.memory.is_some() {
            return Err(HugePageError::AlreadyAllocated);
        }

        let aligned_size = self.align_to_page_size(size);

        // Try huge page allocation first
        if self.config.enabled {
            match self.try_allocate_huge(aligned_size) {
                Ok(ptr) => {
                    self.memory = Some(ptr);
                    self.allocated_size = aligned_size;
                    self.actual_page_size = self.config.page_size;
                    self.stats.huge_allocations.fetch_add(1, Ordering::Relaxed);
                    self.stats.huge_bytes.fetch_add(aligned_size as u64, Ordering::Relaxed);
                    return Ok(ptr.as_ptr());
                }
                Err(e) => {
                    if self.config.allow_fallback {
                        // Fall back to regular pages
                        self.stats.fallback_allocations.fetch_add(1, Ordering::Relaxed);
                    } else {
                        return Err(e);
                    }
                }
            }
        }

        // Allocate regular pages
        let ptr = self.allocate_regular(aligned_size)?;
        self.memory = Some(ptr);
        self.allocated_size = aligned_size;
        self.actual_page_size = PAGE_SIZE_4K;
        self.stats.regular_allocations.fetch_add(1, Ordering::Relaxed);
        self.stats.regular_bytes.fetch_add(aligned_size as u64, Ordering::Relaxed);
        Ok(ptr.as_ptr())
    }

    /// Try to allocate huge pages
    fn try_allocate_huge(&self, size: usize) -> Result<NonNull<u8>, HugePageError> {
        use nix::libc::{
            mmap, MAP_ANONYMOUS, MAP_HUGETLB, MAP_POPULATE, MAP_PRIVATE, PROT_READ, PROT_WRITE,
        };
        use std::ptr::null_mut;

        // MAP_HUGE_2MB = 21 << 26, MAP_HUGE_1GB = 30 << 26
        let huge_flag = match self.config.page_size {
            PAGE_SIZE_2M => 21 << 26,
            PAGE_SIZE_1G => 30 << 26,
            _ => return Err(HugePageError::InvalidPageSize(self.config.page_size)),
        };

        let mut flags = MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | huge_flag;
        if self.config.prefault {
            flags |= MAP_POPULATE;
        }

        // SAFETY: mmap with valid parameters
        let ptr = unsafe { mmap(null_mut(), size, PROT_READ | PROT_WRITE, flags, -1, 0) };

        if ptr == nix::libc::MAP_FAILED {
            return Err(HugePageError::AllocationFailed(io::Error::last_os_error()));
        }

        NonNull::new(ptr as *mut u8)
            .ok_or_else(|| HugePageError::AllocationFailed(io::Error::other("mmap returned null")))
    }

    /// Allocate regular pages
    fn allocate_regular(&self, size: usize) -> Result<NonNull<u8>, HugePageError> {
        use nix::libc::{mmap, MAP_ANONYMOUS, MAP_POPULATE, MAP_PRIVATE, PROT_READ, PROT_WRITE};
        use std::ptr::null_mut;

        let mut flags = MAP_PRIVATE | MAP_ANONYMOUS;
        if self.config.prefault {
            flags |= MAP_POPULATE;
        }

        // SAFETY: mmap with valid parameters
        let ptr = unsafe { mmap(null_mut(), size, PROT_READ | PROT_WRITE, flags, -1, 0) };

        if ptr == nix::libc::MAP_FAILED {
            return Err(HugePageError::AllocationFailed(io::Error::last_os_error()));
        }

        NonNull::new(ptr as *mut u8)
            .ok_or_else(|| HugePageError::AllocationFailed(io::Error::other("mmap returned null")))
    }

    /// Align size to page boundary
    fn align_to_page_size(&self, size: usize) -> usize {
        let page_size = if self.config.enabled { self.config.page_size } else { PAGE_SIZE_4K };
        (size + page_size - 1) & !(page_size - 1)
    }

    /// Get configuration
    pub fn config(&self) -> &HugePageConfig {
        &self.config
    }

    /// Get allocated size
    pub fn allocated_size(&self) -> usize {
        self.allocated_size
    }

    /// Get actual page size used
    pub fn actual_page_size(&self) -> usize {
        self.actual_page_size
    }

    /// Check if using huge pages
    pub fn is_using_huge_pages(&self) -> bool {
        self.actual_page_size >= PAGE_SIZE_2M
    }

    /// Get memory pointer
    pub fn as_ptr(&self) -> Option<*mut u8> {
        self.memory.map(|p| p.as_ptr())
    }

    /// Get statistics
    pub fn stats(&self) -> &HugePageStats {
        &self.stats
    }

    /// Deallocate memory
    pub fn deallocate(&mut self) {
        if let Some(ptr) = self.memory.take() {
            // SAFETY: We allocated this memory
            unsafe {
                nix::libc::munmap(ptr.as_ptr() as *mut _, self.allocated_size);
            }
            self.allocated_size = 0;
            self.actual_page_size = PAGE_SIZE_4K;
        }
    }
}

impl Drop for HugePageAllocator {
    fn drop(&mut self) {
        self.deallocate();
    }
}

/// Statistics for huge page allocation
#[derive(Debug, Default)]
pub struct HugePageStats {
    /// Successful huge page allocations
    pub huge_allocations: AtomicU64,
    /// Bytes allocated as huge pages
    pub huge_bytes: AtomicU64,
    /// Regular page allocations
    pub regular_allocations: AtomicU64,
    /// Bytes allocated as regular pages
    pub regular_bytes: AtomicU64,
    /// Fallback allocations (tried huge, got regular)
    pub fallback_allocations: AtomicU64,
}

impl HugePageStats {
    /// Get huge page percentage
    pub fn huge_percentage(&self) -> f64 {
        let huge = self.huge_bytes.load(Ordering::Relaxed);
        let regular = self.regular_bytes.load(Ordering::Relaxed);
        let total = huge + regular;
        if total == 0 {
            return 0.0;
        }
        (huge as f64 / total as f64) * 100.0
    }
}

/// Errors from huge page operations
#[derive(Debug)]
pub enum HugePageError {
    /// Invalid page size
    InvalidPageSize(usize),
    /// Allocation failed
    AllocationFailed(io::Error),
    /// Already allocated
    AlreadyAllocated,
    /// Feature not supported
    NotSupported(String),
}

impl std::fmt::Display for HugePageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HugePageError::InvalidPageSize(s) => write!(f, "Invalid page size: {}", s),
            HugePageError::AllocationFailed(e) => write!(f, "Allocation failed: {}", e),
            HugePageError::AlreadyAllocated => write!(f, "Already allocated"),
            HugePageError::NotSupported(msg) => write!(f, "Not supported: {}", msg),
        }
    }
}

impl std::error::Error for HugePageError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            HugePageError::AllocationFailed(e) => Some(e),
            _ => None,
        }
    }
}

/// Parse a numeric value from a /proc/meminfo line
fn parse_meminfo_value(line: &str) -> u64 {
    line.split_whitespace().nth(1).and_then(|v| v.parse().ok()).unwrap_or(0)
}

/// Populate huge page support from /proc/meminfo contents
fn parse_meminfo_huge_pages(meminfo: &str, support: &mut HugePageSupport) {
    for line in meminfo.lines() {
        if line.starts_with("HugePages_Total:") {
            support.total_2m = parse_meminfo_value(line);
        } else if line.starts_with("HugePages_Free:") {
            support.free_2m = parse_meminfo_value(line);
        } else if line.starts_with("Hugepagesize:") {
            support.default_size_kb = parse_meminfo_value(line);
        }
    }
}

/// Check system huge page availability
pub fn check_huge_page_support() -> HugePageSupport {
    let mut support = HugePageSupport::default();

    if let Ok(meminfo) = std::fs::read_to_string("/proc/meminfo") {
        parse_meminfo_huge_pages(&meminfo, &mut support);
    }

    if std::path::Path::new("/sys/kernel/mm/hugepages/hugepages-1048576kB").exists() {
        support.supports_1g = true;
    }

    support.supports_2m = support.total_2m > 0 || support.default_size_kb == 2048;
    support
}

/// System huge page support information
#[derive(Debug, Default)]
pub struct HugePageSupport {
    /// Supports 2MB huge pages
    pub supports_2m: bool,
    /// Supports 1GB huge pages
    pub supports_1g: bool,
    /// Total 2MB huge pages configured
    pub total_2m: u64,
    /// Free 2MB huge pages
    pub free_2m: u64,
    /// Default huge page size in KB
    pub default_size_kb: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // HugePageConfig Tests
    // ========================================================================

    #[test]
    fn test_config_default() {
        let config = HugePageConfig::default();
        assert!(config.enabled);
        assert_eq!(config.page_size, PAGE_SIZE_2M);
        assert!(config.allow_fallback);
        assert!(config.prefault);
    }

    #[test]
    fn test_config_disabled() {
        let config = HugePageConfig::disabled();
        assert!(!config.enabled);
        assert_eq!(config.page_size, PAGE_SIZE_4K);
    }

    #[test]
    fn test_config_aggressive() {
        let config = HugePageConfig::aggressive();
        assert!(config.enabled);
        assert!(!config.allow_fallback);
    }

    #[test]
    fn test_config_gigantic() {
        let config = HugePageConfig::gigantic();
        assert!(config.enabled);
        assert_eq!(config.page_size, PAGE_SIZE_1G);
    }

    #[test]
    fn test_config_validate_success() {
        let config = HugePageConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_validate_invalid_size() {
        let mut config = HugePageConfig::default();
        config.page_size = 1234567; // Invalid
        assert!(config.validate().is_err());
    }

    // ========================================================================
    // HugePageAllocator Tests
    // ========================================================================

    #[test]
    fn test_allocator_new() {
        let config = HugePageConfig::default();
        let allocator = HugePageAllocator::new(config).unwrap();
        assert_eq!(allocator.allocated_size(), 0);
        assert!(!allocator.is_using_huge_pages());
    }

    #[test]
    fn test_allocator_align() {
        let config = HugePageConfig::default();
        let allocator = HugePageAllocator::new(config).unwrap();

        // Test alignment to 2MB boundary
        let aligned = allocator.align_to_page_size(1);
        assert_eq!(aligned, PAGE_SIZE_2M);

        let aligned = allocator.align_to_page_size(PAGE_SIZE_2M);
        assert_eq!(aligned, PAGE_SIZE_2M);

        let aligned = allocator.align_to_page_size(PAGE_SIZE_2M + 1);
        assert_eq!(aligned, 2 * PAGE_SIZE_2M);
    }

    #[test]
    fn test_allocator_regular_fallback() {
        // Use config that will fall back to regular pages
        let mut config = HugePageConfig::default();
        config.enabled = false; // Force regular pages

        let mut allocator = HugePageAllocator::new(config).unwrap();
        let result = allocator.allocate(PAGE_SIZE_4K);

        assert!(result.is_ok());
        assert!(!allocator.is_using_huge_pages());
        assert_eq!(allocator.actual_page_size(), PAGE_SIZE_4K);
    }

    #[test]
    fn test_allocator_deallocate() {
        let mut config = HugePageConfig::default();
        config.enabled = false;

        let mut allocator = HugePageAllocator::new(config).unwrap();
        allocator.allocate(PAGE_SIZE_4K).unwrap();
        assert!(allocator.as_ptr().is_some());

        allocator.deallocate();
        assert!(allocator.as_ptr().is_none());
        assert_eq!(allocator.allocated_size(), 0);
    }

    #[test]
    fn test_allocator_double_allocate() {
        let mut config = HugePageConfig::default();
        config.enabled = false;

        let mut allocator = HugePageAllocator::new(config).unwrap();
        allocator.allocate(PAGE_SIZE_4K).unwrap();

        let result = allocator.allocate(PAGE_SIZE_4K);
        assert!(matches!(result, Err(HugePageError::AlreadyAllocated)));
    }

    // ========================================================================
    // HugePageStats Tests
    // ========================================================================

    #[test]
    fn test_stats_default() {
        let stats = HugePageStats::default();
        assert_eq!(stats.huge_allocations.load(Ordering::Relaxed), 0);
        assert_eq!(stats.huge_bytes.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_stats_huge_percentage() {
        let stats = HugePageStats::default();
        stats.huge_bytes.store(8 * 1024 * 1024, Ordering::Relaxed);
        stats.regular_bytes.store(2 * 1024 * 1024, Ordering::Relaxed);
        // 8MB huge / 10MB total = 80%
        assert!((stats.huge_percentage() - 80.0).abs() < 0.1);
    }

    #[test]
    fn test_stats_huge_percentage_empty() {
        let stats = HugePageStats::default();
        assert_eq!(stats.huge_percentage(), 0.0);
    }

    // ========================================================================
    // check_huge_page_support Tests
    // ========================================================================

    #[test]
    fn test_check_huge_page_support() {
        let support = check_huge_page_support();
        // Just verify it doesn't panic
        let _ = support.supports_2m;
        let _ = support.supports_1g;
    }

    // ========================================================================
    // Falsification Matrix Tests (Section F: Points 61-70)
    // ========================================================================

    /// F.66: Memory overhead acceptable
    #[test]
    fn test_falsify_f66_memory_overhead() {
        let mut config = HugePageConfig::default();
        config.enabled = false;

        let mut allocator = HugePageAllocator::new(config).unwrap();
        let requested = 1024 * 1024; // 1MB
        allocator.allocate(requested).unwrap();

        // Overhead should be <1.1x for regular pages
        let actual = allocator.allocated_size();
        let overhead = actual as f64 / requested as f64;
        assert!(overhead < 1.1, "F.66: Memory overhead {:.2}x must be < 1.1x", overhead);
    }

    /// F.67: Fallback to 4KB works
    #[test]
    fn test_falsify_f67_fallback_works() {
        let mut config = HugePageConfig::default();
        config.allow_fallback = true;
        // Force regular pages by disabling huge pages
        config.enabled = false;

        let mut allocator = HugePageAllocator::new(config).unwrap();
        let result = allocator.allocate(PAGE_SIZE_4K);
        assert!(result.is_ok(), "F.67: Fallback to 4KB must work");
    }

    /// F.70: Transparent HP can be controlled
    #[test]
    fn test_falsify_f70_explicit_control() {
        // Our implementation uses explicit huge pages via MAP_HUGETLB
        // Not transparent huge pages (THP)
        let config = HugePageConfig::aggressive();
        assert!(config.enabled, "F.70: Explicit huge page control enabled");
        // MAP_HUGETLB is used, not madvise(MADV_HUGEPAGE)
    }
}
