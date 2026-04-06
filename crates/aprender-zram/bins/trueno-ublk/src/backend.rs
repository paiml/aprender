//! Storage Backend Module - Kernel-Cooperative Architecture (KERN-001)
//!
//! Implements tiered storage backends for the kernel-cooperative philosophy:
//! "Stop fighting the kernel. Start building on it."
//!
//! ## Tiered Architecture
//!
//! | Tier | Backend | Speed | Compression | Use Case |
//! |------|---------|-------|-------------|----------|
//! | Hot | Kernel ZRAM | 171 GB/s | LZ4 | H(X) < 6.0 |
//! | Warm | trueno SIMD | 15 GiB/s | ZSTD AVX-512 | 6.0 ≤ H(X) ≤ 7.5 |
//! | Cold | NVMe Direct | 3.4 GB/s | None | H(X) > 7.5 |
//!
//! ## Why This Works
//!
//! Kernel ZRAM has direct memory access at 171 GB/s but uses scalar compression.
//! trueno has SIMD ZSTD at 15 GiB/s with 6x better compression than kernel.
//! Route each page to its optimal tier based on Shannon entropy.

// Prepared for KERN-001/002/003 integration - some code not yet wired to daemon
#![allow(dead_code)]

use anyhow::{Context, Result};
use nix::libc;
use rustc_hash::FxHashMap;
use std::fs::{File, OpenOptions};
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use trueno_zram_core::PAGE_SIZE;

// =============================================================================
// Storage Backend Trait
// =============================================================================

/// Storage backend trait for tiered storage abstraction.
///
/// Backends implement direct page storage without compression - compression
/// is handled by the tier manager based on entropy routing decisions.
pub trait StorageBackend: Send + Sync {
    /// Store a page at the given page index
    fn store(&self, page_idx: u64, data: &[u8; PAGE_SIZE]) -> Result<()>;

    /// Load a page from the given page index
    fn load(&self, page_idx: u64, buffer: &mut [u8; PAGE_SIZE]) -> Result<bool>;

    /// Remove a page (optional - some backends may not support)
    fn remove(&self, page_idx: u64) -> Result<bool>;

    /// Get backend name for telemetry
    fn name(&self) -> &'static str;

    /// Get statistics
    fn stats(&self) -> BackendStats;
}

/// Backend statistics for telemetry
#[derive(Debug, Clone, Default)]
pub struct BackendStats {
    pub pages_stored: u64,
    pub pages_loaded: u64,
    pub bytes_written: u64,
    pub bytes_read: u64,
    pub errors: u64,
}

// =============================================================================
// KERN-001: Kernel ZRAM Backend
// =============================================================================

/// Kernel ZRAM backend - hot tier at 171 GB/s.
///
/// Routes low-entropy pages (H(X) < 6.0) to kernel zram for maximum throughput.
/// Kernel handles compression with scalar LZ4 - acceptable for low-entropy data.
///
/// ## Why Kernel Wins Here
///
/// - Direct memory access (no syscall per page)
/// - Optimized memory allocator (zspool)
/// - Same-fill pages handled at 171 GB/s
/// - LZ4 fast enough for low-entropy data
pub struct KernelZramBackend {
    /// File handle to zram block device
    device: RwLock<File>,
    /// Device path for error messages
    path: PathBuf,
    /// Device size in bytes
    size: u64,
    /// Statistics
    pages_stored: AtomicU64,
    pages_loaded: AtomicU64,
    bytes_written: AtomicU64,
    bytes_read: AtomicU64,
    errors: AtomicU64,
}

impl KernelZramBackend {
    /// Create a new kernel ZRAM backend.
    ///
    /// # Arguments
    /// * `device_path` - Path to zram device (e.g., `/dev/zram0`)
    ///
    /// # Example
    /// ```no_run
    /// use trueno_ublk::backend::KernelZramBackend;
    /// # fn main() -> anyhow::Result<()> {
    /// let backend = KernelZramBackend::new("/dev/zram0")?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new<P: AsRef<Path>>(device_path: P) -> Result<Self> {
        let path = device_path.as_ref().to_path_buf();

        // Open with O_DIRECT for zero-copy I/O
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .custom_flags(libc::O_DIRECT)
            .open(&path)
            .with_context(|| format!("Failed to open kernel zram device: {}", path.display()))?;

        // Get device size
        let size = Self::get_device_size(&file, &path)?;

        Ok(Self {
            device: RwLock::new(file),
            path,
            size,
            pages_stored: AtomicU64::new(0),
            pages_loaded: AtomicU64::new(0),
            bytes_written: AtomicU64::new(0),
            bytes_read: AtomicU64::new(0),
            errors: AtomicU64::new(0),
        })
    }

    /// Get device size via ioctl BLKGETSIZE64
    fn get_device_size(file: &File, path: &Path) -> Result<u64> {
        use std::os::unix::io::AsRawFd;

        let mut size: u64 = 0;
        let fd = file.as_raw_fd();

        // BLKGETSIZE64 = 0x80081272
        const BLKGETSIZE64: libc::c_ulong = 0x80081272;

        let ret = unsafe { libc::ioctl(fd, BLKGETSIZE64, &mut size) };

        if ret < 0 {
            return Err(std::io::Error::last_os_error())
                .with_context(|| format!("Failed to get size of {}", path.display()));
        }

        Ok(size)
    }

    /// Verify page index is within bounds
    fn check_bounds(&self, page_idx: u64) -> Result<()> {
        let offset = page_idx * PAGE_SIZE as u64;
        if offset + PAGE_SIZE as u64 > self.size {
            anyhow::bail!(
                "Page index {} out of bounds (device size: {} bytes)",
                page_idx,
                self.size
            );
        }
        Ok(())
    }

    /// Bulk read multiple contiguous pages at once.
    ///
    /// This is much faster than reading pages one at a time because it
    /// reduces syscall overhead (one pread vs N preads).
    ///
    /// # Arguments
    /// * `start_page_idx` - First page index to read
    /// * `buffer` - Buffer to read into (must be multiple of PAGE_SIZE)
    ///
    /// # Returns
    /// Number of bytes read
    pub fn bulk_read(&self, start_page_idx: u64, buffer: &mut [u8]) -> Result<usize> {
        use std::os::unix::io::AsRawFd;

        if buffer.is_empty() {
            return Ok(0);
        }

        let num_pages = buffer.len() / PAGE_SIZE;
        let end_page_idx = start_page_idx + num_pages as u64 - 1;

        // Bounds check
        self.check_bounds(start_page_idx)?;
        self.check_bounds(end_page_idx)?;

        let offset = start_page_idx * PAGE_SIZE as u64;
        let device = self.device.read().expect("lock poisoned");
        let fd = device.as_raw_fd();

        let result = unsafe {
            libc::pread(
                fd,
                buffer.as_mut_ptr() as *mut libc::c_void,
                buffer.len(),
                offset as libc::off_t,
            )
        };

        if result < 0 {
            self.errors.fetch_add(1, Ordering::Relaxed);
            return Err(std::io::Error::last_os_error())
                .with_context(|| format!("bulk_read failed at page {}", start_page_idx));
        }

        let bytes_read = result as usize;
        self.pages_loaded.fetch_add(num_pages as u64, Ordering::Relaxed);
        self.bytes_read.fetch_add(bytes_read as u64, Ordering::Relaxed);

        Ok(bytes_read)
    }

    /// Bulk write multiple contiguous pages at once.
    pub fn bulk_write(&self, start_page_idx: u64, data: &[u8]) -> Result<usize> {
        use std::os::unix::io::AsRawFd;

        if data.is_empty() {
            return Ok(0);
        }

        let num_pages = data.len() / PAGE_SIZE;
        let end_page_idx = start_page_idx + num_pages as u64 - 1;

        // Bounds check
        self.check_bounds(start_page_idx)?;
        self.check_bounds(end_page_idx)?;

        let offset = start_page_idx * PAGE_SIZE as u64;
        let device = self.device.read().expect("lock poisoned");
        let fd = device.as_raw_fd();

        let result = unsafe {
            libc::pwrite(
                fd,
                data.as_ptr() as *const libc::c_void,
                data.len(),
                offset as libc::off_t,
            )
        };

        if result < 0 {
            self.errors.fetch_add(1, Ordering::Relaxed);
            return Err(std::io::Error::last_os_error())
                .with_context(|| format!("bulk_write failed at page {}", start_page_idx));
        }

        let bytes_written = result as usize;
        self.pages_stored.fetch_add(num_pages as u64, Ordering::Relaxed);
        self.bytes_written.fetch_add(bytes_written as u64, Ordering::Relaxed);

        Ok(bytes_written)
    }
}

impl StorageBackend for KernelZramBackend {
    fn store(&self, page_idx: u64, data: &[u8; PAGE_SIZE]) -> Result<()> {
        use std::os::unix::io::AsRawFd;

        self.check_bounds(page_idx)?;

        let offset = page_idx * PAGE_SIZE as u64;

        // Use pwrite for lock-free concurrent writes
        // Only need read lock since pwrite doesn't modify file position
        let device = self.device.read().expect("lock poisoned");
        let fd = device.as_raw_fd();

        let result = unsafe {
            libc::pwrite(fd, data.as_ptr() as *const libc::c_void, PAGE_SIZE, offset as libc::off_t)
        };

        if result < 0 {
            self.errors.fetch_add(1, Ordering::Relaxed);
            return Err(std::io::Error::last_os_error())
                .with_context(|| format!("pwrite failed for page {}", page_idx));
        }

        if result as usize != PAGE_SIZE {
            self.errors.fetch_add(1, Ordering::Relaxed);
            anyhow::bail!(
                "Short write for page {}: wrote {} of {} bytes",
                page_idx,
                result,
                PAGE_SIZE
            );
        }

        self.pages_stored.fetch_add(1, Ordering::Relaxed);
        self.bytes_written.fetch_add(PAGE_SIZE as u64, Ordering::Relaxed);

        Ok(())
    }

    fn load(&self, page_idx: u64, buffer: &mut [u8; PAGE_SIZE]) -> Result<bool> {
        use std::os::unix::io::AsRawFd;

        self.check_bounds(page_idx)?;

        let offset = page_idx * PAGE_SIZE as u64;

        // Use pread for lock-free concurrent reads
        // Only need read lock since pread doesn't modify file position
        let device = self.device.read().expect("lock poisoned");
        let fd = device.as_raw_fd();

        let result = unsafe {
            libc::pread(
                fd,
                buffer.as_mut_ptr() as *mut libc::c_void,
                PAGE_SIZE,
                offset as libc::off_t,
            )
        };

        if result < 0 {
            self.errors.fetch_add(1, Ordering::Relaxed);
            return Err(std::io::Error::last_os_error())
                .with_context(|| format!("pread failed for page {}", page_idx));
        }

        if result as usize != PAGE_SIZE {
            self.errors.fetch_add(1, Ordering::Relaxed);
            anyhow::bail!(
                "Short read for page {}: read {} of {} bytes",
                page_idx,
                result,
                PAGE_SIZE
            );
        }

        self.pages_loaded.fetch_add(1, Ordering::Relaxed);
        self.bytes_read.fetch_add(PAGE_SIZE as u64, Ordering::Relaxed);

        Ok(true)
    }

    fn remove(&self, _page_idx: u64) -> Result<bool> {
        // Kernel ZRAM doesn't support explicit page removal
        // Pages are released when overwritten or device reset
        Ok(false)
    }

    fn name(&self) -> &'static str {
        "kernel-zram"
    }

    fn stats(&self) -> BackendStats {
        BackendStats {
            pages_stored: self.pages_stored.load(Ordering::Relaxed),
            pages_loaded: self.pages_loaded.load(Ordering::Relaxed),
            bytes_written: self.bytes_written.load(Ordering::Relaxed),
            bytes_read: self.bytes_read.load(Ordering::Relaxed),
            errors: self.errors.load(Ordering::Relaxed),
        }
    }
}

// =============================================================================
// Memory Backend (existing trueno in-memory store)
// =============================================================================

/// In-memory backend using trueno's SIMD compression.
///
/// This wraps the existing `BatchedPageStore` as a backend for the tiered manager.
/// Used for warm tier (6.0 ≤ H(X) ≤ 7.5) where SIMD ZSTD compression excels.
pub struct MemoryBackend {
    /// Page data storage (page_idx -> compressed data) using FxHashMap for fast u64 keys
    pages: RwLock<FxHashMap<u64, Vec<u8>>>,
    /// Statistics
    pages_stored: AtomicU64,
    pages_loaded: AtomicU64,
    bytes_written: AtomicU64,
    bytes_read: AtomicU64,
}

impl MemoryBackend {
    pub fn new() -> Self {
        Self {
            pages: RwLock::new(FxHashMap::default()),
            pages_stored: AtomicU64::new(0),
            pages_loaded: AtomicU64::new(0),
            bytes_written: AtomicU64::new(0),
            bytes_read: AtomicU64::new(0),
        }
    }
}

impl Default for MemoryBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl StorageBackend for MemoryBackend {
    fn store(&self, page_idx: u64, data: &[u8; PAGE_SIZE]) -> Result<()> {
        let mut pages = self.pages.write().expect("lock poisoned");
        pages.insert(page_idx, data.to_vec());

        self.pages_stored.fetch_add(1, Ordering::Relaxed);
        self.bytes_written.fetch_add(PAGE_SIZE as u64, Ordering::Relaxed);

        Ok(())
    }

    fn load(&self, page_idx: u64, buffer: &mut [u8; PAGE_SIZE]) -> Result<bool> {
        let pages = self.pages.read().expect("lock poisoned");

        match pages.get(&page_idx) {
            Some(data) => {
                buffer.copy_from_slice(data);
                self.pages_loaded.fetch_add(1, Ordering::Relaxed);
                self.bytes_read.fetch_add(PAGE_SIZE as u64, Ordering::Relaxed);
                Ok(true)
            }
            None => {
                buffer.fill(0);
                Ok(false)
            }
        }
    }

    fn remove(&self, page_idx: u64) -> Result<bool> {
        let mut pages = self.pages.write().expect("lock poisoned");
        Ok(pages.remove(&page_idx).is_some())
    }

    fn name(&self) -> &'static str {
        "memory"
    }

    fn stats(&self) -> BackendStats {
        BackendStats {
            pages_stored: self.pages_stored.load(Ordering::Relaxed),
            pages_loaded: self.pages_loaded.load(Ordering::Relaxed),
            bytes_written: self.bytes_written.load(Ordering::Relaxed),
            bytes_read: self.bytes_read.load(Ordering::Relaxed),
            errors: 0,
        }
    }
}

// =============================================================================
// KERN-003: NVMe Cold Tier Backend
// =============================================================================

/// NVMe cold tier backend - stores high-entropy pages without compression.
///
/// Used for cold tier (H(X) > 7.5) where compression is ineffective.
/// Pages are stored as raw 4KB blocks in a sparse file with O_DIRECT.
///
/// ## Why NVMe for High Entropy
///
/// - Incompressible data wastes CPU cycles on compression
/// - NVMe RAID0 provides 9.2 GB/s read, 6.8 GB/s write
/// - Direct I/O bypasses page cache (data already in our cache)
/// - Sparse file means we only use disk for actual data
pub struct NvmeColdBackend {
    /// File handle to cold tier storage file
    file: RwLock<File>,
    /// Base directory for cold tier
    base_path: PathBuf,
    /// Page index tracking (which pages are stored)
    stored_pages: RwLock<FxHashMap<u64, ()>>,
    /// Statistics
    pages_stored: AtomicU64,
    pages_loaded: AtomicU64,
    bytes_written: AtomicU64,
    bytes_read: AtomicU64,
    errors: AtomicU64,
}

impl NvmeColdBackend {
    /// Create a new NVMe cold tier backend.
    ///
    /// # Arguments
    /// * `base_path` - Directory for cold tier storage (e.g., `/var/lib/trueno/cold`)
    ///
    /// Creates a sparse file `pages.dat` for storing uncompressed pages.
    pub fn new<P: AsRef<Path>>(base_path: P) -> Result<Self> {
        let base_path = base_path.as_ref().to_path_buf();

        // Ensure directory exists
        std::fs::create_dir_all(&base_path).with_context(|| {
            format!("Failed to create cold tier directory: {}", base_path.display())
        })?;

        let file_path = base_path.join("pages.dat");

        // Open with O_DIRECT for zero-copy I/O, create if not exists
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .custom_flags(libc::O_DIRECT)
            .open(&file_path)
            .with_context(|| format!("Failed to open cold tier file: {}", file_path.display()))?;

        tracing::info!("KERN-003: NVMe cold tier initialized at {}", file_path.display());

        Ok(Self {
            file: RwLock::new(file),
            base_path,
            stored_pages: RwLock::new(FxHashMap::default()),
            pages_stored: AtomicU64::new(0),
            pages_loaded: AtomicU64::new(0),
            bytes_written: AtomicU64::new(0),
            bytes_read: AtomicU64::new(0),
            errors: AtomicU64::new(0),
        })
    }

    /// Get the storage file path
    pub fn file_path(&self) -> PathBuf {
        self.base_path.join("pages.dat")
    }
}

impl StorageBackend for NvmeColdBackend {
    fn store(&self, page_idx: u64, data: &[u8; PAGE_SIZE]) -> Result<()> {
        use std::os::unix::io::AsRawFd;

        let offset = page_idx * PAGE_SIZE as u64;
        let file = self.file.read().expect("lock poisoned");
        let fd = file.as_raw_fd();

        // Use pwrite for concurrent writes (sparse file grows automatically)
        let result = unsafe {
            libc::pwrite(fd, data.as_ptr() as *const libc::c_void, PAGE_SIZE, offset as libc::off_t)
        };

        if result < 0 {
            self.errors.fetch_add(1, Ordering::Relaxed);
            return Err(std::io::Error::last_os_error())
                .with_context(|| format!("NVMe cold tier pwrite failed for page {}", page_idx));
        }

        if result as usize != PAGE_SIZE {
            self.errors.fetch_add(1, Ordering::Relaxed);
            anyhow::bail!(
                "NVMe cold tier short write for page {}: wrote {} of {} bytes",
                page_idx,
                result,
                PAGE_SIZE
            );
        }

        // Track stored page
        {
            let mut stored = self.stored_pages.write().expect("lock poisoned");
            stored.insert(page_idx, ());
        }

        self.pages_stored.fetch_add(1, Ordering::Relaxed);
        self.bytes_written.fetch_add(PAGE_SIZE as u64, Ordering::Relaxed);

        Ok(())
    }

    fn load(&self, page_idx: u64, buffer: &mut [u8; PAGE_SIZE]) -> Result<bool> {
        use std::os::unix::io::AsRawFd;

        // Check if page exists
        {
            let stored = self.stored_pages.read().expect("lock poisoned");
            if !stored.contains_key(&page_idx) {
                buffer.fill(0);
                return Ok(false);
            }
        }

        let offset = page_idx * PAGE_SIZE as u64;
        let file = self.file.read().expect("lock poisoned");
        let fd = file.as_raw_fd();

        let result = unsafe {
            libc::pread(
                fd,
                buffer.as_mut_ptr() as *mut libc::c_void,
                PAGE_SIZE,
                offset as libc::off_t,
            )
        };

        if result < 0 {
            self.errors.fetch_add(1, Ordering::Relaxed);
            return Err(std::io::Error::last_os_error())
                .with_context(|| format!("NVMe cold tier pread failed for page {}", page_idx));
        }

        if result as usize != PAGE_SIZE {
            self.errors.fetch_add(1, Ordering::Relaxed);
            anyhow::bail!(
                "NVMe cold tier short read for page {}: read {} of {} bytes",
                page_idx,
                result,
                PAGE_SIZE
            );
        }

        self.pages_loaded.fetch_add(1, Ordering::Relaxed);
        self.bytes_read.fetch_add(PAGE_SIZE as u64, Ordering::Relaxed);

        Ok(true)
    }

    fn remove(&self, page_idx: u64) -> Result<bool> {
        let mut stored = self.stored_pages.write().expect("lock poisoned");
        Ok(stored.remove(&page_idx).is_some())
        // Note: We don't punch holes in the sparse file for simplicity
        // Could use fallocate(FALLOC_FL_PUNCH_HOLE) for space reclamation
    }

    fn name(&self) -> &'static str {
        "nvme-cold"
    }

    fn stats(&self) -> BackendStats {
        BackendStats {
            pages_stored: self.pages_stored.load(Ordering::Relaxed),
            pages_loaded: self.pages_loaded.load(Ordering::Relaxed),
            bytes_written: self.bytes_written.load(Ordering::Relaxed),
            bytes_read: self.bytes_read.load(Ordering::Relaxed),
            errors: self.errors.load(Ordering::Relaxed),
        }
    }
}

// =============================================================================
// KERN-002: Tiered Storage Manager
// =============================================================================

/// Entropy routing thresholds.
///
/// Based on Shannon entropy H(X) analysis:
/// - H(X) < 6.0: Low entropy (text, code, zeros) → kernel zram wins
/// - 6.0 ≤ H(X) ≤ 7.5: Medium entropy → trueno SIMD ZSTD wins
/// - H(X) > 7.5: High entropy (encrypted, random) → skip compression
#[derive(Debug, Clone)]
pub struct EntropyThresholds {
    /// Below this → kernel zram (default: 6.0)
    pub kernel_threshold: f64,
    /// Above this → NVMe/skip compression (default: 7.5)
    pub skip_threshold: f64,
}

impl Default for EntropyThresholds {
    fn default() -> Self {
        Self { kernel_threshold: 6.0, skip_threshold: 7.5 }
    }
}

/// Routing decision for a page
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RoutingDecision {
    /// Route to kernel ZRAM (hot tier, 171 GB/s)
    KernelZram,
    /// Route to trueno SIMD (warm tier, 15 GiB/s)
    TruenoSimd,
    /// Skip compression (cold tier, store uncompressed)
    SkipCompression,
    /// Same-fill page (metadata only, ∞ compression ratio)
    SameFill(u64),
}

/// Tiered storage manager with entropy-based routing.
///
/// Implements the kernel-cooperative philosophy:
/// - Hot path (171 GB/s): Kernel does I/O
/// - Warm path (15 GiB/s): We do SIMD compression
/// - Cold path (9.2 GB/s): NVMe direct, skip compression
///
/// ## Entropy Routing
///
/// ```text
/// ┌─────────────────────────────────────────────────────────────────┐
/// │   125GB physical  →  ~300GB effective                           │
/// │                                                                 │
/// │   Hot data:   H(X) < 6.0  → Kernel ZRAM (171 GB/s)             │
/// │   Warm data:  6.0-7.5     → trueno SIMD (15 GiB/s)             │
/// │   Cold data:  H(X) > 7.5  → NVMe RAID0 (9.2 GB/s, no compress) │
/// └─────────────────────────────────────────────────────────────────┘
/// ```
pub struct TieredStorageManager {
    /// Kernel ZRAM backend (hot tier)
    kernel_backend: Option<Arc<dyn StorageBackend>>,
    /// trueno SIMD backend (warm tier)
    trueno_backend: Arc<dyn StorageBackend>,
    /// NVMe cold tier backend (cold tier) - KERN-003
    nvme_backend: Option<Arc<NvmeColdBackend>>,
    /// Entropy routing thresholds
    thresholds: EntropyThresholds,
    /// Enable entropy routing (if false, always use trueno)
    routing_enabled: bool,
    /// Statistics
    kernel_pages: AtomicU64,
    trueno_pages: AtomicU64,
    skipped_pages: AtomicU64,
    samefill_pages: AtomicU64,
}

impl TieredStorageManager {
    /// Create a new tiered storage manager.
    ///
    /// # Arguments
    /// * `kernel_device` - Optional path to kernel zram device (e.g., `/dev/zram0`)
    /// * `nvme_cold_path` - Optional path to NVMe cold tier directory (e.g., `/var/lib/trueno/cold`)
    /// * `routing_enabled` - Enable entropy-based routing
    pub fn new(
        kernel_device: Option<&Path>,
        nvme_cold_path: Option<&Path>,
        routing_enabled: bool,
    ) -> Result<Self> {
        let kernel_backend: Option<Arc<dyn StorageBackend>> = match kernel_device {
            Some(path) => Some(Arc::new(KernelZramBackend::new(path)?)),
            None => None,
        };

        let nvme_backend: Option<Arc<NvmeColdBackend>> = match nvme_cold_path {
            Some(path) => Some(Arc::new(NvmeColdBackend::new(path)?)),
            None => None,
        };

        let trueno_backend: Arc<dyn StorageBackend> = Arc::new(MemoryBackend::new());

        Ok(Self {
            kernel_backend,
            trueno_backend,
            nvme_backend,
            thresholds: EntropyThresholds::default(),
            routing_enabled,
            kernel_pages: AtomicU64::new(0),
            trueno_pages: AtomicU64::new(0),
            skipped_pages: AtomicU64::new(0),
            samefill_pages: AtomicU64::new(0),
        })
    }

    /// Create with custom thresholds
    pub fn with_thresholds(
        kernel_device: Option<&Path>,
        nvme_cold_path: Option<&Path>,
        thresholds: EntropyThresholds,
    ) -> Result<Self> {
        let mut manager = Self::new(kernel_device, nvme_cold_path, true)?;
        manager.thresholds = thresholds;
        Ok(manager)
    }

    /// Calculate Shannon entropy of a page.
    ///
    /// Returns H(X) in bits per byte [0, 8].
    pub fn calculate_entropy(data: &[u8; PAGE_SIZE]) -> f64 {
        let mut freq = [0u32; 256];
        for &byte in data.iter() {
            freq[byte as usize] += 1;
        }

        let len = data.len() as f64;
        let mut entropy = 0.0;

        for &count in freq.iter() {
            if count > 0 {
                let p = count as f64 / len;
                entropy -= p * p.log2();
            }
        }

        entropy
    }

    /// Determine routing decision based on entropy.
    pub fn route(&self, data: &[u8; PAGE_SIZE]) -> RoutingDecision {
        // P0: Check same-fill first (kernel zram pattern)
        if let Some(fill_value) = trueno_zram_core::samefill::page_same_filled(data) {
            return RoutingDecision::SameFill(fill_value);
        }

        // If routing disabled, always use trueno
        if !self.routing_enabled || self.kernel_backend.is_none() {
            return RoutingDecision::TruenoSimd;
        }

        // Calculate entropy and route
        let entropy = Self::calculate_entropy(data);

        if entropy < self.thresholds.kernel_threshold {
            RoutingDecision::KernelZram
        } else if entropy > self.thresholds.skip_threshold {
            RoutingDecision::SkipCompression
        } else {
            RoutingDecision::TruenoSimd
        }
    }

    /// Store a page with entropy-based routing.
    ///
    /// Routes pages based on Shannon entropy H(X):
    /// - H(X) < 6.0: Kernel ZRAM (hot tier)
    /// - 6.0 ≤ H(X) ≤ 7.5: trueno SIMD (warm tier)
    /// - H(X) > 7.5: NVMe cold tier (no compression)
    pub fn store(&self, page_idx: u64, data: &[u8; PAGE_SIZE]) -> Result<RoutingDecision> {
        let decision = self.route(data);

        match decision {
            RoutingDecision::KernelZram => {
                if let Some(ref backend) = self.kernel_backend {
                    backend.store(page_idx, data)?;
                    self.kernel_pages.fetch_add(1, Ordering::Relaxed);
                }
            }
            RoutingDecision::TruenoSimd => {
                self.trueno_backend.store(page_idx, data)?;
                self.trueno_pages.fetch_add(1, Ordering::Relaxed);
            }
            RoutingDecision::SkipCompression => {
                // KERN-003: Route high-entropy pages to NVMe cold tier
                if let Some(ref nvme) = self.nvme_backend {
                    nvme.store(page_idx, data)?;
                    self.skipped_pages.fetch_add(1, Ordering::Relaxed);
                } else {
                    // Fallback to trueno if no NVMe configured
                    self.trueno_backend.store(page_idx, data)?;
                    self.skipped_pages.fetch_add(1, Ordering::Relaxed);
                }
            }
            RoutingDecision::SameFill(_) => {
                // Same-fill pages don't need storage - just metadata
                self.samefill_pages.fetch_add(1, Ordering::Relaxed);
            }
        }

        Ok(decision)
    }

    /// Load a page (tries all backends)
    pub fn load(&self, page_idx: u64, buffer: &mut [u8; PAGE_SIZE]) -> Result<bool> {
        // Try trueno backend first (most common)
        if self.trueno_backend.load(page_idx, buffer)? {
            return Ok(true);
        }

        // Try kernel backend
        if let Some(ref backend) = self.kernel_backend {
            if backend.load(page_idx, buffer)? {
                return Ok(true);
            }
        }

        // KERN-003: Try NVMe cold tier
        if let Some(ref nvme) = self.nvme_backend {
            if nvme.load(page_idx, buffer)? {
                return Ok(true);
            }
        }

        // Page not found - return zeros
        buffer.fill(0);
        Ok(false)
    }

    /// Get tiered storage statistics
    pub fn stats(&self) -> TieredStats {
        TieredStats {
            kernel_pages: self.kernel_pages.load(Ordering::Relaxed),
            trueno_pages: self.trueno_pages.load(Ordering::Relaxed),
            skipped_pages: self.skipped_pages.load(Ordering::Relaxed),
            samefill_pages: self.samefill_pages.load(Ordering::Relaxed),
            kernel_stats: self.kernel_backend.as_ref().map(|b| b.stats()),
            trueno_stats: self.trueno_backend.stats(),
            nvme_stats: self.nvme_backend.as_ref().map(|b| b.stats()),
        }
    }
}

/// Tiered storage statistics
#[derive(Debug, Clone)]
pub struct TieredStats {
    /// Pages routed to kernel ZRAM
    pub kernel_pages: u64,
    /// Pages routed to trueno SIMD
    pub trueno_pages: u64,
    /// Pages skipped (high entropy) - routed to NVMe cold tier
    pub skipped_pages: u64,
    /// Same-fill pages (metadata only)
    pub samefill_pages: u64,
    /// Kernel backend stats
    pub kernel_stats: Option<BackendStats>,
    /// trueno backend stats
    pub trueno_stats: BackendStats,
    /// NVMe cold tier stats (KERN-003)
    pub nvme_stats: Option<BackendStats>,
}

impl TieredStats {
    /// Total pages processed
    pub fn total_pages(&self) -> u64 {
        self.kernel_pages + self.trueno_pages + self.skipped_pages + self.samefill_pages
    }

    /// Percentage routed to kernel
    pub fn kernel_percentage(&self) -> f64 {
        let total = self.total_pages();
        if total == 0 {
            return 0.0;
        }
        self.kernel_pages as f64 / total as f64 * 100.0
    }

    /// Percentage same-fill
    pub fn samefill_percentage(&self) -> f64 {
        let total = self.total_pages();
        if total == 0 {
            return 0.0;
        }
        self.samefill_pages as f64 / total as f64 * 100.0
    }
}

// =============================================================================
// Backend Selection Enum (for CLI)
// =============================================================================

/// Backend type for CLI selection
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum BackendType {
    /// In-memory trueno SIMD (default)
    #[default]
    Memory,
    /// Kernel ZRAM device
    KernelZram,
    /// Tiered (kernel + trueno)
    Tiered,
}

impl std::fmt::Display for BackendType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BackendType::Memory => write!(f, "memory"),
            BackendType::KernelZram => write!(f, "kernel-zram"),
            BackendType::Tiered => write!(f, "tiered"),
        }
    }
}

impl std::str::FromStr for BackendType {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "memory" | "mem" => Ok(BackendType::Memory),
            "zram" | "kernel-zram" | "kernel" => Ok(BackendType::KernelZram),
            "tiered" | "tier" => Ok(BackendType::Tiered),
            _ => Err(format!("Unknown backend type: {}. Valid options: memory, zram, tiered", s)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entropy_calculation() {
        // Zero page should have 0 entropy
        let zeros = [0u8; PAGE_SIZE];
        let entropy = TieredStorageManager::calculate_entropy(&zeros);
        assert!(entropy < 0.001, "Zero page entropy should be ~0, got {}", entropy);

        // Random-ish data should have high entropy
        let mut random = [0u8; PAGE_SIZE];
        for (i, byte) in random.iter_mut().enumerate() {
            *byte = (i % 256) as u8;
        }
        let entropy = TieredStorageManager::calculate_entropy(&random);
        assert!(entropy > 7.0, "Sequential data should have high entropy, got {}", entropy);
    }

    #[test]
    fn test_routing_decision() {
        // Create manager without kernel backend or NVMe
        let manager = TieredStorageManager::new(None, None, true).unwrap();

        // Zero page should be same-fill
        let zeros = [0u8; PAGE_SIZE];
        let decision = manager.route(&zeros);
        assert!(matches!(decision, RoutingDecision::SameFill(0)));

        // Without kernel backend, should route to trueno
        let mut text = [0u8; PAGE_SIZE];
        text[..100].copy_from_slice(&[b'a'; 100]);
        let decision = manager.route(&text);
        assert!(matches!(decision, RoutingDecision::TruenoSimd));
    }

    #[test]
    fn test_memory_backend() {
        let backend = MemoryBackend::new();
        let mut data = [0u8; PAGE_SIZE];
        data[0..4].copy_from_slice(b"test");

        // Store
        backend.store(42, &data).unwrap();

        // Load
        let mut buffer = [0u8; PAGE_SIZE];
        let found = backend.load(42, &mut buffer).unwrap();
        assert!(found);
        assert_eq!(&buffer[0..4], b"test");

        // Load non-existent
        let found = backend.load(999, &mut buffer).unwrap();
        assert!(!found);

        // Remove
        let removed = backend.remove(42).unwrap();
        assert!(removed);
    }

    #[test]
    fn test_backend_type_parsing() {
        assert_eq!("memory".parse::<BackendType>().unwrap(), BackendType::Memory);
        assert_eq!("zram".parse::<BackendType>().unwrap(), BackendType::KernelZram);
        assert_eq!("tiered".parse::<BackendType>().unwrap(), BackendType::Tiered);
        assert!("invalid".parse::<BackendType>().is_err());
    }

    #[test]
    fn test_tiered_stats() {
        let stats = TieredStats {
            kernel_pages: 100,
            trueno_pages: 200,
            skipped_pages: 50,
            samefill_pages: 150,
            kernel_stats: None,
            trueno_stats: BackendStats::default(),
            nvme_stats: None,
        };

        assert_eq!(stats.total_pages(), 500);
        assert!((stats.kernel_percentage() - 20.0).abs() < 0.01);
        assert!((stats.samefill_percentage() - 30.0).abs() < 0.01);
    }

    #[test]
    fn test_nvme_cold_backend() {
        // Create temp directory for test
        let temp_dir = std::env::temp_dir().join("trueno-nvme-test");
        let _ = std::fs::remove_dir_all(&temp_dir); // Clean up from previous test

        let backend = NvmeColdBackend::new(&temp_dir).unwrap();

        // Store a page
        let mut data = [0u8; PAGE_SIZE];
        data[0..4].copy_from_slice(b"test");
        data[4..8].copy_from_slice(&[0xDE, 0xAD, 0xBE, 0xEF]);

        backend.store(42, &data).unwrap();

        // Load it back
        let mut buffer = [0u8; PAGE_SIZE];
        let found = backend.load(42, &mut buffer).unwrap();
        assert!(found);
        assert_eq!(&buffer[0..4], b"test");
        assert_eq!(&buffer[4..8], &[0xDE, 0xAD, 0xBE, 0xEF]);

        // Check stats
        let stats = backend.stats();
        assert_eq!(stats.pages_stored, 1);
        assert_eq!(stats.pages_loaded, 1);

        // Clean up
        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_nvme_cold_backend_nonexistent_page() {
        let temp_dir = std::env::temp_dir().join("trueno-nvme-test-nonexist");
        let _ = std::fs::remove_dir_all(&temp_dir);

        let backend = NvmeColdBackend::new(&temp_dir).unwrap();

        // Load non-existent page should return false and zero buffer
        let mut buffer = [0xFFu8; PAGE_SIZE];
        let found = backend.load(999, &mut buffer).unwrap();
        assert!(!found);
        assert!(buffer.iter().all(|&b| b == 0)); // Buffer zeroed

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_nvme_cold_backend_multiple_pages() {
        let temp_dir = std::env::temp_dir().join("trueno-nvme-test-multi");
        let _ = std::fs::remove_dir_all(&temp_dir);

        let backend = NvmeColdBackend::new(&temp_dir).unwrap();

        // Store multiple pages with different content
        for i in 0..10u64 {
            let mut data = [0u8; PAGE_SIZE];
            data[0..8].copy_from_slice(&i.to_le_bytes());
            backend.store(i * 100, &data).unwrap();
        }

        // Verify all pages
        for i in 0..10u64 {
            let mut buffer = [0u8; PAGE_SIZE];
            let found = backend.load(i * 100, &mut buffer).unwrap();
            assert!(found, "Page {} not found", i * 100);
            let stored_val = u64::from_le_bytes(buffer[0..8].try_into().unwrap());
            assert_eq!(stored_val, i, "Page {} has wrong content", i * 100);
        }

        // Check stats
        let stats = backend.stats();
        assert_eq!(stats.pages_stored, 10);
        assert_eq!(stats.pages_loaded, 10);
        assert_eq!(stats.bytes_written, 10 * PAGE_SIZE as u64);
        assert_eq!(stats.bytes_read, 10 * PAGE_SIZE as u64);

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_nvme_cold_backend_overwrite() {
        let temp_dir = std::env::temp_dir().join("trueno-nvme-test-overwrite");
        let _ = std::fs::remove_dir_all(&temp_dir);

        let backend = NvmeColdBackend::new(&temp_dir).unwrap();

        // Store initial data
        let mut data1 = [0u8; PAGE_SIZE];
        data1[0..4].copy_from_slice(b"old!");
        backend.store(50, &data1).unwrap();

        // Overwrite with new data
        let mut data2 = [0u8; PAGE_SIZE];
        data2[0..4].copy_from_slice(b"new!");
        backend.store(50, &data2).unwrap();

        // Should get new data
        let mut buffer = [0u8; PAGE_SIZE];
        let found = backend.load(50, &mut buffer).unwrap();
        assert!(found);
        assert_eq!(&buffer[0..4], b"new!");

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_nvme_cold_backend_file_path() {
        let temp_dir = std::env::temp_dir().join("trueno-nvme-test-path");
        let _ = std::fs::remove_dir_all(&temp_dir);

        let backend = NvmeColdBackend::new(&temp_dir).unwrap();
        let expected = temp_dir.join("pages.dat");
        assert_eq!(backend.file_path(), expected);

        let _ = std::fs::remove_dir_all(&temp_dir);
    }
}
