//! Kernel-cooperative tiered storage (KERN-001/002/003).
//!
//! Routes pages to the optimal storage tier based on Shannon entropy:
//! - H(X) < 6.0: Kernel ZRAM (171 GB/s, fast LZ4)
//! - 6.0 <= H(X) <= 7.5: Trueno SIMD (15 GiB/s, better ratio)
//! - H(X) > 7.5: NVMe cold tier / skip compression

use parking_lot::RwLock;
use rustc_hash::FxHashSet;
use std::io::Result as IoResult;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;
use trueno_zram_core::{
    samefill::page_same_filled,
    PAGE_SIZE,
};

use crate::backend::{
    BackendType, EntropyThresholds, KernelZramBackend, NvmeColdBackend, StorageBackend,
    TieredStorageManager,
};

use super::batched::BatchedPageStore;
use super::entropy::calculate_entropy;
use super::page_store::{PageStoreTrait, SECTORS_PER_PAGE, SECTOR_SIZE};

use super::batched::BatchedPageStoreStats;

/// Tiered page store configuration
#[derive(Debug, Clone)]
pub struct TieredConfig {
    /// Backend type
    pub backend: BackendType,
    /// Enable entropy-based routing
    pub entropy_routing: bool,
    /// Kernel ZRAM device path
    pub zram_device: Option<std::path::PathBuf>,
    /// NVMe cold tier directory path (KERN-003)
    pub cold_tier: Option<std::path::PathBuf>,
    /// Entropy threshold for kernel routing (H(X) < this goes to kernel)
    pub kernel_threshold: f64,
    /// Entropy threshold for skipping compression (H(X) > this skips)
    pub skip_threshold: f64,
}

impl Default for TieredConfig {
    fn default() -> Self {
        Self {
            backend: BackendType::Memory,
            entropy_routing: false,
            zram_device: None,
            cold_tier: None,
            kernel_threshold: 6.0,
            skip_threshold: 7.5,
        }
    }
}

/// Statistics for tiered page store
#[derive(Debug, Clone, Default)]
pub struct TieredPageStoreStats {
    /// Pages routed to kernel ZRAM (hot tier)
    pub kernel_pages: u64,
    /// Pages routed to trueno SIMD (warm tier)
    pub trueno_pages: u64,
    /// Pages with compression skipped (cold tier)
    pub skipped_pages: u64,
    /// Same-fill pages (no storage needed)
    pub samefill_pages: u64,
    /// Inner BatchedPageStore stats
    pub inner_stats: BatchedPageStoreStats,
}

/// Tiered page store wrapping BatchedPageStore with kernel ZRAM routing.
///
/// Implements the kernel-cooperative philosophy:
/// - H(X) < 6.0: Route to kernel ZRAM (171 GB/s, fast LZ4)
/// - 6.0 <= H(X) <= 7.5: Route to trueno SIMD (15 GiB/s, better ratio)
/// - H(X) > 7.5: NVMe cold tier / skip compression (KERN-003)
pub struct TieredPageStore {
    /// Inner BatchedPageStore for trueno SIMD tier
    inner: Arc<BatchedPageStore>,
    /// Kernel ZRAM backend (optional, for tiered mode)
    kernel_backend: Option<Arc<KernelZramBackend>>,
    /// NVMe cold tier backend (optional, for high-entropy pages - KERN-003)
    nvme_backend: Option<Arc<NvmeColdBackend>>,
    /// Tiered storage manager for entropy routing
    tiered_manager: Option<TieredStorageManager>,
    /// Configuration
    config: TieredConfig,
    /// Track which pages are in kernel ZRAM tier (sector -> true if in kernel)
    /// This allows fast tier lookups without probing inner store
    kernel_tier_pages: RwLock<FxHashSet<u64>>,
    /// Track which pages are in NVMe cold tier (KERN-003)
    nvme_tier_pages: RwLock<FxHashSet<u64>>,
    /// Statistics
    kernel_pages: AtomicU64,
    trueno_pages: AtomicU64,
    skipped_pages: AtomicU64,
    samefill_pages: AtomicU64,
    /// Shutdown flag
    shutdown: AtomicBool,
}

impl TieredPageStore {
    /// Create a new tiered page store.
    ///
    /// # Arguments
    /// * `inner` - The underlying BatchedPageStore for trueno SIMD compression
    /// * `config` - Tiered storage configuration
    pub fn new(inner: Arc<BatchedPageStore>, config: TieredConfig) -> anyhow::Result<Self> {
        let kernel_backend = match (&config.backend, &config.zram_device) {
            (BackendType::KernelZram | BackendType::Tiered, Some(path)) => {
                tracing::info!("KERN-001: Opening kernel ZRAM backend: {}", path.display());
                Some(Arc::new(KernelZramBackend::new(path)?))
            }
            _ => None,
        };

        // KERN-003: NVMe cold tier backend
        let nvme_backend = match &config.cold_tier {
            Some(path) => {
                tracing::info!("KERN-003: Opening NVMe cold tier backend: {}", path.display());
                Some(Arc::new(NvmeColdBackend::new(path)?))
            }
            _ => None,
        };

        let tiered_manager = if config.entropy_routing && kernel_backend.is_some() {
            tracing::info!(
                "KERN-002: Tiered storage enabled (kernel_threshold={}, skip_threshold={})",
                config.kernel_threshold,
                config.skip_threshold
            );
            Some(TieredStorageManager::with_thresholds(
                config.zram_device.as_deref(),
                config.cold_tier.as_deref(),
                EntropyThresholds {
                    kernel_threshold: config.kernel_threshold,
                    skip_threshold: config.skip_threshold,
                },
            )?)
        } else {
            None
        };

        Ok(Self {
            inner,
            kernel_backend,
            nvme_backend,
            tiered_manager,
            config,
            kernel_tier_pages: RwLock::new(FxHashSet::default()),
            nvme_tier_pages: RwLock::new(FxHashSet::default()),
            kernel_pages: AtomicU64::new(0),
            trueno_pages: AtomicU64::new(0),
            skipped_pages: AtomicU64::new(0),
            samefill_pages: AtomicU64::new(0),
            shutdown: AtomicBool::new(false),
        })
    }

    /// Store a page with entropy-based routing.
    pub fn store(&self, sector: u64, data: &[u8; PAGE_SIZE]) -> anyhow::Result<()> {
        // P0: Same-fill fast path (kernel ZRAM pattern)
        if let Some(fill_value) = page_same_filled(data) {
            self.inner.store_same_fill_page(sector, fill_value);
            self.samefill_pages.fetch_add(1, Ordering::Relaxed);
            return Ok(());
        }

        // Check if entropy routing is enabled
        if !self.config.entropy_routing {
            // No routing - use inner store directly
            self.inner.store(sector, data)?;
            self.trueno_pages.fetch_add(1, Ordering::Relaxed);
            return Ok(());
        }

        // Calculate entropy and route
        let entropy = calculate_entropy(data);

        if entropy < self.config.kernel_threshold {
            // Low entropy - route to kernel ZRAM
            if let Some(ref backend) = self.kernel_backend {
                // Convert sector to page index for kernel backend
                let page_idx = sector / SECTORS_PER_PAGE;
                backend.store(page_idx, data)?;
                // Track this page as being in kernel tier for fast lookups
                self.kernel_tier_pages.write().insert(sector);
                self.kernel_pages.fetch_add(1, Ordering::Relaxed);
            } else {
                // Fallback to inner store
                self.inner.store(sector, data)?;
                self.trueno_pages.fetch_add(1, Ordering::Relaxed);
            }
        } else if entropy > self.config.skip_threshold {
            // KERN-003: High entropy - route to NVMe cold tier
            if let Some(ref nvme) = self.nvme_backend {
                let page_idx = sector / SECTORS_PER_PAGE;
                nvme.store(page_idx, data)?;
                // Track this page as being in NVMe tier for fast lookups
                self.nvme_tier_pages.write().insert(sector);
                self.skipped_pages.fetch_add(1, Ordering::Relaxed);
            } else {
                // Fallback to inner store if no NVMe configured
                self.inner.store(sector, data)?;
                self.skipped_pages.fetch_add(1, Ordering::Relaxed);
            }
        } else {
            // Medium entropy - route to trueno SIMD
            self.inner.store(sector, data)?;
            self.trueno_pages.fetch_add(1, Ordering::Relaxed);
        }

        Ok(())
    }

    /// Load a page (checks all tiers)
    pub fn load(&self, sector: u64, buffer: &mut [u8; PAGE_SIZE]) -> anyhow::Result<bool> {
        // Try inner store first (most common)
        if self.inner.load(sector, buffer)? {
            return Ok(true);
        }

        // Try kernel backend if available
        if let Some(ref backend) = self.kernel_backend {
            let page_idx = sector / SECTORS_PER_PAGE;
            if backend.load(page_idx, buffer)? {
                return Ok(true);
            }
        }

        // KERN-003: Try NVMe cold tier
        if let Some(ref nvme) = self.nvme_backend {
            let page_idx = sector / SECTORS_PER_PAGE;
            if nvme.load(page_idx, buffer)? {
                return Ok(true);
            }
        }

        // Not found - return zeros
        buffer.fill(0);
        Ok(false)
    }

    /// Get statistics
    pub fn stats(&self) -> TieredPageStoreStats {
        TieredPageStoreStats {
            kernel_pages: self.kernel_pages.load(Ordering::Relaxed),
            trueno_pages: self.trueno_pages.load(Ordering::Relaxed),
            skipped_pages: self.skipped_pages.load(Ordering::Relaxed),
            samefill_pages: self.samefill_pages.load(Ordering::Relaxed),
            inner_stats: self.inner.stats(),
        }
    }

    /// Get inner BatchedPageStore (for flush thread)
    pub fn inner(&self) -> &Arc<BatchedPageStore> {
        &self.inner
    }

    /// Signal shutdown
    pub fn shutdown(&self) {
        self.shutdown.store(true, Ordering::SeqCst);
        self.inner.shutdown();
    }

    /// Check if shutdown was requested
    pub fn is_shutdown(&self) -> bool {
        self.shutdown.load(Ordering::SeqCst)
    }

    /// Check if should flush (delegates to inner)
    pub fn should_flush(&self) -> bool {
        self.inner.should_flush()
    }

    /// Flush batch (delegates to inner)
    pub fn flush_batch(&self) -> anyhow::Result<()> {
        self.inner.flush_batch()
    }

    // =========================================================================
    // ublk daemon interface methods (compatible with BatchedPageStore)
    // =========================================================================

    /// Read data from store at sector offset.
    ///
    /// This method implements optimized tiered read:
    /// 1. Check tier bitmap to determine which tier the page is in
    /// 2. Use bulk reads for contiguous kernel ZRAM ranges
    /// 3. Return zeros if not found
    pub fn read(&self, start_sector: u64, buffer: &mut [u8]) -> IoResult<usize> {
        // Fast path: try bulk kernel ZRAM read if all pages are in kernel tier
        if buffer.len() >= PAGE_SIZE && self.kernel_backend.is_some() {
            let start_page = (start_sector / SECTORS_PER_PAGE) * SECTORS_PER_PAGE;
            let num_pages = buffer.len() / PAGE_SIZE;

            // Check if ALL pages in this range are in kernel tier
            let kernel_pages = self.kernel_tier_pages.read();
            let mut all_kernel = num_pages > 0;
            for i in 0..num_pages {
                let page_sector = start_page + (i as u64 * SECTORS_PER_PAGE);
                if !kernel_pages.contains(&page_sector) {
                    all_kernel = false;
                    break;
                }
            }
            drop(kernel_pages);

            if all_kernel {
                // All pages are in kernel tier - use bulk read!
                let backend = self
                    .kernel_backend
                    .as_ref()
                    .expect("kernel_backend must be Some when all pages are in kernel tier");
                let page_idx = start_page / SECTORS_PER_PAGE;
                let aligned_len = num_pages * PAGE_SIZE;
                if backend.bulk_read(page_idx, &mut buffer[..aligned_len]).is_ok() {
                    // Handle any remaining partial page
                    if buffer.len() > aligned_len {
                        buffer[aligned_len..].fill(0);
                    }
                    return Ok(buffer.len());
                }
                // Fall through to per-page if bulk fails
            }
        }

        // Standard per-page path: try inner first (fast for same-fill), then kernel
        let mut offset = 0;
        let mut sector = start_sector;

        while offset < buffer.len() {
            let page_sector = (sector / SECTORS_PER_PAGE) * SECTORS_PER_PAGE;
            let sector_offset_in_page = (sector % SECTORS_PER_PAGE) as usize * SECTOR_SIZE as usize;
            let remaining_in_page = PAGE_SIZE - sector_offset_in_page;
            let to_read = (buffer.len() - offset).min(remaining_in_page);

            // For partial page reads or non-page-aligned, use inner
            if to_read < PAGE_SIZE || sector_offset_in_page != 0 {
                self.inner.read(sector, &mut buffer[offset..offset + to_read])?;
            } else {
                // Full page read - try inner first (same-fill is very fast)
                let buf_slice = &mut buffer[offset..offset + PAGE_SIZE];
                let buf_array: &mut [u8; PAGE_SIZE] =
                    buf_slice.try_into().expect("slice is exactly PAGE_SIZE bytes");

                // Try inner store first (same-fill + trueno pages) - NO lock needed
                if !self.inner.load(page_sector, buf_array).unwrap_or(false) {
                    // Not in inner - try kernel backend
                    if let Some(ref backend) = self.kernel_backend {
                        let page_idx = page_sector / SECTORS_PER_PAGE;
                        if !backend.load(page_idx, buf_array).unwrap_or(false) {
                            buf_array.fill(0);
                        }
                    } else {
                        buf_array.fill(0);
                    }
                }
            }

            offset += to_read;
            sector += (to_read / SECTOR_SIZE as usize) as u64;
        }

        Ok(buffer.len())
    }

    /// Write data to store at sector offset
    pub fn write(&self, start_sector: u64, data: &[u8]) -> IoResult<usize> {
        let mut offset = 0;
        let mut sector = start_sector;

        while offset < data.len() {
            let page_sector = (sector / SECTORS_PER_PAGE) * SECTORS_PER_PAGE;
            let sector_offset_in_page = (sector % SECTORS_PER_PAGE) as usize * SECTOR_SIZE as usize;
            let remaining_in_page = PAGE_SIZE - sector_offset_in_page;
            let to_write = (data.len() - offset).min(remaining_in_page);

            if to_write < PAGE_SIZE {
                // Partial page write - delegate to inner for read-modify-write
                self.inner.write(sector, &data[offset..offset + to_write])?;
            } else {
                // Full page write - use tiered routing
                let page_data: &[u8; PAGE_SIZE] = (&data[offset..offset + PAGE_SIZE])
                    .try_into()
                    .expect("slice is exactly PAGE_SIZE bytes");
                self.store(page_sector, page_data)
                    .map_err(|e| std::io::Error::other(e.to_string()))?;
            }

            offset += to_write;
            sector += (to_write / SECTOR_SIZE as usize) as u64;
        }
        Ok(data.len())
    }

    /// Discard sectors
    pub fn discard(&self, start_sector: u64, nr_sectors: u32) -> IoResult<usize> {
        self.inner.discard(start_sector, nr_sectors)
    }

    /// Write zeros to sectors
    pub fn write_zeroes(&self, start_sector: u64, nr_sectors: u32) -> IoResult<usize> {
        self.inner.write_zeroes(start_sector, nr_sectors)
    }
}

// Implement PageStoreTrait for TieredPageStore
impl PageStoreTrait for TieredPageStore {
    fn read(&self, start_sector: u64, buffer: &mut [u8]) -> IoResult<usize> {
        TieredPageStore::read(self, start_sector, buffer)
    }

    fn write(&self, start_sector: u64, data: &[u8]) -> IoResult<usize> {
        TieredPageStore::write(self, start_sector, data)
    }

    fn discard(&self, start_sector: u64, nr_sectors: u32) -> IoResult<usize> {
        TieredPageStore::discard(self, start_sector, nr_sectors)
    }

    fn write_zeroes(&self, start_sector: u64, nr_sectors: u32) -> IoResult<usize> {
        TieredPageStore::write_zeroes(self, start_sector, nr_sectors)
    }

    fn shutdown(&self) {
        TieredPageStore::shutdown(self)
    }
}

/// Spawn background flush thread for tiered page store
pub fn spawn_tiered_flush_thread(store: Arc<TieredPageStore>) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        while !store.is_shutdown() {
            std::thread::sleep(Duration::from_millis(1));

            if store.should_flush() {
                if let Err(e) = store.flush_batch() {
                    tracing::error!("Tiered flush failed: {}", e);
                }
            }
        }

        // Final flush on shutdown
        if let Err(e) = store.flush_batch() {
            tracing::error!("Tiered final flush failed: {}", e);
        }
    })
}
