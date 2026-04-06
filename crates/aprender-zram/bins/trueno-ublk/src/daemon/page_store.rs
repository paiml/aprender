//! Basic compressed page storage.
//!
//! Provides the foundational `PageStore` and `PageStoreTrait` for ublk I/O.

use anyhow::Result;
use rustc_hash::FxHashMap;
use std::io::{Error as IoError, ErrorKind, Result as IoResult};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use trueno_zram_core::{
    samefill::{fill_page_word, page_same_filled},
    Algorithm, CompressedPage as CoreCompressedPage, CompressorBuilder, PageCompressor, PAGE_SIZE,
};

use super::entropy::calculate_entropy;

/// Sector size in bytes
pub(crate) const SECTOR_SIZE: u64 = 512;

/// Pages per sector (4096 / 512 = 8)
pub(crate) const SECTORS_PER_PAGE: u64 = PAGE_SIZE as u64 / SECTOR_SIZE;

// =============================================================================
// KERN-001/002: PageStoreTrait - Common interface for storage backends
// =============================================================================

/// Trait for page storage backends (BatchedPageStore, TieredPageStore).
///
/// This trait allows the ublk daemon to work with different storage backends
/// transparently, enabling kernel-cooperative tiered storage.
pub trait PageStoreTrait: Send + Sync {
    /// Read data from store at sector offset
    fn read(&self, start_sector: u64, buffer: &mut [u8]) -> IoResult<usize>;

    /// Write data to store at sector offset
    fn write(&self, start_sector: u64, data: &[u8]) -> IoResult<usize>;

    /// Discard sectors (trim)
    fn discard(&self, start_sector: u64, nr_sectors: u32) -> IoResult<usize>;

    /// Write zeros to sectors
    fn write_zeroes(&self, start_sector: u64, nr_sectors: u32) -> IoResult<usize>;

    /// Signal shutdown
    fn shutdown(&self);
}

/// Compressed page storage
pub struct PageStore {
    pages: FxHashMap<u64, StoredPage>,
    compressor: Arc<dyn PageCompressor>,
    entropy_threshold: f64,

    // Statistics
    bytes_stored: AtomicU64,
    bytes_compressed: AtomicU64,
    zero_pages: AtomicU64,
    gpu_pages: AtomicU64,
    simd_pages: AtomicU64,
    scalar_pages: AtomicU64,
}

/// Stored page representation - kernel zram style.
///
/// Following the kernel zram approach, same-fill pages are stored as just
/// the fill value (8 bytes) instead of compressed data. This eliminates
/// compression overhead and memory allocation for ~30-40% of typical pages.
///
/// Reference: Linux kernel drivers/block/zram/zram_drv.c
pub(crate) enum StoredPage {
    /// Same-fill page: stores only the u64 fill value (no compression, no allocation).
    /// This is the kernel's `ZRAM_SAME` optimization.
    SameFill(u64),
    /// Compressed page: actual compressed data for non-same-fill pages.
    Compressed(CoreCompressedPage),
}

impl PageStore {
    /// Create a new PageStore with the given device size and algorithm
    pub fn new(dev_size: u64, algorithm: Algorithm) -> Self {
        let compressor: Arc<dyn PageCompressor> = Arc::from(
            CompressorBuilder::new()
                .algorithm(algorithm)
                .build()
                .expect("Failed to create compressor"),
        );
        Self {
            pages: FxHashMap::with_capacity_and_hasher(
                (dev_size / PAGE_SIZE as u64) as usize,
                Default::default(),
            ),
            compressor,
            entropy_threshold: 6.0,
            bytes_stored: AtomicU64::new(0),
            bytes_compressed: AtomicU64::new(0),
            zero_pages: AtomicU64::new(0),
            gpu_pages: AtomicU64::new(0),
            simd_pages: AtomicU64::new(0),
            scalar_pages: AtomicU64::new(0),
        }
    }

    /// Create with custom compressor (for testing)
    pub fn with_compressor(compressor: Arc<dyn PageCompressor>, entropy_threshold: f64) -> Self {
        Self {
            pages: FxHashMap::default(),
            compressor,
            entropy_threshold,
            bytes_stored: AtomicU64::new(0),
            bytes_compressed: AtomicU64::new(0),
            zero_pages: AtomicU64::new(0),
            gpu_pages: AtomicU64::new(0),
            simd_pages: AtomicU64::new(0),
            scalar_pages: AtomicU64::new(0),
        }
    }

    /// Store a page at the given sector offset
    ///
    /// # Kernel ZRAM Optimization (P0)
    ///
    /// Following the kernel zram write path, we check same-fill BEFORE compression:
    /// 1. Check if page is same-fill (all words identical) - O(n) scan
    /// 2. If same-fill: store only the fill value (8 bytes, no allocation)
    /// 3. If not: compress and store normally
    ///
    /// This is critical for performance: ~30-40% of pages are same-fill (mostly zeros).
    /// Kernel ZRAM achieves 171 GB/s on zeros because it skips compression entirely.
    pub fn store(&mut self, sector: u64, data: &[u8]) -> Result<()> {
        debug_assert_eq!(data.len(), PAGE_SIZE);

        // Convert slice to fixed-size array
        let page: &[u8; PAGE_SIZE] = data.try_into().map_err(|_| {
            anyhow::anyhow!("Invalid page size: expected {}, got {}", PAGE_SIZE, data.len())
        })?;

        // P0 CRITICAL: Check same-fill BEFORE compression (kernel zram pattern)
        // This is what allows kernel ZRAM to hit 171 GB/s on zero pages.
        if let Some(fill_value) = page_same_filled(page) {
            // Same-fill page: store only the fill value (no compression!)
            self.pages.insert(sector, StoredPage::SameFill(fill_value));
            self.zero_pages.fetch_add(1, Ordering::Relaxed); // Track as "same-fill" (includes zeros)
            return Ok(());
        }

        // Not same-fill: compress normally
        let entropy = calculate_entropy(data);
        let compressed = self.compressor.compress(page)?;

        // Track which backend was used based on entropy
        if entropy > self.entropy_threshold {
            self.scalar_pages.fetch_add(1, Ordering::Relaxed);
        } else if entropy < 4.0 {
            self.gpu_pages.fetch_add(1, Ordering::Relaxed);
        } else {
            self.simd_pages.fetch_add(1, Ordering::Relaxed);
        }

        self.bytes_stored.fetch_add(data.len() as u64, Ordering::Relaxed);
        self.bytes_compressed.fetch_add(compressed.data.len() as u64, Ordering::Relaxed);

        self.pages.insert(sector, StoredPage::Compressed(compressed));
        Ok(())
    }

    /// Load a page from the given sector offset
    ///
    /// # Kernel ZRAM Three-Branch Read Path
    ///
    /// Following kernel zram read_from_zspool():
    /// 1. Same-fill: just fill with the stored value (fastest - memset_l)
    /// 2. Compressed: decompress normally
    /// 3. Not found: return zeros
    pub fn load(&self, sector: u64, buffer: &mut [u8]) -> Result<bool> {
        debug_assert_eq!(buffer.len(), PAGE_SIZE);

        match self.pages.get(&sector) {
            Some(StoredPage::SameFill(fill_value)) => {
                // Same-fill path: use word-fill (kernel memset_l equivalent)
                let buf_array: &mut [u8; PAGE_SIZE] =
                    buffer.try_into().map_err(|_| anyhow::anyhow!("Buffer size mismatch"))?;
                fill_page_word(buf_array, *fill_value);
                Ok(true)
            }
            Some(StoredPage::Compressed(compressed)) => {
                // Compressed path: decompress normally
                let decompressed = self.compressor.decompress(compressed)?;
                buffer.copy_from_slice(&decompressed);
                Ok(true)
            }
            None => {
                // Page not found - return zeros
                buffer.fill(0);
                Ok(false)
            }
        }
    }

    /// Remove a page
    pub fn remove(&mut self, sector: u64) -> bool {
        self.pages.remove(&sector).is_some()
    }

    // =========================================================================
    // ublk daemon interface methods
    // =========================================================================

    /// Read data from store at sector offset (ublk daemon interface)
    pub fn read(&self, start_sector: u64, buffer: &mut [u8]) -> IoResult<usize> {
        let mut offset = 0;
        let mut sector = start_sector;

        while offset < buffer.len() {
            let page_sector = (sector / SECTORS_PER_PAGE) * SECTORS_PER_PAGE;
            let sector_offset_in_page = (sector % SECTORS_PER_PAGE) as usize * SECTOR_SIZE as usize;
            let remaining_in_page = PAGE_SIZE - sector_offset_in_page;
            let to_read = (buffer.len() - offset).min(remaining_in_page);

            match self.pages.get(&page_sector) {
                Some(StoredPage::SameFill(fill_value)) => {
                    // Same-fill: reconstruct page from fill value, extract slice
                    let mut page_buf = [0u8; PAGE_SIZE];
                    fill_page_word(&mut page_buf, *fill_value);
                    buffer[offset..offset + to_read].copy_from_slice(
                        &page_buf[sector_offset_in_page..sector_offset_in_page + to_read],
                    );
                }
                Some(StoredPage::Compressed(compressed)) => {
                    let decompressed = self
                        .compressor
                        .decompress(compressed)
                        .map_err(|e| IoError::new(ErrorKind::InvalidData, e.to_string()))?;
                    buffer[offset..offset + to_read].copy_from_slice(
                        &decompressed[sector_offset_in_page..sector_offset_in_page + to_read],
                    );
                }
                None => {
                    buffer[offset..offset + to_read].fill(0);
                }
            }

            offset += to_read;
            sector += (to_read / SECTOR_SIZE as usize) as u64;
        }
        Ok(buffer.len())
    }

    /// Write data to store at sector offset (ublk daemon interface)
    pub fn write(&mut self, start_sector: u64, data: &[u8]) -> IoResult<usize> {
        let mut offset = 0;
        let mut sector = start_sector;

        while offset < data.len() {
            let page_sector = (sector / SECTORS_PER_PAGE) * SECTORS_PER_PAGE;
            let sector_offset_in_page = (sector % SECTORS_PER_PAGE) as usize * SECTOR_SIZE as usize;
            let remaining_in_page = PAGE_SIZE - sector_offset_in_page;
            let to_write = (data.len() - offset).min(remaining_in_page);

            if to_write < PAGE_SIZE {
                // Partial page write: read-modify-write
                let mut page_buf = [0u8; PAGE_SIZE];
                if let Some(stored_page) = self.pages.get(&page_sector) {
                    match stored_page {
                        StoredPage::SameFill(fill_value) => {
                            // Reconstruct same-fill page
                            fill_page_word(&mut page_buf, *fill_value);
                        }
                        StoredPage::Compressed(compressed) => {
                            let decompressed = self
                                .compressor
                                .decompress(compressed)
                                .map_err(|e| IoError::new(ErrorKind::InvalidData, e.to_string()))?;
                            page_buf.copy_from_slice(&decompressed);
                        }
                    }
                }
                page_buf[sector_offset_in_page..sector_offset_in_page + to_write]
                    .copy_from_slice(&data[offset..offset + to_write]);
                self.store_page(page_sector, &page_buf)?;
            } else {
                let page_data: &[u8; PAGE_SIZE] = (&data[offset..offset + PAGE_SIZE])
                    .try_into()
                    .expect("slice is exactly PAGE_SIZE bytes");
                self.store_page(page_sector, page_data)?;
            }

            offset += to_write;
            sector += (to_write / SECTOR_SIZE as usize) as u64;
        }
        Ok(data.len())
    }

    /// Discard sectors (ublk daemon interface)
    pub fn discard(&mut self, start_sector: u64, nr_sectors: u32) -> IoResult<usize> {
        let end_sector = start_sector + nr_sectors as u64;
        let mut sector = start_sector;
        while sector < end_sector {
            let page_sector = (sector / SECTORS_PER_PAGE) * SECTORS_PER_PAGE;
            self.pages.remove(&page_sector);
            sector = page_sector + SECTORS_PER_PAGE;
        }
        Ok(0)
    }

    /// Write zeros to sectors (ublk daemon interface)
    ///
    /// Optimized: stores zero fill value directly, NO compression.
    pub fn write_zeroes(&mut self, start_sector: u64, nr_sectors: u32) -> IoResult<usize> {
        let end_sector = start_sector + nr_sectors as u64;
        let mut sector = start_sector;
        while sector < end_sector {
            let page_sector = (sector / SECTORS_PER_PAGE) * SECTORS_PER_PAGE;
            // P0 CRITICAL: Store zero fill value directly - NO COMPRESSION!
            self.pages.insert(page_sector, StoredPage::SameFill(0));
            self.zero_pages.fetch_add(1, Ordering::Relaxed);
            sector = page_sector + SECTORS_PER_PAGE;
        }
        Ok(0)
    }

    /// Store a page (internal method for ublk write path)
    ///
    /// # Kernel ZRAM Optimization (P0)
    ///
    /// Same-fill check runs BEFORE compression - this is critical for performance.
    fn store_page(&mut self, sector: u64, data: &[u8; PAGE_SIZE]) -> IoResult<()> {
        // P0 CRITICAL: Check same-fill BEFORE compression (kernel zram pattern)
        if let Some(fill_value) = page_same_filled(data) {
            self.pages.insert(sector, StoredPage::SameFill(fill_value));
            self.zero_pages.fetch_add(1, Ordering::Relaxed);
            return Ok(());
        }

        // Not same-fill: compress normally
        let entropy = calculate_entropy(data);
        let compressed = self
            .compressor
            .compress(data)
            .map_err(|e| IoError::new(ErrorKind::InvalidData, e.to_string()))?;

        if entropy > self.entropy_threshold {
            self.scalar_pages.fetch_add(1, Ordering::Relaxed);
        } else if entropy < 4.0 {
            self.gpu_pages.fetch_add(1, Ordering::Relaxed);
        } else {
            self.simd_pages.fetch_add(1, Ordering::Relaxed);
        }

        self.bytes_stored.fetch_add(PAGE_SIZE as u64, Ordering::Relaxed);
        self.bytes_compressed.fetch_add(compressed.data.len() as u64, Ordering::Relaxed);
        self.pages.insert(sector, StoredPage::Compressed(compressed));
        Ok(())
    }

    /// Get statistics
    pub fn stats(&self) -> PageStoreStats {
        PageStoreStats {
            pages_stored: self.pages.len() as u64,
            bytes_stored: self.bytes_stored.load(Ordering::Relaxed),
            bytes_compressed: self.bytes_compressed.load(Ordering::Relaxed),
            zero_pages: self.zero_pages.load(Ordering::Relaxed),
            gpu_pages: self.gpu_pages.load(Ordering::Relaxed),
            simd_pages: self.simd_pages.load(Ordering::Relaxed),
            scalar_pages: self.scalar_pages.load(Ordering::Relaxed),
        }
    }
}

pub struct PageStoreStats {
    pub pages_stored: u64,
    pub bytes_stored: u64,
    pub bytes_compressed: u64,
    pub zero_pages: u64,
    pub gpu_pages: u64,
    pub simd_pages: u64,
    pub scalar_pages: u64,
}
