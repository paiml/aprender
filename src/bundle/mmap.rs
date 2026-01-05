//! Memory-Mapped File Support
//!
//! Provides cross-platform memory mapping for efficient model loading.
//!
//! # Toyota Way Principles
//!
//! - **Muda (Waste Elimination)**: Zero-copy access eliminates redundant data movement
//! - **Jidoka (Build Quality In)**: Safe abstractions over unsafe mmap operations
//! - **Heijunka (Level Loading)**: OS demand paging prevents memory spikes
//!
//! # Platform Support
//!
//! - **Native (Linux/macOS/Windows)**: Real mmap via `memmap2` crate
//! - **WASM**: Heap fallback (no mmap available in browser/WASI environments)
//!
//! # Safety (SIGBUS Handling - Jidoka)
//!
//! Memory-mapped files can trigger SIGBUS on Unix if:
//! 1. The file is truncated while mapped
//! 2. The file is on a network filesystem that becomes unavailable
//! 3. Disk errors occur during page fault
//!
//! **Library Policy:** As a library, `aprender` does NOT install global signal
//! handlers. Applications should use crates like `signal-hook` if recovery from
//! storage failures is required during inference.
//!
//! Mitigations:
//! - Open files read-only (prevents external truncation in single-process use)
//! - Validate checksums before accessing data
//! - Document single-writer assumption in API
//!
//! See: Vahalia, U. (1996). "UNIX Internals", Chapter 14 - Memory Mapping
//!
//! # References
//!
//! - `McKusick` & Karels (1988): BSD memory allocator design
//! - Chu, H. (2011): LMDB memory-mapped design
//! - Didona et al. (2022): mmap vs explicit I/O performance analysis

use crate::error::{AprenderError, Result};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

/// Helper to create IO errors with a message.
fn io_error(msg: impl Into<String>) -> AprenderError {
    AprenderError::Other(msg.into())
}

// ============================================================================
// Real mmap implementation (when feature is enabled)
// ============================================================================

/// Memory-mapped file with zero-copy access.
///
/// On native platforms (Linux, macOS, Windows), this uses real OS-level
/// memory mapping via `memmap2`. On WASM, it falls back to standard
/// heap-based file reading.
///
/// # Safety
///
/// The mapped region is valid as long as:
/// 1. The file is not modified externally (single-writer assumption)
/// 2. The `MappedFile` instance is not dropped
/// 3. The underlying storage remains available
///
/// # SIGBUS Warning (Unix)
///
/// If the underlying file is truncated while mapped, accessing the
/// truncated region will generate a SIGBUS signal. This library does
/// not install signal handlers. See module documentation for details.
///
/// # Example
///
/// ```rust,ignore
/// use aprender::bundle::mmap::MappedFile;
///
/// let mapped = MappedFile::open("model.apr")?;
/// let header = mapped.slice(0, 32)?;  // Zero-copy access
/// println!("File size: {} bytes", mapped.len());
/// ```
#[derive(Debug)]
pub struct MappedFile {
    #[cfg(not(target_arch = "wasm32"))]
    mmap: memmap2::Mmap,
    #[cfg(target_arch = "wasm32")]
    data: Vec<u8>,
    path: String,
}

#[cfg(not(target_arch = "wasm32"))]
#[allow(unsafe_code)]
impl MappedFile {
    /// Open a file for memory-mapped read access.
    ///
    /// # Safety Justification (Jidoka)
    ///
    /// The `unsafe` block is required because `memmap2::Mmap::map()` has
    /// undefined behavior if:
    /// - The file is modified by another process while mapped
    /// - The file is truncated while mapped (triggers SIGBUS on Unix)
    ///
    /// We mitigate these risks by:
    /// 1. Opening the file read-only (no write handle to leak)
    /// 2. Documenting the single-writer assumption
    /// 3. Validating checksums before use (in .apr format)
    ///
    /// References:
    /// - Vahalia (1996): SIGBUS from truncated mmap
    /// - memmap2 crate safety documentation
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path_str = path.as_ref().to_string_lossy().to_string();
        let file = File::open(path.as_ref())
            .map_err(|e| io_error(format!("Failed to open file '{path_str}': {e}")))?;

        // SAFETY: File is opened read-only. We document the single-writer
        // assumption. Callers should validate checksums before trusting data.
        // SIGBUS can occur if file is truncated externally - this is documented.
        let mmap = unsafe {
            memmap2::MmapOptions::new()
                .map(&file)
                .map_err(|e| io_error(format!("Failed to mmap file '{path_str}': {e}")))?
        };

        Ok(Self {
            mmap,
            path: path_str,
        })
    }

    /// Get the entire file as a byte slice (zero-copy).
    #[inline]
    #[must_use]
    pub fn as_slice(&self) -> &[u8] {
        &self.mmap
    }

    /// Get a subslice of the file (zero-copy).
    ///
    /// Returns `None` if the range is out of bounds.
    #[inline]
    #[must_use]
    pub fn slice(&self, start: usize, end: usize) -> Option<&[u8]> {
        if start <= end && end <= self.mmap.len() {
            Some(&self.mmap[start..end])
        } else {
            None
        }
    }

    /// File size in bytes.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.mmap.len()
    }

    /// Check if file is empty.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.mmap.is_empty()
    }

    /// Get the file path.
    #[must_use]
    pub fn path(&self) -> &str {
        &self.path
    }

    /// Advise the kernel about sequential access pattern (Unix only).
    ///
    /// This can improve performance for linear scans through the file.
    #[cfg(unix)]
    pub fn advise_sequential(&self) -> Result<()> {
        self.mmap
            .advise(memmap2::Advice::Sequential)
            .map_err(|e| io_error(format!("madvise failed: {e}")))
    }

    /// Advise the kernel about random access pattern (Unix only).
    ///
    /// This can improve performance for random lookups.
    #[cfg(unix)]
    pub fn advise_random(&self) -> Result<()> {
        self.mmap
            .advise(memmap2::Advice::Random)
            .map_err(|e| io_error(format!("madvise failed: {e}")))
    }
}

#[cfg(target_arch = "wasm32")]
impl MappedFile {
    /// Open a file using heap-based fallback (WASM).
    ///
    /// This fallback reads the entire file into memory since WASM
    /// environments typically lack mmap support.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path_str = path.as_ref().to_string_lossy().to_string();
        let mut file = File::open(path.as_ref())
            .map_err(|e| io_error(format!("Failed to open file '{path_str}': {e}")))?;

        let mut data = Vec::new();
        file.read_to_end(&mut data)
            .map_err(|e| io_error(format!("Failed to read file '{path_str}': {e}")))?;

        Ok(Self {
            data,
            path: path_str,
        })
    }

    /// Get the entire file as a byte slice.
    #[inline]
    #[must_use]
    pub fn as_slice(&self) -> &[u8] {
        &self.data
    }

    /// Get a subslice of the file.
    #[inline]
    #[must_use]
    pub fn slice(&self, start: usize, end: usize) -> Option<&[u8]> {
        if start <= end && end <= self.data.len() {
            Some(&self.data[start..end])
        } else {
            None
        }
    }

    /// File size in bytes.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if file is empty.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get the file path.
    #[must_use]
    pub fn path(&self) -> &str {
        &self.path
    }
}

// ============================================================================
// Legacy API (backward compatibility)
// ============================================================================

/// A region of mapped memory representing model data.
///
/// This is the legacy API. For new code, prefer `MappedFile`.
#[derive(Debug)]
pub struct MappedRegion {
    /// Cached data.
    data: Vec<u8>,
    /// Offset in the source file.
    offset: u64,
    /// Length of the region.
    length: usize,
}

impl MappedRegion {
    /// Create a new mapped region.
    #[must_use]
    pub fn new(data: Vec<u8>, offset: u64) -> Self {
        let length = data.len();
        Self {
            data,
            offset,
            length,
        }
    }

    /// Get the data as a slice.
    #[must_use]
    pub fn as_slice(&self) -> &[u8] {
        &self.data
    }

    /// Get the offset in the source file.
    #[must_use]
    pub fn offset(&self) -> u64 {
        self.offset
    }

    /// Get the length of the region.
    #[must_use]
    pub fn len(&self) -> usize {
        self.length
    }

    /// Check if the region is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    /// Get a subslice of the region.
    #[must_use]
    pub fn slice(&self, start: usize, end: usize) -> Option<&[u8]> {
        if end <= self.length && start <= end {
            Some(&self.data[start..end])
        } else {
            None
        }
    }
}

/// Memory-mapped file for efficient model access (legacy API).
///
/// This is the legacy API using simulated mmap. For new code, prefer `MappedFile`.
#[derive(Debug)]
pub struct MemoryMappedFile {
    /// File handle.
    file: File,
    /// File size.
    size: u64,
    /// Path to the file.
    path: String,
    /// Cached regions.
    regions: Vec<MappedRegion>,
}

impl MemoryMappedFile {
    /// Open a file for memory mapping.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path_str = path.as_ref().to_string_lossy().to_string();
        let file = File::open(path.as_ref())
            .map_err(|e| io_error(format!("Failed to open file '{path_str}': {e}")))?;

        let metadata = file
            .metadata()
            .map_err(|e| io_error(format!("Failed to get file metadata: {e}")))?;

        Ok(Self {
            file,
            size: metadata.len(),
            path: path_str,
            regions: Vec::new(),
        })
    }

    /// Get the file size.
    #[must_use]
    pub fn size(&self) -> u64 {
        self.size
    }

    /// Get the file path.
    #[must_use]
    pub fn path(&self) -> &str {
        &self.path
    }

    /// Map a region of the file.
    pub fn map_region(&mut self, offset: u64, length: usize) -> Result<&MappedRegion> {
        // Check if region already mapped
        for (idx, region) in self.regions.iter().enumerate() {
            if region.offset == offset && region.len() >= length {
                return Ok(&self.regions[idx]);
            }
        }

        // Read the region
        self.file
            .seek(SeekFrom::Start(offset))
            .map_err(|e| io_error(format!("Failed to seek: {e}")))?;

        let mut data = vec![0u8; length];
        self.file
            .read_exact(&mut data)
            .map_err(|e| io_error(format!("Failed to read region: {e}")))?;

        let region = MappedRegion::new(data, offset);
        self.regions.push(region);

        Ok(self.regions.last().expect("Just pushed"))
    }

    /// Read bytes at a specific offset.
    pub fn read_at(&mut self, offset: u64, length: usize) -> Result<Vec<u8>> {
        if offset + length as u64 > self.size {
            return Err(io_error("Read past end of file"));
        }

        self.file
            .seek(SeekFrom::Start(offset))
            .map_err(|e| io_error(format!("Failed to seek: {e}")))?;

        let mut data = vec![0u8; length];
        self.file
            .read_exact(&mut data)
            .map_err(|e| io_error(format!("Failed to read: {e}")))?;

        Ok(data)
    }

    /// Clear cached regions.
    pub fn clear_cache(&mut self) {
        self.regions.clear();
    }

    /// Get number of cached regions.
    #[must_use]
    pub fn cached_regions(&self) -> usize {
        self.regions.len()
    }

    /// Get total cached bytes.
    #[must_use]
    pub fn cached_bytes(&self) -> usize {
        self.regions.iter().map(MappedRegion::len).sum()
    }
}

// ============================================================================
// Page Table (for LRU tracking)
// ============================================================================

/// Entry in the page table.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub(crate) struct PageEntry {
    /// Offset in the file.
    pub(crate) offset: u64,
    /// Size of the page.
    pub(crate) size: usize,
    /// Access count.
    pub(crate) access_count: u64,
    /// Last access timestamp (monotonic counter).
    pub(crate) last_access: u64,
}

impl PageEntry {
    /// Create a new page entry.
    #[must_use]
    pub(crate) fn new(offset: u64, size: usize) -> Self {
        Self {
            offset,
            size,
            access_count: 0,
            last_access: 0,
        }
    }

    /// Record an access to this page.
    pub(crate) fn touch(&mut self, timestamp: u64) {
        self.access_count += 1;
        self.last_access = timestamp;
    }
}

/// Page table for tracking memory pages.
#[derive(Debug, Default)]
pub(crate) struct PageTable {
    /// Pages indexed by offset.
    pages: std::collections::HashMap<u64, PageEntry>,
    /// Monotonic timestamp counter.
    timestamp: u64,
}

impl PageTable {
    /// Create a new empty page table.
    #[must_use]
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Add a page to the table.
    pub(crate) fn add_page(&mut self, offset: u64, size: usize) {
        let mut entry = PageEntry::new(offset, size);
        entry.touch(self.next_timestamp());
        self.pages.insert(offset, entry);
    }

    /// Touch a page (record access).
    pub(crate) fn touch(&mut self, offset: u64) {
        let timestamp = self.next_timestamp();
        if let Some(entry) = self.pages.get_mut(&offset) {
            entry.touch(timestamp);
        }
    }

    /// Get the least recently used page offset.
    #[must_use]
    #[allow(dead_code)]
    pub(crate) fn lru_page(&self) -> Option<u64> {
        self.pages
            .iter()
            .min_by_key(|(_, entry)| entry.last_access)
            .map(|(&offset, _)| offset)
    }

    /// Get the least frequently used page offset.
    #[must_use]
    pub(crate) fn lfu_page(&self) -> Option<u64> {
        self.pages
            .iter()
            .min_by_key(|(_, entry)| entry.access_count)
            .map(|(&offset, _)| offset)
    }

    /// Remove a page from the table.
    pub(crate) fn remove(&mut self, offset: u64) -> Option<PageEntry> {
        self.pages.remove(&offset)
    }

    /// Get page entry by offset.
    #[must_use]
    #[allow(dead_code)]
    pub(crate) fn get(&self, offset: u64) -> Option<&PageEntry> {
        self.pages.get(&offset)
    }

    /// Get number of pages.
    #[must_use]
    #[allow(dead_code)]
    pub(crate) fn len(&self) -> usize {
        self.pages.len()
    }

    /// Check if table is empty.
    #[must_use]
    #[allow(dead_code)]
    pub(crate) fn is_empty(&self) -> bool {
        self.pages.is_empty()
    }

    /// Get total size of all pages.
    #[must_use]
    #[allow(dead_code)]
    pub(crate) fn total_size(&self) -> usize {
        self.pages.values().map(|p| p.size).sum()
    }

    /// Get next timestamp.
    fn next_timestamp(&mut self) -> u64 {
        self.timestamp += 1;
        self.timestamp
    }
}

// ============================================================================
// Tests (EXTREME TDD)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    // ========================================================================
    // MappedFile Tests (New API)
    // ========================================================================

    #[test]
    fn test_mapped_file_open_and_read() {
        let mut file = NamedTempFile::new().expect("create temp");
        file.write_all(b"Hello, mmap!").expect("write");

        let mapped = MappedFile::open(file.path()).expect("open");
        assert_eq!(mapped.as_slice(), b"Hello, mmap!");
        assert_eq!(mapped.len(), 12);
        assert!(!mapped.is_empty());
    }

    #[test]
    fn test_mapped_file_slice() {
        let mut file = NamedTempFile::new().expect("create temp");
        file.write_all(b"0123456789").expect("write");

        let mapped = MappedFile::open(file.path()).expect("open");
        assert_eq!(mapped.slice(2, 7), Some(&b"23456"[..]));
        assert_eq!(mapped.slice(0, 3), Some(&b"012"[..]));
        assert_eq!(mapped.slice(8, 10), Some(&b"89"[..]));
    }

    #[test]
    fn test_mapped_file_slice_out_of_bounds() {
        let mut file = NamedTempFile::new().expect("create temp");
        file.write_all(b"short").expect("write");

        let mapped = MappedFile::open(file.path()).expect("open");
        assert!(mapped.slice(0, 100).is_none());
        assert!(mapped.slice(10, 20).is_none());
    }

    #[test]
    fn test_mapped_file_slice_invalid_range() {
        let mut file = NamedTempFile::new().expect("create temp");
        file.write_all(b"test data").expect("write");

        let mapped = MappedFile::open(file.path()).expect("open");
        // end < start should return None
        assert!(mapped.slice(5, 3).is_none());
    }

    #[test]
    fn test_mapped_file_empty() {
        let file = NamedTempFile::new().expect("create temp");
        let mapped = MappedFile::open(file.path()).expect("open");
        assert!(mapped.is_empty());
        assert_eq!(mapped.len(), 0);
        let empty: &[u8] = &[];
        assert_eq!(mapped.as_slice(), empty);
    }

    #[test]
    fn test_mapped_file_path() {
        let file = NamedTempFile::new().expect("create temp");
        let expected_path = file.path().to_string_lossy().to_string();
        let mapped = MappedFile::open(file.path()).expect("open");
        assert_eq!(mapped.path(), expected_path);
    }

    #[test]
    fn test_mapped_file_large_file() {
        let mut file = NamedTempFile::new().expect("create temp");
        let data: Vec<u8> = (0..100_000).map(|i| (i % 256) as u8).collect();
        file.write_all(&data).expect("write");

        let mapped = MappedFile::open(file.path()).expect("open");
        assert_eq!(mapped.len(), 100_000);
        assert_eq!(mapped.slice(50_000, 50_010), Some(&data[50_000..50_010]));
    }

    #[test]
    fn test_mapped_file_nonexistent() {
        let result = MappedFile::open("/nonexistent/path/file.bin");
        assert!(result.is_err());
    }

    // ========================================================================
    // MappedRegion Tests (Legacy API)
    // ========================================================================

    #[test]
    fn test_mapped_region_new() {
        let data = vec![1, 2, 3, 4, 5];
        let region = MappedRegion::new(data.clone(), 100);

        assert_eq!(region.as_slice(), &data);
        assert_eq!(region.offset(), 100);
        assert_eq!(region.len(), 5);
        assert!(!region.is_empty());
    }

    #[test]
    fn test_mapped_region_slice() {
        let data = vec![10, 20, 30, 40, 50];
        let region = MappedRegion::new(data, 0);

        assert_eq!(region.slice(1, 4), Some(&[20, 30, 40][..]));
        assert_eq!(region.slice(0, 2), Some(&[10, 20][..]));
        assert!(region.slice(3, 10).is_none());
    }

    // ========================================================================
    // MemoryMappedFile Tests (Legacy API)
    // ========================================================================

    #[test]
    fn test_memory_mapped_file_open() {
        let mut temp = NamedTempFile::new().expect("create temp");
        temp.write_all(b"test data").expect("write");

        let mmap = MemoryMappedFile::open(temp.path()).expect("open");
        assert_eq!(mmap.size(), 9);
    }

    #[test]
    fn test_memory_mapped_file_read_at() {
        let mut temp = NamedTempFile::new().expect("create temp");
        temp.write_all(b"0123456789").expect("write");

        let mut mmap = MemoryMappedFile::open(temp.path()).expect("open");
        let data = mmap.read_at(2, 5).expect("read");
        assert_eq!(data, b"23456");
    }

    #[test]
    fn test_memory_mapped_file_map_region() {
        let mut temp = NamedTempFile::new().expect("create temp");
        temp.write_all(b"ABCDEFGHIJ").expect("write");

        let mut mmap = MemoryMappedFile::open(temp.path()).expect("open");
        let region = mmap.map_region(3, 4).expect("map");
        assert_eq!(region.as_slice(), b"DEFG");
        assert_eq!(mmap.cached_regions(), 1);
        assert_eq!(mmap.cached_bytes(), 4);
    }

    // ========================================================================
    // PageTable Tests
    // ========================================================================

    #[test]
    fn test_page_entry() {
        let mut entry = PageEntry::new(1000, 256);
        assert_eq!(entry.offset, 1000);
        assert_eq!(entry.size, 256);
        assert_eq!(entry.access_count, 0);

        entry.touch(5);
        assert_eq!(entry.access_count, 1);
        assert_eq!(entry.last_access, 5);

        entry.touch(10);
        assert_eq!(entry.access_count, 2);
        assert_eq!(entry.last_access, 10);
    }

    #[test]
    fn test_page_table_lru() {
        let mut table = PageTable::new();

        table.add_page(100, 10);
        table.add_page(200, 20);
        table.add_page(300, 30);

        // 100 is oldest, should be LRU
        assert_eq!(table.lru_page(), Some(100));

        // Touch 100, now 200 should be LRU
        table.touch(100);
        assert_eq!(table.lru_page(), Some(200));
    }

    #[test]
    fn test_page_table_lfu() {
        let mut table = PageTable::new();

        table.add_page(100, 10);
        table.add_page(200, 20);
        table.add_page(300, 30);

        // Touch pages different numbers of times
        table.touch(100);
        table.touch(100);
        table.touch(200);

        // 300 has fewest accesses (1)
        assert_eq!(table.lfu_page(), Some(300));
    }

    #[test]
    fn test_page_table_remove() {
        let mut table = PageTable::new();
        table.add_page(100, 10);
        table.add_page(200, 20);

        assert_eq!(table.len(), 2);
        assert_eq!(table.total_size(), 30);

        let removed = table.remove(100);
        assert!(removed.is_some());
        assert_eq!(table.len(), 1);
        assert_eq!(table.total_size(), 20);
    }

    // ========================================================================
    // Additional Coverage Tests
    // ========================================================================

    #[test]
    fn test_mapped_file_debug() {
        let mut temp = NamedTempFile::new().expect("create temp");
        temp.write_all(b"test").expect("write");

        let mapped = MappedFile::open(temp.path()).expect("open");
        let debug_str = format!("{:?}", mapped);
        assert!(debug_str.contains("MappedFile"));
    }

    #[test]
    fn test_mapped_region_empty() {
        let region = MappedRegion::new(vec![], 50);
        assert!(region.is_empty());
        assert_eq!(region.len(), 0);
        assert_eq!(region.offset(), 50);
        let empty: &[u8] = &[];
        assert_eq!(region.as_slice(), empty);
    }

    #[test]
    fn test_mapped_region_debug() {
        let region = MappedRegion::new(vec![1, 2, 3], 0);
        let debug_str = format!("{:?}", region);
        assert!(debug_str.contains("MappedRegion"));
    }

    #[test]
    fn test_mapped_region_slice_empty() {
        let region = MappedRegion::new(vec![1, 2, 3, 4, 5], 0);
        // Empty slice at valid position
        assert_eq!(region.slice(2, 2), Some(&[][..]));
    }

    #[test]
    fn test_memory_mapped_file_path() {
        let mut temp = NamedTempFile::new().expect("create temp");
        temp.write_all(b"test").expect("write");

        let mmap = MemoryMappedFile::open(temp.path()).expect("open");
        assert!(mmap.path().contains(temp.path().to_str().unwrap()));
    }

    #[test]
    fn test_memory_mapped_file_clear_cache() {
        let mut temp = NamedTempFile::new().expect("create temp");
        temp.write_all(b"test data for caching").expect("write");

        let mut mmap = MemoryMappedFile::open(temp.path()).expect("open");

        // Map some regions
        let _ = mmap.map_region(0, 5).expect("map");
        let _ = mmap.map_region(5, 5).expect("map");
        assert_eq!(mmap.cached_regions(), 2);

        // Clear cache
        mmap.clear_cache();
        assert_eq!(mmap.cached_regions(), 0);
        assert_eq!(mmap.cached_bytes(), 0);
    }

    #[test]
    fn test_memory_mapped_file_debug() {
        let mut temp = NamedTempFile::new().expect("create temp");
        temp.write_all(b"test").expect("write");

        let mmap = MemoryMappedFile::open(temp.path()).expect("open");
        let debug_str = format!("{:?}", mmap);
        assert!(debug_str.contains("MemoryMappedFile"));
    }

    #[test]
    fn test_memory_mapped_file_read_past_end() {
        let mut temp = NamedTempFile::new().expect("create temp");
        temp.write_all(b"short").expect("write");

        let mut mmap = MemoryMappedFile::open(temp.path()).expect("open");
        let result = mmap.read_at(3, 10); // 3 + 10 = 13 > 5
        assert!(result.is_err());
    }

    #[test]
    fn test_memory_mapped_file_map_region_cached() {
        let mut temp = NamedTempFile::new().expect("create temp");
        temp.write_all(b"ABCDEFGHIJ").expect("write");

        let mut mmap = MemoryMappedFile::open(temp.path()).expect("open");

        // First map
        let _ = mmap.map_region(3, 4).expect("map");
        assert_eq!(mmap.cached_regions(), 1);

        // Same region should reuse cache
        let region = mmap.map_region(3, 4).expect("map again");
        assert_eq!(region.as_slice(), b"DEFG");
        // Still 1 cached region because we reused
        assert_eq!(mmap.cached_regions(), 1);
    }

    #[test]
    fn test_memory_mapped_file_nonexistent() {
        let result = MemoryMappedFile::open("/nonexistent/path/file.bin");
        assert!(result.is_err());
    }

    #[test]
    fn test_page_table_get() {
        let mut table = PageTable::new();
        table.add_page(100, 10);

        let entry = table.get(100);
        assert!(entry.is_some());
        assert_eq!(entry.unwrap().size, 10);

        assert!(table.get(999).is_none());
    }

    #[test]
    fn test_page_table_is_empty() {
        let table = PageTable::new();
        assert!(table.is_empty());

        let mut table2 = PageTable::new();
        table2.add_page(100, 10);
        assert!(!table2.is_empty());
    }

    #[test]
    fn test_page_table_remove_nonexistent() {
        let mut table = PageTable::new();
        table.add_page(100, 10);

        let removed = table.remove(999);
        assert!(removed.is_none());
        assert_eq!(table.len(), 1);
    }

    #[test]
    fn test_page_entry_clone() {
        let entry = PageEntry::new(100, 50);
        let cloned = entry.clone();
        assert_eq!(cloned.offset, entry.offset);
        assert_eq!(cloned.size, entry.size);
    }

    #[test]
    fn test_page_table_empty_lru_lfu() {
        let table = PageTable::new();
        assert!(table.lru_page().is_none());
        assert!(table.lfu_page().is_none());
    }

    #[test]
    fn test_page_table_touch_nonexistent() {
        let mut table = PageTable::new();
        table.add_page(100, 10);

        // Touching nonexistent page should be a no-op
        table.touch(999);
        assert_eq!(table.len(), 1);
    }

    #[cfg(unix)]
    #[test]
    fn test_mapped_file_advise_sequential() {
        let mut temp = NamedTempFile::new().expect("create temp");
        temp.write_all(b"test data for advise").expect("write");

        let mapped = MappedFile::open(temp.path()).expect("open");
        let result = mapped.advise_sequential();
        assert!(result.is_ok());
    }

    #[cfg(unix)]
    #[test]
    fn test_mapped_file_advise_random() {
        let mut temp = NamedTempFile::new().expect("create temp");
        temp.write_all(b"test data for advise").expect("write");

        let mapped = MappedFile::open(temp.path()).expect("open");
        let result = mapped.advise_random();
        assert!(result.is_ok());
    }
}

// ============================================================================
// Property Tests (EXTREME TDD)
// ============================================================================

#[cfg(test)]
mod prop_tests {
    use super::*;
    use proptest::prelude::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    proptest! {
        #[test]
        fn prop_mapped_file_slice_within_bounds(
            data in prop::collection::vec(any::<u8>(), 1..1000),
            start in 0usize..1000,
            len in 0usize..500,
        ) {
            let mut file = NamedTempFile::new().expect("create temp");
            file.write_all(&data).expect("write");

            let mapped = MappedFile::open(file.path()).expect("open");
            let end = start.saturating_add(len);

            if start <= end && end <= data.len() {
                prop_assert_eq!(mapped.slice(start, end), Some(&data[start..end]));
            } else {
                prop_assert!(mapped.slice(start, end).is_none());
            }
        }

        #[test]
        fn prop_mapped_file_full_slice_equals_as_slice(
            data in prop::collection::vec(any::<u8>(), 0..1000),
        ) {
            let mut file = NamedTempFile::new().expect("create temp");
            file.write_all(&data).expect("write");

            let mapped = MappedFile::open(file.path()).expect("open");

            prop_assert_eq!(mapped.slice(0, mapped.len()), Some(mapped.as_slice()));
        }

        #[test]
        fn prop_mapped_file_len_matches_data(
            data in prop::collection::vec(any::<u8>(), 0..10000),
        ) {
            let mut file = NamedTempFile::new().expect("create temp");
            file.write_all(&data).expect("write");

            let mapped = MappedFile::open(file.path()).expect("open");

            prop_assert_eq!(mapped.len(), data.len());
            prop_assert_eq!(mapped.is_empty(), data.is_empty());
        }
    }
}
