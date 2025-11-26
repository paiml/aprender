//! Memory-Mapped File Support
//!
//! Provides cross-platform memory mapping for efficient model loading.
//! Uses the OS's virtual memory system for on-demand paging.

use crate::error::{AprenderError, Result};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

/// Helper to create IO errors with a message.
fn io_error(msg: impl Into<String>) -> AprenderError {
    AprenderError::Other(msg.into())
}

// ============================================================================
// Mapped Region
// ============================================================================

/// A region of mapped memory representing model data.
///
/// In this implementation, we simulate mmap using standard file I/O
/// with caching. For production use, consider using the `memmap2` crate.
#[derive(Debug)]
pub struct MappedRegion {
    /// Cached data (simulated mmap).
    data: Vec<u8>,
    /// Offset in the source file.
    offset: u64,
    /// Length of the region.
    length: usize,
}

impl MappedRegion {
    /// Create a new mapped region.
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

// ============================================================================
// Memory Mapped File
// ============================================================================

/// Memory-mapped file for efficient model access.
///
/// This provides a file-backed memory region that allows
/// efficient random access to model weights.
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
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

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
        assert!(region.slice(3, 10).is_none()); // Out of bounds
    }

    #[test]
    fn test_memory_mapped_file_open() {
        let temp = NamedTempFile::new().expect("Failed to create temp file");
        temp.as_file()
            .write_all(b"test data")
            .expect("Failed to write");

        let mmap = MemoryMappedFile::open(temp.path()).expect("Failed to open");
        assert_eq!(mmap.size(), 9);
    }

    #[test]
    fn test_memory_mapped_file_read_at() {
        let temp = NamedTempFile::new().expect("Failed to create temp file");
        temp.as_file()
            .write_all(b"0123456789")
            .expect("Failed to write");

        let mut mmap = MemoryMappedFile::open(temp.path()).expect("Failed to open");

        let data = mmap.read_at(2, 5).expect("Failed to read");
        assert_eq!(data, b"23456");
    }

    #[test]
    fn test_memory_mapped_file_map_region() {
        let temp = NamedTempFile::new().expect("Failed to create temp file");
        temp.as_file()
            .write_all(b"ABCDEFGHIJ")
            .expect("Failed to write");

        let mut mmap = MemoryMappedFile::open(temp.path()).expect("Failed to open");

        let region = mmap.map_region(3, 4).expect("Failed to map");
        assert_eq!(region.as_slice(), b"DEFG");
        assert_eq!(mmap.cached_regions(), 1);
        assert_eq!(mmap.cached_bytes(), 4);
    }

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
}
