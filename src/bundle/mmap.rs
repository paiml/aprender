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

#[path = "page_table.rs"]
mod page_table;
pub(crate) use page_table::PageTable;
#[path = "mmap_tests.rs"]
mod mmap_tests;
#[path = "mmap_proptests.rs"]
mod mmap_proptests;
