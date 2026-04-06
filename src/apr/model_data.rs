//! Memory-mapped model data abstraction (PMAT-COMPLY)
//!
//! Extracted from mod.rs for file health compliance.
//!
//! # Heijunka - Level Loading
//!
//! References:
//! - Didona et al. (2022): mmap vs read() achieves 2.3x throughput for sequential access
//! - Chu (2011): LMDB design - let kernel manage pages, don't fight the VM subsystem
//! - Vahalia (1996): SIGBUS behavior on truncated mmap
//!
//! This abstraction allows models to be loaded via:
//! 1. Memory mapping (mmap) - zero-copy, kernel manages pages, no zram pressure
//! 2. Heap allocation (Vec<u8>) - required for compressed files after decompression

use std::fs::File;
use std::path::{Path, PathBuf};

use crate::error::{RealizarError, Result};

/// Model data storage abstraction for zero-copy access.
///
/// # Memory Management
///
/// When using `Mmap` variant:
/// - Data is not copied into userspace heap
/// - Kernel demand-pages from disk on access
/// - After GPU transfer, call `release_cpu_pages()` to advise kernel
/// - Pages backed by file (not zram) when evicted
///
/// When using `Heap` variant:
/// - Used for compressed files (must decompress to Vec<u8>)
/// - Standard heap allocation behavior
/// - May be compressed to zram when idle
#[derive(Debug)]
pub enum ModelData {
    /// Memory-mapped file (zero-copy, kernel-managed paging)
    #[cfg(not(target_arch = "wasm32"))]
    Mmap {
        /// Memory-mapped region
        mmap: memmap2::Mmap,
        /// Original file path (for diagnostics)
        path: PathBuf,
    },
    /// Heap-allocated data (for compressed files or WASM)
    Heap(Vec<u8>),
}

impl ModelData {
    /// Open a file with memory mapping.
    ///
    /// # Safety
    ///
    /// Uses `memmap2::Mmap` which requires:
    /// - File must not be truncated while mapped (SIGBUS on Unix)
    /// - File must not be modified while mapped (undefined behavior)
    ///
    /// # References
    ///
    /// - Vahalia (1996): SIGBUS from truncated mmap
    /// - memmap2 crate safety documentation
    #[cfg(not(target_arch = "wasm32"))]
    #[allow(unsafe_code)]
    pub fn open_mmap(path: impl AsRef<Path>) -> Result<Self> {
        let path_ref = path.as_ref();
        let file = File::open(path_ref).map_err(|e| RealizarError::IoError {
            message: format!("Failed to open file '{}': {e}", path_ref.display()),
        })?;

        // SAFETY: File is opened read-only. We document the single-writer
        // assumption. Callers should validate checksums before trusting data.
        // SIGBUS can occur if file is truncated externally - this is documented.
        let mmap = unsafe {
            memmap2::MmapOptions::new()
                .map(&file)
                .map_err(|e| RealizarError::IoError {
                    message: format!("Failed to mmap file '{}': {e}", path_ref.display()),
                })?
        };

        Ok(Self::Mmap {
            mmap,
            path: path_ref.to_path_buf(),
        })
    }

    /// Create from heap-allocated data (for compressed files).
    #[must_use]
    pub fn from_vec(data: Vec<u8>) -> Self {
        Self::Heap(data)
    }

    /// Get the data as a byte slice.
    #[must_use]
    pub fn as_slice(&self) -> &[u8] {
        match self {
            #[cfg(not(target_arch = "wasm32"))]
            Self::Mmap { mmap, .. } => mmap,
            Self::Heap(data) => data,
        }
    }

    /// Get data length.
    #[must_use]
    pub fn len(&self) -> usize {
        self.as_slice().len()
    }

    /// Check if data is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.as_slice().is_empty()
    }

    /// Release CPU pages after GPU transfer (Unix only).
    ///
    /// Calls `madvise(MADV_DONTNEED)` to tell the kernel these pages
    /// are no longer needed. The kernel will:
    /// - Drop pages immediately (not compress to zram)
    /// - Re-fault from disk if accessed again
    ///
    /// # When to Call
    ///
    /// After `cuMemcpy()` completes for all tensors.
    ///
    /// # Safety
    ///
    /// Uses `unchecked_advise` because `MADV_DONTNEED` is in the
    /// `UncheckedAdvice` enum. This is safe for read-only mmaps where
    /// data can be re-faulted from the backing file.
    ///
    /// # References
    ///
    /// - Didona et al. (2022): madvise for memory management
    #[cfg(all(unix, not(target_arch = "wasm32")))]
    #[allow(unsafe_code)]
    pub fn release_cpu_pages(&self) -> Result<()> {
        match self {
            Self::Mmap { mmap, path } => {
                // SAFETY: We opened the file read-only, so MADV_DONTNEED is safe -
                // the kernel will re-fault pages from the backing file if accessed.
                unsafe {
                    mmap.unchecked_advise(memmap2::UncheckedAdvice::DontNeed)
                        .map_err(|e| RealizarError::IoError {
                            message: format!(
                                "madvise(MADV_DONTNEED) failed for '{}': {e}",
                                path.display()
                            ),
                        })
                }
            },
            Self::Heap(_) => {
                // No-op for heap data - kernel manages via normal VM pressure
                Ok(())
            },
        }
    }

    /// No-op on non-Unix platforms (madvise not available).
    #[cfg(not(all(unix, not(target_arch = "wasm32"))))]
    pub fn release_cpu_pages(&self) -> Result<()> {
        Ok(())
    }

    /// Advise sequential access pattern (Unix only).
    ///
    /// Call before linear scan through model data.
    #[cfg(all(unix, not(target_arch = "wasm32")))]
    pub fn advise_sequential(&self) -> Result<()> {
        match self {
            Self::Mmap { mmap, path } => {
                mmap.advise(memmap2::Advice::Sequential)
                    .map_err(|e| RealizarError::IoError {
                        message: format!(
                            "madvise(MADV_SEQUENTIAL) failed for '{}': {e}",
                            path.display()
                        ),
                    })
            },
            Self::Heap(_) => Ok(()),
        }
    }

    /// No-op on non-Unix platforms (madvise not available).
    #[cfg(not(all(unix, not(target_arch = "wasm32"))))]
    pub fn advise_sequential(&self) -> Result<()> {
        Ok(())
    }

    /// Check if this is memory-mapped data.
    #[must_use]
    pub fn is_mmap(&self) -> bool {
        match self {
            #[cfg(not(target_arch = "wasm32"))]
            Self::Mmap { .. } => true,
            Self::Heap(_) => false,
        }
    }
}
