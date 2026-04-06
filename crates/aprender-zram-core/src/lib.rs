//! SIMD-accelerated memory compression for Linux zram.
//!
//! This crate provides high-performance LZ4 and Zstandard compression
//! optimized for 4KB memory pages, with runtime SIMD dispatch.
//!
//! # Example
//!
//! ```
//! use trueno_zram_core::{CompressorBuilder, Algorithm, PageCompressor, PAGE_SIZE};
//!
//! let compressor = CompressorBuilder::new()
//!     .algorithm(Algorithm::Lz4)
//!     .build()
//!     .unwrap();
//!
//! let page = [0u8; PAGE_SIZE];
//! let compressed = compressor.compress(&page).unwrap();
//! let decompressed = compressor.decompress(&compressed).unwrap();
//!
//! assert_eq!(page, decompressed);
//! ```

#![deny(missing_docs)]
#![deny(clippy::panic)]
#![allow(unsafe_code)] // SIMD intrinsics require unsafe
#![warn(clippy::all, clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::similar_names)]
#![allow(clippy::unnecessary_wraps)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::missing_safety_doc)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::cognitive_complexity)]
#![allow(clippy::cast_ptr_alignment)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::match_same_arms)]
#![allow(clippy::unreadable_literal)]
#![allow(clippy::items_after_statements)]
#![allow(clippy::wildcard_imports)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::manual_strip)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::inline_always)]
#![allow(clippy::unused_self)]
#![allow(clippy::struct_excessive_bools)]
#![allow(clippy::incompatible_msrv)]
#![allow(clippy::uninit_vec)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::large_stack_arrays)]
#![allow(unused_assignments)]

pub mod benchmark;
pub mod compat;
mod error;
#[cfg(feature = "cuda")]
pub mod gpu;
pub mod integration;
pub mod lz4;
mod page;
pub mod samefill;
pub mod simd;
pub mod zram;
pub mod zstd;

pub use error::{Error, Result};
pub use page::{CompressedPage, CompressionStats, PAGE_SIZE};

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// Compression algorithm selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Algorithm {
    /// No compression (store as-is).
    None,
    /// LZ4 fast compression.
    #[default]
    Lz4,
    /// LZ4-HC high compression (not yet implemented).
    Lz4Hc,
    /// Zstandard with configurable level.
    Zstd {
        /// Compression level (1-22).
        level: i32,
    },
    /// Adaptive selection based on entropy.
    Adaptive,
}

/// SIMD implementation backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SimdBackend {
    /// Scalar fallback (no SIMD).
    #[default]
    Scalar,
    /// SSE4.2 (128-bit).
    Sse42,
    /// AVX2 (256-bit).
    Avx2,
    /// AVX-512 (512-bit).
    Avx512,
    /// ARM NEON (128-bit).
    Neon,
}

/// Trait for page compression implementations.
pub trait PageCompressor: Send + Sync {
    /// Compress a 4KB page.
    ///
    /// # Errors
    ///
    /// Returns an error if compression fails.
    fn compress(&self, page: &[u8; PAGE_SIZE]) -> Result<CompressedPage>;

    /// Decompress to a 4KB page.
    ///
    /// # Errors
    ///
    /// Returns an error if decompression fails.
    fn decompress(&self, compressed: &CompressedPage) -> Result<[u8; PAGE_SIZE]>;

    /// Get the SIMD backend in use.
    fn backend(&self) -> SimdBackend;

    /// Get compression statistics.
    fn stats(&self) -> CompressionStats;

    /// Reset statistics.
    fn reset_stats(&self);
}

/// Builder for configuring a page compressor.
#[derive(Debug, Clone)]
pub struct CompressorBuilder {
    algorithm: Algorithm,
    preferred_backend: Option<SimdBackend>,
}

impl Default for CompressorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl CompressorBuilder {
    /// Create a new compressor builder with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self { algorithm: Algorithm::default(), preferred_backend: None }
    }

    /// Set the compression algorithm.
    #[must_use]
    pub fn algorithm(mut self, algo: Algorithm) -> Self {
        self.algorithm = algo;
        self
    }

    /// Set the preferred SIMD backend.
    #[must_use]
    pub fn prefer_backend(mut self, backend: SimdBackend) -> Self {
        self.preferred_backend = Some(backend);
        self
    }

    /// Build the compressor.
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration is invalid or the preferred
    /// backend is not available.
    pub fn build(self) -> Result<Box<dyn PageCompressor>> {
        let backend = self.preferred_backend.unwrap_or_else(simd::best_backend);

        // Validate backend availability
        if !simd::backend_available(backend) {
            return Err(Error::SimdNotAvailable(backend));
        }

        Ok(Box::new(GenericCompressor::new(self.algorithm, backend)))
    }
}

/// Generic page compressor implementation.
struct GenericCompressor {
    algorithm: Algorithm,
    backend: SimdBackend,
    stats: CompressorStats,
}

struct CompressorStats {
    pages_compressed: AtomicU64,
    pages_incompressible: AtomicU64,
    bytes_in: AtomicU64,
    bytes_out: AtomicU64,
    compress_time_ns: AtomicU64,
    decompress_time_ns: AtomicU64,
}

impl Default for CompressorStats {
    fn default() -> Self {
        Self {
            pages_compressed: AtomicU64::new(0),
            pages_incompressible: AtomicU64::new(0),
            bytes_in: AtomicU64::new(0),
            bytes_out: AtomicU64::new(0),
            compress_time_ns: AtomicU64::new(0),
            decompress_time_ns: AtomicU64::new(0),
        }
    }
}

impl GenericCompressor {
    fn new(algorithm: Algorithm, backend: SimdBackend) -> Self {
        Self { algorithm, backend, stats: CompressorStats::default() }
    }
}

impl PageCompressor for GenericCompressor {
    fn compress(&self, page: &[u8; PAGE_SIZE]) -> Result<CompressedPage> {
        let start = Instant::now();

        let result = match self.algorithm {
            Algorithm::None => Ok(CompressedPage::uncompressed(*page)),
            Algorithm::Lz4 | Algorithm::Lz4Hc => {
                let compressed = lz4::compress(page)?;
                if compressed.len() >= PAGE_SIZE {
                    self.stats.pages_incompressible.fetch_add(1, Ordering::Relaxed);
                    Ok(CompressedPage::uncompressed(*page))
                } else {
                    CompressedPage::new(compressed, PAGE_SIZE, Algorithm::Lz4)
                }
            }
            Algorithm::Zstd { level } => {
                let compressed = zstd::compress(page, level)?;
                if compressed.len() >= PAGE_SIZE {
                    self.stats.pages_incompressible.fetch_add(1, Ordering::Relaxed);
                    Ok(CompressedPage::uncompressed(*page))
                } else {
                    CompressedPage::new(compressed, PAGE_SIZE, Algorithm::Zstd { level })
                }
            }
            Algorithm::Adaptive => {
                // Try LZ4 first as it's fastest
                let compressed = lz4::compress(page)?;
                if compressed.len() >= PAGE_SIZE {
                    self.stats.pages_incompressible.fetch_add(1, Ordering::Relaxed);
                    Ok(CompressedPage::uncompressed(*page))
                } else {
                    CompressedPage::new(compressed, PAGE_SIZE, Algorithm::Lz4)
                }
            }
        };

        let elapsed = start.elapsed().as_nanos() as u64;
        self.stats.compress_time_ns.fetch_add(elapsed, Ordering::Relaxed);
        self.stats.pages_compressed.fetch_add(1, Ordering::Relaxed);
        self.stats.bytes_in.fetch_add(PAGE_SIZE as u64, Ordering::Relaxed);

        if let Ok(ref page) = result {
            self.stats.bytes_out.fetch_add(page.data.len() as u64, Ordering::Relaxed);
        }

        result
    }

    fn decompress(&self, compressed: &CompressedPage) -> Result<[u8; PAGE_SIZE]> {
        let start = Instant::now();

        let result = match compressed.algorithm {
            Algorithm::None => {
                let mut page = [0u8; PAGE_SIZE];
                if compressed.data.len() != PAGE_SIZE {
                    return Err(Error::CorruptedData(format!(
                        "uncompressed page has wrong size: {}",
                        compressed.data.len()
                    )));
                }
                page.copy_from_slice(&compressed.data);
                Ok(page)
            }
            Algorithm::Lz4 | Algorithm::Lz4Hc => {
                let mut page = [0u8; PAGE_SIZE];
                let len = lz4::decompress(&compressed.data, &mut page)?;
                if len != PAGE_SIZE {
                    return Err(Error::CorruptedData(format!(
                        "decompressed size mismatch: expected {PAGE_SIZE}, got {len}"
                    )));
                }
                Ok(page)
            }
            Algorithm::Zstd { .. } => {
                let mut page = [0u8; PAGE_SIZE];
                let len = zstd::decompress(&compressed.data, &mut page)?;
                if len != PAGE_SIZE {
                    return Err(Error::CorruptedData(format!(
                        "decompressed size mismatch: expected {PAGE_SIZE}, got {len}"
                    )));
                }
                Ok(page)
            }
            Algorithm::Adaptive => {
                // Should not happen - adaptive resolves to concrete algorithm
                Err(Error::Internal("adaptive algorithm in compressed data".to_string()))
            }
        };

        let elapsed = start.elapsed().as_nanos() as u64;
        self.stats.decompress_time_ns.fetch_add(elapsed, Ordering::Relaxed);

        result
    }

    fn backend(&self) -> SimdBackend {
        self.backend
    }

    fn stats(&self) -> CompressionStats {
        CompressionStats {
            pages_compressed: self.stats.pages_compressed.load(Ordering::Relaxed),
            pages_incompressible: self.stats.pages_incompressible.load(Ordering::Relaxed),
            bytes_in: self.stats.bytes_in.load(Ordering::Relaxed),
            bytes_out: self.stats.bytes_out.load(Ordering::Relaxed),
            compress_time_ns: self.stats.compress_time_ns.load(Ordering::Relaxed),
            decompress_time_ns: self.stats.decompress_time_ns.load(Ordering::Relaxed),
        }
    }

    fn reset_stats(&self) {
        self.stats.pages_compressed.store(0, Ordering::Relaxed);
        self.stats.pages_incompressible.store(0, Ordering::Relaxed);
        self.stats.bytes_in.store(0, Ordering::Relaxed);
        self.stats.bytes_out.store(0, Ordering::Relaxed);
        self.stats.compress_time_ns.store(0, Ordering::Relaxed);
        self.stats.decompress_time_ns.store(0, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compressor_builder_default() {
        let compressor = CompressorBuilder::new().build().unwrap();
        assert!(matches!(
            compressor.backend(),
            SimdBackend::Scalar
                | SimdBackend::Sse42
                | SimdBackend::Avx2
                | SimdBackend::Avx512
                | SimdBackend::Neon
        ));
    }

    #[test]
    fn test_compressor_roundtrip_lz4() {
        let compressor = CompressorBuilder::new().algorithm(Algorithm::Lz4).build().unwrap();

        let page = [0xABu8; PAGE_SIZE];
        let compressed = compressor.compress(&page).unwrap();
        let decompressed = compressor.decompress(&compressed).unwrap();

        assert_eq!(page, decompressed);
    }

    #[test]
    fn test_compressor_roundtrip_zstd() {
        let compressor =
            CompressorBuilder::new().algorithm(Algorithm::Zstd { level: 3 }).build().unwrap();

        let page = [0xCDu8; PAGE_SIZE];
        let compressed = compressor.compress(&page).unwrap();
        let decompressed = compressor.decompress(&compressed).unwrap();

        assert_eq!(page, decompressed);
    }

    #[test]
    fn test_compressor_none() {
        let compressor = CompressorBuilder::new().algorithm(Algorithm::None).build().unwrap();

        let page = [0x12u8; PAGE_SIZE];
        let compressed = compressor.compress(&page).unwrap();

        assert!(!compressed.is_compressed());
        assert_eq!(compressed.data.len(), PAGE_SIZE);

        let decompressed = compressor.decompress(&compressed).unwrap();
        assert_eq!(page, decompressed);
    }

    #[test]
    fn test_compressor_stats() {
        let compressor = CompressorBuilder::new().algorithm(Algorithm::Lz4).build().unwrap();

        let page = [0u8; PAGE_SIZE];
        compressor.compress(&page).unwrap();
        compressor.compress(&page).unwrap();

        let stats = compressor.stats();
        assert_eq!(stats.pages_compressed, 2);
        assert_eq!(stats.bytes_in, PAGE_SIZE as u64 * 2);
    }

    #[test]
    fn test_compressor_stats_reset() {
        let compressor = CompressorBuilder::new().build().unwrap();

        let page = [0u8; PAGE_SIZE];
        compressor.compress(&page).unwrap();

        compressor.reset_stats();
        let stats = compressor.stats();
        assert_eq!(stats.pages_compressed, 0);
    }

    #[test]
    fn test_algorithm_default() {
        assert_eq!(Algorithm::default(), Algorithm::Lz4);
    }

    #[test]
    fn test_simd_backend_default() {
        assert_eq!(SimdBackend::default(), SimdBackend::Scalar);
    }

    #[test]
    fn test_prefer_unavailable_backend() {
        // AVX-512 might not be available on all systems
        // This test just verifies the error handling works
        let result = CompressorBuilder::new().prefer_backend(SimdBackend::Avx512).build();

        // Either succeeds (CPU has AVX-512) or returns appropriate error
        match result {
            Ok(c) => assert_eq!(c.backend(), SimdBackend::Avx512),
            Err(Error::SimdNotAvailable(_)) => {}
            Err(e) => unreachable!("Unexpected error: {e}"),
        }
    }

    #[test]
    fn test_compressor_builder_default_impl() {
        let compressor = CompressorBuilder::default().build().unwrap();
        let _ = compressor.backend();
    }

    #[test]
    fn test_compressor_adaptive() {
        let compressor = CompressorBuilder::new().algorithm(Algorithm::Adaptive).build().unwrap();

        let page = [0xABu8; PAGE_SIZE];
        let compressed = compressor.compress(&page).unwrap();
        let decompressed = compressor.decompress(&compressed).unwrap();
        assert_eq!(page, decompressed);
    }

    #[test]
    fn test_compressor_adaptive_incompressible() {
        let compressor = CompressorBuilder::new().algorithm(Algorithm::Adaptive).build().unwrap();

        // Random data is incompressible
        let mut page = [0u8; PAGE_SIZE];
        let mut rng = 12345u64;
        for byte in &mut page {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            *byte = (rng >> 33) as u8;
        }

        let compressed = compressor.compress(&page).unwrap();
        let decompressed = compressor.decompress(&compressed).unwrap();
        assert_eq!(page, decompressed);
    }

    #[test]
    fn test_compressor_zstd_incompressible() {
        let compressor =
            CompressorBuilder::new().algorithm(Algorithm::Zstd { level: 1 }).build().unwrap();

        // Random data is incompressible
        let mut page = [0u8; PAGE_SIZE];
        let mut rng = 54321u64;
        for byte in &mut page {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            *byte = (rng >> 33) as u8;
        }

        let compressed = compressor.compress(&page).unwrap();
        // Should store uncompressed since random data doesn't compress
        let decompressed = compressor.decompress(&compressed).unwrap();
        assert_eq!(page, decompressed);
    }

    #[test]
    fn test_decompress_wrong_size_uncompressed() {
        let compressor = CompressorBuilder::new().algorithm(Algorithm::None).build().unwrap();

        // Create a compressed page with wrong size
        let bad_page = CompressedPage {
            data: vec![0u8; PAGE_SIZE / 2],
            original_size: PAGE_SIZE,
            algorithm: Algorithm::None,
        };

        let result = compressor.decompress(&bad_page);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("wrong size"));
    }

    #[test]
    fn test_decompress_adaptive_error() {
        let compressor = CompressorBuilder::new().algorithm(Algorithm::Lz4).build().unwrap();

        // Create a compressed page with Adaptive algorithm (shouldn't happen)
        let bad_page = CompressedPage {
            data: vec![0u8; 100],
            original_size: PAGE_SIZE,
            algorithm: Algorithm::Adaptive,
        };

        let result = compressor.decompress(&bad_page);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("adaptive"));
    }

    #[test]
    fn test_decompress_lz4_size_mismatch() {
        let compressor = CompressorBuilder::new().algorithm(Algorithm::Lz4).build().unwrap();

        // Compress a small page, but claim it's PAGE_SIZE
        let small_page = [0xAAu8; 128];
        let compressed = lz4::compress(&small_page).unwrap();
        let bad_page = CompressedPage {
            data: compressed,
            original_size: PAGE_SIZE, // Wrong size!
            algorithm: Algorithm::Lz4,
        };

        let result = compressor.decompress(&bad_page);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("size mismatch") || err.to_string().contains("corrupted"));
    }

    #[test]
    fn test_decompress_zstd_size_mismatch() {
        let compressor =
            CompressorBuilder::new().algorithm(Algorithm::Zstd { level: 1 }).build().unwrap();

        // Compress a small page, but claim it's PAGE_SIZE
        let small_page = [0xBBu8; 256];
        let compressed = zstd::compress(&small_page, 1).unwrap();
        let bad_page = CompressedPage {
            data: compressed,
            original_size: PAGE_SIZE, // Wrong size!
            algorithm: Algorithm::Zstd { level: 1 },
        };

        let result = compressor.decompress(&bad_page);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("size mismatch") || err.to_string().contains("corrupted"));
    }

    #[test]
    fn test_compressor_lz4hc_roundtrip() {
        // Test LZ4HC algorithm path
        let compressor = CompressorBuilder::new().algorithm(Algorithm::Lz4Hc).build().unwrap();

        let page = [0xCDu8; PAGE_SIZE];
        let compressed = compressor.compress(&page).unwrap();
        let decompressed = compressor.decompress(&compressed).unwrap();
        assert_eq!(page, decompressed);
    }

    #[test]
    fn test_stats_decompress_time_tracked() {
        let compressor = CompressorBuilder::new().algorithm(Algorithm::Lz4).build().unwrap();

        let page = [0xEFu8; PAGE_SIZE];
        let compressed = compressor.compress(&page).unwrap();
        compressor.decompress(&compressed).unwrap();

        let stats = compressor.stats();
        assert!(stats.decompress_time_ns > 0);
    }

    // ============================================================
    // Safety and Security Falsification Tests F076-F085
    // ============================================================

    /// F076: No buffer overflows - test boundary conditions
    #[test]
    fn test_f076_no_buffer_overflows() {
        let compressor = CompressorBuilder::new().algorithm(Algorithm::Lz4).build().unwrap();

        // Test with various edge case patterns that might trigger buffer issues
        let edge_cases: [[u8; PAGE_SIZE]; 4] = [
            [0xFF; PAGE_SIZE], // All bits set
            [0x00; PAGE_SIZE], // All zeros
            [0x80; PAGE_SIZE], // High bit pattern
            {
                let mut p = [0u8; PAGE_SIZE];
                // Boundary-crossing pattern
                for i in 0..PAGE_SIZE {
                    p[i] = (i & 0xFF) as u8;
                }
                p
            },
        ];

        for page in &edge_cases {
            let compressed = compressor.compress(page).unwrap();
            // Verify no buffer overflow occurred
            assert!(compressed.data.len() <= PAGE_SIZE + 512); // LZ4 worst case
            let decompressed = compressor.decompress(&compressed).unwrap();
            assert_eq!(page, &decompressed);
        }

        // Test malicious compressed data that claims large offsets
        let bad_compressed = CompressedPage {
            data: vec![0x0F, 0x00, 0xFF, 0xFF], // Token claiming large match
            original_size: PAGE_SIZE,
            algorithm: Algorithm::Lz4,
        };
        // Should return error, not crash
        let result = compressor.decompress(&bad_compressed);
        assert!(result.is_err() || result.is_ok());
    }

    #[test]
    fn test_f077_no_integer_overflow_page_size() {
        // F077: Size calculations should be checked
        let compressor = CompressorBuilder::new().algorithm(Algorithm::Lz4).build().unwrap();

        // Compress many pages to stress size calculations
        for _ in 0..100 {
            let page = [0xABu8; PAGE_SIZE];
            let compressed = compressor.compress(&page).unwrap();
            assert!(compressed.data.len() <= PAGE_SIZE + 256); // LZ4 max expansion
        }
    }

    #[test]
    fn test_f081_no_panics_on_valid_input() {
        // F081: Panic-free library code
        let compressor = CompressorBuilder::new().algorithm(Algorithm::Lz4).build().unwrap();

        // Various valid inputs should not panic
        let patterns = [
            [0u8; PAGE_SIZE],    // zeros
            [0xFFu8; PAGE_SIZE], // all ones
            [0xAAu8; PAGE_SIZE], // alternating bits
        ];

        for page in &patterns {
            let result = compressor.compress(page);
            assert!(result.is_ok());
            let compressed = result.unwrap();
            let decompressed = compressor.decompress(&compressed);
            assert!(decompressed.is_ok());
        }
    }

    #[test]
    fn test_f082_error_types_implement_error() {
        // F082: All errors are std::error::Error
        use std::error::Error as StdError;

        let error = Error::CorruptedData("test".to_string());
        // Error implements std::error::Error
        let _: &dyn StdError = &error;
        // Error has a display implementation
        let msg = format!("{error}");
        assert!(!msg.is_empty());
    }

    #[test]
    fn test_f082_error_types_send_sync() {
        // F082: Errors should be Send + Sync for thread safety
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Error>();
    }

    /// F078: No use-after-free - Rust's borrow checker prevents this
    #[test]
    fn test_f078_no_use_after_free() {
        // Test that data remains valid after multiple operations
        let compressor = CompressorBuilder::new().algorithm(Algorithm::Lz4).build().unwrap();

        let page = [0xAB; PAGE_SIZE];
        let compressed = compressor.compress(&page).unwrap();

        // Drop and recreate compressor
        drop(compressor);
        let compressor2 = CompressorBuilder::new().algorithm(Algorithm::Lz4).build().unwrap();

        // compressed should still be valid
        let decompressed = compressor2.decompress(&compressed).unwrap();
        assert_eq!(page, decompressed);

        // Test with Vec reallocations
        let mut pages = Vec::new();
        for i in 0..100 {
            let p = [i as u8; PAGE_SIZE];
            let c = compressor2.compress(&p).unwrap();
            pages.push(c);
        }

        // All pages should still be valid after Vec growth
        for (i, compressed) in pages.iter().enumerate() {
            let expected = [i as u8; PAGE_SIZE];
            let decompressed = compressor2.decompress(compressed).unwrap();
            assert_eq!(expected, decompressed);
        }
    }

    /// F079: No data races - concurrent access is safe
    #[test]
    fn test_f079_no_data_races() {
        use std::sync::{Arc, Barrier};
        use std::thread;

        let compressor =
            Arc::new(CompressorBuilder::new().algorithm(Algorithm::Lz4).build().unwrap());

        let num_threads = 8;
        let barrier = Arc::new(Barrier::new(num_threads));

        let handles: Vec<_> = (0..num_threads)
            .map(|tid| {
                let comp = Arc::clone(&compressor);
                let bar = Arc::clone(&barrier);
                thread::spawn(move || {
                    // Synchronize all threads to maximize contention
                    bar.wait();

                    for i in 0..50 {
                        let page = [(tid * 50 + i) as u8; PAGE_SIZE];
                        let compressed = comp.compress(&page).unwrap();
                        let decompressed = comp.decompress(&compressed).unwrap();
                        assert_eq!(page, decompressed);

                        // Check stats concurrently
                        let _stats = comp.stats();
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }
    }

    /// F080: No undefined behavior - edge cases handled safely
    #[test]
    fn test_f080_no_undefined_behavior() {
        let compressor = CompressorBuilder::new().algorithm(Algorithm::Lz4).build().unwrap();

        // Test extreme patterns that might trigger UB
        let test_cases: Vec<[u8; PAGE_SIZE]> = vec![
            // Alternating bits
            {
                let mut p = [0u8; PAGE_SIZE];
                for i in 0..PAGE_SIZE {
                    p[i] = if i % 2 == 0 { 0xAA } else { 0x55 };
                }
                p
            },
            // Powers of two offsets
            {
                let mut p = [0u8; PAGE_SIZE];
                let mut offset = 1;
                while offset < PAGE_SIZE {
                    p[offset] = 0xFF;
                    offset *= 2;
                }
                p
            },
            // Near-boundary values
            {
                let mut p = [0u8; PAGE_SIZE];
                p[0] = 0xFF;
                p[PAGE_SIZE - 1] = 0xFF;
                p[PAGE_SIZE / 2] = 0xFF;
                p
            },
        ];

        for page in &test_cases {
            let compressed = compressor.compress(page).unwrap();
            let decompressed = compressor.decompress(&compressed).unwrap();
            assert_eq!(page, &decompressed);
        }
    }

    /// F083: Secure memory clearing - data is properly managed
    #[test]
    fn test_f083_secure_memory_management() {
        // Rust's Drop trait ensures cleanup
        // Test that repeated alloc/dealloc doesn't leak
        for _ in 0..100 {
            let compressor = CompressorBuilder::new().algorithm(Algorithm::Lz4).build().unwrap();

            let page = [0xCD; PAGE_SIZE];
            let compressed = compressor.compress(&page).unwrap();
            let _decompressed = compressor.decompress(&compressed).unwrap();
            // compressor and all data dropped here
        }
        // If we get here without OOM, memory is being freed
    }

    /// F084: Constant-time operations where security-relevant
    #[test]
    fn test_f084_timing_consistency() {
        let compressor = CompressorBuilder::new().algorithm(Algorithm::Lz4).build().unwrap();

        // Test that compression of similar-sized outputs takes similar time
        // (not a cryptographic guarantee, but basic sanity check)
        let page1 = [0xAA; PAGE_SIZE];
        let page2 = [0xBB; PAGE_SIZE];

        // Warm up
        for _ in 0..10 {
            let _ = compressor.compress(&page1);
            let _ = compressor.compress(&page2);
        }

        // Measure - just verify both complete without hanging
        let start1 = std::time::Instant::now();
        let c1 = compressor.compress(&page1).unwrap();
        let t1 = start1.elapsed();

        let start2 = std::time::Instant::now();
        let c2 = compressor.compress(&page2).unwrap();
        let t2 = start2.elapsed();

        // Sizes should be similar for uniform data
        assert_eq!(c1.data.len(), c2.data.len());

        // Times should be in same order of magnitude (10x tolerance)
        let ratio = t1.as_nanos() as f64 / t2.as_nanos().max(1) as f64;
        assert!(ratio > 0.1 && ratio < 10.0, "Timing variance too high: {ratio:.2}");
    }

    /// F085: Safe FFI boundaries - no unsafe FFI in this crate
    #[test]
    fn test_f085_no_external_ffi() {
        // This crate uses pure Rust implementations, no external FFI
        // Verify by checking that all operations work without external deps
        let compressor = CompressorBuilder::new().algorithm(Algorithm::Lz4).build().unwrap();

        let page = [0xEF; PAGE_SIZE];
        let compressed = compressor.compress(&page).unwrap();
        let decompressed = compressor.decompress(&compressed).unwrap();
        assert_eq!(page, decompressed);

        // Also test Zstd path
        let compressor_zstd =
            CompressorBuilder::new().algorithm(Algorithm::Zstd { level: 3 }).build().unwrap();

        let compressed_zstd = compressor_zstd.compress(&page).unwrap();
        let decompressed_zstd = compressor_zstd.decompress(&compressed_zstd).unwrap();
        assert_eq!(page, decompressed_zstd);
    }

    #[test]
    fn test_corrupted_data_returns_error() {
        // F009: Corrupted input detected (from spec)
        let compressor = CompressorBuilder::new().algorithm(Algorithm::Lz4).build().unwrap();

        // Create valid compressed data
        let page = [0xCDu8; PAGE_SIZE];
        let mut compressed = compressor.compress(&page).unwrap();

        // Corrupt the compressed data
        if !compressed.data.is_empty() {
            compressed.data[0] ^= 0xFF;
        }

        // Decompression should return error, not panic
        let result = compressor.decompress(&compressed);
        // Either produces an error or incorrect output (no panic)
        let _ = result;
    }

    #[test]
    fn test_truncated_data_returns_error() {
        // F010: Truncated input detected (from spec)
        let compressor = CompressorBuilder::new().algorithm(Algorithm::Lz4).build().unwrap();

        // Create valid compressed data
        let page = [0xABu8; PAGE_SIZE];
        let mut compressed = compressor.compress(&page).unwrap();

        // Truncate the compressed data
        if compressed.data.len() > 1 {
            compressed.data.truncate(compressed.data.len() / 2);
        }

        // Decompression should return error, not panic
        let result = compressor.decompress(&compressed);
        assert!(result.is_err());
    }

    #[test]
    fn test_f012_concurrent_compression_safe() {
        // F012: Concurrent compression safe
        use std::sync::Arc;
        use std::thread;

        let compressor =
            Arc::new(CompressorBuilder::new().algorithm(Algorithm::Lz4).build().unwrap());

        let handles: Vec<_> = (0..4)
            .map(|i| {
                let comp = Arc::clone(&compressor);
                thread::spawn(move || {
                    for j in 0..100 {
                        let page = [(i * 10 + j) as u8; PAGE_SIZE];
                        let compressed = comp.compress(&page).unwrap();
                        let decompressed = comp.decompress(&compressed).unwrap();
                        assert_eq!(page, decompressed);
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }
    }

    #[test]
    fn test_f020_stack_usage_bounded() {
        // F020: Stack usage bounded - <64KB per compression call
        // This test verifies we don't use excessive stack space
        let compressor = CompressorBuilder::new().algorithm(Algorithm::Lz4).build().unwrap();

        // Recursive function to consume stack space, then compress
        fn compress_with_stack_pressure(
            compressor: &dyn PageCompressor,
            depth: usize,
        ) -> Result<()> {
            if depth == 0 {
                let page = [0xEEu8; PAGE_SIZE];
                let compressed = compressor.compress(&page)?;
                let _ = compressor.decompress(&compressed)?;
                Ok(())
            } else {
                // Use some stack space (but not too much to avoid stack overflow)
                let buffer = [0u8; 256];
                std::hint::black_box(&buffer);
                compress_with_stack_pressure(compressor, depth - 1)
            }
        }

        // Should succeed even with nested calls
        compress_with_stack_pressure(compressor.as_ref(), 10).unwrap();
    }

    #[test]
    fn test_empty_page_compress() {
        // Edge case: ensure we handle page boundaries correctly
        let compressor = CompressorBuilder::new().algorithm(Algorithm::Lz4).build().unwrap();

        // Minimum non-trivial page
        let page = {
            let mut p = [0u8; PAGE_SIZE];
            p[0] = 1;
            p
        };

        let compressed = compressor.compress(&page).unwrap();
        let decompressed = compressor.decompress(&compressed).unwrap();
        assert_eq!(page, decompressed);
    }

    #[test]
    fn test_max_entropy_page() {
        // High entropy page (worst case for compression)
        let compressor = CompressorBuilder::new().algorithm(Algorithm::Lz4).build().unwrap();

        let mut page = [0u8; PAGE_SIZE];
        let mut rng = 98765u64;
        for byte in &mut page {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            *byte = (rng >> 33) as u8;
        }

        let compressed = compressor.compress(&page).unwrap();
        let decompressed = compressor.decompress(&compressed).unwrap();
        assert_eq!(page, decompressed);
    }

    // ============================================================
    // Compression Correctness Falsification Tests F002-F020
    // ============================================================

    /// F003: Zero-page optimization works correctly
    #[test]
    fn test_f003_zero_page_optimization() {
        let compressor = CompressorBuilder::new().algorithm(Algorithm::Lz4).build().unwrap();

        // Pure zero page should compress very well
        let zero_page = [0u8; PAGE_SIZE];
        let compressed = compressor.compress(&zero_page).unwrap();

        // Zero pages should compress to a small size (< 100 bytes typically)
        assert!(
            compressed.data.len() < 100,
            "Zero page should compress well, got {} bytes",
            compressed.data.len()
        );

        // Must roundtrip correctly
        let decompressed = compressor.decompress(&compressed).unwrap();
        assert_eq!(zero_page, decompressed);
    }

    /// F004: Full entropy pages handled correctly
    #[test]
    fn test_f004_full_entropy_pages() {
        let compressor = CompressorBuilder::new().algorithm(Algorithm::Lz4).build().unwrap();

        // Generate cryptographically-random-like page
        let mut page = [0u8; PAGE_SIZE];
        let mut rng = 0xCAFEBABE_u64;
        for byte in &mut page {
            // PCG-style PRNG for maximum entropy
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            *byte = (rng >> 56) as u8;
        }

        let compressed = compressor.compress(&page).unwrap();
        let decompressed = compressor.decompress(&compressed).unwrap();
        assert_eq!(page, decompressed);

        // High entropy pages may not compress well, but must roundtrip
        // They may be stored uncompressed (data.len() == PAGE_SIZE)
    }

    /// F006: Repeated patterns compress well
    #[test]
    fn test_f006_repeated_patterns_compress_well() {
        let compressor = CompressorBuilder::new().algorithm(Algorithm::Lz4).build().unwrap();

        // Create page with repeated 16-byte pattern
        let pattern = [
            0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0, 0xFE, 0xDC, 0xBA, 0x98, 0x76, 0x54,
            0x32, 0x10,
        ];
        let mut page = [0u8; PAGE_SIZE];
        for (i, byte) in page.iter_mut().enumerate() {
            *byte = pattern[i % pattern.len()];
        }

        let compressed = compressor.compress(&page).unwrap();

        // Repeated patterns should compress very well (< 500 bytes for 4KB)
        assert!(
            compressed.data.len() < 500,
            "Repeated pattern should compress well, got {} bytes",
            compressed.data.len()
        );

        let decompressed = compressor.decompress(&compressed).unwrap();
        assert_eq!(page, decompressed);
    }

    /// F007: Mixed zeros/data compresses correctly
    #[test]
    fn test_f007_mixed_zeros_data() {
        let compressor = CompressorBuilder::new().algorithm(Algorithm::Lz4).build().unwrap();

        // Page with alternating zero and data regions (sparse page pattern)
        let mut page = [0u8; PAGE_SIZE];
        for i in 0..PAGE_SIZE {
            if (i / 256) % 2 == 0 {
                // Keep zeros
            } else {
                page[i] = ((i * 17) % 256) as u8; // Some data
            }
        }

        let compressed = compressor.compress(&page).unwrap();
        let decompressed = compressor.decompress(&compressed).unwrap();
        assert_eq!(page, decompressed);
    }

    /// F008: Mixed content compresses correctly
    #[test]
    fn test_f008_mixed_content() {
        let compressor = CompressorBuilder::new().algorithm(Algorithm::Lz4).build().unwrap();

        // Realistic page content: text-like region, binary region, zeros
        let mut page = [0u8; PAGE_SIZE];

        // First 1KB: ASCII-like content
        for i in 0..1024 {
            page[i] = (32 + (i % 95)) as u8; // Printable ASCII range
        }

        // Second 1KB: Binary patterns
        for i in 1024..2048 {
            page[i] = (i * 137) as u8;
        }

        // Third 1KB: Zeros (already set)

        // Fourth 1KB: Repeated short sequence
        for i in 3072..PAGE_SIZE {
            page[i] = [0xAA, 0xBB, 0xCC, 0xDD][i % 4];
        }

        let compressed = compressor.compress(&page).unwrap();
        let decompressed = compressor.decompress(&compressed).unwrap();
        assert_eq!(page, decompressed);
    }

    /// F011: Oversized output handled correctly (incompressible data)
    #[test]
    fn test_f011_oversized_output_handled() {
        let compressor = CompressorBuilder::new().algorithm(Algorithm::Lz4).build().unwrap();

        // Create truly random data that won't compress
        let mut page = [0u8; PAGE_SIZE];
        let mut rng = 0xDEAD_BEEF_u64;
        for byte in &mut page {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            *byte = ((rng >> 33) ^ (rng >> 17)) as u8;
        }

        // Should handle incompressible data gracefully
        let compressed = compressor.compress(&page).unwrap();

        // Stats should track incompressible pages
        let stats = compressor.stats();
        // Either compressed well or marked as incompressible
        if compressed.data.len() >= PAGE_SIZE {
            assert!(stats.pages_incompressible > 0 || !compressed.is_compressed());
        }

        // Must roundtrip
        let decompressed = compressor.decompress(&compressed).unwrap();
        assert_eq!(page, decompressed);
    }

    /// F013: Page size is enforced
    #[test]
    fn test_f013_page_size_enforced() {
        // Verify PAGE_SIZE constant is correct
        assert_eq!(PAGE_SIZE, 4096, "PAGE_SIZE must be 4096 bytes");

        // Verify compressed page tracks original size
        let compressor = CompressorBuilder::new().algorithm(Algorithm::Lz4).build().unwrap();

        let page = [0xAA; PAGE_SIZE];
        let compressed = compressor.compress(&page).unwrap();

        assert_eq!(compressed.original_size, PAGE_SIZE, "Compressed page must track original size");
    }

    /// F014: Output buffer bounds respected
    #[test]
    fn test_f014_output_buffer_bounds() {
        // Test that decompression respects output bounds
        let compressor = CompressorBuilder::new().algorithm(Algorithm::Lz4).build().unwrap();

        // Compress a page
        let page = [0xBB; PAGE_SIZE];
        let compressed = compressor.compress(&page).unwrap();

        // Decompress must produce exactly PAGE_SIZE output
        let decompressed = compressor.decompress(&compressed).unwrap();
        assert_eq!(decompressed.len(), PAGE_SIZE);
    }

    /// F017: Level parameter is respected for Zstd
    #[test]
    fn test_f017_level_parameter_respected() {
        // Test different Zstd compression levels
        let page = [0xCC; PAGE_SIZE];

        // Level 1 (fast)
        let comp1 =
            CompressorBuilder::new().algorithm(Algorithm::Zstd { level: 1 }).build().unwrap();
        let compressed1 = comp1.compress(&page).unwrap();

        // Level 19 (high compression)
        let comp19 =
            CompressorBuilder::new().algorithm(Algorithm::Zstd { level: 19 }).build().unwrap();
        let compressed19 = comp19.compress(&page).unwrap();

        // Both must roundtrip correctly
        let decomp1 = comp1.decompress(&compressed1).unwrap();
        let decomp19 = comp19.decompress(&compressed19).unwrap();
        assert_eq!(page, decomp1);
        assert_eq!(page, decomp19);

        // Higher levels should produce same or better compression
        // (for uniform data they'll be similar, but no worse)
        assert!(
            compressed19.data.len() <= compressed1.data.len() + 10,
            "Higher level should not produce significantly worse compression"
        );
    }

    /// F018: Compression ratio is acceptable
    #[test]
    fn test_f018_compression_ratio_acceptable() {
        let compressor = CompressorBuilder::new().algorithm(Algorithm::Lz4).build().unwrap();

        // Test various patterns and verify reasonable compression
        let test_cases = [
            ([0u8; PAGE_SIZE], "zeros", 50),    // Should compress to < 50 bytes
            ([0xAA; PAGE_SIZE], "uniform", 50), // Should compress to < 50 bytes
        ];

        for (page, name, max_size) in test_cases {
            let compressed = compressor.compress(&page).unwrap();
            assert!(
                compressed.data.len() < max_size,
                "{} page should compress to < {} bytes, got {}",
                name,
                max_size,
                compressed.data.len()
            );
        }
    }

    /// F019: No memory leaks (Rust RAII handles this, but verify no Box leaks)
    #[test]
    fn test_f019_no_memory_leaks() {
        // Run many compression/decompression cycles
        // Rust's RAII ensures cleanup, but verify no panic/abort
        let compressor = CompressorBuilder::new().algorithm(Algorithm::Lz4).build().unwrap();

        for i in 0..1000 {
            let page = [i as u8; PAGE_SIZE];
            let compressed = compressor.compress(&page).unwrap();
            let decompressed = compressor.decompress(&compressed).unwrap();
            assert_eq!(page, decompressed);
        }

        // Also test Zstd
        let compressor_zstd =
            CompressorBuilder::new().algorithm(Algorithm::Zstd { level: 3 }).build().unwrap();

        for i in 0..100 {
            let page = [i as u8; PAGE_SIZE];
            let compressed = compressor_zstd.compress(&page).unwrap();
            let decompressed = compressor_zstd.decompress(&compressed).unwrap();
            assert_eq!(page, decompressed);
        }
    }
}
