
// ============================================================================
// Trueno-ZRAM Integration (feature-gated)
// ============================================================================

/// ZRAM compression for KV cache memory optimization
///
/// When the `showcase-zram` feature is enabled, this provides integration with
/// trueno-zram-core for SIMD-accelerated LZ4/ZSTD compression of KV cache pages.
///
/// # Performance Targets
///
/// | Algorithm | Throughput | Compression Ratio |
/// |-----------|------------|-------------------|
/// | Same-Fill | 171 GB/s | N/A (zero pages) |
/// | LZ4 (SIMD) | 3.2 GB/s | 2.1x |
/// | ZSTD | 0.8 GB/s | 2.8x |
///
/// # Usage
///
/// ```rust,ignore
/// use aprender::showcase::zram::{ZramConfig, compress_kv_page};
///
/// let config = ZramConfig::default(); // LZ4 with adaptive selection
/// let compressed = compress_kv_page(&kv_page, &config)?;
/// ```
#[cfg(feature = "showcase-zram")]
pub mod zram {
    use trueno_zram_core::{Algorithm, CompressorBuilder, PAGE_SIZE};

    /// ZRAM configuration for KV cache compression
    #[derive(Debug, Clone)]
    pub struct ZramConfig {
        /// Compression algorithm
        pub algorithm: Algorithm,
        /// Enable adaptive algorithm selection
        pub adaptive: bool,
        /// Minimum savings threshold (0.0-1.0) to keep compressed
        pub min_savings: f64,
    }

    impl Default for ZramConfig {
        fn default() -> Self {
            Self {
                algorithm: Algorithm::Lz4,
                adaptive: true,
                min_savings: 0.1, // Require at least 10% compression
            }
        }
    }

    /// ZRAM compression result
    #[derive(Debug, Clone)]
    pub struct ZramResult {
        /// Original size in bytes
        pub original_size: usize,
        /// Compressed size in bytes
        pub compressed_size: usize,
        /// Compression ratio (original / compressed)
        pub ratio: f64,
        /// Algorithm used
        pub algorithm: String,
        /// Whether zero-page optimization was applied
        pub zero_page: bool,
    }

    impl ZramResult {
        /// Calculate compression ratio
        #[must_use]
        pub fn new(original: usize, compressed: usize, algo: &str, zero: bool) -> Self {
            let ratio = if compressed > 0 {
                original as f64 / compressed as f64
            } else {
                f64::INFINITY
            };
            Self {
                original_size: original,
                compressed_size: compressed,
                ratio,
                algorithm: algo.to_string(),
                zero_page: zero,
            }
        }
    }

    /// Compress a KV cache page using ZRAM
    ///
    /// # Arguments
    /// * `data` - Raw page data (must be PAGE_SIZE bytes)
    /// * `config` - ZRAM configuration
    ///
    /// # Returns
    /// `(compressed_data, ZramResult)` tuple on success
    ///
    /// # Errors
    /// Returns error if compression fails
    pub fn compress_kv_page(
        data: &[u8],
        config: &ZramConfig,
    ) -> Result<(Vec<u8>, ZramResult), String> {
        if data.len() != PAGE_SIZE {
            return Err(format!(
                "Data must be exactly {} bytes, got {}",
                PAGE_SIZE,
                data.len()
            ));
        }

        // Check for zero page first (same-fill optimization)
        if data.iter().all(|&b| b == 0) {
            return Ok((
                vec![0u8; 4],
                ZramResult::new(PAGE_SIZE, 4, "same-fill", true),
            ));
        }

        // Convert slice to fixed-size array (required by trueno-zram-core API)
        let page_array: &[u8; PAGE_SIZE] = data
            .try_into()
            .map_err(|_| "Failed to convert slice to page array")?;

        // Create compressor using builder pattern
        let compressor = CompressorBuilder::new()
            .algorithm(config.algorithm)
            .build()
            .map_err(|e| format!("Failed to create compressor: {e}"))?;

        let compressed = compressor
            .compress(page_array)
            .map_err(|e| format!("Compression failed: {e}"))?;

        let algo_name = match config.algorithm {
            Algorithm::None => "none",
            Algorithm::Lz4 => "lz4",
            Algorithm::Lz4Hc => "lz4hc",
            Algorithm::Zstd { .. } => "zstd",
            Algorithm::Adaptive => "adaptive",
        };
        let result = ZramResult::new(PAGE_SIZE, compressed.data.len(), algo_name, false);

        Ok((compressed.data.clone(), result))
    }

    /// Re-export trueno-zram-core types
    pub use trueno_zram_core::{Algorithm as ZramAlgorithm, PAGE_SIZE as ZRAM_PAGE_SIZE};
}

/// Stub module when trueno-zram-core is not available
#[cfg(not(feature = "showcase-zram"))]
pub mod zram {
    /// Stub config when showcase-zram is disabled
    #[derive(Debug, Clone)]
    pub struct ZramConfig {
        /// Compression algorithm name
        pub algorithm: String,
        /// Enable adaptive selection
        pub adaptive: bool,
        /// Minimum savings threshold
        pub min_savings: f64,
    }

    impl Default for ZramConfig {
        fn default() -> Self {
            Self {
                algorithm: "lz4".to_string(),
                adaptive: true,
                min_savings: 0.1,
            }
        }
    }

    /// Stub result when showcase-zram is disabled
    #[derive(Debug, Clone)]
    pub struct ZramResult {
        /// Original size
        pub original_size: usize,
        /// Compressed size
        pub compressed_size: usize,
        /// Compression ratio
        pub ratio: f64,
        /// Zero page flag
        pub zero_page: bool,
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests;
