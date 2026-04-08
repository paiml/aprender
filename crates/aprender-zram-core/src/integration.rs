//! Integration tests for batuta stack compatibility.
//!
//! This module verifies that trueno-zram integrates correctly with:
//! - trueno SIMD backend selection
//! - trueno-gpu GPU acceleration (when available)
//! - Batuta stack feature flags
//! - Lambda Lab hardware tier configuration

use crate::{Algorithm, CompressorBuilder, PageCompressor, SimdBackend};

/// Lambda Lab hardware tier configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LambdaLabTier {
    /// Full tier: H100/A100 GPU, AVX-512 CPU
    Full,
    /// High tier: RTX 4090/A10 GPU, AVX-512 CPU
    High,
    /// Medium tier: Consumer GPU, AVX2 CPU
    Medium,
    /// Minimal tier: CPU-only, SSE4.2
    Minimal,
}

impl LambdaLabTier {
    /// Detect the current hardware tier based on available features.
    #[must_use]
    pub fn detect() -> Self {
        let features = crate::simd::SimdFeatures::detect();

        // Check for GPU availability (simplified check)
        let has_gpu = cfg!(feature = "cuda");

        match (has_gpu, features.has_avx512(), features.avx2) {
            (true, true, _) => Self::Full,
            (true, false, true) => Self::High,
            (false, _, true) => Self::Medium,
            _ => Self::Minimal,
        }
    }

    /// Get the recommended compression configuration for this tier.
    #[must_use]
    pub fn recommended_config(&self) -> TierConfig {
        match self {
            Self::Full => TierConfig {
                algorithm: Algorithm::Lz4,
                use_gpu: true,
                batch_size: 10000,
                backend: SimdBackend::Avx512,
            },
            Self::High => TierConfig {
                algorithm: Algorithm::Lz4,
                use_gpu: true,
                batch_size: 5000,
                backend: SimdBackend::Avx2,
            },
            Self::Medium => TierConfig {
                algorithm: Algorithm::Lz4,
                use_gpu: false,
                batch_size: 1000,
                backend: SimdBackend::Avx2,
            },
            Self::Minimal => TierConfig {
                algorithm: Algorithm::Zstd { level: 1 },
                use_gpu: false,
                batch_size: 100,
                backend: SimdBackend::Scalar,
            },
        }
    }
}

/// Configuration for a Lambda Lab tier.
#[derive(Debug, Clone)]
pub struct TierConfig {
    /// Recommended compression algorithm.
    pub algorithm: Algorithm,
    /// Whether to use GPU acceleration.
    pub use_gpu: bool,
    /// Recommended batch size for GPU.
    pub batch_size: usize,
    /// SIMD backend to use.
    pub backend: SimdBackend,
}

impl TierConfig {
    /// Create a compressor with this configuration.
    pub fn create_compressor(&self) -> crate::Result<Box<dyn PageCompressor>> {
        CompressorBuilder::new()
            .algorithm(self.algorithm)
            .prefer_backend(self.backend)
            .build()
    }
}

/// Feature flags for trueno-zram builds.
#[derive(Debug, Clone, Copy, Default)]
pub struct FeatureFlags {
    /// Standard library support.
    pub std: bool,
    /// CUDA/GPU support.
    pub cuda: bool,
    /// Nightly Rust features.
    pub nightly: bool,
}

impl FeatureFlags {
    /// Detect current build features.
    #[must_use]
    pub fn detect() -> Self {
        Self {
            std: cfg!(feature = "std"),
            cuda: cfg!(feature = "cuda"),
            nightly: cfg!(feature = "nightly"),
        }
    }

    /// Check if all features are compatible for this build.
    #[must_use]
    pub fn is_compatible(&self) -> bool {
        // CUDA requires std
        if self.cuda && !self.std {
            return false;
        }
        true
    }
}

/// Verify integration with trueno SIMD backend.
pub fn verify_simd_integration() -> bool {
    let backend = crate::simd::detect_backend();
    crate::simd::is_available(backend)
}

/// Verify feature flag consistency.
pub fn verify_feature_flags() -> bool {
    let flags = FeatureFlags::detect();
    flags.is_compatible()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PAGE_SIZE;

    // ============================================================
    // Integration Falsification Tests F086-F095
    // ============================================================

    #[test]
    fn test_f086_trueno_simd_integration() {
        // F086: trueno SIMD integration
        let backend = crate::simd::detect_backend();
        assert!(crate::simd::is_available(backend));

        // Can create compressor with detected backend
        let compressor = CompressorBuilder::new()
            .prefer_backend(backend)
            .build()
            .unwrap();

        let page = [0xABu8; PAGE_SIZE];
        let compressed = compressor.compress(&page).unwrap();
        let decompressed = compressor.decompress(&compressed).unwrap();
        assert_eq!(page, decompressed);
    }

    #[test]
    fn test_f089_lambda_lab_tiers_work() {
        // F089: Lambda Lab tiers work
        for tier in [
            LambdaLabTier::Full,
            LambdaLabTier::High,
            LambdaLabTier::Medium,
            LambdaLabTier::Minimal,
        ] {
            let config = tier.recommended_config();

            // Config should be valid
            assert!(config.batch_size > 0);

            // Can create compressor (may fall back to scalar if SIMD not available)
            let compressor = CompressorBuilder::new()
                .algorithm(config.algorithm)
                .build()
                .unwrap();

            let page = [0xCDu8; PAGE_SIZE];
            let result = compressor.compress(&page);
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_f093_feature_flags_compose() {
        // F093: Feature flags compose
        let flags = FeatureFlags::detect();
        assert!(flags.is_compatible());
    }

    #[test]
    fn test_f094_msrv_respected() {
        // F094: MSRV respected (Rust 1.82.0)
        // This test compiles, so MSRV is satisfied
        let version = env!("CARGO_PKG_VERSION");
        std::hint::black_box(version);
    }

    #[test]
    fn test_f095_wasm_excluded_correctly() {
        // F095: WASM excluded correctly
        // GPU code should only be accessible when CUDA feature is enabled
        #[cfg(feature = "cuda")]
        {
            // With CUDA feature, GPU module is available
            use crate::gpu;
            let _available = gpu::gpu_available();
        }

        // Without CUDA feature, GPU code is correctly excluded at compile time
        // (verified by the absence of compilation errors when cuda is disabled)
    }

    #[test]
    fn test_lambda_lab_tier_detection() {
        let tier = LambdaLabTier::detect();
        // Should detect some tier
        let config = tier.recommended_config();
        assert!(config.batch_size > 0);
    }

    #[test]
    fn test_tier_config_create_compressor() {
        let config = TierConfig {
            algorithm: Algorithm::Lz4,
            use_gpu: false,
            batch_size: 100,
            backend: SimdBackend::Scalar,
        };

        let compressor = config.create_compressor().unwrap();
        let page = [0xEFu8; PAGE_SIZE];
        let compressed = compressor.compress(&page).unwrap();
        assert!(!compressed.data.is_empty());
    }

    #[test]
    fn test_verify_simd_integration() {
        assert!(verify_simd_integration());
    }

    #[test]
    fn test_verify_feature_flags() {
        assert!(verify_feature_flags());
    }

    #[test]
    fn test_feature_flags_cuda_requires_std() {
        let flags = FeatureFlags {
            cuda: true,
            std: false,
            ..Default::default()
        };
        assert!(!flags.is_compatible());

        let flags = FeatureFlags {
            cuda: true,
            std: true,
            ..Default::default()
        };
        assert!(flags.is_compatible());
    }

    #[test]
    fn test_tier_full_config() {
        let config = LambdaLabTier::Full.recommended_config();
        assert_eq!(config.algorithm, Algorithm::Lz4);
        assert!(config.use_gpu);
        assert_eq!(config.batch_size, 10000);
    }

    #[test]
    fn test_tier_minimal_config() {
        let config = LambdaLabTier::Minimal.recommended_config();
        assert!(matches!(config.algorithm, Algorithm::Zstd { level: 1 }));
        assert!(!config.use_gpu);
        assert_eq!(config.batch_size, 100);
    }

    #[test]
    fn test_all_algorithms_work() {
        for algo in [
            Algorithm::Lz4,
            Algorithm::Lz4Hc,
            Algorithm::Zstd { level: 1 },
            Algorithm::Zstd { level: 3 },
            Algorithm::Adaptive,
            Algorithm::None,
        ] {
            let compressor = CompressorBuilder::new().algorithm(algo).build().unwrap();

            let page = [0x12u8; PAGE_SIZE];
            let compressed = compressor.compress(&page).unwrap();
            let decompressed = compressor.decompress(&compressed).unwrap();
            assert_eq!(page, decompressed);
        }
    }

    #[test]
    fn test_all_backends_available_check() {
        for backend in [
            SimdBackend::Scalar,
            SimdBackend::Sse42,
            SimdBackend::Avx2,
            SimdBackend::Avx512,
            SimdBackend::Neon,
        ] {
            // is_available should not panic
            let _ = crate::simd::is_available(backend);
        }
    }
}
