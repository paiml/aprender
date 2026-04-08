//! Page classification for algorithm selection.
//!
//! This module provides intelligent routing of compression workloads to
//! the optimal backend (scalar, SIMD, or GPU) based on batch size and
//! entropy characteristics.

use trueno_zram_core::{Algorithm, PAGE_SIZE};

use crate::entropy::{EntropyCalculator, EntropyLevel};

/// Compute backend selection for batch processing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputeBackend {
    /// Scalar CPU (single-threaded).
    Scalar,
    /// SIMD-accelerated CPU (AVX2/AVX-512/NEON).
    Simd,
    /// GPU-accelerated (CUDA/Vulkan/Metal).
    Gpu,
}

/// Minimum batch size for SIMD to be beneficial.
pub const SIMD_BATCH_THRESHOLD: usize = 4;

/// Minimum batch size for GPU to be beneficial (5x PCIe rule).
pub const GPU_BATCH_THRESHOLD: usize = 1000;

/// Classifier that selects compression algorithm based on page characteristics.
#[derive(Debug, Default)]
pub struct PageClassifier {
    entropy_calc: EntropyCalculator,
}

impl PageClassifier {
    /// Create a new page classifier.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Classify a page and return the recommended algorithm.
    #[must_use]
    pub fn classify(&mut self, page: &[u8; PAGE_SIZE]) -> Algorithm {
        let level = self.entropy_calc.classify(page);

        match level {
            EntropyLevel::VeryLow | EntropyLevel::Low => Algorithm::Lz4,
            EntropyLevel::Medium => Algorithm::Zstd { level: 3 },
            EntropyLevel::High => Algorithm::Zstd { level: 1 },
            EntropyLevel::VeryHigh => Algorithm::None,
        }
    }
}

/// Batch classifier for routing workloads to optimal backend.
///
/// Implements the 5x PCIe rule for GPU routing decisions.
#[derive(Debug, Default)]
pub struct BatchClassifier {
    page_classifier: PageClassifier,
    gpu_available: bool,
    simd_available: bool,
}

impl BatchClassifier {
    /// Create a new batch classifier.
    #[must_use]
    pub fn new() -> Self {
        Self {
            page_classifier: PageClassifier::new(),
            gpu_available: false,
            simd_available: true, // Assume SIMD is available by default
        }
    }

    /// Create a batch classifier with GPU support.
    #[must_use]
    pub fn with_gpu(mut self, available: bool) -> Self {
        self.gpu_available = available;
        self
    }

    /// Create a batch classifier with SIMD support.
    #[must_use]
    pub fn with_simd(mut self, available: bool) -> Self {
        self.simd_available = available;
        self
    }

    /// Select the optimal compute backend for a batch of pages.
    ///
    /// Applies the 5x PCIe rule: GPU is only beneficial when batch size
    /// exceeds the threshold where GPU compute time outweighs transfer overhead.
    #[must_use]
    pub fn select_backend(&self, batch_size: usize) -> ComputeBackend {
        if self.gpu_available && batch_size >= GPU_BATCH_THRESHOLD {
            ComputeBackend::Gpu
        } else if self.simd_available && batch_size >= SIMD_BATCH_THRESHOLD {
            ComputeBackend::Simd
        } else {
            ComputeBackend::Scalar
        }
    }

    /// Classify a batch of pages and return per-page algorithm recommendations.
    ///
    /// Also returns the recommended compute backend for the batch.
    pub fn classify_batch(
        &mut self,
        pages: &[[u8; PAGE_SIZE]],
    ) -> (Vec<Algorithm>, ComputeBackend) {
        let backend = self.select_backend(pages.len());

        let algorithms: Vec<Algorithm> = pages
            .iter()
            .map(|page| self.page_classifier.classify(page))
            .collect();

        (algorithms, backend)
    }

    /// Check if GPU routing is recommended for the given batch size.
    #[must_use]
    pub fn should_use_gpu(&self, batch_size: usize) -> bool {
        self.gpu_available && batch_size >= GPU_BATCH_THRESHOLD
    }

    /// Get the entropy level for a single page.
    #[must_use]
    pub fn entropy_level(&mut self, page: &[u8; PAGE_SIZE]) -> EntropyLevel {
        self.page_classifier.entropy_calc.classify(page)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ============================================================
    // PageClassifier Tests
    // ============================================================

    #[test]
    fn test_zero_page_uses_lz4() {
        let mut classifier = PageClassifier::new();
        let page = [0u8; PAGE_SIZE];
        assert_eq!(classifier.classify(&page), Algorithm::Lz4);
    }

    #[test]
    fn test_low_entropy_uses_lz4() {
        let mut classifier = PageClassifier::new();
        // Create low entropy page with 2-4 unique values
        let mut page = [0u8; PAGE_SIZE];
        for (i, byte) in page.iter_mut().enumerate() {
            *byte = u8::try_from(i % 4).unwrap();
        }
        let result = classifier.classify(&page);
        assert!(matches!(result, Algorithm::Lz4));
    }

    #[test]
    fn test_medium_entropy_uses_zstd() {
        let mut classifier = PageClassifier::new();
        // Create medium entropy page with ~50-100 unique values
        let mut page = [0u8; PAGE_SIZE];
        for (i, byte) in page.iter_mut().enumerate() {
            *byte = u8::try_from(i % 64).unwrap();
        }
        let result = classifier.classify(&page);
        // Medium entropy should use Zstd level 3
        assert!(matches!(result, Algorithm::Zstd { level: 3 }));
    }

    #[test]
    fn test_high_entropy_uses_fast_zstd() {
        let mut classifier = PageClassifier::new();
        // Create high entropy page with many unique values but some pattern
        let mut page = [0u8; PAGE_SIZE];
        for (i, byte) in page.iter_mut().enumerate() {
            *byte = u8::try_from((i * 7) % 200).unwrap();
        }
        let result = classifier.classify(&page);
        // High entropy uses Zstd level 1 (fast) or None
        assert!(matches!(
            result,
            Algorithm::Zstd { level: 1 } | Algorithm::None
        ));
    }

    #[test]
    fn test_random_page_skips_compression() {
        let mut classifier = PageClassifier::new();
        // Create very high entropy (random) page
        let mut page = [0u8; PAGE_SIZE];
        let mut state = 12345u64;
        for byte in &mut page {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            *byte = (state >> 56) as u8; // Take top 8 bits
        }
        assert_eq!(classifier.classify(&page), Algorithm::None);
    }

    #[test]
    fn test_classifier_default() {
        let classifier = PageClassifier::default();
        let debug = format!("{classifier:?}");
        assert!(debug.contains("PageClassifier"));
    }

    // ============================================================
    // BatchClassifier Tests (GPU routing - Falsification F051-F065)
    // ============================================================

    #[test]
    fn test_batch_classifier_new() {
        let classifier = BatchClassifier::new();
        assert!(!classifier.gpu_available);
        assert!(classifier.simd_available);
    }

    #[test]
    fn test_batch_classifier_with_gpu() {
        let classifier = BatchClassifier::new().with_gpu(true);
        assert!(classifier.gpu_available);
    }

    #[test]
    fn test_batch_classifier_with_simd() {
        let classifier = BatchClassifier::new().with_simd(false);
        assert!(!classifier.simd_available);
    }

    #[test]
    fn test_batch_classifier_default() {
        let classifier = BatchClassifier::default();
        let debug = format!("{classifier:?}");
        assert!(debug.contains("BatchClassifier"));
    }

    #[test]
    fn test_select_backend_scalar_small_batch() {
        // F051: Batch size 1 should use scalar
        let classifier = BatchClassifier::new().with_gpu(true);
        assert_eq!(classifier.select_backend(1), ComputeBackend::Scalar);
        assert_eq!(classifier.select_backend(3), ComputeBackend::Scalar);
    }

    #[test]
    fn test_select_backend_simd_medium_batch() {
        // F052: Medium batch should use SIMD
        let classifier = BatchClassifier::new().with_gpu(true);
        assert_eq!(classifier.select_backend(4), ComputeBackend::Simd);
        assert_eq!(classifier.select_backend(100), ComputeBackend::Simd);
        assert_eq!(classifier.select_backend(999), ComputeBackend::Simd);
    }

    #[test]
    fn test_select_backend_gpu_large_batch() {
        // F053: Large batch with GPU should use GPU
        let classifier = BatchClassifier::new().with_gpu(true);
        assert_eq!(classifier.select_backend(1000), ComputeBackend::Gpu);
        assert_eq!(classifier.select_backend(10000), ComputeBackend::Gpu);
    }

    #[test]
    fn test_select_backend_no_gpu_fallback() {
        // F054: Without GPU, large batch should use SIMD
        let classifier = BatchClassifier::new().with_gpu(false);
        assert_eq!(classifier.select_backend(1000), ComputeBackend::Simd);
        assert_eq!(classifier.select_backend(10000), ComputeBackend::Simd);
    }

    #[test]
    fn test_select_backend_no_simd_fallback() {
        // F055: Without SIMD, should use scalar
        let classifier = BatchClassifier::new().with_simd(false);
        assert_eq!(classifier.select_backend(100), ComputeBackend::Scalar);
    }

    #[test]
    fn test_select_backend_boundary_simd() {
        // F056: Boundary at SIMD threshold
        let classifier = BatchClassifier::new();
        assert_eq!(
            classifier.select_backend(SIMD_BATCH_THRESHOLD - 1),
            ComputeBackend::Scalar
        );
        assert_eq!(
            classifier.select_backend(SIMD_BATCH_THRESHOLD),
            ComputeBackend::Simd
        );
    }

    #[test]
    fn test_select_backend_boundary_gpu() {
        // F057: Boundary at GPU threshold
        let classifier = BatchClassifier::new().with_gpu(true);
        assert_eq!(
            classifier.select_backend(GPU_BATCH_THRESHOLD - 1),
            ComputeBackend::Simd
        );
        assert_eq!(
            classifier.select_backend(GPU_BATCH_THRESHOLD),
            ComputeBackend::Gpu
        );
    }

    #[test]
    fn test_should_use_gpu_true() {
        // F058: should_use_gpu returns true for large batches
        let classifier = BatchClassifier::new().with_gpu(true);
        assert!(classifier.should_use_gpu(1000));
        assert!(classifier.should_use_gpu(10000));
    }

    #[test]
    fn test_should_use_gpu_false_small_batch() {
        // F059: should_use_gpu returns false for small batches
        let classifier = BatchClassifier::new().with_gpu(true);
        assert!(!classifier.should_use_gpu(1));
        assert!(!classifier.should_use_gpu(999));
    }

    #[test]
    fn test_should_use_gpu_false_no_gpu() {
        // F060: should_use_gpu returns false when GPU unavailable
        let classifier = BatchClassifier::new().with_gpu(false);
        assert!(!classifier.should_use_gpu(1000));
        assert!(!classifier.should_use_gpu(10000));
    }

    #[test]
    fn test_classify_batch_empty() {
        // F061: Empty batch should work
        let mut classifier = BatchClassifier::new();
        let pages: &[[u8; PAGE_SIZE]] = &[];
        let (algos, backend) = classifier.classify_batch(pages);
        assert!(algos.is_empty());
        assert_eq!(backend, ComputeBackend::Scalar);
    }

    #[test]
    fn test_classify_batch_single_page() {
        // F062: Single page batch
        let mut classifier = BatchClassifier::new();
        let page = [0u8; PAGE_SIZE];
        let (algos, backend) = classifier.classify_batch(&[page]);
        assert_eq!(algos.len(), 1);
        assert_eq!(algos[0], Algorithm::Lz4);
        assert_eq!(backend, ComputeBackend::Scalar);
    }

    #[test]
    fn test_classify_batch_multiple_pages() {
        // F063: Multiple pages get individual classifications
        let mut classifier = BatchClassifier::new();
        let zero_page = [0u8; PAGE_SIZE];
        let mut random_page = [0u8; PAGE_SIZE];
        let mut rng = 12345u64;
        for byte in &mut random_page {
            rng = rng.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            *byte = (rng >> 56) as u8; // Take top 8 bits
        }

        let pages = vec![zero_page, random_page, zero_page, random_page, zero_page];
        let (algos, backend) = classifier.classify_batch(&pages);
        assert_eq!(algos.len(), 5);
        // First page should be LZ4, second should be None (random)
        assert_eq!(algos[0], Algorithm::Lz4);
        assert_eq!(algos[1], Algorithm::None);
        assert_eq!(backend, ComputeBackend::Simd);
    }

    #[test]
    fn test_classify_batch_gpu_routing() {
        // F064: Large batch routes to GPU
        let mut classifier = BatchClassifier::new().with_gpu(true);
        let page = [0u8; PAGE_SIZE];
        let pages: Vec<[u8; PAGE_SIZE]> = (0..1000).map(|_| page).collect();
        let (algos, backend) = classifier.classify_batch(&pages);
        assert_eq!(algos.len(), 1000);
        assert_eq!(backend, ComputeBackend::Gpu);
    }

    #[test]
    fn test_entropy_level() {
        // F065: Entropy level access
        let mut classifier = BatchClassifier::new();
        let page = [0u8; PAGE_SIZE];
        let level = classifier.entropy_level(&page);
        assert_eq!(level, EntropyLevel::VeryLow);
    }

    #[test]
    fn test_compute_backend_equality() {
        assert_eq!(ComputeBackend::Scalar, ComputeBackend::Scalar);
        assert_eq!(ComputeBackend::Simd, ComputeBackend::Simd);
        assert_eq!(ComputeBackend::Gpu, ComputeBackend::Gpu);
        assert_ne!(ComputeBackend::Scalar, ComputeBackend::Gpu);
    }

    #[test]
    fn test_compute_backend_debug() {
        let scalar = format!("{:?}", ComputeBackend::Scalar);
        let simd = format!("{:?}", ComputeBackend::Simd);
        let gpu = format!("{:?}", ComputeBackend::Gpu);
        assert!(scalar.contains("Scalar"));
        assert!(simd.contains("Simd"));
        assert!(gpu.contains("Gpu"));
    }

    #[test]
    fn test_compute_backend_copy_clone() {
        let backend = ComputeBackend::Gpu;
        let copied = backend;
        assert_eq!(backend, copied);
    }

    #[test]
    fn test_thresholds_reasonable() {
        // Verify thresholds are reasonable
        const {
            assert!(SIMD_BATCH_THRESHOLD >= 1);
        }
        const {
            assert!(SIMD_BATCH_THRESHOLD <= 16);
        }
        const {
            assert!(GPU_BATCH_THRESHOLD >= 100);
        }
        const {
            assert!(GPU_BATCH_THRESHOLD <= 10000);
        }
        const {
            assert!(GPU_BATCH_THRESHOLD > SIMD_BATCH_THRESHOLD);
        }
    }
}
