//! Parallel Processing Abstraction (PIXEL-001 v2.1 Phase 9)
//!
//! Provides parallel iteration abstractions that work sequentially by default
//! and can use Rayon when the `parallel` feature is enabled.
//!
//! This module allows the metrics code to be written once and automatically
//! benefit from parallelism when Rayon is available.

use super::config::PerformanceConfig;
use super::heatmap::Rgb;
use super::metrics::{CieDe2000Metric, Lab, SsimMetric};

/// Parallel processing context
#[derive(Debug, Clone, Default)]
pub struct ParallelContext {
    /// Configuration
    config: PerformanceConfig,
}

impl ParallelContext {
    /// Create a new parallel context with default config
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with custom config
    #[must_use]
    pub fn with_config(config: PerformanceConfig) -> Self {
        Self { config }
    }

    /// Check if parallel processing is enabled
    #[must_use]
    pub fn is_parallel(&self) -> bool {
        self.config.parallel
    }

    /// Get thread count (0 = auto)
    #[must_use]
    pub fn thread_count(&self) -> usize {
        if self.config.threads == 0 {
            num_cpus()
        } else {
            self.config.threads
        }
    }
}

/// Get number of CPUs (fallback implementation)
#[must_use]
pub fn num_cpus() -> usize {
    // Simple fallback - in production this would use num_cpus crate or rayon
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

/// Parallel map operation over slices
///
/// When Rayon is available, this uses par_iter().map().
/// Currently falls back to sequential iteration.
pub fn parallel_map<T, U, F>(items: &[T], f: F) -> Vec<U>
where
    T: Sync,
    U: Send,
    F: Fn(&T) -> U + Sync,
{
    // Sequential fallback - replace with rayon::par_iter when available
    items.iter().map(f).collect()
}

/// Parallel reduce operation
///
/// When Rayon is available, this uses par_iter().reduce().
/// Currently falls back to sequential fold.
#[allow(dead_code)]
pub fn parallel_reduce<T, U, M, R>(items: &[T], identity: U, map_fn: M, reduce_fn: R) -> U
where
    T: Sync,
    U: Send + Clone,
    M: Fn(&T) -> U + Sync,
    R: Fn(U, U) -> U + Sync,
{
    // Sequential fallback
    items.iter().map(map_fn).fold(identity, reduce_fn)
}

/// Parallel sum for f32 values
#[allow(dead_code)]
pub fn parallel_sum<T, F>(items: &[T], f: F) -> f32
where
    T: Sync,
    F: Fn(&T) -> f32 + Sync,
{
    parallel_reduce(items, 0.0, f, |a, b| a + b)
}

/// Batch processor for pixel operations
#[derive(Debug)]
pub struct BatchProcessor {
    /// Batch size for processing (used when Rayon is enabled)
    #[allow(dead_code)]
    batch_size: usize,
    /// Parallel context
    ctx: ParallelContext,
}

impl Default for BatchProcessor {
    fn default() -> Self {
        Self {
            batch_size: 1024,
            ctx: ParallelContext::default(),
        }
    }
}

impl BatchProcessor {
    /// Create new batch processor
    #[must_use]
    pub fn new(batch_size: usize) -> Self {
        Self {
            batch_size,
            ..Default::default()
        }
    }

    /// Set parallel context
    #[must_use]
    pub fn with_context(mut self, ctx: ParallelContext) -> Self {
        self.ctx = ctx;
        self
    }

    /// Process pixel pairs computing Delta E values
    #[must_use]
    pub fn compute_delta_e_batch(
        &self,
        reference: &[Rgb],
        generated: &[Rgb],
        metric: &CieDe2000Metric,
    ) -> DeltaEBatchResult {
        if reference.len() != generated.len() {
            return DeltaEBatchResult::default();
        }

        let pairs: Vec<_> = reference.iter().zip(generated.iter()).collect();

        // Compute all delta E values
        let delta_es: Vec<f32> = parallel_map(&pairs, |(r, g)| {
            let lab1 = Lab::from_rgb(r);
            let lab2 = Lab::from_rgb(g);
            metric.delta_e(&lab1, &lab2)
        });

        // Compute statistics
        let sum: f32 = delta_es.iter().sum();
        let max = delta_es.iter().cloned().fold(0.0f32, f32::max);
        let count = delta_es.len();

        let imperceptible = delta_es
            .iter()
            .filter(|&&de| de < metric.jnd_threshold)
            .count();
        let acceptable = delta_es
            .iter()
            .filter(|&&de| de < metric.accept_threshold)
            .count();

        DeltaEBatchResult {
            mean: if count > 0 { sum / count as f32 } else { 0.0 },
            max,
            count,
            imperceptible_count: imperceptible,
            acceptable_count: acceptable,
        }
    }

    /// Process image comparison in batches for memory efficiency
    #[must_use]
    pub fn compute_ssim_batched(
        &self,
        reference: &[Rgb],
        generated: &[Rgb],
        width: u32,
        height: u32,
        metric: &SsimMetric,
    ) -> SsimBatchResult {
        if reference.len() != generated.len() {
            return SsimBatchResult::default();
        }

        // For now, use the standard SSIM calculation
        // Batched processing would split the image into tiles for very large images
        let result = metric.compare(reference, generated, width, height);

        SsimBatchResult {
            score: result.score,
            channel_scores: result.channel_scores,
            batches_processed: 1,
        }
    }
}

/// Result of batch Delta E computation
#[derive(Debug, Clone, Default)]
pub struct DeltaEBatchResult {
    /// Mean delta E
    pub mean: f32,
    /// Maximum delta E
    pub max: f32,
    /// Total pixel count
    pub count: usize,
    /// Count below JND threshold
    pub imperceptible_count: usize,
    /// Count below acceptability threshold
    pub acceptable_count: usize,
}

/// Result of batch SSIM computation
#[derive(Debug, Clone, Default)]
pub struct SsimBatchResult {
    /// Overall SSIM score
    pub score: f32,
    /// Per-channel scores
    pub channel_scores: [f32; 3],
    /// Number of batches processed
    pub batches_processed: usize,
}

/// Downscaler for rapid L1 checks
#[derive(Debug, Clone)]
pub struct Downscaler {
    /// Downscale factor (2 = 50% resolution)
    factor: u32,
}

impl Default for Downscaler {
    fn default() -> Self {
        Self { factor: 2 }
    }
}

impl Downscaler {
    /// Create new downscaler
    #[must_use]
    pub fn new(factor: u32) -> Self {
        Self {
            factor: factor.max(1),
        }
    }

    /// Downscale an image
    #[must_use]
    pub fn downscale(&self, image: &[Rgb], width: u32, height: u32) -> (Vec<Rgb>, u32, u32) {
        let new_width = width / self.factor;
        let new_height = height / self.factor;

        if new_width == 0 || new_height == 0 {
            return (image.to_vec(), width, height);
        }

        let mut result = Vec::with_capacity((new_width * new_height) as usize);

        for y in 0..new_height {
            for x in 0..new_width {
                // Simple box filter (average of factor x factor pixels)
                let mut r_sum = 0u32;
                let mut g_sum = 0u32;
                let mut b_sum = 0u32;
                let mut count = 0u32;

                for dy in 0..self.factor {
                    for dx in 0..self.factor {
                        let src_x = x * self.factor + dx;
                        let src_y = y * self.factor + dy;
                        if src_x < width && src_y < height {
                            let idx = (src_y * width + src_x) as usize;
                            if idx < image.len() {
                                r_sum += image[idx].r as u32;
                                g_sum += image[idx].g as u32;
                                b_sum += image[idx].b as u32;
                                count += 1;
                            }
                        }
                    }
                }

                if count > 0 {
                    result.push(Rgb::new(
                        (r_sum / count) as u8,
                        (g_sum / count) as u8,
                        (b_sum / count) as u8,
                    ));
                }
            }
        }

        (result, new_width, new_height)
    }
}

/// Hash cache for perceptual hashes
#[derive(Debug, Default)]
pub struct HashCache {
    /// Cached hashes (image hash -> perceptual hash)
    cache: std::collections::HashMap<u64, u64>,
}

impl HashCache {
    /// Create new cache
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Get cached hash
    #[must_use]
    pub fn get(&self, image_hash: u64) -> Option<u64> {
        self.cache.get(&image_hash).copied()
    }

    /// Store hash in cache
    pub fn insert(&mut self, image_hash: u64, phash: u64) {
        self.cache.insert(image_hash, phash);
    }

    /// Clear cache
    pub fn clear(&mut self) {
        self.cache.clear();
    }

    /// Get cache size
    #[must_use]
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Check if cache is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    /// Compute simple hash of image data for cache key
    #[must_use]
    pub fn compute_image_hash(image: &[Rgb]) -> u64 {
        // Simple FNV-1a hash for cache lookup
        let mut hash: u64 = 0xcbf29ce484222325;
        for pixel in image {
            hash ^= pixel.r as u64;
            hash = hash.wrapping_mul(0x100000001b3);
            hash ^= pixel.g as u64;
            hash = hash.wrapping_mul(0x100000001b3);
            hash ^= pixel.b as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        hash
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn test_image(size: usize, value: u8) -> Vec<Rgb> {
        vec![Rgb::new(value, value, value); size]
    }

    // =========================================================================
    // Parallel Context Tests (H0-PAR-XX)
    // =========================================================================

    #[test]
    fn h0_par_01_default_context() {
        let ctx = ParallelContext::new();
        assert!(ctx.is_parallel());
        assert!(ctx.thread_count() >= 1);
    }

    #[test]
    fn h0_par_02_custom_threads() {
        let config = PerformanceConfig {
            threads: 4,
            ..Default::default()
        };
        let ctx = ParallelContext::with_config(config);
        assert_eq!(ctx.thread_count(), 4);
    }

    #[test]
    fn h0_par_03_parallel_map() {
        let items = vec![1, 2, 3, 4, 5];
        let result: Vec<i32> = parallel_map(&items, |x| x * 2);
        assert_eq!(result, vec![2, 4, 6, 8, 10]);
    }

    #[test]
    fn h0_par_04_parallel_sum() {
        let items = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let result = parallel_sum(&items, |&x| x);
        assert!((result - 15.0).abs() < f32::EPSILON);
    }

    #[test]
    fn h0_par_05_parallel_reduce() {
        let items = vec![1, 2, 3, 4];
        let result = parallel_reduce(&items, 0, |&x| x, |a, b| a + b);
        assert_eq!(result, 10);
    }

    // =========================================================================
    // Batch Processor Tests (H0-BATCH-XX)
    // =========================================================================

    #[test]
    fn h0_batch_01_delta_e_same() {
        let img = test_image(100, 128);
        let processor = BatchProcessor::default();
        let metric = CieDe2000Metric::default();
        let result = processor.compute_delta_e_batch(&img, &img, &metric);
        assert!(result.mean < f32::EPSILON);
        assert_eq!(result.count, 100);
    }

    #[test]
    fn h0_batch_02_delta_e_different() {
        let img1 = test_image(100, 100);
        let img2 = test_image(100, 150);
        let processor = BatchProcessor::default();
        let metric = CieDe2000Metric::default();
        let result = processor.compute_delta_e_batch(&img1, &img2, &metric);
        assert!(result.mean > 0.0);
        assert_eq!(result.count, 100);
    }

    #[test]
    fn h0_batch_03_ssim_same() {
        let img = test_image(100, 128);
        let processor = BatchProcessor::default();
        let metric = SsimMetric::default();
        let result = processor.compute_ssim_batched(&img, &img, 10, 10, &metric);
        assert!(result.score >= 0.99);
    }

    // =========================================================================
    // Downscaler Tests (H0-DOWN-XX)
    // =========================================================================

    #[test]
    fn h0_down_01_downscale_2x() {
        let img = test_image(100, 128); // 10x10
        let downscaler = Downscaler::new(2);
        let (result, w, h) = downscaler.downscale(&img, 10, 10);
        assert_eq!(w, 5);
        assert_eq!(h, 5);
        assert_eq!(result.len(), 25);
    }

    #[test]
    fn h0_down_02_downscale_preserves_color() {
        let img = test_image(16, 200); // 4x4
        let downscaler = Downscaler::new(2);
        let (result, _, _) = downscaler.downscale(&img, 4, 4);
        // Average of 200 should still be 200
        assert_eq!(result[0].r, 200);
    }

    #[test]
    fn h0_down_03_downscale_factor_1() {
        let img = test_image(25, 100);
        let downscaler = Downscaler::new(1);
        let (result, w, h) = downscaler.downscale(&img, 5, 5);
        assert_eq!(w, 5);
        assert_eq!(h, 5);
        assert_eq!(result.len(), 25);
    }

    // =========================================================================
    // Hash Cache Tests (H0-CACHE-XX)
    // =========================================================================

    #[test]
    fn h0_cache_01_insert_get() {
        let mut cache = HashCache::new();
        cache.insert(12345, 67890);
        assert_eq!(cache.get(12345), Some(67890));
        assert_eq!(cache.get(99999), None);
    }

    #[test]
    fn h0_cache_02_clear() {
        let mut cache = HashCache::new();
        cache.insert(1, 1);
        cache.insert(2, 2);
        assert_eq!(cache.len(), 2);
        cache.clear();
        assert!(cache.is_empty());
    }

    #[test]
    fn h0_cache_03_image_hash() {
        let img1 = test_image(100, 128);
        let img2 = test_image(100, 128);
        let img3 = test_image(100, 129);

        let hash1 = HashCache::compute_image_hash(&img1);
        let hash2 = HashCache::compute_image_hash(&img2);
        let hash3 = HashCache::compute_image_hash(&img3);

        assert_eq!(hash1, hash2); // Same images
        assert_ne!(hash1, hash3); // Different images
    }

    #[test]
    fn h0_cache_04_empty() {
        let cache = HashCache::new();
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }

    // =========================================================================
    // Num CPUs Test
    // =========================================================================

    #[test]
    fn h0_par_06_num_cpus() {
        let cpus = num_cpus();
        assert!(cpus >= 1);
    }
}
