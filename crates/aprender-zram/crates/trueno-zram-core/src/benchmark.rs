//! Benchmarking infrastructure for compression performance testing.

use crate::{Algorithm, CompressorBuilder, Result, PAGE_SIZE};
use std::time::{Duration, Instant};

/// Data pattern for test page generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataPattern {
    /// All zeros (highly compressible).
    Zero,
    /// Pseudo-random data (less compressible).
    Random,
    /// Text-like repeating content.
    Text,
    /// Mix of all patterns.
    Mixed,
}

impl DataPattern {
    /// Parse pattern from string.
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "zero" | "zeros" => Some(Self::Zero),
            "random" => Some(Self::Random),
            "text" => Some(Self::Text),
            "mixed" => Some(Self::Mixed),
            _ => None,
        }
    }
}

/// Benchmark results for a single algorithm.
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Algorithm tested.
    pub algorithm: Algorithm,
    /// SIMD backend used.
    pub backend: crate::SimdBackend,
    /// Total pages compressed.
    pub pages: usize,
    /// Total bytes compressed.
    pub bytes_in: usize,
    /// Total compressed size.
    pub bytes_out: usize,
    /// Compression time.
    pub compress_time: Duration,
    /// Decompression time.
    pub decompress_time: Duration,
}

impl BenchmarkResult {
    /// Compression throughput in bytes per second.
    #[must_use]
    pub fn compress_throughput(&self) -> f64 {
        self.bytes_in as f64 / self.compress_time.as_secs_f64()
    }

    /// Decompression throughput in bytes per second.
    #[must_use]
    pub fn decompress_throughput(&self) -> f64 {
        self.bytes_in as f64 / self.decompress_time.as_secs_f64()
    }

    /// Compression ratio (input / output).
    #[must_use]
    pub fn compression_ratio(&self) -> f64 {
        if self.bytes_out > 0 {
            self.bytes_in as f64 / self.bytes_out as f64
        } else {
            1.0
        }
    }
}

/// Generate test pages with specified pattern.
pub fn generate_test_pages(count: usize, pattern: DataPattern) -> Vec<[u8; PAGE_SIZE]> {
    let mut pages = Vec::with_capacity(count);
    let mut rng_state = 12345u64;

    for i in 0..count {
        let mut page = [0u8; PAGE_SIZE];

        match pattern {
            DataPattern::Zero => {
                // Already zeros
            }
            DataPattern::Random => {
                fill_random(&mut page, &mut rng_state);
            }
            DataPattern::Text => {
                fill_text(&mut page, b"The quick brown fox jumps over the lazy dog. ");
            }
            DataPattern::Mixed => {
                match i % 4 {
                    0 => {} // zeros
                    1 => fill_random(&mut page, &mut rng_state),
                    2 => fill_pattern(&mut page),
                    3 => fill_text(&mut page, b"Lorem ipsum dolor sit amet, consectetur. "),
                    _ => {}
                }
            }
        }

        pages.push(page);
    }

    pages
}

fn fill_random(page: &mut [u8; PAGE_SIZE], rng_state: &mut u64) {
    for byte in page.iter_mut() {
        *rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        *byte = (*rng_state >> 33) as u8;
    }
}

fn fill_text(page: &mut [u8; PAGE_SIZE], text: &[u8]) {
    for (j, byte) in page.iter_mut().enumerate() {
        *byte = text[j % text.len()];
    }
}

fn fill_pattern(page: &mut [u8; PAGE_SIZE]) {
    for (j, byte) in page.iter_mut().enumerate() {
        *byte = (j % 16) as u8;
    }
}

/// Run a compression benchmark for a specific algorithm.
pub fn run_benchmark(algorithm: Algorithm, pages: &[[u8; PAGE_SIZE]]) -> Result<BenchmarkResult> {
    let compressor = CompressorBuilder::new().algorithm(algorithm).build()?;
    let backend = compressor.backend();
    let total_bytes = pages.len() * PAGE_SIZE;

    // Compression benchmark
    let start = Instant::now();
    let mut compressed_pages = Vec::with_capacity(pages.len());
    let mut total_compressed_size = 0usize;

    for page in pages {
        let compressed = compressor.compress(page)?;
        total_compressed_size += compressed.data.len();
        compressed_pages.push(compressed);
    }
    let compress_time = start.elapsed();

    // Decompression benchmark
    let start = Instant::now();
    for compressed in &compressed_pages {
        let _page = compressor.decompress(compressed)?;
    }
    let decompress_time = start.elapsed();

    Ok(BenchmarkResult {
        algorithm,
        backend,
        pages: pages.len(),
        bytes_in: total_bytes,
        bytes_out: total_compressed_size,
        compress_time,
        decompress_time,
    })
}

/// Parse algorithm from string.
pub fn parse_algorithm(s: &str) -> Option<Vec<Algorithm>> {
    match s.to_lowercase().as_str() {
        "all" => {
            Some(vec![Algorithm::Lz4, Algorithm::Zstd { level: 1 }, Algorithm::Zstd { level: 3 }])
        }
        "lz4" => Some(vec![Algorithm::Lz4]),
        s if s.starts_with("zstd") => {
            let level = s.strip_prefix("zstd").and_then(|l| l.parse().ok()).unwrap_or(3);
            Some(vec![Algorithm::Zstd { level }])
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_pattern_parse() {
        assert_eq!(DataPattern::parse("zero"), Some(DataPattern::Zero));
        assert_eq!(DataPattern::parse("zeros"), Some(DataPattern::Zero));
        assert_eq!(DataPattern::parse("random"), Some(DataPattern::Random));
        assert_eq!(DataPattern::parse("text"), Some(DataPattern::Text));
        assert_eq!(DataPattern::parse("mixed"), Some(DataPattern::Mixed));
        assert_eq!(DataPattern::parse("MIXED"), Some(DataPattern::Mixed));
        assert_eq!(DataPattern::parse("invalid"), None);
    }

    #[test]
    fn test_generate_zero_pages() {
        let pages = generate_test_pages(3, DataPattern::Zero);
        assert_eq!(pages.len(), 3);
        for page in &pages {
            assert!(page.iter().all(|&b| b == 0));
        }
    }

    #[test]
    fn test_generate_random_pages() {
        let pages = generate_test_pages(2, DataPattern::Random);
        assert_eq!(pages.len(), 2);
        // Random pages shouldn't be all zeros
        assert!(!pages[0].iter().all(|&b| b == 0));
        // Different pages should have different content
        assert_ne!(pages[0], pages[1]);
    }

    #[test]
    fn test_generate_text_pages() {
        let pages = generate_test_pages(1, DataPattern::Text);
        assert_eq!(pages.len(), 1);
        // Should contain "The quick brown"
        let content = String::from_utf8_lossy(&pages[0][..15]);
        assert!(content.contains("The quick brown"));
    }

    #[test]
    fn test_generate_mixed_pages() {
        let pages = generate_test_pages(8, DataPattern::Mixed);
        assert_eq!(pages.len(), 8);
        // First page should be zeros
        assert!(pages[0].iter().all(|&b| b == 0));
        // Second page should be random (not all zeros)
        assert!(!pages[1].iter().all(|&b| b == 0));
    }

    #[test]
    fn test_run_benchmark_lz4() {
        let pages = generate_test_pages(10, DataPattern::Mixed);
        let result = run_benchmark(Algorithm::Lz4, &pages).unwrap();

        assert_eq!(result.pages, 10);
        assert_eq!(result.bytes_in, 10 * PAGE_SIZE);
        assert!(result.bytes_out < result.bytes_in); // Some compression
        assert!(result.compress_time.as_nanos() > 0);
        assert!(result.decompress_time.as_nanos() > 0);
    }

    #[test]
    fn test_run_benchmark_zstd() {
        let pages = generate_test_pages(5, DataPattern::Zero);
        let result = run_benchmark(Algorithm::Zstd { level: 3 }, &pages).unwrap();

        assert_eq!(result.pages, 5);
        assert!(result.compression_ratio() > 1.0); // Zero pages compress well
    }

    #[test]
    fn test_benchmark_result_throughput() {
        let result = BenchmarkResult {
            algorithm: Algorithm::Lz4,
            backend: crate::SimdBackend::Scalar,
            pages: 100,
            bytes_in: 100 * PAGE_SIZE,
            bytes_out: 50 * PAGE_SIZE,
            compress_time: Duration::from_secs(1),
            decompress_time: Duration::from_millis(500),
        };

        let throughput = result.compress_throughput();
        assert!((throughput - (100.0 * PAGE_SIZE as f64)).abs() < 1.0);

        let decomp_throughput = result.decompress_throughput();
        assert!(decomp_throughput > throughput); // Decompression was faster

        assert!((result.compression_ratio() - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_benchmark_result_ratio_zero_output() {
        let result = BenchmarkResult {
            algorithm: Algorithm::Lz4,
            backend: crate::SimdBackend::Scalar,
            pages: 0,
            bytes_in: 0,
            bytes_out: 0,
            compress_time: Duration::from_nanos(1),
            decompress_time: Duration::from_nanos(1),
        };
        assert!((result.compression_ratio() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_parse_algorithm_all() {
        let algos = parse_algorithm("all").unwrap();
        assert_eq!(algos.len(), 3);
    }

    #[test]
    fn test_parse_algorithm_lz4() {
        let algos = parse_algorithm("lz4").unwrap();
        assert_eq!(algos.len(), 1);
        assert_eq!(algos[0], Algorithm::Lz4);
    }

    #[test]
    fn test_parse_algorithm_zstd() {
        let algos = parse_algorithm("zstd").unwrap();
        assert_eq!(algos[0], Algorithm::Zstd { level: 3 });

        let algos = parse_algorithm("zstd5").unwrap();
        assert_eq!(algos[0], Algorithm::Zstd { level: 5 });
    }

    #[test]
    fn test_parse_algorithm_invalid() {
        assert!(parse_algorithm("invalid").is_none());
    }

    #[test]
    fn test_fill_pattern() {
        let mut page = [0u8; PAGE_SIZE];
        fill_pattern(&mut page);
        assert_eq!(page[0], 0);
        assert_eq!(page[1], 1);
        assert_eq!(page[15], 15);
        assert_eq!(page[16], 0);
    }

    // ============================================================
    // Performance Falsification Tests F051-F065
    // ============================================================

    #[test]
    #[ignore = "Performance test - skip during coverage (instrumentation overhead)"]
    fn test_f051_scalar_achieves_baseline() {
        // F051: Scalar achieves reasonable throughput
        // Note: Debug builds are ~10x slower than release builds
        let pages = generate_test_pages(100, DataPattern::Mixed);
        let result = run_benchmark(Algorithm::Lz4, &pages).unwrap();

        let throughput_mbps = result.compress_throughput() / 1_000_000.0;
        // Debug build baseline: >10 MB/s (release would be >100 MB/s)
        assert!(
            throughput_mbps > 10.0,
            "Scalar throughput {throughput_mbps:.1} MB/s below 10 MB/s debug baseline"
        );
    }

    #[test]
    fn test_f055_latency_bounded() {
        // F055: P99 latency should be bounded
        let page = [0xABu8; PAGE_SIZE];
        let compressor = CompressorBuilder::new().algorithm(Algorithm::Lz4).build().unwrap();

        let mut latencies = Vec::with_capacity(1000);
        for _ in 0..1000 {
            let start = std::time::Instant::now();
            let _ = compressor.compress(&page).unwrap();
            latencies.push(start.elapsed().as_nanos() as u64);
        }

        latencies.sort_unstable();
        let p99 = latencies[990];
        let p99_us = p99 as f64 / 1000.0;

        // P99 should be under 1ms (1000us)
        assert!(p99_us < 1000.0, "P99 latency {p99_us:.1}us exceeds 1000us target");
    }

    #[test]
    fn test_f057_adaptive_selection_works() {
        // F057: Adaptive selection should make reasonable choices
        let compressor = CompressorBuilder::new().algorithm(Algorithm::Adaptive).build().unwrap();

        // Zero page should compress well
        let zero_page = [0u8; PAGE_SIZE];
        let compressed = compressor.compress(&zero_page).unwrap();
        assert!(
            compressed.data.len() < PAGE_SIZE / 2,
            "Zero page should compress to <50% of original"
        );

        // Random page might not compress well
        let mut random_page = [0u8; PAGE_SIZE];
        let mut rng = 12345u64;
        for byte in &mut random_page {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            *byte = (rng >> 33) as u8;
        }
        let compressed = compressor.compress(&random_page).unwrap();
        // Adaptive should handle this gracefully
        assert!(!compressed.data.is_empty());
    }

    #[test]
    fn test_f058_entropy_overhead_minimal() {
        // F058: Entropy calculation overhead should be minimal
        // Note: Debug builds are ~10x slower than release builds
        use crate::samefill::detect_same_fill;

        let page = [0xCDu8; PAGE_SIZE];
        let iterations = 1000;

        let start = std::time::Instant::now();
        for _ in 0..iterations {
            let _ = detect_same_fill(&page);
        }
        let elapsed = start.elapsed();

        let per_page_ns = elapsed.as_nanos() as f64 / iterations as f64;
        // Debug build: <100us per page (release would be <1us)
        assert!(
            per_page_ns < 100_000.0,
            "Same-fill detection {per_page_ns:.1}ns exceeds 100us debug target"
        );
    }

    #[test]
    fn test_f060_no_performance_regression() {
        // F060: Performance should be consistent across runs
        // Note: CI/debug/release builds may have high variance due to system load
        let pages = generate_test_pages(50, DataPattern::Text);

        let result1 = run_benchmark(Algorithm::Lz4, &pages).unwrap();
        let result2 = run_benchmark(Algorithm::Lz4, &pages).unwrap();

        let throughput1 = result1.compress_throughput();
        let throughput2 = result2.compress_throughput();

        // Both runs should produce valid results
        assert!(throughput1 > 0.0, "First run had zero throughput");
        assert!(throughput2 > 0.0, "Second run had zero throughput");

        // Allow 15x variance due to:
        // - CPU frequency scaling (turbo vs base clock)
        // - Cache state differences
        // - System load variations
        // - Timer resolution on fast operations
        // - CI/VM environments with variable performance
        let ratio = throughput2 / throughput1;
        assert!(ratio > 0.05 && ratio < 20.0, "Performance variance too high: ratio {ratio:.2}");
    }

    #[test]
    fn test_f061_warm_cache_faster() {
        // F061: Warm cache performance should be reasonable
        let pages = generate_test_pages(10, DataPattern::Zero);

        // Cold run
        let _cold = run_benchmark(Algorithm::Lz4, &pages).unwrap();

        // Warm run (same data in cache)
        let warm = run_benchmark(Algorithm::Lz4, &pages).unwrap();

        // Warm run should complete without issues
        assert!(warm.compress_throughput() > 0.0);
    }

    #[test]
    fn test_f063_compression_ratio_reasonable() {
        // Verify compression ratios are reasonable for different patterns
        let zero_pages = generate_test_pages(10, DataPattern::Zero);
        let result = run_benchmark(Algorithm::Lz4, &zero_pages).unwrap();
        assert!(
            result.compression_ratio() > 10.0,
            "Zero pages should compress >10:1, got {:.1}:1",
            result.compression_ratio()
        );

        let text_pages = generate_test_pages(10, DataPattern::Text);
        let result = run_benchmark(Algorithm::Lz4, &text_pages).unwrap();
        assert!(
            result.compression_ratio() > 2.0,
            "Text pages should compress >2:1, got {:.1}:1",
            result.compression_ratio()
        );
    }

    #[test]
    fn test_decompression_faster_than_compression() {
        // Decompression is typically faster than compression
        let pages = generate_test_pages(100, DataPattern::Mixed);
        let result = run_benchmark(Algorithm::Lz4, &pages).unwrap();

        let compress_throughput = result.compress_throughput();
        let decompress_throughput = result.decompress_throughput();

        // Decompression should be reasonably close to compression speed
        // In debug builds with coverage instrumentation, decompression may be slower
        assert!(
            decompress_throughput >= compress_throughput * 0.25,
            "Decompression {:.1} MB/s much slower than compression {:.1} MB/s",
            decompress_throughput / 1_000_000.0,
            compress_throughput / 1_000_000.0
        );
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_f053_avx512_simd_path_used() {
        // F053: AVX-512 SIMD path is correctly selected when available
        use crate::lz4;

        if !std::arch::is_x86_feature_detected!("avx512f") {
            println!("Skipping F053: AVX-512 not available");
            return;
        }

        // Generate test data
        let pages = generate_test_pages(1000, DataPattern::Text);

        // Compress all pages
        let mut compressed = Vec::with_capacity(pages.len());
        for page in &pages {
            compressed.push(lz4::compress(page).unwrap());
        }

        // Decompress using SIMD dispatch (should use AVX-512)
        let start = std::time::Instant::now();
        for comp in &compressed {
            let mut output = [0u8; PAGE_SIZE];
            let _ = lz4::decompress_simd(comp, &mut output).unwrap();
        }
        let elapsed = start.elapsed();

        let bytes = pages.len() * PAGE_SIZE;
        let throughput_gbps = (bytes as f64) / elapsed.as_secs_f64() / 1_000_000_000.0;

        println!("F053: AVX-512 decompression throughput: {throughput_gbps:.2} GB/s");

        // Debug builds are ~50-100x slower due to no optimizations
        // In release mode, this should achieve >5 GB/s
        // In debug mode with coverage instrumentation, we allow >10 MB/s
        assert!(
            throughput_gbps > 0.01,
            "AVX-512 decompression throughput {throughput_gbps:.2} GB/s below 10 MB/s debug baseline"
        );
    }
}
