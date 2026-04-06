//! Integration tests for trueno-ublk BlockDevice
//!
//! These tests validate compression roundtrips, throughput, and compression ratios.

use std::time::Instant;
use trueno_zram_core::{Algorithm, CompressorBuilder, PageCompressor, PAGE_SIZE};

// Helper function to create compressor for tests
fn test_compressor(algo: Algorithm) -> Box<dyn PageCompressor> {
    CompressorBuilder::new().algorithm(algo).build().unwrap()
}

/// Test module for roundtrip verification
mod roundtrip {
    use super::*;

    #[test]
    fn test_lz4_roundtrip_all_patterns() {
        test_all_patterns_with_algorithm(Algorithm::Lz4);
    }

    #[test]
    fn test_zstd_roundtrip_all_patterns() {
        test_all_patterns_with_algorithm(Algorithm::Zstd { level: 3 });
    }

    fn test_all_patterns_with_algorithm(algo: Algorithm) {
        let compressor = test_compressor(algo);

        // Test patterns
        let patterns: Vec<(&str, Vec<u8>)> = vec![
            ("zeros", vec![0u8; PAGE_SIZE]),
            ("ones", vec![0xFF; PAGE_SIZE]),
            ("alternating", (0..PAGE_SIZE).map(|i| if i % 2 == 0 { 0xAA } else { 0x55 }).collect()),
            ("sequential", (0..PAGE_SIZE).map(|i| (i % 256) as u8).collect()),
            ("short_pattern", (0..PAGE_SIZE).map(|i| (i % 16) as u8).collect()),
            ("pseudo_random", (0..PAGE_SIZE).map(|i| ((i * 17 + 31) % 256) as u8).collect()),
            (
                "text_like",
                "The quick brown fox jumps over the lazy dog. ".repeat(100).into_bytes()
                    [..PAGE_SIZE]
                    .to_vec(),
            ),
        ];

        for (name, data) in patterns {
            // Convert to fixed-size array
            let page: &[u8; PAGE_SIZE] = data.as_slice().try_into().unwrap();

            // Compress
            let compressed =
                compressor.compress(page).expect(&format!("Compress failed for {}", name));

            // Decompress
            let decompressed = compressor
                .decompress(&compressed)
                .expect(&format!("Decompress failed for {}", name));

            // Verify
            assert_eq!(data, decompressed.as_slice(), "Roundtrip failed for pattern: {}", name);
        }
    }
}

/// Test module for compression ratio validation
mod compression_ratio {
    use super::*;

    #[test]
    fn test_highly_compressible_data_lz4() {
        validate_compression_ratio(Algorithm::Lz4, 10.0); // Expect at least 10:1
    }

    #[test]
    fn test_highly_compressible_data_zstd() {
        validate_compression_ratio(Algorithm::Zstd { level: 3 }, 10.0); // Expect at least 10:1
    }

    fn validate_compression_ratio(algo: Algorithm, min_ratio: f64) {
        let compressor = test_compressor(algo);

        // Highly compressible data (all same byte)
        let data = vec![0xAA; PAGE_SIZE];
        let page: &[u8; PAGE_SIZE] = data.as_slice().try_into().unwrap();

        let compressed = compressor.compress(page).unwrap();
        let ratio = PAGE_SIZE as f64 / compressed.data.len() as f64;

        assert!(
            ratio >= min_ratio,
            "Compression ratio {} is less than expected {} for {:?}",
            ratio,
            min_ratio,
            algo
        );

        println!(
            "Compression ratio for {:?} on repetitive data: {:.2}:1 (compressed {} -> {} bytes)",
            algo,
            ratio,
            PAGE_SIZE,
            compressed.data.len()
        );
    }

    #[test]
    fn test_typical_workload_compression() {
        let compressor = test_compressor(Algorithm::Lz4);

        // Simulate typical workload: mix of compressible data
        let patterns: Vec<Vec<u8>> = vec![
            // Sparse data with some values
            {
                let mut data = vec![0u8; PAGE_SIZE];
                for i in (0..PAGE_SIZE).step_by(64) {
                    data[i] = 0xAB;
                }
                data
            },
            // Struct-like data (repeating patterns)
            (0..PAGE_SIZE).map(|i| ((i / 8) % 256) as u8).collect(),
            // Text data
            "Hello, world! This is some text data that should compress well. "
                .repeat(70)
                .into_bytes()[..PAGE_SIZE]
                .to_vec(),
        ];

        let mut total_orig = 0u64;
        let mut total_compr = 0u64;

        for data in patterns {
            let page: &[u8; PAGE_SIZE] = data.as_slice().try_into().unwrap();
            let compressed = compressor.compress(page).unwrap();
            total_orig += PAGE_SIZE as u64;
            total_compr += compressed.data.len() as u64;
        }

        let overall_ratio = total_orig as f64 / total_compr as f64;

        assert!(
            overall_ratio >= 2.0,
            "Overall compression ratio {} is less than 2:1 for typical workload",
            overall_ratio
        );

        println!("Overall compression ratio for typical workload: {:.2}:1", overall_ratio);
    }
}

/// Test module for throughput benchmarking
mod throughput {
    use super::*;

    #[test]
    fn test_compression_throughput_lz4() {
        benchmark_throughput(Algorithm::Lz4, "LZ4");
    }

    #[test]
    fn test_compression_throughput_zstd1() {
        benchmark_throughput(Algorithm::Zstd { level: 1 }, "Zstd-1");
    }

    fn benchmark_throughput(algo: Algorithm, name: &str) {
        let compressor = test_compressor(algo);

        // Test data: moderately compressible
        let data: Vec<u8> = (0..PAGE_SIZE).map(|i| ((i * 7 + 13) % 256) as u8).collect();
        let page: &[u8; PAGE_SIZE] = data.as_slice().try_into().unwrap();

        // Warmup
        for _ in 0..100 {
            let compressed = compressor.compress(page).unwrap();
            let _ = compressor.decompress(&compressed).unwrap();
        }

        // Benchmark compression
        let iterations = 10000;
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = compressor.compress(page).unwrap();
        }
        let compress_duration = start.elapsed();

        // Benchmark decompression
        let compressed = compressor.compress(page).unwrap();
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = compressor.decompress(&compressed).unwrap();
        }
        let decompress_duration = start.elapsed();

        let bytes_processed = iterations as u64 * PAGE_SIZE as u64;
        let compress_throughput = bytes_processed as f64 / compress_duration.as_secs_f64() / 1e9;
        let decompress_throughput =
            bytes_processed as f64 / decompress_duration.as_secs_f64() / 1e9;

        println!(
            "{} throughput: compress={:.2} GB/s, decompress={:.2} GB/s",
            name, compress_throughput, decompress_throughput
        );

        // We expect at least reasonable throughput (this is a sanity check, actual performance varies)
        assert!(
            compress_throughput > 0.1,
            "{} compression throughput too low: {:.2} GB/s",
            name,
            compress_throughput
        );
        assert!(
            decompress_throughput > 0.1,
            "{} decompression throughput too low: {:.2} GB/s",
            name,
            decompress_throughput
        );
    }

    #[test]
    fn test_batch_compression_throughput() {
        let compressor = test_compressor(Algorithm::Lz4);

        // Create batch of pages with varying data
        let batch_size = 1000;
        let pages: Vec<Vec<u8>> = (0..batch_size)
            .map(|i| (0..PAGE_SIZE).map(|j| ((i + j * 7) % 256) as u8).collect())
            .collect();

        // Benchmark batch compression
        let start = Instant::now();
        let mut compressed_pages = Vec::with_capacity(batch_size);
        for page_data in &pages {
            let page: &[u8; PAGE_SIZE] = page_data.as_slice().try_into().unwrap();
            compressed_pages.push(compressor.compress(page).unwrap());
        }
        let compress_duration = start.elapsed();

        // Benchmark batch decompression
        let start = Instant::now();
        let mut decompressed_pages = Vec::with_capacity(batch_size);
        for compressed in &compressed_pages {
            decompressed_pages.push(compressor.decompress(compressed).unwrap());
        }
        let decompress_duration = start.elapsed();

        // Verify correctness
        for (original, decompressed) in pages.iter().zip(decompressed_pages.iter()) {
            assert_eq!(original.as_slice(), decompressed.as_slice());
        }

        let bytes_processed = batch_size as u64 * PAGE_SIZE as u64;
        let compress_throughput = bytes_processed as f64 / compress_duration.as_secs_f64() / 1e9;
        let decompress_throughput =
            bytes_processed as f64 / decompress_duration.as_secs_f64() / 1e9;

        println!(
            "Batch ({} pages) throughput: compress={:.2} GB/s, decompress={:.2} GB/s",
            batch_size, compress_throughput, decompress_throughput
        );

        // Calculate compression ratio
        let total_compressed: usize = compressed_pages.iter().map(|c| c.data.len()).sum();
        let ratio = bytes_processed as f64 / total_compressed as f64;
        println!("Batch compression ratio: {:.2}:1", ratio);
    }
}

/// Test for edge cases
mod edge_cases {
    use super::*;

    #[test]
    fn test_incompressible_random_data() {
        let compressor = test_compressor(Algorithm::Lz4);

        // Pseudo-random data that looks like real random (high entropy)
        // Using a simple PRNG pattern
        let mut data = vec![0u8; PAGE_SIZE];
        let mut state: u64 = 0xDEADBEEF;
        for byte in data.iter_mut() {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            *byte = (state >> 33) as u8;
        }

        let page: &[u8; PAGE_SIZE] = data.as_slice().try_into().unwrap();
        let compressed = compressor.compress(page).unwrap();
        let decompressed = compressor.decompress(&compressed).unwrap();

        assert_eq!(data, decompressed.as_slice(), "Random data roundtrip failed");

        // High entropy data may actually expand
        println!(
            "High-entropy data: {} -> {} bytes (ratio: {:.2}:1)",
            PAGE_SIZE,
            compressed.data.len(),
            PAGE_SIZE as f64 / compressed.data.len() as f64
        );
    }

    #[test]
    fn test_single_byte_patterns() {
        let compressor = test_compressor(Algorithm::Lz4);

        for byte_val in [0x00, 0x01, 0x7F, 0x80, 0xFE, 0xFF] {
            let data = vec![byte_val; PAGE_SIZE];
            let page: &[u8; PAGE_SIZE] = data.as_slice().try_into().unwrap();

            let compressed = compressor.compress(page).unwrap();
            let decompressed = compressor.decompress(&compressed).unwrap();

            assert_eq!(data, decompressed.as_slice(), "Single byte {} roundtrip failed", byte_val);
        }
    }
}

/// DT-007: Test mlock integration for swap deadlock prevention
mod mlock_integration {
    use trueno_ublk::{is_memory_locked, lock_daemon_memory, MlockStatus};

    #[test]
    fn test_dt007_mlock_available() {
        // Verify duende-mlock API is accessible through trueno-ublk
        let locked = is_memory_locked();
        // Just verify the function returns a boolean without panicking
        println!("DT-007: is_memory_locked() = {}", locked);
    }

    #[test]
    fn test_dt007_lock_daemon_memory() {
        // Test the mlock integration
        match lock_daemon_memory() {
            Ok(MlockStatus::Locked { bytes_locked }) => {
                println!(
                    "DT-007: Memory locked ({} bytes) - swap deadlock prevention active",
                    bytes_locked
                );
                assert!(bytes_locked > 0, "Expected some bytes to be locked");

                // Verify via /proc/self/status
                if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
                    for line in status.lines() {
                        if line.starts_with("VmLck:") {
                            println!("DT-007: Kernel confirms: {}", line);
                        }
                    }
                }
            }
            Ok(MlockStatus::Failed { errno }) => {
                // This is expected in unprivileged environments
                println!(
                    "DT-007: mlock() failed (errno={}) - need CAP_IPC_LOCK or memlock ulimit",
                    errno
                );
                // Common errno values:
                // EPERM (1) = Permission denied (need CAP_IPC_LOCK)
                // ENOMEM (12) = Cannot allocate memory (ulimit too low)
                assert!(errno == 1 || errno == 12, "Unexpected errno: {}", errno);
            }
            Ok(MlockStatus::Unsupported) => {
                println!("DT-007: mlock() not supported on this platform");
            }
            Err(e) => {
                panic!("DT-007: Unexpected error from lock_daemon_memory: {:?}", e);
            }
        }
    }
}
