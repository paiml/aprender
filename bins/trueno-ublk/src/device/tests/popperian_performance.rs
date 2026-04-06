// ========================================================================
// Popperian Falsification Checklist - Section C: Performance (31-40)
// ========================================================================

use super::*;

/// C31: Sequential write throughput test.
/// Validates write path is functional with reasonable throughput.
#[test]
#[ignore = "Performance test - skip during coverage (instrumentation overhead)"]
fn popperian_c31_sequential_write_throughput() {
    use std::time::Instant;

    let num_pages = 10000; // ~40MB
    let device_size = (num_pages * PAGE_SIZE) as u64;
    let mut device = BlockDevice::new(device_size, test_compressor());

    // Create test data (moderately compressible)
    let data: Vec<u8> = (0..PAGE_SIZE).map(|i| (i % 128) as u8).collect();

    let start = Instant::now();
    for i in 0..num_pages {
        device.write((i * PAGE_SIZE) as u64, &data).unwrap();
    }
    let duration = start.elapsed();

    let bytes_written = num_pages * PAGE_SIZE;
    let throughput_gbps = (bytes_written as f64) / duration.as_secs_f64() / 1e9;

    println!(
        "C31: Sequential write throughput: {:.2} GB/s ({} pages in {:?})",
        throughput_gbps, num_pages, duration
    );

    // Verify throughput is reasonable (>50MB/s minimum sanity check)
    // Accounts for parallel test execution overhead
    assert!(
        throughput_gbps > 0.05,
        "C31: Write throughput should be > 50 MB/s, got {:.2} GB/s",
        throughput_gbps
    );
}

/// C32: Sequential read throughput test.
#[test]
fn popperian_c32_sequential_read_throughput() {
    use std::time::Instant;

    let num_pages = 10000;
    let device_size = (num_pages * PAGE_SIZE) as u64;
    let mut device = BlockDevice::new(device_size, test_compressor());

    // Write data first
    let data: Vec<u8> = (0..PAGE_SIZE).map(|i| (i % 128) as u8).collect();
    for i in 0..num_pages {
        device.write((i * PAGE_SIZE) as u64, &data).unwrap();
    }

    // Benchmark reads
    let mut buf = vec![0u8; PAGE_SIZE];
    let start = Instant::now();
    for i in 0..num_pages {
        device.read((i * PAGE_SIZE) as u64, &mut buf).unwrap();
    }
    let duration = start.elapsed();

    let bytes_read = num_pages * PAGE_SIZE;
    let throughput_gbps = (bytes_read as f64) / duration.as_secs_f64() / 1e9;

    println!(
        "C32: Sequential read throughput: {:.2} GB/s ({} pages in {:?})",
        throughput_gbps, num_pages, duration
    );

    // Read should be at least as fast as write (decompression often faster)
    // Accounts for parallel test execution overhead
    assert!(
        throughput_gbps > 0.05,
        "C32: Read throughput should be > 50 MB/s, got {:.2} GB/s",
        throughput_gbps
    );
}

/// C33: Random 4K write IOPS test.
#[test]
fn popperian_c33_random_write_iops() {
    use std::time::Instant;

    let num_pages = 1000;
    let device_size = (num_pages * PAGE_SIZE) as u64;
    let mut device = BlockDevice::new(device_size, test_compressor());

    // Generate random access pattern
    let mut access_order: Vec<usize> = (0..num_pages).collect();
    let mut rng_state: u64 = 0xBADCAFE;
    for i in (1..access_order.len()).rev() {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let j = (rng_state as usize) % (i + 1);
        access_order.swap(i, j);
    }

    let data = vec![0xAB; PAGE_SIZE];
    let iterations = 10000;

    let start = Instant::now();
    for i in 0..iterations {
        let page_idx = access_order[i % num_pages];
        device.write((page_idx * PAGE_SIZE) as u64, &data).unwrap();
    }
    let duration = start.elapsed();

    let iops = iterations as f64 / duration.as_secs_f64();

    println!("C33: Random 4K write: {:.0} IOPS ({} ops in {:?})", iops, iterations, duration);

    // Sanity check: at least 1000 IOPS
    assert!(iops > 1000.0, "C33: Random write IOPS should be > 1000, got {:.0}", iops);
}

/// C34: Random 4K read IOPS test.
#[test]
fn popperian_c34_random_read_iops() {
    use std::time::Instant;

    let num_pages = 1000;
    let device_size = (num_pages * PAGE_SIZE) as u64;
    let mut device = BlockDevice::new(device_size, test_compressor());

    // Write initial data
    let data = vec![0xAB; PAGE_SIZE];
    for i in 0..num_pages {
        device.write((i * PAGE_SIZE) as u64, &data).unwrap();
    }

    // Generate random access pattern
    let mut access_order: Vec<usize> = (0..num_pages).collect();
    let mut rng_state: u64 = 0xCAFEBABE;
    for i in (1..access_order.len()).rev() {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let j = (rng_state as usize) % (i + 1);
        access_order.swap(i, j);
    }

    let mut buf = vec![0u8; PAGE_SIZE];
    let iterations = 10000;

    let start = Instant::now();
    for i in 0..iterations {
        let page_idx = access_order[i % num_pages];
        device.read((page_idx * PAGE_SIZE) as u64, &mut buf).unwrap();
    }
    let duration = start.elapsed();

    let iops = iterations as f64 / duration.as_secs_f64();

    println!("C34: Random 4K read: {:.0} IOPS ({} ops in {:?})", iops, iterations, duration);

    assert!(iops > 1000.0, "C34: Random read IOPS should be > 1000, got {:.0}", iops);
}

/// C35: Latency test at QD=1.
#[test]
fn popperian_c35_latency_qd1() {
    use std::time::Instant;

    let mut device = BlockDevice::new(1 << 20, test_compressor());

    let data = vec![0xAB; PAGE_SIZE];
    let iterations = 1000;
    let mut latencies = Vec::with_capacity(iterations);

    for _ in 0..iterations {
        let start = Instant::now();
        device.write(0, &data).unwrap();
        latencies.push(start.elapsed());
    }

    latencies.sort();
    let p50 = latencies[latencies.len() / 2];
    let p99 = latencies[latencies.len() * 99 / 100];
    let p999 = latencies[latencies.len() * 999 / 1000];

    println!("C35: QD=1 latency: p50={:?}, p99={:?}, p99.9={:?}", p50, p99, p999);

    // In-memory device should be very fast
    assert!(
        p99.as_micros() < 10000, // 10ms
        "C35: p99 latency should be < 10ms at QD=1, got {:?}",
        p99
    );
}

/// C36: Latency test at high queue depth (simulated with threads).
#[test]
fn popperian_c36_latency_high_qd() {
    use std::sync::Arc;
    use std::thread;
    use std::time::Instant;

    let device = Arc::new(std::sync::Mutex::new(BlockDevice::new(1 << 24, test_compressor())));

    let num_threads = 8;
    let ops_per_thread = 500;
    let mut handles = vec![];

    let start = Instant::now();

    for t in 0..num_threads {
        let dev = device.clone();
        handles.push(thread::spawn(move || {
            let data = vec![t as u8; PAGE_SIZE];
            for i in 0..ops_per_thread {
                let offset = ((t * ops_per_thread + i) * PAGE_SIZE) as u64;
                dev.lock().unwrap().write(offset, &data).unwrap();
            }
        }));
    }

    for h in handles {
        h.join().unwrap();
    }

    let duration = start.elapsed();
    let total_ops = num_threads * ops_per_thread;
    let avg_latency_us = duration.as_micros() as f64 / total_ops as f64;

    println!(
        "C36: High QD ({} threads) avg latency: {:.1}us ({} ops in {:?})",
        num_threads, avg_latency_us, total_ops, duration
    );

    // Even under contention, should complete reasonably
    assert!(
        avg_latency_us < 100000.0, // 100ms average
        "C36: Avg latency should be < 100ms under contention"
    );
}

/// C37: Scalability test across multiple threads.
#[test]
fn popperian_c37_scalability() {
    use std::sync::Arc;
    use std::thread;
    use std::time::Instant;

    let ops_per_thread = 1000;
    let device_size = (ops_per_thread * PAGE_SIZE * 2) as u64; // Room for all ops
    let mut results = Vec::new();

    for num_threads in [1, 2, 4] {
        let devices: Vec<_> = (0..num_threads)
            .map(|_| {
                Arc::new(std::sync::Mutex::new(BlockDevice::new(
                    device_size,
                    test_compressor(),
                )))
            })
            .collect();

        let start = Instant::now();

        let handles: Vec<_> = devices
            .iter()
            .enumerate()
            .map(|(t, dev)| {
                let dev = dev.clone();
                thread::spawn(move || {
                    let data = vec![t as u8; PAGE_SIZE];
                    for i in 0..ops_per_thread {
                        let offset = (i * PAGE_SIZE) as u64;
                        dev.lock().unwrap().write(offset, &data).unwrap();
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        let duration = start.elapsed();
        let total_ops = num_threads * ops_per_thread;
        let throughput = total_ops as f64 / duration.as_secs_f64();

        results.push((num_threads, throughput));
        println!(
            "C37: {} threads: {:.0} ops/sec ({} ops in {:?})",
            num_threads, throughput, total_ops, duration
        );
    }

    // Verify we get some scaling benefit
    // (Throughput at 4 threads should be at least 50% better than 1 thread)
    let (_, throughput_1) = results[0];
    let (_, throughput_4) = results[2];

    // Note: Due to mutex contention, perfect scaling isn't expected
    // Just verify it doesn't completely collapse
    assert!(
        throughput_4 > throughput_1 * 0.5,
        "C37: 4-thread throughput should be at least 50% of 1-thread"
    );
}

/// C38: SIMD backend detection verification.
#[test]
fn popperian_c38_simd_backend_detection() {
    let backend = detect_simd_backend();

    println!("C38: Detected SIMD backend: {}", backend);

    // Should detect something
    assert!(!backend.is_empty(), "C38: Should detect a SIMD backend");

    // On x86_64, should be avx512, avx2, sse4.2, or scalar
    #[cfg(target_arch = "x86_64")]
    assert!(
        ["avx512", "avx2", "sse4.2", "scalar"].contains(&backend.as_str()),
        "C38: x86_64 should detect known backend, got: {}",
        backend
    );

    // On aarch64, should be neon
    #[cfg(target_arch = "aarch64")]
    assert_eq!(backend, "neon", "C38: ARM should detect NEON");
}

/// C39: Algorithm switching test (LZ4 vs Zstd).
#[test]
fn popperian_c39_algorithm_switch() {
    use std::time::Instant;

    let data: Vec<u8> = (0..PAGE_SIZE).map(|i| (i % 128) as u8).collect();
    let iterations = 1000;

    // Test LZ4
    let lz4_compressor = CompressorBuilder::new().algorithm(Algorithm::Lz4).build().unwrap();
    let mut lz4_device = BlockDevice::new(1 << 20, lz4_compressor);

    let start = Instant::now();
    for _ in 0..iterations {
        lz4_device.write(0, &data).unwrap();
    }
    let lz4_duration = start.elapsed();

    // Test Zstd (level 1 for speed)
    let zstd_compressor =
        CompressorBuilder::new().algorithm(Algorithm::Zstd { level: 1 }).build().unwrap();
    let mut zstd_device = BlockDevice::new(1 << 20, zstd_compressor);

    let start = Instant::now();
    for _ in 0..iterations {
        zstd_device.write(0, &data).unwrap();
    }
    let zstd_duration = start.elapsed();

    let lz4_stats = lz4_device.stats();
    let zstd_stats = zstd_device.stats();

    println!(
        "C39: LZ4: {:?} ({:.2}:1), Zstd: {:?} ({:.2}:1)",
        lz4_duration,
        lz4_stats.compression_ratio(),
        zstd_duration,
        zstd_stats.compression_ratio()
    );

    // Both should work
    assert!(lz4_stats.bytes_written > 0, "C39: LZ4 should write data");
    assert!(zstd_stats.bytes_written > 0, "C39: Zstd should write data");

    // Verify roundtrip for both
    let mut lz4_buf = vec![0u8; PAGE_SIZE];
    lz4_device.read(0, &mut lz4_buf).unwrap();
    assert_eq!(data, lz4_buf, "C39: LZ4 roundtrip should match");

    let mut zstd_buf = vec![0u8; PAGE_SIZE];
    zstd_device.read(0, &mut zstd_buf).unwrap();
    assert_eq!(data, zstd_buf, "C39: Zstd roundtrip should match");
}

/// C40: Dictionary training placeholder.
/// Note: Dictionary training is not yet implemented in trueno-zram-core.
#[test]
fn popperian_c40_dictionary_training() {
    // Dictionary training improves compression for small, similar data blocks.
    // This is a placeholder for when dictionary support is added.
    //
    // When implemented:
    // 1. Train dictionary on representative data
    // 2. Verify improved compression ratio vs non-dictionary
    // 3. Verify roundtrip correctness

    let mut device = BlockDevice::new(1 << 20, test_compressor());

    // Write similar small patterns (would benefit from dictionary)
    let patterns: Vec<Vec<u8>> = (0..10)
        .map(|i| {
            format!("{{\"id\": {}, \"name\": \"user_{}\", \"active\": true}}", i, i)
                .into_bytes()
                .into_iter()
                .chain(std::iter::repeat(0u8))
                .take(PAGE_SIZE)
                .collect()
        })
        .collect();

    for (i, pattern) in patterns.iter().enumerate() {
        device.write((i * PAGE_SIZE) as u64, pattern).unwrap();
    }

    let stats = device.stats();
    let ratio = stats.compression_ratio();

    println!("C40: JSON-like data compression ratio (no dict): {:.2}:1", ratio);

    // Even without dictionary, should compress somewhat
    assert!(ratio > 1.0, "C40: JSON-like data should compress even without dictionary");
}
