//! Tests for BatchedPageStore batched compression pipeline.

use crate::daemon::batched::{is_zero_page, Backend};
use crate::daemon::page_store::SECTORS_PER_PAGE;
use crate::daemon::{spawn_flush_thread, BatchConfig, BatchedPageStore};
use std::sync::Arc;
use std::time::Duration;
use trueno_zram_core::{Algorithm, PAGE_SIZE};

fn create_batched_test_store() -> BatchedPageStore {
    BatchedPageStore::with_config(
        Algorithm::Lz4,
        BatchConfig {
            batch_threshold: 100, // Lower threshold for testing
            flush_timeout: Duration::from_millis(10),
            gpu_batch_size: 1000,
        },
    )
}

/// G.101: Batch Threshold Test
/// Write 99 pages, verify NOT compressed yet (in pending)
/// Write 1 more page, verify batch ready for flush
/// PERF-002: store() is now non-blocking, so we call flush_batch() explicitly
#[test]
fn test_g101_batch_threshold() {
    let store = create_batched_test_store();

    // Write 99 pages (below threshold of 100)
    for i in 0..99 {
        let mut data = [0u8; PAGE_SIZE];
        data[0] = (i + 1) as u8; // Non-zero to avoid zero-page fast path
        store.store((i * SECTORS_PER_PAGE as usize) as u64, &data).unwrap();
    }

    // Verify pages are still in pending
    let stats = store.stats();
    assert_eq!(stats.pending_pages, 99, "Pages should be pending");
    assert_eq!(stats.batch_flushes, 0, "No flush should have occurred yet");

    // Write one more page to reach threshold
    let mut data = [0u8; PAGE_SIZE];
    data[0] = 100;
    store.store((99 * SECTORS_PER_PAGE as usize) as u64, &data).unwrap();

    // Verify threshold reached (should_flush returns true)
    assert!(store.should_flush(), "should_flush() should return true at threshold");

    // PERF-002: Explicitly flush (in production, background thread does this)
    store.flush_batch().unwrap();

    // Verify batch was flushed
    let stats = store.stats();
    assert_eq!(stats.pending_pages, 0, "Pending should be empty after flush");
    assert_eq!(stats.batch_flushes, 1, "One flush should have occurred");
}

/// G.102: Flush Timer Test
/// Write 50 pages, wait > flush timeout, verify pages are now compressed
#[test]
fn test_g102_flush_timer() {
    let store = Arc::new(BatchedPageStore::with_config(
        Algorithm::Lz4,
        BatchConfig {
            batch_threshold: 1000, // High threshold so timer triggers first
            flush_timeout: Duration::from_millis(5),
            gpu_batch_size: 1000,
        },
    ));

    // Write 50 pages
    for i in 0..50 {
        let mut data = [0u8; PAGE_SIZE];
        data[0] = (i + 1) as u8;
        store.store((i * SECTORS_PER_PAGE as usize) as u64, &data).unwrap();
    }

    // Verify pages are pending
    assert_eq!(store.stats().pending_pages, 50);

    // Wait for flush timeout
    std::thread::sleep(Duration::from_millis(10));

    // Check should_flush returns true
    assert!(store.should_flush(), "Should indicate flush needed");

    // Manually trigger flush (would be done by background thread)
    store.flush_batch().unwrap();

    // Verify pages are now compressed
    let stats = store.stats();
    assert_eq!(stats.pending_pages, 0, "Pending should be empty after timeout flush");
    assert_eq!(stats.batch_flushes, 1, "Flush should have occurred");
}

/// G.103: Read-Before-Flush Test
/// Write 50 pages (not yet flushed), read same pages back
/// Verify correct data returned from pending buffer
#[test]
fn test_g103_read_before_flush() {
    let store = create_batched_test_store();

    // Write pages with unique patterns
    let mut expected_data = Vec::new();
    for i in 0..50 {
        let mut data = [0u8; PAGE_SIZE];
        // Fill with unique pattern
        for j in 0..PAGE_SIZE {
            data[j] = ((i * 17 + j * 31) % 256) as u8;
        }
        expected_data.push(data);
        store.store((i * SECTORS_PER_PAGE as usize) as u64, &data).unwrap();
    }

    // Verify pages are in pending (not flushed)
    assert!(store.stats().pending_pages >= 50 || store.stats().batch_flushes == 0);

    // Read back and verify from pending buffer
    for (i, expected) in expected_data.iter().enumerate() {
        let mut buffer = [0u8; PAGE_SIZE];
        let found = store.load((i * SECTORS_PER_PAGE as usize) as u64, &mut buffer).unwrap();
        assert!(found, "Page {} should be found", i);
        assert_eq!(expected, &buffer, "Page {} data mismatch", i);
    }
}

/// G.104: GPU/SIMD Throughput Test (verify backend selection)
/// PERF-002: store() is non-blocking, explicit flush required
#[test]
fn test_g104_backend_selection() {
    let store = BatchedPageStore::with_config(
        Algorithm::Lz4,
        BatchConfig {
            batch_threshold: 50, // Low threshold for test
            flush_timeout: Duration::from_secs(60),
            gpu_batch_size: 1000,
        },
    );

    // Write 50 non-zero pages to reach threshold
    for i in 0..50 {
        let mut data = [0u8; PAGE_SIZE];
        data[0] = (i + 1) as u8;
        store.store((i * SECTORS_PER_PAGE as usize) as u64, &data).unwrap();
    }

    // PERF-002: Explicitly flush (background thread does this in production)
    store.flush_batch().unwrap();

    let stats = store.stats();
    // With batch size 50, should use SIMD (not GPU which needs 1000+)
    assert!(stats.simd_pages > 0, "SIMD pages should be tracked");
    assert_eq!(stats.batch_flushes, 1, "One batch flush should occur");
}

/// G.105: Hybrid Backend Selection Test
#[test]
fn test_g105_hybrid_backend_selection() {
    let store = BatchedPageStore::with_config(
        Algorithm::Lz4,
        BatchConfig {
            batch_threshold: 50,
            flush_timeout: Duration::from_secs(60),
            gpu_batch_size: 1000,
        },
    );

    // Small batch (< 100) uses Simd for lowest latency
    assert_eq!(store.select_backend(50), Backend::Simd);
    assert_eq!(store.select_backend(99), Backend::Simd);

    // Medium and large batches use SimdParallel (19-24 GB/s)
    assert_eq!(store.select_backend(100), Backend::SimdParallel);
    assert_eq!(store.select_backend(500), Backend::SimdParallel);
    assert_eq!(store.select_backend(2000), Backend::SimdParallel);
    assert_eq!(store.select_backend(10000), Backend::SimdParallel);
}

/// G.106: Zero-Page Fast Path Test
/// PERF-002: store() is non-blocking, explicit flush required
#[test]
fn test_g106_zero_page_fast_path() {
    let store = create_batched_test_store();

    // Write 100 zero pages (should bypass batching)
    for i in 0..100 {
        let data = [0u8; PAGE_SIZE];
        store.store((i * SECTORS_PER_PAGE as usize) as u64, &data).unwrap();
    }

    // Zero pages should be stored immediately, not in pending
    let stats = store.stats();
    assert_eq!(stats.zero_pages, 100, "Zero pages should be counted");
    assert_eq!(stats.pending_pages, 0, "Zero pages should bypass pending");

    // Write 100 non-zero pages
    for i in 100..200 {
        let mut data = [0u8; PAGE_SIZE];
        data[0] = (i - 99) as u8;
        store.store((i * SECTORS_PER_PAGE as usize) as u64, &data).unwrap();
    }

    // PERF-002: Explicitly flush (background thread does this in production)
    store.flush_batch().unwrap();

    // Non-zero pages should trigger batch (threshold is 100)
    let stats = store.stats();
    assert!(stats.batch_flushes >= 1, "Non-zero pages should trigger batch flush");
}

/// Test BatchedPageStore roundtrip with ublk interface
#[test]
fn test_batched_ublk_interface_roundtrip() {
    let store = create_batched_test_store();

    // Write data using ublk interface
    let data = [0xDEu8; PAGE_SIZE];
    store.write(0, &data).expect("write failed");

    // Force flush
    store.flush_batch().unwrap();

    // Read back
    let mut buffer = [0u8; PAGE_SIZE];
    store.read(0, &mut buffer).expect("read failed");

    assert_eq!(data, buffer, "Roundtrip data mismatch");
}

/// Test BatchedPageStore discard
#[test]
fn test_batched_discard() {
    let store = create_batched_test_store();

    // Write data
    let data = [0xABu8; PAGE_SIZE];
    store.write(0, &data).unwrap();
    store.flush_batch().unwrap();

    // Discard
    store.discard(0, SECTORS_PER_PAGE as u32).unwrap();

    // Read should return zeros
    let mut buffer = [0xFFu8; PAGE_SIZE];
    store.read(0, &mut buffer).unwrap();
    assert!(is_zero_page(&buffer), "Discarded sector should return zeros");
}

/// Test BatchedPageStore write_zeroes
#[test]
fn test_batched_write_zeroes() {
    let store = create_batched_test_store();

    // Write non-zero data
    let data = [0xCDu8; PAGE_SIZE];
    store.write(0, &data).unwrap();
    store.flush_batch().unwrap();

    // Write zeros
    store.write_zeroes(0, SECTORS_PER_PAGE as u32).unwrap();

    // Read should return zeros
    let mut buffer = [0xFFu8; PAGE_SIZE];
    store.read(0, &mut buffer).unwrap();
    assert!(is_zero_page(&buffer), "write_zeroes should zero the data");
}

/// Test BatchConfig defaults
#[test]
fn test_batch_config_defaults() {
    let config = BatchConfig::default();
    assert_eq!(config.batch_threshold, 1000);
    assert_eq!(config.flush_timeout, Duration::from_millis(10));
    assert_eq!(config.gpu_batch_size, 4000);
}

/// Test BatchedPageStoreStats
#[test]
fn test_batched_stats() {
    let store = create_batched_test_store();

    let stats = store.stats();
    assert_eq!(stats.pages_stored, 0);
    assert_eq!(stats.pending_pages, 0);
    assert_eq!(stats.bytes_stored, 0);
    assert_eq!(stats.bytes_compressed, 0);
    assert_eq!(stats.zero_pages, 0);
    assert_eq!(stats.batch_flushes, 0);
}

/// Test shutdown flag
#[test]
fn test_batched_shutdown() {
    let store = create_batched_test_store();

    assert!(!store.is_shutdown());
    store.shutdown();
    assert!(store.is_shutdown());
}

/// QA Edge Case: batch_threshold=1 flushes after every page
#[test]
fn test_qa_edge_case_batch_threshold_1() {
    let store = BatchedPageStore::with_config(
        Algorithm::Lz4,
        BatchConfig {
            batch_threshold: 1,
            flush_timeout: Duration::from_secs(60),
            gpu_batch_size: 1,
        },
    );

    for i in 0..5 {
        let mut data = [0u8; PAGE_SIZE];
        data[0] = (i + 1) as u8;
        store.store((i * SECTORS_PER_PAGE as usize) as u64, &data).unwrap();
        store.flush_batch().unwrap();
    }

    let stats = store.stats();
    assert_eq!(stats.batch_flushes, 5, "Each page should trigger a flush");
    assert_eq!(stats.pending_pages, 0, "No pages should be pending");
    assert_eq!(stats.pages_stored, 5, "5 pages should be stored");
}

/// QA Edge Case: flush_timeout=0 considers pages immediately stale
#[test]
fn test_qa_edge_case_flush_timeout_0() {
    let store = BatchedPageStore::with_config(
        Algorithm::Lz4,
        BatchConfig {
            batch_threshold: 1000,
            flush_timeout: Duration::from_millis(0),
            gpu_batch_size: 1000,
        },
    );

    let mut data = [0u8; PAGE_SIZE];
    data[0] = 1;
    store.store(0, &data).unwrap();

    assert!(store.should_flush(), "Timeout=0 should always indicate flush needed");

    store.flush_batch().unwrap();
    assert_eq!(store.stats().pending_pages, 0);
    assert_eq!(store.stats().batch_flushes, 1);
}

/// QA Edge Case: Large batch_threshold doesn't prevent timer flush
#[test]
fn test_qa_large_threshold_timer_still_works() {
    let store = BatchedPageStore::with_config(
        Algorithm::Lz4,
        BatchConfig {
            batch_threshold: 1_000_000,
            flush_timeout: Duration::from_millis(1),
            gpu_batch_size: 1000,
        },
    );

    let mut data = [0u8; PAGE_SIZE];
    data[0] = 1;
    store.store(0, &data).unwrap();

    std::thread::sleep(Duration::from_millis(5));

    assert!(store.should_flush(), "Timer should indicate flush needed");
    store.flush_batch().unwrap();

    assert_eq!(store.stats().pending_pages, 0);
    assert_eq!(store.stats().batch_flushes, 1);
}

/// G.108: SimdParallel Used for Large Batches
#[test]
fn test_g108_simd_parallel_used_for_large_batches() {
    let store = BatchedPageStore::with_config(
        Algorithm::Lz4,
        BatchConfig {
            batch_threshold: 1000,
            flush_timeout: Duration::from_secs(60),
            gpu_batch_size: 1000,
        },
    );

    for batch_size in [100, 500, 1000, 2000, 5000, 10000] {
        assert_eq!(
            store.select_backend(batch_size),
            Backend::SimdParallel,
            "G.108 REFUTED: {} pages should use SimdParallel",
            batch_size
        );
    }
}

/// G.109: SIMD Pages Stat Increments
#[test]
fn test_g109_simd_pages_stat_increments() {
    let store = BatchedPageStore::with_config(
        Algorithm::Lz4,
        BatchConfig {
            batch_threshold: 1000,
            flush_timeout: Duration::from_secs(60),
            gpu_batch_size: 1000,
        },
    );

    for i in 0..1000 {
        let mut data = [0u8; PAGE_SIZE];
        for (j, byte) in data.iter_mut().enumerate() {
            *byte = ((i + j) % 256) as u8;
        }
        store.store((i * SECTORS_PER_PAGE as usize) as u64, &data).unwrap();
    }

    store.flush_batch().unwrap();

    let stats = store.stats();
    assert!(stats.simd_pages > 0, "G.109 REFUTED: simd_pages == 0");
    assert_eq!(stats.batch_flushes, 1, "One batch flush should occur");
}

/// G.110: CPU Parallel Compression Roundtrip
#[test]
fn test_g110_cpu_parallel_compression_roundtrip() {
    let store = BatchedPageStore::with_config(
        Algorithm::Lz4,
        BatchConfig {
            batch_threshold: 1000,
            flush_timeout: Duration::from_secs(60),
            gpu_batch_size: 1000,
        },
    );

    let mut test_pages: Vec<(u64, [u8; PAGE_SIZE])> = Vec::new();
    for i in 0..1000 {
        let mut data = [0u8; PAGE_SIZE];
        for (j, byte) in data.iter_mut().enumerate() {
            *byte = ((i * 17 + j * 31) % 256) as u8;
        }
        let sector = (i * SECTORS_PER_PAGE as usize) as u64;
        test_pages.push((sector, data));
        store.store(sector, &data).unwrap();
    }

    let mut errors = 0;
    for (sector, original) in &test_pages {
        let mut buffer = [0u8; PAGE_SIZE];
        let found = store.load(*sector, &mut buffer).unwrap();
        if !found || buffer != *original {
            errors += 1;
        }
    }

    assert_eq!(errors, 0, "G.110 REFUTED: {} pages failed roundtrip", errors);
}

/// G.111: Single Page Write Latency
#[test]
fn test_g111_single_page_write_latency() {
    let store = BatchedPageStore::with_config(
        Algorithm::Lz4,
        BatchConfig {
            batch_threshold: 10000,
            flush_timeout: Duration::from_secs(60),
            gpu_batch_size: 1000,
        },
    );

    let mut latencies_ns: Vec<u64> = Vec::with_capacity(1000);

    for i in 0..1000 {
        let mut data = [0u8; PAGE_SIZE];
        data[0] = (i + 1) as u8;

        let start = std::time::Instant::now();
        store.store((i * SECTORS_PER_PAGE as usize) as u64, &data).unwrap();
        latencies_ns.push(start.elapsed().as_nanos() as u64);
    }

    latencies_ns.sort();
    let p99 = latencies_ns[990];
    let p99_us = p99 as f64 / 1000.0;

    assert!(p99_us < 100.0, "G.111 REFUTED: p99 latency {:.2}us >= 100us", p99_us);
}

/// G.113: Single Page Read Latency
#[test]
fn test_g113_single_page_read_latency() {
    let store = BatchedPageStore::with_config(
        Algorithm::Lz4,
        BatchConfig {
            batch_threshold: 1000,
            flush_timeout: Duration::from_secs(60),
            gpu_batch_size: 1000,
        },
    );

    for i in 0..1000 {
        let mut data = [0u8; PAGE_SIZE];
        for (j, byte) in data.iter_mut().enumerate() {
            *byte = ((i * 17 + j * 31) % 256) as u8;
        }
        store.store((i * SECTORS_PER_PAGE as usize) as u64, &data).unwrap();
    }

    let mut latencies_ns: Vec<u64> = Vec::with_capacity(1000);
    for i in 0..1000 {
        let mut buffer = [0u8; PAGE_SIZE];
        let start = std::time::Instant::now();
        store.load((i * SECTORS_PER_PAGE as usize) as u64, &mut buffer).unwrap();
        latencies_ns.push(start.elapsed().as_nanos() as u64);
    }

    latencies_ns.sort();
    let p99 = latencies_ns[990];
    let p99_us = p99 as f64 / 1000.0;

    assert!(p99_us < 100.0, "G.113 REFUTED: p99 read latency {:.2}us >= 100us", p99_us);
}

/// G.114: Batch Read Latency
#[test]
fn test_g114_batch_read_latency() {
    let store = BatchedPageStore::with_config(
        Algorithm::Lz4,
        BatchConfig {
            batch_threshold: 1000,
            flush_timeout: Duration::from_secs(60),
            gpu_batch_size: 1000,
        },
    );

    for i in 0..1000 {
        let mut data = [0u8; PAGE_SIZE];
        for (j, byte) in data.iter_mut().enumerate() {
            *byte = ((i * 17 + j * 31) % 256) as u8;
        }
        store.store((i * SECTORS_PER_PAGE as usize) as u64, &data).unwrap();
    }

    let sectors: Vec<u64> = (0..1000).map(|i| (i * SECTORS_PER_PAGE as usize) as u64).collect();
    let mut buffers = vec![[0u8; PAGE_SIZE]; 1000];

    let mut latencies_ms: Vec<f64> = Vec::with_capacity(10);
    for _ in 0..10 {
        let start = std::time::Instant::now();
        store.batch_load_parallel(&sectors, &mut buffers).unwrap();
        latencies_ms.push(start.elapsed().as_secs_f64() * 1000.0);
    }

    latencies_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p99 = latencies_ms[9];

    assert!(p99 < 200.0, "G.114 REFUTED: p99 batch read {:.2}ms >= 200ms", p99);
}

/// PERF-002.1: Store should NOT block on flush threshold
#[test]
fn test_perf002_store_non_blocking() {
    let store = BatchedPageStore::with_config(
        Algorithm::Lz4,
        BatchConfig {
            batch_threshold: 100,
            flush_timeout: Duration::from_secs(60),
            gpu_batch_size: 1000,
        },
    );

    for i in 0..99 {
        let data = [((i + 1) as u8); PAGE_SIZE];
        store.store((i * SECTORS_PER_PAGE as usize) as u64, &data).unwrap();
    }

    let data = [0xABu8; PAGE_SIZE];
    let start = std::time::Instant::now();
    store.store((99 * SECTORS_PER_PAGE as usize) as u64, &data).unwrap();
    let latency = start.elapsed();

    assert!(
        latency.as_micros() < 10_000,
        "PERF-002.1 REFUTED: store() took {}us >= 10000us",
        latency.as_micros()
    );
}

/// PERF-002.2: Flush signal wakes background thread immediately
#[test]
fn test_perf002_flush_signal_immediate_wake() {
    let store = Arc::new(BatchedPageStore::with_config(
        Algorithm::Lz4,
        BatchConfig {
            batch_threshold: 50,
            flush_timeout: Duration::from_secs(60),
            gpu_batch_size: 1000,
        },
    ));

    let store_clone = Arc::clone(&store);
    let _flush_handle = spawn_flush_thread(store_clone);

    for i in 0..50 {
        let data = [((i + 1) as u8); PAGE_SIZE];
        store.store((i * SECTORS_PER_PAGE as usize) as u64, &data).unwrap();
    }

    let mut flushed = false;
    for _ in 0..100 {
        std::thread::sleep(Duration::from_millis(1));
        if store.stats().pending_pages == 0 {
            flushed = true;
            break;
        }
    }

    store.shutdown();
    assert!(flushed, "PERF-002.2 REFUTED: Pending pages not flushed within 100ms");
}
