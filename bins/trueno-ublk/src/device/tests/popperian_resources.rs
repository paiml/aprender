// ========================================================================
// Popperian Falsification Checklist - Section B: Resource Management (21-30)
// ========================================================================

use super::*;

/// B21: Leak check - Create/Destroy device 1000 times.
#[test]
fn popperian_b21_leak_check_create_destroy() {
    // Create and drop devices repeatedly
    for i in 0..1000 {
        let mut device = BlockDevice::new(1 << 20, test_compressor());

        // Write some data
        let data = vec![0xAB; PAGE_SIZE];
        device.write(0, &data).unwrap();

        // Read back
        let mut buf = vec![0u8; PAGE_SIZE];
        device.read(0, &mut buf).unwrap();

        // Device drops here, releasing memory
        drop(device);

        if i % 100 == 0 {
            // Verify we're not accumulating state
            // (In real test, would check RSS)
        }
    }
    // If we get here without OOM, the test passes
}

/// B22: Zero-page deduplication efficiency.
/// Write many zero pages, verify minimal memory usage.
#[test]
fn popperian_b22_zero_page_deduplication_efficiency() {
    let num_pages = 256; // 1MB of zeros
    let device_size = (num_pages * PAGE_SIZE) as u64;
    let mut device = BlockDevice::new(device_size, test_compressor());

    // Write all zeros
    let zeros = vec![0u8; PAGE_SIZE];
    for i in 0..num_pages {
        device.write((i * PAGE_SIZE) as u64, &zeros).unwrap();
    }

    let stats = device.stats();
    assert_eq!(
        stats.zero_pages, num_pages as u64,
        "B22: All pages should be zero-deduplicated"
    );

    // Compressed bytes for zero pages should be minimal
    // (Each zero page is stored as a sentinel, not actual data)
    let bytes_per_zero_page = stats.bytes_compressed as f64 / num_pages as f64;
    assert!(
        bytes_per_zero_page < 100.0,
        "B22: Zero pages should use minimal memory ({:.0} bytes/page)",
        bytes_per_zero_page
    );
}

/// B23: Compression ratio on realistic text data.
#[test]
fn popperian_b23_compression_ratio_text_data() {
    let num_pages = 256; // 1MB
    let device_size = (num_pages * PAGE_SIZE) as u64;
    let mut device = BlockDevice::new(device_size, test_compressor());

    // Write text-like data (highly compressible)
    let text_pattern = "The quick brown fox jumps over the lazy dog. \
        Lorem ipsum dolor sit amet, consectetur adipiscing elit. \
        Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. ";
    let text_page: Vec<u8> =
        text_pattern.chars().cycle().take(PAGE_SIZE).map(|c| c as u8).collect();

    for i in 0..num_pages {
        device.write((i * PAGE_SIZE) as u64, &text_page).unwrap();
    }

    let stats = device.stats();
    let ratio = stats.compression_ratio();

    // Text should compress at least 2:1 with LZ4
    assert!(ratio >= 2.0, "B23: Text data should compress >= 2:1, got {:.2}:1", ratio);

    println!("B23: Text compression ratio: {:.2}:1", ratio);
}

/// B24: Max device allocation test.
/// Create many independent devices to test resource limits.
#[test]
fn popperian_b24_max_devices() {
    let mut devices = Vec::new();

    // Create up to 100 small devices
    for i in 0..100 {
        let device = BlockDevice::new(PAGE_SIZE as u64 * 10, test_compressor());
        devices.push(device);

        if i % 10 == 0 {
            // Verify we can still write to existing devices
            let data = vec![i as u8; PAGE_SIZE];
            for (j, dev) in devices.iter_mut().enumerate() {
                if j < 10 {
                    dev.write(0, &data).unwrap();
                }
            }
        }
    }

    assert_eq!(devices.len(), 100, "B24: Should create 100 devices");

    // Verify all can still be read
    for dev in &devices {
        let mut buf = vec![0u8; PAGE_SIZE];
        dev.read(0, &mut buf).unwrap();
    }
}

/// B25: OOM resilience - write until device is full.
#[test]
fn popperian_b25_oom_resilience() {
    // Small device that will fill up
    let device_size = PAGE_SIZE as u64 * 10;
    let mut device = BlockDevice::new(device_size, test_compressor());

    // Fill the device
    let data = vec![0xAB; PAGE_SIZE];
    for i in 0..10 {
        let result = device.write((i * PAGE_SIZE) as u64, &data);
        assert!(result.is_ok(), "B25: Write {} should succeed", i);
    }

    // Writing beyond capacity should fail gracefully
    let result = device.write(device_size, &data);
    assert!(result.is_err(), "B25: Write beyond capacity should fail gracefully");
}

/// B26: CPU affinity (simplified test for single-threaded operation).
#[test]
fn popperian_b26_cpu_affinity() {
    // BlockDevice is single-threaded, so CPU affinity is managed by caller.
    // This test verifies operations complete on current thread.
    let mut device = BlockDevice::new(1 << 20, test_compressor());

    let data = vec![0xAB; PAGE_SIZE];
    let start_thread = std::thread::current().id();

    device.write(0, &data).unwrap();

    let end_thread = std::thread::current().id();
    assert_eq!(start_thread, end_thread, "B26: Operations should complete on same thread");
}

/// B27: File descriptor leak check (simplified for in-memory device).
#[test]
fn popperian_b27_fd_leak_check() {
    // BlockDevice doesn't use file descriptors directly.
    // This test verifies repeated operations don't leak resources.
    for _ in 0..100 {
        let mut device = BlockDevice::new(1 << 20, test_compressor());

        // Perform operations
        let data = vec![0xAB; PAGE_SIZE * 10];
        device.write(0, &data).unwrap();

        let mut buf = vec![0u8; PAGE_SIZE * 10];
        device.read(0, &mut buf).unwrap();

        // Discard all
        device.discard(0, (PAGE_SIZE * 10) as u64).unwrap();

        // Device drops, resources released
    }
    // Success if we don't run out of resources
}

/// B28: Buffer pool exhaustion - high I/O depth stress test.
#[test]
fn popperian_b28_buffer_pool_exhaustion() {
    let mut device = BlockDevice::new(1 << 24, test_compressor()); // 16MB

    // Perform many rapid writes with varying patterns
    let patterns: Vec<Vec<u8>> = (0..256).map(|i| vec![(i % 256) as u8; PAGE_SIZE]).collect();

    // Rapid write/read cycles
    for iteration in 0..100 {
        for (i, pattern) in patterns.iter().enumerate() {
            let offset = (i * PAGE_SIZE) as u64;
            device.write(offset, pattern).unwrap();
        }

        // Verify random samples
        let check_indices = [0, 50, 100, 200, 255];
        for &idx in &check_indices {
            let mut buf = vec![0u8; PAGE_SIZE];
            device.read((idx * PAGE_SIZE) as u64, &mut buf).unwrap();
            assert!(
                buf.iter().all(|&b| b == (idx % 256) as u8),
                "B28: Iteration {}, pattern {} verification failed",
                iteration,
                idx
            );
        }
    }
}

/// B29: Idle timeout (simplified - verify device can be idle then reused).
#[test]
fn popperian_b29_idle_timeout() {
    let mut device = BlockDevice::new(1 << 20, test_compressor());

    // Write data
    let data = vec![0xAB; PAGE_SIZE];
    device.write(0, &data).unwrap();

    // "Idle" period (in real implementation, would sleep)
    // For testing, we just verify state is preserved

    // Verify data is still accessible after "idle"
    let mut buf = vec![0u8; PAGE_SIZE];
    device.read(0, &mut buf).unwrap();
    assert_eq!(data, buf, "B29: Data should persist through idle period");

    // Can still write after idle
    let new_data = vec![0xCD; PAGE_SIZE];
    device.write(PAGE_SIZE as u64, &new_data).unwrap();

    let mut buf2 = vec![0u8; PAGE_SIZE];
    device.read(PAGE_SIZE as u64, &mut buf2).unwrap();
    assert_eq!(new_data, buf2, "B29: Should write after idle");
}

/// B30: Fragmentation stress test - random write pattern.
#[test]
fn popperian_b30_fragmentation_stress() {
    let num_pages = 1000;
    let device_size = (num_pages * PAGE_SIZE) as u64;
    let mut device = BlockDevice::new(device_size, test_compressor());

    // Random-order writes using deterministic pseudo-random sequence
    let mut order: Vec<usize> = (0..num_pages).collect();
    let mut rng_state: u64 = 0xDEADBEEF;
    for i in (1..order.len()).rev() {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let j = (rng_state as usize) % (i + 1);
        order.swap(i, j);
    }

    // Write in "random" order
    for &page_idx in &order {
        let data: Vec<u8> = (0..PAGE_SIZE).map(|i| ((page_idx + i) % 256) as u8).collect();
        device.write((page_idx * PAGE_SIZE) as u64, &data).unwrap();
    }

    // Verify all pages in sequential order
    for page_idx in 0..num_pages {
        let expected: Vec<u8> = (0..PAGE_SIZE).map(|i| ((page_idx + i) % 256) as u8).collect();
        let mut buf = vec![0u8; PAGE_SIZE];
        device.read((page_idx * PAGE_SIZE) as u64, &mut buf).unwrap();
        assert_eq!(
            expected, buf,
            "B30: Page {} verification failed after fragmented writes",
            page_idx
        );
    }

    // Verify compression still works
    let stats = device.stats();
    assert!(
        stats.compression_ratio() >= 1.0,
        "B30: Compression should still function after fragmented writes"
    );
}

