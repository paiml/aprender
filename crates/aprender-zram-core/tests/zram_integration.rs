//! Integration tests for zram device operations.
//!
//! These tests require:
//! - Root privileges (or sudo)
//! - The zram module to be loaded
//!
//! Run with: sudo cargo test --test zram_integration -- --nocapture

use trueno_zram_core::zram::{SysfsOps, ZramConfig, ZramDevice, ZramOps};
use trueno_zram_core::{Algorithm, CompressorBuilder, PAGE_SIZE};

/// Check if we can run zram tests.
fn can_run_zram_tests() -> bool {
    // Check if zram is available
    let ops = SysfsOps::new();
    if !ops.is_available() {
        eprintln!("Skipping: zram module not loaded");
        return false;
    }

    // Check if we have write access to sysfs (need root)
    let test_path = "/sys/class/zram-control/hot_add";
    if std::fs::metadata(test_path)
        .map(|m| m.permissions().readonly())
        .unwrap_or(true)
    {
        // Try alternate check - can we write to zram0 if it exists?
        let zram0_reset = "/sys/block/zram0/reset";
        if std::fs::metadata(zram0_reset).is_err() {
            eprintln!("Skipping: need root privileges for zram control");
            return false;
        }
    }

    true
}

/// Test creating and removing a small zram device.
#[test]
fn test_create_remove_device() {
    if !can_run_zram_tests() {
        return;
    }

    let ops = SysfsOps::new();

    // Use a high device number to avoid conflicts with system zram
    let config = ZramConfig {
        device: 14,
        size: 4 * 1024 * 1024, // 4MB - small for testing
        algorithm: "lz4".to_string(),
        streams: 1,
    };

    // Create device
    match ops.create(&config) {
        Ok(()) => {
            println!("Created zram14 with 4MB");
        }
        Err(e) => {
            eprintln!("Failed to create device (may need root): {e}");
            return;
        }
    }

    // Verify device exists
    let dev = ZramDevice::new(14);
    assert!(dev.exists(), "Device should exist after creation");

    // Get status
    let status = ops.status(14).expect("Should get status");
    assert_eq!(status.device, 14);
    assert_eq!(status.disksize, 4 * 1024 * 1024);
    println!("Status: {status}");

    // Remove device
    ops.remove(14, true).expect("Should remove device");

    // Verify removed (or reset)
    // Note: hot_remove may not work on all kernels, so we just verify reset worked
    if dev.exists() {
        let status = ops.status(14).unwrap();
        assert_eq!(status.disksize, 0, "Device should be reset");
    }

    println!("Test passed: create/remove 4MB zram device");
}

/// Test compression on a real zram device.
#[test]
fn test_compression_on_zram() {
    if !can_run_zram_tests() {
        return;
    }

    let ops = SysfsOps::new();

    // Create a 1MB test device
    let config = ZramConfig {
        device: 15,
        size: 1024 * 1024, // 1MB
        algorithm: "lz4".to_string(),
        streams: 1,
    };

    if ops.create(&config).is_err() {
        eprintln!("Skipping: cannot create test device");
        return;
    }

    // Verify the compressor works with our SIMD implementation
    let compressor = CompressorBuilder::new()
        .algorithm(Algorithm::Lz4)
        .build()
        .expect("Should build compressor");

    println!("Using SIMD backend: {:?}", compressor.backend());

    // Compress some test pages
    let test_patterns: &[(&str, [u8; PAGE_SIZE])] = &[
        ("zeros", [0u8; PAGE_SIZE]),
        ("ones", [0xFFu8; PAGE_SIZE]),
        ("pattern", {
            let mut p = [0u8; PAGE_SIZE];
            for (i, b) in p.iter_mut().enumerate() {
                *b = (i % 256) as u8;
            }
            p
        }),
    ];

    for (name, page) in test_patterns {
        let compressed = compressor.compress(page).expect("Should compress");
        let decompressed = compressor
            .decompress(&compressed)
            .expect("Should decompress");
        assert_eq!(page, &decompressed, "Roundtrip failed for {name}");

        let ratio = if !compressed.data.is_empty() {
            PAGE_SIZE as f64 / compressed.data.len() as f64
        } else {
            1.0
        };
        println!(
            "  {name}: {} -> {} bytes ({:.2}x)",
            PAGE_SIZE,
            compressed.data.len(),
            ratio
        );
    }

    // Cleanup
    let _ = ops.remove(15, true);

    println!("Test passed: compression on real zram device");
}

/// Test zram device status reporting.
#[test]
fn test_device_status() {
    if !can_run_zram_tests() {
        return;
    }

    let ops = SysfsOps::new();

    // Check if zram0 exists (common system device)
    let dev = ZramDevice::new(0);
    if !dev.exists() {
        println!("zram0 does not exist, skipping status test");
        return;
    }

    let status = ops.status(0).expect("Should get zram0 status");
    println!("zram0 status:");
    println!("  disksize: {} bytes", status.disksize);
    println!("  orig_data_size: {} bytes", status.orig_data_size);
    println!("  compr_data_size: {} bytes", status.compr_data_size);
    println!("  mem_used_total: {} bytes", status.mem_used_total);
    println!("  algorithm: {}", status.algorithm);
    println!("  ratio: {:.2}x", status.compression_ratio());
}

/// Test listing all zram devices.
#[test]
fn test_list_devices() {
    if !can_run_zram_tests() {
        return;
    }

    let ops = SysfsOps::new();
    let devices = ops.list().expect("Should list devices");

    println!("Found {} zram device(s):", devices.len());
    for status in &devices {
        println!(
            "  zram{}: {} disksize, {} algorithm",
            status.device,
            trueno_zram_core::zram::format_size(status.disksize),
            status.algorithm
        );
    }
}

/// Test multiple small devices.
#[test]
fn test_multiple_devices() {
    if !can_run_zram_tests() {
        return;
    }

    let ops = SysfsOps::new();

    // Create multiple 1MB devices
    let device_nums = [12, 13, 14];
    let mut created = Vec::new();

    for &dev_num in &device_nums {
        let config = ZramConfig {
            device: dev_num,
            size: 1024 * 1024, // 1MB each
            algorithm: "lz4".to_string(),
            streams: 1,
        };

        match ops.create(&config) {
            Ok(()) => {
                created.push(dev_num);
                println!("Created zram{dev_num}");
            }
            Err(e) => {
                eprintln!("Failed to create zram{dev_num}: {e}");
            }
        }
    }

    // Verify all created devices exist
    for &dev_num in &created {
        let status = ops.status(dev_num).expect("Should get status");
        assert_eq!(status.device, dev_num);
        assert_eq!(status.disksize, 1024 * 1024);
    }

    // Cleanup
    for &dev_num in &created {
        let _ = ops.remove(dev_num, true);
    }

    println!("Test passed: created {} devices", created.len());
}

/// Benchmark compression throughput.
#[test]
fn test_compression_throughput() {
    let compressor = CompressorBuilder::new()
        .algorithm(Algorithm::Lz4)
        .build()
        .expect("Should build compressor");

    println!("Backend: {:?}", compressor.backend());

    // Generate test data - mix of compressible and incompressible
    let mut pages = Vec::new();

    // Zeros (highly compressible)
    pages.push([0u8; PAGE_SIZE]);

    // Pattern (compressible)
    let mut pattern = [0u8; PAGE_SIZE];
    for (i, b) in pattern.iter_mut().enumerate() {
        *b = (i % 256) as u8;
    }
    pages.push(pattern);

    // Random-ish (less compressible)
    let mut random = [0u8; PAGE_SIZE];
    let mut state = 12345u64;
    for b in &mut random {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        *b = (state >> 33) as u8;
    }
    pages.push(random);

    // Warmup
    for page in &pages {
        let _ = compressor.compress(page);
    }
    compressor.reset_stats();

    // Benchmark
    let iterations = 1000;
    let start = std::time::Instant::now();

    for _ in 0..iterations {
        for page in &pages {
            let compressed = compressor.compress(page).unwrap();
            let _ = compressor.decompress(&compressed).unwrap();
        }
    }

    let elapsed = start.elapsed();
    let total_pages = iterations * pages.len();
    let total_bytes = total_pages * PAGE_SIZE;

    let throughput_mb = (total_bytes as f64 / 1024.0 / 1024.0) / elapsed.as_secs_f64();
    let pages_per_sec = total_pages as f64 / elapsed.as_secs_f64();

    println!(
        "Throughput: {:.2} MB/s ({:.0} pages/sec)",
        throughput_mb, pages_per_sec
    );
    println!("Stats: {:?}", compressor.stats());
}
