use super::*;
use trueno_zram_core::Algorithm;

mod popperian_integrity;
mod popperian_performance;
mod popperian_resources;

// Helper to create a test compressor
pub(super) fn test_compressor() -> Box<dyn PageCompressor> {
    CompressorBuilder::new().algorithm(Algorithm::Lz4).build().unwrap()
}

#[test]
fn test_parse_device_id() {
    let path = PathBuf::from("/dev/ublkb0");
    assert_eq!(UblkDevice::parse_device_id(&path).unwrap(), 0);

    let path = PathBuf::from("/dev/ublkb42");
    assert_eq!(UblkDevice::parse_device_id(&path).unwrap(), 42);
}

#[test]
fn test_detect_simd_backend() {
    let backend = detect_simd_backend();
    assert!(!backend.is_empty());
}

#[test]
fn test_device_stats_default() {
    let stats = DeviceStats::default();
    assert_eq!(stats.orig_data_size, 0);
    assert_eq!(stats.throughput_gbps, 0.0);
}

// ========================================================================
// BlockDevice tests - Pure Rust compression roundtrip verification
// ========================================================================

#[test]
fn test_block_device_creation() {
    let device = BlockDevice::new(1 << 20, test_compressor()); // 1MB
    assert_eq!(device.size(), 1 << 20);
    assert_eq!(device.block_size(), PAGE_SIZE as u32);
}

#[test]
fn test_write_read_roundtrip_single_page() {
    let mut device = BlockDevice::new(1 << 20, test_compressor());

    // Write a single page
    let data = vec![0xAB; PAGE_SIZE];
    device.write(0, &data).unwrap();

    // Read back
    let mut buf = vec![0u8; PAGE_SIZE];
    device.read(0, &mut buf).unwrap();

    assert_eq!(data, buf, "Single page roundtrip failed");
}

#[test]
fn test_write_read_roundtrip_multiple_pages() {
    let mut device = BlockDevice::new(1 << 20, test_compressor());

    // Write multiple pages at different offsets
    for i in 0..10 {
        let data: Vec<u8> = (0..PAGE_SIZE).map(|j| ((i + j) % 256) as u8).collect();
        device.write(i as u64 * PAGE_SIZE as u64, &data).unwrap();
    }

    // Read back and verify
    for i in 0..10 {
        let expected: Vec<u8> = (0..PAGE_SIZE).map(|j| ((i + j) % 256) as u8).collect();
        let mut buf = vec![0u8; PAGE_SIZE];
        device.read(i as u64 * PAGE_SIZE as u64, &mut buf).unwrap();
        assert_eq!(expected, buf, "Page {} roundtrip failed", i);
    }
}

#[test]
fn test_write_read_roundtrip_random_data() {
    let mut device = BlockDevice::new(1 << 20, test_compressor());

    // Create pseudo-random data (high entropy)
    let data: Vec<u8> = (0..PAGE_SIZE).map(|i| (i * 17 + 31) as u8).collect();
    device.write(0, &data).unwrap();

    let mut buf = vec![0u8; PAGE_SIZE];
    device.read(0, &mut buf).unwrap();

    assert_eq!(data, buf, "Random data roundtrip failed");
}

#[test]
fn test_zero_page_deduplication() {
    let mut device = BlockDevice::new(1 << 20, test_compressor());

    // Write zeros to multiple locations
    let zeros = vec![0u8; PAGE_SIZE];
    device.write(0, &zeros).unwrap();
    device.write(PAGE_SIZE as u64, &zeros).unwrap();
    device.write(2 * PAGE_SIZE as u64, &zeros).unwrap();

    let stats = device.stats();
    assert_eq!(stats.zero_pages, 3, "Should have 3 zero pages");

    // Read back and verify
    for i in 0..3 {
        let mut buf = vec![0xFFu8; PAGE_SIZE];
        device.read(i as u64 * PAGE_SIZE as u64, &mut buf).unwrap();
        assert_eq!(zeros, buf, "Zero page {} readback failed", i);
    }
}

#[test]
fn test_compression_ratio_tracking() {
    let mut device = BlockDevice::new(1 << 20, test_compressor());

    // Write compressible but NON-uniform data (avoids same-fill optimization)
    // PERF-013: Same-fill pages are stored without compression, so use varied data
    let mut compressible = vec![0u8; PAGE_SIZE];
    for i in 0..PAGE_SIZE {
        // Repeating pattern but not uniform - still compresses well
        compressible[i] = (i % 16) as u8;
    }
    device.write(0, &compressible).unwrap();

    let stats = device.stats();
    assert!(
        stats.bytes_compressed < stats.bytes_written,
        "Compressible data should compress: written={}, compressed={}",
        stats.bytes_written,
        stats.bytes_compressed
    );
    assert!(
        stats.compression_ratio() > 1.0,
        "Compression ratio should be > 1.0 for compressible data"
    );
}

#[test]
fn test_entropy_routing_high_entropy() {
    // Use threshold of 7.0 for testing
    let mut device = BlockDevice::with_entropy_threshold(1 << 20, test_compressor(), 7.0);

    // Write high-entropy (pseudo-random) data
    let random: Vec<u8> = (0..PAGE_SIZE).map(|i| (i * 17 + 31) as u8).collect();
    device.write(0, &random).unwrap();

    let stats = device.stats();
    assert!(stats.scalar_pages > 0, "High entropy data should use scalar path");
}

#[test]
fn test_entropy_routing_low_entropy() {
    // Use threshold of 7.0 for testing
    let mut device = BlockDevice::with_entropy_threshold(1 << 20, test_compressor(), 7.0);

    // Write low-entropy (repetitive pattern) data
    let repetitive: Vec<u8> = (0..PAGE_SIZE).map(|i| (i % 4) as u8).collect();
    device.write(0, &repetitive).unwrap();

    let stats = device.stats();
    // Low entropy should NOT go to scalar path
    assert_eq!(stats.scalar_pages, 0, "Low entropy data should not use scalar path");
}

#[test]
fn test_discard_operation() {
    let mut device = BlockDevice::new(1 << 20, test_compressor());

    // Write data
    let data = vec![0xAB; PAGE_SIZE];
    device.write(0, &data).unwrap();

    // Verify it's stored
    let stats_before = device.stats();
    assert_eq!(stats_before.pages_stored, 1);

    // Discard
    device.discard(0, PAGE_SIZE as u64).unwrap();

    // Should now read zeros
    let mut buf = vec![0xFFu8; PAGE_SIZE];
    device.read(0, &mut buf).unwrap();
    assert!(buf.iter().all(|&b| b == 0), "Discarded page should read as zeros");
}

#[test]
fn test_overwrite_page() {
    let mut device = BlockDevice::new(1 << 20, test_compressor());

    // Write initial data
    let data1 = vec![0xAA; PAGE_SIZE];
    device.write(0, &data1).unwrap();

    // Overwrite with different data
    let data2 = vec![0xBB; PAGE_SIZE];
    device.write(0, &data2).unwrap();

    // Read back should return new data
    let mut buf = vec![0u8; PAGE_SIZE];
    device.read(0, &mut buf).unwrap();
    assert_eq!(data2, buf, "Overwritten data should be returned");
}

#[test]
fn test_unwritten_page_returns_zeros() {
    let device = BlockDevice::new(1 << 20, test_compressor());

    // Read from unwritten location
    let mut buf = vec![0xFFu8; PAGE_SIZE];
    device.read(0, &mut buf).unwrap();

    assert!(buf.iter().all(|&b| b == 0), "Unwritten page should read as zeros");
}

#[test]
fn test_alignment_validation_offset() {
    let mut device = BlockDevice::new(1 << 20, test_compressor());
    let data = vec![0u8; PAGE_SIZE];

    // Unaligned offset should fail
    let result = device.write(1, &data);
    assert!(result.is_err(), "Unaligned offset should fail");
}

#[test]
fn test_alignment_validation_length() {
    let mut device = BlockDevice::new(1 << 20, test_compressor());
    let data = vec![0u8; PAGE_SIZE - 1]; // Not page-aligned length

    let result = device.write(0, &data);
    assert!(result.is_err(), "Non-page-aligned length should fail");
}

#[test]
fn test_bounds_check() {
    let mut device = BlockDevice::new(PAGE_SIZE as u64 * 10, test_compressor()); // 10 pages
    let data = vec![0u8; PAGE_SIZE];

    // Writing beyond device size should fail
    let result = device.write(PAGE_SIZE as u64 * 10, &data);
    assert!(result.is_err(), "Writing beyond device size should fail");
}

#[test]
fn test_stats_tracking() {
    let mut device = BlockDevice::new(1 << 20, test_compressor());

    // Write some data
    let data = vec![0xAB; PAGE_SIZE * 3];
    device.write(0, &data).unwrap();

    let stats = device.stats();
    assert_eq!(
        stats.bytes_written,
        PAGE_SIZE as u64 * 3,
        "Bytes written should track correctly"
    );
    assert_eq!(stats.pages_stored, 3, "Pages stored should be 3");

    // Read data
    let mut buf = vec![0u8; PAGE_SIZE * 2];
    device.read(0, &mut buf).unwrap();

    let stats = device.stats();
    assert_eq!(stats.bytes_read, PAGE_SIZE as u64 * 2, "Bytes read should track correctly");
}

#[test]
fn test_various_data_patterns() {
    let mut device = BlockDevice::new(1 << 20, test_compressor());
    let patterns: Vec<Vec<u8>> = vec![
        // All zeros
        vec![0u8; PAGE_SIZE],
        // All ones
        vec![0xFF; PAGE_SIZE],
        // Alternating bytes
        (0..PAGE_SIZE).map(|i| if i % 2 == 0 { 0xAA } else { 0x55 }).collect(),
        // Sequential bytes
        (0..PAGE_SIZE).map(|i| (i % 256) as u8).collect(),
        // Repeating short pattern
        (0..PAGE_SIZE).map(|i| (i % 16) as u8).collect(),
        // Text-like data
        "The quick brown fox jumps over the lazy dog. ".repeat(100).into_bytes()[..PAGE_SIZE]
            .to_vec(),
    ];

    for (i, pattern) in patterns.iter().enumerate() {
        let offset = i as u64 * PAGE_SIZE as u64;
        device.write(offset, pattern).unwrap();

        let mut buf = vec![0u8; PAGE_SIZE];
        device.read(offset, &mut buf).unwrap();

        assert_eq!(*pattern, buf, "Pattern {} roundtrip failed", i);
    }
}

#[test]
fn test_block_device_stats_compression_ratio() {
    let stats = BlockDeviceStats {
        pages_stored: 10,
        bytes_written: 40960,
        bytes_read: 0,
        bytes_compressed: 10240,
        zero_pages: 0,
        gpu_pages: 0,
        simd_pages: 0,
        scalar_pages: 0,
    };

    assert!((stats.compression_ratio() - 4.0).abs() < 0.001, "4:1 compression ratio expected");
}

#[test]
fn test_block_device_stats_compression_ratio_no_data() {
    let stats = BlockDeviceStats::default();
    assert!((stats.compression_ratio() - 1.0).abs() < 0.001, "Default ratio should be 1.0");
}

