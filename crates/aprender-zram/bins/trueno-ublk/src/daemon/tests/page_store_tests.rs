//! Tests for PageStore and basic I/O operations.

use crate::daemon::batched::is_zero_page;
use crate::daemon::entropy::calculate_entropy;
use crate::daemon::page_store::SECTORS_PER_PAGE;
use crate::daemon::PageStore;
use trueno_zram_core::{Algorithm, PAGE_SIZE};

// ========================================================================
// Helper functions for tests
// ========================================================================

fn create_test_store() -> PageStore {
    PageStore::new(1024 * 1024 * 1024, Algorithm::Lz4) // 1GB device
}

// ========================================================================
// Basic functionality tests
// ========================================================================

#[test]
fn test_is_zero_page() {
    let zeros = vec![0u8; PAGE_SIZE];
    assert!(is_zero_page(&zeros));

    let mut data = vec![0u8; PAGE_SIZE];
    data[100] = 1;
    assert!(!is_zero_page(&data));
}

#[test]
fn test_entropy_calculation() {
    // All zeros - zero entropy
    let zeros = vec![0u8; PAGE_SIZE];
    assert_eq!(calculate_entropy(&zeros), 0.0);

    // Random-like data - high entropy
    let random: Vec<u8> = (0..PAGE_SIZE).map(|i| (i * 17 + 31) as u8).collect();
    let entropy = calculate_entropy(&random);
    assert!(entropy > 7.0);

    // Repetitive data - low entropy
    let repetitive: Vec<u8> = (0..PAGE_SIZE).map(|i| (i % 4) as u8).collect();
    let entropy = calculate_entropy(&repetitive);
    assert!(entropy < 3.0);
}

// ========================================================================
// Section B: Data Integrity Tests (from Renacer Verification Matrix)
// ========================================================================

/// B.11: Roundtrip: Write 0xDEADBEEF pattern, Read back and verify
#[test]
fn test_roundtrip_deadbeef() {
    let mut store = create_test_store();

    // Create page filled with 0xDEADBEEF pattern
    let mut data = [0u8; PAGE_SIZE];
    for i in (0..PAGE_SIZE).step_by(4) {
        data[i] = 0xDE;
        data[i + 1] = 0xAD;
        data[i + 2] = 0xBE;
        data[i + 3] = 0xEF;
    }

    // Write to sector 0
    store.write(0, &data).expect("write failed");

    // Read back
    let mut buffer = [0u8; PAGE_SIZE];
    store.read(0, &mut buffer).expect("read failed");

    // Verify exact match
    assert_eq!(data, buffer, "Roundtrip data mismatch");
}

/// B.12: Zero-page logic: Write zeros, verify efficient storage
#[test]
fn test_zero_page_optimization() {
    let mut store = create_test_store();

    // Write zero page
    let zeros = [0u8; PAGE_SIZE];
    store.write(0, &zeros).expect("write failed");

    // Verify stats show zero page was detected
    let stats = store.stats();
    assert!(stats.zero_pages > 0, "Zero page not detected");

    // Read back should return zeros
    let mut buffer = [0xFFu8; PAGE_SIZE];
    store.read(0, &mut buffer).expect("read failed");
    assert!(is_zero_page(&buffer), "Zero page not returned correctly");
}

/// B.13: Partial pages: Write 512 bytes (1 sector) -> Read 4KB (1 page)
#[test]
fn test_partial_page_write() {
    let mut store = create_test_store();

    // Write only 512 bytes (1 sector)
    let sector_data = [0xABu8; 512];
    store.write(0, &sector_data).expect("write failed");

    // Read full page (4KB)
    let mut buffer = [0u8; PAGE_SIZE];
    store.read(0, &mut buffer).expect("read failed");

    // First 512 bytes should match, rest should be zeros
    assert_eq!(&buffer[..512], &sector_data[..], "First sector mismatch");
    assert!(buffer[512..].iter().all(|&b| b == 0), "Rest should be zeros");
}

/// B.14: Boundary check: Write to last sector
#[test]
fn test_boundary_last_sector() {
    let mut store = create_test_store();

    // Calculate last page sector
    let dev_size = 1024 * 1024 * 1024u64; // 1GB
    let last_page_sector = (dev_size / PAGE_SIZE as u64 - 1) * SECTORS_PER_PAGE;

    // Write to last page
    let data = [0xEFu8; PAGE_SIZE];
    store.write(last_page_sector, &data).expect("write to last sector failed");

    // Read back
    let mut buffer = [0u8; PAGE_SIZE];
    store.read(last_page_sector, &mut buffer).expect("read from last sector failed");
    assert_eq!(data, buffer, "Last sector data mismatch");
}

/// B.15: Offset check: Write to sector 8, verify sector 0 is unchanged
#[test]
fn test_offset_independence() {
    let mut store = create_test_store();

    // Write to sector 0 (first page)
    let data0 = [0x11u8; PAGE_SIZE];
    store.write(0, &data0).expect("write to sector 0 failed");

    // Write to sector 8 (second page)
    let data8 = [0x22u8; PAGE_SIZE];
    store.write(8, &data8).expect("write to sector 8 failed");

    // Verify sector 0 is unchanged
    let mut buffer = [0u8; PAGE_SIZE];
    store.read(0, &mut buffer).expect("read sector 0 failed");
    assert_eq!(data0, buffer, "Sector 0 was corrupted");

    // Verify sector 8 has correct data
    store.read(8, &mut buffer).expect("read sector 8 failed");
    assert_eq!(data8, buffer, "Sector 8 data mismatch");
}

/// B.16: Data pollution: Fill device with random data, verify checksums
#[test]
fn test_data_pollution_checksums() {
    let mut store = create_test_store();

    // Write 100 pages with pseudo-random data
    let num_pages = 100;
    let mut expected_checksums = Vec::new();

    for i in 0..num_pages {
        let mut data = [0u8; PAGE_SIZE];
        // Fill with pseudo-random pattern based on page index
        for j in 0..PAGE_SIZE {
            data[j] = ((i * 17 + j * 31 + 7) % 256) as u8;
        }

        let checksum: u64 = data.iter().map(|&b| b as u64).sum();
        expected_checksums.push(checksum);

        let sector = (i as u64) * SECTORS_PER_PAGE;
        store.write(sector, &data).expect("write failed");
    }

    // Verify all pages
    for i in 0..num_pages {
        let mut buffer = [0u8; PAGE_SIZE];
        let sector = (i as u64) * SECTORS_PER_PAGE;
        store.read(sector, &mut buffer).expect("read failed");

        let checksum: u64 = buffer.iter().map(|&b| b as u64).sum();
        assert_eq!(checksum, expected_checksums[i], "Checksum mismatch at page {}", i);
    }
}

/// B.19: Discard: Verify discard clears data in PageStore
#[test]
fn test_discard_clears_data() {
    let mut store = create_test_store();

    // Write data
    let data = [0xCDu8; PAGE_SIZE];
    store.write(0, &data).expect("write failed");

    // Verify data exists
    let mut buffer = [0u8; PAGE_SIZE];
    store.read(0, &mut buffer).expect("read failed");
    assert_eq!(data, buffer, "Data not written");

    // Discard the sector
    store.discard(0, SECTORS_PER_PAGE as u32).expect("discard failed");

    // Read should return zeros (unallocated)
    store.read(0, &mut buffer).expect("read after discard failed");
    assert!(is_zero_page(&buffer), "Data not cleared after discard");
}

/// B.20: Write zeros operation
#[test]
fn test_write_zeroes_operation() {
    let mut store = create_test_store();

    // Write non-zero data first
    let data = [0xFFu8; PAGE_SIZE];
    store.write(0, &data).expect("write failed");

    // Issue write_zeroes
    store.write_zeroes(0, SECTORS_PER_PAGE as u32).expect("write_zeroes failed");

    // Read should return zeros
    let mut buffer = [0xFFu8; PAGE_SIZE];
    store.read(0, &mut buffer).expect("read failed");
    assert!(is_zero_page(&buffer), "write_zeroes did not zero the data");
}

/// Test multiple sector write/read spanning pages
#[test]
fn test_multi_sector_io() {
    let mut store = create_test_store();

    // Write 2 pages worth of data (16 sectors)
    let data = vec![0xABu8; PAGE_SIZE * 2];
    store.write(0, &data).expect("write failed");

    // Read back
    let mut buffer = vec![0u8; PAGE_SIZE * 2];
    store.read(0, &mut buffer).expect("read failed");
    assert_eq!(data, buffer, "Multi-sector data mismatch");
}

/// Test read from unallocated sector returns zeros
#[test]
fn test_read_unallocated_returns_zeros() {
    let store = create_test_store();

    let mut buffer = [0xFFu8; PAGE_SIZE];
    store.read(0, &mut buffer).expect("read failed");
    assert!(is_zero_page(&buffer), "Unallocated sector should return zeros");
}

/// Test overwrite existing data
#[test]
fn test_overwrite_data() {
    let mut store = create_test_store();

    // Write initial data
    let data1 = [0x11u8; PAGE_SIZE];
    store.write(0, &data1).expect("first write failed");

    // Overwrite with different data
    let data2 = [0x22u8; PAGE_SIZE];
    store.write(0, &data2).expect("second write failed");

    // Read back should return second data
    let mut buffer = [0u8; PAGE_SIZE];
    store.read(0, &mut buffer).expect("read failed");
    assert_eq!(data2, buffer, "Overwrite failed");
}

/// Test compression ratio (data should be smaller when compressed)
#[test]
fn test_compression_ratio() {
    let mut store = create_test_store();

    // Write highly compressible data (repeated pattern)
    let mut data = [0u8; PAGE_SIZE];
    for i in 0..PAGE_SIZE {
        data[i] = (i % 16) as u8;
    }
    store.write(0, &data).expect("write failed");

    let stats = store.stats();
    // Compressed size should be less than original
    if stats.bytes_stored > 0 {
        assert!(
            stats.bytes_compressed < stats.bytes_stored,
            "Compressible data should compress: stored={} compressed={}",
            stats.bytes_stored,
            stats.bytes_compressed
        );
    }
}

/// Test stats tracking
#[test]
fn test_stats_tracking() {
    let mut store = create_test_store();

    // Write zero page
    let zeros = [0u8; PAGE_SIZE];
    store.write(0, &zeros).expect("write failed");

    let stats = store.stats();
    assert_eq!(stats.pages_stored, 1);
    assert!(stats.zero_pages >= 1);

    // Write non-zero page
    let data = [0x42u8; PAGE_SIZE];
    store.write(8, &data).expect("write failed");

    let stats = store.stats();
    assert_eq!(stats.pages_stored, 2);
}
