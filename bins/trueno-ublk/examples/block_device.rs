//! Basic block device example
//!
//! Demonstrates creating a compressed block device, writing data,
//! and reading it back.
//!
//! Run with: `cargo run --example block_device -p trueno-ublk`

use trueno_ublk::BlockDevice;
use trueno_zram_core::{Algorithm, CompressorBuilder, PAGE_SIZE};

fn main() -> anyhow::Result<()> {
    println!("trueno-ublk Block Device Example");
    println!("=================================\n");

    // Create an LZ4 compressor with SIMD auto-detection
    let compressor = CompressorBuilder::new().algorithm(Algorithm::Lz4).build()?;

    println!("Created LZ4 compressor with SIMD backend");

    // Create a 64MB block device
    let device_size = 64 * 1024 * 1024; // 64 MB
    let mut device = BlockDevice::new(device_size, compressor);

    println!("Created block device: {} MB\n", device_size / (1024 * 1024));

    // Write some test patterns
    println!("Writing test patterns...");

    // Pattern 1: Highly compressible (all same value)
    let compressible = vec![0xAA; PAGE_SIZE];
    device.write(0, &compressible)?;
    println!("  Page 0: Highly compressible (all 0xAA)");

    // Pattern 2: Zero page
    let zeros = vec![0u8; PAGE_SIZE];
    device.write(PAGE_SIZE as u64, &zeros)?;
    println!("  Page 1: Zero page");

    // Pattern 3: Sequential data
    let sequential: Vec<u8> = (0..PAGE_SIZE).map(|i| (i % 256) as u8).collect();
    device.write(2 * PAGE_SIZE as u64, &sequential)?;
    println!("  Page 2: Sequential data");

    // Pattern 4: Pseudo-random data (less compressible)
    let random: Vec<u8> = (0..PAGE_SIZE).map(|i| ((i * 17 + 31) % 256) as u8).collect();
    device.write(3 * PAGE_SIZE as u64, &random)?;
    println!("  Page 3: Pseudo-random data\n");

    // Read back and verify
    println!("Reading back and verifying...");

    let mut buf = vec![0u8; PAGE_SIZE];

    device.read(0, &mut buf)?;
    assert_eq!(buf, compressible, "Compressible pattern mismatch");
    println!("  Page 0: OK");

    device.read(PAGE_SIZE as u64, &mut buf)?;
    assert_eq!(buf, zeros, "Zero page mismatch");
    println!("  Page 1: OK");

    device.read(2 * PAGE_SIZE as u64, &mut buf)?;
    assert_eq!(buf, sequential, "Sequential pattern mismatch");
    println!("  Page 2: OK");

    device.read(3 * PAGE_SIZE as u64, &mut buf)?;
    assert_eq!(buf, random, "Random pattern mismatch");
    println!("  Page 3: OK\n");

    // Display statistics
    let stats = device.stats();
    println!("Device Statistics:");
    println!("  Pages stored:      {}", stats.pages_stored);
    println!("  Bytes written:     {} KB", stats.bytes_written / 1024);
    println!("  Bytes compressed:  {} KB", stats.bytes_compressed / 1024);
    println!("  Compression ratio: {:.2}x", stats.compression_ratio());
    println!("  Zero pages:        {}", stats.zero_pages);

    println!("\nAll tests passed!");

    Ok(())
}
