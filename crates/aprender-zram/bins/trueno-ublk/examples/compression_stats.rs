//! Compression statistics example
//!
//! Demonstrates how trueno-ublk tracks compression statistics and
//! compares different algorithms.
//!
//! Run with: `cargo run --example compression_stats -p trueno-ublk`

use trueno_ublk::BlockDevice;
use trueno_zram_core::{Algorithm, CompressorBuilder, PAGE_SIZE};

fn main() -> anyhow::Result<()> {
    println!("trueno-ublk Compression Statistics Example");
    println!("==========================================\n");

    // Generate test data with different entropy levels
    let test_data: Vec<(&str, Vec<u8>)> = vec![
        ("All zeros", vec![0u8; PAGE_SIZE]),
        ("All ones", vec![0xFF; PAGE_SIZE]),
        ("Repeating pattern", (0..PAGE_SIZE).map(|i| (i % 4) as u8).collect()),
        ("Sequential bytes", (0..PAGE_SIZE).map(|i| (i % 256) as u8).collect()),
        (
            "Mixed pattern",
            (0..PAGE_SIZE)
                .map(|i| if i < PAGE_SIZE / 2 { 0xAA } else { (i % 256) as u8 })
                .collect(),
        ),
        ("High entropy", (0..PAGE_SIZE).map(|i| ((i * 17 + 31) % 256) as u8).collect()),
    ];

    // Test with different algorithms
    for algorithm in [Algorithm::Lz4, Algorithm::Zstd { level: 3 }] {
        println!("Algorithm: {algorithm:?}");
        println!("{:-<60}", "");

        let compressor = CompressorBuilder::new().algorithm(algorithm).build()?;

        let mut device = BlockDevice::new(64 * 1024 * 1024, compressor);

        // Write all test patterns
        for (i, (name, data)) in test_data.iter().enumerate() {
            device.write((i * PAGE_SIZE) as u64, data)?;
            println!("  Wrote: {name}");
        }

        // Get statistics
        let stats = device.stats();

        println!("\n  Results:");
        println!("    Pages stored:      {}", stats.pages_stored);
        println!("    Bytes written:     {} bytes", stats.bytes_written);
        println!("    Bytes compressed:  {} bytes", stats.bytes_compressed);
        println!("    Compression ratio: {:.2}x", stats.compression_ratio());
        println!("    Zero pages:        {}", stats.zero_pages);
        println!(
            "    Space savings:     {:.1}%",
            (1.0 - (stats.bytes_compressed as f64 / stats.bytes_written as f64)) * 100.0
        );
        println!();
    }

    // Demonstrate statistics tracking over time
    println!("Progressive Statistics Demo");
    println!("{:-<60}", "");

    let compressor = CompressorBuilder::new().algorithm(Algorithm::Lz4).build()?;

    let mut device = BlockDevice::new(64 * 1024 * 1024, compressor);

    for i in 0..10 {
        // Alternate between highly compressible and less compressible data
        let data: Vec<u8> = if i % 2 == 0 {
            vec![0xAA; PAGE_SIZE] // Highly compressible
        } else {
            (0..PAGE_SIZE).map(|j| ((j * (i + 1) * 17) % 256) as u8).collect()
        };

        device.write((i * PAGE_SIZE) as u64, &data)?;

        let stats = device.stats();
        println!(
            "  After page {}: ratio={:.2}x, zeros={}, stored={}",
            i,
            stats.compression_ratio(),
            stats.zero_pages,
            stats.pages_stored
        );
    }

    println!("\nDone!");

    Ok(())
}
