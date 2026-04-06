//! Entropy-based routing example
//!
//! Demonstrates how trueno-ublk routes data to different compression
//! backends based on entropy analysis.
//!
//! Run with: `cargo run --example entropy_routing -p trueno-ublk`

use trueno_ublk::BlockDevice;
use trueno_zram_core::{Algorithm, CompressorBuilder, PAGE_SIZE};

/// Calculate Shannon entropy of data (bits per byte)
fn calculate_entropy(data: &[u8]) -> f64 {
    let mut counts = [0u64; 256];
    for &byte in data {
        counts[byte as usize] += 1;
    }

    let len = data.len() as f64;
    let mut entropy = 0.0;

    for &count in &counts {
        if count > 0 {
            let p = count as f64 / len;
            entropy -= p * p.log2();
        }
    }

    entropy
}

fn main() -> anyhow::Result<()> {
    println!("trueno-ublk Entropy Routing Example");
    println!("===================================\n");

    // Create device with custom entropy threshold
    let compressor = CompressorBuilder::new().algorithm(Algorithm::Lz4).build()?;

    // Threshold of 7.0 bits/byte (out of 8.0 max)
    let mut device = BlockDevice::with_entropy_threshold(64 * 1024 * 1024, compressor, 7.0);

    println!("Device created with entropy threshold: 7.0 bits/byte\n");
    println!("Entropy Routing Logic:");
    println!("  - Low entropy (<4.0):   GPU batch compression (highly compressible)");
    println!("  - Medium entropy (4-7): SIMD compression (normal data)");
    println!("  - High entropy (>7.0):  Scalar path (incompressible)\n");

    // Test data with different entropy levels
    let test_cases: Vec<(&str, Vec<u8>)> = vec![
        // Very low entropy - all same value
        ("All zeros (0.0 bits)", vec![0u8; PAGE_SIZE]),
        // Low entropy - simple pattern
        ("2-byte pattern (~1.0 bits)", (0..PAGE_SIZE).map(|i| (i % 2) as u8).collect()),
        // Medium-low entropy
        ("16-value pattern (~4.0 bits)", (0..PAGE_SIZE).map(|i| (i % 16) as u8).collect()),
        // Medium entropy
        ("Sequential bytes (~5.5 bits)", (0..PAGE_SIZE).map(|i| (i % 256) as u8).collect()),
        // Medium-high entropy
        ("LCG pseudo-random (~6.5 bits)", {
            let mut data = vec![0u8; PAGE_SIZE];
            let mut x: u32 = 12345;
            for byte in &mut data {
                x = x.wrapping_mul(1_103_515_245).wrapping_add(12345);
                *byte = (x >> 16) as u8;
            }
            data
        }),
        // High entropy - nearly random
        ("XorShift random (~7.5 bits)", {
            let mut data = vec![0u8; PAGE_SIZE];
            let mut x: u64 = 88_172_645_463_325_252;
            for chunk in data.chunks_mut(8) {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                for (i, byte) in chunk.iter_mut().enumerate() {
                    *byte = (x >> (i * 8)) as u8;
                }
            }
            data
        }),
    ];

    println!("{:<30} {:>12} {:>15}", "Pattern", "Entropy", "Expected Route");
    println!("{:-<60}", "");

    for (i, (name, data)) in test_cases.iter().enumerate() {
        let entropy = calculate_entropy(data);
        let expected_route = if entropy < 4.0 {
            "GPU"
        } else if entropy > 7.0 {
            "Scalar"
        } else {
            "SIMD"
        };

        println!("{name:<30} {entropy:>10.2} bits {expected_route:>12}");

        device.write((i * PAGE_SIZE) as u64, data)?;
    }

    // Check routing statistics
    let stats = device.stats();

    println!("\n{:-<60}", "");
    println!("\nRouting Statistics:");
    println!("  GPU pages:    {} (low entropy, highly compressible)", stats.gpu_pages);
    println!("  SIMD pages:   {} (medium entropy, normal data)", stats.simd_pages);
    println!("  Scalar pages: {} (high entropy, incompressible)", stats.scalar_pages);
    println!("  Zero pages:   {} (all zeros, deduplicated)", stats.zero_pages);

    println!("\nTotal compression ratio: {:.2}x", stats.compression_ratio());

    // Verify routing worked correctly
    println!("\nRouting Verification:");
    if stats.zero_pages >= 1 {
        println!("  [OK] Zero page detected and deduplicated");
    }
    if stats.gpu_pages >= 1 {
        println!("  [OK] Low entropy data routed to GPU path");
    }
    if stats.simd_pages >= 2 {
        println!("  [OK] Medium entropy data routed to SIMD path");
    }
    if stats.scalar_pages >= 1 {
        println!("  [OK] High entropy data routed to scalar path");
    }

    println!("\nDone!");

    Ok(())
}
