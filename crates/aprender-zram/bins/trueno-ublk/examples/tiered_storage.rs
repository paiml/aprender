//! Tiered Storage Demo (KERN-001/002/003)
//!
//! Demonstrates the kernel-cooperative tiered storage architecture.
//!
//! Run with: `cargo run --example tiered_storage -p trueno-ublk`

/// Simulated tier statistics
struct TierStats {
    kernel_zram_pages: u64,
    simd_pages: u64,
    nvme_pages: u64,
    samefill_pages: u64,
    total_bytes: u64,
    compressed_bytes: u64,
}

impl TierStats {
    fn new() -> Self {
        Self {
            kernel_zram_pages: 0,
            simd_pages: 0,
            nvme_pages: 0,
            samefill_pages: 0,
            total_bytes: 0,
            compressed_bytes: 0,
        }
    }

    fn total_pages(&self) -> u64 {
        self.kernel_zram_pages + self.simd_pages + self.nvme_pages + self.samefill_pages
    }

    fn compression_ratio(&self) -> f64 {
        if self.compressed_bytes == 0 {
            return 1.0;
        }
        self.total_bytes as f64 / self.compressed_bytes as f64
    }
}

/// Calculate Shannon entropy
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

/// Determine routing based on entropy
fn route_page(entropy: f64) -> &'static str {
    if entropy < 0.1 {
        "samefill" // Same-fill detection (zeros, patterns)
    } else if entropy < 6.0 {
        "kernel_zram" // Low entropy → kernel ZRAM (fast path)
    } else if entropy < 7.5 {
        "simd" // Medium entropy → SIMD ZSTD (better ratio)
    } else {
        "nvme" // High entropy → skip compression
    }
}

#[allow(clippy::too_many_lines)]
fn main() {
    println!("trueno-ublk Tiered Storage Demo (v3.17.0)");
    println!("=========================================\n");

    println!("Kernel-Cooperative Architecture:");
    println!("--------------------------------");
    println!("  trueno-ublk uses intelligent routing to achieve optimal performance:");
    println!();
    println!("  ┌─────────────────────────────────────────────────────────┐");
    println!("  │                    ENTROPY ROUTER                       │");
    println!("  │                                                         │");
    println!("  │   H(X) < 0.1  →  Same-Fill (metadata only, 171 GB/s)   │");
    println!("  │   H(X) < 6.0  →  Kernel ZRAM (fast LZ4, 171 GB/s I/O)  │");
    println!("  │   H(X) < 7.5  →  SIMD ZSTD (15 GiB/s, better ratio)   │");
    println!("  │   H(X) > 7.5  →  NVMe Direct (skip compression)        │");
    println!("  │                                                         │");
    println!("  └─────────────────────────────────────────────────────────┘");
    println!();

    // Simulate different data types
    let test_cases: Vec<(&str, Vec<u8>, &str)> = vec![
        ("Zero pages", vec![0u8; 4096], "Same-fill detection (∞ ratio)"),
        ("Pattern data", (0..4096u16).map(|i| (i % 4) as u8).collect(), "Same-fill or kernel ZRAM"),
        (
            "Text/code",
            {
                let text = b"fn main() { println!(\"Hello, world!\"); } ";
                let mut data = Vec::with_capacity(4096);
                while data.len() < 4096 {
                    data.extend_from_slice(text);
                }
                data.truncate(4096);
                data
            },
            "Kernel ZRAM (47x ratio typical)",
        ),
        (
            "Mixed data",
            (0..4096u16).map(|i| ((i * 17 + 31) % 256) as u8).collect(),
            "SIMD ZSTD (better ratio)",
        ),
        (
            "Random/encrypted",
            {
                let mut data = vec![0u8; 4096];
                let mut x: u64 = 0xDEAD_BEEF;
                for chunk in data.chunks_mut(8) {
                    x ^= x << 13;
                    x ^= x >> 7;
                    x ^= x << 17;
                    for (i, b) in chunk.iter_mut().enumerate() {
                        *b = (x >> (i * 8)) as u8;
                    }
                }
                data
            },
            "NVMe direct (skip compression)",
        ),
    ];

    let mut stats = TierStats::new();

    println!("{:<20} {:>10} {:>15} {:>20}", "Data Type", "Entropy", "Route", "Rationale");
    println!("{:-<70}", "");

    for (name, data, rationale) in &test_cases {
        let entropy = calculate_entropy(data);
        let route = route_page(entropy);

        println!("{name:<20} {entropy:>8.2} b {route:>15} {rationale:>20}");

        // Update stats
        stats.total_bytes += 4096;
        match route {
            "samefill" => {
                stats.samefill_pages += 1;
                stats.compressed_bytes += 8; // Just store the fill value
            }
            "kernel_zram" => {
                stats.kernel_zram_pages += 1;
                stats.compressed_bytes += 100; // Typical compressed size
            }
            "simd" => {
                stats.simd_pages += 1;
                stats.compressed_bytes += 2000; // Less compressible
            }
            "nvme" => {
                stats.nvme_pages += 1;
                stats.compressed_bytes += 4096; // Stored uncompressed
            }
            _ => {}
        }
    }

    println!();
    println!("Routing Summary");
    println!("---------------");
    println!("  Kernel ZRAM pages: {} (hot tier, 171 GB/s)", stats.kernel_zram_pages);
    println!("  SIMD ZSTD pages:   {} (warm tier, 15 GiB/s)", stats.simd_pages);
    println!("  NVMe direct pages: {} (cold tier, skip compress)", stats.nvme_pages);
    println!("  Same-fill pages:   {} (metadata only)", stats.samefill_pages);
    println!("  Total pages:       {}", stats.total_pages());
    println!("  Compression ratio: {:.1}x", stats.compression_ratio());
    println!();

    println!("CLI Usage");
    println!("---------");
    println!("  # Create tiered device with entropy routing");
    println!("  sudo trueno-ublk create --size 8G \\");
    println!("      --backend tiered \\");
    println!("      --entropy-routing \\");
    println!("      --zram-device /dev/zram0 \\");
    println!("      --entropy-kernel-threshold 6.0 \\");
    println!("      --entropy-skip-threshold 7.5");
    println!();
    println!("  # Use ZSTD for maximum throughput (recommended)");
    println!("  sudo trueno-ublk create --size 8G --algorithm zstd --backend tiered");
    println!();

    println!("Performance Achieved (v3.17.0)");
    println!("------------------------------");
    println!("  Same-fill read:  7.9 GB/s");
    println!("  Kernel ZRAM:     1.3 GB/s (81% of native)");
    println!("  Random 4K IOPS:  666K (--queues 4 --max-perf)");
    println!("  ZSTD compress:   15.4 GiB/s (AVX-512)");
    println!();
}
