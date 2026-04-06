//! Compression benchmark example.
//!
//! Run with: `cargo run --example compress_benchmark --release`
//!
//! This example demonstrates:
//! - SIMD-accelerated compression
//! - Different compression algorithms
//! - Throughput measurement

use trueno_zram_core::benchmark::{generate_test_pages, run_benchmark, DataPattern};
use trueno_zram_core::Algorithm;

fn main() {
    println!("trueno-zram Compression Benchmark");
    println!("==================================\n");

    let page_counts = [100, 1000, 10000];
    let patterns = [
        (DataPattern::Zero, "Zero-filled"),
        (DataPattern::Text, "Text-like"),
        (DataPattern::Random, "Random"),
        (DataPattern::Mixed, "Mixed"),
    ];

    for (pattern, pattern_name) in &patterns {
        println!("Pattern: {pattern_name}");
        println!("{}", "-".repeat(70));
        println!(
            "{:>8} {:>10} {:>12} {:>12} {:>10} {:>10}",
            "Pages", "Algorithm", "Compress", "Decompress", "Ratio", "Backend"
        );

        for &count in &page_counts {
            let pages = generate_test_pages(count, *pattern);

            for algo in [Algorithm::Lz4, Algorithm::Zstd { level: 1 }, Algorithm::Zstd { level: 3 }]
            {
                match run_benchmark(algo, &pages) {
                    Ok(result) => {
                        let compress_gbps = result.compress_throughput() / 1e9;
                        let decompress_gbps = result.decompress_throughput() / 1e9;

                        println!(
                            "{:>8} {:>10} {:>10.2} GB/s {:>10.2} GB/s {:>9.2}x {:>10?}",
                            count,
                            format!("{algo:?}").chars().take(10).collect::<String>(),
                            compress_gbps,
                            decompress_gbps,
                            result.compression_ratio(),
                            result.backend
                        );
                    }
                    Err(e) => {
                        println!("{:>8} {:>10} Error: {}", count, format!("{algo:?}"), e);
                    }
                }
            }
        }
        println!();
    }

    println!("Notes:");
    println!("  - Throughput includes all overhead (memory allocation, etc.)");
    println!("  - Run with --release for accurate performance numbers");
    println!("  - Use --features cuda for GPU acceleration");
}
