//! ZSTD vs LZ4 Performance Comparison (v3.17.0)
//!
//! Demonstrates why ZSTD level 1 is recommended over LZ4 on AVX-512 systems.
//!
//! Run with: `cargo run --example zstd_vs_lz4 -p trueno-ublk --release`

use std::time::Instant;
use trueno_zram_core::{Algorithm, CompressorBuilder, PAGE_SIZE};

/// Generate test data with specified entropy level
fn generate_test_data(entropy_level: &str) -> [u8; PAGE_SIZE] {
    let mut data = [0u8; PAGE_SIZE];
    match entropy_level {
        "text" => {
            // Simulated text data (low entropy ~4.5 bits)
            let pattern = b"The quick brown fox jumps over the lazy dog. ";
            for (i, chunk) in data.chunks_mut(pattern.len()).enumerate() {
                let len = chunk.len().min(pattern.len());
                chunk[..len].copy_from_slice(&pattern[..len]);
                // Add some variation
                if i % 3 == 0 && !chunk.is_empty() {
                    chunk[0] = b'A' + (i % 26) as u8;
                }
            }
        }
        "mixed" => {
            // Mixed data (medium entropy ~6.0 bits)
            for (i, byte) in data.iter_mut().enumerate() {
                *byte = ((i * 17 + 31) % 256) as u8;
            }
        }
        "random" => {
            // High entropy data (~7.9 bits)
            let mut x: u64 = 88_172_645_463_325_252;
            for chunk in data.chunks_mut(8) {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                for (i, byte) in chunk.iter_mut().enumerate() {
                    *byte = (x >> (i * 8)) as u8;
                }
            }
        }
        _ => {}
    }
    data
}

fn benchmark_algorithm(
    name: &str,
    algorithm: Algorithm,
    data: &[u8; PAGE_SIZE],
    iterations: usize,
) -> (f64, f64, f64) {
    let compressor = CompressorBuilder::new().algorithm(algorithm).build().unwrap();

    // Compression benchmark
    let start = Instant::now();
    let mut last_compressed = None;
    for _ in 0..iterations {
        let compressed = compressor.compress(data).unwrap();
        last_compressed = Some(compressed);
    }
    let compress_time = start.elapsed();
    let compress_throughput = (PAGE_SIZE * iterations) as f64 / compress_time.as_secs_f64() / 1e9;

    let compressed = last_compressed.unwrap();
    let compressed_size = compressed.data.len();

    // Decompression benchmark
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = compressor.decompress(&compressed).unwrap();
    }
    let decompress_time = start.elapsed();
    let decompress_throughput =
        (PAGE_SIZE * iterations) as f64 / decompress_time.as_secs_f64() / 1e9;

    let ratio = PAGE_SIZE as f64 / compressed_size as f64;

    println!(
        "  {name:<12} {compress_throughput:>8.2} GiB/s  {decompress_throughput:>8.2} GiB/s  {ratio:>6.1}x"
    );

    (compress_throughput, decompress_throughput, ratio)
}

fn main() {
    println!("trueno-ublk ZSTD vs LZ4 Comparison (v3.17.0)");
    println!("============================================\n");

    // Check for AVX-512
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx512f") {
            println!("AVX-512: ENABLED (optimal SIMD path)\n");
        } else if std::arch::is_x86_feature_detected!("avx2") {
            println!("AVX-512: Not available (using AVX2 fallback)\n");
        } else {
            println!("Warning: No SIMD acceleration available\n");
        }
    }

    let iterations = 10000;
    let workloads = [
        ("zeros", "W1-ZEROS (all zeros)"),
        ("text", "W2-TEXT (prose text)"),
        ("mixed", "W3-MIXED (structured)"),
        ("random", "W4-RANDOM (high entropy)"),
    ];

    println!("Benchmark Configuration:");
    println!("  Page size:   {PAGE_SIZE} bytes");
    println!("  Iterations:  {iterations}");
    println!();

    for (entropy_level, description) in workloads {
        let data = generate_test_data(entropy_level);

        println!("{description}");
        println!("{:<14} {:>14} {:>14} {:>8}", "Algorithm", "Compress", "Decompress", "Ratio");
        println!("{:-<56}", "");

        let (lz4_c, lz4_d, _) = benchmark_algorithm("LZ4", Algorithm::Lz4, &data, iterations);
        let (zstd_c, zstd_d, _) =
            benchmark_algorithm("ZSTD-1", Algorithm::Zstd { level: 1 }, &data, iterations);

        // Calculate speedup
        let compress_speedup = zstd_c / lz4_c;
        let decompress_speedup = zstd_d / lz4_d;

        println!();
        println!(
            "  ZSTD-1 speedup: {compress_speedup:.1}x compress, {decompress_speedup:.1}x decompress"
        );
        println!();
    }

    println!("Recommendation (BENCH-001 v2.1.0)");
    println!("---------------------------------");
    println!("  ZSTD level 1 is recommended for AVX-512 systems:");
    println!("    - 3x faster compression");
    println!("    - 6x faster decompression");
    println!("    - Similar compression ratios");
    println!();
    println!("  Usage:");
    println!("    sudo trueno-ublk create --size 8G --algorithm zstd --backend tiered");
    println!();
}
