//! Quick single-threaded decompression throughput test

use std::time::Instant;
use trueno_zram_core::lz4::{compress, decompress_simd};

const PAGE_SIZE: usize = 4096;
const NUM_PAGES: usize = 10_000;

fn main() {
    println!("Single-threaded LZ4 decompression throughput test\n");

    // Create mixed test data (same as benchmark)
    let pages: Vec<[u8; PAGE_SIZE]> = (0..NUM_PAGES)
        .map(|i| {
            let mut page = [0u8; PAGE_SIZE];
            match i % 4 {
                0 => {} // Zero
                1 => {
                    for (j, b) in page.iter_mut().enumerate() {
                        *b = (j % 256) as u8;
                    }
                }
                _ => {
                    let mut rng = (i as u64).wrapping_mul(0x0005_DEEC_E66D);
                    for b in &mut page {
                        rng = rng.wrapping_mul(0x0005_DEEC_E66D).wrapping_add(0xB);
                        *b = (rng >> 33) as u8;
                    }
                }
            }
            page
        })
        .collect();

    // Compress all pages
    let compressed: Vec<Vec<u8>> = pages.iter().map(|p| compress(p).unwrap()).collect();

    let total_compressed: usize = compressed.iter().map(Vec::len).sum();
    println!(
        "Compressed {} pages: {} → {} bytes ({:.2}x ratio)\n",
        NUM_PAGES,
        NUM_PAGES * PAGE_SIZE,
        total_compressed,
        (NUM_PAGES * PAGE_SIZE) as f64 / total_compressed as f64
    );

    // Pre-allocate output
    let mut output = [0u8; PAGE_SIZE];

    // Warmup
    for c in compressed.iter().take(100) {
        decompress_simd(c, &mut output).unwrap();
    }

    // Time single-threaded decompression
    let start = Instant::now();
    for c in &compressed {
        decompress_simd(c, &mut output).unwrap();
    }
    let elapsed = start.elapsed();

    let bytes = NUM_PAGES * PAGE_SIZE;
    let gbps = bytes as f64 / elapsed.as_secs_f64() / 1e9;
    let pages_per_sec = NUM_PAGES as f64 / elapsed.as_secs_f64();
    let ns_per_page = elapsed.as_nanos() as f64 / NUM_PAGES as f64;

    println!("Results:");
    println!("  {} pages in {:.1} ms", NUM_PAGES, elapsed.as_secs_f64() * 1000.0);
    println!("  Throughput: {gbps:.2} GB/s (single-threaded)");
    println!("  Pages/sec: {pages_per_sec:.0}");
    println!("  ns/page: {ns_per_page:.0}");

    println!("\nProjections:");
    println!("  With 24 threads (perfect scaling): {:.1} GB/s", gbps * 24.0);
    println!("  With 48 threads (perfect scaling): {:.1} GB/s", gbps * 48.0);

    // Check CPU features
    #[cfg(target_arch = "x86_64")]
    {
        println!("\nCPU features:");
        println!("  AVX2: {}", std::arch::is_x86_feature_detected!("avx2"));
        println!("  AVX-512: {}", std::arch::is_x86_feature_detected!("avx512f"));
    }
}
