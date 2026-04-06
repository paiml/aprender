//! PMAT Benchmark: Per-Page vs Batched Compression
//!
//! Compares `PageStore` (per-page) vs `BatchedPageStore` (batched) compression performance.
//! Target: >10 GB/s sequential write throughput with batched compression.
//!
//! Run with:
//! ```bash
//! cargo run --example batched_benchmark -p trueno-ublk
//! ```
//!
//! Run with CUDA:
//! ```bash
//! cargo run --example batched_benchmark -p trueno-ublk --features cuda
//! ```

use std::time::{Duration, Instant};
use trueno_zram_core::{Algorithm, PAGE_SIZE};

// Import from the trueno-ublk crate
use trueno_ublk::daemon::{BatchConfig, BatchedPageStore, PageStore};

#[allow(clippy::too_many_lines)]
fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║     PMAT Benchmark: Per-Page vs Batched Compression              ║");
    println!("║                  trueno-ublk Integration                         ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    // System info
    let num_cpus = std::thread::available_parallelism().map(std::num::NonZero::get).unwrap_or(1);
    println!("CPU Cores: {num_cpus}");

    #[cfg(feature = "cuda")]
    {
        if trueno_zram_core::gpu::gpu_available() {
            println!("GPU: Available (CUDA enabled)");
        } else {
            println!("GPU: Not available");
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        println!("GPU: Disabled (compile with --features cuda)");
    }

    println!("\nTarget: >10 GB/s sequential write throughput with batching\n");

    // Test configurations
    let page_counts = [100, 500, 1000, 2000, 5000, 10000, 20000];

    // =========================================================================
    // Per-Page Compression Benchmark (PageStore)
    // =========================================================================
    println!("═══════════════════════════════════════════════════════════════════");
    println!("                 Per-Page Compression (PageStore)                   ");
    println!("═══════════════════════════════════════════════════════════════════\n");

    let mut per_page_results = Vec::new();

    for &count in &page_counts {
        let pages = generate_mixed_pages(count);
        let result = benchmark_per_page(&pages);

        let throughput_indicator = if result.throughput_gbps >= 10.0 { "✓" } else { "✗" };
        println!(
            "  {:>6} pages: {:>6.2} GB/s, {:>5.2}x ratio, {:>7.2} ms  {} >10 GB/s",
            count, result.throughput_gbps, result.ratio, result.time_ms, throughput_indicator
        );
        per_page_results.push((count, result));
    }

    // =========================================================================
    // Batched Compression Benchmark (BatchedPageStore)
    // =========================================================================
    println!("\n═══════════════════════════════════════════════════════════════════");
    println!("             Batched Compression (BatchedPageStore)                 ");
    println!("           batch_threshold=1000, flush_timeout=10ms                 ");
    println!("═══════════════════════════════════════════════════════════════════\n");

    let mut batched_results = Vec::new();

    for &count in &page_counts {
        let pages = generate_mixed_pages(count);
        let result = benchmark_batched(&pages, 1000);
        batched_results.push((count, result.clone()));

        let throughput_indicator = if result.throughput_gbps >= 10.0 { "✓" } else { "✗" };
        let backend = match result.backend_used.as_str() {
            "simd" => "SIMD",
            "simd_parallel" => "SIMD||",
            "gpu" => "GPU",
            _ => "?",
        };
        println!(
            "  {:>6} pages: {:>6.2} GB/s, {:>5.2}x ratio, {:>7.2} ms  {} >10 GB/s [{}]",
            count,
            result.throughput_gbps,
            result.ratio,
            result.time_ms,
            throughput_indicator,
            backend
        );
    }

    // =========================================================================
    // Batched with Lower Threshold (for comparison)
    // =========================================================================
    println!("\n═══════════════════════════════════════════════════════════════════");
    println!("             Batched Compression (threshold=100)                    ");
    println!("           Lower threshold for latency-sensitive workloads          ");
    println!("═══════════════════════════════════════════════════════════════════\n");

    for &count in &page_counts {
        let pages = generate_mixed_pages(count);
        let result = benchmark_batched(&pages, 100);

        let throughput_indicator = if result.throughput_gbps >= 10.0 { "✓" } else { "✗" };
        println!(
            "  {:>6} pages: {:>6.2} GB/s, {:>5.2}x ratio, {:>7.2} ms  {} >10 GB/s",
            count, result.throughput_gbps, result.ratio, result.time_ms, throughput_indicator
        );
    }

    // =========================================================================
    // Comparison Table
    // =========================================================================
    println!("\n═══════════════════════════════════════════════════════════════════");
    println!("                      Performance Comparison                        ");
    println!("═══════════════════════════════════════════════════════════════════\n");

    println!("┌─────────┬───────────────────┬───────────────────┬──────────────┐");
    println!("│ Pages   │ Per-Page (GB/s)   │ Batched (GB/s)    │ Speedup      │");
    println!("├─────────┼───────────────────┼───────────────────┼──────────────┤");

    for ((count1, per_page), (count2, batched)) in
        per_page_results.iter().zip(batched_results.iter())
    {
        assert_eq!(count1, count2);
        let speedup = batched.throughput_gbps / per_page.throughput_gbps.max(0.001);
        let speedup_indicator = if speedup > 1.5 {
            "⚡"
        } else if speedup > 1.0 {
            "↑"
        } else {
            "  "
        };
        println!(
            "│ {:>6}  │ {:>15.2}   │ {:>15.2}   │ {:>8.2}x {} │",
            count1, per_page.throughput_gbps, batched.throughput_gbps, speedup, speedup_indicator
        );
    }
    println!("└─────────┴───────────────────┴───────────────────┴──────────────┘");

    // =========================================================================
    // Summary
    // =========================================================================
    println!("\n═══════════════════════════════════════════════════════════════════");
    println!("                           Summary                                  ");
    println!("═══════════════════════════════════════════════════════════════════\n");

    // Find best batched result
    let best_batched = batched_results
        .iter()
        .max_by(|a, b| a.1.throughput_gbps.partial_cmp(&b.1.throughput_gbps).unwrap())
        .unwrap();

    let achieved_target = best_batched.1.throughput_gbps >= 10.0;

    println!(
        "Best batched throughput: {:.2} GB/s ({} pages)",
        best_batched.1.throughput_gbps, best_batched.0
    );
    println!("Target: >10 GB/s: {}", if achieved_target { "ACHIEVED ✓" } else { "NOT YET ✗" });

    if achieved_target {
        println!("\n✓ BatchedPageStore meets the >10 GB/s target!");
        println!("  - Use --batched flag with trueno-ublk create");
        println!("  - Optimal for sequential write workloads");
    } else {
        println!("\n⚠ BatchedPageStore did not meet >10 GB/s target");
        println!("  - Consider GPU acceleration (--features cuda)");
        println!("  - Or adjust batch threshold for your workload");
    }

    // Backend selection analysis
    println!("\n Backend Selection Rules:");
    println!("  - < 100 pages:  SIMD sequential (lowest latency)");
    println!("  - 100-999:      SIMD parallel with rayon");
    println!("  - >= 1000:      GPU batch compression (if CUDA available)");

    // Calculate memory pressure stats
    let total_pages: usize = page_counts.iter().sum();
    let total_bytes = total_pages * PAGE_SIZE;
    let total_mb = total_bytes as f64 / 1024.0 / 1024.0;
    println!("\nBenchmark Stats:");
    println!("  Total pages processed: {}", total_pages * 3); // 3 benchmark runs
    println!("  Total data: {:.2} MB", total_mb * 3.0);
}

#[derive(Clone)]
struct BenchmarkResult {
    throughput_gbps: f64,
    ratio: f64,
    time_ms: f64,
    backend_used: String,
}

fn benchmark_per_page(pages: &[[u8; PAGE_SIZE]]) -> BenchmarkResult {
    let mut store = PageStore::new(1024 * 1024 * 1024, Algorithm::Lz4);

    let start = Instant::now();

    for (i, page) in pages.iter().enumerate() {
        let sector = (i * 8) as u64; // 8 sectors per page
        store.store(sector, page).expect("store failed");
    }

    let elapsed = start.elapsed();
    let stats = store.stats();

    let input_bytes = pages.len() * PAGE_SIZE;
    let throughput_gbps = input_bytes as f64 / elapsed.as_secs_f64() / 1e9;
    let ratio = if stats.bytes_compressed > 0 {
        stats.bytes_stored as f64 / stats.bytes_compressed as f64
    } else {
        1.0
    };

    BenchmarkResult {
        throughput_gbps,
        ratio,
        time_ms: elapsed.as_secs_f64() * 1000.0,
        backend_used: "simd".to_string(),
    }
}

fn benchmark_batched(pages: &[[u8; PAGE_SIZE]], batch_threshold: usize) -> BenchmarkResult {
    let config = BatchConfig {
        batch_threshold,
        flush_timeout: Duration::from_millis(10),
        gpu_batch_size: 4000,
    };
    let store = BatchedPageStore::with_config(Algorithm::Lz4, config);

    let start = Instant::now();

    for (i, page) in pages.iter().enumerate() {
        let sector = (i * 8) as u64; // 8 sectors per page
        store.store(sector, page).expect("store failed");
    }

    // Flush any remaining pages
    store.flush_batch().expect("flush failed");

    let elapsed = start.elapsed();
    let stats = store.stats();

    let input_bytes = pages.len() * PAGE_SIZE;
    let throughput_gbps = input_bytes as f64 / elapsed.as_secs_f64() / 1e9;
    let ratio = if stats.bytes_compressed > 0 {
        stats.bytes_stored as f64 / stats.bytes_compressed as f64
    } else {
        1.0
    };

    // Determine which backend was used
    let backend = if stats.gpu_pages > 0 {
        "gpu"
    } else if pages.len() >= 100 {
        "simd_parallel"
    } else {
        "simd"
    };

    BenchmarkResult {
        throughput_gbps,
        ratio,
        time_ms: elapsed.as_secs_f64() * 1000.0,
        backend_used: backend.to_string(),
    }
}

fn generate_mixed_pages(count: usize) -> Vec<[u8; PAGE_SIZE]> {
    (0..count)
        .map(|i| {
            let mut page = [0u8; PAGE_SIZE];
            match i % 5 {
                0 => {
                    // Zero page (highly compressible) - 20%
                }
                1 => {
                    // Sequential pattern (compressible) - 20%
                    for (j, byte) in page.iter_mut().enumerate() {
                        *byte = (j % 256) as u8;
                    }
                }
                2 => {
                    // Repeating short pattern - 20%
                    for (j, byte) in page.iter_mut().enumerate() {
                        *byte = [0xAA, 0xBB, 0xCC, 0xDD][j % 4];
                    }
                }
                3 => {
                    // Text-like data (English-ish distribution) - 20%
                    for (j, byte) in page.iter_mut().enumerate() {
                        // Weighted towards ASCII letters
                        let base = ((i * 31 + j * 17) % 52) as u8;
                        *byte = if base < 26 { b'a' + base } else { b'A' + (base - 26) };
                    }
                }
                _ => {
                    // Semi-random (less compressible) - 20%
                    let mut rng = (i as u64).wrapping_mul(6_364_136_223_846_793_005);
                    for byte in &mut page {
                        rng = rng.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                        *byte = (rng >> 33) as u8;
                    }
                }
            }
            page
        })
        .collect()
}
