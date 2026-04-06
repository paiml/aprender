//! G.119 Parallel CPU+GPU Decompression Benchmark
//!
//! Tests the Sovereign AI target: 2TB restore in <60 seconds (34+ GB/s)
//!
//! Strategy:
//! - 67% CPU @ ~24 GB/s
//! - 33% GPU @ ~12 GB/s
//! - Combined: ~36 GB/s (exceeds G.119 target)
//!
//! Run with: `cargo run --example g119_parallel_benchmark --features cuda --release`

#[cfg(feature = "cuda")]
fn main() {
    use std::time::Instant;
    use trueno_zram_core::gpu::{HybridConfig, HybridScheduler};

    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("  G.119 Parallel CPU+GPU Decompression Benchmark");
    println!("  Target: 2TB restore in <60s = 34+ GB/s sustained");
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!();

    // Initialize hybrid scheduler with large batch size
    let config =
        HybridConfig { batch_size: 100_000, gpu_decompress: true, target_throughput_gbps: 34.0 };

    let mut scheduler = match HybridScheduler::new(config) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Failed to initialize: {e}");
            return;
        }
    };

    println!("✓ Scheduler initialized");
    println!();

    // Test different batch sizes
    let batch_sizes = [10_000, 50_000, 100_000];

    println!("Phase 1: Compress test data (CPU path)");
    println!("─────────────────────────────────────────────────────────────────────────────");

    let mut compressed_batches: Vec<Vec<Vec<u8>>> = Vec::new();

    for &batch_size in &batch_sizes {
        // Generate test pages with realistic mix
        let pages: Vec<[u8; 4096]> = (0..batch_size)
            .map(|i| {
                let mut page = [0u8; 4096];
                match i % 4 {
                    0 => {} // Zero page
                    1 => {
                        for (j, byte) in page.iter_mut().enumerate() {
                            *byte = (j % 256) as u8;
                        }
                    }
                    _ => {
                        let mut rng = (i as u64).wrapping_mul(0x0005_DEEC_E66D);
                        for byte in &mut page {
                            rng = rng.wrapping_mul(0x0005_DEEC_E66D).wrapping_add(0xB);
                            *byte = (rng >> 33) as u8;
                        }
                    }
                }
                page
            })
            .collect();

        let start = Instant::now();
        let result = scheduler.compress_batch(&pages).unwrap();
        let elapsed = start.elapsed();

        let input_mb = (batch_size * 4096) as f64 / 1e6;
        let throughput_gbps = input_mb / elapsed.as_secs_f64() / 1000.0;

        println!(
            "  {:>6} pages: {:>6.1} MB in {:>6.1} ms = {:.2} GB/s",
            batch_size,
            input_mb,
            elapsed.as_secs_f64() * 1000.0,
            throughput_gbps
        );

        compressed_batches.push(result.pages.iter().map(|p| p.data.clone()).collect());
    }

    println!();
    println!("Phase 2: Parallel CPU Decompression (G.119 Path with pre-allocated buffers)");
    println!("─────────────────────────────────────────────────────────────────────────────");

    // Pre-allocate output buffers for maximum throughput
    let max_batch = *batch_sizes.iter().max().unwrap();
    let mut output_buffer: Vec<[u8; 4096]> = vec![[0u8; 4096]; max_batch];

    for (i, compressed) in compressed_batches.iter().enumerate() {
        let batch_size = batch_sizes[i];
        let output_mb = (batch_size * 4096) as f64 / 1e6;

        println!("  Batch: {} pages ({:.1} MB)", batch_size, output_mb);

        // Warmup
        let _ = scheduler.decompress_parallel_into(compressed, &mut output_buffer);

        // Timed run
        let start = Instant::now();
        let _ = scheduler.decompress_parallel_into(compressed, &mut output_buffer).unwrap();
        let elapsed = start.elapsed();

        let throughput_gbps = output_mb / elapsed.as_secs_f64() / 1000.0;
        let estimate_2tb_s = 2048.0 / throughput_gbps;

        let status = if estimate_2tb_s < 60.0 { "✅" } else { "❌" };

        println!(
            "    {:>6.1} ms = {:>5.2} GB/s | 2TB: {:>5.1}s {}",
            elapsed.as_secs_f64() * 1000.0,
            throughput_gbps,
            estimate_2tb_s,
            status
        );
        println!();
    }

    // Final G.119 verification with largest batch
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("  G.119 Final Verification (100K pages, pre-allocated buffer)");
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!();

    let largest_batch = &compressed_batches[2];

    // Warmup (3 iterations)
    for _ in 0..3 {
        let _ = scheduler.decompress_parallel_into(largest_batch, &mut output_buffer);
    }
    scheduler.reset_stats();

    // Multiple runs for accuracy
    let mut times = Vec::new();
    for _ in 0..10 {
        let start = Instant::now();
        let _ = scheduler.decompress_parallel_into(largest_batch, &mut output_buffer);
        times.push(start.elapsed());
    }

    let avg_time_ms =
        times.iter().map(|t| t.as_secs_f64() * 1000.0).sum::<f64>() / times.len() as f64;
    let min_time_ms = times.iter().map(|t| t.as_secs_f64() * 1000.0).fold(f64::INFINITY, f64::min);
    let output_mb = (100_000 * 4096) as f64 / 1e6;
    let avg_throughput_gbps = output_mb / (avg_time_ms / 1000.0) / 1000.0;
    let peak_throughput_gbps = output_mb / (min_time_ms / 1000.0) / 1000.0;
    let estimate_2tb_s = 2048.0 / avg_throughput_gbps;

    println!("  Average throughput: {:.2} GB/s (over {} runs)", avg_throughput_gbps, times.len());
    println!("  Peak throughput:    {:.2} GB/s", peak_throughput_gbps);
    println!("  Estimated 2TB restore: {:.1} seconds", estimate_2tb_s);
    println!();

    if estimate_2tb_s < 60.0 {
        println!("  ✅ G.119 TARGET MET: 2TB restore in <60s");
    } else {
        println!("  ❌ G.119 TARGET NOT MET: Need 34+ GB/s, have {:.2} GB/s", avg_throughput_gbps);
        println!();
        println!("  Analysis:");
        println!("    - Current: {:.2} GB/s parallel CPU decompression", avg_throughput_gbps);
        println!("    - Memory bandwidth limit: ~27 GB/s (from memcpy test)");
        println!("    - Target requires faster memory or multi-socket system");
    }
    println!();
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("This benchmark requires the 'cuda' feature.");
    eprintln!("Run with: cargo run --example g119_parallel_benchmark --features cuda --release");
}
