//! GPU Decompression Benchmark for G.119 Verification
//!
//! Tests the Sovereign AI hybrid architecture:
//! - CPU compress at 24 GB/s (avoids F082)
//! - GPU decompress at 16 GB/s (F082-safe)
//!
//! G.119 Target: 2TB restore in <60s requires ~34 GB/s sustained
//!
//! Run with: `cargo run --example gpu_decompress_benchmark --features cuda --release`

#[cfg(feature = "cuda")]
fn main() {
    use std::time::Instant;
    use trueno_zram_core::gpu::{HybridConfig, HybridScheduler};

    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("  GPU Decompression Benchmark (G.119 Verification)");
    println!("  Target: 2TB restore in <60s = 34+ GB/s sustained");
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!();

    // Initialize hybrid scheduler with pre-allocated buffers for max batch size
    let config = HybridConfig {
        batch_size: 100_000, // Pre-allocate for largest test case
        gpu_decompress: true,
        target_throughput_gbps: 34.0,
    };

    let mut scheduler = match HybridScheduler::new(config) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Failed to initialize GPU: {e}");
            eprintln!("Make sure CUDA is available.");
            return;
        }
    };

    println!("✓ GPU initialized");
    println!();

    // Test different batch sizes
    let batch_sizes = [1_000, 5_000, 10_000, 50_000, 100_000];

    println!("Phase 1: Compress test data (CPU path)");
    println!("─────────────────────────────────────────────────────────────────────────────");

    let mut compressed_batches: Vec<Vec<Vec<u8>>> = Vec::new();

    for &batch_size in &batch_sizes {
        // Generate test pages
        let pages: Vec<[u8; 4096]> = (0..batch_size)
            .map(|i| {
                let mut page = [0u8; 4096];
                // Mix of patterns: 25% zero, 25% sequential, 50% semi-random
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
        let output_mb: f64 = result.pages.iter().map(|p| p.data.len() as f64).sum::<f64>() / 1e6;
        let throughput_gbps = input_mb / elapsed.as_secs_f64() / 1000.0;
        let ratio = input_mb / output_mb;

        println!(
            "  {:>6} pages: {:>6.1} MB → {:>6.1} MB ({:.2}x) in {:>6.1} ms = {:.2} GB/s",
            batch_size,
            input_mb,
            output_mb,
            ratio,
            elapsed.as_secs_f64() * 1000.0,
            throughput_gbps
        );

        // Store compressed data for decompression test
        compressed_batches.push(result.pages.iter().map(|p| p.data.clone()).collect());
    }

    println!();
    println!("Phase 2: Decompress test data (GPU path - F082-safe)");
    println!("─────────────────────────────────────────────────────────────────────────────");

    // Reset stats for decompression phase
    scheduler.reset_stats();

    for (i, compressed) in compressed_batches.iter().enumerate() {
        let batch_size = batch_sizes[i];

        // Warmup
        if i == 0 {
            println!("  Warmup: {} pages...", batch_size);
            match scheduler.decompress_batch_gpu(compressed) {
                Ok(r) => println!("  Warmup OK: {} pages decompressed", r.pages.len()),
                Err(e) => {
                    eprintln!("  Warmup FAILED: {e}");
                    eprintln!("  (Continuing with main benchmark)");
                }
            }
            scheduler.reset_stats();
        }

        println!("  Running: {} pages...", batch_size);
        let start = Instant::now();
        let result = match scheduler.decompress_batch_gpu(compressed) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("  FAILED: {e}");
                continue;
            }
        };
        let elapsed = start.elapsed();

        let output_mb = (result.pages.len() * 4096) as f64 / 1e6;
        let throughput_gbps = output_mb / elapsed.as_secs_f64() / 1000.0;

        println!(
            "  {:>6} pages: {:>6.1} MB decompressed in {:>6.1} ms = {:.2} GB/s",
            batch_size,
            output_mb,
            elapsed.as_secs_f64() * 1000.0,
            throughput_gbps
        );
    }

    println!();
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("  G.119 Analysis: 2TB LLM Checkpoint Restore");
    println!("═══════════════════════════════════════════════════════════════════════════");

    let stats = scheduler.stats();
    let decompress_gbps = stats.decompress_throughput_gbps();
    let estimate_seconds = 2048.0 / decompress_gbps;

    println!();
    println!("  Measured GPU decompression: {:.2} GB/s", decompress_gbps);
    println!("  Estimated 2TB restore time: {:.1} seconds", estimate_seconds);
    println!();

    if estimate_seconds < 60.0 {
        println!("  ✓ G.119 TARGET MET: 2TB restore in <60s");
    } else {
        println!(
            "  ✗ G.119 TARGET NOT MET: Need {:.2} GB/s, have {:.2} GB/s",
            34.0, decompress_gbps
        );
        println!();
        println!("  Note: PCIe 4.0 x16 theoretical max is ~25 GB/s bidirectional.");
        println!("  For 34+ GB/s, consider:");
        println!("    - Streaming DMA pipeline (overlapped H2D/D2H)");
        println!("    - Multiple GPUs");
        println!("    - CPU+GPU parallel decompression");
    }
    println!();
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("This benchmark requires the 'cuda' feature.");
    eprintln!("Run with: cargo run --example gpu_decompress_benchmark --features cuda --release");
}
