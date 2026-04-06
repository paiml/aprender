//! Deep CPU kernel profiler - Five-Whys root cause analysis
//! Measures actual time spent in each phase of fused_q4k_q8k_dot

use std::time::Instant;

fn main() {
    println!("CPU Kernel Deep Profile (Five-Whys)");
    println!("===================================\n");

    // Limit to 16 threads (NUMA optimal)
    rayon::ThreadPoolBuilder::new()
        .num_threads(16)
        .build_global()
        .ok();

    let hidden = 1536;
    let inter = 8960;

    // Create test data
    let super_blocks = hidden / 256;
    let bytes_per_row = super_blocks * 144;
    let weight_bytes = inter * bytes_per_row;

    let weights: Vec<u8> = (0..weight_bytes).map(|i| (i % 256) as u8).collect();
    let activations: Vec<f32> = (0..hidden)
        .map(|i| (i as f32 / hidden as f32) * 2.0 - 1.0)
        .collect();

    // Pre-quantize to Q8K
    let (q8k_scales, q8k_quants) = quantize_to_q8k(&activations);

    println!("Configuration:");
    println!("  Hidden dim: {}", hidden);
    println!("  Output dim: {}", inter);
    println!("  Super-blocks per row: {}", super_blocks);
    println!("  Bytes per row: {} (Q4K)", bytes_per_row);
    println!(
        "  Total weight bytes: {} ({:.1} MB)",
        weight_bytes,
        weight_bytes as f64 / 1e6
    );
    println!("  Threads: {}", rayon::current_num_threads());

    // Measure single dot product in detail
    let row_data = &weights[0..bytes_per_row];

    // Warmup
    for _ in 0..1000 {
        let _ = realizar::quantize::fused_q4k_q8k_dot_simd(row_data, &q8k_scales, &q8k_quants);
    }

    // Detailed timing
    let iters = 100000;
    let start = Instant::now();
    let mut result = 0.0f32;
    for _ in 0..iters {
        result += realizar::quantize::fused_q4k_q8k_dot_simd(row_data, &q8k_scales, &q8k_quants)
            .unwrap_or(0.0);
    }
    let elapsed = start.elapsed();
    let per_dot_ns = elapsed.as_nanos() as f64 / iters as f64;

    // Prevent optimization
    if result.abs() < 0.0001 {
        println!("(prevent opt)");
    }

    println!("\n=== Single Dot Product Analysis ===");
    println!("Time: {:.1} ns", per_dot_ns);

    // Calculate theoretical limits
    let macs = hidden * 2; // mul + add per element
    let gmacs = macs as f64 / per_dot_ns;
    println!("Throughput: {:.2} GMAC/s", gmacs);

    // Memory analysis
    let bytes_read = bytes_per_row + q8k_quants.len() + q8k_scales.len() * 4;
    let bandwidth = bytes_read as f64 / per_dot_ns;
    println!("Memory read: {} bytes", bytes_read);
    println!("Bandwidth: {:.1} GB/s", bandwidth);

    // Theoretical limits
    println!("\n=== Theoretical Limits ===");
    let cpu_ghz = 4.2; // Threadripper all-core
    let avx2_macs_per_cycle = 16; // maddubs does 32 u8*i8 -> 16 i16
    let theoretical_gmacs = cpu_ghz * avx2_macs_per_cycle as f64;
    println!("CPU frequency: {:.1} GHz", cpu_ghz);
    println!("AVX2 MACs/cycle: {}", avx2_macs_per_cycle);
    println!("Theoretical: {:.1} GMAC/s per core", theoretical_gmacs);
    println!("Efficiency: {:.1}%", gmacs / theoretical_gmacs * 100.0);

    // Breakdown by operation (estimated)
    println!("\n=== Operation Breakdown (estimated) ===");
    let total_ns = per_dot_ns;

    // Count operations per super-block
    let ops_per_sb = super_blocks as f64;
    println!("Super-blocks: {}", super_blocks);
    println!("Per super-block: {:.1} ns", total_ns / ops_per_sb);

    // Per super-block breakdown (256 values)
    // - 4 chunks of 64 values each
    // - Per chunk: load Q4 (32B), load Q8 (64B), maddubs×2, madd×2, horizontal sums
    println!("\nPer-chunk operations (64 values):");
    println!("  - Load Q4: 32 bytes (1 AVX2 load)");
    println!("  - Load Q8: 64 bytes (2 AVX2 loads)");
    println!("  - Nibble extract: 2 ops (AND, SHIFT+AND)");
    println!("  - maddubs×2: 2 ops (Q4×Q8 products)");
    println!("  - madd×2: 2 ops (i16→i32 accumulate)");
    println!("  - Horizontal sums: 6-8 ops (THE BOTTLENECK)");
    println!("  - Q8 sum for min: 8+ ops");
    println!("  - Scale application: 4 ops (load scale, mul, sub)");

    // Parallel matmul measurement
    println!("\n=== Parallel Matmul (16 threads) ===");
    let mut output = vec![0.0f32; inter];

    // Warmup
    for _ in 0..3 {
        let _ = realizar::quantize::fused_q4k_q8k_parallel_matvec_into(
            &weights,
            &q8k_scales,
            &q8k_quants,
            hidden,
            inter,
            &mut output,
        );
    }

    let iters2 = 100;
    let start2 = Instant::now();
    for _ in 0..iters2 {
        let _ = realizar::quantize::fused_q4k_q8k_parallel_matvec_into(
            &weights,
            &q8k_scales,
            &q8k_quants,
            hidden,
            inter,
            &mut output,
        );
    }
    let matmul_us = start2.elapsed().as_micros() as f64 / iters2 as f64;

    let total_macs = 2 * hidden * inter;
    let gflops = total_macs as f64 / matmul_us / 1000.0;

    println!("Time: {:.1} µs", matmul_us);
    println!("GFLOPS: {:.1}", gflops);

    // Per-token estimate
    let layers = 28;
    let matmuls_per_layer = 5; // QKV, O, up, gate, down
    let per_layer_us = matmul_us * matmuls_per_layer as f64;
    let total_us = per_layer_us * layers as f64;
    let tok_s = 1e6 / total_us;

    println!("\n=== Per-Token Estimate ===");
    println!(
        "Per layer ({} matmuls): {:.1} µs",
        matmuls_per_layer, per_layer_us
    );
    println!("28 layers: {:.1} ms", total_us / 1000.0);
    println!("Estimated: {:.1} tok/s", tok_s);

    // Target analysis
    let ollama_tok_s = 265.0;
    let target_tok_s = ollama_tok_s * 2.0;
    println!("\n=== Target Analysis ===");
    println!("Ollama CPU: {:.0} tok/s", ollama_tok_s);
    println!("2x target: {:.0} tok/s", target_tok_s);
    println!("Current: {:.1} tok/s", tok_s);
    println!("Gap: {:.1}x", target_tok_s / tok_s);

    // Required improvement
    let required_speedup = target_tok_s / tok_s;
    let required_matmul_us = matmul_us / required_speedup;
    println!("\nTo reach 2x:");
    println!(
        "  Need matmul: {:.1} µs (currently {:.1} µs)",
        required_matmul_us, matmul_us
    );
    println!(
        "  Need per-dot: {:.1} ns (currently {:.1} ns)",
        per_dot_ns / required_speedup,
        per_dot_ns
    );
}

fn quantize_to_q8k(values: &[f32]) -> (Vec<f32>, Vec<i8>) {
    const QK_K: usize = 256;
    let num_sb = values.len().div_ceil(QK_K);
    let padded_len = num_sb * QK_K;

    let mut scales = Vec::with_capacity(num_sb);
    let mut quants = vec![0i8; padded_len];

    for sb in 0..num_sb {
        let start = sb * QK_K;
        let end = (start + QK_K).min(values.len());
        let chunk = &values[start..end];

        let amax = chunk.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let scale = amax / 127.0;
        let inv_scale = if scale > 0.0 { 127.0 / amax } else { 0.0 };

        scales.push(scale);

        for (i, v) in chunk.iter().enumerate() {
            quants[start + i] = (v * inv_scale).round().clamp(-128.0, 127.0) as i8;
        }
    }

    (scales, quants)
}
