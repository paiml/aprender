//! Profile FFN layer to find the 71% roofline gap
//!
//! Breaks down timing of each FFN component:
//! - Q8 quantization of input
//! - Up projection (matvec)
//! - Gate projection (matvec)
//! - SiLU/GELU activation
//! - Down projection (matvec)

use realizar::quantize::fused_q4_0_q8_0_parallel_matvec;
use std::time::Instant;

const WARMUP: usize = 5;
const ITERATIONS: usize = 20;

fn main() {
    println!("=== FFN Layer Profiling ===\n");

    // Use Qwen2.5-Coder dimensions (smaller, faster iteration)
    let hidden_dim = 896;
    let intermediate_dim = 4864;
    let num_layers = 24;

    // Also test Phi-2 dimensions
    let phi2_hidden = 2560;
    let phi2_intermediate = 10240;
    let phi2_layers = 32;

    println!("Model: Qwen2.5-Coder-0.5B");
    println!("  hidden_dim: {}", hidden_dim);
    println!("  intermediate_dim: {}", intermediate_dim);
    println!("  num_layers: {}", num_layers);

    // Calculate weight sizes
    let q4_block_bytes = 18; // 2 byte scale + 16 bytes for 32 values
    let up_blocks = (hidden_dim * intermediate_dim) / 32;
    let down_blocks = (intermediate_dim * hidden_dim) / 32;
    let up_size = up_blocks * q4_block_bytes;
    let down_size = down_blocks * q4_block_bytes;

    println!("\nWeight sizes (Q4_0):");
    println!("  Up projection:   {:.2} MB", up_size as f64 / 1e6);
    println!("  Gate projection: {:.2} MB", up_size as f64 / 1e6);
    println!("  Down projection: {:.2} MB", down_size as f64 / 1e6);
    println!(
        "  Total per layer: {:.2} MB",
        (2 * up_size + down_size) as f64 / 1e6
    );

    // Create fake Q4_0 weights (properly formatted)
    let mut up_weights = vec![0u8; up_blocks * q4_block_bytes];
    let mut gate_weights = vec![0u8; up_blocks * q4_block_bytes];
    let mut down_weights = vec![0u8; down_blocks * q4_block_bytes];

    // Initialize with valid scales and random nibbles
    for block in 0..up_blocks {
        let offset = block * q4_block_bytes;
        // Set scale to small value (f16 bits for ~0.01)
        up_weights[offset] = 0x66;
        up_weights[offset + 1] = 0x23;
        gate_weights[offset] = 0x66;
        gate_weights[offset + 1] = 0x23;
        // Fill nibbles
        for i in 2..q4_block_bytes {
            up_weights[offset + i] = ((block + i) % 256) as u8;
            gate_weights[offset + i] = ((block * 3 + i) % 256) as u8;
        }
    }
    for block in 0..down_blocks {
        let offset = block * q4_block_bytes;
        down_weights[offset] = 0x66;
        down_weights[offset + 1] = 0x23;
        for i in 2..q4_block_bytes {
            down_weights[offset + i] = ((block * 7 + i) % 256) as u8;
        }
    }

    // Create input vector
    let input: Vec<f32> = (0..hidden_dim).map(|i| (i as f32 * 0.01).sin()).collect();
    let intermediate_input: Vec<f32> = (0..intermediate_dim)
        .map(|i| (i as f32 * 0.01).cos())
        .collect();

    println!("\n--- Warming up ({} iterations) ---", WARMUP);
    for _ in 0..WARMUP {
        let _ = fused_q4_0_q8_0_parallel_matvec(&up_weights, &input, hidden_dim, intermediate_dim);
    }

    println!("\n--- Profiling {} iterations ---\n", ITERATIONS);

    // Profile Up projection
    let mut up_times = Vec::with_capacity(ITERATIONS);
    for _ in 0..ITERATIONS {
        let start = Instant::now();
        let _ = fused_q4_0_q8_0_parallel_matvec(&up_weights, &input, hidden_dim, intermediate_dim);
        up_times.push(start.elapsed().as_micros());
    }

    // Profile Gate projection
    let mut gate_times = Vec::with_capacity(ITERATIONS);
    for _ in 0..ITERATIONS {
        let start = Instant::now();
        let _ =
            fused_q4_0_q8_0_parallel_matvec(&gate_weights, &input, hidden_dim, intermediate_dim);
        gate_times.push(start.elapsed().as_micros());
    }

    // Profile Down projection
    let mut down_times = Vec::with_capacity(ITERATIONS);
    for _ in 0..ITERATIONS {
        let start = Instant::now();
        let _ = fused_q4_0_q8_0_parallel_matvec(
            &down_weights,
            &intermediate_input,
            intermediate_dim,
            hidden_dim,
        );
        down_times.push(start.elapsed().as_micros());
    }

    // Profile SiLU activation
    let mut silu_times = Vec::with_capacity(ITERATIONS);
    let gate_result: Vec<f32> = (0..intermediate_dim).map(|i| i as f32 * 0.001).collect();
    let mut up_result: Vec<f32> = (0..intermediate_dim).map(|i| i as f32 * 0.002).collect();
    for _ in 0..ITERATIONS {
        let start = Instant::now();
        for i in 0..intermediate_dim {
            let silu = gate_result[i] / (1.0 + (-gate_result[i]).exp());
            up_result[i] *= silu;
        }
        silu_times.push(start.elapsed().as_micros());
    }

    // Calculate statistics
    fn stats(times: &mut [u128]) -> (u128, u128, u128) {
        times.sort_unstable();
        let min = times[0];
        let median = times[times.len() / 2];
        let avg = times.iter().sum::<u128>() / times.len() as u128;
        (min, median, avg)
    }

    let (up_min, up_med, up_avg) = stats(&mut up_times);
    let (gate_min, gate_med, gate_avg) = stats(&mut gate_times);
    let (down_min, down_med, down_avg) = stats(&mut down_times);
    let (silu_min, silu_med, silu_avg) = stats(&mut silu_times);

    println!("Operation Timing (µs):");
    println!("                        min      median      avg");
    println!(
        "  Up projection:    {:>7}     {:>7}    {:>7}",
        up_min, up_med, up_avg
    );
    println!(
        "  Gate projection:  {:>7}     {:>7}    {:>7}",
        gate_min, gate_med, gate_avg
    );
    println!(
        "  Down projection:  {:>7}     {:>7}    {:>7}",
        down_min, down_med, down_avg
    );
    println!(
        "  SiLU activation:  {:>7}     {:>7}    {:>7}",
        silu_min, silu_med, silu_avg
    );

    let total_ffn = up_med + gate_med + down_med + silu_med;
    println!(
        "\n  Total FFN (median): {} µs ({:.2} ms)",
        total_ffn,
        total_ffn as f64 / 1000.0
    );
    println!(
        "  Per {} layers:      {} µs ({:.2} ms)",
        num_layers,
        total_ffn * num_layers as u128,
        total_ffn as f64 * num_layers as f64 / 1000.0
    );

    // Breakdown percentages
    println!("\nFFN Time Breakdown:");
    println!(
        "  Up projection:   {:>5.1}%",
        up_med as f64 / total_ffn as f64 * 100.0
    );
    println!(
        "  Gate projection: {:>5.1}%",
        gate_med as f64 / total_ffn as f64 * 100.0
    );
    println!(
        "  Down projection: {:>5.1}%",
        down_med as f64 / total_ffn as f64 * 100.0
    );
    println!(
        "  SiLU activation: {:>5.1}%",
        silu_med as f64 / total_ffn as f64 * 100.0
    );

    // Roofline analysis per operation
    println!("\n=== Roofline Analysis (@ 30 GB/s) ===");
    let bandwidth_gbps = 30.0;

    let up_theoretical_ms = (up_size as f64 / 1e9) / bandwidth_gbps * 1000.0;
    let gate_theoretical_ms = (up_size as f64 / 1e9) / bandwidth_gbps * 1000.0;
    let down_theoretical_ms = (down_size as f64 / 1e9) / bandwidth_gbps * 1000.0;

    let up_actual_ms = up_med as f64 / 1000.0;
    let gate_actual_ms = gate_med as f64 / 1000.0;
    let down_actual_ms = down_med as f64 / 1000.0;

    println!("\nPer-operation efficiency:");
    println!("                     Theoretical    Actual    Efficiency");
    println!(
        "  Up projection:     {:>6.2} ms    {:>6.2} ms    {:>5.1}%",
        up_theoretical_ms,
        up_actual_ms,
        up_theoretical_ms / up_actual_ms * 100.0
    );
    println!(
        "  Gate projection:   {:>6.2} ms    {:>6.2} ms    {:>5.1}%",
        gate_theoretical_ms,
        gate_actual_ms,
        gate_theoretical_ms / gate_actual_ms * 100.0
    );
    println!(
        "  Down projection:   {:>6.2} ms    {:>6.2} ms    {:>5.1}%",
        down_theoretical_ms,
        down_actual_ms,
        down_theoretical_ms / down_actual_ms * 100.0
    );

    // Gap analysis
    let total_theoretical = up_theoretical_ms + gate_theoretical_ms + down_theoretical_ms;
    let total_actual = up_actual_ms + gate_actual_ms + down_actual_ms;
    let gap_ms = total_actual - total_theoretical;
    let gap_pct = gap_ms / total_actual * 100.0;

    println!("\n=== Gap Analysis ===");
    println!("  Theoretical (memory-bound): {:.2} ms", total_theoretical);
    println!("  Actual (measured):          {:.2} ms", total_actual);
    println!(
        "  Gap:                        {:.2} ms ({:.1}%)",
        gap_ms, gap_pct
    );

    // Estimate overhead sources
    println!("\nPotential overhead sources:");

    // Q8 quantization overhead (input prep)
    let q8_quant_overhead = hidden_dim as f64 * 4.0 / 1e9 / bandwidth_gbps * 1000.0 * 3.0; // 3 projections
    println!(
        "  - Q8 input quantization: ~{:.2} ms (estimated)",
        q8_quant_overhead
    );

    // Thread sync overhead (rayon)
    let num_threads = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(1);
    println!("  - Rayon thread pool: {} threads", num_threads);
    println!("  - Per-call sync overhead: ~10-50 µs (typical)");

    // Cache effects
    let l3_cache_mb = 24.0; // Typical for Intel Core Ultra 7
    let weights_mb = (2.0 * up_size as f64 + down_size as f64) / 1e6;
    if weights_mb > l3_cache_mb {
        println!(
            "  - Weights ({:.1} MB) exceed L3 cache ({:.0} MB) - main memory bound",
            weights_mb, l3_cache_mb
        );
    } else {
        println!(
            "  - Weights ({:.1} MB) fit in L3 cache ({:.0} MB) - should be cache-resident",
            weights_mb, l3_cache_mb
        );
    }

    // Phi-2 analysis
    println!("\n=== Phi-2 Projection (2.7B) ===");
    let phi2_up_blocks = (phi2_hidden * phi2_intermediate) / 32;
    let phi2_down_blocks = (phi2_intermediate * phi2_hidden) / 32;
    let phi2_up_size = phi2_up_blocks * q4_block_bytes;
    let phi2_down_size = phi2_down_blocks * q4_block_bytes;
    let phi2_layer_size = 2.0 * phi2_up_size as f64 + phi2_down_size as f64;
    let phi2_total_ffn_size = phi2_layer_size * phi2_layers as f64;

    println!("  FFN weights per layer: {:.1} MB", phi2_layer_size / 1e6);
    println!(
        "  FFN weights total:     {:.1} MB",
        phi2_total_ffn_size / 1e6
    );

    let phi2_theoretical = phi2_total_ffn_size / 1e9 / bandwidth_gbps * 1000.0;
    println!("  Theoretical FFN time:  {:.1} ms", phi2_theoretical);

    // If we scale Qwen results to Phi-2
    let scale_factor =
        phi2_total_ffn_size / (total_theoretical * 1e6 / 1000.0 * bandwidth_gbps * 1e9 / 1e6);
    let estimated_phi2_actual =
        total_actual * scale_factor * (phi2_layers as f64 / num_layers as f64);
    println!(
        "  Estimated actual time: {:.1} ms (scaled from Qwen)",
        estimated_phi2_actual
    );
}
