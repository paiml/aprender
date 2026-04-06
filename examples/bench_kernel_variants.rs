//! PAR-126: Compare kernel variants

use std::time::Instant;

fn main() {
    let in_dim: usize = 1536;
    let super_blocks = in_dim / 256; // 6
    let bytes_per_row = super_blocks * 144; // 864

    // Create test data
    let weight_data: Vec<u8> = (0..bytes_per_row).map(|i| (i % 256) as u8).collect();
    let q8k_scales: Vec<f32> = vec![1.0f32; super_blocks];
    let q8k_quants: Vec<i8> = (0..in_dim as i8).collect();

    // Warmup
    for _ in 0..1000 {
        let _ = realizar::quantize::fused_q4k_q8k_dot_simd(&weight_data, &q8k_scales, &q8k_quants);
    }

    // Benchmark current kernel
    let iterations = 100000;
    let start = Instant::now();
    let mut sum: f32 = 0.0;
    for _ in 0..iterations {
        sum += realizar::quantize::fused_q4k_q8k_dot_simd(&weight_data, &q8k_scales, &q8k_quants)
            .unwrap_or(0.0);
    }
    let kernel_ns = start.elapsed().as_nanos() as f64 / iterations as f64;

    println!("=== Kernel Timing Analysis ===\n");
    println!(
        "V2 Kernel: {:.1} ns ({:.3} µs)",
        kernel_ns,
        kernel_ns / 1000.0
    );
    println!("Sum (prevent opt): {}", sum);

    // Calculate operations per call
    // For 1536 elements with Q4K×Q8K:
    // - 1536 multiplies (4-bit × 8-bit)
    // - 1536 additions
    // - 6 scale multiplies
    let flops_per_call = (1536 * 2 + 6) as f64;
    let gflops = flops_per_call / kernel_ns;
    println!("\nFLOPs per call: {}", flops_per_call as usize);
    println!("GFLOPs: {:.2}", gflops);

    // Theoretical peak for AVX-512
    // vpdpbusd: 16 int8 MACs per cycle at 4.8 GHz = 76.8 GMACS/core
    println!("\nTheoretical AVX-512 VNNI:");
    println!("Peak (single core): ~77 GMACS");
    println!("Actual: {:.2} GMACS", flops_per_call / 2.0 / kernel_ns);
    println!(
        "Efficiency: {:.1}%",
        (flops_per_call / 2.0 / kernel_ns) / 77.0 * 100.0
    );

    // For full matmul
    let out_dim = 8960;
    let matmul_time_us = kernel_ns * out_dim as f64 / 1000.0;
    println!("\n=== Full Matmul ({}×{}) ===", out_dim, in_dim);
    println!("Sequential: {:.0} µs", matmul_time_us);
    println!("Parallel (3x): {:.0} µs", matmul_time_us / 3.0);
    println!("Ollama equiv: ~150 µs");
    println!("Gap: {:.1}x", matmul_time_us / 3.0 / 150.0);
}
