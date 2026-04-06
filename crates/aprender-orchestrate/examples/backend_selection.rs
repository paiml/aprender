/// Backend selection demonstration
/// Based on sovereign-ai-spec.md section 2.2
use batuta::backend::BackendSelector;

fn main() {
    println!("🎯 Backend Selection Demo");
    println!("Based on sovereign-ai-spec.md section 2.2 (5× PCIe rule)\n");

    let selector = BackendSelector::new();

    // Example 1: Small matrix multiplication
    println!("Example 1: Matrix Multiplication 64×64");
    let backend = selector.select_for_matmul(64, 64, 64);
    println!("  Selected backend: {}", backend);
    println!("  Rationale: Small matrix, PCIe overhead dominates\n");

    // Example 2: Large matrix multiplication
    println!("Example 2: Matrix Multiplication 512×512");
    let backend = selector.select_for_matmul(512, 512, 512);
    println!("  Selected backend: {}", backend);
    println!("  Rationale: Compute/transfer ratio still < 5×\n");

    // Example 3: Very large matrix
    println!("Example 3: Matrix Multiplication 2048×2048");
    let backend = selector.select_for_matmul(2048, 2048, 2048);
    println!("  Selected backend: {}", backend);
    println!("  Rationale: O(n³) compute begins to justify GPU\n");

    // Example 4: Dot product (memory-bound)
    println!("Example 4: Dot Product (10K elements)");
    let backend = selector.select_for_vector_op(10_000, 2);
    println!("  Selected backend: {}", backend);
    println!("  Rationale: Memory-bound, GPU overhead too high\n");

    // Example 5: Element-wise operations
    println!("Example 5: Element-wise Add (1M elements)");
    let backend = selector.select_for_elementwise(1_000_000);
    println!("  Selected backend: {}", backend);
    println!("  Rationale: Minimal compute, uses SIMD\n");

    // Example 6: Custom workload
    println!("Example 6: Custom Workload");
    let data_bytes = 1_000_000; // 1 MB
    let flops = 1_000_000_000; // 1 GFLOP
    let backend = selector.select_backend(data_bytes, flops);
    println!("  Data: {} bytes", data_bytes);
    println!("  FLOPs: {}", flops);
    println!("  Selected backend: {}", backend);

    // Calculate the ratio
    let pcie_bw = 32e9; // 32 GB/s
    let gpu_gflops = 20e12; // 20 TFLOPS
    let transfer_time = data_bytes as f64 / pcie_bw;
    let compute_time = flops as f64 / gpu_gflops;
    let ratio = compute_time / transfer_time;
    println!("  Compute/Transfer ratio: {:.2}×", ratio);
    println!("  (Need > 5× for GPU benefit)\n");

    println!("✅ Per Gregg & Hazelwood (2011): GPU dispatch when");
    println!("   compute_time > 5× transfer_time");
}
