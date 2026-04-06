//! Backend Selector Example
//!
//! Demonstrates MoE (Mixture-of-Experts) routing for GPU/SIMD acceleration.
//!
//! Run with: cargo run --example backend_selector

use verificar::ml::{BackendSelector, BatchConfig, OpComplexity};

fn main() {
    println!("=== Verificar Backend Selector Example ===\n");

    let selector = BackendSelector::new();

    // Demonstrate MoE routing for different operation complexities
    println!("--- MoE Routing by Complexity ---");

    let test_cases = [
        (OpComplexity::Low, 100, "Small element-wise"),
        (OpComplexity::Low, 1_000_000, "Large element-wise"),
        (OpComplexity::Low, 10_000_000, "Huge element-wise"),
        (OpComplexity::Medium, 1_000, "Small reduction"),
        (OpComplexity::Medium, 50_000, "Medium reduction"),
        (OpComplexity::Medium, 500_000, "Large reduction"),
        (OpComplexity::High, 100, "Small matmul"),
        (OpComplexity::High, 5_000, "Medium matmul"),
        (OpComplexity::High, 100_000, "Large matmul"),
    ];

    for (complexity, size, desc) in test_cases {
        let backend = selector.select_with_moe(complexity, size);
        let stats = selector.selection_stats(complexity, size);
        println!(
            "  {:20} ({:>10} elements): {:6} (~{:.1}x speedup)",
            desc, size, backend, stats.estimated_speedup
        );
    }

    // Matrix multiplication routing
    println!("\n--- Matrix Multiplication Routing ---");

    let matmul_sizes = [
        (10, 10, 10),
        (100, 100, 100),
        (1000, 1000, 1000),
        (5000, 5000, 5000),
    ];

    for (m, n, k) in matmul_sizes {
        let backend = selector.select_for_matmul(m, n, k);
        println!("  {}x{}x{} matmul: {}", m, n, k, backend);
    }

    // Vector operations
    println!("\n--- Vector Operation Routing ---");

    let vector_sizes = [100, 10_000, 1_000_000, 10_000_000];

    for n in vector_sizes {
        let backend = selector.select_for_vector_op(n, 2); // dot product
        println!("  {} element dot product: {}", n, backend);
    }

    // Element-wise operations
    println!("\n--- Element-wise Operation Routing ---");

    for n in vector_sizes {
        let backend = selector.select_for_elementwise(n);
        println!("  {} element add: {}", n, backend);
    }

    // Batch configuration
    println!("\n--- Batch Configuration ---");

    let configs = [
        (1000, OpComplexity::Low),
        (50_000, OpComplexity::Medium),
        (100_000, OpComplexity::High),
    ];

    for (size, complexity) in configs {
        let config = BatchConfig::new(size).with_complexity(complexity);
        println!("  Batch size {}, {:?} complexity:", size, complexity);
        println!("    Recommended: {}", config.recommended_backend());
        println!("    Use GPU: {}", config.should_use_gpu());
        println!("    Use SIMD: {}", config.should_use_simd());
    }

    // Custom selector configuration
    println!("\n--- Custom Selector (Aggressive GPU) ---");

    let aggressive_selector = BackendSelector::new()
        .with_min_dispatch_ratio(2.0) // Less conservative than default 5x
        .with_gpu_gflops(80e12); // Model a faster GPU

    for (complexity, size, desc) in &test_cases[6..9] {
        let default_backend = selector.select_with_moe(*complexity, *size);
        let aggressive_backend = aggressive_selector.select_with_moe(*complexity, *size);
        println!(
            "  {}: default={}, aggressive={}",
            desc, default_backend, aggressive_backend
        );
    }

    println!("\n=== Example Complete ===");
}
