//! Mixture-of-Experts (MoE) Backend Routing Example
//!
//! Demonstrates adaptive backend selection based on operation complexity
//! and data size, implementing the Kaizen principle of continuous optimization.
//!
//! Run with: cargo run --example moe_routing

use batuta::backend::{BackendSelector, OpComplexity};

fn main() {
    println!("🎯 Mixture-of-Experts Backend Routing Demo");
    println!("==========================================\n");

    let selector = BackendSelector::new();

    // Low Complexity Operations (Element-wise: add, multiply, etc.)
    println!("📊 Low Complexity (Element-wise operations)");
    println!("-------------------------------------------");

    let sizes = vec![100, 10_000, 500_000, 2_000_000];
    for size in sizes {
        let backend = selector.select_with_moe(OpComplexity::Low, size);
        println!(
            "  {} elements: {} ({})",
            format_size(size),
            backend,
            explain_low_complexity(size)
        );
    }
    println!();

    // Medium Complexity Operations (Reductions: dot product, sum, etc.)
    println!("📊 Medium Complexity (Reductions: dot, sum)");
    println!("--------------------------------------------");

    let sizes = vec![1_000, 50_000, 150_000, 500_000];
    for size in sizes {
        let backend = selector.select_with_moe(OpComplexity::Medium, size);
        println!(
            "  {} elements: {} ({})",
            format_size(size),
            backend,
            explain_medium_complexity(size)
        );
    }
    println!();

    // High Complexity Operations (Matrix operations: matmul, convolution)
    println!("📊 High Complexity (Matrix operations)");
    println!("---------------------------------------");

    let sizes = vec![500, 5_000, 50_000, 200_000];
    for size in sizes {
        let backend = selector.select_with_moe(OpComplexity::High, size);
        println!(
            "  {} elements: {} ({})",
            format_size(size),
            backend,
            explain_high_complexity(size)
        );
    }
    println!();

    // Practical Examples
    println!("💡 Practical Examples");
    println!("---------------------");

    // Example 1: Image processing (element-wise)
    let pixels = 1920 * 1080; // Full HD image
    let backend = selector.select_with_moe(OpComplexity::Low, pixels);
    println!("  Full HD image ({}px): {}", format_size(pixels), backend);

    // Example 2: Vector dot product (medium)
    let embedding_dim = 768 * 1000; // Typical ML embedding
    let backend = selector.select_with_moe(OpComplexity::Medium, embedding_dim);
    println!("  ML embedding dot product ({}): {}", format_size(embedding_dim), backend);

    // Example 3: Matrix multiplication (high)
    let matrix_size = 512 * 512; // 512x512 matrix
    let backend = selector.select_with_moe(OpComplexity::High, matrix_size);
    println!("  512×512 matrix multiply ({}): {}", format_size(matrix_size), backend);

    println!("\n📈 Performance Insights");
    println!("-----------------------");
    println!("  • Low complexity: GPU rarely beneficial (memory-bound)");
    println!("  • Medium complexity: GPU at 100K+ elements");
    println!("  • High complexity: GPU at 10K+ elements (O(n²) or O(n³))");
    println!("\n  Threshold values based on Trueno performance analysis");
    println!("  and the 5× PCIe dispatch rule (Gregg & Hazelwood 2011)");
}

fn format_size(n: usize) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.0}K", n as f64 / 1_000.0)
    } else {
        n.to_string()
    }
}

fn explain_low_complexity(size: usize) -> &'static str {
    if size > 1_000_000 {
        "SIMD for large arrays"
    } else {
        "Scalar, memory-bound"
    }
}

fn explain_medium_complexity(size: usize) -> &'static str {
    if size > 100_000 {
        "GPU worthwhile"
    } else if size > 10_000 {
        "SIMD optimal"
    } else {
        "Scalar sufficient"
    }
}

fn explain_high_complexity(size: usize) -> &'static str {
    if size > 10_000 {
        "GPU beneficial for O(n²/n³)"
    } else if size > 1_000 {
        "SIMD for medium matrices"
    } else {
        "Scalar for small ops"
    }
}
