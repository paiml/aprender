//! NumPy to Trueno Conversion Example (BATUTA-008)
//!
//! Demonstrates the conversion mapping from Python NumPy operations
//! to Rust Trueno operations with MoE backend selection.
//!
//! Run with: cargo run --example numpy_conversion

use batuta::numpy_converter::{NumPyConverter, NumPyOp};

fn main() {
    println!("🔄 NumPy → Trueno Conversion Demo (BATUTA-008)");
    println!("=============================================\n");

    let converter = NumPyConverter::new();

    // Show conversion mapping
    println!("📋 Operation Conversion Mapping");
    println!("-------------------------------\n");

    let ops = vec![
        (NumPyOp::Add, "np.add(a, b) or a + b"),
        (NumPyOp::Subtract, "np.subtract(a, b) or a - b"),
        (NumPyOp::Multiply, "np.multiply(a, b) or a * b"),
        (NumPyOp::Sum, "np.sum(a)"),
        (NumPyOp::Dot, "np.dot(a, b) or a @ b"),
    ];

    for (op, numpy_code) in ops {
        if let Some(trueno_op) = converter.convert(&op) {
            println!("NumPy:  {}", numpy_code);
            println!("Trueno: {}", trueno_op.code_template);
            println!("        Complexity: {:?}", trueno_op.complexity);
            println!();
        }
    }

    // Show backend recommendations
    println!("🎯 Backend Recommendations by Data Size");
    println!("----------------------------------------\n");

    let operations = vec![
        (NumPyOp::Add, "Element-wise addition"),
        (NumPyOp::Sum, "Reduction (sum)"),
        (NumPyOp::Dot, "Dot product/matmul"),
    ];

    let sizes = vec![100, 10_000, 100_000, 1_000_000];

    for (op, desc) in operations {
        println!("{} ({:?} complexity):", desc, op.complexity());
        for &size in &sizes {
            let backend = converter.recommend_backend(&op, size);
            println!("  {:>8} elements → {}", format_size(size), backend);
        }
        println!();
    }

    // Show Python→Rust conversion example
    println!("💡 Practical Conversion Example");
    println!("-------------------------------\n");

    println!("Python NumPy:");
    println!("```python");
    println!("import numpy as np");
    println!("a = np.array([1.0, 2.0, 3.0, 4.0])");
    println!("b = np.array([5.0, 6.0, 7.0, 8.0])");
    println!("c = np.add(a, b)");
    println!("result = np.sum(c)");
    println!("```\n");

    println!("Rust Trueno (converted):");
    println!("```rust");
    println!("use trueno::Vector;");
    println!();
    println!("let a = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0]);");
    println!("let b = Vector::from_slice(&[5.0, 6.0, 7.0, 8.0]);");
    println!("let c = a.add(&b).unwrap();");
    println!("let result = c.sum();");
    println!("```\n");

    // Show performance analysis
    println!("⚡ Performance Analysis");
    println!("----------------------\n");

    println!("Backend selection via MoE routing:");
    println!(
        "  • 4 elements (element-wise add): {} (memory-bound)",
        converter.recommend_backend(&NumPyOp::Add, 4)
    );
    println!("  • 4 elements (sum reduction): {}", converter.recommend_backend(&NumPyOp::Sum, 4));
    println!(
        "  • 1M elements (element-wise add): {} (vectorized)",
        converter.recommend_backend(&NumPyOp::Add, 1_000_000)
    );
    println!(
        "  • 200K elements (sum reduction): {} (high throughput)",
        converter.recommend_backend(&NumPyOp::Sum, 200_000)
    );

    println!("\n✨ Benefits:");
    println!("  • Semantic preservation: Same numerical results");
    println!("  • Performance: SIMD/GPU acceleration via Trueno");
    println!("  • Safety: Rust's memory safety guarantees");
    println!("  • Adaptive: MoE routing selects optimal backend");
}

fn format_size(n: usize) -> String {
    if n >= 1_000_000 {
        format!("{}M", n / 1_000_000)
    } else if n >= 1_000 {
        format!("{}K", n / 1_000)
    } else {
        n.to_string()
    }
}
