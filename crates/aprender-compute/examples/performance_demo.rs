#![allow(clippy::disallowed_methods)]
//! Performance demonstration comparing Scalar vs SSE2 backends
//!
//! This example demonstrates the performance characteristics documented
//! in PERFORMANCE_GUIDE.md by running actual operations and measuring
//! the time taken.
//!
//! Run with:
//! ```
//! cargo run --release --example performance_demo
//! ```

use std::time::Instant;
use trueno::{Backend, Vector};

fn main() {
    println!("🚀 Trueno Performance Demonstration\n");
    println!("Comparing Scalar vs SSE2 backends across different operations");
    println!("See docs/PERFORMANCE_GUIDE.md for detailed analysis\n");
    println!("{}", "=".repeat(80));

    // Test different vector sizes
    let sizes = vec![100, 1000, 10000];

    for size in sizes {
        println!("\n📊 Vector Size: {} elements\n", size);

        // Generate test data
        let data_a: Vec<f32> = (0..size).map(|i| (i as f32) * 0.5).collect();
        let data_b: Vec<f32> = (0..size).map(|i| (i as f32) * 0.3).collect();

        // Dot Product (Compute-Intensive - Expected: 200-400% speedup)
        demo_operation(
            "Dot Product",
            &data_a,
            &data_b,
            |a, b| a.dot(b).expect("Example should not fail"),
            "Compute-intensive: 340% faster expected",
        );

        // Sum Reduction (Compute-Intensive - Expected: 200-400% speedup)
        demo_operation(
            "Sum Reduction",
            &data_a,
            &data_b,
            |a, _| a.sum().expect("Example should not fail"),
            "Compute-intensive: 315% faster expected",
        );

        // Max Finding (Compute-Intensive - Expected: 200-400% speedup)
        demo_operation(
            "Max Finding",
            &data_a,
            &data_b,
            |a, _| a.max().expect("Example should not fail"),
            "Compute-intensive: 348% faster expected",
        );

        // Element-wise Add (Memory-Bound - Expected: 3-10% speedup)
        demo_operation_vec(
            "Element-wise Add",
            &data_a,
            &data_b,
            |a, b| a.add(b).expect("Example should not fail"),
            "Memory-bound: 3-10% faster expected",
        );

        // Element-wise Mul (Memory-Bound - Expected: 5-6% speedup)
        demo_operation_vec(
            "Element-wise Mul",
            &data_a,
            &data_b,
            |a, b| a.mul(b).expect("Example should not fail"),
            "Memory-bound: 5-6% faster expected",
        );

        println!("{}", "-".repeat(80));
    }

    println!("\n✨ Key Takeaways:\n");
    println!("  ✅ Compute-intensive operations (dot, sum, max): 200-400% faster");
    println!("  ⚠️  Memory-bound operations (add, mul): 3-10% faster");
    println!("\n  💡 Why: SIMD excels at computation but can't overcome memory bandwidth\n");
    println!("  📖 See docs/PERFORMANCE_GUIDE.md for tuning tips and detailed analysis");
    println!("{}", "=".repeat(80));
}

/// Demonstrate operation returning a scalar value
fn demo_operation<F>(name: &str, data_a: &[f32], data_b: &[f32], op: F, description: &str)
where
    F: Fn(&Vector<f32>, &Vector<f32>) -> f32,
{
    const ITERATIONS: usize = 1000;

    // Scalar backend
    let a_scalar = Vector::from_slice_with_backend(data_a, Backend::Scalar);
    let b_scalar = Vector::from_slice_with_backend(data_b, Backend::Scalar);

    let start = Instant::now();
    for _ in 0..ITERATIONS {
        let _ = op(&a_scalar, &b_scalar);
    }
    let scalar_time = start.elapsed();

    // SSE2 backend
    #[cfg(target_arch = "x86_64")]
    let sse2_time = {
        let a_sse2 = Vector::from_slice_with_backend(data_a, Backend::SSE2);
        let b_sse2 = Vector::from_slice_with_backend(data_b, Backend::SSE2);

        let start = Instant::now();
        for _ in 0..ITERATIONS {
            let _ = op(&a_sse2, &b_sse2);
        }
        start.elapsed()
    };

    #[cfg(not(target_arch = "x86_64"))]
    let sse2_time = scalar_time; // Fallback for non-x86_64

    // Calculate speedup
    let speedup = if sse2_time.as_nanos() > 0 {
        (scalar_time.as_nanos() as f64 / sse2_time.as_nanos() as f64 - 1.0) * 100.0
    } else {
        0.0
    };

    // Determine status
    let status = if speedup >= 100.0 {
        "🚀 Excellent"
    } else if speedup >= 10.0 {
        "✅ Good"
    } else if speedup >= 5.0 {
        "⚠️  Modest"
    } else {
        "❌ Limited"
    };

    println!(
        "{:<20} {:>12} {:>12} {:>10.1}% {}",
        format!("  {}:", name),
        format!("{:.2?}", scalar_time / ITERATIONS as u32),
        format!("{:.2?}", sse2_time / ITERATIONS as u32),
        speedup,
        status
    );
    println!("    └─ {}", description);
}

/// Demonstrate operation returning a Vector
fn demo_operation_vec<F>(name: &str, data_a: &[f32], data_b: &[f32], op: F, description: &str)
where
    F: Fn(&Vector<f32>, &Vector<f32>) -> Vector<f32>,
{
    const ITERATIONS: usize = 1000;

    // Scalar backend
    let a_scalar = Vector::from_slice_with_backend(data_a, Backend::Scalar);
    let b_scalar = Vector::from_slice_with_backend(data_b, Backend::Scalar);

    let start = Instant::now();
    for _ in 0..ITERATIONS {
        let _ = op(&a_scalar, &b_scalar);
    }
    let scalar_time = start.elapsed();

    // SSE2 backend
    #[cfg(target_arch = "x86_64")]
    let sse2_time = {
        let a_sse2 = Vector::from_slice_with_backend(data_a, Backend::SSE2);
        let b_sse2 = Vector::from_slice_with_backend(data_b, Backend::SSE2);

        let start = Instant::now();
        for _ in 0..ITERATIONS {
            let _ = op(&a_sse2, &b_sse2);
        }
        start.elapsed()
    };

    #[cfg(not(target_arch = "x86_64"))]
    let sse2_time = scalar_time; // Fallback for non-x86_64

    // Calculate speedup
    let speedup = if sse2_time.as_nanos() > 0 {
        (scalar_time.as_nanos() as f64 / sse2_time.as_nanos() as f64 - 1.0) * 100.0
    } else {
        0.0
    };

    // Determine status
    let status = if speedup >= 100.0 {
        "🚀 Excellent"
    } else if speedup >= 10.0 {
        "✅ Good"
    } else if speedup >= 5.0 {
        "⚠️  Modest"
    } else {
        "❌ Limited"
    };

    println!(
        "{:<20} {:>12} {:>12} {:>10.1}% {}",
        format!("  {}:", name),
        format!("{:.2?}", scalar_time / ITERATIONS as u32),
        format!("{:.2?}", sse2_time / ITERATIONS as u32),
        speedup,
        status
    );
    println!("    └─ {}", description);
}
