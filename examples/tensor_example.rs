//! Tensor operations with SIMD acceleration (v2.0 Phase 3)
//!
//! This example demonstrates:
//! - Creating a `TensorExecutor`
//! - SIMD-accelerated tensor operations (add, mul, dot, etc.)
//! - Performance benefits of vectorized computation

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::many_single_char_names)]

#[cfg(feature = "tensor")]
use repartir::task::Backend;
#[cfg(feature = "tensor")]
use repartir::tensor::{Tensor, TensorExecutor};

#[cfg(feature = "tensor")]
fn main() -> repartir::error::Result<()> {
    println!("=== Repartir v2.0: Tensor Operations Example ===\n");

    // Create tensor executor with CPU backend
    let executor = TensorExecutor::builder().backend(Backend::Cpu).build()?;

    println!("TensorExecutor created (backend: {:?})\n", executor.backend());

    // Example 1: Element-wise addition
    println!("Example 1: Element-wise Addition");
    let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0]);
    let b = Tensor::from_slice(&[5.0, 6.0, 7.0, 8.0]);

    println!("  Input A: {:?}", a.as_slice());
    println!("  Input B: {:?}", b.as_slice());

    let result = executor.add(&a, &b)?;
    println!("  A + B:   {:?}\n", result.as_slice());

    // Example 2: Element-wise multiplication
    println!("Example 2: Element-wise Multiplication");
    let x = Tensor::from_slice(&[2.0, 3.0, 4.0]);
    let y = Tensor::from_slice(&[5.0, 6.0, 7.0]);

    println!("  Input X: {:?}", x.as_slice());
    println!("  Input Y: {:?}", y.as_slice());

    let result = executor.mul(&x, &y)?;
    println!("  X * Y:   {:?}\n", result.as_slice());

    // Example 3: Scalar multiplication
    println!("Example 3: Scalar Multiplication");
    let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let scalar: f32 = 2.5;

    println!("  Input:   {:?}", tensor.as_slice());
    println!("  Scalar:  {scalar}");

    let result = executor.scalar_mul(&tensor, scalar)?;
    println!("  Result:  {:?}\n", result.as_slice());

    // Example 4: Dot product
    println!("Example 4: Dot Product");
    let vec_a = Tensor::from_slice(&[1.0, 2.0, 3.0]);
    let vec_b = Tensor::from_slice(&[4.0, 5.0, 6.0]);

    println!("  Vector A: {:?}", vec_a.as_slice());
    println!("  Vector B: {:?}", vec_b.as_slice());

    let dot_product = executor.dot(&vec_a, &vec_b)?;
    println!("  A . B:    {dot_product}");
    println!("  (1*4 + 2*5 + 3*6 = 32)\n");

    // Example 5: Subtraction and division
    println!("Example 5: Subtraction and Division");
    let numerator = Tensor::from_slice(&[100.0, 200.0, 300.0]);
    let denominator = Tensor::from_slice(&[10.0, 20.0, 30.0]);

    println!("  Numerator:   {:?}", numerator.as_slice());
    println!("  Denominator: {:?}", denominator.as_slice());

    let quotient = executor.div(&numerator, &denominator)?;
    println!("  Quotient:    {:?}\n", quotient.as_slice());

    // Example 6: Complex computation
    println!("Example 6: Complex Computation");
    println!("  Computing: (A + B) * C");

    let a = Tensor::from_slice(&[1.0, 2.0, 3.0]);
    let b = Tensor::from_slice(&[4.0, 5.0, 6.0]);
    let c = Tensor::from_slice(&[2.0, 2.0, 2.0]);

    println!("  A: {:?}", a.as_slice());
    println!("  B: {:?}", b.as_slice());
    println!("  C: {:?}", c.as_slice());

    let sum = executor.add(&a, &b)?;
    let result = executor.mul(&sum, &c)?;

    println!("  (A + B): {:?}", sum.as_slice());
    println!("  Result:  {:?}\n", result.as_slice());

    println!("All tensor operations completed successfully!");
    println!("\nPerformance Note:");
    println!("   Trueno library uses SIMD instructions (AVX2/AVX-512)");
    println!("   for 2-8x speedup vs scalar operations on modern CPUs.\n");

    Ok(())
}

#[cfg(not(feature = "tensor"))]
fn main() {
    eprintln!("Error: This example requires the 'tensor' feature.");
    eprintln!("Run with: cargo run --example tensor_example --features tensor");
    std::process::exit(1);
}
