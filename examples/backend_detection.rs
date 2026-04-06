#![allow(clippy::disallowed_methods)]
//! Example demonstrating runtime CPU feature detection
//!
//! Run with:
//! ```
//! cargo run --example backend_detection
//! ```

use trueno::{select_best_available_backend, Backend, Vector};

fn main() {
    println!("Trueno Backend Detection Example");
    println!("=================================\n");

    // Detect best available backend
    let backend = select_best_available_backend();
    println!("Auto-detected backend: {:?}", backend);

    // Show what features were detected
    #[cfg(target_arch = "x86_64")]
    {
        println!("\nx86_64 CPU Features:");
        println!("  SSE2:    {}", is_x86_feature_detected!("sse2"));
        println!("  AVX:     {}", is_x86_feature_detected!("avx"));
        println!("  AVX2:    {}", is_x86_feature_detected!("avx2"));
        println!("  FMA:     {}", is_x86_feature_detected!("fma"));
        println!("  AVX512F: {}", is_x86_feature_detected!("avx512f"));
    }

    println!("\nBackend Selection Priority:");
    println!("  x86_64: AVX-512 → AVX2+FMA → AVX → SSE2 → Scalar");
    println!("  ARM:    NEON → Scalar");
    println!("  WASM:   SIMD128 → Scalar");

    // Create a vector and show which backend it uses
    println!("\nVector Creation:");
    let v = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0]);
    println!("  Vector::from_slice() uses: {:?}", v.backend());

    // Explicitly create with different backends
    println!("\nExplicit Backend Selection:");
    let v_scalar = Vector::from_slice_with_backend(&[1.0, 2.0], Backend::Scalar);
    println!("  Scalar backend: {:?}", v_scalar.backend());

    let v_auto = Vector::from_slice_with_backend(&[1.0, 2.0], Backend::Auto);
    println!("  Auto backend resolves to: {:?}", v_auto.backend());

    // Demonstrate that operations work the same regardless of backend
    println!("\nBackend Transparency:");
    let a = Vector::from_slice_with_backend(&[1.0, 2.0, 3.0], Backend::Scalar);
    let b = Vector::from_slice_with_backend(&[4.0, 5.0, 6.0], backend);

    let sum = a.add(&b).expect("Example should not fail");
    println!("  Scalar + {:?} = {:?}", backend, sum.as_slice());
    println!("  (Operations work across backends transparently)");

    println!("\n✨ All backends provide the same API and correctness guarantees!");
    println!("   Only performance differs (8x-20x speedup with SIMD/GPU)");
}
