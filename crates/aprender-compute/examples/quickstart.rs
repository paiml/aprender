#![allow(clippy::disallowed_methods)]
//! Trueno Quick Start Example
//!
//! This example demonstrates the core features of Trueno in a single file.
//! Run with: cargo run --example quickstart

use trueno::{Backend, Matrix, SymmetricEigen, Vector};

fn main() {
    println!("=== Trueno Quick Start ===\n");

    // 1. Vector Operations
    println!("1. Vector Operations");
    println!("   -----------------");

    let a = Vector::<f32>::from_slice(&[1.0, 2.0, 3.0, 4.0]);
    let b = Vector::<f32>::from_slice(&[5.0, 6.0, 7.0, 8.0]);

    // Element-wise operations
    let sum = a.add(&b).expect("add");
    println!("   a + b = {:?}", sum.as_slice());

    // Dot product (compute-bound, benefits from AVX-512)
    let dot = a.dot(&b).expect("dot");
    println!("   a · b = {}", dot);

    // Reductions
    let total = a.sum().expect("sum");
    let maximum = a.max().expect("max");
    println!("   sum(a) = {}, max(a) = {}", total, maximum);

    // Norms
    let l2 = a.norm_l2().expect("norm_l2");
    println!("   ||a||₂ = {:.4}", l2);
    println!();

    // 2. Activation Functions (for ML)
    println!("2. Activation Functions");
    println!("   --------------------");

    let x = Vector::<f32>::from_slice(&[-2.0, -1.0, 0.0, 1.0, 2.0]);

    let relu = x.relu().expect("relu");
    println!("   ReLU({:?}) = {:?}", x.as_slice(), relu.as_slice());

    let sigmoid = x.sigmoid().expect("sigmoid");
    println!(
        "   Sigmoid = [{:.3}, {:.3}, {:.3}, {:.3}, {:.3}]",
        sigmoid.as_slice()[0],
        sigmoid.as_slice()[1],
        sigmoid.as_slice()[2],
        sigmoid.as_slice()[3],
        sigmoid.as_slice()[4]
    );

    let gelu = x.gelu().expect("gelu");
    println!(
        "   GELU    = [{:.3}, {:.3}, {:.3}, {:.3}, {:.3}]",
        gelu.as_slice()[0],
        gelu.as_slice()[1],
        gelu.as_slice()[2],
        gelu.as_slice()[3],
        gelu.as_slice()[4]
    );
    println!();

    // 3. Matrix Operations
    println!("3. Matrix Operations");
    println!("   -----------------");

    let m1 = Matrix::<f32>::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("m1");
    let m2 = Matrix::<f32>::from_vec(3, 2, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).expect("m2");

    let product = m1.matmul(&m2).expect("matmul");
    println!("   [2×3] × [3×2] = [2×2]");
    println!(
        "   Result: [{:.0}, {:.0}; {:.0}, {:.0}]",
        product.get(0, 0).unwrap(),
        product.get(0, 1).unwrap(),
        product.get(1, 0).unwrap(),
        product.get(1, 1).unwrap()
    );

    let transposed = m1.transpose();
    println!("   Transpose: [2×3] → [{}×{}]", transposed.rows(), transposed.cols());
    println!();

    // 4. Eigendecomposition
    println!("4. Eigendecomposition");
    println!("   -------------------");

    // Symmetric matrix for PCA-like analysis
    let sym = Matrix::<f32>::from_vec(3, 3, vec![4.0, 2.0, 0.0, 2.0, 5.0, 3.0, 0.0, 3.0, 6.0])
        .expect("sym");

    let eigen = SymmetricEigen::new(&sym).expect("eigen");
    println!("   Eigenvalues: {:?}", eigen.eigenvalues());
    println!(
        "   Largest explains {:.1}% of variance",
        100.0 * eigen.eigenvalues()[0] / eigen.eigenvalues().iter().sum::<f32>()
    );
    println!();

    // 5. Backend Selection
    println!("5. Backend Selection");
    println!("   ------------------");

    let auto_backend = trueno::select_best_available_backend();
    println!("   Auto-detected: {:?}", auto_backend);

    // Force specific backend
    let scalar_vec = Vector::<f32>::from_slice_with_backend(&[1.0, 2.0, 3.0], Backend::Scalar);
    let simd_vec = Vector::<f32>::from_slice(&[1.0, 2.0, 3.0]); // Uses best available

    println!("   Scalar backend: {:?}", scalar_vec.backend());
    println!("   Auto backend:   {:?}", simd_vec.backend());
    println!();

    // 6. Layer Normalization (Transformer component)
    println!("6. Layer Normalization");
    println!("   --------------------");

    let hidden = Vector::<f32>::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    // gamma (scale) = 1.0, beta (shift) = 0.0 for standard normalization
    let gamma = Vector::<f32>::from_slice(&[1.0, 1.0, 1.0, 1.0, 1.0]);
    let beta = Vector::<f32>::from_slice(&[0.0, 0.0, 0.0, 0.0, 0.0]);
    let normalized = hidden.layer_norm(&gamma, &beta, 1e-5).expect("layer_norm");
    let mean: f32 = normalized.as_slice().iter().sum::<f32>() / 5.0;
    let var: f32 = normalized.as_slice().iter().map(|x| (x - mean).powi(2)).sum::<f32>() / 5.0;
    println!("   Input:  {:?}", hidden.as_slice());
    println!(
        "   Output: [{:.3}, {:.3}, {:.3}, {:.3}, {:.3}]",
        normalized.as_slice()[0],
        normalized.as_slice()[1],
        normalized.as_slice()[2],
        normalized.as_slice()[3],
        normalized.as_slice()[4]
    );
    println!("   Mean ≈ {:.6}, Var ≈ {:.6}", mean, var);
    println!();

    println!("=== Quick Start Complete ===");
    println!();
    println!("Next steps:");
    println!("  cargo run --example performance_demo    # See SIMD speedups");
    println!("  cargo run --example activation_functions # All ML activations");
    println!("  cargo run --example symmetric_eigen     # Eigendecomposition details");
    println!("  cargo run --features gpu --example gpu_batch_demo # GPU operations");
}
