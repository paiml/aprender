#![allow(clippy::disallowed_methods)]
//! t-SNE (t-Distributed Stochastic Neighbor Embedding) visualization example
//!
//! Demonstrates dimensionality reduction for visualization of high-dimensional data.
//!
//! Run with:
//! ```bash
//! cargo run --example tsne_visualization
//! ```

use aprender::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== t-SNE Dimensionality Reduction Example ===\n");

    let data_4d = create_4d_data()?;
    let small_data = create_small_data()?;

    example_basic_reduction(&data_4d)?;
    example_perplexity_effects(&small_data)?;
    example_3d_embedding(&data_4d)?;
    example_learning_rate(&small_data)?;
    example_reproducibility(&small_data)?;
    print_comparison_info();

    Ok(())
}

fn create_4d_data() -> Result<Matrix<f32>, Box<dyn std::error::Error>> {
    Matrix::from_vec(
        12,
        4,
        vec![
            // Cluster 1: around (1, 1, 1, 1)
            1.0, 1.0, 1.0, 1.0, 1.1, 1.1, 1.1, 1.1, 0.9, 0.9, 0.9, 0.9, 1.2, 1.2, 1.2, 1.2,
            // Cluster 2: around (5, 5, 5, 5)
            5.0, 5.0, 5.0, 5.0, 5.1, 5.1, 5.1, 5.1, 4.9, 4.9, 4.9, 4.9, 5.2, 5.2, 5.2, 5.2,
            // Cluster 3: around (10, 10, 10, 10)
            10.0, 10.0, 10.0, 10.0, 10.1, 10.1, 10.1, 10.1, 9.9, 9.9, 9.9, 9.9, 10.2, 10.2, 10.2,
            10.2,
        ],
    )
    .map_err(Into::into)
}

fn create_small_data() -> Result<Matrix<f32>, Box<dyn std::error::Error>> {
    Matrix::from_vec(
        8,
        3,
        vec![
            1.0, 1.0, 1.0, 1.1, 1.1, 1.1, 5.0, 5.0, 5.0, 5.1, 5.1, 5.1, 10.0, 10.0, 10.0, 10.1,
            10.1, 10.1, 15.0, 15.0, 15.0, 15.1, 15.1, 15.1,
        ],
    )
    .map_err(Into::into)
}

fn example_basic_reduction(data_4d: &Matrix<f32>) -> Result<(), Box<dyn std::error::Error>> {
    println!("--- Example 1: Basic 4D → 2D Reduction ---");
    println!(
        "Input data: {} samples, {} dimensions",
        data_4d.shape().0,
        data_4d.shape().1
    );

    let mut tsne = TSNE::new(2)
        .with_perplexity(5.0)
        .with_n_iter(300)
        .with_random_state(42);

    let embedding_2d = tsne.fit_transform(data_4d)?;
    println!(
        "Output embedding: {} samples, {} dimensions\n",
        embedding_2d.shape().0,
        embedding_2d.shape().1
    );

    println!("2D Embedding (first 6 points):");
    for i in 0..6 {
        println!(
            "  Point {}: ({:.3}, {:.3})",
            i,
            embedding_2d.get(i, 0),
            embedding_2d.get(i, 1)
        );
    }
    Ok(())
}

fn example_perplexity_effects(small_data: &Matrix<f32>) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Example 2: Perplexity Effects ---");
    println!("Perplexity balances local vs global structure:");
    println!("  - Low perplexity (5-10): Focus on very local structure");
    println!("  - Medium perplexity (20-30): Balanced (default)");
    println!("  - High perplexity (50+): More global structure\n");

    let mut tsne_low_perp = TSNE::new(2)
        .with_perplexity(2.0)
        .with_n_iter(200)
        .with_random_state(42);
    tsne_low_perp.fit_transform(small_data)?;
    println!("✓ Low perplexity (2.0): Fitted successfully");

    let mut tsne_high_perp = TSNE::new(2)
        .with_perplexity(5.0)
        .with_n_iter(200)
        .with_random_state(42);
    tsne_high_perp.fit_transform(small_data)?;
    println!("✓ High perplexity (5.0): Fitted successfully");
    Ok(())
}

fn example_3d_embedding(data_4d: &Matrix<f32>) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Example 3: 3D Embedding ---");

    let mut tsne_3d = TSNE::new(3)
        .with_perplexity(5.0)
        .with_n_iter(250)
        .with_random_state(42);

    let embedding_3d = tsne_3d.fit_transform(data_4d)?;
    println!("3D Embedding shape: {:?}", embedding_3d.shape());
    println!(
        "First point in 3D: ({:.3}, {:.3}, {:.3})",
        embedding_3d.get(0, 0),
        embedding_3d.get(0, 1),
        embedding_3d.get(0, 2)
    );
    Ok(())
}

fn example_learning_rate(small_data: &Matrix<f32>) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Example 4: Learning Rate Effects ---");
    println!("Learning rate controls convergence speed:");
    println!("  - Too low: Slow convergence, may get stuck");
    println!("  - Too high: Unstable, may diverge");
    println!("  - Default (200.0): Good for most cases\n");

    let mut tsne_slow = TSNE::new(2)
        .with_learning_rate(50.0)
        .with_n_iter(100)
        .with_random_state(42);
    tsne_slow.fit_transform(small_data)?;
    println!("✓ Slow learning rate (50.0): Fitted");

    let mut tsne_fast = TSNE::new(2)
        .with_learning_rate(500.0)
        .with_n_iter(100)
        .with_random_state(42);
    tsne_fast.fit_transform(small_data)?;
    println!("✓ Fast learning rate (500.0): Fitted");
    Ok(())
}

fn example_reproducibility(small_data: &Matrix<f32>) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Example 5: Reproducibility ---");

    let mut tsne1 = TSNE::new(2).with_random_state(42).with_n_iter(200);
    let result1 = tsne1.fit_transform(small_data)?;

    let mut tsne2 = TSNE::new(2).with_random_state(42).with_n_iter(200);
    let result2 = tsne2.fit_transform(small_data)?;

    let identical = check_matrices_identical(&result1, &result2);
    println!("Results identical with same random_state: {identical}");
    Ok(())
}

fn check_matrices_identical(m1: &Matrix<f32>, m2: &Matrix<f32>) -> bool {
    for i in 0..m1.shape().0 {
        for j in 0..m1.shape().1 {
            if (m1.get(i, j) - m2.get(i, j)).abs() > 1e-6 {
                return false;
            }
        }
    }
    true
}

fn print_comparison_info() {
    println!("\n--- Example 6: t-SNE vs PCA ---\n");

    println!("t-SNE:");
    println!("  ✓ Non-linear dimensionality reduction");
    println!("  ✓ Preserves local structure excellently");
    println!("  ✓ Best for visualization (2D/3D)");
    println!("  ✗ Slow for large datasets (O(n²))");
    println!("  ✗ Non-deterministic (stochastic)");
    println!("  ✗ Cannot transform new data\n");

    println!("PCA:");
    println!("  ✓ Linear dimensionality reduction");
    println!("  ✓ Fast (O(n·d·k))");
    println!("  ✓ Deterministic");
    println!("  ✓ Can transform new data");
    println!("  ✗ May not capture non-linear patterns");
    println!("  ✗ Focuses on global variance\n");

    println!("Use t-SNE for: Visualization, exploratory data analysis");
    println!("Use PCA for: Feature reduction before modeling, fast preprocessing");

    println!("\n=== Example Complete ===");
    println!("\nKey takeaways:");
    println!("✓ t-SNE reduces high-D data to 2D/3D for visualization");
    println!("✓ Perplexity balances local vs global structure (5-50)");
    println!("✓ Use random_state for reproducibility");
    println!("✓ More iterations = better convergence (but slower)");
    println!("✓ Excellent for finding clusters and patterns visually");
}
