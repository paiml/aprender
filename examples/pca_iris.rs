//! PCA Dimensionality Reduction Example - Iris Dataset
//!
//! This example demonstrates Principal Component Analysis (PCA) on the Iris dataset:
//! - Reducing 4D feature space to 2D for visualization
//! - Analyzing explained variance
//! - Reconstructing original data from reduced dimensions
//! - Understanding information loss from dimensionality reduction
//!
//! Run with: `cargo run --example pca_iris`

use aprender::preprocessing::{StandardScaler, PCA};
use aprender::primitives::Matrix;
use aprender::traits::Transformer;

/// Create the simulated iris-like dataset with species labels.
fn create_iris_data() -> (Matrix<f32>, Vec<&'static str>) {
    let data = Matrix::from_vec(
        30,
        4,
        vec![
            // Setosa (10 samples)
            5.1, 3.5, 1.4, 0.2, 4.9, 3.0, 1.4, 0.2, 4.7, 3.2, 1.3, 0.2, 4.6, 3.1, 1.5, 0.2, 5.0,
            3.6, 1.4, 0.2, 5.4, 3.9, 1.7, 0.4, 4.6, 3.4, 1.4, 0.3, 5.0, 3.4, 1.5, 0.2, 4.4, 2.9,
            1.4, 0.2, 4.9, 3.1, 1.5, 0.1, // Versicolor (10 samples)
            7.0, 3.2, 4.7, 1.4, 6.4, 3.2, 4.5, 1.5, 6.9, 3.1, 4.9, 1.5, 5.5, 2.3, 4.0, 1.3, 6.5,
            2.8, 4.6, 1.5, 5.7, 2.8, 4.5, 1.3, 6.3, 3.3, 4.7, 1.6, 4.9, 2.4, 3.3, 1.0, 6.6, 2.9,
            4.6, 1.3, 5.2, 2.7, 3.9, 1.4, // Virginica (10 samples)
            6.3, 3.3, 6.0, 2.5, 5.8, 2.7, 5.1, 1.9, 7.1, 3.0, 5.9, 2.1, 6.3, 2.9, 5.6, 1.8, 6.5,
            3.0, 5.8, 2.2, 7.6, 3.0, 6.6, 2.1, 4.9, 2.5, 4.5, 1.7, 7.3, 2.9, 6.3, 1.8, 6.7, 2.5,
            5.8, 1.8, 7.2, 3.6, 6.1, 2.5,
        ],
    )
    .expect("Example data should be valid");

    let species = vec![
        "Setosa",
        "Setosa",
        "Setosa",
        "Setosa",
        "Setosa",
        "Setosa",
        "Setosa",
        "Setosa",
        "Setosa",
        "Setosa",
        "Versicolor",
        "Versicolor",
        "Versicolor",
        "Versicolor",
        "Versicolor",
        "Versicolor",
        "Versicolor",
        "Versicolor",
        "Versicolor",
        "Versicolor",
        "Virginica",
        "Virginica",
        "Virginica",
        "Virginica",
        "Virginica",
        "Virginica",
        "Virginica",
        "Virginica",
        "Virginica",
        "Virginica",
    ];

    (data, species)
}

/// Step 1 & 2: Standardize features and apply PCA.
fn standardize_and_reduce(data: &Matrix<f32>) -> (StandardScaler, PCA, Matrix<f32>, Matrix<f32>) {
    let (_, n_features) = data.shape();

    println!("═══════════════════════════════════════════════════════");
    println!("  Step 1: Standardizing Features");
    println!("═══════════════════════════════════════════════════════\n");

    let mut scaler = StandardScaler::new();
    let scaled_data = scaler
        .fit_transform(data)
        .expect("Example data should be valid");

    println!("Why standardize?");
    println!("   PCA is sensitive to feature scales. Standardizing ensures");
    println!("   all features contribute equally to principal components.\n");

    println!("═══════════════════════════════════════════════════════");
    println!("  Step 2: Applying PCA (4D → 2D)");
    println!("═══════════════════════════════════════════════════════\n");

    let mut pca = PCA::new(2);
    let transformed = pca
        .fit_transform(&scaled_data)
        .expect("Example data should be valid");

    println!("Dimensionality Reduction:");
    println!("   Original: {n_features} features");
    println!("   Reduced:  {} principal components\n", 2);

    (scaler, pca, scaled_data, transformed)
}

/// Step 3: Analyze explained variance.
fn analyze_explained_variance(pca: &PCA) -> f32 {
    println!("═══════════════════════════════════════════════════════");
    println!("  Step 3: Explained Variance Analysis");
    println!("═══════════════════════════════════════════════════════\n");

    let explained_var = pca
        .explained_variance()
        .expect("Example data should be valid");
    let explained_ratio = pca
        .explained_variance_ratio()
        .expect("Example data should be valid");

    println!("Explained Variance by Component:");
    for i in 0..2 {
        let variance_pct = explained_ratio[i] * 100.0;
        let bar_length = (variance_pct / 2.0) as usize;
        let bar = "█".repeat(bar_length);
        println!(
            "   PC{}: {:.4} ({:5.2}%) {}",
            i + 1,
            explained_var[i],
            variance_pct,
            bar
        );
    }

    let total_variance: f32 = explained_ratio.iter().sum();
    println!("\nTotal Variance Captured: {:.2}%", total_variance * 100.0);
    println!(
        "Information Lost:        {:.2}%\n",
        (1.0 - total_variance) * 100.0
    );

    println!("💡 Interpretation:");
    println!(
        "   First two components capture {:.1}% of variance",
        total_variance * 100.0
    );
    println!("   This is excellent for 2D visualization!\n");

    total_variance
}

/// Step 4: Display transformed data samples.
fn display_transformed_data(transformed: &Matrix<f32>, species: &[&str]) {
    println!("═══════════════════════════════════════════════════════");
    println!("  Step 4: Transformed Data (First 5 samples per species)");
    println!("═══════════════════════════════════════════════════════\n");

    println!(
        "{:>6} {:>12} {:>10} {:>10}",
        "Sample", "Species", "PC1", "PC2"
    );
    println!("{}", "─".repeat(44));

    for i in &[0, 1, 2, 3, 4, 10, 11, 12, 13, 14, 20, 21, 22, 23, 24] {
        println!(
            "{:>6} {:>12} {:>10.4} {:>10.4}",
            i,
            species[*i],
            transformed.get(*i, 0),
            transformed.get(*i, 1)
        );
    }
    println!();
}

/// Step 5: Reconstruct from reduced dimensions and measure error.
fn reconstruction_analysis(
    data: &Matrix<f32>,
    pca: &PCA,
    scaler: &StandardScaler,
    transformed: &Matrix<f32>,
) -> f32 {
    let (n_samples, n_features) = data.shape();

    println!("═══════════════════════════════════════════════════════");
    println!("  Step 5: Reconstruction from 2D → 4D");
    println!("═══════════════════════════════════════════════════════\n");

    let reconstructed_scaled = pca
        .inverse_transform(transformed)
        .expect("Example data should be valid");
    let reconstructed = scaler
        .inverse_transform(&reconstructed_scaled)
        .expect("Example data should be valid");

    // Calculate reconstruction error
    let mut total_squared_error = 0.0;
    let mut max_error = 0.0;
    for i in 0..n_samples {
        for j in 0..n_features {
            let error = (data.get(i, j) - reconstructed.get(i, j)).abs();
            total_squared_error += error * error;
            if error > max_error {
                max_error = error;
            }
        }
    }
    let mse = total_squared_error / (n_samples * n_features) as f32;
    let rmse = mse.sqrt();

    println!("Reconstruction Error Metrics:");
    println!("   MSE:        {mse:.6}");
    println!("   RMSE:       {rmse:.6}");
    println!("   Max Error:  {max_error:.6}\n");

    println!("Sample Reconstruction (First 3 samples):");
    println!(
        "{:>8} {:>10} {:>14}",
        "Feature", "Original", "Reconstructed"
    );
    println!("{}", "─".repeat(34));

    let feature_names = ["Sepal L", "Sepal W", "Petal L", "Petal W"];
    for sample_idx in 0..3 {
        println!("\nSample {sample_idx}:");
        for (j, fname) in feature_names.iter().enumerate() {
            println!(
                "{:>8} {:>10.4} {:>14.4}",
                fname,
                data.get(sample_idx, j),
                reconstructed.get(sample_idx, j)
            );
        }
    }
    println!();

    println!("💡 Interpretation:");
    println!("   RMSE of {rmse:.4} means typical reconstruction error is small");
    println!("   Dimensionality reduction preserved most information!\n");

    rmse
}

/// Step 6: Show principal component loadings.
fn show_component_loadings(pca: &PCA) {
    println!("═══════════════════════════════════════════════════════");
    println!("  Step 6: Principal Components (Feature Loadings)");
    println!("═══════════════════════════════════════════════════════\n");

    let components = pca.components().expect("Example data should be valid");
    println!(
        "{:>10} {:>10} {:>10} {:>10} {:>10}",
        "Component", "Sepal L", "Sepal W", "Petal L", "Petal W"
    );
    println!("{}", "─".repeat(54));

    for i in 0..2 {
        println!(
            "{:>10} {:>10.4} {:>10.4} {:>10.4} {:>10.4}",
            format!("PC{}", i + 1),
            components.get(i, 0),
            components.get(i, 1),
            components.get(i, 2),
            components.get(i, 3)
        );
    }
    println!();

    println!("💡 Feature Importance:");
    println!("   Larger absolute values = more important for that component");
    println!("   PC1 likely captures overall flower size");
    println!("   PC2 likely captures petal vs sepal differences\n");
}

/// Print the final summary.
fn print_summary(total_variance: f32, rmse: f32) {
    println!("═══════════════════════════════════════════════════════");
    println!("  Summary");
    println!("═══════════════════════════════════════════════════════\n");

    println!("Key Findings:");
    println!(
        "  • Reduced 4D iris data to 2D with {:.1}% variance retained",
        total_variance * 100.0
    );
    println!("  • RMSE of {rmse:.4} shows good reconstruction quality");
    println!("  • 2D representation suitable for visualization");
    println!("  • Three species likely separable in 2D PC space\n");

    println!("When to use PCA:");
    println!("  ✓ Visualizing high-dimensional data");
    println!("  ✓ Reducing features before ML training");
    println!("  ✓ Removing correlated features");
    println!("  ✓ Denoising data");
    println!("  ✓ Compression\n");

    println!("⚠️  PCA Assumptions:");
    println!("  • Linear relationships between features");
    println!("  • High variance = high importance");
    println!("  • Features should be standardized first\n");

    println!("🚀 Performance Notes:");
    println!("  • nalgebra's SymmetricEigen: O(n³) for n×n covariance matrix");
    println!("  • Transform: O(n_samples × n_components × n_features)");
    println!("  • Memory: O(n_components × n_features) for components\n");

    println!("═══════════════════════════════════════════════════════");
    println!("  Example Complete!");
    println!("═══════════════════════════════════════════════════════");
}

fn main() {
    println!("═══════════════════════════════════════════════════════");
    println!("  PCA Dimensionality Reduction - Iris Dataset");
    println!("═══════════════════════════════════════════════════════\n");

    let (data, species) = create_iris_data();

    let (n_samples, n_features) = data.shape();
    println!("📊 Dataset Information:");
    println!("   Samples: {n_samples}");
    println!("   Features: {n_features} (sepal length, sepal width, petal length, petal width)");
    println!("   Species: Setosa, Versicolor, Virginica (10 each)\n");

    let (scaler, pca, _scaled_data, transformed) = standardize_and_reduce(&data);
    let total_variance = analyze_explained_variance(&pca);
    display_transformed_data(&transformed, &species);
    let rmse = reconstruction_analysis(&data, &pca, &scaler, &transformed);
    show_component_loadings(&pca);
    print_summary(total_variance, rmse);
}
