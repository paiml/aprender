#![allow(clippy::expect_used, clippy::unwrap_used)]
//! Heatmap Correlation Matrix Example
//!
//! Demonstrates creating a correlation matrix heatmap, commonly used
//! for visualizing relationships between multiple variables.
//!
//! Run with: `cargo run --example heatmap_correlation`

use trueno_viz::output::PngEncoder;
use trueno_viz::plots::{Heatmap, HeatmapPalette};
use trueno_viz::prelude::WithDimensions;

fn main() {
    println!("Heatmap Correlation Matrix Example");
    println!("===================================\n");

    // Step 1: Create a sample correlation matrix
    println!("Step 1: Creating correlation matrix...");
    let (matrix, labels) = create_correlation_matrix();

    println!("  Matrix size: {}x{}", labels.len(), labels.len());
    println!("  Variables: {labels:?}");

    // Step 2: Build the heatmap visualization
    println!("\nStep 2: Building heatmap...");
    let heatmap = Heatmap::new()
        .data_2d(&matrix)
        .palette(HeatmapPalette::RedBlue) // Diverging palette for correlations
        .dimensions(600, 600)
        .margin(40)
        .borders(true)
        .build()
        .expect("Failed to build heatmap");

    println!("  Cells: {}", heatmap.cell_count());
    println!("  Palette: RedBlue (diverging)");

    // Step 3: Render to framebuffer
    println!("\nStep 3: Rendering...");
    let fb = heatmap.to_framebuffer().expect("Failed to render");

    println!("  Framebuffer: {}x{}", fb.width(), fb.height());

    // Step 4: Save output
    println!("\nStep 4: Saving to PNG...");
    let output_path = "heatmap_correlation.png";
    PngEncoder::write_to_file(&fb, output_path).expect("Failed to write PNG");

    println!("  Saved to: {output_path}");

    // Display the correlation matrix values
    println!("\n--- Correlation Matrix ---");
    print!("        ");
    for label in &labels {
        print!("{label:>8}");
    }
    println!();

    for (i, row) in matrix.iter().enumerate() {
        print!("{:>8}", labels[i]);
        for &val in row {
            print!("{val:>8.2}");
        }
        println!();
    }

    println!("\nHeatmap successfully generated!");
}

/// Create a sample correlation matrix with realistic values.
///
/// Returns the matrix data and variable labels.
fn create_correlation_matrix() -> (Vec<Vec<f32>>, Vec<&'static str>) {
    let labels = vec!["Height", "Weight", "Age", "Income", "Score"];

    // Correlation matrix (symmetric, diagonal = 1.0)
    // Positive correlations: Height-Weight, Income-Score
    // Negative correlations: Age-Score
    let matrix = vec![
        vec![1.00, 0.85, 0.12, 0.23, 0.15],  // Height
        vec![0.85, 1.00, 0.18, 0.31, 0.22],  // Weight
        vec![0.12, 0.18, 1.00, 0.45, -0.35], // Age
        vec![0.23, 0.31, 0.45, 1.00, 0.67],  // Income
        vec![0.15, 0.22, -0.35, 0.67, 1.00], // Score
    ];

    (matrix, labels)
}
