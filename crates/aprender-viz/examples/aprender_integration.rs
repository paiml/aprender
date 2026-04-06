#![allow(clippy::expect_used, clippy::unwrap_used)]
//! Aprender ML Library Integration Example
//!
//! Demonstrates using trueno-viz visualization extensions with aprender
//! data types (Vector, Matrix, `DataFrame`).
//!
//! Run with: `cargo run --example aprender_integration --features ml`

use aprender::data::DataFrame;
use aprender::primitives::{Matrix, Vector};
use trueno_viz::interop::aprender::{DataFrameViz, MatrixViz, VectorViz};
use trueno_viz::output::PngEncoder;

fn main() {
    println!("Aprender ML Integration Example");
    println!("================================\n");

    // Example 1: Vector histogram
    println!("1. Creating histogram from Vector...");
    let values = Vector::from_slice(&[
        1.2, 2.5, 2.1, 3.8, 3.2, 3.5, 4.1, 4.8, 5.0, 2.9, 3.1, 3.7, 4.2, 4.5, 3.9,
    ]);
    let fb = values.to_histogram().expect("Failed to create histogram");
    PngEncoder::write_to_file(&fb, "aprender_histogram.png").expect("Failed to write PNG");
    println!("   Saved: aprender_histogram.png ({}x{})\n", fb.width(), fb.height());

    // Example 2: Predictions vs Actual scatter plot
    println!("2. Creating predictions vs actual scatter plot...");
    let predictions = Vector::from_slice(&[2.1, 4.2, 3.8, 5.1, 6.3, 7.0, 8.2]);
    let actual = Vector::from_slice(&[2.0, 4.0, 4.0, 5.0, 6.0, 7.2, 8.0]);
    let fb = predictions.scatter_vs(&actual).expect("Failed to create scatter");
    PngEncoder::write_to_file(&fb, "aprender_pred_vs_actual.png").expect("Failed to write PNG");
    println!("   Saved: aprender_pred_vs_actual.png ({}x{})\n", fb.width(), fb.height());

    // Example 3: Residual plot
    println!("3. Creating residual plot...");
    let fb = predictions.residual_plot(&actual).expect("Failed to create residual plot");
    PngEncoder::write_to_file(&fb, "aprender_residuals.png").expect("Failed to write PNG");
    println!("   Saved: aprender_residuals.png ({}x{})\n", fb.width(), fb.height());

    // Example 4: Training loss curve
    println!("4. Creating loss curve from Vector...");
    let losses = Vector::from_slice(&[
        2.5, 1.8, 1.4, 1.1, 0.9, 0.75, 0.62, 0.52, 0.45, 0.40, 0.36, 0.33, 0.31, 0.29, 0.28,
    ]);
    let fb = losses.to_line().expect("Failed to create loss curve");
    PngEncoder::write_to_file(&fb, "aprender_loss_curve.png").expect("Failed to write PNG");
    println!("   Saved: aprender_loss_curve.png ({}x{})\n", fb.width(), fb.height());

    // Example 5: Matrix heatmap
    println!("5. Creating heatmap from Matrix...");
    #[rustfmt::skip]
    let matrix_data = vec![
        1.0, 0.8, 0.2, 0.1,
        0.8, 1.0, 0.3, 0.2,
        0.2, 0.3, 1.0, 0.7,
        0.1, 0.2, 0.7, 1.0,
    ];
    let matrix = Matrix::from_vec(4, 4, matrix_data).expect("Failed to create matrix");
    let fb = matrix.to_heatmap().expect("Failed to create heatmap");
    PngEncoder::write_to_file(&fb, "aprender_heatmap.png").expect("Failed to write PNG");
    println!("   Saved: aprender_heatmap.png ({}x{})\n", fb.width(), fb.height());

    // Example 6: Correlation matrix heatmap
    println!("6. Creating correlation heatmap from Matrix...");
    let fb = matrix.correlation_heatmap().expect("Failed to create correlation heatmap");
    PngEncoder::write_to_file(&fb, "aprender_correlation.png").expect("Failed to write PNG");
    println!("   Saved: aprender_correlation.png ({}x{})\n", fb.width(), fb.height());

    // Example 7: DataFrame scatter plot
    println!("7. Creating scatter plot from DataFrame columns...");
    let columns = vec![
        ("feature_1".to_string(), Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])),
        ("feature_2".to_string(), Vector::from_slice(&[2.1, 3.9, 6.2, 7.8, 10.1, 11.9])),
        ("target".to_string(), Vector::from_slice(&[0.5, 1.2, 1.8, 2.5, 3.1, 3.8])),
    ];
    let df = DataFrame::new(columns).expect("Failed to create DataFrame");
    let fb = df.scatter("feature_1", "feature_2").expect("Failed to create scatter");
    PngEncoder::write_to_file(&fb, "aprender_df_scatter.png").expect("Failed to write PNG");
    println!("   Saved: aprender_df_scatter.png ({}x{})\n", fb.width(), fb.height());

    // Example 8: DataFrame histogram
    println!("8. Creating histogram from DataFrame column...");
    let fb = df.histogram("target").expect("Failed to create histogram");
    PngEncoder::write_to_file(&fb, "aprender_df_histogram.png").expect("Failed to write PNG");
    println!("   Saved: aprender_df_histogram.png ({}x{})\n", fb.width(), fb.height());

    // Example 9: DataFrame box plot
    println!("9. Creating box plot from DataFrame columns...");
    let fb = df.boxplot(&["feature_1", "feature_2"]).expect("Failed to create boxplot");
    PngEncoder::write_to_file(&fb, "aprender_df_boxplot.png").expect("Failed to write PNG");
    println!("   Saved: aprender_df_boxplot.png ({}x{})\n", fb.width(), fb.height());

    // Example 10: DataFrame line plot
    println!("10. Creating line plot from DataFrame column...");
    let fb = df.line("target").expect("Failed to create line plot");
    PngEncoder::write_to_file(&fb, "aprender_df_line.png").expect("Failed to write PNG");
    println!("   Saved: aprender_df_line.png ({}x{})\n", fb.width(), fb.height());

    println!("--- Summary ---");
    println!("Generated 10 visualization files demonstrating aprender integration:");
    println!("  - Vector: histogram, scatter_vs, residual_plot, to_line");
    println!("  - Matrix: to_heatmap, correlation_heatmap");
    println!("  - DataFrame: scatter, histogram, boxplot, line");
    println!("\nAll visualizations successfully generated!");
}
