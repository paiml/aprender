#![allow(clippy::expect_used, clippy::unwrap_used)]
//! Basic Scatter Plot Example
//!
//! Demonstrates creating a simple scatter plot with trueno-viz.
//! This example shows the fundamental workflow: data → plot → render.
//!
//! Run with: `cargo run --example scatter_basic`

use trueno_viz::output::PngEncoder;
use trueno_viz::prelude::*;

fn main() {
    println!("Basic Scatter Plot Example");
    println!("==========================\n");

    // Step 1: Prepare sample data
    println!("Step 1: Preparing sample data...");
    let (x_data, y_data) = generate_sample_data();
    println!("  Generated {} data points", x_data.len());

    // Step 2: Create scatter plot using builder pattern
    println!("\nStep 2: Creating scatter plot...");
    let plot = ScatterPlot::new()
        .x(&x_data)
        .y(&y_data)
        .color(Rgba::BLUE)
        .size(6.0)
        .alpha(0.7)
        .dimensions(800, 600)
        .build()
        .expect("Failed to build scatter plot");

    println!("  Plot dimensions: 800x600 pixels");
    println!("  Point count: {}", plot.point_count());

    // Step 3: Render to framebuffer
    println!("\nStep 3: Rendering to framebuffer...");
    let fb = plot.to_framebuffer().expect("Failed to render");

    println!("  Framebuffer size: {}x{}", fb.width(), fb.height());
    println!("  SIMD backend: {:?}", Framebuffer::backend());

    // Step 4: Save to PNG file
    println!("\nStep 4: Saving to PNG...");
    let output_path = "scatter_basic.png";
    PngEncoder::write_to_file(&fb, output_path).expect("Failed to write PNG");

    println!("  Saved to: {output_path}");

    // Display summary
    println!("\n--- Summary ---");
    println!(
        "Data range X: [{:.2}, {:.2}]",
        x_data.iter().copied().fold(f32::INFINITY, f32::min),
        x_data.iter().copied().fold(f32::NEG_INFINITY, f32::max)
    );
    println!(
        "Data range Y: [{:.2}, {:.2}]",
        y_data.iter().copied().fold(f32::INFINITY, f32::min),
        y_data.iter().copied().fold(f32::NEG_INFINITY, f32::max)
    );
    println!("\nScatter plot successfully generated!");
}

/// Generate sample data with a linear relationship plus noise.
fn generate_sample_data() -> (Vec<f32>, Vec<f32>) {
    let n = 100;
    let mut x_data = Vec::with_capacity(n);
    let mut y_data = Vec::with_capacity(n);

    // Simple linear relationship: y = 2x + noise
    for i in 0..n {
        let x = (i as f32) / (n as f32) * 10.0;
        // Add some pseudo-random noise using a simple LCG
        let noise = ((i * 1_103_515_245 + 12345) % 100) as f32 / 50.0 - 1.0;
        let y = 2.0 * x + noise * 2.0;

        x_data.push(x);
        y_data.push(y);
    }

    (x_data, y_data)
}
