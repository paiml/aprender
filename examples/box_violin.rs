#![allow(clippy::expect_used, clippy::unwrap_used)]
//! Box and Violin Plot Example
//!
//! Demonstrates creating box plots and violin plots for
//! visualizing data distributions across multiple groups.
//!
//! Run with: `cargo run --example box_violin`

use trueno_viz::output::PngEncoder;
use trueno_viz::plots::{BoxPlot, BoxStats, ViolinPlot};
use trueno_viz::prelude::WithDimensions;

fn main() {
    println!("Box and Violin Plot Example");
    println!("============================\n");

    // Step 1: Generate sample data for three groups
    println!("Step 1: Generating sample data...");
    let (group_a, group_b, group_c) = generate_sample_data();

    println!("  Group A: {} samples", group_a.len());
    println!("  Group B: {} samples", group_b.len());
    println!("  Group C: {} samples", group_c.len());

    // Step 2: Compute and display statistics
    println!("\nStep 2: Computing statistics...");

    for (name, data) in [("A", &group_a), ("B", &group_b), ("C", &group_c)] {
        if let Some(stats) = BoxStats::from_data(data) {
            println!("\n  Group {name}:");
            println!("    Min:    {:.2}", stats.min);
            println!("    Q1:     {:.2}", stats.q1);
            println!("    Median: {:.2}", stats.median);
            println!("    Q3:     {:.2}", stats.q3);
            println!("    Max:    {:.2}", stats.max);
            println!("    IQR:    {:.2}", stats.iqr);
            if !stats.outliers.is_empty() {
                println!("    Outliers: {:?}", stats.outliers);
            }
        }
    }

    // Step 3: Create box plot
    println!("\nStep 3: Creating box plot...");
    let boxplot = BoxPlot::new()
        .add_group(&group_a, "Control")
        .add_group(&group_b, "Treatment A")
        .add_group(&group_c, "Treatment B")
        .dimensions(600, 400)
        .margin(50)
        .box_width(0.6)
        .show_outliers(true)
        .build()
        .expect("Failed to build box plot");

    println!("  Groups: {}", boxplot.num_groups());

    let fb_box = boxplot.to_framebuffer().expect("Failed to render");
    PngEncoder::write_to_file(&fb_box, "boxplot.png").expect("Failed to write PNG");
    println!("  Saved: boxplot.png");

    // Step 4: Create violin plot
    println!("\nStep 4: Creating violin plot...");
    let violin = ViolinPlot::new()
        .add_group(&group_a, "Control")
        .add_group(&group_b, "Treatment A")
        .add_group(&group_c, "Treatment B")
        .dimensions(600, 400)
        .margin(50)
        .show_box(true)
        .build()
        .expect("Failed to build violin plot");

    println!("  Groups: {}", violin.num_groups());

    let fb_violin = violin.to_framebuffer().expect("Failed to render");
    PngEncoder::write_to_file(&fb_violin, "violin.png").expect("Failed to write PNG");
    println!("  Saved: violin.png");

    // Summary
    println!("\n--- Summary ---");
    println!("Box plots show: min, Q1, median, Q3, max, and outliers");
    println!("Violin plots show: kernel density estimate + inner box plot");
    println!("\nBox and violin plots successfully generated!");
}

/// Generate sample data with different distributions.
fn generate_sample_data() -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    // Group A: Normal-like distribution centered at 50
    let group_a: Vec<f32> = (0i32..100)
        .map(|i| {
            let base = 50.0;
            let noise = (i.wrapping_mul(1103).wrapping_add(12345) % 1000) as f32 / 50.0 - 10.0;
            base + noise
        })
        .collect();

    // Group B: Higher mean, tighter distribution
    let group_b: Vec<f32> = (0i32..100)
        .map(|i| {
            let base = 65.0;
            let noise = (i.wrapping_mul(6361).wrapping_add(1) % 1000) as f32 / 100.0 - 5.0;
            base + noise
        })
        .collect();

    // Group C: Lower mean, with some outliers
    let mut group_c: Vec<f32> = (0i32..95)
        .map(|i| {
            let base = 40.0;
            let noise = (i.wrapping_mul(7919).wrapping_add(1047) % 1000) as f32 / 62.5 - 8.0;
            base + noise
        })
        .collect();
    // Add some outliers
    group_c.extend_from_slice(&[10.0, 12.0, 85.0, 90.0, 95.0]);

    (group_a, group_b, group_c)
}
