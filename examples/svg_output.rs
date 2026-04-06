#![allow(clippy::expect_used, clippy::unwrap_used)]
//! SVG Output Example
//!
//! Demonstrates creating SVG visualizations with both
//! raster (embedded PNG) and vector graphics.
//!
//! Run with: `cargo run --example svg_output`

use trueno_viz::color::Rgba;
use trueno_viz::output::{SvgEncoder, TextAnchor};
use trueno_viz::prelude::*;

fn main() {
    println!("SVG Output Example");
    println!("==================\n");

    // Example 1: Vector SVG (scalable graphics)
    println!("Example 1: Vector SVG");
    println!("---------------------");

    let vector_svg = SvgEncoder::new(400, 300)
        .background(Some(Rgba::new(245, 245, 245, 255)))
        // Title
        .text_anchored(200.0, 30.0, "Vector SVG Chart", 18.0, Rgba::BLACK, TextAnchor::Middle)
        // Axes
        .line(50.0, 250.0, 350.0, 250.0, Rgba::BLACK, 2.0) // X-axis
        .line(50.0, 50.0, 50.0, 250.0, Rgba::BLACK, 2.0) // Y-axis
        // Data bars
        .rect(70.0, 150.0, 40.0, 100.0, Rgba::new(66, 133, 244, 200))
        .rect(130.0, 100.0, 40.0, 150.0, Rgba::new(52, 168, 83, 200))
        .rect(190.0, 180.0, 40.0, 70.0, Rgba::new(251, 188, 4, 200))
        .rect(250.0, 80.0, 40.0, 170.0, Rgba::new(234, 67, 53, 200))
        // Labels
        .text_anchored(90.0, 270.0, "Q1", 12.0, Rgba::BLACK, TextAnchor::Middle)
        .text_anchored(150.0, 270.0, "Q2", 12.0, Rgba::BLACK, TextAnchor::Middle)
        .text_anchored(210.0, 270.0, "Q3", 12.0, Rgba::BLACK, TextAnchor::Middle)
        .text_anchored(270.0, 270.0, "Q4", 12.0, Rgba::BLACK, TextAnchor::Middle)
        // Y-axis labels
        .text_anchored(40.0, 255.0, "0", 10.0, Rgba::BLACK, TextAnchor::End)
        .text_anchored(40.0, 155.0, "50", 10.0, Rgba::BLACK, TextAnchor::End)
        .text_anchored(40.0, 55.0, "100", 10.0, Rgba::BLACK, TextAnchor::End);

    vector_svg.write_to_file("vector_chart.svg").expect("Failed to write SVG");
    println!("  Saved: vector_chart.svg");
    println!("  Size: {} bytes\n", vector_svg.render().len());

    // Example 2: Scatter plot with circles
    println!("Example 2: Scatter Plot (Vector)");
    println!("--------------------------------");

    let mut scatter_svg = SvgEncoder::new(400, 300)
        .background(Some(Rgba::WHITE))
        .text_anchored(200.0, 25.0, "Scatter Plot", 16.0, Rgba::BLACK, TextAnchor::Middle)
        // Plot area
        .rect_outlined(
            50.0,
            40.0,
            300.0,
            220.0,
            Rgba::new(250, 250, 250, 255),
            Rgba::new(200, 200, 200, 255),
            1.0,
        );

    // Add scatter points
    let points = generate_scatter_data();
    for (x, y, size) in &points {
        let color = Rgba::new(66, 133, 244, 180);
        scatter_svg = scatter_svg.circle(*x, *y, *size, color);
    }

    scatter_svg.write_to_file("scatter_vector.svg").expect("Failed to write SVG");
    println!("  Points: {}", points.len());
    println!("  Saved: scatter_vector.svg\n");

    // Example 3: Raster SVG from framebuffer
    println!("Example 3: Raster SVG (embedded PNG)");
    println!("------------------------------------");

    // Create a scatter plot using the trueno-viz rendering
    let x_data: Vec<f32> = (0..50).map(|i| i as f32 / 5.0).collect();
    let y_data: Vec<f32> = x_data.iter().map(|&x| (x * 0.5).sin() * 3.0 + x).collect();

    let plot = ScatterPlot::new()
        .x(&x_data)
        .y(&y_data)
        .color(Rgba::new(200, 50, 50, 255))
        .size(6.0)
        .dimensions(400, 300)
        .build()
        .expect("Failed to build plot");

    let fb = plot.to_framebuffer().expect("Failed to render");
    let raster_svg = SvgEncoder::from_framebuffer(&fb).expect("Failed to create SVG");

    raster_svg.write_to_file("scatter_raster.svg").expect("Failed to write SVG");
    println!("  Framebuffer: {}x{}", fb.width(), fb.height());
    println!("  Saved: scatter_raster.svg");
    println!("  Size: {} bytes (includes embedded PNG)\n", raster_svg.render().len());

    // Example 4: Complex path
    println!("Example 4: SVG Path");
    println!("-------------------");

    let path_svg = SvgEncoder::new(200, 200)
        .background(Some(Rgba::WHITE))
        // Star shape using path
        .path(
            "M 100 10 L 120 80 L 190 80 L 135 120 L 155 190 L 100 150 L 45 190 L 65 120 L 10 80 L 80 80 Z",
            Some(Rgba::new(255, 215, 0, 255)), // Gold fill
            Some(Rgba::new(184, 134, 11, 255)), // Dark gold stroke
            2.0,
        )
        .text_anchored(100.0, 195.0, "Star Path", 12.0, Rgba::BLACK, TextAnchor::Middle);

    path_svg.write_to_file("star_path.svg").expect("Failed to write SVG");
    println!("  Saved: star_path.svg\n");

    // Summary
    println!("--- Summary ---");
    println!("Vector SVG: Scalable, small file size, editable");
    println!("Raster SVG: Preserves exact pixels, larger file size");
    println!("\nSVG output successfully generated!");
}

/// Generate random scatter data points.
fn generate_scatter_data() -> Vec<(f32, f32, f32)> {
    (0i32..30)
        .map(|i| {
            let x = 60.0 + (i.wrapping_mul(17) % 280) as f32;
            let y = 50.0 + (i.wrapping_mul(23) % 200) as f32;
            let size = 3.0 + (i % 5) as f32;
            (x, y, size)
        })
        .collect()
}
