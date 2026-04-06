#![allow(clippy::expect_used, clippy::unwrap_used)]
//! Terminal Output Example
//!
//! Demonstrates rendering visualizations directly to the terminal
//! using ASCII, Unicode half-blocks, and ANSI true color modes.
//!
//! Run with: `cargo run --example terminal_output`

use trueno_viz::output::{TerminalEncoder, TerminalMode};
use trueno_viz::prelude::*;

fn main() {
    println!("Terminal Output Example");
    println!("=======================\n");

    // Step 1: Create a sample scatter plot
    println!("Step 1: Creating scatter plot data...");
    let (x_data, y_data) = generate_sample_data();
    println!("  Generated {} data points\n", x_data.len());

    // Step 2: Build the scatter plot
    println!("Step 2: Building scatter plot...");
    let plot = ScatterPlot::new()
        .x(&x_data)
        .y(&y_data)
        .color(Rgba::BLUE)
        .size(8.0)
        .alpha(0.9)
        .dimensions(200, 100)
        .build()
        .expect("Failed to build scatter plot");

    // Step 3: Render to framebuffer
    let fb = plot.to_framebuffer().expect("Failed to render");
    println!("  Plot rendered to {}x{} framebuffer\n", fb.width(), fb.height());

    // Step 4: ASCII mode (widest compatibility)
    println!("Step 4: ASCII Mode (works in any terminal)");
    println!("{}", "-".repeat(42));

    let ascii_encoder = TerminalEncoder::new().mode(TerminalMode::Ascii).width(40).invert(true); // Dark background terminals benefit from invert

    print!("{}", ascii_encoder.render(&fb));
    println!();

    // Step 5: Unicode half-block mode (2x vertical resolution)
    println!("Step 5: Unicode Half-Block Mode (requires UTF-8 + ANSI)");
    println!("{}", "-".repeat(52));

    let unicode_encoder = TerminalEncoder::new().mode(TerminalMode::UnicodeHalfBlock).width(50);

    unicode_encoder.print(&fb);
    println!();

    // Step 6: ANSI true color mode (24-bit color)
    println!("Step 6: ANSI True Color Mode (requires 24-bit terminal)");
    println!("{}", "-".repeat(62));

    let ansi_encoder = TerminalEncoder::new().mode(TerminalMode::AnsiTrueColor).width(60);

    ansi_encoder.print(&fb);
    println!();

    // Step 7: Demonstrate with a gradient heatmap
    println!("Step 7: Gradient demonstration");
    println!("{}", "-".repeat(42));

    let heatmap_fb = create_gradient_framebuffer();

    let gradient_encoder = TerminalEncoder::new().mode(TerminalMode::UnicodeHalfBlock).width(40);

    gradient_encoder.print(&heatmap_fb);

    // Summary
    println!();
    println!("--- Terminal Modes Summary ---");
    println!("Ascii:            Widest compatibility, 10 gray levels");
    println!("UnicodeHalfBlock: 2x vertical resolution, requires UTF-8 + ANSI");
    println!("AnsiTrueColor:    Full 24-bit color, requires modern terminal");
    println!();
    println!("Terminal output successfully generated!");
}

/// Generate sample scatter plot data.
fn generate_sample_data() -> (Vec<f32>, Vec<f32>) {
    let n = 50;
    let mut x_data = Vec::with_capacity(n);
    let mut y_data = Vec::with_capacity(n);

    // Quadratic relationship with noise: y = 0.5*x^2 + noise
    for i in 0..n {
        let x = (i as f32) / (n as f32) * 10.0 - 5.0;
        let noise = ((i * 1_103_515_245 + 12345) % 100) as f32 / 50.0 - 1.0;
        let y = 0.5 * x * x + noise * 2.0;

        x_data.push(x);
        y_data.push(y);
    }

    (x_data, y_data)
}

/// Create a colorful gradient framebuffer for demonstration.
fn create_gradient_framebuffer() -> Framebuffer {
    let width = 100;
    let height = 40;
    let mut fb = Framebuffer::new(width, height).expect("framebuffer creation should succeed");

    for y in 0..height {
        for x in 0..width {
            // Create a colorful diagonal gradient
            let r = ((x as f32 / width as f32) * 255.0) as u8;
            let g = ((y as f32 / height as f32) * 255.0) as u8;
            let b = (((x + y) as f32 / (width + height) as f32) * 255.0) as u8;
            fb.set_pixel(x, y, Rgba::new(r, g, b, 255));
        }
    }

    fb
}
