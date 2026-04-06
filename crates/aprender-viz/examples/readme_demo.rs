#![allow(clippy::expect_used, clippy::unwrap_used)]
//! Demo for README.md - outputs ASCII scatter plot to stdout.
//! Used by `make readme` to generate reproducible documentation.

use trueno_viz::output::{TerminalEncoder, TerminalMode};
use trueno_viz::prelude::*;

fn main() {
    // Generate scatter data: y = x^2 with noise
    let n = 40;
    let (x, y): (Vec<f32>, Vec<f32>) = (0..n)
        .map(|i| {
            let x = (i as f32 / n as f32) * 6.0 - 3.0;
            let noise = ((i * 7919) % 100) as f32 / 100.0 - 0.5;
            (x, x * x + noise * 2.0)
        })
        .unzip();

    let plot = ScatterPlot::new()
        .x(&x)
        .y(&y)
        .color(Rgba::BLUE)
        .size(4.0)
        .dimensions(200, 120)
        .build()
        .expect("Failed to build plot");

    let fb = plot.to_framebuffer().expect("Failed to render");

    let encoder = TerminalEncoder::new().mode(TerminalMode::Ascii).width(60).invert(true);

    print!("{}", encoder.render(&fb));
}
