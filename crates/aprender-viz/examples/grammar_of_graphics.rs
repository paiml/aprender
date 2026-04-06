#![allow(clippy::expect_used, clippy::unwrap_used)]
//! Grammar of Graphics Example
//!
//! Demonstrates the declarative, composable visualization API
//! based on Wilkinson's Grammar of Graphics.
//!
//! Run with: `cargo run --example grammar_of_graphics`

use trueno_viz::color::Rgba;
use trueno_viz::grammar::{Aes, Coord, DataFrame, GGPlot, Geom, Layer, Theme};
use trueno_viz::output::PngEncoder;

fn main() {
    println!("Grammar of Graphics Example");
    println!("===========================\n");

    example_scatter();
    example_line_points();
    example_bar();
    example_area();
    example_dark_theme();
    example_reference_lines();

    // Print API summary
    println!("--- Grammar of Graphics API ---");
    println!();
    println!("  GGPlot::new()           Create a new plot");
    println!("    .data_xy(&x, &y)      Set x/y data directly");
    println!("    .data(DataFrame)      Set data from DataFrame");
    println!("    .aes(Aes::new()...)   Set global aesthetics");
    println!("    .geom(Geom::...)      Add geometry layer");
    println!("    .coord(Coord::...)    Set coordinate system");
    println!("    .theme(Theme::...)    Set visual theme");
    println!("    .dimensions(w, h)     Set output size");
    println!("    .build()              Build the plot");
    println!();
    println!("  Geometries: point(), line(), area(), bar()");
    println!("  Themes: grey(), minimal(), bw(), classic(), dark(), void()");
    println!("  Coords: cartesian(), polar(), fixed()");
    println!();
    println!("Grammar of Graphics examples generated successfully!");
}

fn example_scatter() {
    println!("Example 1: Basic Scatter Plot");
    println!("-----------------------------");

    let plot = GGPlot::new()
        .data_xy(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            &[2.0, 4.0, 3.0, 5.0, 7.0, 6.0, 8.0, 9.0],
        )
        .geom(Geom::point())
        .aes(Aes::new().color_value(Rgba::new(66, 133, 244, 255)).size_value(10.0))
        .theme(Theme::minimal())
        .dimensions(500, 400)
        .build()
        .expect("Failed to build plot");

    let fb = plot.to_framebuffer().expect("Failed to render");
    PngEncoder::write_to_file(&fb, "ggplot_scatter.png").expect("Failed to write");
    println!("  Saved: ggplot_scatter.png\n");
}

fn example_line_points() {
    println!("Example 2: Line + Points (Layered)");
    println!("----------------------------------");

    let x: Vec<f32> = (0..=10).map(|i| i as f32).collect();
    let y: Vec<f32> = x.iter().map(|&v| v * v / 10.0).collect();

    let plot = GGPlot::new()
        .data_xy(&x, &y)
        .geom(Geom::line())
        .geom(Geom::point().aes(Aes::new().color_value(Rgba::RED).size_value(8.0)))
        .theme(Theme::bw())
        .dimensions(500, 400)
        .build()
        .expect("Failed to build plot");

    let fb = plot.to_framebuffer().expect("Failed to render");
    PngEncoder::write_to_file(&fb, "ggplot_line_points.png").expect("Failed to write");
    println!("  Saved: ggplot_line_points.png (quadratic curve)\n");
}

fn example_bar() {
    println!("Example 3: Bar Chart");
    println!("--------------------");

    let categories: Vec<f32> = (1..=5).map(|i| i as f32).collect();
    let values = vec![35.0, 52.0, 28.0, 45.0, 60.0];

    let plot = GGPlot::new()
        .data_xy(&categories, &values)
        .geom(Geom::bar().aes(Aes::new().color_value(Rgba::new(52, 168, 83, 255))))
        .coord(Coord::cartesian().ylim(0.0, 70.0))
        .theme(Theme::grey())
        .dimensions(500, 400)
        .build()
        .expect("Failed to build plot");

    let fb = plot.to_framebuffer().expect("Failed to render");
    PngEncoder::write_to_file(&fb, "ggplot_bar.png").expect("Failed to write");
    println!("  Saved: ggplot_bar.png\n");
}

fn example_area() {
    println!("Example 4: Area Chart");
    println!("---------------------");

    let x: Vec<f32> = (0..20).map(|i| i as f32).collect();
    let y: Vec<f32> = x.iter().map(|&v| (v * 0.3).sin() * 20.0 + 30.0).collect();

    let plot = GGPlot::new()
        .data_xy(&x, &y)
        .geom(Geom::area().alpha(0.4).aes(Aes::new().color_value(Rgba::new(156, 39, 176, 255))))
        .coord(Coord::cartesian().ylim(0.0, 60.0))
        .theme(Theme::minimal())
        .dimensions(500, 400)
        .build()
        .expect("Failed to build plot");

    let fb = plot.to_framebuffer().expect("Failed to render");
    PngEncoder::write_to_file(&fb, "ggplot_area.png").expect("Failed to write");
    println!("  Saved: ggplot_area.png (sine wave)\n");
}

fn example_dark_theme() {
    println!("Example 5: Dark Theme");
    println!("---------------------");

    let x: Vec<f32> = (0..30).map(|i| i as f32).collect();
    let y1: Vec<f32> = x.iter().map(|&v| (v * 0.2).sin() * 10.0 + 20.0).collect();
    let y2: Vec<f32> = x.iter().map(|&v| (v * 0.2).cos() * 8.0 + 20.0).collect();

    let mut df = DataFrame::new();
    df.add_column_f32("x", &x);
    df.add_column_f32("y1", &y1);
    df.add_column_f32("y2", &y2);

    let plot = GGPlot::new()
        .data(df)
        .aes(Aes::new().x("x"))
        .layer(
            Layer::new(Geom::line())
                .aes(Aes::new().y("y1").color_value(Rgba::new(0, 255, 255, 255))),
        )
        .layer(
            Layer::new(Geom::line())
                .aes(Aes::new().y("y2").color_value(Rgba::new(255, 105, 180, 255))),
        )
        .theme(Theme::dark())
        .dimensions(600, 400)
        .build()
        .expect("Failed to build plot");

    let fb = plot.to_framebuffer().expect("Failed to render");
    PngEncoder::write_to_file(&fb, "ggplot_dark.png").expect("Failed to write");
    println!("  Saved: ggplot_dark.png (two series)\n");
}

fn example_reference_lines() {
    println!("Example 6: Reference Lines");
    println!("--------------------------");

    let x: Vec<f32> = (0..15).map(|i| i as f32).collect();
    let y: Vec<f32> = x.iter().map(|&v| v * 2.0 + 5.0 + (v * 0.5).sin() * 3.0).collect();

    let plot = GGPlot::new()
        .data_xy(&x, &y)
        .geom(Geom::point().aes(Aes::new().size_value(8.0)))
        .geom(Geom::hline(20.0).aes(Aes::new().color_value(Rgba::new(255, 0, 0, 150))))
        .geom(Geom::vline(7.0).aes(Aes::new().color_value(Rgba::new(0, 0, 255, 150))))
        .theme(Theme::classic())
        .dimensions(500, 400)
        .build()
        .expect("Failed to build plot");

    let fb = plot.to_framebuffer().expect("Failed to render");
    PngEncoder::write_to_file(&fb, "ggplot_reflines.png").expect("Failed to write");
    println!("  Saved: ggplot_reflines.png\n");
}
