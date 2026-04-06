#![allow(clippy::expect_used, clippy::unwrap_used)]
//! Text Prompt DSL Example
//!
//! Demonstrates creating visualizations using a simple text-based
//! domain-specific language (DSL) for declarative visualization.
//!
//! Run with: `cargo run --example text_prompt`

use trueno_viz::output::PngEncoder;
use trueno_viz::prompt::{from_prompt, parse_prompt};

fn main() {
    println!("Text Prompt DSL Example");
    println!("=======================\n");

    // Example 1: Scatter plot
    println!("Example 1: Scatter Plot");
    println!("-----------------------");

    let prompt = "scatter x=[1,2,3,4,5,6,7,8] y=[2,4,3,6,5,8,7,9] color=blue size=8";
    println!("  Prompt: {prompt}");

    let spec = parse_prompt(prompt).expect("Failed to parse");
    println!(
        "  Parsed: {} plot, {} points",
        spec.plot_type,
        spec.x_data.as_ref().expect("parsing should succeed").len()
    );

    let fb = spec.render().expect("Failed to render");
    PngEncoder::write_to_file(&fb, "prompt_scatter.png").expect("Failed to write");
    println!("  Saved: prompt_scatter.png\n");

    // Example 2: Line chart with custom dimensions
    println!("Example 2: Line Chart");
    println!("---------------------");

    let prompt = "line x=[0,1,2,3,4,5,6,7,8,9,10] y=[0,1,4,9,16,25,36,49,64,81,100] width=500 height=400 color=red";
    println!("  Prompt: {prompt}");

    let fb = from_prompt(prompt).expect("Failed to create");
    PngEncoder::write_to_file(&fb, "prompt_line.png").expect("Failed to write");
    println!("  Saved: prompt_line.png (quadratic curve)\n");

    // Example 3: Histogram
    println!("Example 3: Histogram");
    println!("--------------------");

    let prompt = "histogram data=[1,2,2,3,3,3,4,4,4,4,5,5,5,6,6,7] color=green";
    println!("  Prompt: {prompt}");

    let fb = from_prompt(prompt).expect("Failed to create");
    PngEncoder::write_to_file(&fb, "prompt_histogram.png").expect("Failed to write");
    println!("  Saved: prompt_histogram.png\n");

    // Example 4: Heatmap with hex color
    println!("Example 4: Heatmap");
    println!("------------------");

    let prompt = "heatmap matrix=[[1,2,3,4],[5,6,7,8],[9,10,11,12]] width=400 height=300";
    println!("  Prompt: {prompt}");

    let fb = from_prompt(prompt).expect("Failed to create");
    PngEncoder::write_to_file(&fb, "prompt_heatmap.png").expect("Failed to write");
    println!("  Saved: prompt_heatmap.png (3x4 matrix)\n");

    // Example 5: Box plot with groups
    println!("Example 5: Box Plot");
    println!("-------------------");

    let prompt = "boxplot groups=[[2,3,4,5,6,7,8],[4,5,6,7,8,9,10],[1,2,3,4,5,6,7,8,9,10,11,12]]";
    println!("  Prompt: {prompt}");

    let fb = from_prompt(prompt).expect("Failed to create");
    PngEncoder::write_to_file(&fb, "prompt_boxplot.png").expect("Failed to write");
    println!("  Saved: prompt_boxplot.png (3 groups)\n");

    // Example 6: Full syntax demonstration
    println!("Example 6: Full Syntax");
    println!("----------------------");

    let prompt = "scatter x=[1,2,3,4,5] y=[5,4,3,2,1] width=300 height=200 color=#ff6600 size=12.0";
    println!("  Prompt: {prompt}");

    let spec = parse_prompt(prompt).expect("Failed to parse");
    println!("  Dimensions: {}x{}", spec.width, spec.height);
    println!("  Color: RGBA({},{},{},{})", spec.color.r, spec.color.g, spec.color.b, spec.color.a);
    println!("  Size: {}", spec.size);

    let fb = spec.render().expect("Failed to render");
    PngEncoder::write_to_file(&fb, "prompt_custom.png").expect("Failed to write");
    println!("  Saved: prompt_custom.png\n");

    // Print DSL syntax reference
    println!("--- DSL Syntax Reference ---");
    println!();
    println!("  <plot_type> <data_spec> [options...]");
    println!();
    println!("  Plot types:");
    println!("    scatter     - Scatter plot (requires x=, y=)");
    println!("    line        - Line chart (requires x=, y=)");
    println!("    histogram   - Histogram (requires data=)");
    println!("    heatmap     - Heatmap (requires matrix=)");
    println!("    boxplot     - Box plot (requires groups=)");
    println!();
    println!("  Data specs:");
    println!("    x=[1,2,3] y=[4,5,6]   - Paired x/y data");
    println!("    data=[1,2,3,4,5]      - Single data array");
    println!("    matrix=[[1,2],[3,4]]  - 2D matrix data");
    println!("    groups=[[1,2],[3,4]]  - Multiple groups");
    println!();
    println!("  Options:");
    println!("    width=800 height=600  - Dimensions in pixels");
    println!("    color=red|blue|#hex   - Colors (named or hex)");
    println!("    size=5.0              - Point/line size");
    println!();
    println!("Text prompt visualizations generated successfully!");
}
