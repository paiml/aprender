# Quick Start

This chapter walks through creating visualizations in under 5 minutes.

## Prerequisites

Add trueno-viz to your `Cargo.toml`:

```toml
[dependencies]
trueno-viz = "0.1"
```

For full features including GPU acceleration:

```toml
[dependencies]
trueno-viz = { version = "0.1", features = ["full"] }
```

## Your First Scatter Plot

```rust
use trueno_viz::prelude::*;
use trueno_viz::plots::ScatterPlot;

fn main() -> Result<()> {
    // Sample data: hours studied vs test score
    let hours = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let scores = vec![52.0, 58.0, 65.0, 71.0, 75.0, 82.0, 88.0, 91.0];

    // Create the plot
    let plot = ScatterPlot::new()
        .x(&hours)
        .y(&scores)
        .color(Rgba::new(66, 133, 244, 255))  // Google Blue
        .title("Study Hours vs Test Score")
        .xlabel("Hours Studied")
        .ylabel("Test Score")
        .build();

    // Render to PNG
    plot.render_to_file("study_scores.png")?;

    Ok(())
}
```

**Verification**: Run `cargo test scatter` to verify scatter plot functionality.

## Creating a Histogram

```rust
use trueno_viz::prelude::*;
use trueno_viz::plots::{Histogram, BinStrategy};

fn main() -> Result<()> {
    // Sample data: response times in milliseconds
    let response_times = vec![
        12.0, 15.0, 18.0, 22.0, 25.0, 28.0, 31.0, 35.0,
        38.0, 42.0, 45.0, 48.0, 52.0, 55.0, 95.0, 120.0,
    ];

    // Auto-binning with Sturges' formula
    let hist = Histogram::new(&response_times)
        .bins(BinStrategy::Sturges)
        .color(Rgba::new(234, 67, 53, 255))  // Google Red
        .title("API Response Times")
        .xlabel("Response Time (ms)")
        .ylabel("Frequency")
        .build();

    hist.render_to_file("response_times.png")?;

    Ok(())
}
```

**Test Reference**: `src/plots/histogram.rs::test_sturges_binning`

## Creating a Heatmap

```rust
use trueno_viz::prelude::*;
use trueno_viz::plots::{Heatmap, HeatmapPalette};

fn main() -> Result<()> {
    // 4x4 correlation matrix
    let data = vec![
        1.0, 0.8, 0.3, -0.2,
        0.8, 1.0, 0.5, 0.1,
        0.3, 0.5, 1.0, 0.6,
        -0.2, 0.1, 0.6, 1.0,
    ];

    let heatmap = Heatmap::new(&data, 4, 4)
        .palette(HeatmapPalette::Viridis)
        .title("Feature Correlation Matrix")
        .build();

    heatmap.render_to_file("correlation.png")?;

    Ok(())
}
```

**Test Reference**: `src/plots/heatmap.rs::test_heatmap_viridis`

## Terminal Output (ASCII Art)

For quick visualization without files:

```rust
use trueno_viz::prelude::*;
use trueno_viz::plots::ScatterPlot;
use trueno_viz::output::TerminalRenderer;

fn main() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![1.0, 4.0, 9.0, 16.0, 25.0];

    let plot = ScatterPlot::new()
        .x(&x)
        .y(&y)
        .build();

    // Render to terminal (80x24 characters)
    let ascii = TerminalRenderer::new(80, 24)
        .render(&plot);

    println!("{}", ascii);
}
```

Output:
```text
                         Scatter Plot
    25 ┤                                            ●
       │
    20 ┤
       │                                  ●
    15 ┤
       │
    10 ┤                        ●
       │
     5 ┤              ●
       │    ●
     0 ┼────┬────┬────┬────┬────┬────┬────┬────┬────┬
       0    1    2    3    4    5    6    7    8    9
```

## Next Steps

- [Installation](./installation.md) - Detailed setup guide
- [Your First Plot](./first-plot.md) - Step-by-step tutorial
- [Grammar of Graphics](../grammar/overview.md) - Understanding the API
