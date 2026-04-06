# Scatter Plots

Scatter plots display the relationship between two continuous variables
using points positioned in 2D space.

## Basic Scatter Plot

```rust
use trueno_viz::prelude::*;
use trueno_viz::plots::ScatterPlot;

let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
let y = vec![2.1, 3.9, 6.2, 7.8, 10.1];

let plot = ScatterPlot::new()
    .x(&x)
    .y(&y)
    .build();

// Verify data was set correctly
assert_eq!(plot.len(), 5);
```

**Test Reference**: `src/plots/scatter.rs::test_scatter_basic`

## Customizing Points

### Color

```rust
use trueno_viz::prelude::*;
use trueno_viz::plots::ScatterPlot;

let plot = ScatterPlot::new()
    .x(&x)
    .y(&y)
    .color(Rgba::new(66, 133, 244, 255))  // Google Blue
    .build();
```

### Size

```rust
use trueno_viz::plots::ScatterPlot;

let plot = ScatterPlot::new()
    .x(&x)
    .y(&y)
    .size(8.0)  // Point radius in pixels
    .build();
```

### Transparency

```rust
use trueno_viz::prelude::*;
use trueno_viz::plots::ScatterPlot;

let plot = ScatterPlot::new()
    .x(&x)
    .y(&y)
    .color(Rgba::new(66, 133, 244, 180))  // Alpha = 180 (semi-transparent)
    .build();
```

## Labels and Title

```rust
use trueno_viz::plots::ScatterPlot;

let plot = ScatterPlot::new()
    .x(&x)
    .y(&y)
    .title("Relationship Between X and Y")
    .xlabel("Independent Variable")
    .ylabel("Dependent Variable")
    .build();
```

## Multiple Series

```rust
use trueno_viz::prelude::*;
use trueno_viz::plots::ScatterPlot;

let x1 = vec![1.0, 2.0, 3.0];
let y1 = vec![1.0, 2.0, 3.0];

let x2 = vec![1.0, 2.0, 3.0];
let y2 = vec![3.0, 2.0, 1.0];

let plot = ScatterPlot::new()
    .series("Series A", &x1, &y1, Rgba::BLUE)
    .series("Series B", &x2, &y2, Rgba::RED)
    .build();
```

## Rendering

### To PNG

```rust
use trueno_viz::plots::ScatterPlot;

let plot = ScatterPlot::new()
    .x(&x)
    .y(&y)
    .build();

// Default size (800x600)
plot.render_to_file("scatter.png").unwrap();

// Custom size
plot.render_to_file_with_size("scatter_large.png", 1200, 900).unwrap();
```

### To SVG

```rust
#[cfg(feature = "svg")]
{
    plot.render_svg_to_file("scatter.svg").unwrap();
}
```

### To Terminal

```rust
#[cfg(feature = "terminal")]
{
    let ascii = plot.render_terminal(80, 24);
    println!("{}", ascii);
}
```

## Statistical Overlays

### Regression Line

```rust
use trueno_viz::plots::ScatterPlot;

let plot = ScatterPlot::new()
    .x(&x)
    .y(&y)
    .regression_line(true)  // Add linear regression line
    .build();
```

### Correlation Display

```rust
use trueno_viz::plots::ScatterPlot;

let plot = ScatterPlot::new()
    .x(&x)
    .y(&y)
    .show_correlation(true)  // Display r² value
    .build();
```

## Performance Considerations

For large datasets, scatter plots use SIMD acceleration:

```rust
use trueno_viz::plots::ScatterPlot;

// 1 million points - SIMD accelerated
let x: Vec<f32> = (0..1_000_000).map(|i| i as f32).collect();
let y: Vec<f32> = x.iter().map(|x| x.sin()).collect();

let plot = ScatterPlot::new()
    .x(&x)
    .y(&y)
    .alpha(0.1)  // Low alpha for density visualization
    .build();
```

**Test Reference**: `src/plots/scatter.rs::test_scatter_large_dataset`

## Complete Example

```rust
use trueno_viz::prelude::*;
use trueno_viz::plots::ScatterPlot;

fn main() -> Result<()> {
    // Iris-like data (sepal measurements)
    let sepal_length = vec![5.1, 4.9, 4.7, 7.0, 6.4, 6.9, 6.3, 5.8, 7.1];
    let sepal_width = vec![3.5, 3.0, 3.2, 3.2, 3.2, 3.1, 3.3, 2.7, 3.0];

    let plot = ScatterPlot::new()
        .x(&sepal_length)
        .y(&sepal_width)
        .color(Rgba::new(102, 178, 102, 220))
        .size(6.0)
        .title("Iris Sepal Dimensions")
        .xlabel("Sepal Length (cm)")
        .ylabel("Sepal Width (cm)")
        .regression_line(true)
        .build();

    plot.render_to_file("iris_scatter.png")?;

    Ok(())
}
```

## Next Chapter

Continue to [Line Charts](./line.md) for visualizing trends over time.
