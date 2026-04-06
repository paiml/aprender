# Line Charts

Line charts display data points connected by straight lines, ideal for
visualizing trends over ordered data (typically time series).

## Basic Line Chart

```rust
use trueno_viz::prelude::*;
use trueno_viz::plots::{LineChart, LineSeries};

let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
let y = vec![10.0, 15.0, 13.0, 17.0, 20.0];

let chart = LineChart::new()
    .x(&x)
    .y(&y)
    .build();

assert_eq!(chart.x_data().len(), 5);
```

**Test Reference**: `src/plots/line.rs::test_line_chart_basic`

## Line Series

For multiple lines on the same chart:

```rust
use trueno_viz::prelude::*;
use trueno_viz::plots::{LineChart, LineSeries};

let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
let revenue = vec![100.0, 120.0, 115.0, 140.0, 150.0];
let costs = vec![80.0, 85.0, 90.0, 95.0, 100.0];
let profit = vec![20.0, 35.0, 25.0, 45.0, 50.0];

let chart = LineChart::new()
    .series(LineSeries::new("Revenue", &time, &revenue)
        .color(Rgba::new(66, 133, 244, 255))
        .line_width(2.0))
    .series(LineSeries::new("Costs", &time, &costs)
        .color(Rgba::new(234, 67, 53, 255))
        .line_width(2.0))
    .series(LineSeries::new("Profit", &time, &profit)
        .color(Rgba::new(52, 168, 83, 255))
        .line_width(2.0))
    .build();
```

## Customizing Lines

### Line Width

```rust
use trueno_viz::plots::LineChart;

let chart = LineChart::new()
    .x(&x)
    .y(&y)
    .line_width(3.0)  // Thicker line
    .build();
```

### Line Style

```rust
use trueno_viz::plots::{LineChart, LineStyle};

// Solid line (default)
let solid = LineChart::new().x(&x).y(&y).style(LineStyle::Solid).build();

// Dashed line
let dashed = LineChart::new().x(&x).y(&y).style(LineStyle::Dashed).build();

// Dotted line
let dotted = LineChart::new().x(&x).y(&y).style(LineStyle::Dotted).build();
```

### With Points

```rust
use trueno_viz::plots::LineChart;

let chart = LineChart::new()
    .x(&x)
    .y(&y)
    .show_points(true)
    .point_size(4.0)
    .build();
```

## Labels and Title

```rust
use trueno_viz::plots::LineChart;

let chart = LineChart::new()
    .x(&x)
    .y(&y)
    .title("Monthly Sales Trend")
    .xlabel("Month")
    .ylabel("Sales ($)")
    .build();
```

## Area Under Line

Fill the area between the line and x-axis:

```rust
use trueno_viz::prelude::*;
use trueno_viz::plots::LineChart;

let chart = LineChart::new()
    .x(&x)
    .y(&y)
    .fill_area(true)
    .fill_color(Rgba::new(66, 133, 244, 100))  // Semi-transparent
    .build();
```

## Time Series Data

Working with time series:

```rust
use trueno_viz::plots::LineChart;

// Using indices as time
let values = vec![10.0, 12.0, 15.0, 14.0, 18.0, 20.0, 19.0];

let chart = LineChart::new()
    .y(&values)  // x-axis auto-generated as indices
    .title("Weekly Values")
    .xlabel("Day")
    .build();
```

## Axis Scaling

### Y-Axis Starting at Zero

```rust
use trueno_viz::plots::LineChart;

let chart = LineChart::new()
    .x(&x)
    .y(&y)
    .y_axis_zero(true)  // Force y-axis to start at 0
    .build();
```

### Custom Axis Limits

```rust
use trueno_viz::plots::LineChart;

let chart = LineChart::new()
    .x(&x)
    .y(&y)
    .x_limits(0.0, 10.0)
    .y_limits(0.0, 100.0)
    .build();
```

## Smooth Lines

Apply smoothing to noisy data:

```rust
use trueno_viz::plots::{LineChart, Smoothing};

let chart = LineChart::new()
    .x(&x)
    .y(&y)
    .smooth(Smoothing::MovingAverage(3))  // 3-point moving average
    .build();
```

## Performance

Line charts benefit from SIMD acceleration for:
- Path computation
- Clipping calculations
- Anti-aliasing

```rust
use trueno_viz::plots::LineChart;

// Large time series (100k points)
let x: Vec<f32> = (0..100_000).map(|i| i as f32 * 0.001).collect();
let y: Vec<f32> = x.iter().map(|t| (t * 10.0).sin()).collect();

let chart = LineChart::new()
    .x(&x)
    .y(&y)
    .build();

// Rendering uses SIMD-accelerated line drawing
chart.render_to_file("large_timeseries.png").unwrap();
```

## Complete Example

```rust
use trueno_viz::prelude::*;
use trueno_viz::plots::{LineChart, LineSeries, LineStyle};

fn main() -> Result<()> {
    // Stock price simulation
    let days: Vec<f32> = (1..=30).map(|d| d as f32).collect();
    let stock_a = vec![
        100.0, 102.0, 101.0, 105.0, 108.0, 107.0, 110.0, 112.0, 115.0, 113.0,
        118.0, 120.0, 119.0, 122.0, 125.0, 123.0, 128.0, 130.0, 132.0, 135.0,
        133.0, 138.0, 140.0, 142.0, 145.0, 143.0, 148.0, 150.0, 152.0, 155.0,
    ];
    let stock_b = vec![
        50.0, 51.0, 52.0, 51.5, 53.0, 54.0, 53.5, 55.0, 56.0, 57.0,
        58.0, 57.5, 59.0, 60.0, 61.0, 62.0, 61.5, 63.0, 64.0, 65.0,
        66.0, 65.5, 67.0, 68.0, 69.0, 70.0, 71.0, 72.0, 73.0, 74.0,
    ];

    let chart = LineChart::new()
        .series(LineSeries::new("Stock A", &days, &stock_a)
            .color(Rgba::new(66, 133, 244, 255))
            .line_width(2.0))
        .series(LineSeries::new("Stock B", &days, &stock_b)
            .color(Rgba::new(52, 168, 83, 255))
            .line_width(2.0)
            .style(LineStyle::Dashed))
        .title("Stock Price Comparison")
        .xlabel("Trading Day")
        .ylabel("Price ($)")
        .y_axis_zero(false)
        .build();

    chart.render_to_file("stock_comparison.png")?;

    Ok(())
}
```

## Next Chapter

Continue to [Histograms](./histogram.md) for distribution visualization.
