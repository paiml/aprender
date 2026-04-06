# Your First Plot

This chapter walks through creating a complete visualization from scratch,
explaining each component along the way.

## The Problem

We have temperature data from a weather station and want to visualize the
daily temperature pattern.

```rust
// Sample data: hours of day and temperatures
let hours: Vec<f32> = (0..24).map(|h| h as f32).collect();
let temps = vec![
    15.0, 14.5, 14.0, 13.5, 13.2, 13.0,  // 0-5am (cooling)
    13.5, 15.0, 17.0, 19.0, 21.0, 23.0,  // 6-11am (warming)
    24.5, 25.0, 25.5, 25.0, 24.0, 22.5,  // 12-5pm (peak)
    21.0, 19.5, 18.0, 17.0, 16.0, 15.5,  // 6-11pm (cooling)
];
```

## Step 1: Import the Prelude

```rust
use trueno_viz::prelude::*;
use trueno_viz::plots::LineChart;
```

The prelude provides:
- `Rgba` - Color type
- `Point`, `Line`, `Rect` - Geometry types
- `Framebuffer` - Pixel buffer
- `Result`, `Error` - Error handling

## Step 2: Create the Data

```rust
let hours: Vec<f32> = (0..24).map(|h| h as f32).collect();
let temps = vec![
    15.0, 14.5, 14.0, 13.5, 13.2, 13.0,
    13.5, 15.0, 17.0, 19.0, 21.0, 23.0,
    24.5, 25.0, 25.5, 25.0, 24.0, 22.5,
    21.0, 19.5, 18.0, 17.0, 16.0, 15.5,
];

// Verify data integrity
assert_eq!(hours.len(), temps.len());
assert_eq!(hours.len(), 24);
```

## Step 3: Build the Line Chart

```rust
let chart = LineChart::new()
    .x(&hours)
    .y(&temps)
    .color(Rgba::new(219, 68, 55, 255))  // Google Red
    .line_width(2.0)
    .title("Daily Temperature Pattern")
    .xlabel("Hour of Day")
    .ylabel("Temperature (°C)")
    .build();
```

### Understanding the Builder Pattern

Each method returns `Self`, enabling fluent chaining:

```rust
impl LineChart {
    pub fn new() -> Self { ... }

    pub fn x(mut self, data: &[f32]) -> Self {
        self.x_data = data.to_vec();
        self
    }

    pub fn y(mut self, data: &[f32]) -> Self {
        self.y_data = data.to_vec();
        self
    }

    pub fn build(self) -> LineChartResult {
        // Validate and construct
        ...
    }
}
```

**Test Reference**: `src/plots/line.rs::test_line_chart_builder`

## Step 4: Render to File

```rust
fn main() -> Result<()> {
    let chart = LineChart::new()
        .x(&hours)
        .y(&temps)
        .title("Daily Temperature Pattern")
        .build();

    // Render to PNG (800x600 pixels)
    chart.render_to_file("temperature.png")?;

    Ok(())
}
```

## Step 5: Verify the Output

The output can be verified programmatically:

```rust
#[test]
fn test_temperature_chart() {
    let hours: Vec<f32> = (0..24).map(|h| h as f32).collect();
    let temps = vec![
        15.0, 14.5, 14.0, 13.5, 13.2, 13.0,
        13.5, 15.0, 17.0, 19.0, 21.0, 23.0,
        24.5, 25.0, 25.5, 25.0, 24.0, 22.5,
        21.0, 19.5, 18.0, 17.0, 16.0, 15.5,
    ];

    let chart = LineChart::new()
        .x(&hours)
        .y(&temps)
        .build();

    // Verify chart properties
    assert_eq!(chart.x_data().len(), 24);
    assert_eq!(chart.y_data().len(), 24);

    // Verify data range detection
    let (x_min, x_max) = chart.x_range();
    assert!((x_min - 0.0).abs() < f32::EPSILON);
    assert!((x_max - 23.0).abs() < f32::EPSILON);

    let (y_min, y_max) = chart.y_range();
    assert!((y_min - 13.0).abs() < f32::EPSILON);
    assert!((y_max - 25.5).abs() < f32::EPSILON);
}
```

## Complete Example

```rust
use trueno_viz::prelude::*;
use trueno_viz::plots::LineChart;

fn main() -> Result<()> {
    // Data
    let hours: Vec<f32> = (0..24).map(|h| h as f32).collect();
    let temps = vec![
        15.0, 14.5, 14.0, 13.5, 13.2, 13.0,
        13.5, 15.0, 17.0, 19.0, 21.0, 23.0,
        24.5, 25.0, 25.5, 25.0, 24.0, 22.5,
        21.0, 19.5, 18.0, 17.0, 16.0, 15.5,
    ];

    // Build chart
    let chart = LineChart::new()
        .x(&hours)
        .y(&temps)
        .color(Rgba::new(219, 68, 55, 255))
        .line_width(2.0)
        .title("Daily Temperature Pattern")
        .xlabel("Hour of Day")
        .ylabel("Temperature (°C)")
        .build();

    // Output to multiple formats
    chart.render_to_file("temperature.png")?;

    #[cfg(feature = "svg")]
    chart.render_svg_to_file("temperature.svg")?;

    #[cfg(feature = "terminal")]
    println!("{}", chart.render_terminal(80, 24));

    Ok(())
}
```

## What's Next?

Now that you've created your first plot, learn about:

- [Grammar of Graphics](../grammar/overview.md) - The underlying theory
- [Plot Types](../plots/scatter.md) - All available visualizations
- [Output Formats](../output/png.md) - PNG, SVG, and terminal output
