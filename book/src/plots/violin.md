# Violin Plots

Violin plots combine box plot statistics with kernel density estimation
to show the full distribution shape alongside quartile information.

## Basic Violin Plot

```rust
use trueno_viz::prelude::*;
use trueno_viz::plots::ViolinPlot;

let data = vec![
    1.0, 1.5, 2.0, 2.0, 2.5, 3.0, 3.0, 3.0, 3.5, 4.0,
    4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0, 9.0, 10.0,
];

let violin = ViolinPlot::new(&data).build();
```

**Test Reference**: `src/plots/boxplot.rs::test_violin_basic`

## Kernel Density Estimation

### Bandwidth

```rust
use trueno_viz::plots::ViolinPlot;

// Auto bandwidth (Silverman's rule)
let auto = ViolinPlot::new(&data).build();

// Custom bandwidth
let custom = ViolinPlot::new(&data)
    .bandwidth(0.5)
    .build();
```

**Test Reference**: `src/plots/boxplot.rs::test_violin_bandwidth`

### Kernel Type

```rust
use trueno_viz::plots::{ViolinPlot, Kernel};

let gaussian = ViolinPlot::new(&data)
    .kernel(Kernel::Gaussian)
    .build();

let epanechnikov = ViolinPlot::new(&data)
    .kernel(Kernel::Epanechnikov)
    .build();

let triangular = ViolinPlot::new(&data)
    .kernel(Kernel::Triangular)
    .build();
```

**Test Reference**: `src/plots/boxplot.rs::test_kde_kernels`

## Multiple Groups

```rust
use trueno_viz::prelude::*;
use trueno_viz::plots::ViolinPlot;

let group_a = vec![1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0];
let group_b = vec![2.0, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0];

let violin = ViolinPlot::new(&[])
    .group("Treatment A", &group_a)
    .group("Treatment B", &group_b)
    .build();
```

## Customization

### Colors

```rust
use trueno_viz::prelude::*;
use trueno_viz::plots::ViolinPlot;

let violin = ViolinPlot::new(&data)
    .fill_color(Rgba::new(102, 178, 102, 180))
    .edge_color(Rgba::new(50, 90, 50, 255))
    .median_color(Rgba::WHITE)
    .build();
```

### Width Scaling

```rust
use trueno_viz::plots::{ViolinPlot, ViolinScale};

// Scale by area (default)
let area = ViolinPlot::new(&data)
    .scale(ViolinScale::Area)
    .build();

// Scale by count
let count = ViolinPlot::new(&data)
    .scale(ViolinScale::Count)
    .build();

// Same width for all
let width = ViolinPlot::new(&data)
    .scale(ViolinScale::Width)
    .build();
```

**Test Reference**: `src/plots/boxplot.rs::test_violin_scale`

## Inner Display

### Box Plot Inside

```rust
use trueno_viz::plots::{ViolinPlot, ViolinInner};

let violin = ViolinPlot::new(&data)
    .inner(ViolinInner::Box)
    .build();
```

### Quartile Lines

```rust
use trueno_viz::plots::{ViolinPlot, ViolinInner};

let violin = ViolinPlot::new(&data)
    .inner(ViolinInner::Quartiles)
    .build();
```

### Points

```rust
use trueno_viz::plots::{ViolinPlot, ViolinInner};

let violin = ViolinPlot::new(&data)
    .inner(ViolinInner::Points)
    .build();
```

### Stick (Line at Each Point)

```rust
use trueno_viz::plots::{ViolinPlot, ViolinInner};

let violin = ViolinPlot::new(&data)
    .inner(ViolinInner::Stick)
    .build();
```

## Split Violins

Compare two groups side-by-side:

```rust
use trueno_viz::plots::ViolinPlot;

let violin = ViolinPlot::new(&[])
    .group("Before", &before_data)
    .group("After", &after_data)
    .split(true)
    .build();
```

## Truncation

```rust
use trueno_viz::plots::ViolinPlot;

// Cut density at data range
let cut = ViolinPlot::new(&data)
    .cut(0.0)  // No extension beyond data
    .build();

// Extend density beyond data range
let extend = ViolinPlot::new(&data)
    .cut(2.0)  // Extend 2 bandwidths beyond data
    .build();
```

## Orientation

```rust
use trueno_viz::plots::{ViolinPlot, Orientation};

let horizontal = ViolinPlot::new(&data)
    .orientation(Orientation::Horizontal)
    .build();
```

## Labels

```rust
use trueno_viz::plots::ViolinPlot;

let violin = ViolinPlot::new(&data)
    .title("Distribution Shape")
    .xlabel("Category")
    .ylabel("Value")
    .build();
```

## Edge Cases

### Single Value

```rust
use trueno_viz::plots::ViolinPlot;

let single = vec![5.0];
let violin = ViolinPlot::new(&single).build();

// Renders as a single point/line
```

**Test Reference**: `src/plots/boxplot.rs::test_violin_single_value`

### Bimodal Distribution

```rust
use trueno_viz::plots::ViolinPlot;

// Two clusters
let bimodal = vec![
    1.0, 1.5, 2.0, 2.5, 3.0,  // First mode
    8.0, 8.5, 9.0, 9.5, 10.0, // Second mode
];

let violin = ViolinPlot::new(&bimodal)
    .bandwidth(0.5)  // Smaller bandwidth to show bimodality
    .build();
```

## Performance

KDE computation uses SIMD acceleration:

```rust
use trueno_viz::plots::ViolinPlot;

// Large dataset (10k points)
let large_data: Vec<f32> = (0..10_000)
    .map(|i| (i as f32 * 0.001).sin() * 10.0)
    .collect();

let violin = ViolinPlot::new(&large_data)
    .n_points(200)  // Sample density at 200 points
    .build();
```

## Complete Example

```rust
use trueno_viz::prelude::*;
use trueno_viz::plots::{ViolinPlot, ViolinInner, ViolinScale, Kernel};

fn main() -> Result<()> {
    // Response times by service
    let service_a = vec![
        12.0, 15.0, 18.0, 20.0, 22.0, 25.0, 28.0, 30.0, 32.0, 35.0,
        38.0, 40.0, 45.0, 50.0, 55.0, 60.0, 80.0, 100.0, 120.0,
    ];
    let service_b = vec![
        8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0,
        28.0, 30.0, 32.0, 34.0, 36.0, 38.0, 40.0,
    ];
    let service_c = vec![
        5.0, 5.0, 6.0, 6.0, 7.0, 7.0, 8.0, 8.0, 9.0, 9.0,
        10.0, 10.0, 11.0, 12.0, 15.0, 20.0, 50.0, 100.0,
    ];

    let violin = ViolinPlot::new(&[])
        .group("Service A", &service_a)
        .group("Service B", &service_b)
        .group("Service C", &service_c)
        .kernel(Kernel::Gaussian)
        .scale(ViolinScale::Area)
        .inner(ViolinInner::Box)
        .fill_color(Rgba::new(66, 133, 244, 150))
        .edge_color(Rgba::new(30, 60, 120, 255))
        .title("Response Time Distribution by Service")
        .xlabel("Service")
        .ylabel("Response Time (ms)")
        .build();

    violin.render_to_file("service_response_times.png")?;

    Ok(())
}
```

## Box Plot vs Violin Plot

| Feature | Box Plot | Violin Plot |
|---------|----------|-------------|
| Shows quartiles | Yes | Yes (with inner=Box) |
| Shows distribution shape | No | Yes |
| Shows bimodality | No | Yes |
| Computation | O(n log n) | O(n × k) |
| Best for | Quick comparison | Detailed analysis |

## Next Chapter

Continue to [ROC Curves](../ml-viz/roc.md) for machine learning evaluation visualizations.
