# Box Plots

Box plots (box-and-whisker plots) display the five-number summary of a
distribution: minimum, first quartile (Q1), median, third quartile (Q3),
and maximum, plus outliers.

## Basic Box Plot

```rust
use trueno_viz::prelude::*;
use trueno_viz::plots::BoxPlot;

let data = vec![
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
    12.0, 15.0, 20.0, 25.0, 100.0,  // Note: 100 is an outlier
];

let boxplot = BoxPlot::new(&data).build();

// Verify statistics
let stats = boxplot.statistics();
assert!((stats.median - 7.0).abs() < 0.1);
```

**Test Reference**: `src/plots/boxplot.rs::test_boxplot_basic`

## Box Plot Statistics

```rust
use trueno_viz::plots::BoxPlot;

let boxplot = BoxPlot::new(&data).build();

let stats = boxplot.statistics();
println!("Min: {}", stats.min);
println!("Q1: {}", stats.q1);
println!("Median: {}", stats.median);
println!("Q3: {}", stats.q3);
println!("Max: {}", stats.max);
println!("Outliers: {:?}", stats.outliers);
```

**Test Reference**: `src/plots/boxplot.rs::test_boxplot_stats`

## Multiple Groups

```rust
use trueno_viz::prelude::*;
use trueno_viz::plots::BoxPlot;

let group_a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
let group_b = vec![3.0, 4.0, 5.0, 6.0, 7.0];
let group_c = vec![5.0, 6.0, 7.0, 8.0, 9.0];

let boxplot = BoxPlot::new(&[])
    .group("Group A", &group_a)
    .group("Group B", &group_b)
    .group("Group C", &group_c)
    .build();
```

**Test Reference**: `src/plots/boxplot.rs::test_boxplot_groups`

## Customization

### Colors

```rust
use trueno_viz::prelude::*;
use trueno_viz::plots::BoxPlot;

let boxplot = BoxPlot::new(&data)
    .fill_color(Rgba::new(66, 133, 244, 200))
    .edge_color(Rgba::new(30, 60, 120, 255))
    .median_color(Rgba::new(255, 0, 0, 255))
    .outlier_color(Rgba::RED)
    .build();
```

### Whisker Style

```rust
use trueno_viz::plots::BoxPlot;

let boxplot = BoxPlot::new(&data)
    .whisker_width(0.5)  // Width relative to box
    .whisker_iqr(1.5)    // IQR multiplier (default: 1.5)
    .build();
```

### Box Width

```rust
use trueno_viz::plots::BoxPlot;

let boxplot = BoxPlot::new(&data)
    .box_width(0.7)  // Width relative to spacing
    .build();
```

## Outlier Detection

The default IQR method: outliers are points beyond Q1 - 1.5×IQR or Q3 + 1.5×IQR.

```rust
use trueno_viz::plots::BoxPlot;

// Stricter outlier detection
let strict = BoxPlot::new(&data)
    .whisker_iqr(1.0)  // Narrower whiskers
    .build();

// More lenient (fewer outliers)
let lenient = BoxPlot::new(&data)
    .whisker_iqr(3.0)  // Wider whiskers
    .build();

// No outliers shown (whiskers to min/max)
let minmax = BoxPlot::new(&data)
    .whisker_iqr(f32::INFINITY)
    .build();
```

## Orientation

```rust
use trueno_viz::plots::{BoxPlot, Orientation};

// Vertical (default)
let vertical = BoxPlot::new(&data)
    .orientation(Orientation::Vertical)
    .build();

// Horizontal
let horizontal = BoxPlot::new(&data)
    .orientation(Orientation::Horizontal)
    .build();
```

## Notched Box Plot

Show confidence interval around median:

```rust
use trueno_viz::plots::BoxPlot;

let boxplot = BoxPlot::new(&data)
    .notch(true)
    .notch_width(0.25)
    .build();
```

## Labels

```rust
use trueno_viz::plots::BoxPlot;

let boxplot = BoxPlot::new(&data)
    .title("Distribution Comparison")
    .xlabel("Category")
    .ylabel("Value")
    .build();
```

## Adding Data Points

Show individual data points alongside box:

```rust
use trueno_viz::plots::BoxPlot;

let boxplot = BoxPlot::new(&data)
    .show_points(true)
    .point_size(3.0)
    .jitter(0.1)  // Random horizontal displacement
    .build();
```

## Edge Cases

### Empty Data

```rust
use trueno_viz::plots::BoxPlot;

let empty: Vec<f32> = vec![];
let boxplot = BoxPlot::new(&empty).build();

assert!(boxplot.statistics().is_empty());
```

### Single Value

```rust
use trueno_viz::plots::BoxPlot;

let single = vec![5.0];
let boxplot = BoxPlot::new(&single).build();

let stats = boxplot.statistics();
assert!((stats.median - 5.0).abs() < f32::EPSILON);
assert!((stats.q1 - 5.0).abs() < f32::EPSILON);
assert!((stats.q3 - 5.0).abs() < f32::EPSILON);
```

**Test Reference**: `src/plots/boxplot.rs::test_boxplot_single_value`

### Two Values

```rust
use trueno_viz::plots::BoxPlot;

let two = vec![1.0, 5.0];
let boxplot = BoxPlot::new(&two).build();

let stats = boxplot.statistics();
assert!((stats.median - 3.0).abs() < f32::EPSILON);
```

## Complete Example

```rust
use trueno_viz::prelude::*;
use trueno_viz::plots::BoxPlot;

fn main() -> Result<()> {
    // Test scores by class
    let class_a = vec![
        72.0, 75.0, 78.0, 80.0, 82.0, 83.0, 85.0, 85.0, 87.0, 88.0,
        90.0, 92.0, 95.0, 45.0,  // Low outlier
    ];
    let class_b = vec![
        65.0, 68.0, 70.0, 72.0, 75.0, 78.0, 80.0, 82.0, 85.0, 88.0,
        90.0, 93.0, 95.0, 98.0,
    ];
    let class_c = vec![
        55.0, 60.0, 62.0, 65.0, 68.0, 70.0, 72.0, 75.0, 78.0, 80.0,
        82.0, 85.0, 88.0, 100.0,  // High outlier
    ];

    let boxplot = BoxPlot::new(&[])
        .group("Class A", &class_a)
        .group("Class B", &class_b)
        .group("Class C", &class_c)
        .fill_color(Rgba::new(66, 133, 244, 180))
        .outlier_color(Rgba::RED)
        .title("Test Score Distribution by Class")
        .xlabel("Class")
        .ylabel("Score")
        .notch(true)
        .show_points(true)
        .jitter(0.05)
        .build();

    boxplot.render_to_file("class_scores.png")?;

    Ok(())
}
```

## Next Chapter

Continue to [Violin Plots](./violin.md) for density-based distribution visualization.
