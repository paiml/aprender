# Histograms

Histograms visualize the distribution of continuous data by dividing values
into bins and displaying frequency counts.

## Basic Histogram

```rust
use trueno_viz::prelude::*;
use trueno_viz::plots::Histogram;

let data = vec![
    1.0, 1.5, 2.0, 2.2, 2.5, 2.8, 3.0, 3.1, 3.2, 3.5,
    3.8, 4.0, 4.2, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0, 10.0,
];

let hist = Histogram::new(&data).build();

assert!(hist.bin_count() > 0);
```

**Test Reference**: `src/plots/histogram.rs::test_histogram_basic`

## Binning Strategies

### Fixed Number of Bins

```rust
use trueno_viz::plots::Histogram;

let hist = Histogram::new(&data)
    .bins(20)  // 20 equal-width bins
    .build();
```

### Sturges' Formula

Optimal for normal distributions: `k = ceil(log2(n) + 1)`

```rust
use trueno_viz::plots::{Histogram, BinStrategy};

let hist = Histogram::new(&data)
    .bins(BinStrategy::Sturges)
    .build();
```

**Test Reference**: `src/plots/histogram.rs::test_sturges_binning`

### Scott's Rule

Based on standard deviation: `h = 3.49σn^(-1/3)`

```rust
use trueno_viz::plots::{Histogram, BinStrategy};

let hist = Histogram::new(&data)
    .bins(BinStrategy::Scott)
    .build();
```

### Freedman-Diaconis Rule

Robust to outliers: `h = 2 × IQR × n^(-1/3)`

```rust
use trueno_viz::plots::{Histogram, BinStrategy};

let hist = Histogram::new(&data)
    .bins(BinStrategy::FreedmanDiaconis)
    .build();
```

**Test Reference**: `src/plots/histogram.rs::test_freedman_diaconis_binning`

### Fixed Bin Width

```rust
use trueno_viz::plots::Histogram;

let hist = Histogram::new(&data)
    .bin_width(0.5)  // Each bin spans 0.5 units
    .build();
```

## Customizing Appearance

### Color

```rust
use trueno_viz::prelude::*;
use trueno_viz::plots::Histogram;

let hist = Histogram::new(&data)
    .color(Rgba::new(66, 133, 244, 255))  // Bar fill
    .edge_color(Rgba::new(30, 60, 120, 255))  // Bar outline
    .build();
```

### Transparency

```rust
use trueno_viz::prelude::*;
use trueno_viz::plots::Histogram;

let hist = Histogram::new(&data)
    .color(Rgba::new(66, 133, 244, 180))  // Semi-transparent
    .build();
```

## Labels and Title

```rust
use trueno_viz::plots::Histogram;

let hist = Histogram::new(&data)
    .title("Response Time Distribution")
    .xlabel("Response Time (ms)")
    .ylabel("Frequency")
    .build();
```

## Normalization

### Density (Area = 1)

```rust
use trueno_viz::plots::{Histogram, Normalization};

let hist = Histogram::new(&data)
    .normalize(Normalization::Density)  // Area integrates to 1
    .build();
```

### Probability (Sum = 1)

```rust
use trueno_viz::plots::{Histogram, Normalization};

let hist = Histogram::new(&data)
    .normalize(Normalization::Probability)  // Heights sum to 1
    .build();
```

## Cumulative Histogram

```rust
use trueno_viz::plots::Histogram;

let hist = Histogram::new(&data)
    .cumulative(true)
    .build();
```

## Overlay with Density Curve

```rust
use trueno_viz::prelude::*;
use trueno_viz::plots::Histogram;

let hist = Histogram::new(&data)
    .bins(BinStrategy::Scott)
    .normalize(Normalization::Density)
    .show_kde(true)  // Kernel density estimate
    .kde_bandwidth(0.5)
    .build();
```

## Multiple Histograms

Overlapping histograms for comparison:

```rust
use trueno_viz::prelude::*;
use trueno_viz::plots::Histogram;

let group_a = vec![1.0, 2.0, 2.5, 3.0, 3.5, 4.0];
let group_b = vec![2.0, 3.0, 3.5, 4.0, 4.5, 5.0];

let hist = Histogram::new(&[])
    .add_series("Group A", &group_a, Rgba::new(66, 133, 244, 150))
    .add_series("Group B", &group_b, Rgba::new(234, 67, 53, 150))
    .build();
```

## Edge Cases

### Empty Data

```rust
use trueno_viz::plots::Histogram;

let empty: Vec<f32> = vec![];
let hist = Histogram::new(&empty).build();

assert_eq!(hist.bin_count(), 0);
```

**Test Reference**: `src/plots/histogram.rs::test_histogram_empty`

### Single Value

```rust
use trueno_viz::plots::Histogram;

let single = vec![5.0];
let hist = Histogram::new(&single).build();

assert_eq!(hist.bin_count(), 1);
```

### All Same Value

```rust
use trueno_viz::plots::Histogram;

let same = vec![3.0, 3.0, 3.0, 3.0];
let hist = Histogram::new(&same).build();
```

**Test Reference**: `src/plots/histogram.rs::test_histogram_same_values`

## Performance

Histograms use SIMD for:
- Finding min/max values
- Bin assignment
- Counting

```rust
use trueno_viz::plots::Histogram;

// 1 million data points
let large_data: Vec<f32> = (0..1_000_000)
    .map(|i| (i as f32 * 0.001).sin())
    .collect();

let hist = Histogram::new(&large_data)
    .bins(100)
    .build();
```

## Complete Example

```rust
use trueno_viz::prelude::*;
use trueno_viz::plots::{Histogram, BinStrategy, Normalization};

fn main() -> Result<()> {
    // Simulated test scores (normally distributed)
    let scores = vec![
        65.0, 70.0, 72.0, 75.0, 78.0, 80.0, 80.0, 82.0, 82.0, 83.0,
        84.0, 85.0, 85.0, 85.0, 86.0, 87.0, 87.0, 88.0, 88.0, 89.0,
        90.0, 90.0, 91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 98.0, 100.0,
    ];

    let hist = Histogram::new(&scores)
        .bins(BinStrategy::Sturges)
        .color(Rgba::new(52, 168, 83, 200))
        .edge_color(Rgba::new(25, 80, 40, 255))
        .title("Test Score Distribution")
        .xlabel("Score")
        .ylabel("Frequency")
        .show_kde(true)
        .build();

    hist.render_to_file("test_scores.png")?;

    Ok(())
}
```

## Next Chapter

Continue to [Heatmaps](./heatmap.md) for 2D density visualization.
