# Heatmaps

Heatmaps visualize 2D data using color to represent values, ideal for
correlation matrices, confusion matrices, and spatial data.

## Basic Heatmap

```rust
use trueno_viz::prelude::*;
use trueno_viz::plots::Heatmap;

// 3x3 matrix (row-major order)
let data = vec![
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0,
    7.0, 8.0, 9.0,
];

let heatmap = Heatmap::new(&data, 3, 3).build();

assert_eq!(heatmap.rows(), 3);
assert_eq!(heatmap.cols(), 3);
```

**Test Reference**: `src/plots/heatmap.rs::test_heatmap_basic`

## Color Palettes

### Built-in Palettes

```rust
use trueno_viz::plots::{Heatmap, HeatmapPalette};

// Sequential palettes (low to high)
let viridis = Heatmap::new(&data, 3, 3)
    .palette(HeatmapPalette::Viridis)
    .build();

let plasma = Heatmap::new(&data, 3, 3)
    .palette(HeatmapPalette::Plasma)
    .build();

let inferno = Heatmap::new(&data, 3, 3)
    .palette(HeatmapPalette::Inferno)
    .build();

// Diverging palettes (negative to positive)
let rdbu = Heatmap::new(&data, 3, 3)
    .palette(HeatmapPalette::RdBu)
    .build();
```

**Test Reference**: `src/plots/heatmap.rs::test_heatmap_viridis`

### Custom Color Scale

```rust
use trueno_viz::prelude::*;
use trueno_viz::plots::Heatmap;

let heatmap = Heatmap::new(&data, 3, 3)
    .color_range(Rgba::BLUE, Rgba::RED)  // Blue (low) to Red (high)
    .build();

// Three-color gradient
let heatmap = Heatmap::new(&data, 3, 3)
    .color_range_diverging(Rgba::BLUE, Rgba::WHITE, Rgba::RED)
    .center(0.0)  // White at 0
    .build();
```

## Value Range

```rust
use trueno_viz::plots::Heatmap;

// Fixed color scale range
let heatmap = Heatmap::new(&data, 3, 3)
    .vmin(0.0)
    .vmax(10.0)
    .build();

// Symmetric around zero (for diverging)
let heatmap = Heatmap::new(&data, 3, 3)
    .symmetric(true)
    .build();
```

## Labels

### Axis Labels

```rust
use trueno_viz::plots::Heatmap;

let heatmap = Heatmap::new(&data, 3, 3)
    .x_labels(&["A", "B", "C"])
    .y_labels(&["X", "Y", "Z"])
    .build();
```

### Cell Annotations

```rust
use trueno_viz::plots::Heatmap;

let heatmap = Heatmap::new(&data, 3, 3)
    .annotate(true)  // Show values in cells
    .annotation_format("{:.2}")  // Two decimal places
    .build();
```

### Title and Labels

```rust
use trueno_viz::plots::Heatmap;

let heatmap = Heatmap::new(&data, 3, 3)
    .title("Correlation Matrix")
    .xlabel("Features")
    .ylabel("Features")
    .build();
```

## Colorbar

```rust
use trueno_viz::plots::Heatmap;

let heatmap = Heatmap::new(&data, 3, 3)
    .show_colorbar(true)
    .colorbar_label("Correlation")
    .build();
```

## Correlation Matrix

Special helper for correlation matrices:

```rust
use trueno_viz::plots::Heatmap;

let correlation_data = vec![
    1.0, 0.8, 0.3, -0.2,
    0.8, 1.0, 0.5, 0.1,
    0.3, 0.5, 1.0, 0.6,
    -0.2, 0.1, 0.6, 1.0,
];

let heatmap = Heatmap::correlation_matrix(&correlation_data, 4)
    .feature_names(&["A", "B", "C", "D"])
    .build();
```

## Confusion Matrix

Special helper for ML confusion matrices:

```rust
use trueno_viz::plots::ConfusionMatrix;

let cm_data = vec![
    50.0, 10.0,  // True Negative, False Positive
    5.0,  35.0,  // False Negative, True Positive
];

let cm = ConfusionMatrix::new(&cm_data, 2)
    .class_names(&["Negative", "Positive"])
    .normalize(true)  // Show percentages
    .build();
```

**Test Reference**: `src/plots/heatmap.rs::test_confusion_matrix`

## Cell Styling

```rust
use trueno_viz::plots::Heatmap;

let heatmap = Heatmap::new(&data, 3, 3)
    .cell_border(true)
    .cell_border_color(Rgba::WHITE)
    .cell_border_width(1.0)
    .build();
```

## Masking

Hide specific cells:

```rust
use trueno_viz::plots::Heatmap;

// Mask upper triangle (for correlation matrix)
let heatmap = Heatmap::new(&data, 3, 3)
    .mask_upper_triangle(true)
    .build();

// Mask diagonal
let heatmap = Heatmap::new(&data, 3, 3)
    .mask_diagonal(true)
    .build();
```

## Performance

Heatmaps use SIMD/GPU for:
- Color mapping
- Value normalization
- Large matrix rendering

```rust
use trueno_viz::plots::Heatmap;

// 1000x1000 heatmap (1M cells)
let large_data: Vec<f32> = (0..1_000_000)
    .map(|i| (i as f32 * 0.001).sin())
    .collect();

let heatmap = Heatmap::new(&large_data, 1000, 1000)
    .palette(HeatmapPalette::Viridis)
    .build();
```

## Complete Example

```rust
use trueno_viz::prelude::*;
use trueno_viz::plots::{Heatmap, HeatmapPalette};

fn main() -> Result<()> {
    // Feature correlation matrix
    let correlations = vec![
        1.00,  0.85,  0.32, -0.15,  0.42,
        0.85,  1.00,  0.45,  0.05,  0.38,
        0.32,  0.45,  1.00,  0.72,  0.55,
       -0.15,  0.05,  0.72,  1.00,  0.68,
        0.42,  0.38,  0.55,  0.68,  1.00,
    ];

    let features = &["Age", "Income", "Education", "Experience", "Score"];

    let heatmap = Heatmap::new(&correlations, 5, 5)
        .palette(HeatmapPalette::RdBu)
        .x_labels(features)
        .y_labels(features)
        .annotate(true)
        .annotation_format("{:.2}")
        .vmin(-1.0)
        .vmax(1.0)
        .mask_upper_triangle(true)
        .title("Feature Correlations")
        .show_colorbar(true)
        .colorbar_label("Correlation")
        .build();

    heatmap.render_to_file("correlation_matrix.png")?;

    Ok(())
}
```

## Next Chapter

Continue to [Box Plots](./boxplot.md) for statistical distribution summaries.
