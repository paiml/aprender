# Statistical Transformations

Statistical transformations (stats) process data before it reaches the
geometric objects. Stats compute summaries, fit models, or transform distributions.

## Available Stats

| Stat | Description | Computed Variables |
|------|-------------|-------------------|
| `identity` | No transformation | - |
| `count` | Count observations | count |
| `bin` | Bin continuous data | count, density |
| `boxplot` | Five-number summary | lower, upper, median, ymin, ymax |
| `density` | Kernel density | density, scaled |
| `smooth` | Regression line | y, ymin, ymax |
| `summary` | Arbitrary summary | user-defined |

## Identity Stat

The default stat - passes data through unchanged:

```rust
use trueno_viz::grammar::{Stat, Geom, GGPlot, DataFrame, Aes};

let df = DataFrame::new()
    .column("x", &[1.0, 2.0, 3.0])
    .column("y", &[4.0, 5.0, 6.0]);

// Explicit identity stat (default)
let plot = GGPlot::new(df)
    .aes(Aes::new().x("x").y("y"))
    .geom(Geom::point().stat(Stat::identity()));
```

**Test Reference**: `src/grammar/stat.rs::test_stat_identity`

## Count Stat

Counts observations per group:

```rust
use trueno_viz::grammar::{Stat, Geom, GGPlot, DataFrame, Aes};

let df = DataFrame::new()
    .column("category", &["A", "A", "B", "B", "B", "C"]);

// Bar chart with automatic counting
let plot = GGPlot::new(df)
    .aes(Aes::new().x("category"))
    .geom(Geom::bar().stat(Stat::count()));
```

Computed variables:
- `count`: Number of observations
- `prop`: Proportion of total

## Bin Stat

Divides continuous data into bins:

```rust
use trueno_viz::grammar::{Stat, BinStrategy};

// Fixed number of bins
let stat = Stat::bin().bins(30);

// Specified bin width
let stat = Stat::bin().binwidth(0.5);

// Sturges' formula (automatic)
let stat = Stat::bin().bins(BinStrategy::Sturges);

// Scott's rule (optimal for normal data)
let stat = Stat::bin().bins(BinStrategy::Scott);

// Freedman-Diaconis rule (robust)
let stat = Stat::bin().bins(BinStrategy::FreedmanDiaconis);
```

**Test Reference**: `src/grammar/stat.rs::test_stat_bin_strategies`

### Binning Algorithms

```rust
use trueno_viz::grammar::stat;

// Sturges: k = ceil(log2(n) + 1)
let sturges = stat::sturges_bins(100);  // → 8 bins for n=100

// Scott: h = 3.49 * std * n^(-1/3)
let scott = stat::scott_binwidth(&data);

// Freedman-Diaconis: h = 2 * IQR * n^(-1/3)
let fd = stat::freedman_diaconis_binwidth(&data);
```

## Boxplot Stat

Computes five-number summary:

```rust
use trueno_viz::grammar::{Stat, Geom, GGPlot, DataFrame, Aes};

let df = DataFrame::new()
    .column("group", &["A", "A", "A", "A", "A"])
    .column("value", &[1.0, 2.0, 3.0, 4.0, 100.0]);  // Note outlier

let plot = GGPlot::new(df)
    .aes(Aes::new().x("group").y("value"))
    .geom(Geom::boxplot().stat(Stat::boxplot()
        .coef(1.5)));  // IQR multiplier for whiskers
```

Computed variables:
- `lower`: 25th percentile (Q1)
- `middle`: Median (Q2)
- `upper`: 75th percentile (Q3)
- `ymin`: Lower whisker
- `ymax`: Upper whisker
- `outliers`: Points beyond whiskers

**Test Reference**: `src/plots/boxplot.rs::test_boxplot_stats`

## Density Stat

Kernel density estimation:

```rust
use trueno_viz::grammar::{Stat, Geom, GGPlot, DataFrame, Aes, Kernel};

let df = DataFrame::new()
    .column("x", &[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]);

let plot = GGPlot::new(df)
    .aes(Aes::new().x("x"))
    .geom(Geom::density().stat(Stat::density()
        .kernel(Kernel::Gaussian)
        .bandwidth(0.5)));
```

### Kernel Options

```rust
use trueno_viz::grammar::{Stat, Kernel};

let gaussian = Stat::density().kernel(Kernel::Gaussian);
let epanechnikov = Stat::density().kernel(Kernel::Epanechnikov);
let triangular = Stat::density().kernel(Kernel::Triangular);
let uniform = Stat::density().kernel(Kernel::Uniform);
```

**Test Reference**: `src/plots/boxplot.rs::test_kde_kernels`

### Bandwidth Selection

```rust
use trueno_viz::grammar::{Stat, BandwidthMethod};

// Silverman's rule of thumb
let silverman = Stat::density().bw(BandwidthMethod::Silverman);

// Scott's rule
let scott = Stat::density().bw(BandwidthMethod::Scott);

// Fixed bandwidth
let fixed = Stat::density().bandwidth(0.3);
```

## Smooth Stat

Fits regression models:

```rust
use trueno_viz::grammar::{Stat, Geom, GGPlot, DataFrame, Aes};

let plot = GGPlot::new(df)
    .aes(Aes::new().x("x").y("y"))
    .geom(Geom::point())
    .geom(Geom::smooth().stat(Stat::smooth()
        .method("lm")         // Linear model
        .se(true)));          // Show confidence interval
```

### Smoothing Methods

```rust
use trueno_viz::grammar::Stat;

// Linear regression
let lm = Stat::smooth().method("lm");

// LOESS (locally weighted)
let loess = Stat::smooth().method("loess").span(0.75);

// Generalized additive model
let gam = Stat::smooth().method("gam");
```

## Summary Stat

Arbitrary summary functions:

```rust
use trueno_viz::grammar::{Stat, Geom, GGPlot, DataFrame, Aes};

let plot = GGPlot::new(df)
    .aes(Aes::new().x("group").y("value"))
    .geom(Geom::point().stat(Stat::summary()
        .fun_y(|v| v.iter().sum::<f32>() / v.len() as f32)));  // Mean
```

### Built-in Summary Functions

```rust
use trueno_viz::grammar::{Stat, SummaryFn};

let mean = Stat::summary().fun_y(SummaryFn::Mean);
let median = Stat::summary().fun_y(SummaryFn::Median);
let min = Stat::summary().fun_y(SummaryFn::Min);
let max = Stat::summary().fun_y(SummaryFn::Max);
let std = Stat::summary().fun_y(SummaryFn::Std);
```

## Stat and Geom Pairing

Some geoms have default stats:

| Geom | Default Stat |
|------|--------------|
| `point` | `identity` |
| `line` | `identity` |
| `bar` | `count` |
| `histogram` | `bin` |
| `boxplot` | `boxplot` |
| `violin` | `density` |
| `smooth` | `smooth` |

Override defaults:

```rust
use trueno_viz::grammar::{Stat, Geom};

// Bar with pre-computed heights
let bar = Geom::bar().stat(Stat::identity());

// Points at bin centers
let points = Geom::point().stat(Stat::bin());
```

## Complete Example

```rust
use trueno_viz::grammar::{Stat, Geom, GGPlot, DataFrame, Aes, Theme};

fn main() {
    // Generate random-looking data
    let data: Vec<f32> = (0..200)
        .map(|i| ((i as f32 * 0.1).sin() + 1.5) * 10.0)
        .collect();

    let df = DataFrame::new()
        .column("value", &data);

    let plot = GGPlot::new(df)
        .aes(Aes::new().x("value"))
        // Histogram with density
        .geom(Geom::histogram()
            .stat(Stat::bin().bins(20))
            .alpha(0.5))
        // Overlay density curve
        .geom(Geom::density()
            .stat(Stat::density().kernel(Kernel::Gaussian))
            .color(Rgba::RED))
        .title("Distribution with Density Overlay")
        .theme(Theme::minimal());

    plot.render_to_file("distribution.png").unwrap();
}
```

## Next Chapter

Continue to [Scales](./scales.md) to learn how data values map to
visual properties.
