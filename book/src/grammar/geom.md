# Geometric Objects

Geometric objects (geoms) are the visual elements that represent data.
Each geom type produces different visual output from the same data.

## Available Geoms

| Geom | Description | Required Aesthetics |
|------|-------------|---------------------|
| `point` | Scatter points | x, y |
| `line` | Connected lines | x, y |
| `bar` | Bar charts | x, y |
| `histogram` | Frequency distribution | x |
| `boxplot` | Box and whisker | x, y |
| `violin` | Density distribution | x, y |
| `area` | Filled area | x, y |
| `path` | Connected path | x, y |
| `text` | Text labels | x, y, label |
| `heatmap` | 2D density | x, y, fill |

## Point Geom

```rust
use trueno_viz::grammar::{Geom, Aes, GGPlot, DataFrame};
use trueno_viz::color::Rgba;

let df = DataFrame::new()
    .column("x", &[1.0, 2.0, 3.0, 4.0, 5.0])
    .column("y", &[2.0, 4.0, 1.0, 5.0, 3.0]);

let plot = GGPlot::new(df)
    .aes(Aes::new().x("x").y("y"))
    .geom(Geom::point()
        .size(5.0)
        .color(Rgba::BLUE)
        .alpha(0.8));
```

**Test Reference**: `src/grammar/geom.rs::test_geom_point`

### Point Shapes

```rust
use trueno_viz::grammar::{Geom, PointShape};

let circle = Geom::point().shape(PointShape::Circle);
let square = Geom::point().shape(PointShape::Square);
let triangle = Geom::point().shape(PointShape::Triangle);
let diamond = Geom::point().shape(PointShape::Diamond);
```

## Line Geom

```rust
use trueno_viz::grammar::{Geom, Aes, GGPlot, DataFrame};

let df = DataFrame::new()
    .column("time", &[0.0, 1.0, 2.0, 3.0, 4.0])
    .column("value", &[10.0, 15.0, 12.0, 18.0, 16.0]);

let plot = GGPlot::new(df)
    .aes(Aes::new().x("time").y("value"))
    .geom(Geom::line()
        .line_width(2.0)
        .line_type(LineType::Solid));
```

### Line Types

```rust
use trueno_viz::grammar::{Geom, LineType};

let solid = Geom::line().line_type(LineType::Solid);
let dashed = Geom::line().line_type(LineType::Dashed);
let dotted = Geom::line().line_type(LineType::Dotted);
let dash_dot = Geom::line().line_type(LineType::DashDot);
```

## Bar Geom

```rust
use trueno_viz::grammar::{Geom, Aes, GGPlot, DataFrame, Position};

let df = DataFrame::new()
    .column("category", &["A", "B", "C", "D"])
    .column("count", &[25.0, 40.0, 30.0, 35.0]);

// Simple bar chart
let plot = GGPlot::new(df)
    .aes(Aes::new().x("category").y("count"))
    .geom(Geom::bar());
```

### Stacked vs Dodged Bars

```rust
use trueno_viz::grammar::{Geom, Position};

// Stacked bars (default)
let stacked = Geom::bar().position(Position::Stack);

// Side-by-side bars
let dodged = Geom::bar().position(Position::Dodge);

// Percentage stacked
let filled = Geom::bar().position(Position::Fill);
```

**Test Reference**: `src/grammar/geom.rs::test_geom_bar_position`

## Histogram Geom

```rust
use trueno_viz::grammar::{Geom, Aes, GGPlot, DataFrame};

let df = DataFrame::new()
    .column("values", &[1.0, 2.1, 2.3, 2.5, 3.0, 3.1, 3.5, 4.0, 4.2, 5.0]);

let plot = GGPlot::new(df)
    .aes(Aes::new().x("values"))
    .geom(Geom::histogram()
        .bins(10)
        .boundary(0.0));
```

### Binning Options

```rust
use trueno_viz::grammar::Geom;

// Fixed number of bins
let fixed = Geom::histogram().bins(30);

// Bin width
let width = Geom::histogram().binwidth(0.5);

// Boundary alignment
let aligned = Geom::histogram()
    .bins(10)
    .boundary(0.0);
```

## Box Plot Geom

```rust
use trueno_viz::grammar::{Geom, Aes, GGPlot, DataFrame};

let df = DataFrame::new()
    .column("group", &["A", "A", "A", "B", "B", "B"])
    .column("value", &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

let plot = GGPlot::new(df)
    .aes(Aes::new().x("group").y("value"))
    .geom(Geom::boxplot()
        .outlier_color(Rgba::RED)
        .whisker_width(0.5));
```

**Test Reference**: `src/plots/boxplot.rs::test_boxplot_builder`

## Violin Geom

```rust
use trueno_viz::grammar::{Geom, Aes, GGPlot, DataFrame};

let plot = GGPlot::new(df)
    .aes(Aes::new().x("group").y("value"))
    .geom(Geom::violin()
        .bandwidth(0.5)
        .scale(ViolinScale::Area));
```

## Area Geom

```rust
use trueno_viz::grammar::{Geom, Aes, GGPlot, DataFrame};
use trueno_viz::color::Rgba;

let df = DataFrame::new()
    .column("x", &[1.0, 2.0, 3.0, 4.0, 5.0])
    .column("y", &[1.0, 3.0, 2.0, 5.0, 4.0]);

let plot = GGPlot::new(df)
    .aes(Aes::new().x("x").y("y"))
    .geom(Geom::area()
        .fill(Rgba::new(66, 133, 244, 128))  // Semi-transparent
        .alpha(0.5));
```

## Text Geom

```rust
use trueno_viz::grammar::{Geom, Aes, GGPlot, DataFrame};

let df = DataFrame::new()
    .column("x", &[1.0, 2.0, 3.0])
    .column("y", &[4.0, 5.0, 6.0])
    .column("name", &["Point A", "Point B", "Point C"]);

let plot = GGPlot::new(df)
    .aes(Aes::new().x("x").y("y").label("name"))
    .geom(Geom::point())
    .geom(Geom::text()
        .nudge_y(0.2)
        .size(12.0));
```

## Combining Geoms

Multiple geoms can be layered:

```rust
use trueno_viz::grammar::{Geom, Aes, GGPlot, DataFrame};

let plot = GGPlot::new(df)
    .aes(Aes::new().x("x").y("y"))
    .geom(Geom::point().size(3.0))        // Points
    .geom(Geom::line().alpha(0.5))         // Connecting line
    .geom(Geom::smooth().method("lm"));    // Regression line
```

## Geom-Specific Aesthetics

Each geom can override plot-level aesthetics:

```rust
use trueno_viz::grammar::{Geom, Aes, GGPlot, DataFrame};

let plot = GGPlot::new(df)
    .aes(Aes::new().x("x").y("y"))
    .geom(Geom::point()
        .aes(Aes::new().color("category")))  // Points colored by category
    .geom(Geom::line()
        .aes(Aes::new().color("series")));   // Lines colored by series
```

## Complete Example

```rust
use trueno_viz::grammar::{Geom, Aes, GGPlot, DataFrame, Theme};
use trueno_viz::color::Rgba;

fn main() {
    let df = DataFrame::new()
        .column("month", &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        .column("revenue", &[100.0, 120.0, 115.0, 140.0, 150.0, 145.0])
        .column("target", &[110.0, 115.0, 120.0, 130.0, 140.0, 150.0]);

    let plot = GGPlot::new(df)
        .aes(Aes::new().x("month"))
        // Actual revenue as bars
        .geom(Geom::bar()
            .aes(Aes::new().y("revenue"))
            .fill(Rgba::new(66, 133, 244, 200)))
        // Target as dashed line
        .geom(Geom::line()
            .aes(Aes::new().y("target"))
            .color(Rgba::RED)
            .line_type(LineType::Dashed)
            .line_width(2.0))
        .title("Revenue vs Target")
        .theme(Theme::minimal());

    plot.render_to_file("revenue.png").unwrap();
}
```

## Next Chapter

Continue to [Statistical Transformations](./stat.md) to learn how
data is transformed before visualization.
