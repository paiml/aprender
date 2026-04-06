# Coordinate Systems

Coordinate systems define how positions are mapped to the plotting area.
Trueno-viz supports Cartesian, polar, and fixed-ratio coordinate systems.

## Cartesian Coordinates

The default coordinate system with perpendicular x and y axes:

```rust
use trueno_viz::grammar::Coord;

let coord = Coord::cartesian();

// With custom limits
let coord = Coord::cartesian()
    .xlim(0.0, 100.0)
    .ylim(-50.0, 50.0);
```

**Test Reference**: `src/grammar/coord.rs::test_coord_cartesian`

### Axis Limits

```rust
use trueno_viz::grammar::Coord;

// Set x-axis limits
let coord = Coord::cartesian().xlim(0.0, 10.0);

// Set y-axis limits
let coord = Coord::cartesian().ylim(-5.0, 5.0);

// Set both
let coord = Coord::cartesian()
    .xlim(0.0, 10.0)
    .ylim(-5.0, 5.0);
```

Verification:

```rust
use trueno_viz::grammar::Coord;

let coord = Coord::cartesian().xlim(0.0, 10.0).ylim(-5.0, 5.0);

match coord {
    Coord::Cartesian { xlim, ylim, flip } => {
        assert_eq!(xlim, Some((0.0, 10.0)));
        assert_eq!(ylim, Some((-5.0, 5.0)));
        assert!(!flip);
    }
    _ => panic!("Expected Cartesian"),
}
```

### Flipped Coordinates

Swap x and y axes (useful for horizontal bar charts):

```rust
use trueno_viz::grammar::{Coord, GGPlot, Aes, Geom, DataFrame};

let df = DataFrame::new()
    .column("category", &["A", "B", "C", "D"])
    .column("value", &[10.0, 25.0, 15.0, 30.0]);

// Horizontal bar chart
let plot = GGPlot::new(df)
    .aes(Aes::new().x("category").y("value"))
    .geom(Geom::bar())
    .coord(Coord::cartesian().flip());
```

**Test Reference**: `src/grammar/coord.rs::test_coord_flip`

## Polar Coordinates

Transform Cartesian coordinates to polar (radial) coordinates:

```rust
use trueno_viz::grammar::Coord;

let coord = Coord::polar();

// With custom start angle (radians)
let coord = Coord::polar()
    .start_angle(std::f32::consts::PI / 2.0);  // Start at top

// Counter-clockwise direction
let coord = Coord::polar()
    .direction(-1);
```

**Test Reference**: `src/grammar/coord.rs::test_coord_polar`

### Polar Coordinate Mapping

```text
Cartesian (x, y) → Polar (θ, r)

x maps to angle (theta)
y maps to radius (r)

      90°
       │
180° ──┼── 0°
       │
      270°
```

### Pie Charts

Polar coordinates with bar geom creates pie charts:

```rust
use trueno_viz::grammar::{Coord, GGPlot, Aes, Geom, DataFrame};

let df = DataFrame::new()
    .column("category", &["A", "B", "C"])
    .column("count", &[30.0, 50.0, 20.0]);

let plot = GGPlot::new(df)
    .aes(Aes::new().x("category").y("count").fill("category"))
    .geom(Geom::bar().position(Position::Stack))
    .coord(Coord::polar());
```

### Rose Diagrams

Wind rose or other radial histograms:

```rust
use trueno_viz::grammar::{Coord, GGPlot, Aes, Geom, DataFrame};

let df = DataFrame::new()
    .column("direction", &[0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0])
    .column("speed", &[5.0, 8.0, 12.0, 6.0, 3.0, 7.0, 10.0, 4.0]);

let plot = GGPlot::new(df)
    .aes(Aes::new().x("direction").y("speed"))
    .geom(Geom::bar())
    .coord(Coord::polar()
        .start_angle(std::f32::consts::PI / 2.0)  // North at top
        .direction(-1));  // Clockwise
```

## Fixed Aspect Ratio

Maintain a specific y/x ratio:

```rust
use trueno_viz::grammar::Coord;

// Equal scaling (1:1 ratio)
let coord = Coord::fixed(1.0);

// 2:1 ratio (y range is twice x range visually)
let coord = Coord::fixed(2.0);
```

**Test Reference**: `src/grammar/coord.rs::test_coord_fixed`

### Use Cases for Fixed Ratio

```rust
use trueno_viz::grammar::{Coord, GGPlot, Aes, Geom, DataFrame};

// Geographic data (lat/lon)
let map_plot = GGPlot::new(geo_data)
    .coord(Coord::fixed(1.0));  // Equal lat/lon scaling

// Physical proportions
let physical_plot = GGPlot::new(measurements)
    .coord(Coord::fixed(1.0));  // True aspect ratio
```

## Coordinate System in GGPlot

```rust
use trueno_viz::grammar::{Coord, GGPlot, Aes, Geom, DataFrame, Theme};

let plot = GGPlot::new(df)
    .aes(Aes::new().x("x").y("y"))
    .geom(Geom::point())
    .coord(Coord::cartesian()
        .xlim(0.0, 100.0)
        .ylim(0.0, 50.0))
    .theme(Theme::minimal());
```

## Coordinate Transformations

### Zooming

```rust
use trueno_viz::grammar::{Coord, GGPlot};

// Zoom into a region without affecting stat calculations
let zoomed = plot.coord(Coord::cartesian()
    .xlim(20.0, 40.0)
    .ylim(10.0, 30.0));
```

### Aspect Ratio Correction

```rust
use trueno_viz::grammar::Coord;

// Correct for different x/y units
let coord = Coord::fixed(0.5);  // y units are half x units visually
```

## Effect on Geoms

Coordinate systems affect how geoms are rendered:

| Geom | Cartesian | Polar |
|------|-----------|-------|
| `bar` | Vertical bars | Pie/donut sectors |
| `line` | Straight lines | Curved lines |
| `point` | Regular points | Radial points |
| `area` | Stacked area | Radial area |

## Complete Example

```rust
use trueno_viz::grammar::{Coord, GGPlot, Aes, Geom, DataFrame, Theme};
use trueno_viz::color::Rgba;

fn main() {
    // Market share data
    let df = DataFrame::new()
        .column("company", &["Apple", "Samsung", "Xiaomi", "Others"])
        .column("share", &[27.0, 21.0, 14.0, 38.0]);

    // Pie chart
    let pie = GGPlot::new(df.clone())
        .aes(Aes::new()
            .x("")  // Single category for full pie
            .y("share")
            .fill("company"))
        .geom(Geom::bar()
            .position(Position::Stack))
        .coord(Coord::polar())
        .title("Smartphone Market Share")
        .theme(Theme::minimal());

    pie.render_to_file("market_share_pie.png").unwrap();

    // Bar chart (same data, different coord)
    let bar = GGPlot::new(df)
        .aes(Aes::new()
            .x("company")
            .y("share")
            .fill("company"))
        .geom(Geom::bar())
        .coord(Coord::cartesian().flip())  // Horizontal bars
        .title("Smartphone Market Share")
        .theme(Theme::minimal());

    bar.render_to_file("market_share_bar.png").unwrap();
}
```

## Next Chapter

Continue to [Faceting](./facet.md) to learn about creating multi-panel
layouts.
