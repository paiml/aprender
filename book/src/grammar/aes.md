# Aesthetic Mappings

Aesthetics define how data variables map to visual properties. The `Aes`
struct connects data columns to visual channels like position, color, and size.

## Core Aesthetics

```rust
use trueno_viz::grammar::Aes;

let aes = Aes::new()
    .x("mpg")           // X position
    .y("horsepower")    // Y position
    .color("cylinders") // Color
    .size("weight")     // Point size
    .shape("origin")    // Point shape
    .alpha("age");      // Transparency
```

## The Aes Builder

Each aesthetic method returns `Self` for fluent chaining:

```rust
use trueno_viz::grammar::Aes;

// Minimal aesthetic (just x and y)
let simple = Aes::new().x("time").y("value");

// Full aesthetic specification
let full = Aes::new()
    .x("x")
    .y("y")
    .color("category")
    .fill("subcategory")
    .size("importance")
    .shape("type")
    .alpha("confidence")
    .group("batch");

// Verify mappings
assert!(simple.x_var().is_some());
assert!(simple.y_var().is_some());
```

**Test Reference**: `src/grammar/aes.rs::test_aes_builder`

## Position Aesthetics

### x and y

Primary position aesthetics for 2D plots:

```rust
use trueno_viz::grammar::{Aes, GGPlot, Geom, DataFrame};

let df = DataFrame::new()
    .column("date", &[1.0, 2.0, 3.0, 4.0])
    .column("price", &[100.0, 105.0, 102.0, 108.0]);

let plot = GGPlot::new(df)
    .aes(Aes::new().x("date").y("price"))
    .geom(Geom::line());
```

### xmin, xmax, ymin, ymax

For range geometries (rectangles, error bars):

```rust
use trueno_viz::grammar::Aes;

let aes = Aes::new()
    .x("center")
    .xmin("lower_bound")
    .xmax("upper_bound")
    .y("value");
```

## Color Aesthetics

### color vs fill

```rust
use trueno_viz::grammar::{Aes, Geom};

// `color` affects outlines/lines
let line_aes = Aes::new().x("x").y("y").color("series");

// `fill` affects interior color
let bar_aes = Aes::new().x("category").y("count").fill("type");
```

Visual difference:

```text
color aesthetic:          fill aesthetic:
┌─────────────┐           ┌─────────────┐
│             │           │█████████████│
│   ┌─────┐   │           │█████████████│
│   │     │   │ ← outline │█████████████│ ← interior
│   │     │   │   colored │█████████████│   colored
│   └─────┘   │           │█████████████│
└─────────────┘           └─────────────┘
```

## Size and Shape

### Size Mapping

```rust
use trueno_viz::grammar::{Aes, GGPlot, Geom, DataFrame};

// Size mapped to a variable (bubble chart)
let df = DataFrame::new()
    .column("gdp", &[1000.0, 5000.0, 2000.0])
    .column("life_exp", &[65.0, 80.0, 72.0])
    .column("population", &[50.0, 300.0, 100.0]);

let plot = GGPlot::new(df)
    .aes(Aes::new()
        .x("gdp")
        .y("life_exp")
        .size("population"))
    .geom(Geom::point());
```

### Shape Mapping

```rust
use trueno_viz::grammar::Aes;

// Categorical shape mapping
let aes = Aes::new()
    .x("x")
    .y("y")
    .shape("species");  // Different shapes per species
```

Available shapes:
- Circle (default)
- Square
- Triangle
- Diamond
- Cross
- Plus

## Alpha (Transparency)

```rust
use trueno_viz::grammar::Aes;

// Transparency mapped to confidence
let aes = Aes::new()
    .x("predicted")
    .y("actual")
    .alpha("confidence");  // 0.0 = transparent, 1.0 = opaque
```

## Grouping

The `group` aesthetic defines how data is split:

```rust
use trueno_viz::grammar::{Aes, GGPlot, Geom, DataFrame};

let df = DataFrame::new()
    .column("time", &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
    .column("value", &[10.0, 15.0, 12.0, 20.0, 25.0, 22.0])
    .column("sensor", &["A", "A", "A", "B", "B", "B"]);

// Without group: single line connecting all points
// With group: separate line per sensor
let plot = GGPlot::new(df)
    .aes(Aes::new()
        .x("time")
        .y("value")
        .group("sensor")
        .color("sensor"))
    .geom(Geom::line());
```

## Aesthetic Inheritance

Aesthetics cascade from plot level to geom level:

```rust
use trueno_viz::grammar::{Aes, GGPlot, Geom, DataFrame};

let df = DataFrame::new()
    .column("x", &[1.0, 2.0, 3.0])
    .column("y", &[1.0, 4.0, 9.0])
    .column("y2", &[1.0, 2.0, 3.0]);

// Plot-level aes inherited by all geoms
let plot = GGPlot::new(df)
    .aes(Aes::new().x("x"))  // Plot-level x
    .geom(Geom::point().aes(Aes::new().y("y")))   // Geom-specific y
    .geom(Geom::line().aes(Aes::new().y("y2")));  // Different y
```

## Setting vs Mapping

**Mapping**: Variable → Visual property
**Setting**: Fixed value → Visual property

```rust
use trueno_viz::grammar::{Aes, Geom};
use trueno_viz::color::Rgba;

// MAPPING: color varies with data
let mapped = Aes::new().color("category");

// SETTING: all points are blue
let geom = Geom::point().color(Rgba::BLUE);
```

## Complete Example

```rust
use trueno_viz::grammar::{Aes, GGPlot, Geom, DataFrame, Theme};

fn main() {
    let df = DataFrame::new()
        .column("sepal_length", &[5.1, 4.9, 7.0, 6.3, 5.8])
        .column("sepal_width", &[3.5, 3.0, 3.2, 3.3, 2.7])
        .column("petal_length", &[1.4, 1.4, 4.7, 6.0, 5.1])
        .column("species", &["setosa", "setosa", "versicolor",
                             "virginica", "virginica"]);

    let plot = GGPlot::new(df)
        .aes(Aes::new()
            .x("sepal_length")
            .y("sepal_width")
            .color("species")
            .size("petal_length"))
        .geom(Geom::point())
        .title("Iris Dataset")
        .theme(Theme::minimal());

    plot.render_to_file("iris.png").unwrap();
}
```

## Next Chapter

Continue to [Geometric Objects](./geom.md) to learn about the visual
elements that represent data.
