# Grammar of Graphics Overview

The Grammar of Graphics (Wilkinson, 2005) provides a formal framework for
describing statistical graphics. Trueno-viz implements this grammar in Rust,
enabling declarative, composable visualizations.

## The Seven Layers

A complete graphic specification consists of seven components:

```text
┌─────────────────────────────────────────────┐
│                   Theme                     │  Visual appearance
├─────────────────────────────────────────────┤
│                   Coord                     │  Coordinate system
├─────────────────────────────────────────────┤
│                   Facet                     │  Panel layout
├─────────────────────────────────────────────┤
│                   Scale                     │  Data → visual mapping
├─────────────────────────────────────────────┤
│                   Stat                      │  Statistical transform
├─────────────────────────────────────────────┤
│                   Geom                      │  Geometric object
├─────────────────────────────────────────────┤
│                   Aes                       │  Aesthetic mapping
├─────────────────────────────────────────────┤
│                   Data                      │  Raw data
└─────────────────────────────────────────────┘
```

## Example: Building a Visualization Layer by Layer

```rust
use trueno_viz::grammar::*;

// Layer 1: Data
let data = DataFrame::new()
    .column("x", &[1.0, 2.0, 3.0, 4.0, 5.0])
    .column("y", &[2.1, 3.9, 6.2, 7.8, 10.1])
    .column("group", &["A", "A", "B", "B", "B"]);

// Layer 2: Aesthetic Mappings
let aes = Aes::new()
    .x("x")
    .y("y")
    .color("group");

// Layer 3: Geometric Object
let geom = Geom::point()
    .size(5.0);

// Layer 4: Statistical Transform (identity = no transform)
let stat = Stat::identity();

// Layer 5: Scales
let scale_x = Scale::linear();
let scale_y = Scale::linear();
let scale_color = Scale::discrete();

// Layer 6: Coordinate System
let coord = Coord::cartesian();

// Layer 7: Theme
let theme = Theme::default();

// Compose into a plot
let plot = GGPlot::new(data)
    .aes(aes)
    .geom(geom)
    .stat(stat)
    .coord(coord)
    .theme(theme);
```

## The GGPlot Builder

Trueno-viz uses the builder pattern for fluent composition:

```rust
use trueno_viz::grammar::*;

let plot = GGPlot::new(data)
    .aes(Aes::new().x("mpg").y("hp"))
    .geom(Geom::point())
    .geom(Geom::line())  // Multiple geoms
    .theme(Theme::minimal())
    .title("MPG vs Horsepower");
```

**Test Reference**: `src/grammar/ggplot.rs::test_ggplot_builder`

## Layer Independence

Each layer is independent and reusable:

```rust
// Define reusable components
let scatter_aes = Aes::new().x("x").y("y");
let line_aes = Aes::new().x("x").y("predicted");

// Same data, different aesthetics
let scatter = GGPlot::new(data.clone())
    .aes(scatter_aes)
    .geom(Geom::point());

let regression = GGPlot::new(data)
    .aes(line_aes)
    .geom(Geom::line());
```

## Key Concepts

### 1. Data Transformation Pipeline

```text
Raw Data → Stat Transform → Scale Transform → Coord Transform → Render
```

### 2. Aesthetic Mapping vs Setting

```rust
// MAPPING: "color" mapped to a data column
let aes = Aes::new().color("species");

// SETTING: fixed color for all points
let geom = Geom::point().color(Rgba::BLUE);
```

### 3. Position Adjustments

```rust
// Stack bars on top of each other
let geom = Geom::bar().position(Position::Stack);

// Place bars side by side
let geom = Geom::bar().position(Position::Dodge);
```

## Chapter Contents

- [Data Layer](./data.md) - DataFrame and data structures
- [Aesthetic Mappings](./aes.md) - Variable to visual property mapping
- [Geometric Objects](./geom.md) - Points, lines, bars, etc.
- [Statistical Transformations](./stat.md) - Binning, smoothing, etc.
- [Scales](./scales.md) - Data to visual range mapping
- [Coordinate Systems](./coord.md) - Cartesian, polar, etc.
- [Faceting](./facet.md) - Multi-panel layouts
- [Themes](./theme.md) - Visual styling

## References

- Wilkinson, L. (2005). *The Grammar of Graphics*. Springer.
- Wickham, H. (2010). "A Layered Grammar of Graphics." *Journal of Computational and Graphical Statistics*.
