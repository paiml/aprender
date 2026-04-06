# Faceting

Faceting creates multi-panel plots by splitting data across multiple
subplots based on categorical variables. This technique is also known
as "small multiples" or "trellis plots."

## Facet Types

| Facet | Description |
|-------|-------------|
| `wrap` | Wrap panels into rows |
| `grid` | 2D grid of rows × columns |

## Facet Wrap

Arranges panels in a wrapped layout:

```rust
use trueno_viz::grammar::Facet;

// Single variable faceting
let facet = Facet::wrap("species");

// Control number of columns
let facet = Facet::wrap("species").ncol(3);

// Control number of rows
let facet = Facet::wrap("species").nrow(2);
```

**Test Reference**: `src/grammar/facet.rs::test_facet_wrap`

### Wrap Example

```rust
use trueno_viz::grammar::{Facet, GGPlot, Aes, Geom, DataFrame, Theme};

let df = DataFrame::new()
    .column("x", &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
    .column("y", &[1.0, 4.0, 9.0, 2.0, 5.0, 8.0, 3.0, 6.0, 7.0])
    .column("group", &["A", "A", "A", "B", "B", "B", "C", "C", "C"]);

let plot = GGPlot::new(df)
    .aes(Aes::new().x("x").y("y"))
    .geom(Geom::point())
    .geom(Geom::line())
    .facet(Facet::wrap("group").ncol(3))
    .title("Trends by Group");
```

Output layout:
```text
┌─────────┬─────────┬─────────┐
│ group=A │ group=B │ group=C │
│    •    │    •    │   •     │
│   •     │   •     │  •      │
│  •      │  •      │ •       │
└─────────┴─────────┴─────────┘
```

## Facet Grid

Creates a 2D grid based on two variables:

```rust
use trueno_viz::grammar::Facet;

// Row and column faceting
let facet = Facet::grid("gender", "age_group");

// Row only
let facet = Facet::grid("gender", ".");

// Column only
let facet = Facet::grid(".", "age_group");
```

**Test Reference**: `src/grammar/facet.rs::test_facet_grid`

### Grid Example

```rust
use trueno_viz::grammar::{Facet, GGPlot, Aes, Geom, DataFrame};

let df = DataFrame::new()
    .column("x", &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
    .column("y", &[2.0, 4.0, 6.0, 3.0, 5.0, 7.0, 1.0, 3.0, 5.0, 4.0, 6.0, 8.0])
    .column("gender", &["M", "M", "M", "M", "M", "M", "F", "F", "F", "F", "F", "F"])
    .column("treatment", &["A", "A", "A", "B", "B", "B", "A", "A", "A", "B", "B", "B"]);

let plot = GGPlot::new(df)
    .aes(Aes::new().x("x").y("y"))
    .geom(Geom::point())
    .facet(Facet::grid("gender", "treatment"));
```

Output layout:
```text
              treatment=A    treatment=B
            ┌─────────────┬─────────────┐
 gender=M   │      •      │      •      │
            │     •       │     •       │
            │    •        │    •        │
            ├─────────────┼─────────────┤
 gender=F   │    •        │      •      │
            │   •         │     •       │
            │  •          │    •        │
            └─────────────┴─────────────┘
```

## Facet Scales

Control whether scales are shared across panels:

```rust
use trueno_viz::grammar::{Facet, FacetScales};

// Fixed scales (same across all panels)
let facet = Facet::wrap("group").scales(FacetScales::Fixed);

// Free scales (each panel has own scale)
let facet = Facet::wrap("group").scales(FacetScales::Free);

// Free x only
let facet = Facet::wrap("group").scales(FacetScales::FreeX);

// Free y only
let facet = Facet::wrap("group").scales(FacetScales::FreeY);
```

### Scale Options Effect

```text
FacetScales::Fixed:          FacetScales::Free:
┌─────────┬─────────┐        ┌─────────┬─────────┐
│ 0-100   │ 0-100   │        │ 0-50    │ 0-100   │
│   •     │       • │        │   •     │       • │
│         │         │        │         │         │
└─────────┴─────────┘        └─────────┴─────────┘
  Same y-axis range           Different y-axis ranges
```

## Facet Labels

Customize panel labels:

```rust
use trueno_viz::grammar::{Facet, Labeller};

// Default: variable value only
let facet = Facet::wrap("species");  // "setosa"

// Both variable and value
let facet = Facet::wrap("species")
    .labeller(Labeller::Both);  // "species: setosa"

// Custom labeller function
let facet = Facet::wrap("species")
    .labeller(Labeller::Custom(|var, val| {
        format!("{} = {}", var.to_uppercase(), val)
    }));  // "SPECIES = setosa"
```

## Facet Spacing

Control spacing between panels:

```rust
use trueno_viz::grammar::Facet;

let facet = Facet::wrap("group")
    .space(FacetSpace::Equal)      // Equal panel sizes
    .margin(10.0);                  // 10px between panels
```

## Facet with Free Space

Adjust panel sizes based on data:

```rust
use trueno_viz::grammar::{Facet, FacetSpace};

// Panels sized proportionally to data range
let facet = Facet::grid("row_var", "col_var")
    .space(FacetSpace::FreeX);  // Column widths vary
```

## Complete Example

```rust
use trueno_viz::grammar::{Facet, GGPlot, Aes, Geom, DataFrame, Theme, FacetScales};

fn main() {
    // Iris dataset (simplified)
    let df = DataFrame::new()
        .column("sepal_length", &[5.1, 4.9, 7.0, 6.5, 5.8, 6.7])
        .column("sepal_width", &[3.5, 3.0, 3.2, 2.8, 2.7, 3.0])
        .column("petal_length", &[1.4, 1.4, 4.7, 4.6, 5.1, 5.2])
        .column("species", &["setosa", "setosa", "versicolor",
                             "versicolor", "virginica", "virginica"]);

    let plot = GGPlot::new(df)
        .aes(Aes::new()
            .x("sepal_length")
            .y("sepal_width")
            .color("species"))
        .geom(Geom::point().size(4.0))
        .facet(Facet::wrap("species")
            .ncol(3)
            .scales(FacetScales::Fixed)
            .labeller(Labeller::Both))
        .title("Iris Sepal Dimensions by Species")
        .theme(Theme::minimal());

    plot.render_to_file("iris_faceted.png").unwrap();
}
```

## When to Use Faceting

| Scenario | Recommended Facet |
|----------|-------------------|
| Single grouping variable | `Facet::wrap` |
| Two grouping variables | `Facet::grid` |
| Many groups (>6) | `Facet::wrap` with `ncol` |
| Compare distributions | `Facet::wrap` with `scales(Free)` |
| Cross-tabulation | `Facet::grid` |

## Next Chapter

Continue to [Themes](./theme.md) to learn about customizing the visual
appearance of plots.
