# Themes

Themes control the non-data visual elements of a plot: backgrounds, fonts,
grid lines, and axis styling.

## Built-in Themes

```rust
use trueno_viz::grammar::Theme;

// Default theme (gray background, white grid)
let theme = Theme::default();

// Minimal theme (white background, no grid)
let theme = Theme::minimal();

// Classic theme (traditional look)
let theme = Theme::classic();

// Dark theme
let theme = Theme::dark();

// Publication-ready theme
let theme = Theme::publication();
```

**Test Reference**: `src/grammar/theme.rs::test_theme_constructors`

## Theme Components

### Background

```rust
use trueno_viz::grammar::{Theme, ThemeElement};
use trueno_viz::color::Rgba;

let theme = Theme::default()
    .panel_background(ThemeElement::rect()
        .fill(Rgba::new(240, 240, 240, 255))
        .color(Rgba::BLACK)
        .size(1.0));
```

### Grid Lines

```rust
use trueno_viz::grammar::{Theme, ThemeElement};
use trueno_viz::color::Rgba;

let theme = Theme::default()
    // Major grid lines
    .panel_grid_major(ThemeElement::line()
        .color(Rgba::new(200, 200, 200, 255))
        .size(0.5))
    // Minor grid lines
    .panel_grid_minor(ThemeElement::line()
        .color(Rgba::new(230, 230, 230, 255))
        .size(0.25))
    // Remove x grid
    .panel_grid_major_x(ThemeElement::blank())
    // Remove y grid
    .panel_grid_minor_y(ThemeElement::blank());
```

### Axis Elements

```rust
use trueno_viz::grammar::{Theme, ThemeElement};
use trueno_viz::color::Rgba;

let theme = Theme::default()
    // Axis lines
    .axis_line(ThemeElement::line()
        .color(Rgba::BLACK)
        .size(1.0))
    // Axis ticks
    .axis_ticks(ThemeElement::line()
        .color(Rgba::BLACK)
        .size(0.5))
    // Axis text (labels)
    .axis_text(ThemeElement::text()
        .color(Rgba::BLACK)
        .size(10.0))
    // Axis titles
    .axis_title(ThemeElement::text()
        .color(Rgba::BLACK)
        .size(12.0)
        .face("bold"));
```

### Text Elements

```rust
use trueno_viz::grammar::{Theme, ThemeElement};
use trueno_viz::color::Rgba;

let theme = Theme::default()
    // Plot title
    .plot_title(ThemeElement::text()
        .color(Rgba::BLACK)
        .size(16.0)
        .face("bold")
        .hjust(0.5))  // Center
    // Subtitle
    .plot_subtitle(ThemeElement::text()
        .color(Rgba::new(100, 100, 100, 255))
        .size(12.0)
        .hjust(0.5))
    // Caption
    .plot_caption(ThemeElement::text()
        .color(Rgba::new(150, 150, 150, 255))
        .size(9.0)
        .hjust(1.0));  // Right align
```

### Legend

```rust
use trueno_viz::grammar::{Theme, ThemeElement, LegendPosition};

let theme = Theme::default()
    // Legend position
    .legend_position(LegendPosition::Right)
    // Legend background
    .legend_background(ThemeElement::rect()
        .fill(Rgba::WHITE)
        .color(Rgba::new(200, 200, 200, 255)))
    // Legend title
    .legend_title(ThemeElement::text()
        .size(11.0)
        .face("bold"))
    // Legend text
    .legend_text(ThemeElement::text()
        .size(10.0));
```

## Theme Element Inheritance

Theme elements inherit from parent elements:

```text
axis_text_x inherits from axis_text
axis_text_y inherits from axis_text
axis_text inherits from text
```

Override specific elements:

```rust
use trueno_viz::grammar::{Theme, ThemeElement};

let theme = Theme::default()
    // Set all axis text
    .axis_text(ThemeElement::text().size(10.0))
    // Override just x-axis text
    .axis_text_x(ThemeElement::text().angle(45.0));
```

## Combining Themes

Themes can be combined by layering:

```rust
use trueno_viz::grammar::Theme;

// Start with minimal, customize
let theme = Theme::minimal()
    .panel_grid_major(ThemeElement::line()
        .color(Rgba::new(220, 220, 220, 255)))
    .plot_title(ThemeElement::text()
        .face("bold"));
```

## Blank Elements

Remove elements with `ThemeElement::blank()`:

```rust
use trueno_viz::grammar::{Theme, ThemeElement};

let theme = Theme::default()
    .panel_grid(ThemeElement::blank())      // No grid
    .axis_ticks(ThemeElement::blank())      // No ticks
    .legend_background(ThemeElement::blank()); // Transparent legend
```

## Theme Presets

### Publication Theme

Optimized for academic papers:

```rust
use trueno_viz::grammar::Theme;

let theme = Theme::publication();
// Equivalent to:
// - White background
// - Black axis lines and text
// - No grid lines
// - Serif fonts
// - High-contrast colors
```

### Dark Theme

For dark mode UIs:

```rust
use trueno_viz::grammar::Theme;

let theme = Theme::dark();
// Equivalent to:
// - Dark gray background
// - Light gray text
// - Subtle grid lines
// - Vibrant accent colors
```

## Complete Example

```rust
use trueno_viz::grammar::{Theme, ThemeElement, GGPlot, Aes, Geom, DataFrame, LegendPosition};
use trueno_viz::color::Rgba;

fn main() {
    let df = DataFrame::new()
        .column("x", &[1.0, 2.0, 3.0, 4.0, 5.0])
        .column("y", &[2.0, 4.0, 3.0, 5.0, 4.5])
        .column("group", &["A", "A", "B", "B", "B"]);

    // Custom professional theme
    let custom_theme = Theme::minimal()
        // Title styling
        .plot_title(ThemeElement::text()
            .size(18.0)
            .face("bold")
            .color(Rgba::new(50, 50, 50, 255)))
        // Axis styling
        .axis_title(ThemeElement::text()
            .size(12.0)
            .color(Rgba::new(80, 80, 80, 255)))
        .axis_text(ThemeElement::text()
            .size(10.0)
            .color(Rgba::new(100, 100, 100, 255)))
        // Grid (horizontal only)
        .panel_grid_major_y(ThemeElement::line()
            .color(Rgba::new(230, 230, 230, 255))
            .size(0.5))
        .panel_grid_major_x(ThemeElement::blank())
        // Legend
        .legend_position(LegendPosition::Bottom);

    let plot = GGPlot::new(df)
        .aes(Aes::new().x("x").y("y").color("group"))
        .geom(Geom::point().size(4.0))
        .geom(Geom::line())
        .title("Professional Looking Chart")
        .theme(custom_theme);

    plot.render_to_file("themed_plot.png").unwrap();
}
```

## Theme Reference

| Element | Description |
|---------|-------------|
| `panel_background` | Plot area background |
| `panel_border` | Plot area border |
| `panel_grid_major` | Major grid lines |
| `panel_grid_minor` | Minor grid lines |
| `axis_line` | Axis lines |
| `axis_ticks` | Axis tick marks |
| `axis_text` | Axis tick labels |
| `axis_title` | Axis titles |
| `plot_title` | Main title |
| `plot_subtitle` | Subtitle |
| `plot_caption` | Caption |
| `legend_background` | Legend background |
| `legend_title` | Legend title |
| `legend_text` | Legend labels |

## Next Chapter

Continue to [Plot Types](../plots/scatter.md) for detailed coverage
of each visualization type.
