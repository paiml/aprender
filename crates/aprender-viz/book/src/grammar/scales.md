# Scales

Scales transform data values into visual properties. They define the
mapping between the data domain and the visual range.

## Scale Types

| Scale | Data Type | Visual Property |
|-------|-----------|-----------------|
| `linear` | Continuous | Position, size |
| `log` | Continuous (positive) | Position |
| `sqrt` | Continuous (non-negative) | Size |
| `discrete` | Categorical | Position, color |
| `color_continuous` | Continuous | Color gradient |
| `color_discrete` | Categorical | Color palette |

## Linear Scale

The default scale for continuous data:

```rust
use trueno_viz::scale::LinearScale;

let scale = LinearScale::new()
    .domain(0.0, 100.0)   // Data range
    .range(0.0, 800.0);   // Pixel range

// Transform values
let pixel = scale.transform(50.0);  // → 400.0
assert!((pixel - 400.0).abs() < f32::EPSILON);

// Inverse transform
let value = scale.inverse(400.0);   // → 50.0
assert!((value - 50.0).abs() < f32::EPSILON);
```

**Test Reference**: `src/scale.rs::test_linear_scale`

### Linear Scale Options

```rust
use trueno_viz::scale::LinearScale;

let scale = LinearScale::new()
    .domain(0.0, 100.0)
    .range(0.0, 800.0)
    .nice()               // Round domain to nice values
    .clamp(true);         // Clamp values outside domain
```

## Log Scale

For data spanning multiple orders of magnitude:

```rust
use trueno_viz::scale::LogScale;

let scale = LogScale::new()
    .domain(1.0, 1000.0)
    .range(0.0, 300.0)
    .base(10.0);

// Equal spacing for powers of 10
let p1 = scale.transform(1.0);    // → 0.0
let p10 = scale.transform(10.0);  // → 100.0
let p100 = scale.transform(100.0); // → 200.0
```

**Test Reference**: `src/scale.rs::test_log_scale`

### Log Scale Bases

```rust
use trueno_viz::scale::LogScale;

let log10 = LogScale::new().base(10.0);
let log2 = LogScale::new().base(2.0);
let ln = LogScale::new().base(std::f32::consts::E);
```

## Color Scales

### Continuous Color Scale

```rust
use trueno_viz::scale::ColorScale;
use trueno_viz::color::Rgba;

let scale = ColorScale::continuous()
    .domain(0.0, 100.0)
    .colors(&[Rgba::BLUE, Rgba::WHITE, Rgba::RED]);

let cold = scale.transform(0.0);    // Blue
let neutral = scale.transform(50.0); // White
let hot = scale.transform(100.0);   // Red
```

**Test Reference**: `src/scale.rs::test_color_scale_continuous`

### Built-in Color Palettes

```rust
use trueno_viz::scale::{ColorScale, Palette};

let viridis = ColorScale::palette(Palette::Viridis);
let plasma = ColorScale::palette(Palette::Plasma);
let inferno = ColorScale::palette(Palette::Inferno);
let magma = ColorScale::palette(Palette::Magma);
let cividis = ColorScale::palette(Palette::Cividis);

// Diverging palettes
let rdbu = ColorScale::palette(Palette::RdBu);  // Red-Blue
let brbg = ColorScale::palette(Palette::BrBG);  // Brown-Blue-Green
```

### Discrete Color Scale

```rust
use trueno_viz::scale::ColorScale;

let scale = ColorScale::discrete()
    .domain(&["A", "B", "C", "D"])
    .palette(Palette::Category10);

let color_a = scale.transform("A");
let color_b = scale.transform("B");
```

## Scale in GGPlot

### Automatic Scales

GGPlot infers scales from data:

```rust
use trueno_viz::grammar::{GGPlot, Aes, Geom, DataFrame};

let plot = GGPlot::new(df)
    .aes(Aes::new()
        .x("continuous_var")   // → LinearScale
        .y("log_var")          // → LinearScale (override for log)
        .color("category"));   // → DiscreteColorScale
```

### Manual Scale Override

```rust
use trueno_viz::grammar::{GGPlot, Aes, Geom, DataFrame, Scale};

let plot = GGPlot::new(df)
    .aes(Aes::new().x("value").y("response"))
    .scale_x(Scale::log().base(10.0))
    .scale_y(Scale::linear().limits(0.0, 100.0))
    .scale_color(Scale::discrete().palette(Palette::Set1));
```

## Scale Transformations

### Limits

```rust
use trueno_viz::scale::LinearScale;

// Expand limits
let scale = LinearScale::new()
    .domain(0.0, 100.0)
    .expand(0.05, 0.05);  // 5% padding on each side

// Fixed limits (clip data)
let scale = LinearScale::new()
    .limits(0.0, 50.0)
    .clamp(true);
```

### Nice Breaks

```rust
use trueno_viz::scale::LinearScale;

let scale = LinearScale::new()
    .domain(3.0, 97.0)
    .nice();  // Rounds to [0, 100]

let breaks = scale.breaks();  // [0, 20, 40, 60, 80, 100]
```

### Custom Breaks

```rust
use trueno_viz::scale::LinearScale;

let scale = LinearScale::new()
    .domain(0.0, 100.0)
    .breaks(&[0.0, 25.0, 50.0, 75.0, 100.0]);
```

## Scale Labels

```rust
use trueno_viz::grammar::{GGPlot, Scale};

let plot = GGPlot::new(df)
    .scale_x(Scale::linear()
        .name("Temperature (°C)")
        .breaks(&[0.0, 20.0, 40.0, 60.0, 80.0, 100.0])
        .labels(&["Freezing", "Cool", "Warm", "Hot", "Very Hot", "Boiling"]));
```

### Formatting Functions

```rust
use trueno_viz::scale::LinearScale;

let scale = LinearScale::new()
    .domain(0.0, 1.0)
    .labels_format(|v| format!("{:.0}%", v * 100.0));
```

## Complete Example

```rust
use trueno_viz::grammar::{GGPlot, Aes, Geom, DataFrame, Scale, Theme};
use trueno_viz::scale::{Palette, LinearScale};

fn main() {
    let df = DataFrame::new()
        .column("gdp", &[1000.0, 5000.0, 20000.0, 50000.0, 100000.0])
        .column("life_exp", &[55.0, 65.0, 75.0, 80.0, 82.0])
        .column("population", &[10.0, 50.0, 100.0, 300.0, 500.0])
        .column("continent", &["Africa", "Asia", "Europe", "Americas", "Oceania"]);

    let plot = GGPlot::new(df)
        .aes(Aes::new()
            .x("gdp")
            .y("life_exp")
            .size("population")
            .color("continent"))
        .geom(Geom::point())
        // Log scale for GDP (spans orders of magnitude)
        .scale_x(Scale::log()
            .base(10.0)
            .name("GDP per Capita (log scale)"))
        // Linear scale for life expectancy
        .scale_y(Scale::linear()
            .limits(50.0, 90.0)
            .name("Life Expectancy"))
        // Size scale
        .scale_size(Scale::sqrt()
            .range(5.0, 30.0))
        // Color scale
        .scale_color(Scale::discrete()
            .palette(Palette::Set2))
        .title("GDP vs Life Expectancy")
        .theme(Theme::minimal());

    plot.render_to_file("gapminder.png").unwrap();
}
```

## Next Chapter

Continue to [Coordinate Systems](./coord.md) to learn about different
ways to position visual elements.
