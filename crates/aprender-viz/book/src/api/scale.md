# Scale Functions

This chapter documents the scale types for data-to-visual mappings.

## LinearScale

Maps continuous data linearly to a visual range.

```rust
use trueno_viz::scale::LinearScale;

let scale = LinearScale::new()
    .domain(0.0, 100.0)   // Data range
    .range(0.0, 800.0);   // Pixel range

// Forward transform
let pixel = scale.transform(50.0);  // 400.0

// Inverse transform
let value = scale.inverse(400.0);   // 50.0

// Options
let scale = LinearScale::new()
    .domain(0.0, 100.0)
    .range(0.0, 800.0)
    .nice()        // Round domain to nice values
    .clamp(true);  // Clamp values outside domain
```

## LogScale

Maps continuous data logarithmically.

```rust
use trueno_viz::scale::LogScale;

let scale = LogScale::new()
    .domain(1.0, 1000.0)
    .range(0.0, 300.0)
    .base(10.0);

// Powers of 10 map evenly
let p1 = scale.transform(1.0);    // 0.0
let p10 = scale.transform(10.0);  // 100.0
let p100 = scale.transform(100.0); // 200.0
let p1000 = scale.transform(1000.0); // 300.0
```

## ColorScale

Maps values to colors.

```rust
use trueno_viz::scale::ColorScale;
use trueno_viz::color::Rgba;

// Continuous gradient
let scale = ColorScale::continuous()
    .domain(0.0, 100.0)
    .colors(&[Rgba::BLUE, Rgba::WHITE, Rgba::RED]);

let cold = scale.transform(0.0);   // Blue
let mid = scale.transform(50.0);   // White
let hot = scale.transform(100.0);  // Red

// Built-in palettes
use trueno_viz::scale::Palette;

let viridis = ColorScale::palette(Palette::Viridis);
let plasma = ColorScale::palette(Palette::Plasma);
```

## Complete API

```rust
impl LinearScale {
    pub fn new() -> Self;
    pub fn domain(self, min: f32, max: f32) -> Self;
    pub fn range(self, min: f32, max: f32) -> Self;
    pub fn nice(self) -> Self;
    pub fn clamp(self, clamp: bool) -> Self;
    pub fn transform(&self, value: f32) -> f32;
    pub fn inverse(&self, value: f32) -> f32;
    pub fn breaks(&self) -> Vec<f32>;
}

impl LogScale {
    pub fn new() -> Self;
    pub fn domain(self, min: f32, max: f32) -> Self;
    pub fn range(self, min: f32, max: f32) -> Self;
    pub fn base(self, base: f32) -> Self;
    pub fn transform(&self, value: f32) -> f32;
    pub fn inverse(&self, value: f32) -> f32;
}

impl ColorScale {
    pub fn continuous() -> Self;
    pub fn discrete() -> Self;
    pub fn palette(palette: Palette) -> Self;
    pub fn domain(self, min: f32, max: f32) -> Self;
    pub fn colors(self, colors: &[Rgba]) -> Self;
    pub fn transform(&self, value: f32) -> Rgba;
}
```
