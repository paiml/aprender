# Color Types

This chapter documents the color types available in trueno-viz.

## Rgba

32-bit RGBA color with 8 bits per channel.

```rust
use trueno_viz::color::Rgba;

// Constructor
let color = Rgba::new(255, 128, 0, 255);  // Orange, fully opaque

// Named constants
let red = Rgba::RED;          // (255, 0, 0, 255)
let green = Rgba::GREEN;      // (0, 255, 0, 255)
let blue = Rgba::BLUE;        // (0, 0, 255, 255)
let white = Rgba::WHITE;      // (255, 255, 255, 255)
let black = Rgba::BLACK;      // (0, 0, 0, 255)
let transparent = Rgba::TRANSPARENT;  // (0, 0, 0, 0)

// Field access
println!("Red: {}", color.r);
println!("Green: {}", color.g);
println!("Blue: {}", color.b);
println!("Alpha: {}", color.a);
```

### From Hex

```rust
use trueno_viz::color::Rgba;

let color = Rgba::from_hex("#ff8000").unwrap();
let with_alpha = Rgba::from_hex("#ff8000cc").unwrap();
```

### To Hex

```rust
let hex = color.to_hex();  // "#ff8000"
let hex_alpha = color.to_hex_alpha();  // "#ff8000ff"
```

### Blending

```rust
let src = Rgba::new(255, 0, 0, 128);  // Semi-transparent red
let dst = Rgba::new(0, 0, 255, 255);  // Opaque blue

let blended = src.blend(dst);  // Alpha-blended result
```

## Hsla

Color in HSL (Hue, Saturation, Lightness) space with alpha.

```rust
use trueno_viz::color::Hsla;

// H: 0-360 degrees, S/L/A: 0.0-1.0
let color = Hsla::new(30.0, 1.0, 0.5, 1.0);  // Orange

// Manipulations
let lighter = color.lighten(0.2);   // Increase lightness
let darker = color.darken(0.2);     // Decrease lightness
let saturated = color.saturate(0.2);
let desaturated = color.desaturate(0.2);
let rotated = color.rotate(180.0);  // Complementary color
```

### Conversions

```rust
use trueno_viz::color::{Rgba, Hsla};

let rgba = Rgba::new(255, 128, 0, 255);
let hsla = Hsla::from(rgba);

let back = Rgba::from(hsla);
```

## Color Palettes

```rust
use trueno_viz::color::palette;

// Sequential palettes
let viridis = palette::viridis(0.5);  // Value 0.0-1.0 → color
let plasma = palette::plasma(0.5);
let inferno = palette::inferno(0.5);

// Diverging palettes
let rdbu = palette::rdbu(0.5);  // Blue → White → Red

// Categorical palettes
let colors = palette::category10();  // 10 distinct colors
let more = palette::category20();    // 20 distinct colors
```

## Complete API

```rust
impl Rgba {
    pub const fn new(r: u8, g: u8, b: u8, a: u8) -> Self;
    pub fn from_hex(hex: &str) -> Result<Self>;
    pub fn to_hex(&self) -> String;
    pub fn to_hex_alpha(&self) -> String;
    pub fn blend(self, dst: Self) -> Self;
    pub fn with_alpha(self, a: u8) -> Self;
    pub fn lerp(self, other: Self, t: f32) -> Self;

    // Constants
    pub const RED: Self;
    pub const GREEN: Self;
    pub const BLUE: Self;
    pub const WHITE: Self;
    pub const BLACK: Self;
    pub const TRANSPARENT: Self;
}

impl Hsla {
    pub fn new(h: f32, s: f32, l: f32, a: f32) -> Self;
    pub fn lighten(self, amount: f32) -> Self;
    pub fn darken(self, amount: f32) -> Self;
    pub fn saturate(self, amount: f32) -> Self;
    pub fn desaturate(self, amount: f32) -> Self;
    pub fn rotate(self, degrees: f32) -> Self;
}
```
