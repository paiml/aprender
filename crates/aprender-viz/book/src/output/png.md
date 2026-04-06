# PNG Encoding

PNG (Portable Network Graphics) is the default output format for trueno-viz,
providing lossless compression and full color support.

## Basic PNG Output

```rust
use trueno_viz::prelude::*;
use trueno_viz::plots::ScatterPlot;

let plot = ScatterPlot::new()
    .x(&[1.0, 2.0, 3.0])
    .y(&[1.0, 4.0, 9.0])
    .build();

// Default size (800x600)
plot.render_to_file("output.png").unwrap();
```

**Test Reference**: `src/output/png_encoder.rs::test_write_to_file`

## Custom Dimensions

```rust
use trueno_viz::plots::ScatterPlot;

let plot = ScatterPlot::new()
    .x(&[1.0, 2.0, 3.0])
    .y(&[1.0, 4.0, 9.0])
    .build();

// Custom size
plot.render_to_file_with_size("large.png", 1920, 1080).unwrap();

// Square
plot.render_to_file_with_size("square.png", 800, 800).unwrap();
```

## To Bytes

For in-memory processing or web responses:

```rust
use trueno_viz::plots::ScatterPlot;

let plot = ScatterPlot::new()
    .x(&[1.0, 2.0, 3.0])
    .y(&[1.0, 4.0, 9.0])
    .build();

let png_bytes: Vec<u8> = plot.render_to_bytes(800, 600).unwrap();
println!("PNG size: {} bytes", png_bytes.len());
```

## DPI Scaling

For high-resolution displays:

```rust
use trueno_viz::output::PngEncoder;

let encoder = PngEncoder::new()
    .width(800)
    .height(600)
    .dpi(300);  // Print quality

let png_bytes = encoder.encode(&framebuffer).unwrap();
```

**Test Reference**: `src/output/png_encoder.rs::test_dimensions`

## Compression Level

```rust
use trueno_viz::output::PngEncoder;

// Fastest encoding (larger file)
let fast = PngEncoder::new().compression(1);

// Best compression (smaller file, slower)
let best = PngEncoder::new().compression(9);

// Default (balanced)
let default = PngEncoder::new().compression(6);
```

## Color Depth

```rust
use trueno_viz::output::{PngEncoder, ColorType};

// 32-bit RGBA (default)
let rgba = PngEncoder::new().color_type(ColorType::Rgba);

// 24-bit RGB (no transparency)
let rgb = PngEncoder::new().color_type(ColorType::Rgb);

// 8-bit grayscale
let gray = PngEncoder::new().color_type(ColorType::Grayscale);
```

## Background Color

```rust
use trueno_viz::prelude::*;
use trueno_viz::output::PngEncoder;

let encoder = PngEncoder::new()
    .background(Rgba::WHITE);  // White background

let encoder = PngEncoder::new()
    .background(Rgba::TRANSPARENT);  // Transparent background
```

## Direct Framebuffer Encoding

```rust
use trueno_viz::prelude::*;
use trueno_viz::output::PngEncoder;

// Create framebuffer
let mut fb = Framebuffer::new(400, 300);

// Draw directly
fb.clear(Rgba::WHITE);
fb.draw_rect(50, 50, 100, 80, Rgba::BLUE);

// Encode to PNG
let encoder = PngEncoder::new();
let bytes = encoder.encode(&fb).unwrap();

std::fs::write("direct.png", bytes).unwrap();
```

## Performance

PNG encoding uses SIMD for:
- Pixel format conversion
- Filter prediction
- Compression preprocessing

```rust
use trueno_viz::plots::Heatmap;
use std::time::Instant;

// Large heatmap
let data: Vec<f32> = (0..1_000_000).map(|i| i as f32).collect();
let heatmap = Heatmap::new(&data, 1000, 1000).build();

let start = Instant::now();
heatmap.render_to_file("large_heatmap.png").unwrap();
println!("Encoding time: {:?}", start.elapsed());
```

## Complete Example

```rust
use trueno_viz::prelude::*;
use trueno_viz::plots::ScatterPlot;
use trueno_viz::output::PngEncoder;

fn main() -> Result<()> {
    // Create plot
    let plot = ScatterPlot::new()
        .x(&[1.0, 2.0, 3.0, 4.0, 5.0])
        .y(&[2.0, 4.0, 3.0, 5.0, 4.5])
        .color(Rgba::new(66, 133, 244, 255))
        .title("My Scatter Plot")
        .build();

    // Standard output
    plot.render_to_file("scatter.png")?;

    // High-resolution for print
    plot.render_to_file_with_size("scatter_print.png", 2400, 1800)?;

    // To bytes for web
    let bytes = plot.render_to_bytes(800, 600)?;
    println!("Generated {} KB PNG", bytes.len() / 1024);

    Ok(())
}
```

## Next Chapter

Continue to [SVG Generation](./svg.md) for vector graphics output.
