# SVG Generation

SVG (Scalable Vector Graphics) output provides resolution-independent
graphics that can be scaled without loss of quality.

## Basic SVG Output

```rust
use trueno_viz::prelude::*;
use trueno_viz::plots::ScatterPlot;

#[cfg(feature = "svg")]
{
    let plot = ScatterPlot::new()
        .x(&[1.0, 2.0, 3.0])
        .y(&[1.0, 4.0, 9.0])
        .build();

    plot.render_svg_to_file("output.svg").unwrap();
}
```

**Test Reference**: `src/output/svg.rs::test_svg_output`

## SVG to String

```rust
#[cfg(feature = "svg")]
{
    let plot = ScatterPlot::new()
        .x(&[1.0, 2.0, 3.0])
        .y(&[1.0, 4.0, 9.0])
        .build();

    let svg_string = plot.render_svg_to_string(800, 600);
    println!("{}", svg_string);
}
```

## Custom Dimensions

```rust
#[cfg(feature = "svg")]
{
    plot.render_svg_to_file_with_size("wide.svg", 1200, 400).unwrap();
}
```

## SVG Elements

The SVG writer generates semantic SVG elements:

```rust
use trueno_viz::output::svg::{SvgWriter, SvgElement};

let mut svg = SvgWriter::new(800, 600);

// Rectangle
svg.rect(50.0, 50.0, 100.0, 80.0)
    .fill(Rgba::BLUE)
    .stroke(Rgba::BLACK)
    .stroke_width(2.0);

// Circle
svg.circle(200.0, 100.0, 30.0)
    .fill(Rgba::RED);

// Line
svg.line(300.0, 50.0, 400.0, 150.0)
    .stroke(Rgba::BLACK)
    .stroke_width(1.0);

// Path
svg.path("M 100 200 L 200 300 L 300 250 Z")
    .fill(Rgba::GREEN);

// Text
svg.text(150.0, 400.0, "Hello SVG!")
    .font_size(16.0)
    .font_family("sans-serif")
    .anchor(TextAnchor::Middle);

let output = svg.to_string();
```

**Test Reference**: `src/output/svg.rs::test_svg_elements`

## Text Anchoring

```rust
use trueno_viz::output::svg::{SvgWriter, TextAnchor};

let mut svg = SvgWriter::new(400, 200);

svg.text(200.0, 100.0, "Start")
    .anchor(TextAnchor::Start);

svg.text(200.0, 120.0, "Middle")
    .anchor(TextAnchor::Middle);

svg.text(200.0, 140.0, "End")
    .anchor(TextAnchor::End);
```

**Test Reference**: `src/output/svg.rs::test_svg_text_anchors`

## Styling

```rust
use trueno_viz::output::svg::SvgWriter;

let mut svg = SvgWriter::new(400, 300);

// Gradient fill
svg.defs()
    .linear_gradient("myGradient")
    .stop(0.0, Rgba::BLUE)
    .stop(1.0, Rgba::RED);

svg.rect(50.0, 50.0, 200.0, 100.0)
    .fill_url("myGradient");
```

## Groups and Transforms

```rust
use trueno_viz::output::svg::SvgWriter;

let mut svg = SvgWriter::new(400, 300);

svg.group()
    .transform("translate(100, 100)")
    .child(|g| {
        g.circle(0.0, 0.0, 30.0).fill(Rgba::BLUE);
        g.circle(50.0, 0.0, 30.0).fill(Rgba::RED);
    });
```

## Embedding in HTML

```html
<!DOCTYPE html>
<html>
<body>
    <!-- Inline SVG -->
    <div id="plot"></div>
    <script>
        document.getElementById('plot').innerHTML = `
            ${svg_string}
        `;
    </script>

    <!-- External SVG -->
    <img src="plot.svg" alt="Plot">

    <!-- Object tag (interactive) -->
    <object data="plot.svg" type="image/svg+xml"></object>
</body>
</html>
```

## Advantages of SVG

| Feature | PNG | SVG |
|---------|-----|-----|
| Scalability | Fixed resolution | Infinite |
| File size (simple) | Larger | Smaller |
| File size (complex) | Smaller | Larger |
| Editability | No | Yes |
| Animation | No | Yes |
| Searchable text | No | Yes |

## Complete Example

```rust
use trueno_viz::prelude::*;
use trueno_viz::plots::LineChart;

fn main() -> Result<()> {
    let x: Vec<f32> = (0..100).map(|i| i as f32 * 0.1).collect();
    let y: Vec<f32> = x.iter().map(|x| x.sin()).collect();

    let chart = LineChart::new()
        .x(&x)
        .y(&y)
        .color(Rgba::new(66, 133, 244, 255))
        .title("Sine Wave")
        .xlabel("x")
        .ylabel("sin(x)")
        .build();

    // PNG for raster display
    chart.render_to_file("sine.png")?;

    // SVG for vector display / web
    #[cfg(feature = "svg")]
    chart.render_svg_to_file("sine.svg")?;

    Ok(())
}
```

## Next Chapter

Continue to [Terminal Rendering](./terminal.md) for ASCII/Unicode output.
