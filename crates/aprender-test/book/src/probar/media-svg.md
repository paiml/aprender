# SVG Export

> **Toyota Way**: Poka-Yoke (Mistake-Proofing) - Scalable vector output

Generate resolution-independent SVG screenshots for documentation and scaling.

## Basic Usage

```rust
use probar::media::{SvgConfig, SvgExporter, SvgShape};

let config = SvgConfig::new(800, 600);
let mut exporter = SvgExporter::new(config);

exporter.add_shape(SvgShape::rect(50.0, 50.0, 200.0, 100.0)
    .with_fill("#3498db")
    .with_stroke("#2980b9"));

let svg_content = exporter.export()?;
```
