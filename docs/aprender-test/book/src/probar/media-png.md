# PNG Screenshots

> **Toyota Way**: Genchi Genbutsu (Go and See) - Visual evidence of test state

Capture high-quality PNG screenshots with metadata and annotations.

## Basic Usage

```rust
use probar::media::{PngExporter, PngMetadata, Annotation, CompressionLevel};

let exporter = PngExporter::new()
    .with_compression(CompressionLevel::Best)
    .with_metadata(PngMetadata::new()
        .with_title("Test Screenshot")
        .with_test_name("login_test"));

let png_data = exporter.export(&screenshot)?;
```

## Annotations

```rust
let annotations = vec![
    Annotation::rectangle(50, 50, 100, 80)
        .with_color(255, 0, 0, 255)
        .with_label("Error area"),
    Annotation::circle(400, 200, 60)
        .with_color(0, 255, 0, 255),
];

let annotated = exporter.export_with_annotations(&screenshot, &annotations)?;
```
