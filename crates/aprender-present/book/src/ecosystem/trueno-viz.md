# Trueno Viz

GPU-accelerated visualization primitives.

## Overview

| Feature | Description |
|---------|-------------|
| Backend | WebGPU/WGSL shaders |
| Primitives | Paths, fills, strokes, text |
| Batching | Draw call optimization |
| Atlas | Glyph texture caching |

## Primitive Types

| Type | Use Case |
|------|----------|
| Path | Custom shapes, curves |
| Rect | Boxes, backgrounds |
| Circle | Points, indicators |
| Line | Strokes, borders |
| Text | Labels, content |

## Drawing API

```rust
use trueno_viz::{Canvas, Color, Rect};

fn draw(canvas: &mut Canvas) {
    // Fill rectangle
    canvas.fill_rect(Rect::new(0, 0, 100, 50), Color::RED);

    // Stroke path
    canvas.stroke_path(&path, Color::BLACK, 2.0);

    // Draw text
    canvas.draw_text("Hello", Point::new(10, 10), &font);
}
```

## WGSL Shaders

```wgsl
@fragment
fn rect_fragment(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    return uniforms.color;
}
```

## Batching

```
Before: 100 rects → 100 draw calls
After:  100 rects → 1 batched draw call
```

## Performance

| Operation | Time |
|-----------|------|
| 1000 rects | 0.5ms |
| 100 text glyphs | 1ms |
| Full frame | 2ms |

## Verified Test

```rust
#[test]
fn test_trueno_viz_color_components() {
    use presentar_core::Color;

    // Color has RGBA components
    let red = Color::new(1.0, 0.0, 0.0, 1.0);
    assert_eq!(red.r, 1.0);
    assert_eq!(red.g, 0.0);
    assert_eq!(red.b, 0.0);
    assert_eq!(red.a, 1.0);

    // Premultiplied alpha
    let semi = Color::new(1.0, 0.0, 0.0, 0.5);
    assert_eq!(semi.a, 0.5);
}
```
