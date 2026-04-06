# Rendering Pipeline

From widgets to pixels.

## Pipeline Stages

```
Widget Tree → Canvas Commands → GPU Primitives → Framebuffer
```

## Stage 1: Widget Paint

Widgets emit draw commands:

```rust
fn paint(&self, canvas: &mut dyn Canvas) {
    canvas.fill_rect(self.bounds, self.background);
    canvas.draw_text(&self.label, self.bounds.center(), &style);
}
```

## Stage 2: Canvas Commands

Commands collected:

```rust
enum DrawCommand {
    FillRect { bounds: Rect, color: Color },
    DrawText { text: String, position: Point, style: TextStyle },
    FillCircle { center: Point, radius: f32, color: Color },
    DrawLine { from: Point, to: Point, color: Color, width: f32 },
}
```

## Stage 3: GPU Primitives

Via Trueno-Viz:

```
FillRect → Quad mesh → Vertex buffer → WGSL shader
DrawText → Glyph atlas → Texture sample → Fragment shader
```

## Stage 4: Framebuffer

Final pixels rendered at 60fps target.

## Recording Canvas

For testing:

```rust
let mut canvas = RecordingCanvas::new();
widget.paint(&mut canvas);

assert_eq!(canvas.command_count(), 2);
let commands = canvas.commands();
```

## Performance

| Stage | Budget |
|-------|--------|
| Paint | <2ms |
| Commands | <1ms |
| GPU | <10ms |
| **Total** | <16ms |

## Verified Test

```rust
#[test]
fn test_rendering_pipeline() {
    use presentar_widgets::Button;
    use presentar_core::{Rect, Widget, RecordingCanvas};

    let mut button = Button::new("Test");
    button.layout(Rect::new(0.0, 0.0, 100.0, 40.0));

    let mut canvas = RecordingCanvas::new();
    button.paint(&mut canvas);

    assert!(canvas.command_count() >= 2);  // Background + text
}
```
