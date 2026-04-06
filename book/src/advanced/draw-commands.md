# Draw Commands

Low-level rendering primitives.

## Command Types

```rust
pub enum DrawCommand {
    FillRect { bounds: Rect, color: Color, radius: CornerRadius },
    StrokeRect { bounds: Rect, color: Color, width: f32 },
    FillCircle { center: Point, radius: f32, color: Color },
    DrawLine { from: Point, to: Point, color: Color, width: f32 },
    DrawText { text: String, position: Point, style: TextStyle },
    SetClip { bounds: Rect },
    ClearClip,
}
```

## Emitting Commands

```rust
fn paint(&self, canvas: &mut dyn Canvas) {
    // Rectangle
    canvas.fill_rect(self.bounds, self.background);

    // Circle
    canvas.fill_circle(center, 10.0, Color::RED);

    // Text
    canvas.draw_text("Hello", position, &TextStyle::default());

    // Line
    canvas.draw_line(Point::new(0.0, 0.0), Point::new(100.0, 100.0), Color::BLACK, 1.0);
}
```

## Recording Canvas

For testing:

```rust
let mut canvas = RecordingCanvas::new();
widget.paint(&mut canvas);

assert_eq!(canvas.command_count(), 2);

for cmd in canvas.commands() {
    match cmd {
        DrawCommand::FillRect { bounds, .. } => {
            assert!(bounds.width > 0.0);
        }
        _ => {}
    }
}
```

## Performance

| Command | Cost |
|---------|------|
| FillRect | Low |
| DrawText | Medium |
| Complex Path | High |

## Batching

Commands batch automatically by type:

```
FillRect → FillRect → FillRect  // One draw call
DrawText → DrawText → DrawText  // One draw call
```

## Verified Test

```rust
#[test]
fn test_draw_commands() {
    use presentar_widgets::Button;
    use presentar_core::{Rect, Widget, RecordingCanvas};

    let mut button = Button::new("Test");
    button.layout(Rect::new(0.0, 0.0, 100.0, 40.0));

    let mut canvas = RecordingCanvas::new();
    button.paint(&mut canvas);

    assert!(canvas.command_count() >= 1);
}
```
