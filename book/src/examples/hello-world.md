# Hello World

Minimal Presentar application.

## Code

```rust
use presentar::widgets::{Column, Text};
use presentar::widgets::row::MainAxisAlignment;
use presentar::{Constraints, Rect, Size, Widget, RecordingCanvas};

fn main() {
    // Build UI
    let mut ui = Column::new()
        .main_axis_alignment(MainAxisAlignment::Center)
        .gap(16.0)
        .child(
            Text::new("Hello, Presentar!")
                .font_size(24.0)
        )
        .child(
            Text::new("A WASM-first visualization framework")
                .font_size(14.0)
        );

    // Measure
    let size = ui.measure(Constraints::loose(Size::new(400.0, 300.0)));
    println!("Size: {}x{}", size.width, size.height);

    // Layout
    ui.layout(Rect::new(0.0, 0.0, size.width, size.height));

    // Paint
    let mut canvas = RecordingCanvas::new();
    ui.paint(&mut canvas);
    println!("Commands: {}", canvas.command_count());
}
```

## Run

```bash
cargo run --example hello_world
```

## Output

```
Size: 302.4x118.4
Commands: 4
```

## Verified Test

```rust
#[test]
fn test_hello_world() {
    use presentar_widgets::{Column, Text};
    use presentar_core::{Constraints, Size, Widget};

    let ui = Column::new()
        .child(Text::new("Hello, Presentar!"));

    let size = ui.measure(Constraints::loose(Size::new(400.0, 300.0)));
    assert!(size.width > 0.0);
}
```
