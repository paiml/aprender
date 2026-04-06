# First App

Build a complete counter application step by step.

## Project Structure

```
counter-app/
├── Cargo.toml
├── src/
│   └── main.rs
└── tests/
    └── counter_test.rs
```

## Dependencies

```toml
[package]
name = "counter-app"
version = "0.1.0"
edition = "2021"

[dependencies]
presentar = "0.1"

[dev-dependencies]
presentar-test = "0.1"
```

## The Counter Widget

```rust
use presentar::widgets::{Button, Column, Text};
use presentar::widgets::row::MainAxisAlignment;
use presentar::{Color, Constraints, Rect, Size, Widget, RecordingCanvas};

fn main() {
    // Build the UI
    let mut ui = Column::new()
        .main_axis_alignment(MainAxisAlignment::Center)
        .gap(16.0)
        .child(Text::new("Counter: 0").font_size(24.0))
        .child(Button::new("+1").with_test_id("increment"))
        .child(Button::new("-1").with_test_id("decrement"));

    // Measure
    let constraints = Constraints::loose(Size::new(400.0, 300.0));
    let size = ui.measure(constraints);

    // Layout
    ui.layout(Rect::new(0.0, 0.0, size.width, size.height));

    // Paint
    let mut canvas = RecordingCanvas::new();
    ui.paint(&mut canvas);

    println!("Drew {} commands", canvas.command_count());
}
```

## Testing

```rust
#[test]
fn test_counter_ui() {
    use presentar_test::Harness;
    use presentar::widgets::{Button, Column, Text};

    let ui = Column::new()
        .child(Text::new("Counter: 0").with_test_id("count"))
        .child(Button::new("+1").with_test_id("increment"));

    let harness = Harness::new(ui);

    harness
        .assert_exists("[data-testid='count']")
        .assert_exists("[data-testid='increment']");
}
```

## Running

```bash
cargo run
cargo test
```

## Next Steps

- Add state management for actual counting
- Style the buttons
- Add keyboard shortcuts

## Verified Test

```rust
#[test]
fn test_first_app_builds() {
    use presentar_widgets::{Button, Column, Text};
    use presentar_core::{Constraints, Size, Widget};

    let ui = Column::new()
        .child(Text::new("Test"))
        .child(Button::new("Click"));

    let size = ui.measure(Constraints::loose(Size::new(400.0, 300.0)));
    assert!(size.width > 0.0);
    assert!(size.height > 0.0);
}
```
