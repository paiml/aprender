# Counter App

Interactive counter with increment/decrement buttons.

## Code

```rust
use presentar::widgets::{Button, Column, Text};
use presentar::widgets::row::MainAxisAlignment;
use presentar::{Constraints, Rect, Size, Widget, RecordingCanvas};

fn main() {
    // Build UI
    let mut ui = Column::new()
        .main_axis_alignment(MainAxisAlignment::Center)
        .gap(16.0)
        .child(
            Text::new("Counter: 0")
                .font_size(32.0)
                .with_test_id("counter-display")
        )
        .child(
            Button::new("+1")
                .with_test_id("increment")
        )
        .child(
            Button::new("-1")
                .with_test_id("decrement")
        );

    // Measure
    let size = ui.measure(Constraints::loose(Size::new(400.0, 400.0)));

    // Layout
    ui.layout(Rect::new(0.0, 0.0, size.width, size.height));

    // Paint
    let mut canvas = RecordingCanvas::new();
    ui.paint(&mut canvas);

    println!("Counter app: {} commands", canvas.command_count());
}
```

## Testing

```rust
#[test]
fn test_counter_ui() {
    use presentar_test::Harness;
    use presentar_widgets::{Button, Column, Text};

    let ui = Column::new()
        .child(Text::new("0").with_test_id("display"))
        .child(Button::new("+").with_test_id("inc"))
        .child(Button::new("-").with_test_id("dec"));

    let harness = Harness::new(ui);

    harness
        .assert_exists("[data-testid='display']")
        .assert_exists("[data-testid='inc']")
        .assert_exists("[data-testid='dec']")
        .assert_count("[data-testid='display']", 1);
}
```

## Verified Test

```rust
#[test]
fn test_counter_builds() {
    use presentar_widgets::{Button, Column, Text};
    use presentar_core::{Constraints, Size, Widget};

    let ui = Column::new()
        .child(Text::new("0"))
        .child(Button::new("+"))
        .child(Button::new("-"));

    let size = ui.measure(Constraints::loose(Size::new(400.0, 400.0)));
    assert!(size.height > 0.0);
}
```
