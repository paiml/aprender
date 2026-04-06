# Event Simulation

Simulate user interactions programmatically.

## Click Events

```rust
// Click by selector
harness.click("[data-testid='submit']");
```

Internally generates: `MouseMove` → `MouseDown` → `MouseUp`

## Text Input

```rust
// Type text into focused widget
harness.type_text("[data-testid='username']", "alice");
```

Generates: `FocusIn` → `TextInput` for each character

## Key Press

```rust
use presentar_core::Key;

harness.press_key(Key::Enter);
harness.press_key(Key::Escape);
harness.press_key(Key::Tab);
```

Generates: `KeyDown` → `KeyUp`

## Scrolling

```rust
harness.scroll("[data-testid='list']", 100.0);  // Scroll down
harness.scroll("[data-testid='list']", -50.0);  // Scroll up
```

## Time Simulation

```rust
harness.tick(1000);  // Advance 1 second
```

Used for animations and debounced events.

## Event Flow

```
click("[data-testid='btn']")
    │
    ├─► MouseMove { position: center }
    ├─► MouseDown { position: center, button: Left }
    └─► MouseUp { position: center, button: Left }
         │
         └─► Widget receives events
              │
              └─► May emit ButtonClicked message
```

## Verified Test

```rust
#[test]
fn test_button_click_flow() {
    use presentar_test::Harness;
    use presentar_widgets::Button;
    use presentar_core::{Event, MouseButton, Point, Rect, Widget};

    let mut button = Button::new("Test").with_test_id("btn");
    button.layout(Rect::new(0.0, 0.0, 100.0, 40.0));

    // Simulate click sequence
    button.event(&Event::MouseDown {
        position: Point::new(50.0, 20.0),
        button: MouseButton::Left,
    });
    let result = button.event(&Event::MouseUp {
        position: Point::new(50.0, 20.0),
        button: MouseButton::Left,
    });

    assert!(result.is_some());  // ButtonClicked emitted
}
```
